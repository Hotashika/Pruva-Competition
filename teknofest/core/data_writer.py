import os
import time
import logging
import threading
from multiprocessing import shared_memory

import cv2
import numpy as np
import rclpy
from std_msgs.msg import String

from teknofest.config.camera_config import RGB_SHAPE
from teknofest.core import shared_state
from teknofest.core.imu_csv_writer import (
    MISSION_RECORDING_TOPIC,
    MissionRecordingState,
    ZedImuCsvWriter,
)
from utils.frame_cadence import FrameCadence
from utils.video_writer import QueuedVideoWriter, close_video_writers
from vision.detection_cache import VisionDetectionCache
from vision.detection_distance import nearest_bbox_median_distance
from vision.render import draw_detections

OUTPUT_DIR = "logs"
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_frames")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")

DEPTH_BIN_PATH = os.path.join(OUTPUT_DIR, "depth_stream.bin")  # single append-only file (disabled for now)

VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "run_{ts}.mp4")

# Şartname için en az 1 Hz yeterli.
VIDEO_FPS = 10
VIDEO_WRITER_CLOSE_TIMEOUT_SEC = 30.0

logger = logging.getLogger("zed_capture")


def setup_output_dirs():
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)


def attach_shared_memory(name, retries=200, delay=0.1):
    """
    ZED kamerasının açılması (özellikle depth mode ile) birkaç saniye
    sürebilir. Eskiden bu fonksiyon sadece 5 saniye bekleyip pes ediyordu,
    bu yüzden kamera henüz hazır olmadan RuntimeError fırlatıp process'i
    çökertiyordu ("kamera açılmıyor" gibi görünen asıl sebeplerden biri
    buydu). Şimdi ~20 saniyeye kadar bekliyor ve ilerlemeyi logluyor.
    """
    last_error = None

    for attempt in range(retries):
        try:
            shm = shared_memory.SharedMemory(name=name)
            if attempt > 0:
                logger.info("%s shared memory %d. denemede bulundu.", name, attempt + 1)
            return shm
        except FileNotFoundError as exc:
            last_error = exc

            if attempt > 0 and attempt % 20 == 0:
                logger.warning(
                    "%s shared memory hala bulunamadı (%d. deneme). "
                    "Kamera/üretici process açık mı ve önce başlatıldı mı kontrol et.",
                    name,
                    attempt,
                )

            time.sleep(delay)

    raise RuntimeError(
        f"{name} shared memory not found after {retries * delay:.1f}s. "
        f"Kamerayı/üretici process'i (ZED capture) bu script'ten önce başlattığından emin ol."
    ) from last_error


def draw_frame_timestamp(frame, timestamp_ms, frame_index):
    """
    Frame üzerine zaman etiketi ve frame numarası yazar.
    """
    timestamp_seconds = timestamp_ms / 1000.0

    timestamp_text = (
        f"Timestamp: {timestamp_ms} ms | "
        f"Time: {timestamp_seconds:.3f} s | "
        f"Frame: {frame_index}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        timestamp_text,
        font,
        font_scale,
        thickness,
    )

    x = 10
    y = 10 + text_height

    # Yazının okunabilmesi için siyah arka plan
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        timestamp_text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame


# noinspection D
def run(
    frame_lock=None,
    frame_ready_event=None,
    stop_event=None,
    on_shutdown_started=None,
):
    setup_output_dirs()

    frame_index = 0
    dropped_frames = 0

    video_recorder = None
    imu_csv_writer = None
    active_imu_session = ""
    recording_state = MissionRecordingState()
    detection_node = None
    detection_spin_thread = None
    owns_rclpy_context = False

    rgb_shm = None
    meta_shm = None
    imu_shm = None

    # Preallocated reusable buffers -> avoids per-frame np/cv2 allocation churn.
    bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)

    h, w = RGB_SHAPE[:2]

    frame_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)

    last_drop_log = 0.0
    last_frame_id = 0
    record_cadence = FrameCadence(VIDEO_FPS)

    try:
        if not rclpy.ok():
            rclpy.init()
            owns_rclpy_context = True
        detection_node = VisionDetectionCache("teknofest_video_detection_cache")
        detection_node.create_subscription(
            String,
            MISSION_RECORDING_TOPIC,
            lambda message: recording_state.update(message.data),
            10,
        )
        detection_spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(detection_node,),
            daemon=True,
        )
        detection_spin_thread.start()

        rgb_shm = attach_shared_memory(shared_state.RGB_SHM_NAME)
        meta_shm = attach_shared_memory(shared_state.META_SHM_NAME)
        imu_shm = attach_shared_memory(shared_state.IMU_SHM_NAME)

        shm_rgb = np.ndarray(
            RGB_SHAPE,
            dtype=np.uint8,
            buffer=rgb_shm.buf,
        )

        shm_meta = np.ndarray(
            shared_state.META_SHAPE,
            dtype=np.int64,
            buffer=meta_shm.buf,
        )

        shm_imu = np.ndarray(
            shared_state.IMU_SHAPE,
            dtype=np.float64,
            buffer=imu_shm.buf,
        )

        video_path = VIDEO_PATH_TEMPLATE.format(ts=int(time.time()))

        video_recorder = QueuedVideoWriter(
            video_path,
            (w, h),
            VIDEO_FPS,
            queue_size=100,
            logger=logger,
            label="TEKNOFEST RGB",
        )

        while stop_event is None or not stop_event.is_set():
            if frame_ready_event is not None:
                frame_ready_event.wait(timeout=0.1)
                frame_ready_event.clear()

            if frame_lock is None:
                current_frame_id = int(shm_meta[0])
                if current_frame_id == 0 or current_frame_id == last_frame_id:
                    continue
                should_record = record_cadence.due(time.monotonic() * 1000.0)
                timestamp_ms = int(shm_meta[1])
                roll, pitch, yaw = shm_imu.tolist()
                if should_record:
                    np.copyto(bgra_buf, shm_rgb)
                if int(shm_meta[0]) != current_frame_id:
                    continue
            else:
                with frame_lock:
                    current_frame_id = int(shm_meta[0])
                    if current_frame_id == 0 or current_frame_id == last_frame_id:
                        continue
                    should_record = record_cadence.due(time.monotonic() * 1000.0)
                    timestamp_ms = int(shm_meta[1])
                    roll, pitch, yaw = shm_imu.tolist()
                    if should_record:
                        np.copyto(bgra_buf, shm_rgb)

            last_frame_id = current_frame_id

            requested_imu_session = recording_state.active_session()
            if requested_imu_session != active_imu_session:
                if imu_csv_writer is not None:
                    logger.info(
                        "ZED IMU recording stopped: %s",
                        imu_csv_writer.csv_path,
                    )
                    try:
                        imu_csv_writer.close()
                    except Exception:
                        logger.exception("ZED IMU CSV could not be closed.")
                    imu_csv_writer = None

                active_imu_session = requested_imu_session
                if active_imu_session:
                    try:
                        imu_csv_writer = ZedImuCsvWriter(active_imu_session)
                        logger.info(
                            "ZED IMU recording started: %s",
                            imu_csv_writer.csv_path,
                        )
                    except Exception:
                        logger.exception(
                            "ZED IMU recording could not be started for %s.",
                            active_imu_session,
                        )

            if imu_csv_writer is not None:
                try:
                    imu_csv_writer.write(
                        frame_id=current_frame_id,
                        camera_timestamp_ms=timestamp_ms,
                        roll_rad=roll,
                        pitch_rad=pitch,
                        yaw_rad=yaw,
                    )
                except Exception:
                    logger.exception(
                        "ZED IMU sample could not be recorded; "
                        "this mission's IMU writer is being closed."
                    )
                    try:
                        imu_csv_writer.close()
                    except Exception:
                        logger.exception("ZED IMU CSV could not be closed.")
                    imu_csv_writer = None

            # ------------------------------------------------------------
            # MP4 KAYDI: İşlenmiş kamera verisi
            # - En az 1 Hz
            # - MP4 formatı
            # - Her frame zaman etiketli
            # - Obje bbox + sınıf + güven + mesafe + varsa takip ID
            # ------------------------------------------------------------
            if should_record:
                detections = detection_node.latest(current_frame_id)
                bbox_median_depth = nearest_bbox_median_distance(detections)
                cv2.cvtColor(
                    bgra_buf,
                    cv2.COLOR_BGRA2BGR,
                    dst=frame_bgr_buf,
                )
                try:
                    processed_frame = draw_detections(
                        frame_bgr_buf,
                        detections,
                    )
                except Exception:
                    logger.exception(
                        "Detection annotation failed. Raw frame will be recorded."
                    )
                    processed_frame = frame_bgr_buf.copy()

                draw_frame_timestamp(
                    processed_frame,
                    timestamp_ms=timestamp_ms,
                    frame_index=current_frame_id,
                )

                with shared_state.frame_condition:
                    shared_state.latest_frame = processed_frame.copy()
                    shared_state.latest_frame_id = current_frame_id
                    shared_state.frame_condition.notify_all()

                if not video_recorder.enqueue(processed_frame):
                    dropped_frames += 1
                    now = time.monotonic()
                    if now - last_drop_log > 1.0:
                        logger.warning(
                            "Disk write speed is lagging, number of dropped frames: %d",
                            dropped_frames,
                        )
                        last_drop_log = now
            else:
                bbox_median_depth = (
                    detection_node.nearest_bbox_median_distance(current_frame_id)
                )

            with shared_state.data_condition:
                # Kept for API compatibility; this is the nearest detection's
                # already-computed bbox median depth, not the center pixel.
                shared_state.latest_center_depth = bbox_median_depth
                shared_state.latest_imu = {
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                }
                shared_state.latest_timestamp = timestamp_ms
                shared_state.latest_data_id = current_frame_id
                shared_state.data_condition.notify_all()

            frame_index += 1

    finally:
        if on_shutdown_started is not None:
            try:
                on_shutdown_started()
            except Exception:
                logger.exception("Could not notify launcher that video shutdown started.")
        print("System shutting down, writing remaining data to disk...")

        writer_results = close_video_writers(
            (video_recorder,),
            timeout=VIDEO_WRITER_CLOSE_TIMEOUT_SEC,
        )
        for result in writer_results:
            if not result.finalized:
                logger.error(
                    "Video close incomplete: requested=%s partial=%s error=%s",
                    result.requested_path,
                    result.partial_path,
                    result.error,
                )

        if imu_csv_writer is not None:
            logger.info(
                "ZED IMU recording stopped during shutdown: %s",
                imu_csv_writer.csv_path,
            )
            try:
                imu_csv_writer.close()
            except Exception:
                logger.exception("ZED IMU CSV could not be closed during shutdown.")

        for shm in (rgb_shm, meta_shm, imu_shm):
            if shm is not None:
                shm.close()

        if detection_node is not None:
            detection_node.destroy_node()
        if owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()
        if detection_spin_thread is not None:
            detection_spin_thread.join(timeout=2.0)
