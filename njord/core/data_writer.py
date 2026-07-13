# import csv  # IMU CSV logging is disabled for now.
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

from njord.config.camera_config import DEPTH_SHAPE, RGB_SHAPE
from njord.core import shared_state
from njord.core.shared_memory_utils import attach_existing_shared_memory
from njord.vision.detector import BuoyDetector, VesselDetector

OUTPUT_DIR = "logs"
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_frames")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
# CSV_PATH = os.path.join(OUTPUT_DIR, "imu_log.csv")  # IMU CSV logging is disabled for now.
DEPTH_BIN_PATH = os.path.join(OUTPUT_DIR, "depth_stream.bin")  # single append-only file (disabled for now)
VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "run_{ts}.mp4")
VIDEO_FPS = 5

TASK_DETECTOR_MAP = {
    "task1": ("buoy",),
    "task2": ("vessel",),
    "task3": ("vessel",),
    "task4": ("buoy",),
    "none": ("buoy", "vessel"),
}

DETECTOR_FACTORIES = {
    "buoy": BuoyDetector,
    "vessel": VesselDetector,
}

logger = logging.getLogger("zed_capture")


def setup_output_dirs():
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)


def attach_shared_memory(name, retries=50, delay=0.1):
    return attach_existing_shared_memory(name, retries=retries, delay=delay)


def detector_names_for_task(active_task):
    task_key = str(active_task or "task1").strip().lower()
    detector_names = TASK_DETECTOR_MAP.get(task_key)

    if detector_names is None:
        logger.warning(
            "Unknown NJORD task '%s' for video annotation, defaulting to task1 detectors.",
            active_task,
        )
        return TASK_DETECTOR_MAP["task1"]

    return detector_names


def create_frame_detectors(active_task, fx=None, cx=None):
    detectors = []

    for detector_name in detector_names_for_task(active_task):
        detector_cls = DETECTOR_FACTORIES[detector_name]
        detectors.append(
            (
                detector_name,
                detector_cls(fx=fx, cx=cx),
            )
        )

    return detectors


def disk_writer_worker(q, video_path, frame_size):
    """
    Writes the captured BGR frames to an .mp4 video file.

    IMU CSV logging and depth persistence are disabled for now (see commented
    blocks below).
    """
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, frame_size
    )

    # --- IMU-to-CSV temporarily disabled ---
    # with open(csv_path, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["timestamp", "pitch", "yaw", "roll", "frame_index"])

    while True:
        item = q.get()
        if item is None:
            break

        frame_bgr = item
        video_writer.write(frame_bgr)

        # --- IMU-to-CSV temporarily disabled ---
        # writer.writerow([timestamp_ms, pitch, yaw, roll, frame_index])

        # --- depth-to-disk temporarily disabled ---
        # depth_bytes = depth_data.tobytes()
        # depth_bin.write(depth_bytes)
        # offset += len(depth_bytes)

        q.task_done()

    video_writer.release()


def draw_frame_timestamp(frame, timestamp_ms, frame_index):
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


def annotate_frame(frame_bgr, depth_array, detectors):
    all_detections = []

    for detector_name, detector in detectors:
        detections = detector.detect(frame_bgr, depth_array)

        for detection in detections:
            detection["type"] = detector_name

        all_detections.extend(detections)

    if not detectors:
        return frame_bgr.copy()

    return detectors[0][1].draw_detections(frame_bgr, all_detections)


# noinspection D
def run(
    frame_lock=None,
    frame_ready_event=None,
    stop_event=None,
    active_task="task1",
    fx=None,
    cx=None,
):
    setup_output_dirs()
    frame_index = 0
    dropped_frames = 0

    write_queue = queue.Queue(maxsize=100)
    writer_thread = None  # started lazily once we know frame size (see below)
    rgb_shm = None
    depth_shm = None
    meta_shm = None
    imu_shm = None
    shm_rgb = None
    shm_depth = None
    shm_meta = None
    shm_imu = None

    # Preallocated reusable buffers -> avoids per-frame np/cv2 allocation churn.
    bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)
    depth_buf = np.empty(DEPTH_SHAPE, dtype=np.float32)
    h, w = RGB_SHAPE[:2]
    frame_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)
    dh, dw = h // 2, w // 2
    downsampled_depth_buf = np.empty((dh, dw), dtype=np.float32)

    last_drop_log = 0.0
    last_frame_id = 0
    record_interval_ms = max(1, int(1000 / VIDEO_FPS))
    last_record_time_ms = None

    try:
        frame_detectors = create_frame_detectors(active_task, fx=fx, cx=cx)
    except Exception:
        logger.exception(
            "NJORD video annotation detectors could not be loaded for task '%s'.",
            active_task,
        )
        raise

    try:
        rgb_shm = attach_shared_memory(shared_state.RGB_SHM_NAME)
        depth_shm = attach_shared_memory(shared_state.DEPTH_SHM_NAME)
        meta_shm = attach_shared_memory(shared_state.META_SHM_NAME)
        imu_shm = attach_shared_memory(shared_state.IMU_SHM_NAME)

        shm_rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=rgb_shm.buf)
        shm_depth = np.ndarray(DEPTH_SHAPE, dtype=np.float32, buffer=depth_shm.buf)
        shm_meta = np.ndarray(shared_state.META_SHAPE, dtype=np.int64, buffer=meta_shm.buf)
        shm_imu = np.ndarray(shared_state.IMU_SHAPE, dtype=np.float64, buffer=imu_shm.buf)

        video_path = VIDEO_PATH_TEMPLATE.format(ts=int(time.time()))
        writer_thread = threading.Thread(
            target=disk_writer_worker,
            args=(write_queue, video_path, (w, h)),
            daemon=True,
        )
        writer_thread.start()

        while stop_event is None or not stop_event.is_set():
            if frame_ready_event is not None:
                frame_ready_event.wait(timeout=0.1)
                frame_ready_event.clear()

            if frame_lock is None:
                current_frame_id = int(shm_meta[0])
                timestamp_ms = int(shm_meta[1])
                pitch, yaw, roll = shm_imu.tolist()
                np.copyto(bgra_buf, shm_rgb)
                np.copyto(depth_buf, shm_depth)
            else:
                with frame_lock:
                    current_frame_id = int(shm_meta[0])
                    timestamp_ms = int(shm_meta[1])
                    pitch, yaw, roll = shm_imu.tolist()
                    np.copyto(bgra_buf, shm_rgb)
                    np.copyto(depth_buf, shm_depth)

            if current_frame_id == 0 or current_frame_id == last_frame_id:
                continue
            last_frame_id = current_frame_id

            # Reuse output buffers via dst= to avoid new allocations every frame
            cv2.cvtColor(bgra_buf, cv2.COLOR_BGRA2BGR, dst=frame_bgr_buf)
            cv2.resize(
                depth_buf, (0, 0), dst=downsampled_depth_buf,
                fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA,
            )
            # float16 conversion disabled along with depth-to-disk writing (see below)
            # np.copyto(downsampled_depth_f16_buf, downsampled_depth_buf, casting="unsafe")

            now_record_time_ms = int(time.monotonic() * 1000)
            should_record = (
                last_record_time_ms is None
                or now_record_time_ms - last_record_time_ms >= record_interval_ms
            )

            if should_record:
                try:
                    processed_frame = annotate_frame(frame_bgr_buf, depth_buf, frame_detectors)
                except Exception:
                    logger.exception("NJORD video annotation failed. Raw frame will be used.")
                    processed_frame = frame_bgr_buf.copy()

                draw_frame_timestamp(
                    processed_frame,
                    timestamp_ms=timestamp_ms,
                    frame_index=current_frame_id,
                )

                # The Flask video server reads this same annotated frame.
                with shared_state.frame_lock:
                    shared_state.latest_frame = processed_frame.copy()

                shared_state.frame_event.set()

                try:
                    write_queue.put_nowait(processed_frame)
                    last_record_time_ms = now_record_time_ms
                except queue.Full:
                    dropped_frames += 1
                    now = time.monotonic()
                    if now - last_drop_log > 1.0:  # rate-limit logging, don't block hot path
                        logger.warning(
                            "Disk write speed is lagging, number of dropped frames: %d", dropped_frames
                        )
                        last_drop_log = now

            # --- minimize time spent holding locks: just pointer/scalar assignment ---
            with shared_state.data_lock:
                shared_state.latest_depth_array = downsampled_depth_buf.copy()
                shared_state.latest_imu = {"pitch": pitch, "yaw": yaw, "roll": roll}
                shared_state.latest_timestamp = timestamp_ms

            shared_state.data_event.set()
            frame_index += 1
    finally:
        print("System shutting down, writing remaining data to disk...")
        if writer_thread is not None:
            write_queue.put(None)
            writer_thread.join()

        shm_rgb = None
        shm_depth = None
        shm_meta = None
        shm_imu = None

        for shm in (rgb_shm, depth_shm, meta_shm, imu_shm):
            if shm is not None:
                shm.close()
