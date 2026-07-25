# import csv  # IMU CSV logging is disabled for now.
import logging
import os
import threading
import time

import cv2
import numpy as np
import rclpy

from njord.config.camera_config import RGB_SHAPE
from njord.core import shared_state
from njord.core.shared_memory_utils import attach_existing_shared_memory
from utils.frame_cadence import FrameCadence
from utils.video_writer import QueuedVideoWriter, close_video_writers
from vision.detection_cache import VisionDetectionCache
from vision.detection_distance import nearest_bbox_median_distance
from vision.render import draw_detections

OUTPUT_DIR = "logs"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
# CSV_PATH = os.path.join(OUTPUT_DIR, "imu_log.csv")  # IMU CSV logging is disabled for now.
VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "run_{ts}.mp4")
DEPTH_VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "depth_run_{ts}.mp4")
VIDEO_FPS = 10
VIDEO_WRITER_CLOSE_TIMEOUT_SEC = 30.0

logger = logging.getLogger("zed_capture")


def setup_output_dirs():
    os.makedirs(VIDEO_DIR, exist_ok=True)


def attach_shared_memory(name, retries=50, delay=0.1):
    return attach_existing_shared_memory(name, retries=retries, delay=delay)


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


def annotate_frame(frame_bgr, detections):
    return draw_detections(frame_bgr, detections)


# noinspection D
def run(
    frame_lock=None,
    frame_ready_event=None,
    stop_event=None,
    active_task="task1",
    fx=None,
    cx=None,
    on_shutdown_started=None,
):
    setup_output_dirs()
    frame_index = 0
    dropped_frames = 0
    dropped_depth_frames = 0

    video_recorder = None
    depth_video_recorder = None
    rgb_shm = None
    depth_vision_shm = None
    meta_shm = None
    imu_shm = None
    shm_rgb = None
    shm_depth_vision = None
    shm_meta = None
    shm_imu = None
    detection_node = None
    detection_spin_thread = None
    owns_rclpy_context = False

    # Preallocated reusable buffers -> avoids per-frame np/cv2 allocation churn.
    bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)
    depth_vision_bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)
    h, w = RGB_SHAPE[:2]
    frame_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)
    depth_vision_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)

    last_drop_log = 0.0
    last_frame_id = 0
    record_cadence = FrameCadence(VIDEO_FPS)

    try:
        if not rclpy.ok():
            rclpy.init()
            owns_rclpy_context = True
        detection_node = VisionDetectionCache("njord_video_detection_cache")
        detection_spin_thread = threading.Thread(
            target=rclpy.spin, args=(detection_node,), daemon=True
        )
        detection_spin_thread.start()

        rgb_shm = attach_shared_memory(shared_state.RGB_SHM_NAME)
        depth_vision_shm = attach_shared_memory(shared_state.DEPTH_VISION_SHM_NAME)
        meta_shm = attach_shared_memory(shared_state.META_SHM_NAME)
        imu_shm = attach_shared_memory(shared_state.IMU_SHM_NAME)

        shm_rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=rgb_shm.buf)
        shm_depth_vision = np.ndarray(
            RGB_SHAPE, dtype=np.uint8, buffer=depth_vision_shm.buf
        )
        shm_meta = np.ndarray(shared_state.META_SHAPE, dtype=np.int64, buffer=meta_shm.buf)
        shm_imu = np.ndarray(shared_state.IMU_SHAPE, dtype=np.float64, buffer=imu_shm.buf)

        run_timestamp = int(time.time())
        video_path = VIDEO_PATH_TEMPLATE.format(ts=run_timestamp)
        depth_video_path = DEPTH_VIDEO_PATH_TEMPLATE.format(ts=run_timestamp)
        video_recorder = QueuedVideoWriter(
            video_path,
            (w, h),
            VIDEO_FPS,
            queue_size=100,
            logger=logger,
            label="NJORD RGB",
        )
        depth_video_recorder = QueuedVideoWriter(
            depth_video_path,
            (w, h),
            VIDEO_FPS,
            queue_size=100,
            logger=logger,
            label="NJORD depth",
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
                    np.copyto(depth_vision_bgra_buf, shm_depth_vision)
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
                        np.copyto(depth_vision_bgra_buf, shm_depth_vision)

            last_frame_id = current_frame_id

            if should_record:
                detections = detection_node.latest(current_frame_id)
                bbox_median_depth = nearest_bbox_median_distance(detections)
                cv2.cvtColor(bgra_buf, cv2.COLOR_BGRA2BGR, dst=frame_bgr_buf)
                cv2.cvtColor(
                    depth_vision_bgra_buf,
                    cv2.COLOR_BGRA2BGR,
                    dst=depth_vision_bgr_buf,
                )
                try:
                    processed_frame = annotate_frame(
                        frame_bgr_buf, detections
                    )
                except Exception:
                    logger.exception("NJORD video annotation failed. Raw frame will be used.")
                    processed_frame = frame_bgr_buf.copy()

                draw_frame_timestamp(
                    processed_frame,
                    timestamp_ms=timestamp_ms,
                    frame_index=current_frame_id,
                )
                depth_record_frame = depth_vision_bgr_buf.copy()
                draw_frame_timestamp(
                    depth_record_frame,
                    timestamp_ms=timestamp_ms,
                    frame_index=current_frame_id,
                )

                # The Flask video server reads this same annotated frame.
                with shared_state.frame_condition:
                    shared_state.latest_frame = processed_frame.copy()
                    shared_state.latest_frame_id = current_frame_id
                    shared_state.frame_condition.notify_all()

                if not video_recorder.enqueue(processed_frame):
                    dropped_frames += 1
                    now = time.monotonic()
                    if now - last_drop_log > 1.0:  # rate-limit logging, don't block hot path
                        logger.warning(
                            "RGB disk writer is lagging; dropped frames: %d",
                            dropped_frames,
                        )
                        last_drop_log = now

                if not depth_video_recorder.enqueue(depth_record_frame):
                    dropped_depth_frames += 1
                    now = time.monotonic()
                    if now - last_drop_log > 1.0:
                        logger.warning(
                            "Depth disk writer is lagging; dropped frames: %d",
                            dropped_depth_frames,
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
                shared_state.latest_imu = {"roll": roll, "pitch": pitch, "yaw": yaw}
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
            (video_recorder, depth_video_recorder),
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

        shm_rgb = None
        shm_depth_vision = None
        shm_meta = None
        shm_imu = None

        for shm in (rgb_shm, depth_vision_shm, meta_shm, imu_shm):
            if shm is not None:
                shm.close()

        if detection_node is not None:
            detection_node.destroy_node()
        if owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()
        if detection_spin_thread is not None:
            detection_spin_thread.join(timeout=2.0)
