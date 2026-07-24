# import csv  # IMU CSV logging is disabled for now.
import logging
import json
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from njord.config.camera_config import DEPTH_SHAPE, RGB_SHAPE
from njord.core import shared_state
from njord.core.shared_memory_utils import attach_existing_shared_memory
from njord.vision.detector import BaseYOLODetector

OUTPUT_DIR = "logs"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
# CSV_PATH = os.path.join(OUTPUT_DIR, "imu_log.csv")  # IMU CSV logging is disabled for now.
VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "run_{ts}.mp4")
DEPTH_VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "depth_run_{ts}.mp4")
VIDEO_FPS = 5

logger = logging.getLogger("zed_capture")


def setup_output_dirs():
    os.makedirs(VIDEO_DIR, exist_ok=True)


def attach_shared_memory(name, retries=50, delay=0.1):
    return attach_existing_shared_memory(name, retries=retries, delay=delay)


class VisionDetectionCache(Node):
    """Cache vision output so recording never reruns the detector models."""

    def __init__(self):
        super().__init__("njord_video_detection_cache")
        self._lock = threading.Lock()
        self._frame_id = None
        self._detections = []
        self.create_subscription(String, "/vision/detections", self._callback, 10)

    def _callback(self, message):
        try:
            payload = json.loads(message.data)
            frame_id = int(payload.get("frame_id"))
            detections = payload.get("detections", [])
            if not isinstance(detections, list):
                return
        except (TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warn("Invalid /vision/detections message ignored.")
            return
        with self._lock:
            self._frame_id = frame_id
            self._detections = detections

    def latest(self, frame_id, max_frame_lag=3):
        with self._lock:
            if self._frame_id is None or abs(int(frame_id) - self._frame_id) > max_frame_lag:
                return []
            return [dict(item) for item in self._detections if isinstance(item, dict)]

def disk_writer_worker(q, video_path, frame_size):
    """
    Writes the captured BGR frames to an .mp4 video file.

    IMU CSV logging is disabled for now (see the commented block below).
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
        # writer.writerow([timestamp_ms, roll, pitch, yaw, frame_index])

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


def annotate_frame(frame_bgr, detections):
    # draw_detections does not access model state; avoid loading a second YOLO model.
    return BaseYOLODetector.draw_detections(None, frame_bgr, detections)


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
    dropped_depth_frames = 0

    write_queue = queue.Queue(maxsize=100)
    depth_write_queue = queue.Queue(maxsize=100)
    writer_thread = None  # started lazily once we know frame size (see below)
    depth_writer_thread = None
    rgb_shm = None
    depth_shm = None
    depth_vision_shm = None
    meta_shm = None
    imu_shm = None
    shm_rgb = None
    shm_depth = None
    shm_depth_vision = None
    shm_meta = None
    shm_imu = None
    detection_node = None
    detection_spin_thread = None
    owns_rclpy_context = False

    # Preallocated reusable buffers -> avoids per-frame np/cv2 allocation churn.
    bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)
    depth_buf = np.empty(DEPTH_SHAPE, dtype=np.float32)
    depth_vision_bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)
    h, w = RGB_SHAPE[:2]
    frame_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)
    depth_vision_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)
    dh, dw = h // 2, w // 2
    downsampled_depth_buf = np.empty((dh, dw), dtype=np.float32)

    last_drop_log = 0.0
    last_frame_id = 0
    record_interval_ms = max(1, int(1000 / VIDEO_FPS))
    last_record_time_ms = None

    try:
        if not rclpy.ok():
            rclpy.init()
            owns_rclpy_context = True
        detection_node = VisionDetectionCache()
        detection_spin_thread = threading.Thread(
            target=rclpy.spin, args=(detection_node,), daemon=True
        )
        detection_spin_thread.start()

        rgb_shm = attach_shared_memory(shared_state.RGB_SHM_NAME)
        depth_shm = attach_shared_memory(shared_state.DEPTH_SHM_NAME)
        depth_vision_shm = attach_shared_memory(shared_state.DEPTH_VISION_SHM_NAME)
        meta_shm = attach_shared_memory(shared_state.META_SHM_NAME)
        imu_shm = attach_shared_memory(shared_state.IMU_SHM_NAME)

        shm_rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=rgb_shm.buf)
        shm_depth = np.ndarray(DEPTH_SHAPE, dtype=np.float32, buffer=depth_shm.buf)
        shm_depth_vision = np.ndarray(
            RGB_SHAPE, dtype=np.uint8, buffer=depth_vision_shm.buf
        )
        shm_meta = np.ndarray(shared_state.META_SHAPE, dtype=np.int64, buffer=meta_shm.buf)
        shm_imu = np.ndarray(shared_state.IMU_SHAPE, dtype=np.float64, buffer=imu_shm.buf)

        run_timestamp = int(time.time())
        video_path = VIDEO_PATH_TEMPLATE.format(ts=run_timestamp)
        depth_video_path = DEPTH_VIDEO_PATH_TEMPLATE.format(ts=run_timestamp)
        writer_thread = threading.Thread(
            target=disk_writer_worker,
            args=(write_queue, video_path, (w, h)),
            daemon=True,
        )
        writer_thread.start()
        depth_writer_thread = threading.Thread(
            target=disk_writer_worker,
            args=(depth_write_queue, depth_video_path, (w, h)),
            daemon=True,
        )
        depth_writer_thread.start()

        while stop_event is None or not stop_event.is_set():
            if frame_ready_event is not None:
                frame_ready_event.wait(timeout=0.1)
                frame_ready_event.clear()

            if frame_lock is None:
                current_frame_id = int(shm_meta[0])
                timestamp_ms = int(shm_meta[1])
                roll, pitch, yaw = shm_imu.tolist()
                np.copyto(bgra_buf, shm_rgb)
                np.copyto(depth_buf, shm_depth)
                np.copyto(depth_vision_bgra_buf, shm_depth_vision)
            else:
                with frame_lock:
                    current_frame_id = int(shm_meta[0])
                    timestamp_ms = int(shm_meta[1])
                    roll, pitch, yaw = shm_imu.tolist()
                    np.copyto(bgra_buf, shm_rgb)
                    np.copyto(depth_buf, shm_depth)
                    np.copyto(depth_vision_bgra_buf, shm_depth_vision)

            if current_frame_id == 0 or current_frame_id == last_frame_id:
                continue
            last_frame_id = current_frame_id

            # Reuse output buffers via dst= to avoid new allocations every frame
            cv2.cvtColor(bgra_buf, cv2.COLOR_BGRA2BGR, dst=frame_bgr_buf)
            cv2.cvtColor(
                depth_vision_bgra_buf,
                cv2.COLOR_BGRA2BGR,
                dst=depth_vision_bgr_buf,
            )
            cv2.resize(
                depth_buf, (0, 0), dst=downsampled_depth_buf,
                fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA,
            )
            now_record_time_ms = int(time.monotonic() * 1000)
            should_record = (
                last_record_time_ms is None
                or now_record_time_ms - last_record_time_ms >= record_interval_ms
            )

            if should_record:
                try:
                    processed_frame = annotate_frame(
                        frame_bgr_buf, detection_node.latest(current_frame_id)
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
                with shared_state.frame_lock:
                    shared_state.latest_frame = processed_frame.copy()

                shared_state.frame_event.set()

                try:
                    write_queue.put_nowait(processed_frame)
                except queue.Full:
                    dropped_frames += 1
                    now = time.monotonic()
                    if now - last_drop_log > 1.0:  # rate-limit logging, don't block hot path
                        logger.warning(
                            "RGB disk writer is lagging; dropped frames: %d",
                            dropped_frames,
                        )
                        last_drop_log = now

                try:
                    depth_write_queue.put_nowait(depth_record_frame)
                except queue.Full:
                    dropped_depth_frames += 1
                    now = time.monotonic()
                    if now - last_drop_log > 1.0:
                        logger.warning(
                            "Depth disk writer is lagging; dropped frames: %d",
                            dropped_depth_frames,
                        )
                        last_drop_log = now

                last_record_time_ms = now_record_time_ms

            # --- minimize time spent holding locks: just pointer/scalar assignment ---
            with shared_state.data_lock:
                shared_state.latest_depth_array = downsampled_depth_buf.copy()
                shared_state.latest_imu = {"roll": roll, "pitch": pitch, "yaw": yaw}
                shared_state.latest_timestamp = timestamp_ms

            shared_state.data_event.set()
            frame_index += 1
    finally:
        print("System shutting down, writing remaining data to disk...")
        if writer_thread is not None:
            write_queue.put(None)
            writer_thread.join()
        if depth_writer_thread is not None:
            depth_write_queue.put(None)
            depth_writer_thread.join()

        shm_rgb = None
        shm_depth = None
        shm_depth_vision = None
        shm_meta = None
        shm_imu = None

        for shm in (rgb_shm, depth_shm, depth_vision_shm, meta_shm, imu_shm):
            if shm is not None:
                shm.close()

        if detection_node is not None:
            detection_node.destroy_node()
        if owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()
        if detection_spin_thread is not None:
            detection_spin_thread.join(timeout=2.0)
