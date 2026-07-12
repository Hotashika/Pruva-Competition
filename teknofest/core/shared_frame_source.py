import time

import cv2
import numpy as np

from teknofest.config.camera_config import DEPTH_SHAPE, RGB_SHAPE
from teknofest.core import shared_state
from teknofest.core.shared_memory_utils import attach_existing_shared_memory


def attach_shared_memory(name, retries=50, delay=0.1):
    return attach_existing_shared_memory(name, retries=retries, delay=delay)


def open_or_start_capture_source(retries=5, delay=0.1):
    try:
        return SharedFrameSource(retries=retries, delay=delay), None, None
    except RuntimeError:
        from teknofest.main import start_capture_process

        capture_process, frame_lock, frame_ready_event, stop_event, _, _ = start_capture_process()
        source = SharedFrameSource(frame_lock, frame_ready_event)
        return source, capture_process, stop_event


def close_capture_source(source, capture_process=None, stop_event=None):
    if source is not None:
        source.close()

    if stop_event is not None:
        stop_event.set()

    if capture_process is not None:
        capture_process.join(timeout=3)
        if capture_process.is_alive():
            capture_process.terminate()
            capture_process.join(timeout=2)


class SharedFrameSource:
    def __init__(self, frame_lock=None, frame_ready_event=None, retries=50, delay=0.1):
        self.frame_lock = frame_lock
        self.frame_ready_event = frame_ready_event
        self.last_frame_id = 0
        self.rgb_shm = None
        self.depth_shm = None
        self.meta_shm = None
        self.imu_shm = None
        self.calib_shm = None

        try:
            self.rgb_shm = attach_shared_memory(shared_state.RGB_SHM_NAME, retries, delay)
            self.depth_shm = attach_shared_memory(shared_state.DEPTH_SHM_NAME, retries, delay)
            self.meta_shm = attach_shared_memory(shared_state.META_SHM_NAME, retries, delay)
            self.imu_shm = attach_shared_memory(shared_state.IMU_SHM_NAME, retries, delay)
            self.calib_shm = attach_shared_memory(shared_state.CALIB_SHM_NAME, retries, delay)

            self.rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=self.rgb_shm.buf)
            self.depth = np.ndarray(DEPTH_SHAPE, dtype=np.float32, buffer=self.depth_shm.buf)
            self.meta = np.ndarray(shared_state.META_SHAPE, dtype=np.int64, buffer=self.meta_shm.buf)
            self.imu = np.ndarray(shared_state.IMU_SHAPE, dtype=np.float64, buffer=self.imu_shm.buf)
            self.calib = np.ndarray(shared_state.CALIB_SHAPE, dtype=np.float64, buffer=self.calib_shm.buf)

            self.bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)
            self.frame_bgr_buf = np.empty((RGB_SHAPE[0], RGB_SHAPE[1], 3), dtype=np.uint8)
            self.depth_buf = np.empty(DEPTH_SHAPE, dtype=np.float32)
        except Exception:
            self.close()
            raise

    def get_calibration(self):
        fx, cx = self.calib.tolist()
        return fx, cx

    def read(self, timeout=1.0):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self.frame_ready_event is not None:
                remaining = max(0.0, min(0.1, deadline - time.monotonic()))
                self.frame_ready_event.wait(timeout=remaining)
                self.frame_ready_event.clear()

            frame = self._copy_latest_if_new()
            if frame is not None:
                return frame

            time.sleep(0.005)

        raise TimeoutError("No new shared ZED frame received.")

    def _copy_latest_if_new(self):
        if self.frame_lock is None:
            return self._copy_without_lock()

        with self.frame_lock:
            frame_id = int(self.meta[0])
            if frame_id == 0 or frame_id == self.last_frame_id:
                return None

            timestamp_ms = int(self.meta[1])
            pitch, yaw, roll = self.imu.tolist()
            np.copyto(self.bgra_buf, self.rgb)
            np.copyto(self.depth_buf, self.depth)

        return self._build_frame(frame_id, timestamp_ms, pitch, yaw, roll)

    def _copy_without_lock(self):
        frame_id = int(self.meta[0])
        if frame_id == 0 or frame_id == self.last_frame_id:
            return None

        timestamp_ms = int(self.meta[1])
        pitch, yaw, roll = self.imu.tolist()
        np.copyto(self.bgra_buf, self.rgb)
        np.copyto(self.depth_buf, self.depth)

        if int(self.meta[0]) != frame_id:
            return None

        return self._build_frame(frame_id, timestamp_ms, pitch, yaw, roll)

    def _build_frame(self, frame_id, timestamp_ms, pitch, yaw, roll):
        self.last_frame_id = frame_id
        cv2.cvtColor(self.bgra_buf, cv2.COLOR_BGRA2BGR, dst=self.frame_bgr_buf)
        return {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "frame_bgr": self.frame_bgr_buf.copy(),
            "depth": self.depth_buf.copy(),
            "imu": {"pitch": pitch, "yaw": yaw, "roll": roll},
        }

    def close(self):
        self.rgb = None
        self.depth = None
        self.meta = None
        self.imu = None
        self.calib = None

        for attr in ("rgb_shm", "depth_shm", "meta_shm", "imu_shm", "calib_shm"):
            shm = getattr(self, attr)
            if shm is not None:
                shm.close()
                setattr(self, attr, None)
