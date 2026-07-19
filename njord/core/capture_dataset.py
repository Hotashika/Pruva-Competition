from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from njord.core.dataset_recorder import DatasetRecorder


class ActiveTaskRecordingGate:
    """Keep a process-safe recording event active while a task heartbeat exists."""

    def __init__(
        self,
        record_event: Any,
        *,
        task_name: str = "task2",
        timeout_sec: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
    ):
        normalized_task = "" if task_name is None else str(task_name).strip().lower()
        if not normalized_task:
            raise ValueError("task_name must be non-empty")

        timeout_sec = float(timeout_sec)
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be a positive finite number")

        self.task_name = normalized_task
        self.timeout_sec = timeout_sec
        self._record_event = record_event
        self._clock = clock
        self._last_heartbeat: Optional[float] = None
        self._lock = threading.Lock()
        self._record_event.clear()

    def observe(self, active_task: str, *, now: Optional[float] = None) -> bool:
        """Apply an active-task heartbeat and return the resulting gate state."""

        normalized_task = (
            "" if active_task is None else str(active_task).strip().lower()
        )
        observed_at = self._clock() if now is None else float(now)
        with self._lock:
            if normalized_task == self.task_name:
                self._last_heartbeat = observed_at
                self._record_event.set()
                return True

            self._last_heartbeat = None
            self._record_event.clear()
            return False

    def expire(self, *, now: Optional[float] = None) -> bool:
        """Clear a stale task heartbeat; return True only when it expires."""

        checked_at = self._clock() if now is None else float(now)
        with self._lock:
            if self._last_heartbeat is None:
                return False
            if checked_at - self._last_heartbeat < self.timeout_sec:
                return False

            self._last_heartbeat = None
            self._record_event.clear()
            return True

    def close(self) -> None:
        with self._lock:
            self._last_heartbeat = None
            self._record_event.clear()


class CaptureDatasetSession:
    """Sample synchronized capture frames into a task-specific dataset run."""

    def __init__(
        self,
        output_root: os.PathLike[str] | str,
        *,
        task_name: str,
        calibration: Mapping[str, Any],
        record_fps: float = 5.0,
        record_right: bool = True,
        queue_size: int = 64,
    ):
        normalized_task = "" if task_name is None else str(task_name).strip().lower()
        if (
            not normalized_task
            or normalized_task in (".", "..")
            or Path(normalized_task).name != normalized_task
            or "/" in normalized_task
            or "\\" in normalized_task
        ):
            raise ValueError("task_name must be a single non-empty directory name")

        record_fps = float(record_fps)
        if not math.isfinite(record_fps) or record_fps <= 0.0:
            raise ValueError("record_fps must be a positive finite number")

        calibration_payload = dict(calibration)
        calibration_payload["dataset_capture"] = {
            "task_name": normalized_task,
            "record_fps": record_fps,
            "record_right": bool(record_right),
        }

        self.task_name = normalized_task
        self.record_fps = record_fps
        self.record_interval_ms = max(1, int(round(1000.0 / record_fps)))
        self.last_record_timestamp_ms: Optional[int] = None
        self.recorder = DatasetRecorder(
            Path(output_root) / normalized_task,
            calibration=calibration_payload,
            record_right=record_right,
            queue_size=queue_size,
        )

    @property
    def run_dir(self) -> Path:
        return self.recorder.run_dir

    def record_frame(
        self,
        *,
        frame_id: int,
        camera_timestamp_ms: int,
        left_image: np.ndarray,
        right_image: Optional[np.ndarray],
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Optional[bool]:
        """Record a due frame.

        ``None`` means the frame was intentionally skipped by the sampling
        interval. A boolean is the underlying recorder's queue result.
        """

        camera_timestamp_ms = int(camera_timestamp_ms)
        if not self.frame_is_due(camera_timestamp_ms):
            return None

        # Advance the sampling clock even if the bounded writer queue drops
        # this frame, otherwise a slow disk would make every camera frame retry.
        self.last_record_timestamp_ms = camera_timestamp_ms
        return self.recorder.record_frame(
            frame_id=frame_id,
            camera_timestamp_ms=camera_timestamp_ms,
            left_image=left_image,
            right_image=right_image,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

    def frame_is_due(self, camera_timestamp_ms: int) -> bool:
        camera_timestamp_ms = int(camera_timestamp_ms)
        if (
            self.last_record_timestamp_ms is not None
            and camera_timestamp_ms <= self.last_record_timestamp_ms
        ):
            raise ValueError("camera_timestamp_ms must be strictly increasing")
        if (
            self.last_record_timestamp_ms is not None
            and camera_timestamp_ms - self.last_record_timestamp_ms
            < self.record_interval_ms
        ):
            return False
        return True

    def close(self, timeout: float = 30.0) -> None:
        self.recorder.close(timeout=timeout)

    def __enter__(self) -> "CaptureDatasetSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
