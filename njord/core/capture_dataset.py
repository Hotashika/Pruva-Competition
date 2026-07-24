from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from njord.core.dataset_recorder import DatasetRecorder


class CaptureDatasetSession:
    """Sample synchronized capture frames into a named manual collection run."""

    def __init__(
        self,
        output_root: os.PathLike[str] | str,
        *,
        collection_name: str,
        calibration: Mapping[str, Any],
        record_fps: float = 5.0,
        record_right: bool = True,
        queue_size: int = 64,
    ):
        normalized_name = (
            "" if collection_name is None else str(collection_name).strip().lower()
        )
        if (
            not normalized_name
            or normalized_name in (".", "..")
            or Path(normalized_name).name != normalized_name
            or "/" in normalized_name
            or "\\" in normalized_name
        ):
            raise ValueError(
                "collection_name must be a single non-empty directory name"
            )

        record_fps = float(record_fps)
        if not math.isfinite(record_fps) or record_fps <= 0.0:
            raise ValueError("record_fps must be a positive finite number")

        calibration_payload = dict(calibration)
        calibration_payload["dataset_capture"] = {
            "collection_name": normalized_name,
            "record_fps": record_fps,
            "record_right": bool(record_right),
            "record_depth": True,
        }

        self.collection_name = normalized_name
        self.record_fps = record_fps
        self.record_interval_ms = max(1, int(round(1000.0 / record_fps)))
        self.last_record_timestamp_ms: Optional[int] = None
        self.recorder = DatasetRecorder(
            Path(output_root) / normalized_name,
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
        depth_map: np.ndarray,
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
            depth_map=depth_map,
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
