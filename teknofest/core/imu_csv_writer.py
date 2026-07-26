"""Mission-scoped ZED orientation CSV recording for TEKNOFEST."""

from __future__ import annotations

import csv
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MISSION_RECORDING_TOPIC = "/mission/recording_state"
MISSION_RECORDING_HEARTBEAT_TIMEOUT_SEC = 3.0

IMU_CSV_HEADER = (
    "timestamp_utc",
    "frame_id",
    "camera_timestamp_ms",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
)


def normalize_session_name(value: object) -> str:
    """Return a filesystem-safe mission session label or an empty stop state."""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9_-]+", "_", text).strip("_")


class MissionRecordingState:
    """Thread-safe, heartbeat-aware recording state shared with the writer loop."""

    def __init__(self, timeout_sec=MISSION_RECORDING_HEARTBEAT_TIMEOUT_SEC):
        self.timeout_sec = float(timeout_sec)
        self._lock = threading.Lock()
        self._session_name = ""
        self._updated_at = None

    def update(self, session_name, *, now=None):
        normalized_name = normalize_session_name(session_name)
        update_time = time.monotonic() if now is None else float(now)
        with self._lock:
            self._session_name = normalized_name
            self._updated_at = update_time

    def active_session(self, *, now=None):
        check_time = time.monotonic() if now is None else float(now)
        with self._lock:
            if not self._session_name or self._updated_at is None:
                return ""
            if check_time - self._updated_at > self.timeout_sec:
                return ""
            return self._session_name


class ZedImuCsvWriter:
    """Write frame-aligned ZED roll, pitch, and yaw samples to one CSV."""

    def __init__(
        self,
        session_name,
        *,
        output_directory: Optional[os.PathLike[str] | str] = None,
    ):
        normalized_name = normalize_session_name(session_name)
        if not normalized_name:
            raise ValueError("session_name must contain a valid label")

        if output_directory is None:
            output_directory = os.getenv(
                "TEKNOFEST_IMU_DIRECTORY",
                str(REPOSITORY_ROOT / "teknofest" / "logs" / "imu"),
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.csv_path = (
            Path(output_directory)
            / f"zed_imu_{normalized_name}_{timestamp}.csv"
        )
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self.csv_path.open("x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._output)
        self._writer.writerow(IMU_CSV_HEADER)
        self._output.flush()
        self._closed = False

    def write(
        self,
        *,
        frame_id,
        camera_timestamp_ms,
        roll_rad,
        pitch_rad,
        yaw_rad,
        timestamp_utc=None,
    ):
        if self._closed:
            raise RuntimeError("ZED IMU CSV writer is closed")

        orientation = tuple(
            float(value) for value in (roll_rad, pitch_rad, yaw_rad)
        )
        if not all(math.isfinite(value) for value in orientation):
            raise ValueError("ZED orientation values must be finite")

        timestamp = timestamp_utc or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp_utc must include timezone information")
        timestamp = timestamp.astimezone(timezone.utc)

        self._writer.writerow(
            (
                timestamp.isoformat(timespec="milliseconds"),
                int(frame_id),
                int(camera_timestamp_ms),
                f"{orientation[0]:.9f}",
                f"{orientation[1]:.9f}",
                f"{orientation[2]:.9f}",
            )
        )
        self._output.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._output.close()
