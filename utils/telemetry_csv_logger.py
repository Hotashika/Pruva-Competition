"""Vehicle telemetry CSV logger that can be imported by any runtime module.

Example:

    from utils.telemetry_csv_logger import TelemetryCsvLogger, TelemetrySample

    logger = TelemetryCsvLogger("vehicle_telemetry.csv", sample_rate_hz=1.0)
    logger.start(
        lambda: TelemetrySample(
            latitude_deg=vehicle.latitude,
            longitude_deg=vehicle.longitude,
            ground_speed_m_s=vehicle.ground_speed,
            roll_deg=vehicle.roll,
            pitch_deg=vehicle.pitch,
            heading_deg=vehicle.heading,
            speed_setpoint_m_s=controller.speed_setpoint,
            heading_setpoint_deg=controller.heading_setpoint,
        )
    )

    # Call logger.close() during application shutdown.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


CSV_HEADER = (
    "timestamp_utc",
    "latitude_deg",
    "longitude_deg",
    "ground_speed_m_s",
    "roll_deg",
    "pitch_deg",
    "heading_deg",
    "speed_setpoint_m_s",
    "heading_setpoint_deg",
)


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One vehicle telemetry sample; angles are in degrees."""

    latitude_deg: float
    longitude_deg: float
    ground_speed_m_s: float
    roll_deg: float
    pitch_deg: float
    heading_deg: float
    speed_setpoint_m_s: float
    heading_setpoint_deg: float
    timestamp_utc: Optional[datetime] = None


class TelemetryCsvLogger:
    """Write vehicle telemetry to CSV manually or at a fixed background rate.

    ``sample_rate_hz`` cannot be lower than 1 Hz. In background mode, ``provider``
    is called immediately and then at the configured rate. It must return either
    a :class:`TelemetrySample` or a mapping with the same field names.
    """

    def __init__(
        self,
        csv_path: os.PathLike[str] | str,
        *,
        sample_rate_hz: float = 1.0,
        append: bool = True,
    ) -> None:
        sample_rate_hz = float(sample_rate_hz)
        if not math.isfinite(sample_rate_hz) or sample_rate_hz < 1.0:
            raise ValueError("sample_rate_hz must be a finite value of at least 1.0")

        self.csv_path = Path(csv_path)
        self.sample_rate_hz = sample_rate_hz
        self._append = bool(append)
        self._period_sec = 1.0 / sample_rate_hz
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._output = None
        self._writer: Optional[csv.writer] = None
        self._failure: Optional[BaseException] = None
        self._closed = False

    @property
    def is_running(self) -> bool:
        """Return whether the background sampling thread is active."""

        return self._thread is not None and self._thread.is_alive()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("telemetry logger is closed")
        if self._output is not None:
            return

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_has_content = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        if self._append and file_has_content:
            with self.csv_path.open("r", encoding="utf-8", newline="") as existing:
                existing_header = tuple(next(csv.reader(existing), ()))
            if existing_header != CSV_HEADER:
                raise ValueError(
                    f"existing CSV header does not match the required header: "
                    f"{self.csv_path}"
                )

        mode = "a" if self._append else "w"
        self._output = self.csv_path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.writer(self._output)
        if not (self._append and file_has_content):
            self._writer.writerow(CSV_HEADER)
            self._output.flush()

    @staticmethod
    def _validated_sample(sample: TelemetrySample) -> TelemetrySample:
        numeric_values = {
            "latitude_deg": sample.latitude_deg,
            "longitude_deg": sample.longitude_deg,
            "ground_speed_m_s": sample.ground_speed_m_s,
            "roll_deg": sample.roll_deg,
            "pitch_deg": sample.pitch_deg,
            "heading_deg": sample.heading_deg,
            "speed_setpoint_m_s": sample.speed_setpoint_m_s,
            "heading_setpoint_deg": sample.heading_setpoint_deg,
        }

        try:
            values = {name: float(value) for name, value in numeric_values.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("all telemetry values must be numeric") from exc

        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("all telemetry values must be finite")
        if not -90.0 <= values["latitude_deg"] <= 90.0:
            raise ValueError("latitude_deg must be in the range [-90, 90]")
        if not -180.0 <= values["longitude_deg"] <= 180.0:
            raise ValueError("longitude_deg must be in the range [-180, 180]")
        if values["ground_speed_m_s"] < 0.0:
            raise ValueError("ground_speed_m_s cannot be negative")

        timestamp = sample.timestamp_utc
        if timestamp is not None:
            if not isinstance(timestamp, datetime):
                raise ValueError("timestamp_utc must be a datetime or None")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("timestamp_utc must include timezone information")
            timestamp = timestamp.astimezone(timezone.utc)

        return TelemetrySample(
            latitude_deg=values["latitude_deg"],
            longitude_deg=values["longitude_deg"],
            ground_speed_m_s=values["ground_speed_m_s"],
            roll_deg=values["roll_deg"],
            pitch_deg=values["pitch_deg"],
            heading_deg=values["heading_deg"] % 360.0,
            speed_setpoint_m_s=values["speed_setpoint_m_s"],
            heading_setpoint_deg=values["heading_setpoint_deg"] % 360.0,
            timestamp_utc=timestamp,
        )

    @staticmethod
    def _coerce_sample(
        sample: TelemetrySample | Mapping[str, Any],
    ) -> TelemetrySample:
        if isinstance(sample, TelemetrySample):
            return sample
        if isinstance(sample, Mapping):
            try:
                return TelemetrySample(**dict(sample))
            except TypeError as exc:
                raise ValueError(
                    "provider mapping does not match TelemetrySample fields"
                ) from exc
        raise TypeError("sample must be TelemetrySample or a compatible mapping")

    def write(
        self,
        sample: TelemetrySample | Mapping[str, Any],
    ) -> None:
        """Validate and immediately append one telemetry row."""

        validated = self._validated_sample(self._coerce_sample(sample))
        timestamp = validated.timestamp_utc or datetime.now(timezone.utc)

        row = (
            timestamp.isoformat(timespec="milliseconds"),
            f"{validated.latitude_deg:.8f}",
            f"{validated.longitude_deg:.8f}",
            f"{validated.ground_speed_m_s:.3f}",
            f"{validated.roll_deg:.3f}",
            f"{validated.pitch_deg:.3f}",
            f"{validated.heading_deg:.3f}",
            f"{validated.speed_setpoint_m_s:.3f}",
            f"{validated.heading_setpoint_deg:.3f}",
        )

        with self._lock:
            self._ensure_open()
            if self._writer is None or self._output is None:
                raise RuntimeError("telemetry CSV could not be opened")
            self._writer.writerow(row)
            self._output.flush()

    def record(
        self,
        *,
        latitude_deg: float,
        longitude_deg: float,
        ground_speed_m_s: float,
        roll_deg: float,
        pitch_deg: float,
        heading_deg: float,
        speed_setpoint_m_s: float,
        heading_setpoint_deg: float,
        timestamp_utc: Optional[datetime] = None,
    ) -> None:
        """Convenience method for callers that do not need TelemetrySample."""

        self.write(
            TelemetrySample(
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
                ground_speed_m_s=ground_speed_m_s,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                heading_deg=heading_deg,
                speed_setpoint_m_s=speed_setpoint_m_s,
                heading_setpoint_deg=heading_setpoint_deg,
                timestamp_utc=timestamp_utc,
            )
        )

    def start(
        self,
        provider: Callable[[], TelemetrySample | Mapping[str, Any]],
    ) -> None:
        """Start automatic CSV recording at ``sample_rate_hz``."""

        if not callable(provider):
            raise TypeError("provider must be callable")

        with self._lock:
            if self._closed:
                raise RuntimeError("telemetry logger is closed")
            if self.is_running:
                raise RuntimeError("telemetry logger is already running")
            self._ensure_open()
            self._failure = None
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(provider,),
                name="telemetry-csv-logger",
                daemon=True,
            )
            self._thread.start()

    def _run(
        self,
        provider: Callable[[], TelemetrySample | Mapping[str, Any]],
    ) -> None:
        next_sample_time = time.monotonic()

        try:
            while not self._stop_event.is_set():
                self.write(provider())
                next_sample_time += self._period_sec
                now = time.monotonic()
                if next_sample_time <= now:
                    next_sample_time = now + self._period_sec
                self._stop_event.wait(next_sample_time - now)
        except BaseException as exc:
            with self._lock:
                self._failure = exc
            self._stop_event.set()

    def raise_if_failed(self) -> None:
        """Raise if the background provider or writer has failed."""

        with self._lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("background telemetry recording failed") from failure

    def stop(self, *, raise_on_error: bool = True) -> None:
        """Stop background sampling, if active."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._thread = None
        if raise_on_error:
            self.raise_if_failed()

    def close(self) -> None:
        """Stop sampling and close the CSV file."""

        failure = None
        try:
            self.stop()
        except RuntimeError as exc:
            failure = exc

        with self._lock:
            if self._output is not None:
                self._output.close()
                self._output = None
                self._writer = None
            self._closed = True

        if failure is not None:
            raise failure

    def __enter__(self) -> TelemetryCsvLogger:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except RuntimeError:
                pass
