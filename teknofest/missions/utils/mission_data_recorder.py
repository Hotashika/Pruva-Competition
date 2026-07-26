"""Shared mission-scoped data recording used by every TEKNOFEST task."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from std_msgs.msg import String

from teknofest.core.imu_csv_writer import (
    MISSION_RECORDING_TOPIC,
    normalize_session_name,
)
from utils.telemetry_csv_logger import TelemetryCsvLogger, TelemetrySample


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class MissionDataRecorder:
    """Record vehicle telemetry and control the ZED IMU recorder lifecycle."""

    def __init__(self, node, session_name):
        self.node = node
        self.session_name = normalize_session_name(session_name)
        if not self.session_name:
            raise ValueError("session_name must contain a valid label")

        self.latest_telemetry_sample = None
        self.telemetry_logger = None
        self.telemetry_csv_path = None
        self.recording_active = False

        self.telemetry_sub = node.create_subscription(
            String,
            "/cube/telemetry",
            self._telemetry_callback,
            10,
        )
        self.recording_state_pub = node.create_publisher(
            String,
            MISSION_RECORDING_TOPIC,
            10,
        )
        self.recording_state_timer = node.create_timer(
            1.0,
            self._publish_recording_state,
        )
        self._publish_recording_state()

    def _telemetry_callback(self, message):
        try:
            payload = json.loads(message.data)
            sample = TelemetrySample(
                latitude_deg=float(payload["latitude_deg"]),
                longitude_deg=float(payload["longitude_deg"]),
                ground_speed_m_s=float(payload["ground_speed_m_s"]),
                roll_deg=float(payload["roll_deg"]),
                pitch_deg=float(payload["pitch_deg"]),
                yaw_deg=float(payload["yaw_deg"]),
                heading_deg=float(payload["heading_deg"]),
                speed_setpoint_m_s=float(payload["speed_setpoint_m_s"]),
                heading_setpoint_deg=float(payload["heading_setpoint_deg"]),
            )
            values = (
                sample.latitude_deg,
                sample.longitude_deg,
                sample.ground_speed_m_s,
                sample.roll_deg,
                sample.pitch_deg,
                sample.yaw_deg,
                sample.heading_deg,
                sample.speed_setpoint_m_s,
                sample.heading_setpoint_deg,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("telemetry contains a non-finite value")
            self.latest_telemetry_sample = sample
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.node.get_logger().warn(
                f"Geçersiz /cube/telemetry mesajı yok sayıldı: {exc}",
                throttle_duration_sec=2.0,
            )

    def _publish_recording_state(self):
        message = String()
        message.data = self.session_name if self.recording_active else ""
        self.recording_state_pub.publish(message)

    def wait_for_complete_telemetry(self, timeout_sec=10.0):
        """Wait until every mandatory vehicle CSV field has arrived."""

        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            if self.latest_telemetry_sample is not None:
                return True
            self.node.get_logger().info(
                "CSV kaydı için hız ve yönelim telemetrisi bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return False

    def start(self):
        """Start vehicle CSV and signal the ZED IMU writer."""

        if self.telemetry_logger is not None:
            return
        if self.latest_telemetry_sample is None:
            raise RuntimeError("tam telemetri alınmadan veri kaydı başlatılamaz")

        output_directory = Path(
            os.getenv(
                "TEKNOFEST_TELEMETRY_DIRECTORY",
                str(REPOSITORY_ROOT / "teknofest" / "logs" / "telemetry"),
            )
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.telemetry_csv_path = (
            output_directory
            / f"vehicle_telemetry_{self.session_name}_{timestamp}.csv"
        )
        logger = TelemetryCsvLogger(
            self.telemetry_csv_path,
            sample_rate_hz=1.0,
            append=False,
        )
        logger.start(lambda: self.latest_telemetry_sample)
        self.telemetry_logger = logger
        self.recording_active = True
        self._publish_recording_state()
        self.node.get_logger().info(
            "Görev veri kaydı başladı: "
            f"telemetri={self.telemetry_csv_path}, "
            f"ZED IMU oturumu={self.session_name}"
        )

    def stop(self):
        """Stop both recorders; safe to call repeatedly."""

        self.recording_active = False
        self._publish_recording_state()

        logger = self.telemetry_logger
        if logger is None:
            return
        self.telemetry_logger = None

        close_error = None
        try:
            if self.latest_telemetry_sample is not None:
                logger.write(self.latest_telemetry_sample)
        except Exception as exc:  # noqa: BLE001 - closing must still continue
            close_error = exc

        try:
            logger.close()
        except Exception as exc:  # noqa: BLE001 - closing must still continue
            if close_error is None:
                close_error = exc

        if close_error is None:
            self.node.get_logger().info(
                f"Görev veri kaydı kapatıldı: {self.telemetry_csv_path}"
            )
        else:
            self.node.get_logger().error(
                f"Görev veri kaydı kapatma hatası: {close_error}"
            )
