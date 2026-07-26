import csv
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from teknofest.core.imu_csv_writer import (
    IMU_CSV_HEADER,
    MissionRecordingState,
    ZedImuCsvWriter,
)
from utils.telemetry_csv_logger import (
    CSV_HEADER,
    TelemetryCsvLogger,
    TelemetrySample,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DataRecordingTests(unittest.TestCase):
    def test_zed_imu_writer_records_frame_aligned_roll_pitch_yaw(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
            writer = ZedImuCsvWriter(
                "Task 1",
                output_directory=directory,
            )
            writer.write(
                frame_id=42,
                camera_timestamp_ms=123456,
                roll_rad=0.1,
                pitch_rad=-0.2,
                yaw_rad=1.5,
                timestamp_utc=datetime(
                    2026,
                    7,
                    26,
                    9,
                    30,
                    tzinfo=timezone.utc,
                ),
            )
            csv_path = writer.csv_path
            writer.close()

            with csv_path.open(encoding="utf-8", newline="") as input_file:
                rows = list(csv.reader(input_file))

            self.assertEqual(list(IMU_CSV_HEADER), rows[0])
            self.assertEqual(
                [
                    "2026-07-26T09:30:00.000+00:00",
                    "42",
                    "123456",
                    "0.100000000",
                    "-0.200000000",
                    "1.500000000",
                ],
                rows[1],
            )
            self.assertTrue(csv_path.name.startswith("zed_imu_task_1_"))

    def test_recording_state_expires_when_mission_heartbeat_stops(self):
        state = MissionRecordingState(timeout_sec=3.0)

        state.update("Competition", now=10.0)

        self.assertEqual("competition", state.active_session(now=12.9))
        self.assertEqual("", state.active_session(now=13.1))

    def test_vehicle_telemetry_csv_contains_pixhawk_yaw(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
            csv_path = Path(directory) / "vehicle.csv"
            logger = TelemetryCsvLogger(csv_path, append=False)
            logger.write(
                TelemetrySample(
                    latitude_deg=37.95,
                    longitude_deg=32.50,
                    ground_speed_m_s=1.2,
                    roll_deg=-2.0,
                    pitch_deg=3.0,
                    yaw_deg=271.5,
                    heading_deg=270.0,
                    speed_setpoint_m_s=1.0,
                    heading_setpoint_deg=269.0,
                )
            )
            logger.close()

            with csv_path.open(encoding="utf-8", newline="") as input_file:
                rows = list(csv.DictReader(input_file))

            self.assertIn("yaw_deg", CSV_HEADER)
            self.assertEqual("-2.000", rows[0]["roll_deg"])
            self.assertEqual("3.000", rows[0]["pitch_deg"])
            self.assertEqual("271.500", rows[0]["yaw_deg"])

    def test_shared_recorder_drives_vehicle_and_zed_for_any_task(self):
        rclpy = types.ModuleType("rclpy")
        rclpy.ok = lambda: True
        rclpy.spin_once = lambda *_args, **_kwargs: None

        class String:
            def __init__(self):
                self.data = ""

        std_msgs = types.ModuleType("std_msgs")
        std_msgs.__path__ = []
        std_msgs_msg = types.ModuleType("std_msgs.msg")
        std_msgs_msg.String = String

        module_name = "teknofest.missions.utils.mission_data_recorder"
        fake_modules = {
            "rclpy": rclpy,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs_msg,
        }

        with mock.patch.dict(sys.modules, fake_modules):
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
            publishers = []
            subscriptions = {}

            class Publisher:
                def __init__(self):
                    self.messages = []

                def publish(self, message):
                    self.messages.append(message.data)

            class Logger:
                def info(self, *_args, **_kwargs):
                    pass

                def warn(self, *_args, **_kwargs):
                    pass

                def error(self, *_args, **_kwargs):
                    pass

            class Node:
                def create_subscription(self, _type, topic, callback, _qos):
                    subscriptions[topic] = callback
                    return object()

                def create_publisher(self, _type, _topic, _qos):
                    publisher = Publisher()
                    publishers.append(publisher)
                    return publisher

                def create_timer(self, _period, callback):
                    return callback

                def get_logger(self):
                    return Logger()

            created_loggers = []

            class FakeTelemetryLogger:
                def __init__(self, csv_path, **_kwargs):
                    self.csv_path = Path(csv_path)
                    self.samples = []
                    self.closed = False
                    created_loggers.append(self)

                def start(self, provider):
                    self.samples.append(provider())

                def write(self, sample):
                    self.samples.append(sample)

                def close(self):
                    self.closed = True

            with tempfile.TemporaryDirectory(
                dir=REPOSITORY_ROOT
            ) as directory, mock.patch.object(
                module,
                "TelemetryCsvLogger",
                FakeTelemetryLogger,
            ), mock.patch.dict(
                os.environ,
                {"TEKNOFEST_TELEMETRY_DIRECTORY": directory},
            ):
                recorder = module.MissionDataRecorder(Node(), "Task 2")
                subscriptions["/cube/telemetry"](
                    types.SimpleNamespace(
                        data=json.dumps(
                            {
                                "latitude_deg": 37.95,
                                "longitude_deg": 32.50,
                                "ground_speed_m_s": 1.2,
                                "roll_deg": -2.0,
                                "pitch_deg": 3.0,
                                "yaw_deg": 271.5,
                                "heading_deg": 270.0,
                                "speed_setpoint_m_s": 1.0,
                                "heading_setpoint_deg": 269.0,
                            }
                        )
                    )
                )

                recorder.start()
                recorder.stop()

            self.assertEqual(["", "task_2", ""], publishers[0].messages)
            self.assertEqual(1, len(created_loggers))
            self.assertEqual(271.5, created_loggers[0].samples[0].yaw_deg)
            self.assertTrue(created_loggers[0].closed)
            self.assertIn(
                "vehicle_telemetry_task_2_",
                created_loggers[0].csv_path.name,
            )
            sys.modules.pop(module_name, None)

    def test_every_teknofest_mission_starts_and_stops_recording(self):
        relative_paths = (
            "teknofest/missions/task1_point_tracking.py",
            (
                "teknofest/missions/"
                "task2_point_tracking_task_in_an_environment_with_obstacle.py"
            ),
            "teknofest/missions/task3_kamikaze_engagement.py",
            "teknofest/missions/competition_mission.py",
        )

        for relative_path in relative_paths:
            with self.subTest(relative_path=relative_path):
                source = (REPOSITORY_ROOT / relative_path).read_text(
                    encoding="utf-8"
                )
                self.assertIn("start_telemetry_recording()", source)
                self.assertIn("stop_telemetry_recording()", source)

    def test_teknofest_capture_uses_roll_pitch_yaw_order_and_radians(self):
        source = (
            REPOSITORY_ROOT / "teknofest" / "core" / "capture_proc.py"
        ).read_text(encoding="utf-8")
        frame_source = (
            REPOSITORY_ROOT
            / "teknofest"
            / "core"
            / "shared_frame_source.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "roll, pitch, yaw = imu_pose.get_euler_angles(radian=True)",
            source,
        )
        self.assertIn("imu_buf[:] = (roll, pitch, yaw)", source)
        self.assertIn("roll, pitch, yaw = self.imu.tolist()", frame_source)

    def test_bridge_publishes_pixhawk_yaw_for_vehicle_csv(self):
        source = (
            REPOSITORY_ROOT / "bridge" / "bridge_node.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '"yaw_deg": math.degrees(float(self.yaw)) % 360.0',
            source,
        )


if __name__ == "__main__":
    unittest.main()
