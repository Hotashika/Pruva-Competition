import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from njord.core.dataset_recorder import (
    DatasetRecorder,
    DatasetRecorderError,
    Task2TestRecorder,
)


def image(value, shape=(24, 32, 3)):
    return np.full(shape, value, dtype=np.uint8)


def depth(value, shape=(24, 32)):
    return np.full(shape, value, dtype=np.float32)


class DatasetRecorderTests(unittest.TestCase):
    def test_task2_test_recorder_keeps_sensor_streams_in_separate_files(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            with Task2TestRecorder(
                temporary_dir,
                run_name="task2_test",
                calibration={"left": {"fx": 700.0}},
            ) as recorder:
                recorder.record_frame(
                    frame_id=10,
                    camera_timestamp_ms=5000,
                    system_timestamp_utc="2026-07-25T10:00:00+00:00",
                    left_image=image(50),
                    right_image=None,
                    depth_map=depth(3.5),
                    roll=0.1,
                    pitch=0.2,
                    yaw=0.3,
                )
                recorder.record_gps(
                    latitude_deg=41.0123,
                    longitude_deg=29.0456,
                    altitude_m=2.5,
                    ros_timestamp_ns=123456789,
                    frame_id=10,
                    camera_timestamp_ms=5000,
                    position_covariance_type=2,
                )
                recorder.record_imu_sample(
                    source="pixhawk",
                    ros_timestamp_ns=123456790,
                    roll_rad=0.4,
                    pitch_rad=0.5,
                    yaw_rad=0.6,
                    angular_velocity_x_rad_s=0.01,
                    linear_acceleration_x_m_s2=9.81,
                )
                recorder.record_kinematics(
                    {
                        "frame_id": 10,
                        "camera_timestamp_ms": 5000,
                        "detected": 1,
                        "track_id": 7,
                        "distance_m": 3.5,
                        "bearing_deg": -4.0,
                        "relative_speed_mps": 0.8,
                        "collision_risk": 1,
                        "collision_reason": "unsafe_cpa",
                    }
                )
                run_dir = recorder.run_dir

            with (run_dir / "frames.csv").open(
                newline="", encoding="utf-8"
            ) as frames_file:
                frame_rows = list(csv.DictReader(frames_file))
            self.assertEqual(1, len(frame_rows))
            self.assertEqual("10", frame_rows[0]["frame_id"])
            self.assertNotIn("roll_rad", frame_rows[0])
            self.assertEqual("depth/00000010.npy", frame_rows[0]["depth_file"])

            with (run_dir / "imu.csv").open(
                newline="", encoding="utf-8"
            ) as imu_file:
                imu_rows = list(csv.DictReader(imu_file))
            self.assertEqual(["pixhawk", "zed"], sorted(row["source"] for row in imu_rows))
            zed_row = next(row for row in imu_rows if row["source"] == "zed")
            self.assertEqual("10", zed_row["frame_id"])
            self.assertEqual("5000", zed_row["camera_timestamp_ms"])
            pixhawk_row = next(
                row for row in imu_rows if row["source"] == "pixhawk"
            )
            self.assertEqual("123456790", pixhawk_row["ros_timestamp_ns"])
            self.assertEqual("0.010000000", pixhawk_row["angular_velocity_x_rad_s"])

            with (run_dir / "gps.csv").open(
                newline="", encoding="utf-8"
            ) as gps_file:
                gps_rows = list(csv.DictReader(gps_file))
            self.assertEqual("41.012300000", gps_rows[0]["latitude_deg"])
            self.assertEqual("29.045600000", gps_rows[0]["longitude_deg"])
            self.assertEqual("10", gps_rows[0]["frame_id"])
            self.assertEqual("5000", gps_rows[0]["camera_timestamp_ms"])

            with (run_dir / "kinematics.csv").open(
                newline="", encoding="utf-8"
            ) as kinematics_file:
                kinematics_rows = list(csv.DictReader(kinematics_file))
            self.assertEqual("7", kinematics_rows[0]["track_id"])
            self.assertEqual("unsafe_cpa", kinematics_rows[0]["collision_reason"])

            with (run_dir / "manifest.json").open(encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual("frames.csv", manifest["metadata_file"])
            self.assertEqual("imu.csv", manifest["imu_file"])
            self.assertEqual("gps.csv", manifest["gps_file"])
            self.assertEqual("kinematics.csv", manifest["kinematics_file"])

    def test_writes_stereo_images_metadata_calibration_and_manifest(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            calibration = {
                "resolution": [1280, 720],
                "left": {"fx": 700.5, "fy": 701.0, "cx": 640.0, "cy": 360.0},
                "baseline_m": 0.12,
            }
            with DatasetRecorder(
                temporary_dir,
                run_name="test_run",
                calibration=calibration,
                jpeg_quality=95,
            ) as recorder:
                accepted = recorder.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=123456,
                    system_timestamp_utc="2026-07-17T12:00:00+00:00",
                    left_image=image(25),
                    right_image=image(200),
                    depth_map=depth(4.5),
                    roll=0.1,
                    pitch=-0.2,
                    yaw=1.5,
                )
                self.assertTrue(accepted)
                run_dir = recorder.run_dir

            left_path = run_dir / "left" / "00000001.jpg"
            right_path = run_dir / "right" / "00000001.jpg"
            depth_path = run_dir / "depth" / "00000001.npy"
            self.assertTrue(left_path.is_file())
            self.assertTrue(right_path.is_file())
            self.assertTrue(depth_path.is_file())
            np.testing.assert_array_equal(depth(4.5), np.load(depth_path))
            with Image.open(left_path) as saved_left:
                self.assertEqual((32, 24), saved_left.size)
                self.assertEqual("RGB", saved_left.mode)
            with Image.open(right_path) as saved_right:
                self.assertEqual((32, 24), saved_right.size)
                self.assertEqual("RGB", saved_right.mode)

            with (run_dir / "metadata.csv").open(
                newline="", encoding="utf-8"
            ) as metadata_file:
                rows = list(csv.DictReader(metadata_file))
            self.assertEqual(1, len(rows))
            self.assertEqual("1", rows[0]["frame_id"])
            self.assertEqual("123456", rows[0]["camera_timestamp_ms"])
            self.assertEqual("0.100000000", rows[0]["roll_rad"])
            self.assertEqual("-0.200000000", rows[0]["pitch_rad"])
            self.assertEqual("1.500000000", rows[0]["yaw_rad"])
            self.assertEqual(
                "2026-07-17T12:00:00+00:00",
                rows[0]["system_timestamp_utc"],
            )
            self.assertNotIn("linear_acceleration_x", rows[0])
            self.assertNotIn("angular_velocity_x", rows[0])
            self.assertEqual("left/00000001.jpg", rows[0]["left_file"])
            self.assertEqual("right/00000001.jpg", rows[0]["right_file"])
            self.assertEqual("depth/00000001.npy", rows[0]["depth_file"])

            with (run_dir / "calibration.yaml").open(encoding="utf-8") as file:
                saved_calibration = json.load(file)
            self.assertTrue(saved_calibration["provided"])
            self.assertEqual(calibration, saved_calibration["camera"])

            with (run_dir / "manifest.json").open(encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual("closed", manifest["status"])
            self.assertEqual(1, manifest["frames_seen"])
            self.assertEqual(1, manifest["frames_accepted"])
            self.assertEqual(1, manifest["frames_written"])
            self.assertEqual(0, manifest["frames_dropped"])
            self.assertEqual(0, manifest["frames_failed"])
            self.assertEqual(0, manifest["writer_error_count"])
            self.assertEqual("depth", manifest["depth_directory"])
            self.assertIsNotNone(manifest["closed_utc"])

    def test_checkpoints_manifest_while_recording(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            recorder = DatasetRecorder(
                temporary_dir,
                run_name="manifest_checkpoint",
                record_right=False,
                manifest_interval_frames=1,
            )
            recorder.record_frame(
                frame_id=1,
                camera_timestamp_ms=1000,
                left_image=image(10),
                right_image=None,
                depth_map=depth(1.0),
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )

            try:
                deadline = time.monotonic() + 5.0
                while (
                    time.monotonic() < deadline
                    and recorder._manifest.get("frames_written") != 1
                ):
                    time.sleep(0.01)

                with recorder.manifest_path.open(encoding="utf-8") as file:
                    manifest = json.load(file)
                self.assertEqual("recording", manifest["status"])
                self.assertEqual(1, manifest["frames_written"])
                self.assertIsNone(manifest["closed_utc"])
                self.assertIsNotNone(manifest["updated_utc"])
            finally:
                recorder.close()

    def test_copies_reusable_capture_buffer_before_async_write(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            source = image(30)
            depth_source = depth(3.0)
            recorder = DatasetRecorder(
                temporary_dir,
                run_name="buffer_copy",
                record_right=False,
                jpeg_quality=100,
            )
            recorder.record_frame(
                frame_id=1,
                camera_timestamp_ms=1000,
                left_image=source,
                right_image=None,
                depth_map=depth_source,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )
            source[:] = 230
            depth_source[:] = 99.0
            recorder.close()

            with Image.open(
                recorder.run_dir / "left" / "00000001.jpg"
            ) as saved_image:
                saved = np.asarray(saved_image)
            self.assertAlmostEqual(30.0, float(saved.mean()), delta=3.0)
            np.testing.assert_array_equal(
                depth(3.0),
                np.load(recorder.run_dir / "depth" / "00000001.npy"),
            )

    def test_converts_bgr_capture_frame_to_rgb_jpeg(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            bgr_frame = np.zeros((24, 32, 3), dtype=np.uint8)
            bgr_frame[:] = (10, 40, 220)
            with DatasetRecorder(
                temporary_dir,
                run_name="bgr_conversion",
                record_right=False,
                jpeg_quality=100,
            ) as recorder:
                recorder.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=1000,
                    left_image=bgr_frame,
                    right_image=None,
                    depth_map=depth(1.0),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )

            with Image.open(
                recorder.run_dir / "left" / "00000001.jpg"
            ) as saved_image:
                red, green, blue = np.asarray(saved_image)[12, 16]
            self.assertAlmostEqual(220, int(red), delta=3)
            self.assertAlmostEqual(40, int(green), delta=3)
            self.assertAlmostEqual(10, int(blue), delta=3)

    def test_ignores_right_image_without_copy_when_right_recording_is_disabled(self):
        class RightImageMustNotBeRead:
            def __array__(self, *args, **kwargs):
                raise AssertionError("right image was converted")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            with DatasetRecorder(
                temporary_dir,
                run_name="ignore_right",
                record_right=False,
            ) as recorder:
                self.assertTrue(
                    recorder.record_frame(
                        frame_id=1,
                        camera_timestamp_ms=1000,
                        left_image=image(1),
                        right_image=RightImageMustNotBeRead(),
                        depth_map=depth(1.0),
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.0,
                    )
                )

    def test_bounded_queue_drops_frame_without_blocking_capture(self):
        writer_started = threading.Event()
        release_writer = threading.Event()

        def slow_writer(path, frame, quality):
            if not writer_started.is_set():
                writer_started.set()
                self.assertTrue(release_writer.wait(timeout=2.0))
            path.write_bytes(b"jpeg")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            recorder = DatasetRecorder(
                temporary_dir,
                run_name="queue_drop",
                record_right=False,
                queue_size=1,
                image_writer=slow_writer,
            )
            self.assertTrue(
                recorder.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=1000,
                    left_image=image(1),
                    right_image=None,
                    depth_map=depth(1.0),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )
            )
            self.assertTrue(writer_started.wait(timeout=2.0))
            self.assertTrue(
                recorder.record_frame(
                    frame_id=2,
                    camera_timestamp_ms=1001,
                    left_image=image(2),
                    right_image=None,
                    depth_map=depth(2.0),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )
            )
            self.assertFalse(
                recorder.record_frame(
                    frame_id=3,
                    camera_timestamp_ms=1002,
                    left_image=image(3),
                    right_image=None,
                    depth_map=depth(3.0),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )
            )
            release_writer.set()
            recorder.close()

            self.assertEqual(3, recorder.frames_seen)
            self.assertEqual(2, recorder.frames_accepted)
            self.assertEqual(2, recorder.frames_written)
            self.assertEqual(1, recorder.frames_dropped)

    def test_writer_failure_is_reported_in_manifest_and_on_close(self):
        def failing_writer(path, frame, quality):
            raise OSError("simulated disk failure")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            recorder = DatasetRecorder(
                temporary_dir,
                run_name="write_error",
                record_right=False,
                image_writer=failing_writer,
            )
            recorder.record_frame(
                frame_id=1,
                camera_timestamp_ms=1000,
                left_image=image(1),
                right_image=None,
                depth_map=depth(1.0),
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )

            with self.assertRaisesRegex(
                DatasetRecorderError,
                "simulated disk failure",
            ):
                recorder.close()

            with recorder.manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual("error", manifest["status"])
            self.assertEqual(1, manifest["writer_error_count"])
            self.assertEqual(0, manifest["frames_written"])
            self.assertEqual(1, manifest["frames_failed"])
            self.assertEqual(1, recorder.frames_failed)
            self.assertTrue(recorder._metadata_file.closed)

    def test_timeout_manifest_is_written_and_worker_eventually_cleans_up(self):
        writer_started = threading.Event()
        release_writer = threading.Event()

        def blocked_writer(path, frame, quality):
            writer_started.set()
            self.assertTrue(release_writer.wait(timeout=2.0))
            path.write_bytes(b"jpeg")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            recorder = DatasetRecorder(
                temporary_dir,
                run_name="close_timeout",
                record_right=False,
                image_writer=blocked_writer,
            )
            recorder.record_frame(
                frame_id=1,
                camera_timestamp_ms=1000,
                left_image=image(1),
                right_image=None,
                depth_map=depth(1.0),
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )
            self.assertTrue(writer_started.wait(timeout=2.0))

            with self.assertRaisesRegex(TimeoutError, "did not stop"):
                recorder.close(timeout=0.05)

            with recorder.manifest_path.open(encoding="utf-8") as file:
                timeout_manifest = json.load(file)
            self.assertEqual("timeout", timeout_manifest["status"])
            self.assertIsNotNone(timeout_manifest["last_close_timeout_utc"])

            release_writer.set()
            recorder._writer_thread.join(timeout=2.0)
            self.assertFalse(recorder._writer_thread.is_alive())
            self.assertTrue(recorder._metadata_file.closed)

            with recorder.manifest_path.open(encoding="utf-8") as file:
                final_manifest = json.load(file)
            self.assertEqual("closed", final_manifest["status"])
            self.assertIsNotNone(final_manifest["closed_utc"])
            self.assertIsNotNone(final_manifest["last_close_timeout_utc"])

    def test_rejects_non_monotonic_ids_timestamps_and_invalid_right_frame(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            recorder = DatasetRecorder(
                temporary_dir,
                run_name="validation",
                record_right=True,
            )
            with self.assertRaisesRegex(ValueError, "right_image"):
                recorder.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=1000,
                    left_image=image(1),
                    right_image=None,
                    depth_map=depth(1.0),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )

            with self.assertRaisesRegex(ValueError, "depth_map"):
                recorder.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=1000,
                    left_image=image(1),
                    right_image=image(2),
                    depth_map=None,
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )
            with self.assertRaisesRegex(ValueError, "dimensions"):
                recorder.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=1000,
                    left_image=image(1),
                    right_image=image(2),
                    depth_map=depth(1.0, shape=(12, 16)),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )

            recorder.record_frame(
                frame_id=1,
                camera_timestamp_ms=1000,
                left_image=image(1),
                right_image=image(2),
                depth_map=depth(1.0),
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )
            with self.assertRaisesRegex(ValueError, "frame_id"):
                recorder.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=1001,
                    left_image=image(1),
                    right_image=image(2),
                    depth_map=depth(1.0),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )
            with self.assertRaisesRegex(ValueError, "camera_timestamp_ms"):
                recorder.record_frame(
                    frame_id=2,
                    camera_timestamp_ms=1000,
                    left_image=image(1),
                    right_image=image(2),
                    depth_map=depth(1.0),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                )
            recorder.close()
            self.assertFalse(recorder._writer_thread.is_alive())
            self.assertTrue(recorder._metadata_file.closed)

    def test_refuses_to_overwrite_existing_run_directory(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            existing = Path(temporary_dir) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                DatasetRecorder(temporary_dir, run_name="existing")


if __name__ == "__main__":
    unittest.main()
