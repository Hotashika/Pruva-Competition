import csv
import json
import tempfile
import unittest

import numpy as np

from njord.core.capture_dataset import CaptureDatasetSession


def image(value):
    return np.full((24, 32, 4), value, dtype=np.uint8)


def depth(value):
    return np.full((24, 32), value, dtype=np.float32)


class CaptureDatasetSessionTests(unittest.TestCase):
    def test_records_synchronized_stereo_and_imu_at_configured_sample_rate(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            session = CaptureDatasetSession(
                temporary_dir,
                collection_name="manual",
                calibration={"left": {"fx": 700.0}, "right": {"fx": 700.0}},
                record_fps=5.0,
            )
            self.assertTrue(
                session.record_frame(
                    frame_id=1,
                    camera_timestamp_ms=1000,
                    left_image=image(10),
                    right_image=image(20),
                    depth_map=depth(1.0),
                    roll=0.1,
                    pitch=0.2,
                    yaw=0.3,
                )
            )
            self.assertIsNone(
                session.record_frame(
                    frame_id=2,
                    camera_timestamp_ms=1100,
                    left_image=image(30),
                    right_image=image(40),
                    depth_map=depth(2.0),
                    roll=0.4,
                    pitch=0.5,
                    yaw=0.6,
                )
            )
            self.assertTrue(
                session.record_frame(
                    frame_id=3,
                    camera_timestamp_ms=1200,
                    left_image=image(50),
                    right_image=image(60),
                    depth_map=depth(3.0),
                    roll=0.7,
                    pitch=0.8,
                    yaw=0.9,
                )
            )
            run_dir = session.run_dir
            session.close()

            self.assertEqual("manual", run_dir.parent.name)
            self.assertTrue((run_dir / "left" / "00000001.jpg").is_file())
            self.assertTrue((run_dir / "right" / "00000001.jpg").is_file())
            self.assertTrue((run_dir / "depth" / "00000001.npy").is_file())
            self.assertFalse((run_dir / "left" / "00000002.jpg").exists())
            self.assertFalse((run_dir / "depth" / "00000002.npy").exists())
            self.assertTrue((run_dir / "left" / "00000003.jpg").is_file())
            self.assertTrue((run_dir / "right" / "00000003.jpg").is_file())
            self.assertTrue((run_dir / "depth" / "00000003.npy").is_file())
            np.testing.assert_array_equal(
                depth(1.0),
                np.load(run_dir / "depth" / "00000001.npy"),
            )
            np.testing.assert_array_equal(
                depth(3.0),
                np.load(run_dir / "depth" / "00000003.npy"),
            )

            with (run_dir / "metadata.csv").open(
                newline="", encoding="utf-8"
            ) as metadata_file:
                rows = list(csv.DictReader(metadata_file))
            self.assertEqual(["1", "3"], [row["frame_id"] for row in rows])
            self.assertEqual(
                ["1000", "1200"],
                [row["camera_timestamp_ms"] for row in rows],
            )
            self.assertEqual("0.100000000", rows[0]["roll_rad"])
            self.assertEqual("0.800000000", rows[1]["pitch_rad"])
            self.assertEqual("0.900000000", rows[1]["yaw_rad"])
            self.assertTrue(all(row["system_timestamp_utc"] for row in rows))
            self.assertEqual("left/00000001.jpg", rows[0]["left_file"])
            self.assertEqual("right/00000001.jpg", rows[0]["right_file"])
            self.assertEqual("depth/00000001.npy", rows[0]["depth_file"])

            with (run_dir / "calibration.yaml").open(encoding="utf-8") as file:
                calibration = json.load(file)["camera"]
            self.assertEqual(
                "manual",
                calibration["dataset_capture"]["collection_name"],
            )
            self.assertEqual(5.0, calibration["dataset_capture"]["record_fps"])

            with (run_dir / "manifest.json").open(encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual("closed", manifest["status"])
            self.assertEqual(2, manifest["frames_written"])

    def test_rejects_invalid_collection_name_and_record_rate(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            with self.assertRaisesRegex(ValueError, "collection_name"):
                CaptureDatasetSession(
                    temporary_dir,
                    collection_name="../manual",
                    calibration={},
                )
            with self.assertRaisesRegex(ValueError, "record_fps"):
                CaptureDatasetSession(
                    temporary_dir,
                    collection_name="manual",
                    calibration={},
                    record_fps=0.0,
                )
if __name__ == "__main__":
    unittest.main()
