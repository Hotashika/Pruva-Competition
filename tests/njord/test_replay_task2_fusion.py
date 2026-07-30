import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from njord.scripts.replay_task2_fusion import run_replay


class FakeReplayDetector:
    def __init__(self):
        self.last_diagnostics = {}

    def detect(self, image, depth, imu=None, now=None):
        self.last_diagnostics = {
            "segmentation_ready": True,
            "fused_count": 1,
        }
        return [
            {
                "type": "fused_obstacle",
                "class": "surface_obstacle_candidate",
                "distance": float(np.median(depth)),
                "angle": 0.0,
            }
        ]


class Task2FusionReplayTests(unittest.TestCase):
    def test_replays_synchronized_rgb_depth_and_imu(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            run_dir = Path(temporary) / "run"
            (run_dir / "left").mkdir(parents=True)
            (run_dir / "depth").mkdir()
            Image.fromarray(
                np.full((12, 16, 3), 50, dtype=np.uint8),
                mode="RGB",
            ).save(run_dir / "left" / "00000001.jpg")
            np.save(
                run_dir / "depth" / "00000001.npy",
                np.full((12, 16), 3.25, dtype=np.float32),
            )
            (run_dir / "calibration.yaml").write_text(
                json.dumps(
                    {
                        "camera": {
                            "left": {
                                "fx": 100.0,
                                "fy": 100.0,
                                "cx": 8.0,
                                "cy": 6.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "metadata_file": "metadata.csv",
                        "imu_file": None,
                    }
                ),
                encoding="utf-8",
            )
            with (run_dir / "metadata.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=(
                        "frame_id",
                        "camera_timestamp_ms",
                        "roll_rad",
                        "pitch_rad",
                        "yaw_rad",
                        "left_file",
                        "depth_file",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "frame_id": 1,
                        "camera_timestamp_ms": 1000,
                        "roll_rad": 0.1,
                        "pitch_rad": 0.2,
                        "yaw_rad": 0.3,
                        "left_file": "left/00000001.jpg",
                        "depth_file": "depth/00000001.npy",
                    }
                )

            summary, output_path = run_replay(
                run_dir,
                detector=FakeReplayDetector(),
            )

            self.assertEqual(1, summary["frame_count"])
            self.assertEqual(1, summary["segmentation_ready_frames"])
            self.assertEqual(
                1,
                summary["detection_type_counts"]["fused_obstacle"],
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                3.25,
                payload["frames"][0]["detections"][0]["distance"],
            )
            self.assertEqual(0.1, payload["frames"][0]["imu_rad"]["roll"])


if __name__ == "__main__":
    unittest.main()
