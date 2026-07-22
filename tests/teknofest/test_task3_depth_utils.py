"""Task 3 ZED duba mesafesi için donanımsız sayısal testler."""

import unittest
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from teknofest.vision.depth_utils import get_distance_from_bbox


class Task3DepthUtilsTests(unittest.TestCase):
    def test_central_median_rejects_background_around_buoy(self):
        depth = np.full((100, 100), 50.0, dtype=np.float32)
        depth[20:80, 20:80] = 4.0

        distance = get_distance_from_bbox(
            depth,
            [0, 0, 100, 100],
            method="central_median",
        )

        self.assertAlmostEqual(distance, 4.0)

    def test_invalid_or_insufficient_depth_is_rejected(self):
        invalid = np.zeros((20, 20), dtype=np.float32)
        invalid[0, 0] = np.nan
        self.assertEqual(
            get_distance_from_bbox(invalid, [0, 0, 20, 20], "central_median"),
            -1.0,
        )

        sparse = np.zeros((10, 10), dtype=np.float32)
        sparse[3:5, 3:5] = 3.0
        self.assertEqual(
            get_distance_from_bbox(sparse, [0, 0, 10, 10], "median"),
            -1.0,
        )

    def test_bbox_outside_image_or_empty_is_rejected(self):
        depth = np.full((10, 10), 2.0, dtype=np.float32)
        self.assertEqual(get_distance_from_bbox(depth, [20, 20, 30, 30]), -1.0)
        self.assertEqual(get_distance_from_bbox(depth, [5, 5, 5, 8]), -1.0)


if __name__ == "__main__":
    unittest.main()
