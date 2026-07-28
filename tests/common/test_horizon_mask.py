import math
import unittest

import numpy as np

from vision.horizon_mask import create_horizon_mask, horizon_rows


class HorizonMaskTests(unittest.TestCase):
    def setUp(self):
        self.width = 320
        self.height = 180
        self.fx = 220.0
        self.fy = 220.0
        self.cx = 160.0
        self.cy = 90.0

    def _mask(self, roll=0.0, pitch=0.0, **kwargs):
        return create_horizon_mask(
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            roll,
            pitch,
            **kwargs,
        )

    def test_level_camera_splits_sky_and_water_at_principal_point(self):
        mask = self._mask()

        self.assertEqual((self.height, self.width), mask.shape)
        self.assertEqual(np.uint8, mask.dtype)
        self.assertTrue(np.all(mask[: int(self.cy) + 1] == 0))
        self.assertTrue(np.all(mask[int(self.cy) + 1 :] == 1))

    def test_roll_tilts_horizon(self):
        rows = horizon_rows(self._mask(roll=math.radians(12.0)))

        self.assertGreater(abs(int(rows[20]) - int(rows[-20])), 20)

    def test_pitch_moves_horizon(self):
        level_row = horizon_rows(self._mask())[self.width // 2]
        pitched_row = horizon_rows(
            self._mask(pitch=math.radians(8.0))
        )[self.width // 2]

        self.assertNotEqual(int(level_row), int(pitched_row))

    def test_rejects_invalid_focal_length(self):
        with self.assertRaises(ValueError):
            create_horizon_mask(
                self.width,
                self.height,
                0.0,
                self.fy,
                self.cx,
                self.cy,
                0.0,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
