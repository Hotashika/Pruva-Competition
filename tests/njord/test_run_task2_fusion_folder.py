import tempfile
import unittest
from pathlib import Path

import numpy as np

from njord.scripts.run_task2_fusion_folder import (
    _paired_inputs,
    _working_depth,
    _working_shape,
    build_parser,
)


class Task2FusionFolderTests(unittest.TestCase):
    def test_pairs_only_images_and_depth_without_reading_existing_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "depth").mkdir()
            (root / "masks").mkdir()
            (root / "images" / "00000001.jpg").write_bytes(b"image")
            (root / "depth" / "00000001.npy").write_bytes(b"depth")
            (root / "masks" / "different_name.png").write_bytes(
                b"old mask must be ignored"
            )

            pairs, pairing = _paired_inputs(root)

            self.assertEqual(1, len(pairs))
            self.assertEqual("00000001", pairs[0][0])
            self.assertEqual(3, len(pairs[0]))
            self.assertEqual([], pairing["image_only"])
            self.assertEqual([], pairing["depth_only"])
            self.assertNotIn("mask_count", pairing)

    def test_pairs_left_camera_directory_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "left").mkdir()
            (root / "depth").mkdir()
            (root / "left" / "00000001.jpg").write_bytes(b"image")
            (root / "depth" / "00000001.npy").write_bytes(b"depth")

            pairs, _pairing = _paired_inputs(
                root,
                image_directory="left",
            )

            self.assertEqual("00000001", pairs[0][0])
            self.assertEqual(root / "left" / "00000001.jpg", pairs[0][1])

    def test_default_camera_height_is_njord_mount_height(self):
        args = build_parser().parse_args(
            [
                "dataset",
                "--output-dir",
                "output",
                "--model",
                "model.onnx",
            ]
        )

        self.assertEqual(0.25, args.camera_height_m)

    def test_default_intrinsics_match_512_by_288_working_geometry(self):
        args = build_parser().parse_args(
            [
                "dataset",
                "--output-dir",
                "output",
                "--model",
                "model.onnx",
            ]
        )

        self.assertEqual(280.0, args.fx)
        self.assertEqual(280.0, args.fy)
        self.assertEqual(256.0, args.cx)
        self.assertEqual(144.0, args.cy)

    def test_working_geometry_preserves_rgb_aspect_ratio(self):
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        depth = np.zeros((384, 512), dtype=np.float32)

        shape = _working_shape(rgb, depth)
        resized_depth = _working_depth(depth, shape)

        self.assertEqual((288, 512), shape)
        self.assertEqual((288, 512), resized_depth.shape)
        self.assertEqual(np.float32, resized_depth.dtype)


if __name__ == "__main__":
    unittest.main()
