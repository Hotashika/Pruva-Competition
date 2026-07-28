import argparse
import unittest

from njord.collect_dataset import (
    DEFAULT_OUTPUT_ROOT,
    _quaternion_to_euler,
    build_parser,
    collection_name,
    non_negative_float,
    positive_float,
)


class CollectDatasetCommandTests(unittest.TestCase):
    def test_defaults_create_an_open_ended_manual_collection(self):
        arguments = build_parser().parse_args([])

        self.assertEqual("manual", arguments.name)
        self.assertEqual(5.0, arguments.fps)
        self.assertEqual(0.0, arguments.duration)
        self.assertEqual(DEFAULT_OUTPUT_ROOT, arguments.output_dir)
        self.assertFalse(arguments.attach)

    def test_accepts_named_fixed_duration_collection(self):
        arguments = build_parser().parse_args(
            ["--name", "Pool_Test", "--fps", "2.5", "--duration", "10"]
        )

        self.assertEqual("pool_test", arguments.name)
        self.assertEqual(2.5, arguments.fps)
        self.assertEqual(10.0, arguments.duration)

    def test_accepts_live_attach_mode(self):
        arguments = build_parser().parse_args(
            ["--attach", "--name", "Task2_Live", "--fps", "5"]
        )

        self.assertTrue(arguments.attach)
        self.assertEqual("task2_live", arguments.name)

    def test_converts_identity_quaternion_to_zero_euler_angles(self):
        roll, pitch, yaw = _quaternion_to_euler(0.0, 0.0, 0.0, 1.0)

        self.assertAlmostEqual(0.0, roll)
        self.assertAlmostEqual(0.0, pitch)
        self.assertAlmostEqual(0.0, yaw)

    def test_rejects_unsafe_names_and_invalid_numbers(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            collection_name("../task2")
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_float("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            non_negative_float("-1")


if __name__ == "__main__":
    unittest.main()
