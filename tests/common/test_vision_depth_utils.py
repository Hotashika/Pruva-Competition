import numpy as np

from vision.depth_utils import get_distance_from_bbox


def test_median_excludes_non_finite_and_non_positive_depths():
    depth = np.array(
        [
            [np.nan, np.inf, -np.inf],
            [0.0, -2.0, 2.0],
            [3.0, 4.0, np.nan],
        ],
        dtype=np.float32,
    )

    assert get_distance_from_bbox(depth, [0, 0, 3, 3]) == 3.0


def test_mean_uses_only_valid_depths():
    depth = np.array(
        [
            [1.0, 2.0, np.inf],
            [3.0, 0.0, np.nan],
        ],
        dtype=np.float32,
    )

    assert get_distance_from_bbox(depth, [0, 0, 3, 2], method="mean") == 2.0


def test_region_without_valid_depth_returns_minus_one():
    depth = np.array(
        [
            [np.nan, np.inf],
            [-np.inf, 0.0],
        ],
        dtype=np.float32,
    )

    assert get_distance_from_bbox(depth, [0, 0, 2, 2]) == -1.0


def test_bbox_is_clamped_before_filtering_depths():
    depth = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    assert get_distance_from_bbox(depth, [-10, -10, 1, 2]) == 2.0
