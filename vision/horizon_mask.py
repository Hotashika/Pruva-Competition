"""IMU-assisted horizon priors for maritime semantic segmentation."""

from __future__ import annotations

import math

import numpy as np


# ZED optical axes: +X right, +Y down, +Z forward.
# Vehicle/body axes follow ROS REP-103: +X forward, +Y left, +Z up.
_BODY_FROM_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


def create_horizon_mask(
    width,
    height,
    fx,
    fy,
    cx,
    cy,
    roll,
    pitch,
    *,
    roll_offset=0.0,
    pitch_offset=0.0,
    flip_roll=False,
    flip_pitch=False,
    invert=False,
) -> np.ndarray:
    """Return WaSR's binary IMU prior: zero above and one below the horizon.

    Angles are radians in ROS REP-103 body axes. Yaw is intentionally omitted
    because rotation around world-up does not move the ideal pinhole horizon.
    """

    width = int(width)
    height = int(height)
    values = np.asarray(
        [fx, fy, cx, cy, roll, pitch, roll_offset, pitch_offset],
        dtype=np.float64,
    )
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Camera intrinsics and IMU angles must be finite.")
    if float(fx) <= 0.0 or float(fy) <= 0.0:
        raise ValueError("Camera focal lengths must be positive.")

    roll = float(roll) * (-1.0 if flip_roll else 1.0) + float(roll_offset)
    pitch = float(pitch) * (-1.0 if flip_pitch else 1.0) + float(pitch_offset)

    world_from_body = _rotation_y(pitch) @ _rotation_x(roll)
    world_up_body = world_from_body.T @ np.array([0.0, 0.0, 1.0])
    camera_up = _BODY_FROM_CAMERA.T @ world_up_body

    yy, xx = np.indices((height, width), dtype=np.float64)
    normalized_x = (xx - float(cx)) / float(fx)
    normalized_y = (yy - float(cy)) / float(fy)
    side = (
        camera_up[0] * normalized_x
        + camera_up[1] * normalized_y
        + camera_up[2]
    )

    mask = (side < 0.0).astype(np.uint8)
    return 1 - mask if invert else mask


def horizon_rows(mask: np.ndarray) -> np.ndarray:
    """Return the first water-side row in every column, or -1 if absent."""

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Horizon mask must be a two-dimensional array.")

    water = mask > 0
    has_water = water.any(axis=0)
    rows = np.full(mask.shape[1], -1, dtype=np.int32)
    rows[has_water] = np.argmax(water[:, has_water], axis=0)
    return rows
