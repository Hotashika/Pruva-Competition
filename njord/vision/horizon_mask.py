import math

import numpy as np


# ZED/ROS camera mounting convention used by the Njord vehicle:
# camera optical X (right)   -> body -Y (right)
# camera optical Y (down)    -> body -Z (down)
# camera optical Z (forward) -> body +X (forward)
_BODY_FROM_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _rotation_x(angle):
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


def _rotation_y(angle):
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
):
    """Create WaSR's binary IMU horizon mask.

    Inputs use radians and ROS REP-103 body axes (X forward, Y left, Z up).
    Output is uint8 HxW: 0 for the sky side and 1 for the water side.
    Yaw is intentionally omitted because rotation around world-up does not move
    the horizon in an ideal pinhole camera.
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

    # Body orientation without yaw. World Z is up in REP-103.
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


def horizon_rows(mask):
    """Return the first water-side row for every column, or -1 if absent."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Horizon mask must be a two-dimensional array.")

    water = mask > 0
    has_water = water.any(axis=0)
    rows = np.full(mask.shape[1], -1, dtype=np.int32)
    rows[has_water] = np.argmax(water[:, has_water], axis=0)
    return rows


def render_horizon_overlay(
    frame_bgr,
    mask,
    *,
    water_alpha=0.28,
    water_color=(255, 0, 0),
    horizon_color=(0, 255, 0),
):
    """Overlay the water side in blue and the computed horizon in green."""
    frame = np.asarray(frame_bgr)
    mask = np.asarray(mask)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("BGR frame must have shape HxWx3.")
    if mask.shape != frame.shape[:2]:
        raise ValueError("Horizon mask and frame dimensions must match.")

    overlay = frame.copy()
    water_pixels = mask > 0
    if np.any(water_pixels):
        color = np.asarray(water_color, dtype=np.float32)
        blended = (
            (1.0 - float(water_alpha)) * overlay[water_pixels].astype(np.float32)
            + float(water_alpha) * color
        )
        overlay[water_pixels] = np.clip(blended, 0, 255).astype(np.uint8)

    rows = horizon_rows(mask)
    columns = np.flatnonzero(rows >= 0)
    if columns.size >= 2:
        color = np.asarray(horizon_color, dtype=np.uint8)
        for row_offset in (-1, 0, 1):
            line_rows = np.clip(rows[columns] + row_offset, 0, mask.shape[0] - 1)
            overlay[line_rows, columns] = color
    return overlay
