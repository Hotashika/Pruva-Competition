import numpy as np


def get_distance_from_bbox(depth_array, bbox, method="median"):
    """
    Return the representative depth inside an ``xyxy`` bounding box.

    NaN, infinite, zero, and negative depth samples are excluded from the
    calculation. Invalid inputs and regions without a valid sample return
    ``-1.0``.
    """
    if depth_array is None or bbox is None:
        return -1.0

    x1, y1, x2, y2 = map(int, bbox)
    h, w = depth_array.shape

    x1_c, x2_c = max(0, x1), min(w, x2)
    y1_c, y2_c = max(0, y1), min(h, y2)

    if y2_c <= y1_c or x2_c <= x1_c:
        return -1.0

    roi_depth = depth_array[y1_c:y2_c, x1_c:x2_c]
    valid_depth = roi_depth[
        np.isfinite(roi_depth) & (roi_depth > 0.0)
    ]
    if valid_depth.size == 0:
        return -1.0

    if method == "mean":
        distance = float(np.mean(valid_depth))
    else:
        distance = float(np.median(valid_depth))

    if not np.isfinite(distance):
        return -1.0

    return distance


def is_valid_distance(distance, min_dist=0.3, max_dist=20.0):
    if distance <= 0.0:
        return False
    return min_dist <= distance <= max_dist
