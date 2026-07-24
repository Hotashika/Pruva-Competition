import numpy as np


def get_distance_from_bbox(depth_array, bbox, method="median"):
    """
    Return the representative depth inside an ``xyxy`` bounding box.

    Invalid inputs and non-finite depth regions return ``-1.0``.
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

    if method == "mean":
        distance = float(np.nanmean(roi_depth))
    else:
        distance = float(np.nanmedian(roi_depth))

    if not np.isfinite(distance):
        return -1.0

    return distance


def is_valid_distance(distance, min_dist=0.3, max_dist=20.0):
    if distance <= 0.0:
        return False
    return min_dist <= distance <= max_dist
