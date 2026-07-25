import math


def nearest_bbox_median_distance(detections):
    """Return the nearest valid bbox-median distance already in detections."""
    nearest = None
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        try:
            distance = float(detection.get("distance"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance) and distance > 0.0:
            nearest = distance if nearest is None else min(nearest, distance)

    return nearest
