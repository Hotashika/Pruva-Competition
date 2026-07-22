import numpy as np


def get_distance_from_bbox(depth_array, bbox, method="median"):
    """
    Input:
        depth_array (np.ndarray): ZED'den gelen 2 boyutlu derinlik matrisi (H, W).
        bbox (list or tuple): [x1, y1, x2, y2] formatında bounding box koordinatları.
        method (str): "median", "mean" veya duba için "central_median".

    Output:
        float: Hesaplanmış mesafe (metre cinsinden). Geçersiz/hatalı durumlarda -1.0 döner.
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

    if method == "central_median":
        # Kutunun kenarları çoğunlukla su/arka plan içerir. Duba gövdesinin
        # bulunma olasılığı yüksek olan merkez %60 bölgeyi kullan.
        roi_h, roi_w = roi_depth.shape
        margin_y = int(roi_h * 0.20)
        margin_x = int(roi_w * 0.20)
        central = roi_depth[
            margin_y:roi_h - margin_y if margin_y else roi_h,
            margin_x:roi_w - margin_x if margin_x else roi_w,
        ]
        if central.size:
            roi_depth = central

    valid_depth = roi_depth[np.isfinite(roi_depth) & (roi_depth > 0.0)]
    minimum_valid_pixels = max(5, int(roi_depth.size * 0.05))
    if valid_depth.size < minimum_valid_pixels:
        return -1.0

    if method in ("median", "central_median"):
        distance = float(np.median(valid_depth))
    elif method == "mean":
        distance = float(np.mean(valid_depth))
    else:
        distance = float(np.median(valid_depth))  # Fallback

    if not np.isfinite(distance):
        return -1.0

    return distance


def is_valid_distance(distance, min_dist=0.3, max_dist=20.0):
    if distance <= 0.0:
        return False
    return min_dist <= distance <= max_dist
