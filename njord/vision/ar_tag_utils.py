"""Dependency-light helpers for Task 3 QR localization and tracking."""

import cv2
import numpy as np


def expanded_bbox(bbox, image_shape, margin_ratio=0.18):
    """Expand an xyxy bbox and clamp it to the image."""
    image_h, image_w = image_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    margin_x = int(max(x2 - x1, 1) * margin_ratio)
    margin_y = int(max(y2 - y1, 1) * margin_ratio)
    return [
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_w, x2 + margin_x),
        min(image_h, y2 + margin_y),
    ]


def normalize_qr_payload(payload):
    normalized = str(payload or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized.replace("middle_birth_", "middle_berth_")


def bbox_to_qr_detection(bbox, payload, confidence):
    x1, y1, x2, y2 = map(int, bbox)
    return {
        "payload": payload,
        "canonical_payload": payload,
        "confidence": float(confidence),
        "center_px": {"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0},
        "bbox_xywh_px": {
            "x": float(x1),
            "y": float(y1),
            "width": float(max(0, x2 - x1)),
            "height": float(max(0, y2 - y1)),
        },
    }


def track_template(image, previous_bbox, template, threshold=0.45):
    """Return (bbox, score) for a local template match, or (None, score)."""
    if template is None or template.size == 0:
        return None, 0.0
    sx1, sy1, sx2, sy2 = expanded_bbox(previous_bbox, image.shape, margin_ratio=0.60)
    search = image[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return None, 0.0
    search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    th, tw = template.shape[:2]
    if search_gray.shape[0] < th or search_gray.shape[1] < tw:
        return None, 0.0
    scores = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_location = cv2.minMaxLoc(scores)
    if not np.isfinite(max_score) or max_score < threshold:
        return None, float(max_score) if np.isfinite(max_score) else 0.0
    x1 = sx1 + max_location[0]
    y1 = sy1 + max_location[1]
    return [x1, y1, x1 + tw, y1 + th], float(max_score)
