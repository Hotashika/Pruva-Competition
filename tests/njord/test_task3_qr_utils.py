import cv2
import numpy as np

from njord.vision.ar_tag_utils import (
    bbox_to_qr_detection,
    expanded_bbox,
    normalize_qr_payload,
    track_template,
)


def test_expanded_bbox_clamps_to_image():
    assert expanded_bbox([2, 3, 22, 13], (20, 30, 3), 0.5) == [0, 0, 30, 18]


def test_qr_payload_and_message_schema_match_task3():
    payload = normalize_qr_payload(" Middle-Birth-1 ")
    detection = bbox_to_qr_detection([10, 20, 50, 80], payload, 0.9)
    assert payload == "middle_berth_1"
    assert detection["center_px"] == {"x": 30.0, "y": 50.0}
    assert detection["bbox_xywh_px"]["width"] == 40.0


def test_template_tracking_follows_translated_tag():
    template_bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.line(template_bgr, (1, 1), (14, 14), (255, 255, 255), 2)
    cv2.circle(template_bgr, (11, 5), 3, (180, 180, 180), -1)
    template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[32:48, 43:59] = template_bgr

    bbox, score = track_template(frame, [35, 28, 51, 44], template)

    assert bbox == [43, 32, 59, 48]
    assert score > 0.99
