import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from njord.config.camera_config import CAMERA_WIDTH
from njord.config.vision_config import DEVICE, BUOY_MODEL_PATH, VESSEL_MODEL_PATH, TOLERANCE_RATIO, TOLARANCE_DEG
from njord.vision.depth_utils import get_distance_from_bbox


class BaseYOLODetector:
    def __init__(self, model_path, device=DEVICE):
        model_p = Path(model_path)
        if not model_p.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            model_p = project_root / model_path
        self.model = YOLO(str(model_p))
        self.device = device
        self.class_names = self.model.names

    def detect(self, bgr_image, depth_array):
        t0 = time.time()
        results = self.model(bgr_image, device=self.device, verbose=False)
        t1 = time.time()

        boxes = results[0].boxes
        detections = []

        if len(boxes) > 0:
            xyxy_all = boxes.xyxy.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy()
            conf_all = boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, xyxy_all[i])
                cls_id = int(cls_all[i])
                conf = float(conf_all[i])
                class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")
                bbox = [x1, y1, x2, y2]
                distance = get_distance_from_bbox(depth_array, bbox, method="median")
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "distance": round(distance, 2),
                    "bbox": bbox
                })
        t2 = time.time()

        return detections

    def draw_detections(self, bgr_image, detections):
        output_frame = bgr_image.copy()
        image_h, image_w = output_frame.shape[:2]

        for detection in detections:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, image_w - 1))
            y1 = max(0, min(y1, image_h - 1))
            x2 = max(0, min(x2, image_w - 1))
            y2 = max(0, min(y2, image_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = detection.get("class", detection.get("type", "unknown"))
            confidence = detection.get("confidence")
            distance = detection.get("distance")
            track_id = detection.get("track_id")

            label_parts = [str(class_name)]

            if confidence is not None:
                try:
                    label_parts.append(f"{float(confidence):.2f}")
                except (TypeError, ValueError):
                    pass

            if distance is not None:
                try:
                    distance_value = float(distance)
                except (TypeError, ValueError):
                    distance_value = float("nan")

                if np.isfinite(distance_value):
                    label_parts.append(f"{distance_value:.2f} m")

            angle = None
            side = None
            for key, value in detection.items():
                if key.endswith(" angle: "):
                    angle = value
                elif key.endswith(" side: "):
                    side = value

            if angle is not None:
                try:
                    label_parts.append(f"{float(angle):.1f} deg")
                except (TypeError, ValueError):
                    pass

            if side is not None:
                label_parts.append(str(side))

            if track_id is not None:
                label_parts.append(f"ID:{track_id}")

            label = " | ".join(label_parts)

            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )

            text_y = max(y1 - 8, text_height + 8)

            cv2.rectangle(
                output_frame,
                (x1, text_y - text_height - 6),
                (x1 + text_width + 6, text_y + baseline),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                output_frame,
                label,
                (x1 + 3, text_y - 3),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return output_frame


def _normalize_intrinsics(fx, cx):
    return fx, cx if cx is not None else CAMERA_WIDTH / 2


def _compute_angle_from_bbox(detection, fx, cx):
    if fx is None:
        return None

    bbox = detection["bbox"]
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    angle_rad = np.arctan2(bbox_center_x - cx, fx)
    return float(np.degrees(angle_rad))


def _compute_side_from_bbox(detection, angle_deg):
    if angle_deg is not None:
        if abs(angle_deg) <= TOLARANCE_DEG:
            return "across"
        if angle_deg > 0:
            return "right"
        return "left"

    bbox = detection["bbox"]
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    image_center_x = CAMERA_WIDTH / 2
    tolerance_px = CAMERA_WIDTH * TOLERANCE_RATIO
    diff = bbox_center_x - image_center_x

    if abs(diff) <= tolerance_px:
        return "across"
    if diff > 0:
        return "right"
    return "left"


def _add_angle_fields(detections, label, fx, cx):
    angle_key = f"{label} angle: "
    side_key = f"{label} side: "

    for det in detections:
        angle_deg = _compute_angle_from_bbox(det, fx, cx)
        det[angle_key] = angle_deg
        det[side_key] = _compute_side_from_bbox(det, angle_deg)

    return detections


class BuoyDetector(BaseYOLODetector):
    def __init__(self, model_path=BUOY_MODEL_PATH, device=DEVICE, fx=None, cx=None):
        super().__init__(model_path, device)
        self.fx, self.cx = _normalize_intrinsics(fx, cx)

    def detect(self, bgr_image, depth_array):
        detections = super().detect(bgr_image, depth_array)
        return _add_angle_fields(detections, "Buoy", self.fx, self.cx)


class VesselDetector(BaseYOLODetector):
    def __init__(self, model_path=VESSEL_MODEL_PATH, device=DEVICE, fx=None, cx=None):
        super().__init__(model_path, device)
        # ZED kalibrasyonundan gelen intrinsics. Kamera açıldıktan sonra
        # zed.get_camera_information() ile okunup buraya geçirilmeli.
        # fx=None kalırsa side hesabı görüntü merkezi fallback'ini kullanır.
        self.fx, self.cx = _normalize_intrinsics(fx, cx)

    def detect(self, bgr_image, depth_array):
        detections = super().detect(bgr_image, depth_array)
        return _add_angle_fields(detections, "Vessel", self.fx, self.cx)
