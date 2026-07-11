import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config.camera_config import CAMERA_WIDTH
from config.vision_config import DEVICE, BUOY_MODEL_PATH, VESSEL_MODEL_PATH, TOLERANCE_RATIO, TOLARANCE_DEG
from vision.depth_utils import get_distance_from_bbox


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
