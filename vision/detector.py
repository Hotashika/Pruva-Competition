from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from vision.depth_utils import get_distance_from_bbox
from vision.render import draw_detections


class BaseYOLODetector:
    def __init__(
            self,
            model_path,
            device=None,
            use_tracking=False,
            tracker=None,
    ):
        model_p = Path(model_path)
        if not model_p.is_absolute():
            repository_root = Path(__file__).resolve().parents[1]
            model_p = repository_root / model_p

        self.model = YOLO(str(model_p))
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else "cpu"
            if device is None
            else device
        )
        self.class_names = self.model.names
        self.use_tracking = use_tracking
        self.tracker = tracker

    def detect(self, bgr_image, depth_array):
        if self.use_tracking:
            track_kwargs = {
                "device": self.device,
                "persist": True,
                "verbose": False,
            }
            if self.tracker is not None:
                track_kwargs["tracker"] = self.tracker
            results = self.model.track(bgr_image, **track_kwargs)
        else:
            results = self.model(
                bgr_image,
                device=self.device,
                verbose=False,
            )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        image_h, image_w = bgr_image.shape[:2]
        xyxy_all = boxes.xyxy.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()
        track_ids = None
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.int().cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, xyxy_all[i])
            x1 = max(0, min(x1, image_w - 1))
            y1 = max(0, min(y1, image_h - 1))
            x2 = max(0, min(x2, image_w - 1))
            y2 = max(0, min(y2, image_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(cls_all[i])
            confidence = float(conf_all[i])
            if hasattr(self.class_names, "get"):
                class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")
            elif 0 <= cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                class_name = f"unknown_{cls_id}"
            bbox = [x1, y1, x2, y2]
            try:
                distance = get_distance_from_bbox(
                    depth_array,
                    bbox,
                    method="median",
                )
            except (AttributeError, TypeError, ValueError):
                distance = float("nan")
            track_id = int(track_ids[i]) if track_ids is not None else None

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "distance": round(float(distance), 2),
                "bbox": bbox,
                "track_id": track_id,
            })

        return detections

    def draw_detections(self, bgr_image, detections):
        return draw_detections(bgr_image, detections)


class DirectionalDetector(BaseYOLODetector):
    detection_label = "Object"

    def __init__(
            self,
            model_path,
            device=None,
            fx=None,
            cx=None,
            camera_width=1280,
            tolerance_ratio=0.05,
            tolerance_deg=5,
            use_tracking=False,
            tracker=None,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            use_tracking=use_tracking,
            tracker=tracker,
        )
        self.fx = fx
        self.camera_width = camera_width
        self.cx = cx if cx is not None else camera_width / 2
        self.tolerance_ratio = tolerance_ratio
        self.tolerance_deg = tolerance_deg

    def detect(self, bgr_image, depth_array):
        detections = super().detect(bgr_image, depth_array)
        angle_key = f"{self.detection_label} angle: "
        side_key = f"{self.detection_label} side: "
        for detection in detections:
            angle_deg = self._compute_angle(detection)
            detection[angle_key] = angle_deg
            detection[side_key] = self._compute_side(detection, angle_deg)
        return detections

    def _compute_angle(self, detection):
        if self.fx is None:
            return None
        bbox = detection["bbox"]
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        angle_rad = np.arctan2(bbox_center_x - self.cx, self.fx)
        return float(np.degrees(angle_rad))

    def _compute_side(self, detection, angle_deg=None):
        if angle_deg is not None:
            if abs(angle_deg) <= self.tolerance_deg:
                return "across"
            if angle_deg > 0:
                return "right"
            return "left"

        bbox = detection["bbox"]
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        image_center_x = self.camera_width / 2
        tolerance_px = self.camera_width * self.tolerance_ratio
        diff = bbox_center_x - image_center_x
        if abs(diff) <= tolerance_px:
            return "across"
        if diff > 0:
            return "right"
        return "left"


class BuoyDetector(DirectionalDetector):
    detection_label = "Buoy"


class ArTagDetector(DirectionalDetector):
    detection_label = "AR tag"
