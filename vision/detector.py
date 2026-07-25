from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from vision.depth_utils import get_distance_from_bbox


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
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
