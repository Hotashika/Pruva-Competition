import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from njord.config.camera_config import DEPTH_SHAPE, RGB_SHAPE
from njord.config.vision_config import AR_TAG_MODEL_PATH, BUOY_MODEL_PATH
from njord.core import shared_state
from njord.core.shared_memory_utils import attach_existing_shared_memory, close_shared_memory_handles
from njord.vision.detector import ArTagDetector, BuoyDetector
from njord.vision.ar_tag_utils import (
    bbox_to_qr_detection,
    expanded_bbox,
    normalize_qr_payload,
    track_template,
)

TASK_DETECTOR_MAP = {
    "task1": {"buoy"},
    "task2": {"buoy"},
    "task3": {"ar_tag"},
    "task4": {"buoy"},
}

DETECTOR_REGISTRY = {
    "buoy": (BuoyDetector, BUOY_MODEL_PATH),
    "ar_tag": (ArTagDetector, AR_TAG_MODEL_PATH),
}


class VisionNode(Node):
    def __init__(self, fx=None, cx=None):
        super().__init__('vision_node')

        self.detectors = {}  # name -> instance
        self.current_task = None
        self.qr_detector = cv2.QRCodeDetector()
        self.qr_pub = self.create_publisher(String, '/njord/task3/qr_detections', 10)
        self.qr_confirmation_history = deque(maxlen=6)
        self.qr_confirmed_payload = None
        self.last_ar_inference_time = 0.0
        self.last_ar_detection = None
        self.last_ar_template = None
        self.ar_confirmed_inference_hz = max(
            1.0, float(os.getenv("NJORD_TASK3_AR_CONFIRMED_HZ", "5.0"))
        )

        self.rgb_shm = self._attach_with_retry(shared_state.RGB_SHM_NAME)
        self.depth_shm = self._attach_with_retry(shared_state.DEPTH_SHM_NAME)
        self.meta_shm = self._attach_with_retry(shared_state.META_SHM_NAME)
        self.calib_shm = self._attach_with_retry(shared_state.CALIB_SHM_NAME)
        self.rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=self.rgb_shm.buf)
        self.depth = np.ndarray(DEPTH_SHAPE, dtype=np.float32, buffer=self.depth_shm.buf)
        self.meta = np.ndarray(shared_state.META_SHAPE, dtype=np.int64, buffer=self.meta_shm.buf)
        self.calib = np.ndarray(
            shared_state.CALIB_SHAPE,
            dtype=np.float64,
            buffer=self.calib_shm.buf,
        )

        calib_fx, _, calib_cx, _ = self.calib.tolist()
        self.fx = float(calib_fx if fx is None else fx)
        self.cx = float(calib_cx if cx is None else cx)

        self.last_frame_id = -1
        self.pub = self.create_publisher(String, '/vision/detections', 10)

        self.create_subscription(String, '/mission/active_task', self.on_task_change, 10)

        self.create_timer(1 / 15, self.process_frame)

    def on_task_change(self, msg: String):
        task = msg.data
        if task == self.current_task:
            return

        wanted = TASK_DETECTOR_MAP.get(task)
        if wanted is None:
            self.get_logger().warn(f"Unknown task: '{task}', detector state unchanged.")
            return

        self.current_task = task
        self._reset_task3_tracking()
        self.get_logger().info(f"Task changed -> '{task}', active detectors: {wanted}")

        for name in list(self.detectors.keys()):
            if name not in wanted:
                self.get_logger().info(f"Closing '{name}' detector...")
                del self.detectors[name]

        for name in wanted:
            if name not in self.detectors:
                cls, model_path = DETECTOR_REGISTRY[name]
                self.get_logger().info(f"Loading '{name}' detector...")
                self.detectors[name] = cls(model_path=model_path, fx=self.fx, cx=self.cx)

        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def _reset_task3_tracking(self):
        self.qr_confirmation_history.clear()
        self.qr_confirmed_payload = None
        self.last_ar_inference_time = 0.0
        self.last_ar_detection = None
        self.last_ar_template = None

    def _should_run_ar_inference(self):
        if self.qr_confirmed_payload is None or self.last_ar_detection is None:
            return True
        return time.monotonic() - self.last_ar_inference_time >= 1.0 / self.ar_confirmed_inference_hz

    @staticmethod
    def _best_ar_detection(detections):
        if not detections:
            return None
        return max(detections, key=lambda item: float(item.get("confidence", 0.0)))

    def _remember_ar_detection(self, image, detection):
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return
        template = image[y1:y2, x1:x2]
        if template.size == 0:
            return
        self.last_ar_detection = dict(detection)
        self.last_ar_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    def _track_last_ar_detection(self, image):
        """Track the last tag cheaply between reduced-rate YOLO inferences."""
        if self.last_ar_detection is None or self.last_ar_template is None:
            return None
        previous = self.last_ar_detection.get("bbox")
        if not previous or len(previous) != 4:
            return None
        tracked_bbox, max_score = track_template(image, previous, self.last_ar_template)
        if tracked_bbox is None:
            return None
        tracked = dict(self.last_ar_detection)
        tracked["bbox"] = tracked_bbox
        tracked["tracked"] = True
        tracked["tracking_score"] = round(float(max_score), 3)
        self.last_ar_detection = tracked
        return tracked

    def _decode_qr_in_ar_roi(self, image, detection):
        if detection is None:
            return None
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = expanded_bbox(bbox, image.shape)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        try:
            payload, _, _ = self.qr_detector.detectAndDecode(roi)
        except cv2.error as exc:
            self.get_logger().warn(f"QR decode failed: {exc}", throttle_duration_sec=2.0)
            return None
        payload = str(payload).strip()
        if not payload:
            return None
        canonical = normalize_qr_payload(payload)
        self.qr_confirmation_history.append(canonical)
        if self.qr_confirmation_history.count(canonical) >= 3:
            self.qr_confirmed_payload = canonical
        return bbox_to_qr_detection(bbox, canonical, detection.get("confidence", 0.0))

    def _publish_task3_qr(self, frame_id, image, qr_detection):
        message = String()
        message.data = json.dumps({
            "frame_id": frame_id,
            "frame_px": {"width": image.shape[1], "height": image.shape[0]},
            "detections": [] if qr_detection is None else [qr_detection],
        })
        self.qr_pub.publish(message)

    def _attach_with_retry(self, name, retries=20, delay=0.5):
        return attach_existing_shared_memory(name, retries=retries, delay=delay)

    def process_frame(self):
        frame_id = int(self.meta[0])
        if frame_id == self.last_frame_id:
            return

        if not self.detectors:
            return

        bgr_image = self.rgb[:, :, :3].copy()
        depth_array = self.depth.copy()
        if int(self.meta[0]) != frame_id:
            return
        self.last_frame_id = frame_id

        all_detections = []
        for name, detector in self.detectors.items():
            if name == "ar_tag" and not self._should_run_ar_inference():
                tracked = self._track_last_ar_detection(bgr_image)
                dets = [] if tracked is None else [tracked]
            else:
                dets = detector.detect(bgr_image, depth_array)
                if name == "ar_tag":
                    self.last_ar_inference_time = time.monotonic()
                    best = self._best_ar_detection(dets)
                    if best is not None:
                        self._remember_ar_detection(bgr_image, best)
                    else:
                        self.last_ar_detection = None
                        self.last_ar_template = None
            for d in dets:
                d["type"] = name
            all_detections += dets

        if self.current_task == "task3":
            best_ar = self._best_ar_detection(
                [item for item in all_detections if item.get("type") == "ar_tag"]
            )
            qr_detection = self._decode_qr_in_ar_roi(bgr_image, best_ar)
            self._publish_task3_qr(frame_id, bgr_image, qr_detection)

        # Bos detection listesi de vision pipeline'in calistigini gosteren bir
        # heartbeat'tir. Mission, mesaj kesilirse bunu failsafe olarak ele alir.
        msg = String()
        msg.data = json.dumps({
            "frame_id": frame_id,
            "detections": all_detections,
        })
        self.pub.publish(msg)

    def destroy_node(self):
        self.rgb = None
        self.depth = None
        self.meta = None
        self.calib = None
        close_shared_memory_handles(
            self.rgb_shm,
            self.depth_shm,
            self.meta_shm,
            self.calib_shm,
        )
        self.rgb_shm = None
        self.depth_shm = None
        self.meta_shm = None
        self.calib_shm = None
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    args = parser.parse_args()

    rclpy.init()
    node = VisionNode(
        fx=args.fx,
        cx=args.cx,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
