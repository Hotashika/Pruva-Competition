import argparse
import importlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vision.ar_tag_utils import (
    bbox_to_qr_detection,
    expanded_bbox,
    normalize_qr_payload,
    track_template,
)
from vision.detector_lifecycle import DetectorRegistry


def load_profile(competition):
    if competition not in {"njord", "teknofest"}:
        raise ValueError(f"Unsupported competition profile: {competition}")
    return importlib.import_module(f"{competition}.config.vision_profile")


class VisionNode(Node):
    def __init__(self, profile, fx=None, fy=None, cx=None, cy=None):
        super().__init__("vision_node")
        self.profile = profile
        self.detectors = {}
        self.active_detector_names = ()
        self.current_task = None
        self.qr_detector = None
        self.qr_pub = None
        self.qr_confirmation_history = deque(maxlen=6)
        self.qr_confirmed_payload = None
        self.last_ar_inference_time = 0.0
        self.last_ar_detection = None
        self.last_ar_template = None
        self.ar_confirmed_inference_hz = 5.0

        if profile.QR_TOPIC:
            self.qr_detector = cv2.QRCodeDetector()
            self.qr_pub = self.create_publisher(String, profile.QR_TOPIC, 10)
            self.ar_confirmed_inference_hz = max(
                1.0,
                float(os.getenv(profile.AR_CONFIRMED_HZ_ENV, "5.0")),
            )

        self.rgb_shm = self._attach_with_retry(profile.shared_state.RGB_SHM_NAME)
        self.depth_shm = self._attach_with_retry(profile.shared_state.DEPTH_SHM_NAME)
        self.meta_shm = self._attach_with_retry(profile.shared_state.META_SHM_NAME)
        self.calib_shm = None
        self.imu_shm = None

        self.rgb = np.ndarray(
            profile.RGB_SHAPE,
            dtype=np.uint8,
            buffer=self.rgb_shm.buf,
        )
        self.depth = np.ndarray(
            profile.DEPTH_SHAPE,
            dtype=np.float32,
            buffer=self.depth_shm.buf,
        )
        self.meta = np.ndarray(
            profile.shared_state.META_SHAPE,
            dtype=np.int64,
            buffer=self.meta_shm.buf,
        )
        self.calib = None
        self.imu = None

        if getattr(profile, "USE_SHARED_IMU", False):
            self.imu_shm = self._attach_with_retry(
                profile.shared_state.IMU_SHM_NAME
            )
            self.imu = np.ndarray(
                profile.shared_state.IMU_SHAPE,
                dtype=np.float64,
                buffer=self.imu_shm.buf,
            )

        if profile.USE_SHARED_CALIBRATION:
            self.calib_shm = self._attach_with_retry(
                profile.shared_state.CALIB_SHM_NAME
            )
            self.calib = np.ndarray(
                profile.shared_state.CALIB_SHAPE,
                dtype=np.float64,
                buffer=self.calib_shm.buf,
            )
            calibration = self.calib.tolist()
            if fx is None:
                fx = calibration[profile.CALIBRATION_FX_INDEX]
            if cx is None:
                cx = calibration[profile.CALIBRATION_CX_INDEX]
            fy_index = getattr(profile, "CALIBRATION_FY_INDEX", None)
            cy_index = getattr(profile, "CALIBRATION_CY_INDEX", None)
            if fy is None and fy_index is not None:
                fy = calibration[fy_index]
            if cy is None and cy_index is not None:
                cy = calibration[cy_index]

        self.fx = None if fx is None else float(fx)
        self.fy = self.fx if fy is None else float(fy)
        self.cx = None if cx is None else float(cx)
        self.cy = (
            profile.DEPTH_SHAPE[0] / 2.0
            if cy is None
            else float(cy)
        )
        try:
            self.detector_registry = DetectorRegistry(
                profile,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
            )
        except Exception:
            self._close_shared_memory_handles()
            raise
        self.detectors = self.detector_registry.detectors
        self.active_detector_names = self.detector_registry.active_names
        self.get_logger().info(
            "Startup detectors loaded; active before task selection: "
            f"{self.active_detector_names}"
        )
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

        self.last_frame_id = -1
        self.pub = self.create_publisher(String, "/vision/detections", 10)
        shadow_topic = getattr(profile, "SHADOW_DETECTIONS_TOPIC", None)
        self.shadow_pub = (
            None
            if not shadow_topic
            else self.create_publisher(String, shadow_topic, 10)
        )
        self.create_subscription(
            String,
            "/mission/active_task",
            self.on_task_change,
            10,
        )
        self.create_timer(1 / 15, self.process_frame)

    def on_task_change(self, msg):
        task = msg.data
        if task == self.current_task:
            return

        if not self.detector_registry.select_task(task):
            self.get_logger().warn(
                f"Unknown task: '{task}', detector state unchanged."
            )
            return

        self.current_task = self.detector_registry.current_task
        self.active_detector_names = self.detector_registry.active_names
        self._reset_qr_tracking()
        self.get_logger().info(
            f"Task changed -> '{task}', active detectors: "
            f"{self.active_detector_names}"
        )

    def _reset_qr_tracking(self):
        self.qr_confirmation_history.clear()
        self.qr_confirmed_payload = None
        self.last_ar_inference_time = 0.0
        self.last_ar_detection = None
        self.last_ar_template = None

    def _should_run_ar_inference(self):
        if self.qr_confirmed_payload is None or self.last_ar_detection is None:
            return True
        elapsed = time.monotonic() - self.last_ar_inference_time
        return elapsed >= 1.0 / self.ar_confirmed_inference_hz

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
        if self.last_ar_detection is None or self.last_ar_template is None:
            return None
        previous = self.last_ar_detection.get("bbox")
        if not previous or len(previous) != 4:
            return None
        tracked_bbox, max_score = track_template(
            image,
            previous,
            self.last_ar_template,
        )
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
            self.get_logger().warn(
                f"QR decode failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        payload = str(payload).strip()
        if not payload:
            return None
        canonical = normalize_qr_payload(payload)
        self.qr_confirmation_history.append(canonical)
        if self.qr_confirmation_history.count(canonical) >= 3:
            self.qr_confirmed_payload = canonical
        return bbox_to_qr_detection(
            bbox,
            canonical,
            detection.get("confidence", 0.0),
            distance_m=detection.get("distance"),
        )

    def _publish_task_qr(self, frame_id, image, qr_detection):
        message = String()
        message.data = json.dumps({
            "frame_id": frame_id,
            "frame_px": {
                "width": image.shape[1],
                "height": image.shape[0],
            },
            "detections": [] if qr_detection is None else [qr_detection],
        })
        self.qr_pub.publish(message)

    def _attach_with_retry(self, name, retries=20, delay=0.5):
        return self.profile.attach_existing_shared_memory(
            name,
            retries=retries,
            delay=delay,
        )

    def process_frame(self):
        frame_id = int(self.meta[0])
        if frame_id == self.last_frame_id or not self.detectors:
            return

        camera_timestamp_ms = int(self.meta[1])
        bgr_image = self.rgb[:, :, :3].copy()
        depth_array = self.depth.copy()
        imu_sample = None if self.imu is None else self.imu.copy()
        if int(self.meta[0]) != frame_id:
            return
        self.last_frame_id = frame_id

        all_detections = []
        shadow_detections = []
        shadow_diagnostics = {}
        for name, detector in self.detector_registry.active_items():
            if (
                    name == "ar_tag"
                    and self.profile.QR_TOPIC
                    and not self._should_run_ar_inference()
            ):
                tracked = self._track_last_ar_detection(bgr_image)
                detections = [] if tracked is None else [tracked]
            else:
                if self.profile.DETECTOR_SPECS[name].get("uses_imu"):
                    detections = detector.detect(
                        bgr_image,
                        depth_array,
                        imu=imu_sample,
                    )
                else:
                    detections = detector.detect(bgr_image, depth_array)
                if name == "ar_tag" and self.profile.QR_TOPIC:
                    self.last_ar_inference_time = time.monotonic()
                    best = self._best_ar_detection(detections)
                    if best is not None:
                        self._remember_ar_detection(bgr_image, best)
                    else:
                        self.last_ar_detection = None
                        self.last_ar_template = None

            for detection in detections:
                detection.setdefault("type", name)
            all_detections.extend(detections)
            detector_shadow = getattr(
                detector,
                "last_shadow_detections",
                None,
            )
            if detector_shadow is not None:
                shadow_detections.extend(detector_shadow)
                shadow_diagnostics[name] = getattr(
                    detector,
                    "last_diagnostics",
                    {},
                )

        if (
                self.qr_pub is not None
                and self.current_task == self.profile.QR_TASK
        ):
            best_ar = self._best_ar_detection(
                [
                    item
                    for item in all_detections
                    if item.get("type") == "ar_tag"
                ]
            )
            qr_detection = self._decode_qr_in_ar_roi(bgr_image, best_ar)
            self._publish_task_qr(frame_id, bgr_image, qr_detection)

        payload = {
            "frame_id": frame_id,
            "detections": all_detections,
        }
        if self.profile.PUBLISH_CAMERA_TIMESTAMP:
            payload["camera_timestamp_ms"] = camera_timestamp_ms

        message = String()
        message.data = json.dumps(payload)
        self.pub.publish(message)

        if self.shadow_pub is not None and shadow_diagnostics:
            shadow_message = String()
            shadow_message.data = json.dumps(
                {
                    "frame_id": frame_id,
                    "camera_timestamp_ms": camera_timestamp_ms,
                    "active_task": self.current_task,
                    "detections": shadow_detections,
                    "diagnostics": shadow_diagnostics,
                }
            )
            self.shadow_pub.publish(shadow_message)

    def _close_shared_memory_handles(self):
        self.rgb = None
        self.depth = None
        self.meta = None
        self.calib = None
        self.imu = None
        handles = [self.rgb_shm, self.depth_shm, self.meta_shm]
        if self.calib_shm is not None:
            handles.append(self.calib_shm)
        if self.imu_shm is not None:
            handles.append(self.imu_shm)
        self.profile.close_shared_memory_handles(*handles)
        self.rgb_shm = None
        self.depth_shm = None
        self.meta_shm = None
        self.calib_shm = None
        self.imu_shm = None

    def destroy_node(self):
        self._close_shared_memory_handles()
        super().destroy_node()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition",
        required=True,
        choices=("njord", "teknofest"),
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    args = parser.parse_args(argv)

    profile = load_profile(args.competition)
    rclpy.init()
    node = VisionNode(
        profile=profile,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
