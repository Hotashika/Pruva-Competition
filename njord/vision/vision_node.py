import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from njord.config.camera_config import DEPTH_SHAPE, RGB_SHAPE
from njord.config.vision_config import AR_TAG_MODEL_PATH, BUOY_MODEL_PATH
from njord.core import shared_state
from njord.core.shared_memory_utils import attach_existing_shared_memory, close_shared_memory_handles
from njord.vision.detector import ArTagDetector, BuoyDetector
from njord.vision.horizon_mask import create_horizon_mask, render_horizon_overlay

TASK_DETECTOR_MAP = {
    "task1": {"buoy", "ar_tag"},
    "task2": {"buoy", "ar_tag"},
    "task3": {"buoy", "ar_tag"},
    "task4": {"buoy", "ar_tag"},
}

DETECTOR_REGISTRY = {
    "buoy": (BuoyDetector, BUOY_MODEL_PATH),
    "ar_tag": (ArTagDetector, AR_TAG_MODEL_PATH),
}


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


class VisionNode(Node):
    def __init__(self, fx=None, cx=None, horizon_debug=False):
        super().__init__('vision_node')

        self.detectors = {}  # name -> instance
        self.current_task = None
        self.horizon_debug = bool(horizon_debug)
        self.horizon_roll_offset = math.radians(
            float(os.getenv("NJORD_HORIZON_ROLL_OFFSET_DEG", "0"))
        )
        self.horizon_pitch_offset = math.radians(
            float(os.getenv("NJORD_HORIZON_PITCH_OFFSET_DEG", "0"))
        )
        self.horizon_flip_roll = _env_flag("NJORD_HORIZON_FLIP_ROLL")
        self.horizon_flip_pitch = _env_flag("NJORD_HORIZON_FLIP_PITCH")
        self.horizon_invert_mask = _env_flag("NJORD_HORIZON_INVERT_MASK")

        self.rgb_shm = self._attach_with_retry(shared_state.RGB_SHM_NAME)
        self.depth_shm = self._attach_with_retry(shared_state.DEPTH_SHM_NAME)
        self.meta_shm = self._attach_with_retry(shared_state.META_SHM_NAME)
        self.imu_shm = self._attach_with_retry(shared_state.IMU_SHM_NAME)
        self.calib_shm = self._attach_with_retry(shared_state.CALIB_SHM_NAME)
        self.rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=self.rgb_shm.buf)
        self.depth = np.ndarray(DEPTH_SHAPE, dtype=np.float32, buffer=self.depth_shm.buf)
        self.meta = np.ndarray(shared_state.META_SHAPE, dtype=np.int64, buffer=self.meta_shm.buf)
        self.imu = np.ndarray(
            shared_state.IMU_SHAPE,
            dtype=np.float64,
            buffer=self.imu_shm.buf,
        )
        self.calib = np.ndarray(
            shared_state.CALIB_SHAPE,
            dtype=np.float64,
            buffer=self.calib_shm.buf,
        )

        calib_fx, self.fy, calib_cx, self.cy = self.calib.tolist()
        self.fx = float(calib_fx if fx is None else fx)
        self.cx = float(calib_cx if cx is None else cx)

        self.last_frame_id = -1
        self.pub = self.create_publisher(String, '/vision/detections', 10)
        self.horizon_pub = None
        if self.horizon_debug:
            self.horizon_pub = self.create_publisher(
                CompressedImage,
                '/vision/horizon_overlay/compressed',
                1,
            )
            self.get_logger().info(
                "IMU horizon debug enabled: /vision/horizon_overlay/compressed"
            )

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
        roll, pitch, yaw = self.imu.tolist()
        if int(self.meta[0]) != frame_id:
            return
        self.last_frame_id = frame_id

        horizon = {"valid": False}
        horizon_mask = None
        try:
            horizon_mask = create_horizon_mask(
                width=bgr_image.shape[1],
                height=bgr_image.shape[0],
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                roll=roll,
                pitch=pitch,
                roll_offset=self.horizon_roll_offset,
                pitch_offset=self.horizon_pitch_offset,
                flip_roll=self.horizon_flip_roll,
                flip_pitch=self.horizon_flip_pitch,
                invert=self.horizon_invert_mask,
            )
            horizon = {
                "valid": True,
                "roll_deg": round(math.degrees(roll), 3),
                "pitch_deg": round(math.degrees(pitch), 3),
                "yaw_deg": round(math.degrees(yaw), 3),
                "water_fraction": round(float(horizon_mask.mean()), 4),
            }
            if self.horizon_debug and frame_id % 5 == 0:
                self._publish_horizon_overlay(bgr_image, horizon_mask)
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(
                f"IMU horizon mask unavailable: {exc}",
                throttle_duration_sec=2.0,
            )

        all_detections = []
        for name, detector in self.detectors.items():
            dets = detector.detect(bgr_image, depth_array)
            for d in dets:
                d["type"] = name
            all_detections += dets

        # Bos detection listesi de vision pipeline'in calistigini gosteren bir
        # heartbeat'tir. Mission, mesaj kesilirse bunu failsafe olarak ele alir.
        msg = String()
        msg.data = json.dumps({
            "frame_id": frame_id,
            "horizon": horizon,
            "detections": all_detections,
        })
        self.pub.publish(msg)

    def _publish_horizon_overlay(self, bgr_image, horizon_mask):
        overlay = render_horizon_overlay(bgr_image, horizon_mask)
        encoded, jpeg = cv2.imencode(
            ".jpg",
            overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not encoded:
            self.get_logger().warn(
                "Could not encode IMU horizon overlay.",
                throttle_duration_sec=2.0,
            )
            return

        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "zed_left_camera"
        message.format = "jpeg"
        message.data = jpeg.tobytes()
        self.horizon_pub.publish(message)

    def destroy_node(self):
        self.rgb = None
        self.depth = None
        self.meta = None
        self.imu = None
        self.calib = None
        close_shared_memory_handles(
            self.rgb_shm,
            self.depth_shm,
            self.meta_shm,
            self.imu_shm,
            self.calib_shm,
        )
        self.rgb_shm = None
        self.depth_shm = None
        self.meta_shm = None
        self.imu_shm = None
        self.calib_shm = None
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--horizon-debug", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = VisionNode(
        fx=args.fx,
        cx=args.cx,
        horizon_debug=args.horizon_debug,
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
