import json
import threading

from rclpy.node import Node
from std_msgs.msg import String

from vision.detection_distance import nearest_bbox_median_distance


class VisionDetectionCache(Node):
    """Cache /vision/detections so video recording never reruns a model."""

    def __init__(self, node_name):
        super().__init__(node_name)
        self._lock = threading.Lock()
        self._frame_id = None
        self._detections = []
        self.create_subscription(String, "/vision/detections", self._callback, 10)

    def _callback(self, message):
        try:
            payload = json.loads(message.data)
            frame_id = int(payload.get("frame_id"))
            detections = payload.get("detections", [])
            if not isinstance(detections, list):
                return
        except (TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warn("Invalid /vision/detections message ignored.")
            return

        with self._lock:
            self._frame_id = frame_id
            self._detections = detections

    def latest(self, frame_id, max_frame_lag=3):
        with self._lock:
            if (
                self._frame_id is None
                or abs(int(frame_id) - self._frame_id) > max_frame_lag
            ):
                return []
            return [
                dict(item)
                for item in self._detections
                if isinstance(item, dict)
            ]

    def nearest_bbox_median_distance(self, frame_id, max_frame_lag=3):
        with self._lock:
            if (
                self._frame_id is None
                or abs(int(frame_id) - self._frame_id) > max_frame_lag
            ):
                return None
            return nearest_bbox_median_distance(self._detections)
