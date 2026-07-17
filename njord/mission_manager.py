#!/usr/bin/env python3

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ROOT = os.path.dirname(PROJECT_ROOT)
if COMPETITION_ROOT not in sys.path:
    sys.path.insert(0, COMPETITION_ROOT)

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int32, String

from utils.mavlink_utilities import parse_bridge_state
from utils.task_selection_state import (
    default_task_selection_file,
    read_task_selection,
    write_task_selection,
)


class MissionManager(Node):
    """
    Keeps MAVLink/system status visible without starting mission algorithms.

    Bridge publishes MAVLink task commands to /mission_start. Mission Manager
    listens to that topic and writes the selected task to the shared JSON file.
    """

    def __init__(self):
        super().__init__("njord_mission_manager")

        self.declare_parameter(
            "mission_start_topic",
            os.getenv("MAVLINK_MISSION_START_TOPIC", "/mission_start"),
        )
        self.declare_parameter(
            "task_selection_file",
            os.getenv("MISSION_SELECTION_FILE", default_task_selection_file()),
        )

        self.mission_start_topic = str(self.get_parameter("mission_start_topic").value)
        self.task_selection_file = str(self.get_parameter("task_selection_file").value)
        self.bridge_state = {}
        self.last_reported_state = None

        self.status_pub = self.create_publisher(String, "/mission_manager/status", 10)
        self.create_subscription(Int32, self.mission_start_topic, self._mission_start_callback, 10)
        self.create_subscription(String, "/cube/state", self._bridge_state_callback, 10)
        self.create_timer(1.0, self._status_loop)

        self._publish_status(
            f"Mission Manager aktif. {self.mission_start_topic} dinleniyor; "
            f"gorev baslatilmiyor, JSON durum dosyasi yaziliyor: {self.task_selection_file}"
        )

    def _bridge_state_callback(self, msg):
        self.bridge_state = parse_bridge_state(msg.data)

    def _mission_start_callback(self, msg):
        try:
            command = int(msg.data)
        except (TypeError, ValueError):
            self._publish_status(f"Invalid mission_start payload ignored: {msg.data}")
            return

        try:
            state = write_task_selection(self.task_selection_file, command)
        except Exception as exc:
            self._publish_status(f"Task selection JSON write failed: {exc}")
            return

        self._publish_status(
            f"{self.mission_start_topic} received: command={command}, "
            f"selected_task={state.get('selected_task')}, status={state.get('status')}"
        )

    def _status_loop(self):
        selection = read_task_selection(self.task_selection_file)
        current_state = {
            "selected_task": selection.get("selected_task"),
            "task_status": selection.get("status"),
            "connected": self.bridge_state.get("connected"),
            "armed": self.bridge_state.get("armed"),
            "mode": self.bridge_state.get("mode"),
        }

        if current_state == self.last_reported_state:
            return

        self.last_reported_state = current_state
        self._publish_status(
            "System status: "
            f"selected_task={current_state['selected_task']}, "
            f"task_status={current_state['task_status']}, "
            f"connected={current_state['connected']}, "
            f"armed={current_state['armed']}, "
            f"mode={current_state['mode']}"
        )

    def _publish_status(self, text):
        self.get_logger().info(text)
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Mission Manager kapatiliyor...")
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
