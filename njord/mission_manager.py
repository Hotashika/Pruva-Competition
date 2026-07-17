#!/usr/bin/env python3

import os
import signal
import subprocess
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
    clear_task_selection,
    default_task_selection_file,
    read_task_selection,
    write_task_selection,
)


MISSION_PATHS = {
    1: os.path.join(PROJECT_ROOT, "missions", "task1_maneuvering_and_path_finding.py"),
    2: os.path.join(PROJECT_ROOT, "missions", "task2_collision_avoidance.py"),
    3: os.path.join(PROJECT_ROOT, "missions", "task3_docking.py"),
    4: os.path.join(PROJECT_ROOT, "missions", "task4_surprise.py"),
}


class MissionManager(Node):
    """
    Starts and stops the selected mission algorithm and reports system status.

    Bridge publishes MAVLink task commands to /mission_start. Mission Manager
    owns the task subprocess and writes the selected task to the shared JSON file.
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
        self.declare_parameter(
            "mission_start_ack_topic",
            os.getenv("MAVLINK_MISSION_START_ACK_TOPIC", "/mission_start_ack"),
        )

        self.mission_start_topic = str(self.get_parameter("mission_start_topic").value)
        self.mission_start_ack_topic = str(
            self.get_parameter("mission_start_ack_topic").value
        )
        self.task_selection_file = str(self.get_parameter("task_selection_file").value)
        clear_task_selection(self.task_selection_file)
        self.bridge_state = {}
        self.last_reported_state = None
        self.active_mission_number = None
        self.active_mission_process = None

        self.status_pub = self.create_publisher(String, "/mission_manager/status", 10)
        self.mission_start_ack_pub = self.create_publisher(
            Int32, self.mission_start_ack_topic, 10
        )
        self.create_subscription(Int32, self.mission_start_topic, self._mission_start_callback, 10)
        self.create_subscription(String, "/cube/state", self._bridge_state_callback, 10)
        self.create_timer(1.0, self._status_loop)

        self._publish_status(
            f"Mission Manager aktif. {self.mission_start_topic} dinleniyor; "
            f"gorev process yonetimi aktif, JSON durum dosyasi idle olarak sifirlandi: "
            f"{self.task_selection_file}"
        )

    def _bridge_state_callback(self, msg):
        self.bridge_state = parse_bridge_state(msg.data)

    def _mission_start_callback(self, msg):
        try:
            command = int(msg.data)
        except (TypeError, ValueError):
            self._publish_status(f"Invalid mission_start payload ignored: {msg.data}")
            return

        if command in MISSION_PATHS:
            if not self._start_mission(command):
                return
        elif command in (90, 99):
            self._stop_active_mission()

        try:
            state = write_task_selection(self.task_selection_file, command)
        except Exception as exc:
            self._publish_status(f"Task selection JSON write failed: {exc}")
            if command in MISSION_PATHS:
                self._stop_active_mission()
            return

        ack = Int32()
        ack.data = command
        self.mission_start_ack_pub.publish(ack)

        self._publish_status(
            f"{self.mission_start_topic} received: command={command}, "
            f"selected_task={state.get('selected_task')}, status={state.get('status')}; "
            f"{self.mission_start_ack_topic}={command} published"
        )

    def _start_mission(self, command):
        mission_path = MISSION_PATHS[command]
        if not os.path.isfile(mission_path):
            self._publish_status(f"Mission script not found: {mission_path}")
            return False

        self._stop_active_mission()
        try:
            self.active_mission_process = subprocess.Popen(
                [sys.executable, mission_path],
                start_new_session=True,
            )
        except Exception as exc:
            self.active_mission_process = None
            self.active_mission_number = None
            self._publish_status(f"Mission M{command} process could not start: {exc}")
            return False

        self.active_mission_number = command
        self._publish_status(
            f"Mission M{command} process started: "
            f"pid={self.active_mission_process.pid}, script={mission_path}"
        )
        return True

    def _stop_active_mission(self):
        process = self.active_mission_process
        mission_number = self.active_mission_number
        self.active_mission_process = None
        self.active_mission_number = None
        if process is None or process.poll() is not None:
            return

        self._publish_status(f"Mission M{mission_number} process stopping: pid={process.pid}")
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self._publish_status(f"Mission M{mission_number} process stopped")

    def destroy_node(self):
        self._stop_active_mission()
        return super().destroy_node()

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
