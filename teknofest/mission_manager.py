#!/usr/bin/env python3

"""Start the continuous TEKNOFEST mission from the MAVLink mission-start topic."""

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


MISSION_START = 1
MISSION_STOP_COMMANDS = (90, 99)
MISSION_PATH = os.path.join(PROJECT_ROOT, "missions", "competition_mission.py")
MISSION_MODULE = "teknofest.missions.competition_mission"


class MissionManager(Node):
    """Own the Task 1 -> Task 2 -> Task 3 competition subprocess."""

    def __init__(self):
        super().__init__("teknofest_mission_manager")
        self.declare_parameter(
            "mission_start_topic",
            os.getenv("MAVLINK_MISSION_START_TOPIC", "/mission_start"),
        )
        self.declare_parameter(
            "mission_start_ack_topic",
            os.getenv("MAVLINK_MISSION_START_ACK_TOPIC", "/mission_start_ack"),
        )
        self.mission_start_topic = str(self.get_parameter("mission_start_topic").value)
        self.mission_start_ack_topic = str(
            self.get_parameter("mission_start_ack_topic").value
        )
        self.active_mission_process = None

        self.status_pub = self.create_publisher(String, "/mission_manager/status", 10)
        self.ack_pub = self.create_publisher(Int32, self.mission_start_ack_topic, 10)
        self.create_subscription(
            Int32, self.mission_start_topic, self._mission_start_callback, 10
        )
        self._publish_status(
            f"TEKNOFEST Mission Manager active; waiting for "
            f"{self.mission_start_topic}=1."
        )

    def _mission_start_callback(self, msg):
        command = int(msg.data)
        if command == MISSION_START:
            if not self._start_competition():
                return
        elif command in MISSION_STOP_COMMANDS:
            self._stop_active_mission()
        else:
            self._publish_status(f"Unsupported mission command ignored: {command}")
            return

        ack = Int32()
        ack.data = command
        self.ack_pub.publish(ack)
        self._publish_status(
            f"{self.mission_start_topic}={command} handled; "
            f"{self.mission_start_ack_topic}={command} published."
        )

    def _start_competition(self):
        if self.active_mission_process is not None:
            if self.active_mission_process.poll() is None:
                self._publish_status("Competition mission is already running.")
                return True
            self.active_mission_process = None

        if not os.path.isfile(MISSION_PATH):
            self._publish_status(f"Competition mission script not found: {MISSION_PATH}")
            return False
        try:
            self.active_mission_process = subprocess.Popen(
                [sys.executable, "-m", MISSION_MODULE],
                cwd=COMPETITION_ROOT,
                start_new_session=True,
            )
        except Exception as exc:
            self.active_mission_process = None
            self._publish_status(f"Competition mission could not start: {exc}")
            return False

        self._publish_status(
            "Competition mission started at Task 1; it will transition internally "
            f"to Task 2 and Task 3. pid={self.active_mission_process.pid}"
        )
        return True

    def _stop_active_mission(self):
        process = self.active_mission_process
        self.active_mission_process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _publish_status(self, text):
        self.get_logger().info(text)
        message = String()
        message.data = text
        self.status_pub.publish(message)

    def destroy_node(self):
        self._stop_active_mission()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
