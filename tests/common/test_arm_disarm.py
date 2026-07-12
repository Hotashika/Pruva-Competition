#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from utils.mavlink_utilities import (
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    publish_cmd_vel,
    stop_vehicle,
    wait_for_mission_services,
)

DEFAULT_SECONDS = 5.0
DEFAULT_LINEAR_X = 0.5
DEFAULT_ANGULAR_Z = 0.0
DEFAULT_MODE = "GUIDED"


class BridgeMotorTestNode(Node):
    def __init__(self):
        super().__init__("bridge_motor_test_node")
        self.clients = create_mission_clients(self)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cube/cmd_vel", 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Arm, send /cube/cmd_vel for a short interval, then disarm through "
            "the bridge services."
        )
    )
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--linear-x", type=float, default=DEFAULT_LINEAR_X)
    parser.add_argument("--angular-z", type=float, default=DEFAULT_ANGULAR_Z)
    parser.add_argument(
        "--normal-arm",
        action="store_true",
        help="Use /cube/arm instead of /cube/force_arm.",
    )
    parser.add_argument(
        "--no-disarm",
        action="store_true",
        help="Leave the vehicle armed after the test.",
    )
    return parser.parse_args()


def run_motion(node, seconds, linear_x, angular_z):
    deadline = time.monotonic() + max(0.0, seconds)
    while rclpy.ok() and time.monotonic() < deadline:
        publish_cmd_vel(node.cmd_vel_pub, linear_x=linear_x, angular_z=angular_z)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.05)


def main():
    args = parse_args()
    rclpy.init()
    node = BridgeMotorTestNode()
    armed_by_test = False

    try:
        wait_for_mission_services(node, node.clients)

        node.get_logger().info(f"Setting vehicle to {args.mode} mode...")
        if call_set_mode(node, node.clients.set_mode_client, args.mode) is False:
            raise RuntimeError(f"Failed to switch to {args.mode} mode.")

        arm_client = node.clients.arm_client if args.normal_arm else node.clients.force_arm_client
        arm_label = "ARM" if args.normal_arm else "FORCE ARM"
        node.get_logger().info(f"{arm_label} requested...")
        if call_trigger_service(node, arm_client, arm_label) is False:
            raise RuntimeError(f"{arm_label} failed.")
        armed_by_test = True

        node.get_logger().info(
            f"Publishing /cube/cmd_vel for {args.seconds:.1f}s: "
            f"linear_x={args.linear_x:.2f}, angular_z={args.angular_z:.2f}"
        )
        run_motion(node, args.seconds, args.linear_x, args.angular_z)

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")

    finally:
        node.get_logger().info("Stopping vehicle...")
        stop_vehicle(node.cmd_vel_pub)

        if armed_by_test and not args.no_disarm:
            call_trigger_service(node, node.clients.disarm_client, "DISARM")
            stop_vehicle(node.cmd_vel_pub)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
