#!/usr/bin/env python3

"""Manual ROS service integration check; not part of the pytest suite."""

import argparse
import sys
import time

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


def normalize_mode(mode_name):
    return str(mode_name).strip().upper()


def parse_bridge_state(text):
    state = {}
    for part in str(text).split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in ("true", "false"):
            state[key] = value.lower() == "true"
        else:
            state[key] = value
    return state


class ModeArmDisarmServiceTest(Node):
    def __init__(self, args):
        super().__init__("mode_arm_disarm_service_test")
        self.args = args
        self.last_state = {}

        self.set_mode_client = self.create_client(SetMode, args.set_mode_service)
        self.arm_client = self.create_client(
            Trigger,
            args.force_arm_service if args.force_arm else args.arm_service,
        )
        self.disarm_client = self.create_client(Trigger, args.disarm_service)
        self.state_sub = self.create_subscription(
            String,
            args.state_topic,
            self._state_callback,
            10,
        )

    def _state_callback(self, msg):
        self.last_state = parse_bridge_state(msg.data)
        self.get_logger().info(f"Bridge state: {msg.data}")

    def wait_for_service(self, client, name):
        deadline = time.monotonic() + self.args.service_timeout
        while time.monotonic() < deadline:
            if client.wait_for_service(timeout_sec=0.25):
                self.get_logger().info(f"{name} service is ready.")
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().error(f"{name} service is not available.")
        return False

    def call_set_mode(self, mode_name):
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = str(mode_name)

        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.args.service_timeout,
        )

        if not future.done():
            self.get_logger().error(f"Set mode timed out: {mode_name}")
            return False

        response = future.result()
        ok = response is not None and bool(response.mode_sent)
        self.get_logger().info(f"Set mode response: mode_sent={ok}")
        return ok

    def call_trigger(self, client, label):
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.args.service_timeout,
        )

        if not future.done():
            self.get_logger().error(f"{label} service timed out.")
            return False

        response = future.result()
        ok = response is not None and bool(response.success)
        message = "" if response is None else str(response.message)
        self.get_logger().info(f"{label} response: success={ok}, message={message}")
        return ok

    def wait_for_state(self, predicate, description):
        if self.args.no_state_check:
            return True

        deadline = time.monotonic() + self.args.state_timeout
        while time.monotonic() < deadline:
            if predicate(self.last_state):
                self.get_logger().info(f"State confirmed: {description}")
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(
            f"State check timed out: {description}, last_state={self.last_state}"
        )
        return False

    def wait_for_mode(self, mode_name):
        expected_mode = normalize_mode(mode_name)
        return self.wait_for_state(
            lambda state: normalize_mode(state.get("mode", "")) == expected_mode,
            f"mode={expected_mode}",
        )

    def wait_for_armed(self, expected_armed):
        return self.wait_for_state(
            lambda state: state.get("armed") is expected_armed,
            f"armed={expected_armed}",
        )

    def wait_for_ready_services(self):
        return (
                self.wait_for_service(self.set_mode_client, self.args.set_mode_service)
                and self.wait_for_service(
            self.arm_client,
            self.args.force_arm_service if self.args.force_arm else self.args.arm_service,
        )
                and self.wait_for_service(self.disarm_client, self.args.disarm_service)
        )

    def run(self):
        if not self.wait_for_ready_services():
            return 1

        armed_by_test = False
        failure = False

        try:
            self.get_logger().info(f"Changing mode to {self.args.mode}...")
            if not self.call_set_mode(self.args.mode):
                raise RuntimeError(f"mode change failed: {self.args.mode}")
            if not self.wait_for_mode(self.args.mode):
                raise RuntimeError(f"mode state was not confirmed: {self.args.mode}")

            arm_label = "FORCE ARM" if self.args.force_arm else "ARM"
            self.get_logger().info(f"Calling {arm_label}...")
            if not self.call_trigger(self.arm_client, arm_label):
                raise RuntimeError(f"{arm_label} failed")
            armed_by_test = True
            if not self.wait_for_armed(True):
                raise RuntimeError("armed=True state was not confirmed")

            if self.args.hold_sec > 0:
                self.get_logger().info(f"Holding armed state for {self.args.hold_sec:.1f}s...")
                time.sleep(self.args.hold_sec)

        except Exception as exc:
            failure = True
            self.get_logger().error(str(exc))

        finally:
            if armed_by_test or self.args.always_disarm:
                self.get_logger().info("Calling DISARM...")
                if not self.call_trigger(self.disarm_client, "DISARM"):
                    failure = True
                if not self.wait_for_armed(False):
                    failure = True

        if failure:
            self.get_logger().error("Mode/arm/disarm service test failed.")
            return 1

        self.get_logger().info("Mode/arm/disarm service test passed.")
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test /cube set-mode, arm, and disarm services."
    )
    parser.add_argument("--mode", default="GUIDED")
    parser.add_argument("--force-arm", action="store_true")
    parser.add_argument("--always-disarm", action="store_true")
    parser.add_argument("--no-state-check", action="store_true")
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--service-timeout", type=float, default=10.0)
    parser.add_argument("--state-timeout", type=float, default=12.0)
    parser.add_argument("--set-mode-service", default="/cube/set_mode_service")
    parser.add_argument("--arm-service", default="/cube/arm")
    parser.add_argument("--force-arm-service", default="/cube/force_arm")
    parser.add_argument("--disarm-service", default="/cube/disarm")
    parser.add_argument("--state-topic", default="/cube/state")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ModeArmDisarmServiceTest(args)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
