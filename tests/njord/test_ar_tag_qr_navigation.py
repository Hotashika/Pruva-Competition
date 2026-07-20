#!/usr/bin/env python3
"""ZED2i + AR-tag/QR + MAVLink bridge hardware integration test.

Run it directly from this directory::

    python3 test_ar_tag_qr_navigation.py

Or from the repository root::

    python3 tests/njord/test_ar_tag_qr_navigation.py

The test starts ``capture_proc.run_capture`` (ZED2i), starts the existing
``VisionNode`` so ``ar_tag.pt`` localizes the tag and OpenCV decodes its QR,
and uses the existing bridge services/topics for mode, ARM, motion, stop, and
DISARM. Press Ctrl+C to stop; cleanup always sends zero motion and DISARM.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BRIDGE_SCRIPT = REPO_ROOT / "bridge" / "bridge_node.py"
MAVLINK_CONNECTION_MODULE = REPO_ROOT / "bridge" / "mavlink_connection.py"
VISION_SCRIPT = REPO_ROOT / "njord" / "vision" / "vision_node.py"

LEFT = "left"
RIGHT = "right"
STRAIGHT = "straight"

# Values decoded from /home/serhatk/Downloads/qr.pdf on 2026-07-20.
PDF_QR_PAYLOADS = (
    "Middle birth 1",
    "Middle birth 2",
    "Middle parallel",
    "Left birth 1",
    "Right birth 2",
    "Right birth 1/ left birth 2",
    "Left parallel",
    "Right parallel",
)


def normalize_qr_payload(payload: object) -> str:
    """Normalize both the PDF spelling (birth) and mission spelling (berth)."""
    normalized = str(payload or "").strip().lower().replace("-", "_")
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = normalized.replace("birth", "berth")
    normalized = normalized.replace("/_", "/").replace("_/", "/")
    return re.sub(r"_+", "_", normalized).strip("_")


def direction_for_qr(payload: object, target_berth: int = 1) -> Optional[str]:
    """Return the requested motion for one of the QR values in ``qr.pdf``.

    The shared marker means "right of berth 1 / left of berth 2", so its
    direction depends on the selected target berth. Unknown values return
    ``None`` and therefore never produce motion.
    """
    if target_berth not in (1, 2):
        raise ValueError("target_berth must be 1 or 2")

    canonical = normalize_qr_payload(payload)
    shared_marker = "right_berth_1/left_berth_2"
    if canonical == shared_marker:
        return RIGHT if target_berth == 1 else LEFT

    exact_directions = {
        "middle_berth_1": STRAIGHT,
        "middle_berth_2": STRAIGHT,
        "middle_parallel": STRAIGHT,
        "left_berth_1": LEFT,
        "left_parallel": LEFT,
        "right_berth_2": RIGHT,
        "right_parallel": RIGHT,
    }
    return exact_directions.get(canonical)


def _validate_qr_pdf_payload_direction_mapping() -> None:
    """Fail before hardware startup if one of the QR mappings is broken."""
    assert all(direction_for_qr(payload, 1) is not None for payload in PDF_QR_PAYLOADS)
    assert direction_for_qr("Middle birth 1") == STRAIGHT
    assert direction_for_qr("Middle birth 2") == STRAIGHT
    assert direction_for_qr("Middle parallel") == STRAIGHT
    assert direction_for_qr("Left birth 1") == LEFT
    assert direction_for_qr("Left parallel") == LEFT
    assert direction_for_qr("Right birth 2") == RIGHT
    assert direction_for_qr("Right parallel") == RIGHT
    assert direction_for_qr("Right birth 1/ left birth 2", 1) == RIGHT
    assert direction_for_qr("Right birth 1/ left birth 2", 2) == LEFT
    assert direction_for_qr("unknown") is None


@dataclass(frozen=True)
class HardwareConfig:
    target_berth: int = 1
    straight_linear_x: float = 0.35
    turn_linear_x: float = 0.20
    turn_angular_z: float = 0.40
    min_confidence: float = 0.20
    confirmation_frames: int = 3
    qr_timeout_sec: float = 1.0
    bridge_state_timeout_sec: float = 2.0
    camera_ready_timeout_sec: float = 25.0
    vision_ready_timeout_sec: float = 45.0
    bridge_ready_timeout_sec: float = 30.0
    force_arm: bool = False


def _validate_config(config: HardwareConfig) -> None:
    if config.target_berth not in (1, 2):
        raise ValueError("target_berth must be 1 or 2")
    if not 0.0 <= config.straight_linear_x <= 1.0:
        raise ValueError("straight_linear_x must be in [0, 1]")
    if not 0.0 <= config.turn_linear_x <= 1.0:
        raise ValueError("turn_linear_x must be in [0, 1]")
    if not 0.0 < config.turn_angular_z <= 1.0:
        raise ValueError("turn_angular_z must be in (0, 1]")
    if config.confirmation_frames < 1:
        raise ValueError("confirmation_frames must be at least 1")
    if config.qr_timeout_sec <= 0.0:
        raise ValueError("qr_timeout_sec must be positive")
    if not 0.0 <= config.min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")


def _start_capture(config: HardwareConfig):
    """Start the repository's ZED capture process and wait for calibration."""
    from njord.core import capture_proc

    context = get_context("spawn")
    stop_event = context.Event()
    ready_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=capture_proc.run_capture,
        kwargs={
            "stop_event": stop_event,
            "ready_queue": ready_queue,
        },
        daemon=False,
        name="njord-ar-qr-zed-capture",
    )
    process.start()

    try:
        ready = ready_queue.get(timeout=config.camera_ready_timeout_sec)
    except queue.Empty as exc:
        stop_event.set()
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        raise RuntimeError("ZED2i capture_proc did not become ready in time") from exc
    except BaseException:
        # Ctrl+C may arrive while the camera is still opening. The process has
        # not yet been returned to the caller, so it must be cleaned up here.
        stop_event.set()
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        raise

    if "error" in ready:
        stop_event.set()
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        raise RuntimeError(f"ZED2i capture_proc failed: {ready['error']}")

    return process, stop_event, float(ready["fx"]), float(ready["cx"])


def _start_child(script: Path, *arguments: object) -> subprocess.Popen:
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [sys.executable, str(script), *(str(item) for item in arguments)]
    return subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )


def _start_bridge_if_needed(node) -> Optional[subprocess.Popen]:
    """Reuse a running bridge or start the repository bridge + MAVLink stack.

    ``mavlink_connection.py`` is a connection library rather than a standalone
    process. ``bridge_node.py`` imports its ``connect_mavlink`` function and
    owns both the initial MAVLink connection and automatic reconnect attempts.
    """
    if node.clients.set_mode_client.wait_for_service(timeout_sec=0.5):
        node.get_logger().info(
            "Running bridge found. Its MAVLink connection/reconnect loop will be used."
        )
        return None

    if not MAVLINK_CONNECTION_MODULE.is_file():
        raise FileNotFoundError(
            f"MAVLink connection module not found: {MAVLINK_CONNECTION_MODULE}"
        )

    node.get_logger().info(
        "Bridge is not running; starting bridge/bridge_node.py. The bridge will "
        "open MAVLink through bridge/mavlink_connection.py."
    )
    return _start_child(BRIDGE_SCRIPT)


def _stop_child(name: str, process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return

    print(f"[AR-QR TEST] Stopping {name} (PID={process.pid})...")
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _stop_capture(process, stop_event) -> None:
    if stop_event is not None:
        stop_event.set()
    if process is None:
        return
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)


def _run_hardware_test(config: HardwareConfig) -> None:
    """Run until Ctrl+C while keeping all real-vehicle cleanup in ``finally``."""
    from njord.config.vision_config import AR_TAG_MODEL_PATH

    ar_tag_model = Path(AR_TAG_MODEL_PATH)
    _validate_qr_pdf_payload_direction_mapping()
    _validate_config(config)
    if not ar_tag_model.is_file():
        raise FileNotFoundError(f"AR-tag model not found: {ar_tag_model}")

    # Hardware/ROS dependencies stay local so displaying --help remains usable
    # on development machines without ZED or ROS 2 installed.
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    try:
        from rclpy.signals import SignalHandlerOptions
    except ImportError:  # Older ROS 2 releases
        SignalHandlerOptions = None

    from utils.mavlink_utilities import (
        call_set_mode,
        call_trigger_service,
        create_mission_clients,
        parse_bridge_state,
        publish_cmd_vel,
        stop_vehicle,
    )

    class ArTagQrNavigationNode(Node):
        def __init__(self):
            super().__init__("njord_ar_tag_qr_navigation_test")
            self.clients = create_mission_clients(self)
            self.cmd_vel_pub = self.create_publisher(Twist, "/cube/cmd_vel", 10)
            self.active_task_pub = self.create_publisher(
                String, "/mission/active_task", 10
            )
            self.create_subscription(String, "/cube/state", self._state_callback, 10)
            self.create_subscription(
                String,
                "/njord/task3/qr_detections",
                self._qr_callback,
                10,
            )

            self.last_bridge_state = ""
            self.last_bridge_state_time: Optional[float] = None
            self.vision_message_received = False
            self.motion_enabled = False
            self.candidate_payload: Optional[str] = None
            self.candidate_count = 0
            self.confirmed_payload: Optional[str] = None
            self.confirmed_direction: Optional[str] = None
            self.last_detection_time: Optional[float] = None
            self.last_command: Optional[tuple[float, float]] = None
            self.last_active_task_publish = 0.0
            self.last_wait_log = 0.0

        def _state_callback(self, message) -> None:
            self.last_bridge_state = message.data
            self.last_bridge_state_time = time.monotonic()

        def _qr_callback(self, message) -> None:
            self.vision_message_received = True
            try:
                payload = json.loads(message.data)
            except (json.JSONDecodeError, TypeError):
                self._clear_detection("invalid QR JSON")
                return

            detections = payload.get("detections") if isinstance(payload, dict) else None
            if not isinstance(detections, list) or not detections:
                return

            usable = []
            for item in detections:
                if not isinstance(item, dict):
                    continue
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    continue
                qr_payload = item.get("canonical_payload") or item.get("payload")
                direction = direction_for_qr(qr_payload, config.target_berth)
                if confidence >= config.min_confidence and direction is not None:
                    usable.append((confidence, normalize_qr_payload(qr_payload), direction))

            if not usable:
                self._clear_detection("unknown or low-confidence QR payload")
                return

            confidence, canonical, direction = max(usable, key=lambda item: item[0])
            now = time.monotonic()
            self.last_detection_time = now
            if canonical != self.candidate_payload:
                self.candidate_payload = canonical
                self.candidate_count = 1
                self.confirmed_payload = None
                self.confirmed_direction = None
                self._publish_stop()
                self.get_logger().info(
                    f"QR candidate: payload={canonical!r}, direction={direction}, "
                    f"confidence={confidence:.2f} (1/{config.confirmation_frames})"
                )
                return

            self.candidate_count += 1
            if self.candidate_count < config.confirmation_frames:
                return
            if self.confirmed_payload != canonical:
                self.confirmed_payload = canonical
                self.confirmed_direction = direction
                self.get_logger().info(
                    f"QR CONFIRMED: payload={canonical!r}, direction={direction}, "
                    f"samples={self.candidate_count}"
                )

        def _clear_detection(self, reason: str) -> None:
            if self.confirmed_direction is not None:
                self.get_logger().warn(f"QR motion cancelled: {reason}")
            self.candidate_payload = None
            self.candidate_count = 0
            self.confirmed_payload = None
            self.confirmed_direction = None
            self.last_detection_time = None
            self._publish_stop()

        def reset_detection(self) -> None:
            self._clear_detection("vehicle preparation")

        def publish_active_task(self, force: bool = False) -> None:
            now = time.monotonic()
            if not force and now - self.last_active_task_publish < 0.5:
                return
            message = String()
            message.data = "task3"
            self.active_task_pub.publish(message)
            self.last_active_task_publish = now

        def _publish_stop(self) -> None:
            if self.last_command != (0.0, 0.0):
                publish_cmd_vel(self.cmd_vel_pub, 0.0, 0.0)
                self.last_command = (0.0, 0.0)

        def control_step(self) -> None:
            self.publish_active_task()
            if not self.motion_enabled:
                self._publish_stop()
                return

            bridge_state = parse_bridge_state(self.last_bridge_state)
            bridge_state_stale = (
                self.last_bridge_state_time is None
                or time.monotonic() - self.last_bridge_state_time
                > config.bridge_state_timeout_sec
            )
            if (
                bridge_state_stale
                or bridge_state.get("connected") is not True
                or bridge_state.get("armed") is not True
                or str(bridge_state.get("mode", "")).upper() != "GUIDED"
            ):
                self.safe_stop()
                raise RuntimeError(
                    "Bridge failsafe: expected fresh connected=True, armed=True, "
                    f"mode=GUIDED; state={self.last_bridge_state!r}"
                )

            if (
                self.last_detection_time is None
                or time.monotonic() - self.last_detection_time > config.qr_timeout_sec
            ):
                self._clear_detection("QR detection timeout")
                return

            if self.confirmed_direction == STRAIGHT:
                command = (config.straight_linear_x, 0.0)
            elif self.confirmed_direction == RIGHT:
                # This repository's bridge contract defines +angular_z as right.
                command = (config.turn_linear_x, config.turn_angular_z)
            elif self.confirmed_direction == LEFT:
                command = (config.turn_linear_x, -config.turn_angular_z)
            else:
                self._publish_stop()
                return

            publish_cmd_vel(
                self.cmd_vel_pub,
                linear_x=command[0],
                angular_z=command[1],
            )
            if command != self.last_command:
                self.get_logger().info(
                    f"Motion: direction={self.confirmed_direction}, "
                    f"linear_x={command[0]:.2f}, angular_z={command[1]:.2f}"
                )
            self.last_command = command

        def wait_for_vision(self, process: subprocess.Popen) -> None:
            deadline = time.monotonic() + config.vision_ready_timeout_sec
            self.get_logger().info(
                "Waiting for VisionNode task3 heartbeat and ar_tag model startup..."
            )
            while rclpy.ok() and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"VisionNode exited during startup: return_code={process.returncode}"
                    )
                self.publish_active_task(force=True)
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.vision_message_received:
                    self.get_logger().info("VisionNode task3 pipeline is ready.")
                    return
            raise TimeoutError("VisionNode did not publish task3 QR heartbeat in time")

        def wait_for_bridge(self, process: Optional[subprocess.Popen]) -> None:
            clients = (
                self.clients.set_mode_client,
                self.clients.arm_client,
                self.clients.force_arm_client,
                self.clients.disarm_client,
            )
            deadline = time.monotonic() + config.bridge_ready_timeout_sec
            while rclpy.ok() and time.monotonic() < deadline:
                if process is not None and process.poll() is not None:
                    raise RuntimeError(
                        f"bridge_node.py exited during startup: return_code={process.returncode}"
                    )
                services_ready = all(client.service_is_ready() for client in clients)
                state = parse_bridge_state(self.last_bridge_state)
                if services_ready and state.get("connected") is True:
                    self.get_logger().info(
                        f"Bridge ready: {self.last_bridge_state}"
                    )
                    return
                now = time.monotonic()
                if now - self.last_wait_log >= 5.0:
                    self.get_logger().info(
                        f"Waiting for bridge: services_ready={services_ready}, "
                        f"state={self.last_bridge_state!r}"
                    )
                    self.last_wait_log = now
                rclpy.spin_once(self, timeout_sec=0.1)
            raise TimeoutError("MAVLink bridge did not become connected in time")

        def safe_stop(self) -> None:
            self.motion_enabled = False
            stop_vehicle(self.cmd_vel_pub)
            self.last_command = (0.0, 0.0)

    capture_process = None
    capture_stop_event = None
    vision_process = None
    bridge_process = None
    node = None
    rclpy_initialized = False
    bridge_ready = False

    try:
        print("[AR-QR TEST] Opening ZED2i through njord.core.capture_proc...")
        capture_process, capture_stop_event, fx, cx = _start_capture(config)
        print(f"[AR-QR TEST] ZED2i ready: fx={fx:.2f}, cx={cx:.2f}")

        # Keep the ROS context alive during Ctrl+C cleanup so HOLD and DISARM
        # service calls can still complete before rclpy.shutdown().
        if SignalHandlerOptions is None:
            rclpy.init()
        else:
            try:
                rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            except TypeError:  # ROS 2 API compatibility
                rclpy.init()
        signal.signal(signal.SIGINT, signal.default_int_handler)
        rclpy_initialized = True
        node = ArTagQrNavigationNode()

        bridge_process = _start_bridge_if_needed(node)

        node.get_logger().info(
            f"Starting VisionNode with AR model {ar_tag_model}..."
        )
        vision_process = _start_child(VISION_SCRIPT, "--fx", fx, "--cx", cx)

        # Confirm camera -> shared memory -> model -> QR topic is functional
        # before allowing the vehicle to arm.
        node.wait_for_vision(vision_process)
        node.wait_for_bridge(bridge_process)
        bridge_ready = True

        initial_state = parse_bridge_state(node.last_bridge_state)
        if initial_state.get("armed") is True:
            node.get_logger().warn(
                "Vehicle was already armed; stopping and disarming before test."
            )
            node.safe_stop()
            if not call_trigger_service(
                node, node.clients.disarm_client, "PRE-TEST DISARM"
            ):
                raise RuntimeError("Pre-test DISARM failed")

        required_mode = "GUIDED"
        node.get_logger().info(f"Setting required mode: {required_mode}")
        if not call_set_mode(node, node.clients.set_mode_client, required_mode):
            raise RuntimeError(f"Could not switch vehicle to {required_mode}")

        arm_client = (
            node.clients.force_arm_client if config.force_arm else node.clients.arm_client
        )
        arm_name = "FORCE ARM" if config.force_arm else "ARM"
        if not call_trigger_service(node, arm_client, arm_name):
            raise RuntimeError(f"{arm_name} failed")

        node.reset_detection()
        node.motion_enabled = True
        node.get_logger().warn(
            "REAL VEHICLE ACTIVE. Show an AR-tag/QR to steer; press Ctrl+C to stop "
            "and DISARM. Unknown, unconfirmed, or stale QR values stop the vehicle."
        )

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.control_step()

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Ctrl+C received; safe shutdown starting.")
    finally:
        if node is not None:
            try:
                node.safe_stop()
            except Exception as exc:
                print(f"[AR-QR TEST] Stop command failed: {exc!r}")

            if bridge_ready:
                try:
                    call_set_mode(
                        node,
                        node.clients.set_mode_client,
                        "HOLD",
                        timeout_sec=2.0,
                    )
                except Exception as exc:
                    node.get_logger().warn(f"Shutdown HOLD failed: {exc!r}")
                try:
                    if not call_trigger_service(
                        node,
                        node.clients.disarm_client,
                        "DISARM",
                        timeout_sec=4.0,
                    ):
                        node.get_logger().error("Shutdown DISARM was not confirmed.")
                except Exception as exc:
                    node.get_logger().error(f"Shutdown DISARM failed: {exc!r}")
                try:
                    node.safe_stop()
                except Exception:
                    pass

        # Vision must detach shared memory before capture unlinks it.
        _stop_child("VisionNode", vision_process)
        _stop_capture(capture_process, capture_stop_event)
        _stop_child("bridge_node.py", bridge_process)

        if node is not None:
            node.destroy_node()
        if rclpy_initialized and rclpy.ok():
            rclpy.shutdown()
        print("[AR-QR TEST] Camera, vision, motion, and bridge cleanup complete.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open ZED2i, run ar_tag.pt + QR decoding, arm through the MAVLink "
            "bridge, and steer left/right/straight until Ctrl+C."
        )
    )
    parser.add_argument("--target-berth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--straight-linear-x", type=float, default=0.35)
    parser.add_argument("--turn-linear-x", type=float, default=0.20)
    parser.add_argument("--turn-angular-z", type=float, default=0.40)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--confirmation-frames", type=int, default=3)
    parser.add_argument("--qr-timeout-sec", type=float, default=1.0)
    parser.add_argument(
        "--force-arm",
        action="store_true",
        help="Use /cube/force_arm instead of the safer normal /cube/arm service.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_hardware_test(
        HardwareConfig(
            target_berth=args.target_berth,
            straight_linear_x=args.straight_linear_x,
            turn_linear_x=args.turn_linear_x,
            turn_angular_z=args.turn_angular_z,
            min_confidence=args.min_confidence,
            confirmation_frames=args.confirmation_frames,
            qr_timeout_sec=args.qr_timeout_sec,
            force_arm=args.force_arm,
        )
    )


if __name__ == "__main__":
    main()
