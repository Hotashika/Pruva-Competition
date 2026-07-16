#!/usr/bin/env python3

import argparse
from collections import deque
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from utils.mavlink_utilities import (
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    parse_bridge_state,
    publish_cmd_vel,
    stop_vehicle,
)

BRIDGE_SCRIPT = REPO_ROOT / "bridge" / "bridge_node.py"
DEFAULT_SECONDS = 5.0
DEFAULT_LINEAR_X = 0.5
DEFAULT_ANGULAR_Z = 0.0
DEFAULT_MODE = "GUIDED"
DEFAULT_BRIDGE_READY_TIMEOUT = 0.0
BRIDGE_WAIT_LOG_INTERVAL = 5.0
BRIDGE_PROCESS_STOP_TIMEOUT = 5.0


class BridgeMotorTestNode(Node):
    def __init__(self):
        super().__init__("bridge_motor_test_node")
        self.mission_clients = create_mission_clients(self)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cube/cmd_vel", 10)
        self.last_bridge_state = None
        self.last_bridge_error = None
        self.bridge_diagnostics = deque(maxlen=100)
        self._state_sub = self.create_subscription(
            String,
            "/cube/state",
            self._state_callback,
            10,
        )
        self._error_sub = self.create_subscription(
            String,
            "/cube/error",
            self._error_callback,
            10,
        )
        self._diagnostics_sub = self.create_subscription(
            String,
            "/cube/diagnostics",
            self._diagnostics_callback,
            10,
        )

    def _state_callback(self, msg):
        if msg.data != self.last_bridge_state:
            self.get_logger().info(
                f'BRIDGE STATE RX /cube/state: raw={msg.data!r}'
            )
        self.last_bridge_state = msg.data

    def _error_callback(self, msg):
        self.last_bridge_error = msg.data
        self.get_logger().error(
            f'BRIDGE ERROR RX /cube/error: raw={msg.data!r}'
        )

    def _diagnostics_callback(self, msg):
        self.bridge_diagnostics.append(msg.data)
        self.get_logger().info(
            f'BRIDGE RAW RX /cube/diagnostics: {msg.data}'
        )

    def wait_for_bridge_ready(self, timeout_sec=DEFAULT_BRIDGE_READY_TIMEOUT):
        timeout_sec = max(0.0, float(timeout_sec))
        deadline = (
            time.monotonic() + timeout_sec
            if timeout_sec > 0.0
            else None
        )
        next_wait_log = time.monotonic()

        if deadline is None:
            self.get_logger().info(
                'Bridge MAVLink baglantisi bekleniyor. Testin baslattigi bridge '
                'connected=True oldugunda test otomatik devam edecek. Durdurmak '
                'icin Ctrl+C kullanin.'
            )
        else:
            self.get_logger().info(
                f'Bridge MAVLink baglantisi en fazla {timeout_sec:.1f}s beklenecek.'
            )

        while rclpy.ok():
            state = parse_bridge_state(self.last_bridge_state or '')
            if state.get('connected') is True:
                self.get_logger().info(
                    f'Bridge MAVLink baglantisi hazir: {self.last_bridge_state}'
                )
                return True

            now = time.monotonic()
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0.0:
                    break
                spin_timeout = min(0.1, remaining)
            else:
                spin_timeout = 0.1

            if now >= next_wait_log:
                self.get_logger().info(
                    'Bridge bekleniyor: '
                    f'last_bridge_state={self.last_bridge_state!r}'
                )
                next_wait_log = now + BRIDGE_WAIT_LOG_INTERVAL

            rclpy.spin_once(self, timeout_sec=spin_timeout)

        if self.last_bridge_state is None:
            self.get_logger().error(
                f'/cube/state {timeout_sec:.1f}s icinde alinmadi. Servis sunucusu '
                'var olsa bile bridge telemetri timeri calismiyor veya eski bridge '
                'surumu calisiyor olabilir.'
            )
        else:
            self.get_logger().error(
                f'Bridge servisleri bulundu ancak MAVLink baglantisi '
                f'{timeout_sec:.1f}s icinde hazir olmadi: '
                f'last_bridge_state={self.last_bridge_state!r}'
            )
        return False

    def collect_diagnostics(self, duration_sec=0.5):
        deadline = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, remaining)))

    def log_failure_summary(self, action):
        self.collect_diagnostics()
        self.get_logger().error(f'{action} ICIN TESHIS OZETI')
        self.get_logger().error(
            f'last_bridge_state={self.last_bridge_state!r}'
        )
        self.get_logger().error(
            f'last_bridge_error={self.last_bridge_error!r}'
        )
        self.get_logger().error(
            f'raw_diagnostic_count={len(self.bridge_diagnostics)}, '
            f'last_raw={self.bridge_diagnostics[-1]!r}'
            if self.bridge_diagnostics
            else 'raw_diagnostic_count=0, last_raw=None'
        )

        try:
            cube_services = [
                f'{name}:{"|".join(types)}'
                for name, types in self.get_service_names_and_types()
                if name.startswith('/cube/')
            ]
            self.get_logger().error(
                'ROS graph /cube services=' + (
                    ', '.join(sorted(cube_services)) if cube_services else '<none>'
                )
            )
        except Exception as exc:
            self.get_logger().warn(f'ROS servis grafigi okunamadi: {exc!r}')

        if self.last_bridge_state and 'connected=False' in self.last_bridge_state:
            self.get_logger().error(
                'Muhtemel neden: ROS bridge servisleri acik, fakat bridge Orange '
                'Cube MAVLink baglantisini/heartbeatini hazir gormuyor.'
            )
        elif self.last_bridge_error is None and not self.bridge_diagnostics:
            self.get_logger().error(
                'Bridge ayrinti topiclerinden veri gelmedi. Bridge prosesini yeni '
                'kodla yeniden baslatin ve /orange_cube_bridge nodeunu kontrol edin.'
            )


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
        "--bridge-ready-timeout",
        type=float,
        default=DEFAULT_BRIDGE_READY_TIMEOUT,
        help=(
            "Maximum seconds to wait for /cube/state connected=True after the "
            "ROS services become available. Use 0 (default) to wait until the "
            "user starts bridge_node.py and the MAVLink connection is ready."
        ),
    )
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


def start_bridge_if_needed(node):
    set_mode_client = node.mission_clients.set_mode_client
    if set_mode_client.wait_for_service(timeout_sec=0.5):
        node.get_logger().info(
            'Calisan bridge servisi bulundu; ikinci bridge prosesi baslatilmayacak.'
        )
        return None

    if not BRIDGE_SCRIPT.is_file():
        raise RuntimeError(f'Bridge dosyasi bulunamadi: {BRIDGE_SCRIPT}')

    node.get_logger().info(
        f'Bridge servisi bulunamadi; otomatik baslatiliyor: {BRIDGE_SCRIPT}'
    )
    try:
        return subprocess.Popen(
            [sys.executable, str(BRIDGE_SCRIPT)],
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        raise RuntimeError(
            f'bridge_node.py otomatik baslatilamadi: {exc!r}'
        ) from exc


def wait_for_bridge_services(node, bridge_process):
    services = (
        ('/cube/set_mode_service', node.mission_clients.set_mode_client),
        ('/cube/arm', node.mission_clients.arm_client),
        ('/cube/force_arm', node.mission_clients.force_arm_client),
        ('/cube/disarm', node.mission_clients.disarm_client),
    )
    next_wait_log = 0.0

    while rclpy.ok():
        missing = [name for name, client in services if not client.service_is_ready()]
        if not missing:
            node.get_logger().info('Bridge servisleri hazir.')
            return

        if bridge_process is not None:
            return_code = bridge_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    'Otomatik baslatilan bridge_node.py servisler hazir olmadan '
                    f'kapandi: return_code={return_code}'
                )

        now = time.monotonic()
        if now >= next_wait_log:
            node.get_logger().info(
                'Bridge servisleri bekleniyor: ' + ', '.join(missing)
            )
            next_wait_log = now + BRIDGE_WAIT_LOG_INTERVAL
        rclpy.spin_once(node, timeout_sec=0.1)

    raise RuntimeError('ROS kapanirken bridge servisleri beklenemedi.')


def stop_started_bridge(node, bridge_process):
    if bridge_process is None or bridge_process.poll() is not None:
        return

    node.get_logger().info('Testin baslattigi bridge prosesi kapatiliyor...')
    bridge_process.terminate()
    try:
        bridge_process.wait(timeout=BRIDGE_PROCESS_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        node.get_logger().warn(
            'Bridge prosesi zamaninda kapanmadi; zorla sonlandiriliyor.'
        )
        bridge_process.kill()
        bridge_process.wait()


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
    disarm_required = False
    bridge_process = None

    try:
        node.get_logger().info(
            'Test hazir. Gerekiyorsa bridge_node.py otomatik baslatilacak; mod, '
            'ARM, hareket ve DISARM adimlari otomatik uygulanacak.'
        )
        bridge_process = start_bridge_if_needed(node)
        wait_for_bridge_services(node, bridge_process)
        if not node.wait_for_bridge_ready(args.bridge_ready_timeout):
            node.log_failure_summary("BRIDGE READY")
            raise RuntimeError("Bridge services are available but MAVLink is not ready.")

        initial_state = parse_bridge_state(node.last_bridge_state or '')
        if initial_state.get('armed') is True:
            disarm_required = True
            node.get_logger().warn(
                'Arac test baslangicinda zaten ARMED. Guvenli baslangic icin '
                'hareket sifirlanip once DISARM dogrulanacak.'
            )
            stop_vehicle(node.cmd_vel_pub)
            if call_trigger_service(
                node,
                node.mission_clients.disarm_client,
                "PRE-TEST DISARM",
            ) is False:
                node.log_failure_summary("PRE-TEST DISARM")
                raise RuntimeError(
                    "Vehicle was already armed and pre-test DISARM failed."
                )
            disarm_required = False

        node.get_logger().info(f"Setting vehicle to {args.mode} mode...")
        if call_set_mode(node, node.mission_clients.set_mode_client, args.mode) is False:
            node.log_failure_summary(f"SET_MODE {args.mode}")
            raise RuntimeError(f"Failed to switch to {args.mode} mode.")

        arm_client = (
            node.mission_clients.arm_client
            if args.normal_arm
            else node.mission_clients.force_arm_client
        )
        arm_label = "ARM" if args.normal_arm else "FORCE ARM"
        node.get_logger().info(f"{arm_label} requested...")
        if call_trigger_service(node, arm_client, arm_label) is False:
            node.log_failure_summary(arm_label)
            raise RuntimeError(f"{arm_label} failed.")
        disarm_required = True

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

        if disarm_required and not args.no_disarm:
            call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
            stop_vehicle(node.cmd_vel_pub)

        if args.no_disarm and bridge_process is not None:
            node.get_logger().warn(
                '--no-disarm kullanildigi icin otomatik baslatilan bridge prosesi '
                'acik birakiliyor; bridge kapanirken shutdown DISARM uygular.'
            )
        else:
            stop_started_bridge(node, bridge_process)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
