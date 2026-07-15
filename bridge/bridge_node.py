#!/usr/bin/env python3

import math
import os
import signal
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from pymavlink import mavutil

from sensor_msgs.msg import Imu, NavSatFix, BatteryState
from std_msgs.msg import String, Float32

from bridge.mavlink_connection import (
    DEFAULT_BAUD,
    DEFAULT_CONNECTION_STRING,
    DEFAULT_HEARTBEAT_TIMEOUT,
    connect_mavlink,
)
from utils.mavlink_utilities import create_bridge_topics, create_bridge_services

MAV_CMD_MISSION_START = mavutil.mavlink.MAV_CMD_USER_1


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def set_position(connection, destination, boot_time):
    """Send position command (global relative alt)."""
    time_boot_ms = int(1000 * (time.time() - boot_time))
    mask = 0b110111111100
    lat = int(destination[0] * 1e7)
    lon = int(destination[1] * 1e7)
    connection.mav.send(
        mavutil.mavlink.MAVLink_set_position_target_global_int_message(
            time_boot_ms,
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mask,
            lat, lon, 20,
            0, 0, 0,
            0, 0, 0,
            1.57, 0.5
        )
    )


class OrangeCubeBridgeNode(Node):
    def __init__(self):
        super().__init__("orange_cube_bridge")

        self.declare_parameter(
            "connection_string",
            os.getenv("MAVLINK_CONNECTION_STRING", DEFAULT_CONNECTION_STRING),
        )
        self.declare_parameter("baud", int(os.getenv("MAVLINK_BAUD", str(DEFAULT_BAUD))))
        self.declare_parameter(
            "heartbeat_timeout",
            int(os.getenv("MAVLINK_HEARTBEAT_TIMEOUT", str(DEFAULT_HEARTBEAT_TIMEOUT))),
        )
        self.declare_parameter(
            "connection_timeout_sec",
            float(os.getenv("MAVLINK_CONNECTION_TIMEOUT", "5.0")),
        )
        self.declare_parameter(
            "reconnect_interval_sec",
            float(os.getenv("MAVLINK_RECONNECT_INTERVAL", "3.0")),
        )
        self.declare_parameter(
            "reconnect_heartbeat_timeout",
            float(os.getenv("MAVLINK_RECONNECT_HEARTBEAT_TIMEOUT", "3.0")),
        )
        self.declare_parameter(
            "command_confirmation_timeout_sec",
            float(os.getenv("MAVLINK_COMMAND_CONFIRM_TIMEOUT", "4.0")),
        )
        self.declare_parameter(
            "disarm_on_shutdown",
            os.getenv("MAVLINK_DISARM_ON_SHUTDOWN", "1").lower()
            not in ("0", "false", "no", "off"),
        )
        self.declare_parameter(
            "mission_launch_enabled",
            os.getenv("MAVLINK_MISSION_LAUNCH_ENABLED", "0").lower()
            in ("1", "true", "yes", "on"),
        )

        self.connection_string = self.get_parameter("connection_string").value
        self.baud = int(self.get_parameter("baud").value)
        self.heartbeat_timeout = int(self.get_parameter("heartbeat_timeout").value)
        self.connection_timeout_sec = float(self.get_parameter("connection_timeout_sec").value)
        self.reconnect_interval_sec = float(self.get_parameter("reconnect_interval_sec").value)
        self.reconnect_heartbeat_timeout = float(
            self.get_parameter("reconnect_heartbeat_timeout").value
        )
        self.command_confirmation_timeout_sec = float(
            self.get_parameter("command_confirmation_timeout_sec").value
        )
        self.disarm_on_shutdown = bool(self.get_parameter("disarm_on_shutdown").value)
        self.mission_launch_enabled = bool(
            self.get_parameter("mission_launch_enabled").value
        )

        self.master = None
        self.connected = False
        self.armed = False
        self.mode = "UNKNOWN"
        self.last_heartbeat_time = 0.0
        self.last_connection_attempt = 0.0
        self.connection_lost_reported = False
        self.cmd_vel_ignored_reported = False

        self.gps_lat = None
        self.gps_lon = None
        self.gps_alt = None
        self.relative_alt = None
        self.heading_deg = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.voltage_v = None
        self.current_a = None
        self.battery_remaining = None

        self.boot_time = time.time()
        self.last_cmd_vel_time = 0.0
        self.cmd_timeout_sec = 0.5
        self.last_target_q = self._yaw_to_mavlink_quaternion(0.0)
        self.last_thrust = 0.0
        self.last_attitude_tx_active = False
        self.last_position_target_time = 0.0
        self.position_target_timeout_sec = 0.5
        self.active_mission_name = None
        self.active_mission_process = None
        self.mission_paths = self._load_mission_paths_from_env()
        self.mission_waypoint_paths = self._load_mission_waypoint_paths_from_env()
        self.mission_sequence = self._load_mission_sequence_from_env()
        self.active_sequence = []
        self.active_sequence_index = 0

        self.topics = create_bridge_topics(
            self,
            self._cmd_vel_callback,
            self._set_position_callback,
        )
        self.bridge_services = create_bridge_services(
            self,
            self._set_mode_callback,
            self._arm_callback,
            self._force_arm_callback,
            self._disarm_callback,
        )

        self.mission_command_pub = self.create_publisher(
            String,
            "/mission_command",
            10,
        )

        # self._connect()

        self.create_timer(1.0, self._connect_if_needed)
        self.create_timer(0.02, self._read_mavlink_messages)
        self.create_timer(0.2, self._publish_telemetry)
        self.create_timer(0.1, self._send_attitude_target_loop)
        self.create_timer(1.0, self._connection_watchdog)
        self.create_timer(1.0, self._send_companion_heartbeat)
        self.create_timer(1.0, self._mission_process_watchdog)

        if self.mission_launch_enabled:
            configured = ", ".join(
                f"M{number}={path}"
                for number, path in sorted(self.mission_paths.items())
            )
            waypoint_configured = ", ".join(
                f"M{number}={path}"
                for number, path in sorted(self.mission_waypoint_paths.items())
            )
            sequence_text = (
                " sequence=" + ",".join(f"M{number}" for number in self.mission_sequence)
                if self.mission_sequence
                else ""
            )
            self.get_logger().info(
                f"MAVLink mission launch aktif. Komut: MAV_CMD_USER_1 param1=1..4.{sequence_text} {configured}"
            )
            if waypoint_configured:
                self.get_logger().info(
                    f"Pixhawk waypoint sync aktif: {waypoint_configured}"
                )
        else:
            self.get_logger().info(
                "MAVLink mission launch pasif. Sadece /mission_command yayinlanacak."
            )

        self.get_logger().info("/cube topic ve servisleri aktif.")

    @staticmethod
    def _load_mission_paths_from_env():
        paths = {}
        for number in range(1, 5):
            path = os.getenv(f"MAVLINK_MISSION_{number}_PATH", "").strip()
            if path:
                paths[number] = path
        return paths

    @staticmethod
    def _load_mission_waypoint_paths_from_env():
        paths = {}
        for number in range(1, 5):
            path = os.getenv(f"MAVLINK_MISSION_{number}_WAYPOINT_PATH", "").strip()
            if path:
                paths[number] = path
        return paths

    @staticmethod
    def _load_mission_sequence_from_env():
        raw = os.getenv("MAVLINK_MISSION_SEQUENCE", "").strip()
        if not raw:
            return []

        sequence = []
        for part in raw.split(","):
            text = part.strip().upper()
            if text.startswith("M"):
                text = text[1:]
            try:
                number = int(text)
            except ValueError:
                continue
            if 1 <= number <= 4 and number not in sequence:
                sequence.append(number)
        return sequence

    def _send_command_ack(self, command, result):
        if self.master is None:
            return

        try:
            self.master.mav.command_ack_send(command, result)
        except Exception as exc:
            self.get_logger().warn(f"COMMAND_ACK gonderilemedi: {exc}")

    def _stop_active_mission(self, timeout_sec=7.0):
        process = self.active_mission_process
        mission_name = self.active_mission_name
        if process is None:
            return

        if process.poll() is not None:
            self.active_mission_process = None
            self.active_mission_name = None
            return

        self.get_logger().info(f"{mission_name} gorevi durduruluyor...")
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except AttributeError:
            process.send_signal(signal.SIGINT)

        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"{mission_name} SIGINT ile kapanmadi, SIGTERM gonderiliyor.")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except AttributeError:
                process.terminate()
            process.wait(timeout=2)

        self.active_mission_process = None
        self.active_mission_name = None
        self.active_sequence = []
        self.active_sequence_index = 0

    def _start_mission_process(self, mission_number):
        mission_name = f"M{mission_number}"

        mission_path = self.mission_paths.get(mission_number)
        if not mission_path:
            self.get_logger().warn(f"{mission_name} icin mission path tanimli degil.")
            return False

        if not os.path.isfile(mission_path):
            self.get_logger().error(f"{mission_name} script bulunamadi: {mission_path}")
            return False

        if (
            self.active_mission_process is not None
            and self.active_mission_process.poll() is None
        ):
            if self.active_mission_name == mission_name:
                self.get_logger().warn(f"{mission_name} zaten calisiyor; ikinci kez baslatilmadi.")
                return True
            self._stop_active_mission()

        try:
            self.active_mission_process = subprocess.Popen(
                [sys.executable, mission_path],
                start_new_session=True,
            )
            self.active_mission_name = mission_name
            self.get_logger().info(
                f"{mission_name} Jetson uzerinde baslatildi: PID={self.active_mission_process.pid}"
            )
            return True
        except Exception as exc:
            self.active_mission_process = None
            self.active_mission_name = None
            self._publish_error(f"{mission_name} baslatilamadi: {exc}")
            return False

    def _request_mission_item(self, target_system, target_component, seq):
        try:
            self.master.mav.mission_request_int_send(
                target_system,
                target_component,
                seq,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except AttributeError:
            self.master.mav.mission_request_send(target_system, target_component, seq)
        except TypeError:
            self.master.mav.mission_request_int_send(target_system, target_component, seq)

    def _read_pixhawk_mission_waypoints(self):
        if not self._has_valid_link():
            raise RuntimeError("MAVLink baglantisi hazir degil.")

        target_system = self.master.target_system
        target_component = self.master.target_component

        try:
            self.master.mav.mission_request_list_send(
                target_system,
                target_component,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            self.master.mav.mission_request_list_send(target_system, target_component)

        count_msg = self.master.recv_match(
            type="MISSION_COUNT",
            blocking=True,
            timeout=5.0,
        )
        if count_msg is None:
            raise RuntimeError("Pixhawk MISSION_COUNT gondermedi.")

        count = int(getattr(count_msg, "count", 0) or 0)
        if count <= 0:
            raise RuntimeError("Pixhawk mission listesi bos.")

        waypoints = []
        for seq in range(count):
            self._request_mission_item(target_system, target_component, seq)
            item_msg = None
            deadline = time.time() + 5.0
            while time.time() < deadline:
                candidate = self.master.recv_match(
                    type=("MISSION_ITEM_INT", "MISSION_ITEM"),
                    blocking=True,
                    timeout=1.0,
                )
                if candidate is None:
                    continue
                if int(getattr(candidate, "seq", -1)) == seq:
                    item_msg = candidate
                    break

            if item_msg is None:
                raise RuntimeError(f"Pixhawk mission item {seq} gondermedi.")

            command = int(
                getattr(
                    item_msg,
                    "command",
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                )
            )
            if command != mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
                continue

            if item_msg.get_type() == "MISSION_ITEM_INT":
                lat = float(getattr(item_msg, "x", 0)) / 1e7
                lon = float(getattr(item_msg, "y", 0)) / 1e7
            else:
                lat = float(getattr(item_msg, "x", getattr(item_msg, "lat", 0.0)))
                lon = float(getattr(item_msg, "y", getattr(item_msg, "lon", 0.0)))

            alt = float(getattr(item_msg, "z", 0.0) or 0.0)
            if abs(lat) < 1e-6 and abs(lon) < 1e-6:
                continue

            # GUI upload path adds a HOME item at seq=0. Mission algorithms should
            # consume only the route coordinates selected in the GUI.
            if seq == 0 and count > 1:
                continue

            waypoints.append({"lat": lat, "lon": lon, "alt": alt})

        if not waypoints:
            raise RuntimeError("Pixhawk mission icinde gecerli waypoint yok.")

        return waypoints

    @staticmethod
    def _write_qgc_waypoint_file(path, waypoints):
        lines = ["QGC WPL 110\n"]
        lines.append("0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\n")
        for index, waypoint in enumerate(waypoints, start=1):
            lines.append(
                "{seq}\t0\t3\t16\t0.00000000\t0.00000000\t0.00000000\t0.00000000\t"
                "{lat:.8f}\t{lon:.8f}\t{alt:.6f}\t1\n".format(
                    seq=index,
                    lat=float(waypoint["lat"]),
                    lon=float(waypoint["lon"]),
                    alt=float(waypoint.get("alt", 0.0) or 0.0),
                )
            )

        with open(path, "w", encoding="utf-8", newline="") as waypoint_file:
            waypoint_file.writelines(lines)

    def _sync_pixhawk_waypoints_for_missions(self, mission_numbers):
        waypoint_paths = [
            self.mission_waypoint_paths[number]
            for number in mission_numbers
            if number in self.mission_waypoint_paths
        ]
        if not waypoint_paths:
            return True

        try:
            waypoints = self._read_pixhawk_mission_waypoints()
            for waypoint_path in waypoint_paths:
                self._write_qgc_waypoint_file(waypoint_path, waypoints)
                self.get_logger().info(
                    f"Pixhawk mission waypointleri yazildi: {waypoint_path} ({len(waypoints)} WP)"
                )
            return True
        except Exception as exc:
            self._publish_error(f"Pixhawk waypoint sync basarisiz: {exc}")
            return False

    def _launch_mission(self, mission_number):
        mission_name = f"M{mission_number}"

        if not self.mission_launch_enabled:
            self.get_logger().info(
                f"{mission_name} MAVLink komutu alindi; mission launch pasif oldugu icin sadece yayinlandi."
            )
            return True

        if self.mission_sequence:
            self._stop_active_mission()
            self.active_sequence = list(self.mission_sequence)
            self.active_sequence_index = 0
            first_mission_number = self.active_sequence[self.active_sequence_index]
            if not self._sync_pixhawk_waypoints_for_missions(self.active_sequence):
                self.active_sequence = []
                self.active_sequence_index = 0
                return False
            self.get_logger().info(
                f"{mission_name} komutu alindi; sirali gorev M{first_mission_number}'den baslatiliyor."
            )
            return self._start_mission_process(first_mission_number)

        if not self._sync_pixhawk_waypoints_for_missions([mission_number]):
            return False

        return self._start_mission_process(mission_number)

    def _mission_process_watchdog(self):
        process = self.active_mission_process
        if process is None:
            return

        return_code = process.poll()
        if return_code is None:
            return

        finished_mission = self.active_mission_name
        self.get_logger().info(
            f"{finished_mission} process bitti. return_code={return_code}"
        )
        self.active_mission_process = None
        self.active_mission_name = None

        if not self.active_sequence:
            return

        if return_code != 0:
            self.get_logger().error(
                f"{finished_mission} hata kodu ile bitti; sirali gorev durduruldu."
            )
            self.active_sequence = []
            self.active_sequence_index = 0
            return

        self.active_sequence_index += 1
        if self.active_sequence_index >= len(self.active_sequence):
            self.get_logger().info("Sirali gorevlerin tamami bitti.")
            self.active_sequence = []
            self.active_sequence_index = 0
            return

        next_mission_number = self.active_sequence[self.active_sequence_index]
        self.get_logger().info(f"Siradaki gorev baslatiliyor: M{next_mission_number}")
        if not self._start_mission_process(next_mission_number):
            self.get_logger().error(
                f"M{next_mission_number} baslatilamadi; sirali gorev durduruldu."
            )
            self.active_sequence = []
            self.active_sequence_index = 0

    def _publish_error(self, text):
        msg = String()
        msg.data = str(text)
        self.topics.error_pub.publish(msg)
        self.get_logger().error(str(text))

    def _connect(self):
        self.last_connection_attempt = time.time()
        try:
            self._close_master()
            self.master = connect_mavlink(
                connection_string=self.connection_string,
                baud=self.baud,
                heartbeat_timeout=self.heartbeat_timeout,
                logger=self.get_logger(),
            )
            self.connected = True
            self.last_heartbeat_time = time.time()
            self.connection_lost_reported = False
            self.cmd_vel_ignored_reported = False
            self._request_data_streams()
        except Exception as exc:
            self.connected = False
            self.master = None
            self.last_heartbeat_time = 0.0
            self._reset_vehicle_state()
            self._neutralize_outputs()
            self._publish_error(f"MAVLink baglanti hatasi: {exc}")

    def _connect_if_needed(self):
        if self.master is not None and self.connected:
            return

        try:
            self._connect()
        except Exception as exc:
            self.connected = False
            self.get_logger().warn(f"MAVLink reconnect waiting: {exc}")

    def _neutralize_outputs(self):
        self.last_thrust = 0.0
        self.last_attitude_tx_active = False
        self.last_position_target_time = 0.0
        if self.yaw is not None:
            self.last_target_q = self._yaw_to_mavlink_quaternion(self.yaw)
        self.last_cmd_vel_time = 0.0

    @staticmethod
    def _yaw_to_mavlink_quaternion(yaw_rad):
        qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, yaw_rad)
        return (qw, qx, qy, qz)

    def _reset_vehicle_state(self):
        self.armed = False
        self.mode = "UNKNOWN"
        self.gps_lat = None
        self.gps_lon = None
        self.gps_alt = None
        self.relative_alt = None
        self.heading_deg = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.voltage_v = None
        self.current_a = None
        self.battery_remaining = None

    def _close_master(self):
        if self.master is None:
            return
        try:
            self.master.close()
        except Exception:
            pass
        finally:
            self.master = None

    def _has_valid_link(self):
        return (
                self.master is not None
                and self.connected
                and self.master.target_system not in (None, 0)
                and self.master.target_component not in (None, 0)
        )

    def _has_valid_gps(self):
        if self.gps_lat is None or self.gps_lon is None:
            return False
        return abs(self.gps_lat) > 1e-6 or abs(self.gps_lon) > 1e-6

    def _message_from_target(self, msg):
        if self.master is None or self.master.target_system in (None, 0):
            return False
        if not hasattr(msg, "get_srcSystem"):
            return True
        if msg.get_srcSystem() != self.master.target_system:
            return False
        if not hasattr(msg, "get_srcComponent"):
            return True
        return msg.get_srcComponent() == self.master.target_component

    @staticmethod
    def _normalize_mode_name(mode_name):
        return str(mode_name or "UNKNOWN").strip().upper()

    @staticmethod
    def _mav_result_name(result):
        try:
            return mavutil.mavlink.enums["MAV_RESULT"][int(result)].name
        except (KeyError, TypeError, ValueError, AttributeError):
            return f"MAV_RESULT_{result}"

    def _update_vehicle_state_from_heartbeat(self, msg, source="heartbeat"):
        previous_state = (self.connected, self.armed, self.mode)
        self.mode = mavutil.mode_string_v10(msg)
        self.armed = bool(
            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        self.last_heartbeat_time = time.time()
        self.connected = True
        self.connection_lost_reported = False
        self.cmd_vel_ignored_reported = False

        current_state = (self.connected, self.armed, self.mode)
        if current_state != previous_state:
            self.get_logger().info(
                "Orange Cube state transition "
                f"({source}): connected={self.connected}, "
                f"armed={self.armed}, mode={self.mode}"
            )

    def _log_command_ack(self, msg):
        command = int(getattr(msg, "command", -1))
        result = int(getattr(msg, "result", -1))
        result_name = self._mav_result_name(result)
        log_text = f"MAVLink RX COMMAND_ACK: command={command}, result={result_name}({result})"
        accepted_results = {
            mavutil.mavlink.MAV_RESULT_ACCEPTED,
            mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
        }
        if result in accepted_results:
            self.get_logger().info(log_text)
        else:
            self.get_logger().warn(log_text)
        return command, result

    def _wait_for_vehicle_state(
            self,
            predicate,
            description,
            expected_command=None,
            timeout_sec=None,
    ):
        timeout_sec = (
            self.command_confirmation_timeout_sec
            if timeout_sec is None
            else float(timeout_sec)
        )
        deadline = time.time() + timeout_sec
        self.get_logger().info(
            f"Orange Cube confirmation waiting: {description}, timeout={timeout_sec:.1f}s"
        )

        while time.time() < deadline:
            if predicate():
                self.get_logger().info(
                    f"Orange Cube confirmation received: {description}"
                )
                return True

            msg = self.master.recv_match(
                type=["HEARTBEAT", "COMMAND_ACK", "STATUSTEXT"],
                blocking=True,
                timeout=min(0.25, max(0.0, deadline - time.time())),
            )
            if msg is None or not self._message_from_target(msg):
                continue

            msg_type = msg.get_type()
            if msg_type == "HEARTBEAT":
                self._update_vehicle_state_from_heartbeat(msg, source="command confirmation")
                continue

            if msg_type == "STATUSTEXT":
                self._log_statustext(msg)
                continue

            command, result = self._log_command_ack(msg)
            if expected_command is None or command != expected_command:
                continue

            accepted_results = {
                mavutil.mavlink.MAV_RESULT_ACCEPTED,
                mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
            }
            if result not in accepted_results:
                self.get_logger().error(
                    f"Orange Cube command rejected before state confirmation: {description}"
                )
                return False

        self.get_logger().error(
            "Orange Cube confirmation timeout: "
            f"expected={description}, connected={self.connected}, "
            f"armed={self.armed}, mode={self.mode}"
        )
        return False

    def _vehicle_ready_for_guided_motion(self, action_name):
        mode_name = self._normalize_mode_name(self.mode)
        if self.armed and mode_name == "GUIDED":
            return True

        self.get_logger().warn(
            f"{action_name} ignored: Orange Cube is not motion-ready "
            f"(connected={self.connected}, armed={self.armed}, mode={self.mode}).",
            throttle_duration_sec=1.0,
        )
        return False

    def _try_reconnect(self):
        now = time.time()
        if now - self.last_connection_attempt < self.reconnect_interval_sec:
            return

        self.last_connection_attempt = now
        self.get_logger().info("MAVLink yeniden baglanti deneniyor...")
        try:
            self._close_master()
            self.master = connect_mavlink(
                connection_string=self.connection_string,
                baud=self.baud,
                heartbeat_timeout=self.reconnect_heartbeat_timeout,
                logger=self.get_logger(),
            )
            self.connected = True
            self.last_heartbeat_time = time.time()
            self.connection_lost_reported = False
            self.cmd_vel_ignored_reported = False
            self._request_data_streams()
            self.get_logger().info("MAVLink yeniden baglandi.")
        except Exception as exc:
            self.connected = False
            self.master = None
            self.last_heartbeat_time = 0.0
            self._reset_vehicle_state()
            self._neutralize_outputs()
            if not self.connection_lost_reported:
                self.connection_lost_reported = True
                self._publish_error(f"MAVLink yeniden baglanti basarisiz: {exc}")

    def _request_message_interval(self, message_id, frequency_hz):
        if not self._has_valid_link():
            return

        interval_us = int(1_000_000 / frequency_hz) if frequency_hz > 0 else -1
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    def _request_data_streams(self):
        if not self._has_valid_link():
            return

        requested_messages = (
            (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5),
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 10),
            (mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 5),
            (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 1),
        )

        for message_id, frequency_hz in requested_messages:
            try:
                self._request_message_interval(message_id, frequency_hz)
            except Exception as exc:
                self.get_logger().warn(f"Telemetry istegi gonderilemedi: {exc}")

    def _connection_watchdog(self):
        if self.master is None:
            self.connected = False
            if not self.connection_lost_reported:
                self.connection_lost_reported = True
                self.get_logger().warn("MAVLink baglantisi yok: master None")
            self._try_reconnect()
            return

        if self.last_heartbeat_time == 0.0:
            self._try_reconnect()
            return

        elapsed = time.time() - self.last_heartbeat_time
        if elapsed <= self.connection_timeout_sec:
            return

        self.connected = False
        self._reset_vehicle_state()
        self._neutralize_outputs()
        if not self.connection_lost_reported:
            self.connection_lost_reported = True
            self._publish_error(
                f"MAVLink heartbeat kesildi. Son heartbeat {elapsed:.1f} saniye once alindi."
            )
        self._try_reconnect()

    def _reject_without_link(self, action_name):
        if self._has_valid_link():
            return False

        self.get_logger().warn(
            f"{action_name} reddedildi: MAVLink baglantisi hazir degil."
        )
        return True

    def _send_companion_heartbeat(self):
        if self.master is None:
            return

        try:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )
        except Exception as exc:
            self.get_logger().warn(f"Companion heartbeat gonderilemedi: {exc}")

    # noinspection D
    def _read_mavlink_messages(self):
        if self.master is None:
            return

        for _ in range(50):
            try:
                msg = self.master.recv_match(blocking=False)
                if msg is None:
                    return

                msg_type = msg.get_type()
                if msg_type == "BAD_DATA":
                    continue

                if msg_type == "COMMAND_LONG":
                    self._handle_command_long(msg)
                    continue

                if not self._message_from_target(msg):
                    continue

                if msg_type == "HEARTBEAT":
                    self._update_vehicle_state_from_heartbeat(msg)

                elif msg_type == "COMMAND_ACK":
                    self._log_command_ack(msg)

                elif msg_type == "GLOBAL_POSITION_INT":
                    self.gps_lat = msg.lat / 1e7
                    self.gps_lon = msg.lon / 1e7
                    self.gps_alt = msg.alt / 1000.0
                    self.relative_alt = msg.relative_alt / 1000.0
                    if hasattr(msg, "hdg") and msg.hdg != 65535:
                        self.heading_deg = msg.hdg / 100.0

                elif msg_type == "VFR_HUD" and hasattr(msg, "heading"):
                    self.heading_deg = float(msg.heading)

                elif msg_type == "ATTITUDE":
                    self.roll = float(msg.roll)
                    self.pitch = float(msg.pitch)
                    self.yaw = float(msg.yaw)

                elif msg_type == "SYS_STATUS":
                    if msg.voltage_battery != 65535:
                        self.voltage_v = msg.voltage_battery / 1000.0
                    if msg.current_battery != -1:
                        self.current_a = msg.current_battery / 100.0
                    if msg.battery_remaining != -1:
                        self.battery_remaining = float(msg.battery_remaining)

                elif msg_type == "STATUSTEXT":
                    self._log_statustext(msg)


            except Exception as exc:
                self._publish_error(f"MAVLink okuma hatasi: {exc}")
                self.connected = False
                self._reset_vehicle_state()
                self._neutralize_outputs()
                self._close_master()
                return

    def _handle_command_long(self, msg):
        command = int(getattr(msg, "command", -1))

        if command != MAV_CMD_MISSION_START:
            return

        mission_number = int(getattr(msg, "param1", 0))

        if mission_number < 1 or mission_number > 4:
            self.get_logger().warn(
                f"Gecersiz gorev komutu alindi: M{mission_number}"
            )
            self._send_command_ack(
                command,
                mavutil.mavlink.MAV_RESULT_DENIED,
            )
            return

        mission_name = f"M{mission_number}"

        mission_msg = String()
        mission_msg.data = mission_name
        self.mission_command_pub.publish(mission_msg)

        launch_ok = self._launch_mission(mission_number)
        ack_result = (
            mavutil.mavlink.MAV_RESULT_ACCEPTED
            if launch_ok
            else mavutil.mavlink.MAV_RESULT_FAILED
        )
        self._send_command_ack(command, ack_result)

        self.get_logger().info(f"MAVLink mission command received: {mission_name}")

    def _log_statustext(self, msg):
        text = getattr(msg, "text", "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        text = str(text).strip("\x00 ")
        if not text:
            return

        severity = int(getattr(msg, "severity", 6))
        if severity <= mavutil.mavlink.MAV_SEVERITY_WARNING:
            self.get_logger().warn(f"MAVLink STATUSTEXT[{severity}]: {text}")
        else:
            self.get_logger().info(f"MAVLink STATUSTEXT[{severity}]: {text}")

    # noinspection D
    def _publish_telemetry(self):
        now = self.get_clock().now().to_msg()
        link_ready = self._has_valid_link()

        if link_ready and self._has_valid_gps():
            gps_msg = NavSatFix()
            gps_msg.header.stamp = now
            gps_msg.header.frame_id = "gps"
            gps_msg.latitude = float(self.gps_lat)
            gps_msg.longitude = float(self.gps_lon)
            gps_msg.altitude = float(self.gps_alt) if self.gps_alt is not None else 0.0
            self.topics.gps_pub.publish(gps_msg)

        if link_ready and self.heading_deg is not None:
            heading_msg = Float32()
            heading_msg.data = float(self.heading_deg)
            self.topics.gps_heading_pub.publish(heading_msg)

        if link_ready and self.relative_alt is not None:
            alt_msg = Float32()
            alt_msg.data = float(self.relative_alt)
            self.topics.relative_alt_pub.publish(alt_msg)

        if link_ready and self.roll is not None and self.pitch is not None and self.yaw is not None:
            imu_msg = Imu()
            imu_msg.header.stamp = now
            imu_msg.header.frame_id = "base_link"
            qx, qy, qz, qw = euler_to_quaternion(self.roll, self.pitch, self.yaw)
            imu_msg.orientation.x = qx
            imu_msg.orientation.y = qy
            imu_msg.orientation.z = qz
            imu_msg.orientation.w = qw
            self.topics.imu_pub.publish(imu_msg)

        if link_ready and self.voltage_v is not None:
            battery_msg = BatteryState()
            battery_msg.header.stamp = now
            battery_msg.voltage = float(self.voltage_v)
            if self.current_a is not None:
                battery_msg.current = float(self.current_a)
            if self.battery_remaining is not None:
                battery_msg.percentage = float(self.battery_remaining) / 100.0
            self.topics.battery_pub.publish(battery_msg)

        state_msg = String()
        state_msg.data = (
            f"connected={self.connected}, armed={self.armed}, mode={self.mode}"
        )
        self.topics.state_pub.publish(state_msg)

    def _set_mode_callback(self, request, response):
        if self._reject_without_link("Mod komutu"):
            response.mode_sent = False
            return response

        mode_name = self._normalize_mode_name(request.custom_mode)
        mapping = self.master.mode_mapping()
        if mode_name not in mapping:
            self.get_logger().error(f"Bilinmeyen mod: {mode_name}")
            response.mode_sent = False
            return response

        if self._normalize_mode_name(self.mode) == mode_name:
            self.get_logger().info(
                f"Mode confirmation already valid: mode={self.mode}"
            )
            response.mode_sent = True
            return response

        try:
            self.get_logger().info(
                f"MAVLink TX SET_MODE: requested={mode_name}, current={self.mode}"
            )
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mapping[mode_name],
            )
            response.mode_sent = self._wait_for_vehicle_state(
                lambda: self._normalize_mode_name(self.mode) == mode_name,
                f"mode={mode_name}",
                expected_command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            )
            return response
        except Exception as exc:
            self._publish_error(f"Mod degistirme hatasi: {exc}")
            response.mode_sent = False
            return response

    def _arm_callback(self, request, response):
        success = self._arm_disarm(True)
        response.success = success
        response.message = "ARM heartbeat ile dogrulandi." if success else "ARM dogrulanamadi."
        return response

    def _force_arm_callback(self, request, response):
        success = self._arm_disarm(True, force=True)
        response.success = success
        response.message = (
            "FORCE ARM heartbeat ile dogrulandi."
            if success
            else "FORCE ARM dogrulanamadi."
        )
        return response

    def _disarm_callback(self, request, response):
        success = self._arm_disarm(False)
        response.success = success
        response.message = (
            "DISARM heartbeat ile dogrulandi."
            if success
            else "DISARM dogrulanamadi."
        )
        return response

    def _arm_disarm(self, arm, force=False):
        if self._reject_without_link("ARM/DISARM komutu"):
            return False

        requested_state = "armed=True" if arm else "armed=False"
        if self.armed == bool(arm):
            self.get_logger().info(
                f"Arm-state confirmation already valid: {requested_state}"
            )
            return True

        force_code = 21196 if arm and force else 0
        try:
            action_name = "FORCE ARM" if arm and force else ("ARM" if arm else "DISARM")
            self.get_logger().info(
                f"MAVLink TX {action_name}: current_armed={self.armed}, mode={self.mode}"
            )
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1 if arm else 0,
                force_code,
                0,
                0,
                0,
                0,
                0,
            )
            return self._wait_for_vehicle_state(
                lambda: self.armed == bool(arm),
                requested_state,
                expected_command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            )
        except Exception as exc:
            self._publish_error(f"ARM/DISARM hatasi: {exc}")
            return False

    def _send_zero_attitude_target_once(self):
        if not self._has_valid_link():
            return

        q = (
            self._yaw_to_mavlink_quaternion(self.yaw)
            if self.yaw is not None
            else self.last_target_q
        )
        time_boot_ms = int(1000 * (time.time() - self.boot_time))
        type_mask = 0b00100111

        self.master.mav.set_attitude_target_send(
            time_boot_ms,
            self.master.target_system,
            self.master.target_component,
            type_mask,
            q,
            0,
            0,
            0,
            0.0,
        )

    def shutdown_vehicle(self):
        if not self.disarm_on_shutdown:
            return
        if not self._has_valid_link():
            self.get_logger().warn("Shutdown DISARM atlandi: MAVLink baglantisi hazir degil.")
            return

        self.get_logger().info("Shutdown: arac durduruluyor ve DISARM deneniyor...")
        self._neutralize_outputs()

        try:
            for _ in range(3):
                self._send_zero_attitude_target_once()
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                time.sleep(0.15)

            deadline = time.time() + 3.0
            while time.time() < deadline:
                msg = self.master.recv_match(
                    type="HEARTBEAT",
                    blocking=True,
                    timeout=0.25,
                )
                if msg is None or not self._message_from_target(msg):
                    continue

                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.armed = armed
                self.mode = mavutil.mode_string_v10(msg)
                if not armed:
                    self.get_logger().info("Shutdown DISARM dogrulandi.")
                    return

            self.get_logger().warn("Shutdown DISARM dogrulanamadi; komut gonderildi ama armed heartbeat devam ediyor.")
        except Exception as exc:
            self._publish_error(f"Shutdown DISARM hatasi: {exc}")

    def _set_position_callback(self, msg):
        if not self._has_valid_link():
            self.get_logger().warn(
                "MAVLink baglantisi yok. /cube/set_position komutu yok sayiliyor.",
                throttle_duration_sec=2.0,
            )
            return

        if not self._vehicle_ready_for_guided_motion("/cube/set_position"):
            return

        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            self.get_logger().warn(
                "Gecersiz set_position hedefi (0,0) yok sayiliyor.",
                throttle_duration_sec=2.0,
            )
            return

        try:
            set_position(self.master, (lat, lon), self.boot_time)
            self.last_position_target_time = time.time()
            self.get_logger().info(
                "MAVLink TX SET_POSITION_TARGET_GLOBAL_INT: "
                f"lat={lat:.7f}, lon={lon:.7f}, armed={self.armed}, mode={self.mode}",
                throttle_duration_sec=1.0,
            )
        except Exception as exc:
            self._publish_error(f"Set position hatasi: {exc}")

    def _cmd_vel_callback(self, msg):
        if not self._has_valid_link():
            self._neutralize_outputs()

            if not self.cmd_vel_ignored_reported:
                self.cmd_vel_ignored_reported = True
                self.get_logger().warn(
                    "MAVLink baglantisi yok. /cube/cmd_vel komutlari yok sayiliyor."
                )
            return

        if not self._vehicle_ready_for_guided_motion("/cube/cmd_vel"):
            self._neutralize_outputs()
            return

        self.cmd_vel_ignored_reported = False
        self.last_position_target_time = 0.0

        linear_x = max(-1.0, min(1.0, float(msg.linear.x)))
        angular_z = max(-1.0, min(1.0, float(msg.angular.z)))

        target_yaw_rad = (self.yaw if self.yaw is not None else 0.0) + angular_z
        self.last_target_q = self._yaw_to_mavlink_quaternion(target_yaw_rad)
        self.last_thrust = linear_x

        self.last_cmd_vel_time = time.time()

        self.get_logger().info(
            f"cmd_vel accepted: linear={linear_x:.2f}, angular={angular_z:.2f} | "
            f"target_yaw={math.degrees(target_yaw_rad):.1f}deg, "
            f"thrust={self.last_thrust:.2f}, armed={self.armed}, mode={self.mode}",
            throttle_duration_sec=1.0,
        )

    def _send_attitude_target_loop(self):
        if self.master is None:
            return

        if not self._has_valid_link():
            self._neutralize_outputs()
            return

        command_age = time.time() - self.last_cmd_vel_time
        if not self.armed or self._normalize_mode_name(self.mode) != "GUIDED":
            if command_age <= self.cmd_timeout_sec or self.last_attitude_tx_active:
                self._vehicle_ready_for_guided_motion("SET_ATTITUDE_TARGET TX")
            self.last_attitude_tx_active = False
            return

        if time.time() - self.last_position_target_time <= self.position_target_timeout_sec:
            return

        command_active = command_age <= self.cmd_timeout_sec
        if not command_active:
            q = (
                self._yaw_to_mavlink_quaternion(self.yaw)
                if self.yaw is not None
                else self.last_target_q
            )
            thrust = 0.0
        else:
            q = self.last_target_q
            thrust = self.last_thrust

        try:
            time_boot_ms = int(1000 * (time.time() - self.boot_time))
            type_mask = 0b00100111
            self.master.mav.set_attitude_target_send(
                time_boot_ms,
                self.master.target_system,
                self.master.target_component,
                type_mask,
                q,
                0,
                0,
                0,
                thrust,
            )
            if command_active:
                self.get_logger().info(
                    "MAVLink TX SET_ATTITUDE_TARGET: "
                    f"thrust={thrust:.2f}, cmd_age={command_age:.2f}s, "
                    f"armed={self.armed}, mode={self.mode}",
                    throttle_duration_sec=1.0,
                )
            elif self.last_attitude_tx_active:
                self.get_logger().info(
                    "MAVLink TX SET_ATTITUDE_TARGET stop: "
                    f"cmd_vel timeout ({command_age:.2f}s), thrust=0.00"
                )
            self.last_attitude_tx_active = command_active
        except Exception as exc:
            self._publish_error(f"Attitude target hatasi: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = OrangeCubeBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Bridge kapatiliyor...")
    except ExternalShutdownException:
        pass
    finally:
        node._stop_active_mission()
        node.shutdown_vehicle()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
