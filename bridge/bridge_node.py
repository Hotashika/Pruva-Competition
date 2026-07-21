#!/usr/bin/env python3

import math
import os
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
from std_msgs.msg import Float32, Int32, String

from bridge.mavlink_connection import (
    DEFAULT_BAUD,
    DEFAULT_CONNECTION_STRING,
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_SOURCE_COMPONENT,
    DEFAULT_SOURCE_SYSTEM,
    connect_mavlink,
)
from utils.mavlink_utilities import create_bridge_topics, create_bridge_services
from utils.pixhawk_waypoints import mission_items_to_qgc
from utils.waypoint_server import DEFAULT_WAYPOINT_DIRECTORY, overwrite_waypoint_file
from utils.battery import battery_percentage_from_voltage

MISSION_PARAM_NAME = "SCR_USER1"
MISSION_IDLE = 0
MISSION_1 = 1
MISSION_2 = 2
MISSION_3 = 3
MISSION_4 = 4
MISSION_STOP = 90
MISSION_EMERGENCY = 99
VALID_MISSION_COMMANDS = {
    MISSION_IDLE,
    MISSION_1,
    MISSION_2,
    MISSION_3,
    MISSION_4,
    MISSION_STOP,
    MISSION_EMERGENCY,
}


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
            "source_system",
            int(os.getenv("MAVLINK_SOURCE_SYSTEM", str(DEFAULT_SOURCE_SYSTEM))),
        )
        self.declare_parameter(
            "source_component",
            int(os.getenv("MAVLINK_SOURCE_COMPONENT", str(DEFAULT_SOURCE_COMPONENT))),
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
            "mission_start_topic",
            os.getenv("MAVLINK_MISSION_START_TOPIC", "/mission_start"),
        )
        self.declare_parameter(
            "mission_start_ack_topic",
            os.getenv("MAVLINK_MISSION_START_ACK_TOPIC", "/mission_start_ack"),
        )
        self.declare_parameter(
            "mission_start_retry_sec",
            float(os.getenv("MAVLINK_MISSION_START_RETRY_SEC", "1.0")),
        )

        self.connection_string = self.get_parameter("connection_string").value
        self.baud = int(self.get_parameter("baud").value)
        self.heartbeat_timeout = int(self.get_parameter("heartbeat_timeout").value)
        self.source_system = int(self.get_parameter("source_system").value)
        self.source_component = int(self.get_parameter("source_component").value)
        self.connection_timeout_sec = float(self.get_parameter("connection_timeout_sec").value)
        self.reconnect_interval_sec = float(self.get_parameter("reconnect_interval_sec").value)
        self.reconnect_heartbeat_timeout = float(
            self.get_parameter("reconnect_heartbeat_timeout").value
        )
        self.command_confirmation_timeout_sec = float(
            self.get_parameter("command_confirmation_timeout_sec").value
        )
        self.disarm_on_shutdown = bool(self.get_parameter("disarm_on_shutdown").value)
        self.mission_start_topic = str(self.get_parameter("mission_start_topic").value)
        self.mission_start_ack_topic = str(self.get_parameter("mission_start_ack_topic").value)
        self.mission_start_retry_sec = float(
            self.get_parameter("mission_start_retry_sec").value
        )

        self.master = None
        self.connected = False
        self.armed = False
        self.mode = "UNKNOWN"
        self.last_heartbeat_time = 0.0
        self.last_connection_attempt = 0.0
        self.connection_lost_reported = False
        self.cmd_vel_ignored_reported = False
        self.last_mavlink_rx_time = 0.0
        self.last_mission_command_time = 0.0
        self.last_mission_command_wait_log_time = 0.0
        self.mission_parameter_initialized = False
        self.mission_parameter_startup_reset_pending = False
        self.last_mission_parameter_value = MISSION_IDLE
        self.pending_mission_command = None
        self.pending_mission_command_first_publish_time = 0.0
        self.pending_mission_command_last_publish_time = 0.0
        self.mission_download_task = None
        self.mission_download_count = None
        self.mission_download_items = {}
        self.mission_download_last_request_time = 0.0
        self.mission_download_retry_count = 0

        self.gps_lat = None
        self.gps_lon = None
        self.gps_alt = None
        self.relative_alt = None
        self.last_gps_sample_time = 0.0
        self.gps_fix_type = None
        self.last_gps_fix_time = 0.0
        # GLOBAL_POSITION_INT.hdg hareket yonu/course bilgisidir ve arac
        # yerinde donerken degismeyebilir. Task 3'te 20 derecelik donusu
        # dogrulamak icin Pixhawk EKF attitude yaw degerini ayri tutuyoruz.
        self.heading_deg = None
        self.attitude_heading_deg = None
        self.last_attitude_time = 0.0
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.imu_linear_acceleration = None
        self.imu_angular_velocity = None
        self.last_imu_sample_time = 0.0
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

        self.mission_start_pub = self.create_publisher(
            Int32,
            self.mission_start_topic,
            10,
        )
        self.create_subscription(
            Int32,
            self.mission_start_ack_topic,
            self._mission_start_ack_callback,
            10,
        )

        # self._connect()

        self.create_timer(1.0, self._connect_if_needed)
        self.create_timer(0.02, self._read_mavlink_messages)
        self.create_timer(0.2, self._publish_telemetry)
        self.create_timer(0.1, self._send_attitude_target_loop)
        self.create_timer(1.0, self._connection_watchdog)
        self.create_timer(1.0, self._send_companion_heartbeat)
        self.create_timer(0.5, self._request_mission_command_parameter)
        self.create_timer(5.0, self._mission_command_rx_watchdog)
        self.create_timer(0.2, self._mission_start_ack_watchdog)
        self.create_timer(0.5, self._mission_download_watchdog)

        self.get_logger().info(
            f"MAVLink Bridge aktif. {MISSION_PARAM_NAME}=1..4/90/99 okunup "
            f"{self.mission_start_topic} topic'ine Int32 olarak yayinlanacak; "
            f"{self.mission_start_ack_topic} ack geldikten sonra {MISSION_PARAM_NAME}=0 yapilacak."
        )

        self.get_logger().info("/cube topic ve servisleri aktif.")

    def _send_command_ack(self, command, result):
        if self.master is None:
            return

        try:
            self.master.mav.command_ack_send(command, result)
        except Exception as exc:
            self.get_logger().warn(f"COMMAND_ACK gonderilemedi: {exc}")

    def _send_status_text(self, text, severity=None):
        if self.master is None:
            return

        if severity is None:
            severity = mavutil.mavlink.MAV_SEVERITY_INFO

        message = f"JETSON: {text}"[:50]
        try:
            self.master.mav.statustext_send(severity, message.encode("utf-8"))
        except TypeError:
            self.master.mav.statustext_send(severity, message)
        except Exception as exc:
            self.get_logger().warn(f"STATUSTEXT gonderilemedi: {exc}")

    def _mission_command_rx_watchdog(self):
        if self.master is None or not self.connected:
            return

        now = time.time()
        if now - self.last_mission_command_wait_log_time < 15.0:
            return

        if self.last_mission_command_time > 0.0:
            return

        mavlink_rx_age = (
            now - self.last_mavlink_rx_time
            if self.last_mavlink_rx_time > 0.0
            else None
        )
        if mavlink_rx_age is None or mavlink_rx_age > 10.0:
            return

        self.last_mission_command_wait_log_time = now
        self.get_logger().warn(
            f"{MISSION_PARAM_NAME} mission command not seen yet: telemetry is arriving, "
            "but no non-zero parameter command has been read."
        )
        self._send_status_text(
            f"telemetry OK, waiting {MISSION_PARAM_NAME}",
            mavutil.mavlink.MAV_SEVERITY_WARNING,
        )

    def _publish_error(self, text):
        msg = String()
        msg.data = str(text)
        self.topics.error_pub.publish(msg)
        self.get_logger().error(str(text))

    def _publish_diagnostic(self, text):
        msg = String()
        msg.data = str(text)
        self.topics.diagnostics_pub.publish(msg)

    def _link_diagnostic_text(self):
        target_system = getattr(self.master, "target_system", None)
        target_component = getattr(self.master, "target_component", None)
        heartbeat_age = (
            time.time() - self.last_heartbeat_time
            if self.last_heartbeat_time > 0.0
            else None
        )
        heartbeat_age_text = (
            f"{heartbeat_age:.3f}s"
            if heartbeat_age is not None
            else "never"
        )
        return (
            f"master_present={self.master is not None}, connected={self.connected}, "
            f"target_system={target_system}, target_component={target_component}, "
            f"last_heartbeat_age={heartbeat_age_text}, armed={self.armed}, mode={self.mode}"
        )

    def _connect(self):
        self.last_connection_attempt = time.time()
        try:
            self._close_master()
            self.master = connect_mavlink(
                connection_string=self.connection_string,
                baud=self.baud,
                heartbeat_timeout=self.heartbeat_timeout,
                source_system=self.source_system,
                source_component=self.source_component,
                logger=self.get_logger(),
            )
            initial_heartbeat = getattr(self.master, "initial_vehicle_heartbeat", None)
            if initial_heartbeat is None:
                raise ConnectionError("Baglanti heartbeat'i bridge durumuna aktarilamadi.")
            self._update_vehicle_state_from_heartbeat(
                initial_heartbeat, source="initial connection"
            )
            self.last_mavlink_rx_time = time.time()
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
        self.last_gps_sample_time = 0.0
        self.gps_fix_type = None
        self.last_gps_fix_time = 0.0
        self.heading_deg = None
        self.attitude_heading_deg = None
        self.last_attitude_time = 0.0
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.imu_linear_acceleration = None
        self.imu_angular_velocity = None
        self.last_imu_sample_time = 0.0
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
        if self.last_gps_sample_time <= 0.0 or time.time() - self.last_gps_sample_time > 1.0:
            return False
        # GLOBAL_POSITION_INT, EKF'in son konum tahminini GPS fix'i koptuktan
        # sonra da bir süre taşıyabilir. Gerçek hareket için taze 3D GPS fix'i
        # (GPS_RAW_INT.fix_type >= 3) ayrıca doğrulanmalıdır.
        if self.gps_fix_type is None or self.gps_fix_type < 3:
            return False
        if self.last_gps_fix_time <= 0.0 or time.time() - self.last_gps_fix_time > 2.0:
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
        log_text = (
            "MAVLink RX COMMAND_ACK raw: "
            f"src_system={getattr(msg, 'get_srcSystem', lambda: None)()}, "
            f"src_component={getattr(msg, 'get_srcComponent', lambda: None)()}, "
            f"command={command}, result={result_name}({result}), "
            f"progress={getattr(msg, 'progress', None)}, "
            f"result_param2={getattr(msg, 'result_param2', None)}, "
            f"target_system={getattr(msg, 'target_system', None)}, "
            f"target_component={getattr(msg, 'target_component', None)}"
        )
        self._publish_diagnostic(log_text)
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
        wait_text = (
            f"Orange Cube confirmation waiting: {description}, timeout={timeout_sec:.1f}s"
        )
        self.get_logger().info(wait_text)
        self._publish_diagnostic(wait_text)

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
                heartbeat_text = (
                    "MAVLink RX HEARTBEAT raw (command confirmation): "
                    f"src_system={getattr(msg, 'get_srcSystem', lambda: None)()}, "
                    f"src_component={getattr(msg, 'get_srcComponent', lambda: None)()}, "
                    f"type={getattr(msg, 'type', None)}, "
                    f"autopilot={getattr(msg, 'autopilot', None)}, "
                    f"base_mode={getattr(msg, 'base_mode', None)}, "
                    f"custom_mode={getattr(msg, 'custom_mode', None)}, "
                    f"system_status={getattr(msg, 'system_status', None)}, "
                    f"mavlink_version={getattr(msg, 'mavlink_version', None)}, "
                    f"decoded_mode={mavutil.mode_string_v10(msg)}"
                )
                self.get_logger().info(heartbeat_text)
                self._publish_diagnostic(heartbeat_text)
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
                self._publish_error(
                    "Orange Cube command rejected before state confirmation: "
                    f"expected={description}, command={command}, "
                    f"result={self._mav_result_name(result)}({result}), "
                    f"{self._link_diagnostic_text()}"
                )
                return False

        self._publish_error(
            "Orange Cube confirmation timeout: "
            f"expected={description}, timeout={timeout_sec:.1f}s, "
            f"{self._link_diagnostic_text()}"
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
                source_system=self.source_system,
                source_component=self.source_component,
                logger=self.get_logger(),
            )
            initial_heartbeat = getattr(self.master, "initial_vehicle_heartbeat", None)
            if initial_heartbeat is None:
                raise ConnectionError("Reconnect heartbeat'i bridge durumuna aktarilamadi.")
            self._update_vehicle_state_from_heartbeat(
                initial_heartbeat, source="reconnect"
            )
            self.last_mavlink_rx_time = time.time()
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
            (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5),
            (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5),
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 10),
            (mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, 20),
            (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU, 20),
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

        reason = (
            f"{action_name} reddedildi: MAVLink baglantisi hazir degil; "
            f"{self._link_diagnostic_text()}"
        )
        self._publish_error(reason)
        self._publish_diagnostic(reason)
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

    def _request_mission_command_parameter(self):
        if self.master is None or not self.connected:
            return

        try:
            self.master.mav.param_request_read_send(
                self.master.target_system,
                self.master.target_component,
                MISSION_PARAM_NAME.encode("ascii"),
                -1,
            )
        except Exception as exc:
            self.get_logger().warn(f"{MISSION_PARAM_NAME} okunamadi: {exc}")

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

                self.last_mavlink_rx_time = time.time()

                if not self._message_from_target(msg):
                    continue

                if msg_type == "HEARTBEAT":
                    self._update_vehicle_state_from_heartbeat(msg)

                elif msg_type == "PARAM_VALUE":
                    self._handle_mission_parameter_value(msg)

                elif msg_type == "COMMAND_ACK":
                    self._log_command_ack(msg)

                elif msg_type == "MISSION_COUNT":
                    self._handle_mission_count(msg)

                elif msg_type in ("MISSION_ITEM_INT", "MISSION_ITEM"):
                    self._handle_mission_item(msg)

                elif msg_type == "GLOBAL_POSITION_INT":
                    self.gps_lat = msg.lat / 1e7
                    self.gps_lon = msg.lon / 1e7
                    self.gps_alt = msg.alt / 1000.0
                    self.relative_alt = msg.relative_alt / 1000.0
                    self.last_gps_sample_time = time.time()
                    if hasattr(msg, "hdg") and msg.hdg != 65535:
                        self.heading_deg = msg.hdg / 100.0

                elif msg_type == "GPS_RAW_INT":
                    self.gps_fix_type = int(getattr(msg, "fix_type", 0))
                    self.last_gps_fix_time = time.time()

                elif msg_type == "VFR_HUD" and hasattr(msg, "heading"):
                    self.heading_deg = float(msg.heading)

                elif msg_type == "ATTITUDE":
                    self.roll = float(msg.roll)
                    self.pitch = float(msg.pitch)
                    self.yaw = float(msg.yaw)
                    # MAVLink ATTITUDE.yaw NED ekseninde radyandir. Dereceye
                    # cevirip 0..360 araligina alinan bu deger, arac yerinde
                    # donerken de gercek Pixhawk pusula/EKF basligini izler.
                    self.attitude_heading_deg = math.degrees(self.yaw) % 360.0
                    self.last_attitude_time = time.time()

                elif msg_type == "HIGHRES_IMU":
                    # HIGHRES_IMU: ivme m/s^2, acisal hiz rad/s.
                    self.imu_linear_acceleration = (
                        float(msg.xacc),
                        float(msg.yacc),
                        float(msg.zacc),
                    )
                    self.imu_angular_velocity = (
                        float(msg.xgyro),
                        float(msg.ygyro),
                        float(msg.zgyro),
                    )
                    self.last_imu_sample_time = time.time()

                elif msg_type == "SCALED_IMU":
                    # SCALED_IMU: ivme mG, acisal hiz millirad/s.
                    accel_scale = 9.80665 / 1000.0
                    gyro_scale = 1.0 / 1000.0
                    self.imu_linear_acceleration = (
                        float(msg.xacc) * accel_scale,
                        float(msg.yacc) * accel_scale,
                        float(msg.zacc) * accel_scale,
                    )
                    self.imu_angular_velocity = (
                        float(msg.xgyro) * gyro_scale,
                        float(msg.ygyro) * gyro_scale,
                        float(msg.zgyro) * gyro_scale,
                    )
                    self.last_imu_sample_time = time.time()

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

    def _handle_mission_parameter_value(self, msg):
        param_name = getattr(msg, "param_id", "")
        if isinstance(param_name, bytes):
            param_name = param_name.decode("utf-8", errors="ignore")
        param_name = str(param_name).rstrip("\x00")
        if param_name != MISSION_PARAM_NAME:
            return

        try:
            command = int(round(float(getattr(msg, "param_value", MISSION_IDLE))))
        except (TypeError, ValueError):
            return

        if self.mission_parameter_startup_reset_pending:
            if command == MISSION_IDLE:
                self.mission_parameter_startup_reset_pending = False
                self.last_mission_parameter_value = MISSION_IDLE
                self.get_logger().info(
                    f"Baslangic {MISSION_PARAM_NAME} sifirlamasi dogrulandi; "
                    "yeni gorev komutu bekleniyor."
                )
            return

        if not self.mission_parameter_initialized:
            self.mission_parameter_initialized = True
            self.last_mission_parameter_value = command
            if command != MISSION_IDLE:
                self.get_logger().warn(
                    f"Baslangicta {MISSION_PARAM_NAME}={command} bulundu; "
                    "kalici/eski komut olarak kabul edilip 0'a sifirlaniyor; "
                    "gorev baslatilmayacak."
                )
                self.mission_parameter_startup_reset_pending = True
                self._set_mission_parameter(MISSION_IDLE)
            return

        if command == self.last_mission_parameter_value:
            return

        self.get_logger().info(
            f"{MISSION_PARAM_NAME} degisti: {self.last_mission_parameter_value} -> {command}"
        )
        self.last_mission_parameter_value = command

        if command == MISSION_IDLE:
            return

        self._process_mission_parameter_command(command)

    def _process_mission_parameter_command(self, command):
        self.last_mission_command_time = time.time()

        if command == MISSION_STOP:
            self.get_logger().info(
                "Normal gorev durdurma komutu alindi; /mission_start=90 yayinlaniyor, ack bekleniyor."
            )
            self._publish_mission_start(command, track_ack=True)
            self._send_status_text("mission stop published, waiting ack")
            return

        if command == MISSION_EMERGENCY:
            self.get_logger().warn(
                "Jetson acil gorev iptal komutu alindi; /mission_start=99 yayinlaniyor, ack bekleniyor."
            )
            self._publish_mission_start(command, track_ack=True)
            self._send_status_text(
                "emergency cancel published, waiting ack",
                mavutil.mavlink.MAV_SEVERITY_WARNING,
            )
            return

        if command not in (MISSION_1, MISSION_2, MISSION_3, MISSION_4):
            self.get_logger().warn(
                f"Gecersiz {MISSION_PARAM_NAME} gorev komutu alindi: {command}"
            )
            self._send_status_text(
                f"invalid {MISSION_PARAM_NAME} command {command}",
                mavutil.mavlink.MAV_SEVERITY_WARNING,
            )
            self._acknowledge_mission_parameter()
            return

        mission_number = command
        mission_name = f"M{mission_number}"

        if mission_number in (MISSION_1, MISSION_2, MISSION_4):
            self._start_mission_download(mission_number)
            self._send_status_text(
                f"{MISSION_PARAM_NAME} received: {mission_name}, downloading mission",
                mavutil.mavlink.MAV_SEVERITY_INFO,
            )
            self.get_logger().info(
                f"{MISSION_PARAM_NAME} mission command received: {mission_name}; "
                "waypoint dosyasi yazildiktan sonra mission_start yayinlanacak."
            )
            return

        self._publish_downloaded_mission_start(mission_number)

    def _publish_downloaded_mission_start(self, mission_number):
        self._publish_mission_start(mission_number, track_ack=True)
        self._send_status_text(
            f"mission M{mission_number} ready, waiting ack",
            mavutil.mavlink.MAV_SEVERITY_INFO,
        )
        self.get_logger().info(
            f"Mission M{mission_number} hazir; "
            f"{self.mission_start_topic}={mission_number} yayinlandi, "
            f"{self.mission_start_ack_topic} ack bekleniyor."
        )

    def _start_mission_download(self, mission_number):
        if self.master is None:
            self.get_logger().warn("Waypoint senkronizasyonu baslatilamadi: MAVLink yok.")
            return

        self.mission_download_task = int(mission_number)
        self.mission_download_count = None
        self.mission_download_items = {}
        self.mission_download_retry_count = 0
        self._request_mission_list()
        self.get_logger().info(
            f"Pixhawk mission listesi njord_task{mission_number}.waypoints icin isteniyor."
        )

    def _request_mission_list(self):
        try:
            self.master.mav.mission_request_list_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            self.master.mav.mission_request_list_send(
                self.master.target_system,
                self.master.target_component,
            )
        self.mission_download_last_request_time = time.time()

    def _request_mission_item(self, sequence):
        try:
            self.master.mav.mission_request_int_send(
                self.master.target_system,
                self.master.target_component,
                int(sequence),
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            self.master.mav.mission_request_int_send(
                self.master.target_system,
                self.master.target_component,
                int(sequence),
            )
        self.mission_download_last_request_time = time.time()

    def _handle_mission_count(self, msg):
        if self.mission_download_task is None:
            return
        self.mission_download_count = int(msg.count)
        self.mission_download_retry_count = 0
        if self.mission_download_count <= 0:
            self.get_logger().warn("Pixhawk mission listesi bos; waypoint dosyasi degistirilmedi.")
            self._reset_mission_download()
            return
        self._request_mission_item(0)

    def _handle_mission_item(self, msg):
        if self.mission_download_task is None or self.mission_download_count is None:
            return
        sequence = int(msg.seq)
        if not 0 <= sequence < self.mission_download_count:
            return
        self.mission_download_items[sequence] = msg
        self.mission_download_retry_count = 0

        missing = next(
            (seq for seq in range(self.mission_download_count)
             if seq not in self.mission_download_items),
            None,
        )
        if missing is not None:
            self._request_mission_item(missing)
            return

        task_number = self.mission_download_task
        filename = f"njord_task{task_number}.waypoints"
        try:
            content = mission_items_to_qgc(self.mission_download_items.values())
            destination = overwrite_waypoint_file(
                DEFAULT_WAYPOINT_DIRECTORY, filename, content
            )
            save_text = (
                f"Pixhawk mission dosyaya yazildi: path={destination.resolve()}, "
                f"items={self.mission_download_count}, bytes={len(content.encode('utf-8'))}"
            )
            self.get_logger().info(save_text)
            self._publish_diagnostic(save_text)
            try:
                self.master.mav.mission_ack_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_MISSION_ACCEPTED,
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                )
            except TypeError:
                self.master.mav.mission_ack_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_MISSION_ACCEPTED,
                )
            self._publish_downloaded_mission_start(task_number)
        except Exception as exc:
            self.get_logger().error(f"Pixhawk waypoint dosyasi yazilamadi: {exc}")
        finally:
            self._reset_mission_download()

    def _mission_download_watchdog(self):
        if self.mission_download_task is None:
            return
        if time.time() - self.mission_download_last_request_time < 2.0:
            return
        self.mission_download_retry_count += 1
        if self.mission_download_retry_count > 4:
            self.get_logger().error("Pixhawk mission indirme zaman asimina ugradi.")
            self._reset_mission_download()
            return
        if self.mission_download_count is None:
            self._request_mission_list()
            return
        missing = next(
            (seq for seq in range(self.mission_download_count)
             if seq not in self.mission_download_items),
            None,
        )
        if missing is not None:
            self._request_mission_item(missing)

    def _reset_mission_download(self):
        self.mission_download_task = None
        self.mission_download_count = None
        self.mission_download_items = {}
        self.mission_download_last_request_time = 0.0
        self.mission_download_retry_count = 0

    def _publish_mission_start(self, command, track_ack=False):
        mission_msg = Int32()
        mission_msg.data = int(command)
        self.mission_start_pub.publish(mission_msg)
        now = time.time()
        if track_ack:
            previous_pending_command = self.pending_mission_command
            self.pending_mission_command = int(command)
            if (
                self.pending_mission_command_first_publish_time <= 0.0
                or previous_pending_command != self.pending_mission_command
            ):
                self.pending_mission_command_first_publish_time = now
            self.pending_mission_command_last_publish_time = now

    def _mission_start_ack_callback(self, msg):
        try:
            ack_command = int(msg.data)
        except (TypeError, ValueError):
            self.get_logger().warn(f"Gecersiz mission_start_ack verisi: {msg.data}")
            return

        if self.pending_mission_command is None:
            self.get_logger().info(
                f"{self.mission_start_ack_topic}={ack_command} alindi ama bekleyen komut yok."
            )
            return

        if ack_command != self.pending_mission_command:
            self.get_logger().warn(
                f"{self.mission_start_ack_topic} uyusmadi: beklenen={self.pending_mission_command}, "
                f"gelen={ack_command}"
            )
            return

        self.get_logger().info(
            f"{self.mission_start_ack_topic}={ack_command} alindi; {MISSION_PARAM_NAME}=0 yapiliyor."
        )
        self._acknowledge_mission_parameter()

    def _mission_start_ack_watchdog(self):
        if self.pending_mission_command is None:
            return

        now = time.time()
        if now - self.pending_mission_command_last_publish_time < self.mission_start_retry_sec:
            return

        pending_command = self.pending_mission_command
        self.get_logger().warn(
            f"{self.mission_start_ack_topic} bekleniyor; "
            f"{self.mission_start_topic}={pending_command} tekrar yayinlaniyor."
        )
        self._publish_mission_start(pending_command, track_ack=True)

    def _acknowledge_mission_parameter(self):
        self._set_mission_parameter(MISSION_IDLE)
        self.last_mission_parameter_value = MISSION_IDLE
        self.pending_mission_command = None
        self.pending_mission_command_first_publish_time = 0.0
        self.pending_mission_command_last_publish_time = 0.0

    def _set_mission_parameter(self, value):
        if self.master is None:
            return

        try:
            self.master.mav.param_set_send(
                self.master.target_system,
                self.master.target_component,
                MISSION_PARAM_NAME.encode("ascii"),
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            self.get_logger().info(f"{MISSION_PARAM_NAME} = {value} gonderildi.")
        except Exception as exc:
            self.get_logger().warn(f"{MISSION_PARAM_NAME} sifirlanamadi: {exc}")

    def _log_statustext(self, msg):
        text = getattr(msg, "text", "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        text = str(text).strip("\x00 ")
        if not text:
            return

        severity = int(getattr(msg, "severity", 6))
        raw_text = (
            "MAVLink RX STATUSTEXT raw: "
            f"src_system={getattr(msg, 'get_srcSystem', lambda: None)()}, "
            f"src_component={getattr(msg, 'get_srcComponent', lambda: None)()}, "
            f"severity={severity}, id={getattr(msg, 'id', None)}, "
            f"chunk_seq={getattr(msg, 'chunk_seq', None)}, text={text!r}"
        )
        self._publish_diagnostic(raw_text)
        if severity <= mavutil.mavlink.MAV_SEVERITY_WARNING:
            self.get_logger().warn(raw_text)
        else:
            self.get_logger().info(raw_text)

    # noinspection D
    def _publish_telemetry(self):
        now = self.get_clock().now().to_msg()
        link_ready = self._has_valid_link()

        # Gecerli heartbeat mode'u olmadan bagli durum yayinlama. Boylece
        # arayuzde gecici UNKNOWN/GUIDED salinimi gorunmez.
        if self.connected and self._normalize_mode_name(self.mode) == "UNKNOWN":
            self.get_logger().warn(
                "connected=True fakat heartbeat mode henuz dogrulanmadi; "
                "/cube/state yayini bu tur icin atlandi.",
                throttle_duration_sec=2.0,
            )
            return

        if link_ready and self._has_valid_gps():
            gps_msg = NavSatFix()
            gps_msg.header.stamp = now
            gps_msg.header.frame_id = "gps"
            gps_msg.latitude = float(self.gps_lat)
            gps_msg.longitude = float(self.gps_lon)
            gps_msg.altitude = float(self.gps_alt) if self.gps_alt is not None else 0.0
            self.topics.gps_pub.publish(gps_msg)

        # Arama kontrolunde yalnizca taze attitude yaw kullan. GPS course'a
        # geri donmek yerinde donen aracta heading'i tekrar sabitleyip motorun
        # durmadan donmesine yol acabilir. Akis kesilirse yayin da kesilir ve
        # Task 3 heading watchdog'u araci guvenli durdurur.
        attitude_is_fresh = (
            self.attitude_heading_deg is not None
            and self.last_attitude_time > 0.0
            and time.time() - self.last_attitude_time <= 1.0
        )
        if link_ready and attitude_is_fresh:
            heading_msg = Float32()
            heading_msg.data = float(self.attitude_heading_deg)
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
            imu_data_fresh = (
                self.last_imu_sample_time > 0.0
                and time.time() - self.last_imu_sample_time <= 0.5
                and self.imu_linear_acceleration is not None
                and self.imu_angular_velocity is not None
            )
            if imu_data_fresh:
                (
                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y,
                    imu_msg.linear_acceleration.z,
                ) = self.imu_linear_acceleration
                (
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z,
                ) = self.imu_angular_velocity
            else:
                # ROS Imu sozlesmesi: ilk covariance -1 ise alan mevcut degildir.
                imu_msg.linear_acceleration_covariance[0] = -1.0
                imu_msg.angular_velocity_covariance[0] = -1.0
            self.topics.imu_pub.publish(imu_msg)

        if link_ready and self.voltage_v is not None:
            battery_msg = BatteryState()
            battery_msg.header.stamp = now
            battery_msg.voltage = float(self.voltage_v)
            if self.current_a is not None:
                battery_msg.current = float(self.current_a)
            battery_msg.percentage = battery_percentage_from_voltage(self.voltage_v)
            self.topics.battery_pub.publish(battery_msg)

        state_msg = String()
        state_msg.data = (
            f"connected={self.connected}, armed={self.armed}, mode={self.mode}"
        )
        self.topics.state_pub.publish(state_msg)

    def _set_mode_callback(self, request, response):
        request_text = (
            "ROS RX SET_MODE request: "
            f"base_mode={request.base_mode}, custom_mode={request.custom_mode!r}, "
            f"{self._link_diagnostic_text()}"
        )
        self.get_logger().info(request_text)
        self._publish_diagnostic(request_text)

        if self._reject_without_link("Mod komutu"):
            response.mode_sent = False
            return response

        mode_name = self._normalize_mode_name(request.custom_mode)
        try:
            mapping = self.master.mode_mapping() or {}
        except Exception as exc:
            self._publish_error(
                f"Mode mapping okunamadi: requested={mode_name}, exception={exc!r}"
            )
            response.mode_sent = False
            return response

        if mode_name not in mapping:
            available_modes = ",".join(sorted(mapping)) if mapping else "<empty>"
            self._publish_error(
                f"Bilinmeyen mod: requested={mode_name}, available_modes={available_modes}"
            )
            response.mode_sent = False
            return response

        if self._normalize_mode_name(self.mode) == mode_name:
            self.get_logger().info(
                f"Mode confirmation already valid: mode={self.mode}"
            )
            response.mode_sent = True
            return response

        try:
            tx_text = (
                "MAVLink TX MAV_CMD_DO_SET_MODE raw: "
                f"target_system={self.master.target_system}, "
                f"target_component={self.master.target_component}, "
                f"command={mavutil.mavlink.MAV_CMD_DO_SET_MODE}, confirmation=0, "
                f"param1={mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED}, "
                f"param2={mapping[mode_name]}, requested={mode_name}, "
                f"current={self.mode}"
            )
            self.get_logger().info(tx_text)
            self._publish_diagnostic(tx_text)
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mapping[mode_name],
                0,
                0,
                0,
                0,
                0,
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
            tx_text = (
                f"MAVLink TX {action_name} raw: "
                f"target_system={self.master.target_system}, "
                f"target_component={self.master.target_component}, "
                f"command={mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM}, "
                f"confirmation=0, param1={1 if arm else 0}, param2={force_code}, "
                f"current_armed={self.armed}, mode={self.mode}"
            )
            self.get_logger().info(tx_text)
            self._publish_diagnostic(tx_text)
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
            tx_text = (
                "MAVLink TX SHUTDOWN DISARM raw: "
                f"target_system={self.master.target_system}, "
                f"target_component={self.master.target_component}, "
                f"command={mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM}, "
                "confirmation=0, param1=0, param2=0"
            )
            self.get_logger().info(tx_text)
            self._publish_diagnostic(tx_text)
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

            success = self._wait_for_vehicle_state(
                lambda: not self.armed,
                "armed=False (shutdown)",
                expected_command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                timeout_sec=3.0,
            )
            if success:
                self.get_logger().info("Shutdown DISARM dogrulandi.")
            else:
                self.get_logger().warn(
                    "Shutdown DISARM dogrulanamadi; ayrinti icin COMMAND_ACK ve "
                    "STATUSTEXT loglarina bakin."
                )
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
        node.shutdown_vehicle()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
