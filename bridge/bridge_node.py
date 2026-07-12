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
from std_msgs.msg import String, Float32

from bridge.mavlink_connection import (
    DEFAULT_BAUD,
    DEFAULT_CONNECTION_STRING,
    DEFAULT_HEARTBEAT_TIMEOUT,
    connect_mavlink,
)
from utils.mavlink_utilities import (
    calculate_angle_error_deg,
    calculate_bearing,
    calculate_gps_distance,
    create_bridge_services,
    create_bridge_topics,
)

MAV_CMD_NJORD_MISSION_START = mavutil.mavlink.MAV_CMD_USER_1


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
            "disarm_on_shutdown",
            os.getenv("MAVLINK_DISARM_ON_SHUTDOWN", "1").lower()
            not in ("0", "false", "no", "off"),
        )
        self.declare_parameter(
            "align_before_set_position",
            os.getenv("MAVLINK_ALIGN_BEFORE_SET_POSITION", "1").lower()
            not in ("0", "false", "no", "off"),
        )
        self.declare_parameter(
            "set_position_heading_tolerance_deg",
            float(os.getenv("MAVLINK_SET_POSITION_HEADING_TOLERANCE_DEG", "8.0")),
        )
        self.declare_parameter(
            "set_position_heading_min_distance_m",
            float(os.getenv("MAVLINK_SET_POSITION_HEADING_MIN_DISTANCE_M", "1.5")),
        )

        self.connection_string = self.get_parameter("connection_string").value
        self.baud = int(self.get_parameter("baud").value)
        self.heartbeat_timeout = int(self.get_parameter("heartbeat_timeout").value)
        self.connection_timeout_sec = float(self.get_parameter("connection_timeout_sec").value)
        self.reconnect_interval_sec = float(self.get_parameter("reconnect_interval_sec").value)
        self.reconnect_heartbeat_timeout = float(
            self.get_parameter("reconnect_heartbeat_timeout").value
        )
        self.disarm_on_shutdown = bool(self.get_parameter("disarm_on_shutdown").value)
        self.align_before_set_position = bool(
            self.get_parameter("align_before_set_position").value
        )
        self.set_position_heading_tolerance_deg = float(
            self.get_parameter("set_position_heading_tolerance_deg").value
        )
        self.set_position_heading_min_distance_m = float(
            self.get_parameter("set_position_heading_min_distance_m").value
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
        self.last_position_target_time = 0.0
        self.position_target_timeout_sec = 0.5
        self.position_target_aligned_key = None

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

        self.get_logger().info("/cube topic ve servisleri aktif.")

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
        self.position_target_aligned_key = None

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
                if not self._message_from_target(msg):
                    continue

                if msg_type == "HEARTBEAT":
                    self.mode = mavutil.mode_string_v10(msg)
                    self.armed = bool(
                        msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    )
                    self.last_heartbeat_time = time.time()
                    if not self.connected or self.connection_lost_reported:
                        self.get_logger().info("MAVLink heartbeat tekrar alindi.")
                    self.connected = True
                    self.connection_lost_reported = False
                    self.cmd_vel_ignored_reported = False

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

                elif msg_type == "COMMAND_LONG":
                    self._handle_command_long(msg)

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

        if command != MAV_CMD_NJORD_MISSION_START:
            return

        mission_number = int(getattr(msg, "param1", 0))

        if mission_number < 1 or mission_number > 4:
            self.get_logger().warn(
                f"Gecersiz gorev komutu alindi: M{mission_number}"
            )
            return

        mission_name = f"M{mission_number}"

        mission_msg = String()
        mission_msg.data = mission_name
        self.mission_command_pub.publish(mission_msg)

        self.get_logger().info(
            f"NJORD mission command received from MAVLink: {mission_name}"
        )

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

        mode_name = request.custom_mode
        mapping = self.master.mode_mapping()
        if mode_name not in mapping:
            self.get_logger().error(f"Bilinmeyen mod: {mode_name}")
            response.mode_sent = False
            return response

        try:
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mapping[mode_name],
            )
            self.get_logger().info(f"Mod komutu gonderildi: {mode_name}")
            response.mode_sent = True
            return response
        except Exception as exc:
            self._publish_error(f"Mod degistirme hatasi: {exc}")
            response.mode_sent = False
            return response

    def _arm_callback(self, request, response):
        success = self._arm_disarm(True)
        response.success = success
        response.message = "ARM komutu gonderildi." if success else "ARM komutu basarisiz."
        return response

    def _force_arm_callback(self, request, response):
        success = self._arm_disarm(True, force=True)
        response.success = success
        response.message = "FORCE ARM komutu gonderildi." if success else "FORCE ARM komutu basarisiz."
        return response

    def _disarm_callback(self, request, response):
        success = self._arm_disarm(False)
        response.success = success
        response.message = "DISARM komutu gonderildi." if success else "DISARM komutu basarisiz."
        return response

    def _arm_disarm(self, arm, force=False):
        if self._reject_without_link("ARM/DISARM komutu"):
            return False

        force_code = 21196 if arm and force else 0
        try:
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
            if arm and force:
                self.get_logger().info("FORCE ARM komutu gonderildi.")
            else:
                self.get_logger().info("ARM komutu gonderildi." if arm else "DISARM komutu gonderildi.")
            return True
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

        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            self.get_logger().warn(
                "Gecersiz set_position hedefi (0,0) yok sayiliyor.",
                throttle_duration_sec=2.0,
            )
            return

        try:
            if not self._align_before_set_position(lat, lon):
                return

            set_position(self.master, (lat, lon), self.boot_time)
            self.last_position_target_time = time.time()
            self.get_logger().info(
                f"set_position: lat={lat:.7f}, lon={lon:.7f}",
                throttle_duration_sec=1.0,
            )
        except Exception as exc:
            self._publish_error(f"Set position hatasi: {exc}")

    @staticmethod
    def _position_target_key(lat, lon):
        return (round(float(lat), 7), round(float(lon), 7))

    def _align_before_set_position(self, lat, lon):
        if not self.align_before_set_position:
            return True

        target_key = self._position_target_key(lat, lon)
        if self.position_target_aligned_key == target_key:
            return True

        if self.gps_lat is None or self.gps_lon is None or self.heading_deg is None:
            self._neutralize_outputs()
            self.get_logger().warn(
                "set_position oncesi heading hizalamasi icin GPS/heading bekleniyor.",
                throttle_duration_sec=2.0,
            )
            return False

        distance_m = calculate_gps_distance(self.gps_lat, self.gps_lon, lat, lon)
        if distance_m <= self.set_position_heading_min_distance_m:
            self.position_target_aligned_key = target_key
            return True

        target_bearing = calculate_bearing(self.gps_lat, self.gps_lon, lat, lon)
        heading_error = calculate_angle_error_deg(target_bearing, self.heading_deg)

        if abs(heading_error) <= self.set_position_heading_tolerance_deg:
            self.position_target_aligned_key = target_key
            self._neutralize_outputs()
            self.get_logger().info(
                f"set_position heading hizalandi: target={target_bearing:.1f}deg, "
                f"current={self.heading_deg:.1f}deg, error={heading_error:.1f}deg",
                throttle_duration_sec=1.0,
            )
            return True

        target_yaw_rad = math.radians(target_bearing)
        self.last_position_target_time = 0.0
        self.last_target_q = self._yaw_to_mavlink_quaternion(target_yaw_rad)
        self.last_thrust = 0.0
        self.last_cmd_vel_time = time.time()

        self.get_logger().info(
            f"set_position oncesi heading hizalaniyor: target={target_bearing:.1f}deg, "
            f"current={self.heading_deg:.1f}deg, error={heading_error:.1f}deg",
            throttle_duration_sec=1.0,
        )
        return False

    def _cmd_vel_callback(self, msg):
        if not self._has_valid_link():
            self._neutralize_outputs()

            if not self.cmd_vel_ignored_reported:
                self.cmd_vel_ignored_reported = True
                self.get_logger().warn(
                    "MAVLink baglantisi yok. /cube/cmd_vel komutlari yok sayiliyor."
                )
            return

        self.cmd_vel_ignored_reported = False
        self.last_position_target_time = 0.0

        linear_x = max(-1.0, min(1.0, float(msg.linear.x)))
        angular_z = max(-1.0, min(1.0, float(msg.angular.z)))
        if abs(linear_x) > 1e-3 or abs(angular_z) > 1e-3:
            self.position_target_aligned_key = None

        target_yaw_rad = (self.yaw if self.yaw is not None else 0.0) + angular_z
        self.last_target_q = self._yaw_to_mavlink_quaternion(target_yaw_rad)
        self.last_thrust = linear_x

        self.last_cmd_vel_time = time.time()

        self.get_logger().info(
            f"cmd_vel: linear={linear_x:.2f}, angular={angular_z:.2f} | "
            f"target_yaw={math.degrees(target_yaw_rad):.1f}deg, "
            f"thrust={self.last_thrust:.2f}",
            throttle_duration_sec=1.0,
        )

    def _send_attitude_target_loop(self):
        if self.master is None:
            return

        if not self._has_valid_link():
            self._neutralize_outputs()
            return

        if time.time() - self.last_position_target_time <= self.position_target_timeout_sec:
            return

        if time.time() - self.last_cmd_vel_time > self.cmd_timeout_sec:
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
        except Exception as exc:
            self._publish_error(f"Attitude target hatasi: {exc}")

    def _correction_yaw(self):
        ...


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
