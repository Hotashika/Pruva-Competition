from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String

try:
    from utils.mavlink_utilities import (
        call_set_mode,
        call_trigger_service,
        calculate_gps_distance,
        create_mission_clients,
        create_mission_topics,
        publish_cmd_vel,
        publish_set_position,
        stop_vehicle,
        wait_for_mission_services,
    )
except Exception:  # pragma: no cover - repo dışında statik inceleme için fallback
    call_set_mode = None
    call_trigger_service = None
    create_mission_clients = None
    create_mission_topics = None
    wait_for_mission_services = None


    def publish_cmd_vel(pub: Any, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        pub.publish(msg)


    def stop_vehicle(pub: Any) -> None:
        publish_cmd_vel(pub, 0.0, 0.0)


    def publish_set_position(pub: Any, lat: float, lon: float, altitude: float = 20.0) -> None:
        msg = NavSatFix()
        msg.latitude = float(lat)
        msg.longitude = float(lon)
        msg.altitude = float(altitude)
        pub.publish(msg)


    def calculate_gps_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_m = 6378137.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
        return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


ACTIVE_TASK_NAME = "task3"
HOLD_MODE_NAME = "HOLD"
MIN_VALID_ABS_COORD = 1e-6
REAL_RUN_ACK_VALUE = "YES_I_ACCEPT_REAL_ROBOT_RISK"


class DockingState(Enum):
    WAIT_START = auto()
    GO_TO_APPROACH_POINT = auto()
    SEARCH_DOCK = auto()
    ALIGN_TO_TAG = auto()
    FINAL_APPROACH = auto()
    HOLD_POSITION = auto()
    REVERSE_EXIT = auto()
    GO_TO_EXIT_POINT = auto()
    MODE_FINISHED = auto()
    FINISHED = auto()
    FAILSAFE = auto()


@dataclass
class GpsPoint:
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass
class ModeConfig:
    name: str
    hold_seconds: float
    berth_width_m: float
    berth_length_m: float
    approach_point: GpsPoint = field(default_factory=GpsPoint)
    exit_point: GpsPoint = field(default_factory=GpsPoint)
    reverse_seconds: float = 3.0
    stop_area_ratio: float = 0.060
    search_timeout_seconds: float = 20.0
    final_approach_timeout_seconds: float = 18.0
    target_payloads: Tuple[str, ...] = field(default_factory=tuple)

    # Opsiyonel global liman/dock açısı.
    # Verilirse ALIGN_TO_TAG aşamasında önce tekne bu global heading'e hizalanır,
    # sonra QR görüntü merkezleme ile ince hizalama yapılır.
    # Verilmezse eski davranış korunur: sadece QR merkezleme kullanılır.
    dock_global_heading_deg: Optional[float] = None
    dock_heading_tolerance_deg: float = 8.0
    dock_heading_kp: float = 0.015


@dataclass
class Task3Config:
    dry_run: bool = True
    auto_start: bool = True
    arm_vehicle: bool = True
    set_guided_mode: bool = True
    require_gps_for_visual_docking: bool = False
    real_run_acknowledged: bool = False
    gps_timeout_sec: float = 2.0
    heading_timeout_sec: float = 2.0
    bridge_state_timeout_sec: float = 2.0
    geofence_radius_m: float = 150.0
    waypoint_tolerance_m: float = 1.2
    control_hz: float = 10.0
    qr_topic: str = "/njord/task3/qr_detections"
    docking_state_topic: str = "/njord/task3/docking_state"
    active_task_topic: str = "/mission/active_task"
    min_tag_confidence: float = 0.20
    qr_detection_timeout_sec: float = 1.0
    allowed_payloads: Tuple[str, ...] = ("middle_berth_1", "middle_berth_2", "middle_parallel")
    image_center_deadband_norm: float = 0.08
    align_kp_yaw: float = 0.45
    final_kp_yaw: float = 0.28
    max_yaw_speed: float = 0.35
    search_yaw_speed: float = 0.18
    search_sweep_seconds: float = 4.0
    approach_linear_speed: float = 0.5
    final_linear_speed: float = 0.5
    reverse_linear_speed: float = -0.14
    max_frame_age_ms: float = 500.0
    require_qr_confirmation: bool = True
    confirmation_window_size: int = 6
    confirmation_required_count: int = 3
    confirmation_max_age_sec: float = 1.5
    sequence: Tuple[str, ...] = ("normal", "parallel")
    modes: Dict[str, ModeConfig] = field(default_factory=dict)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def create_default_config() -> Task3Config:
    cfg = Task3Config()
    cfg.dry_run = _env_bool("TASK3_DRY_RUN", cfg.dry_run)
    cfg.arm_vehicle = _env_bool("TASK3_ARM_VEHICLE", cfg.arm_vehicle)
    cfg.set_guided_mode = _env_bool("TASK3_SET_GUIDED_MODE", cfg.set_guided_mode)
    cfg.real_run_acknowledged = os.environ.get("TASK3_REAL_RUN_ACK") == REAL_RUN_ACK_VALUE
    cfg.modes = {
        "normal": ModeConfig(
            name="normal",
            hold_seconds=10.0,
            berth_width_m=2.0,
            berth_length_m=2.0,
            reverse_seconds=3.0,
            stop_area_ratio=0.060,
            final_approach_timeout_seconds=18.0,
            target_payloads=("middle_berth_1", "middle_berth_2"),
        ),
        "parallel": ModeConfig(
            name="parallel",
            hold_seconds=5.0,
            berth_width_m=2.0,
            berth_length_m=4.0,
            reverse_seconds=2.0,
            stop_area_ratio=0.045,
            final_approach_timeout_seconds=16.0,
            target_payloads=("middle_parallel",),
        ),
    }
    return cfg


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_diff_deg(target_deg: float, current_deg: float) -> float:
    """[-180, 180] aralığında hedef açı - mevcut açı farkını döndürür."""
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return diff


def normalize_payload(payload: Optional[str]) -> Optional[str]:
    if not payload:
        return None
    normalized = str(payload).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("middle_birth_", "middle_berth_")
    return normalized


@dataclass
class QrDetection:
    timestamp: float
    canonical_payload: Optional[str]
    confidence: float
    center_x: float
    center_y: float
    bbox_width: float
    bbox_height: float
    frame_width: float
    frame_height: float
    payload_valid: bool

    @property
    def area_ratio(self) -> float:
        if self.frame_width <= 0 or self.frame_height <= 0:
            return 0.0
        return (self.bbox_width * self.bbox_height) / (self.frame_width * self.frame_height)

    @property
    def x_error_norm(self) -> float:
        if self.frame_width <= 0:
            return 0.0
        image_center_x = self.frame_width / 2.0
        return (self.center_x - image_center_x) / max(image_center_x, 1.0)


class Task3DockingMission:
    def __init__(self, node: Node, config: Task3Config, cmd_vel_pub: Any, position_target_pub: Any):
        self.node = node
        self.logger = node.get_logger()
        self.config = config
        self.cmd_vel_pub = cmd_vel_pub
        self.position_target_pub = position_target_pub

        self.state = DockingState.WAIT_START
        self.finished = False
        self.mode_index = 0
        self.current_mode_name = self.config.sequence[0]
        self.state_enter_time = time.monotonic()
        self.mode_started_at = self.state_enter_time

        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.current_heading: Optional[float] = None
        self.home_lat: Optional[float] = None
        self.home_lon: Optional[float] = None
        self.last_gps_time: Optional[float] = None
        self.last_heading_time: Optional[float] = None
        self.last_qr_detection: Optional[QrDetection] = None
        self.last_qr_message_time: Optional[float] = None
        self.qr_history: List[QrDetection] = []
        self.last_rejected_qr_payload: Optional[str] = None
        self.bridge_connected = False
        self.last_bridge_state_time: Optional[float] = None
        self.last_cmd_log_time = 0.0
        self.last_position_log_time = 0.0
        self._missing_config_logged: set[str] = set()

    @property
    def current_mode(self) -> ModeConfig:
        return self.config.modes[self.current_mode_name]

    def set_state(self, new_state: DockingState, reason: str = "") -> None:
        if self.state == new_state:
            return
        old = self.state
        self.state = new_state
        self.state_enter_time = time.monotonic()
        suffix = f" | {reason}" if reason else ""
        self.logger.info(f"Task3 state: {old.name} -> {new_state.name}{suffix}")

    def update_gps(self, lat: float, lon: float) -> None:
        self.current_lat = lat
        self.current_lon = lon
        self.last_gps_time = time.monotonic()
        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Task3 home position set: {lat:.7f}, {lon:.7f}")

    def update_heading(self, heading_deg: float) -> None:
        self.current_heading = heading_deg % 360.0
        self.last_heading_time = time.monotonic()

    def update_bridge_state(self, data: str) -> None:
        normalized = data.replace(" ", "").lower()
        self.bridge_connected = "connected=true" in normalized
        self.last_bridge_state_time = time.monotonic()

    def update_qr_from_json(self, payload: Dict[str, Any]) -> None:
        detection = self._select_best_qr_detection(payload)
        self.last_qr_message_time = time.monotonic()
        if detection is not None:
            self.last_qr_detection = detection
            self.qr_history.append(detection)
            max_keep = max(self.config.confirmation_window_size * 3, 20)
            self.qr_history = self.qr_history[-max_keep:]

    def _select_best_qr_detection(self, payload: Dict[str, Any]) -> Optional[QrDetection]:
        frame = payload.get("frame_px") or {}
        frame_width = _coerce_float(frame.get("width")) or 1280.0
        frame_height = _coerce_float(frame.get("height")) or 720.0
        detections = payload.get("detections") or []
        if not isinstance(detections, list):
            return None

        candidates: List[QrDetection] = []
        now = time.monotonic()
        for item in detections:
            if not isinstance(item, dict):
                continue
            confidence = _coerce_float(item.get("confidence")) or 0.0
            if confidence < self.config.min_tag_confidence:
                continue
            center = item.get("center_px") or {}
            bbox = item.get("bbox_xywh_px") or {}
            center_x = _coerce_float(center.get("x"))
            center_y = _coerce_float(center.get("y"))
            bbox_w = _coerce_float(bbox.get("width"))
            bbox_h = _coerce_float(bbox.get("height"))
            if None in (center_x, center_y, bbox_w, bbox_h):
                continue

            canonical = normalize_payload(item.get("canonical_payload") or item.get("payload"))
            payload_valid = canonical in self.config.allowed_payloads
            candidates.append(
                QrDetection(
                    timestamp=now,
                    canonical_payload=canonical,
                    confidence=confidence,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    bbox_width=float(bbox_w),
                    bbox_height=float(bbox_h),
                    frame_width=float(frame_width),
                    frame_height=float(frame_height),
                    payload_valid=payload_valid,
                )
            )

        if not candidates:
            return None

        # Öncelik: geçerli payload + yüksek confidence + görüntü merkezine yakınlık.
        return max(
            candidates,
            key=lambda d: (
                1 if d.payload_valid else 0,
                d.confidence,
                -abs(d.x_error_norm),
                d.area_ratio,
            ),
        )

    def _qr_is_fresh(self) -> bool:
        if self.last_qr_detection is None:
            return False
        return (time.monotonic() - self.last_qr_detection.timestamp) <= self.config.qr_detection_timeout_sec

    def _confirmation_count(self, detection: Optional[QrDetection]) -> int:
        if detection is None or not detection.payload_valid or detection.canonical_payload is None:
            return 0
        now = time.monotonic()
        recent = [
            item
            for item in self.qr_history[-max(self.config.confirmation_window_size, 1):]
            if (now - item.timestamp) <= self.config.confirmation_max_age_sec
        ]
        return sum(1 for item in recent if item.canonical_payload == detection.canonical_payload)

    def _payload_confirmed(self, detection: Optional[QrDetection]) -> bool:
        if not self.config.require_qr_confirmation:
            return True
        return self._confirmation_count(detection) >= max(self.config.confirmation_required_count, 1)

    def _payload_matches_current_mode(self, detection: Optional[QrDetection]) -> bool:
        if detection is None:
            return False
        if not detection.payload_valid:
            return False
        targets = self.current_mode.target_payloads
        if not targets:
            return True
        return detection.canonical_payload in targets

    def _qr_is_usable_for_current_mode(self) -> bool:
        detection = self.last_qr_detection
        if not self._qr_is_fresh() or not self._payload_matches_current_mode(detection):
            if detection is not None and self._qr_is_fresh():
                self.last_rejected_qr_payload = detection.canonical_payload
            return False
        if not self._payload_confirmed(detection):
            return False
        return True

    def _search_yaw_command(self) -> float:
        sweep = max(self.config.search_sweep_seconds, 0.5)
        phase = int(self._state_age() // sweep)
        direction = 1.0 if phase % 2 == 0 else -1.0
        return direction * self.config.search_yaw_speed

    def _send_cmd(self, linear_x: float, angular_z: float, reason: str = "") -> None:
        linear_x = float(linear_x)
        angular_z = float(angular_z)
        if self.config.dry_run:
            now = time.monotonic()
            if now - self.last_cmd_log_time > 1.0:
                self.logger.info(
                    f"[DRY-RUN] cmd_vel linear.x={linear_x:.3f}, angular.z={angular_z:.3f} {reason}"
                )
                self.last_cmd_log_time = now
            return
        publish_cmd_vel(self.cmd_vel_pub, linear_x=linear_x, angular_z=angular_z)

    def stop(self, reason: str = "") -> None:
        if self.config.dry_run:
            self._send_cmd(0.0, 0.0, reason=reason or "stop")
        else:
            stop_vehicle(self.cmd_vel_pub)

    def _send_position_target(self, lat: float, lon: float, target_name: str, distance_m: float) -> None:
        if self.config.dry_run:
            now = time.monotonic()
            if now - self.last_position_log_time > 1.0:
                self.logger.info(
                    f"[DRY-RUN] set_position {target_name}: "
                    f"lat={lat:.7f}, lon={lon:.7f}, distance={distance_m:.2f}m"
                )
                self.last_position_log_time = now
            return

        publish_set_position(self.position_target_pub, lat, lon)
        self.logger.info(
            f"set_position {target_name}: lat={lat:.7f}, lon={lon:.7f}, distance={distance_m:.2f}m",
            throttle_duration_sec=1.0,
        )

    def _watchdog_ok(self) -> bool:
        now = time.monotonic()
        if not self.config.dry_run:
            if self.last_bridge_state_time is None:
                self.logger.info("Waiting for bridge state...", throttle_duration_sec=2.0)
                self.stop("waiting_bridge_state")
                return False
            if now - self.last_bridge_state_time > self.config.bridge_state_timeout_sec:
                self._enter_failsafe(f"Bridge state timeout > {self.config.bridge_state_timeout_sec}s")
                return False
            if not self.bridge_connected:
                self._enter_failsafe("Bridge reports MAVLink disconnected")
                return False

        # GPS noktalarına gitme aktifse GPS zorunlu. Sadece görsel dry-run için config ile esnetilebilir.
        gps_required = self.config.require_gps_for_visual_docking or self._current_gps_target() is not None
        if gps_required:
            if self.last_gps_time is None or self.current_lat is None or self.current_lon is None:
                self.logger.info("Waiting for GPS data...", throttle_duration_sec=2.0)
                self.stop("waiting_gps")
                return False
            if now - self.last_gps_time > self.config.gps_timeout_sec:
                self._enter_failsafe(f"GPS timeout > {self.config.gps_timeout_sec}s")
                return False

        if self.last_heading_time is not None and now - self.last_heading_time > self.config.heading_timeout_sec:
            self._enter_failsafe(f"Heading timeout > {self.config.heading_timeout_sec}s")
            return False

        if self.home_lat is not None and self.current_lat is not None:
            dist = calculate_gps_distance(self.home_lat, self.home_lon, self.current_lat, self.current_lon)
            if dist > self.config.geofence_radius_m:
                self._enter_failsafe(f"Geofence violation: {dist:.1f}m")
                return False

        return True

    def _enter_failsafe(self, reason: str) -> None:
        if self.state != DockingState.FAILSAFE:
            self.logger.error(f"Task3 FAILSAFE: {reason}")
        self.set_state(DockingState.FAILSAFE, reason)
        self.stop("failsafe")

    def _current_gps_target(self) -> Optional[GpsPoint]:
        if self.state == DockingState.GO_TO_APPROACH_POINT and self.current_mode.approach_point.is_valid:
            return self.current_mode.approach_point
        if self.state == DockingState.GO_TO_EXIT_POINT and self.current_mode.exit_point.is_valid:
            return self.current_mode.exit_point
        return None

    def _navigate_to_point(self, point: GpsPoint, target_name: str) -> bool:
        if self.current_lat is None or self.current_lon is None:
            self.stop("waiting_position")
            return False

        distance = calculate_gps_distance(self.current_lat, self.current_lon, point.latitude, point.longitude)
        if distance <= self.config.waypoint_tolerance_m:
            self.logger.info(f"Reached {target_name}: remaining={distance:.2f}m")
            self.stop(f"reached_{target_name}")
            return True

        self._send_position_target(point.latitude, point.longitude, target_name, distance)
        return False

    def _global_dock_heading_error(self) -> Optional[float]:
        target_heading = self.current_mode.dock_global_heading_deg
        if target_heading is None or self.current_heading is None:
            return None
        return angle_diff_deg(target_heading, self.current_heading)

    def _align_to_global_dock_heading_if_needed(self) -> bool:
        """
        Global dock açısı config'te verildiyse önce bu açıya hizalanır.

        True dönerse bu tick içinde komut gönderilmiştir ve state devam etmelidir.
        False dönerse global heading hizası tamamdır veya heading config'i yoktur.
        """
        target_heading = self.current_mode.dock_global_heading_deg
        if target_heading is None:
            return False
        if self.current_heading is None:
            self.stop("waiting_heading_for_dock_global_alignment")
            return True
        error = angle_diff_deg(target_heading, self.current_heading)
        tolerance = max(self.current_mode.dock_heading_tolerance_deg, 0.5)
        if abs(error) <= tolerance:
            return False
        # Repo içindeki waypoint kontrolüyle aynı işaret mantığı korunuyor:
        # hedef - mevcut pozitifse angular_z negatif yönde uygulanıyor.
        angular_z = clamp(
            -self.current_mode.dock_heading_kp * error,
            -self.config.max_yaw_speed,
            self.config.max_yaw_speed,
        )
        self._send_cmd(0.0, angular_z, reason=f"align_global_dock_heading error={error:.1f}deg")
        return True

    def _align_command(self, detection: QrDetection, kp: float, forward_speed: float) -> Tuple[float, float, bool]:
        x_error = detection.x_error_norm
        aligned = abs(x_error) <= self.config.image_center_deadband_norm
        angular_z = clamp(-kp * x_error, -self.config.max_yaw_speed, self.config.max_yaw_speed)
        if not aligned and abs(x_error) > 0.35:
            forward_speed = min(forward_speed, 0.06)
        return forward_speed, angular_z, aligned

    def _state_age(self) -> float:
        return time.monotonic() - self.state_enter_time

    def _finish_current_mode(self) -> None:
        self.logger.info(f"Task3 mode finished: {self.current_mode_name}")
        self.mode_index += 1
        if self.mode_index >= len(self.config.sequence):
            self.set_state(DockingState.FINISHED, "all_modes_done")
            self.finished = True
            self.stop("finished")
            return
        self.current_mode_name = self.config.sequence[self.mode_index]
        self.mode_started_at = time.monotonic()
        self.logger.info(f"Task3 next mode: {self.current_mode_name}")
        self.set_state(DockingState.GO_TO_APPROACH_POINT, "next_mode")

    def update(self) -> None:
        if self.state == DockingState.FINISHED:
            self.stop("finished")
            return
        if self.state == DockingState.FAILSAFE:
            self.stop("failsafe")
            return
        if not self._watchdog_ok():
            return

        if self.state == DockingState.WAIT_START:
            if self.config.auto_start:
                self.set_state(DockingState.GO_TO_APPROACH_POINT, "auto_start")
            else:
                self.stop("wait_start")
            return

        if self.state == DockingState.GO_TO_APPROACH_POINT:
            target = self.current_mode.approach_point
            if target.is_valid:
                if self._navigate_to_point(target, f"{self.current_mode_name}_approach"):
                    self.set_state(DockingState.SEARCH_DOCK, "approach_reached")
                return
            key = f"{self.current_mode_name}_approach_missing"
            if key not in self._missing_config_logged:
                self.logger.warn(
                    f"{self.current_mode_name}: approach_point config missing. "
                    "Skipping GPS approach and using QR visual docking."
                )
                self._missing_config_logged.add(key)
            self.set_state(DockingState.SEARCH_DOCK, "no_approach_point")
            return

        if self.state == DockingState.SEARCH_DOCK:
            if self._qr_is_usable_for_current_mode():
                self.set_state(DockingState.ALIGN_TO_TAG, "target_qr_detected")
                return
            if self._qr_is_fresh() and not self._payload_matches_current_mode(self.last_qr_detection):
                det = self.last_qr_detection
                self.logger.warn(
                    f"QR detected but not valid for mode={self.current_mode_name}: "
                    f"payload={det.canonical_payload if det else None}",
                    throttle_duration_sec=2.0,
                )
            if self._state_age() > self.current_mode.search_timeout_seconds:
                self._enter_failsafe(f"Target QR not detected within {self.current_mode.search_timeout_seconds}s")
                return
            self._send_cmd(0.0, self._search_yaw_command(), reason="search_target_qr")
            return

        if self.state == DockingState.ALIGN_TO_TAG:
            if not self._qr_is_usable_for_current_mode():
                self.set_state(DockingState.SEARCH_DOCK, "target_qr_lost_or_wrong")
                return
            if self._align_to_global_dock_heading_if_needed():
                return
            detection = self.last_qr_detection
            linear_x, angular_z, aligned = self._align_command(
                detection, kp=self.config.align_kp_yaw, forward_speed=0.0
            )
            self._send_cmd(linear_x, angular_z, reason="align_to_tag")
            if aligned:
                self.set_state(DockingState.FINAL_APPROACH, "tag_centered")
            return

        if self.state == DockingState.FINAL_APPROACH:
            if not self._qr_is_usable_for_current_mode():
                self.set_state(DockingState.SEARCH_DOCK, "target_qr_lost_or_wrong_in_final")
                return
            if self._state_age() > self.current_mode.final_approach_timeout_seconds:
                self._enter_failsafe(
                    f"Final approach timeout > {self.current_mode.final_approach_timeout_seconds}s"
                )
                return
            detection = self.last_qr_detection
            if detection.area_ratio >= self.current_mode.stop_area_ratio:
                self.stop("dock_reached_by_bbox_area")
                self.set_state(DockingState.HOLD_POSITION, f"area_ratio={detection.area_ratio:.3f}")
                return
            linear_x, angular_z, _ = self._align_command(
                detection, kp=self.config.final_kp_yaw, forward_speed=self.config.final_linear_speed
            )
            self._send_cmd(linear_x, angular_z, reason="final_approach")
            return

        if self.state == DockingState.HOLD_POSITION:
            self.stop("hold_position")
            if self._state_age() >= self.current_mode.hold_seconds:
                self.set_state(DockingState.REVERSE_EXIT, "hold_complete")
            return

        if self.state == DockingState.REVERSE_EXIT:
            if self._state_age() < self.current_mode.reverse_seconds:
                self._send_cmd(self.config.reverse_linear_speed, 0.0, reason="reverse_exit")
                return
            self.stop("reverse_complete")
            if self.current_mode.exit_point.is_valid:
                self.set_state(DockingState.GO_TO_EXIT_POINT, "reverse_complete")
            else:
                self.set_state(DockingState.MODE_FINISHED, "no_exit_point")
            return

        if self.state == DockingState.GO_TO_EXIT_POINT:
            if self._navigate_to_point(self.current_mode.exit_point, f"{self.current_mode_name}_exit"):
                self.set_state(DockingState.MODE_FINISHED, "exit_reached")
            return

        if self.state == DockingState.MODE_FINISHED:
            self._finish_current_mode()
            return

    def status_payload(self) -> Dict[str, Any]:
        detection = self.last_qr_detection
        return {
            "active_task": ACTIVE_TASK_NAME,
            "mode": self.current_mode_name,
            "mode_index": self.mode_index,
            "sequence": list(self.config.sequence),
            "state": self.state.name,
            "dock_global_heading_deg": self.current_mode.dock_global_heading_deg,
            "dock_heading_error_deg": self._global_dock_heading_error(),
            "dock_heading_tolerance_deg": self.current_mode.dock_heading_tolerance_deg,
            "dry_run": self.config.dry_run,
            "real_run_acknowledged": self.config.real_run_acknowledged,
            "bridge_connected": self.bridge_connected,
            "bridge_state_age_sec": None
            if self.last_bridge_state_time is None
            else time.monotonic() - self.last_bridge_state_time,
            "finished": self.finished,
            "gps_available": self.current_lat is not None and self.current_lon is not None,
            "heading_available": self.current_heading is not None,
            "qr_fresh": self._qr_is_fresh(),
            "qr": None
            if detection is None
            else {
                "canonical_payload": detection.canonical_payload,
                "payload_valid": detection.payload_valid,
                "payload_matches_current_mode": self._payload_matches_current_mode(detection),
                "payload_confirmed": self._payload_confirmed(detection),
                "confirmation_count": self._confirmation_count(detection),
                "confirmation_required_count": self.config.confirmation_required_count,
                "target_payloads": list(self.current_mode.target_payloads),
                "confidence": detection.confidence,
                "x_error_norm": detection.x_error_norm,
                "area_ratio": detection.area_ratio,
                "last_rejected_payload": self.last_rejected_qr_payload,
            },
        }


class Task3DockingNode(Node):
    def __init__(self, config: Task3Config):
        super().__init__("task3_docking_node")
        self.config = config
        self.get_logger().info("Task 3 Docking Node starting...")

        if create_mission_topics is not None:
            self.mission_topics = create_mission_topics(
                self,
                gps_callback=self.gps_callback,
                heading_callback=self.heading_callback,
                state_callback=self.bridge_state_callback,
            )
            self.cmd_vel_pub = self.mission_topics.cmd_vel_pub
            self.position_target_pub = self.mission_topics.position_target_pub
        else:
            self.cmd_vel_pub = self.create_publisher(Twist, "/cube/cmd_vel", 10)
            self.position_target_pub = self.create_publisher(NavSatFix, "/cube/set_position", 10)
            self.create_subscription(NavSatFix, "/cube/gps", self.gps_callback, 10)
            self.create_subscription(Float32, "/cube/gps/heading", self.heading_callback, 10)
            self.create_subscription(String, "/cube/state", self.bridge_state_callback, 10)

        self.mission_clients = None
        if create_mission_clients is not None:
            self.mission_clients = create_mission_clients(self)
            if not self.config.dry_run and wait_for_mission_services is not None:
                wait_for_mission_services(self, self.mission_clients)

        self.qr_sub = self.create_subscription(String, self.config.qr_topic, self.qr_callback, 10)
        self.active_task_pub = self.create_publisher(String, self.config.active_task_topic, 10)
        self.docking_state_pub = self.create_publisher(String, self.config.docking_state_topic, 10)

        self.mission = Task3DockingMission(
            self,
            self.config,
            self.cmd_vel_pub,
            self.position_target_pub,
        )
        self.control_timer = self.create_timer(1.0 / max(self.config.control_hz, 1.0), self.control_tick)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.active_task_timer = self.create_timer(1.0, self.publish_active_task)

        self.get_logger().warn(
            "Task3 V3.1 single package loaded. dry_run=%s. "
            "Set dry_run=false only after bench test."
            % self.config.dry_run
        )

    def publish_active_task(self) -> None:
        msg = String()
        msg.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(msg)

    def gps_callback(self, msg: NavSatFix) -> None:
        if abs(msg.latitude) < MIN_VALID_ABS_COORD and abs(msg.longitude) < MIN_VALID_ABS_COORD:
            self.get_logger().warn("Invalid GPS (0,0) ignored.", throttle_duration_sec=2.0)
            return
        self.mission.update_gps(msg.latitude, msg.longitude)

    def heading_callback(self, msg: Float32) -> None:
        self.mission.update_heading(float(msg.data))

    def bridge_state_callback(self, msg: String) -> None:
        self.mission.update_bridge_state(msg.data)

    def qr_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid QR JSON ignored: {exc}", throttle_duration_sec=2.0)
            return
        if not isinstance(payload, dict):
            self.get_logger().warn("QR JSON root is not object; ignored.", throttle_duration_sec=2.0)
            return
        self.mission.update_qr_from_json(payload)

    def control_tick(self) -> None:
        try:
            self.mission.update()
        except Exception as exc:  # noqa: BLE001 - failsafe için geniş yakalama
            self.get_logger().error(f"Unexpected Task3 control error: {exc}")
            self.mission._enter_failsafe(str(exc))

    def publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self.mission.status_payload(), ensure_ascii=False)
        self.docking_state_pub.publish(msg)

    def prepare_vehicle(self) -> bool:
        if self.config.dry_run:
            self.get_logger().warn("dry_run=true: vehicle mode/arm commands will not be sent.")
            return True

        if not self.config.real_run_acknowledged:
            self.get_logger().error(
                f"dry_run=false blocked: export TASK3_REAL_RUN_ACK={REAL_RUN_ACK_VALUE} before real run."
            )
            return False

        if self.mission_clients is None:
            self.get_logger().error("Mission clients are unavailable; cannot prepare vehicle.")
            return False
        if self.config.set_guided_mode and call_set_mode is not None:
            ok = call_set_mode(self, self.mission_clients.set_mode_client, "GUIDED")
            if ok is False:
                self.get_logger().error("Failed to switch to GUIDED mode.")
                return False
        if self.config.arm_vehicle and call_trigger_service is not None:
            ok = call_trigger_service(self, self.mission_clients.force_arm_client, "FORCE ARM")
            if ok is False:
                self.get_logger().error("FORCE ARM failed.")
                return False
        return True

    def shutdown_vehicle(self) -> None:
        try:
            self.mission.stop("shutdown")
        finally:
            if not self.config.dry_run and self.mission_clients is not None and call_set_mode is not None:
                try:
                    call_set_mode(self, self.mission_clients.set_mode_client, HOLD_MODE_NAME, timeout_sec=2.0)
                except Exception:  # noqa: BLE001
                    pass
            if not self.config.dry_run and self.config.arm_vehicle and call_trigger_service is not None:
                try:
                    call_trigger_service(self, self.mission_clients.disarm_client, "DISARM")
                except Exception:  # noqa: BLE001
                    pass


def main(args: Optional[List[str]] = None) -> None:
    config = create_default_config()
    rclpy.init(args=args)
    node = Task3DockingNode(config)
    try:
        if not node.prepare_vehicle():
            return
        node.publish_active_task()
        while rclpy.ok() and not node.mission.finished and node.mission.state != DockingState.FAILSAFE:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Task3 docking interrupted manually.")
    finally:
        node.shutdown_vehicle()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
