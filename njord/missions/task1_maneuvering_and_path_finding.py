import json
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from std_msgs.msg import String

from njord.core.mission_decision import DECISION_TOPIC, mission_decision_json
from njord.config.mission_config import WAYPOINT_DIRECTORY
from utils.mavlink_utilities import (
    align_heading_to_gps_target,
    create_mission_topics,
    create_mission_clients,
    wait_for_mission_services,
    call_set_mode,
    call_trigger_service,
    parse_bridge_state,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    calculate_gps_distance
)
from utils.read_waypoints import parse_qgc_waypoints

WAYPOINT_PATH = WAYPOINT_DIRECTORY / "njord_task1.waypoints"
ACTIVE_TASK_NAME = "task1"

# ============================================================
# GÜVENLİK PARAMETRELERİ
# ============================================================
GPS_TIMEOUT_SEC = 2.0  # Bu sure GPS gelmezse dur ve HOLD moda gecmeyi dene
HEADING_TIMEOUT_SEC = 2.0  # Bu sure heading gelmezse dur ve HOLD moda gecmeyi dene
BRIDGE_STATE_TIMEOUT_SEC = 10.0  # /cube/state bu sure gelmezse FAILSAFE + HOLD
GEOFENCE_RADIUS_M = 150.0  # Başlangıç noktasından max uzaklık
MIN_VALID_ABS_COORD = 1e-6
HOLD_MODE_NAME = "HOLD"

# ============================================================
# NAVİGASYON PARAMETRELERİ
# ============================================================
WAYPOINT_TOLERANCE_M = 1.0
WAYPOINT_SETTLE_SEC = 0.75  # Her ana GPS noktasinda kesin durus suresi
WAYPOINT_HEADING_TOLERANCE_DEG = 15.0  # Kucuk heading farklarinda gereksiz salinimi onler

# ============================================================
# KAÇINMA PARAMETRELERİ
# ============================================================
# Engel bu mesafeye veya daha yakına geldiğinde kaçınma başlatılır.
AVOIDANCE_START_DISTANCE_M = 3.0
AVOIDANCE_EXIT_DISTANCE_M = 5.0
AVOIDANCE_PASS_CLEARANCE_M = 2.5
AVOIDANCE_EMERGENCY_DISTANCE_M = 1.5
AVOIDANCE_MIN_LINEAR_SPEED = 0.2
AVOIDANCE_MAX_LINEAR_SPEED = 0.6
AVOIDANCE_MAX_ANGULAR_Z = 0.7
AVOIDANCE_TURN_SPEED_REDUCTION = 0.5
AVOIDANCE_MIN_DURATION_SEC = 0.8
AVOIDANCE_CLEAR_DURATION_SEC = 0.5
AVOIDANCE_TIMEOUT_SEC = 8.0

# ============================================================
# VISION / ENGEL EŞLEŞTİRME PARAMETRELERİ
# ============================================================
VISION_DETECTION_TIMEOUT_SEC = 1.0
MIN_OBSTACLE_CONFIDENCE = 0.45
OBSTACLE_CONFIRMATION_MAX_GAP_SEC = 0.75
OBSTACLE_FILTER_ALPHA = 0.40
OBSTACLE_MATCH_MAX_ANGLE_DELTA_DEG = 25.0
OBSTACLE_MATCH_MAX_DISTANCE_DELTA_M = 2.0
OBSTACLE_BBOX_MIN_IOU = 0.05
SIDE_FALLBACK_ANGLE_DEG = 15.0
RED_BUOY_CLASS = "red_buoys"
GREEN_BUOY_CLASS = "green_buoys"
EAST_CARDINAL_CLASS = "east_buoys"
WEST_CARDINAL_CLASS = "west_buoys"
OBSTACLE_CLASS_ALIASES = {
    "red_buoy": RED_BUOY_CLASS, "red_buoys": RED_BUOY_CLASS,
    "green_buoy": GREEN_BUOY_CLASS, "green_buoys": GREEN_BUOY_CLASS,
    "east_buoy": EAST_CARDINAL_CLASS, "east_buoys": EAST_CARDINAL_CLASS,
    "east_cardinal": EAST_CARDINAL_CLASS,
    "east_cardinal_buoy": EAST_CARDINAL_CLASS,
    "west_buoy": WEST_CARDINAL_CLASS, "west_buoys": WEST_CARDINAL_CLASS,
    "west_cardinal": WEST_CARDINAL_CLASS,
    "west_cardinal_buoy": WEST_CARDINAL_CLASS,
}
CARDINAL_PASS_SIDES = {
    EAST_CARDINAL_CLASS: "east",
    WEST_CARDINAL_CLASS: "west",
}
BUOY_PASS_SIDES = {
    RED_BUOY_CLASS: "starboard",
    GREEN_BUOY_CLASS: "port",
}
RELEVANT_OBSTACLE_CLASSES = (
    RED_BUOY_CLASS,
    GREEN_BUOY_CLASS,
    EAST_CARDINAL_CLASS,
    WEST_CARDINAL_CLASS,
)
DETECTION_ANGLE_KEYS = (
    "angle_deg",
    "Buoy angle: ",
    "Buoy angle",
    "angle_from_center",
    "angle",
)
DETECTION_SIDE_KEYS = ("Buoy side: ", "side", "buoy_side")


class MissionState(Enum):
    INIT = auto()  # Başlangıç konumu bekleniyor / WP0 doğrulanıyor
    NAVIGATING = auto()  # Normal waypoint takibi
    AVOIDING = auto()  # Şamandıra kaçınma
    FINISHED = auto()  # Görev tamamlandı
    FAILSAFE = auto()  # GPS kaybı / geofence ihlali / beklenmeyen hata


# ============================================================
# MISSION LOGIC
# ============================================================
class Task1Maneuvering:
    # Gorev durumunu, waypointleri ve guvenlik degiskenlerini hazirlar.
    def __init__(self, node, mission_topics, mission_clients):
        self.node = node
        self.logger = node.get_logger()

        self.topics = mission_topics
        self.clients = mission_clients

        self.logger.info(f"[INIT-DEBUG] Waypoint path: {WAYPOINT_PATH.resolve()}")

        self.waypoints = parse_qgc_waypoints(WAYPOINT_PATH)
        self.current_target_index = 0
        self.waypoint_tolerance = WAYPOINT_TOLERANCE_M

        self.logger.info(f"[INIT-DEBUG] Parsed waypoints: {self.waypoints}")

        # Anlık konum verileri
        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self.last_angular_z = 0.0
        self.finished = False

        # --- Güvenlik / state machine alanları ---
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self.last_bridge_state_time = None
        self.home_lat = None
        self.home_lon = None
        self.avoiding_class = None  # RELEVANT_OBSTACLE_CLASSES icinden biri veya None
        self.avoiding_track_id = None
        self.active_obstacle_reference = None
        self.pending_obstacle = None
        self.pending_obstacle_time = None
        self.pending_obstacle_count = 0
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.active_pass_side = None
        self.last_avoidance_linear_x = 0.0
        self.last_avoidance_angular_z = 0.0
        self.aligned_target_key = None
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None
        self.waiting_for_sensor_text = "GPS Data"
        self.hold_mode_requested = False
        self.hold_mode_future = None

    # GPS/heading bilgisini gunceller ve ilk konumu home noktasi yapar.
    def update_gps(self, lat, lon, heading):
        """ROS 2 Node'undan gelen güncel GPS ve yönelim verilerini kaydeder."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading
        self.last_gps_time = time.monotonic()

        if self.home_lat is None:
            # İlk GPS okuması home/geofence merkezi olarak kaydedilir
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Home position set: {lat:.6f}, {lon:.6f}")

    def update_bridge_state(self, connected, armed, mode, now=None):
        """Bridge heartbeat durumunu görev güvenlik denetimine aktarır."""
        self.bridge_connected = bool(connected)
        self.bridge_armed = bool(armed)
        self.bridge_mode = str(mode or "UNKNOWN").strip().upper()
        self.last_bridge_state_time = (
            time.monotonic() if now is None else float(now)
        )

    # GPS veya heading verisi gecikirse gorevi FAILSAFE durumuna alir.
    def _request_hold_mode(self):
        """FAILSAFE durumunda araci HOLD moda almak icin tek seferlik istek gonderir."""
        if self.hold_mode_requested:
            return

        self.hold_mode_requested = True
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = HOLD_MODE_NAME

        try:
            self.hold_mode_future = self.clients.set_mode_client.call_async(req)
            self.hold_mode_future.add_done_callback(self._hold_mode_done)
            self.logger.warn(f"Requesting {HOLD_MODE_NAME} mode due to failsafe.")
        except Exception as exc:  # noqa: BLE001 - failsafe mod istegi kesinlikle loglanmali
            self.logger.error(f"Failed to request {HOLD_MODE_NAME} mode: {exc}")

    def _hold_mode_done(self, future):
        """HOLD mod servis cevabini loglar."""
        try:
            res = future.result()
        except Exception as exc:  # noqa: BLE001 - ROS future hatasi loglanmali
            self.logger.error(f"{HOLD_MODE_NAME} mode response failed: {exc}")
            return

        if res is not None and getattr(res, "mode_sent", False):
            self.logger.warn(
                f"{HOLD_MODE_NAME} mode confirmed by Orange Cube heartbeat."
            )
        else:
            self.logger.error(
                f"{HOLD_MODE_NAME} mode could not be confirmed by Orange Cube."
            )

    def _enter_failsafe(self, reason, request_hold=False):
        """Araci FAILSAFE'e alir; gerekirse HOLD moda gecis istegi yollar."""
        if self.state != MissionState.FAILSAFE:
            self.logger.error(reason)

        self.state = MissionState.FAILSAFE

        if request_hold:
            self._request_hold_mode()

    def _check_watchdog(self):
        """Navigasyon ve araç durumu güvenliyse True döndürür."""
        now = time.monotonic()

        if self.last_gps_time is None:
            self.waiting_for_sensor_text = "GPS Data"
            return False

        if self.last_heading_time is None:
            self.waiting_for_sensor_text = "Heading Data"
            return False

        if (now - self.last_gps_time) > GPS_TIMEOUT_SEC:
            self._enter_failsafe(
                f"GPS DATA NOT RECEIVED FOR OVER {GPS_TIMEOUT_SEC}s! FAILSAFE + HOLD.",
                request_hold=True
            )
            return False

        if (now - self.last_heading_time) > HEADING_TIMEOUT_SEC:
            self._enter_failsafe(
                f"HEADING DATA NOT RECEIVED FOR OVER {HEADING_TIMEOUT_SEC}s! FAILSAFE + HOLD.",
                request_hold=True
            )
            return False

        if self.last_bridge_state_time is None:
            self._enter_failsafe(
                "BRIDGE STATE NOT RECEIVED! FAILSAFE + HOLD.",
                request_hold=True,
            )
            return False

        bridge_state_age = now - self.last_bridge_state_time
        if bridge_state_age > BRIDGE_STATE_TIMEOUT_SEC:
            self._enter_failsafe(
                f"BRIDGE STATE NOT RECEIVED FOR {bridge_state_age:.2f}s "
                f"(limit {BRIDGE_STATE_TIMEOUT_SEC:.1f}s)! FAILSAFE + HOLD.",
                request_hold=True,
            )
            return False

        if not self.bridge_connected:
            self._enter_failsafe(
                "MAVLINK BRIDGE DISCONNECTED! FAILSAFE + HOLD.",
                request_hold=True,
            )
            return False

        if self.bridge_mode != "GUIDED":
            self._enter_failsafe(
                f"ORANGE CUBE LEFT GUIDED MODE (mode={self.bridge_mode})! "
                "FAILSAFE + HOLD.",
                request_hold=True,
            )
            return False

        if not self.bridge_armed:
            self._enter_failsafe(
                "ORANGE CUBE IS NO LONGER ARMED! FAILSAFE + HOLD.",
                request_hold=True,
            )
            return False

        return True

    # Arac home merkezli izinli alanin disina cikti mi kontrol eder.
    def _check_geofence(self):
        """Home noktasından çok uzaklaşıldıysa FAILSAFE'e geç. True dönerse sınır içinde."""
        if self.home_lat is None or self.current_lat is None:
            return True

        dist_from_home = calculate_gps_distance(
            self.home_lat, self.home_lon,
            self.current_lat, self.current_lon
        )

        if dist_from_home > GEOFENCE_RADIUS_M:
            self._enter_failsafe(
                f"GEOFENCE VIOLATION! {dist_from_home:.1f}m away from home "
                f"(limit {GEOFENCE_RADIUS_M}m). FAILSAFE + HOLD.",
                request_hold=True
            )
            return False

        return True

    @staticmethod
    def _safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _normalize_obstacle(self, obj):
        if not isinstance(obj, dict):
            return None
        name = obj.get("class") or obj.get("class_name") or obj.get("label")
        name = OBSTACLE_CLASS_ALIASES.get(str(name).strip().lower())
        if name is None:
            return None
        distance = self._safe_float(obj.get("distance"))
        if distance is None:
            distance = self._safe_float(obj.get("distance_m"))
        confidence = self._safe_float(obj.get("confidence"))
        if confidence is None:
            confidence = self._safe_float(obj.get("conf"))
        if confidence is None:
            confidence = 1.0
        normalized = dict(obj)
        normalized.update({"class": name, "distance": distance,
                           "confidence": confidence})
        angle_deg = self._detection_angle_deg(normalized)
        side = None
        for key in DETECTION_SIDE_KEYS:
            side = self._normalize_side_text(normalized.get(key))
            if side is not None:
                break
        if angle_deg is None and side is not None:
            angle_deg = {
                "left": -SIDE_FALLBACK_ANGLE_DEG,
                "right": SIDE_FALLBACK_ANGLE_DEG,
                "center": 0.0,
            }[side]
        if angle_deg is not None:
            normalized["angle_deg"] = angle_deg
        if side is not None:
            normalized["side"] = side
        return normalized

    @staticmethod
    def _normalize_side_text(value):
        text = str(value or "").strip().lower()
        if text in ("left", "sol", "port"):
            return "left"
        if text in ("right", "sag", "sağ", "starboard"):
            return "right"
        if text in (
                "across", "center", "centre", "middle", "orta", "front",
        ):
            return "center"
        return None

    @staticmethod
    def _bbox_iou(first, second):
        if not (
                isinstance(first, (list, tuple))
                and isinstance(second, (list, tuple))
                and len(first) >= 4
                and len(second) >= 4
        ):
            return None
        try:
            ax1, ay1, ax2, ay2 = map(float, first[:4])
            bx1, by1, bx2, by2 = map(float, second[:4])
        except (TypeError, ValueError):
            return None
        intersection = (
            max(0.0, min(ax2, bx2) - max(ax1, bx1))
            * max(0.0, min(ay2, by2) - max(ay1, by1))
        )
        union = (
            max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            - intersection
        )
        return intersection / union if union > 0.0 else None

    def _same_obstacle(self, candidate, reference):
        if reference is None or candidate.get("class") != reference.get("class"):
            return reference is None

        candidate_track = candidate.get("track_id")
        reference_track = reference.get("track_id")
        if candidate_track is not None and reference_track is not None:
            if candidate_track == reference_track:
                return True

        bbox_iou = self._bbox_iou(
            candidate.get("bbox"),
            reference.get("bbox"),
        )
        bbox_match = bbox_iou is not None and bbox_iou >= OBSTACLE_BBOX_MIN_IOU

        angle = self._detection_angle_deg(candidate)
        reference_angle = self._detection_angle_deg(reference)
        distance = self._safe_float(candidate.get("distance"))
        reference_distance = self._safe_float(reference.get("distance"))
        motion_match = (
            angle is not None
            and reference_angle is not None
            and distance is not None
            and reference_distance is not None
            and abs(angle - reference_angle) <= OBSTACLE_MATCH_MAX_ANGLE_DELTA_DEG
            and abs(distance - reference_distance)
            <= OBSTACLE_MATCH_MAX_DISTANCE_DELTA_M
        )
        return bbox_match or motion_match

    def _obstacle_match_score(self, candidate, reference):
        if reference is None:
            return float(candidate["distance"])
        candidate_track = candidate.get("track_id")
        reference_track = reference.get("track_id")
        if candidate_track is not None and reference_track is not None:
            if candidate_track == reference_track:
                return 0.0
            track_penalty = 1.0
        elif reference_track is not None:
            track_penalty = 0.5
        else:
            track_penalty = 0.0

        bbox_iou = self._bbox_iou(
            candidate.get("bbox"),
            reference.get("bbox"),
        )
        bbox_penalty = 1.0 if bbox_iou is None else 1.0 - bbox_iou
        angle = self._detection_angle_deg(candidate)
        reference_angle = self._detection_angle_deg(reference)
        angle_penalty = (
            1.0
            if angle is None or reference_angle is None
            else abs(angle - reference_angle) / OBSTACLE_MATCH_MAX_ANGLE_DELTA_DEG
        )
        distance_penalty = (
            abs(float(candidate["distance"]) - float(reference["distance"]))
            / OBSTACLE_MATCH_MAX_DISTANCE_DELTA_M
        )
        return (
            track_penalty
            + bbox_penalty
            + angle_penalty
            + distance_penalty
        )

    def _filter_obstacle(self, obstacle, reference):
        filtered = dict(obstacle)
        if reference is None:
            return filtered
        distance = self._safe_float(obstacle.get("distance"))
        reference_distance = self._safe_float(reference.get("distance"))
        if distance is not None and reference_distance is not None:
            filtered["distance"] = (
                OBSTACLE_FILTER_ALPHA * distance
                + (1.0 - OBSTACLE_FILTER_ALPHA) * reference_distance
            )
        angle = self._detection_angle_deg(obstacle)
        reference_angle = self._detection_angle_deg(reference)
        if angle is not None and reference_angle is not None:
            filtered["angle_deg"] = (
                OBSTACLE_FILTER_ALPHA * angle
                + (1.0 - OBSTACLE_FILTER_ALPHA) * reference_angle
            )
        return filtered

    def _nearest_relevant_obstacle(self, detections, now=None):
        candidates = []
        for raw in detections or []:
            obj = self._normalize_obstacle(raw)
            if obj is None or obj["confidence"] < MIN_OBSTACLE_CONFIDENCE:
                continue
            if obj["distance"] is None or not 0 < obj["distance"] < AVOIDANCE_EXIT_DISTANCE_M:
                continue
            if self._detection_angle_deg(obj) is None:
                continue
            candidates.append(obj)
        return min(candidates, key=lambda x: x["distance"]) if candidates else None

    def _confirmed_obstacle(self, obstacle, now):
        if obstacle is None:
            self.pending_obstacle = None
            self.pending_obstacle_count = 0
            return None
        previous = getattr(self, "pending_obstacle", None)
        pending_time = getattr(self, "pending_obstacle_time", None)
        continuous = (
            previous is not None
            and pending_time is not None
            and now - pending_time <= OBSTACLE_CONFIRMATION_MAX_GAP_SEC
            and self._same_obstacle(obstacle, previous)
        )
        if continuous:
            obstacle = self._filter_obstacle(obstacle, previous)
        self.pending_obstacle_count = self.pending_obstacle_count + 1 if continuous else 1
        self.pending_obstacle = obstacle
        self.pending_obstacle_time = now
        if self.pending_obstacle_count < 2:
            return None
        self.pending_obstacle = None
        self.pending_obstacle_count = 0
        return obstacle

    @staticmethod
    def _detection_angle_deg(obstacle):
        """Detector ciktisindaki aci alanini derece olarak okur."""
        for key in DETECTION_ANGLE_KEYS:
            value = obstacle.get(key)
            if value is None:
                continue
            try:
                angle_deg = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(angle_deg):
                return angle_deg
        return None

    def _pass_side_and_body_offset(self, obstacle_class):
        """Sinif kuralini arac eksenindeki sanal gecis offsetine cevirir."""
        buoy_side = BUOY_PASS_SIDES.get(obstacle_class)
        if buoy_side is not None:
            starboard_m = (
                AVOIDANCE_PASS_CLEARANCE_M
                if buoy_side == "starboard"
                else -AVOIDANCE_PASS_CLEARANCE_M
            )
            return buoy_side, 0.0, starboard_m

        cardinal_side = CARDINAL_PASS_SIDES.get(obstacle_class)
        if cardinal_side is None or self.current_heading is None:
            return None

        east_m = (
            AVOIDANCE_PASS_CLEARANCE_M
            if cardinal_side == "east"
            else -AVOIDANCE_PASS_CLEARANCE_M
        )
        heading_rad = math.radians(float(self.current_heading))
        # Geographic east offsetini body forward/starboard eksenine dondur.
        forward_m = east_m * math.sin(heading_rad)
        starboard_m = east_m * math.cos(heading_rad)
        return cardinal_side, forward_m, starboard_m

    def _calculate_avoidance_command(self, obstacle):
        """Aci ve derinlikten dinamik ileri hiz ve heading offseti hesaplar."""
        obstacle = self._normalize_obstacle(obstacle)
        if obstacle is None:
            return None

        distance_m = self._safe_float(obstacle.get("distance"))
        angle_deg = self._detection_angle_deg(obstacle)
        pass_offset = self._pass_side_and_body_offset(obstacle.get("class"))
        if (
                distance_m is None
                or distance_m <= 0.0
                or angle_deg is None
                or pass_offset is None
        ):
            return None

        pass_side, offset_forward_m, offset_starboard_m = pass_offset
        angle_rad = math.radians(angle_deg)
        # Kameradaki engel konumu + sinifin zorunlu gecis acikligi.
        target_forward_m = distance_m * math.cos(angle_rad) + offset_forward_m
        target_starboard_m = (
            distance_m * math.sin(angle_rad) + offset_starboard_m
        )
        desired_heading_offset_rad = math.atan2(
            target_starboard_m,
            target_forward_m,
        )
        # Bridge angular_z degerini mevcut yaw'a radyan heading offseti olarak ekler.
        angular_z = max(
            -AVOIDANCE_MAX_ANGULAR_Z,
            min(AVOIDANCE_MAX_ANGULAR_Z, desired_heading_offset_rad),
        )

        if distance_m <= AVOIDANCE_EMERGENCY_DISTANCE_M:
            linear_x = 0.0
        else:
            distance_span_m = (
                AVOIDANCE_START_DISTANCE_M - AVOIDANCE_EMERGENCY_DISTANCE_M
            )
            distance_factor = min(
                1.0,
                max(
                    0.0,
                    (distance_m - AVOIDANCE_EMERGENCY_DISTANCE_M)
                    / distance_span_m,
                ),
            )
            distance_speed = (
                AVOIDANCE_MIN_LINEAR_SPEED
                + distance_factor
                * (AVOIDANCE_MAX_LINEAR_SPEED - AVOIDANCE_MIN_LINEAR_SPEED)
            )
            turn_factor = min(
                1.0,
                abs(angular_z) / AVOIDANCE_MAX_ANGULAR_Z,
            )
            linear_x = max(
                AVOIDANCE_MIN_LINEAR_SPEED,
                distance_speed
                * (1.0 - AVOIDANCE_TURN_SPEED_REDUCTION * turn_factor),
            )

        return {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "pass_side": pass_side,
            "distance": distance_m,
            "angle_deg": angle_deg,
        }

    def _publish_avoidance_command(self, command):
        self.last_avoidance_linear_x = command["linear_x"]
        self.last_avoidance_angular_z = command["angular_z"]
        self.active_pass_side = command["pass_side"]
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=command["linear_x"],
            angular_z=command["angular_z"],
        )

    def _republish_last_avoidance_command(self):
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=self.last_avoidance_linear_x,
            angular_z=self.last_avoidance_angular_z,
        )

    def _matching_avoidance_obstacle(self, detections):
        """Aktif kaçınma sınıfından hâlâ yakın görünen objeyi döndürür."""
        if self.avoiding_class is None:
            return None

        reference = getattr(self, "active_obstacle_reference", None)
        candidates = []
        for raw in detections or []:
            obj = self._normalize_obstacle(raw)
            if (
                    obj is None
                    or obj["confidence"] < MIN_OBSTACLE_CONFIDENCE
            ):
                continue
            if obj.get("class") != self.avoiding_class:
                continue
            if not self._same_obstacle(
                    obj,
                    reference,
            ):
                continue
            try:
                distance_m = float(obj.get("distance"))
            except (TypeError, ValueError):
                continue
            if (
                    0 < distance_m < AVOIDANCE_EXIT_DISTANCE_M
                    and self._detection_angle_deg(obj) is not None
            ):
                score = self._obstacle_match_score(obj, reference)
                candidates.append((score, obj))

        if not candidates:
            return None
        obstacle = min(candidates, key=lambda item: item[0])[1]
        filtered = self._filter_obstacle(obstacle, reference)
        self.active_obstacle_reference = filtered
        if obstacle.get("track_id") is not None:
            self.avoiding_track_id = obstacle["track_id"]
        return filtered

    def _reset_avoidance_state(self):
        self.avoiding_class = None
        self.avoiding_track_id = None
        self.active_obstacle_reference = None
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.active_pass_side = None
        self.last_avoidance_linear_x = 0.0
        self.last_avoidance_angular_z = 0.0
        self.aligned_target_key = None
        self.state = MissionState.NAVIGATING

    def _begin_waypoint_hold(self, waypoint_name):
        """Ana GPS noktasinda araci durdurup heading gecisi icin sabitler."""
        stop_vehicle(self.topics.cmd_vel_pub)
        self.waypoint_hold_until = time.monotonic() + WAYPOINT_SETTLE_SEC
        self.waypoint_hold_name = waypoint_name
        self.aligned_target_key = None
        self.logger.info(
            f"{waypoint_name} reached; vehicle stopped for "
            f"{WAYPOINT_SETTLE_SEC:.2f}s before next heading alignment."
        )

    def _waypoint_hold_active(self):
        """Planli waypoint durusu devam ediyorsa sifir hareket komutu basar."""
        if self.waypoint_hold_until is None:
            return False

        remaining = self.waypoint_hold_until - time.monotonic()
        if remaining > 0.0:
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=0.0)
            self.logger.info(
                f"Holding at {self.waypoint_hold_name}: {remaining:.2f}s remaining.",
                throttle_duration_sec=0.5,
            )
            return True

        completed_name = self.waypoint_hold_name
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None
        self.logger.info(
            f"{completed_name} stop stabilized; proceeding to next mission step."
        )
        return False

    # GPS hedefine MAVLink position target komutu basar.
    def _set_position_to_gps_target(self, target_lat, target_lon, target_name, tolerance_m):
        """Verilen GPS hedefine SET_POSITION_TARGET_GLOBAL_INT ile gider."""
        distance = calculate_gps_distance(
            self.current_lat, self.current_lon,
            target_lat, target_lon
        )

        if distance < tolerance_m:
            self.logger.info(f"Reached {target_name}! Remaining: {distance:.2f}m")
            return True

        target_key = (
            target_name,
            round(float(target_lat), 7),
            round(float(target_lon), 7),
        )
        if self.aligned_target_key != target_key:
            if not align_heading_to_gps_target(
                    self.topics.cmd_vel_pub,
                    self.current_lat,
                    self.current_lon,
                    self.current_heading,
                    target_lat,
                    target_lon,
                    logger=self.logger,
                    target_name=target_name,
                    tolerance_deg=WAYPOINT_HEADING_TOLERANCE_DEG,
            ):
                return False
            self.aligned_target_key = target_key

        publish_set_position(
            self.topics.position_target_pub,
            target_lat,
            target_lon
        )
        self.last_angular_z = 0.0

        self.logger.info(
            f"Target {target_name} | Distance: {distance:.2f}m | set_position sent",
            throttle_duration_sec=1.0
        )
        return False

    def _prepare_update(self):
        """Görev adımından önce güvenlik, rota ve bekleme koşullarını doğrular."""
        safety_ok = self._check_watchdog()

        if self.state == MissionState.FAILSAFE:
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.warn("FAILSAFE active, vehicle stopped.", throttle_duration_sec=2.0)
            return

        if not safety_ok:
            # Henüz zorunlu sensörlerden biri gelmediyse bekle.
            self.logger.info(f"Waiting for {self.waiting_for_sensor_text}...", throttle_duration_sec=2.0)
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=0.0)
            return

        if not self._check_geofence():
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if not self.waypoints:
            self.logger.warn("Mission list is empty! Please check the route.", throttle_duration_sec=5.0)
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if self._waypoint_hold_active():
            return

        if self.current_target_index >= len(self.waypoints):
            if not self.finished:
                self.logger.info("ALL WAYPOINTS REACHED! MISSION COMPLETED!")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.finished = True
                self.state = MissionState.FINISHED
            return False

        return True

    def _update_active_avoidance(self, detections, now):
        """Aktif kaçınmayı günceller; bu tick tüketildiyse True döndürür."""
        elapsed = (
            0.0
            if self.avoid_started_time is None
            else now - self.avoid_started_time
        )
        if elapsed >= AVOIDANCE_TIMEOUT_SEC:
            self._enter_failsafe(
                f"Obstacle did not clear within {AVOIDANCE_TIMEOUT_SEC:.1f}s. "
                "FAILSAFE + HOLD.",
                request_hold=True,
            )
            stop_vehicle(self.topics.cmd_vel_pub)
            return True

        obstacle = self._matching_avoidance_obstacle(detections)
        if obstacle is None:
            if self.avoid_clear_started_time is None:
                self.avoid_clear_started_time = now

            clear_duration = now - self.avoid_clear_started_time
            if (
                    elapsed >= AVOIDANCE_MIN_DURATION_SEC
                    and clear_duration >= AVOIDANCE_CLEAR_DURATION_SEC
            ):
                completed_class = self.avoiding_class
                completed_side = self.active_pass_side
                stop_vehicle(self.topics.cmd_vel_pub)
                self._reset_avoidance_state()
                self.logger.info(
                    f"{completed_class} cleared for {clear_duration:.2f}s; "
                    f"{completed_side} dynamic pass completed, "
                    "resuming main GNSS route."
                )
                return True

            self._republish_last_avoidance_command()
            self.logger.info(
                f"Obstacle temporarily out of frame; holding last dynamic command "
                f"for clear confirmation ({clear_duration:.2f}/"
                f"{AVOIDANCE_CLEAR_DURATION_SEC:.2f}s).",
                throttle_duration_sec=0.5,
            )
            return True

        self.avoid_clear_started_time = None
        command = self._calculate_avoidance_command(obstacle)
        if command is None:
            self._enter_failsafe(
                "Active obstacle has invalid angle/depth. FAILSAFE + HOLD.",
                request_hold=True,
            )
            stop_vehicle(self.topics.cmd_vel_pub)
            return True

        self._publish_avoidance_command(command)
        self.logger.info(
            f"Dynamic avoidance {obstacle['class']}: "
            f"distance={command['distance']:.2f}m, "
            f"angle={command['angle_deg']:.1f}deg, "
            f"side={command['pass_side']}, "
            f"linear={command['linear_x']:.2f}, "
            f"angular={command['angular_z']:.2f}.",
            throttle_duration_sec=0.5,
        )
        return True

    def _start_avoidance(self, obstacle, now):
        """Yeni bir engel için dinamik kamera manevrasını başlatır."""
        obstacle = self._normalize_obstacle(obstacle)
        if obstacle is None:
            return False
        command = self._calculate_avoidance_command(obstacle)
        if command is None:
            self.logger.warn(
                "Obstacle detection has no valid angle/depth; "
                "dynamic avoidance was not started.",
                throttle_duration_sec=1.0,
            )
            return False

        self.state = MissionState.AVOIDING
        self.avoiding_class = obstacle["class"]
        self.avoiding_track_id = obstacle.get("track_id")
        self.active_obstacle_reference = obstacle
        self.avoid_started_time = now
        self.avoid_clear_started_time = None
        self._publish_avoidance_command(command)
        side_reference = (
            "Geographic" if obstacle["class"] in CARDINAL_PASS_SIDES
            else "Vehicle-relative"
        )
        self.logger.info(
            f"{obstacle['class']} dynamic avoidance started: "
            f"distance={command['distance']:.2f}m, "
            f"angle={command['angle_deg']:.1f}deg, "
            f"{side_reference} side={command['pass_side']}, "
            f"linear={command['linear_x']:.2f}, "
            f"angular={command['angular_z']:.2f}, "
            f"clearance={AVOIDANCE_PASS_CLEARANCE_M:.1f}m."
        )
        return True

    def update(self, detections):
        """Güvenlik, kaçınma ve waypoint akışlarının ana kontrol döngüsü."""
        if not self._prepare_update():
            return

        target_gps = self.waypoints[self.current_target_index]
        target_lat = target_gps["lat"]
        target_lon = target_gps["lon"]

        distance = calculate_gps_distance(
            self.current_lat, self.current_lon,
            target_lat, target_lon
        )

        # ---------------------------------------------------------
        # 1. ENGELLERDEN KAÇINMA KONTROLÜ (dinamik kamera manevrasi)
        # ---------------------------------------------------------
        now = time.monotonic()
        nearest = self._nearest_relevant_obstacle(detections, now)

        if self.state == MissionState.AVOIDING:
            if self._update_active_avoidance(detections, now):
                return

        elif (
                nearest is not None
                and nearest["distance"] <= AVOIDANCE_START_DISTANCE_M
        ):
            confirmed = self._confirmed_obstacle(nearest, now)
            if confirmed is not None:
                if self._start_avoidance(confirmed, now):
                    return
        else:
            self._confirmed_obstacle(None, now)
        # ---------------------------------------------------------
        # 2. WP0 / MISSION BAŞLANGIÇ KONTROLÜ
        # ---------------------------------------------------------
        if self.state == MissionState.INIT:
            if self.current_target_index == 0 and distance < (self.waypoint_tolerance + 2.0):
                self.logger.info("WP0 (Start) point verified, mission starting.")
                self._begin_waypoint_hold("WP0 (Start)")
                self.current_target_index += 1
                self.state = MissionState.NAVIGATING
                return
            else:
                # Henüz start noktasında değiliz; WP0'a doğru ilerlemeye devam et,
                # ama mission'ı NAVIGATING'e geçirmeden (WP0'ı atlamadan).
                pass

        self.state = MissionState.NAVIGATING if self.state == MissionState.INIT else self.state

        # ---------------------------------------------------------
        # 3. MESAFE VE HEDEF KONTROLÜ
        # ---------------------------------------------------------
        if self._set_position_to_gps_target(
                target_lat,
                target_lon,
                f"WP{self.current_target_index}",
                self.waypoint_tolerance
        ):
            self._begin_waypoint_hold(f"WP{self.current_target_index}")
            self.current_target_index += 1
            return


# ============================================================
# ROS 2 NODE (GÖREV YÖNETİCİSİ)
# ============================================================
class Task1Node(Node):
    # ROS node'unu, servisleri, topicleri ve periyodik kontrol timer'ini kurar.
    def __init__(self):
        super().__init__('task1_mission_node')
        self.get_logger().info("Task 1 (Maneuvering) Node Starting...")

        # 1. Servis İstemcilerini (Clients) Oluştur ve Bekle
        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)

        # 2. Topic Aboneliklerini (Subscribers/Publishers) Oluştur
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback
        )

        self.latest_detections = []
        self.last_detection_time = None
        self.vision_sub = self.create_subscription(
            String,
            '/vision/detections',
            self.vision_callback,
            10
        )
        self.active_task_pub = self.create_publisher(
            String,
            '/mission/active_task',
            10
        )
        self.decision_pub = self.create_publisher(String, DECISION_TOPIC, 10)

        # 3. Görev Sınıfını Başlat
        self.task = Task1Maneuvering(self, self.mission_topics, self.mission_clients)

        # Anlık Yönelim Değişkeni (GPS Callback'e aktarmak için)
        self.current_heading = None
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self._last_logged_bridge_state = None
        self.mission_active = False
        self.valid_gps_received = False
        self.valid_heading_received = False

        # 4. Ana Kontrol Döngüsünü Başlat (Saniyede 10 kez çalışır: 0.1 sn)
        self.control_timer = self.create_timer(0.1, self.timer_callback)
        self.active_task_timer = self.create_timer(1.0, self.publish_active_task)
        self.decision_timer = self.create_timer(0.5, self.publish_decision)
        self.publish_active_task()

    # Vision node'a aktif gorevin task1 oldugunu bildirir.
    def publish_active_task(self):
        msg = String()
        msg.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(msg)

    def publish_decision(self):
        action = None
        reason = None
        if self.task.state == MissionState.AVOIDING:
            side = self.task.active_pass_side or "selected side"
            action = f"Dynamic camera pass on {side}"
            obstacle = self.task.avoiding_class or "course marker"
            reason = (
                f"{obstacle} detected; "
                f"linear={self.task.last_avoidance_linear_x:.2f}, "
                f"angular={self.task.last_avoidance_angular_z:.2f}"
            )
        msg = String()
        msg.data = mission_decision_json(
            1,
            self.task.state,
            current_target=self.task.current_target_index,
            target_count=len(self.task.waypoints),
            action=action,
            reason=reason,
        )
        self.decision_pub.publish(msg)

    # Vision detection JSON mesajlarini saklar.
    def vision_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f"Gecersiz vision JSON yok sayildi: {exc}",
                throttle_duration_sec=2.0
            )
            return

        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            self.get_logger().warn(
                "Vision detections alani liste degil, mesaj yok sayildi.",
                throttle_duration_sec=2.0
            )
            return

        self.latest_detections = detections
        self.last_detection_time = time.monotonic()

    # Eski vision mesajlarini kullanmamak icin guncel detection listesini dondurur.
    def _current_detections(self):
        if self.last_detection_time is None:
            return []

        if (time.monotonic() - self.last_detection_time) > VISION_DETECTION_TIMEOUT_SEC:
            return []

        return self.latest_detections

    # GPS mesajlarini dogrular ve gorev mantigina aktarir.
    def gps_callback(self, msg):
        """Araçtan gelen NavSatFix verisini dinler."""
        if abs(msg.latitude) < MIN_VALID_ABS_COORD and abs(msg.longitude) < MIN_VALID_ABS_COORD:
            self.get_logger().warn(
                "Gecersiz GPS (0,0) yok sayiliyor.",
                throttle_duration_sec=2.0
            )
            return

        self.valid_gps_received = True
        self.task.update_gps(msg.latitude, msg.longitude, self.current_heading)

    # Heading mesajini saklar ve watchdog zamanini tazeler.
    def heading_callback(self, msg):
        """Araçtan gelen Float32 yön verisini dinler."""
        try:
            heading = float(msg.data)
        except (TypeError, ValueError):
            heading = float("nan")

        if not math.isfinite(heading):
            self.get_logger().warn(
                "Gecersiz heading verisi yok sayiliyor.",
                throttle_duration_sec=2.0,
            )
            return

        self.current_heading = heading % 360.0
        self.valid_heading_received = True
        self.task.current_heading = self.current_heading
        self.task.last_heading_time = time.monotonic()

    # Bridge durumundan MAVLink baglantisinin hazir olup olmadigini izler.
    def state_callback(self, msg):
        """Bridge durumunu ayrıştırır, değişiklikleri loglar ve göreve aktarır."""
        state = parse_bridge_state(msg.data)
        required_keys = {"connected", "armed", "mode"}
        if not required_keys.issubset(state):
            self.get_logger().warn(
                f"Incomplete /cube/state ignored: {msg.data}",
                throttle_duration_sec=2.0,
            )
            return

        self.bridge_connected = state["connected"] is True
        self.bridge_armed = state["armed"] is True
        self.bridge_mode = str(state["mode"] or "UNKNOWN").strip().upper()

        current_state = (
            self.bridge_connected,
            self.bridge_armed,
            self.bridge_mode,
        )
        if current_state != self._last_logged_bridge_state:
            self.get_logger().info(
                "Task1 bridge state: "
                f"connected={self.bridge_connected}, "
                f"armed={self.bridge_armed}, mode={self.bridge_mode}"
            )
            self._last_logged_bridge_state = current_state

        self.task.update_bridge_state(
            self.bridge_connected,
            self.bridge_armed,
            self.bridge_mode,
        )

    # Mission baslamadan once bridge heartbeat bilgisini bekler.
    def wait_for_bridge_connection(self, timeout_sec=30.0):
        """Bridge servisleri hazir olsa bile MAVLink heartbeat gelene kadar bekler."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            state_fresh = (
                self.task.last_bridge_state_time is not None
                and now - self.task.last_bridge_state_time
                <= BRIDGE_STATE_TIMEOUT_SEC
            )
            if self.bridge_connected and state_fresh:
                return True

            self.get_logger().info(
                "Bridge MAVLink baglantisi bekleniyor...",
                throttle_duration_sec=2.0
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    # ARM oncesi sifir olmayan gecerli GPS konumu bekler.
    def wait_for_valid_navigation_data(self, timeout_sec=30.0):
        """Mission ARM olmadan once gercek GPS ve heading verisini bekler."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            gps_fresh = (
                self.task.last_gps_time is not None
                and now - self.task.last_gps_time <= GPS_TIMEOUT_SEC
            )
            heading_fresh = (
                self.task.last_heading_time is not None
                and now - self.task.last_heading_time <= HEADING_TIMEOUT_SEC
            )
            if (
                    self.valid_gps_received
                    and self.valid_heading_received
                    and gps_fresh
                    and heading_fresh
            ):
                return True

            self.get_logger().info(
                "Gecerli GPS ve heading verisi bekleniyor...",
                throttle_duration_sec=2.0
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    def wait_for_vehicle_state(
            self,
            expected_mode=None,
            expected_armed=None,
            timeout_sec=6.0,
    ):
        """Beklenen mode/armed değerlerini taze /cube/state üzerinden doğrular."""
        expected_mode = (
            None
            if expected_mode is None
            else str(expected_mode).strip().upper()
        )
        deadline = time.monotonic() + float(timeout_sec)
        expected_parts = ["connected=True"]
        if expected_mode is not None:
            expected_parts.append(f"mode={expected_mode}")
        if expected_armed is not None:
            expected_parts.append(f"armed={bool(expected_armed)}")
        expected_text = ", ".join(expected_parts)

        self.get_logger().info(
            f"Task1 waiting for confirmed vehicle state: {expected_text}"
        )

        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            state_fresh = (
                self.task.last_bridge_state_time is not None
                and now - self.task.last_bridge_state_time
                <= BRIDGE_STATE_TIMEOUT_SEC
            )
            mode_ok = expected_mode is None or self.bridge_mode == expected_mode
            armed_ok = (
                expected_armed is None
                or self.bridge_armed == bool(expected_armed)
            )
            if self.bridge_connected and state_fresh and mode_ok and armed_ok:
                self.get_logger().info(
                    f"Task1 vehicle state confirmed: {expected_text}"
                )
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(
            "Task1 vehicle-state confirmation timeout: "
            f"expected=({expected_text}), actual=(connected={self.bridge_connected}, "
            f"armed={self.bridge_armed}, mode={self.bridge_mode})"
        )
        return False

    def wait_for_operational_readiness(self, timeout_sec=3.0):
        """ARM sonrasında tüm görev girdilerinin hâlâ taze olduğunu doğrular."""
        deadline = time.monotonic() + float(timeout_sec)
        gps_fresh = False
        heading_fresh = False
        state_fresh = False
        vision_fresh = False
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            gps_fresh = (
                self.task.last_gps_time is not None
                and now - self.task.last_gps_time <= GPS_TIMEOUT_SEC
            )
            heading_fresh = (
                self.task.last_heading_time is not None
                and now - self.task.last_heading_time <= HEADING_TIMEOUT_SEC
            )
            state_fresh = (
                self.task.last_bridge_state_time is not None
                and now - self.task.last_bridge_state_time
                <= BRIDGE_STATE_TIMEOUT_SEC
            )
            vision_fresh = (
                self.last_detection_time is not None
                and now - self.last_detection_time
                <= VISION_DETECTION_TIMEOUT_SEC
            )
            if (
                    self.bridge_connected
                    and self.bridge_armed
                    and self.bridge_mode == "GUIDED"
                    and gps_fresh
                    and heading_fresh
                    and state_fresh
                    and vision_fresh
            ):
                self.get_logger().info(
                    "Task1 operational readiness confirmed: "
                    "GPS/heading/vision/bridge fresh, armed=True, mode=GUIDED"
                )
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(
            "Task1 operational-readiness timeout: "
            f"connected={self.bridge_connected}, armed={self.bridge_armed}, "
            f"mode={self.bridge_mode}, gps_fresh={gps_fresh}, "
            f"heading_fresh={heading_fresh}, state_fresh={state_fresh}, "
            f"vision_fresh={vision_fresh}"
        )
        return False

    def wait_for_vision(self, timeout_sec=30.0):
        """ARM oncesi vision node'dan en az bir guncel frame mesaji bekler."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if (
                    self.last_detection_time is not None
                    and time.monotonic() - self.last_detection_time
                    <= VISION_DETECTION_TIMEOUT_SEC
            ):
                return True

            self.get_logger().info(
                "Vision heartbeat bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    # Timer tick'lerinde aktif gorevi calistirir ve hatada araci durdurur.
    def timer_callback(self):
        """Görev mantığını sürekli tetikler.

        KRİTİK: Bu fonksiyon içinde beklenmeyen bir hata (örn. bozuk detection
        formatı) fırlarsa, düzeltilmezse araç son verilen cmd_vel komutuyla
        donmuş halde sürüklenmeye devam eder. Bu yüzden her tick try/except
        ile korunuyor ve hata durumunda araç durduruluyor.
        """
        # Vision cache guncel degilse bos liste doner; eski detection ile manevra yapilmaz.
        if not self.mission_active:
            return

        vision_age = (
            None
            if self.last_detection_time is None
            else time.monotonic() - self.last_detection_time
        )
        if vision_age is None or vision_age > VISION_DETECTION_TIMEOUT_SEC:
            stop_vehicle(self.mission_topics.cmd_vel_pub)
            age_text = "hic gelmedi" if vision_age is None else f"{vision_age:.2f}s eski"
            self.task._enter_failsafe(
                f"VISION HEARTBEAT LOST ({age_text})! FAILSAFE + HOLD.",
                request_hold=True,
            )
            return

        current_detections = self._current_detections()

        try:
            self.task.update(detections=current_detections)
        except Exception as exc:  # noqa: BLE001 - kasıtlı geniş yakalama, failsafe için
            self.get_logger().error(f"Unexpected error in timer_callback: {exc}")
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub)
            except Exception as stop_exc:  # noqa: BLE001
                self.get_logger().error(f"Failed to stop vehicle: {stop_exc}")
            self.task.state = MissionState.FAILSAFE


# ============================================================
# ANA ÇALIŞTIRMA BLOĞU
# ============================================================
# ROS 2 node yasam dongusunu baslatir, araci hazirlar ve spin'e girer.
# noinspection D
def main(args=None):
    rclpy.init(args=args)

    node = Task1Node()

    try:
        if not node.wait_for_bridge_connection(timeout_sec=30.0):
            node.get_logger().error("Bridge MAVLink baglantisi hazir degil! Mission not starting.")
            return

        if not node.wait_for_valid_navigation_data(timeout_sec=30.0):
            node.get_logger().error("Gecerli GPS/heading verisi yok! Mission not starting.")
            return

        if not node.wait_for_vision(timeout_sec=30.0):
            node.get_logger().error("Vision heartbeat yok! Mission not starting.")
            return

        node.get_logger().info("Setting vehicle to GUIDED mode...")
        # ------------------------------------------------------------
        mode_ok = call_set_mode(node, node.mission_clients.set_mode_client, "GUIDED")
        if mode_ok is False:
            node.get_logger().error("Failed to switch to GUIDED mode! Mission not starting.")
            return
        if not node.wait_for_vehicle_state(
                expected_mode="GUIDED",
                timeout_sec=6.0,
        ):
            node.get_logger().error(
                "GUIDED was not confirmed on /cube/state; mission not starting."
            )
            return

        node.get_logger().info("Force arming vehicle...")
        arm_ok = call_trigger_service(
            node,
            node.mission_clients.force_arm_client,
            "FORCE ARM"
        )

        if arm_ok is False:
            node.get_logger().error("FORCE ARM failed! Mission not starting.")
            return

        if not node.wait_for_vehicle_state(
                expected_mode="GUIDED",
                expected_armed=True,
                timeout_sec=6.0,
        ):
            node.get_logger().error(
                "armed=True and mode=GUIDED were not confirmed; mission not starting."
            )
            return

        if not node.wait_for_operational_readiness(timeout_sec=3.0):
            node.get_logger().error(
                "Fresh GPS/heading/vision/bridge data was not restored after arming; "
                "mission not starting."
            )
            return

        node.mission_active = True
        node.publish_active_task()
        node.get_logger().info(
            "Task 1 mission loop started with confirmed vehicle state: "
            f"connected={node.bridge_connected}, armed={node.bridge_armed}, "
            f"mode={node.bridge_mode}"
        )

        while rclpy.ok() and not node.task.finished and node.task.state != MissionState.FAILSAFE:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.mission_active = False
        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error(
                "Mission terminated due to FAILSAFE. Vehicle will stay in HOLD if mode change succeeds.")
            stop_vehicle(node.mission_topics.cmd_vel_pub)

            if node.task.hold_mode_future is not None:
                rclpy.spin_until_future_complete(
                    node,
                    node.task.hold_mode_future,
                    timeout_sec=2.0
                )
                if not node.task.hold_mode_future.done():
                    node.get_logger().error("HOLD mode request did not complete before shutdown.")
            else:
                call_set_mode(
                    node,
                    node.mission_clients.set_mode_client,
                    HOLD_MODE_NAME,
                    timeout_sec=2.0
                )
            return

        node.get_logger().info("Mission finished. Stopping vehicle.")
        stop_vehicle(node.mission_topics.cmd_vel_pub)

        node.get_logger().info("Disarming vehicle...")
        call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")

    except KeyboardInterrupt:
        node.get_logger().info("Mission terminated manually.")
        node.mission_active = False
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        try:
            call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
        except Exception:  # noqa: BLE001
            pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
