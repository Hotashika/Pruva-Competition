#!/usr/bin/env python3
"""
TEKNOFEST Görev 2
-----------------
Waypoint takibi + sarı dubalardan açı/derinlik tabanlı dinamik kaçınma.

Beklenen kamera topic'i:
    /vision/detections   (std_msgs/String, JSON)

Desteklenen detection örnekleri:
    {
        "detections": [
            {
                "class": "yellow_buoy",
                "confidence": 0.91,
                "distance": 2.7,
                "bbox": [x1, y1, x2, y2],
                "Buoy angle: ": -8.4,
                "Buoy side: ": "left"
            }
        ]
    }

Ayrıca class_name / angle_from_center / side alanlarını kullanan eski format da
kabul edilir.
"""

import json
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_TEXT = str(REPO_ROOT)
while REPO_ROOT_TEXT in sys.path:
    sys.path.remove(REPO_ROOT_TEXT)
sys.path.insert(0, REPO_ROOT_TEXT)

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from teknofest.config.mission_config import WAYPOINT_DIRECTORY
from teknofest.missions.utils.mission_data_recorder import MissionDataRecorder
from utils.mavlink_utilities import (
    align_heading_to_gps_target,
    calculate_gps_distance,
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    create_mission_topics,
    publish_cmd_vel,
    publish_set_position,
    parse_bridge_state,
    stop_vehicle,
    wait_for_mission_services,
)
from utils.read_waypoints import parse_qgc_waypoints
from teknofest.missions.utils.yellow_buoy_course_keeper import (
    YellowBuoyCourseConfig,
    YellowBuoyCourseKeeper,
)

WAYPOINT_PATH = WAYPOINT_DIRECTORY / "teknofest_task2.waypoints"

# ============================================================
# ROS / VISION PARAMETRELERİ
# ============================================================
DETECTION_TOPIC = "/vision/detections"
VISION_DETECTION_TIMEOUT_SEC = 3.00

# Görev 2 parkurundaki bütün engeller sarı dubadır. Mevcut buoy.pt modelinin
# class adı dışında hiçbir tespit engel kaçınmasını tetiklemez.
OBSTACLE_CLASS_NAMES = ("yellow_buoy",)
MIN_OBSTACLE_CONFIDENCE = 0.45

# ============================================================
# GÜVENLİK PARAMETRELERİ
# ============================================================
GPS_TIMEOUT_SEC = 2.0
HEADING_TIMEOUT_SEC = 2.0
BRIDGE_STATE_TIMEOUT_SEC = 10.0
HOLD_MODE_NAME = "HOLD"
GEOFENCE_RADIUS_M = 150.0
MIN_VALID_ABS_COORD = 1e-6

# ============================================================
# NAVİGASYON PARAMETRELERİ
# ============================================================
WAYPOINT_TOLERANCE_M = 1.0
WAYPOINT_SETTLE_SEC = 0.75
WAYPOINT_HEADING_TOLERANCE_DEG = 15.0

# ============================================================
# SARI DUBA PARKUR PARAMETRELERİ
# ============================================================
# Gorev basinda iki sari duba bulunamazsa bu sure boyunca ana GPS hedefine git.
INITIAL_YELLOW_SEARCH_GRACE_SEC = 3.0

# Ikinci en yakin sari dubaya yonelirken uretilecek kisa GPS hedefinin ust mesafesi.
YELLOW_COURSE_LOOKAHEAD_M = 5.0

# Sari parkur bir kez bulunduktan sonra tespit kaybinda son yonu koruma suresi.
YELLOW_TARGET_MEMORY_SEC = 1.0

# ============================================================
# KAÇINMA PARAMETRELERİ
# ============================================================
# Sarı duba bu mesafeye veya daha yakına geldiğinde kaçınma başlatılır.
AVOIDANCE_START_DISTANCE_M = 2.5
# Aktif engel bu mesafeye çıktığında temiz görüş süresi başlar.
AVOIDANCE_EXIT_DISTANCE_M = 5.0
AVOIDANCE_PASS_CLEARANCE_M = 2.5
AVOIDANCE_EMERGENCY_DISTANCE_M = 1.5
AVOIDANCE_MIN_LINEAR_SPEED = 0.2
AVOIDANCE_MAX_LINEAR_SPEED = 0.6
AVOIDANCE_MAX_ANGULAR_Z = 0.7
AVOIDANCE_TURN_SPEED_REDUCTION = 0.5
AVOIDANCE_CLEAR_DURATION_SEC = 0.2
AVOIDANCE_TIMEOUT_SEC = 8.0
SIDE_FALLBACK_ANGLE_DEG = 15.0

# ============================================================
# ENGEL EŞLEŞTİRME / FİLTRE PARAMETRELERİ
# ============================================================
OBSTACLE_CONFIRMATION_MAX_GAP_SEC = 0.75
OBSTACLE_FILTER_ALPHA = 0.40
OBSTACLE_MATCH_MAX_ANGLE_DELTA_DEG = 25.0
OBSTACLE_MATCH_MAX_DISTANCE_DELTA_M = 2.0
OBSTACLE_BBOX_MIN_IOU = 0.05

# Duba kameranın tam ortasındaysa seçilecek kaçış yönü.
DEFAULT_CENTER_AVOIDANCE_SIDE = "right"



class MissionState(Enum):
    INIT = auto()
    NAVIGATING = auto()
    AVOIDING = auto()
    FINISHED = auto()
    FAILSAFE = auto()


class Task2PointTrackingWithObstacleAvoidance:
    def __init__(self, node, mission_topics, mission_clients):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.clients = mission_clients

        self.logger.info(f"[INIT] Waypoint dosyası: {WAYPOINT_PATH.resolve()}")
        self.waypoints = parse_qgc_waypoints(WAYPOINT_PATH)
        self.logger.info(f"[INIT] Waypoint sayısı: {len(self.waypoints)}")

        self.current_target_index = 0
        self.waypoint_tolerance = WAYPOINT_TOLERANCE_M

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None

        self.last_angular_z = 0.0
        self.finished = False
        self.state = MissionState.INIT
        self.course_keeper = YellowBuoyCourseKeeper(YellowBuoyCourseConfig(
            min_confidence=MIN_OBSTACLE_CONFIDENCE,
            lookahead_m=YELLOW_COURSE_LOOKAHEAD_M,
            target_memory_sec=YELLOW_TARGET_MEMORY_SEC,
        ))
        self.yellow_course_acquired = False
        self.yellow_initial_search_started_time = None
        self.aligned_target_key = None
        self.resume_navigation_without_alignment = False
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None

        # Kaçınma durumu
        self.avoidance_side = None
        self.avoided_obstacle_side = None
        self.avoidance_started_time = None
        self.avoidance_clear_started_time = None
        self.last_avoidance_linear_x = 0.0
        self.last_avoidance_angular_z = 0.0
        self.avoiding_track_id = None
        self.active_obstacle_reference = None
        self.pending_obstacle = None
        self.pending_obstacle_time = None
        self.pending_obstacle_count = 0
        self.obstacle_data_uncertain = False
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self.last_bridge_state_time = None
        self.hold_mode_requested = False
        self.hold_mode_future = None

    def update_bridge_state(self, connected, armed, mode, now=None):
        self.bridge_connected = bool(connected)
        self.bridge_armed = bool(armed)
        self.bridge_mode = str(mode or "UNKNOWN").strip().upper()
        self.last_bridge_state_time = time.monotonic() if now is None else float(now)

    def _request_hold_mode(self):
        if self.hold_mode_requested:
            return
        self.hold_mode_requested = True
        request = SetMode.Request()
        request.base_mode = 0
        request.custom_mode = HOLD_MODE_NAME
        try:
            self.hold_mode_future = self.clients.set_mode_client.call_async(request)
            self.logger.warn("FAILSAFE nedeniyle HOLD modu isteniyor.")
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"HOLD modu istenemedi: {exc}")

    def _enter_failsafe(self, reason):
        if self.state != MissionState.FAILSAFE:
            self.logger.error(reason)
        self.state = MissionState.FAILSAFE
        stop_vehicle(self.topics.cmd_vel_pub)
        self._request_hold_mode()

    def reset_geofence_origin(self, lat, lon):
        self.home_lat = float(lat)
        self.home_lon = float(lon)
        self.logger.info(
            f"Task 2 geofence merkezi yenilendi: "
            f"{self.home_lat:.7f}, {self.home_lon:.7f}"
        )

    # ========================================================
    # VERİ GÜNCELLEME
    # ========================================================
    def update_gps(self, lat, lon):
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.last_gps_time = time.monotonic()

        if self.home_lat is None:
            self.home_lat = self.current_lat
            self.home_lon = self.current_lon
            self.logger.info(
                f"Home konumu ayarlandı: {self.home_lat:.7f}, {self.home_lon:.7f}"
            )

    def update_heading(self, heading):
        self.current_heading = float(heading) % 360.0
        self.last_heading_time = time.monotonic()

    # ========================================================
    # GÜVENLİK
    # ========================================================
    def _check_watchdog(self):
        now = time.monotonic()

        if self.last_gps_time is None or self.last_heading_time is None:
            return False

        if now - self.last_gps_time > GPS_TIMEOUT_SEC:
            self.logger.error(
                f"GPS verisi {GPS_TIMEOUT_SEC:.1f} saniyeden uzun süredir gelmiyor. FAILSAFE."
            )
            self.state = MissionState.FAILSAFE
            return False

        if now - self.last_heading_time > HEADING_TIMEOUT_SEC:
            self.logger.error(
                f"Heading verisi {HEADING_TIMEOUT_SEC:.1f} saniyeden uzun süredir gelmiyor. FAILSAFE."
            )
            self.state = MissionState.FAILSAFE
            return False

        if self.last_bridge_state_time is None:
            self._enter_failsafe("BRIDGE STATE alınamadı. FAILSAFE + HOLD.")
            return False
        if now - self.last_bridge_state_time > BRIDGE_STATE_TIMEOUT_SEC:
            self._enter_failsafe("BRIDGE STATE zaman aşımı. FAILSAFE + HOLD.")
            return False
        if not self.bridge_connected:
            self._enter_failsafe("MAVLink bridge bağlantısı kesildi. FAILSAFE + HOLD.")
            return False
        if not self.bridge_armed:
            self._enter_failsafe("Araç ARM durumundan çıktı. FAILSAFE + HOLD.")
            return False
        if self.bridge_mode != "GUIDED":
            self._enter_failsafe(
                f"Araç GUIDED modundan çıktı (mode={self.bridge_mode}). FAILSAFE + HOLD."
            )
            return False

        return True

    def _check_geofence(self):
        if self.home_lat is None or self.current_lat is None:
            return True

        distance_from_home = calculate_gps_distance(
            self.home_lat,
            self.home_lon,
            self.current_lat,
            self.current_lon,
        )

        if distance_from_home > GEOFENCE_RADIUS_M:
            self.logger.error(
                f"Geofence ihlali: home noktasından {distance_from_home:.1f} m uzaklıkta "
                f"(limit={GEOFENCE_RADIUS_M:.1f} m). FAILSAFE."
            )
            self.state = MissionState.FAILSAFE
            return False

        return True

    def _begin_waypoint_hold(self, waypoint_name):
        """Ana waypoint'te aracı durdurup heading için sabitler."""
        stop_vehicle(self.topics.cmd_vel_pub)
        self.waypoint_hold_until = time.monotonic() + WAYPOINT_SETTLE_SEC
        self.waypoint_hold_name = waypoint_name
        self.aligned_target_key = None
        self.resume_navigation_without_alignment = False
        self.logger.info(
            f"{waypoint_name} ulaşıldı; araç {WAYPOINT_SETTLE_SEC:.2f}s durduruldu."
        )

    def _waypoint_hold_active(self):
        if self.waypoint_hold_until is None:
            return False

        remaining = self.waypoint_hold_until - time.monotonic()
        if remaining > 0.0:
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=0.0)
            self.logger.info(
                f"{self.waypoint_hold_name} noktasında bekleniyor: {remaining:.2f}s.",
                throttle_duration_sec=0.5,
            )
            return True

        completed_name = self.waypoint_hold_name
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None
        self.logger.info(
            f"{completed_name} duruşu sabitlendi; sonraki görev adımına geçiliyor."
        )
        return False

    # ========================================================
    # DETECTION NORMALİZASYONU
    # ========================================================
    @staticmethod
    def _safe_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None
        return number

    @staticmethod
    def _normalize_side_text(value):
        if value is None:
            return None

        text = str(value).strip().lower()

        if text in ("left", "sol", "port"):
            return "left"
        if text in ("right", "sag", "sağ", "starboard"):
            return "right"
        if text in ("across", "center", "centre", "middle", "orta", "front"):
            return "center"

        return None

    # noinspection D
    def _normalize_detection(self, obj):
        if not isinstance(obj, dict):
            return None

        class_name = obj.get("class")
        if class_name is None:
            class_name = obj.get("class_name")
        if class_name is None:
            class_name = obj.get("label", "obstacle")
        class_name = str(class_name).strip().lower()

        confidence = self._safe_float(obj.get("confidence"))
        if confidence is None:
            confidence = self._safe_float(obj.get("conf"))
        if confidence is None:
            confidence = 1.0

        distance = self._safe_float(obj.get("distance"))
        if distance is None:
            distance = self._safe_float(obj.get("distance_m"))
        if distance is None:
            distance = self._safe_float(obj.get("depth"))

        side = self._normalize_side_text(obj.get("Buoy side: "))
        if side is None:
            side = self._normalize_side_text(obj.get("side"))
        if side is None:
            side = self._normalize_side_text(obj.get("buoy_side"))

        angle = self._safe_float(obj.get("Buoy angle: "))
        if angle is None:
            angle = self._safe_float(obj.get("angle_from_center"))
        if angle is None:
            angle = self._safe_float(obj.get("angle"))

        # Side alanı gelmediyse açıdan üret.
        if side is None and angle is not None:
            if angle < -2.0:
                side = "left"
            elif angle > 2.0:
                side = "right"
            else:
                side = "center"

        # Son yedek: bbox merkezi ve image_width bilgisi.
        bbox = obj.get("bbox")
        image_width = self._safe_float(obj.get("image_width"))
        if (
                side is None
                and isinstance(bbox, (list, tuple))
                and len(bbox) >= 4
                and image_width is not None
                and image_width > 0
        ):
            x1 = self._safe_float(bbox[0])
            x2 = self._safe_float(bbox[2])
            if x1 is not None and x2 is not None:
                center_x = (x1 + x2) / 2.0
                diff = center_x - image_width / 2.0
                tolerance_px = image_width * 0.05
                if abs(diff) <= tolerance_px:
                    side = "center"
                elif diff < 0:
                    side = "left"
                else:
                    side = "right"

        # Eski detector formatında yalnız yön varsa yaklaşık kamera açısı kullan.
        if angle is None and side is not None:
            angle = {
                "left": -SIDE_FALLBACK_ANGLE_DEG,
                "right": SIDE_FALLBACK_ANGLE_DEG,
                "center": 0.0,
            }[side]

        return {
            "class": class_name,
            "confidence": confidence,
            "distance": distance,
            "side": side,
            "angle": angle,
            "bbox": bbox,
            "track_id": obj.get("track_id"),
            "raw": obj,
        }

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
            return candidate_track == reference_track

        bbox_iou = self._bbox_iou(
            candidate.get("bbox"),
            reference.get("bbox"),
        )
        bbox_match = bbox_iou is not None and bbox_iou >= OBSTACLE_BBOX_MIN_IOU
        angle = self._safe_float(candidate.get("angle"))
        reference_angle = self._safe_float(reference.get("angle"))
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
            return 0.0 if candidate_track == reference_track else math.inf

        bbox_iou = self._bbox_iou(
            candidate.get("bbox"),
            reference.get("bbox"),
        )
        bbox_penalty = 1.0 if bbox_iou is None else 1.0 - bbox_iou
        angle = self._safe_float(candidate.get("angle"))
        reference_angle = self._safe_float(reference.get("angle"))
        angle_penalty = (
            1.0
            if angle is None or reference_angle is None
            else abs(angle - reference_angle) / OBSTACLE_MATCH_MAX_ANGLE_DELTA_DEG
        )
        distance_penalty = (
            abs(float(candidate["distance"]) - float(reference["distance"]))
            / OBSTACLE_MATCH_MAX_DISTANCE_DELTA_M
        )
        return bbox_penalty + angle_penalty + distance_penalty

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
        angle = self._safe_float(obstacle.get("angle"))
        reference_angle = self._safe_float(reference.get("angle"))
        if angle is not None and reference_angle is not None:
            filtered["angle"] = (
                OBSTACLE_FILTER_ALPHA * angle
                + (1.0 - OBSTACLE_FILTER_ALPHA) * reference_angle
            )
        return filtered

    def _nearest_relevant_obstacle(self, detections, now=None):
        candidates = []
        self.obstacle_data_uncertain = False

        for raw_detection in detections or []:
            obstacle = self._normalize_detection(raw_detection)
            if obstacle is None:
                continue

            if obstacle["confidence"] < MIN_OBSTACLE_CONFIDENCE:
                continue

            if OBSTACLE_CLASS_NAMES and obstacle["class"] not in OBSTACLE_CLASS_NAMES:
                continue

            distance = obstacle["distance"]
            if distance is None or distance <= 0.0:
                self.obstacle_data_uncertain = True
                continue
            if distance >= AVOIDANCE_EXIT_DISTANCE_M:
                continue

            # Yakın sarı dubanın açısı/yönü yoksa ilerlemek güvenli değildir.
            if obstacle["side"] is None or obstacle["angle"] is None:
                self.obstacle_data_uncertain = True
                continue

            candidates.append(obstacle)

        if not candidates:
            return None

        return min(candidates, key=lambda item: item["distance"])

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

    def _matching_avoidance_obstacle(self, detections):
        reference = getattr(self, "active_obstacle_reference", None)
        candidates = []
        for raw in detections or []:
            obstacle = self._normalize_detection(raw)
            if (
                    obstacle is None
                    or obstacle["class"] not in OBSTACLE_CLASS_NAMES
                    or obstacle["confidence"] < MIN_OBSTACLE_CONFIDENCE
            ):
                continue
            if (
                    obstacle["distance"] is None
                    or obstacle["angle"] is None
                    or not 0 < obstacle["distance"] < AVOIDANCE_EXIT_DISTANCE_M
            ):
                continue
            avoiding_track_id = getattr(self, "avoiding_track_id", None)
            if (avoiding_track_id is not None and
                    obstacle.get("track_id") != avoiding_track_id):
                continue
            if avoiding_track_id is None and not self._same_obstacle(
                    obstacle,
                    reference,
            ):
                continue
            score = self._obstacle_match_score(obstacle, reference)
            candidates.append((score, obstacle))
        if not candidates:
            return None
        obstacle = min(candidates, key=lambda item: item[0])[1]
        filtered = self._filter_obstacle(obstacle, reference)
        self.active_obstacle_reference = filtered
        if (
                getattr(self, "avoiding_track_id", None) is None
                and obstacle.get("track_id") is not None
        ):
            self.avoiding_track_id = obstacle["track_id"]
        return filtered

    # ========================================================
    # DİNAMİK KAÇINMA
    # ========================================================
    @staticmethod
    def _choose_avoidance_side(obstacle_side):
        """
        obstacle_side kamerada dubanın bulunduğu taraftır.

        Duba soldaysa araç dubanın sağındadır -> sağdan geçmeye devam et.
        Duba sağdaysa araç dubanın solundadır -> soldan geçmeye devam et.
        Duba ortadaysa sabit varsayılan yön kullan.
        """
        if obstacle_side == "left":
            return "right"
        if obstacle_side == "right":
            return "left"
        return DEFAULT_CENTER_AVOIDANCE_SIDE

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def _calculate_avoidance_command(self, obstacle):
        obstacle = self._normalize_detection(obstacle)
        if (
                obstacle is None
                or obstacle["distance"] is None
                or obstacle["distance"] <= 0.0
                or obstacle["side"] is None
                or obstacle["angle"] is None
        ):
            return None

        avoidance_side = self.avoidance_side
        if avoidance_side is None:
            avoidance_side = self._choose_avoidance_side(obstacle["side"])

        distance = float(obstacle["distance"])
        angle_deg = float(obstacle["angle"])
        angle_rad = math.radians(angle_deg)
        target_forward = distance * math.cos(angle_rad)
        target_starboard = distance * math.sin(angle_rad)
        target_starboard += (
            AVOIDANCE_PASS_CLEARANCE_M
            if avoidance_side == "right"
            else -AVOIDANCE_PASS_CLEARANCE_M
        )

        angular_z = self._clamp(
            math.atan2(target_starboard, target_forward),
            -AVOIDANCE_MAX_ANGULAR_Z,
            AVOIDANCE_MAX_ANGULAR_Z,
        )
        if distance <= AVOIDANCE_EMERGENCY_DISTANCE_M:
            linear_x = 0.0
        else:
            distance_ratio = self._clamp(
                (
                    distance - AVOIDANCE_EMERGENCY_DISTANCE_M
                ) / (
                    AVOIDANCE_START_DISTANCE_M
                    - AVOIDANCE_EMERGENCY_DISTANCE_M
                ),
                0.0,
                1.0,
            )
            base_speed = (
                AVOIDANCE_MIN_LINEAR_SPEED
                + distance_ratio
                * (AVOIDANCE_MAX_LINEAR_SPEED - AVOIDANCE_MIN_LINEAR_SPEED)
            )
            turn_ratio = min(
                1.0,
                abs(angular_z) / AVOIDANCE_MAX_ANGULAR_Z,
            )
            linear_x = max(
                AVOIDANCE_MIN_LINEAR_SPEED,
                base_speed
                * (1.0 - AVOIDANCE_TURN_SPEED_REDUCTION * turn_ratio),
            )

        return {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "distance": distance,
            "angle": angle_deg,
            "avoidance_side": avoidance_side,
        }

    def _publish_avoidance_command(self, command):
        self.avoidance_side = command["avoidance_side"]
        self.last_avoidance_linear_x = command["linear_x"]
        self.last_avoidance_angular_z = command["angular_z"]
        self.last_angular_z = command["angular_z"]
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
        self.last_angular_z = self.last_avoidance_angular_z

    def _start_avoidance(
            self,
            obstacle,
            now=None,
    ):
        obstacle = self._normalize_detection(obstacle)
        command = self._calculate_avoidance_command(obstacle)
        if command is None:
            publish_cmd_vel(
                self.topics.cmd_vel_pub,
                linear_x=0.0,
                angular_z=0.0,
            )
            return False

        self.avoided_obstacle_side = obstacle["side"]
        self.avoidance_side = command["avoidance_side"]
        self.avoidance_started_time = (
            time.monotonic() if now is None else float(now)
        )
        self.avoidance_clear_started_time = None
        self.avoiding_track_id = obstacle.get("track_id")
        self.active_obstacle_reference = obstacle
        self.state = MissionState.AVOIDING
        self.aligned_target_key = None
        self.resume_navigation_without_alignment = False
        self._publish_avoidance_command(command)

        self.logger.warn(
            f"ENGEL: class={obstacle['class']}, distance={obstacle['distance']:.2f} m, "
            f"camera_side={obstacle['side']}, angle={obstacle['angle']:.1f}°. "
            f"Geçiş={self.avoidance_side}; dinamik manevra="
            f"linear_x={command['linear_x']:.2f}, "
            f"angular_z={command['angular_z']:+.2f}"
        )
        return True

    def _reset_avoidance_state(self):
        self.avoidance_side = None
        self.avoided_obstacle_side = None
        self.avoidance_started_time = None
        self.avoidance_clear_started_time = None
        self.last_avoidance_linear_x = 0.0
        self.last_avoidance_angular_z = 0.0
        self.avoiding_track_id = None
        self.active_obstacle_reference = None
        self.pending_obstacle = None
        self.pending_obstacle_time = None
        self.pending_obstacle_count = 0
        self.last_angular_z = 0.0
        self.aligned_target_key = None
        self.state = MissionState.NAVIGATING

    def _finish_avoidance(self):
        completed_side = self.avoidance_side
        self._reset_avoidance_state()
        self.resume_navigation_without_alignment = True
        self.logger.info(
            f"Engel kesintisiz {AVOIDANCE_CLEAR_DURATION_SEC:.1f}s temiz. "
            f"{completed_side} taraftan dinamik geçiş tamamlandı; "
            "durmadan aynı ana waypoint ve sarı duba parkur takibine dönülüyor."
        )

    def _avoidance_timed_out(self, now=None):
        if self.avoidance_started_time is None:
            return False
        now = time.monotonic() if now is None else float(now)
        return now - self.avoidance_started_time >= AVOIDANCE_TIMEOUT_SEC

    def _update_active_avoidance(self, detections, now=None):
        """Kaçınma bu çevrimi tüketiyorsa True, tamamlandıysa False döndürür."""
        now = time.monotonic() if now is None else float(now)
        if self.avoidance_started_time is None:
            self._enter_failsafe("AVOIDING başlangıç zamanı yok; FAILSAFE + HOLD.")
            return True
        if self._avoidance_timed_out(now):
            self._enter_failsafe("Kaçınma zaman aşımı; FAILSAFE + HOLD.")
            return True

        # Eksik derinlik/yön temiz görüş değildir: araç durur, clear sayacı sıfırlanır.
        self._nearest_relevant_obstacle(detections, now=now)
        if self.obstacle_data_uncertain:
            self.avoidance_clear_started_time = None
            publish_cmd_vel(
                self.topics.cmd_vel_pub,
                linear_x=0.0,
                angular_z=0.0,
            )
            self.last_angular_z = 0.0
            self.logger.warn(
                "Aktif kaçınmada sarı duba verisi belirsiz; araç veri "
                "düzelene kadar durduruldu.",
                throttle_duration_sec=1.0,
            )
            return True

        obstacle = self._matching_avoidance_obstacle(detections)
        if obstacle is None:
            if self.avoidance_clear_started_time is None:
                self.avoidance_clear_started_time = now
            clear_elapsed = now - self.avoidance_clear_started_time
            if clear_elapsed >= AVOIDANCE_CLEAR_DURATION_SEC:
                self._finish_avoidance()
                return False

            # Kısa tespit kaybında son güvenli komutu clear süresince koru.
            self._republish_last_avoidance_command()
            return True

        self.avoidance_clear_started_time = None
        command = self._calculate_avoidance_command(obstacle)
        if command is None:
            publish_cmd_vel(
                self.topics.cmd_vel_pub,
                linear_x=0.0,
                angular_z=0.0,
            )
            self.last_angular_z = 0.0
            return True

        self._publish_avoidance_command(command)
        self.logger.info(
            f"Dinamik kaçınma: class={obstacle['class']}, "
            f"geçiş={command['avoidance_side']}, "
            f"distance={command['distance']:.2f} m, "
            f"angle={command['angle']:+.1f}°, "
            f"linear_x={command['linear_x']:.2f}, "
            f"angular_z={command['angular_z']:+.2f}",
            throttle_duration_sec=0.5,
        )
        return True

    # ========================================================
    # GPS HEDEF TAKİBİ
    # ========================================================
    def _navigate_to_gps_target(
            self,
            target_lat,
            target_lon,
            target_name,
            tolerance_m,
            detections,
            follow_yellow_course=True,
    ):
        distance = calculate_gps_distance(
            self.current_lat,
            self.current_lon,
            target_lat,
            target_lon,
        )

        if distance < tolerance_m:
            self.logger.info(f"{target_name} ulaşıldı. Kalan mesafe: {distance:.2f} m")
            return True

        navigation_lat = target_lat
        navigation_lon = target_lon
        navigation_status = "direct_main_waypoint"
        if follow_yellow_course:
            now = time.monotonic()
            course_decision = self.course_keeper.compute(
                detections=detections,
                current_lat=self.current_lat,
                current_lon=self.current_lon,
                current_heading=self.current_heading,
                now=now,
            )
            if course_decision.should_stop:
                initial_search_active = (
                    not self.yellow_course_acquired
                    and course_decision.reason == "fewer_than_two_yellow_buoys"
                )
                if initial_search_active:
                    if self.yellow_initial_search_started_time is None:
                        self.yellow_initial_search_started_time = now
                    search_elapsed = now - self.yellow_initial_search_started_time
                    if search_elapsed < INITIAL_YELLOW_SEARCH_GRACE_SEC:
                        navigation_status = (
                            f"initial_yellow_search/{search_elapsed:.1f}s/"
                            "direct_main_waypoint"
                        )
                    else:
                        initial_search_active = False

                if not initial_search_active:
                    if self.resume_navigation_without_alignment:
                        navigation_status = (
                            "post_avoidance_main_waypoint_fallback/"
                            f"{course_decision.reason}"
                        )
                    else:
                        publish_cmd_vel(
                            self.topics.cmd_vel_pub,
                            linear_x=0.0,
                            angular_z=0.0,
                        )
                        self.logger.warn(
                            f"Sarı duba parkur hedefi hesaplanamadı "
                            f"({course_decision.reason}); araç bekletiliyor.",
                            throttle_duration_sec=1.0,
                        )
                        return False
            if (
                    course_decision.target_lat is not None
                    and course_decision.target_lon is not None
            ):
                if course_decision.status == "live":
                    self.yellow_course_acquired = True
                navigation_lat = course_decision.target_lat
                navigation_lon = course_decision.target_lon
                navigation_status = (
                    f"yellow_course/{course_decision.status}/"
                    f"{course_decision.reason}"
                )

        target_key = (
            target_name,
            round(float(navigation_lat), 7),
            round(float(navigation_lon), 7),
        )
        if self.resume_navigation_without_alignment:
            self.aligned_target_key = target_key
        elif self.aligned_target_key != target_key:
            if not align_heading_to_gps_target(
                    self.topics.cmd_vel_pub,
                    self.current_lat,
                    self.current_lon,
                    self.current_heading,
                    navigation_lat,
                    navigation_lon,
                    logger=self.logger,
                    target_name=target_name,
                    tolerance_deg=WAYPOINT_HEADING_TOLERANCE_DEG,
            ):
                return False
            self.aligned_target_key = target_key

        publish_set_position(
            self.topics.position_target_pub,
            navigation_lat,
            navigation_lon,
        )
        self.last_angular_z = 0.0

        self.logger.info(
            f"Hedef={target_name} | mesafe={distance:.2f} m | "
            f"navigation={navigation_status}",
            throttle_duration_sec=1.0,
        )
        return False

    # ========================================================
    # ANA STATE MACHINE
    # ========================================================
    # noinspection D
    def update(self, detections):
        if self.state == MissionState.FAILSAFE:
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.warn(
                "FAILSAFE aktif; araç durduruldu.",
                throttle_duration_sec=2.0,
            )
            return

        if not self._check_watchdog():
            if self.state != MissionState.FAILSAFE:
                self.logger.info(
                    "Geçerli GPS ve heading verisi bekleniyor...",
                    throttle_duration_sec=2.0,
                )
                publish_cmd_vel(
                    self.topics.cmd_vel_pub,
                    linear_x=0.0,
                    angular_z=0.0,
                )
            return

        if not self._check_geofence():
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if not self.waypoints:
            self.logger.error("Waypoint listesi boş. Araç durduruldu.")
            self.state = MissionState.FAILSAFE
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if self._waypoint_hold_active():
            return

        if self.current_target_index >= len(self.waypoints):
            if not self.finished:
                self.logger.info("BÜTÜN WAYPOINT'LER TAMAMLANDI. GÖREV BİTTİ.")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.finished = True
                self.state = MissionState.FINISHED
            return

        now = time.monotonic()

        # Devam eden kaçınma yalnız /cube/cmd_vel kullanır. Temiz geçiş
        # doğrulanırsa aynı çevrimde course keeper ve GPS navigasyonu sürer.
        if self.state == MissionState.AVOIDING:
            if self._update_active_avoidance(detections, now=now):
                return

        target_gps = self.waypoints[self.current_target_index]
        target_lat = target_gps["lat"]
        target_lon = target_gps["lon"]
        nearest_obstacle = self._nearest_relevant_obstacle(detections, now=now)

        if self.obstacle_data_uncertain:
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.warn(
                "Yakın sarı duba görüldü fakat mesafe/yön güvenilir değil; "
                "araç veri düzelene kadar bekletiliyor.",
                throttle_duration_sec=1.0,
            )
            return

        # ----------------------------------------------------
        # 1. BAŞLANGIÇ WP0 KONTROLÜ
        # ----------------------------------------------------
        if self.state == MissionState.INIT:
            reached_wp0 = self._navigate_to_gps_target(
                target_lat,
                target_lon,
                "WP0 (başlangıç)",
                self.waypoint_tolerance + 2.0,
                detections,
                follow_yellow_course=False,
            )

            if reached_wp0:
                self.logger.info("WP0 doğrulandı; görev navigasyonu başlıyor.")
                self._begin_waypoint_hold("WP0 (başlangıç)")
                self.current_target_index += 1
                self.last_angular_z = 0.0
                self.state = MissionState.NAVIGATING
            return

        # ----------------------------------------------------
        # 2. YENİ ENGEL TETİKLEME
        # ----------------------------------------------------
        if (
                nearest_obstacle is not None
                and nearest_obstacle["distance"] <= AVOIDANCE_START_DISTANCE_M
        ):
            confirmed = self._confirmed_obstacle(nearest_obstacle, now)
            if confirmed is None:
                return
            self._start_avoidance(confirmed, now=now)
            return
        else:
            self._confirmed_obstacle(None, now)

        # ----------------------------------------------------
        # 3. NORMAL WAYPOINT TAKİBİ
        # ----------------------------------------------------
        reached_main_wp = self._navigate_to_gps_target(
            target_lat,
            target_lon,
            f"WP{self.current_target_index}",
            self.waypoint_tolerance,
            detections,
        )

        if reached_main_wp:
            self._begin_waypoint_hold(f"WP{self.current_target_index}")
            self.current_target_index += 1
            self.last_angular_z = 0.0


class Task2Node(Node):
    def __init__(self):
        super().__init__("task2_mission_node")
        self.get_logger().info(
            "Görev 2 node'u başlatılıyor: waypoint takibi + dinamik engelden kaçınma."
        )

        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)

        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback,
        )

        self.task = Task2PointTrackingWithObstacleAvoidance(
            self,
            self.mission_topics,
            self.mission_clients,
        )

        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self.mission_active = False
        self.valid_gps_received = False
        self.valid_heading_received = False

        self.latest_detections = []
        self.last_detection_message_time = None

        detection_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.detection_sub = self.create_subscription(
            String,
            DETECTION_TOPIC,
            self.detection_callback,
            detection_qos,
        )

        self.active_task_pub = self.create_publisher(
            String,
            "/mission/active_task",
            10,
        )
        self.active_task_timer = self.create_timer(1.0, self._publish_active_task)
        self._publish_active_task()

        self.data_recorder = MissionDataRecorder(self, "task2")
        self.control_timer = self.create_timer(0.1, self.timer_callback)

    # ========================================================
    # CALLBACKS
    # ========================================================
    def gps_callback(self, msg):
        if (
                abs(msg.latitude) < MIN_VALID_ABS_COORD
                and abs(msg.longitude) < MIN_VALID_ABS_COORD
        ):
            self.get_logger().warn(
                "Geçersiz GPS (0,0) yok sayıldı.",
                throttle_duration_sec=2.0,
            )
            return

        self.valid_gps_received = True
        self.task.update_gps(msg.latitude, msg.longitude)

    def heading_callback(self, msg):
        heading = Task2PointTrackingWithObstacleAvoidance._safe_float(msg.data)
        if heading is None:
            self.get_logger().warn(
                "Geçersiz heading verisi yok sayıldı.",
                throttle_duration_sec=2.0,
            )
            return

        self.valid_heading_received = True
        self.task.update_heading(heading)

    def state_callback(self, msg):
        state = parse_bridge_state(msg.data)
        if not {"connected", "armed", "mode"}.issubset(state):
            self.get_logger().warn("Eksik /cube/state mesajı yok sayıldı.", throttle_duration_sec=2.0)
            return
        self.bridge_connected = state["connected"] is True
        self.bridge_armed = state["armed"] is True
        self.bridge_mode = str(state["mode"] or "UNKNOWN").strip().upper()
        self.task.update_bridge_state(
            self.bridge_connected, self.bridge_armed, self.bridge_mode
        )

    def _publish_active_task(self):
        msg = String()
        msg.data = "task2"
        self.active_task_pub.publish(msg)

    def wait_for_complete_telemetry(self, timeout_sec=10.0):
        return self.data_recorder.wait_for_complete_telemetry(timeout_sec)

    def start_telemetry_recording(self):
        self.data_recorder.start()

    def stop_telemetry_recording(self):
        self.data_recorder.stop()

    @staticmethod
    def _parse_detections_payload(payload):
        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            detections = payload.get("detections")
            if isinstance(detections, list):
                return detections

            objects = payload.get("objects")
            if isinstance(objects, list):
                return objects

        return []

    def detection_callback(self, msg):
        try:
            parsed = json.loads(msg.data)
            detections = self._parse_detections_payload(parsed)
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().error(
                f"Detection JSON ayrıştırılamadı: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        self.latest_detections = detections
        self.last_detection_message_time = time.monotonic()

    def _get_fresh_detections(self):
        if self.last_detection_message_time is None:
            return []

        age = time.monotonic() - self.last_detection_message_time
        if age > VISION_DETECTION_TIMEOUT_SEC:
            return []

        return list(self.latest_detections)

    # ========================================================
    # BAŞLANGIÇ BEKLEMELERİ
    # ========================================================
    def wait_for_bridge_connection(self, timeout_sec=30.0):
        deadline = time.monotonic() + timeout_sec

        while rclpy.ok() and time.monotonic() < deadline:
            if self.bridge_connected:
                return True

            self.get_logger().info(
                "Bridge MAVLink bağlantısı bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    def wait_for_operational_vehicle_state(self, timeout_sec=6.0):
        """Servis cevabından sonra heartbeat'te GUIDED ve ARM durumunu doğrular."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            state_fresh = (
                self.task.last_bridge_state_time is not None
                and time.monotonic() - self.task.last_bridge_state_time
                <= BRIDGE_STATE_TIMEOUT_SEC
            )
            if (
                    self.bridge_connected
                    and self.bridge_armed
                    and self.bridge_mode == "GUIDED"
                    and state_fresh
            ):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(
            "Araç durumu doğrulanamadı: "
            f"connected={self.bridge_connected}, armed={self.bridge_armed}, "
            f"mode={self.bridge_mode}"
        )
        return False

    def wait_for_valid_navigation_data(self, timeout_sec=30.0):
        deadline = time.monotonic() + timeout_sec

        while rclpy.ok() and time.monotonic() < deadline:
            if self.valid_gps_received and self.valid_heading_received:
                return True

            self.get_logger().info(
                "Geçerli GPS ve heading verisi bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    def wait_for_vision(self, timeout_sec=30.0):
        """ARM öncesinde vision pipeline'dan güncel heartbeat bekler."""
        deadline = time.monotonic() + timeout_sec

        while rclpy.ok() and time.monotonic() < deadline:
            if (
                    self.last_detection_message_time is not None
                    and time.monotonic() - self.last_detection_message_time
                    <= VISION_DETECTION_TIMEOUT_SEC
            ):
                return True

            self.get_logger().info(
                "Vision heartbeat bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    # ========================================================
    # ANA TIMER
    # ========================================================
    def timer_callback(self):
        if not self.mission_active:
            return

        vision_age = (
            None
            if self.last_detection_message_time is None
            else time.monotonic() - self.last_detection_message_time
        )
        if vision_age is None or vision_age > VISION_DETECTION_TIMEOUT_SEC:
            stop_vehicle(self.mission_topics.cmd_vel_pub)
            self.task.state = MissionState.FAILSAFE
            age_text = "hiç gelmedi" if vision_age is None else f"{vision_age:.2f}s eski"
            self.get_logger().error(
                f"VISION HEARTBEAT KAYBI ({age_text}). Araç durduruldu; FAILSAFE."
            )
            return

        current_detections = self._get_fresh_detections()

        try:
            self.task.update(detections=current_detections)
        except Exception as exc:  # noqa: BLE001 - failsafe için geniş yakalama
            self.get_logger().error(f"Görev timer hatası: {exc}")
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub)
            except Exception as stop_exc:  # noqa: BLE001
                self.get_logger().error(f"Araç durdurulamadı: {stop_exc}")
            self.task.state = MissionState.FAILSAFE


def main(args=None):
    rclpy.init(args=args)
    node = Task2Node()

    try:
        if not node.wait_for_bridge_connection(timeout_sec=30.0):
            node.get_logger().error(
                "Bridge MAVLink bağlantısı hazır değil. Görev başlatılmadı."
            )
            return

        if not node.wait_for_valid_navigation_data(timeout_sec=30.0):
            node.get_logger().error(
                "Geçerli GPS/heading verisi yok. Görev başlatılmadı."
            )
            return

        if not node.wait_for_complete_telemetry(timeout_sec=10.0):
            node.get_logger().error(
                "Roll/pitch/yaw ve araç telemetrisi hazır değil. Görev başlatılmadı."
            )
            return

        if not node.wait_for_vision(timeout_sec=30.0):
            node.get_logger().error(
                "Vision heartbeat yok. Görev başlatılmadı."
            )
            return

        node.get_logger().info("Araç GUIDED moda alınıyor...")
        mode_ok = call_set_mode(
            node,
            node.mission_clients.set_mode_client,
            "GUIDED",
        )
        if mode_ok is False:
            node.get_logger().error("GUIDED moda geçilemedi. Görev başlatılmadı.")
            return

        node.get_logger().info("Araç FORCE ARM ediliyor...")
        arm_ok = call_trigger_service(
            node,
            node.mission_clients.force_arm_client,
            "FORCE ARM",
        )
        if arm_ok is False:
            node.get_logger().error("FORCE ARM başarısız. Görev başlatılmadı.")
            return

        if not node.wait_for_operational_vehicle_state(timeout_sec=6.0):
            node.get_logger().error(
                "GUIDED/ARM heartbeat teyit edilemedi. Görev başlatılmadı."
            )
            return

        try:
            node.start_telemetry_recording()
        except (OSError, RuntimeError, ValueError) as exc:
            node.get_logger().error(f"Görev veri kaydı başlatılamadı: {exc}")
            stop_vehicle(node.mission_topics.cmd_vel_pub)
            call_trigger_service(
                node,
                node.mission_clients.disarm_client,
                "DISARM",
            )
            return
        node.mission_active = True
        node.get_logger().info("Görev 2 kontrol döngüsü başladı.")

        while (
                rclpy.ok()
                and not node.task.finished
                and node.task.state != MissionState.FAILSAFE
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        node.mission_active = False
        node.stop_telemetry_recording()
        stop_vehicle(node.mission_topics.cmd_vel_pub)

        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error("Görev FAILSAFE nedeniyle sonlandırıldı.")
        else:
            node.get_logger().info("Görev tamamlandı; araç durduruldu.")

        node.get_logger().info("Araç DISARM ediliyor...")
        call_trigger_service(
            node,
            node.mission_clients.disarm_client,
            "DISARM",
        )

    except KeyboardInterrupt:
        node.get_logger().info("Görev kullanıcı tarafından durduruldu.")
        node.mission_active = False
        node.stop_telemetry_recording()
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        try:
            call_trigger_service(
                node,
                node.mission_clients.disarm_client,
                "DISARM",
            )
        except Exception:  # noqa: BLE001
            pass

    finally:
        node.stop_telemetry_recording()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
