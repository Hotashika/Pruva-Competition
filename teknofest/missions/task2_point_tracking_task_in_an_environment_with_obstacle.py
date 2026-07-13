#!/usr/bin/env python3
"""
TEKNOFEST Görev 2
-----------------
Waypoint takibi + tek renk engel dubasından kameradaki sağ/sol konumuna göre kaçınma.

Beklenen kamera topic'i:
    /vision/detections   (std_msgs/String, JSON)

Desteklenen detection örnekleri:
    {
        "detections": [
            {
                "class": "obstacle_buoy",
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from utils.mavlink_utilities import (
    calculate_bearing,
    calculate_gps_distance,
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    create_mission_topics,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    wait_for_mission_services,
)
from utils.read_waypoints import parse_qgc_waypoints


BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = BASE_DIR / "waypoints" / "teknofest_task2.waypoints"

# ============================================================
# ROS / VISION PARAMETRELERİ
# ============================================================
DETECTION_TOPIC = "/vision/detections"
DETECTION_STALE_SEC = 0.75
VISION_HEALTH_TIMEOUT_SEC = 2.0
VISION_WARNING_SEC = 5.0
ACTIVE_TASK_NAME = "task2"

# Model yalnızca engel dubasını algılıyorsa boş bırakılabilir; bu durumda geçerli
# mesafesi olan bütün tespitler engel kabul edilir.
# Model başka nesneleri de algılıyorsa örneğin:
# OBSTACLE_CLASS_NAMES = ("obstacle_buoy", "yellow_buoy")
OBSTACLE_CLASS_NAMES = ()
MIN_OBSTACLE_CONFIDENCE = 0.45

# ============================================================
# GÜVENLİK PARAMETRELERİ
# ============================================================
GPS_TIMEOUT_SEC = 2.0
HEADING_TIMEOUT_SEC = 2.0
BRIDGE_STATE_TIMEOUT_SEC = 2.0
GEOFENCE_RADIUS_M = 150.0
MIN_VALID_ABS_COORD = 1e-6
HOLD_MODE_NAME = "HOLD"

# ============================================================
# KAÇINMA PARAMETRELERİ
# ============================================================
AVOID_ENTER_DIST_M = 3.0
AVOID_EXIT_DIST_M = 4.0
AVOID_FORWARD_DIST_M = 5.0
AVOID_SIDE_DIST_M = 3.0
AVOID_WAYPOINT_TOLERANCE_M = 1.0
AVOID_RETRIGGER_DELAY_SEC = 3.0

# Duba kameranın tam ortasındaysa seçilecek kaçış yönü.
DEFAULT_CENTER_AVOIDANCE_SIDE = "right"

EARTH_RADIUS_M = 6378137.0


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
        self.waypoint_tolerance = 1.0

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None
        self.bridge_connected = False
        self.bridge_mode = None
        self.last_bridge_state_time = None
        self.hold_mode_requested = False
        self.hold_mode_future = None

        self.last_angular_z = 0.0
        self.finished = False
        self.state = MissionState.INIT

        # Kaçınma durumu
        self.avoidance_target = None
        self.avoidance_side = None
        self.avoided_obstacle_side = None
        self.avoidance_block_until = 0.0

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

    def update_bridge_state(self, state_text):
        state_map = {}
        for part in str(state_text).split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            state_map[key.strip().lower()] = value.strip()

        connected_text = state_map.get("connected")
        if connected_text is not None:
            self.bridge_connected = connected_text.lower() == "true"

        mode_text = state_map.get("mode")
        if mode_text:
            self.bridge_mode = mode_text.upper()

        self.last_bridge_state_time = time.monotonic()

    def _request_hold_mode(self):
        if self.hold_mode_requested:
            return

        self.hold_mode_requested = True
        request = SetMode.Request()
        request.base_mode = 0
        request.custom_mode = HOLD_MODE_NAME

        try:
            self.hold_mode_future = self.clients.set_mode_client.call_async(request)
            self.hold_mode_future.add_done_callback(self._hold_mode_done)
            self.logger.warn(f"Failsafe: {HOLD_MODE_NAME} mode requested.")
        except Exception as exc:  # noqa: BLE001 - failsafe request must be logged
            self.logger.error(f"Failed to request {HOLD_MODE_NAME} mode: {exc}")

    def _hold_mode_done(self, future):
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001 - ROS future error must be logged
            self.logger.error(f"{HOLD_MODE_NAME} mode request failed: {exc}")
            return

        if response is not None and getattr(response, "mode_sent", False):
            self.logger.warn(f"{HOLD_MODE_NAME} mode request accepted; awaiting telemetry confirmation.")
        else:
            self.logger.error(f"{HOLD_MODE_NAME} mode request rejected.")

    def _enter_failsafe(self, reason):
        if self.state != MissionState.FAILSAFE:
            self.logger.error(reason)
        self.state = MissionState.FAILSAFE
        stop_vehicle(self.topics.cmd_vel_pub)
        self._request_hold_mode()

    # ========================================================
    # GÜVENLİK
    # ========================================================
    def _check_watchdog(self):
        now = time.monotonic()

        if self.last_gps_time is None or self.last_heading_time is None:
            return False

        if self.last_bridge_state_time is None:
            return False

        if now - self.last_bridge_state_time > BRIDGE_STATE_TIMEOUT_SEC:
            self._enter_failsafe(
                f"Bridge state {BRIDGE_STATE_TIMEOUT_SEC:.1f} saniyeden uzun suredir gelmiyor. "
                "FAILSAFE + HOLD."
            )
            return False

        if not self.bridge_connected:
            self._enter_failsafe("MAVLink bridge baglantisi kesildi. FAILSAFE + HOLD.")
            return False

        if now - self.last_gps_time > GPS_TIMEOUT_SEC:
            self._enter_failsafe(
                f"GPS verisi {GPS_TIMEOUT_SEC:.1f} saniyeden uzun suredir gelmiyor. "
                "FAILSAFE + HOLD."
            )
            return False

        if now - self.last_heading_time > HEADING_TIMEOUT_SEC:
            self._enter_failsafe(
                f"Heading verisi {HEADING_TIMEOUT_SEC:.1f} saniyeden uzun suredir gelmiyor. "
                "FAILSAFE + HOLD."
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
            self._enter_failsafe(
                f"Geofence ihlali: home noktasından {distance_from_home:.1f} m uzaklıkta "
                f"(limit={GEOFENCE_RADIUS_M:.1f} m). FAILSAFE + HOLD."
            )
            return False

        return True

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

    def _normalize_detection(self, obj):
        if not isinstance(obj, dict):
            return None

        class_name = obj.get("class")
        if class_name is None:
            class_name = obj.get("class_name")
        if class_name is None:
            class_name = obj.get("label", "obstacle")
        class_name = str(class_name)

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

        return {
            "class": class_name,
            "confidence": confidence,
            "distance": distance,
            "side": side,
            "angle": angle,
            "bbox": bbox,
            "raw": obj,
        }

    def _nearest_relevant_obstacle(self, detections):
        candidates = []

        for raw_detection in detections or []:
            obstacle = self._normalize_detection(raw_detection)
            if obstacle is None:
                continue

            if obstacle["confidence"] < MIN_OBSTACLE_CONFIDENCE:
                continue

            if OBSTACLE_CLASS_NAMES and obstacle["class"] not in OBSTACLE_CLASS_NAMES:
                continue

            distance = obstacle["distance"]
            if distance is None or not (0.0 < distance < AVOID_EXIT_DIST_M):
                continue

            # Sağ/sol bilgisi yoksa güvenli karar veremeyiz.
            if obstacle["side"] is None:
                continue

            candidates.append(obstacle)

        if not candidates:
            return None

        return min(candidates, key=lambda item: item["distance"])

    # ========================================================
    # KAÇINMA HEDEFİ
    # ========================================================
    @staticmethod
    def _offset_gps(lat, lon, north_m, east_m):
        lat_rad = math.radians(lat)
        new_lat = lat + math.degrees(north_m / EARTH_RADIUS_M)

        cos_lat = math.cos(lat_rad)
        if abs(cos_lat) < 1e-6:
            cos_lat = 1e-6 if cos_lat >= 0 else -1e-6

        new_lon = lon + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
        return {"lat": new_lat, "lon": new_lon}

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

    def _create_avoidance_target(self, avoidance_side, main_target_lat, main_target_lon):
        route_bearing = calculate_bearing(
            self.current_lat,
            self.current_lon,
            main_target_lat,
            main_target_lon,
        )
        bearing_rad = math.radians(route_bearing)

        # Ana rota yönünde ileri bileşen.
        forward_north = AVOID_FORWARD_DIST_M * math.cos(bearing_rad)
        forward_east = AVOID_FORWARD_DIST_M * math.sin(bearing_rad)

        # Compass bearing için sağ = +90°, sol = -90°.
        side_sign = 1.0 if avoidance_side == "right" else -1.0
        side_north = side_sign * AVOID_SIDE_DIST_M * (-math.sin(bearing_rad))
        side_east = side_sign * AVOID_SIDE_DIST_M * math.cos(bearing_rad)

        return self._offset_gps(
            self.current_lat,
            self.current_lon,
            forward_north + side_north,
            forward_east + side_east,
        )

    def _start_avoidance(self, obstacle, main_target_lat, main_target_lon):
        avoidance_side = self._choose_avoidance_side(obstacle["side"])
        target = self._create_avoidance_target(
            avoidance_side,
            main_target_lat,
            main_target_lon,
        )

        self.avoided_obstacle_side = obstacle["side"]
        self.avoidance_side = avoidance_side
        self.avoidance_target = target
        self.state = MissionState.AVOIDING
        self.last_angular_z = 0.0

        vehicle_side_text = (
            "right of buoy"
            if obstacle["side"] == "left"
            else "left of buoy"
            if obstacle["side"] == "right"
            else "in front of buoy"
        )

        self.logger.warn(
            f"ENGEL: class={obstacle['class']}, distance={obstacle['distance']:.2f} m, "
            f"camera_side={obstacle['side']}, vehicle_position={vehicle_side_text}. "
            f"Kaçınma yönü={avoidance_side}; geçici WP="
            f"{target['lat']:.7f}, {target['lon']:.7f}"
        )

    def _finish_avoidance(self):
        self.logger.info(
            f"Kaçınma WP'sine ulaşıldı. {self.avoidance_side} taraftan geçiş tamamlandı; "
            "ana rotaya dönülüyor."
        )
        self.avoidance_target = None
        self.avoidance_side = None
        self.avoided_obstacle_side = None
        self.avoidance_block_until = time.monotonic() + AVOID_RETRIGGER_DELAY_SEC
        self.last_angular_z = 0.0
        self.state = MissionState.NAVIGATING

    # ========================================================
    # GPS HEDEF TAKİBİ
    # ========================================================
    def _navigate_to_gps_target(self, target_lat, target_lon, target_name, tolerance_m):
        distance = calculate_gps_distance(
            self.current_lat,
            self.current_lon,
            target_lat,
            target_lon,
        )

        if distance < tolerance_m:
            self.logger.info(f"{target_name} ulaşıldı. Kalan mesafe: {distance:.2f} m")
            return True

        publish_set_position(
            self.topics.position_target_pub,
            target_lat,
            target_lon,
        )
        self.last_angular_z = 0.0

        self.logger.info(
            f"Hedef={target_name} | mesafe={distance:.2f} m | set_position gönderildi",
            throttle_duration_sec=1.0,
        )
        return False

    # ========================================================
    # ANA STATE MACHINE
    # ========================================================
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
            self._enter_failsafe("Waypoint listesi bos. FAILSAFE + HOLD.")
            return

        if self.current_target_index >= len(self.waypoints):
            if not self.finished:
                self.logger.info("BÜTÜN WAYPOINT'LER TAMAMLANDI. GÖREV BİTTİ.")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.finished = True
                self.state = MissionState.FINISHED
            return

        target_gps = self.waypoints[self.current_target_index]
        target_lat = target_gps["lat"]
        target_lon = target_gps["lon"]

        # ----------------------------------------------------
        # 1. BAŞLANGIÇ WP0 KONTROLÜ
        # ----------------------------------------------------
        if self.state == MissionState.INIT:
            reached_wp0 = self._navigate_to_gps_target(
                target_lat,
                target_lon,
                "WP0 (başlangıç)",
                self.waypoint_tolerance + 2.0,
            )

            if reached_wp0:
                self.logger.info("WP0 doğrulandı; görev navigasyonu başlıyor.")
                self.current_target_index += 1
                self.last_angular_z = 0.0
                self.state = MissionState.NAVIGATING
            return

        # ----------------------------------------------------
        # 2. AKTİF KAÇINMA
        # ----------------------------------------------------
        if self.state == MissionState.AVOIDING:
            if self.avoidance_target is None:
                self._enter_failsafe(
                    "AVOIDING durumunda gecici waypoint yok. FAILSAFE + HOLD."
                )
                return

            reached_avoidance_wp = self._navigate_to_gps_target(
                self.avoidance_target["lat"],
                self.avoidance_target["lon"],
                f"kaçınma WP ({self.avoidance_side})",
                AVOID_WAYPOINT_TOLERANCE_M,
            )

            if reached_avoidance_wp:
                self._finish_avoidance()
            return

        # ----------------------------------------------------
        # 3. YENİ ENGEL TETİKLEME
        # ----------------------------------------------------
        nearest_obstacle = self._nearest_relevant_obstacle(detections)

        if (
            nearest_obstacle is not None
            and nearest_obstacle["distance"] < AVOID_ENTER_DIST_M
            and time.monotonic() >= self.avoidance_block_until
        ):
            self._start_avoidance(
                nearest_obstacle,
                target_lat,
                target_lon,
            )

            # State değiştiği tick'te ilk kaçınma komutunu hemen gönder.
            self._navigate_to_gps_target(
                self.avoidance_target["lat"],
                self.avoidance_target["lon"],
                f"kaçınma WP ({self.avoidance_side})",
                AVOID_WAYPOINT_TOLERANCE_M,
            )
            return

        # ----------------------------------------------------
        # 4. NORMAL WAYPOINT TAKİBİ
        # ----------------------------------------------------
        reached_main_wp = self._navigate_to_gps_target(
            target_lat,
            target_lon,
            f"WP{self.current_target_index}",
            self.waypoint_tolerance,
        )

        if reached_main_wp:
            self.current_target_index += 1
            self.last_angular_z = 0.0


class Task2Node(Node):
    def __init__(self):
        super().__init__("task2_mission_node")
        self.get_logger().info(
            "Görev 2 node'u başlatılıyor: waypoint takibi + konuma göre engelden kaçınma."
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
        self.mission_active = False
        self.valid_gps_received = False
        self.valid_heading_received = False

        self.latest_detections = []
        self.last_detection_message_time = None
        self.node_start_time = time.monotonic()
        self.vision_warning_emitted = False

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
        self.active_task_pub = self.create_publisher(String, "/mission/active_task", 10)

        self.control_timer = self.create_timer(0.1, self.timer_callback)
        self.active_task_timer = self.create_timer(1.0, self.publish_active_task)

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
        self.task.update_bridge_state(msg.data)
        self.bridge_connected = self.task.bridge_connected

    def publish_active_task(self):
        message = String()
        message.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(message)

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
        if age > DETECTION_STALE_SEC:
            return []

        return list(self.latest_detections)

    def _vision_is_healthy(self):
        if self.last_detection_message_time is None:
            return False
        return (
            time.monotonic() - self.last_detection_message_time
            <= VISION_HEALTH_TIMEOUT_SEC
        )

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
        deadline = time.monotonic() + timeout_sec

        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_active_task()
            if self._vision_is_healthy():
                self.get_logger().info("Vision detection stream is healthy.")
                return True

            self.get_logger().info(
                f"{DETECTION_TOPIC} vision stream bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    def wait_for_hold_confirmation(self, timeout_sec=3.0):
        self.task._request_hold_mode()

        future = self.task.hold_mode_future
        if future is not None and not future.done():
            rclpy.spin_until_future_complete(self, future, timeout_sec=min(timeout_sec, 2.0))

        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.task.bridge_mode == HOLD_MODE_NAME:
                self.get_logger().warn(f"Failsafe mode confirmed: {HOLD_MODE_NAME}.")
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(
            f"Failsafe mode could not be confirmed as {HOLD_MODE_NAME}; "
            f"last reported mode={self.task.bridge_mode}."
        )
        return False

    # ========================================================
    # ANA TIMER
    # ========================================================
    def timer_callback(self):
        if not self.mission_active:
            return

        if not self._vision_is_healthy():
            age_text = (
                "never received"
                if self.last_detection_message_time is None
                else f"{time.monotonic() - self.last_detection_message_time:.2f}s old"
            )
            self.task._enter_failsafe(
                f"Vision stream unhealthy ({age_text}). FAILSAFE + HOLD."
            )
            return

        if (
            self.last_detection_message_time is None
            and not self.vision_warning_emitted
            and time.monotonic() - self.node_start_time > VISION_WARNING_SEC
        ):
            self.vision_warning_emitted = True
            self.get_logger().warn(
                f"{DETECTION_TOPIC} topic'inden henüz detection mesajı gelmedi. "
                "Waypoint takibi devam eder fakat kaçınma tetiklenemez."
            )

        current_detections = self._get_fresh_detections()

        try:
            self.task.update(detections=current_detections)
        except Exception as exc:  # noqa: BLE001 - failsafe için geniş yakalama
            self.get_logger().error(f"Görev timer hatası: {exc}")
            self.task._enter_failsafe(f"Unexpected Task 2 control error: {exc}")


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

        if not node.wait_for_vision(timeout_sec=30.0):
            node.get_logger().error(
                "Vision detection stream hazir degil. Gorev ARM edilmeden durduruldu."
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

        node.mission_active = True
        node.publish_active_task()
        node.get_logger().info("Görev 2 kontrol döngüsü başladı.")

        while (
            rclpy.ok()
            and not node.task.finished
            and node.task.state != MissionState.FAILSAFE
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        node.mission_active = False
        stop_vehicle(node.mission_topics.cmd_vel_pub)

        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error("Görev FAILSAFE nedeniyle sonlandırıldı.")
            node.wait_for_hold_confirmation(timeout_sec=3.0)
            return
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
