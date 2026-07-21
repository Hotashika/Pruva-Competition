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
GPS_TIMEOUT_SEC = 3.0  # Bu süre GPS gelmezse dur ve HOLD moda geçmeyi dene
HEADING_TIMEOUT_SEC = 3.0  # Bu süre heading gelmezse dur ve HOLD moda geçmeyi dene
BRIDGE_STATE_TIMEOUT_SEC = 12.0  # /cube/state bu süre gelmezse FAILSAFE + HOLD
GEOFENCE_RADIUS_M = 100.0  # Başlangıç noktasından izin verilen maksimum uzaklık
WAYPOINT_SETTLE_SEC = 0.50  # Her ana GPS noktasında kesin duruş süresi
WAYPOINT_HEADING_TOLERANCE_DEG = 15.0  # Küçük heading farklarında gereksiz salınımı önler

AVOID_ENTER_DIST_M = 3.0  # Kırmızı/yeşil şamandıra kaçınma tetiklenme mesafesi
AVOID_EXIT_DIST_M = 6.0  # Aktif kaçınmada şamandıranın izleneceği maksimum mesafe
AVOID_MIN_CONFIDENCE = 0.45  # Düşük güvenli detection'ların manevra başlatmasını önler
AVOID_CONFIRMATION_FRAMES = 3  # Manevra öncesi gereken ardışık vision frame sayısı
AVOID_ASSOCIATION_MAX_ANGLE_DEG = 18.0  # Track ID yoksa frame'ler arası açı kapısı
AVOID_ASSOCIATION_MAX_DISTANCE_M = 2.5  # Track ID yoksa frame'ler arası mesafe kapısı
AVOID_FILTER_ALPHA = 0.35  # Açı ve mesafe ölçümleri için EMA katsayısı

AVOID_MIN_LINEAR_X = 0.10  # Yakın engelde izin verilen minimum ileri komut
AVOID_MAX_LINEAR_X = 0.35  # Güvenli mesafedeki maksimum ileri komut
AVOID_STOP_DIST_M = 0.80  # Bu mesafenin altında ileri komut kesilir
AVOID_DESIRED_SIDE_ANGLE_DEG = 35.0  # Şamandıranın tutulacağı görüntü açısı
AVOID_TURN_KP = 0.012  # Görüntü açı hatasından dönüş komutuna kazanç
AVOID_MAX_TURN_Z = 0.35  # Dönüş komutu sınırı
AVOID_TURN_Z = AVOID_MAX_TURN_Z  # Eski sabit adını kullanan dış kod için uyumluluk
AVOID_LINEAR_X = AVOID_MAX_LINEAR_X  # Eski sabit adını kullanan dış kod için uyumluluk

AVOID_MANEUVER_MIN_SEC = 0.5  # Temizlenme kabul edilmeden önce minimum manevra süresi
AVOID_MANEUVER_MAX_SEC = 8.0  # Aşılır ve engel hâlâ yakınsa FAILSAFE'e geçilir
AVOID_CLEAR_DURATION_SEC = 0.5  # Obje güvenli tarafta kaldıktan sonraki doğrulama süresi
AVOID_LOST_CONFIRM_SEC = 0.8  # Obje kaybolduktan sonra temiz kabul edilme süresi
AVOID_CLEAR_ANGLE_DEG = 35.0  # Obje bu açının dışına çıkınca yanda kabul edilir
AVOID_CLEAR_MIN_DIST_M = 2.0  # Çok yakın obje yalnızca açıyla temiz kabul edilmez
AVOID_RECEDING_MARGIN_M = 0.5  # Obje en yakın ölçümden sonra bu kadar uzaklaşmalı
AVOID_CONFLICT_DIST_M = 1.5  # Farklı yakın engelde manevrayı kesip HOLD'a geç

CARDINAL_ENTER_DIST_M = 8.0  # Kardinal rota üretimini daha güvenli mesafede başlat
CARDINAL_TRACK_DIST_M = 12.0  # Aktif kardinal geçişinde ölçüm güncelleme menzili
CARDINAL_PASS_CLEARANCE_M = 4.0  # Cardinal marker'ın güvenli tarafındaki hedef mesafesi
CARDINAL_MIN_ROUTE_CLEARANCE_M = 3.0  # Giriş segmentinin marker'a minimum mesafesi
CARDINAL_ENTRY_SEARCH_LIMIT_M = 20.0  # Güvenli giriş hedefi arama sınırı
CARDINAL_EXIT_LEAD_M = 4.0  # Çıkış hedefinin güvenli tarafa ek uzaklığı
CARDINAL_REPLAN_THRESHOLD_M = 0.75  # Marker tahmini bu kadar değişirse giriş rotasını yenile
CARDINAL_TARGET_TOLERANCE_M = 1.0  # Geçiş GPS hedefinin tamamlanma toleransı
CARDINAL_PASS_TIMEOUT_SEC = 25.0  # Geçiş hedefleri bu sürede alınmazsa FAILSAFE + HOLD

VISION_DETECTION_TIMEOUT_SEC = 1.0  # Son vision mesajı bu süreden eskiyse yok say
EARTH_RADIUS_M = 6378137.0
MIN_VALID_ABS_COORD = 1e-6

HOLD_MODE_NAME = "HOLD"
RED_BUOY_CLASS = "red_buoys"
GREEN_BUOY_CLASS = "green_buoys"
EAST_CARDINAL_CLASS = "east_buoys"
WEST_CARDINAL_CLASS = "west_buoys"
CARDINAL_PASS_SIDES = {
    EAST_CARDINAL_CLASS: "east",
    WEST_CARDINAL_CLASS: "west",
}
RELEVANT_OBSTACLE_CLASSES = (
    RED_BUOY_CLASS,
    GREEN_BUOY_CLASS,
    EAST_CARDINAL_CLASS,
    WEST_CARDINAL_CLASS,
)
DETECTION_ANGLE_KEYS = ("Buoy angle: ", "Buoy angle", "angle_deg", "angle")


class MissionState(Enum):
    INIT = auto()  # Başlangıç konumu bekleniyor / WP0 doğrulanıyor
    NAVIGATING = auto()  # Normal waypoint takibi
    AVOIDING = auto()  # Şamandıra kaçınma
    FINISHED = auto()  # Görev tamamlandı
    FAILSAFE = auto()  # GPS kaybı / geofence ihlali / beklenmeyen hata


# ============================================================
# GÖREV MANTIĞI
# ============================================================
class Task1Maneuvering:
    # Görev durumunu, waypointleri ve güvenlik değişkenlerini hazırlar.
    def __init__(self, node, mission_topics, mission_clients):
        self.node = node
        self.logger = node.get_logger()

        self.topics = mission_topics
        self.clients = mission_clients

        self.logger.info(f"[INIT-DEBUG] Waypoint path: {WAYPOINT_PATH.resolve()}")

        self.waypoints = parse_qgc_waypoints(WAYPOINT_PATH)
        self.current_target_index = 0
        self.waypoint_tolerance = 1

        self.logger.info(f"[INIT-DEBUG] Parsed waypoints: {self.waypoints}")

        # Anlık konum verileri
        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self.last_angular_z = 0.0
        self.finished = False

        # Güvenlik ve durum makinesi alanları
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self.last_bridge_state_time = None
        self.home_lat = None
        self.home_lon = None
        self.avoiding_class = None  # RELEVANT_OBSTACLE_CLASSES içinden biri veya None
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = 0.0  # +1.0 sağ/starboard, -1.0 sol/port
        self.avoiding_track_id = None
        self.filtered_obstacle_angle = None
        self.filtered_obstacle_distance = None
        self.minimum_obstacle_distance = None
        self.confirmation_candidate_key = None
        self.confirmation_candidate_obstacle = None
        self.confirmation_count = 0
        self.last_confirmation_frame_token = None
        self.synthetic_frame_token = 0
        self.cardinal_marker_estimate = None
        self.cardinal_route_marker = None
        self.cardinal_pass_targets = []
        self.cardinal_target_index = 0
        self.cardinal_pass_target = None  # Aktif giriş veya çıkış GPS hedefi
        self.aligned_target_key = None
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None
        self.waiting_for_sensor_text = "GPS Data"
        self.hold_mode_requested = False
        self.hold_mode_future = None

    # GPS/heading bilgisini günceller ve ilk konumu home noktası yapar.
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

    # GPS veya heading verisi gecikirse görevi FAILSAFE durumuna alır.
    def _request_hold_mode(self):
        """FAILSAFE durumunda aracı HOLD moda almak için tek seferlik istek gönderir."""
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
        except Exception as exc:  # noqa: BLE001 - failsafe mod isteği kesinlikle loglanmalı
            self.logger.error(f"Failed to request {HOLD_MODE_NAME} mode: {exc}")

    def _hold_mode_done(self, future):
        """HOLD mod servis cevabını loglar."""
        try:
            res = future.result()
        except Exception as exc:  # noqa: BLE001 - ROS future hatası loglanmalı
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
        """Aracı FAILSAFE'e alır; gerekirse HOLD moda geçiş isteği yollar."""
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

    # Aracın home merkezli izinli alanın dışına çıkıp çıkmadığını kontrol eder.
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

    # Kaçınma için dikkate alınacak en yakın şamandırayı seçer.
    @staticmethod
    def _obstacle_tracking_limit(obstacle_class):
        if obstacle_class in CARDINAL_PASS_SIDES:
            return CARDINAL_TRACK_DIST_M
        return AVOID_EXIT_DIST_M

    @staticmethod
    def _obstacle_enter_distance(obstacle_class):
        if obstacle_class in CARDINAL_PASS_SIDES:
            return CARDINAL_ENTER_DIST_M
        return AVOID_ENTER_DIST_M

    def _valid_relevant_obstacles(self, detections):
        """Detection alanlarını doğrular ve sayısal değerleri normalize eder."""
        valid = []
        for raw in detections:
            if not isinstance(raw, dict):
                continue

            obstacle_class = raw.get("class")
            if obstacle_class not in RELEVANT_OBSTACLE_CLASSES:
                continue

            try:
                distance_m = float(raw.get("distance"))
                confidence = float(raw.get("confidence", 1.0))
            except (TypeError, ValueError):
                continue

            if (
                    not math.isfinite(distance_m)
                    or not math.isfinite(confidence)
                    or distance_m <= 0.0
                    or distance_m >= self._obstacle_tracking_limit(obstacle_class)
                    or confidence < AVOID_MIN_CONFIDENCE
            ):
                continue

            obstacle = dict(raw)
            obstacle["distance"] = distance_m
            obstacle["confidence"] = confidence
            valid.append(obstacle)
        return valid

    def _nearest_relevant_obstacle(self, detections):
        """Kaçınma menzilindeki en yakın ilgili şamandırayı döndürür (None yoksa)."""
        candidates = self._valid_relevant_obstacles(detections)
        if not candidates:
            return None
        return min(candidates, key=lambda o: o["distance"])

    def _nearest_trigger_obstacle(self, detections):
        candidates = [
            obstacle
            for obstacle in self._valid_relevant_obstacles(detections)
            if obstacle["distance"] <= self._obstacle_enter_distance(
                obstacle["class"]
            )
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda obstacle: obstacle["distance"])

    @staticmethod
    def _obstacle_confirmation_key(obstacle):
        track_id = obstacle.get("track_id")
        if track_id is None:
            return obstacle["class"], None
        return obstacle["class"], str(track_id)

    def _same_confirmation_candidate(self, previous, current):
        if previous is None or previous.get("class") != current.get("class"):
            return False

        previous_track_id = previous.get("track_id")
        current_track_id = current.get("track_id")
        if previous_track_id is not None and current_track_id is not None:
            return str(previous_track_id) == str(current_track_id)

        if abs(previous["distance"] - current["distance"]) > (
                AVOID_ASSOCIATION_MAX_DISTANCE_M
        ):
            return False

        previous_angle = self._detection_angle_deg(previous)
        current_angle = self._detection_angle_deg(current)
        if previous_angle is None or current_angle is None:
            return True
        return abs(previous_angle - current_angle) <= AVOID_ASSOCIATION_MAX_ANGLE_DEG

    def _confirmed_nearest_obstacle(self, detections, frame_token=None):
        """Aynı engel ardışık vision frame'lerinde görülürse manevraya izin verir."""
        nearest = self._nearest_trigger_obstacle(detections)

        if frame_token is None:
            self.synthetic_frame_token += 1
            frame_token = ("synthetic", self.synthetic_frame_token)

        if frame_token == self.last_confirmation_frame_token:
            if nearest is None:
                return None
            key = self._obstacle_confirmation_key(nearest)
            if key == self.confirmation_candidate_key and (
                    self.confirmation_count >= AVOID_CONFIRMATION_FRAMES
            ):
                return nearest
            return None

        self.last_confirmation_frame_token = frame_token
        if nearest is None:
            self.confirmation_candidate_key = None
            self.confirmation_candidate_obstacle = None
            self.confirmation_count = 0
            return None

        candidate_key = self._obstacle_confirmation_key(nearest)
        if (
                candidate_key == self.confirmation_candidate_key
                and self._same_confirmation_candidate(
                    self.confirmation_candidate_obstacle,
                    nearest,
                )
        ):
            self.confirmation_count += 1
        else:
            self.confirmation_candidate_key = candidate_key
            self.confirmation_count = 1
        self.confirmation_candidate_obstacle = nearest

        if self.confirmation_count >= AVOID_CONFIRMATION_FRAMES:
            return nearest
        return None

    # Lokal metre offsetini yaklaşık GPS koordinatına dönüştürür.
    @staticmethod
    def _offset_gps(lat, lon, north_m, east_m):
        """Metre cinsinden lokal north/east offset'i yaklaşık GPS koordinatına çevirir."""
        lat_rad = math.radians(lat)
        new_lat = lat + math.degrees(north_m / EARTH_RADIUS_M)

        cos_lat = math.cos(lat_rad)
        if abs(cos_lat) < 1e-6:
            # Kutuplara yakın kullanım beklenmiyor; sıfıra bölmeyi engelle.
            cos_lat = 1e-6 if cos_lat >= 0 else -1e-6

        new_lon = lon + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
        return {"lat": new_lat, "lon": new_lon}

    @staticmethod
    def _detection_angle_deg(obstacle):
        """Detector çıktısındaki açı alanını derece olarak okur."""
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

    def _obstacle_offset_from_detection(self, obstacle):
        """Açı ve mesafe ile şamandıranın araca göre north/east offsetini hesaplar."""
        try:
            distance_m = float(obstacle.get("distance"))
        except (TypeError, ValueError):
            return None

        angle_deg = self._detection_angle_deg(obstacle)
        if (
                angle_deg is None
                or self.current_heading is None
                or not math.isfinite(distance_m)
                or distance_m <= 0
        ):
            return None

        # Kamera araç ekseniyle hizalı kabul edilir: pozitif açı sağ/starboard tarafıdır.
        obstacle_bearing = (self.current_heading + angle_deg) % 360.0
        obstacle_bearing_rad = math.radians(obstacle_bearing)
        obstacle_north = distance_m * math.cos(obstacle_bearing_rad)
        obstacle_east = distance_m * math.sin(obstacle_bearing_rad)
        return obstacle_north, obstacle_east

    @staticmethod
    def _gps_offset_m(origin_lat, origin_lon, target_lat, target_lon):
        """Yakın GPS noktaları arasındaki north/east farkını metre olarak döndürür."""
        mean_lat_rad = math.radians((origin_lat + target_lat) / 2.0)
        north_m = math.radians(target_lat - origin_lat) * EARTH_RADIUS_M
        east_m = (
            math.radians(target_lon - origin_lon)
            * EARTH_RADIUS_M
            * math.cos(mean_lat_rad)
        )
        return north_m, east_m

    @staticmethod
    def _point_to_segment_distance(point, start, end):
        """İki boyutlu bir noktanın doğru parçasına en kısa mesafesini hesaplar."""
        segment_north = end[0] - start[0]
        segment_east = end[1] - start[1]
        segment_length_sq = segment_north ** 2 + segment_east ** 2
        if segment_length_sq <= 1e-9:
            return math.hypot(point[0] - start[0], point[1] - start[1])

        projection = (
            (point[0] - start[0]) * segment_north
            + (point[1] - start[1]) * segment_east
        ) / segment_length_sq
        projection = max(0.0, min(1.0, projection))
        nearest = (
            start[0] + projection * segment_north,
            start[1] + projection * segment_east,
        )
        return math.hypot(point[0] - nearest[0], point[1] - nearest[1])

    def _marker_gps_from_detection(self, obstacle):
        obstacle_offset = self._obstacle_offset_from_detection(obstacle)
        if obstacle_offset is None:
            return None
        return self._offset_gps(
            self.current_lat,
            self.current_lon,
            north_m=obstacle_offset[0],
            east_m=obstacle_offset[1],
        )

    def _build_cardinal_pass_targets(self, marker_gps, pass_side):
        """Marker çevresinde güvenli giriş ve çıkış hedefleri üretir."""
        if pass_side not in ("east", "west"):
            return []

        marker_north, marker_east = self._gps_offset_m(
            self.current_lat,
            self.current_lon,
            marker_gps["lat"],
            marker_gps["lon"],
        )
        start_relative = (-marker_north, -marker_east)
        start_distance = math.hypot(*start_relative)
        if start_distance + 1e-6 < CARDINAL_MIN_ROUTE_CLEARANCE_M:
            return []

        side_sign = 1.0 if pass_side == "east" else -1.0
        preferred_sign = 1.0 if start_relative[0] >= 0.0 else -1.0
        if abs(start_relative[0]) < 0.5 and self.current_heading is not None:
            heading_north = math.cos(math.radians(float(self.current_heading)))
            if abs(heading_north) >= 0.1:
                preferred_sign = -1.0 if heading_north > 0.0 else 1.0

        candidates = []
        search_step = 1.0
        search_count = int(CARDINAL_ENTRY_SEARCH_LIMIT_M / search_step)
        for index in range(int(CARDINAL_PASS_CLEARANCE_M), search_count + 1):
            magnitude = index * search_step
            for north_sign in (preferred_sign, -preferred_sign):
                entry_relative = (
                    north_sign * magnitude,
                    side_sign * CARDINAL_PASS_CLEARANCE_M,
                )
                route_clearance = self._point_to_segment_distance(
                    (0.0, 0.0),
                    start_relative,
                    entry_relative,
                )
                if route_clearance + 1e-6 < CARDINAL_MIN_ROUTE_CLEARANCE_M:
                    continue

                route_length = math.hypot(
                    entry_relative[0] - start_relative[0],
                    entry_relative[1] - start_relative[1],
                )
                direction_penalty = 0.0 if north_sign == preferred_sign else 2.0
                candidates.append((
                    route_length + direction_penalty + magnitude * 0.05,
                    entry_relative,
                    route_clearance,
                ))

        if not candidates:
            return []

        _, entry_relative, route_clearance = min(candidates, key=lambda item: item[0])
        exit_relative = (
            0.0,
            side_sign * (CARDINAL_PASS_CLEARANCE_M + CARDINAL_EXIT_LEAD_M),
        )

        targets = []
        for phase, offset in (("entry", entry_relative), ("exit", exit_relative)):
            target = self._offset_gps(
                marker_gps["lat"],
                marker_gps["lon"],
                north_m=offset[0],
                east_m=offset[1],
            )
            target.update({
                "phase": phase,
                "side": pass_side,
                "marker_lat": marker_gps["lat"],
                "marker_lon": marker_gps["lon"],
                "route_clearance_m": route_clearance,
            })
            targets.append(target)
        return targets

    def _create_cardinal_pass_targets(self, obstacle):
        """Detection'dan iki aşamalı cardinal geçiş rotası üretir."""
        pass_side = CARDINAL_PASS_SIDES.get(obstacle.get("class"))
        marker_gps = self._marker_gps_from_detection(obstacle)
        if pass_side is None or marker_gps is None:
            return []
        return self._build_cardinal_pass_targets(marker_gps, pass_side)

    def _create_cardinal_pass_target(self, obstacle):
        """Eski tek-hedef API'si için güvenli giriş hedefini döndürür."""
        targets = self._create_cardinal_pass_targets(obstacle)
        return targets[0] if targets else None

    def _matching_avoidance_obstacle(self, detections):
        """Aktif kaçınma sınıfından hâlâ yakın görünen objeyi döndürür."""
        if self.avoiding_class is None:
            return None

        candidates = []
        for obj in self._valid_relevant_obstacles(detections):
            if obj.get("class") != self.avoiding_class:
                continue
            track_id = obj.get("track_id")
            if (
                    self.avoiding_track_id is not None
                    and track_id is not None
                    and str(track_id) != self.avoiding_track_id
            ):
                continue
            candidates.append((obj["distance"], obj))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _conflicting_close_obstacle(self, detections):
        """Aktif hedeften farklı ve acil mesafedeki engeli döndürür."""
        for obstacle in self._valid_relevant_obstacles(detections):
            if obstacle["distance"] > AVOID_CONFLICT_DIST_M:
                continue
            if obstacle.get("class") != self.avoiding_class:
                return obstacle
            track_id = obstacle.get("track_id")
            if (
                    self.avoiding_track_id is not None
                    and track_id is not None
                    and str(track_id) != self.avoiding_track_id
            ):
                return obstacle
        return None

    def _update_filtered_obstacle(self, obstacle, initialize=False):
        """Açı ve mesafedeki tek-frame sıçramalarını üstel ortalamayla azaltır."""
        if obstacle is None:
            return

        distance_m = float(obstacle["distance"])
        angle_deg = self._detection_angle_deg(obstacle)
        if initialize or self.filtered_obstacle_distance is None:
            self.filtered_obstacle_distance = distance_m
        else:
            self.filtered_obstacle_distance = (
                AVOID_FILTER_ALPHA * distance_m
                + (1.0 - AVOID_FILTER_ALPHA) * self.filtered_obstacle_distance
            )
        if initialize or self.minimum_obstacle_distance is None:
            self.minimum_obstacle_distance = self.filtered_obstacle_distance
        else:
            self.minimum_obstacle_distance = min(
                self.minimum_obstacle_distance,
                self.filtered_obstacle_distance,
            )

        if angle_deg is None:
            return
        if initialize or self.filtered_obstacle_angle is None:
            self.filtered_obstacle_angle = angle_deg
        else:
            self.filtered_obstacle_angle = (
                AVOID_FILTER_ALPHA * angle_deg
                + (1.0 - AVOID_FILTER_ALPHA) * self.filtered_obstacle_angle
            )

    def _filtered_obstacle(self):
        if self.filtered_obstacle_distance is None:
            return None
        obstacle = {
            "class": self.avoiding_class,
            "distance": self.filtered_obstacle_distance,
        }
        if self.filtered_obstacle_angle is not None:
            obstacle["angle_deg"] = self.filtered_obstacle_angle
        return obstacle

    def _avoid_turn_direction_for_obstacle(self, obstacle):
        """Normal şamandıra veya cardinal GPS fallback manevrasının yönünü seçer."""
        obstacle_class = obstacle.get("class")

        if obstacle_class == RED_BUOY_CLASS:
            return 1.0
        if obstacle_class == GREEN_BUOY_CLASS:
            return -1.0

        pass_side = CARDINAL_PASS_SIDES.get(obstacle_class)
        if pass_side is not None and self.current_heading is not None:
            target_bearing = 90.0 if pass_side == "east" else 270.0
            heading_error = (
                                    target_bearing - float(self.current_heading) + 180.0
                            ) % 360.0 - 180.0
            if abs(heading_error) < 1e-6:
                return 0.0
            return 1.0 if heading_error > 0.0 else -1.0

        angle_deg = self._detection_angle_deg(obstacle)
        if angle_deg is not None:
            # Engel sağdaysa sola, soldaysa sağa açıl.
            return -1.0 if angle_deg > 0 else 1.0

        return 1.0

    @staticmethod
    def _avoid_direction_text(turn_direction, obstacle_class):
        if turn_direction > 0:
            turn_text = "starboard/right"
        elif turn_direction < 0:
            turn_text = "port/left"
        else:
            turn_text = "straight"

        if obstacle_class == EAST_CARDINAL_CLASS:
            return f"east side via {turn_text}"
        if obstacle_class == WEST_CARDINAL_CLASS:
            return f"west side via {turn_text}"
        return turn_text

    def _is_avoidance_clear(self, obstacle):
        """Obje görüntü merkezinden çıktıysa veya artık görünmüyorsa True döner."""
        if obstacle is None:
            return True

        angle_deg = self._detection_angle_deg(obstacle)
        if angle_deg is None:
            return False

        try:
            distance_m = float(obstacle.get("distance"))
        except (TypeError, ValueError):
            return False
        if distance_m < AVOID_CLEAR_MIN_DIST_M:
            return False
        minimum_distance = self.minimum_obstacle_distance
        if (
                minimum_distance is not None
                and distance_m < minimum_distance + AVOID_RECEDING_MARGIN_M
        ):
            return False

        if self.avoid_turn_direction > 0:
            return angle_deg <= -AVOID_CLEAR_ANGLE_DEG
        if self.avoid_turn_direction < 0:
            return angle_deg >= AVOID_CLEAR_ANGLE_DEG
        return abs(angle_deg) >= AVOID_CLEAR_ANGLE_DEG

    def _reset_avoidance_state(self):
        self.avoiding_class = None
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = 0.0
        self.avoiding_track_id = None
        self.filtered_obstacle_angle = None
        self.filtered_obstacle_distance = None
        self.minimum_obstacle_distance = None
        self.confirmation_candidate_key = None
        self.confirmation_candidate_obstacle = None
        self.confirmation_count = 0
        self.cardinal_marker_estimate = None
        self.cardinal_route_marker = None
        self.cardinal_pass_targets = []
        self.cardinal_target_index = 0
        self.cardinal_pass_target = None
        self.aligned_target_key = None
        self.state = MissionState.NAVIGATING

    def _update_cardinal_marker_estimate(self, obstacle):
        """Yeni detection ile marker konumunu yumuşatır ve gerekirse girişi yeniler."""
        measurement = self._marker_gps_from_detection(obstacle)
        if measurement is None:
            return

        if self.cardinal_marker_estimate is None:
            self.cardinal_marker_estimate = measurement
        else:
            self.cardinal_marker_estimate = {
                "lat": (
                    AVOID_FILTER_ALPHA * measurement["lat"]
                    + (1.0 - AVOID_FILTER_ALPHA) * self.cardinal_marker_estimate["lat"]
                ),
                "lon": (
                    AVOID_FILTER_ALPHA * measurement["lon"]
                    + (1.0 - AVOID_FILTER_ALPHA) * self.cardinal_marker_estimate["lon"]
                ),
            }

        if self.cardinal_target_index != 0 or self.cardinal_route_marker is None:
            return

        marker_delta = self._gps_offset_m(
            self.cardinal_route_marker["lat"],
            self.cardinal_route_marker["lon"],
            self.cardinal_marker_estimate["lat"],
            self.cardinal_marker_estimate["lon"],
        )
        if math.hypot(*marker_delta) < CARDINAL_REPLAN_THRESHOLD_M:
            return

        pass_side = CARDINAL_PASS_SIDES[self.avoiding_class]
        targets = self._build_cardinal_pass_targets(
            self.cardinal_marker_estimate,
            pass_side,
        )
        if not targets:
            return

        self.cardinal_pass_targets = targets
        self.cardinal_pass_target = targets[0]
        self.cardinal_route_marker = dict(self.cardinal_marker_estimate)
        self.aligned_target_key = None
        self.logger.info(
            f"{pass_side.upper()} cardinal entry replanned after marker update."
        )

    def _update_cardinal_pass(self, detections, now):
        """Aktif cardinal hedefini mevcut GNSS position akışıyla takip eder."""
        target = self.cardinal_pass_target
        if target is None:
            return False

        active_obstacle = self._matching_avoidance_obstacle(detections)
        if active_obstacle is not None:
            self._update_filtered_obstacle(active_obstacle)
            self._update_cardinal_marker_estimate(active_obstacle)
            target = self.cardinal_pass_target

        elapsed = 0.0 if self.avoid_started_time is None else now - self.avoid_started_time
        if elapsed >= CARDINAL_PASS_TIMEOUT_SEC:
            self._enter_failsafe(
                f"{target['side'].upper()} CARDINAL PASS TARGET TIMEOUT "
                f"({elapsed:.1f}s)! FAILSAFE + HOLD.",
                request_hold=True,
            )
            stop_vehicle(self.topics.cmd_vel_pub)
            return True

        phase = target["phase"]
        target_name = f"{target['side'].upper()} cardinal {phase}"
        if self._set_position_to_gps_target(
                target["lat"],
                target["lon"],
                target_name,
                CARDINAL_TARGET_TOLERANCE_M,
        ):
            if phase == "entry":
                self.cardinal_target_index = 1
                self.cardinal_pass_target = self.cardinal_pass_targets[1]
                self.aligned_target_key = None
                self.logger.info(
                    f"{target_name} completed; proceeding to cardinal exit."
                )
                return True

            marker = self.cardinal_marker_estimate or {
                "lat": target["marker_lat"],
                "lon": target["marker_lon"],
            }
            marker_to_vehicle = self._gps_offset_m(
                marker["lat"],
                marker["lon"],
                self.current_lat,
                self.current_lon,
            )
            side_sign = 1.0 if target["side"] == "east" else -1.0
            safe_side_distance = side_sign * marker_to_vehicle[1]
            marker_distance = math.hypot(*marker_to_vehicle)
            if (
                    safe_side_distance
                    < CARDINAL_PASS_CLEARANCE_M - CARDINAL_TARGET_TOLERANCE_M
                    or marker_distance < CARDINAL_MIN_ROUTE_CLEARANCE_M
            ):
                self._enter_failsafe(
                    f"{target_name} reached without verified safe clearance! "
                    "FAILSAFE + HOLD.",
                    request_hold=True,
                )
                stop_vehicle(self.topics.cmd_vel_pub)
                return True

            self.logger.info(f"{target_name} completed; resuming main GNSS route.")
            self._reset_avoidance_state()
        return True

    def _publish_avoidance_maneuver(self):
        """Şamandıra açısı ve mesafesine göre hız/dönüş komutunu günceller."""
        angle_deg = self.filtered_obstacle_angle
        distance_m = self.filtered_obstacle_distance
        desired_angle = (
            -AVOID_DESIRED_SIDE_ANGLE_DEG
            if self.avoiding_class == RED_BUOY_CLASS
            else AVOID_DESIRED_SIDE_ANGLE_DEG
        )

        if angle_deg is None:
            angular_z = self.avoid_turn_direction * AVOID_MAX_TURN_Z
            angle_error = AVOID_DESIRED_SIDE_ANGLE_DEG
        else:
            angle_error = angle_deg - desired_angle
            angular_z = max(
                -AVOID_MAX_TURN_Z,
                min(AVOID_MAX_TURN_Z, AVOID_TURN_KP * angle_error),
            )

        if distance_m is None or distance_m <= AVOID_STOP_DIST_M:
            linear_x = 0.0
        else:
            distance_ratio = (
                (distance_m - AVOID_STOP_DIST_M)
                / (AVOID_ENTER_DIST_M - AVOID_STOP_DIST_M)
            )
            distance_ratio = max(0.0, min(1.0, distance_ratio))
            linear_x = (
                AVOID_MIN_LINEAR_X
                + distance_ratio * (AVOID_MAX_LINEAR_X - AVOID_MIN_LINEAR_X)
            )
            steering_factor = max(
                0.35,
                1.0 - min(abs(angle_error) / 90.0, 0.65),
            )
            linear_x *= steering_factor

        self.last_angular_z = angular_z
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=linear_x,
            angular_z=angular_z
        )

    def _begin_waypoint_hold(self, waypoint_name):
        """Ana GPS noktasında aracı durdurup heading geçişi için sabitler."""
        stop_vehicle(self.topics.cmd_vel_pub)
        self.waypoint_hold_until = time.monotonic() + WAYPOINT_SETTLE_SEC
        self.waypoint_hold_name = waypoint_name
        self.aligned_target_key = None
        self.logger.info(
            f"{waypoint_name} reached; vehicle stopped for "
            f"{WAYPOINT_SETTLE_SEC:.2f}s before next heading alignment."
        )

    def _waypoint_hold_active(self):
        """Planlı waypoint duruşu devam ediyorsa sıfır hareket komutu yayınlar."""
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

    # GPS hedefine MAVLink position target komutu yayınlar.
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
        conflicting = self._conflicting_close_obstacle(detections)
        if conflicting is not None:
            self._enter_failsafe(
                f"Conflicting obstacle {conflicting['class']} at "
                f"{conflicting['distance']:.1f}m during avoidance! FAILSAFE + HOLD.",
                request_hold=True,
            )
            stop_vehicle(self.topics.cmd_vel_pub)
            return True

        if self._update_cardinal_pass(detections, now):
            return True

        elapsed = 0.0
        if self.avoid_started_time is not None:
            elapsed = now - self.avoid_started_time

        active_obstacle = self._matching_avoidance_obstacle(detections)
        if active_obstacle is not None:
            self._update_filtered_obstacle(active_obstacle)
        avoidance_done = False

        if elapsed >= AVOID_MANEUVER_MAX_SEC:
            filtered = self._filtered_obstacle()
            if active_obstacle is not None and not self._is_avoidance_clear(filtered):
                self._enter_failsafe(
                    "Avoidance timeout while obstacle is still close! "
                    "FAILSAFE + HOLD.",
                    request_hold=True,
                )
                stop_vehicle(self.topics.cmd_vel_pub)
                return True
            self.logger.info("Avoidance timeout reached after obstacle clearance.")
            avoidance_done = True
        else:
            filtered = self._filtered_obstacle() if active_obstacle is not None else None
            clear_enough = elapsed >= AVOID_MANEUVER_MIN_SEC and (
                active_obstacle is None or self._is_avoidance_clear(filtered)
            )
            if clear_enough:
                if self.avoid_clear_started_time is None:
                    self.avoid_clear_started_time = now
                clear_duration = (
                    AVOID_LOST_CONFIRM_SEC
                    if active_obstacle is None
                    else AVOID_CLEAR_DURATION_SEC
                )
                if (now - self.avoid_clear_started_time) >= clear_duration:
                    self.logger.info("Obstacle cleared, returning to main route.")
                    avoidance_done = True
            else:
                self.avoid_clear_started_time = None

        if avoidance_done:
            self._reset_avoidance_state()
            return False

        self._publish_avoidance_maneuver()
        return True

    def _start_avoidance(self, obstacle, now):
        """Yeni bir engel için mevcut kaçınma davranışını başlatır."""
        self.state = MissionState.AVOIDING
        self.avoiding_class = obstacle["class"]
        track_id = obstacle.get("track_id")
        self.avoiding_track_id = None if track_id is None else str(track_id)
        self.avoid_started_time = now
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = self._avoid_turn_direction_for_obstacle(obstacle)
        self._update_filtered_obstacle(obstacle, initialize=True)
        self.cardinal_pass_targets = self._create_cardinal_pass_targets(obstacle)
        self.cardinal_target_index = 0
        self.cardinal_pass_target = (
            self.cardinal_pass_targets[0]
            if self.cardinal_pass_targets
            else None
        )

        if self.cardinal_pass_target is not None:
            target = self.cardinal_pass_target
            self.cardinal_marker_estimate = {
                "lat": target["marker_lat"],
                "lon": target["marker_lon"],
            }
            self.cardinal_route_marker = dict(self.cardinal_marker_estimate)
            angle_deg = self._detection_angle_deg(obstacle)
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.info(
                f"{obstacle['class']} ({obstacle['distance']:.1f}m, "
                f"angle={angle_deg:.1f} deg)! Geographic {target['side']} "
                f"entry/exit route created; minimum planned clearance="
                f"{target['route_clearance_m']:.1f}m."
            )
            return

        if obstacle["class"] in CARDINAL_PASS_SIDES:
            self._enter_failsafe(
                f"Safe route could not be created for {obstacle['class']} at "
                f"{obstacle['distance']:.1f}m! FAILSAFE + HOLD.",
                request_hold=True,
            )
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        direction_text = self._avoid_direction_text(
            self.avoid_turn_direction,
            obstacle["class"],
        )
        angle_deg = self._detection_angle_deg(obstacle)
        angle_text = "unknown" if angle_deg is None else f"{angle_deg:.1f} deg"
        self.logger.info(
            f"{obstacle['class']} ({obstacle['distance']:.1f}m, angle={angle_text})! "
            f"Hybrid avoidance maneuver started toward {direction_text}."
        )
        self._publish_avoidance_maneuver()

    def update(self, detections, detection_frame_token=None):
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
        # 1. Engelden kaçınma kontrolü: süre ve detection temizlenme durumu
        # ---------------------------------------------------------
        now = time.monotonic()

        if self.state == MissionState.AVOIDING:
            if self._update_active_avoidance(detections, now):
                return

        else:
            confirmed = self._confirmed_nearest_obstacle(
                detections,
                frame_token=detection_frame_token,
            )
            if confirmed is not None:
                self._start_avoidance(confirmed, now)
                return
        # ---------------------------------------------------------
        # 2. WP0 / görev başlangıç kontrolü
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
                # ama görevi NAVIGATING'e geçirmeden (WP0'ı atlamadan).
                pass

        self.state = MissionState.NAVIGATING if self.state == MissionState.INIT else self.state

        # ---------------------------------------------------------
        # 3. Mesafe ve hedef kontrolü
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
    # ROS node'unu, servisleri, topicleri ve periyodik kontrol timer'ını kurar.
    def __init__(self):
        super().__init__('task1_mission_node')
        self.get_logger().info("Task 1 (Maneuvering) Node Starting...")

        # Servis istemcilerini oluştur ve hazır olmalarını bekle.
        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)

        # Topic aboneliklerini ve yayıncılarını oluştur.
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback
        )

        self.latest_detections = []
        self.last_detection_time = None
        self.latest_detection_frame_token = None
        self.detection_message_sequence = 0
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

        # Görev sınıfını başlat.
        self.task = Task1Maneuvering(self, self.mission_topics, self.mission_clients)

        # Anlık yönelim değişkeni; GPS callback'e aktarılır.
        self.current_heading = None
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self._last_logged_bridge_state = None
        self.mission_active = False
        self.valid_gps_received = False
        self.valid_heading_received = False

        # Ana kontrol döngüsünü başlat; saniyede 10 kez çalışır.
        self.control_timer = self.create_timer(0.1, self.timer_callback)
        self.active_task_timer = self.create_timer(1.0, self.publish_active_task)
        self.publish_active_task()

    # Vision node'a aktif görevin task1 olduğunu bildirir.
    def publish_active_task(self):
        msg = String()
        msg.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(msg)

    # Vision detection JSON mesajlarını saklar.
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
        self.detection_message_sequence += 1
        payload_frame_id = payload.get("frame_id")
        self.latest_detection_frame_token = (
            payload_frame_id
            if payload_frame_id is not None
            else ("message", self.detection_message_sequence)
        )

    # Eski vision mesajlarını kullanmamak için güncel detection listesini döndürür.
    def _current_detections(self):
        if self.last_detection_time is None:
            return []

        if (time.monotonic() - self.last_detection_time) > VISION_DETECTION_TIMEOUT_SEC:
            return []

        return self.latest_detections

    # GPS mesajlarını doğrular ve görev mantığına aktarır.
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

    # Heading mesajını saklar ve watchdog zamanını tazeler.
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

    # Bridge durumundan MAVLink bağlantısının hazır olup olmadığını izler.
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

    # Mission başlamadan önce bridge heartbeat bilgisini bekler.
    def wait_for_bridge_connection(self, timeout_sec=30.0):
        """Bridge servisleri hazır olsa bile MAVLink heartbeat gelene kadar bekler."""
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

    # ARM öncesi sıfır olmayan geçerli GPS konumu bekler.
    def wait_for_valid_navigation_data(self, timeout_sec=30.0):
        """Mission ARM olmadan önce gerçek GPS ve heading verisini bekler."""
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
        """ARM öncesi vision node'dan en az bir güncel frame mesajı bekler."""
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

    # Timer tick'lerinde aktif görevi çalıştırır ve hatada aracı durdurur.
    def timer_callback(self):
        """Görev mantığını sürekli tetikler.

        KRİTİK: Bu fonksiyon içinde beklenmeyen bir hata (örn. bozuk detection
        formatı) fırlarsa, düzeltilmezse araç son verilen cmd_vel komutuyla
        donmuş halde sürüklenmeye devam eder. Bu yüzden her tick try/except
        ile korunuyor ve hata durumunda araç durduruluyor.
        """
        # Vision cache güncel değilse boş liste döner; eski detection ile manevra yapılmaz.
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
            self.task.update(
                detections=current_detections,
                detection_frame_token=self.latest_detection_frame_token,
            )
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
# ROS 2 node yaşam döngüsünü başlatır, aracı hazırlar ve spin'e girer.
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
