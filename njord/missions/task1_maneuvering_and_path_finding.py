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

BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = BASE_DIR / "waypoints" / "njord_task1.waypoints"
ACTIVE_TASK_NAME = "task1"

# ============================================================
# SAFETY PARAMS
# ============================================================
GPS_TIMEOUT_SEC = 2.0  # Bu sure GPS gelmezse dur ve HOLD moda gecmeyi dene
HEADING_TIMEOUT_SEC = 2.0  # Bu sure heading gelmezse dur ve HOLD moda gecmeyi dene
BRIDGE_STATE_TIMEOUT_SEC = 10.0  # /cube/state bu sure gelmezse FAILSAFE + HOLD
GEOFENCE_RADIUS_M = 150.0  # Başlangıç noktasından max uzaklık
WAYPOINT_SETTLE_SEC = 0.75  # Her ana GPS noktasinda kesin durus suresi
WAYPOINT_HEADING_TOLERANCE_DEG = 15.0  # Kucuk heading farklarinda gereksiz salinimi onler

AVOID_ENTER_DIST_M = 3.0  # Kaçınma tetiklenme mesafesi
AVOID_EXIT_DIST_M = 5.0  # Kaçınma için dikkate alınacak maksimum engel mesafesi

AVOID_LINEAR_X = 0.3  # Kacinma manevrasinda ileri hiz
AVOID_TURN_Z = 0.15  # Kacinma manevrasinda sag/sol donus komutu buyuklugu

AVOID_MANEUVER_MIN_SEC = 0.5  # Temizlenme kabul edilmeden once minimum manevra suresi
AVOID_MANEUVER_MAX_SEC = 1.5  # Tek kacinma manevrasinin maksimum suresi

AVOID_CLEAR_DURATION_SEC = 0.3  # Obje temiz gorundukten sonra ana rotaya donus bekleme suresi
AVOID_CLEAR_ANGLE_DEG = 25.0  # Obje bu acinin disina cikinca merkezden temiz kabul edilir

CARDINAL_PASS_CLEARANCE_M = 4.0  # Cardinal marker'in dogu/bati tarafindaki gecis mesafesi
CARDINAL_TARGET_TOLERANCE_M = 1.0  # Gecis GPS hedefinin tamamlanma toleransi
CARDINAL_PASS_TIMEOUT_SEC = 20.0  # Gecis hedefi bu surede alinmazsa FAILSAFE + HOLD

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
        self.waypoint_tolerance = 1

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
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = 0.0  # +1.0 right/starboard, -1.0 left/port
        self.cardinal_pass_target = None  # Marker'in coğrafi dogu/batisindaki GPS hedefi
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

    # Kacinma icin dikkate alinacak en yakin samandirayi secer.
    def _nearest_relevant_obstacle(self, detections):
        """Kaçınma menzilindeki en yakın ilgili şamandırayı döndürür (None yoksa)."""
        candidates = [
            obj for obj in detections
            if obj.get("class") in RELEVANT_OBSTACLE_CLASSES
               and obj.get("distance") is not None
               and 0 < obj["distance"] < AVOID_EXIT_DIST_M
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda o: o["distance"])

    # Lokal metre offsetini yaklasik GPS koordinatina donusturur.
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

    def _obstacle_offset_from_detection(self, obstacle):
        """Aci+mesafe ile samandiranin araca gore north/east offsetini hesaplar."""
        try:
            distance_m = float(obstacle.get("distance"))
        except (TypeError, ValueError):
            return None

        angle_deg = self._detection_angle_deg(obstacle)
        if angle_deg is None or not math.isfinite(distance_m) or distance_m <= 0:
            return None

        # Kamera arac ekseniyle hizali kabul edilir: pozitif aci starboard/right tarafidir.
        obstacle_bearing = (self.current_heading + angle_deg) % 360.0
        obstacle_bearing_rad = math.radians(obstacle_bearing)
        obstacle_north = distance_m * math.cos(obstacle_bearing_rad)
        obstacle_east = distance_m * math.sin(obstacle_bearing_rad)
        return obstacle_north, obstacle_east

    def _create_cardinal_pass_target(self, obstacle):
        """Cardinal marker'in istenen coğrafi tarafinda geçici bir GPS hedefi üretir."""
        pass_side = CARDINAL_PASS_SIDES.get(obstacle.get("class"))
        obstacle_offset = self._obstacle_offset_from_detection(obstacle)
        if pass_side is None or obstacle_offset is None:
            return None

        obstacle_north, obstacle_east = obstacle_offset
        marker_gps = self._offset_gps(
            self.current_lat,
            self.current_lon,
            north_m=obstacle_north,
            east_m=obstacle_east,
        )
        pass_east_offset = (
            CARDINAL_PASS_CLEARANCE_M
            if pass_side == "east"
            else -CARDINAL_PASS_CLEARANCE_M
        )
        target_gps = self._offset_gps(
            marker_gps["lat"],
            marker_gps["lon"],
            north_m=0.0,
            east_m=pass_east_offset,
        )
        target_gps.update({
            "side": pass_side,
            "marker_lat": marker_gps["lat"],
            "marker_lon": marker_gps["lon"],
        })
        return target_gps

    def _matching_avoidance_obstacle(self, detections):
        """Aktif kaçınma sınıfından hâlâ yakın görünen objeyi döndürür."""
        if self.avoiding_class is None:
            return None

        candidates = []
        for obj in detections:
            if obj.get("class") != self.avoiding_class:
                continue
            try:
                distance_m = float(obj.get("distance"))
            except (TypeError, ValueError):
                continue
            if 0 < distance_m < AVOID_EXIT_DIST_M:
                candidates.append((distance_m, obj))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

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
            # Engel sagdaysa sola, soldaysa saga acil.
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

        if self.avoid_turn_direction > 0:
            return angle_deg < -AVOID_CLEAR_ANGLE_DEG
        if self.avoid_turn_direction < 0:
            return angle_deg > AVOID_CLEAR_ANGLE_DEG
        return abs(angle_deg) > AVOID_CLEAR_ANGLE_DEG

    def _reset_avoidance_state(self):
        self.avoiding_class = None
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = 0.0
        self.cardinal_pass_target = None
        self.aligned_target_key = None
        self.state = MissionState.NAVIGATING

    def _update_cardinal_pass(self, now):
        """Aktif cardinal hedefini mevcut GNSS position akışıyla takip eder."""
        target = self.cardinal_pass_target
        if target is None:
            return False

        elapsed = 0.0 if self.avoid_started_time is None else now - self.avoid_started_time
        if elapsed >= CARDINAL_PASS_TIMEOUT_SEC:
            self._enter_failsafe(
                f"{target['side'].upper()} CARDINAL PASS TARGET TIMEOUT "
                f"({elapsed:.1f}s)! FAILSAFE + HOLD.",
                request_hold=True,
            )
            stop_vehicle(self.topics.cmd_vel_pub)
            return True

        target_name = f"{target['side'].upper()} cardinal pass"
        if self._set_position_to_gps_target(
                target["lat"],
                target["lon"],
                target_name,
                CARDINAL_TARGET_TOLERANCE_M,
        ):
            self.logger.info(
                f"{target_name} completed; resuming main GNSS route."
            )
            self._reset_avoidance_state()
        return True

    def _publish_avoidance_maneuver(self):
        angular_z = self.avoid_turn_direction * AVOID_TURN_Z
        self.last_angular_z = angular_z
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=AVOID_LINEAR_X,
            angular_z=angular_z
        )

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
        if self._update_cardinal_pass(now):
            return True

        elapsed = 0.0
        if self.avoid_started_time is not None:
            elapsed = now - self.avoid_started_time

        active_obstacle = self._matching_avoidance_obstacle(detections)
        avoidance_done = False

        if elapsed >= AVOID_MANEUVER_MAX_SEC:
            self.logger.info(
                "Avoidance maneuver max duration reached, returning to main route."
            )
            avoidance_done = True
        else:
            clear_enough = (
                elapsed >= AVOID_MANEUVER_MIN_SEC
                and self._is_avoidance_clear(active_obstacle)
            )
            if clear_enough:
                if self.avoid_clear_started_time is None:
                    self.avoid_clear_started_time = now
                elif (now - self.avoid_clear_started_time) >= AVOID_CLEAR_DURATION_SEC:
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
        self.avoid_started_time = now
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = self._avoid_turn_direction_for_obstacle(obstacle)
        self.cardinal_pass_target = self._create_cardinal_pass_target(obstacle)

        if self.cardinal_pass_target is not None:
            target = self.cardinal_pass_target
            angle_deg = self._detection_angle_deg(obstacle)
            self.logger.info(
                f"{obstacle['class']} ({obstacle['distance']:.1f}m, "
                f"angle={angle_deg:.1f} deg)! Geographic {target['side']} pass "
                f"target created at ({target['lat']:.7f}, {target['lon']:.7f}), "
                f"clearance={CARDINAL_PASS_CLEARANCE_M:.1f}m."
            )
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
        # 1. ENGELLERDEN KAÇINMA KONTROLÜ (süre + detection temizlenme state'i)
        # ---------------------------------------------------------
        nearest = self._nearest_relevant_obstacle(detections)
        now = time.monotonic()

        if self.state == MissionState.AVOIDING:
            if self._update_active_avoidance(detections, now):
                return

        elif nearest is not None and nearest["distance"] <= AVOID_ENTER_DIST_M:
            self._start_avoidance(nearest, now)
            return
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
        self.publish_active_task()

    # Vision node'a aktif gorevin task1 oldugunu bildirir.
    def publish_active_task(self):
        msg = String()
        msg.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(msg)

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
