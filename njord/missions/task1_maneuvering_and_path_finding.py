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
    create_mission_topics,
    create_mission_clients,
    wait_for_mission_services,
    call_set_mode,
    call_trigger_service,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    calculate_gps_distance
)
from utils.read_waypoints import parse_qgc_waypoints

BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = BASE_DIR / "waypoints" / "njord1_2_maneuvering.waypoints"
ACTIVE_TASK_NAME = "task1"

# ============================================================
# SAFETY PARAMS
# ============================================================
GPS_TIMEOUT_SEC = 2.0  # Bu sure GPS gelmezse dur ve HOLD moda gecmeyi dene
HEADING_TIMEOUT_SEC = 2.0  # Bu sure heading gelmezse dur ve HOLD moda gecmeyi dene
GEOFENCE_RADIUS_M = 150.0  # Başlangıç noktasından max uzaklık
AVOID_ENTER_DIST_M = 10.0  # Kaçınma tetiklenme mesafesi
AVOID_EXIT_DIST_M = 12.0  # Kaçınma için dikkate alınacak maksimum engel mesafesi
AVOID_LINEAR_X = 0.5  # Kacinma manevrasinda ileri hiz
AVOID_TURN_Z = 0.35  # Kacinma manevrasinda sag/sol donus komutu buyuklugu
AVOID_MANEUVER_MIN_SEC = 1.2  # Temizlenme kabul edilmeden once minimum manevra suresi
AVOID_MANEUVER_MAX_SEC = 3.0  # Tek kacinma manevrasinin maksimum suresi
AVOID_CLEAR_DURATION_SEC = 0.7  # Obje temiz gorundukten sonra ana rotaya donus bekleme suresi
AVOID_CLEAR_ANGLE_DEG = 25.0  # Obje bu acinin disina cikinca merkezden temiz kabul edilir
VISION_DETECTION_TIMEOUT_SEC = 1.0  # Son vision mesajı bu süreden eskiyse yok say
EARTH_RADIUS_M = 6378137.0
MIN_VALID_ABS_COORD = 1e-6
HOLD_MODE_NAME = "HOLD"
RELEVANT_OBSTACLE_CLASSES = (
    "red_buoy",
    "green_buoy",
    "east_cardinal",
    "west_cardinal",
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
        self.is_armed = False
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
        self.current_heading = 0.0
        self.last_angular_z = 0.0
        self.finished = False

        # --- Güvenlik / state machine alanları ---
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None
        self.avoiding_class = None  # RELEVANT_OBSTACLE_CLASSES icinden biri veya None
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = 0.0  # -1.0 right/starboard, +1.0 left/port
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
            self.logger.warn(f"{HOLD_MODE_NAME} mode request accepted.")
        else:
            self.logger.error(f"{HOLD_MODE_NAME} mode request rejected.")

    def _enter_failsafe(self, reason, request_hold=False):
        """Araci FAILSAFE'e alir; gerekirse HOLD moda gecis istegi yollar."""
        if self.state != MissionState.FAILSAFE:
            self.logger.error(reason)

        self.state = MissionState.FAILSAFE

        if request_hold:
            self._request_hold_mode()

    def _check_watchdog(self):
        """GPS/heading verisi zamanında gelmiyorsa FAILSAFE'e geç. True dönerse devam edilebilir."""
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
        """Kaçınma için dönüş yönünü seçer: -1 sağ/starboard, +1 sol/port."""
        obstacle_class = obstacle.get("class")

        if obstacle_class == "red_buoy":
            return -1.0
        if obstacle_class == "green_buoy":
            return 1.0

        if obstacle_class in ("east_cardinal", "west_cardinal"):
            desired_east = 1.0 if obstacle_class == "east_cardinal" else -1.0
            heading_rad = math.radians(self.current_heading)
            starboard_east_component = math.cos(heading_rad)
            side_score = desired_east * starboard_east_component
            if abs(side_score) >= 0.15:
                return -1.0 if side_score > 0 else 1.0

        angle_deg = self._detection_angle_deg(obstacle)
        if angle_deg is not None:
            return 1.0 if angle_deg > 0 else -1.0

        return -1.0

    @staticmethod
    def _avoid_direction_text(turn_direction, obstacle_class):
        if turn_direction < 0:
            turn_text = "starboard/right"
        elif turn_direction > 0:
            turn_text = "port/left"
        else:
            turn_text = "straight"

        if obstacle_class == "east_cardinal":
            return f"east side via {turn_text}"
        if obstacle_class == "west_cardinal":
            return f"west side via {turn_text}"
        return turn_text

    def _is_avoidance_clear(self, obstacle):
        """Obje görüntü merkezinden çıktıysa veya artık görünmüyorsa True döner."""
        if obstacle is None:
            return True

        angle_deg = self._detection_angle_deg(obstacle)
        if angle_deg is None:
            return False

        if self.avoid_turn_direction < 0:
            return angle_deg < -AVOID_CLEAR_ANGLE_DEG
        if self.avoid_turn_direction > 0:
            return angle_deg > AVOID_CLEAR_ANGLE_DEG
        return abs(angle_deg) > AVOID_CLEAR_ANGLE_DEG

    def _reset_avoidance_state(self):
        self.avoiding_class = None
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = 0.0
        self.state = MissionState.NAVIGATING

    def _publish_avoidance_maneuver(self):
        angular_z = self.avoid_turn_direction * AVOID_TURN_Z
        self.last_angular_z = angular_z
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=AVOID_LINEAR_X,
            angular_z=angular_z
        )
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

    # noinspection D
    # Guvenlik, kacinma ve waypoint akislarini tek kontrol dongusunde yurutur.
    def update(self, detections):
        """Sürekli çalışan ana kontrol döngüsü."""

        # ---------------------------------------------------------
        # 0. GÜVENLİK KONTROLLERİ (her şeyden önce)
        # ---------------------------------------------------------
        gps_ok = self._check_watchdog()

        if self.state == MissionState.FAILSAFE:
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.warn("FAILSAFE active, vehicle stopped.", throttle_duration_sec=2.0)
            return

        if not gps_ok:
            # Henüz hiç GPS gelmedi (başlangıç), bekle
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

        if self.current_target_index >= len(self.waypoints):
            if not self.finished:
                self.logger.info("ALL WAYPOINTS REACHED! MISSION COMPLETED!")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.finished = True
                self.state = MissionState.FINISHED
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
            else:
                self._publish_avoidance_maneuver()
                return

        elif nearest is not None and nearest["distance"] < AVOID_ENTER_DIST_M:
            self.state = MissionState.AVOIDING
            self.avoiding_class = nearest["class"]
            self.avoid_started_time = now
            self.avoid_clear_started_time = None
            self.avoid_turn_direction = self._avoid_turn_direction_for_obstacle(nearest)

            direction_text = self._avoid_direction_text(
                self.avoid_turn_direction,
                nearest["class"]
            )
            angle_deg = self._detection_angle_deg(nearest)
            angle_text = "unknown" if angle_deg is None else f"{angle_deg:.1f} deg"

            self.logger.info(
                f"{nearest['class']} ({nearest['distance']:.1f}m, angle={angle_text})! "
                f"Hybrid avoidance maneuver started toward {direction_text}."
            )
            self._publish_avoidance_maneuver()
            return
        # ---------------------------------------------------------
        # 2. WP0 / MISSION BAŞLANGIÇ KONTROLÜ
        # ---------------------------------------------------------
        if self.state == MissionState.INIT:
            if self.current_target_index == 0 and distance < (self.waypoint_tolerance + 2.0):
                self.logger.info("WP0 (Start) point verified, mission starting.")
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
        self.current_heading = 0.0
        self.bridge_connected = False
        self.mission_active = False
        self.valid_gps_received = False

        # 4. Ana Kontrol Döngüsünü Başlat (Saniyede 10 kez çalışır: 0.1 sn)
        self.control_timer = self.create_timer(0.1, self.timer_callback)
        self.active_task_timer = self.create_timer(1.0, self.publish_active_task)

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
        self.current_heading = msg.data
        self.task.last_heading_time = time.monotonic()

    # Bridge durumundan MAVLink baglantisinin hazir olup olmadigini izler.
    def state_callback(self, msg):
        """Bridge'den gelen durum mesajlarını dinler (Gerekirse kullanılır)."""
        self.bridge_connected = "connected=True" in msg.data

    # Mission baslamadan once bridge heartbeat bilgisini bekler.
    def wait_for_bridge_connection(self, timeout_sec=30.0):
        """Bridge servisleri hazir olsa bile MAVLink heartbeat gelene kadar bekler."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.bridge_connected:
                return True

            self.get_logger().info(
                "Bridge MAVLink baglantisi bekleniyor...",
                throttle_duration_sec=2.0
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    # ARM oncesi sifir olmayan gecerli GPS konumu bekler.
    def wait_for_valid_gps(self, timeout_sec=30.0):
        """Mission ARM olmadan once gercek GPS konumu bekler."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.valid_gps_received:
                return True

            self.get_logger().info(
                "Gecerli GPS konumu bekleniyor...",
                throttle_duration_sec=2.0
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

        if not node.wait_for_valid_gps(timeout_sec=30.0):
            node.get_logger().error("Gecerli GPS konumu yok! Mission not starting.")
            return

        node.get_logger().info("Setting vehicle to GUIDED mode...")
        # ------------------------------------------------------------
        mode_ok = call_set_mode(node, node.mission_clients.set_mode_client, "GUIDED")
        if mode_ok is False:
            node.get_logger().error("Failed to switch to GUIDED mode! Mission not starting.")
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

        node.mission_active = True
        node.publish_active_task()
        node.get_logger().info("Mission loop started.")

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
