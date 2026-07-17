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

# Yardımcı fonksiyonlar (Kendi yazdıklarımız ve mavlink_utilities içindekiler)
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
    calculate_gps_distance,
)
from utils.read_waypoints import parse_qgc_waypoints
from teknofest.missions.utils.orange_boundary_guard import OrangeBoundaryGuard

BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = BASE_DIR / "waypoints" / "teknofest_task1.waypoints"

# ============================================================
# SAFETY PARAMS
# ============================================================
GPS_TIMEOUT_SEC = 2.0  # Bu süre GPS/heading gelmezse dur
BRIDGE_STATE_TIMEOUT_SEC = 10.0
HOLD_MODE_NAME = "HOLD"
GEOFENCE_RADIUS_M = 150.0  # Başlangıç noktasından max uzaklık
MIN_VALID_ABS_COORD = 1e-6
WAYPOINT_SETTLE_SEC = 0.75
WAYPOINT_HEADING_TOLERANCE_DEG = 15.0

DETECTION_TOPIC = "/vision/detections"
DETECTION_STALE_SEC = 3.00


class MissionState(Enum):
    INIT = auto()  # Başlangıç konumu bekleniyor / WP0 doğrulanıyor
    NAVIGATING = auto()  # Normal waypoint takibi
    FINISHED = auto()  # Görev tamamlandı
    FAILSAFE = auto()  # GPS kaybı / geofence ihlali / beklenmeyen hata


# ============================================================
# MISSION LOGIC
# ============================================================
class Task1Maneuvering:
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
        self.current_heading = None
        self.last_angular_z = 0.0
        self.finished = False
        self.boundary_guard = OrangeBoundaryGuard()
        self.aligned_target_key = None
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None

        # --- Güvenlik / state machine alanları ---
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None
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

    def update_gps(self, lat, lon):
        """ROS 2 Node'undan gelen güncel GPS verisini kaydeder."""
        self.current_lat = lat
        self.current_lon = lon
        self.last_gps_time = time.monotonic()

        if self.home_lat is None:
            # İlk GPS okuması home/geofence merkezi olarak kaydedilir
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Home position set: {lat:.6f}, {lon:.6f}")

    def update_heading(self, heading):
        """Koridor hedefini global GPS'e çevirmek için güncel heading'i kaydeder."""
        self.current_heading = float(heading) % 360.0
        self.last_heading_time = time.monotonic()

    def _check_watchdog(self):
        """GPS/heading verisi zamanında gelmiyorsa FAILSAFE'e geç. True dönerse devam edilebilir."""
        now = time.monotonic()

        if self.last_gps_time is None:
            # Henüz hiç veri gelmedi, bu normal başlangıç durumu (FAILSAFE değil)
            return False

        if (now - self.last_gps_time) > GPS_TIMEOUT_SEC:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(
                    f"GPS DATA NOT RECEIVED FOR OVER {GPS_TIMEOUT_SEC}s! FAILSAFE."
                )
            self.state = MissionState.FAILSAFE
            return False

        if self.last_heading_time is None:
            return False

        if (now - self.last_heading_time) > GPS_TIMEOUT_SEC:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(
                    f"HEADING DATA NOT RECEIVED FOR OVER {GPS_TIMEOUT_SEC}s! FAILSAFE."
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
        """Home noktasından çok uzaklaşıldıysa FAILSAFE'e geç. True dönerse sınır içinde."""
        if self.home_lat is None or self.current_lat is None:
            return True

        dist_from_home = calculate_gps_distance(
            self.home_lat, self.home_lon,
            self.current_lat, self.current_lon
        )

        if dist_from_home > GEOFENCE_RADIUS_M:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(
                    f"GEOFENCE VIOLATION! {dist_from_home:.1f}m away from home "
                    f"(limit {GEOFENCE_RADIUS_M}m). FAILSAFE."
                )
            self.state = MissionState.FAILSAFE
            return False

        return True

    def _begin_waypoint_hold(self, waypoint_name):
        """Ana waypoint'e varista araci durdurup sonraki heading icin sabitler."""
        stop_vehicle(self.topics.cmd_vel_pub)
        self.waypoint_hold_until = time.monotonic() + WAYPOINT_SETTLE_SEC
        self.waypoint_hold_name = waypoint_name
        self.aligned_target_key = None
        self.logger.info(
            f"{waypoint_name} reached; vehicle stopped for "
            f"{WAYPOINT_SETTLE_SEC:.2f}s before next heading alignment."
        )

    def _waypoint_hold_active(self):
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

    def _set_position_to_gps_target(
            self,
            target_lat,
            target_lon,
            target_name,
            tolerance_m,
            detections,
    ):
        """GPS hedefine yalnız turuncu duba koridorunun içinden gider."""
        distance = calculate_gps_distance(
            self.current_lat, self.current_lon,
            target_lat, target_lon
        )

        if distance < tolerance_m:
            self.logger.info(f"Reached {target_name}! Remaining: {distance:.2f}m")
            return True

        boundary_decision = self._stay_in(detections, target_lat, target_lon)
        if boundary_decision.should_stop:
            publish_cmd_vel(
                self.topics.cmd_vel_pub,
                linear_x=0.0,
                angular_z=0.0,
            )
            self.logger.warn(
                f"Turuncu parkur sınırı güvenle hesaplanamadı "
                f"({boundary_decision.reason}); araç bekletiliyor.",
                throttle_duration_sec=1.0,
            )
            return False

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
                    boundary_decision.target_lat,
                    boundary_decision.target_lon,
                    logger=self.logger,
                    target_name=target_name,
                    tolerance_deg=WAYPOINT_HEADING_TOLERANCE_DEG,
            ):
                return False
            self.aligned_target_key = target_key

        publish_set_position(
            self.topics.position_target_pub,
            boundary_decision.target_lat,
            boundary_decision.target_lon,
        )
        self.last_angular_z = 0.0

        self.logger.info(
            f"Target {target_name} | Distance: {distance:.2f}m | "
            f"boundary={boundary_decision.status}/{boundary_decision.reason} | "
            f"safe_angle={boundary_decision.relative_bearing_deg:.1f}° | "
            f"corridor={boundary_decision.corridor_width_m:.1f}m",
            throttle_duration_sec=1.0
        )
        return False

    def _stay_in(self, detections, target_lat, target_lon):
        """Turuncu dubalardan koridor çıkarıp güvenli kısa GPS hedefi üretir."""
        return self.boundary_guard.compute(
            detections=detections,
            current_lat=self.current_lat,
            current_lon=self.current_lon,
            current_heading=self.current_heading,
            main_target_lat=target_lat,
            main_target_lon=target_lon,
        )

    # noinspection D
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
            self.logger.info("Waiting for GPS Data...", throttle_duration_sec=2.0)
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
            return

        target_gps = self.waypoints[self.current_target_index]
        target_lat = target_gps["lat"]
        target_lon = target_gps["lon"]

        distance = calculate_gps_distance(
            self.current_lat, self.current_lon,
            target_lat, target_lon
        )

        # ---------------------------------------------------------
        # 1. WP0 / MISSION BAŞLANGIÇ KONTROLÜ
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
        # 2. MESAFE VE HEDEF KONTROLÜ
        # ---------------------------------------------------------
        if self._set_position_to_gps_target(
                target_lat,
                target_lon,
                f"WP{self.current_target_index}",
                self.waypoint_tolerance,
                detections,
        ):
            self._begin_waypoint_hold(f"WP{self.current_target_index}")
            self.current_target_index += 1
            return


# ============================================================
# ROS 2 NODE (GÖREV YÖNETİCİSİ)
# ============================================================
class Task1Node(Node):
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

        # 3. Görev Sınıfını Başlat
        self.task = Task1Maneuvering(self, self.mission_topics, self.mission_clients)

        # Anlık Yönelim Değişkeni (GPS Callback'e aktarmak için)
        self.current_heading = None
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

        # 4. Ana Kontrol Döngüsünü Başlat (Saniyede 10 kez çalışır: 0.1 sn)
        self.control_timer = self.create_timer(0.1, self.timer_callback)

    def gps_callback(self, msg):
        """Araçtan gelen NavSatFix verisini dinler."""
        if abs(msg.latitude) < MIN_VALID_ABS_COORD and abs(msg.longitude) < MIN_VALID_ABS_COORD:
            self.get_logger().warn(
                "Gecersiz GPS (0,0) yok sayiliyor.",
                throttle_duration_sec=2.0
            )
            return

        self.valid_gps_received = True
        self.task.update_gps(msg.latitude, msg.longitude)

    def heading_callback(self, msg):
        """Araçtan gelen Float32 yön verisini dinler."""
        try:
            heading = float(msg.data)
        except (TypeError, ValueError):
            heading = float("nan")

        if not math.isfinite(heading):
            self.get_logger().warn(
                "Geçersiz heading verisi yok sayılıyor.",
                throttle_duration_sec=2.0,
            )
            return

        self.current_heading = heading % 360.0
        self.valid_heading_received = True
        self.task.update_heading(self.current_heading)

    def state_callback(self, msg):
        """Bridge heartbeat'ini ayrıştırıp sürekli görev watchdog'una aktarır."""
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
        msg.data = "task1"
        self.active_task_pub.publish(msg)

    @staticmethod
    def _parse_detections_payload(payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("detections", "objects"):
                if isinstance(payload.get(key), list):
                    return payload[key]
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
        if time.monotonic() - self.last_detection_message_time > DETECTION_STALE_SEC:
            return []
        return list(self.latest_detections)

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
        """Mission ARM olmadan önce gerçek GPS ve heading verisini bekler."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.valid_gps_received and self.valid_heading_received:
                return True

            self.get_logger().info(
                "Geçerli GPS ve heading verisi bekleniyor...",
                throttle_duration_sec=2.0
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
                    <= DETECTION_STALE_SEC
            ):
                return True

            self.get_logger().info(
                "Vision heartbeat bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        return False

    def timer_callback(self):
        """Görev mantığını sürekli tetikler.

        KRİTİK: Bu fonksiyon içinde beklenmeyen bir hata (örn. bozuk detection
        formatı) fırlarsa, düzeltilmezse araç son verilen cmd_vel komutuyla
        donmuş halde sürüklenmeye devam eder. Bu yüzden her tick try/except
        ile korunuyor ve hata durumunda araç durduruluyor.
        """
        if not self.mission_active:
            return

        vision_age = (
            None
            if self.last_detection_message_time is None
            else time.monotonic() - self.last_detection_message_time
        )
        if vision_age is None or vision_age > DETECTION_STALE_SEC:
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
def main(args=None):
    rclpy.init(args=args)

    node = Task1Node()

    try:
        if not node.wait_for_bridge_connection(timeout_sec=30.0):
            node.get_logger().error("Bridge MAVLink baglantisi hazir degil! Mission not starting.")
            return

        if not node.wait_for_valid_navigation_data(timeout_sec=30.0):
            node.get_logger().error("Geçerli GPS/heading verisi yok! Mission not starting.")
            return

        if not node.wait_for_vision(timeout_sec=30.0):
            node.get_logger().error("Vision heartbeat yok! Mission not starting.")
            return

        node.get_logger().info("Setting vehicle to GUIDED mode...")
        mode_ok = call_set_mode(node, node.mission_clients.set_mode_client, "GUIDED")
        if mode_ok is False:
            node.get_logger().error("Failed to switch to GUIDED mode! Mission not starting.")
            return

        node.get_logger().info("Force arming vehicle...")
        arm_ok = call_trigger_service(node, node.mission_clients.force_arm_client, "FORCE ARM")
        if arm_ok is False:
            node.get_logger().error("FORCE ARM failed! Mission not starting.")
            return

        if not node.wait_for_operational_vehicle_state(timeout_sec=6.0):
            node.get_logger().error("GUIDED/ARM heartbeat teyit edilemedi! Mission not starting.")
            return

        node.mission_active = True
        node.get_logger().info("Mission loop started.")

        while rclpy.ok() and not node.task.finished and node.task.state != MissionState.FAILSAFE:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.mission_active = False
        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error("Mission terminated due to FAILSAFE.")
        else:
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
