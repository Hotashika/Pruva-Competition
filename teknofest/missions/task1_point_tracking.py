import time
from enum import Enum, auto
from pathlib import Path

import rclpy
from rclpy.node import Node

# Yardımcı fonksiyonlar (Kendi yazdıklarımız ve mavlink_utilities içindekiler)
from utils.mavlink_utilities import (
    create_mission_topics,
    create_mission_clients,
    wait_for_mission_services,
    call_set_mode,
    call_trigger_service,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    calculate_gps_distance,
)
from utils.read_waypoints import parse_qgc_waypoints

BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = BASE_DIR / "waypoints" / "teknofest_task1.waypoints"

# ============================================================
# SAFETY PARAMS
# ============================================================
GPS_TIMEOUT_SEC = 2.0  # Bu süre GPS/heading gelmezse dur
GEOFENCE_RADIUS_M = 150.0  # Başlangıç noktasından max uzaklık
MIN_VALID_ABS_COORD = 1e-6


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
        self.current_heading = 0.0
        self.last_angular_z = 0.0
        self.finished = False

        # --- Güvenlik / state machine alanları ---
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None

    def update_gps(self, lat, lon, heading):
        """ROS 2 Node'undan gelen güncel GPS ve yönelim verilerini kaydeder."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading
        self.last_gps_time = time.monotonic()
        self.last_heading_time = time.monotonic()

        if self.home_lat is None:
            # İlk GPS okuması home/geofence merkezi olarak kaydedilir
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Home position set: {lat:.6f}, {lon:.6f}")

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

    def _set_position_to_gps_target(self, target_lat, target_lon, target_name, tolerance_m):
        """Verilen GPS hedefine bridge set_position hattı ile gider."""
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
    def update(self):
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
                self.waypoint_tolerance
        ):
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
        self.current_heading = 0.0
        self.bridge_connected = False
        self.mission_active = False
        self.valid_gps_received = False

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
        self.task.update_gps(msg.latitude, msg.longitude, self.current_heading)

    def heading_callback(self, msg):
        """Araçtan gelen Float32 yön verisini dinler."""
        self.current_heading = msg.data
        self.task.last_heading_time = time.monotonic()

    def state_callback(self, msg):
        """Bridge'den gelen durum mesajlarını dinler (Gerekirse kullanılır)."""
        self.bridge_connected = "connected=True" in msg.data

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

    def timer_callback(self):
        """Görev mantığını sürekli tetikler.

        KRİTİK: Bu fonksiyon içinde beklenmeyen bir hata (örn. bozuk detection
        formatı) fırlarsa, düzeltilmezse araç son verilen cmd_vel komutuyla
        donmuş halde sürüklenmeye devam eder. Bu yüzden her tick try/except
        ile korunuyor ve hata durumunda araç durduruluyor.
        """
        if not self.mission_active:
            return

        try:
            self.task.update()
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

        if not node.wait_for_valid_gps(timeout_sec=30.0):
            node.get_logger().error("Gecerli GPS konumu yok! Mission not starting.")
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

