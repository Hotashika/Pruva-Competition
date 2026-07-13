import json
import sys
import time
from enum import Enum, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from utils.mavlink_utilities import (
    create_mission_topics,
    create_mission_clients,
    wait_for_mission_services,
    call_set_mode,
    call_trigger_service,
    stop_vehicle,
    calculate_gps_distance, publish_cmd_vel,
)

# YENİ: arama aşaması bu modülden geliyor
from teknofest.missions.arama import AramaGorevi

# ============================================================
# GÜVENLİK PARAMETRELERİ
# ============================================================
GPS_TIMEOUT_SEC = 2.0  # Bu süre boyunca GPS gelmezse aracı durdur
GEOFENCE_RADIUS_M = 150.0  # Başlangıç noktasından maksimum uzaklık sınırı

DRIVE_MODE = "GUIDED"

# ------------------------------------------------------------
# ÇARPILACAK DUBA RENGİ
# ------------------------------------------------------------
# Yarışma günü renk hakem tarafından belirlenir ve 'carpilacak_duba'
# parametresiyle dışarıdan verilir (ör. --ros-args -p carpilacak_duba:=black).
#
# ŞİMDİLİK: Gerçek sahada/teknede test yapabilmek için geçici bir varsayılan
# renk tanımlıyoruz -> "red". Yani parametre hiç verilmezse sistem artık
# hata verip durmuyor, doğrudan red_buoy ile çalışmaya başlıyor.
# Yarışma günü bu varsayılanı kullanmayın -- hakemin söylediği rengi MUTLAKA
# parametre olarak açıkça verin, aşağıdaki TEST_DEFAULT_TARGET_COLOR'ı
# değiştirmeyin (o gün unutma riskini azaltmak için).
TEST_DEFAULT_TARGET_COLOR = "red"
VALID_TARGET_COLORS = ("red", "green", "black")


class MissionState(Enum):
    INIT = auto()  # Başlangıç konumu bekleniyor
    FAILSAFE = auto()  # GPS kaybı / geofence ihlali / beklenmeyen hata


class Task3KamikazeEngagement:
    def __init__(self, node, mission_topics, mission_clients, target_class):
        self.node = node
        self.is_armed = False
        self.logger = node.get_logger()

        self.topics = mission_topics
        self.clients = mission_clients

        # Hedef duba sınıfı (ör. "red_buoy"), Task3Node üzerinden,
        # 'carpilacak_duba' ROS2 parametresinden gelir. ZORUNLUDUR.
        self.target_class = target_class

        # Anlık konum verileri
        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0

        # Güvenlik ve durum alanları
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None

        # Bridge state telemetry
        self.bridge_connected = None
        self.bridge_armed = None
        self.bridge_mode = None

        # Arama aşaması nesnesi. handle_vision_detections ile AYNI
        # target_class'ı kullanıyor, böylece renk her yerde tutarlı olur.
        self.arama = AramaGorevi(node, mission_topics, target_class=self.target_class)

    def update_gps(self, lat, lon, heading):
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading
        self.last_gps_time = time.monotonic()
        self.last_heading_time = time.monotonic()

        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Home konumu ayarlandı: {lat:.6f}, {lon:.6f}")

        self.arama.update_gps(lat, lon, heading)

    def _check_watchdog(self):
        now = time.monotonic()
        if self.last_gps_time is None:
            return False

        if (now - self.last_gps_time) > GPS_TIMEOUT_SEC:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(f"GPS VERİSİ {GPS_TIMEOUT_SEC} SANİYEDİR GELMİYOR! FAILSAFE.")
            self.state = MissionState.FAILSAFE
            return False
        return True

    def _check_geofence(self):
        if self.home_lat is None or self.current_lat is None:
            return True

        dist_from_home = calculate_gps_distance(
            self.home_lat, self.home_lon,
            self.current_lat, self.current_lon
        )

        if dist_from_home > GEOFENCE_RADIUS_M:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(
                    f"GEOFENCE İHLALİ! Evden {dist_from_home:.1f}m uzaktasınız "
                    f"(Limit {GEOFENCE_RADIUS_M}m). FAILSAFE."
                )
            self.state = MissionState.FAILSAFE
            return False
        return True

    # noinspection D
    def update_bridge_state(self, state_text):
        try:
            parts = [p.strip() for p in state_text.split(",")]
            state_map = {}
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                state_map[key.strip()] = value.strip()

            if "connected" in state_map:
                self.bridge_connected = state_map["connected"].lower() == "true"
            if "armed" in state_map:
                self.bridge_armed = state_map["armed"].lower() == "true"
            if "mode" in state_map:
                self.bridge_mode = state_map["mode"]
        except Exception as exc:
            self.logger.warn(f"Bridge state parse edilemedi: {exc}", throttle_duration_sec=2.0)

    def handle_vision_detections(self, detections):
        if not detections:
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        targets = [
            d for d in detections
            if d.get("class") == self.target_class
            and d.get("distance") is not None
            and d.get("Buoy angle: ") is not None
        ]

        if not targets:
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        target = min(targets, key=lambda d: d["distance"])
        distance = target["distance"]
        angle = target["Buoy angle: "]

        if distance <= 1.0:
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=0.0)
            return

        kp = 0.03
        angular_z = -kp * angle
        angular_z = max(-0.5, min(0.5, angular_z))

        linear_x = 0.50 if abs(angle) < 8 else 0.50

        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=linear_x,
            angular_z=angular_z,
        )

    # noinspection D
    def update(self, detections):
        gps_ok = self._check_watchdog()

        if self.bridge_mode is not None and self.bridge_mode != DRIVE_MODE:
            self.logger.warn(
                f"Bridge mode={self.bridge_mode}. Attitude target komutları icin beklenen mode={DRIVE_MODE}.",
                throttle_duration_sec=2.0,
            )

        if self.bridge_armed is False:
            self.logger.warn("Arac arm degil; cmd_vel etkisiz olabilir.", throttle_duration_sec=2.0)

        if self.state == MissionState.FAILSAFE:
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.warn("FAILSAFE aktif, araç durduruldu.", throttle_duration_sec=2.0)
            return

        if not gps_ok:
            self.logger.info("GPS verisi bekleniyor...", throttle_duration_sec=2.0)
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if not self._check_geofence():
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        # Hedef bulunana kadar ARAMA çalışır; bulununca YAKLAŞMA/ÇARPMA'ya geçilir
        if not self.arama.finished:
            self.arama.update(detections)
            return

        self.handle_vision_detections(detections)


# ============================================================
# ROS 2 NODE (GÖREV YÖNETİCİSİ)
# ============================================================
class Task3Node(Node):
    def __init__(self):
        super().__init__('task3_kamikaze_engagement_node')
        self.get_logger().info("Task 3 Kamikaze Engagement düğümü başlatılıyor...")

        # ------------------------------------------------------
        # ÇARPILACAK DUBA rengi, node çalıştırılırken dışarıdan verilir.
        # Örnek kullanım:
        #   ros2 run teknofest task3_kamikaze_engagement --ros-args -p carpilacak_duba:=black
        #
        # ŞİMDİLİK (gerçek sahada/teknede test aşaması): parametre hiç
        # verilmezse ROS2, declare_parameter'a burada verdiğimiz
        # TEST_DEFAULT_TARGET_COLOR ("red") değerini otomatik/sessizce
        # kullanır -- yani şu an parametre vermeden de doğrudan
        # test edebilirsiniz.
        #
        # Geçersiz bir değer girilirse (red/green/black dışında bir şey,
        # ör. yazım hatasıyla "blue" veya "mavi") görev YİNE DE
        # BAŞLATILMAZ -- hata basılır, motorlar arm edilmez.
        #
        # YARIŞMA GÜNÜ: Hakemin söylediği rengi MUTLAKA parametre olarak
        # açıkça verin (ör. carpilacak_duba:=black), varsayılana güvenmeyin.
        # ------------------------------------------------------
        self.declare_parameter('carpilacak_duba', TEST_DEFAULT_TARGET_COLOR)
        color = self.get_parameter('carpilacak_duba').get_parameter_value().string_value
        color = color.strip().lower()

        if color not in VALID_TARGET_COLORS:
            self.get_logger().error(
                f"'carpilacak_duba' parametresi geçersiz (girilen: '{color}'). "
                f"Geçerli değerler: {VALID_TARGET_COLORS}. "
                f"Örnek: --ros-args -p carpilacak_duba:=red"
            )
            raise SystemExit(1)

        self.target_class = f"{color}_buoy"
        self.get_logger().info(f"Bu tur için çarpılacak duba: {self.target_class}")

        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)

        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback
        )

        self.vision_sub = self.create_subscription(
            String,
            '/vision/detections',
            self.vision_callback,
            10
        )

        self.task = Task3KamikazeEngagement(
            self,
            self.mission_topics,
            self.mission_clients,
            target_class=self.target_class,
        )
        self.current_heading = 0.0
        self.current_detections = []
        self.current_vision_frame_id = None
        self.last_detection_time = None

        self.control_timer = self.create_timer(0.1, self.timer_callback)

    def gps_callback(self, msg):
        self.task.update_gps(msg.latitude, msg.longitude, self.current_heading)

    def heading_callback(self, msg):
        self.current_heading = msg.data
        self.task.last_heading_time = time.monotonic()

    def state_callback(self, msg):
        self.task.update_bridge_state(msg.data)

    def vision_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            frame_id = payload.get("frame_id")
            detections = payload.get("detections", [])
            if not isinstance(detections, list):
                self.get_logger().warn("Vision detections list formatında değil.", throttle_duration_sec=2.0)
                return

            self.current_vision_frame_id = frame_id
            self.current_detections = detections
            self.last_detection_time = time.monotonic()
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Vision JSON parse edilemedi: {exc}", throttle_duration_sec=2.0)
        except Exception as exc:
            self.get_logger().error(f"Vision callback hatası: {exc}")

    def timer_callback(self):
        try:
            self.task.update(detections=self.current_detections)
        except Exception as exc:
            self.get_logger().error(f"Zamanlayıcı döngüsünde beklenmeyen hata: {exc}")
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub)
            except Exception as stop_exc:
                self.get_logger().error(f"Araç durdurulamadı: {stop_exc}")
            self.task.state = MissionState.FAILSAFE


# ============================================================
# ANA ÇALIŞTIRMA BLOĞU
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    try:
        node = Task3Node()
    except SystemExit:
        # 'carpilacak_duba' parametresi eksik/geçersizdi; Task3Node.__init__
        # içinde zaten hata mesajı basıldı. Görev başlatılmadan çıkılıyor.
        rclpy.shutdown()
        return

    try:
        node.get_logger().info(f"Aracı {DRIVE_MODE} moduna alınıyor...")
        mode_ok = call_set_mode(node, node.mission_clients.set_mode_client, DRIVE_MODE)
        if mode_ok is False:
            node.get_logger().error("Mod geçişi başarısız! Görev başlatılamadı.")
            return

        node.get_logger().info("Motorlar FORCE ARM ediliyor...")
        arm_ok = call_trigger_service(node, node.mission_clients.force_arm_client, "FORCE ARM")
        if arm_ok is False:
            node.get_logger().error("FORCE ARM başarısız! Görev başlatılamadı.")
            return

        node.get_logger().info("Task3 Kamikaze Engagement döngüsü başladı.")

        while rclpy.ok() and node.task.state != MissionState.FAILSAFE:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error("Görev FAILSAFE sebebiyle sonlandırıldı.")

        stop_vehicle(node.mission_topics.cmd_vel_pub)

        node.get_logger().info("Motorlar DISARM ediliyor...")
        call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")

    except KeyboardInterrupt:
        node.get_logger().info("Görev kullanıcı tarafından manuel kesildi (Ctrl+C).")
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        try:
            call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
        except Exception:
            pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()