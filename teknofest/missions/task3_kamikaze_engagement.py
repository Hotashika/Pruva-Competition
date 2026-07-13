#!/usr/bin/env python3
"""
Task-3 Kamikaze Angajman Görevi — Ana Modül
GERÇEK HAYAT TESTİ İÇİN TAMAMLANMIŞ VERSİYON

Bu modül ROS2 node'u olarak çalışır ve 3 aşamalı görevi yönetir:
1. ARAMA: AramaGorevi sınıfı ile hedef duba aranır
2. YAKLAŞMA: Hedef bulunduğunda yaklaşma başlar
3. ÇARPMA: Hedefe çarpma (1m mesafede durur)

Hedef kaybolursa otomatik olarak arama aşamasına geri dönülür.
"""

import json
import sys
import time
from enum import Enum, auto
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Utilities
from utils.mavlink_utilities import (
    create_mission_topics,
    create_mission_clients,
    wait_for_mission_services,
    call_set_mode,
    call_trigger_service,
    stop_vehicle,
    calculate_gps_distance,
    publish_cmd_vel,
)

# ARAMA aşaması bu modülden geliyor
from teknofest.missions.arama import AramaGorevi

# ============================================================
# GÜVENLİK PARAMETRELERİ
# ============================================================
GPS_TIMEOUT_SEC = 3.0              # GPS kaybı timeout
GEOFENCE_RADIUS_M = 150.0          # Maksimum uzaklık sınırı
DRIVE_MODE = "GUIDED"
TARGET_LOSS_TIMEOUT_FRAMES = 20   # Hedef kaç frame kaybolursa aramaya dön

# ============================================================
# TEST PARAMETRELERİ
# ============================================================
TEST_DEFAULT_TARGET_COLOR = "red"
VALID_TARGET_COLORS = ("red", "green", "black")
TEST_MODE = True                   # Test modu (detaylı loglar)
SAFETY_STOP_DISTANCE = 1.0        # Çarpma mesafesi


class MissionState(Enum):
    INIT = auto()          # Başlangıç konumu bekleniyor
    SEARCHING = auto()     # Arama aşaması
    APPROACHING = auto()   # Yaklaşma aşaması
    FAILSAFE = auto()      # GPS kaybı / geofence ihlali


class Task3KamikazeEngagement:
    """Görev yöneticisi sınıfı - 3 aşamayı koordine eder."""
    
    def __init__(self, node, mission_topics, mission_clients, target_class, test_mode=False):
        self.node = node
        self.is_armed = False
        self.logger = node.get_logger()
        self.test_mode = test_mode

        self.topics = mission_topics
        self.clients = mission_clients
        self.target_class = target_class

        # Konum verileri
        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0

        # Güvenlik
        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.home_lat = None
        self.home_lon = None

        # Bridge state telemetry
        self.bridge_connected = None
        self.bridge_armed = None
        self.bridge_mode = None

        # ARAMA aşaması nesnesi (test modu ile)
        self.arama = AramaGorevi(node, mission_topics, target_class, test_mode=test_mode)

        # Hedef kaybı takibi
        self.target_loss_counter = 0
        self.max_target_loss_frames = TARGET_LOSS_TIMEOUT_FRAMES
        self.approach_active = False
        self.approach_start_time = None

    def update_gps(self, lat, lon, heading):
        """GPS verilerini günceller."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading
        self.last_gps_time = time.monotonic()
        self.last_heading_time = time.monotonic()

        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Home konumu ayarlandı: {lat:.6f}, {lon:.6f}")
            # INIT'ten SEARCHING'e geç
            if self.state == MissionState.INIT:
                self.state = MissionState.SEARCHING

        self.arama.update_gps(lat, lon, heading)

    def _check_watchdog(self):
        """GPS watchdog kontrolü."""
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
        """Geofence kontrolü."""
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

    def update_bridge_state(self, state_text):
        """Bridge durum bilgisini günceller."""
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

    def reset_search(self):
        """Aramayı yeniden başlatır."""
        self.approach_active = False
        self.target_loss_counter = 0
        self.state = MissionState.SEARCHING
        self.arama.reset_search()
        self.logger.info("[ARAMA] Yeniden arama başlatıldı.")

    def handle_vision_detections(self, detections):
        """
        Hedefe yaklaşma ve çarpma mantığı.
        Hedef kaybolursa aramayı yeniden başlatır.
        """
        if not detections:
            # Tespit yok, kayıp sayacını artır
            self.target_loss_counter += 1
            stop_vehicle(self.topics.cmd_vel_pub)

            # Yaklaşma aşamasındaysak ve hedef belirli süredir kayıpsa, aramaya dön
            if self.approach_active and self.target_loss_counter > self.max_target_loss_frames:
                self.logger.warn(
                    f"[YAKLAŞMA] Hedef {self.max_target_loss_frames} frame kayboldu, "
                    f"yeniden aramaya başlanıyor."
                )
                self.reset_search()
            return

        # Tespit geldiğinde kayıp sayacını sıfırla
        if self.target_loss_counter > 0:
            self.target_loss_counter = 0

        # Hedef sınıfına göre filtrele
        targets = [
            d for d in detections
            if d.get("class") == self.target_class
            and d.get("distance") is not None
            and d.get("distance", -1) > 0.5
            and d.get("Buoy angle: ") is not None
        ]

        if not targets:
            # Hedef sınıfı eşleşen tespit yok
            self.target_loss_counter += 1
            stop_vehicle(self.topics.cmd_vel_pub)

            if self.approach_active and self.target_loss_counter > self.max_target_loss_frames:
                self.logger.warn(f"[YAKLAŞMA] Hedef sınıfı {self.target_class} kayboldu.")
                self.reset_search()
            return

        # Hedef bulundu, yaklaşma aktif
        if not self.approach_active:
            self.approach_active = True
            self.approach_start_time = time.monotonic()
            self.logger.info("[YAKLAŞMA] Yaklaşma başlıyor!")

        self.state = MissionState.APPROACHING

        # En yakın hedefi seç
        target = min(targets, key=lambda d: d["distance"])
        distance = target["distance"]
        angle = target["Buoy angle: "]

        # Test modunda detaylı log
        if self.test_mode:
            self.logger.info(f"[TEST] Mesafe: {distance:.2f}m, Açı: {angle:.1f}°")

        # Çarpma mesafesine ulaşıldı mı?
        if distance <= SAFETY_STOP_DISTANCE:
            self.logger.info(f"[ÇARPMA] Hedefe ulaşıldı! Mesafe: {distance:.2f}m. Durduruluyor.")
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=0.0)
            
            # Görevi tamamlandı olarak işaretle
            if self.test_mode:
                elapsed = time.monotonic() - self.approach_start_time if self.approach_start_time else 0
                self.logger.info(f"[TEST] Yaklaşma süresi: {elapsed:.1f} sn")
            return

        # PID benzeri kontrol
        kp = 0.035
        angular_z = -kp * angle
        angular_z = max(-0.5, min(0.5, angular_z))

        # Açıya göre hız ayarlama
        if abs(angle) < 10:
            linear_x = 0.60
        elif abs(angle) < 30:
            linear_x = 0.35
        else:
            linear_x = 0.15

        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=linear_x,
            angular_z=angular_z,
        )

    def update(self, detections):
        """Ana güncelleme döngüsü. Her 0.1 saniyede çağrılır."""
        
        # --- GÜVENLİK KONTROLLERİ ---
        gps_ok = self._check_watchdog()

        if self.bridge_mode is not None and self.bridge_mode != DRIVE_MODE:
            self.logger.warn(
                f"Bridge mode={self.bridge_mode}. Beklenen mode={DRIVE_MODE}.",
                throttle_duration_sec=2.0,
            )

        if self.bridge_armed is False:
            self.logger.warn("Araç arm değil; cmd_vel etkisiz olabilir.", throttle_duration_sec=2.0)

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

        # INIT durumunda home ayarlanmamış olabilir
        if self.state == MissionState.INIT:
            return

        # --- STATE MAKİNESİ ---
        if self.state == MissionState.SEARCHING:
            # Hedef bulunana kadar ARAMA çalışır
            if not self.arama.finished:
                self.arama.update(detections)
                return
            else:
                # Hedef bulundu, yaklaşma aşamasına geç
                self.state = MissionState.APPROACHING
                self.approach_active = True
                self.approach_start_time = time.monotonic()
                self.logger.info("[DURUM] Arama tamamlandı, yaklaşma aşamasına geçiliyor.")

        if self.state == MissionState.APPROACHING:
            # Yaklaşma ve çarpma
            self.handle_vision_detections(detections)


# ============================================================
# ROS 2 NODE (GÖREV YÖNETİCİSİ)
# ============================================================
class Task3Node(Node):
    """ROS2 Node - Task 3 Kamikaze Engagement"""
    
    def __init__(self):
        super().__init__('task3_kamikaze_engagement_node')
        self.get_logger().info("=" * 60)
        self.get_logger().info("Task 3 Kamikaze Engagement düğümü başlatılıyor...")
        self.get_logger().info("=" * 60)

        # ------------------------------------------------------
        # ROS2 PARAMETRELERİ
        # ------------------------------------------------------
        self.declare_parameter('carpilacak_duba', TEST_DEFAULT_TARGET_COLOR)
        self.declare_parameter('test_mode', TEST_MODE)
        self.declare_parameter('safety_stop_distance', SAFETY_STOP_DISTANCE)
        
        color = self.get_parameter('carpilacak_duba').get_parameter_value().string_value
        color = color.strip().lower()
        self.test_mode = self.get_parameter('test_mode').get_parameter_value().bool_value
        self.safety_stop_distance = self.get_parameter('safety_stop_distance').get_parameter_value().double_value

        # Parametre doğrulama
        if color not in VALID_TARGET_COLORS:
            self.get_logger().error(
                f"'carpilacak_duba' parametresi geçersiz (girilen: '{color}'). "
                f"Geçerli değerler: {VALID_TARGET_COLORS}. "
                f"Örnek: --ros-args -p carpilacak_duba:=red"
            )
            raise SystemExit(1)

        self.target_class = f"{color}_buoy"
        self.get_logger().info(f"🎯 Çarpılacak duba: {self.target_class}")
        self.get_logger().info(f"🧪 Test modu: {self.test_mode}")
        self.get_logger().info(f"🛑 Durma mesafesi: {self.safety_stop_distance}m")

        # ------------------------------------------------------
        # MISSION CLIENT'LAR
        # ------------------------------------------------------
        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)

        # ------------------------------------------------------
        # MISSION TOPIC'LER
        # ------------------------------------------------------
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback
        )

        # ------------------------------------------------------
        # VISION SUBSCRIBER
        # ------------------------------------------------------
        self.vision_sub = self.create_subscription(
            String,
            '/vision/detections',
            self.vision_callback,
            10
        )

        # ------------------------------------------------------
        # TASK NESNESİ
        # ------------------------------------------------------
        self.task = Task3KamikazeEngagement(
            self,
            self.mission_topics,
            self.mission_clients,
            target_class=self.target_class,
            test_mode=self.test_mode,
        )

        # Durum değişkenleri
        self.current_heading = 0.0
        self.current_detections = []
        self.current_vision_frame_id = None
        self.last_detection_time = None

        # ------------------------------------------------------
        # TIMER'LAR
        # ------------------------------------------------------
        # 0.1 saniyede bir kontrol
        self.control_timer = self.create_timer(0.1, self.timer_callback)

        # Test modunda 5 saniyede bir durum raporu
        if self.test_mode:
            self.status_timer = self.create_timer(5.0, self.status_callback)

        self.get_logger().info("✅ Task 3 düğümü başlatıldı, görev başlıyor...")

    # ------------------------------------------------------
    # CALLBACK'LER
    # ------------------------------------------------------
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
        """Ana kontrol döngüsü."""
        try:
            self.task.update(detections=self.current_detections)
        except Exception as exc:
            self.get_logger().error(f"Zamanlayıcı döngüsünde beklenmeyen hata: {exc}")
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub)
            except Exception as stop_exc:
                self.get_logger().error(f"Araç durdurulamadı: {stop_exc}")
            self.task.state = MissionState.FAILSAFE

    def status_callback(self):
        """Test modunda periyodik durum raporu."""
        status = self.task.arama.get_search_status() if hasattr(self.task, 'arama') else {}
        
        self.get_logger().info(
            f"[TEST STATUS] "
            f"State: {self.task.state.name}, "
            f"Arama: {status.get('state', 'N/A')}, "
            f"Tamamlandı: {status.get('finished', False)}, "
            f"Onaylandı: {status.get('target_confirmed', False)}, "
            f"Ardışık: {status.get('consecutive_detections', 0)}, "
            f"İstasyon: {status.get('visited_positions', 0)}, "
            f"Süre: {status.get('elapsed_time', 0):.1f}s"
        )


# ============================================================
# ANA ÇALIŞTIRMA BLOĞU
# ============================================================
def main(args=None):
    """Ana program başlangıç noktası."""
    rclpy.init(args=args)

    try:
        node = Task3Node()
    except SystemExit:
        # 'carpilacak_duba' parametresi eksik/geçersizdi
        rclpy.shutdown()
        return

    try:
        node.get_logger().info("=" * 60)
        node.get_logger().info(f"Aracı {DRIVE_MODE} moduna alınıyor...")
        mode_ok = call_set_mode(node, node.mission_clients.set_mode_client, DRIVE_MODE)
        if mode_ok is False:
            node.get_logger().error("❌ Mod geçişi başarısız! Görev başlatılamadı.")
            rclpy.shutdown()
            return
        node.get_logger().info("✅ Mod geçişi başarılı.")

        node.get_logger().info("Motorlar FORCE ARM ediliyor...")
        arm_ok = call_trigger_service(node, node.mission_clients.force_arm_client, "FORCE ARM")
        if arm_ok is False:
            node.get_logger().error("❌ FORCE ARM başarısız! Görev başlatılamadı.")
            rclpy.shutdown()
            return
        node.get_logger().info("✅ FORCE ARM başarılı.")

        node.get_logger().info("=" * 60)
        node.get_logger().info("🚀 Task 3 Kamikaze Engagement görevi başlıyor!")
        node.get_logger().info("=" * 60)

        # ROS2 spin (olay döngüsü)
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("⚠️ Kullanıcı tarafından durduruldu.")
    except Exception as exc:
        node.get_logger().error(f"❌ Beklenmeyen hata: {exc}")
    finally:
        node.get_logger().info("Araç durduruluyor...")
        try:
            stop_vehicle(node.mission_topics.cmd_vel_pub)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("✅ Task 3 sonlandırıldı.")


if __name__ == "__main__":
    main()