#!/usr/bin/env python3
"""
Task-3 Kamikaze Angajman Görevi — Ana Modül
GERÇEK HAYAT TESTİ İÇİN DÜZELTİLMİŞ VERSİYON (v5 — yaklaşma.py ve carpma.py
gerçek görev akışına bağlandı)

v4'ten devralınan doğru tasarım kararları (arama.py v5 ile hizalı):
  - update_heading(): AramaGorevi'nin heading'i artık sadece update_gps()
    üzerinden değil, ayrıca doğrudan bridge'in /cube/gps/heading topic'inden
    de güncelleniyor.

v5'te YAPILAN ASIL DEĞİŞİKLİK:
  Eskiden bu dosyada handle_vision_detections() adında, YaklasmaGorevi/
  CarpmaGorevi hiç kullanılmadan yazılmış BASİT bir yaklaşma taklidi vardı;
  çarpma aşaması (3 vuruş) hiç yoktu — hedefe SAFETY_STOP_DISTANCE kadar
  yaklaşılınca görev "bitmiş" sayılıyordu. Artık:

  1. YaklasmaGorevi ve CarpmaGorevi GERÇEKTEN kullanılıyor (arama.py'nin
     import edildiği paketten, aynı düzende):
         from teknofest.missions.yaklasma import YaklasmaGorevi
         from teknofest.missions.carpma import CarpmaGorevi
     (Dosya yolu arama.py ile aynı klasörde olduğu varsayılıyor:
     teknofest/missions/yaklasma.py ve teknofest/missions/carpma.py.
     Farklıysa bu iki import satırını kendi paket yapınıza göre düzeltin.)

  2. State machine genişletildi: INIT -> SEARCHING -> APPROACHING -> CARPMA
     -> DONE (veya herhangi bir aşamada hedef kaybolursa tekrar SEARCHING).

  3. update_gps() / update_heading() artık arama + yaklaşma + çarpma
     modüllerinin ÜÇÜNE DE aynı anda iletiliyor. Bridge, GPS ve heading'i
     BAĞIMSIZ topic'ler olarak yayınladığından (bkz. bridge_node.py,
     arama.py v5 notları) her üç modülün de kendi update_heading()'i
     olmalı — yaklasma.py ve carpma.py'ye bu v2'lerinde eklendi.

  4. YENİ: IMU verisi artık gerçekten dinleniyor. create_mission_topics()
     (utils/mavlink_utilities.py) bir /cube/imu aboneliği İÇERMİYOR — bu
     dosya mavlink_utilities.py'yi DEĞİŞTİRMEDEN, Task3Node içinde DOĞRUDAN
     '/cube/imu' (sensor_msgs/Imu) aboneliği açıyor ve
     Task3KamikazeEngagement.update_imu() üzerinden yaklaşma+çarpma
     modüllerine dağıtıyor. Bu olmadan carpma.py'nin IMU tabanlı çarpma
     algılaması ve yaklasma.py'nin update_imu() çağrısı hiçbir zaman veri
     almıyordu.

  5. reset_search() artık üç modülü de (arama, yaklaşma, çarpma) sıfırlıyor;
     eskiden sadece arama sıfırlanıyordu, yaklaşma/çarpma yoktu.

  ÖNEMLİ NOT: bridge_node.py ve utils/mavlink_utilities.py bu düzeltmede
  HİÇ değiştirilmedi (kullanıcı talebi üzerine); sadece bu üç görev dosyası
  (arama.py zaten v5'te düzeltilmişti, burada yaklasma/carpma/task3) revize
  edildi.
"""

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
from sensor_msgs.msg import Imu

from utils.mavlink_utilities import (
    create_mission_topics,
    create_mission_clients,
    wait_for_mission_services,
    call_set_mode,
    call_trigger_service,
    stop_vehicle,
    calculate_gps_distance,
)

from teknofest.missions.arama import AramaGorevi
from teknofest.missions.yaklasma import YaklasmaGorevi
from teknofest.missions.carpma import CarpmaGorevi

# ============================================================
# GÜVENLİK PARAMETRELERİ
# ============================================================
GPS_TIMEOUT_SEC = 3.0
HEADING_TIMEOUT_SEC = 2.0
VISION_TIMEOUT_SEC = 1.0
GEOFENCE_RADIUS_M = 150.0
DRIVE_MODE = "GUIDED"

TEST_DEFAULT_TARGET_COLOR = "red"
VALID_TARGET_COLORS = ("red", "green", "black")
TEST_MODE = True
SAFETY_STOP_DISTANCE = 1.0


class MissionState(Enum):
    INIT = auto()
    SEARCHING = auto()
    APPROACHING = auto()
    CARPMA = auto()
    DONE = auto()
    FAILSAFE = auto()


class Task3KamikazeEngagement:
    """Görev yöneticisi sınıfı - 3 aşamayı (arama/yaklaşma/çarpma) koordine eder."""

    def __init__(self, node, mission_topics, mission_clients, target_class,
                 test_mode=False, safety_stop_distance=None):
        self.node = node
        self.is_armed = False
        self.logger = node.get_logger()
        self.test_mode = test_mode

        self.topics = mission_topics
        self.clients = mission_clients
        self.target_class = target_class

        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0

        self.state = MissionState.INIT
        self.last_gps_time = None
        self.last_heading_time = None
        self.last_vision_time = None
        self.home_lat = None
        self.home_lon = None

        self.bridge_connected = None
        self.bridge_armed = None
        self.bridge_mode = None

        self.arama = AramaGorevi(node, mission_topics, target_class, test_mode=test_mode)
        self.yaklasma = YaklasmaGorevi(
            node, mission_topics, target_class,
            safe_stop_distance=safety_stop_distance,
        )
        self.carpma = CarpmaGorevi(node, mission_topics, target_class)

    # --------------------------------------------------------
    def update_gps(self, lat, lon, heading):
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading
        self.last_gps_time = time.monotonic()

        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.logger.info(f"Home konumu ayarlandı: {lat:.6f}, {lon:.6f}")
            if self.state == MissionState.INIT:
                self.state = MissionState.SEARCHING

        # Üç göreve de GPS bilgisini dağıt.
        self.arama.update_gps(lat, lon, heading)
        self.yaklasma.update_gps(lat, lon, heading)
        self.carpma.update_gps(lat, lon, heading)

    # --------------------------------------------------------
    def update_heading(self, heading):
        """heading_callback tarafından çağrılır; bridge bunu DERECE olarak
        yayınlıyor (/cube/gps/heading). Bu topic /cube/gps'ten BAĞIMSIZ
        yayınlanıyor (bridge_node._publish_telemetry), bu yüzden üç göreve
        de DOĞRUDAN iletiyoruz — sadece update_gps()'e güvenmek, GPS geçici
        kaybolduğunda hepsinin heading'inin bayatlamasına yol açardı
        (bkz. arama.py / yaklasma.py / carpma.py update_heading)."""
        self.current_heading = heading
        self.last_heading_time = time.monotonic()
        self.arama.update_heading(heading)
        self.yaklasma.update_heading(heading)
        self.carpma.update_heading(heading)

    # --------------------------------------------------------
    def update_imu(self, gyro_z, accel_x, accel_y, accel_z):
        """YENİ: /cube/imu callback'inden çağrılır. Yaklaşma ve çarpma
        modüllerine ilgili IMU verilerini dağıtır (bkz. Task3Node.imu_callback)."""
        self.yaklasma.update_imu(gyro_z, accel_x, accel_y)
        self.carpma.update_imu(accel_x, accel_y, accel_z)

    # --------------------------------------------------------
    def update_vision_timestamp(self):
        self.last_vision_time = time.monotonic()

    # --------------------------------------------------------
    def _check_watchdog(self):
        """GPS + heading watchdog kontrolü."""
        now = time.monotonic()

        if self.last_gps_time is None:
            return False
        if (now - self.last_gps_time) > GPS_TIMEOUT_SEC:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(f"GPS VERİSİ {GPS_TIMEOUT_SEC} SANİYEDİR GELMİYOR! FAILSAFE.")
            self.state = MissionState.FAILSAFE
            return False

        if self.last_heading_time is not None and (now - self.last_heading_time) > HEADING_TIMEOUT_SEC:
            if self.state != MissionState.FAILSAFE:
                self.logger.error(f"HEADING VERİSİ {HEADING_TIMEOUT_SEC} SANİYEDİR GELMİYOR! FAILSAFE.")
            self.state = MissionState.FAILSAFE
            return False

        return True

    # --------------------------------------------------------
    def is_vision_stale(self):
        if self.last_vision_time is None:
            return True
        return (time.monotonic() - self.last_vision_time) > VISION_TIMEOUT_SEC

    # --------------------------------------------------------
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

    # --------------------------------------------------------
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

    # --------------------------------------------------------
    def reset_search(self):
        """Hedef kaybedildiğinde (yaklaşma veya çarpma sırasında) tüm
        görevi aramaya geri döndürür. Üç modülün de sıfırlanması gerekir;
        eskiden sadece arama sıfırlanıyordu."""
        self.state = MissionState.SEARCHING
        self.arama.reset_search()
        self.yaklasma.reset_approach()
        self.carpma.reset_carpma()
        self.logger.info("[GÖREV] Yeniden arama başlatıldı (arama+yaklaşma+çarpma sıfırlandı).")

    # --------------------------------------------------------
    def update(self, detections):
        """Ana güncelleme döngüsü. Her 0.1 saniyede çağrılır."""

        gps_ok = self._check_watchdog()

        if self.bridge_mode is not None and self.bridge_mode != DRIVE_MODE:
            self.logger.warn(
                f"Bridge mode={self.bridge_mode}. Beklenen mode={DRIVE_MODE}.",
                throttle_duration_sec=2.0,
            )

        if self.bridge_armed is False:
            self.logger.warn("Araç arm değil; cmd_vel etkisiz olabilir.", throttle_duration_sec=2.0)

        if self.state == MissionState.FAILSAFE:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.logger.warn("FAILSAFE aktif, araç durduruldu.", throttle_duration_sec=2.0)
            return

        if not gps_ok:
            self.logger.info("GPS/heading verisi bekleniyor veya kayıp...", throttle_duration_sec=2.0)
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return

        if not self._check_geofence():
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return

        if self.state == MissionState.INIT:
            return

        if self.is_vision_stale():
            self.logger.warn("Vision verisi bayat, tespit yokmuş gibi davranılıyor.",
                              throttle_duration_sec=2.0)
            detections = []

        # ---------------- SEARCHING ----------------
        if self.state == MissionState.SEARCHING:
            if not self.arama.finished:
                self.arama.update(detections)
                return
            self.state = MissionState.APPROACHING
            self.logger.info("[DURUM] Arama tamamlandı, yaklaşma aşamasına geçiliyor.")

        # ---------------- APPROACHING ----------------
        if self.state == MissionState.APPROACHING:
            approach_done = self.yaklasma.update(detections)

            if self.yaklasma.should_return_to_search():
                self.logger.warn("[DURUM] Yaklaşma sırasında hedef kayboldu, aramaya dönülüyor.")
                self.reset_search()
                return

            if approach_done:
                self.state = MissionState.CARPMA
                self.logger.info("[DURUM] Yaklaşma+emin olma tamamlandı, çarpma aşamasına geçiliyor.")
            return

        # ---------------- CARPMA ----------------
        if self.state == MissionState.CARPMA:
            carpma_done = self.carpma.update(detections)

            if self.carpma.should_retry_search():
                self.logger.warn("[DURUM] Çarpma başarısız/hedef kayboldu, aramaya dönülüyor.")
                self.reset_search()
                return

            if carpma_done:
                self.state = MissionState.DONE
                if self.carpma.success:
                    self.logger.info("[DURUM] 🎉 GÖREV BAŞARIYLA TAMAMLANDI (3 çarpma).")
                else:
                    self.logger.error("[DURUM] Görev tamamlanamadı (zaman aşımı).")
            return

        # ---------------- DONE ----------------
        if self.state == MissionState.DONE:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return


class Task3Node(Node):
    """ROS2 Node - Task 3 Kamikaze Engagement"""

    def __init__(self):
        super().__init__('task3_kamikaze_engagement_node')
        self.get_logger().info("=" * 60)
        self.get_logger().info("Task 3 Kamikaze Engagement düğümü başlatılıyor...")
        self.get_logger().info("=" * 60)

        self.declare_parameter('carpilacak_duba', TEST_DEFAULT_TARGET_COLOR)
        self.declare_parameter('test_mode', TEST_MODE)
        self.declare_parameter('safety_stop_distance', SAFETY_STOP_DISTANCE)

        color = self.get_parameter('carpilacak_duba').get_parameter_value().string_value
        color = color.strip().lower()
        self.test_mode = self.get_parameter('test_mode').get_parameter_value().bool_value
        self.safety_stop_distance = self.get_parameter('safety_stop_distance').get_parameter_value().double_value

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

        # YENİ: create_mission_topics() (mavlink_utilities.py) bir IMU
        # aboneliği içermiyor ve bu dosya mavlink_utilities.py'yi
        # DEĞİŞTİRMİYOR. Bu yüzden bridge'in '/cube/imu' topic'ine
        # doğrudan burada abone oluyoruz. Bu abonelik olmadan
        # carpma.py'nin IMU tabanlı çarpma algılaması ve yaklasma.py'nin
        # update_imu() çağrısı hiçbir zaman veri almıyordu.
        self.imu_sub = self.create_subscription(
            Imu,
            '/cube/imu',
            self.imu_callback,
            10,
        )

        # vision_node, /mission/active_task topic'i "task3" ALMADAN
        # detektör YÜKLEMİYOR (bkz. VisionNode.on_task_change /
        # TASK_DETECTOR_MAP). Periyodik olarak (tek seferlik değil)
        # gönderiyoruz çünkü vision_node bu node'dan sonra başlarsa
        # tek seferlik bir yayın kaçırılabilir.
        self.active_task_pub = self.create_publisher(String, '/mission/active_task', 10)
        self.active_task_timer = self.create_timer(1.0, self._publish_active_task)

        self.task = Task3KamikazeEngagement(
            self,
            self.mission_topics,
            self.mission_clients,
            target_class=self.target_class,
            test_mode=self.test_mode,
            safety_stop_distance=self.safety_stop_distance,
        )

        self.current_heading = 0.0
        self.current_detections = []
        self.current_vision_frame_id = None
        self.last_detection_time = None

        self.control_timer = self.create_timer(0.1, self.timer_callback)

        if self.test_mode:
            self.status_timer = self.create_timer(5.0, self.status_callback)

        self.get_logger().info("✅ Task 3 düğümü başlatıldı, görev başlıyor...")

    # --------------------------------------------------------
    def _publish_active_task(self):
        msg = String()
        msg.data = "task3"
        self.active_task_pub.publish(msg)

    # --------------------------------------------------------
    def gps_callback(self, msg):
        self.task.update_gps(msg.latitude, msg.longitude, self.current_heading)

    # --------------------------------------------------------
    def heading_callback(self, msg):
        self.current_heading = msg.data
        self.task.update_heading(msg.data)

    # --------------------------------------------------------
    def imu_callback(self, msg):
        """/cube/imu (sensor_msgs/Imu) -> gyro_z + ivme bileşenlerini
        çıkarıp Task3KamikazeEngagement.update_imu()'ya iletir."""
        gyro_z = msg.angular_velocity.z
        accel_x = msg.linear_acceleration.x
        accel_y = msg.linear_acceleration.y
        accel_z = msg.linear_acceleration.z
        self.task.update_imu(gyro_z, accel_x, accel_y, accel_z)

    # --------------------------------------------------------
    def state_callback(self, msg):
        self.task.update_bridge_state(msg.data)

    # --------------------------------------------------------
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
            self.task.update_vision_timestamp()
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Vision JSON parse edilemedi: {exc}", throttle_duration_sec=2.0)
        except Exception as exc:
            self.get_logger().error(f"Vision callback hatası: {exc}")

    # --------------------------------------------------------
    def timer_callback(self):
        """Ana kontrol döngüsü."""
        try:
            self.task.update(detections=self.current_detections)
        except Exception as exc:
            self.get_logger().error(f"Zamanlayıcı döngüsünde beklenmeyen hata: {exc}")
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub, repeat_count=1)
            except Exception as stop_exc:
                self.get_logger().error(f"Araç durdurulamadı: {stop_exc}")
            self.task.state = MissionState.FAILSAFE

    # --------------------------------------------------------
    def status_callback(self):
        """Test modunda periyodik durum raporu (arama + yaklaşma + çarpma)."""
        arama_status = self.task.arama.get_search_status() if hasattr(self.task, 'arama') else {}
        yaklasma_status = self.task.yaklasma.get_status() if hasattr(self.task, 'yaklasma') else {}
        carpma_status = self.task.carpma.get_status() if hasattr(self.task, 'carpma') else {}

        self.get_logger().info(
            f"[TEST STATUS] Genel: {self.task.state.name} | "
            f"Arama: {arama_status.get('state', 'N/A')} "
            f"bitti={arama_status.get('finished', False)} "
            f"onaylı={arama_status.get('target_confirmed', False)} | "
            f"Yaklaşma: {yaklasma_status.get('state', 'N/A')} "
            f"bitti={yaklasma_status.get('finished', False)} "
            f"nihai_onay={yaklasma_status.get('final_confirmed', False)} "
            f"mesafe={yaklasma_status.get('distance')} | "
            f"Çarpma: {carpma_status.get('state', 'N/A')} "
            f"vuruş={carpma_status.get('hit_count', 0)}/{carpma_status.get('required_hits', 0)}"
        )


def main(args=None):
    """Ana program başlangıç noktası."""
    rclpy.init(args=args)

    try:
        node = Task3Node()
    except SystemExit:
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

        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("⚠️ Kullanıcı tarafından durduruldu.")
    except Exception as exc:
        node.get_logger().error(f"❌ Beklenmeyen hata: {exc}")
    finally:
        node.get_logger().info("Araç durduruluyor...")
        try:
            stop_vehicle(node.mission_topics.cmd_vel_pub, repeat_count=1)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("✅ Task 3 sonlandırıldı.")


if __name__ == "__main__":
    main()