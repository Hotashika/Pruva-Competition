#!/usr/bin/env python3
"""
Task-3 Kamikaze Angajman Görevi — Ana Modül
GERÇEK HAYAT TESTİ İÇİN DÜZELTİLMİŞ VERSİYON (v6 — arama.py/yaklasma.py/carpma.py
ile birlikte güncellendi)

v6 NOTU: Bu dosyada FONKSİYONEL bir değişiklik YAPILMADI. arama.py, yaklasma.py
ve carpma.py'ye eklenen düzeltmeler (yerinde dönüş, dönüş retry limiti, kare-kare
süreklilik filtresi, saldırıda saf yaw düzeltmesi, yaklaşma fazı toplam süre/segment
sınırı) hepsi ilgili modüllerin İÇİNDE, mevcut public arayüzlerini (update_gps,
update_heading, update_imu, update, reset_*, get_*status) DEĞİŞTİRMEDEN yapıldı.
Bu yüzden bu orkestrasyon dosyasının üç modülü çağırma şekli aynı kalabildi;
dosya bütünlük için burada aynen tekrar veriliyor.

v5'ten devralınan tasarım:
  - Task3KamikazeEngagement, AramaGorevi -> YaklasmaGorevi -> CarpmaGorevi
    sırasını bir state machine ile yönetir (INIT -> SEARCHING -> APPROACHING
    -> CARPMA -> DONE), herhangi bir aşamada hedef kaybolursa SEARCHING'e
    geri döner.
  - update_gps() / update_heading() / update_imu() üç modüle de AYNI ANDA
    iletilir (bridge bunları bağımsız topic'ler olarak yayınladığı için).
  - reset_search() üç modülü de sıfırlar.
  - IMU verisi Task3Node içinde doğrudan '/cube/imu' aboneliğiyle alınır
    (create_mission_topics() bunu içermez) ve update_imu() üzerinden
    yaklaşma+çarpma modüllerine dağıtılır.
"""

import json
import math
import os
import sys
import time
from enum import Enum, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = str(REPO_ROOT)
# Bu dosya dogrudan calistirildiginda script dizini
# ``teknofest/missions`` sys.path[0] olur. Buradaki missions/utils paketi,
# depo kokundeki ortak utils paketini golgelememeli. Kok zaten listede olsa
# bile ilk siraya tasimak bu isim cakismasini kesin olarak engeller.
while repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
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

# Yarışma denemesinde vurulacak tek renk. Sonradan renk değişecekse
# yalnızca bu sabiti ve yeni modelin sınıf adını güncelleyin.
ACTIVE_TARGET_COLOR = "red"
ACTIVE_TARGET_CLASS = f"{ACTIVE_TARGET_COLOR}_buoy"
TEST_MODE = False  # Yalnızca ayrıntılı log içindir; sensör verisi üretmez
SAFETY_STOP_DISTANCE = 1.0
MIN_TARGET_CONFIDENCE = 0.65
IMPACT_THRESHOLD_MPS2 = 4.0
USE_FORCE_ARM = False


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


AUTO_START_DEFAULT = _env_flag("TASK3_AUTO_START", False)


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
                 test_mode=False, safety_stop_distance=None,
                 min_target_confidence=MIN_TARGET_CONFIDENCE,
                 impact_delta_threshold=IMPACT_THRESHOLD_MPS2):
        self.node = node
        self.is_armed = False
        self.logger = node.get_logger()
        self.test_mode = test_mode

        self.topics = mission_topics
        self.clients = mission_clients
        self.target_class = target_class
        self.min_target_confidence = float(min_target_confidence)
        self.impact_delta_threshold = float(impact_delta_threshold)

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None

        self.state = MissionState.INIT
        self.mission_enabled = False
        self.last_gps_time = None
        self.last_heading_time = None
        self.last_vision_time = None
        self.home_lat = None
        self.home_lon = None

        self.bridge_connected = None
        self.bridge_armed = None
        self.bridge_mode = None

        self.arama = AramaGorevi(
            node, mission_topics, target_class,
            test_mode=test_mode,
            min_target_confidence=self.min_target_confidence,
        )
        self.yaklasma = YaklasmaGorevi(
            node, mission_topics, target_class,
            safe_stop_distance=safety_stop_distance,
            min_target_confidence=self.min_target_confidence,
        )
        self.carpma = CarpmaGorevi(
            node, mission_topics, target_class,
            min_target_confidence=self.min_target_confidence,
            impact_delta_threshold=self.impact_delta_threshold,
        )

    # --------------------------------------------------------
    def update_gps(self, lat, lon):
        """Yalnızca Pixhawk/bridge üzerinden gelen gerçek GPS fix'ini kaydeder."""
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.last_gps_time = time.monotonic()
        self._activate_with_real_pose_if_ready()

    def _activate_with_real_pose_if_ready(self):
        """Gerçek GPS ve gerçek heading birlikte gelmeden görevi başlatmaz."""
        if self.current_lat is None or self.current_lon is None or self.current_heading is None:
            return False

        if self.home_lat is None:
            self.home_lat = self.current_lat
            self.home_lon = self.current_lon
            self.logger.info(
                f"[GERÇEK VERİ] Home: {self.home_lat:.7f}, {self.home_lon:.7f}, "
                f"heading={self.current_heading:.2f}°"
            )

        self.arama.update_gps(self.current_lat, self.current_lon, self.current_heading)
        self.yaklasma.update_gps(self.current_lat, self.current_lon, self.current_heading)
        self.carpma.update_gps(self.current_lat, self.current_lon, self.current_heading)

        # Sensörler hazır olsa bile komut sistemi START göndermeden görev başlamaz.
        return True

    def start_mission(self):
        """Komut sistemi tarafından çağrılır; sensör ölçümü üretmez."""
        if self.current_lat is None or self.current_lon is None or self.current_heading is None:
            return False, "Gerçek GPS ve heading henüz hazır değil."
        self.reset_search()
        self.mission_enabled = True
        self.state = MissionState.SEARCHING
        self.logger.info("[GÖREV] Komut sisteminden START alındı; Task 3 araması başladı.")
        return True, "Task 3 başlatıldı."

    def stop_mission(self, reason="Komut sisteminden STOP komutu"):
        stop_vehicle(self.topics.cmd_vel_pub, repeat_count=2)
        self.mission_enabled = False
        self.state = MissionState.INIT
        self.arama.reset_search()
        self.yaklasma.reset_approach()
        self.carpma.reset_carpma()
        self.logger.warning(f"[GÖREV] {reason}; araç durduruldu.")

    # --------------------------------------------------------
    def update_heading(self, heading):
        """heading_callback tarafından çağrılır; bridge bunu DERECE olarak
        yayınlıyor (/cube/gps/heading). Bu topic /cube/gps'ten BAĞIMSIZ
        yayınlanıyor (bridge_node._publish_telemetry), bu yüzden üç göreve
        de DOĞRUDAN iletiyoruz — sadece update_gps()'e güvenmek, GPS geçici
        kaybolduğunda hepsinin heading'inin bayatlamasına yol açardı
        (bkz. arama.py / yaklasma.py / carpma.py update_heading)."""
        heading = float(heading) % 360.0
        self.current_heading = heading
        self.last_heading_time = time.monotonic()
        self.arama.update_heading(heading)
        self.yaklasma.update_heading(heading)
        self.carpma.update_heading(heading)
        self._activate_with_real_pose_if_ready()

    # --------------------------------------------------------
    def update_imu(self, gyro_z, accel_x, accel_y, accel_z):
        """/cube/imu callback'inden çağrılır. Yaklaşma ve çarpma
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

        if self.last_heading_time is None:
            return False
        if (now - self.last_heading_time) > HEADING_TIMEOUT_SEC:
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
    def update(self, detections, frame_id=None):
        """Ana güncelleme döngüsü.

        frame_id yalnızca yeni bir vision_callback geldiğinde değişir. Böylece
        aynı kamera mesajı 10 Hz kontrol timer'ında tekrar tekrar "farklı kare"
        olarak sayılamaz.
        """

        if not self.mission_enabled:
            # Pasif durumda hareket komutu yayimlama. Gercek STOP komutu
            # stop_mission() yolundan, kapanis ise node cleanup yolundan araci
            # acikca durdurur. Boylece gorev baslamadan Bridge'e 10 Hz sifir
            # hiz komutu gonderilmez.
            return

        gps_ok = self._check_watchdog()

        if self.bridge_connected is False:
            self.logger.error("Bridge bağlantısı kesildi; görev FAILSAFE.")
            self.state = MissionState.FAILSAFE
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=2)
            return

        if self.bridge_mode is not None and self.bridge_mode != DRIVE_MODE:
            self.logger.warn(
                f"Bridge mode={self.bridge_mode}. Beklenen mode={DRIVE_MODE}; "
                "görev komutu gönderilmiyor.",
                throttle_duration_sec=2.0,
            )
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return

        if self.bridge_armed is False:
            self.logger.warn(
                "Araç arm değil; görev komutu gönderilmiyor.",
                throttle_duration_sec=2.0,
            )
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return

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
                self.arama.update(detections, frame_id=frame_id)
                if self.arama.should_fail():
                    self.state = MissionState.FAILSAFE
                    stop_vehicle(self.topics.cmd_vel_pub, repeat_count=2)
                return
            self.state = MissionState.APPROACHING
            self.logger.info("[DURUM] Arama tamamlandı, yaklaşma aşamasına geçiliyor.")

        # ---------------- APPROACHING ----------------
        if self.state == MissionState.APPROACHING:
            approach_done = self.yaklasma.update(detections, frame_id=frame_id)

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
            carpma_done = self.carpma.update(detections, frame_id=frame_id)

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

        self.declare_parameter('carpilacak_duba', ACTIVE_TARGET_COLOR)
        self.declare_parameter('test_mode', TEST_MODE)
        self.declare_parameter('safety_stop_distance', SAFETY_STOP_DISTANCE)
        self.declare_parameter('min_target_confidence', MIN_TARGET_CONFIDENCE)
        self.declare_parameter('impact_delta_threshold', IMPACT_THRESHOLD_MPS2)
        self.declare_parameter('use_force_arm', USE_FORCE_ARM)
        self.declare_parameter('auto_start', AUTO_START_DEFAULT)

        color = self.get_parameter('carpilacak_duba').get_parameter_value().string_value
        color = color.strip().lower()
        self.test_mode = self.get_parameter('test_mode').get_parameter_value().bool_value
        self.safety_stop_distance = self.get_parameter('safety_stop_distance').get_parameter_value().double_value
        self.min_target_confidence = self.get_parameter('min_target_confidence').get_parameter_value().double_value
        self.impact_delta_threshold = self.get_parameter('impact_delta_threshold').get_parameter_value().double_value
        self.use_force_arm = self.get_parameter('use_force_arm').get_parameter_value().bool_value
        self.auto_start = self.get_parameter('auto_start').get_parameter_value().bool_value

        if not 0.0 < self.min_target_confidence <= 1.0:
            raise ValueError("min_target_confidence 0 ile 1 arasında olmalıdır.")
        if self.impact_delta_threshold <= 0.0:
            raise ValueError("impact_delta_threshold pozitif olmalıdır.")

        if color != ACTIVE_TARGET_COLOR:
            self.get_logger().error(
                f"Bu denemede yalnızca '{ACTIVE_TARGET_COLOR}' duba etkin "
                f"(girilen: '{color}'). Hedefi değiştirmek için "
                "ACTIVE_TARGET_COLOR sabitini ve YOLO model sınıfını birlikte güncelleyin."
            )
            raise SystemExit(1)

        self.target_class = ACTIVE_TARGET_CLASS
        self.get_logger().info(f"🎯 Çarpılacak duba: {self.target_class}")
        self.get_logger().info(f"🧪 Test modu: {self.test_mode}")
        self.get_logger().info(f"🛑 Durma mesafesi: {self.safety_stop_distance}m")
        self.get_logger().info(f"📷 Minimum tespit güveni: {self.min_target_confidence:.2f}")
        self.get_logger().info(f"💥 IMU temas eşiği: {self.impact_delta_threshold:.2f} m/s²")
        self.get_logger().info(f"🔐 ARM yöntemi: {'FORCE ARM' if self.use_force_arm else 'normal ARM'}")
        self.get_logger().info(f"Otomatik başlangıç: {'AÇIK' if self.auto_start else 'KAPALI'}")

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

        # create_mission_topics() (mavlink_utilities.py) bir IMU aboneliği
        # içermiyor ve bu dosya mavlink_utilities.py'yi DEĞİŞTİRMİYOR. Bu
        # yüzden bridge'in '/cube/imu' topic'ine doğrudan burada abone
        # oluyoruz. Bu abonelik olmadan carpma.py'nin IMU tabanlı çarpma
        # algılaması ve yaklasma.py'nin update_imu() çağrısı hiçbir zaman
        # veri almıyordu.
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

        # Bridge gerçek Pixhawk MIS_START parametresini Int32 /mission_start
        # mesajına çevirir. Farklı isim/türde ikinci bir sahte komut kanalı yoktur.
        self.mission_start_sub = self.create_subscription(
            Int32, '/mission_start', self.mission_start_callback, 10
        )
        self.mission_start_ack_pub = self.create_publisher(Int32, '/mission_start_ack', 10)

        self.task = Task3KamikazeEngagement(
            self,
            self.mission_topics,
            self.mission_clients,
            target_class=self.target_class,
            test_mode=self.test_mode,
            safety_stop_distance=self.safety_stop_distance,
            min_target_confidence=self.min_target_confidence,
            impact_delta_threshold=self.impact_delta_threshold,
        )

        self.current_heading = None
        self.current_detections = []
        self.current_vision_frame_id = None
        self.vision_frame_sequence = 0
        self.last_detection_time = None

        self.control_timer = self.create_timer(0.1, self.timer_callback)

        self.auto_start_attempted = False
        self.auto_start_timer = None
        if self.auto_start:
            self.auto_start_timer = self.create_timer(0.5, self._auto_start_callback)
            self.get_logger().warning(
                "[AUTO START] Etkin. Gerçek GPS, heading, kamera ve Bridge "
                "hazır olunca bir kez GUIDED + normal ARM denenecek."
            )

        if self.test_mode:
            self.status_timer = self.create_timer(5.0, self.status_callback)

        self.get_logger().info("✅ Task 3 hazır. Pixhawk/bridge /mission_start=3 komutu bekleniyor...")

    # --------------------------------------------------------
    def _publish_active_task(self):
        msg = String()
        msg.data = "task3"
        self.active_task_pub.publish(msg)

    def _arm_and_start(self):
        now = time.monotonic()
        if self.task.last_gps_time is None or self.task.last_heading_time is None:
            return False, "GPS/heading alınmadı."
        if now - self.task.last_gps_time > GPS_TIMEOUT_SEC or now - self.task.last_heading_time > HEADING_TIMEOUT_SEC:
            return False, "GPS/heading bayat."
        if self.task.bridge_connected is False:
            return False, "Orange Cube bridge bağlı değil."
        if self.task.last_vision_time is None:
            return False, "Gerçek kamera/vision verisi henüz alınmadı."
        if now - self.task.last_vision_time > VISION_TIMEOUT_SEC:
            return False, "Gerçek kamera/vision verisi bayat."
        if call_set_mode(self, self.mission_clients.set_mode_client, DRIVE_MODE) is False:
            return False, f"{DRIVE_MODE} moduna geçilemedi."
        arm_client = (
            self.mission_clients.force_arm_client
            if self.use_force_arm
            else self.mission_clients.arm_client
        )
        arm_label = "FORCE ARM" if self.use_force_arm else "ARM"
        if call_trigger_service(self, arm_client, arm_label) is False:
            return False, f"{arm_label} başarısız."
        return self.task.start_mission()

    def _auto_start_callback(self):
        """Gerçek sensörler hazır olduktan sonra yalnızca bir kez başlat."""
        if self.auto_start_attempted or self.task.mission_enabled:
            return

        now = time.monotonic()
        navigation_ready = (
            self.task.last_gps_time is not None
            and self.task.last_heading_time is not None
            and now - self.task.last_gps_time <= GPS_TIMEOUT_SEC
            and now - self.task.last_heading_time <= HEADING_TIMEOUT_SEC
        )
        vision_ready = (
            self.task.last_vision_time is not None
            and now - self.task.last_vision_time <= VISION_TIMEOUT_SEC
        )
        bridge_ready = self.task.bridge_connected is True

        if not (navigation_ready and vision_ready and bridge_ready):
            self.get_logger().info(
                "[AUTO START] Gerçek veriler bekleniyor: "
                f"gps_heading={navigation_ready}, vision={vision_ready}, "
                f"bridge={bridge_ready}",
                throttle_duration_sec=2.0,
            )
            return

        # GUIDED/ARM aşamasına bir kez girilir. Başarısız ARM'ı otomatik
        # tekrarlamak güvenli değildir; yeni deneme için süreç yeniden açılır.
        self.auto_start_attempted = True
        if self.auto_start_timer is not None:
            self.auto_start_timer.cancel()

        ok, message = self._arm_and_start()
        if ok:
            self.get_logger().info(f"[AUTO START] {message}")
        else:
            self.get_logger().error(
                f"[AUTO START] Başlatılamadı: {message}. "
                "Güvenlik için otomatik ARM tekrar denenmeyecek."
            )

    def _ack_mission_command(self, command):
        ack = Int32()
        ack.data = int(command)
        self.mission_start_ack_pub.publish(ack)

    def mission_start_callback(self, msg):
        command = int(msg.data)
        if command == 3:
            if self.task.mission_enabled:
                self._ack_mission_command(command)
                return
            ok, text = self._arm_and_start()
            (self.get_logger().info if ok else self.get_logger().error)(f"[KOMUT SİSTEMİ] {text}")
            if ok:
                self._ack_mission_command(command)
        elif command in (90, 99):
            self.task.stop_mission("Pixhawk/bridge durdurma komutu")
            if command == 99:
                disarm_ok = call_trigger_service(
                    self, self.mission_clients.disarm_client, "ACİL DISARM"
                )
                if disarm_ok is False:
                    self.get_logger().error("[ACİL DURDURMA] DISARM doğrulanamadı.")
            self._ack_mission_command(command)

    # --------------------------------------------------------
    def gps_callback(self, msg):
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        status = getattr(getattr(msg, "status", None), "status", 0)
        if status < 0 or not math.isfinite(lat) or not math.isfinite(lon):
            self.get_logger().warn("Geçersiz GPS fix yok sayıldı.", throttle_duration_sec=2.0)
            return
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            self.get_logger().warn("GPS (0,0) yok sayıldı.", throttle_duration_sec=2.0)
            return
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            self.get_logger().warn("GPS koordinatı aralık dışında.", throttle_duration_sec=2.0)
            return
        self.task.update_gps(lat, lon)

    # --------------------------------------------------------
    def heading_callback(self, msg):
        heading = float(msg.data)
        if not math.isfinite(heading):
            self.get_logger().warn("Geçersiz heading yok sayıldı.", throttle_duration_sec=2.0)
            return
        heading %= 360.0
        self.current_heading = heading
        self.task.update_heading(heading)

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
            # Öncelik vision_node'un gerçek frame_id/zaman damgasıdır.
            # Bunlar yoksa callback sıra numarası yalnızca mesajları ayırmak için kullanılır.
            source_frame_id = payload.get("frame_id", payload.get("timestamp"))
            if source_frame_id is None:
                self.vision_frame_sequence += 1
                frame_id = ("callback", self.vision_frame_sequence)
            else:
                frame_id = ("camera", str(source_frame_id))
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
            self.task.update(
                detections=self.current_detections,
                frame_id=self.current_vision_frame_id,
            )
        except Exception as exc:
            self.get_logger().error(f"Zamanlayıcı döngüsünde beklenmeyen hata: {exc}")
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub, repeat_count=1)
            except Exception as stop_exc:
                self.get_logger().error(f"Araç durdurulamadı: {stop_exc}")
            self.task.state = MissionState.FAILSAFE

    def wait_for_real_navigation_data(self, timeout_sec=30.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.task.last_gps_time is not None and self.task.last_heading_time is not None:
                return True
            self.get_logger().info(
                "Pixhawk'tan gerçek GPS ve heading bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

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
            f"retry={arama_status.get('turn_retry_count', 0)} | "
            f"Yaklaşma: {yaklasma_status.get('state', 'N/A')} "
            f"bitti={yaklasma_status.get('finished', False)} "
            f"mesafe={yaklasma_status.get('distance')} "
            f"segment={yaklasma_status.get('segment_count', 0)} | "
            f"Çarpma: {carpma_status.get('state', 'N/A')} "
            f"vuruş={carpma_status.get('hit_count', 0)}/{carpma_status.get('required_hits', 0)}"
        )

def main(args=None):
    """Node pasif başlar; yalnızca gerçek /mission_start=3 komutu ARM eder."""
    rclpy.init(args=args)
    node = None

    try:
        node = Task3Node()

        rclpy.spin(node)

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Kullanıcı tarafından durduruldu.")

    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Beklenmeyen hata: {exc}")


    finally:
        if node is not None:
            try:
                node.task.stop_mission("Node kapatılıyor")
            except Exception:
                pass

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
