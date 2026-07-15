"""
Task-3 Kamikaze Angajman Görevi — Aşama 1: ARAMA
GERÇEK HAYAT TESTİ İÇİN DÜZELTİLMİŞ VERSİYON (v4)

v3'ten (senin yazdığın) devralınan, bridge_node.py ile uyumlu doğru
tasarım kararları:
  - angular_z, bridge'de "mevcut yaw'a eklenecek radyan ofset"tir
    (target_yaw_rad = bridge.yaw + angular_z). Bu yüzden SCANNING sırasında
    sabit bir angular_z göndermek yerine, her tick'te küçük bir delta
    (ANGULAR_STEP_DEG_PER_TICK) hesaplanıp gönderiliyor.
  - Heading birimi: bridge /cube/gps/heading'i DERECE yayınlıyor
    (GLOBAL_POSITION_INT.hdg/100 veya VFR_HUD.heading) — tüm iç hesaplar
    derece cinsinde tutuluyor, sadece bridge'e giden angular_z radyana
    çevriliyor.
  - stop_vehicle(..., repeat_count=1): repeat_count=10 ile her çağrıda
    art arda 10 mesaj göndermenin fonksiyonel bir kazancı yok (bridge zaten
    son mesajı işler), sadece gereksiz topic trafiği yaratıyor — SEARCHING
    döngüsünde her tick çağrıldığından bu trafik anlamlı büyüyor.

v4'te EKLENEN/DÜZELTİLEN:
  - Tespit onay penceresi artık TICK sayısı değil GERÇEK ZAMAN
    (DETECTION_WINDOW_SEC) bazlı. Önceki tick-tabanlı pencere, vision
    kontrol döngüsünden (10Hz) daha yavaş geldiğinde (ör. 3Hz kamera)
    neredeyse hiç onay veremiyordu — bunu kendi yazdığımız simülasyon
    testinde (low_vision_rate senaryosu) yakaladık.
  - "Hedef kayboldu" kontrolü de tick sayacı yerine gerçek zaman
    (MAX_DETECTION_GAP_SEC) kullanıyor, aynı nedenle.
"""

import math
import time
from enum import Enum, auto
from collections import deque

from utils.mavlink_utilities import (
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    calculate_gps_distance,
)

# ============================================================
# ARAMA PARAMETRELERİ
# ============================================================
SEARCH_STEP_DEG = 20.0             # Her adımda dönülecek açı (derece)
SEARCH_ANGULAR_SPEED_DEG_S = 30.0  # Hedeflenen dönüş hızı (derece/saniye)

TICK_RATE_HZ = 10.0                # update() çağrı frekansı — bridge loop 0.1sn
ANGULAR_STEP_DEG_PER_TICK = SEARCH_ANGULAR_SPEED_DEG_S / TICK_RATE_HZ
# Her tick'te yaw hedefine eklenecek delta (derece). Bridge bunu
# math.radians(delta) olarak yaw_rad'a ekler → kararlı artımlı dönüş.

STEP_SETTLE_SEC = 1.5              # Her adımdan sonra bekleme (tespit için)
STEP_MAX_DURATION_SEC = 6.0        # Dönüş adımı için üst süre sınırı
STATION_TIMEOUT_SEC = 20.0         # Bir konumda en fazla kalma süresi
MAX_SEARCH_ROTATION_DEG = 360.0    # Bir konumda toplam taranacak açı

STATION_MOVE_DISTANCE_M = 10.0
STATION_MIN_SEPARATION_M = 8.0
RELOCATE_TOLERANCE_M = 3.0

SEARCH_AREA_RADIUS_M = 80.0
SEARCH_AREA_EXPAND_FACTOR = 1.3
GOLDEN_ANGLE_DEG = 137.5

# HEDEF TESPİT GÜVENİRLİK PARAMETRELERİ
# DÜZELTME (v4): tick sayısı yerine gerçek zaman penceresi. Vision'ın
# kontrol döngüsünden (10Hz) daha yavaş gelmesi (ör. 3Hz kamera) gerçek
# hayatta çok olası; tick-tabanlı pencerede bu durumda MIN_CONSECUTIVE_
# DETECTIONS'a neredeyse hiç ulaşılamıyordu.
MIN_CONSECUTIVE_DETECTIONS = 5
# NOT: bu değeri sahadaki gerçek kamera FPS'ine göre kalibre edin.
# Kural of thumb: DETECTION_WINDOW_SEC >= MIN_CONSECUTIVE_DETECTIONS /
# (beklenen_vision_fps * (1 - beklenen_kaçırma_oranı)). Örn. 3Hz kamera,
# %30 kaçırma toleransı için: 5 / (3*0.7) ≈ 2.4sn -> 2.5sn güvenli pay.
DETECTION_WINDOW_SEC = 2.5         # Bu süre içinde en az N pozitif tespit ara
DETECTION_HISTORY_MAXLEN = 200     # Bellek için güvenlik üst sınırı (tick değil)
MAX_DETECTION_GAP_SEC = 1.5        # Onaylı hedef bu süre görülmezse TARGET_LOST

MAX_SEARCH_RETRIES = 5


class SearchState(Enum):
    SCANNING = auto()
    STEP_PAUSE = auto()
    RELOCATING = auto()
    TARGET_FOUND = auto()
    TARGET_LOST = auto()


class AramaGorevi:
    def __init__(self, node, mission_topics, target_class, test_mode=False):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class
        self.test_mode = test_mode

        self.state = SearchState.SCANNING
        self.finished = False
        self.found_target = None

        # (zaman_damgası, bool) çiftleri — pencere GERÇEK ZAMANA göre kırpılır
        self.detection_history = deque(maxlen=DETECTION_HISTORY_MAXLEN)
        self.last_detection_time = None
        self.target_confirmed = False

        self.home_lat = None
        self.home_lon = None
        self.current_lat = None
        self.current_lon = None

        # Bridge heading_deg yayınlıyor (DERECE, 0-360). Tüm iç
        # hesaplamalar derece cinsinden yapılıyor.
        self.current_heading_deg = 0.0

        self.visited_positions = []
        self.station_index = 0

        self.rotated_deg_this_station = 0.0
        self.step_start_heading_deg = None
        self.step_start_time = None
        self.station_start_time = None
        self.step_pause_until = None

        # Hedeflenen yaw (derece). Dönüş sırasında her tick artırılır,
        # her yeni adımda mevcut heading'e sıfırlanır (runaway'i önler).
        self.target_heading_deg = None

        self.relocation_target = None

        self.target_lost_start_time = None
        self.search_retry_count = 0
        self.max_search_retries = MAX_SEARCH_RETRIES
        self.current_search_radius = SEARCH_AREA_RADIUS_M

        self.total_rotation_completed = 0.0
        self.search_start_time = None

    # ----------------------------------------------------------
    def update_gps(self, lat, lon, heading_deg):
        """heading_deg: bridge'in /cube/gps/heading topic'inden gelen
        DERECE cinsinden yön (0-360, kuzey=0, saat yönü pozitif)."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading_deg = heading_deg

        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.visited_positions.append((lat, lon))
            self.station_start_time = time.monotonic()
            self.search_start_time = time.monotonic()
            self.target_heading_deg = heading_deg
            self.logger.info(
                f"[ARAMA] İlk konum: {lat:.6f}, {lon:.6f}, Heading: {heading_deg:.1f}°"
            )

    # ----------------------------------------------------------
    def _select_target(self, detections):
        if not detections:
            return None

        candidates = [
            d for d in detections
            if d.get("class") == self.target_class
            and d.get("distance") is not None
            and d.get("distance", -1) > 0.5
            and d.get("Buoy angle: ") is not None
        ]

        if self.test_mode and candidates:
            self.logger.info(f"[TEST] {len(candidates)} aday tespit:")
            for c in candidates:
                self.logger.info(f"  Mesafe: {c['distance']:.2f}m, Açı: {c['Buoy angle: ']:.1f}°")

        return min(candidates, key=lambda d: d["distance"]) if candidates else None

    # ----------------------------------------------------------
    def _update_detection_history(self, target):
        """Pencere-içi (windowed) onay — DÜZELTME (v4): artık gerçek zaman
        bazlı. Eskiden 'son N tick' kullanılıyordu; vision kontrol
        döngüsünden yavaşsa (örn. 3Hz kamera / 10Hz kontrol) bu pencere
        yeterli sayıda gerçek kareyi kapsamıyordu ve onay neredeyse hiç
        gerçekleşmiyordu."""
        now = time.monotonic()
        self.detection_history.append((now, target is not None))
        while self.detection_history and (now - self.detection_history[0][0]) > DETECTION_WINDOW_SEC:
            self.detection_history.popleft()

        positive_count = sum(1 for _, v in self.detection_history if v)

        if target is not None:
            self.last_detection_time = now
            self.target_lost_start_time = None

            if not self.target_confirmed and positive_count >= MIN_CONSECUTIVE_DETECTIONS:
                self.target_confirmed = True
                if self.test_mode:
                    self.logger.info(
                        f"[TEST] Son {DETECTION_WINDOW_SEC:.1f}sn'de {positive_count} "
                        f"pozitif tespit → Hedef onaylandı!"
                    )
            return self.target_confirmed

        else:
            if self.target_confirmed and self.last_detection_time is not None:
                gap = now - self.last_detection_time
                if gap >= MAX_DETECTION_GAP_SEC:
                    self.logger.warning(
                        f"[ARAMA] Hedef {gap:.1f}sn görülmedi, TARGET_LOST."
                    )
                    self.target_confirmed = False
                    self.state = SearchState.TARGET_LOST
                    self.finished = False
                    self.found_target = None
                    self._register_search_retry()
                    return False
            return False

    def _register_search_retry(self):
        self.search_retry_count += 1
        if self.search_retry_count % self.max_search_retries == 0:
            self.current_search_radius = min(
                self.current_search_radius * SEARCH_AREA_EXPAND_FACTOR,
                SEARCH_AREA_RADIUS_M * 1.8,  # geofence(150m) aşımını önlemek için üst sınır
            )
            self.logger.warning(
                f"[ARAMA] {self.search_retry_count} kez hedef kaybedildi. "
                f"Yarıçap → {self.current_search_radius:.1f}m"
            )

    def _reset_search(self):
        self.rotated_deg_this_station = 0.0
        self.step_start_heading_deg = None
        self.step_start_time = None
        self.station_start_time = time.monotonic()
        self.state = SearchState.SCANNING
        self.finished = False
        self.detection_history.clear()
        self.last_detection_time = None
        self.target_confirmed = False
        self.target_heading_deg = self.current_heading_deg

    # ----------------------------------------------------------
    @staticmethod
    def _heading_diff_deg(a_deg, b_deg):
        """b - a farkını [-180, 180] aralığına sarar."""
        return (b_deg - a_deg + 180.0) % 360.0 - 180.0

    # ----------------------------------------------------------
    def _send_rotation_command(self):
        """
        Bridge semantiği: target_yaw_rad = bridge.yaw + angular_z

        target_heading_deg her tick'te ANGULAR_STEP_DEG_PER_TICK kadar
        ilerletilir; gönderilen açı her zaman GERÇEK mevcut heading'e göre
        hesaplanan bir delta'dır (bridge.yaw ile arama'nın current_heading_deg
        değeri arasında küçük bir gecikme/fark olabilir ama pratikte ihmal
        edilebilir düzeydedir — USV'lerde roll/pitch minimal olduğundan
        ATTITUDE.yaw ile GPS/kompas heading birbirine yakın seyreder).
        """
        if self.target_heading_deg is None:
            self.target_heading_deg = self.current_heading_deg

        self.target_heading_deg = (self.target_heading_deg + ANGULAR_STEP_DEG_PER_TICK) % 360.0

        delta_rad = math.radians(
            self._heading_diff_deg(self.current_heading_deg, self.target_heading_deg)
        )
        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=delta_rad)

    # ----------------------------------------------------------
    @staticmethod
    def _project_gps(lat, lon, bearing_deg, distance_m):
        R = 6378137.0
        bearing_rad = math.radians(bearing_deg)
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ang_dist = distance_m / R

        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(ang_dist)
            + math.cos(lat_rad) * math.sin(ang_dist) * math.cos(bearing_rad)
        )
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(ang_dist) * math.cos(lat_rad),
            math.cos(ang_dist) - math.sin(lat_rad) * math.sin(new_lat_rad),
        )
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

    def _is_far_enough_from_visited(self, lat, lon):
        for vlat, vlon in self.visited_positions:
            if calculate_gps_distance(lat, lon, vlat, vlon) < STATION_MIN_SEPARATION_M:
                return False
        return True

    def _next_station_target(self):
        target_lat, target_lon = None, None
        for attempt in range(30):
            idx = self.station_index + attempt
            bearing = (idx * GOLDEN_ANGLE_DEG) % 360.0
            distance = min(
                STATION_MOVE_DISTANCE_M * (1 + idx * 0.3),
                self.current_search_radius * 0.8,
            )
            target_lat, target_lon = self._project_gps(self.home_lat, self.home_lon, bearing, distance)
            if self._is_far_enough_from_visited(target_lat, target_lon):
                if self.test_mode:
                    self.logger.info(
                        f"[TEST] Yeni istasyon: {target_lat:.6f}, {target_lon:.6f} "
                        f"({distance:.1f}m, {bearing:.1f}°)"
                    )
                break
        self.station_index += 1
        return target_lat, target_lon

    def _start_relocation(self):
        self.state = SearchState.RELOCATING
        self.rotated_deg_this_station = 0.0
        self.step_start_heading_deg = None
        self.step_start_time = None

        target_lat, target_lon = self._next_station_target()
        self.relocation_target = (target_lat, target_lon)
        self.logger.info(
            f"[ARAMA] {STATION_TIMEOUT_SEC:.0f}sn'de bulunamadı, "
            f"yeni konum: {target_lat:.6f}, {target_lon:.6f}"
        )

    def _do_relocation(self):
        target_lat, target_lon = self.relocation_target
        distance = calculate_gps_distance(self.current_lat, self.current_lon, target_lat, target_lon)

        if self.test_mode and distance > 5.0:
            self.logger.info(f"[TEST] Hedefe: {distance:.1f}m kaldı")

        if distance < RELOCATE_TOLERANCE_M:
            self.logger.info("[ARAMA] Yeni konuma ulaşıldı, tarama başlıyor.")
            self.visited_positions.append((self.current_lat, self.current_lon))
            self.station_start_time = time.monotonic()
            self.target_heading_deg = self.current_heading_deg
            self.state = SearchState.SCANNING
            return

        # NOT: bridge, /cube/set_position'a yakın zamanda (son 0.5sn
        # içinde) bir mesaj geldiyse cmd_vel/attitude komutlarını
        # yayınlamayı DURDURUR (bkz. _send_attitude_target_loop'taki
        # position_target_timeout_sec kontrolü). Bu yüzden bu fonksiyon
        # her tick çağrılıp waypoint'i tazelemeli — aksi halde bridge
        # 0.5sn sonra eski/stale attitude komutuna geri döner.
        publish_set_position(self.topics.position_target_pub, target_lat, target_lon)

    # ----------------------------------------------------------
    def update(self, detections):
        """
        Returns:
            True  → Hedef onaylandı, yaklaşma aşamasına geç.
            False → Arama devam ediyor.
        """
        if self.current_lat is None:
            return False

        target = self._select_target(detections)
        is_confirmed = self._update_detection_history(target)

        if is_confirmed and target is not None:
            self.state = SearchState.TARGET_FOUND
            self.found_target = target
            self.finished = True
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)

            elapsed = time.monotonic() - self.search_start_time if self.search_start_time else 0
            self.logger.info(
                f"[ARAMA] Hedef onaylandı! mesafe={target['distance']:.2f}m "
                f"açı={target['Buoy angle: ']:.1f}° → YAKLAŞMA"
            )
            if self.test_mode:
                self.logger.info(f"[TEST] Süre: {elapsed:.1f}sn, Ziyaret: {len(self.visited_positions)} konum")
            return True

        if self.state == SearchState.TARGET_LOST:
            self.logger.warning("[ARAMA] Hedef kayboldu, yeniden arama başlıyor...")
            self._reset_search()
            return False

        if self.finished:
            return False

        now = time.monotonic()
        if self.station_start_time is None:
            self.station_start_time = now

        elapsed = now - self.station_start_time
        if self.state != SearchState.RELOCATING and elapsed > STATION_TIMEOUT_SEC:
            self._start_relocation()

        if self.state == SearchState.RELOCATING:
            self._do_relocation()
            return False

        if self.state == SearchState.STEP_PAUSE:
            if self.step_pause_until is not None and now < self.step_pause_until:
                stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
                return False
            self.state = SearchState.SCANNING

        if self.step_start_heading_deg is None:
            self.step_start_heading_deg = self.current_heading_deg
            self.step_start_time = now
            self.target_heading_deg = self.current_heading_deg  # bu adım için sıfırla

        rotated_now = abs(self._heading_diff_deg(self.step_start_heading_deg, self.current_heading_deg))
        step_elapsed = now - self.step_start_time if self.step_start_time else 0.0

        step_done = rotated_now >= SEARCH_STEP_DEG
        step_timed_out = step_elapsed >= STEP_MAX_DURATION_SEC

        if step_done or step_timed_out:
            if step_timed_out and not step_done:
                self.logger.warning(
                    f"[ARAMA] Adım {STEP_MAX_DURATION_SEC:.1f}sn'de tamamlanamadı "
                    f"(heading donmuş olabilir), zorla ilerletiliyor."
                )
            increment = max(rotated_now, SEARCH_STEP_DEG if step_timed_out else 0.0)
            self.rotated_deg_this_station += increment
            self.total_rotation_completed += rotated_now
            self.step_start_heading_deg = None
            self.step_start_time = None

            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.state = SearchState.STEP_PAUSE
            self.step_pause_until = now + STEP_SETTLE_SEC

            if self.test_mode:
                self.logger.info(
                    f"[TEST] Adım tamamlandı: {rotated_now:.1f}° | "
                    f"Toplam: {self.rotated_deg_this_station:.1f}°"
                )

            if self.rotated_deg_this_station >= MAX_SEARCH_ROTATION_DEG:
                self._start_relocation()
            return False

        self._send_rotation_command()
        return False

    # ----------------------------------------------------------
    def get_search_status(self):
        elapsed = time.monotonic() - self.search_start_time if self.search_start_time else 0
        positive_count = sum(1 for _, v in self.detection_history if v)
        return {
            "state": self.state.name,
            "finished": self.finished,
            "target_confirmed": self.target_confirmed,
            "positive_in_window": positive_count,
            "window_sec": DETECTION_WINDOW_SEC,
            "rotated_deg": self.rotated_deg_this_station,
            "total_rotation_completed": self.total_rotation_completed,
            "target_heading_deg": self.target_heading_deg,
            "current_heading_deg": self.current_heading_deg,
            "visited_positions": len(self.visited_positions),
            "search_retry_count": self.search_retry_count,
            "current_search_radius": self.current_search_radius,
            "elapsed_time": elapsed,
            "current_position": (self.current_lat, self.current_lon),
            "target_class": self.target_class,
        }

    def reset_search(self):
        """Dışarıdan çağrılabilir tam sıfırlama. visited_positions
        korunuyor — tekrar aynı yerleri taramasın."""
        self._reset_search()
        self.search_start_time = time.monotonic()
        self.logger.info("[ARAMA] Arama sıfırlandı, yeniden başlıyor.")