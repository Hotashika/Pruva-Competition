"""
Task-3 Kamikaze Angajman Görevi — Aşama 1: ARAMA
GERÇEK HAYAT TESTİ İÇİN DÜZENLENMİŞ VERSİYON
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
# ARAMA PARAMETRELERİ (GERÇEK SAHA İÇİN AYARLANDI)
# ============================================================
SEARCH_STEP_DEG = 20.0             # Her adımda dönülecek açı
SEARCH_ANGULAR_SPEED = 0.3         # Dönüş sırasında angular_z (rad/s)
STEP_SETTLE_SEC = 1.5              # Her adımdan sonra bekleme (tespit için)
STATION_TIMEOUT_SEC = 20.0         # Bir konumda en fazla kalma süresi
MAX_SEARCH_ROTATION_DEG = 360.0    # Bir konumda toplam taranacak açı

STATION_MOVE_DISTANCE_M = 10.0     # Yeni konuma geçerken kat edilecek mesafe
STATION_MIN_SEPARATION_M = 8.0     # Yeni konum ziyaret edilenlerden uzak olmalı
RELOCATE_TOLERANCE_M = 3.0         # Yeni konuma "ulaşıldı" sayılacak mesafe

SEARCH_AREA_RADIUS_M = 80.0        # Arama alanı yarıçapı
GOLDEN_ANGLE_DEG = 137.5           # Yeni istasyon yönleri arasında iyi dağılım

# HEDEF TESPİT GÜVENİRLİK PARAMETRELERİ
MIN_CONSECUTIVE_DETECTIONS = 5     # Hedef sayılması için kaç ardışık karede görülmeli
MAX_DETECTION_GAP = 10             # Kaç kare boyunca hedef görülmezse yeniden aramaya dön
DETECTION_HISTORY_SIZE = 15        # Son kaç karelik tespit hafızada tutulacak

# GERÇEK HAYAT TESTİ İÇİN EK PARAMETRELER
TEST_MODE = True                   # Test modu aktif mi?
SIMULATE_VISION = False            # Vision simülasyonu (gerçek sensör yoksa)
MANUAL_OVERRIDE = False            # Manuel müdahale için


class SearchState(Enum):
    SCANNING = auto()      # Bulunduğu konumda 20°'lik adımlarla dönerek tarıyor
    STEP_PAUSE = auto()    # Bir adım döndükten sonra kısa bekleme (tespit için)
    RELOCATING = auto()    # Yeni bir arama konumuna ilerliyor
    TARGET_FOUND = auto()  # Hedef bulundu, arama bitti
    TARGET_LOST = auto()   # Hedef kayboldu, yeniden aramaya başlanacak


class AramaGorevi:
    def __init__(self, node, mission_topics, target_class, test_mode=False):
        """
        target_class: ZORUNLU. "red_buoy", "green_buoy" veya "black_buoy"
        test_mode: Test modu aktifse loglamalar daha detaylı olur
        """
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class
        self.test_mode = test_mode

        self.state = SearchState.SCANNING
        self.finished = False
        self.found_target = None

        # Hedef tespit geçmişi (ardışık tespit kontrolü için)
        self.detection_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.consecutive_detections = 0
        self.last_detection_frame = 0
        self.frame_counter = 0
        self.target_confirmed = False

        # Konum verileri
        self.home_lat = None
        self.home_lon = None
        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0
        self.visited_positions = []  # [(lat, lon), ...]
        self.station_index = 0

        # Tarama takibi
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None
        self.station_start_time = None
        self.step_pause_until = None

        # Yer değiştirme hedefi
        self.relocation_target = None

        # Hedef kaybı takibi
        self.target_lost_start_time = None
        self.search_retry_count = 0
        self.max_search_retries = 5

        # Test modu için ek bilgiler
        self.total_rotation_completed = 0
        self.search_start_time = None

    def update_gps(self, lat, lon, heading):
        """GPS verilerini günceller."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.visited_positions.append((lat, lon))
            self.station_start_time = time.monotonic()
            self.search_start_time = time.monotonic()
            self.logger.info(f"[ARAMA] İlk arama konumu kaydedildi: {lat:.6f}, {lon:.6f}")

            if self.test_mode:
                self.logger.info(f"[TEST] Home konumu: {lat:.6f}, {lon:.6f}")

    def _select_target(self, detections):
        """Tespitler arasından geçerli, en yakın duba adayını seçer."""
        if not detections:
            return None

        candidates = [
            d for d in detections
            if d.get("class") == self.target_class
            and d.get("distance") is not None
            and d.get("distance", -1) > 0.5  # Minimum mesafe filtresi
            and d.get("Buoy angle: ") is not None
        ]

        if self.test_mode and candidates:
            self.logger.info(f"[TEST] {len(candidates)} aday duba bulundu")
            for c in candidates:
                self.logger.info(f"[TEST] Mesafe: {c['distance']:.2f}m, Açı: {c['Buoy angle: ']:.1f}°")

        if not candidates:
            return None
        return min(candidates, key=lambda d: d["distance"])

    def _update_detection_history(self, target):
        """Hedef tespit geçmişini günceller ve ardışık tespit sayısını hesaplar."""
        self.frame_counter += 1
        self.detection_history.append(target is not None)

        if target is not None:
            self.consecutive_detections += 1
            self.last_detection_frame = self.frame_counter
            self.target_lost_start_time = None

            if self.consecutive_detections >= MIN_CONSECUTIVE_DETECTIONS:
                self.target_confirmed = True
                if self.test_mode:
                    self.logger.info(f"[TEST] Hedef {MIN_CONSECUTIVE_DETECTIONS} ardışık karede görüldü, onaylandı!")
                return True
        else:
            self.consecutive_detections = 0
            if self.target_confirmed:
                if self.target_lost_start_time is None:
                    self.target_lost_start_time = time.monotonic()
                    self.logger.warning("[ARAMA] Hedef kayboldu, yeniden tespit edilmeye çalışılıyor...")

                # Belirli süre boyunca hedef görülmezse yeniden aramaya dön
                if time.monotonic() - self.target_lost_start_time > MAX_DETECTION_GAP * 0.1:
                    self.logger.warning("[ARAMA] Hedef çok uzun süredir görülmüyor, yeniden aramaya dönülüyor...")
                    self.target_confirmed = False
                    self.state = SearchState.TARGET_LOST
                    self.finished = False
                    self.found_target = None
                    self._reset_search()
                    return False

        return self.target_confirmed and target is not None

    def _reset_search(self):
        """Aramayı sıfırla, yeniden başlangıç durumuna getir."""
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None
        self.station_start_time = time.monotonic()
        self.state = SearchState.SCANNING
        self.finished = False
        self.consecutive_detections = 0
        self.detection_history.clear()
        self.target_confirmed = False

    @staticmethod
    def _heading_diff(a, b):
        """b - a farkını [-180, 180] aralığına sararak döner (derece)."""
        diff = (b - a + 180.0) % 360.0 - 180.0
        return diff

    @staticmethod
    def _project_gps(lat, lon, bearing_deg, distance_m):
        """Verilen konumdan, bearing ve mesafeye göre yeni bir GPS noktası hesaplar."""
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
        """Home merkezli, ziyaret edilmemiş, alan sınırı içinde yeni bir durak üretir."""
        target_lat, target_lon = None, None

        for attempt in range(30):
            idx = self.station_index + attempt
            bearing = (idx * GOLDEN_ANGLE_DEG) % 360.0
            distance = min(
                STATION_MOVE_DISTANCE_M * (1 + idx * 0.3),
                SEARCH_AREA_RADIUS_M * 0.8,
            )
            target_lat, target_lon = self._project_gps(
                self.home_lat, self.home_lon, bearing, distance
            )
            if self._is_far_enough_from_visited(target_lat, target_lon):
                if self.test_mode:
                    self.logger.info(f"[TEST] Yeni istasyon: {target_lat:.6f}, {target_lon:.6f}, "
                                   f"Mesafe: {distance:.1f}m, Açı: {bearing:.1f}°")
                break

        self.station_index += 1
        return target_lat, target_lon

    def _start_relocation(self):
        self.state = SearchState.RELOCATING
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None

        target_lat, target_lon = self._next_station_target()
        self.relocation_target = (target_lat, target_lon)

        self.logger.info(
            f"[ARAMA] {STATION_TIMEOUT_SEC:.0f}sn'de hedef bulunamadı, "
            f"yeni konuma geçiliyor: {target_lat:.6f}, {target_lon:.6f}"
        )

    def _do_relocation(self):
        target_lat, target_lon = self.relocation_target
        distance = calculate_gps_distance(
            self.current_lat, self.current_lon, target_lat, target_lon
        )

        if self.test_mode and distance > 5.0:
            self.logger.info(f"[TEST] Yeni konuma ilerleniyor, mesafe: {distance:.1f}m")

        if distance < RELOCATE_TOLERANCE_M:
            self.logger.info("[ARAMA] Yeni konuma ulaşıldı, tarama yeniden başlıyor.")
            self.visited_positions.append((self.current_lat, self.current_lon))
            self.station_start_time = time.monotonic()
            self.state = SearchState.SCANNING
            return

        publish_set_position(self.topics.position_target_pub, target_lat, target_lon)

    def update(self, detections):
        """Her tick'te (0.1 sn) çağrılır.
        Returns:
            bool: True eğer hedef onaylandıysa ve yaklaşma aşamasına geçilebilirse
        """
        if self.current_lat is None:
            return False

        # 1) Hedef tespitini kontrol et ve geçmişi güncelle
        target = self._select_target(detections)
        target_visible = target is not None

        is_confirmed = self._update_detection_history(target)

        if is_confirmed and target_visible:
            self.state = SearchState.TARGET_FOUND
            self.found_target = target
            self.finished = True
            stop_vehicle(self.topics.cmd_vel_pub)

            elapsed_time = time.monotonic() - self.search_start_time if self.search_start_time else 0
            self.logger.info(
                f"[ARAMA] Hedef onaylandı! mesafe={target['distance']:.2f}m "
                f"açı={target['Buoy angle: ']:.1f}° -> YAKLAŞMA'ya geçiliyor."
            )
            if self.test_mode:
                self.logger.info(f"[TEST] Toplam arama süresi: {elapsed_time:.1f} sn, "
                               f"ziyaret edilen konum: {len(self.visited_positions)}")
            return True

        if self.state == SearchState.TARGET_LOST:
            self.logger.info("[ARAMA] Hedef kayboldu, yeniden arama başlatılıyor...")
            self.state = SearchState.SCANNING
            self.finished = False
            self.rotated_deg_this_station = 0.0
            self.step_start_heading = self.current_heading
            self.station_start_time = time.monotonic()
            return False

        # 3) Arama devam ediyor
        if not self.finished:
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
                    stop_vehicle(self.topics.cmd_vel_pub)
                    return False
                self.state = SearchState.SCANNING

            if self.step_start_heading is None:
                self.step_start_heading = self.current_heading

            rotated_now = abs(self._heading_diff(self.step_start_heading, self.current_heading))

            if rotated_now >= SEARCH_STEP_DEG:
                self.rotated_deg_this_station += rotated_now
                self.step_start_heading = None
                stop_vehicle(self.topics.cmd_vel_pub)
                self.state = SearchState.STEP_PAUSE
                self.step_pause_until = now + STEP_SETTLE_SEC

                if self.test_mode:
                    self.logger.info(f"[TEST] {SEARCH_STEP_DEG}° dönüldü, "
                                   f"toplam dönüş: {self.rotated_deg_this_station:.1f}°")

                if self.rotated_deg_this_station >= MAX_SEARCH_ROTATION_DEG:
                    self._start_relocation()
                return False

            # Dönmeye devam et
            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=SEARCH_ANGULAR_SPEED)

        return False

    def get_search_status(self):
        """Arama durum bilgilerini döndürür."""
        elapsed = time.monotonic() - self.search_start_time if self.search_start_time else 0
        return {
            "state": self.state.name,
            "finished": self.finished,
            "target_confirmed": self.target_confirmed,
            "consecutive_detections": self.consecutive_detections,
            "rotated_deg": self.rotated_deg_this_station,
            "visited_positions": len(self.visited_positions),
            "search_retry_count": self.search_retry_count,
            "elapsed_time": elapsed,
            "current_position": (self.current_lat, self.current_lon),
            "target_class": self.target_class
        }

    def reset_search(self):
        """Arama görevini tamamen sıfırlar (dışarıdan çağrılabilir)."""
        self._reset_search()
        self.search_start_time = time.monotonic()
        self.visited_positions = [(self.current_lat, self.current_lon)] if self.current_lat else []
        self.logger.info("[ARAMA] Arama tamamen sıfırlandı, yeniden başlıyor.")