#!/usr/bin/env python3
"""
ARAMA MODÜLÜ TEST SİMÜLASYONU - ROS2 BAĞIMSIZ
Gerçek donanım ve ROS2 olmadan arama algoritmasını test etmek için

Kullanım:
    python3 test_arama_simulation.py
    python3 test_arama_simulation.py --target red --test-mode
"""

import sys
import time
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================
# ROS2 BAĞIMSIZLIK İÇİN MOCK SINIFLAR
# ============================================================

class MockLogger:
    """ROS2 logger mock."""

    def info(self, msg, *args, **kwargs):
        print(f"[INFO] {msg}")

    def warn(self, msg, *args, **kwargs):
        print(f"[WARN] {msg}")

    def error(self, msg, *args, **kwargs):
        print(f"[ERROR] {msg}")

    def debug(self, msg, *args, **kwargs):
        if args and args[0] == 'throttle_duration_sec':
            pass  # Throttle parametresini yoksay
        print(f"[DEBUG] {msg}")


class MockNode:
    """ROS2 node mock."""

    def __init__(self):
        self.logger = MockLogger()

    def get_logger(self):
        return self.logger


class MockTopics:
    """Mission topics mock."""

    def __init__(self):
        self.cmd_vel_pub = MockPublisher()
        self.position_target_pub = MockPublisher()


class MockPublisher:
    """ROS2 publisher mock."""

    def __init__(self):
        self.last_msg = None
        self.messages = []

    def publish(self, msg):
        self.last_msg = msg
        self.messages.append(msg)
        return msg


# ============================================================
# UTILITY FONKSİYONLAR (MAVLINK UTILITIES'DEN BAĞIMSIZ)
# ============================================================

def calculate_gps_distance(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki mesafeyi hesapla (haversine)."""
    R = 6371000  # Dünya yarıçapı (metre)

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * \
        math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def project_gps(lat, lon, bearing_deg, distance_m):
    """GPS noktasını bearing ve mesafeye göre projekte et."""
    R = 6378137.0
    bearing_rad = math.radians(bearing_deg)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ang_dist = distance_m / R

    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(ang_dist) +
        math.cos(lat_rad) * math.sin(ang_dist) * math.cos(bearing_rad)
    )
    new_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(ang_dist) * math.cos(lat_rad),
        math.cos(ang_dist) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)


# ============================================================
# ARAMA MODÜLÜ (BAĞIMSIZ KOPYA)
# ============================================================

class SearchState(Enum):
    SCANNING = 1
    STEP_PAUSE = 2
    RELOCATING = 3
    TARGET_FOUND = 4
    TARGET_LOST = 5


class AramaGorevi:
    """ARAMA modülü - ROS2 bağımsız versiyon."""

    def __init__(self, node, mission_topics, target_class, test_mode=False):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class
        self.test_mode = test_mode

        # ARAMA PARAMETRELERİ
        self.SEARCH_STEP_DEG = 20.0
        self.SEARCH_ANGULAR_SPEED = 0.3
        self.STEP_SETTLE_SEC = 1.2
        self.STATION_TIMEOUT_SEC = 18.0
        self.MAX_SEARCH_ROTATION_DEG = 360.0
        self.STATION_MOVE_DISTANCE_M = 8.0
        self.STATION_MIN_SEPARATION_M = 5.0
        self.RELOCATE_TOLERANCE_M = 2.0
        self.SEARCH_AREA_RADIUS_M = 60.0
        self.GOLDEN_ANGLE_DEG = 137.5

        # HEDEF TESPİT GÜVENİRLİK PARAMETRELERİ
        self.MIN_CONSECUTIVE_DETECTIONS = 3
        self.MAX_DETECTION_GAP = 5
        self.DETECTION_HISTORY_SIZE = 10

        # Durum
        self.state = SearchState.SCANNING
        self.finished = False
        self.found_target = None

        # Hedef tespit geçmişi
        self.detection_history = []
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
        self.visited_positions = []
        self.station_index = 0

        # Tarama takibi
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None
        self.station_start_time = None
        self.step_pause_until = None
        self.relocation_target = None
        self.target_lost_start_time = None

        # İstatistikler
        self.search_start_time = time.monotonic()
        self.total_rotation = 0.0

        self.logger.info(f"[ARAMA] Başlatıldı, hedef: {target_class}")

    # --------------------------------------------------------
    def update_gps(self, lat, lon, heading):
        """GPS verilerini güncelle."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.visited_positions.append((lat, lon))
            self.station_start_time = time.monotonic()
            self.logger.info(f"[ARAMA] İlk konum: {lat:.6f}, {lon:.6f}")

    # --------------------------------------------------------
    def _select_target(self, detections):
        """Tespitler arasından hedefi seç."""
        if not detections:
            return None

        candidates = [
            d for d in detections
            if d.get("class") == self.target_class
               and d.get("distance") is not None
               and d.get("distance", -1) > 0
               and d.get("Buoy angle: ") is not None
        ]

        if not candidates:
            return None
        return min(candidates, key=lambda d: d["distance"])

    # --------------------------------------------------------
    def _update_detection_history(self, target):
        """Tespit geçmişini güncelle."""
        self.frame_counter += 1
        self.detection_history.append(target is not None)

        if len(self.detection_history) > self.DETECTION_HISTORY_SIZE:
            self.detection_history.pop(0)

        if target is not None:
            self.consecutive_detections += 1
            self.last_detection_frame = self.frame_counter
            self.target_lost_start_time = None

            if self.consecutive_detections >= self.MIN_CONSECUTIVE_DETECTIONS:
                self.target_confirmed = True
                return True
        else:
            self.consecutive_detections = 0
            if self.target_confirmed:
                if self.target_lost_start_time is None:
                    self.target_lost_start_time = time.monotonic()

                if time.monotonic() - self.target_lost_start_time > self.MAX_DETECTION_GAP * 0.1:
                    self.target_confirmed = False
                    self.state = SearchState.TARGET_LOST
                    self.finished = False
                    self.found_target = None
                    self._reset_search()
                    return False

        return self.target_confirmed and target is not None

    # --------------------------------------------------------
    def _reset_search(self):
        """Aramayı sıfırla."""
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None
        self.station_start_time = time.monotonic()
        self.state = SearchState.SCANNING
        self.finished = False
        self.consecutive_detections = 0
        self.target_confirmed = False

    # --------------------------------------------------------
    def _heading_diff(self, a, b):
        """İki açı arasındaki fark."""
        diff = (b - a + 180.0) % 360.0 - 180.0
        return diff

    # --------------------------------------------------------
    def _is_far_enough_from_visited(self, lat, lon):
        """Ziyaret edilen konumlardan uzak mı?"""
        for vlat, vlon in self.visited_positions:
            if calculate_gps_distance(lat, lon, vlat, vlon) < self.STATION_MIN_SEPARATION_M:
                return False
        return True

    # --------------------------------------------------------
    def _next_station_target(self):
        """Yeni istasyon hedefi üret."""
        target_lat, target_lon = None, None

        for attempt in range(20):
            idx = self.station_index + attempt
            bearing = (idx * self.GOLDEN_ANGLE_DEG) % 360.0
            distance = min(
                self.STATION_MOVE_DISTANCE_M * (1 + idx * 0.4),
                self.SEARCH_AREA_RADIUS_M * 0.9,
            )
            target_lat, target_lon = project_gps(
                self.home_lat, self.home_lon, bearing, distance
            )
            if self._is_far_enough_from_visited(target_lat, target_lon):
                break

        self.station_index += 1
        return target_lat, target_lon

    # --------------------------------------------------------
    def _start_relocation(self):
        """Yeni konuma geçiş."""
        self.state = SearchState.RELOCATING
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None

        target_lat, target_lon = self._next_station_target()
        self.relocation_target = (target_lat, target_lon)

        self.logger.info(f"[ARAMA] Yeni konuma geçiliyor: {target_lat:.6f}, {target_lon:.6f}")

    # --------------------------------------------------------
    def _do_relocation(self):
        """Yeni konuma git."""
        target_lat, target_lon = self.relocation_target
        distance = calculate_gps_distance(
            self.current_lat, self.current_lon, target_lat, target_lon
        )

        if distance < self.RELOCATE_TOLERANCE_M:
            self.logger.info("[ARAMA] Yeni konuma ulaşıldı.")
            self.visited_positions.append((self.current_lat, self.current_lon))
            self.station_start_time = time.monotonic()
            self.state = SearchState.SCANNING
            return

        # Mock: konumu güncelle (gerçekte publish_set_position yapılır)
        self.logger.debug(f"[ARAMA] İlerleniyor, mesafe: {distance:.1f}m")

    # --------------------------------------------------------
    def _publish_cmd_vel(self, linear_x, angular_z):
        """Hız komutu yayınla."""
        self.topics.cmd_vel_pub.publish({'linear_x': linear_x, 'angular_z': angular_z})

        # GPS'i güncelle (simülasyon için)
        if abs(angular_z) > 0.01:
            self.total_rotation += abs(angular_z) * 0.1
        if abs(linear_x) > 0.01:
            # Konumu güncelle
            distance = linear_x * 0.1
            new_lat, new_lon = project_gps(
                self.current_lat, self.current_lon,
                self.current_heading, distance
            )
            self.current_lat = new_lat
            self.current_lon = new_lon
            self.current_heading += angular_z * 0.1
            self.current_heading %= 360.0

    # --------------------------------------------------------
    def update(self, detections):
        """Ana güncelleme döngüsü."""
        if self.current_lat is None:
            return False

        # Hedef tespiti
        target = self._select_target(detections)
        is_confirmed = self._update_detection_history(target)

        if is_confirmed and target is not None:
            self.state = SearchState.TARGET_FOUND
            self.found_target = target
            self.finished = True
            self._publish_cmd_vel(0.0, 0.0)
            self.logger.info(f"[ARAMA] ✅ HEDEF BULUNDU! Mesafe: {target['distance']:.2f}m")
            return True

        if self.state == SearchState.TARGET_LOST:
            self.logger.info("[ARAMA] Hedef kayboldu, yeniden arama başlıyor...")
            self.state = SearchState.SCANNING
            self.finished = False
            self.step_start_heading = self.current_heading
            self.station_start_time = time.monotonic()
            return False

        if not self.finished:
            now = time.monotonic()
            if self.station_start_time is None:
                self.station_start_time = now

            # Zaman aşımı
            elapsed = now - self.station_start_time
            if self.state != SearchState.RELOCATING and elapsed > self.STATION_TIMEOUT_SEC:
                self._start_relocation()

            # Yer değiştirme
            if self.state == SearchState.RELOCATING:
                self._do_relocation()
                return False

            # Adım bekleme
            if self.state == SearchState.STEP_PAUSE:
                if self.step_pause_until is not None and now < self.step_pause_until:
                    self._publish_cmd_vel(0.0, 0.0)
                    return False
                self.state = SearchState.SCANNING

            # Tarama
            if self.step_start_heading is None:
                self.step_start_heading = self.current_heading

            rotated_now = abs(self._heading_diff(self.step_start_heading, self.current_heading))

            if rotated_now >= self.SEARCH_STEP_DEG:
                self.rotated_deg_this_station += rotated_now
                self.step_start_heading = None
                self._publish_cmd_vel(0.0, 0.0)
                self.state = SearchState.STEP_PAUSE
                self.step_pause_until = now + self.STEP_SETTLE_SEC

                if self.rotated_deg_this_station >= self.MAX_SEARCH_ROTATION_DEG:
                    self._start_relocation()
                return False

            # Dönmeye devam
            self._publish_cmd_vel(0.0, self.SEARCH_ANGULAR_SPEED)

        return False

    def get_search_status(self):
        """Durum bilgilerini döndür."""
        elapsed = time.monotonic() - self.search_start_time
        return {
            "state": self.state.name if hasattr(self.state, 'name') else str(self.state),
            "finished": self.finished,
            "target_confirmed": self.target_confirmed,
            "consecutive_detections": self.consecutive_detections,
            "rotated_deg": self.rotated_deg_this_station,
            "visited_positions": len(self.visited_positions),
            "elapsed_time": elapsed,
            "current_position": (self.current_lat, self.current_lon),
            "target_class": self.target_class,
            "total_rotation": self.total_rotation
        }


# ============================================================
# SİMÜLASYON SINIFLARI
# ============================================================

@dataclass
class SimConfig:
    """Simülasyon konfigürasyonu."""
    target_class: str = "red_buoy"
    test_mode: bool = True
    home_lat: float = 40.0
    home_lon: float = 30.0
    search_area_radius: float = 50.0
    simulation_duration: float = 120.0
    noise_level: float = 0.1
    target_visible: bool = True
    target_distance: float = 20.0
    vision_update_rate: float = 0.1


class GPSSimulator:
    """GPS simülatörü."""

    def __init__(self, home_lat, home_lon, noise_level=0.1):
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.noise_level = noise_level
        self.current_lat = home_lat
        self.current_lon = home_lon
        self.current_heading = 0.0
        self.total_distance = 0.0
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0

    def update_from_commands(self, linear_x, angular_z, dt=0.1):
        """Komutlardan GPS'i güncelle."""
        self.last_linear_x = linear_x
        self.last_angular_z = angular_z

        self.current_heading += angular_z * dt
        self.current_heading %= 360.0

        if abs(linear_x) > 0.001:
            distance = linear_x * dt
            self.total_distance += abs(distance)

            new_lat, new_lon = project_gps(
                self.current_lat, self.current_lon,
                self.current_heading, distance
            )
            self.current_lat = new_lat
            self.current_lon = new_lon

        # Gürültü
        if self.noise_level > 0:
            self.current_lat += random.uniform(-self.noise_level, self.noise_level) * 1e-6
            self.current_lon += random.uniform(-self.noise_level, self.noise_level) * 1e-6

        return self.get_gps()

    def get_gps(self):
        return {
            'lat': self.current_lat,
            'lon': self.current_lon,
            'heading': self.current_heading
        }

    def get_distance_to(self, lat, lon):
        return calculate_gps_distance(self.current_lat, self.current_lon, lat, lon)


class VisionSimulator:
    """Vision simülatörü."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.target_class = config.target_class
        self.detection_range = 40.0
        self.field_of_view = 60.0
        self.noise_level = config.noise_level
        self.target_visible = config.target_visible

        # Hedef konumunu belirle
        self.target_lat = None
        self.target_lon = None
        self._setup_target()

        self.detections_sent = 0
        self.total_attempts = 0

    def _setup_target(self):
        """Hedef konumunu belirle."""
        angle = random.uniform(0, 360)
        distance = random.uniform(10, self.config.search_area_radius)
        self.target_lat, self.target_lon = project_gps(
            self.config.home_lat,
            self.config.home_lon,
            angle,
            distance
        )

    def update(self, gps: dict) -> List[Dict]:
        """Tespitleri üret."""
        self.total_attempts += 1

        if not self.target_visible:
            return []

        distance = calculate_gps_distance(
            gps['lat'], gps['lon'],
            self.target_lat, self.target_lon
        )

        if distance > self.detection_range:
            return []

        # Hedef açısı
        bearing = self._calculate_bearing(
            gps['lat'], gps['lon'],
            self.target_lat, self.target_lon
        )
        angle = (bearing - gps['heading']) % 360.0
        if angle > 180:
            angle -= 360

        if abs(angle) > self.field_of_view / 2:
            return []

        # Gürültü
        if self.noise_level > 0:
            distance += random.uniform(-self.noise_level, self.noise_level)
            angle += random.uniform(-2, 2)

        distance = max(0.5, distance)

        detection = {
            'class': self.target_class,
            'distance': distance,
            'Buoy angle: ': angle,
            'confidence': random.uniform(0.7, 1.0)
        }

        self.detections_sent += 1
        return [detection]

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """İki nokta arasındaki bearing."""
        dlon = math.radians(lon2 - lon1)
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)

        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360


# ============================================================
# ANA TEST SINIFI
# ============================================================

class AramaTestSimulator:
    """Ana test simülatörü."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.node = MockNode()
        self.topics = MockTopics()

        # Simülasyon nesneleri
        self.gps = GPSSimulator(
            config.home_lat,
            config.home_lon,
            config.noise_level
        )
        self.vision = VisionSimulator(config)

        # Arama görevi
        self.arama = AramaGorevi(
            self.node,
            self.topics,
            config.target_class,
            config.test_mode
        )

        # İstatistikler
        self.stats = {
            'total_ticks': 0,
            'search_time': 0,
            'target_found': False,
            'visited_positions': 0,
            'total_rotation': 0.0,
            'total_distance': 0.0,
            'detections_received': 0
        }

        self.log_data = []
        self.start_time = None

        print(f"\n{'=' * 60}")
        print(f"ARAMA TEST SİMÜLASYONU (ROS2 BAĞIMSIZ)")
        print(f"{'=' * 60}")
        print(f"Hedef: {config.target_class}")
        print(f"Başlangıç: {config.home_lat:.6f}, {config.home_lon:.6f}")
        print(f"Arama alanı: {config.search_area_radius}m")
        print(f"Hedef görünür: {config.target_visible}")
        print(f"Test modu: {config.test_mode}")
        print(f"{'=' * 60}\n")

    def run(self):
        """Simülasyonu çalıştır."""
        self.start_time = time.monotonic()

        # İlk GPS
        gps_data = self.gps.get_gps()
        self.arama.update_gps(
            gps_data['lat'],
            gps_data['lon'],
            gps_data['heading']
        )

        print("Simülasyon başlıyor...\n")

        while True:
            tick_start = time.monotonic()
            self.stats['total_ticks'] += 1

            # 1. Vision tespitleri
            gps_data = self.gps.get_gps()
            detections = self.vision.update(gps_data)
            if detections:
                self.stats['detections_received'] += 1

            # 2. Arama güncelle
            self.arama.update(detections)

            # 3. GPS'i güncelle (komutları oku)
            if self.topics.cmd_vel_pub.last_msg:
                cmd = self.topics.cmd_vel_pub.last_msg
                self.gps.update_from_commands(
                    cmd.get('linear_x', 0),
                    cmd.get('angular_z', 0)
                )

            # 4. Arama'ya GPS'i güncelle
            gps_data = self.gps.get_gps()
            self.arama.update_gps(
                gps_data['lat'],
                gps_data['lon'],
                gps_data['heading']
            )

            # 5. İstatistikler
            self.stats['search_time'] = time.monotonic() - self.start_time
            status = self.arama.get_search_status()
            self.stats['visited_positions'] = status['visited_positions']
            self.stats['total_rotation'] = status.get('total_rotation', 0)
            self.stats['total_distance'] = self.gps.total_distance

            # 6. Durum log'u
            if self.stats['total_ticks'] % 50 == 0:
                self._log_status()

            # 7. Tamamlama kontrolü
            if self.arama.finished:
                self.stats['target_found'] = True
                self._log_status()
                print(f"\n✅ HEDEF BULUNDU! Süre: {self.stats['search_time']:.1f}s")
                break

            if self.stats['search_time'] > self.config.simulation_duration:
                print(f"\n⏰ ZAMAN AŞIMI! {self.config.simulation_duration}s")
                break

            # 8. Bekle
            tick_duration = time.monotonic() - tick_start
            sleep_time = max(0, 0.1 - tick_duration)
            time.sleep(sleep_time)

        # Sonuçlar
        print(f"\n{'=' * 60}")
        print("📊 TEST SONUÇLARI")
        print(f"{'=' * 60}")
        print(f"Hedef bulundu: {'✅' if self.stats['target_found'] else '❌'}")
        print(f"Toplam tick: {self.stats['total_ticks']}")
        print(f"Arama süresi: {self.stats['search_time']:.1f}s")
        print(f"Ziyaret edilen konum: {self.stats['visited_positions']}")
        print(f"Toplam dönüş: {self.stats['total_rotation']:.1f}°")
        print(f"Toplam mesafe: {self.stats['total_distance']:.1f}m")
        print(f"Tespit alınan: {self.stats['detections_received']}")
        print(f"{'=' * 60}")

        return self.stats

    def _log_status(self):
        """Durum log'u."""
        status = self.arama.get_search_status()
        print(
            f"[{self.stats['search_time']:.1f}s] "
            f"State: {status['state']}, "
            f"Finished: {status['finished']}, "
            f"Onay: {status['target_confirmed']}, "
            f"Ardışık: {status['consecutive_detections']}, "
            f"İstasyon: {status['visited_positions']}, "
            f"Dönüş: {status['rotated_deg']:.1f}°"
        )


# ============================================================
# ANA FONKSİYON
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Arama test simülatörü")
    parser.add_argument('--target', type=str, default='red',
                        choices=['red', 'green', 'black'],
                        help='Hedef duba rengi')
    parser.add_argument('--test-mode', action='store_true', default=True,
                        help='Test modu')
    parser.add_argument('--duration', type=float, default=120.0,
                        help='Maksimum süre (saniye)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Rastgele tohum')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Sensör gürültüsü')
    parser.add_argument('--target-visible', action='store_true', default=True,
                        help='Hedef görünür')

    args = parser.parse_args()
    random.seed(args.seed)

    config = SimConfig(
        target_class=f"{args.target}_buoy",
        test_mode=args.test_mode,
        simulation_duration=args.duration,
        noise_level=args.noise,
        target_visible=args.target_visible
    )

    simulator = AramaTestSimulator(config)
    simulator.run()


if __name__ == "__main__":
    main()