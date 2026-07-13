#!/usr/bin/env python3
"""
Task-3 Kamikaze Angajman Görevi — Aşama 3: ÇARPMA
GERÇEK HAYATTA ÇALIŞACAK ŞEKİLDE İYİLEŞTİRİLMİŞ VERSİYON

Bu modül bağımsızdır ve yaklasma.py ile aynı mimariyi kullanır.
3 çarpma başarısız olursa aramaya dönülmesi için sinyal gönderir.
"""

import time
import math
from collections import deque
from enum import Enum, auto

from utils.mavlink_utilities import (
    publish_cmd_vel,
    stop_vehicle,
    calculate_bearing,
    calculate_gps_distance,
)


# ============================================================
# GERÇEK HAYAT PARAMETRELERİ (SAHAYA GÖRE KALİBRE EDİN)
# ============================================================

# --- ÇARPMA HEDEFLERİ ---
REQUIRED_HITS = 3                   # Kaç çarpma istiyoruz
CREEP_SPEED = 0.20                  # Yaklaşma hızı
CREEP_ANGULAR_KP = 0.03             # Yön düzeltme katsayısı
MAX_CREEP_ANGULAR_Z = 0.4           # Maksimum dönüş hızı

# --- IMU ÇARPMA ALGILAMA (SAHADA KALİBRE EDİN!) ---
IMU_BASELINE_WINDOW = 20            # Kaç örnek baz alınacak
IMPACT_ACCEL_THRESHOLD = 5.0        # m/s² - SAHADA AYARLAYIN
IMPACT_CONSECUTIVE_SAMPLES = 3      # Kaç ardışık örnek
IMPACT_MIN_SPEED = 0.1              # Çarpma için minimum hız

# --- ÇARPMA SONRASI ---
BACKOFF_SPEED = -0.20               # Geri çekilme hızı
BACKOFF_DURATION_SEC = 1.5          # Geri çekilme süresi
COOLDOWN_SEC = 1.0                  # Bekleme süresi (IMU sakinleşsin)

# --- ZAMAN AŞIMLARI ---
PER_ATTEMPT_TIMEOUT_SEC = 15.0      # Tek deneme timeout
TOTAL_CARPMA_TIMEOUT_SEC = 45.0     # Toplam timeout
TARGET_VISIBILITY_TIMEOUT = 3.0     # Hedef ne kadar kaybolabilir

# --- DEAD RECKONING ---
DEAD_RECKONING_TIMEOUT = 5.0        # Son konum ne kadar geçerli


class CarpmaState(Enum):
    CREEPING = auto()          # Hedefe yaklaşıyor
    BACKING_OFF = auto()       # Çarptı, geri çekiliyor
    COOLDOWN = auto()          # Bekleme (IMU sakinleşsin)
    COMPLETE = auto()          # 3 çarpma tamam
    MISSED = auto()            # Başarısız (aramaya dön)


class CarpmaGorevi:
    """Çarpma görevi - GERÇEK HAYAT İÇİN OPTİMİZE"""
    
    def __init__(self, node, mission_topics, target_class):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class

        # Durum
        self.state = CarpmaState.CREEPING
        self.finished = False
        self.success = False
        self.hit_count = 0

        # Konum
        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0
        self.last_known_target_lat = None
        self.last_known_target_lon = None
        self.last_target_update_time = None

        # IMU / çarpma algılama
        self.accel_history = deque(maxlen=IMU_BASELINE_WINDOW)
        self.accel_magnitude_history = deque(maxlen=IMU_BASELINE_WINDOW)
        self.consecutive_spikes = 0
        self.impact_armed = True
        self.current_speed = 0.0

        # Zamanlama
        self.mission_start_time = None
        self.attempt_start_time = None
        self.backoff_start_time = None
        self.cooldown_start_time = None
        self.target_last_seen_time = None

        # İstatistikler
        self.total_attempts = 0
        self.impact_detection_count = 0

        self.logger.info(f"[ÇARPMA] Başlatıldı, hedef: {self.target_class}")

    # --------------------------------------------------------
    def update_gps(self, lat, lon, heading):
        """GPS verilerini güncelle."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

    # --------------------------------------------------------
    def update_imu(self, accel_x, accel_y, accel_z):
        """IMU verilerini güncelle ve çarpma algıla."""
        if not self.impact_armed:
            return

        # İvme büyüklüğü
        accel_mag = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        self.accel_magnitude_history.append(accel_mag)

        # Yeterli örnek yoksa bekle
        if len(self.accel_history) < IMU_BASELINE_WINDOW // 2:
            self.accel_history.append(accel_mag)
            return

        # Baseline hesapla (hareketli ortalama)
        baseline = sum(self.accel_history) / len(self.accel_history)

        # Çarpma tespiti
        if self.current_speed >= IMPACT_MIN_SPEED:  # Sadece hareket halindeyken
            delta = abs(accel_mag - baseline)
            if delta > IMPACT_ACCEL_THRESHOLD:
                self.consecutive_spikes += 1
            else:
                self.consecutive_spikes = 0

            # Ardışık spike kontrolü
            if self.consecutive_spikes >= IMPACT_CONSECUTIVE_SAMPLES:
                self._register_hit()
                return

        # Baseline'ı güncelle (yavaşça)
        self.accel_history.append(accel_mag)

    # --------------------------------------------------------
    def reset_carpma(self):
        """Çarpmayı sıfırla (aramaya dönmek için)."""
        self.state = CarpmaState.CREEPING
        self.finished = False
        self.success = False
        self.hit_count = 0
        self.consecutive_spikes = 0
        self.impact_armed = True
        self.accel_history.clear()
        self.accel_magnitude_history.clear()
        self.last_known_target_lat = None
        self.last_known_target_lon = None
        self.last_target_update_time = None
        self.mission_start_time = time.monotonic()
        self.attempt_start_time = time.monotonic()
        self.total_attempts = 0
        self.logger.info("[ÇARPMA] Sıfırlandı, aramaya dönülebilir.")

    # --------------------------------------------------------
    def should_retry_search(self):
        """Başarısız oldu mu? Aramaya dönülmeli mi?"""
        return self.state == CarpmaState.MISSED

    # --------------------------------------------------------
    def _register_hit(self):
        """Çarpma kaydet."""
        self.hit_count += 1
        self.impact_detection_count += 1
        self.consecutive_spikes = 0
        self.accel_history.clear()
        self.impact_armed = False

        self.logger.info(
            f"[ÇARPMA] 💥 TEMAS! ({self.hit_count}/{REQUIRED_HITS}) "
            f"ivme delta: ~{self.consecutive_spikes:.1f}"
        )

        stop_vehicle(self.topics.cmd_vel_pub)

        if self.hit_count >= REQUIRED_HITS:
            self.state = CarpmaState.COMPLETE
            self.finished = True
            self.success = True
            self.logger.info("[ÇARPMA] 🎉 3 çarpma tamamlandı! GÖREV BAŞARILI!")
            return

        # Yeni çarpma için geri çekil
        self.state = CarpmaState.BACKING_OFF
        self.backoff_start_time = time.monotonic()
        self.total_attempts += 1

    # --------------------------------------------------------
    def _select_target(self, detections):
        """Tespitlerden hedefi seç."""
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
    def _update_target_position(self, detections):
        """Hedef GPS konumunu güncelle."""
        target = self._select_target(detections)
        if target is None or self.current_lat is None:
            return None

        # Hedefin mutlak açısını hesapla
        absolute_bearing = (self.current_heading + target["Buoy angle: "]) % 360.0

        # Hedefin GPS konumunu tahmin et
        target_lat, target_lon = self._project_gps(
            self.current_lat,
            self.current_lon,
            absolute_bearing,
            target["distance"]
        )

        self.last_known_target_lat = target_lat
        self.last_known_target_lon = target_lon
        self.last_target_update_time = time.monotonic()
        self.target_last_seen_time = time.monotonic()

        return target

    # --------------------------------------------------------
    @staticmethod
    def _project_gps(lat, lon, bearing_deg, distance_m):
        """GPS projeksiyonu."""
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

    # --------------------------------------------------------
    def _creep_towards_target(self, visible_target):
        """Hedefe doğru sürün."""
        if visible_target is not None:
            # Görsel takip
            angle = visible_target["Buoy angle: "]
            angular_z = -CREEP_ANGULAR_KP * angle
            angular_z = max(-MAX_CREEP_ANGULAR_Z, min(MAX_CREEP_ANGULAR_Z, angular_z))
            self.current_speed = CREEP_SPEED

        elif self.last_known_target_lat is not None:
            # Dead reckoning
            if time.monotonic() - self.last_target_update_time > DEAD_RECKONING_TIMEOUT:
                self.logger.warn("[ÇARPMA] Son hedef konumu çok eski, düz git...")
                angular_z = 0.0
                self.current_speed = CREEP_SPEED * 0.5
            else:
                bearing = calculate_bearing(
                    self.current_lat, self.current_lon,
                    self.last_known_target_lat, self.last_known_target_lon
                )
                heading_error = (bearing - self.current_heading + 180) % 360 - 180
                angular_z = -CREEP_ANGULAR_KP * heading_error
                angular_z = max(-MAX_CREEP_ANGULAR_Z, min(MAX_CREEP_ANGULAR_Z, angular_z))
                self.current_speed = CREEP_SPEED

        else:
            # Hiçbir bilgi yok, düz git
            angular_z = 0.0
            self.current_speed = CREEP_SPEED * 0.3

        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=self.current_speed, angular_z=angular_z)

    # --------------------------------------------------------
    def _check_timeouts(self, now):
        """Zaman aşımı kontrolleri."""
        if self.mission_start_time is None:
            self.mission_start_time = now
            self.attempt_start_time = now
            return False

        # Toplam timeout
        if now - self.mission_start_time > TOTAL_CARPMA_TIMEOUT_SEC:
            self.logger.error(
                f"[ÇARPMA] Toplam zaman aşımı! "
                f"{self.hit_count}/{REQUIRED_HITS} çarpma ile kaldı."
            )
            stop_vehicle(self.topics.cmd_vel_pub)
            self.state = CarpmaState.MISSED
            self.finished = True
            self.success = False
            return True

        # Deneme timeout (sadece CREEPING'de)
        if self.state == CarpmaState.CREEPING:
            if now - self.attempt_start_time > PER_ATTEMPT_TIMEOUT_SEC:
                self.logger.warn(
                    f"[ÇARPMA] Bu deneme {PER_ATTEMPT_TIMEOUT_SEC}s oldu, "
                    f"ilerlemeye devam..."
                )
                # Devam et, ama timeout'u yenile
                self.attempt_start_time = now

        return False

    # --------------------------------------------------------
    def update(self, detections):
        """Ana güncelleme döngüsü."""
        if self.current_lat is None:
            return False

        now = time.monotonic()
        if self.mission_start_time is None:
            self.mission_start_time = now
            self.attempt_start_time = now

        # Zaman aşımı kontrolü
        if self._check_timeouts(now):
            return True

        # Tamamlandı veya başarısız
        if self.state in (CarpmaState.COMPLETE, CarpmaState.MISSED):
            stop_vehicle(self.topics.cmd_vel_pub)
            return True