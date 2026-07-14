#!/usr/bin/env python3
"""
Task-3 Kamikaze Angajman Görevi — Aşama 3: ÇARPMA
GERÇEK HAYATTA ÇALIŞACAK ŞEKİLDE İYİLEŞTİRİLMİŞ VERSİYON (TAMAMLANMIŞ)

Bu modül bağımsızdır ve yaklasma.py ile aynı mimariyi kullanır.
3 çarpma başarısız olursa aramaya dönülmesi için sinyal gönderir.

TAMAMLANAN / DÜZELTİLEN KISIMLAR:
  1. update() fonksiyonu yarım kalmıştı (CREEPING / BACKING_OFF / COOLDOWN
     durumları hiç işlenmiyordu) -> tamamlandı.
  2. _register_hit() log mesajı yanlış değeri basıyordu (sayaç sıfırlandıktan
     sonra loglanıyordu) -> gerçek ivme deltası saklanıp loglanıyor.
  3. update_imu() içinde, henüz 3 ardışığa ulaşmamış "spike" örnekleri bile
     baseline'a ekleniyordu -> çarpma anında baseline kirleniyor, hassasiyet
     düşüyordu. Artık sadece spike OLMAYAN örnekler baseline'a giriyor.
  4. TARGET_VISIBILITY_TIMEOUT parametresi tanımlı ama hiç kullanılmıyordu
     -> artık "hiç hedef görülmedi" ve "hedef kısa süre kayboldu" ayrımı
     için kullanılıyor.
  5. IMU eşiği bir sebepten (yumuşak/teğet temas) çarpmayı yakalayamazsa diye
     mesafe tabanlı yedek temas kontrolü eklendi (SOFT_CONTACT_DISTANCE_M).
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
REQUIRED_HITS = 3
CREEP_SPEED = 0.20
CREEP_ANGULAR_KP = 0.03
MAX_CREEP_ANGULAR_Z = 0.4

# --- IMU ÇARPMA ALGILAMA (SAHADA KALİBRE EDİN!) ---
IMU_BASELINE_WINDOW = 20
IMPACT_ACCEL_THRESHOLD = 5.0        # m/s² - SAHADA AYARLAYIN
IMPACT_CONSECUTIVE_SAMPLES = 3
IMPACT_MIN_SPEED = 0.1

# --- YEDEK TEMAS ALGILAMA (IMU spike'ı kaçırırsa) ---
SOFT_CONTACT_DISTANCE_M = 0.4       # Bu mesafede görsel olarak "değmiş" say

# --- ÇARPMA SONRASI ---
BACKOFF_SPEED = -0.20
BACKOFF_DURATION_SEC = 1.5
COOLDOWN_SEC = 1.0

# --- ZAMAN AŞIMLARI ---
PER_ATTEMPT_TIMEOUT_SEC = 15.0
TOTAL_CARPMA_TIMEOUT_SEC = 45.0
TARGET_VISIBILITY_TIMEOUT = 3.0     # Bu süre hiç görülmezse uyar / dikkatli ol

# --- DEAD RECKONING ---
DEAD_RECKONING_TIMEOUT = 5.0        # Bu süreden eskiyse tahmini konumu terk et


class CarpmaState(Enum):
    CREEPING = auto()
    BACKING_OFF = auto()
    COOLDOWN = auto()
    COMPLETE = auto()
    MISSED = auto()


class CarpmaGorevi:
    """Çarpma görevi - GERÇEK HAYAT İÇİN OPTİMİZE"""

    def __init__(self, node, mission_topics, target_class):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class

        self.state = CarpmaState.CREEPING
        self.finished = False
        self.success = False
        self.hit_count = 0

        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0
        self.last_known_target_lat = None
        self.last_known_target_lon = None
        self.last_target_update_time = None

        self.accel_history = deque(maxlen=IMU_BASELINE_WINDOW)
        self.accel_magnitude_history = deque(maxlen=IMU_BASELINE_WINDOW)
        self.consecutive_spikes = 0
        self.last_impact_delta = 0.0
        self.impact_armed = True
        self.current_speed = 0.0

        self.mission_start_time = None
        self.attempt_start_time = None
        self.backoff_start_time = None
        self.cooldown_start_time = None
        self.target_last_seen_time = None

        self.total_attempts = 0
        self.impact_detection_count = 0

        self.logger.info(f"[ÇARPMA] Başlatıldı, hedef: {self.target_class}")

    # --------------------------------------------------------
    def update_gps(self, lat, lon, heading):
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

    # --------------------------------------------------------
    def update_imu(self, accel_x, accel_y, accel_z):
        """IMU verilerini güncelle ve çarpma algıla."""
        if not self.impact_armed:
            return

        accel_mag = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        self.accel_magnitude_history.append(accel_mag)

        if len(self.accel_history) < IMU_BASELINE_WINDOW // 2:
            self.accel_history.append(accel_mag)
            return

        baseline = sum(self.accel_history) / len(self.accel_history)

        is_spike = False
        if self.current_speed >= IMPACT_MIN_SPEED:
            delta = abs(accel_mag - baseline)
            if delta > IMPACT_ACCEL_THRESHOLD:
                is_spike = True
                self.consecutive_spikes += 1
                self.last_impact_delta = delta
            else:
                self.consecutive_spikes = 0

            if self.consecutive_spikes >= IMPACT_CONSECUTIVE_SAMPLES:
                self._register_hit()
                return

        # DÜZELTME: sadece spike OLMAYAN örnekler baseline'a giriyor.
        # Aksi halde tam çarpma anında baseline kirlenip hassasiyet düşüyordu.
        if not is_spike:
            self.accel_history.append(accel_mag)

    # --------------------------------------------------------
    def reset_carpma(self):
        """Çarpmayı sıfırla (aramaya dönmek için)."""
        self.state = CarpmaState.CREEPING
        self.finished = False
        self.success = False
        self.hit_count = 0
        self.consecutive_spikes = 0
        self.last_impact_delta = 0.0
        self.impact_armed = True
        self.accel_history.clear()
        self.accel_magnitude_history.clear()
        self.last_known_target_lat = None
        self.last_known_target_lon = None
        self.last_target_update_time = None
        self.target_last_seen_time = None
        self.mission_start_time = time.monotonic()
        self.attempt_start_time = time.monotonic()
        self.backoff_start_time = None
        self.cooldown_start_time = None
        self.total_attempts = 0
        self.logger.info("[ÇARPMA] Sıfırlandı, aramaya dönülebilir.")

    # --------------------------------------------------------
    def should_retry_search(self):
        return self.state == CarpmaState.MISSED

    # --------------------------------------------------------
    def _register_hit(self, soft=False):
        """Çarpma kaydet."""
        self.hit_count += 1
        self.impact_detection_count += 1
        delta = self.last_impact_delta
        self.consecutive_spikes = 0
        self.accel_history.clear()
        self.impact_armed = False

        # DÜZELTME: eskiden consecutive_spikes burada sıfırlandıktan SONRA
        # loglanıyordu, yani log her zaman 0.0 basıyordu. Artık gerçek
        # ivme deltası saklanıp loglanıyor.
        kaynak = "mesafe (yedek)" if soft else "IMU"
        self.logger.info(
            f"[ÇARPMA] 💥 TEMAS! ({self.hit_count}/{REQUIRED_HITS}) "
            f"kaynak: {kaynak}, ivme delta: ~{delta:.2f} m/s²"
        )

        stop_vehicle(self.topics.cmd_vel_pub)

        if self.hit_count >= REQUIRED_HITS:
            self.state = CarpmaState.COMPLETE
            self.finished = True
            self.success = True
            self.logger.info("[ÇARPMA] 🎉 3 çarpma tamamlandı! GÖREV BAŞARILI!")
            return

        self.state = CarpmaState.BACKING_OFF
        self.backoff_start_time = time.monotonic()
        self.total_attempts += 1

    # --------------------------------------------------------
    def _select_target(self, detections):
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

        absolute_bearing = (self.current_heading + target["Buoy angle: "]) % 360.0

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
    def _creep_towards_target(self, visible_target, now):
        """Hedefe doğru sürün.
        Returns:
            bool: True -> normal şekilde ilerlendi (görsel veya dead-reckoning)
                  False -> konum bilgisi çok eski/yok, çağıran taraf MISSED'e
                           geçmeli.
        """
        if visible_target is not None:
            angle = visible_target["Buoy angle: "]
            angular_z = -CREEP_ANGULAR_KP * angle
            angular_z = max(-MAX_CREEP_ANGULAR_Z, min(MAX_CREEP_ANGULAR_Z, angular_z))
            self.current_speed = CREEP_SPEED

            # YENİ: yedek temas kontrolü. IMU eşiği yumuşak/teğet bir
            # temasta tetiklenmeyebilir; çok yakın mesafede görsel olarak
            # da "değmiş" sayıyoruz.
            if visible_target["distance"] <= SOFT_CONTACT_DISTANCE_M:
                publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=0.0)
                self._register_hit(soft=True)
                return True

            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=self.current_speed, angular_z=angular_z)
            return True

        if self.last_known_target_lat is not None:
            age = now - self.last_target_update_time
            if age > DEAD_RECKONING_TIMEOUT:
                # Konum bilgisi çok eski; körlemesine ilerlemek yerine
                # (eski davranış: düz git) çağırana MISSED sinyali ver.
                return False

            if age > TARGET_VISIBILITY_TIMEOUT:
                self.logger.warn(
                    f"[ÇARPMA] Hedef {age:.1f}sn'dir görünmüyor, "
                    f"tahmini konuma göre düşük hızda ilerleniyor...",
                    throttle_duration_sec=1.0,
                )
                speed_scale = 0.5
            else:
                speed_scale = 1.0

            bearing = calculate_bearing(
                self.current_lat, self.current_lon,
                self.last_known_target_lat, self.last_known_target_lon
            )
            heading_error = (bearing - self.current_heading + 180) % 360 - 180
            angular_z = -CREEP_ANGULAR_KP * heading_error
            angular_z = max(-MAX_CREEP_ANGULAR_Z, min(MAX_CREEP_ANGULAR_Z, angular_z))
            self.current_speed = CREEP_SPEED * speed_scale

            publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=self.current_speed, angular_z=angular_z)
            return True

        # Hiçbir bilgi yok
        return False

    # --------------------------------------------------------
    def _check_timeouts(self, now):
        if self.mission_start_time is None:
            self.mission_start_time = now
            self.attempt_start_time = now
            return False

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

        if self.state == CarpmaState.CREEPING:
            if now - self.attempt_start_time > PER_ATTEMPT_TIMEOUT_SEC:
                self.logger.warn(
                    f"[ÇARPMA] Bu deneme {PER_ATTEMPT_TIMEOUT_SEC}s oldu, "
                    f"ilerlemeye devam..."
                )
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

        if self._check_timeouts(now):
            return True

        if self.state in (CarpmaState.COMPLETE, CarpmaState.MISSED):
            stop_vehicle(self.topics.cmd_vel_pub)
            return True

        # -------- YENİ: eksik olan durum makinesi tamamlandı --------
        if self.state == CarpmaState.BACKING_OFF:
            if self.backoff_start_time is None:
                self.backoff_start_time = now
            if now - self.backoff_start_time < BACKOFF_DURATION_SEC:
                publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=BACKOFF_SPEED, angular_z=0.0)
                return False
            stop_vehicle(self.topics.cmd_vel_pub)
            self.state = CarpmaState.COOLDOWN
            self.cooldown_start_time = now
            return False

        if self.state == CarpmaState.COOLDOWN:
            stop_vehicle(self.topics.cmd_vel_pub)
            if now - self.cooldown_start_time >= COOLDOWN_SEC:
                self.impact_armed = True
                self.consecutive_spikes = 0
                self.accel_history.clear()
                self.state = CarpmaState.CREEPING
                self.attempt_start_time = now
                self.logger.info(
                    f"[ÇARPMA] Yeniden denemeye hazır ({self.hit_count}/{REQUIRED_HITS})"
                )
            return False

        if self.state == CarpmaState.CREEPING:
            target = self._select_target(detections)
            if target is not None:
                self._update_target_position(detections)

            if target is None and self.last_known_target_lat is None:
                # Hiç hedef görülmedi. Deneme başından beri kısa bir süre
                # bekle, sonra vazgeç.
                if now - self.attempt_start_time < TARGET_VISIBILITY_TIMEOUT:
                    stop_vehicle(self.topics.cmd_vel_pub)
                    return False
                self.logger.error("[ÇARPMA] Hedef hiç görülemedi, aramaya dönülüyor.")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.state = CarpmaState.MISSED
                self.finished = True
                self.success = False
                return True

            ok = self._creep_towards_target(target, now)
            if self.state != CarpmaState.CREEPING:
                # _creep_towards_target içinde SOFT_CONTACT ile hit
                # kaydedilmiş ve state değişmiş olabilir (BACKING_OFF/COMPLETE)
                return self.state in (CarpmaState.COMPLETE, CarpmaState.MISSED)
            if not ok:
                self.logger.error("[ÇARPMA] Hedef konum bilgisi çok eski, aramaya dönülüyor.")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.state = CarpmaState.MISSED
                self.finished = True
                self.success = False
                return True
            return False

        return False

    # --------------------------------------------------------
    def get_status(self):
        return {
            "state": self.state.name,
            "finished": self.finished,
            "success": self.success,
            "hit_count": self.hit_count,
            "required_hits": REQUIRED_HITS,
            "total_attempts": self.total_attempts,
            "impact_armed": self.impact_armed,
        }