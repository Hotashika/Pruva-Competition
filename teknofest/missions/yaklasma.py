#!/usr/bin/env python3
"""
Task-3 Kamikaze Angajman Görevi — Aşama 2: YAKLAŞMA + EMİN OLMA
GERÇEK HAYATTA ÇALIŞACAK ŞEKİLDE İYİLEŞTİRİLMİŞ VERSİYON

Bu modül bağımsızdır ve arama.py ile aynı mimariyi kullanır.
Hedef kaybolursa otomatik olarak aramaya dönülmesi için sinyal gönderir.
"""

import time
import math
from collections import deque
from enum import Enum, auto

from utils.mavlink_utilities import (
    publish_cmd_vel,
    stop_vehicle,
    calculate_gps_distance,
)

# ============================================================
# GERÇEK HAYAT PARAMETRELERİ (SAHAYA GÖRE AYARLAYIN)
# ============================================================

# --- GÜVENLİK MESAFELERİ ---
SAFE_STOP_DISTANCE_M = 1.0          # Bu mesafede dur
EMERGENCY_STOP_DISTANCE_M = 0.5     # Acil durma mesafesi (güvenlik)
CONFIRM_TRIGGER_DISTANCE_M = 3.0    # Emin olma başlangıç mesafesi

# --- EMİN OLMA (CONFIRMATION) ---
MIN_CONSECUTIVE_DETECTIONS = 5      # Kaç ardışık karede görülmeli
CONFIRM_HOLD_SEC = 1.5              # En az bu süre boyunca görülmeli
DETECTION_HISTORY_SIZE = 10         # Kaç karelik geçmiş tutulacak

# --- HEDEF KAYBI TOLERANSI ---
TARGET_LOST_TOLERANCE_SEC = 0.8     # Bu süre kaybolursa aramaya dön
MAX_LOST_FRAMES = 8                 # Maksimum kayıp frame sayısı

# --- PID KATSAYILARI (YÖN KONTROLÜ) ---
YAW_KP = 0.035
YAW_KI = 0.001
YAW_KD = 0.008
YAW_I_LIMIT = 0.15
MAX_ANGULAR_Z = 0.5

# --- HIZ KONTROLÜ ---
SURGE_MAX_LINEAR_X = 0.55           # Maksimum ileri hız
SURGE_MIN_LINEAR_X = 0.12           # Minimum ileri hız
SURGE_SLOWDOWN_START_M = 5.0        # Yavaşlama başlangıç mesafesi

# --- RÜZGAR/AKINTI TELAFİSİ ---
DRIFT_KP = 0.04
MAX_DRIFT_COMPENSATION = 0.12

# --- KOMUT YUMUŞATMA ---
MAX_ANGULAR_STEP = 0.06
MAX_LINEAR_STEP = 0.05

# --- MESAFE YUMUŞATMA ---
DISTANCE_SMOOTH_WINDOW = 5


class ApproachState(Enum):
    TRACKING = auto()          # Normal takip
    CONFIRMING = auto()        # Emin olma aşaması
    TARGET_LOST_WAIT = auto()  # Hedef kayboldu, bekleniyor
    STOPPING = auto()          # Durma manevrası
    DONE = auto()              # Yaklaşma tamamlandı
    LOST = auto()              # Hedef kayboldu (aramaya dön)


class YaklasmaGorevi:
    """Yaklaşma ve emin olma görevi - GERÇEK HAYAT İÇİN OPTİMİZE"""
    
    def __init__(self, node, mission_topics, target_class):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class

        # Durum
        self.state = ApproachState.TRACKING
        self.finished = False
        self.target_lost = False
        self.approach_start_time = None

        # Hedef tespit geçmişi (ARDİŞIK KONTROL İÇİN)
        self.detection_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.consecutive_detections = 0
        self.last_detection_frame = 0
        self.frame_counter = 0
        self.target_confirmed = False
        self.confirm_start_time = None

        # Mesafe yumuşatma
        self.distance_buffer = deque(maxlen=DISTANCE_SMOOTH_WINDOW)

        # PID durumu
        self.yaw_integral = 0.0
        self.yaw_prev_error = None
        self.prev_tick_time = None

        # Kayıp takibi
        self.lost_since = None
        self.lost_frame_count = 0

        # Son komutlar (rate limiting)
        self.last_angular_z = 0.0
        self.last_linear_x = 0.0

        # Durma (STOPPING) fazı
        self.stopping_since = None
        self.reverse_start_time = None
        self.reverse_active = False

        # IMU verisi
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0

        # Konum
        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0

        # İstatistikler
        self.total_detections = 0
        self.total_frames = 0

        self.logger.info(f"[YAKLAŞMA] Başlatıldı, hedef: {self.target_class}")

    # --------------------------------------------------------
    def update_gps(self, lat, lon, heading):
        """GPS verilerini güncelle."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

    # --------------------------------------------------------
    def update_imu(self, gyro_z, accel_x, accel_y):
        """IMU verilerini güncelle."""
        self.gyro_z = gyro_z
        self.accel_x = accel_x
        self.accel_y = accel_y

    # --------------------------------------------------------
    def reset_approach(self):
        """Yaklaşmayı sıfırla (aramaya dönmek için)."""
        self.state = ApproachState.TRACKING
        self.finished = False
        self.target_lost = False
        self.target_confirmed = False
        self.consecutive_detections = 0
        self.confirm_start_time = None
        self.lost_since = None
        self.lost_frame_count = 0
        self.distance_buffer.clear()
        self.yaw_integral = 0.0
        self.yaw_prev_error = None
        self.last_angular_z = 0.0
        self.last_linear_x = 0.0
        self.reverse_active = False
        self.detection_history.clear()
        self.logger.info("[YAKLAŞMA] Sıfırlandı, aramaya dönülebilir.")

    # --------------------------------------------------------
    def should_return_to_search(self):
        """Hedef kayboldu mu? Aramaya dönülmeli mi?"""
        return self.state == ApproachState.LOST or self.target_lost

    # --------------------------------------------------------
    def _select_target(self, detections):
        """Tespitler arasından hedefi seç."""
        if not detections:
            return None

        candidates = [
            d for d in detections
            if d.get("class") == self.target_class
            and d.get("distance") is not None
            and d.get("distance", -1) > 0.3
            and d.get("Buoy angle: ") is not None
        ]

        if not candidates:
            return None

        # En yakın hedefi seç
        return min(candidates, key=lambda d: d["distance"])

    # --------------------------------------------------------
    def _update_detection_history(self, target):
        """Tespit geçmişini güncelle ve ardışık tespit kontrolü yap."""
        self.frame_counter += 1
        self.total_frames += 1

        if target is not None:
            self.total_detections += 1
            self.consecutive_detections += 1
            self.detection_history.append(True)
            self.lost_frame_count = 0
            self.lost_since = None

            # Ardışık tespit sayısı yeterli mi?
            if self.consecutive_detections >= MIN_CONSECUTIVE_DETECTIONS:
                if not self.target_confirmed:
                    self.target_confirmed = True
                    self.confirm_start_time = time.monotonic()
                    self.logger.info(
                        f"[EMİN OLMA] {MIN_CONSECUTIVE_DETECTIONS} ardışık tespit! "
                        f"Hedef doğrulandı."
                    )
                return True
        else:
            self.detection_history.append(False)
            self.consecutive_detections = 0

            # Hedef kaybı kontrolü
            if self.target_confirmed:
                self.lost_frame_count += 1
                if self.lost_since is None:
                    self.lost_since = time.monotonic()
                    self.logger.warn("[YAKLAŞMA] Hedef kayboldu, tolerans başladı...")

        return self.target_confirmed and target is not None

    # --------------------------------------------------------
    def _smoothed_distance(self, raw_distance):
        """Mesafeyi yumuşat."""
        self.distance_buffer.append(raw_distance)
        return sum(self.distance_buffer) / len(self.distance_buffer)

    # --------------------------------------------------------
    def _compute_yaw_command(self, angle_error_deg, dt):
        """Yaw PID kontrolü."""
        self.yaw_integral += angle_error_deg * dt
        self.yaw_integral = max(-YAW_I_LIMIT, min(YAW_I_LIMIT, self.yaw_integral))

        if self.yaw_prev_error is None:
            derivative = 0.0
        else:
            derivative = (angle_error_deg - self.yaw_prev_error) / (dt + 0.001)

        self.yaw_prev_error = angle_error_deg

        raw = -(YAW_KP * angle_error_deg + YAW_KI * self.yaw_integral + YAW_KD * derivative)
        return max(-MAX_ANGULAR_Z, min(MAX_ANGULAR_Z, raw))

    # --------------------------------------------------------
    def _compute_surge_command(self, distance_m):
        """Hız komutunu hesapla."""
        if distance_m >= SURGE_SLOWDOWN_START_M:
            return SURGE_MAX_LINEAR_X

        span = SURGE_SLOWDOWN_START_M - SAFE_STOP_DISTANCE_M
        if span <= 0:
            return SURGE_MIN_LINEAR_X

        ratio = (distance_m - SAFE_STOP_DISTANCE_M) / span
        ratio = max(0.0, min(1.0, ratio))
        return SURGE_MIN_LINEAR_X + ratio * (SURGE_MAX_LINEAR_X - SURGE_MIN_LINEAR_X)

    # --------------------------------------------------------
    def _drift_compensation(self):
        """Rüzgar/akıntı telafisi."""
        comp = -DRIFT_KP * self.accel_y
        return max(-MAX_DRIFT_COMPENSATION, min(MAX_DRIFT_COMPENSATION, comp))

    # --------------------------------------------------------
    def _rate_limit(self, target, last, max_step):
        """Komut yumuşatma."""
        delta = target - last
        if delta > max_step:
            return last + max_step
        if delta < -max_step:
            return last - max_step
        return target

    # --------------------------------------------------------
    def _start_stopping(self, now):
        """Durma manevrasını başlat."""
        self.state = ApproachState.STOPPING
        self.stopping_since = now
        self.reverse_active = True
        self.reverse_start_time = now
        self.logger.info("[YAKLAŞMA] Güvenli mesafe, durma başlıyor!")

    # --------------------------------------------------------
    def _do_stopping(self, now):
        """Durma manevrasını uygula."""
        # İlk olarak ters itki (0.5 sn)
        if self.reverse_active:
            reverse_duration = 0.5
            if now - self.reverse_start_time < reverse_duration:
                publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=-0.15, angular_z=0.0)
                return False
            
            self.reverse_active = False
            self.logger.info("[YAKLAŞMA] Ters itki tamamlandı, duruluyor...")

        # Tam dur
        stop_vehicle(self.topics.cmd_vel_pub)
        self.state = ApproachState.DONE
        self.finished = True
        self.logger.info("[YAKLAŞMA] ✅ Yaklaşma tamamlandı!")
        return True

    # --------------------------------------------------------
    def update(self, detections):
        """Ana güncelleme döngüsü."""
        now = time.monotonic()
        dt = 0.1 if self.prev_tick_time is None else max(0.01, now - self.prev_tick_time)
        self.prev_tick_time = now

        # Başlangıç zamanını kaydet
        if self.approach_start_time is None:
            self.approach_start_time = now

        # --- DURUM KONTROLLERİ ---
        if self.state == ApproachState.STOPPING:
            return self._do_stopping(now)

        if self.state in (ApproachState.DONE, ApproachState.LOST):
            return True

        # --- HEDEF TESPİTİ ---
        target = self._select_target(detections)
        is_confirmed = self._update_detection_history(target)

        # HEDEF KAYBI KONTROLÜ
        if target is None:
            if self.target_confirmed:
                # Hedef onaylanmış ama kayboldu
                if self.lost_frame_count >= MAX_LOST_FRAMES:
                    self.logger.error(
                        f"[YAKLAŞMA] Hedef {MAX_LOST_FRAMES} frame kayboldu! "
                        f"Aramaya dönülüyor."
                    )
                    stop_vehicle(self.topics.cmd_vel_pub)
                    self.state = ApproachState.LOST
                    self.target_lost = True
                    return True

                # Tolerans içinde: son komutla devam et
                coast_linear = self.last_linear_x * 0.7
                coast_angular = self.last_angular_z * 0.7
                publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=coast_linear, angular_z=coast_angular)
                self.last_linear_x = coast_linear
                self.last_angular_z = coast_angular
                return False

            # Hedef hiç görülmemiş veya onaylanmamış
            stop_vehicle(self.topics.cmd_vel_pub)
            self.last_linear_x = 0.0
            self.last_angular_z = 0.0
            return False

        # --- HEDEF GÖRÜNÜYOR ---
        raw_distance = target["distance"]
        angle_error = target["Buoy angle: "]
        distance = self._smoothed_distance(raw_distance)

        # --- EMİN OLMA (CONFIRMATION) KONTROLÜ ---
        if not self.target_confirmed and distance <= CONFIRM_TRIGGER_DISTANCE_M:
            if self.state != ApproachState.CONFIRMING:
                self.state = ApproachState.CONFIRMING
                self.logger.info(
                    f"[EMİN OLMA] {CONFIRM_TRIGGER_DISTANCE_M:.1f}m içinde, "
                    f"doğrulama başlıyor..."
                )

            # Ardışık tespit sayısı kontrolü
            if self.consecutive_detections >= MIN_CONSECUTIVE_DETECTIONS:
                self.target_confirmed = True
                self.state = ApproachState.TRACKING
                self.logger.info(
                    f"[EMİN OLMA] ✅ Hedef doğrulandı! "
                    f"({self.consecutive_detections} ardışık tespit)"
                )
            else:
                # Doğrulama tamamlanmadı, ilerleme yok
                yaw_cmd = self._compute_yaw_command(angle_error, dt)
                angular_z = self._rate_limit(yaw_cmd, self.last_angular_z, MAX_ANGULAR_STEP)
                publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=angular_z)
                self.last_angular_z = angular_z
                self.last_linear_x = 0.0
                self.logger.info(
                    f"[EMİN OLMA] Bekleniyor... ({self.consecutive_detections}/{MIN_CONSECUTIVE_DETECTIONS})",
                    throttle_duration_sec=0.5
                )
                return False

        # --- GÜVENLİ MESAFE KONTROLÜ ---
        if distance <= SAFE_STOP_DISTANCE_M:
            if self.target_confirmed:
                self._start_stopping(now)
                return self._do_stopping(now)
            else:
                # HEDEF DOĞRULANMADAN ÇOK YAKLAŞMA (GÜVENLİK)
                self.logger.warn(
                    f"[GÜVENLİK] Hedef doğrulanmadı ama {distance:.2f}m yakınında! "
                    f"Durduruluyor, aramaya dönülüyor."
                )
                stop_vehicle(self.topics.cmd_vel_pub)
                self.state = ApproachState.LOST
                self.target_lost = True
                return True

        # --- ACİL DURMA (EMERGENCY STOP) ---
        if distance <= EMERGENCY_STOP_DISTANCE_M:
            self.logger.error(f"[ACİL] Çok yakın! {distance:.2f}m, acil dur!")
            stop_vehicle(self.topics.cmd_vel_pub)
            self.state = ApproachState.STOPPING
            self.stopping_since = now
            self.reverse_active = True
            self.reverse_start_time = now
            return False

        # --- PID KONTROL (NORMAL YAKLAŞMA) ---
        yaw_cmd = self._compute_yaw_command(angle_error, dt)
        yaw_cmd += self._drift_compensation()
        angular_z = self._rate_limit(yaw_cmd, self.last_angular_z, MAX_ANGULAR_STEP)

        surge_cmd = self._compute_surge_command(distance)
        linear_x = self._rate_limit(surge_cmd, self.last_linear_x, MAX_LINEAR_STEP)

        # Komutları uygula
        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=linear_x, angular_z=angular_z)
        self.last_angular_z = angular_z
        self.last_linear_x = linear_x

        return False

    # --------------------------------------------------------
    def get_status(self):
        """Durum bilgilerini döndür."""
        return {
            "state": self.state.name,
            "finished": self.finished,
            "target_lost": self.target_lost,
            "target_confirmed": self.target_confirmed,
            "consecutive_detections": self.consecutive_detections,
            "distance": list(self.distance_buffer)[-1] if self.distance_buffer else None,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "detection_rate": self.total_detections / max(1, self.total_frames)
        }