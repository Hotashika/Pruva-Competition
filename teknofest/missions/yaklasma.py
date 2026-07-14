#!/usr/bin/env python3
"""
Task-3 Kamikaze Angajman Görevi — Aşama 2: YAKLAŞMA + EMİN OLMA
GERÇEK HAYATTA ÇALIŞACAK ŞEKİLDE İYİLEŞTİRİLMİŞ VERSİYON (DÜZELTİLMİŞ)

Bu modül bağımsızdır ve arama.py ile aynı mimariyi kullanır.
Hedef kaybolursa otomatik olarak aramaya dönülmesi için sinyal gönderir.

DÜZELTİLEN NOKTALAR:
  1. EMERGENCY_STOP_DISTANCE_M bloğu SAFE_STOP_DISTANCE_M'den (1.0m) sonra
     kontrol ediliyordu; 0.5m her zaman 1.0m şartını da sağladığından acil
     durma bloğuna asla sıra gelmiyordu (dead code). Sıra değiştirildi.
  2. "Emin olma" (CONFIRMING) mesafe kapısı işlevsizdi: target_confirmed
     zaten _update_detection_history içinde, mesafeden bağımsız olarak
     uzaktan da True oluyordu. Artık iki ayrı bayrak var:
       - target_confirmed: genel takip güveni (uzaktan da olabilir)
       - final_confirmed:  sadece CONFIRM_TRIGGER_DISTANCE_M içinde,
         CONFIRM_HOLD_SEC kadar KESİNTİSİZ tutulursa açılır ve asıl
         "dur/çarp" kararı buna bakar.
     CONFIRM_HOLD_SEC artık gerçekten kullanılıyor.
  3. TARGET_LOST_TOLERANCE_SEC tanımlı ama hiç kullanılmıyordu; artık
     gerçek zaman + kare sayısı birlikte (ilk tetiklenen kazanır) kontrol
     ediliyor -> sabit 10Hz varsayımına daha az bağımlı.
  4. Acil durma bloğu state'i set edip bir sonraki tick'e bırakıyordu;
     normal STOPPING ile tutarlı olacak şekilde aynı tick'te uygulanıyor.
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
SAFE_STOP_DISTANCE_M = 1.0
EMERGENCY_STOP_DISTANCE_M = 0.5
CONFIRM_TRIGGER_DISTANCE_M = 3.0

# --- EMİN OLMA (CONFIRMATION) ---
MIN_CONSECUTIVE_DETECTIONS = 5
CONFIRM_HOLD_SEC = 1.5
DETECTION_HISTORY_SIZE = 10

# --- HEDEF KAYBI TOLERANSI ---
TARGET_LOST_TOLERANCE_SEC = 0.8
MAX_LOST_FRAMES = 8

# --- PID KATSAYILARI (YÖN KONTROLÜ) ---
YAW_KP = 0.035
YAW_KI = 0.001
YAW_KD = 0.008
YAW_I_LIMIT = 0.15
MAX_ANGULAR_Z = 0.5

# --- HIZ KONTROLÜ ---
SURGE_MAX_LINEAR_X = 0.55
SURGE_MIN_LINEAR_X = 0.12
SURGE_SLOWDOWN_START_M = 5.0

# --- RÜZGAR/AKINTI TELAFİSİ ---
DRIFT_KP = 0.04
MAX_DRIFT_COMPENSATION = 0.12

# --- KOMUT YUMUŞATMA ---
MAX_ANGULAR_STEP = 0.06
MAX_LINEAR_STEP = 0.05

# --- MESAFE YUMUŞATMA ---
DISTANCE_SMOOTH_WINDOW = 5


class ApproachState(Enum):
    TRACKING = auto()
    CONFIRMING = auto()
    TARGET_LOST_WAIT = auto()
    STOPPING = auto()
    DONE = auto()
    LOST = auto()


class YaklasmaGorevi:
    """Yaklaşma ve emin olma görevi - GERÇEK HAYAT İÇİN OPTİMİZE"""

    def __init__(self, node, mission_topics, target_class):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class

        self.state = ApproachState.TRACKING
        self.finished = False
        self.target_lost = False
        self.approach_start_time = None

        self.detection_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.consecutive_detections = 0
        self.last_detection_frame = 0
        self.frame_counter = 0

        # YENİ: iki ayrı bayrak. target_confirmed genel takip güveni,
        # final_confirmed ise yalnızca yakın mesafede + hold süresi
        # boyunca kesintisiz tutulan gerçek "emin olma" onayı.
        self.target_confirmed = False
        self.final_confirmed = False
        self.confirm_start_time = None

        self.distance_buffer = deque(maxlen=DISTANCE_SMOOTH_WINDOW)

        self.yaw_integral = 0.0
        self.yaw_prev_error = None
        self.prev_tick_time = None

        self.lost_since = None
        self.lost_frame_count = 0

        self.last_angular_z = 0.0
        self.last_linear_x = 0.0

        self.stopping_since = None
        self.reverse_start_time = None
        self.reverse_active = False

        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0

        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0

        self.total_detections = 0
        self.total_frames = 0

        self.logger.info(f"[YAKLAŞMA] Başlatıldı, hedef: {self.target_class}")

    # --------------------------------------------------------
    def update_gps(self, lat, lon, heading):
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

    # --------------------------------------------------------
    def update_imu(self, gyro_z, accel_x, accel_y):
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
        self.final_confirmed = False
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
        return self.state == ApproachState.LOST or self.target_lost

    # --------------------------------------------------------
    def _select_target(self, detections):
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

        return min(candidates, key=lambda d: d["distance"])

    # --------------------------------------------------------
    def _update_detection_history(self, target):
        """Tespit geçmişini güncelle ve genel takip güvenini
        (target_confirmed) hesapla. Bu, yalnızca 'bu nesneye güvenerek
        peşinden gidebilirim miyim' sorusuna cevap verir; asıl güvenlik
        kararı (dur/çarp) final_confirmed'e bakar, o da update() içinde
        mesafe + hold süresiyle ayrıca kontrol edilir."""
        self.frame_counter += 1
        self.total_frames += 1

        if target is not None:
            self.total_detections += 1
            self.consecutive_detections += 1
            self.detection_history.append(True)
            self.lost_frame_count = 0
            self.lost_since = None

            if self.consecutive_detections >= MIN_CONSECUTIVE_DETECTIONS:
                if not self.target_confirmed:
                    self.target_confirmed = True
                    self.logger.info(
                        f"[TAKİP] {MIN_CONSECUTIVE_DETECTIONS} ardışık tespit, "
                        f"hedef takibe alındı (henüz nihai onay değil)."
                    )
                return True
        else:
            self.detection_history.append(False)
            self.consecutive_detections = 0

            if self.target_confirmed:
                self.lost_frame_count += 1
                if self.lost_since is None:
                    self.lost_since = time.monotonic()
                    self.logger.warn("[YAKLAŞMA] Hedef kayboldu, tolerans başladı...")

        return self.target_confirmed and target is not None

    # --------------------------------------------------------
    def _smoothed_distance(self, raw_distance):
        self.distance_buffer.append(raw_distance)
        return sum(self.distance_buffer) / len(self.distance_buffer)

    # --------------------------------------------------------
    def _compute_yaw_command(self, angle_error_deg, dt):
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
        comp = -DRIFT_KP * self.accel_y
        return max(-MAX_DRIFT_COMPENSATION, min(MAX_DRIFT_COMPENSATION, comp))

    # --------------------------------------------------------
    def _rate_limit(self, target, last, max_step):
        delta = target - last
        if delta > max_step:
            return last + max_step
        if delta < -max_step:
            return last - max_step
        return target

    # --------------------------------------------------------
    def _start_stopping(self, now):
        self.state = ApproachState.STOPPING
        self.stopping_since = now
        self.reverse_active = True
        self.reverse_start_time = now
        self.logger.info("[YAKLAŞMA] Güvenli mesafe, durma başlıyor!")

    # --------------------------------------------------------
    def _do_stopping(self, now):
        if self.reverse_active:
            reverse_duration = 0.5
            if now - self.reverse_start_time < reverse_duration:
                publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=-0.15, angular_z=0.0)
                return False

            self.reverse_active = False
            self.logger.info("[YAKLAŞMA] Ters itki tamamlandı, duruluyor...")

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

        if self.approach_start_time is None:
            self.approach_start_time = now

        if self.state == ApproachState.STOPPING:
            return self._do_stopping(now)

        if self.state in (ApproachState.DONE, ApproachState.LOST):
            return True

        target = self._select_target(detections)
        self._update_detection_history(target)

        # --- HEDEF KAYBI KONTROLÜ ---
        # DÜZELTME: artık hem gerçek geçen süre (TARGET_LOST_TOLERANCE_SEC)
        # hem de kare sayısı (MAX_LOST_FRAMES) birlikte kontrol ediliyor;
        # hangisi önce dolarsa aramaya dönülüyor. Böylece sabit 10Hz tick
        # varsayımına daha az bağımlı oluyoruz.
        if target is None:
            if self.target_confirmed:
                lost_elapsed = (now - self.lost_since) if self.lost_since else 0.0
                if self.lost_frame_count >= MAX_LOST_FRAMES or lost_elapsed >= TARGET_LOST_TOLERANCE_SEC * 4:
                    # Not: TARGET_LOST_TOLERANCE_SEC kısa bir tolerans
                    # penceresidir (coast/yumuşatma için); gerçek "vazgeç"
                    # kararını MAX_LOST_FRAMES ile birlikte, ondan biraz
                    # daha uzun bir süre penceresinde veriyoruz.
                    self.logger.error(
                        f"[YAKLAŞMA] Hedef {self.lost_frame_count} karedir / "
                        f"{lost_elapsed:.1f}sn'dir kayıp! Aramaya dönülüyor."
                    )
                    stop_vehicle(self.topics.cmd_vel_pub)
                    self.state = ApproachState.LOST
                    self.target_lost = True
                    return True

                coast_linear = self.last_linear_x * 0.7
                coast_angular = self.last_angular_z * 0.7
                publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=coast_linear, angular_z=coast_angular)
                self.last_linear_x = coast_linear
                self.last_angular_z = coast_angular
                return False

            stop_vehicle(self.topics.cmd_vel_pub)
            self.last_linear_x = 0.0
            self.last_angular_z = 0.0
            return False

        # --- HEDEF GÖRÜNÜYOR ---
        raw_distance = target["distance"]
        angle_error = target["Buoy angle: "]
        distance = self._smoothed_distance(raw_distance)

        # --- EMİN OLMA (CONFIRMING) — nihai onay burada veriliyor ---
        if not self.final_confirmed and distance <= CONFIRM_TRIGGER_DISTANCE_M:
            if self.state != ApproachState.CONFIRMING:
                self.state = ApproachState.CONFIRMING
                self.confirm_start_time = None
                self.logger.info(
                    f"[EMİN OLMA] {CONFIRM_TRIGGER_DISTANCE_M:.1f}m içinde, "
                    f"doğrulama başlıyor..."
                )

            if self.consecutive_detections >= MIN_CONSECUTIVE_DETECTIONS:
                if self.confirm_start_time is None:
                    self.confirm_start_time = now

                held = now - self.confirm_start_time
                if held >= CONFIRM_HOLD_SEC:
                    self.final_confirmed = True
                    self.state = ApproachState.TRACKING
                    self.logger.info(
                        f"[EMİN OLMA] ✅ Hedef nihai onaylandı! "
                        f"({held:.1f}sn kesintisiz tutuldu)"
                    )
                else:
                    yaw_cmd = self._compute_yaw_command(angle_error, dt)
                    angular_z = self._rate_limit(yaw_cmd, self.last_angular_z, MAX_ANGULAR_STEP)
                    publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=angular_z)
                    self.last_angular_z = angular_z
                    self.last_linear_x = 0.0
                    self.logger.info(
                        f"[EMİN OLMA] Tutuluyor... ({held:.1f}/{CONFIRM_HOLD_SEC:.1f}sn)",
                        throttle_duration_sec=0.5
                    )
                    return False
            else:
                # Ardışık tespit zinciri bozuldu -> hold süresi sıfırlanır
                self.confirm_start_time = None
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

        # --- ACİL DURMA (EMERGENCY STOP) ---
        # DÜZELTME: bu kontrol artık SAFE_STOP_DISTANCE_M kontrolünden ÖNCE
        # yapılıyor. Eskiden 0.5m her zaman 1.0m şartını da sağladığından
        # bu blok hiçbir zaman çalışmıyordu (dead code).
        if distance <= EMERGENCY_STOP_DISTANCE_M:
            self.logger.error(f"[ACİL] Çok yakın! {distance:.2f}m, acil dur!")
            self._start_stopping(now)
            return self._do_stopping(now)

        # --- GÜVENLİ MESAFE KONTROLÜ ---
        if distance <= SAFE_STOP_DISTANCE_M:
            if self.final_confirmed:
                self._start_stopping(now)
                return self._do_stopping(now)
            else:
                self.logger.warn(
                    f"[GÜVENLİK] Hedef nihai olarak doğrulanmadı ama {distance:.2f}m "
                    f"yakınında! Durduruluyor, aramaya dönülüyor."
                )
                stop_vehicle(self.topics.cmd_vel_pub)
                self.state = ApproachState.LOST
                self.target_lost = True
                return True

        # --- PID KONTROL (NORMAL YAKLAŞMA) ---
        yaw_cmd = self._compute_yaw_command(angle_error, dt)
        yaw_cmd += self._drift_compensation()
        angular_z = self._rate_limit(yaw_cmd, self.last_angular_z, MAX_ANGULAR_STEP)

        surge_cmd = self._compute_surge_command(distance)
        linear_x = self._rate_limit(surge_cmd, self.last_linear_x, MAX_LINEAR_STEP)

        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=linear_x, angular_z=angular_z)
        self.last_angular_z = angular_z
        self.last_linear_x = linear_x

        return False

    # --------------------------------------------------------
    def get_status(self):
        return {
            "state": self.state.name,
            "finished": self.finished,
            "target_lost": self.target_lost,
            "target_confirmed": self.target_confirmed,
            "final_confirmed": self.final_confirmed,
            "consecutive_detections": self.consecutive_detections,
            "distance": list(self.distance_buffer)[-1] if self.distance_buffer else None,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "detection_rate": self.total_detections / max(1, self.total_frames)
        }