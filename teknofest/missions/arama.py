#!/usr/bin/env python3
"""Task 3 arama: gerçek heading/GPS/kamera ile 20° adımlı 360° tarama.

v6 değişiklikleri:
  - Tekne fark itkili (skid-steer) olduğundan tarama dönüşü artık YERİNDE
    yapılıyor (TURN_LINEAR_X = 0.0). Eskiden dönüş sırasında ileri hız da
    veriliyordu (TURN_THRUST=0.18); bu, 18 adımlık tam tur boyunca home
    noktasından fark edilir sürüklenmeye yol açabiliyordu.
  - Bir arama adımının dönüşü TURN_TIMEOUT_SEC içinde art arda
    TURN_MAX_RETRIES kez doğrulanamazsa artık sonsuz döngüye girmek yerine
    SearchState.FAILED'e düşülüyor (relocation'daki retry mantığıyla
    tutarlı).
  - Hedef seçiminde artık yaklaşma/çarpma modüllerindeki gibi kare-kare
    süreklilik filtresi var: bir önceki onaylı kareye göre açı/mesafede ani
    sıçrama gösteren tespitler (örn. yakında duran farklı bir aynı renkli
    duba) elenip karışma riski azaltılıyor.
"""
import math
import time
from collections import deque
from enum import Enum, auto

from utils.mavlink_utilities import (
    calculate_bearing,
    calculate_gps_distance,
    publish_cmd_vel,
    stop_vehicle,
)

SEARCH_STEP_DEG = 20.0
STEP_HOLD_SEC = 5.0
TURN_TIMEOUT_SEC = 12.0
HEADING_TOLERANCE_DEG = 3.0
HOLD_MAX_HEADING_DRIFT_DEG = 3.0
# Skid-steer mikserinde saat yönü steering ile birlikte küçük pozitif taban
# itki, dıştaki (sol) motoru ileri sürüp içteki (sağ) motoru durdurmak içindir.
# Saf thrust=0 komutu sahada iki ESC tarafından aynı yönde uygulanmıştı.
TURN_LINEAR_X = 0.18
MAX_YAW_OFFSET_RAD = 0.18
TURN_MAX_RETRIES = 3
TURN_PROGRESS_TIMEOUT_SEC = 3.0
TURN_MIN_PROGRESS_DEG = 2.0
FULL_SCAN_STEPS = 18
RELOCATION_DISTANCE_M = 2.0
RELOCATION_THRUST = 0.20
RELOCATION_TIMEOUT_SEC = 20.0
RELOCATION_CONFIRM_GPS_SAMPLES = 3
RELOCATION_MAX_LATERAL_M = 1.0
SEARCH_CONFIRM_FRAMES = 5
SEARCH_CONFIRM_WINDOW_SEC = 5.0
MAX_CONFIRM_ANGLE_SPREAD_DEG = 18.0
MAX_CONFIRM_DISTANCE_RATIO = 0.45
SEARCH_MAX_TRACK_ANGLE_JUMP_DEG = 30.0
SEARCH_MAX_TRACK_DISTANCE_RATIO = 0.60
DEFAULT_MIN_TARGET_CONFIDENCE = 0.65
class SearchState(Enum):
    START_STEP = auto()
    TURNING = auto()
    HOLDING = auto()
    RELOCATING = auto()
    TARGET_FOUND = auto()
    FAILED = auto()


class AramaGorevi:
    def __init__(self, node, mission_topics, target_class, test_mode=False,
                 min_target_confidence=DEFAULT_MIN_TARGET_CONFIDENCE):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class
        self.test_mode = test_mode
        self.min_target_confidence = float(min_target_confidence)
        self.state = SearchState.START_STEP
        self.finished = False
        self.failed = False
        self.found_target = None
        self.current_lat = self.current_lon = self.current_heading_deg = None
        self.home_lat = self.home_lon = None
        self.step_target_heading = self.step_start_time = self.hold_until = None
        self.turn_start_heading_deg = None
        self.turn_start_error_deg = None
        self.scan_origin_heading_deg = None
        self.completed_steps = 0
        self.turn_retry_count = 0
        self.gps_update_sequence = 0
        self.relocation_start_lat = self.relocation_start_lon = None
        self.relocation_heading_deg = self.relocation_start_time = None
        self.relocation_confirm_count = 0
        self.relocation_last_checked_gps_sequence = None
        self.last_processed_frame_id = None
        self.confirmations = deque(maxlen=SEARCH_CONFIRM_FRAMES)

    def update_gps(self, lat, lon, heading_deg=None):
        self.current_lat, self.current_lon = float(lat), float(lon)
        self.gps_update_sequence += 1
        if heading_deg is not None:
            self.update_heading(heading_deg)
        if self.home_lat is None:
            self.home_lat, self.home_lon = self.current_lat, self.current_lon

    def update_heading(self, heading_deg):
        self.current_heading_deg = float(heading_deg) % 360.0

    @staticmethod
    def _angle_error(target_deg, current_deg):
        return (target_deg - current_deg + 180.0) % 360.0 - 180.0

    def _select_target(self, detections):
        # Süreklilik filtresi: bir önceki onaylı kareye göre ani sıçrayan
        # (başka bir cisme ait olabilecek) tespitleri ele.
        reference = self.confirmations[-1][2] if self.confirmations else None
        valid = []
        for det in detections or []:
            try:
                if det.get("class") != self.target_class:
                    continue
                distance = float(det["distance"])
                angle = float(det["Buoy angle: "])
                confidence = float(det.get("confidence", 0.0))
                if confidence < self.min_target_confidence:
                    continue
                if not (math.isfinite(distance) and distance > 0 and math.isfinite(angle)):
                    continue
                if reference is not None:
                    ref_distance = float(reference["distance"])
                    ref_angle = float(reference["Buoy angle: "])
                    if abs(angle - ref_angle) > SEARCH_MAX_TRACK_ANGLE_JUMP_DEG:
                        continue
                    if abs(distance - ref_distance) > max(0.8, ref_distance * SEARCH_MAX_TRACK_DISTANCE_RATIO):
                        continue
                valid.append((confidence, det))
            except (KeyError, TypeError, ValueError):
                continue
        return max(valid, key=lambda item: item[0])[1] if valid else None

    def _process_camera_frame(self, detections, frame_id, now):
        if frame_id is None or frame_id == self.last_processed_frame_id:
            return False
        self.last_processed_frame_id = frame_id
        target = self._select_target(detections)
        if target is None:
            self.confirmations.clear()
            return False
        self.confirmations.append((now, frame_id, target))
        while self.confirmations and now - self.confirmations[0][0] > SEARCH_CONFIRM_WINDOW_SEC:
            self.confirmations.popleft()
        if len(self.confirmations) < SEARCH_CONFIRM_FRAMES:
            return False
        distances = [float(item[2]["distance"]) for item in self.confirmations]
        angles = [float(item[2]["Buoy angle: "]) for item in self.confirmations]
        mean_distance = sum(distances) / len(distances)
        distance_ok = max(distances) - min(distances) <= max(0.4, mean_distance * MAX_CONFIRM_DISTANCE_RATIO)
        angle_ok = max(angles) - min(angles) <= MAX_CONFIRM_ANGLE_SPREAD_DEG
        if not (distance_ok and angle_ok):
            self.confirmations.clear()
            return False
        self.found_target = self.confirmations[-1][2]
        self.finished = True
        self.state = SearchState.TARGET_FOUND
        stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
        self.logger.info("[ARAMA] Hedef 5 farklı kamera karesinde doğrulandı; yaklaşmaya geçiliyor.")
        return True

    def _start_turn(self, now):
        if self.current_heading_deg is None:
            return
        # Hedefleri her seferinde mevcut heading uzerinden kurmak, 3 derecelik
        # toleransi 18 adim boyunca biriktirip eksik bir "360 derece" taramaya
        # yol acar. Tum adimlar ilk tarama basligina sabitlenir.
        if self.scan_origin_heading_deg is None:
            self.scan_origin_heading_deg = self.current_heading_deg
        self.step_target_heading = (
            self.scan_origin_heading_deg
            + (self.completed_steps + 1) * SEARCH_STEP_DEG
        ) % 360.0
        self.turn_start_heading_deg = self.current_heading_deg
        self.turn_start_error_deg = abs(
            self._angle_error(self.step_target_heading, self.current_heading_deg)
        )
        self.step_start_time = now
        self.state = SearchState.TURNING

    def _turn(self, now):
        error_deg = self._angle_error(self.step_target_heading, self.current_heading_deg)
        if abs(error_deg) <= HEADING_TOLERANCE_DEG:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.turn_retry_count = 0
            self.hold_until = now + STEP_HOLD_SEC
            self.state = SearchState.HOLDING
            self.confirmations.clear()
            self.logger.info(
                f"[ARAMA] {self.completed_steps + 1}/{FULL_SCAN_STEPS} açıya ulaşıldı; "
                "araç durdu ve 5 sn tarama başladı."
            )
            return
        if now - self.step_start_time > TURN_TIMEOUT_SEC:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.turn_retry_count += 1
            if self.turn_retry_count >= TURN_MAX_RETRIES:
                self.failed = True
                self.state = SearchState.FAILED
                self.logger.error(
                    f"[ARAMA] {TURN_MAX_RETRIES} denemede 20° dönüş doğrulanamadı; arama güvenli biçimde durduruldu."
                )
                return
            self.state = SearchState.START_STEP
            self.logger.error(
                f"[ARAMA] 20° dönüş doğrulanamadı ({self.turn_retry_count}/{TURN_MAX_RETRIES}); aynı adım yeniden denenecek."
            )
            return
        if now - self.step_start_time >= TURN_PROGRESS_TIMEOUT_SEC:
            remaining_error_deg = abs(
                self._angle_error(self.step_target_heading, self.current_heading_deg)
            )
            error_reduction_deg = self.turn_start_error_deg - remaining_error_deg
            if error_reduction_deg < TURN_MIN_PROGRESS_DEG:
                stop_vehicle(self.topics.cmd_vel_pub, repeat_count=2)
                self.failed = True
                self.state = SearchState.FAILED
                self.logger.error(
                    "[ARAMA] Dönüş komutuna rağmen heading 3 sn içinde "
                    f"hedefe doğru {TURN_MIN_PROGRESS_DEG:.1f}° ilerlemedi "
                    f"(hata azalması={error_reduction_deg:.1f}°); motorlar durduruldu."
                )
                return
        yaw_offset = max(-MAX_YAW_OFFSET_RAD, min(MAX_YAW_OFFSET_RAD, math.radians(error_deg)))
        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=TURN_LINEAR_X, angular_z=yaw_offset)

    def _start_relocation(self, now):
        self.relocation_start_lat = self.current_lat
        self.relocation_start_lon = self.current_lon
        self.relocation_heading_deg = self.current_heading_deg
        self.relocation_start_time = now
        self.relocation_confirm_count = 0
        self.relocation_last_checked_gps_sequence = self.gps_update_sequence
        self.confirmations.clear()
        self.state = SearchState.RELOCATING
        self.logger.info(
            "[ARAMA] 360° taramada hedef bulunamadı; gerçek GPS ile "
            f"{RELOCATION_DISTANCE_M:.1f} m yeni konuma ilerleniyor."
        )

    def _relocate(self, now):
        if now - self.relocation_start_time > RELOCATION_TIMEOUT_SEC:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=2)
            self.failed = True
            self.state = SearchState.FAILED
            self.logger.error(
                f"[ARAMA] {RELOCATION_DISTANCE_M:.1f} m yer değiştirme "
                f"{RELOCATION_TIMEOUT_SEC:.0f} sn içinde GPS ile doğrulanamadı; "
                "araç güvenli biçimde durduruldu."
            )
            return
        distance_m = calculate_gps_distance(
            self.relocation_start_lat,
            self.relocation_start_lon,
            self.current_lat,
            self.current_lon,
        )
        movement_bearing_deg = calculate_bearing(
            self.relocation_start_lat,
            self.relocation_start_lon,
            self.current_lat,
            self.current_lon,
        )
        movement_angle_rad = math.radians(
            self._angle_error(movement_bearing_deg, self.relocation_heading_deg)
        )
        forward_progress_m = distance_m * math.cos(movement_angle_rad)
        lateral_offset_m = abs(distance_m * math.sin(movement_angle_rad))

        # Kontrol döngüsü GPS'ten hızlıdır; aynı GPS örneğini üç kez sayma.
        if self.gps_update_sequence != self.relocation_last_checked_gps_sequence:
            self.relocation_last_checked_gps_sequence = self.gps_update_sequence
            if (
                forward_progress_m >= RELOCATION_DISTANCE_M
                and lateral_offset_m <= RELOCATION_MAX_LATERAL_M
            ):
                self.relocation_confirm_count += 1
            else:
                self.relocation_confirm_count = 0

        if self.relocation_confirm_count >= RELOCATION_CONFIRM_GPS_SAMPLES:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=2)
            self.completed_steps = 0
            self.turn_retry_count = 0
            self.state = SearchState.START_STEP
            self.logger.info(
                f"[ARAMA] Yeni konum {self.relocation_confirm_count} ardışık "
                f"GPS örneğiyle doğrulandı (ileri={forward_progress_m:.2f} m, "
                f"yanal={lateral_offset_m:.2f} m); yeni 360° tarama başlıyor."
            )
            return

        heading_error_deg = self._angle_error(
            self.relocation_heading_deg,
            self.current_heading_deg,
        )
        yaw_offset = max(
            -MAX_YAW_OFFSET_RAD,
            min(MAX_YAW_OFFSET_RAD, math.radians(heading_error_deg)),
        )
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=RELOCATION_THRUST,
            angular_z=yaw_offset,
        )

    def update(self, detections, frame_id=None):
        if self.finished or self.failed:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return self.finished
        now = time.monotonic()
        if None in (self.current_heading_deg, self.current_lat, self.current_lon):
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return False
        if self.state == SearchState.START_STEP:
            # Bir tam turda hedef bulunmazsa aynı GPS konumunda ikinci turu
            # başlatma; gerçek GPS ile 2 m ilerleyip yeni konumda tekrar tara.
            if self.completed_steps >= FULL_SCAN_STEPS:
                self._start_relocation(now)
            else:
                self._start_turn(now)
        elif self.state == SearchState.TURNING:
            self._turn(now)
        elif self.state == SearchState.HOLDING:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            hold_error_deg = abs(
                self._angle_error(self.step_target_heading, self.current_heading_deg)
            )
            if hold_error_deg > HOLD_MAX_HEADING_DRIFT_DEG:
                # Dalga/atalet araci dondurmeye devam ederse hareketli kareleri
                # tarama onayina katma. Ayni mutlak 20 derece hedefine geri don.
                self.confirmations.clear()
                self.turn_start_heading_deg = self.current_heading_deg
                self.turn_start_error_deg = hold_error_deg
                self.step_start_time = now
                self.state = SearchState.TURNING
                self.logger.warning(
                    f"[ARAMA] 5 sn tarama sırasında heading {hold_error_deg:.1f}° "
                    "saptı; görüntüler reddedildi ve aynı açı düzeltiliyor."
                )
                return False
            # Dönüş sırasındaki bulanık/değişken kareler hedef onayına
            # katılmaz. Kamera yalnız motor komutu sıfırken ve heading sabitken
            # değerlendirilir.
            if self._process_camera_frame(detections, frame_id, now):
                return True
            if now >= self.hold_until:
                # Adımı hedef açıya ilk varışta değil, ancak 5 saniyelik sabit
                # görüş taraması bitince say. HOLD sapmasını düzeltmek böylece
                # aynı açıyı ikinci kez tamamlanmış gibi saymaz.
                self.completed_steps += 1
                self.state = SearchState.START_STEP
        elif self.state == SearchState.RELOCATING:
            self._relocate(now)
        return False

    def reset_search(self):
        stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
        self.state = SearchState.START_STEP
        self.finished = self.failed = False
        self.found_target = None
        self.completed_steps = 0
        self.step_target_heading = self.step_start_time = self.hold_until = None
        self.turn_start_heading_deg = None
        self.turn_start_error_deg = None
        self.scan_origin_heading_deg = None
        self.turn_retry_count = 0
        self.relocation_start_lat = self.relocation_start_lon = None
        self.relocation_heading_deg = self.relocation_start_time = None
        self.relocation_confirm_count = 0
        self.relocation_last_checked_gps_sequence = None
        self.last_processed_frame_id = None
        self.confirmations.clear()


    def should_fail(self):
        return self.failed

    def get_search_status(self):
        return {
            "state": self.state.name,
            "finished": self.finished,
            "failed": self.failed,
            "completed_steps": self.completed_steps,
            "turn_retry_count": self.turn_retry_count,
            "relocation_distance_m": RELOCATION_DISTANCE_M,
            "relocation_confirm_count": self.relocation_confirm_count,
        }
