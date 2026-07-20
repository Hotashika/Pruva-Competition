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

from utils.mavlink_utilities import publish_cmd_vel, stop_vehicle

SEARCH_STEP_DEG = 20.0
STEP_HOLD_SEC = 5.0
TURN_TIMEOUT_SEC = 12.0
HEADING_TOLERANCE_DEG = 3.0
TURN_LINEAR_X = 0.0  # Fark itkili (skid-steer) tekne yerinde döner; ileri hıza gerek yok.
MAX_YAW_OFFSET_RAD = 0.18
TURN_MAX_RETRIES = 3
FULL_SCAN_STEPS = 18
SEARCH_CONFIRM_FRAMES = 5
SEARCH_CONFIRM_WINDOW_SEC = 5.0
MAX_CONFIRM_ANGLE_SPREAD_DEG = 18.0
MAX_CONFIRM_DISTANCE_RATIO = 0.45
SEARCH_MAX_TRACK_ANGLE_JUMP_DEG = 30.0
SEARCH_MAX_TRACK_DISTANCE_RATIO = 0.60
class SearchState(Enum):
    START_STEP = auto()
    TURNING = auto()
    HOLDING = auto()
    TARGET_FOUND = auto()
    FAILED = auto()


class AramaGorevi:
    def __init__(self, node, mission_topics, target_class, test_mode=False):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class
        self.test_mode = test_mode
        self.state = SearchState.START_STEP
        self.finished = False
        self.failed = False
        self.found_target = None
        self.current_lat = self.current_lon = self.current_heading_deg = None
        self.home_lat = self.home_lon = None
        self.step_target_heading = self.step_start_time = self.hold_until = None
        self.completed_steps = 0
        self.turn_retry_count = 0
        self.last_processed_frame_id = None
        self.confirmations = deque(maxlen=SEARCH_CONFIRM_FRAMES)

    def update_gps(self, lat, lon, heading_deg=None):
        self.current_lat, self.current_lon = float(lat), float(lon)
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
        self.step_target_heading = (self.current_heading_deg + SEARCH_STEP_DEG) % 360.0
        self.step_start_time = now
        self.state = SearchState.TURNING

    def _turn(self, now):
        error_deg = self._angle_error(self.step_target_heading, self.current_heading_deg)
        if abs(error_deg) <= HEADING_TOLERANCE_DEG:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.completed_steps += 1
            self.turn_retry_count = 0
            self.hold_until = now + STEP_HOLD_SEC
            self.state = SearchState.HOLDING
            self.confirmations.clear()
            self.logger.info(f"[ARAMA] {self.completed_steps}/{FULL_SCAN_STEPS} adım tamamlandı; 5 sn tarama.")
            return
        if now - self.step_start_time > TURN_TIMEOUT_SEC:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.turn_retry_count += 1
            if self.turn_retry_count > TURN_MAX_RETRIES:
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
        yaw_offset = max(-MAX_YAW_OFFSET_RAD, min(MAX_YAW_OFFSET_RAD, math.radians(error_deg)))
        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=TURN_LINEAR_X, angular_z=yaw_offset)

    def update(self, detections, frame_id=None):
        if self.finished or self.failed:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return self.finished
        now = time.monotonic()
        # Dönüş sırasındaki bulanık/değişken kareler hedef onayına
        # katılmaz. Kamera yalnızca motor komutu sıfırken, 5 saniyelik
        # sabit bakış penceresinde değerlendirilir.
        if self.state == SearchState.HOLDING and self._process_camera_frame(detections, frame_id, now):
            return True
        if None in (self.current_heading_deg, self.current_lat, self.current_lon):
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return False
        if self.state == SearchState.START_STEP:
            # 360 derece tamamlanınca aynı gerçek konumda yeni bir 360 derece
            # tarama turuna başla. Konum/tespit verisi uydurulmaz ve GPS hedefi
            # üretilmez; her bakış açısında en fazla 5 saniye kalınır.
            if self.completed_steps >= FULL_SCAN_STEPS:
                self.completed_steps = 0
                self.logger.info("[ARAMA] 360° tarama tamamlandı; yeni tarama turu başlıyor.")
            self._start_turn(now)
        elif self.state == SearchState.TURNING:
            self._turn(now)
        elif self.state == SearchState.HOLDING:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            if now >= self.hold_until:
                self.state = SearchState.START_STEP
        return False

    def reset_search(self):
        stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
        self.state = SearchState.START_STEP
        self.finished = self.failed = False
        self.found_target = None
        self.completed_steps = 0
        self.step_target_heading = self.step_start_time = self.hold_until = None
        self.turn_retry_count = 0
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
        }
