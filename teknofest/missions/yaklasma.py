#!/usr/bin/env python3
"""Task 3 yaklaşma: yalnızca gerçek kamera, GPS ve heading verisi kullanır."""

import math
import time
from enum import Enum, auto

from utils.mavlink_utilities import calculate_gps_distance, publish_cmd_vel, stop_vehicle

REQUIRED_DISTINCT_FRAMES = 5
TARGET_LOST_TIMEOUT_SEC = 1.0
CONFIRM_WINDOW_SEC = 5.0
SEGMENT_FRACTION = 1.0 / 3.0
MIN_SEGMENT_M = 0.40
MAX_SEGMENT_M = 6.0
IMPACT_ENTRY_DISTANCE_M = 1.5
DISTANCE_CONSISTENCY_RATIO = 0.30
MAX_CONFIRM_ANGLE_SPREAD_DEG = 18.0
ALIGN_TOLERANCE_DEG = 4.0
ANGLE_KP = 0.02
MAX_ANGULAR_Z = 0.30
APPROACH_SPEED = 0.30
SEGMENT_TIMEOUT_SEC = 12.0
STALL_TIMEOUT_SEC = 4.0
MIN_GPS_PROGRESS_M = 0.25
MAX_TRACK_ANGLE_JUMP_DEG = 30.0
MAX_TRACK_DISTANCE_RATIO = 0.60
MAX_APPROACH_SEGMENTS = 8
APPROACH_TOTAL_TIMEOUT_SEC = 60.0


class ApproachState(Enum):
    CONFIRMING_TARGET = auto()
    ALIGNING = auto()
    MOVING_STRAIGHT = auto()
    CONFIRMING_RESULT = auto()
    DONE = auto()
    LOST = auto()


class YaklasmaGorevi:
    def __init__(self, node, mission_topics, target_class, safe_stop_distance=None):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class
        self.impact_entry_distance = float(safe_stop_distance or IMPACT_ENTRY_DISTANCE_M)
        self.current_lat = self.current_lon = self.current_heading = None
        self.state = ApproachState.CONFIRMING_TARGET
        self.finished = False
        self.target_lost = False
        self.latest_target = None
        self.last_seen_time = None
        self.last_processed_frame_id = None
        self.confirmations = []
        self.confirmed_distance = None
        self.confirmed_angle = None
        self.segment_goal_m = None
        self.segment_start_lat = self.segment_start_lon = None
        self.segment_start_time = self.last_progress_time = None
        self.best_travelled = 0.0
        self.approach_start_time = None
        self.segment_count = 0

    def update_gps(self, lat, lon, heading=None):
        self.current_lat, self.current_lon = float(lat), float(lon)
        if heading is not None:
            self.update_heading(heading)

    def update_heading(self, heading):
        self.current_heading = float(heading) % 360.0

    def update_imu(self, gyro_z, accel_x, accel_y):
        # Yaklaşma kararında sentetik IMU tahmini kullanılmaz.
        return None

    def _select_target(self, detections):
        candidates = []
        reference = self.latest_target
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
                    old_distance = float(reference["distance"])
                    old_angle = float(reference["Buoy angle: "])
                    if abs(angle - old_angle) > MAX_TRACK_ANGLE_JUMP_DEG:
                        continue
                    if abs(distance - old_distance) > max(0.8, old_distance * MAX_TRACK_DISTANCE_RATIO):
                        continue
                candidates.append((confidence, det))
            except (KeyError, TypeError, ValueError):
                continue
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _clear_confirmation(self):
        self.confirmations.clear()

    def _process_frame(self, detections, frame_id, now):
        if frame_id is None or frame_id == self.last_processed_frame_id:
            return
        self.last_processed_frame_id = frame_id
        target = self._select_target(detections)
        if target is None:
            self._clear_confirmation()
            return
        self.latest_target = target
        self.last_seen_time = now
        if self.state not in (ApproachState.CONFIRMING_TARGET, ApproachState.CONFIRMING_RESULT):
            return
        self.confirmations.append((now, frame_id, float(target["distance"]), float(target["Buoy angle: "])))
        self.confirmations = [item for item in self.confirmations if now - item[0] <= CONFIRM_WINDOW_SEC]
        if len(self.confirmations) >= REQUIRED_DISTINCT_FRAMES:
            self._finish_confirmation(now)

    def _finish_confirmation(self, now):
        samples = self.confirmations[-REQUIRED_DISTINCT_FRAMES:]
        distances = [item[2] for item in samples]
        angles = [item[3] for item in samples]
        mean_distance = sum(distances) / len(distances)
        mean_angle = sum(angles) / len(angles)
        distance_ok = max(distances) - min(distances) <= max(0.30, mean_distance * DISTANCE_CONSISTENCY_RATIO)
        angle_ok = max(angles) - min(angles) <= MAX_CONFIRM_ANGLE_SPREAD_DEG
        self._clear_confirmation()
        if not (distance_ok and angle_ok):
            self._lose_target("5 farklı kamera karesi tutarlı değil")
            return
        if self.state == ApproachState.CONFIRMING_RESULT and mean_distance <= self.impact_entry_distance:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.state = ApproachState.DONE
            self.finished = True
            self.logger.info("[YAKLAŞMA] 5 farklı kare ile çarpma mesafesi doğrulandı.")
            return
        if self.segment_count >= MAX_APPROACH_SEGMENTS:
            self._lose_target("azami yaklaşma segmentine ulaşıldı")
            return
        self.confirmed_distance = mean_distance
        self.confirmed_angle = mean_angle
        self.segment_goal_m = max(MIN_SEGMENT_M, min(MAX_SEGMENT_M, mean_distance * SEGMENT_FRACTION))
        self.state = ApproachState.ALIGNING
        self.logger.info(
            f"[YAKLAŞMA] 5 kare ortalaması: mesafe={mean_distance:.2f}m, "
            f"açı={mean_angle:.1f}°; düz ilerleme={self.segment_goal_m:.2f}m."
        )

    def _align(self, now):
        if self.latest_target is None:
            return
        angle = float(self.latest_target["Buoy angle: "])
        if abs(angle) <= ALIGN_TOLERANCE_DEG:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.segment_count += 1
            self.segment_start_lat, self.segment_start_lon = self.current_lat, self.current_lon
            self.segment_start_time = self.last_progress_time = now
            self.best_travelled = 0.0
            self.state = ApproachState.MOVING_STRAIGHT
            return
        angular = max(-MAX_ANGULAR_Z, min(MAX_ANGULAR_Z, ANGLE_KP * angle))
        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=angular)

    def _move_straight(self, now):
        if None in (self.current_lat, self.current_lon, self.segment_start_lat, self.segment_start_lon):
            self._lose_target("gerçek GPS ilerlemesi alınamadı")
            return
        travelled = calculate_gps_distance(
            self.segment_start_lat, self.segment_start_lon, self.current_lat, self.current_lon
        )
        if travelled > self.best_travelled + MIN_GPS_PROGRESS_M:
            self.best_travelled = travelled
            self.last_progress_time = now
        if travelled >= self.segment_goal_m:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            self.state = ApproachState.CONFIRMING_RESULT
            self._clear_confirmation()
            return
        if now - self.segment_start_time > SEGMENT_TIMEOUT_SEC or now - self.last_progress_time > STALL_TIMEOUT_SEC:
            self._lose_target("GPS ile düz ilerleme doğrulanamadı")
            return
        # Hizalama tamamlandıktan sonra segment boyunca yalnızca düz git.
        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=APPROACH_SPEED, angular_z=0.0)

    def _lose_target(self, reason):
        stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
        self.state = ApproachState.LOST
        self.target_lost = True
        self.finished = False
        self.logger.warning(f"[YAKLAŞMA] {reason}; aramaya dönülüyor.")

    def update(self, detections, frame_id=None):
        now = time.monotonic()
        if self.approach_start_time is None:
            self.approach_start_time = now
        self._process_frame(detections, frame_id, now)
        if self.state in (ApproachState.DONE, ApproachState.LOST):
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
            return self.state == ApproachState.DONE
        if now - self.approach_start_time > APPROACH_TOTAL_TIMEOUT_SEC:
            self._lose_target("yaklaşma toplam süre sınırını aştı")
            return False
        if self.last_seen_time is None or now - self.last_seen_time > TARGET_LOST_TIMEOUT_SEC:
            self._lose_target("hedef kamerada kayboldu")
            return False
        if self.state == ApproachState.ALIGNING:
            self._align(now)
        elif self.state == ApproachState.MOVING_STRAIGHT:
            self._move_straight(now)
        else:
            stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
        return self.finished

    def should_return_to_search(self):
        return self.state == ApproachState.LOST

    def reset_approach(self):
        stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
        node, topics, target, distance = self.node, self.topics, self.target_class, self.impact_entry_distance
        self.__init__(node, topics, target, safe_stop_distance=distance)


    def get_status(self):
        return {
            "state": self.state.name,
            "finished": self.finished,
            "distance": None if self.latest_target is None else self.latest_target.get("distance"),
            "segment_count": self.segment_count,
        }
