"""TEKNOFEST Task 3: confirmed buoy search, approach, and repeated impact."""

import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_TEXT = str(REPO_ROOT)
while REPO_ROOT_TEXT in sys.path:
    sys.path.remove(REPO_ROOT_TEXT)
sys.path.insert(0, REPO_ROOT_TEXT)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from utils.mavlink_utilities import (
    calculate_gps_distance,
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    create_mission_topics,
    parse_bridge_state,
    publish_cmd_vel,
    stop_vehicle,
    wait_for_mission_services,
)
from teknofest.missions.utils.task3_impact_controller import (
    ImpactAction,
    Task3ImpactController,
)
from teknofest.missions.utils.task3_search_controller import (
    SearchPhase,
    Task3SearchController,
    angle_error_deg,
)
from teknofest.missions.utils.task3_targeting import (
    filter_target,
    median,
    select_target,
    target_is_consistent,
)

# ============================================================
# GÖREV / NAVİGASYON SABİTLERİ
# ============================================================
DRIVE_MODE = "GUIDED"
ACTIVE_TASK_NAME = "task3"

# Task3 yalnızca bu listedeki kırmızı ve turuncu dubaları hedef alır.
TASK3_TARGET_BUOY_CLASSES = (
    "red_buoy",
    "red_buoys",
    "orange_buoy",
    "orange_buoys",
)


@dataclass(frozen=True)
class Task3Config:
    # Hedef parametreleri
    target_classes: tuple[str, ...] = TASK3_TARGET_BUOY_CLASSES
    required_impact_count: int = 3
    min_confidence: float = 0.45

    # Güvenlik parametreleri
    gps_timeout_sec: float = 2.0
    heading_timeout_sec: float = 2.0
    bridge_state_timeout_sec: float = 10.0
    vision_detection_timeout_sec: float = 3.0
    geofence_radius_m: float = 25.0
    mission_timeout_sec: float = 240.0

    # Arama parametreleri
    entry_settle_sec: float = 1.0
    search_linear_x: float = 0.25
    search_angular_z: float = 0.18
    search_initial_sweep_deg: float = 20.0
    search_sweep_increment_deg: float = 10.0
    search_max_sweep_deg: float = 180.0
    search_forward_duration_sec: float = 2.5
    search_heading_tolerance_deg: float = 2.0
    search_turn_timeout_min_sec: float = 6.0

    # Hedef doğrulama parametreleri
    confirmation_window_size: int = 2
    confirmation_required: int = 2
    confirmation_max_gap_sec: float = 0.5
    target_filter_alpha: float = 0.40
    confirmation_angle_spread_deg: float = 12.0
    confirmation_distance_spread_ratio: float = 0.45
    acquire_timeout_sec: float = 2.0

    # Yaklaşma / dümen parametreleri
    align_tolerance_deg: float = 6.0
    realign_threshold_deg: float = 12.0
    steering_kp: float = 0.025
    max_angular_z: float = 0.45
    far_approach_distance_m: float = 4.0
    final_confirm_distance_m: float = 1.4
    far_approach_speed: float = 0.55
    medium_approach_speed: float = 0.35
    near_approach_speed: float = 0.20

    # Hedef sürekliliği parametreleri
    final_confirmation_required: int = 3
    final_distance_spread_m: float = 0.35
    target_angle_jump_deg: float = 25.0
    target_distance_jump_ratio: float = 0.60
    target_bbox_min_iou: float = 0.05
    target_lost_timeout_sec: float = 1.0
    reacquire_linear_x: float = 0.15
    reacquire_angular_z: float = 0.16

    # Temas / geri çekilme parametreleri
    ram_speed: float = 0.75
    ram_duration_sec: float = 1.6
    contact_hold_sec: float = 0.7
    retreat_speed: float = 0.25
    retreat_min_sec: float = 0.6
    retreat_max_sec: float = 1.5
    retreat_target_distance_m: float = 2.5
    retreat_heading_max_angular_z: float = 0.25

    def __post_init__(self):
        if self.required_impact_count < 1:
            raise ValueError("required_impact_count must be at least 1")
        if self.confirmation_required > self.confirmation_window_size:
            raise ValueError(
                "confirmation_required cannot exceed confirmation_window_size"
            )
        if self.search_linear_x <= 0.0:
            raise ValueError("search_linear_x must be positive")
        if self.search_angular_z <= 0.0:
            raise ValueError("search_angular_z must be positive")
        if self.search_initial_sweep_deg <= 0.0:
            raise ValueError("search_initial_sweep_deg must be positive")
        if self.search_sweep_increment_deg <= 0.0:
            raise ValueError("search_sweep_increment_deg must be positive")
        if (
                self.search_max_sweep_deg < self.search_initial_sweep_deg
                or self.search_max_sweep_deg > 180.0
        ):
            raise ValueError(
                "search_max_sweep_deg must be between "
                "search_initial_sweep_deg and 180"
            )
        if self.search_forward_duration_sec <= 0.0:
            raise ValueError("search_forward_duration_sec must be positive")
        if self.search_heading_tolerance_deg <= 0.0:
            raise ValueError("search_heading_tolerance_deg must be positive")
        if self.search_turn_timeout_min_sec <= 0.0:
            raise ValueError("search_turn_timeout_min_sec must be positive")


class MissionState(Enum):
    INIT = auto()
    ENTRY_SETTLE = auto()
    SEARCH = auto()
    ACQUIRE_CONFIRM = auto()
    ALIGN = auto()
    APPROACH = auto()
    FINAL_CONFIRM = auto()
    RAM = auto()
    CONTACT_HOLD = auto()
    RETREAT = auto()
    REACQUIRE = auto()
    FINISHED = auto()
    FAILSAFE = auto()


class Task3KamikazeEngagement:
    def __init__(
            self,
            node,
            mission_topics,
            mission_clients,
            config=None,
    ):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.clients = mission_clients
        self.config = config or Task3Config()
        self.search_controller = Task3SearchController(self.config)
        self.impact_controller = Task3ImpactController(self.config)

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self.last_gps_time = None
        self.last_heading_time = None

        self.bridge_connected = None
        self.bridge_armed = None
        self.bridge_mode = None
        self.last_bridge_state_time = None

        self.home_lat = None
        self.home_lon = None
        self.entry_heading = None
        self.mission_started_at = None
        self.state_started_at = None
        self.state = MissionState.INIT
        self.finished = False

        self.impact_count = 0
        self.last_target = None
        self.last_target_angle = 0.0
        self.confirmation_samples = deque(
            maxlen=self.config.confirmation_window_size
        )
        self.confirmation_last_time = None
        self.distance_history = deque(
            maxlen=self.config.confirmation_window_size
        )
        self.final_confirmation_samples = deque(
            maxlen=self.config.final_confirmation_required
        )

        self.retreat_heading = None

        self.target_data_uncertain = False
        self.target_data_uncertain_reason = None
        self.target_rejection_reason = None
        self.last_observed_classes = ()

    @property
    def impact_count(self):
        return self.impact_controller.impact_count

    @impact_count.setter
    def impact_count(self, value):
        self.impact_controller.impact_count = int(value)

    @property
    def retreat_heading(self):
        return self.impact_controller.retreat_heading

    @retreat_heading.setter
    def retreat_heading(self, value):
        self.impact_controller.retreat_heading = value

    @staticmethod
    def _now(now):
        return time.monotonic() if now is None else float(now)

    @staticmethod
    def _angle_error_deg(target_deg, current_deg):
        return angle_error_deg(target_deg, current_deg)

    @staticmethod
    def _median(values):
        return median(values)

    def _set_state(self, state, now, reason=None):
        if state == self.state:
            return
        previous = self.state
        self.state = state
        self.state_started_at = now
        if reason:
            self.logger.info(
                f"Task 3 state: {previous.name} -> {state.name} ({reason})"
            )
        else:
            self.logger.info(f"Task 3 state: {previous.name} -> {state.name}")

    def _stop(self):
        stop_vehicle(self.topics.cmd_vel_pub)

    def _publish_motion(self, linear_x, angular_z, reason):
        linear_x = max(-1.0, min(1.0, float(linear_x)))
        angular_z = max(
            -self.config.max_angular_z,
            min(self.config.max_angular_z, float(angular_z)),
        )
        retreat_allowed = (
            self.state == MissionState.RETREAT
            and self.impact_count > 0
        )
        if linear_x < 0.0 and not retreat_allowed:
            self._enter_failsafe(
                "Task 3 güvenlik ihlali: doğrulanmış temas dışında "
                "negatif linear_x engellendi."
            )
            return False

        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=linear_x,
            angular_z=angular_z,
        )
        target_text = (
            "none"
            if self.last_target is None
            else (
                f"{self.last_target['class']}/"
                f"{self.last_target['distance']:.2f}m/"
                f"{self.last_target['angle']:+.1f}deg"
            )
        )
        self.logger.info(
            f"Task3 decision: state={self.state.name}, reason={reason}, "
            f"observed={list(self.last_observed_classes)}, target={target_text}, "
            f"rejected={self.target_rejection_reason}, "
            f"linear_x={linear_x:+.2f}, angular_z={angular_z:+.2f}",
            throttle_duration_sec=0.5,
        )
        return True

    def _enter_failsafe(self, reason):
        if self.state != MissionState.FAILSAFE:
            self.logger.error(reason)
        self.state = MissionState.FAILSAFE
        self.finished = False
        self._stop()

    def reset_for_entry(self, lat, lon, heading, now=None):
        now = self._now(now)
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.current_heading = float(heading)
        self.last_gps_time = now
        self.last_heading_time = now
        self.home_lat = self.current_lat
        self.home_lon = self.current_lon
        self.entry_heading = self.current_heading
        self.mission_started_at = now
        self.finished = False
        self.impact_controller.reset()
        self.last_target = None
        self.last_target_angle = 0.0
        self.confirmation_samples.clear()
        self.confirmation_last_time = None
        self.distance_history.clear()
        self.final_confirmation_samples.clear()
        self.search_controller.reset_for_entry(
            self.current_heading,
            now,
        )
        self.target_data_uncertain = False
        self.target_data_uncertain_reason = None
        self.target_rejection_reason = None
        self.last_observed_classes = ()
        self.state = MissionState.ENTRY_SETTLE
        self.state_started_at = now
        self._stop()
        self.logger.info(
            "Task 3 giriş durumu sıfırlandı: "
            f"lat={self.current_lat:.7f}, lon={self.current_lon:.7f}, "
            f"heading={self.current_heading:.1f}, "
            f"targets={list(self.config.target_classes)}, "
            f"impact_count={self.config.required_impact_count}"
        )

    def update_gps(self, lat, lon, heading=None, now=None):
        now = self._now(now)
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.last_gps_time = now
        if heading is not None:
            self.update_heading(heading, now=now)
        if self.home_lat is None:
            self.home_lat = self.current_lat
            self.home_lon = self.current_lon

    def update_heading(self, heading, now=None):
        self.current_heading = float(heading)
        self.last_heading_time = self._now(now)

    def update_bridge_state(self, state_text, now=None):
        try:
            state = parse_bridge_state(state_text)
            if "connected" in state:
                self.bridge_connected = state["connected"] is True
            if "armed" in state:
                self.bridge_armed = state["armed"] is True
            if "mode" in state:
                self.bridge_mode = str(
                    state["mode"] or "UNKNOWN"
                ).strip().upper()
            self.last_bridge_state_time = self._now(now)
        except Exception as exc:
            self.logger.warn(
                f"Bridge state parse edilemedi: {exc}",
                throttle_duration_sec=2.0,
            )

    def _check_navigation(self, now):
        if self.last_gps_time is None or self.last_heading_time is None:
            self._stop()
            self.logger.info(
                "Task 3 GPS ve heading verisi bekliyor.",
                throttle_duration_sec=2.0,
            )
            return False
        if now - self.last_gps_time > self.config.gps_timeout_sec:
            self._enter_failsafe("Task 3 GPS watchdog zaman aşımı.")
            return False
        if now - self.last_heading_time > self.config.heading_timeout_sec:
            self._enter_failsafe("Task 3 heading watchdog zaman aşımı.")
            return False
        return True

    def _check_bridge(self, now):
        if (
                self.last_bridge_state_time is not None
                and now - self.last_bridge_state_time
                > self.config.bridge_state_timeout_sec
        ):
            self._enter_failsafe("Task 3 bridge state watchdog zaman aşımı.")
            return False
        if self.bridge_connected is False:
            self._stop()
            self.logger.warn(
                "Task 3 bridge bağlantısı yok; araç bekletiliyor.",
                throttle_duration_sec=2.0,
            )
            return False
        if self.bridge_armed is False:
            self._stop()
            self.logger.warn(
                "Task 3 araç ARM değil; araç bekletiliyor.",
                throttle_duration_sec=2.0,
            )
            return False
        if self.bridge_mode not in (None, DRIVE_MODE):
            self._stop()
            self.logger.warn(
                f"Task 3 bridge mode={self.bridge_mode}; "
                f"beklenen={DRIVE_MODE}.",
                throttle_duration_sec=2.0,
            )
            return False
        return True

    def _check_geofence(self):
        if (
                self.home_lat is None
                or self.home_lon is None
                or self.current_lat is None
                or self.current_lon is None
        ):
            return True
        distance = calculate_gps_distance(
            self.home_lat,
            self.home_lon,
            self.current_lat,
            self.current_lon,
        )
        if distance > self.config.geofence_radius_m:
            self._enter_failsafe(
                f"Task 3 yerel geofence ihlali: "
                f"{distance:.1f}m > {self.config.geofence_radius_m:.1f}m."
            )
            return False
        return True

    def _select_target(self, detections):
        result = select_target(
            detections,
            target_classes=self.config.target_classes,
            min_confidence=self.config.min_confidence,
            last_target=self.last_target,
        )
        self.last_observed_classes = result.observed_classes
        self.target_data_uncertain = result.data_uncertain
        self.target_data_uncertain_reason = result.data_uncertain_reason
        self.target_rejection_reason = result.rejection_reason
        return result.target

    def _target_is_consistent(self, target):
        return target_is_consistent(
            target,
            self.last_target,
            bbox_min_iou=self.config.target_bbox_min_iou,
            angle_jump_deg=self.config.target_angle_jump_deg,
            distance_jump_ratio=self.config.target_distance_jump_ratio,
        )

    def _filter_target(self, target, previous):
        return filter_target(
            target,
            previous,
            self.config.target_filter_alpha,
        )

    def _record_confirmation(self, target, now):
        if target is None:
            self.confirmation_samples.clear()
            self.confirmation_last_time = None
            return None

        previous = (
            self.confirmation_samples[-1]
            if self.confirmation_samples
            else None
        )
        continuous = (
            previous is not None
            and self.confirmation_last_time is not None
            and now - self.confirmation_last_time
            <= self.config.confirmation_max_gap_sec
        )
        if continuous:
            self.last_target = previous
            continuous = self._target_is_consistent(target)

        if not continuous:
            self.confirmation_samples.clear()
        else:
            target = self._filter_target(target, previous)

        self.confirmation_samples.append(target)
        self.confirmation_last_time = now
        if len(self.confirmation_samples) < self.config.confirmation_required:
            return None

        recent = list(self.confirmation_samples)[
            -self.config.confirmation_required:
        ]
        angles = [sample["angle"] for sample in recent]
        distances = [sample["distance"] for sample in recent]
        distance_median = self._median(distances)
        distance_spread_ratio = (
            (max(distances) - min(distances)) / max(distance_median, 0.1)
        )
        if (
                max(angles) - min(angles)
                > self.config.confirmation_angle_spread_deg
                or distance_spread_ratio
                > self.config.confirmation_distance_spread_ratio
        ):
            self.confirmation_samples.clear()
            self.confirmation_samples.append(target)
            return None

        return dict(recent[-1])

    def _wait_for_target_data(self):
        if not self.target_data_uncertain:
            return False
        self._stop()
        self.logger.warn(
            f"Task3 target verisi belirsiz: "
            f"{self.target_data_uncertain_reason}; state={self.state.name}, "
            f"observed={list(self.last_observed_classes)}, "
            f"rejected={self.target_rejection_reason}, "
            f"linear_x=+0.00, angular_z=+0.00. Araç bekletiliyor.",
            throttle_duration_sec=1.0,
        )
        return True

    def _begin_acquisition(self, target, now, reason):
        self._stop()
        self.confirmation_samples.clear()
        self.confirmation_samples.append(target)
        self.confirmation_last_time = now
        self.last_target = target
        self.last_target_angle = target["angle"]
        self._set_state(MissionState.ACQUIRE_CONFIRM, now, reason)
        self.logger.info(
            f"Task3 decision: state={self.state.name}, reason={reason}, "
            f"observed={list(self.last_observed_classes)}, "
            f"target={target['class']}/{target['distance']:.2f}m/"
            f"{target['angle']:+.1f}deg, rejected=None, "
            f"linear_x=+0.00, angular_z=+0.00",
            throttle_duration_sec=0.5,
        )

    def _enter_search(self, now, reason):
        self._stop()
        self.search_controller.enter_search(self.current_heading, now)
        self.confirmation_samples.clear()
        self.confirmation_last_time = None
        self.distance_history.clear()
        self.final_confirmation_samples.clear()
        self.last_target = None
        self._set_state(MissionState.SEARCH, now, reason)

    def _enter_reacquire(self, now, reason):
        self._stop()
        self.retreat_heading = None
        if self.last_target is not None:
            self.last_target_angle = self.last_target["angle"]
        self.confirmation_samples.clear()
        self.confirmation_last_time = None
        self.final_confirmation_samples.clear()
        self._set_state(MissionState.REACQUIRE, now, reason)

    def _steering_command(self, angle):
        command = self.config.steering_kp * float(angle)
        return max(
            -self.config.max_angular_z,
            min(self.config.max_angular_z, command),
        )

    def _approach_speed(self, distance):
        if distance > self.config.far_approach_distance_m:
            return self.config.far_approach_speed
        midpoint = (
            self.config.final_confirm_distance_m
            + self.config.far_approach_distance_m
        ) / 2.0
        if distance > midpoint:
            return self.config.medium_approach_speed
        return self.config.near_approach_speed

    def _update_search(self, detections, now):
        target = self._select_target(detections)
        if target is not None:
            self._begin_acquisition(target, now, "arama sırasında hedef adayı")
            return
        if self._wait_for_target_data():
            self.search_controller.pause(self.current_heading, now)
            return

        decision = self.search_controller.step(self.current_heading, now)
        if decision.failed:
            self._enter_failsafe(
                f"Task 3 arama denetleyicisi başarısız: {decision.reason}"
            )
            return
        if decision.phase_changed:
            self.logger.info(
                f"Task3 arama fazı: "
                f"{self.search_controller.phase.name}, "
                f"sweep={self.search_controller.sweep_deg:.1f}deg, "
                f"cycle={self.search_controller.cycle_index}."
            )

        self._publish_motion(
            linear_x=decision.linear_x,
            angular_z=decision.angular_z,
            reason=decision.reason,
        )

    def _update_acquire(self, detections, now):
        self._stop()
        target = self._select_target(detections)
        if self._wait_for_target_data():
            self._record_confirmation(None, now)
            return

        confirmed = self._record_confirmation(target, now)
        if target is not None:
            latest = (
                self.confirmation_samples[-1]
                if self.confirmation_samples
                else target
            )
            self.last_target = latest
            self.last_target_angle = latest["angle"]
        if confirmed is not None:
            self.last_target = confirmed
            self.last_target_angle = confirmed["angle"]
            self.distance_history.clear()
            self._set_state(
                MissionState.ALIGN,
                now,
                "çok kareli hedef teyidi tamamlandı",
            )
            return

        if now - self.state_started_at >= self.config.acquire_timeout_sec:
            self._enter_search(now, "hedef adayı teyit edilemedi")

    def _update_align(self, detections, now):
        target = self._select_target(detections)
        if self._wait_for_target_data():
            return
        if target is None or not self._target_is_consistent(target):
            self._enter_reacquire(now, "hizalama sırasında hedef kayboldu")
            return
        target = self._filter_target(target, self.last_target)
        self.last_target = target
        self.last_target_angle = target["angle"]

        if abs(target["angle"]) <= self.config.align_tolerance_deg:
            self._stop()
            self.distance_history.clear()
            self._set_state(
                MissionState.APPROACH,
                now,
                "hedef kamera merkezine hizalandı",
            )
            return

        self._publish_motion(
            linear_x=0.0,
            angular_z=self._steering_command(target["angle"]),
            reason="target alignment",
        )

    def _update_approach(self, detections, now):
        target = self._select_target(detections)
        if self._wait_for_target_data():
            return
        if target is None or not self._target_is_consistent(target):
            self._enter_reacquire(now, "yaklaşma sırasında hedef kayboldu")
            return
        observed_distance = target["distance"]
        target = self._filter_target(target, self.last_target)
        self.last_target = target
        self.last_target_angle = target["angle"]

        if abs(target["angle"]) > self.config.realign_threshold_deg:
            self._stop()
            self._set_state(
                MissionState.ALIGN,
                now,
                "yaklaşma açı hatası büyüdü",
            )
            return

        if observed_distance <= self.config.final_confirm_distance_m:
            self._stop()
            self.final_confirmation_samples.clear()
            self._set_state(
                MissionState.FINAL_CONFIRM,
                now,
                "yakın mesafe eşiğine ulaşıldı",
            )
            return

        self.distance_history.append(target["distance"])
        distance = self._median(self.distance_history)

        self._publish_motion(
            linear_x=self._approach_speed(distance),
            angular_z=self._steering_command(target["angle"]),
            reason="target approach",
        )

    def _update_final_confirm(self, detections, now):
        self._stop()
        target = self._select_target(detections)
        if self._wait_for_target_data():
            self.final_confirmation_samples.clear()
            return
        if target is None or not self._target_is_consistent(target):
            self._enter_reacquire(now, "son teyitte hedef kayboldu")
            return
        observed_target = target
        target = self._filter_target(target, self.last_target)
        self.last_target = target
        self.last_target_angle = target["angle"]

        if abs(observed_target["angle"]) > self.config.align_tolerance_deg:
            self.final_confirmation_samples.clear()
            self._set_state(
                MissionState.ALIGN,
                now,
                "son teyitte hedef merkezden çıktı",
            )
            return
        if (
                observed_target["distance"]
                > self.config.final_confirm_distance_m
                + self.config.final_distance_spread_m
        ):
            self.final_confirmation_samples.clear()
            self._set_state(
                MissionState.APPROACH,
                now,
                "son teyitte hedef uzaklaştı",
            )
            return

        self.final_confirmation_samples.append(observed_target)
        if (
                len(self.final_confirmation_samples)
                < self.config.final_confirmation_required
        ):
            return

        distances = [
            sample["distance"]
            for sample in self.final_confirmation_samples
        ]
        angles = [
            sample["angle"]
            for sample in self.final_confirmation_samples
        ]
        if (
                max(distances) - min(distances)
                > self.config.final_distance_spread_m
                or max(abs(angle) for angle in angles)
                > self.config.align_tolerance_deg
        ):
            self.final_confirmation_samples.clear()
            return

        self.last_target = dict(target)
        self.last_target["distance"] = self._median(distances)
        self.last_target["angle"] = self._median(angles)
        self._set_state(
            MissionState.RAM,
            now,
            f"{self.impact_count + 1}. temas için son teyit tamamlandı",
        )

    def _update_ram(self, now):
        elapsed = now - self.state_started_at
        decision = self.impact_controller.ram_decision(
            elapsed,
            self.current_heading,
        )
        if decision.action == ImpactAction.RAM_MOTION:
            self._publish_motion(
                linear_x=decision.linear_x,
                angular_z=decision.angular_z,
                reason=decision.reason,
            )
            return

        self._stop()
        self._set_state(
            MissionState.CONTACT_HOLD,
            now,
            decision.reason,
        )

    def _update_contact_hold(self, now):
        self._stop()
        decision = self.impact_controller.contact_hold_decision(
            now - self.state_started_at
        )
        if decision.action == ImpactAction.HOLD:
            return
        if decision.action == ImpactAction.FINISH:
            self.finished = True
            self._set_state(
                MissionState.FINISHED,
                now,
                decision.reason,
            )
            return
        self._set_state(
            MissionState.RETREAT,
            now,
            decision.reason,
        )

    def _update_retreat(self, detections, now):
        elapsed = now - self.state_started_at
        if self.impact_count <= 0:
            decision = self.impact_controller.retreat_decision(
                elapsed=elapsed,
                target_far_enough=False,
                current_heading=self.current_heading,
            )
            self._enter_failsafe(decision.reason)
            return

        target = self._select_target(detections)
        if self._wait_for_target_data():
            if self.impact_controller.retreat_timeout_reached(elapsed):
                self._enter_reacquire(
                    now,
                    "geri çekilme veri beklerken zaman sınırına ulaştı",
                )
            return
        target_far_enough = (
            target is not None
            and target["distance"] >= self.config.retreat_target_distance_m
        )
        decision = self.impact_controller.retreat_decision(
            elapsed=elapsed,
            target_far_enough=target_far_enough,
            current_heading=self.current_heading,
        )
        if decision.action == ImpactAction.REACQUIRE:
            if target is not None:
                self.last_target = target
                self.last_target_angle = target["angle"]
            self._enter_reacquire(
                now,
                decision.reason,
            )
            return

        self._publish_motion(
            linear_x=decision.linear_x,
            angular_z=decision.angular_z,
            reason=decision.reason,
        )

    def _update_reacquire(self, detections, now):
        target = self._select_target(detections)
        if target is not None:
            self._begin_acquisition(
                target,
                now,
                "hedef yeniden bulundu",
            )
            return
        if self._wait_for_target_data():
            return

        if now - self.state_started_at >= self.config.target_lost_timeout_sec:
            self._enter_search(now, "lokal yeniden arama başarısız")
            return

        direction = 1.0 if self.last_target_angle >= 0.0 else -1.0
        self._publish_motion(
            linear_x=self.config.reacquire_linear_x,
            angular_z=direction * self.config.reacquire_angular_z,
            reason="local forward reacquire",
        )

    def update(self, detections, now=None, vision_fresh=True):
        now = self._now(now)
        if self.state == MissionState.FINISHED:
            self._stop()
            return
        if self.state == MissionState.FAILSAFE:
            self._stop()
            return

        if not vision_fresh:
            self._enter_failsafe("Task 3 vision heartbeat zaman aşımı.")
            return
        if not self._check_navigation(now):
            return
        if not self._check_bridge(now):
            return
        if not self._check_geofence():
            return

        if self.mission_started_at is None:
            self.reset_for_entry(
                self.current_lat,
                self.current_lon,
                self.current_heading,
                now=now,
            )
            return
        if (
                now - self.mission_started_at
                > self.config.mission_timeout_sec
        ):
            self._enter_failsafe("Task 3 toplam görev zaman aşımı.")
            return

        if self.state == MissionState.ENTRY_SETTLE:
            self._stop()
            if now - self.state_started_at >= self.config.entry_settle_sec:
                self._enter_search(now, "Task 2 çıkış pozu sabitlendi")
            return
        if self.state == MissionState.SEARCH:
            self._update_search(detections, now)
            return
        if self.state == MissionState.ACQUIRE_CONFIRM:
            self._update_acquire(detections, now)
            return
        if self.state == MissionState.ALIGN:
            self._update_align(detections, now)
            return
        if self.state == MissionState.APPROACH:
            self._update_approach(detections, now)
            return
        if self.state == MissionState.FINAL_CONFIRM:
            self._update_final_confirm(detections, now)
            return
        if self.state == MissionState.RAM:
            self._update_ram(now)
            return
        if self.state == MissionState.CONTACT_HOLD:
            self._update_contact_hold(now)
            return
        if self.state == MissionState.RETREAT:
            self._update_retreat(detections, now)
            return
        if self.state == MissionState.REACQUIRE:
            self._update_reacquire(detections, now)
            return

        self._enter_failsafe(f"Task 3 beklenmeyen state: {self.state!r}")


class Task3Node(Node):
    def __init__(self, config=None):
        super().__init__("task3_kamikaze_engagement_node")
        self.get_logger().info("Task 3 Kamikaze Engagement düğümü başlatılıyor...")

        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback,
        )

        self.config = config or Task3Config()
        self.task = Task3KamikazeEngagement(
            self,
            self.mission_topics,
            self.mission_clients,
            config=self.config,
        )
        self.mission_active = False
        self.current_detections = []
        self.last_detection_time = None

        self.vision_sub = self.create_subscription(
            String,
            "/vision/detections",
            self.vision_callback,
            10,
        )
        self.active_task_pub = self.create_publisher(
            String,
            "/mission/active_task",
            10,
        )
        self.control_timer = self.create_timer(0.1, self.timer_callback)
        self.active_task_timer = self.create_timer(
            1.0,
            self.publish_active_task,
        )
        self.publish_active_task()

    def publish_active_task(self):
        message = String()
        message.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(message)

    def gps_callback(self, msg):
        self.task.update_gps(msg.latitude, msg.longitude)

    def heading_callback(self, msg):
        self.task.update_heading(msg.data)

    def state_callback(self, msg):
        self.task.update_bridge_state(msg.data)

    def vision_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            detections = payload.get("detections", [])
            if not isinstance(detections, list):
                self.get_logger().warn(
                    "Vision detections list formatında değil.",
                    throttle_duration_sec=2.0,
                )
                return
            self.current_detections = detections
            self.last_detection_time = time.monotonic()
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().warn(
                f"Vision JSON parse edilemedi: {exc}",
                throttle_duration_sec=2.0,
            )

    def vision_is_fresh(self):
        return (
            self.last_detection_time is not None
            and time.monotonic() - self.last_detection_time
            <= self.config.vision_detection_timeout_sec
        )

    def wait_for_vision(self, timeout_sec=30.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.vision_is_fresh():
                return True
            self.publish_active_task()
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def timer_callback(self):
        if not self.mission_active:
            return
        try:
            self.task.update(
                detections=self.current_detections,
                vision_fresh=self.vision_is_fresh(),
            )
        except Exception as exc:
            self.get_logger().error(
                f"Zamanlayıcı döngüsünde beklenmeyen hata: {exc}"
            )
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub)
            except Exception as stop_exc:
                self.get_logger().error(f"Araç durdurulamadı: {stop_exc}")
            self.task.state = MissionState.FAILSAFE


def main(args=None):
    rclpy.init(args=args)
    node = Task3Node()

    try:
        if not node.wait_for_vision(timeout_sec=30.0):
            node.get_logger().error("Vision hazır değil; Task 3 başlatılmadı.")
            return

        node.get_logger().info(f"Araç {DRIVE_MODE} moduna alınıyor...")
        if call_set_mode(
                node,
                node.mission_clients.set_mode_client,
                DRIVE_MODE,
        ) is False:
            node.get_logger().error("Mod geçişi başarısız.")
            return

        node.get_logger().info("Motorlar FORCE ARM ediliyor...")
        if call_trigger_service(
                node,
                node.mission_clients.force_arm_client,
                "FORCE ARM",
        ) is False:
            node.get_logger().error("FORCE ARM başarısız.")
            return

        if not node.wait_for_vision(timeout_sec=5.0):
            node.get_logger().error(
                "ARM sonrasında güncel vision doğrulanamadı; Task 3 başlatılmadı."
            )
            return

        node.mission_active = True
        node.get_logger().info("Task 3 kontrol döngüsü başladı.")
        while (
                rclpy.ok()
                and not node.task.finished
                and node.task.state != MissionState.FAILSAFE
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.task.finished:
            node.get_logger().info(
                "Task 3 üç temas tamamlandı; görev başarıyla bitti."
            )
        elif node.task.state == MissionState.FAILSAFE:
            node.get_logger().error("Task 3 FAILSAFE ile sonlandı.")
    except KeyboardInterrupt:
        node.get_logger().info("Task 3 kullanıcı tarafından durduruldu.")
    finally:
        node.mission_active = False
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        call_trigger_service(
            node,
            node.mission_clients.disarm_client,
            "DISARM",
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
