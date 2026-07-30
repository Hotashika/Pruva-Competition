"""TEKNOFEST Task 3: direct buoy attack and GPS-anchored repeated impact."""

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_TEXT = str(REPO_ROOT)
while REPO_ROOT_TEXT in sys.path:
    sys.path.remove(REPO_ROOT_TEXT)
sys.path.insert(0, REPO_ROOT_TEXT)

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from std_msgs.msg import String

from teknofest.missions.utils.mission_data_recorder import MissionDataRecorder
from teknofest.missions.utils.task3_impact_controller import (
    ImpactAction,
    Task3ImpactController,
)
from teknofest.missions.utils.task3_search_controller import (
    SearchPhase,
    Task3SearchController,
)
from teknofest.missions.utils.task3_targeting import (
    filter_target,
    select_target,
    target_is_consistent,
)
from utils.mavlink_utilities import (
    calculate_gps_distance,
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    create_mission_topics,
    parse_bridge_state,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    wait_for_mission_services,
)


DRIVE_MODE = "GUIDED"
HOLD_MODE = "LOITER"
ACTIVE_TASK_NAME = "task3"
MISSION_STATE_HEARTBEAT_SEC = 1.0

TASK3_TARGET_BUOY_CLASSES = (
    "red_buoy",
    "red_buoys",
    "orange_buoy",
    "orange_buoys",
)


@dataclass(frozen=True)
class Task3Config:
    # Hedef
    target_classes: Tuple[str, ...] = TASK3_TARGET_BUOY_CLASSES
    required_impact_count: int = 3
    min_confidence: float = 0.45

    # Güvenlik
    gps_timeout_sec: float = 2.0
    heading_timeout_sec: float = 2.0
    bridge_state_timeout_sec: float = 10.0
    vision_detection_timeout_sec: float = 12.0
    geofence_radius_m: float = 25.0
    mission_timeout_sec: float = 240.0
    mode_transition_timeout_sec: float = 5.0
    loiter_mode: str = HOLD_MODE

    # Arama
    search_linear_x: float = 0.25
    search_angular_z: float = 0.18
    search_initial_sweep_deg: float = 20.0
    search_sweep_increment_deg: float = 10.0
    search_max_sweep_deg: float = 180.0
    search_advance_distance_m: float = 1.5
    search_heading_tolerance_deg: float = 2.0
    search_heading_settle_sec: float = 0.3
    search_heading_kp: float = 0.018
    search_min_angular_z: float = 0.04
    search_turn_timeout_sec: float = 25.0
    search_advance_timeout_sec: float = 15.0
    search_no_progress_timeout_sec: float = 6.0
    search_progress_min_m: float = 0.15
    search_cross_track_limit_m: float = 2.0
    search_cross_track_kp_deg_per_m: float = 8.0
    search_advance_heading_limit_deg: float = 25.0

    # Altı ayrı karelik düşük hızlı saldırı teyidi
    confirmation_window_size: int = 6
    confirmation_required: int = 6
    confirmation_max_gap_sec: float = 0.5
    confirmation_angle_spread_deg: float = 12.0
    confirmation_distance_spread_ratio: float = 0.45
    target_filter_alpha: float = 0.40
    target_angle_jump_deg: float = 25.0
    target_distance_jump_ratio: float = 0.60
    target_bbox_min_iou: float = 0.05
    attack_confirm_timeout_sec: float = 2.0
    attack_confirm_speed: float = 0.15
    steering_kp: float = 0.018
    max_angular_z: float = 0.35

    # Temas ve kayıtlı GPS'e dönüş
    ram_speed: float = 0.85
    ram_duration_sec: float = 2.0
    post_impact_forward_speed: float = 0.85
    post_impact_forward_duration_sec: float = 2.5
    impact_return_tolerance_m: float = 1.0
    impact_return_timeout_sec: float = 20.0

    def __post_init__(self):
        if self.required_impact_count < 1:
            raise ValueError("required_impact_count must be at least 1")
        if not (
                1
                <= self.confirmation_required
                <= self.confirmation_window_size
        ):
            raise ValueError(
                "confirmation_required must be between 1 and "
                "confirmation_window_size"
            )
        for name in (
                "search_linear_x",
                "search_angular_z",
                "search_initial_sweep_deg",
                "search_sweep_increment_deg",
                "search_advance_distance_m",
                "search_heading_tolerance_deg",
                "search_heading_settle_sec",
                "search_heading_kp",
                "search_min_angular_z",
                "search_turn_timeout_sec",
                "search_advance_timeout_sec",
                "search_no_progress_timeout_sec",
                "search_progress_min_m",
                "search_cross_track_limit_m",
                "search_cross_track_kp_deg_per_m",
                "search_advance_heading_limit_deg",
                "attack_confirm_speed",
                "ram_speed",
                "ram_duration_sec",
                "post_impact_forward_speed",
                "post_impact_forward_duration_sec",
                "impact_return_tolerance_m",
                "impact_return_timeout_sec",
                "mode_transition_timeout_sec",
                "confirmation_max_gap_sec",
                "attack_confirm_timeout_sec",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
                self.search_max_sweep_deg < self.search_initial_sweep_deg
                or self.search_max_sweep_deg > 180.0
        ):
            raise ValueError(
                "search_max_sweep_deg must be between "
                "search_initial_sweep_deg and 180"
            )
        if self.search_min_angular_z > self.search_angular_z:
            raise ValueError(
                "search_min_angular_z cannot exceed search_angular_z"
            )
        if (
                self.search_advance_heading_limit_deg
                <= self.search_heading_tolerance_deg
                or self.search_advance_heading_limit_deg >= 90.0
        ):
            raise ValueError(
                "search_advance_heading_limit_deg must be between "
                "search_heading_tolerance_deg and 90"
            )
        if str(self.loiter_mode).strip().upper() != HOLD_MODE:
            raise ValueError("loiter_mode must be LOITER")


class MissionState(Enum):
    INIT = auto()
    WAIT_GUIDED_SEARCH = auto()
    SEARCH = auto()
    ATTACK_CONFIRM = auto()
    RAM = auto()
    POST_IMPACT_ADVANCE = auto()
    RETURN_TO_IMPACT = auto()
    FINAL_LOITER = auto()
    FAILSAFE_LOITER = auto()
    FINISHED = auto()
    FAILSAFE = auto()


class Task3KamikazeEngagement:
    _GUIDED_STATES = {
        MissionState.SEARCH,
        MissionState.ATTACK_CONFIRM,
        MissionState.RAM,
        MissionState.POST_IMPACT_ADVANCE,
        MissionState.RETURN_TO_IMPACT,
    }

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
        self.last_state_publish_at = None
        self.finished = False

        self.last_target = None
        self.last_target_angle = 0.0
        self.confirmation_samples = deque(
            maxlen=self.config.confirmation_window_size
        )
        self.confirmation_last_time = None
        self.confirmation_last_frame_id = None

        self.impact_target_gps = None
        self.impact_events = []
        self.impact_return_departed = False

        self.target_data_uncertain = False
        self.target_data_uncertain_reason = None
        self.target_rejection_reason = None
        self.last_observed_classes = ()

        self.pending_mode_name = None
        self.pending_mode_future = None
        self.pending_mode_started_at = None
        self.failsafe_reason = None

    @property
    def impact_count(self):
        return self.impact_controller.impact_count

    @impact_count.setter
    def impact_count(self, value):
        self.impact_controller.impact_count = int(value)

    @staticmethod
    def _now(now):
        return time.monotonic() if now is None else float(now)

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
        self._publish_state(now, force=True)

    def _publish_state(self, now=None, force=False):
        publisher = getattr(self.topics, "mission_state_pub", None)
        if publisher is None:
            return
        now = self._now(now)
        if (
                not force
                and self.last_state_publish_at is not None
                and now - self.last_state_publish_at
                < MISSION_STATE_HEARTBEAT_SEC
        ):
            return
        message = String()
        message.data = self.state.name
        publisher.publish(message)
        self.last_state_publish_at = now

    def _stop(self):
        stop_vehicle(self.topics.cmd_vel_pub)

    def _steering_command(self, angle):
        command = self.config.steering_kp * float(angle)
        return max(
            -self.config.max_angular_z,
            min(self.config.max_angular_z, command),
        )

    def _publish_motion(self, linear_x, angular_z, reason):
        linear_x = max(-1.0, min(1.0, float(linear_x)))
        angular_z = max(
            -self.config.max_angular_z,
            min(self.config.max_angular_z, float(angular_z)),
        )
        if linear_x < 0.0:
            self._enter_failsafe(
                "Task 3 güvenlik ihlali: negatif linear_x engellendi."
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

    def _clear_pending_mode(self):
        self.pending_mode_name = None
        self.pending_mode_future = None
        self.pending_mode_started_at = None

    def _cancel_pending_mode(self):
        future = self.pending_mode_future
        if future is not None and not future.done():
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
        self._clear_pending_mode()

    def _mode_transition_failed(self, desired_mode, reason, now):
        self._clear_pending_mode()
        self._stop()
        self.logger.error(reason)
        if desired_mode == self.config.loiter_mode:
            self.finished = False
            self._set_state(MissionState.FAILSAFE, now, reason)
            return
        self._enter_failsafe(reason, now=now)

    def _ensure_mode(self, desired_mode, now):
        desired_mode = str(desired_mode).strip().upper()
        if self.bridge_mode == desired_mode:
            if self.pending_mode_name == desired_mode:
                self._clear_pending_mode()
            return True

        if (
                self.pending_mode_name is not None
                and self.pending_mode_name != desired_mode
        ):
            self._cancel_pending_mode()

        if self.pending_mode_name is None:
            client = getattr(self.clients, "set_mode_client", None)
            if client is None:
                self._mode_transition_failed(
                    desired_mode,
                    f"Task 3 {desired_mode} modu istenemedi: client yok.",
                    now,
                )
                return False
            request = SetMode.Request()
            request.base_mode = 0
            request.custom_mode = desired_mode
            try:
                self.pending_mode_future = client.call_async(request)
            except Exception as exc:
                self._mode_transition_failed(
                    desired_mode,
                    f"Task 3 {desired_mode} modu istenemedi: {exc}",
                    now,
                )
                return False
            self.pending_mode_name = desired_mode
            self.pending_mode_started_at = now
            self.logger.info(
                f"Task 3 mode request sent: {desired_mode}; "
                "heartbeat confirmation waiting."
            )
            return False

        future = self.pending_mode_future
        if future is not None and future.done():
            try:
                response = future.result()
            except Exception as exc:
                self._mode_transition_failed(
                    desired_mode,
                    f"Task 3 {desired_mode} service error: {exc}",
                    now,
                )
                return False
            if response is None or not bool(
                    getattr(response, "mode_sent", False)
            ):
                self._mode_transition_failed(
                    desired_mode,
                    f"Task 3 {desired_mode} mode request was rejected.",
                    now,
                )
                return False

        if (
                self.pending_mode_started_at is not None
                and now - self.pending_mode_started_at
                >= self.config.mode_transition_timeout_sec
        ):
            self._mode_transition_failed(
                desired_mode,
                f"Task 3 {desired_mode} heartbeat confirmation timed out.",
                now,
            )
        return False

    def _enter_failsafe(self, reason, now=None):
        now = self._now(now)
        if self.state not in (
                MissionState.FAILSAFE_LOITER,
                MissionState.FAILSAFE,
        ):
            self.logger.error(reason)
        self.failsafe_reason = reason
        self.finished = False
        self._cancel_pending_mode()
        self._stop()
        self._set_state(
            MissionState.FAILSAFE_LOITER,
            now,
            "FAILSAFE nedeniyle LOITER isteniyor",
        )

    def request_failsafe_loiter(self, reason, now=None):
        """Stop immediately and dispatch the Task 3 LOITER request."""
        now = self._now(now)
        self._enter_failsafe(reason, now=now)
        self._update_failsafe_loiter(now)

    def reset_for_entry(self, lat, lon, heading, now=None):
        now = self._now(now)
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.current_heading = float(heading) % 360.0
        self.last_gps_time = now
        self.last_heading_time = now
        self.home_lat = self.current_lat
        self.home_lon = self.current_lon
        self.entry_heading = self.current_heading
        self.mission_started_at = now
        self.finished = False
        self.failsafe_reason = None
        self.impact_controller.reset()
        self.last_target = None
        self.last_target_angle = 0.0
        self.confirmation_samples.clear()
        self.confirmation_last_time = None
        self.confirmation_last_frame_id = None
        self.impact_target_gps = None
        self.impact_events = []
        self.impact_return_departed = False
        self.search_controller.reset_for_entry(
            self.current_heading,
            self.current_lat,
            self.current_lon,
            now,
        )
        self.target_data_uncertain = False
        self.target_data_uncertain_reason = None
        self.target_rejection_reason = None
        self.last_observed_classes = ()
        self._cancel_pending_mode()
        self.state = MissionState.WAIT_GUIDED_SEARCH
        self.state_started_at = now
        self._publish_state(now, force=True)
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
        try:
            heading = float(heading)
        except (TypeError, ValueError):
            heading = float("nan")
        if not math.isfinite(heading):
            self.logger.warn(
                "Task 3 geçersiz heading verisi yok sayıldı.",
                throttle_duration_sec=2.0,
            )
            return False
        self.current_heading = heading % 360.0
        self.last_heading_time = self._now(now)
        return True

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
            return False
        if now - self.last_gps_time > self.config.gps_timeout_sec:
            self._enter_failsafe(
                "Task 3 GPS watchdog zaman aşımı.",
                now=now,
            )
            return False
        if now - self.last_heading_time > self.config.heading_timeout_sec:
            self._enter_failsafe(
                "Task 3 heading watchdog zaman aşımı.",
                now=now,
            )
            return False
        return True

    def _check_bridge(self, now):
        if self.last_bridge_state_time is None:
            self._stop()
            return False
        if (
                now - self.last_bridge_state_time
                > self.config.bridge_state_timeout_sec
        ):
            self._enter_failsafe(
                "Task 3 bridge state watchdog zaman aşımı.",
                now=now,
            )
            return False
        if self.bridge_connected is not True:
            self._enter_failsafe(
                "Task 3 bridge bağlantısı yok.",
                now=now,
            )
            return False
        if self.bridge_armed is not True:
            self._enter_failsafe(
                "Task 3 araç ARM değil.",
                now=now,
            )
            return False
        return True

    def _check_geofence(self, now):
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
                f"{distance:.1f}m > {self.config.geofence_radius_m:.1f}m.",
                now=now,
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

    def _record_confirmation(self, target, now, frame_id=None):
        if target is None:
            self.confirmation_samples.clear()
            self.confirmation_last_time = None
            self.confirmation_last_frame_id = None
            return None

        if (
                frame_id is not None
                and self.confirmation_last_frame_id is not None
        ):
            try:
                is_new_frame = (
                    float(frame_id)
                    > float(self.confirmation_last_frame_id)
                )
            except (TypeError, ValueError):
                is_new_frame = frame_id != self.confirmation_last_frame_id
            if not is_new_frame:
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
        self.confirmation_last_frame_id = frame_id
        if len(self.confirmation_samples) < self.config.confirmation_required:
            return None

        recent = list(self.confirmation_samples)[
            -self.config.confirmation_required:
        ]
        angles = [sample["angle"] for sample in recent]
        distances = [sample["distance"] for sample in recent]
        distance_median = sorted(distances)[len(distances) // 2]
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

    def _enter_search(self, now, reason, *, recenter=False):
        self.search_controller.enter_search(
            self.current_heading,
            self.current_lat,
            self.current_lon,
            now,
            recenter=recenter,
        )
        self.confirmation_samples.clear()
        self.confirmation_last_time = None
        self.confirmation_last_frame_id = None
        self.last_target = None
        self._set_state(MissionState.SEARCH, now, reason)

    def _begin_attack_confirmation(self, target, now, frame_id=None):
        self.confirmation_samples.clear()
        self.confirmation_samples.append(target)
        self.confirmation_last_time = now
        self.confirmation_last_frame_id = frame_id
        self.last_target = target
        self.last_target_angle = target["angle"]
        self._set_state(
            MissionState.ATTACK_CONFIRM,
            now,
            f"ilk hedef karesi; {self.config.confirmation_required} "
            "ayrı tutarlı kare bekleniyor",
        )
        self._publish_motion(
            linear_x=self.config.attack_confirm_speed,
            angular_z=self._steering_command(target["angle"]),
            reason=(
                f"low-speed target confirmation 1/"
                f"{self.config.confirmation_required}"
            ),
        )

    def _pause_for_uncertain_target_data(self, now, reason):
        self.search_controller.pause(self.current_heading, now)
        self.confirmation_samples.clear()
        self.confirmation_last_time = None
        self.confirmation_last_frame_id = None
        self.last_target = None
        self._stop()
        if self.state != MissionState.SEARCH:
            self._set_state(
                MissionState.SEARCH,
                now,
                reason,
            )
        self.logger.warn(
            f"Task 3 GUIDED arama veri nedeniyle duraklatıldı: {reason}",
            throttle_duration_sec=1.0,
        )

    def _update_wait_guided_search(self, now):
        self._stop()
        if not self._ensure_mode(DRIVE_MODE, now):
            return
        self._enter_search(
            now,
            "GUIDED teyit edildi; sabit eksenli arama başlıyor",
        )

    def _update_search(self, detections, now, vision_frame_id=None):
        target = self._select_target(detections)
        if self.target_data_uncertain:
            self._pause_for_uncertain_target_data(
                now,
                f"hedef verisi belirsiz: {self.target_data_uncertain_reason}",
            )
            return
        if target is not None:
            self._begin_attack_confirmation(
                target,
                now,
                frame_id=vision_frame_id,
            )
            return

        decision = self.search_controller.step(
            self.current_heading,
            self.current_lat,
            self.current_lon,
            now,
        )
        if decision.failed:
            self._enter_failsafe(
                f"Task 3 arama denetleyicisi başarısız: {decision.reason}",
                now=now,
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

    def _update_attack_confirm(
            self,
            detections,
            now,
            vision_frame_id=None,
    ):
        target = self._select_target(detections)
        if self.target_data_uncertain:
            self._pause_for_uncertain_target_data(
                now,
                f"saldırı teyidinde veri belirsiz: "
                f"{self.target_data_uncertain_reason}",
            )
            return
        if target is None:
            self._enter_search(
                now,
                "altı karelik teyit sırasında hedef kayboldu",
                recenter=True,
            )
            return

        confirmed = self._record_confirmation(
            target,
            now,
            frame_id=vision_frame_id,
        )
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
            self._set_state(
                MissionState.RAM,
                now,
                f"{self.config.confirmation_required} ayrı tutarlı kare; "
                "doğrudan RAM",
            )
            self._publish_motion(
                linear_x=self.config.ram_speed,
                angular_z=self._steering_command(confirmed["angle"]),
                reason="direct ram start",
            )
            return

        if (
                now - self.state_started_at
                >= self.config.attack_confirm_timeout_sec
        ):
            self._enter_search(
                now,
                "altı karelik saldırı teyidi zaman aşımı",
                recenter=True,
            )
            return

        self._publish_motion(
            linear_x=self.config.attack_confirm_speed,
            angular_z=self._steering_command(latest["angle"]),
            reason=(
                f"low-speed target confirmation "
                f"{len(self.confirmation_samples)}/"
                f"{self.config.confirmation_required}"
            ),
        )

    def _ram_steering(self, detections):
        target = self._select_target(detections)
        if target is None or self.target_data_uncertain:
            return 0.0
        if self.last_target is not None and self._target_is_consistent(target):
            target = self._filter_target(target, self.last_target)
        self.last_target = target
        self.last_target_angle = target["angle"]
        return self._steering_command(target["angle"])

    def _valid_current_gps(self):
        return (
            self.current_lat is not None
            and self.current_lon is not None
            and math.isfinite(float(self.current_lat))
            and math.isfinite(float(self.current_lon))
            and not (
                abs(float(self.current_lat)) < 1e-6
                and abs(float(self.current_lon)) < 1e-6
            )
        )

    def _register_impact(self, now, source):
        if not self._valid_current_gps():
            self._enter_failsafe(
                "Task 3 çarpışma GPS'i geçersiz; temas kaydedilemedi.",
                now=now,
            )
            return False

        impact_count = self.impact_controller.register_impact()
        event = {
            "lat": float(self.current_lat),
            "lon": float(self.current_lon),
            "recorded_at": float(now),
            "impact_count": impact_count,
            "source": str(source),
        }
        self.impact_events.append(event)
        if self.impact_target_gps is None:
            self.impact_target_gps = dict(event)
            self.logger.info(
                "Task 3 çarpışma GPS çapası kaydedildi: "
                f"lat={event['lat']:.7f}, lon={event['lon']:.7f}, "
                f"impact_count={impact_count}."
            )
        else:
            self.logger.info(
                "Task 3 kayıtlı GPS'e dönüş teması: "
                f"lat={event['lat']:.7f}, lon={event['lon']:.7f}, "
                f"impact_count={impact_count}."
            )
        return True

    def _enter_final_loiter(self, now, reason):
        self._stop()
        self._set_state(MissionState.FINAL_LOITER, now, reason)

    def _update_ram(self, detections, now):
        elapsed = now - self.state_started_at
        decision = self.impact_controller.ram_decision(
            elapsed,
            angular_z=self._ram_steering(detections),
        )
        if decision.action == ImpactAction.RAM_MOTION:
            self._publish_motion(
                linear_x=decision.linear_x,
                angular_z=decision.angular_z,
                reason=decision.reason,
            )
            return

        if not self._register_impact(now, "initial_ram"):
            return
        if self.impact_count >= self.config.required_impact_count:
            self._enter_final_loiter(now, decision.reason)
            return

        self._set_state(
            MissionState.POST_IMPACT_ADVANCE,
            now,
            decision.reason,
        )
        self._publish_motion(
            linear_x=self.config.post_impact_forward_speed,
            angular_z=0.0,
            reason="post-impact forward advance start",
        )

    def _publish_impact_return_target(self):
        publish_set_position(
            self.topics.position_target_pub,
            self.impact_target_gps["lat"],
            self.impact_target_gps["lon"],
        )

    def _update_post_impact_advance(self, now):
        decision = self.impact_controller.post_impact_decision(
            now - self.state_started_at
        )
        if decision.action == ImpactAction.POST_IMPACT_MOTION:
            self._publish_motion(
                linear_x=decision.linear_x,
                angular_z=decision.angular_z,
                reason=decision.reason,
            )
            return

        if self.impact_target_gps is None:
            self._enter_failsafe(
                "Task 3 kayıtlı çarpışma GPS'i yok.",
                now=now,
            )
            return
        self._stop()
        departure_distance = calculate_gps_distance(
            self.current_lat,
            self.current_lon,
            self.impact_target_gps["lat"],
            self.impact_target_gps["lon"],
        )
        self.impact_return_departed = (
            departure_distance > self.config.impact_return_tolerance_m
        )
        self._set_state(
            MissionState.RETURN_TO_IMPACT,
            now,
            decision.reason,
        )
        self._publish_impact_return_target()

    def _update_return_to_impact(self, now):
        if self.impact_target_gps is None:
            self._enter_failsafe(
                "Task 3 kayıtlı çarpışma GPS'i yok.",
                now=now,
            )
            return
        if (
                now - self.state_started_at
                >= self.config.impact_return_timeout_sec
        ):
            self._enter_failsafe(
                "Task 3 çarpışma GPS'ine dönüş zaman aşımı.",
                now=now,
            )
            return

        distance = calculate_gps_distance(
            self.current_lat,
            self.current_lon,
            self.impact_target_gps["lat"],
            self.impact_target_gps["lon"],
        )
        if distance > self.config.impact_return_tolerance_m:
            self.impact_return_departed = True
        if (
                self.impact_return_departed
                and distance <= self.config.impact_return_tolerance_m
        ):
            if not self._register_impact(now, "gps_return"):
                return
            if self.impact_count >= self.config.required_impact_count:
                self._enter_final_loiter(
                    now,
                    "gerekli GPS dönüş temasları tamamlandı",
                )
                return
            self._set_state(
                MissionState.POST_IMPACT_ADVANCE,
                now,
                f"{self.impact_count}. temas; yeniden ileri çıkılıyor",
            )
            self._publish_motion(
                linear_x=self.config.post_impact_forward_speed,
                angular_z=0.0,
                reason="next post-impact forward advance",
            )
            return

        self._publish_impact_return_target()
        departure_status = (
            "confirmed"
            if self.impact_return_departed
            else "waiting"
        )
        self.logger.info(
            "Task 3 kayıtlı çarpışma GPS'ine dönüyor: "
            f"remaining={distance:.2f}m, "
            f"departure={departure_status}, "
            f"target=({self.impact_target_gps['lat']:.7f}, "
            f"{self.impact_target_gps['lon']:.7f}).",
            throttle_duration_sec=1.0,
        )

    def _update_final_loiter(self, now):
        self._stop()
        if (
                self.bridge_connected is not True
                or self.bridge_armed is not True
                or self.last_bridge_state_time is None
                or now - self.last_bridge_state_time
                > self.config.bridge_state_timeout_sec
        ):
            self.logger.error(
                "Task 3 final LOITER için güncel/operasyonel bridge state yok."
            )
            self.finished = False
            self._set_state(
                MissionState.FAILSAFE,
                now,
                "final LOITER bridge state doğrulanamadı",
            )
            return
        if not self._ensure_mode(self.config.loiter_mode, now):
            return
        self.finished = True
        self._set_state(
            MissionState.FINISHED,
            now,
            "LOITER heartbeat teyit edildi; görev tamamlandı",
        )

    def _update_failsafe_loiter(self, now):
        self._stop()
        if (
                self.bridge_connected is not True
                or self.last_bridge_state_time is None
                or now - self.last_bridge_state_time
                > self.config.bridge_state_timeout_sec
        ):
            self.logger.error(
                "Task 3 FAILSAFE LOITER bridge üzerinden doğrulanamadı."
            )
            self._set_state(
                MissionState.FAILSAFE,
                now,
                "failsafe LOITER bridge üzerinden doğrulanamadı",
            )
            return
        if not self._ensure_mode(self.config.loiter_mode, now):
            return
        self._set_state(
            MissionState.FAILSAFE,
            now,
            "LOITER heartbeat teyit edildi",
        )

    def update(
            self,
            detections,
            now=None,
            vision_fresh=True,
            vision_frame_id=None,
    ):
        now = self._now(now)
        self._publish_state(now)
        if self.state == MissionState.FINISHED:
            self._stop()
            return
        if self.state == MissionState.FAILSAFE:
            self._stop()
            return
        if self.state == MissionState.FINAL_LOITER:
            self._update_final_loiter(now)
            return
        if self.state == MissionState.FAILSAFE_LOITER:
            self._update_failsafe_loiter(now)
            return

        if not self._check_navigation(now):
            return
        if not self._check_bridge(now):
            return
        if not self._check_geofence(now):
            return

        if self.mission_started_at is None:
            self.reset_for_entry(
                self.current_lat,
                self.current_lon,
                self.current_heading,
                now=now,
            )
            return
        if now - self.mission_started_at > self.config.mission_timeout_sec:
            self._enter_failsafe(
                "Task 3 toplam görev zaman aşımı.",
                now=now,
            )
            return

        vision_required = self.impact_target_gps is None
        if vision_required and not vision_fresh:
            self._enter_failsafe(
                "Task 3 vision heartbeat zaman aşımı.",
                now=now,
            )
            return

        if self.state in self._GUIDED_STATES:
            if not self._ensure_mode(DRIVE_MODE, now):
                self._stop()
                return

        if self.state == MissionState.WAIT_GUIDED_SEARCH:
            self._update_wait_guided_search(now)
            return
        if self.state == MissionState.SEARCH:
            self._update_search(
                detections,
                now,
                vision_frame_id=vision_frame_id,
            )
            return
        if self.state == MissionState.ATTACK_CONFIRM:
            self._update_attack_confirm(
                detections,
                now,
                vision_frame_id=vision_frame_id,
            )
            return
        if self.state == MissionState.RAM:
            self._update_ram(detections, now)
            return
        if self.state == MissionState.POST_IMPACT_ADVANCE:
            self._update_post_impact_advance(now)
            return
        if self.state == MissionState.RETURN_TO_IMPACT:
            self._update_return_to_impact(now)
            return
        self._enter_failsafe(
            f"Task 3 beklenmeyen state: {self.state!r}",
            now=now,
        )


class Task3Node(Node):
    def __init__(self, config=None):
        super().__init__("task3_kamikaze_engagement_node")
        self.get_logger().info(
            "Task 3 Kamikaze Engagement düğümü başlatılıyor..."
        )

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
        self.current_detection_frame_id = None
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
        self.data_recorder = MissionDataRecorder(self, ACTIVE_TASK_NAME)

    def publish_active_task(self):
        message = String()
        message.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(message)

    def wait_for_complete_telemetry(self, timeout_sec=10.0):
        return self.data_recorder.wait_for_complete_telemetry(timeout_sec)

    def start_telemetry_recording(self):
        self.data_recorder.start()

    def stop_telemetry_recording(self):
        self.data_recorder.stop()

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
            frame_id = payload.get("frame_id")
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                self.get_logger().warn(
                    "Vision frame_id eksik/geçersiz; kare teyidi bekletiliyor.",
                    throttle_duration_sec=2.0,
                )
                frame_id = -1
            self.current_detections = detections
            self.current_detection_frame_id = frame_id
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
                vision_frame_id=self.current_detection_frame_id,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Zamanlayıcı döngüsünde beklenmeyen hata: {exc}"
            )
            try:
                stop_vehicle(self.mission_topics.cmd_vel_pub)
            except Exception as stop_exc:
                self.get_logger().error(f"Araç durdurulamadı: {stop_exc}")
            self.task._enter_failsafe(
                f"Task 3 timer exception: {exc}",
            )


def main(args=None):
    rclpy.init(args=args)
    node = Task3Node()

    try:
        if not node.wait_for_complete_telemetry(timeout_sec=30.0):
            node.get_logger().error(
                "Roll/pitch/yaw ve araç telemetrisi hazır değil; "
                "Task 3 başlatılmadı."
            )
            return

        if not node.wait_for_vision(timeout_sec=30.0):
            node.get_logger().error("Vision hazır değil; Task 3 başlatılmadı.")
            return

        node.get_logger().info(f"Araç {DRIVE_MODE} moduna alınıyor...")
        if call_set_mode(
                node,
                node.mission_clients.set_mode_client,
                DRIVE_MODE,
        ) is False:
            node.get_logger().error("Başlangıç GUIDED geçişi başarısız.")
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
                "ARM sonrasında güncel vision doğrulanamadı; "
                "Task 3 başlatılmadı."
            )
            return

        try:
            node.start_telemetry_recording()
        except (OSError, RuntimeError, ValueError) as exc:
            node.get_logger().error(f"Görev veri kaydı başlatılamadı: {exc}")
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
                "Task 3 üç temas ve final LOITER tamamlandı."
            )
        elif node.task.state == MissionState.FAILSAFE:
            node.get_logger().error("Task 3 FAILSAFE ile sonlandı.")
    except KeyboardInterrupt:
        node.get_logger().info("Task 3 kullanıcı tarafından durduruldu.")
    finally:
        node.mission_active = False
        node.stop_telemetry_recording()
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
