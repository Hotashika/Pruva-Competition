"""ROS-independent search state and decisions for TEKNOFEST Task 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


class SearchPhase(Enum):
    MOVE_TO_ARC_START = auto()
    SWEEP_ARC = auto()
    RETURN_TO_BASE_HEADING = auto()
    ADVANCE = auto()


@dataclass(frozen=True)
class SearchDecision:
    linear_x: float = 0.0
    angular_z: float = 0.0
    reason: str | None = None
    phase_changed: bool = False
    failed: bool = False


def angle_error_deg(target_deg, current_deg) -> float:
    return (
        float(target_deg) - float(current_deg) + 180.0
    ) % 360.0 - 180.0


class Task3SearchController:
    def __init__(self, config):
        self.config = config
        self.phase = SearchPhase.MOVE_TO_ARC_START
        self.base_heading = None
        self.sweep_deg = float(config.search_initial_sweep_deg)
        self.sweep_progress_deg = 0.0
        self.last_heading = None
        self.last_update_at = None
        self.phase_started_at = None
        self.cycle_index = 0

    def reset_for_entry(self, current_heading, now):
        self.sweep_deg = float(self.config.search_initial_sweep_deg)
        self.cycle_index = 0
        self._start_arc(current_heading, now)

    def enter_search(self, current_heading, now):
        self._start_arc(current_heading, now)

    def _start_arc(self, current_heading, now):
        heading = float(current_heading) % 360.0
        self.phase = SearchPhase.MOVE_TO_ARC_START
        self.base_heading = heading
        self.sweep_progress_deg = 0.0
        self.last_heading = heading
        self.last_update_at = now
        self.phase_started_at = now

    def pause(self, current_heading, now):
        previous_update_at = self.last_update_at
        self.last_update_at = now
        if self.phase_started_at is not None and previous_update_at is not None:
            self.phase_started_at += max(0.0, now - previous_update_at)
        self.last_heading = float(current_heading) % 360.0

    def _set_phase(self, phase, current_heading, now):
        self.phase = phase
        self.phase_started_at = now
        self.last_update_at = now
        self.last_heading = float(current_heading) % 360.0
        if phase == SearchPhase.SWEEP_ARC:
            self.sweep_progress_deg = 0.0

    def _turn_timeout_sec(self, span_deg):
        commanded_rate_deg_sec = max(
            1.0,
            math.degrees(abs(float(self.config.search_angular_z))),
        )
        return (
            float(self.config.search_turn_timeout_min_sec)
            + 1.5 * abs(float(span_deg)) / commanded_rate_deg_sec
        )

    def _phase_timed_out(self, now, span_deg):
        if self.phase_started_at is None:
            return False
        return (
            now - self.phase_started_at
            >= self._turn_timeout_sec(span_deg)
        )

    def _heading_command(self, target_heading, current_heading):
        error_deg = angle_error_deg(target_heading, current_heading)
        limit = abs(float(self.config.search_angular_z))
        angular_z = max(
            -limit,
            min(limit, math.radians(error_deg)),
        )
        return error_deg, angular_z

    def _failed_turn(self, description):
        return SearchDecision(
            reason=(
                f"search heading watchdog timeout during {description}; "
                f"phase={self.phase.name}, sweep={self.sweep_deg:.1f}deg"
            ),
            failed=True,
        )

    def _move_to_arc_start(self, current_heading, now):
        half_sweep = self.sweep_deg / 2.0
        target_heading = (self.base_heading - half_sweep) % 360.0
        error_deg, angular_z = self._heading_command(
            target_heading,
            current_heading,
        )
        if abs(error_deg) <= self.config.search_heading_tolerance_deg:
            self._set_phase(SearchPhase.SWEEP_ARC, current_heading, now)
            return SearchDecision(
                reason=(
                    f"arc start reached; sweep={self.sweep_deg:.1f}deg, "
                    f"base={self.base_heading:.1f}deg"
                ),
                phase_changed=True,
            )
        if self._phase_timed_out(now, half_sweep):
            return self._failed_turn("arc-start alignment")
        return SearchDecision(
            angular_z=angular_z,
            reason=(
                f"move to arc start; sweep={self.sweep_deg:.1f}deg, "
                f"target={target_heading:.1f}deg"
            ),
        )

    def _sweep_arc(self, current_heading, now):
        heading_delta = angle_error_deg(current_heading, self.last_heading)
        self.sweep_progress_deg += max(0.0, heading_delta)
        self.last_heading = float(current_heading) % 360.0
        if (
                self.sweep_progress_deg
                >= self.sweep_deg - self.config.search_heading_tolerance_deg
        ):
            self._set_phase(
                SearchPhase.RETURN_TO_BASE_HEADING,
                current_heading,
                now,
            )
            return SearchDecision(
                reason=(
                    f"arc sweep complete; swept="
                    f"{self.sweep_progress_deg:.1f}deg"
                ),
                phase_changed=True,
            )
        if self._phase_timed_out(now, self.sweep_deg):
            return self._failed_turn("arc sweep")
        return SearchDecision(
            angular_z=abs(float(self.config.search_angular_z)),
            reason=(
                f"arc sweep; progress={self.sweep_progress_deg:.1f}/"
                f"{self.sweep_deg:.1f}deg"
            ),
        )

    def _return_to_base_heading(self, current_heading, now):
        error_deg, angular_z = self._heading_command(
            self.base_heading,
            current_heading,
        )
        if abs(error_deg) <= self.config.search_heading_tolerance_deg:
            self._set_phase(SearchPhase.ADVANCE, current_heading, now)
            return SearchDecision(
                reason=(
                    f"base heading restored; advance for "
                    f"{self.config.search_forward_duration_sec:.1f}s"
                ),
                phase_changed=True,
            )
        if self._phase_timed_out(now, self.sweep_deg / 2.0):
            return self._failed_turn("base-heading return")
        return SearchDecision(
            angular_z=angular_z,
            reason=(
                f"return to base heading; "
                f"target={self.base_heading:.1f}deg"
            ),
        )

    def _advance(self, current_heading, now):
        if (
                now - self.phase_started_at
                >= self.config.search_forward_duration_sec
        ):
            self.cycle_index += 1
            self.sweep_deg = min(
                float(self.config.search_max_sweep_deg),
                self.sweep_deg
                + float(self.config.search_sweep_increment_deg),
            )
            self._start_arc(current_heading, now)
            return SearchDecision(
                reason=(
                    f"search advance complete; next sweep="
                    f"{self.sweep_deg:.1f}deg"
                ),
                phase_changed=True,
            )

        _, angular_z = self._heading_command(
            self.base_heading,
            current_heading,
        )
        return SearchDecision(
            linear_x=float(self.config.search_linear_x),
            angular_z=angular_z,
            reason=(
                f"heading-held search advance; "
                f"base={self.base_heading:.1f}deg"
            ),
        )

    def step(self, current_heading, now) -> SearchDecision:
        current_heading = float(current_heading) % 360.0
        self.last_update_at = now
        if self.base_heading is None:
            self._start_arc(current_heading, now)

        if self.phase == SearchPhase.MOVE_TO_ARC_START:
            return self._move_to_arc_start(current_heading, now)
        if self.phase == SearchPhase.SWEEP_ARC:
            return self._sweep_arc(current_heading, now)
        if self.phase == SearchPhase.RETURN_TO_BASE_HEADING:
            return self._return_to_base_heading(current_heading, now)
        if self.phase == SearchPhase.ADVANCE:
            return self._advance(current_heading, now)

        return SearchDecision(
            reason=f"unknown search phase: {self.phase!r}",
            failed=True,
        )
