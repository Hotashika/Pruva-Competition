"""ROS-independent, GPS-bounded search control for TEKNOFEST Task 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


EARTH_RADIUS_M = 6378137.0


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
    target_heading: float | None = None
    heading_error_deg: float | None = None
    along_track_m: float | None = None
    cross_track_m: float | None = None


def angle_error_deg(target_deg, current_deg) -> float:
    return (
        float(target_deg) - float(current_deg) + 180.0
    ) % 360.0 - 180.0


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class Task3SearchController:
    """Scan a widening arc around one immutable mission-entry heading."""

    def __init__(self, config):
        self.config = config
        self.phase = SearchPhase.MOVE_TO_ARC_START
        self.base_heading = None
        self.entry_lat = None
        self.entry_lon = None
        self.sweep_deg = float(config.search_initial_sweep_deg)
        self.cycle_index = 0

        self.phase_started_at = None
        self.last_update_at = None
        self.heading_settle_started_at = None
        self.recovery_recenter = False

        self.advance_started_along_m = None
        self.advance_best_progress_m = 0.0
        self.advance_last_progress_at = None

    def reset_for_entry(
            self,
            current_heading,
            current_lat,
            current_lon,
            now,
    ):
        heading = self._valid_heading(current_heading)
        lat, lon = self._valid_gps(current_lat, current_lon)
        self.base_heading = heading
        self.entry_lat = lat
        self.entry_lon = lon
        self.sweep_deg = float(self.config.search_initial_sweep_deg)
        self.cycle_index = 0
        self.recovery_recenter = False
        self._set_phase(
            SearchPhase.MOVE_TO_ARC_START,
            heading,
            lat,
            lon,
            now,
        )

    def enter_search(
            self,
            current_heading,
            current_lat,
            current_lon,
            now,
            *,
            recenter=False,
    ):
        heading = self._valid_heading(current_heading)
        lat, lon = self._valid_gps(current_lat, current_lon)
        if self.base_heading is None:
            self.reset_for_entry(heading, lat, lon, now)
            return

        self.recovery_recenter = bool(recenter)
        phase = (
            SearchPhase.RETURN_TO_BASE_HEADING
            if recenter
            else SearchPhase.MOVE_TO_ARC_START
        )
        self._set_phase(phase, heading, lat, lon, now)

    def pause(self, current_heading, now):
        self._valid_heading(current_heading)
        now = float(now)
        if self.last_update_at is None:
            self.last_update_at = now
            return

        paused_for = max(0.0, now - self.last_update_at)
        self.last_update_at = now
        if self.phase_started_at is not None:
            self.phase_started_at += paused_for
        if self.heading_settle_started_at is not None:
            self.heading_settle_started_at += paused_for
        if self.advance_last_progress_at is not None:
            self.advance_last_progress_at += paused_for

    @staticmethod
    def _valid_heading(value):
        heading = _finite_float(value)
        if heading is None:
            raise ValueError("search heading is not finite")
        return heading % 360.0

    @staticmethod
    def _valid_gps(lat, lon):
        latitude = _finite_float(lat)
        longitude = _finite_float(lon)
        if latitude is None or longitude is None:
            raise ValueError("search GPS is not finite")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("search latitude is outside [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("search longitude is outside [-180, 180]")
        if abs(latitude) < 1e-6 and abs(longitude) < 1e-6:
            raise ValueError("search GPS cannot be (0, 0)")
        return latitude, longitude

    def _position_on_search_axis(self, lat, lon):
        lat, lon = self._valid_gps(lat, lon)
        mean_lat_rad = math.radians((self.entry_lat + lat) / 2.0)
        north_m = math.radians(lat - self.entry_lat) * EARTH_RADIUS_M
        east_m = (
            math.radians(lon - self.entry_lon)
            * EARTH_RADIUS_M
            * math.cos(mean_lat_rad)
        )
        axis_rad = math.radians(self.base_heading)
        along_track_m = (
            north_m * math.cos(axis_rad)
            + east_m * math.sin(axis_rad)
        )
        cross_track_m = (
            -north_m * math.sin(axis_rad)
            + east_m * math.cos(axis_rad)
        )
        return along_track_m, cross_track_m

    def _set_phase(self, phase, current_heading, lat, lon, now):
        self.phase = phase
        self.phase_started_at = float(now)
        self.last_update_at = float(now)
        self.heading_settle_started_at = None
        if phase == SearchPhase.ADVANCE:
            along_track_m, _ = self._position_on_search_axis(lat, lon)
            self.advance_started_along_m = along_track_m
            self.advance_best_progress_m = 0.0
            self.advance_last_progress_at = float(now)
        else:
            self.advance_started_along_m = None
            self.advance_best_progress_m = 0.0
            self.advance_last_progress_at = None

    def _phase_timed_out(self, now):
        if self.phase_started_at is None:
            return False
        return (
            float(now) - self.phase_started_at
            >= float(self.config.search_turn_timeout_sec)
        )

    def _angular_command_for_error(self, error_deg):
        tolerance = float(self.config.search_heading_tolerance_deg)
        if abs(error_deg) <= tolerance:
            return 0.0

        limit = abs(float(self.config.search_angular_z))
        minimum = min(
            limit,
            abs(float(self.config.search_min_angular_z)),
        )
        angular_z = max(
            -limit,
            min(
                limit,
                float(self.config.search_heading_kp) * error_deg,
            ),
        )
        if abs(angular_z) < minimum:
            angular_z = math.copysign(minimum, error_deg)
        return angular_z

    def _heading_command(self, target_heading, current_heading):
        error_deg = angle_error_deg(target_heading, current_heading)
        return error_deg, self._angular_command_for_error(error_deg)

    def _heading_is_settled(self, error_deg, now):
        if abs(error_deg) > self.config.search_heading_tolerance_deg:
            self.heading_settle_started_at = None
            return False
        if self.heading_settle_started_at is None:
            self.heading_settle_started_at = float(now)
            return False
        return (
            float(now) - self.heading_settle_started_at
            >= float(self.config.search_heading_settle_sec)
        )

    def _turn_decision(
            self,
            target_heading,
            current_heading,
            now,
            description,
            *,
            directed_error_deg=None,
    ):
        if directed_error_deg is None:
            error_deg, angular_z = self._heading_command(
                target_heading,
                current_heading,
            )
        else:
            error_deg = float(directed_error_deg)
            angular_z = self._angular_command_for_error(error_deg)
        if (
                self._phase_timed_out(now)
                and abs(error_deg)
                > float(self.config.search_heading_tolerance_deg)
        ):
            return SearchDecision(
                reason=(
                    f"search heading watchdog timeout during {description}; "
                    f"phase={self.phase.name}, target={target_heading:.1f}deg, "
                    f"current={current_heading:.1f}deg"
                ),
                failed=True,
                target_heading=target_heading,
                heading_error_deg=error_deg,
            )
        return SearchDecision(
            angular_z=angular_z,
            reason=(
                f"{description}; target={target_heading:.1f}deg, "
                f"error={error_deg:+.1f}deg"
            ),
            target_heading=target_heading,
            heading_error_deg=error_deg,
        )

    def _move_to_arc_start(self, current_heading, lat, lon, now):
        target_heading = (
            self.base_heading - self.sweep_deg / 2.0
        ) % 360.0
        decision = self._turn_decision(
            target_heading,
            current_heading,
            now,
            "move to fixed-axis arc start",
        )
        if decision.failed:
            return decision
        if not self._heading_is_settled(decision.heading_error_deg, now):
            return decision

        self._set_phase(
            SearchPhase.SWEEP_ARC,
            current_heading,
            lat,
            lon,
            now,
        )
        return SearchDecision(
            reason=(
                f"arc start settled; sweep={self.sweep_deg:.1f}deg, "
                f"axis={self.base_heading:.1f}deg"
            ),
            phase_changed=True,
            target_heading=target_heading,
            heading_error_deg=decision.heading_error_deg,
        )

    def _sweep_arc(self, current_heading, lat, lon, now):
        start_heading = (
            self.base_heading - self.sweep_deg / 2.0
        ) % 360.0
        target_heading = (
            self.base_heading + self.sweep_deg / 2.0
        ) % 360.0
        clockwise_progress_deg = (
            current_heading - start_heading
        ) % 360.0
        tolerance = float(self.config.search_heading_tolerance_deg)
        if clockwise_progress_deg >= 360.0 - tolerance:
            clockwise_progress_deg = 0.0
        remaining_deg = self.sweep_deg - clockwise_progress_deg
        directed_error_deg = (
            remaining_deg
            if remaining_deg > tolerance
            else angle_error_deg(target_heading, current_heading)
        )
        decision = self._turn_decision(
            target_heading,
            current_heading,
            now,
            "sweep to fixed absolute endpoint",
            directed_error_deg=directed_error_deg,
        )
        if decision.failed:
            return decision
        if not self._heading_is_settled(decision.heading_error_deg, now):
            return decision

        self._set_phase(
            SearchPhase.RETURN_TO_BASE_HEADING,
            current_heading,
            lat,
            lon,
            now,
        )
        return SearchDecision(
            reason=(
                f"arc endpoint settled; sweep={self.sweep_deg:.1f}deg, "
                f"axis={self.base_heading:.1f}deg"
            ),
            phase_changed=True,
            target_heading=target_heading,
            heading_error_deg=decision.heading_error_deg,
        )

    def _return_to_base_heading(self, current_heading, lat, lon, now):
        target_heading = self.base_heading
        decision = self._turn_decision(
            target_heading,
            current_heading,
            now,
            "return to immutable search axis",
        )
        if decision.failed:
            return decision
        if not self._heading_is_settled(decision.heading_error_deg, now):
            return decision

        if self.recovery_recenter:
            self.recovery_recenter = False
            next_phase = SearchPhase.MOVE_TO_ARC_START
            reason = "search recovery recentered; restarting current arc"
        else:
            next_phase = SearchPhase.ADVANCE
            reason = (
                f"search axis settled; advancing "
                f"{self.config.search_advance_distance_m:.1f}m"
            )
        self._set_phase(next_phase, current_heading, lat, lon, now)
        return SearchDecision(
            reason=reason,
            phase_changed=True,
            target_heading=target_heading,
            heading_error_deg=decision.heading_error_deg,
        )

    def _advance(self, current_heading, lat, lon, now):
        along_track_m, cross_track_m = self._position_on_search_axis(
            lat,
            lon,
        )
        progress_m = along_track_m - self.advance_started_along_m
        if progress_m >= float(self.config.search_advance_distance_m):
            self.cycle_index += 1
            self.sweep_deg = min(
                float(self.config.search_max_sweep_deg),
                self.sweep_deg
                + float(self.config.search_sweep_increment_deg),
            )
            self._set_phase(
                SearchPhase.MOVE_TO_ARC_START,
                current_heading,
                lat,
                lon,
                now,
            )
            return SearchDecision(
                reason=(
                    f"GPS search advance complete; progress={progress_m:.2f}m, "
                    f"next_sweep={self.sweep_deg:.1f}deg"
                ),
                phase_changed=True,
                along_track_m=along_track_m,
                cross_track_m=cross_track_m,
            )

        if abs(cross_track_m) > float(
                self.config.search_cross_track_limit_m
        ):
            return SearchDecision(
                reason=(
                    f"search corridor exceeded; "
                    f"cross_track={cross_track_m:+.2f}m, "
                    f"limit={self.config.search_cross_track_limit_m:.2f}m"
                ),
                failed=True,
                along_track_m=along_track_m,
                cross_track_m=cross_track_m,
            )

        if (
                progress_m
                >= self.advance_best_progress_m
                + float(self.config.search_progress_min_m)
        ):
            self.advance_best_progress_m = progress_m
            self.advance_last_progress_at = float(now)

        if (
                float(now) - self.advance_last_progress_at
                >= float(self.config.search_no_progress_timeout_sec)
        ):
            return SearchDecision(
                reason=(
                    f"search advance made no GPS progress; "
                    f"progress={progress_m:.2f}m"
                ),
                failed=True,
                along_track_m=along_track_m,
                cross_track_m=cross_track_m,
            )

        if (
                float(now) - self.phase_started_at
                >= float(self.config.search_advance_timeout_sec)
        ):
            return SearchDecision(
                reason=(
                    f"search advance timeout; progress={progress_m:.2f}/"
                    f"{self.config.search_advance_distance_m:.2f}m"
                ),
                failed=True,
                along_track_m=along_track_m,
                cross_track_m=cross_track_m,
            )

        correction_deg = max(
            -float(self.config.search_advance_heading_limit_deg),
            min(
                float(self.config.search_advance_heading_limit_deg),
                -cross_track_m
                * float(self.config.search_cross_track_kp_deg_per_m),
            ),
        )
        target_heading = (self.base_heading + correction_deg) % 360.0
        error_deg, angular_z = self._heading_command(
            target_heading,
            current_heading,
        )
        linear_x = float(self.config.search_linear_x)
        if abs(error_deg) > float(
                self.config.search_advance_heading_limit_deg
        ):
            linear_x = 0.0

        return SearchDecision(
            linear_x=linear_x,
            angular_z=angular_z,
            reason=(
                f"GPS-bounded search advance; progress={progress_m:.2f}/"
                f"{self.config.search_advance_distance_m:.2f}m, "
                f"cross_track={cross_track_m:+.2f}m, "
                f"target={target_heading:.1f}deg"
            ),
            target_heading=target_heading,
            heading_error_deg=error_deg,
            along_track_m=along_track_m,
            cross_track_m=cross_track_m,
        )

    def step(
            self,
            current_heading,
            current_lat,
            current_lon,
            now,
    ) -> SearchDecision:
        try:
            heading = self._valid_heading(current_heading)
            lat, lon = self._valid_gps(current_lat, current_lon)
        except ValueError as exc:
            return SearchDecision(
                reason=f"invalid search navigation input: {exc}",
                failed=True,
            )

        if self.base_heading is None:
            try:
                self.reset_for_entry(heading, lat, lon, now)
            except ValueError as exc:
                return SearchDecision(
                    reason=f"search initialization failed: {exc}",
                    failed=True,
                )
        self.last_update_at = float(now)

        if self.phase == SearchPhase.MOVE_TO_ARC_START:
            return self._move_to_arc_start(heading, lat, lon, now)
        if self.phase == SearchPhase.SWEEP_ARC:
            return self._sweep_arc(heading, lat, lon, now)
        if self.phase == SearchPhase.RETURN_TO_BASE_HEADING:
            return self._return_to_base_heading(heading, lat, lon, now)
        if self.phase == SearchPhase.ADVANCE:
            return self._advance(heading, lat, lon, now)

        return SearchDecision(
            reason=f"unknown search phase: {self.phase!r}",
            failed=True,
        )
