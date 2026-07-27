"""ROS-independent search state and decisions for TEKNOFEST Task 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class SearchDecision:
    relocate: bool
    linear_x: Optional[float] = None
    angular_z: Optional[float] = None
    reason: Optional[str] = None
    leg_changed: bool = False


def angle_error_deg(target_deg, current_deg) -> float:
    return (
        float(target_deg) - float(current_deg) + 180.0
    ) % 360.0 - 180.0


def offset_gps(lat, lon, bearing_deg, distance_m):
    bearing_rad = math.radians(float(bearing_deg))
    north_m = float(distance_m) * math.cos(bearing_rad)
    east_m = float(distance_m) * math.sin(bearing_rad)
    latitude = float(lat) + math.degrees(north_m / EARTH_RADIUS_M)
    cos_lat = math.cos(math.radians(float(lat)))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6 if cos_lat >= 0.0 else -1e-6
    longitude = float(lon) + math.degrees(
        east_m / (EARTH_RADIUS_M * cos_lat)
    )
    return {"lat": latitude, "lon": longitude}


class Task3SearchController:
    def __init__(self, config):
        self.config = config
        self.last_heading = None
        self.last_update_at = None
        self.accumulated_degrees = 0.0
        self.leg_started_at = None
        self.leg_index = 0
        self.direction = 1.0
        self.cycle_index = 0
        self.point_index = 0
        self.target = None

    def reset_for_entry(self, current_heading, now):
        self.last_heading = current_heading
        self.last_update_at = now
        self.accumulated_degrees = 0.0
        self.leg_started_at = now
        self.leg_index = 0
        self.direction = 1.0
        self.cycle_index = 0
        self.point_index = 0
        self.target = None

    def enter_search(self, current_heading, now):
        self.last_heading = current_heading
        self.last_update_at = now
        self.accumulated_degrees = 0.0
        self.leg_started_at = now
        self.leg_index = 0
        self.direction = 1.0 if self.cycle_index % 2 == 0 else -1.0
        self.cycle_index += 1
        self.target = None

    def pause(self, current_heading, now):
        previous_update_at = self.last_update_at
        self.last_update_at = now
        if self.leg_started_at is not None and previous_update_at is not None:
            self.leg_started_at += max(0.0, now - previous_update_at)
        self.last_heading = current_heading

    def step(self, current_heading, now) -> SearchDecision:
        self.last_update_at = now
        if self.last_heading is None:
            self.last_heading = current_heading
        if self.leg_started_at is None:
            self.leg_started_at = now

        heading_delta = abs(
            angle_error_deg(current_heading, self.last_heading)
        )
        self.accumulated_degrees += heading_delta
        self.last_heading = current_heading

        leg_changed = False
        if (
                self.accumulated_degrees
                >= self.config.search_leg_sweep_deg
                or now - self.leg_started_at
                >= self.config.search_leg_timeout_sec
        ):
            self.leg_index += 1
            if self.leg_index >= self.config.search_legs_per_cycle:
                return SearchDecision(relocate=True)
            self.direction *= -1.0
            self.last_heading = current_heading
            self.accumulated_degrees = 0.0
            self.leg_started_at = now
            leg_changed = True

        return SearchDecision(
            relocate=False,
            linear_x=self.config.search_linear_x,
            angular_z=self.direction * self.config.search_angular_z,
            reason=(
                f"S-search leg {self.leg_index + 1}/"
                f"{self.config.search_legs_per_cycle}"
            ),
            leg_changed=leg_changed,
        )

    def next_relocation_target(
            self,
            home_lat,
            home_lon,
            entry_heading,
    ):
        completed_rings = (
            self.point_index // self.config.search_points_per_ring
        )
        max_ring = max(
            1,
            int(
                self.config.search_max_radius_m
                // self.config.search_radius_step_m
            ),
        )
        ring = min(completed_rings + 1, max_ring)
        point_in_ring = (
            self.point_index % self.config.search_points_per_ring
        )
        radius = ring * self.config.search_radius_step_m
        bearing_step = 360.0 / self.config.search_points_per_ring
        bearing = (
            float(entry_heading or 0.0)
            + point_in_ring * bearing_step
        ) % 360.0
        self.point_index += 1
        self.target = offset_gps(
            home_lat,
            home_lon,
            bearing,
            radius,
        )
        return self.target
