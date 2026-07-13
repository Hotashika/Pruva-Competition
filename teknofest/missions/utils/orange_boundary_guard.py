"""Turuncu parkur dubalarindan guvenli, kisa GPS hedefi uretir.

Bu modul ROS'a bagimli degildir. Kamera tespitlerini arac merkezli yerel bir
duzleme (x=sag, y=ileri) cevirir, sol/sag duba siralarina cizgi uydurur ve ana
GPS hedefini bu iki sinirin guvenli araligina kirpar.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

EARTH_RADIUS_M = 6378137.0


def _normalize_label(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


ORANGE_BOUNDARY_CLASS_NAMES = frozenset(
    {
        "orange",
        "orange_buoy",
        "orangebuoy",
        "orange_boundary_buoy",
        "orange_cone",
        "orange_marker",
        "turuncu",
        "turuncu_duba",
        "turuncu_samandira",
    }
)


def detection_class_name(detection) -> str:
    """Farkli detector semalarindan normalize sinif adi dondurur."""
    if not isinstance(detection, dict):
        return ""

    for key in ("class", "class_name", "label", "name"):
        if detection.get(key) is not None:
            return _normalize_label(detection[key])
    return ""


def is_orange_boundary_detection(detection) -> bool:
    """Tespit turuncu parkur siniriysa True dondurur."""
    return detection_class_name(detection) in ORANGE_BOUNDARY_CLASS_NAMES


def _safe_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _wrap_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = (
            math.cos(lat1_rad) * math.sin(lat2_rad)
            - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    )
    return math.degrees(math.atan2(y, x)) % 360.0


def _offset_gps(lat: float, lon: float, bearing_deg: float, distance_m: float):
    bearing_rad = math.radians(bearing_deg)
    north_m = distance_m * math.cos(bearing_rad)
    east_m = distance_m * math.sin(bearing_rad)

    lat_rad = math.radians(lat)
    new_lat = lat + math.degrees(north_m / EARTH_RADIUS_M)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6 if cos_lat >= 0.0 else -1e-6
    new_lon = lon + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
    return new_lat, new_lon


@dataclass(frozen=True)
class BoundaryGuardConfig:
    """Sahada parkur olculerine gore ayarlanabilecek parametreler."""

    min_confidence: float = 0.45
    min_detection_distance_m: float = 0.4
    max_detection_distance_m: float = 25.0
    max_detection_angle_deg: float = 78.0
    image_width_px: float = 1920.0
    fallback_horizontal_fov_deg: float = 90.0

    lookahead_m: float = 5.0
    boundary_clearance_m: float = 1.5
    default_corridor_width_m: float = 8.0
    min_corridor_width_m: float = 3.5
    max_corridor_width_m: float = 20.0
    corridor_width_ema_alpha: float = 0.25

    route_weight: float = 0.70
    max_relative_bearing_deg: float = 50.0
    steering_smoothing_alpha: float = 0.45
    boundary_memory_sec: float = 1.0


@dataclass(frozen=True)
class BoundaryGuardDecision:
    status: str
    reason: str
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    relative_bearing_deg: Optional[float] = None
    global_bearing_deg: Optional[float] = None
    corridor_width_m: Optional[float] = None
    left_count: int = 0
    right_count: int = 0
    inferred_boundary: bool = False

    @property
    def should_stop(self) -> bool:
        return self.status == "blocked"

    @property
    def has_target(self) -> bool:
        return self.target_lat is not None and self.target_lon is not None


@dataclass(frozen=True)
class _BoundaryPoint:
    x_right_m: float
    y_forward_m: float
    confidence: float


class OrangeBoundaryGuard:
    """Turuncu duba koridorunu takip eden durumlu guvenlik katmani."""

    def __init__(self, config: Optional[BoundaryGuardConfig] = None):
        self.config = config or BoundaryGuardConfig()
        self.estimated_corridor_width_m = self.config.default_corridor_width_m
        self.last_live_time = None
        self.last_global_bearing_deg = None
        self.last_relative_bearing_deg = None
        self.last_corridor_width_m = None

    def _extract_angle(self, detection) -> Optional[float]:
        for key in (
                "Buoy angle: ",
                "angle_from_center",
                "angle",
                "bearing_angle",
        ):
            angle = _safe_float(detection.get(key))
            if angle is not None:
                return angle

        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        x1 = _safe_float(bbox[0])
        x2 = _safe_float(bbox[2])
        image_width = _safe_float(detection.get("image_width"))
        if image_width is None:
            image_width = self.config.image_width_px
        if x1 is None or x2 is None or image_width <= 0.0:
            return None

        normalized_x = ((x1 + x2) / 2.0 - image_width / 2.0) / (image_width / 2.0)
        return normalized_x * (self.config.fallback_horizontal_fov_deg / 2.0)

    def _to_boundary_point(self, detection) -> Optional[_BoundaryPoint]:
        if not is_orange_boundary_detection(detection):
            return None

        confidence = _safe_float(detection.get("confidence"))
        if confidence is None:
            confidence = _safe_float(detection.get("conf"))
        if confidence is None:
            confidence = 1.0
        if confidence < self.config.min_confidence:
            return None

        distance = None
        for key in ("distance", "distance_m", "depth"):
            distance = _safe_float(detection.get(key))
            if distance is not None:
                break

        angle_deg = self._extract_angle(detection)
        if distance is None or angle_deg is None:
            return None
        if not (
                self.config.min_detection_distance_m
                <= distance
                <= self.config.max_detection_distance_m
        ):
            return None
        if abs(angle_deg) > self.config.max_detection_angle_deg:
            return None

        angle_rad = math.radians(angle_deg)
        x_right_m = distance * math.sin(angle_rad)
        y_forward_m = distance * math.cos(angle_rad)
        if y_forward_m <= 0.15 or abs(x_right_m) <= 0.05:
            return None

        return _BoundaryPoint(x_right_m, y_forward_m, confidence)

    @staticmethod
    def _fit_x_at_y(points: Iterable[_BoundaryPoint], y_eval: float):
        points = list(points)
        if not points:
            return None

        weights = [point.confidence / max(1.0, point.y_forward_m) for point in points]
        weight_sum = sum(weights)
        mean_y = sum(w * p.y_forward_m for w, p in zip(weights, points)) / weight_sum
        mean_x = sum(w * p.x_right_m for w, p in zip(weights, points)) / weight_sum
        denominator = sum(
            w * (p.y_forward_m - mean_y) ** 2 for w, p in zip(weights, points)
        )

        if len(points) < 2 or denominator < 1e-6:
            slope = 0.0
        else:
            slope = sum(
                w * (p.y_forward_m - mean_y) * (p.x_right_m - mean_x)
                for w, p in zip(weights, points)
            ) / denominator
            slope = _clamp(slope, -2.0, 2.0)

        intercept = mean_x - slope * mean_y
        return intercept + slope * y_eval, slope

    def _blocked_or_memory(
            self,
            reason,
            current_lat,
            current_lon,
            current_heading,
            now,
            left_count=0,
            right_count=0,
    ):
        if (
                self.last_live_time is not None
                and self.last_global_bearing_deg is not None
                and now - self.last_live_time <= self.config.boundary_memory_sec
        ):
            target_lat, target_lon = _offset_gps(
                current_lat,
                current_lon,
                self.last_global_bearing_deg,
                self.config.lookahead_m,
            )
            return BoundaryGuardDecision(
                status="memory",
                reason=reason,
                target_lat=target_lat,
                target_lon=target_lon,
                relative_bearing_deg=_wrap_angle_deg(
                    self.last_global_bearing_deg - current_heading
                ),
                global_bearing_deg=self.last_global_bearing_deg,
                corridor_width_m=self.last_corridor_width_m,
                left_count=left_count,
                right_count=right_count,
            )

        return BoundaryGuardDecision(
            status="blocked",
            reason=reason,
            corridor_width_m=self.last_corridor_width_m,
            left_count=left_count,
            right_count=right_count,
        )

    def compute(
            self,
            detections,
            current_lat,
            current_lon,
            current_heading,
            main_target_lat,
            main_target_lon,
            now=None,
    ) -> BoundaryGuardDecision:
        """Ana GPS hedefini gorulen turuncu duba koridoruna kirpar."""
        now = time.monotonic() if now is None else float(now)
        numeric_inputs = (
            current_lat,
            current_lon,
            current_heading,
            main_target_lat,
            main_target_lon,
        )
        if any(_safe_float(value) is None for value in numeric_inputs):
            return BoundaryGuardDecision(status="blocked", reason="invalid_navigation_data")

        current_lat = float(current_lat)
        current_lon = float(current_lon)
        current_heading = float(current_heading) % 360.0
        main_target_lat = float(main_target_lat)
        main_target_lon = float(main_target_lon)

        points = []
        for detection in detections or []:
            point = self._to_boundary_point(detection)
            if point is not None:
                points.append(point)

        left_points = [point for point in points if point.x_right_m < 0.0]
        right_points = [point for point in points if point.x_right_m > 0.0]
        left_fit = self._fit_x_at_y(left_points, self.config.lookahead_m)
        right_fit = self._fit_x_at_y(right_points, self.config.lookahead_m)

        if left_fit is None and right_fit is None:
            return self._blocked_or_memory(
                "orange_boundaries_not_visible",
                current_lat,
                current_lon,
                current_heading,
                now,
            )

        inferred_boundary = False
        if left_fit is not None:
            left_x, left_slope = left_fit
        else:
            right_x, right_slope = right_fit
            left_x = right_x - self.estimated_corridor_width_m
            left_slope = right_slope
            inferred_boundary = True

        if right_fit is not None:
            right_x, right_slope = right_fit
        else:
            right_x = left_x + self.estimated_corridor_width_m
            right_slope = left_slope
            inferred_boundary = True

        measured_width = right_x - left_x
        if not (
                self.config.min_corridor_width_m
                <= measured_width
                <= self.config.max_corridor_width_m
        ):
            return self._blocked_or_memory(
                "invalid_corridor_width",
                current_lat,
                current_lon,
                current_heading,
                now,
                len(left_points),
                len(right_points),
            )

        if not inferred_boundary:
            alpha = self.config.corridor_width_ema_alpha
            self.estimated_corridor_width_m = (
                    (1.0 - alpha) * self.estimated_corridor_width_m
                    + alpha * measured_width
            )

        safe_left_x = left_x + self.config.boundary_clearance_m
        safe_right_x = right_x - self.config.boundary_clearance_m
        if safe_left_x >= safe_right_x:
            return self._blocked_or_memory(
                "corridor_too_narrow_for_clearance",
                current_lat,
                current_lon,
                current_heading,
                now,
                len(left_points),
                len(right_points),
            )

        route_bearing = _bearing_deg(
            current_lat,
            current_lon,
            main_target_lat,
            main_target_lon,
        )
        route_relative = _wrap_angle_deg(route_bearing - current_heading)
        route_relative = _clamp(route_relative, -75.0, 75.0)
        route_x = self.config.lookahead_m * math.tan(math.radians(route_relative))

        clamped_route_x = _clamp(route_x, safe_left_x, safe_right_x)
        corridor_center_x = (left_x + right_x) / 2.0
        desired_x = (
                self.config.route_weight * clamped_route_x
                + (1.0 - self.config.route_weight) * corridor_center_x
        )
        desired_x = _clamp(desired_x, safe_left_x, safe_right_x)

        desired_relative = math.degrees(
            math.atan2(desired_x, self.config.lookahead_m)
        )
        desired_relative = _clamp(
            desired_relative,
            -self.config.max_relative_bearing_deg,
            self.config.max_relative_bearing_deg,
        )

        safe_left_angle = math.degrees(
            math.atan2(safe_left_x, self.config.lookahead_m)
        )
        safe_right_angle = math.degrees(
            math.atan2(safe_right_x, self.config.lookahead_m)
        )

        if self.last_relative_bearing_deg is not None:
            delta = _wrap_angle_deg(
                desired_relative - self.last_relative_bearing_deg
            )
            desired_relative = self.last_relative_bearing_deg + (
                    self.config.steering_smoothing_alpha * delta
            )

        allowed_left_angle = max(
            safe_left_angle,
            -self.config.max_relative_bearing_deg,
        )
        allowed_right_angle = min(
            safe_right_angle,
            self.config.max_relative_bearing_deg,
        )
        if allowed_left_angle > allowed_right_angle:
            return self._blocked_or_memory(
                "corridor_requires_excessive_turn",
                current_lat,
                current_lon,
                current_heading,
                now,
                len(left_points),
                len(right_points),
            )

        desired_relative = _clamp(
            desired_relative,
            allowed_left_angle,
            allowed_right_angle,
        )
        global_bearing = (current_heading + desired_relative) % 360.0
        target_distance = self.config.lookahead_m / max(
            0.2,
            math.cos(math.radians(desired_relative)),
        )
        target_lat, target_lon = _offset_gps(
            current_lat,
            current_lon,
            global_bearing,
            target_distance,
        )

        self.last_live_time = now
        self.last_global_bearing_deg = global_bearing
        self.last_relative_bearing_deg = desired_relative
        self.last_corridor_width_m = measured_width

        reason = "both_boundaries" if not inferred_boundary else "single_boundary_inferred"
        return BoundaryGuardDecision(
            status="live",
            reason=reason,
            target_lat=target_lat,
            target_lon=target_lon,
            relative_bearing_deg=desired_relative,
            global_bearing_deg=global_bearing,
            corridor_width_m=measured_width,
            left_count=len(left_points),
            right_count=len(right_points),
            inferred_boundary=inferred_boundary,
        )
