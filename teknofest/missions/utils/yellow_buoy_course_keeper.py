"""İkinci en yakın sarı dubaya yönelen dinamik parkur hedefi üretir.

Modül ROS'a bağlı değildir. Her ``compute`` çağrısında geçerli sarı duba
tespitlerini mesafeye göre yeniden sıralar, ikinci sıradaki dubanın kamera
açısını araç heading'iyle birleştirir ve kısa bir GPS hedefi döndürür.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional


EARTH_RADIUS_M = 6378137.0


def _normalize_label(value) -> str:
    text = str(value or "").replace("ı", "i").replace("İ", "I")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


YELLOW_BUOY_CLASS_NAMES = frozenset(
    {
        "yellow",
        "yellow_buoy",
        "yellowbuoy",
        "yellow_marker",
        "sari",
        "sari_duba",
        "sari_samandira",
    }
)


def detection_class_name(detection) -> str:
    """Farklı detector şemalarından normalize edilmiş sınıf adı döndürür."""
    if not isinstance(detection, dict):
        return ""

    for key in ("class", "class_name", "label", "name"):
        if detection.get(key) is not None:
            return _normalize_label(detection[key])
    return ""


def is_yellow_buoy_detection(detection) -> bool:
    """Tespit sarı parkur dubasıysa True döndürür."""
    return detection_class_name(detection) in YELLOW_BUOY_CLASS_NAMES


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
class YellowBuoyCourseConfig:
    """Sahada kamera ve kontrol davranışına göre ayarlanabilecek parametreler."""

    min_confidence: float = 0.45
    min_detection_distance_m: float = 0.4
    max_detection_distance_m: float = 30.0
    max_detection_angle_deg: float = 78.0
    image_width_px: float = 1920.0
    fallback_horizontal_fov_deg: float = 90.0
    lookahead_m: float = 5.0
    min_target_distance_m: float = 1.0
    max_relative_bearing_deg: float = 70.0
    steering_smoothing_alpha: float = 0.55
    target_memory_sec: float = 1.0


@dataclass(frozen=True)
class YellowBuoyCourseDecision:
    status: str
    reason: str
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    relative_bearing_deg: Optional[float] = None
    global_bearing_deg: Optional[float] = None
    selected_distance_m: Optional[float] = None
    candidate_count: int = 0

    @property
    def should_stop(self) -> bool:
        return self.status == "blocked"

    @property
    def has_target(self) -> bool:
        return self.target_lat is not None and self.target_lon is not None


@dataclass(frozen=True)
class _YellowBuoy:
    distance_m: float
    angle_deg: float


class YellowBuoyCourseKeeper:
    """Her iterasyonda ikinci en yakın sarı dubayı rota referansı yapar."""

    def __init__(self, config: Optional[YellowBuoyCourseConfig] = None):
        self.config = config or YellowBuoyCourseConfig()
        self.last_live_time = None
        self.last_global_bearing_deg = None
        self.last_relative_bearing_deg = None
        self.last_selected_distance_m = None

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

    def _to_yellow_buoy(self, detection) -> Optional[_YellowBuoy]:
        if not is_yellow_buoy_detection(detection):
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

        return _YellowBuoy(distance_m=distance, angle_deg=angle_deg)

    def _blocked_or_memory(
            self,
            reason,
            current_lat,
            current_lon,
            current_heading,
            now,
            candidate_count,
    ):
        if (
                self.last_live_time is not None
                and self.last_global_bearing_deg is not None
                and now - self.last_live_time <= self.config.target_memory_sec
        ):
            relative_bearing = _wrap_angle_deg(
                self.last_global_bearing_deg - current_heading
            )
            target_lat, target_lon = _offset_gps(
                current_lat,
                current_lon,
                self.last_global_bearing_deg,
                self.config.lookahead_m,
            )
            return YellowBuoyCourseDecision(
                status="memory",
                reason=reason,
                target_lat=target_lat,
                target_lon=target_lon,
                relative_bearing_deg=relative_bearing,
                global_bearing_deg=self.last_global_bearing_deg,
                selected_distance_m=self.last_selected_distance_m,
                candidate_count=candidate_count,
            )

        return YellowBuoyCourseDecision(
            status="blocked",
            reason=reason,
            candidate_count=candidate_count,
        )

    def compute(
            self,
            detections,
            current_lat,
            current_lon,
            current_heading,
            now=None,
    ) -> YellowBuoyCourseDecision:
        """İkinci en yakın geçerli sarı dubaya doğru kısa GPS hedefi üretir."""
        now = time.monotonic() if now is None else float(now)
        numeric_inputs = (current_lat, current_lon, current_heading)
        if any(_safe_float(value) is None for value in numeric_inputs):
            return YellowBuoyCourseDecision(
                status="blocked",
                reason="invalid_navigation_data",
            )

        current_lat = float(current_lat)
        current_lon = float(current_lon)
        current_heading = float(current_heading) % 360.0

        candidates = []
        for detection in detections or []:
            buoy = self._to_yellow_buoy(detection)
            if buoy is not None:
                candidates.append(buoy)
        candidates.sort(key=lambda buoy: buoy.distance_m)

        if len(candidates) < 2:
            return self._blocked_or_memory(
                "fewer_than_two_yellow_buoys",
                current_lat,
                current_lon,
                current_heading,
                now,
                len(candidates),
            )

        selected = candidates[1]
        desired_relative = _clamp(
            selected.angle_deg,
            -self.config.max_relative_bearing_deg,
            self.config.max_relative_bearing_deg,
        )
        if self.last_relative_bearing_deg is not None:
            delta = _wrap_angle_deg(desired_relative - self.last_relative_bearing_deg)
            desired_relative = self.last_relative_bearing_deg + (
                    self.config.steering_smoothing_alpha * delta
            )
            desired_relative = _clamp(
                desired_relative,
                -self.config.max_relative_bearing_deg,
                self.config.max_relative_bearing_deg,
            )

        global_bearing = (current_heading + desired_relative) % 360.0
        target_distance = _clamp(
            selected.distance_m,
            self.config.min_target_distance_m,
            self.config.lookahead_m,
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
        self.last_selected_distance_m = selected.distance_m

        return YellowBuoyCourseDecision(
            status="live",
            reason="second_nearest_yellow_buoy",
            target_lat=target_lat,
            target_lon=target_lon,
            relative_bearing_deg=desired_relative,
            global_bearing_deg=global_bearing,
            selected_distance_m=selected.distance_m,
            candidate_count=len(candidates),
        )
