"""ROS-independent target handling for TEKNOFEST Task 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class TargetNormalizationResult:
    target: Optional[dict] = None
    data_uncertain_reason: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class TargetSelectionResult:
    target: Optional[dict]
    observed_classes: tuple[str, ...]
    data_uncertain: bool = False
    data_uncertain_reason: Optional[str] = None
    rejection_reason: Optional[str] = None


def finite_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def canonical_class_name(value) -> str:
    text = str(value or "").strip().lower()
    return text.replace("-", "_").replace(" ", "_")


def class_alias_key(value) -> str:
    canonical = canonical_class_name(value)
    return canonical[:-1] if canonical.endswith("_buoys") else canonical


def normalize_side(value) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in ("left", "sol", "port"):
        return "left"
    if text in ("right", "sag", "sağ", "starboard"):
        return "right"
    if text in (
            "across", "center", "centre", "middle", "orta", "front",
    ):
        return "center"
    return None


def median(values) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def bbox_iou(first, second) -> Optional[float]:
    if not (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == 4
            and len(second) == 4
    ):
        return None
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = (
        max(0.0, min(ax2, bx2) - max(ax1, bx1))
        * max(0.0, min(ay2, by2) - max(ay1, by1))
    )
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - intersection
    )
    return intersection / union if union > 0.0 else None


def normalize_target(
        detection,
        *,
        target_classes: Iterable[str],
        min_confidence: float,
) -> TargetNormalizationResult:
    if not isinstance(detection, dict):
        return TargetNormalizationResult()

    class_name = canonical_class_name(
        first_present(detection, ("class", "class_name", "label"))
    )
    configured_class_keys = {
        class_alias_key(configured_class)
        for configured_class in target_classes
    }
    if class_alias_key(class_name) not in configured_class_keys:
        return TargetNormalizationResult()

    confidence = finite_float(
        first_present(detection, ("confidence", "conf"))
    )
    if confidence is None:
        reason = f"{class_name}: eksik/geçersiz confidence"
        return TargetNormalizationResult(rejection_reason=reason)
    if confidence < min_confidence:
        reason = f"confidence {confidence:.2f} < {min_confidence:.2f}"
        return TargetNormalizationResult(rejection_reason=reason)

    distance = finite_float(
        first_present(detection, ("distance", "distance_m", "depth"))
    )
    side = normalize_side(
        first_present(detection, ("Buoy side: ", "side", "buoy_side"))
    )
    angle = finite_float(
        first_present(
            detection,
            ("Buoy angle: ", "angle_from_center", "angle"),
        )
    )
    if angle is None and side is not None:
        angle = {
            "left": -15.0,
            "right": 15.0,
            "center": 0.0,
        }[side]

    invalid_fields = []
    if distance is None or distance <= 0.0:
        invalid_fields.append("distance")
    if angle is None:
        invalid_fields.append("angle/side")
    if invalid_fields:
        reason = (
            f"{class_name}: eksik/geçersiz {', '.join(invalid_fields)}"
        )
        return TargetNormalizationResult(
            data_uncertain_reason=reason,
            rejection_reason=reason,
        )

    bbox = detection.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            bbox = None
        else:
            try:
                bbox = [int(value) for value in bbox]
            except (TypeError, ValueError):
                bbox = None
            if (
                    bbox is not None
                    and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1])
            ):
                bbox = None

    return TargetNormalizationResult(
        target={
            "class": class_name,
            "confidence": confidence,
            "distance": distance,
            "angle": angle,
            "side": side,
            "bbox": bbox,
            "track_id": detection.get("track_id"),
            "raw": detection,
        }
    )


def select_target(
        detections,
        *,
        target_classes: Iterable[str],
        min_confidence: float,
        last_target=None,
) -> TargetSelectionResult:
    detections = detections or []
    target_classes = tuple(target_classes)
    observed_classes = tuple(
        sorted(
            {
                canonical_class_name(
                    first_present(
                        detection,
                        ("class", "class_name", "label"),
                    )
                )
                for detection in detections
                if isinstance(detection, dict)
            }
            - {""}
        )
    )

    candidates = []
    data_uncertain = False
    data_uncertain_reason = None
    rejection_reason = None
    for detection in detections:
        result = normalize_target(
            detection,
            target_classes=target_classes,
            min_confidence=min_confidence,
        )
        if result.data_uncertain_reason is not None:
            data_uncertain = True
            data_uncertain_reason = result.data_uncertain_reason
        if result.rejection_reason is not None:
            rejection_reason = result.rejection_reason
        if result.target is not None:
            candidates.append(result.target)

    if not candidates:
        if detections and rejection_reason is None:
            rejection_reason = (
                f"configured targets={list(target_classes)} "
                f"not in observed={list(observed_classes)}"
            )
        return TargetSelectionResult(
            target=None,
            observed_classes=observed_classes,
            data_uncertain=data_uncertain,
            data_uncertain_reason=data_uncertain_reason,
            rejection_reason=rejection_reason,
        )

    if last_target is None:
        selected = min(
            candidates,
            key=lambda target: (
                abs(target["angle"]),
                target["distance"],
                -target["confidence"],
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda target: (
                abs(target["angle"] - last_target["angle"]),
                abs(target["distance"] - last_target["distance"]),
                -target["confidence"],
            ),
        )

    return TargetSelectionResult(
        target=selected,
        observed_classes=observed_classes,
    )


def target_is_consistent(
        target,
        previous,
        *,
        bbox_min_iou: float,
        angle_jump_deg: float,
        distance_jump_ratio: float,
) -> bool:
    if target is None or previous is None:
        return target is not None

    target_track_id = target.get("track_id")
    previous_track_id = previous.get("track_id")
    if (
            target_track_id is not None
            and previous_track_id is not None
            and target_track_id == previous_track_id
    ):
        return True

    overlap = bbox_iou(target.get("bbox"), previous.get("bbox"))
    if overlap is not None and overlap >= bbox_min_iou:
        return True

    angle_jump = abs(target["angle"] - previous["angle"])
    distance_base = max(previous["distance"], 0.1)
    observed_distance_jump_ratio = (
        abs(target["distance"] - previous["distance"])
        / distance_base
    )
    return (
        angle_jump <= angle_jump_deg
        and observed_distance_jump_ratio <= distance_jump_ratio
    )


def filter_target(target, previous, alpha: float):
    if previous is None:
        return dict(target)
    filtered = dict(target)
    filtered["angle"] = (
        alpha * target["angle"]
        + (1.0 - alpha) * previous["angle"]
    )
    filtered["distance"] = (
        alpha * target["distance"]
        + (1.0 - alpha) * previous["distance"]
    )
    return filtered
