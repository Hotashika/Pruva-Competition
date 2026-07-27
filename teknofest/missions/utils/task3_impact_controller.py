"""ROS-independent impact and retreat decisions for TEKNOFEST Task 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ImpactAction(Enum):
    RAM_MOTION = auto()
    CONTACT_HOLD = auto()
    HOLD = auto()
    FINISH = auto()
    RETREAT = auto()
    RETREAT_MOTION = auto()
    REACQUIRE = auto()
    FAILSAFE = auto()


@dataclass(frozen=True)
class ImpactDecision:
    action: ImpactAction
    linear_x: Optional[float] = None
    angular_z: Optional[float] = None
    reason: Optional[str] = None


def _angle_error_deg(target_deg, current_deg) -> float:
    return (
        float(target_deg) - float(current_deg) + 180.0
    ) % 360.0 - 180.0


class Task3ImpactController:
    def __init__(self, config):
        self.config = config
        self.impact_count = 0
        self.retreat_heading = None

    def reset(self):
        self.impact_count = 0
        self.retreat_heading = None

    def ram_decision(self, elapsed, current_heading) -> ImpactDecision:
        if elapsed < self.config.ram_duration_sec:
            return ImpactDecision(
                action=ImpactAction.RAM_MOTION,
                linear_x=self.config.ram_speed,
                angular_z=0.0,
                reason="confirmed ram",
            )

        self.impact_count += 1
        self.retreat_heading = current_heading
        return ImpactDecision(
            action=ImpactAction.CONTACT_HOLD,
            reason=f"{self.impact_count}. temas komutu tamamlandı",
        )

    def contact_hold_decision(self, elapsed) -> ImpactDecision:
        if elapsed < self.config.contact_hold_sec:
            return ImpactDecision(action=ImpactAction.HOLD)
        if self.impact_count >= self.config.required_impact_count:
            return ImpactDecision(
                action=ImpactAction.FINISH,
                reason="gerekli temas sayısı tamamlandı",
            )
        return ImpactDecision(
            action=ImpactAction.RETREAT,
            reason="yeniden yaklaşmak için geri çekilme",
        )

    def retreat_timeout_reached(self, elapsed) -> bool:
        return elapsed >= self.config.retreat_max_sec

    def retreat_decision(
            self,
            *,
            elapsed,
            target_far_enough,
            current_heading,
    ) -> ImpactDecision:
        if self.impact_count <= 0:
            return ImpactDecision(
                action=ImpactAction.FAILSAFE,
                reason=(
                    "Task 3 RETREAT doğrulanmış temas olmadan başlatıldı."
                ),
            )

        if (
                elapsed >= self.config.retreat_max_sec
                or (
                    elapsed >= self.config.retreat_min_sec
                    and target_far_enough
                )
        ):
            return ImpactDecision(
                action=ImpactAction.REACQUIRE,
                reason=(
                    "geri çekilme tamamlandı; "
                    "hedef yeniden teyit edilecek"
                ),
            )

        if self.retreat_heading is None:
            self.retreat_heading = current_heading
        heading_error_deg = _angle_error_deg(
            self.retreat_heading,
            current_heading,
        )
        heading_correction = max(
            -self.config.retreat_heading_max_angular_z,
            min(
                self.config.retreat_heading_max_angular_z,
                math.radians(heading_error_deg),
            ),
        )
        return ImpactDecision(
            action=ImpactAction.RETREAT_MOTION,
            linear_x=-self.config.retreat_speed,
            angular_z=heading_correction,
            reason="post-contact straight retreat",
        )
