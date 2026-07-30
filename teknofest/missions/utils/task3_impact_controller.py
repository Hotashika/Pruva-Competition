"""ROS-independent impact timing for TEKNOFEST Task 3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ImpactAction(Enum):
    RAM_MOTION = auto()
    IMPACT_RECORDED = auto()
    POST_IMPACT_MOTION = auto()
    RETURN_TO_IMPACT = auto()


@dataclass(frozen=True)
class ImpactDecision:
    action: ImpactAction
    linear_x: Optional[float] = None
    angular_z: Optional[float] = None
    reason: Optional[str] = None


class Task3ImpactController:
    def __init__(self, config):
        self.config = config
        self.impact_count = 0

    def reset(self):
        self.impact_count = 0

    def register_impact(self) -> int:
        self.impact_count += 1
        return self.impact_count

    def ram_decision(self, elapsed, angular_z=0.0) -> ImpactDecision:
        if elapsed < self.config.ram_duration_sec:
            return ImpactDecision(
                action=ImpactAction.RAM_MOTION,
                linear_x=self.config.ram_speed,
                angular_z=float(angular_z),
                reason="direct ram",
            )
        return ImpactDecision(
            action=ImpactAction.IMPACT_RECORDED,
            reason="RAM süresi tamamlandı; çarpışma GPS'i kaydediliyor",
        )

    def post_impact_decision(self, elapsed) -> ImpactDecision:
        if elapsed < self.config.post_impact_forward_duration_sec:
            return ImpactDecision(
                action=ImpactAction.POST_IMPACT_MOTION,
                linear_x=self.config.post_impact_forward_speed,
                angular_z=0.0,
                reason="post-impact forward advance",
            )
        return ImpactDecision(
            action=ImpactAction.RETURN_TO_IMPACT,
            reason="ileri çıkış tamamlandı; kayıtlı GPS'e dönülüyor",
        )
