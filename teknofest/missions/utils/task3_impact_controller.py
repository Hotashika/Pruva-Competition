"""ROS-independent repeated-impact decisions for TEKNOFEST Task 3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ImpactAction(Enum):
    RAM_MOTION = auto()
    CONTACT_HOLD = auto()
    HOLD = auto()
    FINISH = auto()
    FORWARD_CLEAR = auto()
    FORWARD_CLEAR_MOTION = auto()
    IMPACT_RETURN = auto()


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

    def ram_decision(self, elapsed) -> ImpactDecision:
        if elapsed < self.config.ram_duration_sec:
            return ImpactDecision(
                action=ImpactAction.RAM_MOTION,
                linear_x=self.config.ram_speed,
                angular_z=0.0,
                reason="confirmed ram",
            )

        self.impact_count += 1
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
            action=ImpactAction.FORWARD_CLEAR,
            reason="yeniden yaklaşmak için ileri ayrılma",
        )

    def forward_clear_decision(self, elapsed) -> ImpactDecision:
        if elapsed >= self.config.post_impact_forward_duration_sec:
            return ImpactDecision(
                action=ImpactAction.IMPACT_RETURN,
                reason="ileri ayrılma tamamlandı; kayıtlı hedefe dönülüyor",
            )
        return ImpactDecision(
            action=ImpactAction.FORWARD_CLEAR_MOTION,
            linear_x=self.config.post_impact_forward_speed,
            angular_z=0.0,
            reason="post-contact forward clear",
        )
