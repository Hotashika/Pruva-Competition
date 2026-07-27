from types import SimpleNamespace

import pytest

from teknofest.missions.utils.task3_impact_controller import (
    ImpactAction,
    Task3ImpactController,
)
from teknofest.missions.utils.task3_search_controller import (
    Task3SearchController,
)
from teknofest.missions.utils.task3_targeting import select_target


def _controller_config(**overrides):
    values = {
        "search_linear_x": 0.25,
        "search_angular_z": 0.18,
        "search_leg_sweep_deg": 70.0,
        "search_leg_timeout_sec": 0.1,
        "search_legs_per_cycle": 4,
        "search_radius_step_m": 2.0,
        "search_max_radius_m": 6.0,
        "search_points_per_ring": 4,
        "ram_speed": 0.75,
        "ram_duration_sec": 0.2,
        "contact_hold_sec": 0.1,
        "required_impact_count": 3,
        "retreat_speed": 0.25,
        "retreat_min_sec": 0.6,
        "retreat_max_sec": 1.5,
        "retreat_heading_max_angular_z": 0.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _target(class_name, distance, angle):
    return {
        "class": class_name,
        "confidence": 0.9,
        "distance": distance,
        "Buoy angle: ": angle,
    }


def test_targeting_selects_supported_candidate_closest_to_last_target():
    last_target = {
        "class": "red_buoy",
        "confidence": 0.9,
        "distance": 2.5,
        "angle": 8.0,
    }

    result = select_target(
        [
            _target("yellow_buoy", distance=2.0, angle=0.0),
            _target("red_buoys", distance=3.0, angle=-10.0),
            _target("orange_buoy", distance=2.4, angle=7.0),
        ],
        target_classes=(
            "red_buoy",
            "red_buoys",
            "orange_buoy",
            "orange_buoys",
        ),
        min_confidence=0.45,
        last_target=last_target,
    )

    assert result.target["class"] == "orange_buoy"
    assert result.observed_classes == (
        "orange_buoy",
        "red_buoys",
        "yellow_buoy",
    )
    assert result.data_uncertain is False


def test_search_controller_alternates_legs_then_requests_relocation():
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(current_heading=10.0, now=0.0)
    controller.enter_search(current_heading=10.0, now=0.0)

    decisions = [
        controller.step(current_heading=10.0, now=timestamp)
        for timestamp in (0.01, 0.11, 0.22, 0.33, 0.44)
    ]

    motion_decisions = [
        decision for decision in decisions if not decision.relocate
    ]
    assert [decision.linear_x for decision in motion_decisions] == pytest.approx(
        [0.25, 0.25, 0.25, 0.25]
    )
    assert [
        1 if decision.angular_z > 0.0 else -1
        for decision in motion_decisions
    ] == [1, -1, 1, -1]
    assert decisions[-1].relocate is True


def test_impact_controller_owns_ram_hold_and_retreat_decisions():
    controller = Task3ImpactController(_controller_config())

    ram = controller.ram_decision(elapsed=0.1, current_heading=42.0)
    assert ram.action is ImpactAction.RAM_MOTION
    assert ram.linear_x == pytest.approx(0.75)

    contact = controller.ram_decision(elapsed=0.21, current_heading=42.0)
    assert contact.action is ImpactAction.CONTACT_HOLD
    assert controller.impact_count == 1
    assert controller.retreat_heading == pytest.approx(42.0)

    hold = controller.contact_hold_decision(elapsed=0.05)
    assert hold.action is ImpactAction.HOLD
    retreat = controller.contact_hold_decision(elapsed=0.11)
    assert retreat.action is ImpactAction.RETREAT

    retreat_motion = controller.retreat_decision(
        elapsed=0.2,
        target_far_enough=False,
        current_heading=47.0,
    )
    assert retreat_motion.action is ImpactAction.RETREAT_MOTION
    assert retreat_motion.linear_x == pytest.approx(-0.25)
    assert retreat_motion.angular_z < 0.0

    reacquire = controller.retreat_decision(
        elapsed=1.5,
        target_far_enough=False,
        current_heading=47.0,
    )
    assert reacquire.action is ImpactAction.REACQUIRE
