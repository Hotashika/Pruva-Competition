from types import SimpleNamespace

import pytest

from teknofest.missions.utils.task3_impact_controller import (
    ImpactAction,
    Task3ImpactController,
)
from teknofest.missions.utils.task3_search_controller import (
    SearchPhase,
    Task3SearchController,
)
from teknofest.missions.utils.task3_targeting import select_target


def _controller_config(**overrides):
    values = {
        "search_linear_x": 0.25,
        "search_angular_z": 0.18,
        "search_initial_sweep_deg": 20.0,
        "search_sweep_increment_deg": 10.0,
        "search_max_sweep_deg": 180.0,
        "search_forward_duration_sec": 2.5,
        "search_heading_tolerance_deg": 2.0,
        "search_turn_timeout_min_sec": 6.0,
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


def test_search_controller_scans_centered_arc_then_advances_on_saved_heading():
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(current_heading=100.0, now=0.0)
    controller.enter_search(current_heading=100.0, now=0.0)

    move_left = controller.step(current_heading=100.0, now=0.1)
    assert controller.phase is SearchPhase.MOVE_TO_ARC_START
    assert move_left.linear_x == 0.0
    assert move_left.angular_z < 0.0

    at_start = controller.step(current_heading=90.0, now=0.2)
    assert at_start.phase_changed is True
    assert controller.phase is SearchPhase.SWEEP_ARC
    assert at_start.linear_x == 0.0

    sweep = controller.step(current_heading=90.0, now=0.3)
    assert sweep.linear_x == 0.0
    assert sweep.angular_z > 0.0

    at_end = controller.step(current_heading=110.0, now=0.4)
    assert at_end.phase_changed is True
    assert controller.phase is SearchPhase.RETURN_TO_BASE_HEADING

    return_to_base = controller.step(current_heading=110.0, now=0.5)
    assert return_to_base.linear_x == 0.0
    assert return_to_base.angular_z < 0.0

    centered = controller.step(current_heading=100.0, now=0.6)
    assert centered.phase_changed is True
    assert controller.phase is SearchPhase.ADVANCE

    advance = controller.step(current_heading=95.0, now=0.7)
    assert advance.linear_x == pytest.approx(0.25)
    assert advance.angular_z > 0.0

    next_arc = controller.step(current_heading=100.0, now=3.11)
    assert next_arc.phase_changed is True
    assert controller.phase is SearchPhase.MOVE_TO_ARC_START
    assert controller.base_heading == pytest.approx(100.0)
    assert controller.sweep_deg == pytest.approx(30.0)


def test_search_controller_sweeps_across_heading_wraparound():
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(current_heading=359.0, now=0.0)
    controller.enter_search(current_heading=359.0, now=0.0)

    controller.step(current_heading=349.0, now=0.1)
    assert controller.phase is SearchPhase.SWEEP_ARC

    first_half = controller.step(current_heading=1.0, now=0.2)
    assert first_half.angular_z > 0.0
    assert controller.sweep_progress_deg == pytest.approx(12.0)

    controller.step(current_heading=9.0, now=0.3)
    assert controller.phase is SearchPhase.RETURN_TO_BASE_HEADING


def test_search_sweep_growth_is_capped_at_180_degrees():
    controller = Task3SearchController(
        _controller_config(
            search_initial_sweep_deg=175.0,
            search_sweep_increment_deg=10.0,
        )
    )
    controller.reset_for_entry(current_heading=40.0, now=0.0)
    controller.phase = SearchPhase.ADVANCE
    controller.base_heading = 40.0
    controller.phase_started_at = 0.0

    controller.step(current_heading=40.0, now=2.5)
    assert controller.sweep_deg == pytest.approx(180.0)

    controller.phase = SearchPhase.ADVANCE
    controller.phase_started_at = 2.5
    controller.step(current_heading=40.0, now=5.0)
    assert controller.sweep_deg == pytest.approx(180.0)


def test_search_reentry_preserves_sweep_and_fresh_entry_resets_it():
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(current_heading=20.0, now=0.0)
    controller.sweep_deg = 60.0

    controller.enter_search(current_heading=25.0, now=1.0)
    assert controller.sweep_deg == pytest.approx(60.0)
    assert controller.base_heading == pytest.approx(25.0)

    controller.reset_for_entry(current_heading=30.0, now=2.0)
    assert controller.sweep_deg == pytest.approx(20.0)
    assert controller.base_heading == pytest.approx(30.0)


def test_search_turn_watchdog_reports_failure():
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(current_heading=100.0, now=0.0)
    controller.enter_search(current_heading=100.0, now=0.0)

    decision = controller.step(current_heading=100.0, now=100.0)

    assert decision.failed is True
    assert "watchdog timeout" in decision.reason


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
