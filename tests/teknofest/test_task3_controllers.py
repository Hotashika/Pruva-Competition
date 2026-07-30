import math
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
        "search_advance_distance_m": 1.0,
        "search_heading_tolerance_deg": 2.0,
        "search_heading_settle_sec": 0.1,
        "search_heading_kp": 0.018,
        "search_min_angular_z": 0.04,
        "search_turn_timeout_sec": 5.0,
        "search_advance_timeout_sec": 5.0,
        "search_no_progress_timeout_sec": 2.0,
        "search_progress_min_m": 0.1,
        "search_cross_track_limit_m": 1.0,
        "search_cross_track_kp_deg_per_m": 8.0,
        "search_advance_heading_limit_deg": 25.0,
        "ram_speed": 0.75,
        "ram_duration_sec": 0.2,
        "required_impact_count": 3,
        "post_impact_forward_speed": 0.85,
        "post_impact_forward_duration_sec": 1.5,
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


def _offset_gps(lat, lon, bearing_deg, distance_m):
    radius_m = 6378137.0
    bearing_rad = math.radians(bearing_deg)
    north_m = distance_m * math.cos(bearing_rad)
    east_m = distance_m * math.sin(bearing_rad)
    target_lat = lat + math.degrees(north_m / radius_m)
    target_lon = lon + math.degrees(
        east_m / (radius_m * math.cos(math.radians(lat)))
    )
    return target_lat, target_lon


def _settle_heading(controller, heading, lat, lon, now):
    controller.step(heading, lat, lon, now)
    return controller.step(heading, lat, lon, now + 0.11)


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


def test_search_controller_scans_fixed_arc_then_advances_by_gps_distance():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(100.0, lat, lon, now=0.0)
    controller.enter_search(100.0, lat, lon, now=0.0)

    move_left = controller.step(100.0, lat, lon, now=0.1)
    assert controller.phase is SearchPhase.MOVE_TO_ARC_START
    assert move_left.linear_x == 0.0
    assert move_left.angular_z < 0.0
    assert move_left.target_heading == pytest.approx(90.0)

    at_start = _settle_heading(controller, 90.0, lat, lon, now=0.2)
    assert at_start.phase_changed is True
    assert controller.phase is SearchPhase.SWEEP_ARC
    assert at_start.linear_x == 0.0

    sweep = controller.step(90.0, lat, lon, now=0.4)
    assert sweep.linear_x == 0.0
    assert sweep.angular_z > 0.0
    assert sweep.target_heading == pytest.approx(110.0)

    at_end = _settle_heading(controller, 110.0, lat, lon, now=0.5)
    assert at_end.phase_changed is True
    assert controller.phase is SearchPhase.RETURN_TO_BASE_HEADING

    return_to_base = controller.step(110.0, lat, lon, now=0.7)
    assert return_to_base.linear_x == 0.0
    assert return_to_base.angular_z < 0.0

    centered = _settle_heading(controller, 100.0, lat, lon, now=0.8)
    assert centered.phase_changed is True
    assert controller.phase is SearchPhase.ADVANCE

    advance = controller.step(95.0, lat, lon, now=1.0)
    assert advance.linear_x == pytest.approx(0.25)
    assert advance.angular_z > 0.0
    assert advance.target_heading == pytest.approx(100.0)

    advanced_lat, advanced_lon = _offset_gps(lat, lon, 100.0, 1.1)
    next_arc = controller.step(
        100.0,
        advanced_lat,
        advanced_lon,
        now=1.1,
    )
    assert next_arc.phase_changed is True
    assert controller.phase is SearchPhase.MOVE_TO_ARC_START
    assert controller.base_heading == pytest.approx(100.0)
    assert controller.sweep_deg == pytest.approx(30.0)


def test_search_controller_sweeps_across_heading_wraparound():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(359.0, lat, lon, now=0.0)
    controller.enter_search(359.0, lat, lon, now=0.0)

    _settle_heading(controller, 349.0, lat, lon, now=0.1)
    assert controller.phase is SearchPhase.SWEEP_ARC

    first_half = controller.step(1.0, lat, lon, now=0.3)
    assert first_half.angular_z > 0.0
    assert first_half.target_heading == pytest.approx(9.0)

    _settle_heading(controller, 9.0, lat, lon, now=0.4)
    assert controller.phase is SearchPhase.RETURN_TO_BASE_HEADING


def test_search_heading_jitter_cannot_complete_sweep_before_endpoint():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(100.0, lat, lon, now=0.0)
    controller.enter_search(100.0, lat, lon, now=0.0)
    _settle_heading(controller, 90.0, lat, lon, now=0.1)

    for index, heading in enumerate((91.0, 89.5, 92.0, 90.5) * 5):
        decision = controller.step(
            heading,
            lat,
            lon,
            now=0.3 + index * 0.1,
        )
        assert decision.failed is False
        assert controller.phase is SearchPhase.SWEEP_ARC


def test_search_180_degree_sweep_keeps_commanding_starboard():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(
        _controller_config(
            search_initial_sweep_deg=180.0,
            search_max_sweep_deg=180.0,
        )
    )
    controller.reset_for_entry(0.0, lat, lon, now=0.0)
    controller.enter_search(0.0, lat, lon, now=0.0)
    _settle_heading(controller, 270.0, lat, lon, now=0.1)

    decision = controller.step(269.0, lat, lon, now=0.3)

    assert controller.phase is SearchPhase.SWEEP_ARC
    assert decision.target_heading == pytest.approx(90.0)
    assert decision.heading_error_deg == pytest.approx(180.0)
    assert decision.angular_z > 0.0


def test_search_sweep_growth_is_capped_at_180_degrees():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(
        _controller_config(
            search_initial_sweep_deg=175.0,
            search_sweep_increment_deg=10.0,
        )
    )
    controller.reset_for_entry(40.0, lat, lon, now=0.0)
    controller._set_phase(SearchPhase.ADVANCE, 40.0, lat, lon, now=0.0)
    first_lat, first_lon = _offset_gps(lat, lon, 40.0, 1.1)

    controller.step(40.0, first_lat, first_lon, now=0.2)
    assert controller.sweep_deg == pytest.approx(180.0)

    controller._set_phase(
        SearchPhase.ADVANCE,
        40.0,
        first_lat,
        first_lon,
        now=0.3,
    )
    second_lat, second_lon = _offset_gps(lat, lon, 40.0, 2.2)
    controller.step(40.0, second_lat, second_lon, now=0.5)
    assert controller.sweep_deg == pytest.approx(180.0)


def test_search_reentry_recenters_without_reanchoring_search_axis():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(20.0, lat, lon, now=0.0)
    controller.sweep_deg = 60.0

    controller.enter_search(
        75.0,
        lat,
        lon,
        now=1.0,
        recenter=True,
    )
    assert controller.sweep_deg == pytest.approx(60.0)
    assert controller.base_heading == pytest.approx(20.0)
    assert controller.phase is SearchPhase.RETURN_TO_BASE_HEADING

    recentered = _settle_heading(
        controller,
        20.0,
        lat,
        lon,
        now=1.1,
    )
    assert recentered.phase_changed is True
    assert controller.phase is SearchPhase.MOVE_TO_ARC_START
    assert controller.base_heading == pytest.approx(20.0)

    controller.reset_for_entry(30.0, lat, lon, now=2.0)
    assert controller.sweep_deg == pytest.approx(20.0)
    assert controller.base_heading == pytest.approx(30.0)


def test_search_advance_corrects_cross_track_error_toward_fixed_axis():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(0.0, lat, lon, now=0.0)
    controller._set_phase(SearchPhase.ADVANCE, 0.0, lat, lon, now=0.0)
    east_lat, east_lon = _offset_gps(lat, lon, 90.0, 0.5)

    decision = controller.step(0.0, east_lat, east_lon, now=0.2)

    assert decision.failed is False
    assert decision.linear_x == pytest.approx(0.25)
    assert decision.cross_track_m == pytest.approx(0.5, abs=0.02)
    assert decision.target_heading == pytest.approx(356.0, abs=0.2)
    assert decision.angular_z < 0.0


def test_search_advance_fails_when_corridor_is_exceeded():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(0.0, lat, lon, now=0.0)
    controller._set_phase(SearchPhase.ADVANCE, 0.0, lat, lon, now=0.0)
    east_lat, east_lon = _offset_gps(lat, lon, 90.0, 1.2)

    decision = controller.step(0.0, east_lat, east_lon, now=0.2)

    assert decision.failed is True
    assert "corridor exceeded" in decision.reason


def test_search_advance_fails_without_gps_progress():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(0.0, lat, lon, now=0.0)
    controller._set_phase(SearchPhase.ADVANCE, 0.0, lat, lon, now=0.0)

    decision = controller.step(0.0, lat, lon, now=2.0)

    assert decision.failed is True
    assert "no GPS progress" in decision.reason


def test_search_axis_does_not_drift_across_many_cycles():
    lat = 37.95125
    lon = 32.50090
    heading = 123.0
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(heading, lat, lon, now=0.0)
    controller.enter_search(heading, lat, lon, now=0.0)
    now = 0.1
    distance_m = 0.0

    for _ in range(10):
        left_heading = (heading - controller.sweep_deg / 2.0) % 360.0
        right_heading = (heading + controller.sweep_deg / 2.0) % 360.0
        current_lat, current_lon = _offset_gps(
            lat,
            lon,
            heading,
            distance_m,
        )
        _settle_heading(
            controller,
            left_heading,
            current_lat,
            current_lon,
            now,
        )
        now += 0.3
        _settle_heading(
            controller,
            right_heading,
            current_lat,
            current_lon,
            now,
        )
        now += 0.3
        _settle_heading(
            controller,
            heading,
            current_lat,
            current_lon,
            now,
        )
        now += 0.3
        distance_m += 1.1
        current_lat, current_lon = _offset_gps(
            lat,
            lon,
            heading,
            distance_m,
        )
        completed = controller.step(
            heading,
            current_lat,
            current_lon,
            now=now,
        )
        assert completed.phase_changed is True
        assert controller.base_heading == pytest.approx(heading)
        now += 0.2

    assert controller.cycle_index == 10
    assert controller.base_heading == pytest.approx(heading)


def test_search_closed_loop_plant_stays_on_entry_axis():
    lat = 37.95125
    lon = 32.50090
    entry_heading = 42.0
    heading = entry_heading
    controller = Task3SearchController(
        _controller_config(search_advance_distance_m=0.5)
    )
    controller.reset_for_entry(heading, lat, lon, now=0.0)
    controller.enter_search(heading, lat, lon, now=0.0)
    now = 0.0
    last_decision = None

    for _ in range(1000):
        last_decision = controller.step(heading, lat, lon, now)
        assert last_decision.failed is False, last_decision.reason

        heading = (
            heading + math.degrees(last_decision.angular_z) * 0.35
        ) % 360.0
        if last_decision.linear_x > 0.0:
            lat, lon = _offset_gps(
                lat,
                lon,
                heading,
                last_decision.linear_x * 0.1,
            )
        now += 0.1
        if controller.cycle_index >= 4:
            break

    assert controller.cycle_index == 4
    assert controller.base_heading == pytest.approx(entry_heading)
    assert abs(last_decision.cross_track_m) < 0.2


def test_search_turn_watchdog_reports_failure():
    lat = 37.95125
    lon = 32.50090
    controller = Task3SearchController(_controller_config())
    controller.reset_for_entry(100.0, lat, lon, now=0.0)
    controller.enter_search(100.0, lat, lon, now=0.0)

    decision = controller.step(100.0, lat, lon, now=5.0)

    assert decision.failed is True
    assert "watchdog timeout" in decision.reason


def test_impact_controller_owns_ram_and_post_impact_timing():
    controller = Task3ImpactController(_controller_config())

    ram = controller.ram_decision(elapsed=0.1, angular_z=0.12)
    assert ram.action is ImpactAction.RAM_MOTION
    assert ram.linear_x == pytest.approx(0.75)
    assert ram.angular_z == pytest.approx(0.12)

    impact = controller.ram_decision(elapsed=0.21)
    assert impact.action is ImpactAction.IMPACT_RECORDED
    assert controller.impact_count == 0

    assert controller.register_impact() == 1
    assert controller.impact_count == 1

    forward_motion = controller.post_impact_decision(elapsed=0.2)
    assert forward_motion.action is ImpactAction.POST_IMPACT_MOTION
    assert forward_motion.linear_x == pytest.approx(0.85)
    assert forward_motion.angular_z == pytest.approx(0.0)

    impact_return = controller.post_impact_decision(elapsed=1.5)
    assert impact_return.action is ImpactAction.RETURN_TO_IMPACT
