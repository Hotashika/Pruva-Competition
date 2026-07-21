import math

import pytest

from teknofest.missions.utils.yellow_buoy_course_keeper import (
    YellowBuoyCourseConfig,
    YellowBuoyCourseKeeper,
    is_yellow_buoy_detection,
)


CURRENT_LAT = 37.95125
CURRENT_LON = 32.50090


def yellow(distance_m, angle_deg, confidence=0.9, class_name="yellow_buoy"):
    return {
        "class": class_name,
        "confidence": confidence,
        "distance": distance_m,
        "Buoy angle: ": angle_deg,
    }


def keeper_without_smoothing():
    return YellowBuoyCourseKeeper(
        YellowBuoyCourseConfig(steering_smoothing_alpha=1.0)
    )


def test_yellow_class_aliases_are_recognized():
    assert is_yellow_buoy_detection({"class": "yellow_buoy"})
    assert is_yellow_buoy_detection({"label": "Sarı Duba"})
    assert not is_yellow_buoy_detection({"class": "orange_buoy"})


def test_second_nearest_buoy_is_selected_after_distance_sorting():
    decision = keeper_without_smoothing().compute(
        [
            yellow(11.0, -30.0),
            yellow(3.0, -10.0),
            yellow(7.0, 24.0),
        ],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.0,
    )

    assert decision.status == "live"
    assert decision.reason == "second_nearest_yellow_buoy"
    assert decision.candidate_count == 3
    assert decision.selected_distance_m == pytest.approx(7.0)
    assert decision.relative_bearing_deg == pytest.approx(24.0)
    assert decision.target_lat > CURRENT_LAT
    assert decision.target_lon > CURRENT_LON


def test_target_is_reselected_on_every_iteration():
    keeper = keeper_without_smoothing()
    first = keeper.compute(
        [yellow(2.0, -20.0), yellow(5.0, 30.0), yellow(9.0, 5.0)],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.0,
    )
    second = keeper.compute(
        [yellow(8.0, -20.0), yellow(3.0, 30.0), yellow(6.0, -35.0)],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.1,
    )

    assert first.selected_distance_m == pytest.approx(5.0)
    assert first.relative_bearing_deg == pytest.approx(30.0)
    assert second.selected_distance_m == pytest.approx(6.0)
    assert second.relative_bearing_deg == pytest.approx(-35.0)
    assert (first.target_lat, first.target_lon) != (
        second.target_lat,
        second.target_lon,
    )


def test_invalid_candidates_do_not_affect_second_nearest_selection():
    decision = keeper_without_smoothing().compute(
        [
            yellow(1.0, 5.0, confidence=0.1),
            yellow(float("nan"), 7.0),
            yellow(2.0, None),
            yellow(4.0, -12.0),
            yellow(8.0, 18.0),
            yellow(3.0, 0.0, class_name="orange_buoy"),
        ],
        CURRENT_LAT,
        CURRENT_LON,
        15.0,
        now=10.0,
    )

    assert decision.status == "live"
    assert decision.candidate_count == 2
    assert decision.selected_distance_m == pytest.approx(8.0)
    assert decision.relative_bearing_deg == pytest.approx(18.0)
    assert decision.global_bearing_deg == pytest.approx(33.0)


def test_short_detection_dropout_uses_memory_then_stops():
    keeper = keeper_without_smoothing()
    keeper.compute(
        [yellow(3.0, -5.0), yellow(6.0, 15.0)],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.0,
    )

    memory = keeper.compute(
        [yellow(3.0, -5.0)],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.5,
    )
    blocked = keeper.compute(
        [],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=11.1,
    )

    assert memory.status == "memory"
    assert memory.reason == "fewer_than_two_yellow_buoys"
    assert memory.has_target
    assert blocked.status == "blocked"
    assert blocked.should_stop


def test_fewer_than_two_initial_buoys_blocks_without_target():
    decision = keeper_without_smoothing().compute(
        [yellow(4.0, 10.0)],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.0,
    )

    assert decision.status == "blocked"
    assert not decision.has_target
    assert decision.candidate_count == 1


def test_all_live_navigation_outputs_are_finite():
    decision = keeper_without_smoothing().compute(
        [yellow(3.0, -10.0), yellow(9.0, 20.0)],
        CURRENT_LAT,
        CURRENT_LON,
        350.0,
        now=10.0,
    )

    assert decision.has_target
    assert math.isfinite(decision.target_lat)
    assert math.isfinite(decision.target_lon)
    assert decision.global_bearing_deg == pytest.approx(10.0)
