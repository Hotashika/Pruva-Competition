import math

import pytest

from teknofest.missions.utils.yellow_buoy_course_keeper import (
    ORANGE_BUOY_CLASS_NAMES,
    YellowBuoyCourseConfig,
    YellowBuoyCourseKeeper,
    is_course_buoy_detection,
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


@pytest.mark.parametrize(
    "class_name",
    [
        "yellow_buoy",
        "yellow_buoys",
        "Sarı Duba",
        "green_buoy",
        "green_buoys",
        "Yeşil Duba",
        "red_buoy",
        "red_buoys",
        "Kırmızı Duba",
        "orange_buoy",
        "orange_buoys",
        "Turuncu Duba",
    ],
)
def test_supported_course_buoy_class_aliases_are_recognized(class_name):
    assert is_course_buoy_detection({"class": class_name})


def test_unrelated_class_is_not_recognized_as_course_buoy():
    assert not is_course_buoy_detection({"class": "blue_boat"})


def test_yellow_specific_helper_rejects_other_course_colors():
    assert is_yellow_buoy_detection({"class": "yellow_buoy"})
    assert not is_yellow_buoy_detection({"class": "green_buoy"})


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
    assert decision.reason == "second_nearest_course_buoy"
    assert decision.candidate_count == 3
    assert decision.selected_distance_m == pytest.approx(7.0)
    assert decision.relative_bearing_deg == pytest.approx(24.0)
    assert decision.target_lat > CURRENT_LAT
    assert decision.target_lon > CURRENT_LON


def test_different_buoy_colors_share_the_same_course_selection():
    decision = keeper_without_smoothing().compute(
        [
            yellow(3.0, -10.0, class_name="green_buoy"),
            yellow(7.0, 24.0, class_name="red_buoy"),
            yellow(11.0, -30.0, class_name="orange_buoy"),
        ],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.0,
    )

    assert decision.status == "live"
    assert decision.candidate_count == 3
    assert decision.selected_distance_m == pytest.approx(7.0)
    assert decision.relative_bearing_deg == pytest.approx(24.0)


def test_configured_orange_only_keeper_ignores_other_buoy_colors():
    keeper = YellowBuoyCourseKeeper(YellowBuoyCourseConfig(
        steering_smoothing_alpha=1.0,
        course_buoy_class_names=ORANGE_BUOY_CLASS_NAMES,
    ))

    decision = keeper.compute(
        [
            yellow(2.0, -15.0, class_name="green_buoy"),
            yellow(4.0, 10.0, class_name="orange_buoy"),
            yellow(6.0, -25.0, class_name="red_buoy"),
            yellow(8.0, 30.0, class_name="Turuncu Duba"),
        ],
        CURRENT_LAT,
        CURRENT_LON,
        0.0,
        now=10.0,
    )

    assert decision.status == "live"
    assert decision.candidate_count == 2
    assert decision.selected_distance_m == pytest.approx(8.0)
    assert decision.relative_bearing_deg == pytest.approx(30.0)


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
            yellow(3.0, 0.0, class_name="blue_boat"),
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
    assert memory.reason == "fewer_than_two_course_buoys"
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
