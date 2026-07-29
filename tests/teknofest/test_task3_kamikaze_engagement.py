import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK3_PATH = (
    REPO_ROOT
    / "teknofest"
    / "missions"
    / "task3_kamikaze_engagement.py"
)


@pytest.fixture()
def task3_module(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = type("Node", (), {})

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})
    std_msgs.msg = std_msgs_msg

    mavlink_utilities = types.ModuleType("utils.mavlink_utilities")
    utility_names = (
        "call_set_mode",
        "call_trigger_service",
        "create_mission_clients",
        "create_mission_topics",
        "parse_bridge_state",
        "publish_cmd_vel",
        "stop_vehicle",
        "wait_for_mission_services",
    )
    for name in utility_names:
        setattr(mavlink_utilities, name, lambda *args, **kwargs: None)
    mavlink_utilities.parse_bridge_state = lambda value: value
    mavlink_utilities.calculate_gps_distance = _gps_distance_m

    for name, module in {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "utils.mavlink_utilities": mavlink_utilities,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "task3_kamikaze_test_module"
    spec = importlib.util.spec_from_file_location(module_name, TASK3_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _gps_distance_m(lat1, lon1, lat2, lon2):
    radius_m = 6378137.0
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    north_m = math.radians(lat2 - lat1) * radius_m
    east_m = math.radians(lon2 - lon1) * radius_m * math.cos(mean_lat)
    return math.hypot(north_m, east_m)


def test_target_classes_annotation_is_python38_compatible():
    source = TASK3_PATH.read_text(encoding="utf-8")

    assert "target_classes: Tuple[str, ...]" in source
    assert "target_classes: tuple[" not in source


def test_task3_node_initializes_mission_data_recorder():
    source = TASK3_PATH.read_text(encoding="utf-8")

    assert (
        "from teknofest.missions.utils.mission_data_recorder "
        "import MissionDataRecorder"
    ) in source
    assert (
        "self.data_recorder = MissionDataRecorder(self, ACTIVE_TASK_NAME)"
        in source
    )


def _logger():
    sink = lambda *args, **kwargs: None
    return types.SimpleNamespace(
        info=sink,
        warn=sink,
        error=sink,
    )


def _mission(task3_module, **config_overrides):
    config_values = {
        "entry_settle_sec": 0.0,
        "confirmation_window_size": 3,
        "confirmation_required": 2,
        "final_confirmation_required": 2,
        "ram_duration_sec": 0.2,
        "contact_hold_sec": 0.1,
        "post_impact_forward_duration_sec": 0.2,
        "impact_return_timeout_sec": 1.0,
        "target_lost_timeout_sec": 0.2,
        "gps_timeout_sec": 100.0,
        "heading_timeout_sec": 100.0,
        "bridge_state_timeout_sec": 100.0,
        "mission_timeout_sec": 100.0,
    }
    config_values.update(config_overrides)
    node = types.SimpleNamespace(get_logger=_logger)
    topics = types.SimpleNamespace(
        cmd_vel_pub="cmd_vel",
        position_target_pub="position_target",
    )
    mission = task3_module.Task3KamikazeEngagement(
        node,
        topics,
        mission_clients=types.SimpleNamespace(),
        config=task3_module.Task3Config(**config_values),
    )
    mission.reset_for_entry(37.95125, 32.50090, 15.0, now=0.0)
    mission.update([], now=0.0)
    assert mission.state is task3_module.MissionState.SEARCH
    return mission


def _target(
        distance=3.0,
        angle=0.0,
        class_name="red_buoy",
        confidence=0.9,
        **overrides,
):
    detection = {
        "class": class_name,
        "confidence": confidence,
        "distance": distance,
        "bbox": [500, 200, 650, 500],
        "track_id": 7,
        "Buoy angle: ": angle,
        "Buoy side: ": "across",
    }
    detection.update(overrides)
    return detection


def _acquire_and_reach_final_confirm(mission, task3_module, now):
    far_target = [_target(distance=2.5)]
    mission.update(far_target, now=now)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM
    now += 0.05
    mission.update(far_target, now=now)
    assert mission.state is task3_module.MissionState.ALIGN
    now += 0.05
    mission.update(far_target, now=now)
    assert mission.state is task3_module.MissionState.APPROACH

    for _ in range(mission.config.approach_distance_required):
        now += 0.05
        mission.update([_target(distance=1.2)], now=now)
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    now += 0.05
    mission.update([_target(distance=1.2)], now=now)
    now += 0.05
    mission.update([_target(distance=1.2)], now=now)
    assert mission.state is task3_module.MissionState.RAM
    return now


def _complete_one_impact(mission, task3_module, now):
    assert mission.state is task3_module.MissionState.RAM
    now += mission.config.ram_duration_sec + 0.01
    mission.update([], now=now)
    assert mission.state is task3_module.MissionState.CONTACT_HOLD
    now += mission.config.contact_hold_sec + 0.01
    mission.update([], now=now)
    return now


def test_search_pivots_for_wrong_class_and_stops_for_invalid_target_data(
        task3_module,
):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    task3_module.stop_vehicle = (
        lambda publisher: commands.append((0.0, 0.0))
    )
    mission = _mission(task3_module)
    commands.clear()

    mission.update([_target(class_name="yellow_buoy")], now=0.1)
    assert commands[-1][0] == pytest.approx(0.0)
    assert commands[-1][1] < 0.0

    mission.update([_target(distance=-1.0)], now=0.2)
    assert commands[-1] == (0.0, 0.0)

    mission.update([_target(distance=float("inf"))], now=0.3)
    missing_direction = _target(distance=2.0, angle=None)
    missing_direction.pop("Buoy side: ")
    mission.update([missing_direction], now=0.4)
    assert commands[-1] == (0.0, 0.0)

    mission.update([_target(confidence=None)], now=0.5)

    assert mission.state is task3_module.MissionState.SEARCH
    assert commands[-1][0] == pytest.approx(0.0)
    assert commands[-1][1] < 0.0


@pytest.mark.parametrize(
    ("class_field", "class_name"),
    [
        ("class", "red_buoy"),
        ("class_name", "red_buoys"),
        ("label", "orange_buoy"),
        ("class", "orange_buoys"),
    ],
)
def test_target_classes_accept_red_and_orange_labels(
        task3_module,
        class_field,
        class_name,
):
    mission = _mission(task3_module)
    detection = {
        class_field: class_name,
        "conf": 0.91,
        "distance_m": 2.2,
        "angle_from_center": -4.0,
        "bbox": [500, 200, 650, 500],
    }

    target = mission._select_target([detection])

    assert task3_module.TASK3_TARGET_BUOY_CLASSES == (
        "red_buoy",
        "red_buoys",
        "orange_buoy",
        "orange_buoys",
    )
    assert target["class"] == class_name
    assert target["distance"] == pytest.approx(2.2)
    assert target["angle"] == pytest.approx(-4.0)

    mission.update([detection], now=0.1)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM


@pytest.mark.parametrize("class_name", ["green_buoy", "yellow_buoys"])
def test_target_classes_reject_other_buoy_labels(task3_module, class_name):
    mission = _mission(task3_module)

    target = mission._select_target([_target(class_name=class_name)])

    assert target is None


@pytest.mark.parametrize(
    ("side", "expected_angle"),
    [
        ("left", -15.0),
        ("right", 15.0),
        ("across", 0.0),
    ],
)
def test_side_fallback_supplies_missing_angle(
        task3_module,
        side,
        expected_angle,
):
    mission = _mission(task3_module)
    detection = _target(angle=None)
    detection["Buoy side: "] = side

    target = mission._select_target([detection])

    assert target["angle"] == pytest.approx(expected_angle)


def test_target_requires_multiple_consistent_frames(task3_module):
    mission = _mission(task3_module)
    target = [_target(distance=3.0, angle=2.0)]

    mission.update(target, now=0.1)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM

    mission.update([], now=0.2)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM

    mission.update(target, now=0.3)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM

    mission.update(target, now=0.4)
    assert mission.state is task3_module.MissionState.ALIGN


def test_confirmation_gap_over_half_second_resets_counter(task3_module):
    mission = _mission(task3_module)
    target = [_target(distance=3.0, angle=2.0)]

    mission.update(target, now=0.1)
    mission.update(target, now=0.7)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM

    mission.update(target, now=0.8)
    assert mission.state is task3_module.MissionState.ALIGN


def test_confirmation_uses_track_continuity_and_ema(task3_module):
    mission = _mission(task3_module, target_filter_alpha=0.4)
    first = [_target(distance=3.0, angle=-10.0, track_id=12)]
    second = [_target(distance=2.0, angle=-6.0, track_id=12)]

    mission.update(first, now=0.1)
    mission.update(second, now=0.2)

    assert mission.state is task3_module.MissionState.ALIGN
    assert mission.last_target["distance"] == pytest.approx(2.6)
    assert mission.last_target["angle"] == pytest.approx(-8.4)


def test_confirmation_falls_back_to_bbox_when_track_id_changes(task3_module):
    mission = _mission(task3_module)
    first = [
        _target(
            distance=3.0,
            angle=-5.0,
            track_id=12,
            bbox=[500, 200, 650, 500],
        )
    ]
    second = [
        _target(
            distance=2.9,
            angle=-4.0,
            track_id=99,
            bbox=[510, 205, 660, 505],
        )
    ]

    mission.update(first, now=0.1)
    mission.update(second, now=0.2)

    assert mission.state is task3_module.MissionState.ALIGN


def test_confirmation_falls_back_to_angle_and_distance_continuity(task3_module):
    mission = _mission(task3_module)
    first = [_target(distance=3.0, angle=-5.0, track_id=None, bbox=None)]
    second = [_target(distance=2.9, angle=-4.0, track_id=None, bbox=None)]

    mission.update(first, now=0.1)
    mission.update(second, now=0.2)

    assert mission.state is task3_module.MissionState.ALIGN


def test_front_target_stops_immediately_then_approaches_forward(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    task3_module.stop_vehicle = (
        lambda publisher: commands.append((0.0, 0.0))
    )
    mission = _mission(task3_module)
    commands.clear()
    target = [_target(distance=3.0, angle=0.0, class_name="red_buoys")]

    mission.update(target, now=0.1)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM
    assert commands[-1] == (0.0, 0.0)

    mission.update(target, now=0.2)
    assert mission.state is task3_module.MissionState.ALIGN

    mission.update(target, now=0.3)
    assert mission.state is task3_module.MissionState.APPROACH

    mission.update(target, now=0.4)
    assert commands[-1][0] > 0.0
    assert commands[-1][1] == pytest.approx(0.0)


def test_single_close_sample_does_not_end_approach(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    target = [_target(distance=2.5)]

    mission.update(target, now=0.1)
    mission.update(target, now=0.2)
    mission.update(target, now=0.3)
    assert mission.state is task3_module.MissionState.APPROACH

    mission.update([_target(distance=1.2)], now=0.4)

    assert mission.state is task3_module.MissionState.APPROACH
    assert list(mission.distance_history) == [pytest.approx(1.2)]
    assert commands[-1][0] > 0.0


def test_approach_uses_seven_sample_median_before_final_confirm(task3_module):
    mission = _mission(task3_module)
    target = [_target(distance=2.5)]

    mission.update(target, now=0.1)
    mission.update(target, now=0.2)
    mission.update(target, now=0.3)
    assert mission.state is task3_module.MissionState.APPROACH

    approach_distances = (2.5, 2.3, 1.0, 2.1, 1.9, 1.3, 1.2)
    for index, distance in enumerate(approach_distances, start=4):
        mission.update(
            [_target(distance=distance)],
            now=index / 10.0,
        )

    assert mission.state is task3_module.MissionState.APPROACH
    assert mission._median(mission.distance_history) == pytest.approx(1.9)

    mission.update([_target(distance=1.1)], now=1.1)
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM
    assert mission._median(mission.distance_history) == pytest.approx(1.3)


def test_approach_keeps_forward_motion_for_moderate_angle_error(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    target = [_target(distance=2.5)]

    mission.update(target, now=0.1)
    mission.update(target, now=0.2)
    mission.update(target, now=0.3)
    assert mission.state is task3_module.MissionState.APPROACH

    mission.update([_target(distance=3.0, angle=20.0)], now=0.4)

    assert mission.state is task3_module.MissionState.APPROACH
    assert commands[-1][0] == pytest.approx(
        mission.config.medium_approach_speed
    )
    assert commands[-1][1] > 0.0


def test_final_confirmation_keeps_advancing_and_starts_ram_immediately(
        task3_module,
):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    now = 0.1

    mission.update([_target(distance=2.5)], now=now)
    mission.update([_target(distance=2.5)], now=now + 0.1)
    mission.update([_target(distance=2.5)], now=now + 0.2)
    for offset in range(mission.config.approach_distance_required):
        mission.update(
            [_target(distance=1.2)],
            now=now + 0.3 + offset * 0.1,
        )
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    mission.update([_target(distance=1.2)], now=now + 0.9)
    assert commands[-1][0] == pytest.approx(
        mission.config.final_confirm_forward_speed
    )

    mission.update([_target(distance=1.2)], now=now + 1.0)
    assert mission.state is task3_module.MissionState.RAM
    assert commands[-1] == pytest.approx((mission.config.ram_speed, 0.0))


def test_lost_target_stops_and_enters_reacquire(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    stopped = []
    task3_module.stop_vehicle = lambda publisher: stopped.append(publisher)
    mission = _mission(task3_module)
    now = 0.1
    mission.update([_target()], now=now)
    mission.update([_target()], now=now + 0.1)
    mission.update([_target()], now=now + 0.2)
    assert mission.state is task3_module.MissionState.APPROACH

    mission.update([], now=now + 0.3)

    assert mission.state is task3_module.MissionState.REACQUIRE
    assert stopped[-1] == "cmd_vel"

    mission.update([], now=now + 0.35)
    assert commands[-1][0] == pytest.approx(
        mission.config.reacquire_linear_x
    )
    assert commands[-1][0] > 0.0

    mission.update(
        [],
        now=now + 0.3 + mission.config.target_lost_timeout_sec,
    )
    assert mission.state is task3_module.MissionState.SEARCH


def test_centered_arc_returns_to_saved_heading_then_advances(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    commands.clear()

    mission.update([], now=0.01)
    assert commands[-1][0] == pytest.approx(0.0)
    assert commands[-1][1] < 0.0

    mission.update_heading(5.0, now=0.02)
    mission.update([], now=0.02)
    assert mission.search_controller.phase is task3_module.SearchPhase.SWEEP_ARC

    mission.update([], now=0.03)
    assert commands[-1] == pytest.approx(
        (0.0, mission.config.search_angular_z)
    )

    mission.update_heading(25.0, now=0.04)
    mission.update([], now=0.04)
    assert (
        mission.search_controller.phase
        is task3_module.SearchPhase.RETURN_TO_BASE_HEADING
    )

    mission.update([], now=0.05)
    assert commands[-1][0] == pytest.approx(0.0)
    assert commands[-1][1] < 0.0

    mission.update_heading(15.0, now=0.06)
    mission.update([], now=0.06)
    assert mission.search_controller.phase is task3_module.SearchPhase.ADVANCE

    mission.update_heading(12.0, now=0.07)
    mission.update([], now=0.07)
    assert commands[-1][0] == pytest.approx(mission.config.search_linear_x)
    assert commands[-1][1] > 0.0

    mission.update_heading(15.0, now=2.57)
    mission.update([], now=2.57)
    assert (
        mission.search_controller.phase
        is task3_module.SearchPhase.MOVE_TO_ARC_START
    )
    assert mission.search_controller.sweep_deg == pytest.approx(30.0)
    assert mission.search_controller.base_heading == pytest.approx(15.0)
    assert mission.state is task3_module.MissionState.SEARCH


def test_invalid_target_data_pauses_search_phase_timeout(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    task3_module.stop_vehicle = (
        lambda publisher: commands.append((0.0, 0.0))
    )
    mission = _mission(task3_module)
    commands.clear()

    mission.update([], now=0.4)
    mission.update([_target(distance=None)], now=10.4)
    mission.update([_target(distance=None)], now=20.4)
    mission.update([], now=20.5)

    assert mission.state is task3_module.MissionState.SEARCH
    assert (
        mission.search_controller.phase
        is task3_module.SearchPhase.MOVE_TO_ARC_START
    )
    assert commands[-1][0] == pytest.approx(0.0)
    assert commands[-1][1] < 0.0


def test_target_interrupts_heading_held_search_advance(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    mission.search_controller.phase = task3_module.SearchPhase.ADVANCE
    mission.search_controller.base_heading = mission.current_heading
    mission.search_controller.phase_started_at = 0.1
    commands.clear()

    mission.update([_target(class_name="red_buoys")], now=0.2)

    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM
    assert commands == []


def test_search_heading_watchdog_enters_failsafe(task3_module):
    stopped = []
    task3_module.stop_vehicle = lambda publisher: stopped.append(publisher)
    mission = _mission(
        task3_module,
        gps_timeout_sec=200.0,
        heading_timeout_sec=200.0,
        mission_timeout_sec=200.0,
    )

    mission.update([], now=100.0)

    assert mission.state is task3_module.MissionState.FAILSAFE
    assert stopped[-1] == "cmd_vel"


def test_final_confirmation_rejects_unstable_distance(task3_module):
    mission = _mission(task3_module, final_distance_spread_m=0.2)
    now = 0.1
    mission.update([_target(distance=2.5)], now=now)
    mission.update([_target(distance=2.5)], now=now + 0.05)
    mission.update([_target(distance=2.5)], now=now + 0.10)
    for offset in range(mission.config.approach_distance_required):
        mission.update(
            [_target(distance=1.2)],
            now=now + 0.15 + offset * 0.05,
        )
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    mission.update([_target(distance=1.2)], now=now + 0.45)
    mission.update([_target(distance=1.5)], now=now + 0.50)

    assert mission.state is task3_module.MissionState.FINAL_CONFIRM
    assert mission.impact_count == 0


def test_production_defaults_use_aggressive_approach_and_three_impacts(
        task3_module,
):
    defaults = task3_module.Task3Config()
    assert defaults.approach_distance_window_size == 7
    assert defaults.approach_distance_required == 5
    assert defaults.realign_threshold_deg == pytest.approx(30.0)
    assert defaults.far_approach_speed == pytest.approx(0.70)
    assert defaults.medium_approach_speed == pytest.approx(0.55)
    assert defaults.near_approach_speed == pytest.approx(0.40)
    assert defaults.final_confirmation_required == 2
    assert defaults.final_confirm_forward_speed == pytest.approx(0.35)
    assert defaults.ram_speed == pytest.approx(0.85)
    assert defaults.ram_duration_sec == pytest.approx(2.0)
    assert defaults.required_impact_count == 3

    mission = _mission(
        task3_module,
        final_confirmation_required=2,
    )
    far_target = [_target(distance=2.5)]
    near_target = [_target(distance=1.2)]
    mission.update(far_target, now=0.1)
    mission.update(far_target, now=0.2)
    mission.update(far_target, now=0.3)
    for index in range(mission.config.approach_distance_required):
        mission.update(near_target, now=0.4 + index * 0.1)
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    mission.update(near_target, now=1.0)
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    mission.update(near_target, now=1.1)
    assert mission.state is task3_module.MissionState.RAM


def test_negative_motion_is_always_rejected(task3_module):
    commands = []
    stopped = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    task3_module.stop_vehicle = lambda publisher: stopped.append(publisher)
    mission = _mission(task3_module)

    accepted = mission._publish_motion(
        linear_x=-0.1,
        angular_z=0.0,
        reason="invalid test command",
    )

    assert accepted is False
    assert mission.state is task3_module.MissionState.FAILSAFE
    assert commands == []
    assert stopped[-1] == "cmd_vel"


def test_impact_gps_is_saved_and_vehicle_clears_forward(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    mission.state = task3_module.MissionState.RAM
    mission.state_started_at = 1.0
    mission.current_lat = 37.95126
    mission.current_lon = 32.50091
    commands.clear()

    contact_time = 1.0 + mission.config.ram_duration_sec + 0.01
    mission.update([], now=contact_time)

    assert mission.state is task3_module.MissionState.CONTACT_HOLD
    assert mission.impact_target_gps["lat"] == pytest.approx(37.95126)
    assert mission.impact_target_gps["lon"] == pytest.approx(32.50091)
    assert mission.impact_target_gps["recorded_at"] == pytest.approx(
        contact_time
    )
    assert mission.impact_target_gps["impact_count"] == 1

    hold_end = contact_time + mission.config.contact_hold_sec + 0.01
    mission.update([], now=hold_end)
    assert mission.state is task3_module.MissionState.FORWARD_CLEAR

    mission.update([], now=hold_end + 0.01)
    assert commands[-1] == pytest.approx(
        (mission.config.post_impact_forward_speed, 0.0)
    )
    assert all(linear >= 0.0 for linear, _ in commands)


def test_saved_impact_gps_guides_return_until_camera_reacquires(
        task3_module,
):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    mission.state = task3_module.MissionState.IMPACT_RETURN
    mission.state_started_at = 1.0
    mission.impact_count = 1
    mission.impact_target_gps = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
    }
    mission.update_gps(37.95127, 32.50090, now=1.1)
    mission.update_heading(0.0, now=1.1)
    commands.clear()

    mission.update([], now=1.1)
    assert commands[-1][0] == pytest.approx(0.0)
    assert commands[-1][1] != pytest.approx(0.0)

    mission.update_heading(180.0, now=1.2)
    mission.update([], now=1.2)
    assert commands[-1][0] == pytest.approx(
        mission.config.impact_return_speed
    )
    assert commands[-1][0] > 0.0

    mission.update([_target(distance=2.0)], now=1.3)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM
    assert all(linear >= 0.0 for linear, _ in commands)


def test_exactly_three_confirmed_impacts_finish_task(task3_module):
    ram_commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        ram_commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module, required_impact_count=3)
    now = 0.1

    for expected_count in range(1, 4):
        now = _acquire_and_reach_final_confirm(mission, task3_module, now)
        now = _complete_one_impact(mission, task3_module, now)
        assert mission.impact_count == expected_count

        if expected_count < 3:
            assert mission.state is task3_module.MissionState.FORWARD_CLEAR
            now += mission.config.post_impact_forward_duration_sec + 0.01
            mission.update([], now=now)
            assert mission.state is task3_module.MissionState.IMPACT_RETURN
            now += 0.01
        else:
            assert mission.state is task3_module.MissionState.FINISHED
            assert mission.finished is True

    positive_ram_commands = [
        command
        for command in ram_commands
        if command[0] == mission.config.ram_speed
    ]
    assert len(positive_ram_commands) == 3

    mission.update([_target(distance=1.0)], now=now + 1.0)
    assert mission.impact_count == 3
    assert mission.state is task3_module.MissionState.FINISHED


def test_stale_vision_enters_failsafe(task3_module):
    mission = _mission(task3_module)

    mission.update([], now=0.1, vision_fresh=False)

    assert mission.state is task3_module.MissionState.FAILSAFE
    assert mission.finished is False


def test_standalone_timer_does_not_run_before_mission_is_active(task3_module):
    node = task3_module.Task3Node.__new__(task3_module.Task3Node)
    node.mission_active = False
    node.task = types.SimpleNamespace(
        update=lambda *args, **kwargs: pytest.fail(
            "inactive standalone timer must not update the mission"
        )
    )

    node.timer_callback()
