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
        "publish_set_position",
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
        "retreat_min_sec": 0.1,
        "retreat_max_sec": 0.2,
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
    mission.update([], now=now)
    assert mission.state is task3_module.MissionState.RAM
    now += mission.config.ram_duration_sec + 0.01
    mission.update([], now=now)
    assert mission.state is task3_module.MissionState.CONTACT_HOLD
    now += mission.config.contact_hold_sec + 0.01
    mission.update([], now=now)
    return now


def test_search_moves_forward_for_wrong_class_and_stops_for_invalid_target_data(
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
    assert commands[-1] == pytest.approx(
        (mission.config.search_linear_x, mission.config.search_angular_z)
    )

    mission.update([_target(distance=-1.0)], now=0.2)
    assert commands[-1] == (0.0, 0.0)

    mission.update([_target(distance=float("inf"))], now=0.3)
    missing_direction = _target(distance=2.0, angle=None)
    missing_direction.pop("Buoy side: ")
    mission.update([missing_direction], now=0.4)
    assert commands[-1] == (0.0, 0.0)

    mission.update([_target(confidence=None)], now=0.5)

    assert mission.state is task3_module.MissionState.SEARCH
    assert commands[-1][0] == pytest.approx(
        mission.config.search_linear_x
    )


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


def test_s_search_alternates_forward_legs_then_relocates(task3_module):
    commands = []
    gps_targets = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    task3_module.publish_set_position = (
        lambda publisher, lat, lon:
        gps_targets.append((lat, lon))
    )
    mission = _mission(
        task3_module,
        search_leg_timeout_sec=0.1,
        search_legs_per_cycle=4,
    )
    commands.clear()

    mission.update([], now=0.01)
    mission.update([], now=0.11)
    mission.update([], now=0.22)
    mission.update([], now=0.33)
    mission.update([], now=0.44)

    assert mission.state is task3_module.MissionState.SEARCH_RELOCATE
    assert [command[0] for command in commands] == pytest.approx(
        [mission.config.search_linear_x] * 4
    )
    assert [1 if command[1] > 0.0 else -1 for command in commands] == [
        1, -1, 1, -1,
    ]
    assert _gps_distance_m(
        mission.home_lat,
        mission.home_lon,
        mission.search_target["lat"],
        mission.search_target["lon"],
    ) == pytest.approx(mission.config.search_radius_step_m, abs=0.02)

    mission.update([], now=0.45)
    assert gps_targets == [
        (
            mission.search_target["lat"],
            mission.search_target["lon"],
        )
    ]

    mission._enter_search(now=0.5, reason="next search cycle")
    assert mission.search_direction == -1.0


def test_invalid_target_data_pauses_search_leg_timeout(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    task3_module.stop_vehicle = (
        lambda publisher: commands.append((0.0, 0.0))
    )
    mission = _mission(
        task3_module,
        search_leg_timeout_sec=1.0,
    )
    commands.clear()

    mission.update([], now=0.4)
    mission.update([_target(distance=None)], now=1.4)
    mission.update([_target(distance=None)], now=2.4)
    mission.update([], now=2.5)

    assert mission.state is task3_module.MissionState.SEARCH
    assert mission.search_leg_index == 0
    assert commands[-1][0] == pytest.approx(
        mission.config.search_linear_x
    )


def test_target_interrupts_search_relocation_without_gps_publish(task3_module):
    mission = _mission(task3_module)
    mission._enter_search_relocate(now=0.1)
    task3_module.publish_set_position = lambda *args, **kwargs: pytest.fail(
        "visible target must interrupt GPS search relocation"
    )

    mission.update([_target(class_name="red_buoys")], now=0.2)

    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM


def test_final_confirmation_rejects_unstable_distance(task3_module):
    mission = _mission(task3_module, final_distance_spread_m=0.2)
    now = 0.1
    mission.update([_target(distance=2.5)], now=now)
    mission.update([_target(distance=2.5)], now=now + 0.05)
    mission.update([_target(distance=2.5)], now=now + 0.10)
    mission.update([_target(distance=1.2)], now=now + 0.15)
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    mission.update([_target(distance=1.2)], now=now + 0.20)
    mission.update([_target(distance=1.5)], now=now + 0.25)

    assert mission.state is task3_module.MissionState.FINAL_CONFIRM
    assert mission.impact_count == 0


def test_production_defaults_require_three_frames_and_three_impacts(task3_module):
    defaults = task3_module.Task3Config()
    assert defaults.final_confirmation_required == 3
    assert defaults.required_impact_count == 3

    mission = _mission(
        task3_module,
        final_confirmation_required=3,
    )
    far_target = [_target(distance=2.5)]
    near_target = [_target(distance=1.2)]
    mission.update(far_target, now=0.1)
    mission.update(far_target, now=0.2)
    mission.update(far_target, now=0.3)
    mission.update(near_target, now=0.4)
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    mission.update(near_target, now=0.5)
    mission.update(near_target, now=0.6)
    assert mission.state is task3_module.MissionState.FINAL_CONFIRM

    mission.update(near_target, now=0.7)
    assert mission.state is task3_module.MissionState.RAM


def test_negative_motion_is_rejected_outside_confirmed_retreat(task3_module):
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


def test_confirmed_retreat_is_short_straight_and_heading_corrected(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(
        task3_module,
        retreat_min_sec=0.6,
        retreat_max_sec=1.5,
    )
    mission.state = task3_module.MissionState.RETREAT
    mission.state_started_at = 1.0
    mission.impact_count = 1
    mission.retreat_heading = 15.0
    mission.current_heading = 20.0
    commands.clear()

    mission.update([], now=1.2)

    assert commands[-1][0] == pytest.approx(-0.25)
    assert commands[-1][1] == pytest.approx(math.radians(-5.0))

    mission.update([], now=2.5)

    assert mission.state is task3_module.MissionState.REACQUIRE
    assert len([linear for linear, _ in commands if linear < 0.0]) == 1


def test_retreat_heading_is_captured_when_contact_completes(task3_module):
    mission = _mission(task3_module)
    mission.state = task3_module.MissionState.RAM
    mission.state_started_at = 1.0
    mission.current_heading = 42.0

    contact_time = 1.0 + mission.config.ram_duration_sec + 0.01
    mission.update([], now=contact_time)

    assert mission.state is task3_module.MissionState.CONTACT_HOLD
    assert mission.retreat_heading == pytest.approx(42.0)

    mission.current_heading = 80.0
    mission.update(
        [],
        now=contact_time + mission.config.contact_hold_sec + 0.01,
    )
    assert mission.state is task3_module.MissionState.RETREAT
    assert mission.retreat_heading == pytest.approx(42.0)


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
            assert mission.state is task3_module.MissionState.RETREAT
            now += mission.config.retreat_max_sec + 0.01
            mission.update([], now=now)
            assert mission.state is task3_module.MissionState.REACQUIRE
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
