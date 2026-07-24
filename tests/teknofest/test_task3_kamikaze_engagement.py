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
        "bridge_timeout_sec": 100.0,
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


def _target(distance=3.0, angle=0.0, class_name="red_buoy", confidence=0.9):
    return {
        "class": class_name,
        "confidence": confidence,
        "distance": distance,
        "bbox": [500, 200, 650, 500],
        "Buoy angle: ": angle,
        "Buoy side: ": "across",
    }


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


def test_search_rejects_wrong_class_and_invalid_depth(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)

    mission.update([_target(class_name="yellow_buoy")], now=0.1)
    mission.update([_target(distance=-1.0)], now=0.2)
    mission.update([_target(distance=float("inf"))], now=0.3)

    assert mission.state is task3_module.MissionState.SEARCH
    assert all(linear_x == 0.0 for linear_x, _ in commands)
    assert commands[-1][1] == pytest.approx(mission.config.search_angular_z)


def test_target_requires_multiple_consistent_frames(task3_module):
    mission = _mission(task3_module)
    target = [_target(distance=3.0, angle=2.0)]

    mission.update(target, now=0.1)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM

    mission.update([], now=0.2)
    assert mission.state is task3_module.MissionState.ACQUIRE_CONFIRM

    mission.update(target, now=0.3)
    assert mission.state is task3_module.MissionState.ALIGN


def test_lost_target_stops_and_enters_reacquire(task3_module):
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


def test_full_scan_relocates_to_bounded_search_point(task3_module):
    gps_targets = []
    task3_module.publish_set_position = (
        lambda publisher, lat, lon:
        gps_targets.append((lat, lon))
    )
    mission = _mission(
        task3_module,
        search_scan_timeout_sec=0.1,
    )

    mission.update([], now=0.2)
    assert mission.state is task3_module.MissionState.SEARCH_RELOCATE
    assert _gps_distance_m(
        mission.home_lat,
        mission.home_lon,
        mission.search_target["lat"],
        mission.search_target["lon"],
    ) == pytest.approx(mission.config.search_radius_step_m, abs=0.02)

    mission.update([], now=0.25)
    assert gps_targets == [
        (
            mission.search_target["lat"],
            mission.search_target["lon"],
        )
    ]


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
