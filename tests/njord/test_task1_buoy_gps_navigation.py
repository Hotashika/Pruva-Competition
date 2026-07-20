import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK1_PATH = REPO_ROOT / "njord" / "missions" / "task1_maneuvering_and_path_finding.py"


@pytest.fixture()
def task1_module(monkeypatch):
    rclpy_module = types.ModuleType("rclpy")
    rclpy_node_module = types.ModuleType("rclpy.node")
    rclpy_node_module.Node = type("Node", (), {})
    rclpy_module.node = rclpy_node_module

    mavros_module = types.ModuleType("mavros_msgs")
    mavros_srv_module = types.ModuleType("mavros_msgs.srv")
    mavros_srv_module.SetMode = type(
        "SetMode",
        (),
        {"Request": type("Request", (), {})},
    )
    mavros_module.srv = mavros_srv_module

    std_msgs_module = types.ModuleType("std_msgs")
    std_msgs_msg_module = types.ModuleType("std_msgs.msg")
    std_msgs_msg_module.String = type("String", (), {})
    std_msgs_module.msg = std_msgs_msg_module

    mavlink_utilities = types.ModuleType("utils.mavlink_utilities")
    for name in (
        "align_heading_to_gps_target",
        "create_mission_topics",
        "create_mission_clients",
        "wait_for_mission_services",
        "call_set_mode",
        "call_trigger_service",
        "parse_bridge_state",
        "publish_cmd_vel",
        "publish_set_position",
        "stop_vehicle",
        "calculate_gps_distance",
    ):
        setattr(mavlink_utilities, name, lambda *args, **kwargs: None)

    read_waypoints = types.ModuleType("utils.read_waypoints")
    read_waypoints.parse_qgc_waypoints = lambda path: []

    for name, module in {
        "rclpy": rclpy_module,
        "rclpy.node": rclpy_node_module,
        "mavros_msgs": mavros_module,
        "mavros_msgs.srv": mavros_srv_module,
        "std_msgs": std_msgs_module,
        "std_msgs.msg": std_msgs_msg_module,
        "utils.mavlink_utilities": mavlink_utilities,
        "utils.read_waypoints": read_waypoints,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("task1_buoy_gps_test_module", TASK1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mission_without_ros(task1_module, heading):
    mission = task1_module.Task1Maneuvering.__new__(task1_module.Task1Maneuvering)
    mission.current_lat = 63.4305
    mission.current_lon = 10.3951
    mission.current_heading = heading
    mission.aligned_target_key = None
    return mission


def _gps_offset_m(task1_module, origin_lat, origin_lon, target_lat, target_lon):
    north_m = math.radians(target_lat - origin_lat) * task1_module.EARTH_RADIUS_M
    east_m = (
        math.radians(target_lon - origin_lon)
        * task1_module.EARTH_RADIUS_M
        * math.cos(math.radians(origin_lat))
    )
    return north_m, east_m


@pytest.mark.parametrize(
    ("obstacle_class", "heading", "expected_side", "lateral_angle"),
    (
        ("red_buoys", 0.0, "starboard", 90.0),
        ("red_buoys", 90.0, "starboard", 180.0),
        ("green_buoys", 0.0, "port", -90.0),
        ("green_buoys", 90.0, "port", 0.0),
    ),
)
def test_buoy_target_uses_vehicle_relative_pass_side(
        task1_module,
        obstacle_class,
        heading,
        expected_side,
        lateral_angle,
):
    mission = _mission_without_ros(task1_module, heading)

    target = mission._create_buoy_pass_target({
        "class": obstacle_class,
        "distance": 3.0,
        "angle_deg": 0.0,
    })

    assert target["pass_type"] == "buoy"
    assert target["side"] == expected_side
    north_m, east_m = _gps_offset_m(
        task1_module,
        target["marker_lat"],
        target["marker_lon"],
        target["lat"],
        target["lon"],
    )
    assert north_m == pytest.approx(
        task1_module.BUOY_PASS_CLEARANCE_M
        * math.cos(math.radians(lateral_angle)),
        abs=0.01,
    )
    assert east_m == pytest.approx(
        task1_module.BUOY_PASS_CLEARANCE_M
        * math.sin(math.radians(lateral_angle)),
        abs=0.01,
    )
    assert task1_module.BUOY_PASS_CLEARANCE_M == 2.5


@pytest.mark.parametrize(
    ("obstacle_class", "expected_east_m"),
    (
        ("east_buoys", 2.5),
        ("west_buoys", -2.5),
    ),
)
def test_cardinal_target_uses_2_5_meter_clearance(
        task1_module,
        obstacle_class,
        expected_east_m,
):
    mission = _mission_without_ros(task1_module, heading=0.0)

    target = mission._create_cardinal_pass_target({
        "class": obstacle_class,
        "distance": 3.0,
        "angle_deg": 0.0,
    })

    north_m, east_m = _gps_offset_m(
        task1_module,
        target["marker_lat"],
        target["marker_lon"],
        target["lat"],
        target["lon"],
    )
    assert north_m == pytest.approx(0.0, abs=0.01)
    assert east_m == pytest.approx(expected_east_m, abs=0.01)
    assert task1_module.CARDINAL_PASS_CLEARANCE_M == 2.5


@pytest.mark.parametrize("obstacle_class", ("red_buoys", "green_buoys"))
def test_buoy_avoidance_follows_generated_gps_target(task1_module, obstacle_class):
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    mission.topics = types.SimpleNamespace(cmd_vel_pub=object())
    gps_targets = []
    mission._set_position_to_gps_target = (
        lambda lat, lon, name, tolerance: gps_targets.append(
            (lat, lon, name, tolerance)
        ) or False
    )

    mission._start_avoidance({
        "class": obstacle_class,
        "distance": 2.5,
        "angle_deg": 0.0,
    }, now=10.0)

    initial_target = mission.cardinal_pass_target
    assert initial_target["pass_type"] == "buoy"
    assert initial_target["reference_heading"] == 0.0

    mission.current_heading = 30.0
    assert mission._update_active_avoidance([{
        "class": obstacle_class,
        "distance": 2.0,
        "angle_deg": 20.0,
    }], now=11.0)

    refreshed_target = mission.cardinal_pass_target
    assert refreshed_target["reference_heading"] == 0.0
    assert refreshed_target["lat"] != initial_target["lat"]
    assert refreshed_target["lon"] != initial_target["lon"]
    assert len(gps_targets) == 1
    assert gps_targets[0][:2] == (
        refreshed_target["lat"],
        refreshed_target["lon"],
    )
    assert gps_targets[0][2].endswith("buoy pass")
    assert gps_targets[0][3] == task1_module.CARDINAL_TARGET_TOLERANCE_M


def test_buoy_avoidance_finishes_only_when_temporary_target_is_reached(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    mission.state = task1_module.MissionState.AVOIDING
    mission.avoiding_class = "red_buoys"
    mission.avoid_started_time = 10.0
    mission.cardinal_pass_target = mission._create_buoy_pass_target({
        "class": "red_buoys",
        "distance": 2.0,
        "angle_deg": 0.0,
    })
    reached = {"value": False}
    mission._set_position_to_gps_target = (
        lambda *args, **kwargs: reached["value"]
    )

    assert mission._update_active_avoidance([], now=11.0)
    assert mission.state is task1_module.MissionState.AVOIDING

    reached["value"] = True
    assert mission._update_active_avoidance([], now=12.0)
    assert mission.state is task1_module.MissionState.NAVIGATING
    assert mission.cardinal_pass_target is None


def test_missing_detection_angle_fails_safe_without_direct_maneuver(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    mission.topics = types.SimpleNamespace(cmd_vel_pub=object())
    failsafe_requests = []

    def enter_failsafe(reason, request_hold=False):
        failsafe_requests.append((reason, request_hold))
        mission.state = task1_module.MissionState.FAILSAFE

    mission._enter_failsafe = enter_failsafe
    mission._start_avoidance({
        "class": "red_buoys",
        "distance": 2.0,
    }, now=10.0)

    assert mission.state is task1_module.MissionState.FAILSAFE
    assert mission.cardinal_pass_target is None
    assert failsafe_requests[0][1] is True
