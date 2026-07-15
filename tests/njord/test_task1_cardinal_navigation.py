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

    spec = importlib.util.spec_from_file_location("task1_cardinal_test_module", TASK1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mission_without_ros(task1_module, heading):
    mission = task1_module.Task1Maneuvering.__new__(task1_module.Task1Maneuvering)
    mission.current_lat = 63.4305
    mission.current_lon = 10.3951
    mission.current_heading = heading
    return mission


def _east_delta_m(task1_module, lat, lon_from, lon_to):
    return (
        math.radians(lon_to - lon_from)
        * task1_module.EARTH_RADIUS_M
        * math.cos(math.radians(lat))
    )


@pytest.mark.parametrize(
    ("obstacle_class", "expected_side", "expected_sign"),
    (
        ("east_buoys", "east", 1.0),
        ("west_buoys", "west", -1.0),
    ),
)
def test_cardinal_target_is_on_geographic_side(
        task1_module,
        obstacle_class,
        expected_side,
        expected_sign,
):
    mission = _mission_without_ros(task1_module, heading=0.0)
    target = mission._create_cardinal_pass_target({
        "class": obstacle_class,
        "distance": 3.0,
        "Buoy angle: ": 0.0,
    })

    assert target is not None
    assert target["side"] == expected_side
    assert target["marker_lat"] > mission.current_lat

    east_delta_m = _east_delta_m(
        task1_module,
        target["marker_lat"],
        target["marker_lon"],
        target["lon"],
    )
    assert east_delta_m == pytest.approx(
        expected_sign * task1_module.CARDINAL_PASS_CLEARANCE_M,
        abs=0.01,
    )


@pytest.mark.parametrize(
    ("heading", "obstacle_class", "expected_turn"),
    (
        (0.0, "east_buoys", 1.0),
        (180.0, "east_buoys", -1.0),
        (0.0, "west_buoys", -1.0),
        (180.0, "west_buoys", 1.0),
    ),
)
def test_cardinal_fallback_turn_depends_on_heading(
        task1_module,
        heading,
        obstacle_class,
        expected_turn,
):
    mission = _mission_without_ros(task1_module, heading=heading)

    turn = mission._avoid_turn_direction_for_obstacle({"class": obstacle_class})

    assert turn == expected_turn
