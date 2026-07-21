import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK1_PATH = REPO_ROOT / "teknofest" / "missions" / "task1_point_tracking.py"


@pytest.fixture()
def task1_module(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_node.Node = type("Node", (), {})
    rclpy_qos.QoSHistoryPolicy = type("QoSHistoryPolicy", (), {"KEEP_LAST": 1})
    rclpy_qos.QoSReliabilityPolicy = type(
        "QoSReliabilityPolicy",
        (),
        {"BEST_EFFORT": 1},
    )
    rclpy_qos.QoSProfile = lambda **kwargs: kwargs

    mavros_msgs = types.ModuleType("mavros_msgs")
    mavros_srv = types.ModuleType("mavros_msgs.srv")
    mavros_srv.SetMode = type("SetMode", (), {"Request": type("Request", (), {})})
    mavros_msgs.srv = mavros_srv

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})
    std_msgs.msg = std_msgs_msg

    mavlink_utilities = types.ModuleType("utils.mavlink_utilities")
    utility_names = (
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
    )
    for name in utility_names:
        setattr(mavlink_utilities, name, lambda *args, **kwargs: None)

    read_waypoints = types.ModuleType("utils.read_waypoints")
    read_waypoints.parse_qgc_waypoints = lambda path: []

    for name, module in {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "mavros_msgs": mavros_msgs,
        "mavros_msgs.srv": mavros_srv,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "utils.mavlink_utilities": mavlink_utilities,
        "utils.read_waypoints": read_waypoints,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("task1_direct_gps_test_module", TASK1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mission(task1_module):
    mission = task1_module.Task1Maneuvering.__new__(task1_module.Task1Maneuvering)
    mission.current_lat = 37.95125
    mission.current_lon = 32.50090
    mission.current_heading = 15.0
    mission.aligned_target_key = None
    mission.last_angular_z = 1.0
    mission.topics = types.SimpleNamespace(
        cmd_vel_pub=object(),
        position_target_pub=object(),
    )
    mission.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    return mission


def test_waypoint_coordinates_are_published_without_vision_adjustment(task1_module):
    mission = _mission(task1_module)
    target_lat = 37.95200
    target_lon = 32.50150
    aligned_targets = []
    published_targets = []
    task1_module.calculate_gps_distance = lambda *args: 10.0
    task1_module.align_heading_to_gps_target = (
        lambda publisher, current_lat, current_lon, heading, lat, lon, **kwargs:
        aligned_targets.append((lat, lon)) or True
    )
    task1_module.publish_set_position = (
        lambda publisher, lat, lon: published_targets.append((lat, lon))
    )

    reached = mission._set_position_to_gps_target(
        target_lat,
        target_lon,
        "WP1",
        1.0,
    )

    assert reached is False
    assert aligned_targets == [(target_lat, target_lon)]
    assert published_targets == [(target_lat, target_lon)]
    assert mission.last_angular_z == 0.0


def test_reached_waypoint_is_not_published_again(task1_module):
    mission = _mission(task1_module)
    published_targets = []
    task1_module.calculate_gps_distance = lambda *args: 0.4
    task1_module.publish_set_position = (
        lambda publisher, lat, lon: published_targets.append((lat, lon))
    )

    reached = mission._set_position_to_gps_target(
        37.95200,
        32.50150,
        "WP1",
        1.0,
    )

    assert reached is True
    assert published_targets == []
