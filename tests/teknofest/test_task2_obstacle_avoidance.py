import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK2_PATH = (
    REPO_ROOT
    / "teknofest"
    / "missions"
    / "task2_point_tracking_task_in_an_environment_with_obstacle.py"
)


@pytest.fixture()
def task2_module(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_node.Node = type("Node", (), {})
    rclpy_qos.QoSHistoryPolicy = type("QoSHistoryPolicy", (), {"KEEP_LAST": 1})
    rclpy_qos.QoSReliabilityPolicy = type("QoSReliabilityPolicy", (), {"BEST_EFFORT": 1})
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
        "align_heading_to_gps_target", "calculate_bearing",
        "calculate_gps_distance", "call_set_mode", "call_trigger_service",
        "create_mission_clients", "create_mission_topics", "parse_bridge_state",
        "publish_cmd_vel", "publish_set_position", "stop_vehicle",
        "wait_for_mission_services",
    )
    for name in utility_names:
        setattr(mavlink_utilities, name, lambda *args, **kwargs: None)
    mavlink_utilities.calculate_bearing = lambda *args, **kwargs: 0.0

    read_waypoints = types.ModuleType("utils.read_waypoints")
    read_waypoints.parse_qgc_waypoints = lambda path: []

    modules = {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "mavros_msgs": mavros_msgs,
        "mavros_msgs.srv": mavros_srv,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "utils.mavlink_utilities": mavlink_utilities,
        "utils.read_waypoints": read_waypoints,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("task2_avoidance_test_module", TASK2_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mission(task2_module):
    mission = task2_module.Task2PointTrackingWithObstacleAvoidance.__new__(
        task2_module.Task2PointTrackingWithObstacleAvoidance
    )
    mission.current_lat = 37.95125
    mission.current_lon = 32.50090
    mission.current_heading = 0.0
    mission.obstacle_data_uncertain = False
    mission.avoidance_target = None
    mission.avoidance_side = None
    mission.avoided_obstacle_side = None
    mission.last_angular_z = 0.0
    mission.state = task2_module.MissionState.NAVIGATING
    mission.logger = types.SimpleNamespace(
        warn=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    return mission


def test_close_yellow_buoy_without_direction_stops_as_uncertain(task2_module):
    mission = _mission(task2_module)
    obstacle = mission._nearest_relevant_obstacle([
        {"class": "yellow_buoy", "confidence": 0.9, "distance": 2.0}
    ])

    assert obstacle is None
    assert mission.obstacle_data_uncertain is True


def test_avoidance_target_is_created_on_opposite_side(task2_module):
    mission = _mission(task2_module)
    mission._start_avoidance(
        {"class": "yellow_buoy", "distance": 2.0, "side": "left"},
        mission.current_lat + 0.001,
        mission.current_lon,
    )

    assert mission.state is task2_module.MissionState.AVOIDING
    assert mission.avoidance_side == "right"
    assert mission.avoidance_target["lat"] > mission.current_lat
    assert mission.avoidance_target["lon"] > mission.current_lon


def test_finishing_avoidance_clears_target_and_resumes_route(task2_module):
    mission = _mission(task2_module)
    mission.state = task2_module.MissionState.AVOIDING
    mission.avoidance_side = "right"
    mission.avoidance_target = {"lat": 1.0, "lon": 2.0}
    mission.avoided_obstacle_side = "left"
    held = []
    mission._begin_waypoint_hold = held.append

    mission._finish_avoidance()

    assert mission.state is task2_module.MissionState.NAVIGATING
    assert mission.avoidance_target is None
    assert mission.avoidance_side is None
    assert mission.avoided_obstacle_side is None
    assert held == ["kaçınma WP (right)"]
