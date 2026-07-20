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
    mission.aligned_target_key = None
    mission.yellow_course_acquired = False
    mission.yellow_initial_search_started_time = None
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


def test_orange_buoy_does_not_affect_task2_navigation(task2_module):
    mission = _mission(task2_module)

    obstacle = mission._nearest_relevant_obstacle([{
        "class": "orange_buoy",
        "confidence": 0.99,
        "distance": 1.0,
        "angle": 0.0,
    }])

    assert obstacle is None
    assert mission.obstacle_data_uncertain is False


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
    marker = {
        "lat": mission.avoidance_target["marker_lat"],
        "lon": mission.avoidance_target["marker_lon"],
    }
    assert mission._gps_target_shift_m(
        marker,
        mission.avoidance_target,
    ) == pytest.approx(task2_module.AVOID_PASS_CLEARANCE_M, abs=0.01)
    assert task2_module.AVOID_PASS_CLEARANCE_M == 2.5


def test_active_avoidance_target_is_refreshed_from_vision(task2_module):
    mission = _mission(task2_module)
    main_target_lat = mission.current_lat + 0.001
    main_target_lon = mission.current_lon
    mission._start_avoidance(
        {
            "class": "yellow_buoy",
            "distance": 2.5,
            "side": "left",
            "angle": 0.0,
        },
        main_target_lat,
        main_target_lon,
    )
    initial_target = dict(mission.avoidance_target)

    mission.current_heading = 25.0
    mission._refresh_avoidance_target(
        {
            "class": "yellow_buoy",
            "distance": 1.8,
            "side": "left",
            "angle": 15.0,
        },
        main_target_lat,
        main_target_lon,
    )

    assert mission.avoidance_target["reference_heading"] == initial_target[
        "reference_heading"
    ]
    assert mission.avoidance_target["lat"] != initial_target["lat"]
    assert mission.avoidance_target["lon"] != initial_target["lon"]


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


def test_active_avoidance_bypasses_yellow_course_target(task2_module):
    mission = _mission(task2_module)
    published_targets = []
    mission.topics = types.SimpleNamespace(
        cmd_vel_pub=object(),
        position_target_pub=object(),
    )
    mission.course_keeper = types.SimpleNamespace(
        compute=lambda **kwargs: pytest.fail(
            "course keeper must not override an active avoidance target"
        )
    )
    task2_module.calculate_gps_distance = lambda *args, **kwargs: 10.0
    task2_module.align_heading_to_gps_target = lambda *args, **kwargs: True
    task2_module.publish_set_position = (
        lambda publisher, lat, lon: published_targets.append((lat, lon))
    )

    reached = mission._navigate_to_gps_target(
        37.95200,
        32.50150,
        "kaçınma WP (right)",
        1.0,
        detections=[],
        follow_yellow_course=False,
    )

    assert reached is False
    assert published_targets == [(37.95200, 32.50150)]


def test_normal_navigation_publishes_dynamic_yellow_course_target(task2_module):
    mission = _mission(task2_module)
    published_targets = []
    received_detections = []
    mission.topics = types.SimpleNamespace(
        cmd_vel_pub=object(),
        position_target_pub=object(),
    )

    def compute_course(**kwargs):
        received_detections.extend(kwargs["detections"])
        return types.SimpleNamespace(
            should_stop=False,
            target_lat=37.95160,
            target_lon=32.50120,
            status="live",
            reason="second_nearest_yellow_buoy",
            relative_bearing_deg=12.0,
            selected_distance_m=6.0,
            candidate_count=3,
        )

    mission.course_keeper = types.SimpleNamespace(compute=compute_course)
    task2_module.calculate_gps_distance = lambda *args, **kwargs: 10.0
    task2_module.align_heading_to_gps_target = lambda *args, **kwargs: True
    task2_module.publish_set_position = (
        lambda publisher, lat, lon: published_targets.append((lat, lon))
    )
    detections = [
        {"class": "yellow_buoy", "distance": 3.0},
        {"class": "yellow_buoy", "distance": 6.0},
    ]

    reached = mission._navigate_to_gps_target(
        37.95200,
        32.50150,
        "WP1",
        1.0,
        detections=detections,
    )

    assert reached is False
    assert received_detections == detections
    assert published_targets == [(37.95160, 32.50120)]
    assert mission.yellow_course_acquired is True


def test_initial_yellow_search_uses_main_waypoint_then_stops(task2_module):
    mission = _mission(task2_module)
    published_targets = []
    velocity_commands = []
    clock = {"now": 10.0}
    mission.topics = types.SimpleNamespace(
        cmd_vel_pub=object(),
        position_target_pub=object(),
    )
    mission.course_keeper = types.SimpleNamespace(
        compute=lambda **kwargs: types.SimpleNamespace(
            should_stop=True,
            status="blocked",
            reason="fewer_than_two_yellow_buoys",
            target_lat=None,
            target_lon=None,
        )
    )
    task2_module.time.monotonic = lambda: clock["now"]
    task2_module.calculate_gps_distance = lambda *args, **kwargs: 10.0
    task2_module.align_heading_to_gps_target = lambda *args, **kwargs: True
    task2_module.publish_set_position = (
        lambda publisher, lat, lon: published_targets.append((lat, lon))
    )
    task2_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z: velocity_commands.append(
            (linear_x, angular_z)
        )
    )
    main_target = (37.95200, 32.50150)

    mission._navigate_to_gps_target(
        *main_target,
        "WP1",
        1.0,
        detections=[],
    )
    clock["now"] = 12.9
    mission._navigate_to_gps_target(
        *main_target,
        "WP1",
        1.0,
        detections=[],
    )
    clock["now"] = 13.1
    mission._navigate_to_gps_target(
        *main_target,
        "WP1",
        1.0,
        detections=[],
    )

    assert published_targets == [main_target, main_target]
    assert velocity_commands == [(0.0, 0.0)]
