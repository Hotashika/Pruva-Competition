import importlib.util
import math
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


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


@pytest.fixture()
def task2_module(monkeypatch):
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

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

    class Vector3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:
        def __init__(self):
            self.linear = Vector3()
            self.angular = Vector3()

    geometry_msgs_msg.Twist = Twist
    geometry_msgs.msg = geometry_msgs_msg

    mavros_msgs = types.ModuleType("mavros_msgs")
    mavros_srv = types.ModuleType("mavros_msgs.srv")
    mavros_srv.SetMode = type(
        "SetMode",
        (),
        {"Request": type("Request", (), {})},
    )
    mavros_msgs.srv = mavros_srv

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})
    std_msgs.msg = std_msgs_msg

    mavlink_utilities = types.ModuleType("utils.mavlink_utilities")
    utility_names = (
        "align_heading_to_gps_target",
        "calculate_gps_distance",
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

    read_waypoints = types.ModuleType("utils.read_waypoints")
    read_waypoints.parse_qgc_waypoints = lambda path: []

    modules = {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "mavros_msgs": mavros_msgs,
        "mavros_msgs.srv": mavros_srv,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "utils.mavlink_utilities": mavlink_utilities,
        "utils.read_waypoints": read_waypoints,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "task2_avoidance_test_module",
        TASK2_PATH,
    )
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
    mission.current_target_index = 0
    mission.waypoint_tolerance = 1.0
    mission.waypoints = [{"lat": 37.95200, "lon": 32.50150}]
    mission.finished = False

    mission.obstacle_data_uncertain = False
    mission.avoidance_side = None
    mission.avoided_obstacle_side = None
    mission.avoidance_started_time = None
    mission.avoidance_phase = None
    mission.avoidance_phase_started_time = None
    mission.avoidance_entry_heading = None
    mission.avoidance_clear_started_time = None
    mission.last_avoidance_speed_m_s = 0.0
    mission.avoiding_track_id = None
    mission.active_obstacle_reference = None
    mission.pending_obstacle = None
    mission.pending_obstacle_time = None
    mission.pending_obstacle_count = 0

    mission.last_angular_z = 0.0
    mission.aligned_target_key = None
    mission.resume_navigation_without_alignment = False
    mission.waypoint_hold_until = None
    mission.waypoint_hold_name = None
    mission.state = task2_module.MissionState.NAVIGATING
    mission.topics = types.SimpleNamespace(
        cmd_vel_pub=object(),
        position_target_pub=object(),
        avoidance_velocity_pub=FakePublisher(),
    )
    mission.clients = types.SimpleNamespace(
        set_mode_client=types.SimpleNamespace(call_async=lambda request: None),
    )
    mission.logger = types.SimpleNamespace(
        warn=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    mission._check_watchdog = lambda: True
    mission._check_geofence = lambda: True
    mission._waypoint_hold_active = lambda: False
    return mission


def _yellow(**overrides):
    detection = {
        "class": "yellow_buoy",
        "confidence": 0.9,
        "distance": 2.5,
        "side": "left",
        "angle": -10.0,
        "bbox": [100, 100, 140, 160],
        "track_id": 7,
    }
    detection.update(overrides)
    return detection


def test_close_yellow_buoy_without_angle_or_direction_is_uncertain(task2_module):
    mission = _mission(task2_module)

    obstacle = mission._nearest_relevant_obstacle([
        {"class": "yellow_buoy", "confidence": 0.9, "distance": 2.0}
    ])

    assert obstacle is None
    assert mission.obstacle_data_uncertain is True


def test_only_yellow_buoy_triggers_obstacle_logic(task2_module):
    mission = _mission(task2_module)

    obstacle = mission._nearest_relevant_obstacle([_yellow(distance=1.0)])

    assert obstacle["class"] == "yellow_buoy"
    assert mission.obstacle_data_uncertain is False


@pytest.mark.parametrize(
    "class_name",
    ["red_buoys", "green_buoy", "orange_buoys"],
)
def test_non_yellow_buoys_do_not_trigger_obstacle_logic(
        task2_module,
        class_name,
):
    mission = _mission(task2_module)

    obstacle = mission._nearest_relevant_obstacle([{
        "class": class_name,
        "confidence": 0.99,
        "distance": 1.0,
        "angle": 0.0,
    }])

    assert obstacle is None
    assert mission.obstacle_data_uncertain is False


def test_non_buoy_detection_does_not_trigger_obstacle_logic(task2_module):
    mission = _mission(task2_module)

    obstacle = mission._nearest_relevant_obstacle([{
        "class": "boat",
        "confidence": 0.99,
        "distance": 1.0,
        "angle": 0.0,
    }])

    assert obstacle is None


def test_avoidance_candidate_boundaries_are_four_and_five(
        task2_module,
):
    mission = _mission(task2_module)

    start_boundary = mission._nearest_relevant_obstacle([
        _yellow(distance=4.0)
    ])
    clear_boundary = mission._nearest_relevant_obstacle([
        _yellow(distance=5.0)
    ])

    assert task2_module.AVOIDANCE_START_DISTANCE_M == 4.0
    assert start_boundary["distance"] == 4.0
    assert clear_boundary is None
    assert mission.obstacle_data_uncertain is False


@pytest.mark.parametrize(
    ("side", "expected_angle"),
    [
        ("left", -15.0),
        ("right", 15.0),
        ("center", 0.0),
    ],
)
def test_side_only_detection_uses_angle_fallback(
        task2_module,
        side,
        expected_angle,
):
    mission = _mission(task2_module)

    normalized = mission._normalize_detection(
        _yellow(side=side, angle=None)
    )

    assert normalized["angle"] == expected_angle


def test_real_angle_takes_priority_over_side_fallback(task2_module):
    mission = _mission(task2_module)

    normalized = mission._normalize_detection(
        _yellow(side="left", angle=-4.5)
    )

    assert normalized["angle"] == -4.5


@pytest.mark.parametrize(
    ("obstacle_side", "angle", "pass_side", "expected_bearing"),
    [
        ("left", -15.0, "right", 45.0),
        ("right", 15.0, "left", 315.0),
        ("center", 0.0, "right", 45.0),
    ],
)
def test_pass_side_policy_produces_expected_timed_velocity_direction(
        task2_module,
        obstacle_side,
        angle,
        pass_side,
        expected_bearing,
):
    mission = _mission(task2_module)

    assert mission._start_avoidance(
        _yellow(side=obstacle_side, angle=angle, distance=3.0),
        now=10.0,
    )

    velocity = mission.topics.avoidance_velocity_pub.messages[-1]
    expected_rad = math.radians(expected_bearing)
    expected_speed = mission.last_avoidance_speed_m_s
    assert mission.avoidance_side == pass_side
    assert velocity.linear.x == pytest.approx(
        expected_speed * math.cos(expected_rad)
    )
    assert velocity.linear.y == pytest.approx(
        expected_speed * math.sin(expected_rad)
    )


def test_legacy_dynamic_speed_profile_is_preserved(task2_module):
    mission = _mission(task2_module)
    mission.avoidance_side = "right"
    farther = mission._normalize_detection(
        _yellow(distance=3.0, angle=-10.0)
    )
    nearer = mission._normalize_detection(
        _yellow(distance=2.0, angle=10.0)
    )
    emergency = mission._normalize_detection(
        _yellow(distance=1.0, angle=0.0)
    )

    farther_speed = mission._calculate_avoidance_speed(farther)
    nearer_speed = mission._calculate_avoidance_speed(nearer)

    assert nearer_speed != pytest.approx(farther_speed)
    assert (
        task2_module.AVOIDANCE_MIN_LINEAR_SPEED
        <= farther_speed
        <= task2_module.AVOIDANCE_MAX_LINEAR_SPEED
    )
    assert (
        task2_module.AVOIDANCE_MIN_LINEAR_SPEED
        <= nearer_speed
        <= task2_module.AVOIDANCE_MAX_LINEAR_SPEED
    )
    assert mission._calculate_avoidance_speed(emergency) == 0.0


def test_confirmation_applies_ema_to_range_and_angle(task2_module):
    mission = _mission(task2_module)
    first = mission._normalize_detection(
        _yellow(distance=2.8, angle=-10.0, track_id=None)
    )
    second = mission._normalize_detection(
        _yellow(
            distance=2.0,
            angle=-6.0,
            bbox=[104, 100, 144, 160],
            track_id=None,
        )
    )

    assert mission._confirmed_obstacle(first, now=1.0) is None
    confirmed = mission._confirmed_obstacle(second, now=1.2)

    assert confirmed["distance"] == pytest.approx(2.48)
    assert confirmed["angle"] == pytest.approx(-8.4)


def test_active_obstacle_uses_bbox_or_angle_distance_continuity(task2_module):
    mission = _mission(task2_module)
    mission.active_obstacle_reference = mission._normalize_detection(
        _yellow(distance=2.5, angle=0.0, track_id=None)
    )

    matched = mission._matching_avoidance_obstacle([
        _yellow(
            distance=1.0,
            angle=60.0,
            bbox=[300, 100, 340, 160],
            track_id=None,
        ),
        _yellow(
            distance=2.2,
            angle=4.0,
            bbox=[105, 100, 145, 160],
            track_id=None,
        ),
    ])

    assert matched["distance"] == pytest.approx(2.38)
    assert matched["angle"] == pytest.approx(1.6)


def test_active_obstacle_prefers_exact_track_id(task2_module):
    mission = _mission(task2_module)
    mission.avoiding_track_id = 7
    mission.active_obstacle_reference = mission._normalize_detection(
        _yellow(distance=2.5, angle=0.0, track_id=7)
    )

    matched = mission._matching_avoidance_obstacle([
        _yellow(distance=1.0, angle=0.0, track_id=8),
        _yellow(distance=2.2, angle=3.0, track_id=7),
    ])

    assert matched["track_id"] == 7


def test_two_frames_start_timed_velocity_avoidance_without_gps(
        task2_module,
        monkeypatch,
):
    mission = _mission(task2_module)
    monkeypatch.setattr(task2_module.time, "monotonic", lambda: 10.0)
    task2_module.publish_set_position = lambda *args, **kwargs: pytest.fail(
        "active avoidance must not publish a GPS target"
    )
    mission._navigate_to_gps_target = lambda *args, **kwargs: pytest.fail(
        "active avoidance must not enter normal/course navigation"
    )
    detection = _yellow(distance=2.5, side="left", angle=-10.0)

    mission.update([detection])
    assert mission.state is task2_module.MissionState.NAVIGATING
    assert mission.topics.avoidance_velocity_pub.messages == []

    mission.update([detection])

    assert mission.state is task2_module.MissionState.AVOIDING
    assert mission.avoidance_side == "right"
    assert len(mission.topics.avoidance_velocity_pub.messages) == 1
    velocity = mission.topics.avoidance_velocity_pub.messages[-1]
    assert velocity.linear.x > 0.0
    assert velocity.linear.y > 0.0

    mission.update([_yellow(distance=2.2, side="left", angle=-6.0)])

    assert mission.state is task2_module.MissionState.AVOIDING
    assert len(mission.topics.avoidance_velocity_pub.messages) == 2


def test_timed_avoidance_turns_then_moves_forward_and_resumes_route(
        task2_module,
):
    mission = _mission(task2_module)
    detection = _yellow(distance=2.5, side="left", angle=-10.0)
    assert mission._start_avoidance(detection, now=0.0)
    expected_speed = mission.last_avoidance_speed_m_s

    assert mission._update_active_avoidance([], now=0.1)
    assert mission.avoidance_phase == "right"
    assert mission.avoidance_clear_started_time is None

    assert mission._update_active_avoidance(
        [_yellow(distance=2.2, side="left", angle=-6.0)],
        now=3.9,
    )
    assert mission.avoidance_phase == "right"

    assert mission._update_active_avoidance([], now=4.0)
    assert mission.avoidance_phase == "forward"
    forward = mission.topics.avoidance_velocity_pub.messages[-1]
    assert forward.linear.x == pytest.approx(expected_speed)
    assert forward.linear.y == pytest.approx(0.0, abs=1e-9)

    assert mission._update_active_avoidance([], now=6.9)
    assert mission.state is task2_module.MissionState.AVOIDING
    assert not mission._update_active_avoidance([], now=7.0)
    assert mission.state is task2_module.MissionState.NAVIGATING
    assert mission.avoidance_phase is None


def test_forward_phase_finishes_when_obstacle_is_behind(task2_module):
    mission = _mission(task2_module)
    obstacle = _yellow(distance=2.5, side="left", angle=-10.0)
    assert mission._start_avoidance(obstacle, now=0.0)
    assert mission._update_active_avoidance([obstacle], now=4.0)

    behind = _yellow(distance=1.5, side="left", angle=180.0)
    mission.active_obstacle_reference = mission._normalize_detection(behind)

    assert not mission._update_active_avoidance([behind], now=7.0)
    assert mission.state is task2_module.MissionState.NAVIGATING


def test_uncertain_depth_stops_without_advancing_clear_timer(task2_module):
    mission = _mission(task2_module)
    velocity_commands = []
    task2_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z: velocity_commands.append(
            (linear_x, angular_z)
        )
    )
    assert mission._start_avoidance(_yellow(), now=0.0)

    uncertain = _yellow()
    uncertain.pop("distance")
    mission._update_active_avoidance([uncertain], now=0.6)

    assert mission.state is task2_module.MissionState.AVOIDING
    assert mission.avoidance_clear_started_time is None
    assert velocity_commands[-1] == (0.0, 0.0)


def test_clear_view_resumes_same_main_waypoint_in_same_tick_without_alignment(
        task2_module,
        monkeypatch,
):
    mission = _mission(task2_module)
    published_targets = []
    clock = {"now": 7.0}
    monkeypatch.setattr(
        task2_module.time,
        "monotonic",
        lambda: clock["now"],
    )
    stops = []
    task2_module.stop_vehicle = lambda publisher: stops.append(True)
    task2_module.calculate_gps_distance = lambda *args, **kwargs: 10.0
    task2_module.align_heading_to_gps_target = (
        lambda *args, **kwargs: pytest.fail(
            "post-avoidance navigation must not realign"
        )
    )
    task2_module.publish_set_position = (
        lambda publisher, lat, lon: published_targets.append((lat, lon))
    )
    assert mission._start_avoidance(_yellow(), now=0.0)
    mission.avoidance_phase = "forward"
    mission.avoidance_phase_started_time = 4.0
    mission.avoidance_clear_started_time = 4.0
    mission.update([])

    assert mission.state is task2_module.MissionState.NAVIGATING
    assert published_targets == [
        (
            mission.waypoints[0]["lat"],
            mission.waypoints[0]["lon"],
        ),
    ]
    assert mission.resume_navigation_without_alignment
    assert stops == []


def test_avoidance_does_not_start_above_four_metres(
        task2_module,
        monkeypatch,
):
    mission = _mission(task2_module)
    mission._navigate_to_gps_target = lambda *args, **kwargs: False
    clock = {"now": 1.0}
    monkeypatch.setattr(
        task2_module.time,
        "monotonic",
        lambda: clock["now"],
    )

    for now in (1.0, 1.1):
        clock["now"] = now
        mission.update([_yellow(distance=4.01)])

    assert mission.state is task2_module.MissionState.NAVIGATING


def test_forty_second_timeout_enters_failsafe_and_requests_hold(task2_module):
    mission = _mission(task2_module)
    stopped = []
    hold_requests = []
    task2_module.stop_vehicle = lambda publisher: stopped.append(True)
    mission._request_hold_mode = lambda: hold_requests.append(True)
    assert mission._start_avoidance(_yellow(), now=10.0)

    mission._update_active_avoidance([_yellow()], now=50.0)

    assert mission.state is task2_module.MissionState.FAILSAFE
    assert stopped == [True]
    assert hold_requests == [True]


def test_normal_navigation_publishes_only_main_waypoint(task2_module):
    mission = _mission(task2_module)
    published_targets = []
    task2_module.calculate_gps_distance = lambda *args, **kwargs: 10.0
    task2_module.align_heading_to_gps_target = lambda *args, **kwargs: True
    task2_module.publish_set_position = (
        lambda publisher, lat, lon: published_targets.append((lat, lon))
    )
    reached = mission._navigate_to_gps_target(
        37.95200,
        32.50150,
        "WP1",
        1.0,
    )

    assert reached is False
    assert published_targets == [(37.95200, 32.50150)]


def test_repeated_navigation_keeps_following_main_waypoint(
        task2_module,
        monkeypatch,
):
    mission = _mission(task2_module)
    published_targets = []
    velocity_commands = []
    clock = {"now": 10.0}
    monkeypatch.setattr(
        task2_module.time,
        "monotonic",
        lambda: clock["now"],
    )
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
    )
    clock["now"] = 12.9
    mission._navigate_to_gps_target(
        *main_target,
        "WP1",
        1.0,
    )
    clock["now"] = 13.1
    mission._navigate_to_gps_target(
        *main_target,
        "WP1",
        1.0,
    )

    assert published_targets == [main_target, main_target, main_target]
    assert velocity_commands == []
