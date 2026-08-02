"""Task 1 detection continuity and timed-avoidance regression tests."""

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK1_PATH = REPO_ROOT / "njord" / "missions" / "task1_maneuvering_and_path_finding.py"


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


@pytest.fixture()
def task1_module(monkeypatch):
    rclpy_module = types.ModuleType("rclpy")
    rclpy_node_module = types.ModuleType("rclpy.node")
    rclpy_node_module.Node = type("Node", (), {})
    rclpy_module.node = rclpy_node_module

    geometry_msgs_module = types.ModuleType("geometry_msgs")
    geometry_msgs_msg_module = types.ModuleType("geometry_msgs.msg")

    class Vector3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:
        def __init__(self):
            self.linear = Vector3()
            self.angular = Vector3()

    geometry_msgs_msg_module.Twist = Twist
    geometry_msgs_module.msg = geometry_msgs_msg_module

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
        "geometry_msgs": geometry_msgs_module,
        "geometry_msgs.msg": geometry_msgs_msg_module,
        "mavros_msgs": mavros_module,
        "mavros_msgs.srv": mavros_srv_module,
        "std_msgs": std_msgs_module,
        "std_msgs.msg": std_msgs_msg_module,
        "utils.mavlink_utilities": mavlink_utilities,
        "utils.read_waypoints": read_waypoints,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "task1_timed_avoidance_test_module",
        TASK1_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mission_without_ros(task1_module, heading=0.0):
    mission = task1_module.Task1Maneuvering.__new__(
        task1_module.Task1Maneuvering
    )
    mission.current_lat = 63.4305
    mission.current_lon = 10.3951
    mission.current_heading = heading
    mission.aligned_target_key = None
    mission.avoiding_class = None
    mission.avoiding_track_id = None
    mission.active_obstacle_reference = None
    mission.pending_obstacle = None
    mission.pending_obstacle_time = None
    mission.pending_obstacle_count = 0
    mission.recently_avoided_obstacles = []
    mission.avoidance_phase = None
    mission.avoid_started_time = None
    mission.avoidance_phase_started_time = None
    mission.avoidance_entry_heading = None
    mission.avoidance_clear_since = None
    mission.avoidance_marker_gps = None
    mission.state = task1_module.MissionState.NAVIGATING
    mission.topics = types.SimpleNamespace(
        cmd_vel_pub=FakePublisher(),
        position_target_pub=FakePublisher(),
        avoidance_velocity_pub=FakePublisher(),
    )
    mission.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    return mission


@pytest.mark.parametrize(
    ("obstacle_class", "expected_side", "expected_bearing"),
    (
        ("red_buoys", "starboard", 45.0),
        ("green_buoys", "port", 315.0),
    ),
)
def test_buoy_timed_avoidance_preserves_pass_side(
        task1_module,
        obstacle_class,
        expected_side,
        expected_bearing,
):
    mission = _mission_without_ros(task1_module)

    mission._start_avoidance({
        "class": obstacle_class,
        "distance": 2.5,
        "angle_deg": 0.0,
    }, now=10.0)

    assert mission.state is task1_module.MissionState.AVOIDING
    assert mission.avoidance_phase == expected_side
    assert mission.avoidance_entry_heading == 0.0
    velocity = mission.topics.avoidance_velocity_pub.messages[-1]
    expected_rad = math.radians(expected_bearing)
    assert velocity.linear.x == pytest.approx(
        task1_module.AVOIDANCE_TARGET_SPEED_M_S * math.cos(expected_rad)
    )
    assert velocity.linear.y == pytest.approx(
        task1_module.AVOIDANCE_TARGET_SPEED_M_S * math.sin(expected_rad)
    )
    assert mission.topics.position_target_pub.messages == []


@pytest.mark.parametrize(
    ("obstacle_class", "heading", "expected_side"),
    (
        ("east_buoys", 0.0, "starboard"),
        ("west_buoys", 0.0, "port"),
        ("east_buoys", 180.0, "port"),
        ("west_buoys", 180.0, "starboard"),
    ),
)
def test_cardinal_timed_avoidance_preserves_geographic_side(
        task1_module,
        obstacle_class,
        heading,
        expected_side,
):
    mission = _mission_without_ros(task1_module, heading)

    mission._start_avoidance({
        "class": obstacle_class,
        "distance": 2.5,
        "angle_deg": 0.0,
    }, now=10.0)

    assert mission.avoidance_phase == expected_side
    assert mission.topics.position_target_pub.messages == []


def test_timed_avoidance_turns_then_moves_forward_and_resumes_route(task1_module):
    mission = _mission_without_ros(task1_module)
    obstacle = {
        "class": "red_buoys",
        "distance": 2.5,
        "angle_deg": 0.0,
    }
    mission._start_avoidance(obstacle, now=10.0)

    mission._update_active_avoidance([obstacle], now=13.9)
    assert mission.avoidance_phase == "starboard"

    mission._update_active_avoidance([obstacle], now=14.0)
    assert mission.avoidance_phase == "forward"
    forward_velocity = mission.topics.avoidance_velocity_pub.messages[-1]
    assert forward_velocity.linear.x == pytest.approx(
        task1_module.AVOIDANCE_TARGET_SPEED_M_S
    )
    assert forward_velocity.linear.y == pytest.approx(0.0, abs=1e-9)

    mission._update_active_avoidance([], now=14.1)
    mission._update_active_avoidance([], now=16.9)
    assert mission.state is task1_module.MissionState.AVOIDING

    mission._update_active_avoidance([], now=17.0)
    assert mission.state is task1_module.MissionState.NAVIGATING
    assert mission.avoidance_phase is None
    assert mission.recently_avoided_obstacles
    assert mission.topics.position_target_pub.messages == []


@pytest.mark.parametrize(
    ("phase", "expected_action"),
    (
        ("starboard", "Pass obstacle on starboard"),
        ("port", "Pass obstacle on port"),
        ("forward", "Continue forward until obstacle clears"),
    ),
)
def test_publish_decision_uses_timed_avoidance_phase(
        task1_module,
        phase,
        expected_action,
):
    node = task1_module.Task1Node.__new__(task1_module.Task1Node)
    node.task = types.SimpleNamespace(
        state=task1_module.MissionState.AVOIDING,
        avoidance_phase=phase,
        avoiding_class="red_buoys",
        current_target_index=1,
        waypoints=[{}, {}],
    )
    node.decision_pub = FakePublisher()

    node.publish_decision()

    decision = json.loads(node.decision_pub.messages[-1].data)
    assert decision["action"] == expected_action
    assert decision["reason"] == "red_buoys detected on planned route"


def test_obstacle_behind_completes_after_minimum_forward_leg(task1_module):
    mission = _mission_without_ros(task1_module)
    obstacle = {
        "class": "green_buoys",
        "distance": 2.5,
        "angle_deg": 0.0,
        "track_id": 7,
    }
    mission._start_avoidance(obstacle, now=10.0)
    mission._update_active_avoidance([obstacle], now=14.0)

    behind = dict(obstacle, distance=1.5, angle_deg=180.0)
    mission.active_obstacle_reference = dict(behind)
    mission._update_active_avoidance([behind], now=17.0)

    assert mission.state is task1_module.MissionState.NAVIGATING
    assert mission.topics.position_target_pub.messages == []


def test_timed_avoidance_timeout_enters_failsafe(task1_module):
    mission = _mission_without_ros(task1_module)
    failsafe_requests = []

    def enter_failsafe(reason, request_hold=False):
        failsafe_requests.append((reason, request_hold))
        mission.state = task1_module.MissionState.FAILSAFE

    mission._enter_failsafe = enter_failsafe
    mission._start_avoidance({
        "class": "red_buoys",
        "distance": 2.5,
        "angle_deg": 0.0,
    }, now=10.0)
    mission._update_active_avoidance([], now=50.0)

    assert mission.state is task1_module.MissionState.FAILSAFE
    assert failsafe_requests[-1][1] is True


def test_missing_detection_angle_fails_safe_without_maneuver(task1_module):
    mission = _mission_without_ros(task1_module)
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
    assert mission.topics.avoidance_velocity_pub.messages == []
    assert failsafe_requests[0][1] is True


def test_singular_and_plural_class_aliases_are_normalized(task1_module):
    mission = _mission_without_ros(task1_module)

    singular = mission._normalize_obstacle({
        "class": "red_buoy",
        "confidence": 0.9,
        "distance": "2.0",
    })
    plural = mission._normalize_obstacle({
        "class": "red_buoys",
        "confidence": 0.9,
        "distance": 2.0,
    })

    assert singular["class"] == task1_module.RED_BUOY_CLASS
    assert plural["class"] == task1_module.RED_BUOY_CLASS
    assert singular["distance"] == 2.0


def test_active_obstacle_uses_bbox_or_angle_distance_continuity(task1_module):
    mission = _mission_without_ros(task1_module)
    mission.avoiding_class = task1_module.RED_BUOY_CLASS
    mission.active_obstacle_reference = {
        "class": task1_module.RED_BUOY_CLASS,
        "distance": 2.5,
        "angle_deg": 0.0,
        "bbox": [100, 100, 140, 160],
    }

    matched = mission._matching_avoidance_obstacle([
        {
            "class": "red_buoy",
            "confidence": 0.9,
            "distance": 1.0,
            "angle_deg": 60.0,
            "bbox": [300, 100, 340, 160],
        },
        {
            "class": "red_buoy",
            "confidence": 0.9,
            "distance": 2.2,
            "angle_deg": 4.0,
            "bbox": [105, 100, 145, 160],
        },
    ])

    assert matched["distance"] == pytest.approx(2.38)
    assert matched["angle_deg"] == pytest.approx(1.6)


def test_active_obstacle_prefers_exact_track_id(task1_module):
    mission = _mission_without_ros(task1_module)
    mission.avoiding_class = task1_module.RED_BUOY_CLASS
    mission.avoiding_track_id = 7
    mission.active_obstacle_reference = {
        "class": task1_module.RED_BUOY_CLASS,
        "distance": 2.5,
        "angle_deg": 0.0,
        "track_id": 7,
    }

    matched = mission._matching_avoidance_obstacle([
        {
            "class": "red_buoy",
            "confidence": 0.9,
            "distance": 1.0,
            "angle_deg": 0.0,
            "track_id": 8,
        },
        {
            "class": "red_buoy",
            "confidence": 0.9,
            "distance": 2.2,
            "angle_deg": 3.0,
            "track_id": 7,
        },
    ])

    assert matched["track_id"] == 7


def test_confirmation_applies_ema_to_range_and_angle(task1_module):
    mission = _mission_without_ros(task1_module)
    first = mission._normalize_obstacle({
        "class": "green_buoy",
        "distance": 2.8,
        "Buoy angle: ": -10.0,
        "bbox": [100, 100, 140, 160],
    })
    second = mission._normalize_obstacle({
        "class": "green_buoy",
        "distance": 2.0,
        "Buoy angle: ": -6.0,
        "bbox": [104, 100, 144, 160],
    })

    assert mission._confirmed_obstacle(first, now=1.0) is None
    confirmed = mission._confirmed_obstacle(second, now=1.2)

    assert confirmed["distance"] == pytest.approx(2.48)
    assert mission._detection_angle_deg(confirmed) == pytest.approx(-8.4)


def test_task1_accepts_low_confidence_buoy_within_detection_range(task1_module):
    mission = _mission_without_ros(task1_module)

    detected = mission._nearest_relevant_obstacle([{
        "class": "green_buoy",
        "confidence": 0.25,
        "distance": 4.5,
        "Buoy angle: ": -4.0,
    }], now=1.0)

    assert detected is not None
    assert detected["class"] == task1_module.GREEN_BUOY_CLASS


def test_confirmation_survives_brief_detection_gap(task1_module):
    mission = _mission_without_ros(task1_module)
    obstacle = mission._normalize_obstacle({
        "class": "green_buoy",
        "confidence": 0.25,
        "distance": 4.5,
        "Buoy angle: ": -4.0,
    })

    assert mission._confirmed_obstacle(obstacle, now=1.0) is None
    assert mission._confirmed_obstacle(None, now=1.5) is None
    assert mission._confirmed_obstacle(obstacle, now=2.0) is not None


def test_trackless_recent_marker_is_suppressed_by_position(task1_module):
    mission = _mission_without_ros(task1_module)
    detection = mission._normalize_obstacle({
        "class": "red_buoy",
        "confidence": 0.9,
        "distance": 2.0,
        "angle_deg": 0.0,
    })
    marker = mission._estimated_marker_gps(detection)
    mission.recently_avoided_obstacles = [{
        "class": task1_module.RED_BUOY_CLASS,
        "track_id": None,
        "marker_lat": marker["lat"],
        "marker_lon": marker["lon"],
        "expires_at": 10.0,
    }]

    assert mission._nearest_relevant_obstacle([detection], now=5.0) is None
