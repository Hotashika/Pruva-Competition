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
    mission.confirmation_candidate_key = None
    mission.confirmation_candidate_obstacle = None
    mission.confirmation_count = 0
    mission.last_confirmation_frame_token = None
    mission.synthetic_frame_token = 0
    mission.minimum_obstacle_distance = None
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


@pytest.mark.parametrize("heading", (0.0, 90.0, 180.0, 270.0))
@pytest.mark.parametrize("detection_angle", (-30.0, 0.0, 30.0))
@pytest.mark.parametrize("obstacle_class", ("east_buoys", "west_buoys"))
def test_cardinal_entry_and_exit_segments_keep_minimum_clearance(
        task1_module,
        heading,
        detection_angle,
        obstacle_class,
):
    mission = _mission_without_ros(task1_module, heading=heading)
    targets = mission._create_cardinal_pass_targets({
        "class": obstacle_class,
        "distance": task1_module.CARDINAL_ENTER_DIST_M,
        "confidence": 0.9,
        "Buoy angle: ": detection_angle,
    })

    assert [target["phase"] for target in targets] == ["entry", "exit"]
    marker = mission._gps_offset_m(
        mission.current_lat,
        mission.current_lon,
        targets[0]["marker_lat"],
        targets[0]["marker_lon"],
    )
    entry = mission._gps_offset_m(
        mission.current_lat,
        mission.current_lon,
        targets[0]["lat"],
        targets[0]["lon"],
    )
    exit_target = mission._gps_offset_m(
        mission.current_lat,
        mission.current_lon,
        targets[1]["lat"],
        targets[1]["lon"],
    )

    entry_clearance = mission._point_to_segment_distance(
        marker,
        (0.0, 0.0),
        entry,
    )
    exit_clearance = mission._point_to_segment_distance(
        marker,
        entry,
        exit_target,
    )
    assert (
        entry_clearance + 0.01
        >= task1_module.CARDINAL_MIN_ROUTE_CLEARANCE_M
    )
    assert (
        exit_clearance + 0.01
        >= task1_module.CARDINAL_MIN_ROUTE_CLEARANCE_M
    )


def test_obstacle_requires_distinct_confirmed_frames(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    detection = {
        "class": "red_buoys",
        "distance": 2.5,
        "confidence": 0.9,
        "Buoy angle: ": 0.0,
    }

    assert mission._confirmed_nearest_obstacle([detection], frame_token=10) is None
    assert mission._confirmed_nearest_obstacle([detection], frame_token=10) is None
    assert mission._confirmed_nearest_obstacle([detection], frame_token=11) is None
    assert mission._confirmed_nearest_obstacle([detection], frame_token=12) == detection


def test_low_confidence_obstacle_does_not_confirm(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    detection = {
        "class": "red_buoys",
        "distance": 2.0,
        "confidence": task1_module.AVOID_MIN_CONFIDENCE - 0.01,
    }

    for frame_token in range(3):
        assert mission._confirmed_nearest_obstacle(
            [detection],
            frame_token=frame_token,
        ) is None


def test_same_class_detection_must_still_match_angle_gate(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    left_detection = {
        "class": "red_buoys",
        "distance": 2.5,
        "confidence": 0.9,
        "angle_deg": -30.0,
    }
    right_detection = dict(left_detection, angle_deg=30.0)

    assert mission._confirmed_nearest_obstacle(
        [left_detection], frame_token=1
    ) is None
    assert mission._confirmed_nearest_obstacle(
        [right_detection], frame_token=2
    ) is None
    assert mission.confirmation_count == 1
    assert mission._confirmed_nearest_obstacle(
        [right_detection], frame_token=3
    ) is None
    assert mission._confirmed_nearest_obstacle(
        [right_detection], frame_token=4
    ) == right_detection


@pytest.mark.parametrize(
    ("obstacle_class", "expected_turn_sign", "desired_angle"),
    (
        ("red_buoys", 1.0, -35.0),
        ("green_buoys", -1.0, 35.0),
    ),
)
def test_buoy_command_reacts_to_angle_and_distance(
        task1_module,
        obstacle_class,
        expected_turn_sign,
        desired_angle,
        monkeypatch,
):
    commands = []
    monkeypatch.setattr(
        task1_module,
        "publish_cmd_vel",
        lambda publisher, linear_x, angular_z: commands.append(
            (float(linear_x), float(angular_z))
        ),
    )
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.topics = types.SimpleNamespace(cmd_vel_pub=object())
    mission.avoiding_class = obstacle_class
    mission.avoid_turn_direction = expected_turn_sign
    mission.filtered_obstacle_angle = 0.0
    mission.filtered_obstacle_distance = task1_module.AVOID_ENTER_DIST_M

    mission._publish_avoidance_maneuver()
    far_linear, initial_turn = commands[-1]
    assert far_linear > 0.0
    assert initial_turn * expected_turn_sign > 0.0

    mission.filtered_obstacle_angle = desired_angle
    mission._publish_avoidance_maneuver()
    assert commands[-1][1] == pytest.approx(0.0)

    mission.filtered_obstacle_distance = task1_module.AVOID_STOP_DIST_M - 0.1
    mission._publish_avoidance_maneuver()
    assert commands[-1][0] == 0.0


def test_close_obstacle_is_not_clear_based_on_angle_only(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.avoid_turn_direction = 1.0

    assert not mission._is_avoidance_clear({
        "distance": task1_module.AVOID_CLEAR_MIN_DIST_M - 0.1,
        "angle_deg": -45.0,
    })
    assert mission._is_avoidance_clear({
        "distance": task1_module.AVOID_CLEAR_MIN_DIST_M + 0.1,
        "angle_deg": -45.0,
    })


def test_obstacle_must_be_receding_before_it_is_clear(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.avoid_turn_direction = 1.0
    mission.minimum_obstacle_distance = 2.0

    assert not mission._is_avoidance_clear({
        "distance": 2.4,
        "angle_deg": -45.0,
    })
    assert mission._is_avoidance_clear({
        "distance": 2.6,
        "angle_deg": -45.0,
    })


def test_avoidance_timeout_holds_if_obstacle_is_still_close(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.topics = types.SimpleNamespace(cmd_vel_pub=object())
    mission.avoiding_class = "red_buoys"
    mission.avoiding_track_id = None
    mission.avoid_started_time = 0.0
    mission.avoid_clear_started_time = None
    mission.avoid_turn_direction = 1.0
    mission.filtered_obstacle_angle = 0.0
    mission.filtered_obstacle_distance = 2.0
    mission.minimum_obstacle_distance = 2.0
    mission.cardinal_pass_target = None
    failsafe_calls = []
    mission._enter_failsafe = lambda reason, request_hold=False: failsafe_calls.append(
        (reason, request_hold)
    )

    consumed = mission._update_active_avoidance(
        [{
            "class": "red_buoys",
            "distance": 2.0,
            "confidence": 0.9,
            "angle_deg": 0.0,
        }],
        now=task1_module.AVOID_MANEUVER_MAX_SEC + 0.1,
    )

    assert consumed
    assert failsafe_calls
    assert failsafe_calls[-1][1] is True


def test_lost_obstacle_resumes_navigation_after_confirmation(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    mission.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    mission.topics = types.SimpleNamespace(cmd_vel_pub=object())
    mission.state = task1_module.MissionState.AVOIDING
    mission.avoiding_class = "red_buoys"
    mission.avoiding_track_id = None
    mission.avoid_started_time = 0.0
    mission.avoid_clear_started_time = 1.0
    mission.avoid_turn_direction = 1.0
    mission.filtered_obstacle_angle = -45.0
    mission.filtered_obstacle_distance = 3.0
    mission.minimum_obstacle_distance = 2.0
    mission.cardinal_marker_estimate = None
    mission.cardinal_route_marker = None
    mission.cardinal_pass_targets = []
    mission.cardinal_target_index = 0
    mission.cardinal_pass_target = None
    mission.aligned_target_key = ("old",)

    consumed = mission._update_active_avoidance(
        [],
        now=1.0 + task1_module.AVOID_LOST_CONFIRM_SEC,
    )

    assert not consumed
    assert mission.state == task1_module.MissionState.NAVIGATING
    assert mission.avoiding_class is None


def test_cardinal_route_advances_entry_then_exit(task1_module):
    mission = _mission_without_ros(task1_module, heading=0.0)
    targets = mission._create_cardinal_pass_targets({
        "class": "east_buoys",
        "distance": task1_module.CARDINAL_ENTER_DIST_M,
        "confidence": 0.9,
        "angle_deg": 0.0,
    })
    mission.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    mission.topics = types.SimpleNamespace(cmd_vel_pub=object())
    mission.state = task1_module.MissionState.AVOIDING
    mission.avoiding_class = "east_buoys"
    mission.avoiding_track_id = None
    mission.avoid_started_time = 0.0
    mission.avoid_clear_started_time = None
    mission.avoid_turn_direction = 1.0
    mission.filtered_obstacle_angle = 0.0
    mission.filtered_obstacle_distance = task1_module.CARDINAL_ENTER_DIST_M
    mission.minimum_obstacle_distance = task1_module.CARDINAL_ENTER_DIST_M
    mission.cardinal_pass_targets = targets
    mission.cardinal_target_index = 0
    mission.cardinal_pass_target = targets[0]
    mission.cardinal_marker_estimate = {
        "lat": targets[0]["marker_lat"],
        "lon": targets[0]["marker_lon"],
    }
    mission.cardinal_route_marker = dict(mission.cardinal_marker_estimate)
    mission.aligned_target_key = ("old",)
    mission._set_position_to_gps_target = lambda *args, **kwargs: True

    mission.current_lat = targets[0]["lat"]
    mission.current_lon = targets[0]["lon"]
    assert mission._update_cardinal_pass([], now=1.0)
    assert mission.cardinal_target_index == 1
    assert mission.cardinal_pass_target["phase"] == "exit"

    mission.current_lat = targets[1]["lat"]
    mission.current_lon = targets[1]["lon"]
    assert mission._update_cardinal_pass([], now=2.0)
    assert mission.state == task1_module.MissionState.NAVIGATING
    assert mission.cardinal_pass_target is None
