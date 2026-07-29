import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK1_PATH = REPO_ROOT / "njord" / "missions" / "task1_maneuvering_and_path_finding.py"


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


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

    spec = importlib.util.spec_from_file_location(
        "task1_dynamic_avoidance_test_module",
        TASK1_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mission_without_ros(task1_module, heading=0.0):
    mission = task1_module.Task1Maneuvering.__new__(
        task1_module.Task1Maneuvering
    )
    mission.logger = FakeLogger()
    mission.topics = types.SimpleNamespace(
        cmd_vel_pub=object(),
        position_target_pub=object(),
    )
    mission.current_lat = 63.4305
    mission.current_lon = 10.3951
    mission.current_heading = heading
    mission.current_target_index = 2
    mission.aligned_target_key = None
    mission.resume_navigation_without_alignment = False
    mission.state = task1_module.MissionState.NAVIGATING
    mission.avoiding_class = None
    mission.avoiding_track_id = None
    mission.active_obstacle_reference = None
    mission.pending_obstacle = None
    mission.pending_obstacle_time = None
    mission.pending_obstacle_count = 0
    mission.avoid_started_time = None
    mission.avoid_clear_started_time = None
    mission.active_pass_side = None
    mission.last_avoidance_linear_x = 0.0
    mission.last_avoidance_angular_z = 0.0
    return mission


def _detection(obstacle_class, distance=3.0, angle=0.0, **extra):
    return {
        "class": obstacle_class,
        "confidence": 0.9,
        "distance": distance,
        "angle_deg": angle,
        **extra,
    }


@pytest.mark.parametrize(
    ("obstacle_class", "heading", "expected_side", "expected_sign"),
    (
        ("red_buoys", 0.0, "starboard", 1),
        ("red_buoys", 90.0, "starboard", 1),
        ("green_buoys", 0.0, "port", -1),
        ("green_buoys", 90.0, "port", -1),
    ),
)
def test_buoy_command_uses_vehicle_relative_class_side(
        task1_module,
        obstacle_class,
        heading,
        expected_side,
        expected_sign,
):
    mission = _mission_without_ros(task1_module, heading)

    command = mission._calculate_avoidance_command(
        _detection(obstacle_class)
    )

    assert command["pass_side"] == expected_side
    assert command["angular_z"] * expected_sign > 0.0


@pytest.mark.parametrize(
    ("obstacle_class", "heading", "expected_forward", "expected_starboard"),
    (
        ("east_buoys", 0.0, 0.0, 2.5),
        ("east_buoys", 90.0, 2.5, 0.0),
        ("east_buoys", 180.0, 0.0, -2.5),
        ("west_buoys", 0.0, 0.0, -2.5),
        ("west_buoys", 90.0, -2.5, 0.0),
        ("west_buoys", 180.0, 0.0, 2.5),
    ),
)
def test_cardinal_offset_preserves_geographic_side(
        task1_module,
        obstacle_class,
        heading,
        expected_forward,
        expected_starboard,
):
    mission = _mission_without_ros(task1_module, heading)

    side, forward_m, starboard_m = mission._pass_side_and_body_offset(
        obstacle_class
    )

    assert side == ("east" if obstacle_class == "east_buoys" else "west")
    assert forward_m == pytest.approx(expected_forward, abs=1e-6)
    assert starboard_m == pytest.approx(expected_starboard, abs=1e-6)


def test_angle_and_depth_change_dynamic_command(task1_module):
    mission = _mission_without_ros(task1_module)

    far = mission._calculate_avoidance_command(
        _detection("red_buoys", distance=4.0, angle=0.0)
    )
    close = mission._calculate_avoidance_command(
        _detection("red_buoys", distance=2.0, angle=0.0)
    )
    left = mission._calculate_avoidance_command(
        _detection("red_buoys", distance=3.0, angle=-20.0)
    )
    right = mission._calculate_avoidance_command(
        _detection("red_buoys", distance=3.0, angle=20.0)
    )

    assert close["angular_z"] > far["angular_z"]
    assert close["linear_x"] < far["linear_x"]
    assert left["angular_z"] < right["angular_z"]


def test_dynamic_avoidance_uses_reduced_linear_speed_range(task1_module):
    mission = _mission_without_ros(task1_module)

    maximum_speed = mission._calculate_avoidance_command(
        _detection("red_buoys", distance=4.0, angle=0.0)
    )
    minimum_speed = mission._calculate_avoidance_command(
        _detection("red_buoys", distance=1.6, angle=45.0)
    )

    assert task1_module.AVOIDANCE_MIN_LINEAR_SPEED == pytest.approx(0.15)
    assert task1_module.AVOIDANCE_MAX_LINEAR_SPEED == pytest.approx(0.4)
    assert 0.15 <= maximum_speed["linear_x"] <= 0.4
    assert minimum_speed["linear_x"] == pytest.approx(0.15)


def test_emergency_distance_stops_forward_motion_and_clamps_turn(task1_module):
    mission = _mission_without_ros(task1_module)

    command = mission._calculate_avoidance_command(
        _detection("red_buoys", distance=1.5, angle=45.0)
    )

    assert command["linear_x"] == 0.0
    assert abs(command["angular_z"]) <= task1_module.AVOIDANCE_MAX_ANGULAR_Z


def test_start_avoidance_publishes_cmd_vel_without_gps_target(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    velocity_commands = []
    gps_targets = []
    monkeypatch.setattr(
        task1_module,
        "publish_cmd_vel",
        lambda _publisher, linear_x, angular_z: velocity_commands.append(
            (linear_x, angular_z)
        ),
    )
    monkeypatch.setattr(
        task1_module,
        "publish_set_position",
        lambda *args, **kwargs: gps_targets.append((args, kwargs)),
    )

    started = mission._start_avoidance(
        _detection("red_buoys", distance=2.5, angle=5.0),
        now=10.0,
    )

    assert started
    assert mission.state is task1_module.MissionState.AVOIDING
    assert mission.active_pass_side == "starboard"
    assert len(velocity_commands) == 1
    assert gps_targets == []


def test_missing_angle_is_ignored_without_starting_maneuver(task1_module):
    mission = _mission_without_ros(task1_module)
    missing_direction = {
        "class": "red_buoys",
        "confidence": 0.9,
        "distance": 2.0,
    }
    missing_depth = {
        "class": "red_buoys",
        "confidence": 0.9,
        "Buoy angle: ": 0.0,
    }

    assert mission._nearest_relevant_obstacle([missing_direction]) is None
    assert not mission._start_avoidance(missing_direction, now=10.0)
    assert mission._nearest_relevant_obstacle([missing_depth]) is None
    assert not mission._start_avoidance(missing_depth, now=10.1)
    assert mission.state is task1_module.MissionState.NAVIGATING


@pytest.mark.parametrize(
    ("side", "expected_angle"),
    (
        ("left", -15.0),
        ("right", 15.0),
        ("center", 0.0),
        ("across", 0.0),
        ("sol", -15.0),
        ("starboard", 15.0),
    ),
)
def test_side_fallback_supplies_missing_angle(
        task1_module,
        side,
        expected_angle,
):
    mission = _mission_without_ros(task1_module)
    detection = {
        "class_name": "red_buoy",
        "conf": 0.9,
        "distance_m": 2.0,
        "Buoy side: ": side,
    }

    normalized = mission._normalize_obstacle(detection)

    assert normalized["angle_deg"] == pytest.approx(expected_angle)


@pytest.mark.parametrize("side_key", ("Buoy side: ", "side", "buoy_side"))
def test_side_fallback_supports_all_side_field_names(
        task1_module,
        side_key,
):
    mission = _mission_without_ros(task1_module)
    detection = {
        "label": "green_buoys",
        "confidence": 0.9,
        "distance_m": 2.0,
        side_key: "left",
    }

    normalized = mission._normalize_obstacle(detection)

    assert normalized["angle_deg"] == pytest.approx(-15.0)


def test_real_angle_takes_precedence_over_side_fallback(task1_module):
    mission = _mission_without_ros(task1_module)
    normalized = mission._normalize_obstacle({
        "class": "red_buoys",
        "confidence": 0.9,
        "distance": 2.0,
        "angle_from_center": 7.5,
        "side": "left",
    })

    assert normalized["angle_deg"] == pytest.approx(7.5)


def test_side_only_detection_confirms_and_starts_avoidance(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    commands = []
    monkeypatch.setattr(
        task1_module,
        "publish_cmd_vel",
        lambda _publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z)),
    )
    detection = {
        "class": "red_buoys",
        "confidence": 0.9,
        "distance": 2.5,
        "Buoy side: ": "right",
    }

    first = mission._nearest_relevant_obstacle([detection])
    assert mission._confirmed_obstacle(first, now=1.0) is None
    second = mission._nearest_relevant_obstacle([detection])
    confirmed = mission._confirmed_obstacle(second, now=1.1)

    assert confirmed is not None
    assert mission._start_avoidance(confirmed, now=1.1)
    assert mission.state is task1_module.MissionState.AVOIDING
    assert commands[-1][0] > 0.0
    assert commands[-1][1] > 0.0


def test_only_current_four_classes_trigger_avoidance(task1_module):
    mission = _mission_without_ros(task1_module)

    accepted = [
        mission._normalize_obstacle(_detection(name))
        for name in (
            "red_buoys",
            "green_buoys",
            "east_buoys",
            "west_buoys",
        )
    ]

    assert all(item is not None for item in accepted)
    assert mission._normalize_obstacle(_detection("north_buoys")) is None
    assert mission._normalize_obstacle(_detection("south_buoys")) is None


def test_singular_and_plural_class_aliases_are_normalized(task1_module):
    mission = _mission_without_ros(task1_module)

    singular = mission._normalize_obstacle(
        _detection("red_buoy", distance="2.0")
    )
    plural = mission._normalize_obstacle(
        _detection("red_buoys", distance=2.0)
    )

    assert singular["class"] == task1_module.RED_BUOY_CLASS
    assert plural["class"] == task1_module.RED_BUOY_CLASS
    assert singular["distance"] == 2.0


def test_confirmation_applies_ema_to_range_and_angle(task1_module):
    mission = _mission_without_ros(task1_module)
    first = mission._normalize_obstacle(
        _detection(
            "green_buoy",
            distance=2.8,
            angle=-10.0,
            bbox=[100, 100, 140, 160],
        )
    )
    second = mission._normalize_obstacle(
        _detection(
            "green_buoy",
            distance=2.0,
            angle=-6.0,
            bbox=[104, 100, 144, 160],
        )
    )

    assert mission._confirmed_obstacle(first, now=1.0) is None
    confirmed = mission._confirmed_obstacle(second, now=1.2)

    assert confirmed["distance"] == pytest.approx(2.48)
    assert mission._detection_angle_deg(confirmed) == pytest.approx(-8.4)


def test_active_obstacle_prefers_exact_track_id(task1_module):
    mission = _mission_without_ros(task1_module)
    mission.avoiding_class = task1_module.RED_BUOY_CLASS
    mission.avoiding_track_id = 7
    mission.active_obstacle_reference = _detection(
        "red_buoys",
        distance=2.5,
        track_id=7,
    )

    matched = mission._matching_avoidance_obstacle(
        [
            _detection("red_buoy", distance=1.0, track_id=8),
            _detection(
                "red_buoy",
                distance=2.2,
                angle=3.0,
                track_id=7,
            ),
        ]
    )

    assert matched["track_id"] == 7


def test_active_obstacle_accepts_new_track_id_with_bbox_continuity(
        task1_module,
):
    mission = _mission_without_ros(task1_module)
    mission.avoiding_class = task1_module.RED_BUOY_CLASS
    mission.avoiding_track_id = 7
    mission.active_obstacle_reference = _detection(
        "red_buoys",
        distance=2.5,
        angle=2.0,
        track_id=7,
        bbox=[100, 100, 200, 220],
    )

    matched = mission._matching_avoidance_obstacle([
        _detection(
            "red_buoys",
            distance=2.3,
            angle=3.0,
            track_id=19,
            bbox=[108, 104, 208, 224],
        ),
    ])

    assert matched is not None
    assert matched["track_id"] == 19
    assert mission.avoiding_track_id == 19


def test_changed_track_id_rejects_unrelated_obstacle(task1_module):
    mission = _mission_without_ros(task1_module)
    mission.avoiding_class = task1_module.RED_BUOY_CLASS
    mission.avoiding_track_id = 7
    mission.active_obstacle_reference = _detection(
        "red_buoys",
        distance=2.5,
        angle=0.0,
        track_id=7,
        bbox=[100, 100, 200, 220],
    )

    matched = mission._matching_avoidance_obstacle([
        _detection(
            "red_buoys",
            distance=4.8,
            angle=60.0,
            track_id=19,
            bbox=[400, 100, 500, 220],
        ),
    ])

    assert matched is None
    assert mission.avoiding_track_id == 7


def test_short_detection_loss_republishes_then_resumes_without_stop(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    published = []
    stops = []
    monkeypatch.setattr(
        task1_module,
        "publish_cmd_vel",
        lambda _publisher, linear_x, angular_z: published.append(
            (linear_x, angular_z)
        ),
    )
    monkeypatch.setattr(
        task1_module,
        "stop_vehicle",
        lambda _publisher: stops.append(True),
    )
    mission._start_avoidance(
        _detection("red_buoys", distance=2.5, angle=0.0),
        now=10.0,
    )
    initial_command = published[-1]

    assert mission._update_active_avoidance([], now=10.1)
    assert mission.state is task1_module.MissionState.AVOIDING
    assert published[-1] == initial_command

    assert mission._update_active_avoidance([], now=10.29)
    assert mission.state is task1_module.MissionState.AVOIDING

    assert not mission._update_active_avoidance([], now=10.31)
    assert mission.state is task1_module.MissionState.NAVIGATING
    assert mission.current_target_index == 2
    assert mission.resume_navigation_without_alignment
    assert stops == []


def test_clear_view_resumes_same_waypoint_in_same_tick_without_alignment(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    target = {"lat": 63.4310, "lon": 10.3960}
    mission.waypoints = [{}, {}, target]
    mission.waypoint_tolerance = 1.0
    mission._prepare_update = lambda: True
    published_targets = []
    stops = []
    monkeypatch.setattr(task1_module.time, "monotonic", lambda: 10.31)
    monkeypatch.setattr(
        task1_module,
        "calculate_gps_distance",
        lambda *args, **kwargs: 10.0,
    )
    monkeypatch.setattr(
        task1_module,
        "publish_cmd_vel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        task1_module,
        "stop_vehicle",
        lambda _publisher: stops.append(True),
    )
    monkeypatch.setattr(
        task1_module,
        "align_heading_to_gps_target",
        lambda *args, **kwargs: pytest.fail(
            "post-avoidance navigation must not realign"
        ),
    )
    monkeypatch.setattr(
        task1_module,
        "publish_set_position",
        lambda _publisher, lat, lon: published_targets.append((lat, lon)),
    )
    assert mission._start_avoidance(
        _detection("red_buoys", distance=2.5, angle=0.0),
        now=10.0,
    )
    mission.avoid_clear_started_time = 10.1

    mission.update([])
    mission.update([])

    assert mission.state is task1_module.MissionState.NAVIGATING
    assert mission.current_target_index == 2
    assert mission.resume_navigation_without_alignment
    assert published_targets == [
        (target["lat"], target["lon"]),
        (target["lat"], target["lon"]),
    ]
    assert stops == []


def test_avoidance_starts_only_at_two_and_a_half_metres(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    mission.waypoints = [{}, {}, {"lat": 63.4310, "lon": 10.3960}]
    mission.waypoint_tolerance = 1.0
    mission._prepare_update = lambda: True
    mission._set_position_to_gps_target = lambda *args, **kwargs: False
    clock = {"now": 1.0}
    monkeypatch.setattr(
        task1_module.time,
        "monotonic",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        task1_module,
        "publish_cmd_vel",
        lambda *args, **kwargs: None,
    )

    for now in (1.0, 1.1):
        clock["now"] = now
        mission.update([
            _detection("red_buoys", distance=2.51, angle=0.0)
        ])
    assert mission.state is task1_module.MissionState.NAVIGATING

    for now in (1.2, 1.3):
        clock["now"] = now
        mission.update([
            _detection("red_buoys", distance=2.5, angle=0.0)
        ])

    assert task1_module.AVOIDANCE_START_DISTANCE_M == 2.5
    assert mission.state is task1_module.MissionState.AVOIDING


def test_active_invalid_data_uses_short_loss_grace(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    published = []
    monkeypatch.setattr(
        task1_module,
        "publish_cmd_vel",
        lambda _publisher, linear_x, angular_z:
        published.append((linear_x, angular_z)),
    )
    mission._start_avoidance(
        _detection(
            "red_buoys",
            distance=2.5,
            angle=0.0,
            track_id=3,
        ),
        now=10.0,
    )
    initial_command = published[-1]
    missing_depth = {
        "class": "red_buoys",
        "confidence": 0.9,
        "distance": None,
        "Buoy side: ": "right",
        "track_id": 3,
    }

    mission._update_active_avoidance([missing_depth], now=10.2)

    assert mission.state is task1_module.MissionState.AVOIDING
    assert mission.avoid_clear_started_time == 10.2
    assert published[-1] == initial_command


def test_reacquired_obstacle_resets_clear_confirmation(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    monkeypatch.setattr(task1_module, "publish_cmd_vel", lambda *args, **kwargs: None)
    mission._start_avoidance(
        _detection("red_buoys", distance=2.5, angle=0.0, track_id=3),
        now=10.0,
    )

    mission._update_active_avoidance([], now=10.2)
    assert mission.avoid_clear_started_time == 10.2

    mission._update_active_avoidance(
        [_detection("red_buoys", distance=2.3, angle=3.0, track_id=3)],
        now=10.4,
    )
    assert mission.avoid_clear_started_time is None


def test_persistent_obstacle_timeout_enters_failsafe_hold(
        task1_module,
        monkeypatch,
):
    mission = _mission_without_ros(task1_module)
    failsafe_requests = []
    stops = []
    monkeypatch.setattr(task1_module, "publish_cmd_vel", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        task1_module,
        "stop_vehicle",
        lambda _publisher: stops.append(True),
    )

    def enter_failsafe(reason, request_hold=False):
        failsafe_requests.append((reason, request_hold))
        mission.state = task1_module.MissionState.FAILSAFE

    mission._enter_failsafe = enter_failsafe
    mission._start_avoidance(
        _detection("red_buoys", distance=2.5, angle=0.0),
        now=10.0,
    )

    assert mission._update_active_avoidance(
        [_detection("red_buoys", distance=2.0, angle=0.0)],
        now=18.0,
    )
    assert mission.state is task1_module.MissionState.FAILSAFE
    assert failsafe_requests[0][1] is True
    assert stops == [True]


def test_decision_payload_reports_dynamic_class_side_and_command(task1_module):
    node = task1_module.Task1Node.__new__(task1_module.Task1Node)
    node.task = types.SimpleNamespace(
        state=task1_module.MissionState.AVOIDING,
        active_pass_side="east",
        avoiding_class="east_buoys",
        last_avoidance_linear_x=0.31,
        last_avoidance_angular_z=0.42,
        current_target_index=2,
        waypoints=[{}, {}, {}],
    )
    node.decision_pub = FakePublisher()

    task1_module.Task1Node.publish_decision(node)

    payload = json.loads(node.decision_pub.messages[-1].data)
    assert payload["action"] == "Dynamic camera pass on east"
    assert "east_buoys" in payload["reason"]
    assert "linear=0.31" in payload["reason"]
    assert "angular=0.42" in payload["reason"]
