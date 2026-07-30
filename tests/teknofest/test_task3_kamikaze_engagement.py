import importlib.util
import json
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


class _ModeRequest:
    def __init__(self):
        self.base_mode = 0
        self.custom_mode = ""


class _ModeFuture:
    def __init__(self, mode_sent=True, done=True, exception=None):
        self.mode_sent = mode_sent
        self._done = done
        self._exception = exception
        self.cancelled = False

    def done(self):
        return self._done

    def result(self):
        if self._exception is not None:
            raise self._exception
        return types.SimpleNamespace(mode_sent=self.mode_sent)

    def cancel(self):
        self.cancelled = True
        self._done = True
        return True


class _ModeClient:
    def __init__(self, future_factory=None):
        self.requests = []
        self.future_factory = future_factory or (lambda: _ModeFuture())

    def call_async(self, request):
        self.requests.append((request.base_mode, request.custom_mode))
        return self.future_factory()


class _MessagePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


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

    mavros_msgs = types.ModuleType("mavros_msgs")
    mavros_msgs_srv = types.ModuleType("mavros_msgs.srv")
    mavros_msgs_srv.SetMode = type(
        "SetMode",
        (),
        {"Request": _ModeRequest},
    )
    mavros_msgs.srv = mavros_msgs_srv

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
        "mavros_msgs": mavros_msgs,
        "mavros_msgs.srv": mavros_msgs_srv,
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


def _mission(task3_module, mode_client=None, **config_overrides):
    config_values = {
        "ram_duration_sec": 0.2,
        "post_impact_forward_duration_sec": 0.2,
        "impact_return_timeout_sec": 1.0,
        "mode_transition_timeout_sec": 0.5,
        "gps_timeout_sec": 100.0,
        "heading_timeout_sec": 100.0,
        "bridge_state_timeout_sec": 100.0,
        "mission_timeout_sec": 100.0,
    }
    config_values.update(config_overrides)
    node = types.SimpleNamespace(get_logger=_logger)
    topics = types.SimpleNamespace(
        cmd_vel_pub="cmd_vel",
        position_target_pub="position_target",
        mission_state_pub=_MessagePublisher(),
    )
    clients = types.SimpleNamespace(
        set_mode_client=mode_client or _ModeClient()
    )
    mission = task3_module.Task3KamikazeEngagement(
        node,
        topics,
        mission_clients=clients,
        config=task3_module.Task3Config(**config_values),
    )
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "LOITER"},
        now=0.0,
    )
    mission.reset_for_entry(37.95125, 32.50090, 15.0, now=0.0)
    mission.update([], now=0.0)
    assert mission.state is task3_module.MissionState.WAIT_GUIDED_SEARCH
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=0.0,
    )
    mission.update([], now=0.0)
    assert mission.state is task3_module.MissionState.SEARCH
    return mission


def _target(
        distance=3.0,
        angle=0.0,
        class_name="red_buoy",
        confidence=0.9,
        **overrides,
):
    detection = {
        "class": class_name,
        "confidence": confidence,
        "distance": distance,
        "bbox": [500, 200, 650, 500],
        "track_id": 7,
        "Buoy angle: ": angle,
        "Buoy side: ": "across",
    }
    detection.update(overrides)
    return detection


def _enter_ram(mission, task3_module, now=0.1, angle=0.0):
    detections = [_target(distance=8.0, angle=angle)]
    for index in range(mission.config.confirmation_required):
        mission.update(
            detections,
            now=now + index * 0.1,
            vision_frame_id=100 + index,
        )
        if index < mission.config.confirmation_required - 1:
            assert mission.state is task3_module.MissionState.ATTACK_CONFIRM
    assert mission.state is task3_module.MissionState.RAM
    return now + (
        mission.config.confirmation_required - 1
    ) * 0.1


def test_target_classes_annotation_is_python38_compatible():
    source = TASK3_PATH.read_text(encoding="utf-8")

    assert "target_classes: Tuple[str, ...]" in source
    assert "target_classes: tuple[" not in source


def test_task3_node_initializes_mission_data_recorder():
    source = TASK3_PATH.read_text(encoding="utf-8")

    assert (
        "from teknofest.missions.utils.mission_data_recorder "
        "import MissionDataRecorder"
    ) in source
    assert (
        "self.data_recorder = MissionDataRecorder(self, ACTIVE_TASK_NAME)"
        in source
    )


def test_task3_node_preserves_vision_frame_id(task3_module):
    node = task3_module.Task3Node.__new__(task3_module.Task3Node)
    node.current_detections = []
    node.current_detection_frame_id = None
    node.last_detection_time = None
    node.get_logger = _logger
    message = types.SimpleNamespace(
        data=json.dumps({
            "frame_id": 321,
            "detections": [_target()],
        })
    )

    node.vision_callback(message)

    assert node.current_detection_frame_id == 321
    assert node.current_detections == [_target()]
    assert node.last_detection_time is not None


@pytest.mark.parametrize(
    ("class_field", "class_name"),
    [
        ("class", "red_buoy"),
        ("class_name", "red_buoys"),
        ("label", "orange_buoy"),
        ("class", "orange_buoys"),
    ],
)
def test_target_classes_accept_red_and_orange_labels(
        task3_module,
        class_field,
        class_name,
):
    mission = _mission(task3_module)
    detection = {
        class_field: class_name,
        "conf": 0.91,
        "distance_m": 8.0,
        "angle_from_center": -4.0,
        "bbox": [500, 200, 650, 500],
    }

    target = mission._select_target([detection])

    assert target["class"] == class_name
    assert target["distance"] == pytest.approx(8.0)
    assert target["angle"] == pytest.approx(-4.0)


@pytest.mark.parametrize("class_name", ["green_buoy", "yellow_buoys"])
def test_target_classes_reject_other_buoy_labels(task3_module, class_name):
    mission = _mission(task3_module)

    assert mission._select_target([_target(class_name=class_name)]) is None


def test_six_distinct_consistent_frames_start_ram_at_low_confirm_speed(
        task3_module,
):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    commands.clear()
    target = [_target(distance=8.0, angle=10.0)]

    mission.update(target, now=0.1, vision_frame_id=100)

    assert mission.state is task3_module.MissionState.ATTACK_CONFIRM
    assert mission.config.attack_confirm_speed == pytest.approx(0.15)
    assert commands[-1][0] == pytest.approx(0.15)
    assert commands[-1][1] > 0.0

    for now in (0.2, 0.3, 0.4):
        mission.update(target, now=now, vision_frame_id=100)
        assert mission.state is task3_module.MissionState.ATTACK_CONFIRM
        assert len(mission.confirmation_samples) == 1
        assert commands[-1][0] == pytest.approx(0.15)

    for frame_id, now in zip(range(101, 105), (0.5, 0.6, 0.7, 0.8)):
        mission.update(target, now=now, vision_frame_id=frame_id)
        assert mission.state is task3_module.MissionState.ATTACK_CONFIRM

    assert len(mission.confirmation_samples) == 5
    mission.update(target, now=0.9, vision_frame_id=105)
    assert mission.state is task3_module.MissionState.RAM
    assert commands[-1][0] == pytest.approx(mission.config.ram_speed)
    assert commands[-1][1] > 0.0


def test_single_frame_disappearance_resumes_search(task3_module):
    mission = _mission(task3_module)

    mission.update(
        [_target(distance=8.0)],
        now=0.1,
        vision_frame_id=1,
    )
    mission.update([], now=0.2, vision_frame_id=2)

    assert mission.state is task3_module.MissionState.SEARCH
    assert mission.impact_count == 0
    assert mission.search_controller.base_heading == pytest.approx(15.0)
    assert (
        mission.search_controller.phase
        is task3_module.SearchPhase.RETURN_TO_BASE_HEADING
    )


def test_false_target_turn_cannot_reanchor_search_direction(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    mission.update(
        [_target(distance=8.0, angle=20.0)],
        now=0.1,
        vision_frame_id=1,
    )
    mission.update_heading(70.0, now=0.2)

    mission.update([], now=0.2, vision_frame_id=2)

    assert mission.state is task3_module.MissionState.SEARCH
    assert mission.search_controller.base_heading == pytest.approx(15.0)
    assert (
        mission.search_controller.phase
        is task3_module.SearchPhase.RETURN_TO_BASE_HEADING
    )

    commands.clear()
    mission.update([], now=0.3)

    assert commands[-1][0] == pytest.approx(0.0)
    assert commands[-1][1] < 0.0


@pytest.mark.parametrize("invalid_heading", [float("nan"), "invalid", None])
def test_invalid_heading_does_not_replace_last_valid_direction(
        task3_module,
        invalid_heading,
):
    mission = _mission(task3_module)
    last_heading_time = mission.last_heading_time

    accepted = mission.update_heading(invalid_heading, now=0.5)

    assert accepted is False
    assert mission.current_heading == pytest.approx(15.0)
    assert mission.last_heading_time == pytest.approx(last_heading_time)


def test_state_topic_reports_transitions_and_periodic_heartbeat(task3_module):
    mission = _mission(task3_module)
    publisher = mission.topics.mission_state_pub

    assert publisher.messages[:2] == ["WAIT_GUIDED_SEARCH", "SEARCH"]

    published_count = len(publisher.messages)
    mission.update([], now=0.5)
    assert len(publisher.messages) == published_count

    mission.update([], now=1.0)
    assert publisher.messages[-1] == "SEARCH"
    assert len(publisher.messages) == published_count + 1

    mission.update([_target()], now=1.1)
    assert publisher.messages[-1] == "ATTACK_CONFIRM"


def test_direct_attack_has_no_legacy_approach_states_or_config(task3_module):
    state_names = {state.name for state in task3_module.MissionState}
    defaults = task3_module.Task3Config()

    assert {
        "ALIGN",
        "APPROACH",
        "FINAL_CONFIRM",
        "CONTACT_HOLD",
        "RETREAT",
        "REACQUIRE",
    }.isdisjoint(state_names)
    assert not hasattr(defaults, "approach_distance_required")
    assert defaults.confirmation_required == 6


def test_ram_steers_while_target_visible_and_continues_straight_if_lost(
        task3_module,
):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    now = _enter_ram(mission, task3_module, angle=12.0)
    commands.clear()

    mission.update([_target(distance=4.0, angle=8.0)], now=now + 0.05)
    assert commands[-1][0] == pytest.approx(mission.config.ram_speed)
    assert commands[-1][1] > 0.0

    mission.update([], now=now + 0.10)
    assert commands[-1] == pytest.approx((mission.config.ram_speed, 0.0))


def test_ram_end_saves_gps_and_immediately_advances(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    now = _enter_ram(mission, task3_module)
    mission.update_gps(37.95126, 32.50091, now=now + 0.2)
    mission.update_heading(15.0, now=now + 0.2)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=now + 0.2,
    )
    commands.clear()

    impact_time = now + mission.config.ram_duration_sec + 0.01
    mission.update([], now=impact_time)

    assert mission.state is task3_module.MissionState.POST_IMPACT_ADVANCE
    assert mission.impact_count == 1
    assert mission.impact_target_gps["lat"] == pytest.approx(37.95126)
    assert mission.impact_target_gps["lon"] == pytest.approx(32.50091)
    assert mission.impact_target_gps["recorded_at"] == pytest.approx(
        impact_time
    )
    assert commands[-1] == pytest.approx(
        (mission.config.post_impact_forward_speed, 0.0)
    )


def test_post_impact_advance_returns_with_global_gps_target(task3_module):
    positions = []
    task3_module.publish_set_position = (
        lambda publisher, lat, lon:
        positions.append((publisher, lat, lon))
    )
    mission = _mission(task3_module)
    mission.impact_count = 1
    mission.impact_target_gps = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
        "source": "initial_ram",
    }
    mission.state = task3_module.MissionState.POST_IMPACT_ADVANCE
    mission.state_started_at = 1.0

    mission.update(
        [],
        now=1.0 + mission.config.post_impact_forward_duration_sec + 0.01,
        vision_fresh=False,
    )

    assert mission.state is task3_module.MissionState.RETURN_TO_IMPACT
    assert positions[-1] == pytest.approx(
        ("position_target", 37.95125, 32.50090)
    )


def test_return_arrival_counts_next_impact_without_overwriting_anchor(
        task3_module,
):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)
    anchor = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
        "source": "initial_ram",
    }
    mission.impact_count = 1
    mission.impact_target_gps = dict(anchor)
    mission.impact_return_departed = True
    mission.state = task3_module.MissionState.RETURN_TO_IMPACT
    mission.state_started_at = 1.0
    mission.update_gps(anchor["lat"], anchor["lon"], now=1.1)
    mission.update_heading(180.0, now=1.1)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=1.1,
    )

    mission.update([], now=1.1, vision_fresh=False)

    assert mission.impact_count == 2
    assert mission.state is task3_module.MissionState.POST_IMPACT_ADVANCE
    assert mission.impact_target_gps == anchor
    assert mission.impact_events[-1]["source"] == "gps_return"
    assert commands[-1] == pytest.approx(
        (mission.config.post_impact_forward_speed, 0.0)
    )


def test_three_impacts_finish_only_after_loiter_confirmation(task3_module):
    client = _ModeClient()
    mission = _mission(task3_module, mode_client=client)
    anchor = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
        "source": "initial_ram",
    }
    mission.impact_count = 2
    mission.impact_target_gps = dict(anchor)
    mission.impact_return_departed = True
    mission.state = task3_module.MissionState.RETURN_TO_IMPACT
    mission.state_started_at = 1.0
    mission.update_gps(anchor["lat"], anchor["lon"], now=1.1)
    mission.update_heading(180.0, now=1.1)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=1.1,
    )

    mission.update([], now=1.1, vision_fresh=False)

    assert mission.impact_count == 3
    assert mission.state is task3_module.MissionState.FINAL_LOITER
    assert mission.finished is False

    mission.update([], now=1.2, vision_fresh=False)
    assert client.requests[-1] == (0, "LOITER")
    assert mission.finished is False

    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "LOITER"},
        now=1.3,
    )
    mission.update([], now=1.3, vision_fresh=False)

    assert mission.state is task3_module.MissionState.FINISHED
    assert mission.finished is True


def test_return_target_republished_until_one_meter_arrival(task3_module):
    positions = []
    task3_module.publish_set_position = (
        lambda publisher, lat, lon:
        positions.append((lat, lon))
    )
    mission = _mission(task3_module)
    mission.impact_count = 1
    mission.impact_target_gps = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
        "source": "initial_ram",
    }
    mission.state = task3_module.MissionState.RETURN_TO_IMPACT
    mission.state_started_at = 1.0
    mission.update_gps(37.95127, 32.50090, now=1.1)
    mission.update_heading(180.0, now=1.1)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=1.1,
    )

    mission.update([], now=1.1, vision_fresh=False)

    assert mission.state is task3_module.MissionState.RETURN_TO_IMPACT
    assert positions[-1] == pytest.approx((37.95125, 32.50090))
    assert mission.impact_count == 1


def test_return_does_not_count_again_without_departing_anchor(task3_module):
    positions = []
    task3_module.publish_set_position = (
        lambda publisher, lat, lon: positions.append((lat, lon))
    )
    mission = _mission(task3_module)
    anchor = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
        "source": "initial_ram",
    }
    mission.impact_count = 1
    mission.impact_target_gps = dict(anchor)
    mission.impact_return_departed = False
    mission.state = task3_module.MissionState.RETURN_TO_IMPACT
    mission.state_started_at = 1.0
    mission.update_gps(anchor["lat"], anchor["lon"], now=1.1)
    mission.update_heading(180.0, now=1.1)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=1.1,
    )

    mission.update([], now=1.1, vision_fresh=False)

    assert mission.impact_count == 1
    assert mission.state is task3_module.MissionState.RETURN_TO_IMPACT
    assert positions


def test_return_timeout_enters_failsafe_and_requests_loiter(task3_module):
    client = _ModeClient()
    mission = _mission(
        task3_module,
        mode_client=client,
        impact_return_timeout_sec=0.2,
    )
    mission.impact_count = 1
    mission.impact_target_gps = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
        "source": "initial_ram",
    }
    mission.state = task3_module.MissionState.RETURN_TO_IMPACT
    mission.state_started_at = 1.0
    mission.update_gps(37.95130, 32.50090, now=1.3)
    mission.update_heading(180.0, now=1.3)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=1.3,
    )

    mission.update([], now=1.3, vision_fresh=False)
    assert mission.state is task3_module.MissionState.FAILSAFE_LOITER

    mission.update([], now=1.4, vision_fresh=False)
    assert client.requests[-1] == (0, "LOITER")

    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "LOITER"},
        now=1.5,
    )
    mission.update([], now=1.5, vision_fresh=False)
    assert mission.state is task3_module.MissionState.FAILSAFE


def test_guided_mode_is_required_before_search_motion(task3_module):
    commands = []
    client = _ModeClient()
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module, mode_client=client)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "LOITER"},
        now=0.1,
    )
    commands.clear()

    mission.update([], now=0.1)

    assert client.requests[-1] == (0, "GUIDED")
    assert commands == []

    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=0.2,
    )
    mission.update([], now=0.2)

    assert commands


def test_mode_request_is_not_duplicated_while_heartbeat_is_pending(
        task3_module,
):
    pending = _ModeFuture(done=False)
    client = _ModeClient(future_factory=lambda: pending)
    mission = _mission(task3_module)
    mission.clients.set_mode_client = client
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "LOITER"},
        now=0.1,
    )

    mission.update([], now=0.1)
    mission.update([], now=0.2)
    mission.update([], now=0.3)

    assert client.requests == [(0, "GUIDED")]


def test_uncertain_target_data_pauses_without_leaving_guided_search(
        task3_module,
):
    commands = []
    client = _ModeClient()
    task3_module.stop_vehicle = (
        lambda publisher: commands.append((0.0, 0.0))
    )
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module, mode_client=client)
    invalid = [_target(distance=None)]
    commands.clear()

    mission.update(invalid, now=0.1)
    assert mission.state is task3_module.MissionState.SEARCH
    assert commands[-1] == pytest.approx((0.0, 0.0))
    assert all(request != (0, "LOITER") for request in client.requests)

    mission.update([], now=0.2)
    assert mission.state is task3_module.MissionState.SEARCH
    assert commands[-1] != pytest.approx((0.0, 0.0))


def test_stale_vision_before_first_impact_enters_failsafe_loiter(
        task3_module,
):
    mission = _mission(task3_module)

    mission.update([], now=0.1, vision_fresh=False)

    assert mission.state is task3_module.MissionState.FAILSAFE_LOITER
    assert mission.finished is False


def test_stale_vision_does_not_interrupt_gps_return(task3_module):
    positions = []
    task3_module.publish_set_position = (
        lambda publisher, lat, lon: positions.append((lat, lon))
    )
    mission = _mission(task3_module)
    mission.impact_count = 1
    mission.impact_target_gps = {
        "lat": 37.95125,
        "lon": 32.50090,
        "recorded_at": 0.5,
        "impact_count": 1,
        "source": "initial_ram",
    }
    mission.state = task3_module.MissionState.RETURN_TO_IMPACT
    mission.state_started_at = 1.0
    mission.update_gps(37.95127, 32.50090, now=1.1)
    mission.update_heading(180.0, now=1.1)
    mission.update_bridge_state(
        {"connected": True, "armed": True, "mode": "GUIDED"},
        now=1.1,
    )

    mission.update([], now=1.1, vision_fresh=False)

    assert mission.state is task3_module.MissionState.RETURN_TO_IMPACT
    assert positions


def test_negative_motion_is_rejected(task3_module):
    commands = []
    task3_module.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z:
        commands.append((linear_x, angular_z))
    )
    mission = _mission(task3_module)

    accepted = mission._publish_motion(
        linear_x=-0.1,
        angular_z=0.0,
        reason="invalid test command",
    )

    assert accepted is False
    assert mission.state is task3_module.MissionState.FAILSAFE_LOITER
    assert commands == []


def test_production_defaults_match_direct_attack_plan(task3_module):
    defaults = task3_module.Task3Config()

    assert defaults.search_advance_distance_m == pytest.approx(1.5)
    assert defaults.search_cross_track_limit_m == pytest.approx(2.0)
    assert defaults.search_max_sweep_deg == pytest.approx(180.0)
    assert defaults.confirmation_required == 6
    assert defaults.confirmation_window_size == 6
    assert defaults.attack_confirm_speed == pytest.approx(0.15)
    assert defaults.ram_speed == pytest.approx(0.85)
    assert defaults.ram_duration_sec == pytest.approx(2.0)
    assert defaults.post_impact_forward_speed == pytest.approx(0.85)
    assert defaults.post_impact_forward_duration_sec == pytest.approx(2.5)
    assert defaults.impact_return_tolerance_m == pytest.approx(1.0)
    assert defaults.impact_return_timeout_sec == pytest.approx(20.0)
    assert defaults.mode_transition_timeout_sec == pytest.approx(5.0)
    assert defaults.loiter_mode == "LOITER"
    assert defaults.required_impact_count == 3


def test_standalone_timer_does_not_run_before_mission_is_active(task3_module):
    node = task3_module.Task3Node.__new__(task3_module.Task3Node)
    node.mission_active = False
    node.task = types.SimpleNamespace(
        update=lambda *args, **kwargs: pytest.fail(
            "inactive standalone timer must not update the mission"
        )
    )

    node.timer_callback()


def test_standalone_main_starts_in_guided_without_entry_loiter(task3_module):
    requested_modes = []
    node = types.SimpleNamespace(
        mission_active=False,
        mission_clients=types.SimpleNamespace(
            set_mode_client="set_mode",
            force_arm_client="force_arm",
            disarm_client="disarm",
        ),
        mission_topics=types.SimpleNamespace(cmd_vel_pub="cmd_vel"),
        task=types.SimpleNamespace(
            finished=True,
            state=task3_module.MissionState.FINISHED,
        ),
        get_logger=_logger,
        wait_for_complete_telemetry=lambda timeout_sec: True,
        wait_for_vision=lambda timeout_sec: True,
        start_telemetry_recording=lambda: None,
        stop_telemetry_recording=lambda: None,
        destroy_node=lambda: None,
    )
    task3_module.Task3Node = lambda: node
    task3_module.rclpy.init = lambda args=None: None
    task3_module.rclpy.shutdown = lambda: None
    task3_module.call_set_mode = (
        lambda _node, _client, mode:
        requested_modes.append(mode) or True
    )
    task3_module.call_trigger_service = lambda *_args, **_kwargs: True
    task3_module.stop_vehicle = lambda _publisher: None

    task3_module.main()

    assert requested_modes == ["GUIDED"]
