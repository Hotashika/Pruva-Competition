import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK3_PATH = REPO_ROOT / "njord" / "missions" / "task3_docking.py"


class _Vector:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Twist:
    def __init__(self):
        self.linear = _Vector()
        self.angular = _Vector()


class _Message:
    def __init__(self):
        self.data = None
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _Node:
    def get_logger(self):
        return _Logger()


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def task3_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        _module(
            "rclpy",
            init=lambda *args, **kwargs: None,
            ok=lambda: False,
            shutdown=lambda: None,
            spin_once=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rclpy.node",
        _module("rclpy.node", Node=object),
    )
    monkeypatch.setitem(sys.modules, "geometry_msgs", _module("geometry_msgs"))
    monkeypatch.setitem(
        sys.modules,
        "geometry_msgs.msg",
        _module("geometry_msgs.msg", Twist=_Twist),
    )
    monkeypatch.setitem(sys.modules, "sensor_msgs", _module("sensor_msgs"))
    monkeypatch.setitem(
        sys.modules,
        "sensor_msgs.msg",
        _module("sensor_msgs.msg", NavSatFix=_Message),
    )
    monkeypatch.setitem(sys.modules, "std_msgs", _module("std_msgs"))
    monkeypatch.setitem(
        sys.modules,
        "std_msgs.msg",
        _module("std_msgs.msg", Float32=_Message, String=_Message),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.mavlink_utilities",
        _module(
            "utils.mavlink_utilities",
            call_set_mode=lambda *args, **kwargs: True,
            call_trigger_service=lambda *args, **kwargs: True,
            calculate_gps_distance=lambda *args, **kwargs: 0.0,
            create_mission_clients=lambda *args, **kwargs: None,
            create_mission_topics=lambda *args, **kwargs: None,
            publish_cmd_vel=lambda *args, **kwargs: None,
            publish_set_position=lambda *args, **kwargs: None,
            stop_vehicle=lambda *args, **kwargs: None,
            wait_for_mission_services=lambda *args, **kwargs: True,
        ),
    )

    spec = importlib.util.spec_from_file_location(
        "task3_docking_modes_test_module",
        TASK3_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _detection_payload(name, distance_m):
    return {
        "frame_px": {"width": 640, "height": 480},
        "detections": [
            {
                "canonical_payload": name,
                "confidence": 0.95,
                "distance_m": distance_m,
                "center_px": {"x": 320.0, "y": 240.0},
                "bbox_xywh_px": {
                    "x": 280.0,
                    "y": 200.0,
                    "width": 80.0,
                    "height": 80.0,
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("environment_mode", "sequence", "targets", "parallel_enabled"),
    [
        (
            "normal",
            ("normal",),
            ("middle_berth_1", "middle_berth_2"),
            False,
        ),
        ("parallel", ("parallel",), ("middle_parallel",), True),
    ],
)
def test_mode_environment_builds_single_mode_task3_config(
    monkeypatch,
    task3_module,
    environment_mode,
    sequence,
    targets,
    parallel_enabled,
):
    monkeypatch.setenv("TASK3_DOCKING_MODE", environment_mode)
    monkeypatch.delenv("TASK3_ALLOWED_PAYLOADS", raising=False)

    config = task3_module.create_default_config()

    assert config.sequence == sequence
    assert config.allowed_payloads == targets
    assert config.modes[environment_mode].target_payloads == targets
    assert config.modes[environment_mode].reverse_after_hold is False
    assert config.parallel_maneuver_enabled is parallel_enabled


def test_serial_depth_stop_enters_hold_and_finishes_without_reverse(
    monkeypatch,
    task3_module,
):
    monkeypatch.setenv("TASK3_DOCKING_MODE", "normal")
    monkeypatch.delenv("TASK3_ALLOWED_PAYLOADS", raising=False)
    config = task3_module.create_default_config()
    config.require_qr_confirmation = False
    mission = task3_module.Task3DockingMission(
        _Node(), config, object(), object()
    )
    mission.state = task3_module.DockingState.FINAL_APPROACH

    for _ in range(3):
        mission.update_qr_from_json(
            _detection_payload("middle_berth_1", distance_m=1.3)
        )
    mission._update_final_approach()

    assert mission.state is task3_module.DockingState.HOLD_POSITION
    mission.state_enter_time = time.monotonic() - config.modes["normal"].hold_seconds
    mission._update_hold_position()
    assert mission.state is task3_module.DockingState.MODE_FINISHED


def test_parallel_depth_stop_starts_left_heading_manoeuvre(
    monkeypatch,
    task3_module,
):
    monkeypatch.setenv("TASK3_DOCKING_MODE", "parallel")
    monkeypatch.delenv("TASK3_ALLOWED_PAYLOADS", raising=False)
    config = task3_module.create_default_config()
    config.require_qr_confirmation = False
    mission = task3_module.Task3DockingMission(
        _Node(), config, object(), object()
    )
    mission.state = task3_module.DockingState.FINAL_APPROACH
    mission.update_heading(0.0)

    for _ in range(3):
        mission.update_qr_from_json(
            _detection_payload("middle_parallel", distance_m=1.3)
        )
    mission._update_final_approach()

    assert mission.state is task3_module.DockingState.PARALLEL_TURN_FIRST
    assert mission.parallel_initial_heading == 0.0
    assert mission.parallel_first_target_heading == 270.0
    assert mission.parallel_final_target_heading == 270.0
    error, angular_z = mission._parallel_heading_command(270.0)
    assert error == -90.0
    assert angular_z < 0.0


def test_vehicle_preparation_uses_standard_guided_and_arm_services(
    monkeypatch,
    task3_module,
):
    calls = []
    node = types.SimpleNamespace(
        mission_clients=types.SimpleNamespace(
            set_mode_client="set_mode",
            force_arm_client="force_arm",
        ),
        get_logger=lambda: _Logger(),
    )
    monkeypatch.setattr(
        task3_module,
        "call_set_mode",
        lambda _node, client, mode, **_kwargs:
        calls.append((client, mode)) or True,
    )
    monkeypatch.setattr(
        task3_module,
        "call_trigger_service",
        lambda _node, client, action:
        calls.append((client, action)) or True,
    )

    prepared = task3_module.Task3DockingNode.prepare_vehicle(node)

    assert prepared is True
    assert calls == [("set_mode", "GUIDED"), ("force_arm", "FORCE ARM")]
