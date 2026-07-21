import importlib
import sys
import types
from pathlib import Path

import pytest


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_teknofest_main(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "teknofest.core.capture_proc", _module("capture_proc")
    )
    monkeypatch.setitem(
        sys.modules, "teknofest.core.data_writer", _module("data_writer")
    )
    monkeypatch.setitem(
        sys.modules, "teknofest.servers.data_server", _module("data_server")
    )
    monkeypatch.setitem(
        sys.modules, "utils.waypoint_server", _module("waypoint_server")
    )
    monkeypatch.delitem(sys.modules, "teknofest.main", raising=False)
    return importlib.import_module("teknofest.main")


def _load_mission_manager(monkeypatch):
    rclpy = _module("rclpy", init=lambda **_kwargs: None)
    rclpy.__path__ = []
    rclpy.ok = lambda: False
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda _node: None
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(
        sys.modules,
        "rclpy.executors",
        _module("rclpy.executors", ExternalShutdownException=RuntimeError),
    )
    monkeypatch.setitem(
        sys.modules,
        "rclpy.node",
        _module("rclpy.node", Node=object),
    )
    class Message:
        def __init__(self):
            self.data = None

    monkeypatch.setitem(
        sys.modules,
        "std_msgs.msg",
        _module("std_msgs.msg", Int32=Message, String=Message),
    )
    std_msgs = _module("std_msgs")
    std_msgs.__path__ = []
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(
        sys.modules,
        "utils.mavlink_utilities",
        _module("utils.mavlink_utilities", parse_bridge_state=lambda value: value),
    )
    monkeypatch.delitem(sys.modules, "teknofest.mission_manager", raising=False)
    return importlib.import_module("teknofest.mission_manager")


@pytest.mark.parametrize(
    ("argv", "expected_task"),
    [
        ([], None),
        (["--competition"], "competition"),
        (["--task-1"], "task1"),
        (["--task1"], "task1"),
        (["--task-2"], "task2"),
        (["--task2"], "task2"),
        (["--task-3"], "task3"),
        (["--task3"], "task3"),
    ],
)
def test_cli_selection_and_interface_default(monkeypatch, argv, expected_task):
    main = _load_teknofest_main(monkeypatch)

    assert main.parse_args(argv).task == expected_task


def test_interface_command_mapping_matches_requested_order(monkeypatch):
    manager = _load_mission_manager(monkeypatch)

    assert manager.MISSION_NAMES == {
        1: "Competition",
        2: "task1",
        3: "task2",
        4: "task3",
    }
    assert {
        command: Path(path).name
        for command, path in manager.MISSION_PATHS.items()
    } == {
        1: "competition_mission.py",
        2: "task1_point_tracking.py",
        3: "task2_point_tracking_task_in_an_environment_with_obstacle.py",
        4: "task3_kamikaze_engagement.py",
    }


def test_interface_waypoint_sync_uses_teknofest_files(monkeypatch):
    _load_teknofest_main(monkeypatch)
    from teknofest.config import mission_config

    assert mission_config.MISSION_WAYPOINT_FILES == {
        1: "teknofest.waypoints",
        2: "teknofest_task1.waypoints",
        3: "teknofest_task2.waypoints",
    }
    assert mission_config.WAYPOINT_DIRECTORY == (
        Path(__file__).resolve().parents[2] / "waypoints" / "teknofest"
    )
    assert {
        "teknofest.waypoints",
        "teknofest_task1.waypoints",
        "teknofest_task2.waypoints",
    } == {
        path.name for path in mission_config.WAYPOINT_DIRECTORY.glob("*.waypoints")
    }


def test_teknofest_config_owns_cli_and_interface_mission_specs():
    from teknofest.config.mission_config import MISSION_COMMANDS, MISSION_SPECS

    assert MISSION_COMMANDS == {
        1: ("competition", "Competition", "competition_mission.py"),
        2: ("task1", "task1", "task1_point_tracking.py"),
        3: (
            "task2",
            "task2",
            "task2_point_tracking_task_in_an_environment_with_obstacle.py",
        ),
        4: ("task3", "task3", "task3_kamikaze_engagement.py"),
    }
    assert MISSION_SPECS["competition"] == (
        "Competition",
        "competition_mission.py",
    )


def test_teknofest_profile_replaces_stale_njord_waypoint_mapping(monkeypatch):
    main = _load_teknofest_main(monkeypatch)
    monkeypatch.setenv(
        "MAVLINK_MISSION_WAYPOINT_FILES",
        "1:njord_task1.waypoints,2:njord_task2.waypoints",
    )
    monkeypatch.setenv(
        "MAVLINK_MISSION_WAYPOINT_DIRECTORY",
        str(Path(__file__).resolve().parents[2] / "waypoints" / "njord"),
    )

    main.configure_mavlink_bridge_environment()

    assert main.os.environ["MAVLINK_MISSION_WAYPOINT_FILES"] == (
        "1:teknofest.waypoints,"
        "2:teknofest_task1.waypoints,"
        "3:teknofest_task2.waypoints"
    )
    assert Path(main.os.environ["MAVLINK_MISSION_WAYPOINT_DIRECTORY"]) == (
        Path(__file__).resolve().parents[2] / "waypoints" / "teknofest"
    )


class _FakeProcess:
    def __init__(self, return_code=None, pid=4321):
        self.return_code = return_code
        self.pid = pid

    def poll(self):
        return self.return_code


class _RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


def test_duplicate_teknofest_command_republishes_ack_without_restart(
    monkeypatch, tmp_path
):
    manager_module = _load_mission_manager(monkeypatch)
    manager = manager_module.MissionManager.__new__(manager_module.MissionManager)
    manager.active_command = 2
    manager.active_task_key = "task1"
    manager.active_mission_process = _FakeProcess()
    manager.task_selection_file = str(tmp_path / "mission.json")
    manager.mission_start_ack_pub = _RecordingPublisher()
    statuses = []
    manager._publish_status = statuses.append
    manager._start_mission = lambda _command: pytest.fail(
        "duplicate command restarted the mission"
    )

    message = types.SimpleNamespace(data=2)
    manager._mission_start_callback(message)

    assert manager.mission_start_ack_pub.messages == [2]
    assert manager.active_mission_process.pid == 4321
    assert any("yeniden baslatilmadan" in status for status in statuses)


def test_teknofest_manager_clears_finished_process(monkeypatch, tmp_path):
    manager_module = _load_mission_manager(monkeypatch)
    manager = manager_module.MissionManager.__new__(manager_module.MissionManager)
    manager.active_command = 3
    manager.active_task_key = "task2"
    manager.active_mission_process = _FakeProcess(return_code=7)
    manager.task_selection_file = str(tmp_path / "mission.json")
    statuses = []
    manager._publish_status = statuses.append

    assert manager._reap_finished_mission() is True
    assert manager.active_command is None
    assert manager.active_task_key is None
    assert manager.active_mission_process is None
    assert any("return_code=7" in status for status in statuses)
