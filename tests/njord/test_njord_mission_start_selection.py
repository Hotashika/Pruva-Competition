import importlib
import sys
import types
from pathlib import Path

import pytest


def _module(name):
    return types.ModuleType(name)


def _load_njord_main(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "njord.core.capture_proc", _module("capture_proc")
    )
    monkeypatch.setitem(
        sys.modules, "njord.core.data_writer", _module("data_writer")
    )
    monkeypatch.setitem(
        sys.modules, "njord.servers.data_server", _module("data_server")
    )
    monkeypatch.setitem(
        sys.modules, "njord.servers.video_server", _module("video_server")
    )
    monkeypatch.setitem(
        sys.modules, "utils.waypoint_server", _module("waypoint_server")
    )
    monkeypatch.delitem(sys.modules, "njord.main", raising=False)
    return importlib.import_module("njord.main")


def _load_mission_manager(monkeypatch):
    rclpy = _module("rclpy")
    rclpy.__path__ = []
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    executors = _module("rclpy.executors")
    executors.ExternalShutdownException = RuntimeError
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors)
    node = _module("rclpy.node")
    node.Node = object
    monkeypatch.setitem(sys.modules, "rclpy.node", node)

    class Message:
        def __init__(self):
            self.data = None

    std_msgs = _module("std_msgs")
    std_msgs.__path__ = []
    std_msgs_msg = _module("std_msgs.msg")
    std_msgs_msg.Int32 = Message
    std_msgs_msg.String = Message
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)
    mavlink_utilities = _module("utils.mavlink_utilities")
    mavlink_utilities.parse_bridge_state = lambda value: value
    monkeypatch.setitem(
        sys.modules,
        "utils.mavlink_utilities",
        mavlink_utilities,
    )
    monkeypatch.delitem(sys.modules, "njord.mission_manager", raising=False)
    return importlib.import_module("njord.mission_manager")


def test_njord_interface_syncs_all_four_waypoint_files(monkeypatch):
    _load_njord_main(monkeypatch)
    from njord.config import mission_config

    assert mission_config.MISSION_WAYPOINT_FILES == {
        1: "njord_task1.waypoints",
        2: "njord_task2.waypoints",
        3: "njord_task3.waypoints",
        4: "njord_task4.waypoints",
    }
    assert mission_config.WAYPOINT_DIRECTORY == (
        Path(__file__).resolve().parents[2] / "waypoints" / "njord"
    )
    assert {
        "njord_task1.waypoints",
        "njord_task2.waypoints",
        "njord_task3.waypoints",
        "njord_task4.waypoints",
    } <= {
        path.name for path in mission_config.WAYPOINT_DIRECTORY.glob("*.waypoints")
    }


def test_njord_config_owns_interface_mission_specs():
    from njord.config.mission_config import MISSION_COMMANDS

    assert MISSION_COMMANDS == {
        1: ("task1", "task1", "task1_maneuvering_and_path_finding.py"),
        2: ("task2", "task2", "task2_collision_avoidance.py"),
        3: ("task3", "task3", "task3_docking.py"),
        4: ("task4", "task4", "task4_surprise.py"),
    }


def test_njord_profile_replaces_stale_teknofest_waypoint_mapping(monkeypatch):
    main = _load_njord_main(monkeypatch)
    monkeypatch.setenv(
        "MAVLINK_MISSION_WAYPOINT_FILES",
        "1:teknofest.waypoints,2:teknofest_task1.waypoints",
    )
    monkeypatch.setenv(
        "MAVLINK_MISSION_WAYPOINT_DIRECTORY",
        str(Path(__file__).resolve().parents[2] / "waypoints" / "teknofest"),
    )

    main.configure_mavlink_bridge_environment()

    assert main.os.environ["MAVLINK_MISSION_WAYPOINT_FILES"] == (
        "1:njord_task1.waypoints,"
        "2:njord_task2.waypoints,"
        "3:njord_task3.waypoints,"
        "4:njord_task4.waypoints"
    )
    assert Path(main.os.environ["MAVLINK_MISSION_WAYPOINT_DIRECTORY"]) == (
        Path(__file__).resolve().parents[2] / "waypoints" / "njord"
    )


class _FakeProcess:
    def __init__(self, return_code=None, pid=1234):
        self.return_code = return_code
        self.pid = pid

    def poll(self):
        return self.return_code


class _RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


def test_duplicate_njord_command_republishes_ack_without_restart(
    monkeypatch, tmp_path
):
    manager_module = _load_mission_manager(monkeypatch)
    manager = manager_module.MissionManager.__new__(manager_module.MissionManager)
    manager.active_mission_number = 2
    manager.active_mission_process = _FakeProcess()
    manager.task_selection_file = str(tmp_path / "mission.json")
    manager.mission_start_topic = "/mission_start"
    manager.mission_start_ack_topic = "/mission_start_ack"
    manager.mission_start_ack_pub = _RecordingPublisher()
    statuses = []
    manager._publish_status = statuses.append
    manager._start_mission = lambda _command: pytest.fail(
        "duplicate command restarted the mission"
    )

    message = types.SimpleNamespace(data=2)
    manager._mission_start_callback(message)

    assert manager.mission_start_ack_pub.messages == [2]
    assert manager.active_mission_process.pid == 1234
    assert any("without restarting" in status for status in statuses)


def test_njord_manager_clears_finished_process(monkeypatch, tmp_path):
    manager_module = _load_mission_manager(monkeypatch)
    manager = manager_module.MissionManager.__new__(manager_module.MissionManager)
    manager.active_mission_number = 4
    manager.active_mission_process = _FakeProcess(return_code=9)
    manager.task_selection_file = str(tmp_path / "mission.json")
    statuses = []
    manager._publish_status = statuses.append

    assert manager._reap_finished_mission() is True
    assert manager.active_mission_number is None
    assert manager.active_mission_process is None
    assert any("return_code=9" in status for status in statuses)
