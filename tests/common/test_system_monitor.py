import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_MONITOR_PATH = REPO_ROOT / "utils" / "system_monitor.py"


@pytest.fixture()
def system_monitor_module(monkeypatch):
    class Node:
        def __init__(self, node_name):
            self.node_name = node_name
            self.subscriptions = []

        def create_subscription(
                self,
                message_type,
                topic,
                callback,
                queue_size,
        ):
            subscription = (message_type, topic, callback, queue_size)
            self.subscriptions.append(subscription)
            return subscription

    class String:
        def __init__(self, data=""):
            self.data = data

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = Node

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.BatteryState = type("BatteryState", (), {})
    sensor_msgs_msg.NavSatFix = type("NavSatFix", (), {})
    sensor_msgs.msg = sensor_msgs_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float32 = type("Float32", (), {})
    std_msgs_msg.String = String
    std_msgs.msg = std_msgs_msg

    mavlink_utilities = types.ModuleType("utils.mavlink_utilities")
    mavlink_utilities.parse_bridge_state = lambda value: value

    for name, module in {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "utils.mavlink_utilities": mavlink_utilities,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "system_monitor_test_module"
    spec = importlib.util.spec_from_file_location(
        module_name,
        SYSTEM_MONITOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_monitor_subscribes_to_and_tracks_mission_state(
        system_monitor_module,
):
    node = system_monitor_module.SystemMonitorNode(
        "test_monitor",
        subscribe_mission_status=False,
    )
    topics = [subscription[1] for subscription in node.subscriptions]

    assert "/mission/state" in topics

    message = system_monitor_module.String()
    message.data = "SEARCH"
    node._mission_state_callback(message)

    assert node.mission_state == "SEARCH"
    assert node.events[0][1] == "Mission state: SEARCH"

    event_count = len(node.events)
    node._mission_state_callback(message)
    assert len(node.events) == event_count


def test_monitor_draws_current_mission_state(
        system_monitor_module,
        monkeypatch,
):
    node = system_monitor_module.SystemMonitorNode(
        "test_monitor",
        subscribe_mission_status=False,
    )
    message = system_monitor_module.String()
    message.data = "RETURN_TO_IMPACT"
    node._mission_state_callback(message)

    drawn = []
    monkeypatch.setattr(
        system_monitor_module,
        "_safe_add",
        lambda _screen, row, column, text, attributes=0:
        drawn.append((row, column, str(text), attributes)),
    )
    monkeypatch.setattr(
        system_monitor_module.curses,
        "color_pair",
        lambda _index: 0,
    )
    screen = types.SimpleNamespace(
        erase=lambda: None,
        getmaxyx=lambda: (24, 100),
        refresh=lambda: None,
    )

    system_monitor_module._draw(screen, node, "TEST")

    assert any(
        row == 9 and column == 18 and text == "RETURN_TO_IMPACT"
        for row, column, text, _attributes in drawn
    )
