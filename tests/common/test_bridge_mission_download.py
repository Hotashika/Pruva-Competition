import importlib
import sys
import types


MISSION_CONTENT = (
    "QGC WPL 110\n"
    "0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\n"
    "1\t0\t3\t16\t0\t0\t0\t0\t37.1\t32.1\t0\t1\n"
)


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_bridge(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", _module("rclpy"))
    monkeypatch.setitem(sys.modules, "rclpy.node", _module("rclpy.node", Node=object))
    monkeypatch.setitem(
        sys.modules,
        "rclpy.executors",
        _module("rclpy.executors", ExternalShutdownException=RuntimeError),
    )

    mavlink = types.SimpleNamespace(
        MAV_MISSION_ACCEPTED=0,
        MAV_MISSION_TYPE_MISSION=0,
        MAV_SEVERITY_ERROR=3,
        MAV_SEVERITY_INFO=6,
        MAV_SEVERITY_WARNING=4,
    )
    pymavlink = _module(
        "pymavlink",
        mavutil=types.SimpleNamespace(mavlink=mavlink),
    )
    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink)

    sensor_msgs = _module("sensor_msgs")
    sensor_msgs.__path__ = []
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(
        sys.modules,
        "sensor_msgs.msg",
        _module(
            "sensor_msgs.msg",
            Imu=object,
            NavSatFix=object,
            BatteryState=object,
        ),
    )
    std_msgs = _module("std_msgs")
    std_msgs.__path__ = []
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(
        sys.modules,
        "std_msgs.msg",
        _module("std_msgs.msg", Float32=object, Int32=object, String=object),
    )

    connection = _module(
        "bridge.mavlink_connection",
        DEFAULT_BAUD=921600,
        DEFAULT_CONNECTION_STRING="test",
        DEFAULT_HEARTBEAT_TIMEOUT=5,
        DEFAULT_SOURCE_COMPONENT=191,
        DEFAULT_SOURCE_SYSTEM=1,
        connect_mavlink=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "bridge.mavlink_connection", connection)
    monkeypatch.setitem(
        sys.modules,
        "utils.mavlink_utilities",
        _module(
            "utils.mavlink_utilities",
            create_bridge_topics=lambda *_args, **_kwargs: None,
            create_bridge_services=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.delitem(sys.modules, "bridge.bridge_node", raising=False)
    return importlib.import_module("bridge.bridge_node")


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)

    def info(self, _message):
        pass


class _Mav:
    def __init__(self):
        self.acks = []

    def mission_ack_send(self, *args):
        self.acks.append(args)


def _download_node(bridge_module):
    node = bridge_module.OrangeCubeBridgeNode.__new__(
        bridge_module.OrangeCubeBridgeNode
    )
    node.mission_download_task = 2
    node.mission_download_count = 1
    node.mission_download_items = {}
    node.mission_download_last_request_time = 0.0
    node.mission_download_retry_count = 0
    node.last_mission_parameter_value = 2
    node.pending_mission_command = None
    node.pending_mission_command_first_publish_time = 0.0
    node.pending_mission_command_last_publish_time = 0.0
    node.get_logger = lambda: _Logger()
    node._publish_diagnostic = lambda _message: None
    node._send_status_text = lambda *_args, **_kwargs: None
    return node


def test_bridge_writes_download_to_configured_profile_directory(monkeypatch, tmp_path):
    bridge_module = _load_bridge(monkeypatch)
    monkeypatch.setattr(
        bridge_module,
        "mission_items_to_qgc",
        lambda _items: MISSION_CONTENT,
    )
    node = _download_node(bridge_module)
    node.mission_waypoint_files = {2: "teknofest_task1.waypoints"}
    node.mission_waypoint_directory = tmp_path / "waypoints" / "teknofest"
    mav = _Mav()
    node.master = types.SimpleNamespace(
        target_system=1,
        target_component=1,
        mav=mav,
    )
    published = []
    node._publish_downloaded_mission_start = published.append

    node._handle_mission_item(types.SimpleNamespace(seq=0))

    destination = node.mission_waypoint_directory / "teknofest_task1.waypoints"
    assert destination.read_text(encoding="utf-8") == MISSION_CONTENT
    assert published == [2]
    assert len(mav.acks) == 1
    assert node.mission_download_task is None


def test_bridge_download_timeout_resets_command_for_operator_retry(
    monkeypatch, tmp_path
):
    bridge_module = _load_bridge(monkeypatch)
    node = _download_node(bridge_module)
    node.mission_download_task = 1
    node.mission_download_count = None
    node.mission_download_retry_count = 4
    node.mission_waypoint_files = {1: "teknofest.waypoints"}
    node.mission_waypoint_directory = tmp_path
    node.master = object()
    reset_values = []
    node._set_mission_parameter = reset_values.append

    node._mission_download_watchdog()

    assert reset_values == [bridge_module.MISSION_IDLE]
    assert node.last_mission_parameter_value == 2
    assert node.mission_download_task is None


def test_bridge_empty_mission_resets_command(monkeypatch):
    bridge_module = _load_bridge(monkeypatch)
    node = _download_node(bridge_module)
    node.master = object()
    reset_values = []
    node._set_mission_parameter = reset_values.append

    node._handle_mission_count(types.SimpleNamespace(count=0))

    assert reset_values == [bridge_module.MISSION_IDLE]
    assert node.mission_download_task is None
