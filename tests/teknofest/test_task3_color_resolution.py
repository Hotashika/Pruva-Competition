"""Task 3 hedef rengi çözümlemesini ROS 2/donanım olmadan doğrular."""

import importlib
import math
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
PARAM_OVERRIDES = {}


class FakeParameterValue:
    def __init__(self, value):
        self.string_value = value if isinstance(value, str) else ""
        self.bool_value = value if isinstance(value, bool) else False
        self.double_value = float(value) if isinstance(value, (int, float)) else 0.0


class FakeParameter:
    def __init__(self, value):
        self.value = value

    def get_parameter_value(self):
        return FakeParameterValue(self.value)


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    warn = warning

    def error(self, *args, **kwargs):
        pass


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.logger = FakeLogger()
        self.parameters = {}

    def get_logger(self):
        return self.logger

    def declare_parameter(self, name, default):
        self.parameters[name] = PARAM_OVERRIDES.get(name, default)

    def get_parameter(self, name):
        return FakeParameter(self.parameters[name])

    def create_subscription(self, *args, **kwargs):
        return object()

    def create_timer(self, *args, **kwargs):
        return object()

    def create_publisher(self, *args, **kwargs):
        return types.SimpleNamespace(publish=lambda msg: None)


class FakeMessage:
    def __init__(self):
        self.data = None


class FakeService:
    class Request:
        pass


fake_rclpy = types.ModuleType("rclpy")
fake_rclpy.spin_until_future_complete = lambda *args, **kwargs: None
fake_rclpy_node = types.ModuleType("rclpy.node")
fake_rclpy_node.Node = FakeNode
fake_rclpy.node = fake_rclpy_node
sys.modules["rclpy"] = fake_rclpy
sys.modules["rclpy.node"] = fake_rclpy_node

for parent_name, child_name, names in (
    ("std_msgs", "std_msgs.msg", ("Int32", "String")),
    ("sensor_msgs", "sensor_msgs.msg", ("Imu",)),
):
    parent = types.ModuleType(parent_name)
    child = types.ModuleType(child_name)
    for name in names:
        setattr(child, name, FakeMessage)
    parent.msg = child
    sys.modules[parent_name] = parent
    sys.modules[child_name] = child

for parent_name, child_name, service_name in (
    ("mavros_msgs", "mavros_msgs.srv", "SetMode"),
    ("std_srvs", "std_srvs.srv", "Trigger"),
):
    parent = types.ModuleType(parent_name)
    child = types.ModuleType(child_name)
    setattr(child, service_name, FakeService)
    parent.srv = child
    sys.modules[parent_name] = parent
    sys.modules[child_name] = child


def calculate_gps_distance(lat1, lon1, lat2, lon2):
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat1))
    return math.hypot(north, east)


def calculate_bearing(lat1, lon1, lat2, lon2):
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat1))
    return math.degrees(math.atan2(east, north)) % 360.0


fake_mavlink = types.ModuleType("utils.mavlink_utilities")
fake_mavlink.publish_cmd_vel = lambda *args, **kwargs: None
fake_mavlink.stop_vehicle = lambda *args, **kwargs: None
fake_mavlink.calculate_gps_distance = calculate_gps_distance
fake_mavlink.calculate_bearing = calculate_bearing
fake_mavlink.create_mission_topics = lambda *args, **kwargs: types.SimpleNamespace(
    cmd_vel_pub=object()
)


class FakeFuture:
    def add_done_callback(self, callback):
        callback(self)

    def exception(self):
        return None

    def done(self):
        return True

    def result(self):
        return types.SimpleNamespace(success=True, message="disarmed")


class FakeClient:
    def call_async(self, request):
        return FakeFuture()


class RecordingPublisher:
    def __init__(self):
        self.values = []

    def publish(self, message):
        self.values.append(message.data)


fake_mavlink.create_mission_clients = lambda *args, **kwargs: types.SimpleNamespace(
    set_mode_client=FakeClient(),
    arm_client=FakeClient(),
    force_arm_client=FakeClient(),
    disarm_client=FakeClient(),
)
fake_mavlink.wait_for_mission_services = lambda *args, **kwargs: None
sys.modules["utils.mavlink_utilities"] = fake_mavlink

t3 = importlib.import_module("teknofest.missions.task3_kamikaze_engagement")


class Task3ColorResolutionTests(unittest.TestCase):
    def tearDown(self):
        PARAM_OVERRIDES.clear()

    def test_default_color_is_red(self):
        node = t3.Task3Node()
        self.assertEqual(node.target_class, "red_buoy")

    def test_official_target_colors_are_supported(self):
        for color in ("red", "green", "black"):
            with self.subTest(color=color):
                PARAM_OVERRIDES["carpilacak_duba"] = color
                node = t3.Task3Node()
                self.assertEqual(node.target_class, f"{color}_buoy")

    def test_unknown_color_is_rejected_before_mission_start(self):
        PARAM_OVERRIDES["carpilacak_duba"] = "blue"
        with self.assertRaises(SystemExit):
            t3.Task3Node()

    def test_shutdown_path_disarms(self):
        node = t3.Task3Node()
        self.assertTrue(node.shutdown_and_disarm())
        self.assertIs(node.task.bridge_armed, False)

    def test_stop_ack_is_sent_only_after_disarm_succeeds(self):
        node = t3.Task3Node()
        publisher = RecordingPublisher()
        node.mission_start_ack_pub = publisher
        node.command_disarm_in_progress = True

        node._command_disarm_worker(99)

        self.assertEqual(publisher.values, [99])
        self.assertFalse(node.command_disarm_in_progress)

        publisher.values.clear()
        node.command_disarm_in_progress = True
        node._disarm_with_retries = lambda label: False
        node._command_disarm_worker(90)
        self.assertEqual(publisher.values, [])
        self.assertFalse(node.command_disarm_in_progress)

    def test_disarm_is_retried_three_times_before_failure(self):
        node = t3.Task3Node()
        outcomes = iter((False, False, True))
        calls = []

        def fake_trigger(client, label):
            calls.append(label)
            return next(outcomes)

        node._trigger_and_wait = fake_trigger
        self.assertTrue(node._disarm_with_retries("TEST DISARM"))
        self.assertEqual(len(calls), 3)
        self.assertIs(node.task.bridge_armed, False)


if __name__ == "__main__":
    unittest.main()
