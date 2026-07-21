import importlib
import sys
import types
import unittest


def _install_ros_stubs():
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda *args, **kwargs: None
    rclpy.shutdown = lambda: None
    rclpy.spin_once = lambda *args, **kwargs: None
    rclpy.spin_until_future_complete = lambda *args, **kwargs: None

    class StubLogger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    class StubPublisher:
        def publish(self, message):
            pass

    class StubNode:
        def __init__(self, *args, **kwargs):
            pass

        def get_logger(self):
            return StubLogger()

        def create_subscription(self, *args, **kwargs):
            return object()

        def create_publisher(self, *args, **kwargs):
            return StubPublisher()

        def create_timer(self, *args, **kwargs):
            return object()

    node_module = types.ModuleType("rclpy.node")
    node_module.Node = StubNode
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = node_module

    mavros_msgs = types.ModuleType("mavros_msgs")
    mavros_srv = types.ModuleType("mavros_msgs.srv")

    class SetMode:
        class Request:
            def __init__(self):
                self.base_mode = 0
                self.custom_mode = ""

    mavros_srv.SetMode = SetMode
    mavros_msgs.srv = mavros_srv
    sys.modules["mavros_msgs"] = mavros_msgs
    sys.modules["mavros_msgs.srv"] = mavros_srv

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")

    class String:
        def __init__(self):
            self.data = ""

    std_msgs_msg.String = String
    std_msgs.msg = std_msgs_msg
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg

    utilities = types.ModuleType("utils.mavlink_utilities")
    utilities.align_heading_to_gps_target = lambda *args, **kwargs: True
    utilities.calculate_gps_distance = lambda lat1, lon1, lat2, lon2: 100.0
    utilities.call_set_mode = lambda *args, **kwargs: True
    utilities.call_trigger_service = lambda *args, **kwargs: True
    utilities.create_mission_clients = lambda node: None
    utilities.create_mission_topics = lambda *args, **kwargs: None
    utilities.parse_bridge_state = lambda text: {
        key.strip(): (
            value.strip().lower() == "true"
            if value.strip().lower() in ("true", "false")
            else value.strip()
        )
        for part in str(text).split(",")
        if "=" in part
        for key, value in [part.split("=", 1)]
    }
    utilities.wait_for_mission_services = lambda *args, **kwargs: None
    utilities.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z: publisher.publish(
            ("cmd_vel", float(linear_x), float(angular_z))
        )
    )
    utilities.publish_set_position = (
        lambda publisher, lat, lon, altitude=20.0: publisher.publish(
            ("set_position", float(lat), float(lon), float(altitude))
        )
    )
    utilities.stop_vehicle = lambda publisher: publisher.publish(("cmd_vel", 0.0, 0.0))
    sys.modules["utils.mavlink_utilities"] = utilities

    waypoint_reader = types.ModuleType("utils.read_waypoints")
    waypoint_reader.parse_qgc_waypoints = lambda path: [
        {"seq": 0, "lat": 1.0, "lon": 1.0, "alt": 0.0},
        {"seq": 1, "lat": 2.0, "lon": 2.0, "alt": 0.0},
    ]
    sys.modules["utils.read_waypoints"] = waypoint_reader


_STUB_MODULE_NAMES = (
    "rclpy",
    "rclpy.node",
    "mavros_msgs",
    "mavros_msgs.srv",
    "std_msgs",
    "std_msgs.msg",
    "utils.mavlink_utilities",
    "utils.read_waypoints",
)
_missing_module = object()
_original_modules = {
    name: sys.modules.get(name, _missing_module)
    for name in _STUB_MODULE_NAMES
}
_install_ros_stubs()
try:
    task2 = importlib.import_module("njord.missions.task2_collision_avoidance")
finally:
    for _name, _module in _original_modules.items():
        if _module is _missing_module:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


class FakeLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeNode:
    def get_logger(self):
        return FakeLogger()


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeTopics:
    def __init__(self):
        self.cmd_vel_pub = FakePublisher()
        self.position_target_pub = FakePublisher()


class FakeSetModeClient:
    def call_async(self, request):
        return object()


class FakeClients:
    def __init__(self):
        self.set_mode_client = FakeSetModeClient()


class Task2CollisionAvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.topics = FakeTopics()
        self.mission = task2.Task2CollisionAvoidance(
            FakeNode(),
            self.topics,
            FakeClients(),
            [{"lat": 10.0, "lon": 20.0, "alt": 0.0, "seq": 1}],
        )
        self._refresh_sensors(10.0)
        self.mission.state = task2.MissionState.NAVIGATING

    def _refresh_sensors(self, now):
        self.mission.update_gps(1.0, 1.0, now=now)
        self.mission.update_heading(0.0, now=now)
        self.mission.update_bridge_state(True, True, "GUIDED", now=now)

    @staticmethod
    def _vessel(distance, angle):
        return {
            "type": "vessel",
            "class": "unknown_model_label",
            "distance": distance,
            "Vessel angle: ": angle,
        }

    @staticmethod
    def _buoy(color, distance, angle):
        return {
            "type": "buoy",
            "class": f"{color}_buoys",
            "distance": distance,
            "Buoy angle: ": angle,
        }

    def _update(self, distance, angle, now, record=True):
        self._refresh_sensors(now)
        self.mission.update(
            [self._vessel(distance, angle)],
            now=now,
            record_observation=record,
        )

    def test_task2_waypoint_loader_discards_qgc_home(self):
        waypoints = task2.load_task2_waypoints("unused.waypoints")
        self.assertEqual([1], [waypoint["seq"] for waypoint in waypoints])

    def test_task2_node_accepts_initial_bridge_state(self):
        node = task2.Task2Node()
        message = task2.String()
        message.data = "connected=True,armed=False,mode=GUIDED"

        node.state_callback(message)

        self.assertTrue(node.bridge_connected)
        self.assertFalse(node.bridge_armed)
        self.assertEqual("GUIDED", node.bridge_mode)
        self.assertEqual((True, False, "GUIDED"), node._last_logged_bridge_state)

    def test_receding_vessel_does_not_trigger_avoidance(self):
        self._update(6.0, 0.0, 10.0)
        self._update(6.5, 0.0, 10.3)
        self._update(7.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.NAVIGATING, self.mission.state)
        self.assertNotIn(("cmd_vel", 0.5, -0.6), self.topics.cmd_vel_pub.messages)

    def test_head_on_collision_risk_always_commands_starboard(self):
        self._update(6.0, 0.0, 10.0)
        self._update(5.0, 0.0, 10.3)
        self._update(4.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual(("cmd_vel", 0.5, -0.6), self.topics.cmd_vel_pub.messages[-1])

    def test_closing_buoy_is_used_as_collision_target(self):
        for distance, now in ((6.0, 10.0), (5.0, 10.3), (4.0, 10.6)):
            self._refresh_sensors(now)
            self.mission.update(
                [self._buoy("red", distance, 0.0)],
                now=now,
                record_observation=True,
            )

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual(("cmd_vel", 0.5, -0.6), self.topics.cmd_vel_pub.messages[-1])

    def test_current_buoy_model_classes_are_collision_targets(self):
        expected_classes = {
            "green_buoys",
            "red_buoys",
            "north_buoys",
            "east_buoys",
            "south_buoys",
            "west_buoys",
        }

        self.assertEqual(expected_classes, task2.BUOY_MODEL_TYPES)
        for model_class in expected_classes:
            with self.subTest(model_class=model_class):
                self.assertTrue(
                    self.mission._is_vessel(
                        {"type": "buoy", "class": model_class}
                    )
                )

    def test_unknown_buoy_class_is_not_used_as_collision_target(self):
        self.assertFalse(
            self.mission._is_vessel(
                {"type": "buoy", "class": "unknown_buoy"}
            )
        )

    def test_port_side_risk_stands_on_then_uses_starboard_fallback(self):
        self._update(5.0, -25.0, 10.0)
        self._update(4.7, -25.0, 10.3)
        self._update(4.4, -25.0, 10.6)

        self.assertEqual(task2.MissionState.STAND_ON, self.mission.state)
        self.assertNotIn(("cmd_vel", 0.5, -0.6), self.topics.cmd_vel_pub.messages)

        self._update(4.4, -25.0, 13.2, record=False)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual(("cmd_vel", 0.5, -0.6), self.topics.cmd_vel_pub.messages[-1])

    def test_avoidance_resumes_same_waypoint_only_after_clear_duration(self):
        self._update(6.0, 0.0, 10.0)
        self._update(5.0, 0.0, 10.3)
        self._update(4.0, 0.0, 10.6)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)

        self._refresh_sensors(11.5)
        self.mission.update([], now=11.5)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)

        self._refresh_sensors(12.6)
        self.mission.update([], now=12.6)
        self.assertEqual(task2.MissionState.NAVIGATING, self.mission.state)
        self.assertEqual(0, self.mission.current_target_index)

    def test_next_waypoint_waits_then_aligns_before_position_command(self):
        self.mission.waypoints = [
            {"lat": 10.0, "lon": 20.0, "alt": 0.0, "seq": 1},
            {"lat": 11.0, "lon": 21.0, "alt": 0.0, "seq": 2},
        ]
        original_distance = task2.calculate_gps_distance
        original_align = task2.align_heading_to_gps_target
        alignment_ready = {"value": False}
        alignment_targets = []

        def fake_distance(lat1, lon1, lat2, lon2):
            return 0.5 if float(lat2) == 10.0 else 10.0

        def fake_align(*args, **kwargs):
            alignment_targets.append(kwargs.get("target_name"))
            return alignment_ready["value"]

        task2.calculate_gps_distance = fake_distance
        task2.align_heading_to_gps_target = fake_align
        self.addCleanup(setattr, task2, "calculate_gps_distance", original_distance)
        self.addCleanup(setattr, task2, "align_heading_to_gps_target", original_align)

        self.mission.update([], now=10.0)
        self.assertEqual(1, self.mission.current_target_index)
        self.assertEqual(("cmd_vel", 0.0, 0.0), self.topics.cmd_vel_pub.messages[-1])

        self._refresh_sensors(10.5)
        self.mission.update([], now=10.5)
        self.assertEqual([], alignment_targets)
        self.assertEqual([], self.topics.position_target_pub.messages)

        self._refresh_sensors(10.8)
        self.mission.update([], now=10.8)
        self.assertEqual(["WP1"], alignment_targets)
        self.assertEqual([], self.topics.position_target_pub.messages)

        alignment_ready["value"] = True
        self._refresh_sensors(10.9)
        self.mission.update([], now=10.9)
        self.assertEqual(
            ("set_position", 11.0, 21.0, 0.0),
            self.topics.position_target_pub.messages[-1],
        )

    def test_avoidance_forces_realignment_with_current_waypoint(self):
        self.mission.aligned_target_key = ("WP0", 10.0, 20.0)

        self._update(6.0, 0.0, 10.0)
        self._update(5.0, 0.0, 10.3)
        self._update(4.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertIsNone(self.mission.aligned_target_key)


if __name__ == "__main__":
    unittest.main()
