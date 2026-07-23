

import importlib
import itertools
import math
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
    node_module = types.ModuleType("rclpy.node")
    node_module.Node = object
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
    utilities.calculate_gps_distance = lambda *args: 0.0
    utilities.call_set_mode = lambda *args, **kwargs: True
    utilities.call_trigger_service = lambda *args, **kwargs: True
    utilities.create_mission_clients = lambda node: None
    utilities.create_mission_topics = lambda *args, **kwargs: None
    utilities.wait_for_mission_services = lambda *args, **kwargs: None
    utilities.publish_cmd_vel = (
        lambda publisher, linear_x, angular_z: publisher.publish(
            ("cmd_vel", float(linear_x), float(angular_z))
        )
    )
    utilities.publish_set_position = (
        lambda publisher, lat, lon, altitude=0.0: publisher.publish(
            ("set_position", float(lat), float(lon), float(altitude))
        )
    )
    utilities.stop_vehicle = lambda publisher: publisher.publish(
        ("cmd_vel", 0.0, 0.0)
    )
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
    task4 = importlib.import_module("njord.missions.task4_surprise")
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


def waypoint(seq, north_m, east_m, origin=(37.0, 32.0)):
    lat = origin[0] + math.degrees(north_m / task4.EARTH_RADIUS_M)
    lon = origin[1] + math.degrees(
        east_m / (task4.EARTH_RADIUS_M * math.cos(math.radians(origin[0])))
    )
    return {"seq": seq, "lat": lat, "lon": lon, "alt": 0.0}


class Task4RouteOptimizerTests(unittest.TestCase):
    def test_exact_solver_matches_brute_force_open_route(self):
        start = (37.0, 32.0)
        points = [
            waypoint(1, 8, 2),
            waypoint(2, -4, 13),
            waypoint(3, 15, 16),
            waypoint(4, -12, -3),
            waypoint(5, 3, -14),
            waypoint(6, 19, -5),
        ]
        solution = task4.optimize_waypoint_order(start, points)
        brute_distance = min(
            task4.route_distance_m(start, permutation)
            for permutation in itertools.permutations(points)
        )
        self.assertEqual("held-karp-exact", solution.method)
        self.assertAlmostEqual(brute_distance, solution.distance_m, places=6)
        self.assertEqual(
            {point["seq"] for point in points},
            {point["seq"] for point in solution.waypoints},
        )

    def test_large_route_heuristic_never_worsens_nearest_neighbour_seed(self):
        start = (37.0, 32.0)
        points = [waypoint(index, index * 3 % 17, index * 7 % 23) for index in range(1, 16)]
        seed = task4._nearest_neighbour_route(start, points)
        solution = task4.optimize_waypoint_order(start, points, exact_limit=5)
        self.assertEqual("nearest-neighbour+2-opt", solution.method)
        self.assertLessEqual(
            solution.distance_m,
            task4.route_distance_m(start, seed) + 1e-6,
        )

    def test_loader_discards_qgc_home(self):
        self.assertEqual([1], [item["seq"] for item in task4.load_task4_waypoints("unused")])


class Task4AvoidanceTests(unittest.TestCase):
    def setUp(self):
        self.topics = FakeTopics()
        self.mission = task4.Task4FastRoute(
            FakeNode(),
            self.topics,
            FakeClients(),
            [waypoint(1, 20, 0), waypoint(2, 30, 10)],
        )
        self._refresh_sensors(10.0)
        self.mission.state = task4.MissionState.NAVIGATING

    def _refresh_sensors(self, now):
        self.mission.update_gps(37.0, 32.0, now=now)
        self.mission.update_heading(0.0, now=now)
        self.mission.update_bridge_state(True, now=now)

    @staticmethod
    def _buoy(distance, angle):
        return {
            "type": "buoy",
            "class": "red_buoy",
            "distance": distance,
            "Buoy angle: ": angle,
        }

    def test_buoy_on_starboard_causes_port_turn(self):
        self.mission.update([self._buoy(4.0, 15.0)], now=10.0)
        self.assertEqual(task4.MissionState.AVOIDING, self.mission.state)
        self.assertEqual(
            ("cmd_vel", task4.AVOID_LINEAR_X, task4.AVOID_TURN_Z),
            self.topics.cmd_vel_pub.messages[-1],
        )

    def test_avoidance_clears_and_resumes_same_optimized_target(self):
        self.mission.update([self._buoy(4.0, -15.0)], now=10.0)
        self._refresh_sensors(11.0)
        self.mission.update([], now=11.0)
        self._refresh_sensors(11.6)
        self.mission.update([], now=11.6)
        self.assertEqual(task4.MissionState.NAVIGATING, self.mission.state)
        self.assertEqual(0, self.mission.current_target_index)

    def test_persistent_obstacle_enters_hold_failsafe(self):
        self.mission.update([self._buoy(4.0, 0.0)], now=10.0)
        self._refresh_sensors(16.1)
        self.mission.update([self._buoy(4.0, 0.0)], now=16.1)
        self.assertEqual(task4.MissionState.FAILSAFE, self.mission.state)
        self.assertTrue(self.mission.hold_mode_requested)

    def test_readiness_returns_immediately_if_empty_route_enters_failsafe(self):
        mission = task4.Task4FastRoute(
            FakeNode(),
            FakeTopics(),
            FakeClients(),
            [],
        )
        mission.update_gps(37.0, 32.0, now=10.0)
        fake_node = types.SimpleNamespace(
            task=mission,
            bridge_connected=True,
            valid_gps_received=True,
            valid_heading_received=True,
            get_logger=lambda: FakeLogger(),
        )

        self.assertFalse(
            task4.Task4Node.wait_until_ready(fake_node, timeout_sec=30.0)
        )
        self.assertEqual(task4.MissionState.FAILSAFE, mission.state)


if __name__ == "__main__":
    unittest.main()
