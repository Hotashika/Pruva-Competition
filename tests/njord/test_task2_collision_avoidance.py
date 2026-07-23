import csv
import importlib
import math
import sys
import tempfile
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
    def _vessel(distance, angle, track_id=7):
        detection = {
            "type": "vessel",
            "class": "unknown_model_label",
            "distance": distance,
            "Vessel angle: ": angle,
        }
        if track_id is not None:
            detection["track_id"] = track_id
        return detection

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

    @staticmethod
    def _target_clearance_m(target):
        mean_lat = math.radians((target["marker_lat"] + target["lat"]) / 2.0)
        north_m = (
            math.radians(target["lat"] - target["marker_lat"])
            * task2.EARTH_RADIUS_M
        )
        east_m = (
            math.radians(target["lon"] - target["marker_lon"])
            * task2.EARTH_RADIUS_M
            * math.cos(mean_lat)
        )
        return math.hypot(north_m, east_m)

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
        self.assertIsNone(self.mission.avoidance_target)

    def test_head_on_collision_risk_creates_starboard_gps_target(self):
        self._update(6.0, 0.0, 10.0)
        self._update(5.0, 0.0, 10.3)
        self._update(4.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        target = self.mission.avoidance_target
        self.assertIsNotNone(target)
        self.assertEqual("starboard", target["side"])
        self.assertAlmostEqual(
            task2.AVOID_PASS_CLEARANCE_M,
            self._target_clearance_m(target),
            places=3,
        )
        self.assertEqual(
            ("set_position", target["lat"], target["lon"], 20.0),
            self.topics.position_target_pub.messages[-1],
        )

    def test_closing_buoy_is_used_as_collision_target(self):
        for distance, now in ((6.0, 10.0), (5.0, 10.3), (4.0, 10.6)):
            self._refresh_sensors(now)
            self.mission.update(
                [self._buoy("red", distance, 0.0)],
                now=now,
                record_observation=True,
            )

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        target = self.mission.avoidance_target
        self.assertIsNotNone(target)
        self.assertAlmostEqual(
            task2.AVOID_PASS_CLEARANCE_M,
            self._target_clearance_m(target),
            places=3,
        )

    def test_current_buoy_model_classes_are_collision_targets(self):
        expected_classes = {
            "red_buoy",
            "green_buoy",
            "black_buoy",
            "orange_buoy",
            "yellow_buoy",
        }

        self.assertTrue(expected_classes.issubset(task2.BUOY_MODEL_TYPES))
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

    def test_port_side_risk_stands_on_then_uses_starboard_gps_target(self):
        self._update(5.0, -25.0, 10.0)
        self._update(4.7, -25.0, 10.3)
        self._update(4.4, -25.0, 10.6)

        self.assertEqual(task2.MissionState.STAND_ON, self.mission.state)
        self.assertIsNone(self.mission.avoidance_target)

        self._update(4.4, -25.0, 13.2, record=False)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("starboard", self.mission.avoidance_target["side"])
        self.assertTrue(self.topics.position_target_pub.messages)

    def test_avoidance_resumes_same_waypoint_only_after_temporary_target(self):
        self._update(6.0, 0.0, 10.0)
        self._update(5.0, 0.0, 10.3)
        self._update(4.0, 0.0, 10.6)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        target = dict(self.mission.avoidance_target)

        self._refresh_sensors(11.5)
        self.mission.update([], now=11.5)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)

        original_distance = task2.calculate_gps_distance
        task2.calculate_gps_distance = lambda lat1, lon1, lat2, lon2: (
            0.0
            if lat2 == target["lat"] and lon2 == target["lon"]
            else 100.0
        )
        self.addCleanup(setattr, task2, "calculate_gps_distance", original_distance)

        self._refresh_sensors(12.0)
        self.mission.update([], now=12.0)
        self.assertEqual(task2.MissionState.NAVIGATING, self.mission.state)
        self.assertIsNone(self.mission.avoidance_target)
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

    def test_avoidance_aligns_with_temporary_target(self):
        self.mission.aligned_target_key = ("WP0", 10.0, 20.0)

        self._update(6.0, 0.0, 10.0)
        self._update(5.0, 0.0, 10.3)
        self._update(4.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual(
            "starboard avoidance WP",
            self.mission.aligned_target_key[0],
        )

    def test_estimates_relative_and_true_vessel_speed_and_course(self):
        base_latitude = 1.0
        for index, now in enumerate((10.0, 10.5, 11.0)):
            elapsed = now - 10.0
            own_north_m = 0.4 * elapsed
            target_north_m = 10.0 - 0.6 * elapsed
            relative_distance_m = target_north_m - own_north_m
            latitude = base_latitude + math.degrees(
                own_north_m / task2.EARTH_RADIUS_M
            )
            self.mission.update_gps(latitude, 1.0, now=now)
            self.mission.update_heading(0.0, now=now)
            self.mission.update_bridge_state(True, True, "GUIDED", now=now)
            self.mission.update(
                [self._vessel(relative_distance_m, 0.0, track_id=42)],
                now=now,
                frame_id=100 + index,
                camera_timestamp_ms=int(now * 1000),
            )

        estimate = self.mission.latest_kinematics
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(1.0, estimate.relative_speed_mps, places=3)
        self.assertAlmostEqual(180.0, abs(estimate.relative_course_deg), places=3)
        self.assertAlmostEqual(0.6, estimate.true_speed_mps, places=3)
        self.assertAlmostEqual(180.0, estimate.true_course_deg, places=3)

    def test_writes_timestamped_vessel_kinematics_csv(self):
        captured = []
        self.mission.kinematics_callback = (
            lambda observation, kinematics, assessment, frame_id,
            camera_timestamp_ms: captured.append(
                (
                    observation,
                    kinematics,
                    assessment,
                    frame_id,
                    camera_timestamp_ms,
                )
            )
        )
        for index, (distance, now) in enumerate(
            ((10.0, 10.0), (9.5, 10.5), (9.0, 11.0))
        ):
            self._refresh_sensors(now)
            self.mission.update(
                [self._vessel(distance, 0.0, track_id=9)],
                now=now,
                frame_id=200 + index,
                camera_timestamp_ms=int(now * 1000),
            )

        self.assertEqual(3, len(captured))
        observation, estimate, assessment, frame_id, camera_timestamp_ms = captured[-1]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary_dir:
            recorder = task2.VesselKinematicsCsvRecorder(
                temporary_dir,
                run_name="kinematics.csv",
            )
            recorder.record(
                None,
                None,
                task2.CollisionAssessment(False, "no_vessel"),
                frame_id=199,
                camera_timestamp_ms=9500,
            )
            recorder.record(
                observation,
                estimate,
                assessment,
                frame_id=frame_id,
                camera_timestamp_ms=camera_timestamp_ms,
            )
            recorder.close()

            with recorder.path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(2, len(rows))
        self.assertTrue(rows[0]["system_timestamp_utc"])
        self.assertEqual("9500", rows[0]["camera_timestamp_ms"])
        self.assertEqual("199", rows[0]["frame_id"])
        self.assertEqual("0", rows[0]["detected"])
        for field in task2.KINEMATICS_CSV_FIELDS[4:]:
            self.assertEqual(0.0, float(rows[0][field]))

        self.assertEqual("11000", rows[1]["camera_timestamp_ms"])
        self.assertEqual("202", rows[1]["frame_id"])
        self.assertEqual("1", rows[1]["detected"])
        self.assertEqual("9", rows[1]["track_id"])
        self.assertAlmostEqual(1.0, float(rows[1]["relative_speed_mps"]), places=3)
        self.assertAlmostEqual(1.0, float(rows[1]["true_speed_mps"]), places=3)
        self.assertAlmostEqual(
            180.0,
            float(rows[1]["true_course_deg"]),
            places=3,
        )

    def test_missing_vessel_emits_zero_kinematics_sample(self):
        captured = []
        self.mission.kinematics_callback = lambda *values: captured.append(values)

        self.mission.update(
            [],
            now=10.0,
            frame_id=300,
            camera_timestamp_ms=12000,
        )

        self.assertEqual(1, len(captured))
        observation, estimate, assessment, frame_id, timestamp_ms = captured[0]
        self.assertIsNone(observation)
        self.assertIsNone(estimate)
        self.assertEqual("no_vessel", assessment.reason)
        self.assertEqual(300, frame_id)
        self.assertEqual(12000, timestamp_ms)


if __name__ == "__main__":
    unittest.main()
