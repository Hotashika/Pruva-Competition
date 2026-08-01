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

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

    class Vector:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:
        def __init__(self):
            self.linear = Vector()
            self.angular = Vector()

    geometry_msgs_msg.Twist = Twist
    geometry_msgs.msg = geometry_msgs_msg
    sys.modules["geometry_msgs"] = geometry_msgs
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

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
    utilities.calculate_angle_error_deg = (
        lambda target, current:
        (float(target) - float(current) + 180.0) % 360.0 - 180.0
    )
    utilities.calculate_bearing = lambda *args, **kwargs: 0.0
    utilities.calculate_gps_distance = lambda lat1, lon1, lat2, lon2: 100.0
    utilities.call_set_mode = lambda *args, **kwargs: True
    utilities.call_trigger_service = lambda *args, **kwargs: True
    utilities.create_mission_clients = lambda node: None
    utilities.create_mission_topics = lambda *args, **kwargs: types.SimpleNamespace(
        cmd_vel_pub=StubPublisher(),
        position_target_pub=StubPublisher(),
    )
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
    "geometry_msgs",
    "geometry_msgs.msg",
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
        self.task2_velocity_pub = FakePublisher()


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
    def _buoy(color, distance, angle, *, bbox=None, track_id=None):
        detection = {
            "type": "buoy",
            "class": f"{color}_buoy",
            "distance": distance,
            "Buoy angle: ": angle,
        }
        if bbox is not None:
            detection["bbox"] = bbox
        if track_id is not None:
            detection["track_id"] = track_id
        return detection

    @staticmethod
    def _depth_obstacle(
            distance,
            angle,
            *,
            forward_m=None,
            lateral_m=None,
            width_m=None,
    ):
        detection = {
            "type": "depth_obstacle",
            "class": "surface_obstacle_candidate",
            "distance": distance,
            "angle": angle,
        }
        if forward_m is not None:
            detection["forward_m"] = forward_m
        if lateral_m is not None:
            detection["lateral_m"] = lateral_m
        if width_m is not None:
            detection["width_m"] = width_m
        return detection

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

    def test_task2_node_publishes_kinematics_without_starting_a_csv(self):
        node = task2.Task2Node()

        self.assertIsNone(node.kinematics_recorder)
        self.assertIsNotNone(node.task.kinematics_callback)

    def test_completed_mission_disarms_before_manual_without_arming(self):
        events = []

        class CompletionNode:
            mission_topics = types.SimpleNamespace(cmd_vel_pub=object())
            mission_clients = types.SimpleNamespace(
                disarm_client=object(),
                set_mode_client=object(),
            )

            def get_logger(self):
                return FakeLogger()

            def wait_for_vehicle_state(self, **expected):
                events.append(("wait", expected))
                return True

        original_stop_vehicle = task2.stop_vehicle
        original_call_trigger_service = task2.call_trigger_service
        original_call_set_mode = task2.call_set_mode
        self.addCleanup(setattr, task2, "stop_vehicle", original_stop_vehicle)
        self.addCleanup(
            setattr,
            task2,
            "call_trigger_service",
            original_call_trigger_service,
        )
        self.addCleanup(setattr, task2, "call_set_mode", original_call_set_mode)

        task2.stop_vehicle = lambda publisher: events.append(("stop", publisher))
        task2.call_trigger_service = (
            lambda node, client, name: events.append(("trigger", name)) or True
        )
        task2.call_set_mode = (
            lambda node, client, mode: events.append(("mode", mode)) or True
        )

        completed = task2.finalize_completed_mission(CompletionNode())

        self.assertTrue(completed)
        self.assertEqual("stop", events[0][0])
        self.assertEqual(("trigger", "DISARM"), events[1])
        self.assertEqual(
            ("wait", {"expected_armed": False, "timeout_sec": 6.0}),
            events[2],
        )
        self.assertEqual(("mode", "MANUAL"), events[3])
        self.assertEqual(
            (
                "wait",
                {
                    "expected_mode": "MANUAL",
                    "expected_armed": False,
                    "timeout_sec": 6.0,
                },
            ),
            events[4],
        )
        self.assertFalse(any(event[1:] == ("ARM",) for event in events))

    def test_builds_manual_recorder_kinematics_payload(self):
        observation = task2.VesselObservation(
            timestamp=10.0,
            camera_timestamp_ms=5000,
            frame_id=42,
            track_id=7,
            distance_m=4.2,
            angle_deg=-3.0,
            forward_m=4.19,
            starboard_m=-0.22,
            latitude=41.0123,
            longitude=29.0456,
            heading_deg=90.0,
        )
        kinematics = task2.VesselKinematics(
            relative_course_deg=180.0,
            relative_speed_mps=0.8,
            true_course_deg=175.0,
            true_speed_mps=0.6,
        )
        assessment = task2.CollisionAssessment(
            True,
            "unsafe_cpa",
            closing_rate_mps=0.5,
            tcpa_sec=3.0,
            dcpa_m=1.2,
        )

        payload = task2.build_kinematics_payload(
            observation,
            kinematics,
            assessment,
        )

        self.assertEqual(42, payload["frame_id"])
        self.assertEqual(5000, payload["camera_timestamp_ms"])
        self.assertEqual(41.0123, payload["latitude_deg"])
        self.assertEqual(0.8, payload["relative_speed_mps"])
        self.assertEqual(1, payload["collision_risk"])
        self.assertEqual("unsafe_cpa", payload["collision_reason"])

    def test_receding_vessel_does_not_trigger_avoidance(self):
        self._update(6.0, 0.0, 10.0)
        self._update(6.5, 0.0, 10.3)
        self._update(7.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.NAVIGATING, self.mission.state)
        self.assertIsNone(self.mission.avoidance_phase)

    def test_collision_target_is_monitored_only_within_eight_metres(self):
        at_limit = self.mission._nearest_vessel([
            self._buoy("red", 8.0, 0.0),
        ])
        beyond_limit = self.mission._nearest_vessel([
            self._buoy("red", 8.01, 0.0),
        ])

        self.assertIsNotNone(at_limit)
        self.assertIsNone(beyond_limit)

    def test_closing_target_waits_until_three_metres_before_avoidance(self):
        self._update(5.0, 0.0, 10.0)
        self._update(4.0, 0.0, 10.3)
        self._update(3.01, 0.0, 10.6)

        self.assertEqual(task2.MissionState.NAVIGATING, self.mission.state)

        self._update(3.0, 0.0, 10.9)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)

    def test_head_on_collision_risk_starts_fixed_starboard_velocity(self):
        self._update(5.0, 0.0, 10.0)
        self._update(4.0, 0.0, 10.3)
        self._update(3.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("starboard", self.mission.avoidance_phase)
        self.assertEqual(0.0, self.mission.avoidance_entry_heading)
        velocity = self.topics.task2_velocity_pub.messages[-1]
        self.assertAlmostEqual(
            task2.TASK_TARGET_SPEED_M_S,
            math.hypot(velocity.linear.x, velocity.linear.y),
            places=6,
        )
        self.assertEqual([], self.topics.position_target_pub.messages)
        self.assertAlmostEqual(
            velocity.linear.x,
            velocity.linear.y,
            places=6,
        )

    def test_avoidance_bearing_is_always_right_of_entry_heading(self):
        for heading in (0.0, 90.0, 225.0):
            with self.subTest(heading=heading):
                self.mission.avoidance_entry_heading = heading
                self.mission.avoidance_phase = "starboard"
                self.mission._publish_avoidance_velocity()
                velocity = self.topics.task2_velocity_pub.messages[-1]
                actual_bearing = math.degrees(math.atan2(
                    velocity.linear.y,
                    velocity.linear.x,
                )) % 360.0
                expected_bearing = (
                    heading + task2.AVOIDANCE_STARBOARD_ANGLE_DEG
                ) % 360.0
                self.assertAlmostEqual(expected_bearing, actual_bearing)

    def test_closing_buoy_is_used_as_collision_target(self):
        for distance, now in ((5.0, 10.0), (4.0, 10.3), (3.0, 10.6)):
            self._refresh_sensors(now)
            self.mission.update(
                [self._buoy("red", distance, 0.0)],
                now=now,
                record_observation=True,
            )

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("starboard", self.mission.avoidance_phase)

    def test_closing_green_buoy_starts_fixed_port_velocity(self):
        for distance, now in ((5.0, 10.0), (4.0, 10.3), (3.0, 10.6)):
            self._refresh_sensors(now)
            self.mission.update(
                [self._buoy("green", distance, 0.0, track_id=21)],
                now=now,
                record_observation=True,
            )

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("port", self.mission.avoidance_phase)
        velocity = self.topics.task2_velocity_pub.messages[-1]
        actual_bearing = math.degrees(math.atan2(
            velocity.linear.y,
            velocity.linear.x,
        )) % 360.0
        self.assertAlmostEqual(
            360.0 - task2.AVOIDANCE_PORT_ANGLE_DEG,
            actual_bearing,
        )

    def test_nearest_distinct_obstacle_has_priority_over_farther_buoy(self):
        nearest = self.mission._nearest_vessel([
            self._buoy("red", 4.0, 15.0),
            self._depth_obstacle(2.0, -20.0),
        ])

        self.assertEqual("depth_obstacle", nearest["detector_type"])
        self.assertAlmostEqual(2.0, nearest["distance"])

    def test_buoy_semantics_override_matching_generic_depth_duplicate(self):
        nearest = self.mission._nearest_vessel([
            self._buoy(
                "green",
                2.9,
                -4.0,
                bbox=[100, 80, 180, 220],
                track_id=21,
            ),
            {
                **self._depth_obstacle(3.0, -3.0),
                "bbox": [110, 100, 170, 215],
            },
        ])

        self.assertEqual("buoy", nearest["detector_type"])
        self.assertEqual("green_buoy", nearest["model_class"])
        self.assertEqual(21, nearest["track_id"])

    def test_green_avoidance_without_track_id_ignores_new_closer_red_buoy(self):
        for distance, now in ((5.0, 10.0), (4.0, 10.3), (3.0, 10.6)):
            self._refresh_sensors(now)
            self.mission.update(
                [self._buoy("green", distance, 0.0)],
                now=now,
            )

        self._refresh_sensors(10.7)
        self.mission.update(
            [
                self._buoy("red", 1.0, 20.0),
                self._buoy("green", 2.8, 1.0),
            ],
            now=10.7,
        )

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("port", self.mission.avoidance_phase)
        self.assertIsNone(self.mission.avoiding_track_id)
        self.assertEqual(
            "green_buoy",
            self.mission.active_obstacle_reference["model_class"],
        )

    def test_active_buoy_can_fallback_to_matching_depth_obstacle(self):
        reference = self.mission._normalized_vessel(
            self._buoy("green", 3.0, 0.0, track_id=21)
        )
        self.mission.avoiding_track_id = 21
        self.mission.active_obstacle_reference = reference
        self.mission.active_obstacle_bearing_deg = 0.0

        matched = self.mission._matching_avoidance_vessel([
            self._depth_obstacle(2.8, 2.0),
            self._depth_obstacle(1.0, 80.0),
        ])

        self.assertEqual("depth_obstacle", matched["detector_type"])
        self.assertAlmostEqual(2.8, matched["distance"])

    def test_task2_red_and_green_buoys_are_collision_targets(self):
        expected_classes = {"red_buoy", "green_buoy"}

        self.assertTrue(expected_classes.issubset(task2.BUOY_MODEL_TYPES))
        for model_class in expected_classes:
            with self.subTest(model_class=model_class):
                self.assertTrue(
                    self.mission._is_vessel(
                        {"type": "buoy", "class": model_class}
                    )
                )

        for model_class in ("black_buoy", "orange_buoy", "yellow_buoy"):
            with self.subTest(ignored_model_class=model_class):
                self.assertFalse(
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

    def test_depth_obstacle_is_used_as_collision_target(self):
        self.assertTrue(
            self.mission._is_vessel({
                "type": "depth_obstacle",
                "class": "surface_obstacle_candidate",
            })
        )

    def test_metric_fusion_obstacles_are_collision_targets(self):
        for detection_type in ("fused_obstacle", "seg_depth_obstacle"):
            with self.subTest(detection_type=detection_type):
                self.assertTrue(
                    self.mission._is_vessel({
                        "type": detection_type,
                        "class": "surface_obstacle_candidate",
                        "distance": 3.5,
                    })
                )

    def test_metric_geometry_is_preserved_for_clearance_detection(self):
        vessel = self.mission._normalized_vessel(
            self._depth_obstacle(
                3.0,
                0.0,
                forward_m=4.0,
                lateral_m=-0.5,
                width_m=2.0,
            )
        )

        self.assertAlmostEqual(4.0, vessel["forward_m"])
        self.assertAlmostEqual(-0.5, vessel["starboard_m"])
        self.assertAlmostEqual(2.0, vessel["width_m"])
        self.assertFalse(self.mission._obstacle_is_behind(vessel))

    def test_active_avoidance_keeps_tracking_the_same_vessel(self):
        vessel = self.mission._normalized_vessel(
            self._vessel(4.0, 0.0, track_id=7)
        )
        self.mission.avoiding_track_id = 7
        self.mission.active_obstacle_reference = vessel

        matched = self.mission._matching_avoidance_vessel([
            self._vessel(1.5, 0.0, track_id=8),
            self._vessel(3.8, 4.0, track_id=7),
        ])

        self.assertEqual(7, matched["track_id"])
        self.assertAlmostEqual(3.8, matched["distance"])

    def test_geometry_matching_survives_heading_change_without_track_id(self):
        vessel = self.mission._normalized_vessel(
            self._depth_obstacle(4.0, 0.0)
        )
        self.mission.active_obstacle_reference = vessel
        self.mission.active_obstacle_bearing_deg = 0.0
        self.mission.update_heading(30.0, now=10.1)

        matched = self.mission._matching_avoidance_vessel([
            self._depth_obstacle(2.0, 0.0),
            self._depth_obstacle(4.0, -30.0),
        ])

        self.assertAlmostEqual(4.0, matched["distance"])
        self.assertAlmostEqual(-30.0, matched["angle"])

    def test_avoidance_does_not_create_or_publish_a_gps_target(self):
        self._update(5.0, 0.0, 10.0)
        self._update(4.0, 0.0, 10.3)
        self._update(3.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual([], self.topics.position_target_pub.messages)
        self.assertFalse(hasattr(self.mission, "avoidance_target"))

    def test_visual_only_segment_is_not_a_collision_target(self):
        self.assertFalse(
            self.mission._is_vessel({
                "type": "visual_obstacle_candidate",
                "class": "surface_obstacle_candidate",
                "distance": None,
            })
        )

    def test_port_side_risk_stands_on_then_uses_timed_starboard_leg(self):
        self._update(3.2, -25.0, 10.0)
        self._update(3.1, -25.0, 10.3)
        self._update(3.0, -25.0, 10.6)

        self.assertEqual(task2.MissionState.STAND_ON, self.mission.state)
        self.assertIsNone(self.mission.avoidance_phase)

        self._update(3.0, -25.0, 13.2, record=False)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("starboard", self.mission.avoidance_phase)
        self.assertTrue(self.topics.task2_velocity_pub.messages)
        self.assertEqual([], self.topics.position_target_pub.messages)

    def test_avoidance_runs_starboard_then_forward_before_rejoining(self):
        self._update(5.0, 0.0, 10.0)
        self._update(4.0, 0.0, 10.3)
        self._update(3.0, 0.0, 10.6)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("starboard", self.mission.avoidance_phase)

        transition_time = 10.6 + task2.AVOIDANCE_STARBOARD_DURATION_SEC
        self._refresh_sensors(transition_time - 0.01)
        self.mission.update([], now=transition_time - 0.01)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("starboard", self.mission.avoidance_phase)

        self._refresh_sensors(transition_time)
        self.mission.update([], now=transition_time)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("forward", self.mission.avoidance_phase)
        forward_velocity = self.topics.task2_velocity_pub.messages[-1]
        self.assertAlmostEqual(
            task2.TASK_TARGET_SPEED_M_S,
            forward_velocity.linear.x,
            places=6,
        )
        self.assertAlmostEqual(0.0, forward_velocity.linear.y, places=6)
        self.assertEqual(0, self.mission.current_target_index)

        before_minimum = (
            transition_time + task2.AVOIDANCE_FORWARD_MIN_DURATION_SEC - 0.01
        )
        self._refresh_sensors(before_minimum)
        self.mission.update([], now=before_minimum)
        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)

        complete_time = (
            transition_time + task2.AVOIDANCE_FORWARD_MIN_DURATION_SEC
        )
        self._refresh_sensors(complete_time)
        self.mission.update([], now=complete_time)
        self.assertEqual(task2.MissionState.NAVIGATING, self.mission.state)
        self.assertIsNone(self.mission.avoidance_phase)
        self.assertEqual(0, self.mission.current_target_index)

    def test_forward_leg_extends_while_obstacle_remains_ahead(self):
        self._update(5.0, 0.0, 10.0)
        self._update(4.0, 0.0, 10.3)
        self._update(3.0, 0.0, 10.6)
        transition_time = 10.6 + task2.AVOIDANCE_STARBOARD_DURATION_SEC

        self._refresh_sensors(transition_time)
        self.mission.update(
            [self._vessel(3.0, -45.0)],
            now=transition_time,
        )
        forward_complete = (
            transition_time + task2.AVOIDANCE_FORWARD_MIN_DURATION_SEC
        )
        self._refresh_sensors(forward_complete)
        self.mission.update(
            [self._vessel(2.0, -80.0)],
            now=forward_complete,
        )

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertEqual("forward", self.mission.avoidance_phase)

        self._refresh_sensors(forward_complete + 0.1)
        self.mission.update(
            [self._vessel(2.0, 180.0)],
            now=forward_complete + 0.1,
        )
        self.assertEqual(task2.MissionState.NAVIGATING, self.mission.state)

    def test_timed_avoidance_timeout_enters_failsafe(self):
        self._update(5.0, 0.0, 10.0)
        self._update(4.0, 0.0, 10.3)
        self._update(3.0, 0.0, 10.6)
        timeout_time = 10.6 + task2.AVOIDANCE_TIMEOUT_SEC

        self._refresh_sensors(timeout_time)
        self.mission.update(
            [self._vessel(2.0, -80.0)],
            now=timeout_time,
        )

        self.assertEqual(task2.MissionState.FAILSAFE, self.mission.state)
        self.assertTrue(self.mission.hold_mode_requested)
        self.assertEqual(
            ("cmd_vel", 0.0, 0.0),
            self.topics.cmd_vel_pub.messages[-1],
        )

    def test_next_waypoint_waits_then_uses_task2_velocity(self):
        self.mission.waypoints = [
            {"lat": 10.0, "lon": 20.0, "alt": 0.0, "seq": 1},
            {"lat": 11.0, "lon": 21.0, "alt": 0.0, "seq": 2},
        ]
        original_distance = task2.calculate_gps_distance

        def fake_distance(lat1, lon1, lat2, lon2):
            return 0.5 if float(lat2) == 10.0 else 10.0

        task2.calculate_gps_distance = fake_distance
        self.addCleanup(setattr, task2, "calculate_gps_distance", original_distance)

        self.mission.update([], now=10.0)
        self.assertEqual(1, self.mission.current_target_index)
        self.assertEqual(("cmd_vel", 0.0, 0.0), self.topics.cmd_vel_pub.messages[-1])

        self._refresh_sensors(10.5)
        self.mission.update([], now=10.5)
        self.assertEqual([], self.topics.task2_velocity_pub.messages)
        self.assertEqual([], self.topics.position_target_pub.messages)

        self._refresh_sensors(10.8)
        self.mission.update([], now=10.8)
        velocity = self.topics.task2_velocity_pub.messages[-1]
        self.assertAlmostEqual(
            task2.TASK_TARGET_SPEED_M_S,
            math.hypot(velocity.linear.x, velocity.linear.y),
            places=6,
        )
        self.assertEqual([], self.topics.position_target_pub.messages)

    def test_avoidance_uses_only_task2_velocity_topic(self):
        self.mission.aligned_target_key = ("WP0", 10.0, 20.0)

        self._update(5.0, 0.0, 10.0)
        self._update(4.0, 0.0, 10.3)
        self._update(3.0, 0.0, 10.6)

        self.assertEqual(task2.MissionState.AVOIDING, self.mission.state)
        self.assertIsNone(self.mission.aligned_target_key)
        velocity = self.topics.task2_velocity_pub.messages[-1]
        self.assertAlmostEqual(
            task2.TASK_TARGET_SPEED_M_S,
            math.hypot(velocity.linear.x, velocity.linear.y),
            places=6,
        )
        self.assertAlmostEqual(velocity.linear.x, velocity.linear.y, places=6)
        self.assertEqual([], self.topics.position_target_pub.messages)

    def test_sharp_turn_reduces_only_task2_metric_velocity(self):
        original_bearing = task2.calculate_bearing
        task2.calculate_bearing = lambda *args: 90.0
        self.addCleanup(
            setattr,
            task2,
            "calculate_bearing",
            original_bearing,
        )

        commanded_speed = self.mission._publish_task2_velocity_target(
            1.0,
            2.0,
        )
        velocity = self.topics.task2_velocity_pub.messages[-1]

        self.assertAlmostEqual(
            task2.TASK_TARGET_SPEED_M_S
            * task2.TASK_MIN_TURN_SPEED_FRACTION,
            commanded_speed,
            places=6,
        )
        self.assertAlmostEqual(0.0, velocity.linear.x, places=6)
        self.assertAlmostEqual(commanded_speed, velocity.linear.y, places=6)
        self.assertEqual([], self.topics.cmd_vel_pub.messages)
        self.assertEqual([], self.topics.position_target_pub.messages)

    def test_estimates_relative_and_true_vessel_speed_and_course(self):
        base_latitude = 1.0
        for index, now in enumerate((10.0, 10.5, 11.0)):
            elapsed = now - 10.0
            own_north_m = 0.4 * elapsed
            target_north_m = 8.0 - 0.6 * elapsed
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
            ((8.0, 10.0), (7.5, 10.5), (7.0, 11.0))
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
