import importlib
import math
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


commands = []


def publish_cmd_vel(pub, linear_x, angular_z):
    commands.append((float(linear_x), float(angular_z)))


def stop_vehicle(pub, repeat_count=1):
    commands.append((0.0, 0.0))


def calculate_gps_distance(lat1, lon1, lat2, lon2):
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat1))
    return math.hypot(north, east)


def calculate_bearing(lat1, lon1, lat2, lon2):
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat1))
    return math.degrees(math.atan2(east, north)) % 360.0


fake_mavlink = types.ModuleType("utils.mavlink_utilities")
fake_mavlink.publish_cmd_vel = publish_cmd_vel
fake_mavlink.stop_vehicle = stop_vehicle
fake_mavlink.calculate_gps_distance = calculate_gps_distance
fake_mavlink.calculate_bearing = calculate_bearing
fake_mavlink.create_mission_topics = lambda *args, **kwargs: None
fake_mavlink.create_mission_clients = lambda *args, **kwargs: None
fake_mavlink.wait_for_mission_services = lambda *args, **kwargs: None
sys.modules["utils.mavlink_utilities"] = fake_mavlink


class _RosNode:
    pass


class _Message:
    def __init__(self):
        self.data = None


class _Service:
    class Request:
        pass


fake_rclpy = types.ModuleType("rclpy")
fake_rclpy_node = types.ModuleType("rclpy.node")
fake_rclpy_node.Node = _RosNode
fake_rclpy.node = fake_rclpy_node
sys.modules.setdefault("rclpy", fake_rclpy)
sys.modules.setdefault("rclpy.node", fake_rclpy_node)

for module_name, message_names in (
    ("std_msgs.msg", ("Int32", "String")),
    ("sensor_msgs.msg", ("Imu",)),
):
    parent_name = module_name.split(".")[0]
    parent = types.ModuleType(parent_name)
    child = types.ModuleType(module_name)
    for message_name in message_names:
        setattr(child, message_name, _Message)
    setattr(parent, "msg", child)
    sys.modules.setdefault(parent_name, parent)
    sys.modules.setdefault(module_name, child)

for module_name, service_name in (
    ("mavros_msgs.srv", "SetMode"),
    ("std_srvs.srv", "Trigger"),
):
    parent_name = module_name.split(".")[0]
    parent = types.ModuleType(parent_name)
    child = types.ModuleType(module_name)
    setattr(child, service_name, _Service)
    setattr(parent, "srv", child)
    sys.modules.setdefault(parent_name, parent)
    sys.modules.setdefault(module_name, child)

arama = importlib.import_module("teknofest.missions.arama")
yaklasma = importlib.import_module("teknofest.missions.yaklasma")
carpma = importlib.import_module("teknofest.missions.carpma")
t3 = importlib.import_module("teknofest.missions.task3_kamikaze_engagement")


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


clock = FakeClock()
fake_time = types.SimpleNamespace(monotonic=clock.monotonic)
arama.time = fake_time
yaklasma.time = fake_time
carpma.time = fake_time
t3.time = fake_time


class Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    warn = warning

    def error(self, *args, **kwargs):
        pass


class Node:
    def get_logger(self):
        return Logger()


class Topics:
    cmd_vel_pub = object()


def detection(distance=9.0, angle=0.0):
    return [{
        "class": "red_buoy",
        "confidence": 0.95,
        "distance": distance,
        "Buoy angle: ": angle,
    }]


class Task3RealSensorFlowTests(unittest.TestCase):
    def setUp(self):
        commands.clear()
        clock.now = 100.0

    def _ready_task_manager(self):
        manager = t3.Task3KamikazeEngagement(
            Node(), Topics(), object(), "red_buoy",
        )
        manager.update_gps(41.0, 29.0)
        manager.update_heading(0.0)
        manager.update_imu(0.0, 0.0, 0.0, 9.81)
        manager.update_vision_timestamp()
        manager.update_bridge_state("connected=true,armed=true,mode=GUIDED")
        ok, _ = manager.start_mission()
        self.assertTrue(ok)
        return manager

    def _refresh_navigation_and_imu(self, manager):
        manager.update_gps(manager.current_lat, manager.current_lon)
        manager.update_heading(manager.current_heading)
        manager.update_imu(0.0, 0.0, 0.0, 9.81)

    def _manager_step(self, manager, detections, frame_id, seconds=0.1):
        clock.advance(seconds)
        self._refresh_navigation_and_imu(manager)
        manager.update_vision_timestamp()
        manager.update(detections, frame_id=frame_id)

    def test_manager_does_not_move_before_real_start(self):
        manager = t3.Task3KamikazeEngagement(
            Node(), Topics(), object(), "red_buoy",
        )
        manager.update_gps(41.0, 29.0)
        manager.update_heading(0.0)
        manager.update_imu(0.0, 0.0, 0.0, 9.81)
        manager.update_vision_timestamp()
        manager.update_bridge_state("connected=true,armed=false,mode=MANUAL")
        commands.clear()

        manager.update(detection(), frame_id=1)

        self.assertEqual(manager.state, t3.MissionState.INIT)
        self.assertEqual(commands, [])

    def test_manager_latches_home_at_start_not_at_power_on(self):
        manager = t3.Task3KamikazeEngagement(
            Node(), Topics(), object(), "red_buoy",
        )
        manager.update_heading(45.0)
        manager.update_gps(41.0, 29.0)
        manager.update_gps(41.0002, 29.0003)

        ok, _ = manager.start_mission()

        self.assertTrue(ok)
        self.assertEqual(manager.home_lat, 41.0002)
        self.assertEqual(manager.home_lon, 29.0003)

    def test_heading_updates_do_not_recount_the_last_gps_sample(self):
        manager = t3.Task3KamikazeEngagement(
            Node(), Topics(), object(), "red_buoy",
        )
        manager.update_gps(41.0, 29.0)
        gps_sequence = manager.arama.gps_update_sequence

        manager.update_heading(10.0)
        manager.update_heading(11.0)
        manager.update_heading(12.0)

        self.assertEqual(manager.arama.gps_update_sequence, gps_sequence)
        manager.update_gps(41.00001, 29.0)
        self.assertEqual(manager.arama.gps_update_sequence, gps_sequence + 1)

    def test_state_transitions_do_not_reuse_the_previous_camera_frame(self):
        manager = self._ready_task_manager()
        manager.arama.finished = True
        manager.arama.state = arama.SearchState.TARGET_FOUND
        manager.arama.last_processed_frame_id = 50

        manager.update(detection(4.0, 0.0), frame_id=50)

        self.assertEqual(manager.state, t3.MissionState.APPROACHING)
        self.assertEqual(len(manager.yaklasma.confirmations), 0)
        self.assertFalse(manager.yaklasma.target_lost)

        manager.update(detection(4.0, 0.0), frame_id=51)
        self.assertEqual(len(manager.yaklasma.confirmations), 1)

        manager.yaklasma.finished = True
        manager.yaklasma.state = yaklasma.ApproachState.DONE
        manager.yaklasma.last_processed_frame_id = 60
        manager.update(detection(1.0, 0.0), frame_id=60)

        self.assertEqual(manager.state, t3.MissionState.CARPMA)
        manager.update(detection(1.0, 0.0), frame_id=60)
        self.assertEqual(len(manager.carpma.confirm_frame_ids), 0)
        self.assertEqual(manager.carpma.state, carpma.CarpmaState.CAMERA_CONFIRM)

        manager.update(detection(1.0, 0.0), frame_id=61)
        self.assertEqual(len(manager.carpma.confirm_frame_ids), 1)

    def test_manager_failsafe_on_stale_camera_or_imu(self):
        manager = self._ready_task_manager()
        clock.advance(t3.VISION_TIMEOUT_SEC + 0.1)
        self._refresh_navigation_and_imu(manager)
        manager.update([], frame_id=1)
        self.assertEqual(manager.state, t3.MissionState.FAILSAFE)
        self.assertEqual(commands[-1], (0.0, 0.0))

        manager = self._ready_task_manager()
        clock.advance(t3.IMU_TIMEOUT_SEC + 0.1)
        manager.update_gps(41.0, 29.0)
        manager.update_heading(0.0)
        manager.update_vision_timestamp()
        manager.update([], frame_id=1)
        self.assertEqual(manager.state, t3.MissionState.FAILSAFE)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_manager_failsafe_on_stale_gps_or_heading(self):
        manager = self._ready_task_manager()
        clock.advance(t3.GPS_TIMEOUT_SEC + 0.1)
        manager.update_heading(0.0)
        manager.update_imu(0.0, 0.0, 0.0, 9.81)
        manager.update_vision_timestamp()
        manager.update([], frame_id=1)
        self.assertEqual(manager.state, t3.MissionState.FAILSAFE)
        self.assertEqual(commands[-1], (0.0, 0.0))

        manager = self._ready_task_manager()
        clock.advance(t3.HEADING_TIMEOUT_SEC + 0.1)
        manager.update_gps(41.0, 29.0)
        manager.update_imu(0.0, 0.0, 0.0, 9.81)
        manager.update_vision_timestamp()
        manager.update([], frame_id=1)
        self.assertEqual(manager.state, t3.MissionState.FAILSAFE)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_manager_failsafe_on_mode_or_arm_loss(self):
        manager = self._ready_task_manager()
        manager.update_bridge_state("connected=true,armed=true,mode=MANUAL")
        manager.update([], frame_id=1)
        self.assertEqual(manager.state, t3.MissionState.FAILSAFE)

        manager = self._ready_task_manager()
        manager.update_bridge_state("connected=true,armed=false,mode=GUIDED")
        manager.update([], frame_id=1)
        self.assertEqual(manager.state, t3.MissionState.FAILSAFE)

    def test_manager_runs_search_approach_and_three_impacts_end_to_end(self):
        manager = self._ready_task_manager()

        # 20 derece dön, dur ve yalnız sabit bakışta beş farklı kareyi say.
        self._manager_step(manager, [], 0)
        manager.update_heading(20.0)
        self._manager_step(manager, [], 1)
        self.assertEqual(manager.arama.state, arama.SearchState.HOLDING)
        for frame_id in range(2, 7):
            self._manager_step(manager, detection(4.0, 0.0), frame_id)
        self.assertTrue(manager.arama.finished)

        # Aramadan yaklaşmaya geç; hedef zaten 1 m içinde olduğunda ileri
        # segment zorlamadan beş yeni kamera karesiyle çarpma aşamasına gir.
        for frame_id in range(7, 12):
            self._manager_step(manager, detection(1.0, 0.0), frame_id)
        self.assertEqual(manager.state, t3.MissionState.CARPMA)

        # Kamera onayı sırasında gerçek IMU taban çizgisini doldur.
        for _ in range(carpma.BASELINE_WINDOW):
            manager.update_imu(0.0, 0.0, 0.0, 9.81)

        next_frame = 20
        for expected_hit in range(1, carpma.REQUIRED_HITS + 1):
            for _ in range(carpma.CAMERA_CONFIRM_FRAMES):
                self._manager_step(
                    manager,
                    detection(1.0, 0.0),
                    next_frame,
                )
                next_frame += 1
            self.assertEqual(manager.carpma.state, carpma.CarpmaState.STRIKING)

            for _ in range(carpma.IMPACT_CONSECUTIVE_SAMPLES):
                manager.update_imu(0.0, 9.81 + 8.0, 0.0, 0.0)
            self.assertEqual(manager.carpma.hit_count, expected_hit)

            if expected_hit == carpma.REQUIRED_HITS:
                break

            # Temastan sonra kamerada mesafe artışı gerçek geri çıkışı
            # doğrular; cooldown bitmeden yeni vuruş sayılmaz.
            manager.update_gps(
                manager.current_lat - 0.70 / 111_320.0,
                manager.current_lon,
            )
            self._manager_step(manager, detection(1.6, 0.0), next_frame)
            next_frame += 1
            self.assertEqual(manager.carpma.state, carpma.CarpmaState.COOLDOWN)
            self._manager_step(
                manager,
                detection(1.6, 0.0),
                next_frame,
                seconds=carpma.COOLDOWN_SEC + 0.1,
            )
            next_frame += 1
            self.assertEqual(manager.carpma.state, carpma.CarpmaState.CAMERA_CONFIRM)

        self._manager_step(manager, detection(1.0, 0.0), next_frame)
        self.assertEqual(manager.state, t3.MissionState.DONE)
        self.assertFalse(manager.mission_enabled)

    def test_search_accepts_five_distinct_frames_only_while_holding(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.state = arama.SearchState.TURNING
        mission.step_target_heading = 20.0
        mission.step_start_time = arama.time.monotonic()
        for frame_id in range(1, 6):
            mission.update(detection(), frame_id=frame_id)
        self.assertFalse(mission.finished)

        mission.update_heading(20.0)
        mission.state = arama.SearchState.HOLDING
        mission.hold_until = float("inf")
        for frame_id in range(6, 10):
            mission.update(detection(), frame_id=frame_id)
            self.assertFalse(mission.finished)
        mission.update(detection(), frame_id=10)
        self.assertTrue(mission.finished)

    def test_search_completes_360_degrees_in_18_twenty_degree_steps(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        frame_id = 0
        for step in range(1, arama.FULL_SCAN_STEPS + 1):
            mission.update([], frame_id=frame_id)
            self.assertEqual(mission.state, arama.SearchState.TURNING)
            self.assertAlmostEqual(mission.step_target_heading, (step * 20.0) % 360.0)

            # Saat yönü dönüşte skid-steer mikserinin dış motoru sürmesi için
            # küçük pozitif taban itki ve pozitif yaw birlikte gönderilir.
            mission.update([], frame_id=frame_id)
            self.assertGreater(commands[-1][0], 0.0)
            self.assertGreater(commands[-1][1], 0.0)
            mission.update_heading((step * 20.0) % 360.0)
            clock.advance(0.1)
            mission.update([], frame_id=frame_id)
            self.assertEqual(mission.state, arama.SearchState.HOLDING)
            self.assertAlmostEqual(mission.hold_until - clock.monotonic(), 5.0)

            clock.advance(5.0)
            frame_id += 1
            mission.update([], frame_id=frame_id)
            self.assertEqual(mission.state, arama.SearchState.START_STEP)

        self.assertEqual(mission.completed_steps, 18)
        mission.update([], frame_id=frame_id + 1)
        self.assertEqual(mission.completed_steps, 18)
        self.assertEqual(mission.state, arama.SearchState.RELOCATING)

        # Aynı kontrol döngüsündeki tekrarlı okumalar değil, üç farklı gerçek
        # GPS örneği 2 m yer değiştirmeyi doğrulamalı.
        relocated_lat = 41.0 + 2.10 / 111_320.0
        for _ in range(arama.RELOCATION_CONFIRM_GPS_SAMPLES):
            mission.update_gps(relocated_lat, 29.0, 0.0)
            clock.advance(0.2)
            mission.update([], frame_id=frame_id + 2)

        self.assertEqual(mission.completed_steps, 0)
        self.assertEqual(mission.state, arama.SearchState.START_STEP)
        mission.update([], frame_id=frame_id + 3)
        self.assertEqual(mission.state, arama.SearchState.TURNING)

    def test_search_tolerance_does_not_accumulate_across_full_scan(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for step in range(arama.FULL_SCAN_STEPS):
            mission.update([], frame_id=step)
            reached = (
                mission.step_target_heading
                - (arama.HEADING_TOLERANCE_DEG - 0.1)
            ) % 360.0
            mission.update_heading(reached)
            clock.advance(0.1)
            mission.update([], frame_id=step)
            clock.advance(arama.STEP_HOLD_SEC)
            mission.update([], frame_id=100 + step)

        # Son hedef ilk heading'in 360 derece sonraki karsiligidir. Tolerans
        # her adimda birikmemeli; yalnız son adimin toleransi kalabilir.
        final_error = abs(mission._angle_error(0.0, mission.current_heading_deg))
        self.assertLessEqual(final_error, arama.HEADING_TOLERANCE_DEG)

    def test_search_turns_clockwise_across_zero_degree_boundary(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 350.0)

        mission.update([], frame_id=1)
        self.assertAlmostEqual(mission.step_target_heading, 10.0)
        mission.update([], frame_id=1)
        self.assertGreater(commands[-1][0], 0.0)
        self.assertGreater(commands[-1][1], 0.0)

        mission.update_heading(10.0)
        clock.advance(0.1)
        mission.update([], frame_id=2)
        self.assertEqual(mission.state, arama.SearchState.HOLDING)

    def test_search_rejects_frames_if_heading_drifts_during_hold(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.update([], frame_id=0)
        mission.update_heading(20.0)
        clock.advance(0.1)
        mission.update([], frame_id=0)
        self.assertEqual(mission.state, arama.SearchState.HOLDING)

        mission.update_heading(30.0)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(), frame_id=frame_id)

        self.assertFalse(mission.finished)
        self.assertEqual(mission.state, arama.SearchState.TURNING)
        self.assertEqual(mission.completed_steps, 0)
        self.assertEqual(len(mission.confirmations), 0)

        # Aynı 20 derece hedefine geri dönülünce bu açı ikinci kez sayılmamalı.
        mission.update_heading(20.0)
        clock.advance(0.1)
        mission.update([], frame_id=20)
        self.assertEqual(mission.state, arama.SearchState.HOLDING)
        self.assertEqual(mission.completed_steps, 0)
        clock.advance(arama.STEP_HOLD_SEC)
        mission.update([], frame_id=21)
        self.assertEqual(mission.completed_steps, 1)

    def test_search_relocation_timeout_stops_and_fails(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.completed_steps = arama.FULL_SCAN_STEPS
        mission.update([], frame_id=1)
        self.assertEqual(mission.state, arama.SearchState.RELOCATING)

        clock.advance(arama.RELOCATION_TIMEOUT_SEC + 0.1)
        mission.update([], frame_id=2)
        self.assertTrue(mission.failed)
        self.assertEqual(mission.state, arama.SearchState.FAILED)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_search_relocation_requires_forward_not_lateral_gps_progress(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.completed_steps = arama.FULL_SCAN_STEPS
        mission.scan_origin_heading_deg = 0.0
        mission.update([], frame_id=1)
        self.assertEqual(mission.state, arama.SearchState.RELOCATING)

        lateral_lon = 29.0 + 2.2 / (111_320.0 * math.cos(math.radians(41.0)))
        for frame_id in range(2, 2 + arama.RELOCATION_CONFIRM_GPS_SAMPLES):
            mission.update_gps(41.0, lateral_lon, 0.0)
            clock.advance(0.2)
            mission.update([], frame_id=frame_id)
        self.assertEqual(mission.state, arama.SearchState.RELOCATING)

        forward_lat = 41.0 + 2.2 / 111_320.0
        for frame_id in range(10, 10 + arama.RELOCATION_CONFIRM_GPS_SAMPLES):
            mission.update_gps(forward_lat, 29.0, 0.0)
            clock.advance(0.2)
            mission.update([], frame_id=frame_id)
        self.assertEqual(mission.state, arama.SearchState.START_STEP)

    def test_search_stops_if_clockwise_heading_does_not_progress(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.update([], frame_id=1)
        self.assertEqual(mission.state, arama.SearchState.TURNING)

        mission.update([], frame_id=2)
        self.assertGreater(commands[-1][0], 0.0)
        clock.advance(arama.TURN_PROGRESS_TIMEOUT_SEC + 0.1)
        mission.update([], frame_id=3)

        self.assertTrue(mission.failed)
        self.assertEqual(mission.state, arama.SearchState.FAILED)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_approach_averages_five_frames_then_moves_one_third_straight(self):
        mission = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for frame_id, distance in enumerate((9.0, 9.1, 8.9, 9.0, 9.0), start=1):
            mission.update(detection(distance, 0.0), frame_id=frame_id)
        self.assertEqual(mission.state, yaklasma.ApproachState.MOVING_STRAIGHT)
        self.assertAlmostEqual(mission.segment_goal_m, 3.0, places=1)

        mission.update(detection(8.8, 0.0), frame_id=6)
        self.assertEqual(commands[-1][1], 0.0)
        self.assertGreater(commands[-1][0], 0.0)

    def test_approach_does_not_move_forward_if_initial_target_is_already_close(self):
        mission = yaklasma.YaklasmaGorevi(
            Node(), Topics(), "red_buoy", safe_stop_distance=1.0
        )
        mission.update_gps(41.0, 29.0, 0.0)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(0.8, 0.0), frame_id=frame_id)
        self.assertTrue(mission.finished)
        self.assertEqual(mission.state, yaklasma.ApproachState.DONE)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_approach_corrects_heading_while_moving_straight(self):
        mission = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(9.0, 0.0), frame_id=frame_id)
        self.assertEqual(mission.state, yaklasma.ApproachState.MOVING_STRAIGHT)

        mission.update_heading(10.0)
        clock.advance(0.1)
        mission.update(detection(8.8, 0.0), frame_id=6)
        self.assertGreater(commands[-1][0], 0.0)
        self.assertLess(commands[-1][1], 0.0)

    def test_approach_stops_immediately_on_a_fresh_missing_frame(self):
        mission = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(9.0, 0.0), frame_id=frame_id)
        self.assertEqual(mission.state, yaklasma.ApproachState.MOVING_STRAIGHT)

        clock.advance(0.1)
        mission.update([], frame_id=6)

        self.assertTrue(mission.should_return_to_search())
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_approach_requires_camera_distance_to_confirm_gps_progress(self):
        mission = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(9.0, 0.0), frame_id=frame_id)

        mission.update_gps(41.0 + 3.1 / 111_320.0, 29.0, 0.0)
        clock.advance(0.1)
        mission.update(detection(9.0, 0.0), frame_id=6)
        self.assertEqual(mission.state, yaklasma.ApproachState.CONFIRMING_RESULT)
        for frame_id in range(7, 12):
            clock.advance(0.1)
            mission.update(detection(9.0, 0.0), frame_id=frame_id)

        self.assertTrue(mission.should_return_to_search())
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_approach_stops_early_if_camera_reports_collision_distance(self):
        mission = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(3.0, 0.0), frame_id=frame_id)
        self.assertEqual(mission.state, yaklasma.ApproachState.MOVING_STRAIGHT)

        clock.advance(0.1)
        mission.update(detection(1.4, 0.0), frame_id=6)

        self.assertEqual(mission.state, yaklasma.ApproachState.CONFIRMING_RESULT)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_approach_rejects_reused_frame_id(self):
        mission = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for _ in range(10):
            mission.update(detection(), frame_id=1)
        self.assertEqual(mission.state, yaklasma.ApproachState.CONFIRMING_TARGET)
        self.assertEqual(len(mission.confirmations), 1)

    def test_complete_approach_reaches_collision_distance_with_real_gps_progress(self):
        mission = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        lat, lon = 41.0, 29.0
        mission.update_gps(lat, lon, 0.0)
        frame_id = 0

        def five_frames(distance):
            nonlocal frame_id
            for _ in range(5):
                frame_id += 1
                clock.advance(0.1)
                mission.update(detection(distance, 0.0), frame_id=frame_id)

        distance = 9.0
        five_frames(distance)
        for expected_distance in (6.0, 4.0, 2.67, 1.78, 1.19):
            self.assertEqual(mission.state, yaklasma.ApproachState.MOVING_STRAIGHT)
            north_m = mission.segment_goal_m + 0.05
            lat += north_m / 111_320.0
            distance = expected_distance
            mission.update_gps(lat, lon, 0.0)
            frame_id += 1
            clock.advance(0.1)
            mission.update(detection(distance, 0.0), frame_id=frame_id)
            self.assertEqual(mission.state, yaklasma.ApproachState.CONFIRMING_RESULT)
            five_frames(distance)

        self.assertTrue(mission.finished)
        self.assertEqual(mission.state, yaklasma.ApproachState.DONE)

    def test_collision_requires_three_distinct_physical_imu_impacts(self):
        mission = carpma.CarpmaGorevi(Node(), Topics(), "red_buoy")
        lat, lon = 41.0, 29.0
        mission.update_gps(lat, lon, 0.0)
        frame_id = 0

        for _ in range(carpma.BASELINE_WINDOW):
            mission.update_imu(0.0, 0.0, 9.81)

        for expected_hit in range(1, 4):
            for _ in range(carpma.CAMERA_CONFIRM_FRAMES):
                frame_id += 1
                clock.advance(0.1)
                mission.update(detection(1.0, 0.0), frame_id=frame_id)
                mission.update_imu(0.0, 0.0, 9.81)
            self.assertEqual(mission.state, carpma.CarpmaState.STRIKING)

            mission.update(detection(1.0, 0.0), frame_id=frame_id + 1)
            mission.update_imu(20.0, 0.0, 0.0)
            mission.update_imu(20.0, 0.0, 0.0)
            self.assertEqual(mission.hit_count, expected_hit)

            if expected_hit == carpma.REQUIRED_HITS:
                break

            self.assertEqual(mission.state, carpma.CarpmaState.BACKING_OFF)
            lat -= 0.70 / 111_320.0
            mission.update_gps(lat, lon, 0.0)
            frame_id += 2
            clock.advance(0.1)
            mission.update(detection(1.5, 0.0), frame_id=frame_id)
            self.assertEqual(mission.state, carpma.CarpmaState.COOLDOWN)
            clock.advance(carpma.COOLDOWN_SEC + 0.1)
            frame_id += 1
            mission.update(detection(1.5, 0.0), frame_id=frame_id)
            self.assertEqual(mission.state, carpma.CarpmaState.CAMERA_CONFIRM)

        self.assertTrue(mission.finished)
        self.assertTrue(mission.success)
        self.assertEqual(mission.state, carpma.CarpmaState.COMPLETE)

    def test_collision_backoff_holds_the_heading_of_the_impact(self):
        mission = carpma.CarpmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.state = carpma.CarpmaState.BACKING_OFF
        mission.backoff_start_time = clock.monotonic()
        mission.backoff_start_lat = 41.0
        mission.backoff_start_lon = 29.0
        mission.backoff_heading_deg = 0.0
        mission.update_heading(15.0)

        mission.update(detection(1.0, 0.0), frame_id=1)

        self.assertLess(commands[-1][0], 0.0)
        self.assertLess(commands[-1][1], 0.0)

    def test_collision_stops_immediately_if_target_disappears_during_strike(self):
        mission = carpma.CarpmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for _ in range(carpma.BASELINE_WINDOW):
            mission.update_imu(0.0, 0.0, 9.81)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(1.0, 0.0), frame_id=frame_id)
        self.assertEqual(mission.state, carpma.CarpmaState.STRIKING)

        clock.advance(0.1)
        mission.update([], frame_id=6)

        self.assertEqual(mission.state, carpma.CarpmaState.MISSED)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_collision_does_not_count_impact_with_stale_camera_target(self):
        mission = carpma.CarpmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        for _ in range(carpma.BASELINE_WINDOW):
            mission.update_imu(0.0, 0.0, 9.81)
        for frame_id in range(1, 6):
            clock.advance(0.1)
            mission.update(detection(1.0, 0.0), frame_id=frame_id)
        self.assertEqual(mission.state, carpma.CarpmaState.STRIKING)

        clock.advance(carpma.IMPACT_TARGET_FRESH_SEC + 0.1)
        for _ in range(carpma.IMPACT_CONSECUTIVE_SAMPLES):
            mission.update_imu(9.81 + 8.0, 0.0, 0.0)

        self.assertEqual(mission.hit_count, 0)
        self.assertEqual(mission.state, carpma.CarpmaState.STRIKING)

    def test_collision_backoff_requires_both_gps_and_camera_confirmation(self):
        mission = carpma.CarpmaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.state = carpma.CarpmaState.BACKING_OFF
        mission.backoff_start_time = clock.monotonic()
        mission.backoff_start_lat = 41.0
        mission.backoff_start_lon = 29.0
        mission.backoff_heading_deg = 0.0
        mission.backoff_start_distance = 1.0

        mission.update(detection(1.6, 0.0), frame_id=1)
        self.assertEqual(mission.state, carpma.CarpmaState.BACKING_OFF)

        mission.update_gps(41.0 - 0.70 / 111_320.0, 29.0, 0.0)
        clock.advance(0.1)
        mission.update(detection(1.6, 0.0), frame_id=2)
        self.assertEqual(mission.state, carpma.CarpmaState.COOLDOWN)

    def test_manager_failsafe_when_total_mission_time_expires(self):
        manager = self._ready_task_manager()
        clock.advance(t3.MISSION_TOTAL_TIMEOUT_SEC + 0.1)
        self._refresh_navigation_and_imu(manager)
        manager.update_vision_timestamp()

        manager.update([], frame_id=1)

        self.assertEqual(manager.state, t3.MissionState.FAILSAFE)
        self.assertEqual(commands[-1], (0.0, 0.0))

    def test_low_confidence_and_target_loss_are_rejected(self):
        search = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        search.update_gps(41.0, 29.0, 0.0)
        search.step_target_heading = 0.0
        search.state = arama.SearchState.HOLDING
        search.hold_until = float("inf")
        low_confidence = detection()
        low_confidence[0]["confidence"] = 0.20
        for frame_id in range(1, 8):
            search.update(low_confidence, frame_id=frame_id)
        self.assertFalse(search.finished)

        approach = yaklasma.YaklasmaGorevi(Node(), Topics(), "red_buoy")
        approach.update_gps(41.0, 29.0, 0.0)
        approach.update(detection(), frame_id=1)
        clock.advance(yaklasma.TARGET_LOST_TIMEOUT_SEC + 0.1)
        approach.update([], frame_id=2)
        self.assertTrue(approach.should_return_to_search())


if __name__ == "__main__":
    unittest.main()
