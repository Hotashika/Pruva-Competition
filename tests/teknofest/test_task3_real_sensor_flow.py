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


fake_mavlink = types.ModuleType("utils.mavlink_utilities")
fake_mavlink.publish_cmd_vel = publish_cmd_vel
fake_mavlink.stop_vehicle = stop_vehicle
fake_mavlink.calculate_gps_distance = calculate_gps_distance
sys.modules["utils.mavlink_utilities"] = fake_mavlink

arama = importlib.import_module("teknofest.missions.arama")
yaklasma = importlib.import_module("teknofest.missions.yaklasma")
carpma = importlib.import_module("teknofest.missions.carpma")


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

    def test_search_accepts_five_distinct_frames_only_while_holding(self):
        mission = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        mission.update_gps(41.0, 29.0, 0.0)
        mission.state = arama.SearchState.TURNING
        mission.step_target_heading = 20.0
        mission.step_start_time = arama.time.monotonic()
        for frame_id in range(1, 6):
            mission.update(detection(), frame_id=frame_id)
        self.assertFalse(mission.finished)

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

            # Dönüş komutu ileri itki içermemeli.
            mission.update([], frame_id=frame_id)
            self.assertEqual(commands[-1][0], 0.0)
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
            lat += 0.70 / 111_320.0
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

    def test_low_confidence_and_target_loss_are_rejected(self):
        search = arama.AramaGorevi(Node(), Topics(), "red_buoy")
        search.update_gps(41.0, 29.0, 0.0)
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
