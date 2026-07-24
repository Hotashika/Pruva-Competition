import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGER_PATH = REPO_ROOT / "teknofest" / "missions" / "mission_manager.py"
MAIN_PATH = REPO_ROOT / "teknofest" / "main.py"
BRIDGE_PATH = REPO_ROOT / "bridge" / "bridge_node.py"
DETECTOR_PATH = REPO_ROOT / "teknofest" / "vision" / "detector.py"
MAVLINK_UTILITIES_PATH = REPO_ROOT / "utils" / "mavlink_utilities.py"
TASK3_PATH = (
    REPO_ROOT
    / "teknofest"
    / "missions"
    / "task3_kamikaze_engagement.py"
)


class MissionManagerContractTests(unittest.TestCase):
    def test_manager_source_is_valid_python(self):
        ast.parse(MANAGER_PATH.read_text(encoding="utf-8"))

    def test_main_launches_one_manager_instead_of_a_task3_only_node(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn('"mission_manager.py"', source)
        self.assertIn("p_mission_manager", source)
        self.assertNotIn("p_teknofest_task3", source)

    def test_manager_accepts_only_the_three_teknofest_tasks(self):
        source = MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn("VALID_COMMANDS = (1, 2, 3)", source)
        self.assertIn("Task1Maneuvering", source)
        self.assertIn("Task2PointTrackingWithObstacleAvoidance", source)
        self.assertIn("Task3KamikazeEngagement", source)

    def test_manager_uses_normal_arm_for_task3_and_force_for_task1_task2(self):
        source = MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.mission_clients.force_arm_client", source)
        self.assertIn("self.mission_clients.arm_client", source)
        self.assertIn("if command == 3:", source)
        self.assertIn('"NORMAL ARM"', source)
        self.assertIn('"FORCE ARM"', source)
        task3_source = TASK3_PATH.read_text(encoding="utf-8")
        self.assertIn("USE_FORCE_ARM = False", task3_source)

    def test_failed_start_waits_for_bridge_zero_before_new_start(self):
        source = MANAGER_PATH.read_text(encoding="utf-8")
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("START_FAILED = auto()", source)
        self.assertIn("if command == 0:", source)
        self.assertIn("self._publish_ack(command)", source)
        self.assertIn(
            "self._publish_mission_start(MISSION_IDLE, track_ack=False)",
            bridge_source,
        )

    def test_failed_disarm_keeps_manager_in_failsafe(self):
        source = MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '"FAILSAFE kilidi korunuyor.',
            source,
        )
        self.assertIn("self.state = ManagerState.FAILSAFE", source)

    def test_task3_requires_real_imu_before_start(self):
        source = MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn("command == 3 and not self._fresh(self.last_imu_time", source)
        self.assertIn("self.task.update_imu(*self.latest_imu)", source)

    def test_task3_target_defaults_to_red_but_is_explicitly_configurable(self):
        source = MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn('"TASK3_TARGET_COLOR"', source)
        self.assertIn("ACTIVE_TARGET_COLOR", source)
        self.assertIn("SUPPORTED_TARGET_COLORS", source)

    def test_bridge_downloads_teknofest_waypoint_names_when_main_runs(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        bridge_source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertIn("MAVLINK_WAYPOINT_PREFIX=teknofest", main_source)
        self.assertIn(
            'filename = f"{self.waypoint_prefix}_task{task_number}.waypoints"',
            bridge_source,
        )

    def test_task1_and_task2_reject_stale_local_waypoint_files(self):
        source = MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn("def _validate_fresh_waypoint_download(", source)
        self.assertIn("WAYPOINT_DOWNLOAD_MAX_AGE_SEC = 15.0", source)
        self.assertIn("Mission Planner", source)

    def test_model_contract_covers_all_task_colors(self):
        source = DETECTOR_PATH.read_text(encoding="utf-8")
        for class_name in (
                "red_buoy",
                "green_buoy",
                "black_buoy",
                "orange_buoy",
                "yellow_buoy",
        ):
            self.assertIn(f'"{class_name}"', source)

    def test_all_tasks_share_the_same_limited_skid_steer_turn_contract(self):
        source = MAVLINK_UTILITIES_PATH.read_text(encoding="utf-8")
        self.assertIn("def publish_skid_steer_turn(", source)
        self.assertIn(
            "max_angular_z=DEFAULT_SKID_STEER_MAX_YAW_OFFSET",
            source,
        )


if __name__ == "__main__":
    unittest.main()
