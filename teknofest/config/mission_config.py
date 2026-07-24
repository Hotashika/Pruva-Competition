"""TEKNOFEST görev seçimi ve MAVLink bridge profil ayarları."""

from pathlib import Path

from utils.mission_waypoint_files import format_mission_waypoint_files
from utils.task_selection_state import default_task_selection_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WAYPOINT_DIRECTORY = REPOSITORY_ROOT / "waypoints" / "teknofest"


# Mission Planner'da SCR_USER1=1 seçildiğinde ayrı Task 1 süreci yerine
# Task 1 -> Task 2 -> Task 3 zincirini tek ARM/GUIDED oturumunda çalıştır.
# Diğer komutlar görevleri bakım/test amacıyla tek başına başlatır.
#
# command: (task_key, display_name, mission_filename)
MISSION_COMMANDS = {
    1: ("competition", "task1->task2->task3", "competition_mission.py"),
    2: ("task1", "task1 (standalone)", "task1_point_tracking.py"),
    3: (
        "task2",
        "task2 (standalone)",
        "task2_point_tracking_task_in_an_environment_with_obstacle.py",
    ),
    4: ("task3", "task3 (standalone)", "task3_kamikaze_engagement.py"),
}

MISSION_SPECS = {
    task_key: (display_name, filename)
    for task_key, display_name, filename in MISSION_COMMANDS.values()
}

MISSION_WAYPOINT_FILES = {
    1: "teknofest.waypoints",
    2: "teknofest_task1.waypoints",
    3: "teknofest_task2.waypoints",
}

MAVLINK_BRIDGE_DEFAULTS = {
    "MAVLINK_CONNECTION_STRING": "/dev/ttyACM0",
    "MAVLINK_BAUD": "921600",
    "MAVLINK_SOURCE_SYSTEM": "1",
    "MAVLINK_SOURCE_COMPONENT": "191",
    "MAVLINK_MISSION_START_TOPIC": "/mission_start",
    "MAVLINK_MISSION_START_ACK_TOPIC": "/mission_start_ack",
    "MISSION_SELECTION_FILE": default_task_selection_file(),
}

# Profil sahibi bu değerleri eski shell environment değerlerinin üzerine yazar.
MAVLINK_BRIDGE_OVERRIDES = {
    "MAVLINK_MISSION_WAYPOINT_DIRECTORY": str(WAYPOINT_DIRECTORY),
    "MAVLINK_MISSION_WAYPOINT_FILES": format_mission_waypoint_files(
        MISSION_WAYPOINT_FILES
    ),
}
