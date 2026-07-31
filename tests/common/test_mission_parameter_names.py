import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

MISSION_FILES = {
    "njord_task1": REPO_ROOT / "njord/missions/task1_maneuvering_and_path_finding.py",
    "njord_task2": REPO_ROOT / "njord/missions/task2_collision_avoidance.py",
    "njord_task4": REPO_ROOT / "njord/missions/task4_surprise.py",
    "teknofest_task1": REPO_ROOT / "teknofest/missions/task1_point_tracking.py",
    "teknofest_task2": (
        REPO_ROOT
        / "teknofest/missions/task2_point_tracking_task_in_an_environment_with_obstacle.py"
    ),
}

AVOIDANCE_TASKS = {
    "njord_task1",
    "njord_task2",
    "njord_task4",
    "teknofest_task2",
}

DISTANCE_HYSTERESIS_TASKS = {
    "njord_task1",
    "njord_task4",
    "teknofest_task2",
}

GPS_TARGET_AVOIDANCE_TASKS = set()

FORBIDDEN_PARAMETER_NAMES = {
    "AVOID_ENTER_DISTANCE_M",
    "AVOID_EXIT_DISTANCE_M",
    "AVOID_ENTER_DIST_M",
    "AVOID_EXIT_DIST_M",
    "AVOID_LINEAR_X",
    "AVOID_TURN_Z",
    "AVOID_MIN_DURATION_SEC",
    "AVOID_CLEAR_DURATION_SEC",
    "AVOID_MAX_DURATION_SEC",
    "AVOID_PASS_CLEARANCE_M",
    "AVOID_TARGET_TOLERANCE_M",
    "AVOID_TARGET_REFRESH_MIN_SHIFT_M",
    "AVOID_TARGET_TIMEOUT_SEC",
    "AVOID_WAYPOINT_TOLERANCE_M",
    "AVOID_TIMEOUT_SEC",
    "DETECTION_STALE_SEC",
}


def _assigned_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_common_mission_parameter_names_are_consistent():
    for task_name, path in MISSION_FILES.items():
        names = _assigned_names(path)
        assert "GPS_TIMEOUT_SEC" in names, task_name
        assert "HEADING_TIMEOUT_SEC" in names, task_name
        assert "BRIDGE_STATE_TIMEOUT_SEC" in names, task_name
        assert "VISION_DETECTION_TIMEOUT_SEC" in names, task_name
        assert "WAYPOINT_TOLERANCE_M" in names, task_name
        assert not names.intersection(FORBIDDEN_PARAMETER_NAMES), task_name


def test_avoidance_parameter_names_are_consistent():
    for task_name in AVOIDANCE_TASKS:
        names = _assigned_names(MISSION_FILES[task_name])
        assert "AVOIDANCE_START_DISTANCE_M" in names, task_name
        assert "AVOIDANCE_TIMEOUT_SEC" in names, task_name
    for task_name in DISTANCE_HYSTERESIS_TASKS:
        names = _assigned_names(MISSION_FILES[task_name])
        assert "AVOIDANCE_EXIT_DISTANCE_M" in names, task_name
    for task_name in GPS_TARGET_AVOIDANCE_TASKS:
        names = _assigned_names(MISSION_FILES[task_name])
        assert "AVOIDANCE_PASS_CLEARANCE_M" in names, task_name
        assert "AVOIDANCE_WAYPOINT_TOLERANCE_M" in names, task_name
        assert "AVOIDANCE_TARGET_REFRESH_MIN_SHIFT_M" in names, task_name


def test_parameter_sections_are_grouped():
    for task_name, path in MISSION_FILES.items():
        source = path.read_text(encoding="utf-8")
        assert "GÜVENLİK PARAMETRELERİ" in source, task_name
        assert "NAVİGASYON PARAMETRELERİ" in source, task_name
        assert "VISION" in source and "PARAMETRELERİ" in source, task_name
        if task_name in AVOIDANCE_TASKS:
            assert "KAÇINMA" in source and "PARAMETRELERİ" in source, task_name


def test_task3_config_uses_standard_timeout_field_names():
    path = REPO_ROOT / "teknofest/missions/task3_kamikaze_engagement.py"
    source = path.read_text(encoding="utf-8")
    assert "bridge_state_timeout_sec" in source
    assert "vision_detection_timeout_sec" in source
    assert "bridge_timeout_sec" not in source
    assert "vision_stale_sec" not in source
