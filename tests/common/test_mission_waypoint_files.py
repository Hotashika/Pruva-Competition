import pytest

from utils.mission_waypoint_files import (
    format_mission_waypoint_files,
    parse_mission_waypoint_files,
    resolve_mission_waypoint_directory,
)


def test_mission_waypoint_file_mapping_round_trip():
    mapping = {
        1: "teknofest.waypoints",
        2: "teknofest_task1.waypoints",
        3: "teknofest_task2.waypoints",
    }

    assert parse_mission_waypoint_files(
        format_mission_waypoint_files(mapping)
    ) == mapping


def test_empty_mapping_keeps_shared_bridge_profile_neutral():
    assert parse_mission_waypoint_files("") == {}


@pytest.mark.parametrize(
    "value",
    [
        "5:invalid.waypoints",
        "1:../outside.waypoints",
        "1:not-a-waypoint.txt",
        "missing-command",
    ],
)
def test_invalid_mission_waypoint_file_mapping_is_rejected(value):
    with pytest.raises(ValueError):
        parse_mission_waypoint_files(value)


def test_waypoint_sync_requires_explicit_directory(tmp_path):
    mapping = {1: "teknofest.waypoints"}

    with pytest.raises(ValueError, match="mission_waypoint_directory"):
        resolve_mission_waypoint_directory(mapping, "")

    assert resolve_mission_waypoint_directory(mapping, tmp_path) == tmp_path
    assert resolve_mission_waypoint_directory({}, "") is None
