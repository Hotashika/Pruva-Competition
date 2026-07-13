from utils.read_waypoints import parse_qgc_waypoints


def _write_waypoints(tmp_path, home_command):
    waypoint_file = tmp_path / "route.waypoints"
    waypoint_file.write_text(
        "QGC WPL 110\n"
        f"0\t1\t0\t{home_command}\t0\t0\t0\t0\t37.8730996\t32.4872262\t1030.88\t1\n"
        "1\t0\t3\t16\t0\t0\t0\t0\t37.9513328\t32.5007451\t0\t1\n"
        "2\t0\t3\t16\t0\t0\t0\t0\t37.9512504\t32.5010381\t0\t1\n",
        encoding="utf-8",
    )
    return waypoint_file


def test_qgc_nav_waypoint_home_row_is_skipped(tmp_path):
    waypoints = parse_qgc_waypoints(_write_waypoints(tmp_path, home_command=16))

    assert [waypoint["name"] for waypoint in waypoints] == ["WP0", "WP1"]
    assert [waypoint["seq"] for waypoint in waypoints] == [1, 2]
    assert waypoints[0]["lat"] == 37.9513328


def test_bridge_generated_home_row_is_skipped(tmp_path):
    waypoints = parse_qgc_waypoints(_write_waypoints(tmp_path, home_command=0))

    assert [waypoint["seq"] for waypoint in waypoints] == [1, 2]


def test_route_item_at_sequence_zero_is_not_skipped_without_home_markers(tmp_path):
    waypoint_file = tmp_path / "route_without_home.waypoints"
    waypoint_file.write_text(
        "QGC WPL 110\n"
        "0\t1\t3\t16\t0\t0\t0\t0\t37.9513328\t32.5007451\t0\t1\n",
        encoding="utf-8",
    )

    waypoints = parse_qgc_waypoints(waypoint_file)

    assert len(waypoints) == 1
    assert waypoints[0]["name"] == "WP0"
    assert waypoints[0]["seq"] == 0
