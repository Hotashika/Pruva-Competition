from pathlib import Path

import pytest

from teknofest.missions.utils.competition_waypoints import (
    GN_WAYPOINT_PATH,
    build_competition_routes,
    load_competition_points,
)


def _write_waypoints(path, points):
    lines = [
        "QGC WPL 110",
        "0\t1\t0\t16\t0\t0\t0\t0\t37.0\t32.0\t0\t1",
    ]
    for seq, (lat, lon) in enumerate(points, start=1):
        lines.append(
            f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t0\t1"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_competition_points_are_named_in_required_order(tmp_path):
    path = tmp_path / "competition.waypoints"
    _write_waypoints(
        path,
        [
            (37.1, 32.1),
            (37.2, 32.2),
            (37.3, 32.3),
            (37.4, 32.4),
            (37.5, 32.5),
        ],
    )

    points = load_competition_points(path)

    assert tuple(points) == ("GN1", "GN2", "GN3", "GN4", "GN5")
    assert points["GN4"]["lat"] == pytest.approx(37.4)
    assert points["GN5"]["lon"] == pytest.approx(32.5)


def test_default_competition_source_is_teknofest_waypoints():
    assert GN_WAYPOINT_PATH.name == "teknofest.waypoints"


def test_competition_routes_use_gn_boundaries():
    points = {
        "GN1": {"name": "GN1"},
        "GN2": {"name": "GN2"},
        "GN3": {"name": "GN3"},
        "GN4": {"name": "GN4"},
        "GN5": {"name": "GN5"},
    }

    routes = build_competition_routes(points)

    assert [point["name"] for point in routes["task1"]] == [
        "GN1",
        "GN2",
        "GN3",
        "GN4",
    ]
    assert [point["name"] for point in routes["task2"]] == ["GN4", "GN5"]


def test_competition_file_requires_exactly_five_points(tmp_path):
    path = Path(tmp_path) / "competition.waypoints"
    _write_waypoints(
        path,
        [(37.1, 32.1), (37.2, 32.2), (37.3, 32.3), (37.4, 32.4)],
    )

    with pytest.raises(ValueError, match="tam 5 nokta"):
        load_competition_points(path)
