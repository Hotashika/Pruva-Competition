"""Kesintisiz yarış modu için GN waypoint dosyası sözleşmesi."""

from pathlib import Path

from utils.read_waypoints import parse_qgc_waypoints


GN_WAYPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "waypoints"
    / "teknofest.waypoints"
)
GN_NAMES = ("GN1", "GN4", "GN5")
COMPETITION_ROUTE_NAMES = {
    "task1": ("GN1", "GN4"),
    "task2": ("GN4", "GN5"),
}


def load_competition_points(path=GN_WAYPOINT_PATH):
    """HOME satırından sonra sırasıyla GN1, GN4 ve GN5'i yükler."""
    points = parse_qgc_waypoints(path)
    if len(points) != len(GN_NAMES):
        raise ValueError(
            f"{path} HOME satırından sonra tam 3 nokta içermeli: "
            f"GN1, GN4, GN5; bulunan={len(points)}"
        )
    return {
        name: {**point, "name": name}
        for name, point in zip(GN_NAMES, points)
    }


def build_competition_routes(points):
    """GN noktalarından yalnız competition modunda kullanılacak görev rotalarını üretir."""
    missing = set(GN_NAMES).difference(points)
    if missing:
        raise ValueError(f"Competition GN noktaları eksik: {sorted(missing)}")
    return {
        task_name: [points[point_name] for point_name in point_names]
        for task_name, point_names in COMPETITION_ROUTE_NAMES.items()
    }
