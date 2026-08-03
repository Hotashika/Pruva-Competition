import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parsed_source(relative_path):
    path = REPOSITORY_ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def test_teknofest_recorder_uses_cached_vision_without_detector_inference():
    source, tree = _parsed_source("teknofest/core/data_writer.py")
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "VisionDetectionCache" in source
    assert "BuoyDetector" not in source
    assert "detect" not in called_attributes


def test_both_recorders_target_ten_fps_and_publish_scalar_depth():
    for relative_path in (
        "njord/core/data_writer.py",
        "teknofest/core/data_writer.py",
    ):
        source, _tree = _parsed_source(relative_path)
        assert "VIDEO_FPS = 10" in source
        assert "latest_center_depth" in source
        assert "latest_depth_array" not in source
        assert "shm_depth[" not in source
        assert "nearest_bbox_median_distance" in source


def test_detector_distance_is_computed_as_bbox_median():
    source, _tree = _parsed_source("vision/detector.py")

    assert 'method="median"' in source


def test_teknofest_buoy_detector_uses_persistent_tracking():
    source, _tree = _parsed_source("teknofest/config/vision_profile.py")

    assert '"kwargs": {"use_tracking": True}' in source
