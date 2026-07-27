import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from vision.detector_lifecycle import DetectorRegistry


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RecordingDetector:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)


class BrokenDetector:
    def __init__(self, **_kwargs):
        raise OSError("weights are unavailable")


def detector_spec(detector_class=RecordingDetector, **extra):
    spec = {"class": detector_class}
    spec.update(extra)
    return spec


def fake_profile(startup_detectors, task_detector_map, detector_specs):
    return SimpleNamespace(
        STARTUP_DETECTORS=startup_detectors,
        TASK_DETECTOR_MAP=task_detector_map,
        DETECTOR_SPECS=detector_specs,
        CAMERA_WIDTH=1280,
        DEVICE="cpu",
        TOLERANCE_RATIO=0.05,
        TOLERANCE_DEG=5,
    )


def assigned_literal(relative_path, variable_name):
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{variable_name} was not found in {relative_path}")


class DetectorRegistryTests(unittest.TestCase):
    def setUp(self):
        RecordingDetector.instances = []

    def test_njord_detectors_load_once_and_switch_without_reconstruction(self):
        profile = fake_profile(
            ("buoy", "usv_3d", "ar_tag"),
            {
                "task1": {"buoy"},
                "task2": {"usv_3d"},
                "task3": {"ar_tag"},
                "task4": {"buoy"},
            },
            {
                "buoy": detector_spec(model_path="buoy.pt"),
                "usv_3d": detector_spec(
                    uses_full_intrinsics=True,
                    kwargs={"detection_max_range_m": 12.0},
                ),
                "ar_tag": detector_spec(model_path="ar_tag.pt"),
            },
        )

        registry = DetectorRegistry(
            profile,
            fx=700.0,
            fy=701.0,
            cx=640.0,
            cy=360.0,
        )
        original_detector_ids = {
            name: id(detector)
            for name, detector in registry.detectors.items()
        }

        self.assertEqual(
            ("buoy", "usv_3d", "ar_tag"),
            registry.active_names,
        )
        self.assertEqual(3, len(RecordingDetector.instances))

        for task, expected in (
            ("task1", ("buoy",)),
            ("task2", ("usv_3d",)),
            ("task3", ("ar_tag",)),
            ("task4", ("buoy",)),
            ("task4", ("buoy",)),
        ):
            self.assertTrue(registry.select_task(task))
            self.assertEqual(expected, registry.active_names)

        self.assertEqual(original_detector_ids, {
            name: id(detector)
            for name, detector in registry.detectors.items()
        })
        self.assertEqual(3, len(RecordingDetector.instances))

    def test_teknofest_buoy_detector_is_active_before_and_during_tasks(self):
        profile = fake_profile(
            ("buoy",),
            {
                "task1": {"buoy"},
                "task2": {"buoy"},
                "task3": {"buoy"},
            },
            {"buoy": detector_spec(model_path="buoy.pt")},
        )

        registry = DetectorRegistry(profile, fx=700.0, cx=640.0)
        detector_id = id(registry.detectors["buoy"])

        self.assertEqual(("buoy",), registry.active_names)
        for task in ("task1", "task2", "task3"):
            self.assertTrue(registry.select_task(task))
            self.assertEqual(("buoy",), registry.active_names)
            self.assertEqual(detector_id, id(registry.detectors["buoy"]))
        self.assertEqual(1, len(RecordingDetector.instances))

    def test_unknown_task_keeps_existing_selection(self):
        profile = fake_profile(
            ("buoy", "ar_tag"),
            {"task1": {"buoy"}, "task3": {"ar_tag"}},
            {
                "buoy": detector_spec(model_path="buoy.pt"),
                "ar_tag": detector_spec(model_path="ar_tag.pt"),
            },
        )
        registry = DetectorRegistry(profile)
        self.assertTrue(registry.select_task("task3"))

        self.assertFalse(registry.select_task("not-a-task"))
        self.assertEqual("task3", registry.current_task)
        self.assertEqual(("ar_tag",), registry.active_names)

    def test_detector_load_failure_names_the_failed_detector(self):
        profile = fake_profile(
            ("buoy",),
            {"task1": {"buoy"}},
            {
                "buoy": detector_spec(
                    BrokenDetector,
                    model_path="missing-buoy.pt",
                )
            },
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to load detector 'buoy'.*missing-buoy.pt",
        ) as context:
            DetectorRegistry(profile)

        self.assertIsInstance(context.exception.__cause__, OSError)

    def test_startup_detector_list_must_include_every_spec_once(self):
        profile = fake_profile(
            ("buoy",),
            {"task1": {"buoy"}},
            {
                "buoy": detector_spec(),
                "ar_tag": detector_spec(),
            },
        )

        with self.assertRaisesRegex(ValueError, "missing=.*ar_tag"):
            DetectorRegistry(profile)


class CompetitionProfileContractTests(unittest.TestCase):
    def test_profiles_define_deterministic_startup_detectors(self):
        self.assertEqual(
            ("buoy", "usv_3d", "ar_tag"),
            assigned_literal(
                "njord/config/vision_profile.py",
                "STARTUP_DETECTORS",
            ),
        )
        self.assertEqual(
            ("buoy",),
            assigned_literal(
                "teknofest/config/vision_profile.py",
                "STARTUP_DETECTORS",
            ),
        )

    def test_vision_node_uses_preloaded_registry_for_inference(self):
        source = (
            REPOSITORY_ROOT / "vision" / "vision_node.py"
        ).read_text(encoding="utf-8")

        self.assertIn("DetectorRegistry(", source)
        self.assertIn("self.detector_registry.select_task(task)", source)
        self.assertIn("self.detector_registry.active_items()", source)
        self.assertNotIn("del self.detectors[name]", source)


if __name__ == "__main__":
    unittest.main()
