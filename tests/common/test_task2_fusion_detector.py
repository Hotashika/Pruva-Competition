import types
import unittest
from pathlib import Path

import numpy as np

from vision.ewasr_segmenter import EWaSRResult, EWaSRSegmenter
from vision.task2_fusion_detector import (
    FUSED_OBSTACLE_TYPE,
    SEGMENTATION_DEPTH_OBSTACLE_TYPE,
    VISUAL_OBSTACLE_TYPE,
    Task2FusionDetector,
)


IMAGE_SHAPE = (12, 16)


def semantic_result(obstacle_slice=None):
    labels = np.full(IMAGE_SHAPE, 1, dtype=np.uint8)
    if obstacle_slice is not None:
        labels[obstacle_slice] = 0
    return EWaSRResult(
        label_map=labels,
        confidence_map=np.full(IMAGE_SHAPE, 0.9, dtype=np.float32),
    )


class FakeSegmenter:
    def __init__(self, result):
        self.result = result
        self.ready = True
        self.last_error = None
        self.calls = 0

    def detect(self, image, imu_mask):
        self.calls += 1
        return self.result


class FakeBackend:
    def predict(self, image, imu_mask):
        result = semantic_result((slice(3, 9), slice(5, 11)))
        return result.label_map, result.confidence_map


class FakeDepthDetector:
    def __init__(self, detections, geometry_pixels=None):
        self.detections = detections
        self.last_depth_shape = IMAGE_SHAPE
        if geometry_pixels:
            rows, columns = zip(*geometry_pixels)
            self.last_result = types.SimpleNamespace(
                pixel_rows=np.asarray(rows),
                pixel_columns=np.asarray(columns),
                cluster_labels=np.zeros(len(rows), dtype=np.int32),
            )
        else:
            self.last_result = None

    def detect(self, image, depth):
        return [dict(item) for item in self.detections]


def depth_detection():
    return {
        "type": "depth_obstacle",
        "class": "surface_obstacle_candidate",
        "confidence": 0.8,
        "geometry_confidence": 0.8,
        "distance": 3.5,
        "angle": 0.0,
        "bbox": [5, 3, 11, 9],
        "geometry_id": 0,
        "track_id": None,
    }


class Task2FusionDetectorTests(unittest.TestCase):
    def _detector(
        self,
        *,
        segmentation,
        depth_detections=(),
        geometry_pixels=None,
        shadow_mode=False,
        **kwargs,
    ):
        return Task2FusionDetector(
            fx=100.0,
            fy=100.0,
            cx=8.0,
            cy=6.0,
            depth_detector=FakeDepthDetector(
                depth_detections,
                geometry_pixels,
            ),
            segmenter=FakeSegmenter(segmentation),
            shadow_mode=shadow_mode,
            segmentation_hz=100.0,
            min_semantic_pixels=4,
            min_mask_depth_pixels=4,
            segmentation_erode_radius=1,
            **kwargs,
        )

    def test_shadow_mode_preserves_depth_contract_and_reports_fusion(self):
        pixels = [(row, column) for row in range(4, 8) for column in range(6, 10)]
        detector = self._detector(
            segmentation=semantic_result((slice(3, 9), slice(5, 11))),
            depth_detections=[depth_detection()],
            geometry_pixels=pixels,
            shadow_mode=True,
        )

        live = detector.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            np.full(IMAGE_SHAPE, 3.5, dtype=np.float32),
            imu=(0.0, 0.0, 0.0),
            now=1.0,
        )

        self.assertEqual("depth_obstacle", live[0]["type"])
        self.assertEqual(
            FUSED_OBSTACLE_TYPE,
            detector.last_shadow_detections[0]["type"],
        )
        self.assertEqual(
            "confirmed",
            detector.last_shadow_detections[0]["fusion_status"],
        )
        self.assertEqual(1, detector.last_diagnostics["fused_count"])

    def test_depth_detection_is_never_vetoed_by_segmentation(self):
        detector = self._detector(
            segmentation=semantic_result(),
            depth_detections=[depth_detection()],
        )

        detections = detector.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            np.full(IMAGE_SHAPE, 3.5, dtype=np.float32),
            now=1.0,
        )

        self.assertEqual("depth_obstacle", detections[0]["type"])
        self.assertEqual("depth_only", detections[0]["fusion_status"])
        self.assertFalse(detections[0]["segmentation_confirmed"])

    def test_unmatched_semantic_component_uses_masked_metric_depth(self):
        detector = self._detector(
            segmentation=semantic_result((slice(3, 9), slice(5, 11))),
        )
        depth = np.full(IMAGE_SHAPE, np.nan, dtype=np.float32)
        depth[3:9, 5:11] = 4.2

        detections = detector.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            depth,
            now=1.0,
        )

        self.assertEqual(1, len(detections))
        self.assertEqual(
            SEGMENTATION_DEPTH_OBSTACLE_TYPE,
            detections[0]["type"],
        )
        self.assertAlmostEqual(4.2, detections[0]["distance"], places=1)
        self.assertEqual("segmentation_depth", detections[0]["fusion_status"])

    def test_semantic_component_without_depth_stays_visual_only(self):
        detector = self._detector(
            segmentation=semantic_result((slice(3, 9), slice(5, 11))),
        )

        detections = detector.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            np.full(IMAGE_SHAPE, np.nan, dtype=np.float32),
            now=1.0,
        )

        self.assertEqual(VISUAL_OBSTACLE_TYPE, detections[0]["type"])
        self.assertIsNone(detections[0]["distance"])

    def test_water_plane_geometry_rejects_reflection_metric_distance(self):
        detector = self._detector(
            segmentation=semantic_result((slice(3, 9), slice(5, 11))),
        )
        rows, columns = np.indices(IMAGE_SHAPE)
        detector.depth_detector.last_result = types.SimpleNamespace(
            pixel_rows=rows.ravel(),
            pixel_columns=columns.ravel(),
            obstacle_mask=np.zeros(rows.size, dtype=bool),
        )

        detections = detector.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            np.full(IMAGE_SHAPE, 4.2, dtype=np.float32),
            now=1.0,
        )

        self.assertEqual(VISUAL_OBSTACLE_TYPE, detections[0]["type"])
        self.assertIsNone(detections[0]["distance"])
        self.assertTrue(detections[0]["geometry_gated"])

    def test_segmenter_accepts_an_injected_backend_without_model_file(self):
        segmenter = EWaSRSegmenter(
            model_path=None,
            backend=FakeBackend(),
        )

        result = segmenter.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            np.ones(IMAGE_SHAPE, dtype=np.uint8),
        )

        self.assertTrue(segmenter.ready)
        self.assertEqual(36, int(np.count_nonzero(result.obstacle_mask)))

    def test_missing_imu_uses_level_fallback_but_is_reported_invalid(self):
        detector = self._detector(
            segmentation=semantic_result(),
        )

        detector.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            np.full(IMAGE_SHAPE, 3.5, dtype=np.float32),
            imu=None,
            now=1.0,
        )

        self.assertFalse(detector.last_diagnostics["imu_valid"])

    def test_missing_model_fails_open_to_existing_depth_detection(self):
        missing_model = Path(__file__).with_name(
            "missing_ewasr_model.torchscript"
        )
        segmenter = EWaSRSegmenter(model_path=missing_model)
        detector = Task2FusionDetector(
            fx=100.0,
            fy=100.0,
            cx=8.0,
            cy=6.0,
            depth_detector=FakeDepthDetector([depth_detection()]),
            segmenter=segmenter,
            shadow_mode=True,
        )

        detections = detector.detect(
            np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8),
            np.full(IMAGE_SHAPE, 3.5, dtype=np.float32),
            imu=(0.0, 0.0, 0.0),
            now=1.0,
        )

        self.assertEqual("depth_obstacle", detections[0]["type"])
        self.assertFalse(detector.last_diagnostics["segmentation_ready"])
        self.assertIn(
            "missing",
            detector.last_diagnostics["segmentation_error"].lower(),
        )


if __name__ == "__main__":
    unittest.main()
