import unittest
from unittest.mock import patch

import numpy as np

from vision.usv_3d_detector import (
    Detection,
    DetectorConfig,
    USV3DDetector,
    USV3DObstacleDetector,
)


def synthetic_depth():
    height, width = 240, 320
    fx = fy = 220.0
    cx, cy = width / 2.0, height / 2.0
    rows, columns = np.indices((height, width))
    ray_y = (rows - cy) / fy

    depth = np.full((height, width), np.nan, dtype=np.float32)
    water = ray_y > 0.0
    depth[water] = (0.25 / ray_y[water]).astype(np.float32)
    depth[(depth < 0.30) | (depth > 12.0)] = np.nan

    target = (
        (columns >= 130)
        & (columns <= 190)
        & (rows >= 105)
        & (rows <= 137)
    )
    depth[target] = 3.0

    rng = np.random.default_rng(4)
    valid = np.isfinite(depth)
    depth[valid] += rng.normal(0.0, 0.003, np.count_nonzero(valid))
    return depth


def geometry_detection():
    return Detection(
        detection_id=2,
        point_count=150,
        cell_count=12,
        lateral_m=-0.3,
        forward_m=2.8,
        range_m=2.82,
        nearest_range_m=2.65,
        bearing_deg=-6.1,
        camera_xyz_m=(-0.3, 0.0, 2.8),
        pixel_bbox_xyxy=(120, 80, 180, 180),
        width_m=0.45,
        length_m=0.30,
        height_m=0.65,
        max_height_above_water_m=0.72,
        confidence=0.78,
    )


class USV3DDetectorTest(unittest.TestCase):
    def test_detects_target_and_estimates_water_plane(self):
        detector = USV3DDetector(
            DetectorConfig(
                fx=220.0,
                fy=220.0,
                camera_height_m=0.25,
                max_depth_m=12.0,
                plane_fit_max_points=8_000,
                plane_ransac_iterations=350,
                min_cluster_points=20,
            )
        )

        result = detector.detect(synthetic_depth())

        self.assertAlmostEqual(
            result.plane.camera_distance_m,
            0.25,
            delta=0.03,
        )
        self.assertGreater(result.plane.confidence, 0.45)
        self.assertGreaterEqual(len(result.detections), 1)
        target = result.detections[0]
        self.assertAlmostEqual(target.range_m, 3.0, delta=0.20)
        self.assertAlmostEqual(target.bearing_deg, 0.0, delta=3.0)
        self.assertGreater(target.max_height_above_water_m, 0.20)

    def test_geometry_detection_uses_live_mission_contract(self):
        live = USV3DObstacleDetector._geometry_to_live_detection(
            geometry_detection(),
            plane_confidence=0.84,
            bbox=[120, 80, 180, 180],
        )

        self.assertEqual("surface_obstacle_candidate", live["class"])
        self.assertEqual("depth_obstacle", live["type"])
        self.assertEqual(2.65, live["distance"])
        self.assertEqual(-6.1, live["angle"])
        self.assertEqual("left", live["side"])
        self.assertIsNone(live["track_id"])

    def test_depth_only_live_detector_does_not_need_yolo_model(self):
        detector = USV3DObstacleDetector(
            fx=220.0,
            fy=220.0,
            cx=160.0,
            cy=120.0,
            camera_height_m=0.25,
            detection_max_range_m=12.0,
            plane_ransac_iterations=350,
        )

        detections = detector.detect(
            np.zeros((240, 320, 3), dtype=np.uint8),
            synthetic_depth(),
        )

        self.assertGreaterEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual("surface_obstacle_candidate", detection["class"])
        self.assertEqual("depth_obstacle", detection["type"])
        self.assertEqual("usv_3d", detection["source"])
        self.assertIsNone(detection["track_id"])
        self.assertIsNotNone(detector.last_result)

    def test_pixel_rays_are_cached_for_repeated_resolution(self):
        detector = USV3DDetector(
            DetectorConfig(
                fx=220.0,
                fy=220.0,
                cx=160.0,
                cy=120.0,
            )
        )
        depth = synthetic_depth()

        detector._depth_to_points(depth)
        cached_ray_x = detector._ray_x
        cached_rows = detector._pixel_rows
        detector._depth_to_points(depth.copy())

        self.assertIs(cached_ray_x, detector._ray_x)
        self.assertIs(cached_rows, detector._pixel_rows)
        self.assertEqual(depth.shape, detector._ray_cache_shape)

    def test_tracks_plane_between_periodic_full_ransac_updates(self):
        detector = USV3DDetector(
            DetectorConfig(
                fx=220.0,
                fy=220.0,
                camera_height_m=0.25,
                max_depth_m=12.0,
                plane_fit_max_points=8_000,
                plane_ransac_iterations=100,
                plane_full_ransac_interval=3,
                min_cluster_points=20,
            )
        )

        with patch.object(
            detector,
            "_estimate_water_plane_ransac",
            wraps=detector._estimate_water_plane_ransac,
        ) as full_ransac:
            results = [
                detector.detect(synthetic_depth())
                for _ in range(4)
            ]

        self.assertEqual(2, full_ransac.call_count)
        self.assertTrue(
            all(result.plane.confidence > 0.35 for result in results)
        )

    def test_live_detector_downsamples_depth_and_scales_intrinsics(self):
        detector = USV3DObstacleDetector(
            fx=220.0,
            fy=220.0,
            cx=160.0,
            cy=120.0,
            camera_height_m=0.25,
            detection_max_range_m=12.0,
            plane_ransac_iterations=100,
            depth_downsample_factor=2,
        )

        detections = detector.detect(
            np.zeros((240, 320, 3), dtype=np.uint8),
            synthetic_depth(),
        )

        self.assertAlmostEqual(110.0, detector.geometry_detector.config.fx)
        self.assertAlmostEqual(80.0, detector.geometry_detector.config.cx)
        self.assertEqual(
            (120, 160),
            detector.geometry_detector._ray_cache_shape,
        )
        self.assertGreaterEqual(len(detections), 1)
        x1, y1, x2, y2 = detections[0]["bbox"]
        self.assertLessEqual(x1, 135)
        self.assertGreaterEqual(x2, 185)
        self.assertLessEqual(y1, 110)
        self.assertGreaterEqual(y2, 128)

    def test_live_detector_reuses_result_between_geometry_ticks(self):
        detector = USV3DObstacleDetector(
            fx=220.0,
            fy=220.0,
            cx=160.0,
            cy=120.0,
            camera_height_m=0.25,
            detection_max_range_m=12.0,
            plane_ransac_iterations=100,
            geometry_hz=10.0,
        )
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = synthetic_depth()

        with (
            patch(
                "vision.usv_3d_detector.time.monotonic",
                side_effect=(10.0, 10.05, 10.11),
            ),
            patch.object(
                detector.geometry_detector,
                "detect",
                wraps=detector.geometry_detector.detect,
            ) as geometry_detect,
        ):
            first = detector.detect(image, depth)
            cached = detector.detect(image, depth)
            refreshed = detector.detect(image, depth)

        self.assertEqual(2, geometry_detect.call_count)
        self.assertEqual(first, cached)
        self.assertEqual(first, refreshed)
        self.assertIsNot(first, cached)


if __name__ == "__main__":
    unittest.main()
