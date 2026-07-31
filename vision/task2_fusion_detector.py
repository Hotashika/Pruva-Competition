"""Shadow-safe fusion of eWaSR semantics and Njord metric-depth geometry."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from vision.ewasr_segmenter import EWaSRResult, EWaSRSegmenter
from vision.horizon_mask import create_horizon_mask
from vision.usv_3d_detector import USV3DObstacleDetector


FUSED_OBSTACLE_TYPE = "fused_obstacle"
SEGMENTATION_DEPTH_OBSTACLE_TYPE = "seg_depth_obstacle"
VISUAL_OBSTACLE_TYPE = "visual_obstacle_candidate"
SURFACE_OBSTACLE_CLASS = "surface_obstacle_candidate"


@dataclass(frozen=True)
class _SemanticComponent:
    component_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    pixel_count: int
    confidence: float


def _nearest_resize(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    if array.shape[:2] == shape:
        return array
    source_height, source_width = array.shape[:2]
    rows = np.minimum(
        (np.arange(target_height) * source_height / target_height).astype(int),
        source_height - 1,
    )
    columns = np.minimum(
        (np.arange(target_width) * source_width / target_width).astype(int),
        source_width - 1,
    )
    return array[rows[:, None], columns[None, :]]


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    output = np.asarray(mask, dtype=bool).copy()
    if radius == 0 or not np.any(output):
        return output

    padded = np.pad(output, radius, mode="constant", constant_values=False)
    height, width = output.shape
    for row_offset in range(radius * 2 + 1):
        for column_offset in range(radius * 2 + 1):
            output &= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return output


def _numpy_connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Small dependency-free fallback used when OpenCV is unavailable."""

    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    component_id = 0

    for start_row, start_column in np.argwhere(mask):
        if labels[start_row, start_column] != 0:
            continue
        component_id += 1
        labels[start_row, start_column] = component_id
        stack = [(int(start_row), int(start_column))]
        while stack:
            row, column = stack.pop()
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    next_row = row + row_offset
                    next_column = column + column_offset
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and labels[next_row, next_column] == 0
                    ):
                        labels[next_row, next_column] = component_id
                        stack.append((next_row, next_column))
    return labels, component_id + 1


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        import cv2
    except ImportError:
        return _numpy_connected_components(mask)

    count, labels = cv2.connectedComponents(
        np.asarray(mask, dtype=np.uint8),
        connectivity=8,
    )
    return labels.astype(np.int32, copy=False), int(count)


class Task2FusionDetector:
    """Keep depth authoritative while evaluating eWaSR in shadow mode."""

    def __init__(
        self,
        *,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        camera_width=1280,
        camera_height_m=0.25,
        detection_max_range_m=12.0,
        plane_ransac_iterations=350,
        model_path=None,
        device=None,
        segmentation_enabled=True,
        shadow_mode=True,
        segmentation_hz=5.0,
        segmentation_cache_max_age_sec=0.30,
        geometry_overlap_threshold=0.25,
        min_semantic_pixels=20,
        min_mask_depth_pixels=20,
        min_mask_depth_ratio=0.03,
        segmentation_erode_radius=2,
        semantic_waterline_filter=True,
        waterline_quantile=0.20,
        min_water_intrusion_pixels=2,
        min_depth_m=0.30,
        max_depth_m=20.0,
        depth_detector=None,
        segmenter=None,
        **_unused,
    ):
        self.fx = self._positive_float(fx, 700.0)
        self.fy = self._positive_float(fy, self.fx)
        self.cx = self._optional_float(cx)
        self.cy = self._optional_float(cy)
        self.camera_width = int(camera_width)
        self.shadow_mode = bool(shadow_mode)
        self.segmentation_enabled = bool(segmentation_enabled)
        self.segmentation_interval_sec = 1.0 / max(
            float(segmentation_hz),
            0.1,
        )
        self.segmentation_cache_max_age_sec = max(
            float(segmentation_cache_max_age_sec),
            self.segmentation_interval_sec,
        )
        self.geometry_overlap_threshold = float(
            geometry_overlap_threshold
        )
        self.min_semantic_pixels = max(1, int(min_semantic_pixels))
        self.min_mask_depth_pixels = max(1, int(min_mask_depth_pixels))
        self.min_mask_depth_ratio = max(0.0, float(min_mask_depth_ratio))
        self.segmentation_erode_radius = max(
            0,
            int(segmentation_erode_radius),
        )
        self.semantic_waterline_filter = bool(
            semantic_waterline_filter
        )
        self.waterline_quantile = min(
            max(float(waterline_quantile), 0.0),
            1.0,
        )
        self.min_water_intrusion_pixels = max(
            0,
            int(min_water_intrusion_pixels),
        )
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

        self.depth_detector = depth_detector or USV3DObstacleDetector(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            camera_width=self.camera_width,
            camera_height_m=camera_height_m,
            detection_max_range_m=detection_max_range_m,
            plane_ransac_iterations=plane_ransac_iterations,
        )
        self.segmenter = segmenter or EWaSRSegmenter(
            model_path=model_path,
            enabled=self.segmentation_enabled,
            device=device,
        )

        self._cached_segmentation: EWaSRResult | None = None
        self._cached_segmentation_time: float | None = None
        self._last_segmentation_attempt_time: float | None = None
        self.last_segmentation: EWaSRResult | None = None
        self.last_shadow_detections: list[dict] = []
        self.last_diagnostics: dict = {}

    @staticmethod
    def _positive_float(value, fallback):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return number if math.isfinite(number) and number > 0.0 else float(fallback)

    @staticmethod
    def _optional_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _side_from_angle(angle_deg: float) -> str:
        if angle_deg < -2.0:
            return "left"
        if angle_deg > 2.0:
            return "right"
        return "across"

    @staticmethod
    def _imu_angles(imu) -> tuple[float, float, float, bool]:
        if isinstance(imu, dict):
            values = (
                imu.get("roll", imu.get("roll_rad")),
                imu.get("pitch", imu.get("pitch_rad")),
                imu.get("yaw", imu.get("yaw_rad", 0.0)),
            )
        elif imu is None:
            return 0.0, 0.0, 0.0, False
        else:
            try:
                values = tuple(imu)
            except TypeError:
                values = (0.0, 0.0, 0.0)

        if len(values) < 3:
            values = (*values, *(0.0 for _ in range(3 - len(values))))
        try:
            roll, pitch, yaw = (float(value) for value in values[:3])
        except (TypeError, ValueError):
            return 0.0, 0.0, 0.0, False
        valid = all(math.isfinite(value) for value in (roll, pitch, yaw))
        return (
            (roll, pitch, yaw, True)
            if valid
            else (0.0, 0.0, 0.0, False)
        )

    def _horizon_mask(self, image_shape, imu) -> tuple[np.ndarray, bool]:
        height, width = image_shape
        roll, pitch, _yaw, imu_valid = self._imu_angles(imu)
        cx = width / 2.0 if self.cx is None else self.cx
        cy = height / 2.0 if self.cy is None else self.cy
        mask = create_horizon_mask(
            width=width,
            height=height,
            fx=self.fx,
            fy=self.fy,
            cx=cx,
            cy=cy,
            roll=roll,
            pitch=pitch,
        )
        return mask, imu_valid

    def _segmentation_for_frame(
        self,
        bgr_image: np.ndarray,
        imu,
        now: float,
    ) -> tuple[EWaSRResult | None, bool]:
        should_run = (
            self._last_segmentation_attempt_time is None
            or now - self._last_segmentation_attempt_time
            >= self.segmentation_interval_sec
        )
        imu_valid = self._imu_angles(imu)[3]
        if should_run:
            self._last_segmentation_attempt_time = now
            try:
                horizon_mask, imu_valid = self._horizon_mask(
                    bgr_image.shape[:2],
                    imu,
                )
                result = self.segmenter.detect(bgr_image, horizon_mask)
            except (RuntimeError, TypeError, ValueError) as exc:
                result = None
                self.segmenter.last_error = str(exc)

            if result is not None:
                self._cached_segmentation = result
                self._cached_segmentation_time = now

        if (
            self._cached_segmentation_time is None
            or now - self._cached_segmentation_time
            > self.segmentation_cache_max_age_sec
        ):
            self._cached_segmentation = None
            self._cached_segmentation_time = None
        return self._cached_segmentation, imu_valid

    def _semantic_components(
        self,
        segmentation: EWaSRResult,
        image_shape: tuple[int, int],
    ) -> list[_SemanticComponent]:
        obstacle_mask = _nearest_resize(
            segmentation.obstacle_mask,
            image_shape,
        ).astype(bool, copy=False)
        confidence_map = _nearest_resize(
            segmentation.confidence_map,
            image_shape,
        )
        if self.semantic_waterline_filter:
            water_mask = _nearest_resize(
                segmentation.water_mask,
                image_shape,
            ).astype(bool, copy=False)
            water_rows = np.nonzero(water_mask)[0]
            if water_rows.size:
                waterline = int(
                    round(
                        float(
                            np.quantile(
                                water_rows,
                                self.waterline_quantile,
                            )
                        )
                    )
                )
                minimum_row = (
                    waterline + self.min_water_intrusion_pixels
                )
                obstacle_mask = obstacle_mask.copy()
                obstacle_mask[:minimum_row] = False
        labels, count = _connected_components(obstacle_mask)
        components = []
        for component_id in range(1, count):
            component_mask = labels == component_id
            pixel_count = int(np.count_nonzero(component_mask))
            if pixel_count < self.min_semantic_pixels:
                continue
            rows, columns = np.nonzero(component_mask)
            bbox = (
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            )
            confidence = float(np.mean(confidence_map[component_mask]))
            components.append(
                _SemanticComponent(
                    component_id=component_id,
                    mask=component_mask,
                    bbox=bbox,
                    pixel_count=pixel_count,
                    confidence=confidence,
                )
            )
        return components

    def _geometry_mask(
        self,
        detection: dict,
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        result = getattr(self.depth_detector, "last_result", None)
        geometry_id = detection.get("geometry_id")
        if result is None or geometry_id is None:
            return None

        selected = result.cluster_labels == int(geometry_id)
        if not np.any(selected):
            return None

        depth_shape = getattr(self.depth_detector, "last_depth_shape", None)
        if depth_shape is None:
            depth_shape = image_shape
        depth_height, depth_width = map(int, depth_shape)
        rows = result.pixel_rows[selected]
        columns = result.pixel_columns[selected]
        image_height, image_width = image_shape
        scaled_rows = np.clip(
            np.rint(rows * image_height / max(depth_height, 1)).astype(int),
            0,
            image_height - 1,
        )
        scaled_columns = np.clip(
            np.rint(columns * image_width / max(depth_width, 1)).astype(int),
            0,
            image_width - 1,
        )
        mask = np.zeros(image_shape, dtype=bool)
        mask[scaled_rows, scaled_columns] = True
        return mask

    @staticmethod
    def _bbox_mask(detection: dict, image_shape) -> np.ndarray | None:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        height, width = image_shape
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros(image_shape, dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask

    def _above_water_mask(
        self,
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        result = getattr(self.depth_detector, "last_result", None)
        if result is None:
            return None
        point_mask = getattr(result, "obstacle_mask", None)
        rows = getattr(result, "pixel_rows", None)
        columns = getattr(result, "pixel_columns", None)
        if point_mask is None or rows is None or columns is None:
            return None

        point_mask = np.asarray(point_mask, dtype=bool)
        rows = np.asarray(rows)
        columns = np.asarray(columns)
        if (
            point_mask.ndim != 1
            or rows.shape != point_mask.shape
            or columns.shape != point_mask.shape
        ):
            return None

        selected_rows = rows[point_mask]
        selected_columns = columns[point_mask]
        mask = np.zeros(image_shape, dtype=bool)
        if not selected_rows.size:
            return mask

        depth_shape = getattr(self.depth_detector, "last_depth_shape", None)
        if depth_shape is None:
            depth_shape = image_shape
        depth_height, depth_width = map(int, depth_shape)
        image_height, image_width = image_shape
        scaled_rows = np.clip(
            np.rint(
                selected_rows
                * image_height
                / max(depth_height, 1)
            ).astype(int),
            0,
            image_height - 1,
        )
        scaled_columns = np.clip(
            np.rint(
                selected_columns
                * image_width
                / max(depth_width, 1)
            ).astype(int),
            0,
            image_width - 1,
        )
        mask[scaled_rows, scaled_columns] = True
        return mask

    def _match_component(
        self,
        detection: dict,
        components: list[_SemanticComponent],
        image_shape: tuple[int, int],
    ) -> tuple[_SemanticComponent | None, float]:
        geometry_mask = self._geometry_mask(detection, image_shape)
        if geometry_mask is None:
            geometry_mask = self._bbox_mask(detection, image_shape)
        if geometry_mask is None:
            return None, 0.0

        support = int(np.count_nonzero(geometry_mask))
        if support == 0:
            return None, 0.0

        best_component = None
        best_overlap = 0.0
        for component in components:
            overlap = float(
                np.count_nonzero(geometry_mask & component.mask) / support
            )
            if overlap > best_overlap:
                best_component = component
                best_overlap = overlap

        if best_overlap < self.geometry_overlap_threshold:
            return None, best_overlap
        return best_component, best_overlap

    def _fused_detection(
        self,
        depth_detection: dict,
        component: _SemanticComponent,
        overlap: float,
    ) -> dict:
        detection = dict(depth_detection)
        geometry_confidence = self._optional_float(
            detection.get(
                "geometry_confidence",
                detection.get("confidence"),
            )
        )
        scores = [component.confidence]
        if geometry_confidence is not None:
            scores.append(geometry_confidence)
        detection.update(
            {
                "type": FUSED_OBSTACLE_TYPE,
                "class": SURFACE_OBSTACLE_CLASS,
                "source": "ewasr+usv_3d",
                "fusion_status": "confirmed",
                "segmentation_confidence": round(
                    component.confidence,
                    3,
                ),
                "geometry_overlap": round(overlap, 3),
                "confidence": round(float(np.mean(scores)), 3),
            }
        )
        return detection

    def _semantic_detection(
        self,
        component: _SemanticComponent,
        depth_array: np.ndarray,
        image_shape: tuple[int, int],
    ) -> dict:
        mask = component.mask
        if depth_array.shape != image_shape:
            mask = _nearest_resize(mask, depth_array.shape).astype(
                bool,
                copy=False,
            )
        eroded = _binary_erode(mask, self.segmentation_erode_radius)
        above_water_mask = self._above_water_mask(depth_array.shape)
        metric_support = (
            eroded
            if above_water_mask is None
            else eroded & above_water_mask
        )
        valid = (
            metric_support
            & np.isfinite(depth_array)
            & (depth_array >= self.min_depth_m)
            & (depth_array <= self.max_depth_m)
        )
        valid_count = int(np.count_nonzero(valid))
        valid_ratio = valid_count / max(int(np.count_nonzero(eroded)), 1)
        common = {
            "class": SURFACE_OBSTACLE_CLASS,
            "bbox": list(component.bbox),
            "track_id": None,
            "segmentation_confidence": round(component.confidence, 3),
            "confidence": round(component.confidence, 3),
            "valid_depth_pixels": valid_count,
            "valid_depth_ratio": round(valid_ratio, 4),
            "geometry_gated": above_water_mask is not None,
        }

        if (
            valid_count < self.min_mask_depth_pixels
            or valid_ratio < self.min_mask_depth_ratio
        ):
            common.update(
                {
                    "type": VISUAL_OBSTACLE_TYPE,
                    "source": "ewasr",
                    "fusion_status": "visual_only",
                    "distance": None,
                    "angle": None,
                    "side": "unknown",
                }
            )
            return common

        distance = float(np.quantile(depth_array[valid], 0.10))
        rows, columns = np.nonzero(valid)
        center_x_depth = float(np.median(columns))
        image_width = image_shape[1]
        center_x_image = (
            center_x_depth * image_width / max(depth_array.shape[1], 1)
        )
        cx = image_width / 2.0 if self.cx is None else self.cx
        angle = math.degrees(math.atan2(center_x_image - cx, self.fx))
        common.update(
            {
                "type": SEGMENTATION_DEPTH_OBSTACLE_TYPE,
                "source": "ewasr+mask_depth",
                "fusion_status": "segmentation_depth",
                "distance": round(distance, 2),
                "angle": round(angle, 3),
                "bearing_deg": round(angle, 3),
                "side": self._side_from_angle(angle),
            }
        )
        return common

    def _fusion_detections(
        self,
        depth_detections: list[dict],
        segmentation: EWaSRResult,
        depth_array: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[list[dict], dict]:
        components = self._semantic_components(segmentation, image_shape)
        matched_components: set[int] = set()
        detections = []
        counts = {
            "semantic_component_count": len(components),
            "fused_count": 0,
            "depth_only_count": 0,
            "segmentation_depth_count": 0,
            "visual_only_count": 0,
        }

        for depth_detection in depth_detections:
            component, overlap = self._match_component(
                depth_detection,
                components,
                image_shape,
            )
            if component is None:
                detection = dict(depth_detection)
                detection.setdefault("fusion_status", "depth_only")
                detection["segmentation_confirmed"] = False
                detections.append(detection)
                counts["depth_only_count"] += 1
                continue

            matched_components.add(component.component_id)
            detections.append(
                self._fused_detection(
                    depth_detection,
                    component,
                    overlap,
                )
            )
            counts["fused_count"] += 1

        for component in components:
            if component.component_id in matched_components:
                continue
            detection = self._semantic_detection(
                component,
                depth_array,
                image_shape,
            )
            detections.append(detection)
            if detection["type"] == SEGMENTATION_DEPTH_OBSTACLE_TYPE:
                counts["segmentation_depth_count"] += 1
            else:
                counts["visual_only_count"] += 1

        return detections, counts

    def detect(self, bgr_image, depth_array, imu=None, now=None):
        now = time.monotonic() if now is None else float(now)
        depth_detections = self.depth_detector.detect(
            bgr_image,
            depth_array,
        )
        depth_detections = [
            dict(detection) for detection in depth_detections
        ]

        segmentation, imu_valid = self._segmentation_for_frame(
            bgr_image,
            imu,
            now,
        )
        self.last_segmentation = segmentation
        cache_age = (
            None
            if self._cached_segmentation_time is None
            else max(0.0, now - self._cached_segmentation_time)
        )
        diagnostics = {
            "shadow_mode": self.shadow_mode,
            "segmentation_enabled": self.segmentation_enabled,
            "segmentation_ready": bool(self.segmenter.ready),
            "segmentation_error": self.segmenter.last_error,
            "segmentation_age_sec": (
                None if cache_age is None else round(cache_age, 3)
            ),
            "imu_valid": imu_valid,
            "depth_detection_count": len(depth_detections),
            "semantic_component_count": 0,
            "fused_count": 0,
            "depth_only_count": len(depth_detections),
            "segmentation_depth_count": 0,
            "visual_only_count": 0,
        }

        if segmentation is None:
            shadow_detections = [
                {
                    **detection,
                    "fusion_status": detection.get(
                        "fusion_status",
                        "depth_only",
                    ),
                    "segmentation_confirmed": False,
                }
                for detection in depth_detections
            ]
        else:
            shadow_detections, fusion_counts = self._fusion_detections(
                depth_detections,
                segmentation,
                depth_array,
                bgr_image.shape[:2],
            )
            diagnostics.update(fusion_counts)

        self.last_shadow_detections = shadow_detections
        self.last_diagnostics = diagnostics
        return depth_detections if self.shadow_mode else shadow_detections
