"""Metric depth tabanli deniz duzlemi ve su ustu engel algilama.

Bu modul ``usv_3d_detector.zip`` paketindeki NumPy tabanli 3B cekirdegi
canli vision hattina uyarlar. ``USV3DObstacleDetector`` yalnizca metrik
derinlik kullanir ve YOLO modeline bagli degildir.

Kamera koordinatlari:
    x: goruntude saga
    y: goruntude asagi
    z: kameradan ileri
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import acos, degrees

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    """Algilayicinin metre ve piksel cinsinden parametreleri."""

    fx: float = 700.0
    fy: float = 700.0
    cx: float | None = None
    cy: float | None = None

    camera_height_m: float = 0.25
    min_depth_m: float = 0.30
    max_depth_m: float = 20.0
    detection_max_range_m: float = 8.0

    plane_roi_top_ratio: float = 0.35
    plane_roi_bottom_ratio: float = 0.98
    plane_fit_max_points: int = 20_000
    plane_ransac_iterations: int = 800
    plane_inlier_threshold_m: float = 0.04
    plane_min_inlier_ratio: float = 0.08
    plane_max_tilt_deg: float = 50.0
    plane_height_tolerance_m: float = 0.20
    plane_height_score_sigma_m: float = 0.10
    plane_height_prior_weight: float = 0.20
    plane_smoothing_alpha: float = 0.40
    plane_prior_angle_sigma_deg: float = 12.0
    plane_prior_distance_sigma_m: float = 0.08

    min_obstacle_height_m: float = 0.08
    water_noise_multiplier: float = 2.5
    height_threshold_range_slope: float = 0.002
    max_obstacle_height_m: float = 2.50

    grid_cell_size_m: float = 0.08
    grid_neighbor_radius: int = 1
    min_points_per_cell: int = 1
    min_cluster_points: int = 20
    min_cluster_cells: int = 3
    min_cluster_mean_points_per_cell: float = 8.0
    min_cluster_height_m: float = 0.12
    max_cluster_width_m: float = 3.0
    max_cluster_length_m: float = 6.0
    max_detections: int = 30

    random_seed: int = 7


@dataclass(frozen=True)
class PlaneEstimate:
    normal: np.ndarray
    d: float
    inlier_ratio: float
    residual_sigma_m: float
    camera_distance_m: float
    confidence: float
    used_previous_plane: bool = False

    def signed_height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d

    def to_dict(self) -> dict:
        return {
            "normal": self.normal.tolist(),
            "d": float(self.d),
            "inlier_ratio": float(self.inlier_ratio),
            "residual_sigma_m": float(self.residual_sigma_m),
            "camera_distance_m": float(self.camera_distance_m),
            "confidence": float(self.confidence),
            "used_previous_plane": bool(self.used_previous_plane),
        }


@dataclass(frozen=True)
class Detection:
    detection_id: int
    point_count: int
    cell_count: int
    lateral_m: float
    forward_m: float
    range_m: float
    nearest_range_m: float
    bearing_deg: float
    camera_xyz_m: tuple[float, float, float]
    pixel_bbox_xyxy: tuple[int, int, int, int]
    width_m: float
    length_m: float
    height_m: float
    max_height_above_water_m: float
    confidence: float
    semantic_label: str = "surface_obstacle_candidate"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameResult:
    plane: PlaneEstimate
    detections: list[Detection]
    points: np.ndarray
    pixel_rows: np.ndarray
    pixel_columns: np.ndarray
    signed_heights: np.ndarray
    obstacle_mask: np.ndarray
    cluster_labels: np.ndarray
    water_basis_right: np.ndarray
    water_basis_forward: np.ndarray

    def summary(self) -> dict:
        return {
            "plane": self.plane.to_dict(),
            "detection_count": len(self.detections),
            "detections": [item.to_dict() for item in self.detections],
        }


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Sifira yakin vektor normalize edilemez.")
    return vector / norm


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return degrees(acos(cosine))


def _robust_sigma(values: np.ndarray, minimum: float = 0.005) -> float:
    if values.size == 0:
        return minimum
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, minimum)


class USV3DDetector:
    """Kareler arasinda deniz duzlemini koruyan durumlu algilayici."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._previous_plane: PlaneEstimate | None = None

    def reset(self) -> None:
        self._previous_plane = None

    def detect(self, depth: np.ndarray) -> FrameResult:
        if depth.ndim != 2:
            raise ValueError(
                f"Depth verisi iki boyutlu olmali; gelen sekil: {depth.shape}"
            )

        points, rows, columns = self._depth_to_points(depth)
        if len(points) < 100:
            raise ValueError(
                "Duzlem kestirimi icin yeterli gecerli depth noktasi yok."
            )

        plane_points = self._plane_roi_points(points, rows, depth.shape[0])
        plane = self._estimate_water_plane(plane_points)
        self._previous_plane = plane

        right, forward_axis = self._water_plane_basis(plane.normal)
        signed_heights = plane.signed_height(points)
        lateral = points @ right
        forward = points @ forward_axis
        planar_range = np.hypot(lateral, forward)

        height_threshold = (
            self.config.min_obstacle_height_m
            + self.config.water_noise_multiplier * plane.residual_sigma_m
            + self.config.height_threshold_range_slope * planar_range
        )
        obstacle_mask = (
            (signed_heights > height_threshold)
            & (signed_heights < self.config.max_obstacle_height_m)
            & (forward > self.config.min_depth_m)
            & (planar_range < self.config.detection_max_range_m)
        )

        labels = np.full(len(points), -1, dtype=np.int32)
        candidate_indices = np.flatnonzero(obstacle_mask)
        detections: list[Detection] = []

        if candidate_indices.size:
            candidate_labels, component_cells = self._cluster_bev(
                lateral[candidate_indices],
                forward[candidate_indices],
            )
            detections, accepted_components = self._build_detections(
                points=points,
                point_rows=rows,
                point_columns=columns,
                signed_heights=signed_heights,
                lateral=lateral,
                forward=forward,
                candidate_indices=candidate_indices,
                candidate_labels=candidate_labels,
                component_cells=component_cells,
                plane_confidence=plane.confidence,
            )
            for component_id, detection_id in accepted_components.items():
                component_points = candidate_indices[
                    candidate_labels == component_id
                ]
                labels[component_points] = detection_id

        return FrameResult(
            plane=plane,
            detections=detections,
            points=points,
            pixel_rows=rows,
            pixel_columns=columns,
            signed_heights=signed_heights,
            obstacle_mask=obstacle_mask,
            cluster_labels=labels,
            water_basis_right=right,
            water_basis_forward=forward_axis,
        )

    def _depth_to_points(
        self,
        depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = depth.shape
        cx = width / 2.0 if self.config.cx is None else self.config.cx
        cy = height / 2.0 if self.config.cy is None else self.config.cy

        rows, columns = np.indices(depth.shape)
        z_image = depth.astype(np.float64, copy=False)
        valid = (
            np.isfinite(z_image)
            & (z_image > self.config.min_depth_m)
            & (z_image < self.config.max_depth_m)
        )

        z = z_image[valid]
        x = (columns[valid] - cx) * z / self.config.fx
        y = (rows[valid] - cy) * z / self.config.fy
        return np.column_stack((x, y, z)), rows[valid], columns[valid]

    def _plane_roi_points(
        self,
        points: np.ndarray,
        rows: np.ndarray,
        image_height: int,
    ) -> np.ndarray:
        first_row = int(image_height * self.config.plane_roi_top_ratio)
        last_row = int(image_height * self.config.plane_roi_bottom_ratio)
        roi = points[(rows >= first_row) & (rows < last_row)]
        if len(roi) < 100:
            raise ValueError("Deniz ROI bolgesinde yeterli gecerli nokta yok.")
        return roi

    def _estimate_water_plane(self, roi_points: np.ndarray) -> PlaneEstimate:
        sample_size = min(len(roi_points), self.config.plane_fit_max_points)
        sample_indices = self._rng.choice(
            len(roi_points),
            size=sample_size,
            replace=False,
        )
        sample = roi_points[sample_indices]

        best_score = -np.inf
        best_model: tuple[np.ndarray, float, np.ndarray] | None = None
        min_vertical_component = np.cos(
            np.deg2rad(self.config.plane_max_tilt_deg)
        )

        for _ in range(self.config.plane_ransac_iterations):
            selected = sample[
                self._rng.choice(sample_size, size=3, replace=False)
            ]
            normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal /= norm
            d = -float(np.dot(normal, selected[0]))

            if normal[1] > 0.0:
                normal = -normal
                d = -d

            if abs(normal[1]) < min_vertical_component or d <= 0.0:
                continue

            camera_distance = abs(d)
            height_error = abs(
                camera_distance - self.config.camera_height_m
            )
            if height_error > self.config.plane_height_tolerance_m:
                continue

            distances = np.abs(sample @ normal + d)
            inliers = distances < self.config.plane_inlier_threshold_m
            inlier_count = int(np.count_nonzero(inliers))
            height_weight = np.exp(
                -0.5
                * (
                    height_error
                    / max(self.config.plane_height_score_sigma_m, 1e-6)
                )
                ** 2
            )
            prior_weight = self._previous_plane_weight(normal, d)
            score = inlier_count * height_weight * prior_weight

            if score > best_score:
                best_score = score
                best_model = (normal.copy(), d, inliers)

        if best_model is None:
            return self._reuse_previous_or_fail(
                "Kamera yuksekligiyle uyumlu deniz duzlemi bulunamadi."
            )

        normal, d, sample_inliers = best_model
        inlier_ratio = float(np.mean(sample_inliers))
        if inlier_ratio < self.config.plane_min_inlier_ratio:
            return self._reuse_previous_or_fail(
                "Deniz duzlemi destegi guvenilir esigin altinda."
            )

        full_distances = np.abs(roi_points @ normal + d)
        refine_points = roi_points[
            full_distances < self.config.plane_inlier_threshold_m
        ]
        if len(refine_points) >= 3:
            normal, d = self._least_squares_plane(refine_points, normal)

        weight = float(
            np.clip(self.config.plane_height_prior_weight, 0.0, 1.0)
        )
        d = (1.0 - weight) * d + weight * self.config.camera_height_m

        refined_residuals = roi_points @ normal + d
        refined_inliers = (
            np.abs(refined_residuals)
            < self.config.plane_inlier_threshold_m
        )
        if np.any(refined_inliers):
            residual_sigma = _robust_sigma(refined_residuals[refined_inliers])
            inlier_ratio = float(np.mean(refined_inliers))
        else:
            residual_sigma = self.config.plane_inlier_threshold_m

        normal, d = self._smooth_with_previous(normal, d)
        height_error = abs(abs(d) - self.config.camera_height_m)
        confidence = self._plane_confidence(
            inlier_ratio,
            residual_sigma,
            height_error,
        )

        return PlaneEstimate(
            normal=normal,
            d=float(d),
            inlier_ratio=inlier_ratio,
            residual_sigma_m=residual_sigma,
            camera_distance_m=abs(float(d)),
            confidence=confidence,
        )

    def _least_squares_plane(
        self,
        points: np.ndarray,
        reference_normal: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        center = np.mean(points, axis=0)
        covariance = (points - center).T @ (points - center)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = _normalize(eigenvectors[:, 0])
        if np.dot(normal, reference_normal) < 0.0:
            normal = -normal
        if normal[1] > 0.0:
            normal = -normal
        d = -float(np.dot(normal, center))
        return normal, d

    def _previous_plane_weight(self, normal: np.ndarray, d: float) -> float:
        if self._previous_plane is None:
            return 1.0
        angle = _angle_deg(normal, self._previous_plane.normal)
        distance_delta = abs(d - self._previous_plane.d)
        angle_sigma = max(self.config.plane_prior_angle_sigma_deg, 1e-6)
        distance_sigma = max(
            self.config.plane_prior_distance_sigma_m,
            1e-6,
        )
        return float(
            np.exp(
                -0.5 * (angle / angle_sigma) ** 2
                -0.5 * (distance_delta / distance_sigma) ** 2
            )
        )

    def _smooth_with_previous(
        self,
        normal: np.ndarray,
        d: float,
    ) -> tuple[np.ndarray, float]:
        if self._previous_plane is None:
            return normal, d
        alpha = float(
            np.clip(self.config.plane_smoothing_alpha, 0.0, 1.0)
        )
        blended_normal = _normalize(
            (1.0 - alpha) * self._previous_plane.normal + alpha * normal
        )
        blended_d = (1.0 - alpha) * self._previous_plane.d + alpha * d
        return blended_normal, float(blended_d)

    def _reuse_previous_or_fail(self, message: str) -> PlaneEstimate:
        if self._previous_plane is None:
            raise RuntimeError(message)
        previous = self._previous_plane
        return PlaneEstimate(
            normal=previous.normal.copy(),
            d=previous.d,
            inlier_ratio=previous.inlier_ratio,
            residual_sigma_m=previous.residual_sigma_m,
            camera_distance_m=previous.camera_distance_m,
            confidence=previous.confidence * 0.65,
            used_previous_plane=True,
        )

    def _plane_confidence(
        self,
        inlier_ratio: float,
        residual_sigma: float,
        height_error: float,
    ) -> float:
        support = min(inlier_ratio / 0.30, 1.0)
        residual_score = np.exp(
            -residual_sigma
            / max(self.config.plane_inlier_threshold_m, 1e-6)
        )
        height_score = np.exp(
            -height_error
            / max(self.config.plane_height_score_sigma_m, 1e-6)
        )
        return float(
            np.clip(support * residual_score * height_score, 0.0, 1.0)
        )

    def _water_plane_basis(
        self,
        up: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        camera_right = np.array([1.0, 0.0, 0.0])
        right = camera_right - np.dot(camera_right, up) * up
        right = _normalize(right)
        forward = _normalize(np.cross(up, right))
        if forward[2] < 0.0:
            forward = -forward
        return right, forward

    def _cluster_bev(
        self,
        lateral: np.ndarray,
        forward: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, int]]:
        cell_size = self.config.grid_cell_size_m
        cells = np.floor(
            np.column_stack((lateral, forward)) / cell_size
        ).astype(np.int64)
        unique_cells, inverse, counts = np.unique(
            cells,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        usable = counts >= self.config.min_points_per_cell
        usable_indices = np.flatnonzero(usable)
        cell_lookup = {
            tuple(unique_cells[index]): int(index)
            for index in usable_indices
        }

        cell_components = np.full(len(unique_cells), -1, dtype=np.int32)
        component_id = 0
        radius = self.config.grid_neighbor_radius

        for starting_index in usable_indices:
            if cell_components[starting_index] >= 0:
                continue

            cell_components[starting_index] = component_id
            queue: deque[int] = deque([int(starting_index)])

            while queue:
                current_index = queue.popleft()
                current_x, current_z = unique_cells[current_index]
                for delta_x in range(-radius, radius + 1):
                    for delta_z in range(-radius, radius + 1):
                        if delta_x == 0 and delta_z == 0:
                            continue
                        neighbor = (
                            int(current_x + delta_x),
                            int(current_z + delta_z),
                        )
                        neighbor_index = cell_lookup.get(neighbor)
                        if (
                            neighbor_index is not None
                            and cell_components[neighbor_index] < 0
                        ):
                            cell_components[neighbor_index] = component_id
                            queue.append(neighbor_index)

            component_id += 1

        point_components = cell_components[inverse]
        component_cells = {
            identifier: int(
                np.count_nonzero(cell_components == identifier)
            )
            for identifier in range(component_id)
        }
        return point_components, component_cells

    def _build_detections(
        self,
        points: np.ndarray,
        point_rows: np.ndarray,
        point_columns: np.ndarray,
        signed_heights: np.ndarray,
        lateral: np.ndarray,
        forward: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_labels: np.ndarray,
        component_cells: dict[int, int],
        plane_confidence: float,
    ) -> tuple[list[Detection], dict[int, int]]:
        candidates: list[tuple[int, Detection]] = []

        for component_id in np.unique(
            candidate_labels[candidate_labels >= 0]
        ):
            local_mask = candidate_labels == component_id
            indices = candidate_indices[local_mask]
            cell_count = component_cells[int(component_id)]

            if (
                len(indices) < self.config.min_cluster_points
                or cell_count < self.config.min_cluster_cells
                or (
                    len(indices) / cell_count
                    < self.config.min_cluster_mean_points_per_cell
                )
            ):
                continue

            cluster_lateral = lateral[indices]
            cluster_forward = forward[indices]
            cluster_height = signed_heights[indices]
            low_lat, high_lat = np.percentile(
                cluster_lateral,
                [2.0, 98.0],
            )
            low_fwd, high_fwd = np.percentile(
                cluster_forward,
                [2.0, 98.0],
            )
            low_h, high_h = np.percentile(
                cluster_height,
                [2.0, 98.0],
            )
            width = float(high_lat - low_lat)
            length = float(high_fwd - low_fwd)
            height = float(high_h - low_h)
            max_height = float(np.percentile(cluster_height, 98.0))

            if (
                max_height < self.config.min_cluster_height_m
                or width > self.config.max_cluster_width_m
                or length > self.config.max_cluster_length_m
            ):
                continue

            median_lateral = float(np.median(cluster_lateral))
            median_forward = float(np.median(cluster_forward))
            planar_ranges = np.hypot(cluster_lateral, cluster_forward)
            range_m = float(np.hypot(median_lateral, median_forward))
            nearest_range = float(np.percentile(planar_ranges, 10.0))
            bearing = float(
                np.degrees(np.arctan2(median_lateral, median_forward))
            )
            camera_center = np.median(points[indices], axis=0)
            bbox_left, bbox_right = np.percentile(
                point_columns[indices],
                [1.0, 99.0],
            )
            bbox_top, bbox_bottom = np.percentile(
                point_rows[indices],
                [1.0, 99.0],
            )
            pixel_bbox = (
                int(np.floor(bbox_left)),
                int(np.floor(bbox_top)),
                int(np.ceil(bbox_right)),
                int(np.ceil(bbox_bottom)),
            )

            evidence = min(
                1.0,
                0.35 * np.log1p(len(indices)) / np.log(200.0)
                + 0.35 * min(max_height / 0.40, 1.0)
                + 0.30 * min(cell_count / 20.0, 1.0),
            )
            confidence = float(
                np.clip(plane_confidence * evidence, 0.0, 1.0)
            )

            candidates.append(
                (
                    int(component_id),
                    Detection(
                        detection_id=-1,
                        point_count=int(len(indices)),
                        cell_count=cell_count,
                        lateral_m=median_lateral,
                        forward_m=median_forward,
                        range_m=range_m,
                        nearest_range_m=nearest_range,
                        bearing_deg=bearing,
                        camera_xyz_m=tuple(
                            float(value) for value in camera_center
                        ),
                        pixel_bbox_xyxy=pixel_bbox,
                        width_m=width,
                        length_m=length,
                        height_m=height,
                        max_height_above_water_m=max_height,
                        confidence=confidence,
                    ),
                )
            )

        candidates.sort(key=lambda item: item[1].range_m)
        candidates = candidates[: self.config.max_detections]
        detections: list[Detection] = []
        accepted_components: dict[int, int] = {}
        for detection_id, (component_id, item) in enumerate(candidates):
            detections.append(
                Detection(
                    **{**item.to_dict(), "detection_id": detection_id}
                )
            )
            accepted_components[component_id] = detection_id
        return detections, accepted_components


class USV3DObstacleDetector:
    """YOLO yuklemeden yalnizca metrik derinlikten engel uretir."""

    detection_label = "Depth obstacle"

    def __init__(
        self,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        camera_width=1280,
        camera_height_m=0.25,
        detection_max_range_m=8.0,
        plane_ransac_iterations=350,
    ):
        # VisionNode ile ortak kurucu sozlesmesi icin tutulur. Olcekleme,
        # gercek RGB ve depth kare boyutlarindan dinamik hesaplanir.
        _ = camera_width
        resolved_fx = self._positive_float(fx, 700.0)
        resolved_fy = self._positive_float(fy, resolved_fx)
        resolved_cx = self._optional_float(cx)
        resolved_cy = self._optional_float(cy)
        self.geometry_detector = USV3DDetector(
            DetectorConfig(
                fx=resolved_fx,
                fy=resolved_fy,
                cx=resolved_cx,
                cy=resolved_cy,
                camera_height_m=float(camera_height_m),
                detection_max_range_m=float(detection_max_range_m),
                plane_ransac_iterations=int(plane_ransac_iterations),
            )
        )
        self.last_geometry_error: str | None = None
        self.last_result: FrameResult | None = None
        self.last_depth_shape: tuple[int, int] | None = None
        self.last_image_shape: tuple[int, int] | None = None

    @staticmethod
    def _positive_float(value, fallback):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return number if np.isfinite(number) and number > 0.0 else float(fallback)

    @staticmethod
    def _optional_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _side_from_angle(angle_deg: float) -> str:
        if angle_deg < -2.0:
            return "left"
        if angle_deg > 2.0:
            return "right"
        return "across"

    @staticmethod
    def _scaled_bbox(
        bbox: tuple[int, int, int, int],
        depth_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> list[int]:
        depth_height, depth_width = depth_shape
        image_height, image_width = image_shape
        scale_x = image_width / max(depth_width, 1)
        scale_y = image_height / max(depth_height, 1)
        x1, y1, x2, y2 = bbox
        return [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        ]

    @classmethod
    def _geometry_to_live_detection(
        cls,
        detection: Detection,
        plane_confidence: float,
        bbox: list[int],
    ) -> dict:
        angle = float(detection.bearing_deg)
        return {
            "class": detection.semantic_label,
            "type": "depth_obstacle",
            "confidence": round(float(detection.confidence), 3),
            "distance": round(float(detection.nearest_range_m), 2),
            "angle": round(angle, 3),
            "side": cls._side_from_angle(angle),
            "bbox": bbox,
            "track_id": None,
            "source": "usv_3d",
            "geometry_id": int(detection.detection_id),
            "range_m": round(float(detection.range_m), 3),
            "nearest_range_m": round(
                float(detection.nearest_range_m),
                3,
            ),
            "bearing_deg": round(angle, 3),
            "lateral_m": round(float(detection.lateral_m), 3),
            "forward_m": round(float(detection.forward_m), 3),
            "width_m": round(float(detection.width_m), 3),
            "length_m": round(float(detection.length_m), 3),
            "height_m": round(float(detection.height_m), 3),
            "max_height_above_water_m": round(
                float(detection.max_height_above_water_m),
                3,
            ),
            "geometry_confidence": round(
                float(detection.confidence),
                3,
            ),
            "plane_confidence": round(float(plane_confidence), 3),
        }

    def detect(self, bgr_image, depth_array):
        self.last_image_shape = getattr(bgr_image, "shape", (None, None))[:2]
        self.last_depth_shape = getattr(depth_array, "shape", None)
        if depth_array is None or getattr(depth_array, "ndim", None) != 2:
            self.last_geometry_error = "Gecerli bir metrik depth karesi yok."
            self.last_result = None
            self.last_depth_shape = None
            return []

        try:
            result = self.geometry_detector.detect(depth_array)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.last_geometry_error = str(exc)
            self.last_result = None
            return []

        self.last_geometry_error = None
        self.last_result = result
        image_shape = bgr_image.shape[:2]
        detections = []
        for item in result.detections:
            bbox = self._scaled_bbox(
                item.pixel_bbox_xyxy,
                depth_array.shape,
                image_shape,
            )
            detections.append(
                self._geometry_to_live_detection(
                    item,
                    result.plane.confidence,
                    bbox,
                )
            )
        return detections
