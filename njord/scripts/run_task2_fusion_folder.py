#!/usr/bin/env python3
"""Run fresh eWaSR inference and Njord Task 2 fusion on image/depth pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vision.ewasr_segmenter import EWaSRResult, EWaSRSegmenter
from vision.task2_fusion_detector import Task2FusionDetector
from vision.usv_3d_detector import USV3DObstacleDetector


SEMANTIC_PALETTE = np.asarray(
    [
        [247, 195, 37],  # obstacle
        [41, 167, 224],  # water
        [90, 75, 164],  # sky
    ],
    dtype=np.uint8,
)
DETECTION_COLORS = {
    "fused_obstacle": (0, 230, 70),
    "seg_depth_obstacle": (255, 145, 0),
    "depth_obstacle": (0, 170, 255),
    "visual_obstacle_candidate": (190, 190, 190),
}
DETECTION_CSV_FIELDS = (
    "frame_id",
    "type",
    "class",
    "distance_m",
    "bearing_deg",
    "side",
    "confidence",
    "segmentation_confidence",
    "geometry_confidence",
    "geometry_overlap",
    "fusion_status",
    "source",
    "bbox",
)


def _files_by_stem(directory: Path, suffixes) -> dict[str, Path]:
    suffixes = {suffix.lower() for suffix in suffixes}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _paired_inputs(root: Path, image_directory="images"):
    image_dir = root / str(image_directory)
    depth_dir = root / "depth"
    for path in (image_dir, depth_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Required input directory is missing: {path}")

    images = _files_by_stem(image_dir, {".jpg", ".jpeg", ".png"})
    depths = _files_by_stem(depth_dir, {".npy"})
    common = sorted(set(images) & set(depths))
    if not common:
        raise RuntimeError("No image/depth pairs share the same stem")
    return [
        (stem, images[stem], depths[stem])
        for stem in common
    ], {
        "image_only": sorted(set(images) - set(depths)),
        "depth_only": sorted(set(depths) - set(images)),
        "image_count": len(images),
        "depth_count": len(depths),
    }


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _working_bgr(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray(rgb, mode="RGB").resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )
    resized = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(resized[:, :, ::-1])


def _working_depth(
    depth: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    if depth.shape == shape:
        return np.ascontiguousarray(depth, dtype=np.float32)
    image = Image.fromarray(
        np.asarray(depth, dtype=np.float32),
        mode="F",
    ).resize(
        (width, height),
        Image.Resampling.NEAREST,
    )
    return np.ascontiguousarray(
        np.asarray(image, dtype=np.float32)
    )


def _working_shape(
    rgb: np.ndarray,
    depth: np.ndarray,
) -> tuple[int, int]:
    width = int(depth.shape[1])
    height = max(
        1,
        int(round(rgb.shape[0] * width / rgb.shape[1])),
    )
    return height, width


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return output
    values = depth[valid]
    low = float(np.quantile(values, 0.02))
    high = float(np.quantile(values, 0.98))
    scale = max(high - low, 1e-6)
    normalized = np.clip((depth - low) / scale, 0.0, 1.0)
    red = 255.0 * normalized
    green = 255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))
    blue = 255.0 * (1.0 - normalized)
    output[valid] = np.stack(
        (red[valid], green[valid], blue[valid]),
        axis=1,
    ).astype(np.uint8)
    return output


def _scaled_bbox(bbox, source_shape, target_shape):
    if not bbox or len(bbox) != 4:
        return None
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    scale_x = target_width / max(source_width, 1)
    scale_y = target_height / max(source_height, 1)
    x1, y1, x2, y2 = bbox
    return (
        int(round(float(x1) * scale_x)),
        int(round(float(y1) * scale_y)),
        int(round(float(x2) * scale_x)),
        int(round(float(y2) * scale_y)),
    )


def _label(detection):
    detection_type = str(detection.get("type", "obstacle"))
    distance = detection.get("distance")
    bearing = detection.get("bearing_deg", detection.get("angle"))
    parts = [detection_type]
    if distance is not None:
        parts.append(f"{float(distance):.2f}m")
    if bearing is not None:
        parts.append(f"{float(bearing):+.1f}deg")
    return " | ".join(parts)


def _draw_detections(image: Image.Image, detections, source_shape):
    draw = ImageDraw.Draw(image)
    target_shape = (image.height, image.width)
    for detection in detections:
        bbox = _scaled_bbox(
            detection.get("bbox"),
            source_shape,
            target_shape,
        )
        if bbox is None:
            continue
        color = DETECTION_COLORS.get(
            str(detection.get("type")),
            (255, 255, 255),
        )
        draw.rectangle(bbox, outline=color, width=4)
        label = _label(detection)
        text_box = draw.textbbox((bbox[0], bbox[1]), label)
        text_height = text_box[3] - text_box[1] + 6
        text_width = text_box[2] - text_box[0] + 6
        text_y = max(0, bbox[1] - text_height)
        draw.rectangle(
            (bbox[0], text_y, bbox[0] + text_width, text_y + text_height),
            fill=(0, 0, 0),
        )
        draw.text((bbox[0] + 3, text_y + 3), label, fill=color)


def _semantic_overlay(
    rgb: np.ndarray,
    segmentation: EWaSRResult,
    detections,
    working_shape,
) -> Image.Image:
    semantic = Image.fromarray(
        SEMANTIC_PALETTE[segmentation.label_map],
        mode="RGB",
    ).resize(
        (rgb.shape[1], rgb.shape[0]),
        Image.Resampling.NEAREST,
    )
    source = Image.fromarray(rgb, mode="RGB")
    overlay = Image.blend(source, semantic, 0.28)
    _draw_detections(overlay, detections, working_shape)
    return overlay


def _detection_csv_row(frame_id, detection):
    return {
        "frame_id": frame_id,
        "type": detection.get("type"),
        "class": detection.get("class"),
        "distance_m": detection.get("distance"),
        "bearing_deg": detection.get(
            "bearing_deg",
            detection.get("angle"),
        ),
        "side": detection.get("side"),
        "confidence": detection.get("confidence"),
        "segmentation_confidence": detection.get(
            "segmentation_confidence"
        ),
        "geometry_confidence": detection.get("geometry_confidence"),
        "geometry_overlap": detection.get("geometry_overlap"),
        "fusion_status": detection.get("fusion_status"),
        "source": detection.get("source"),
        "bbox": json.dumps(detection.get("bbox")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_folder(
    root,
    output_dir,
    *,
    model_path,
    fx,
    fy,
    cx,
    cy,
    camera_height_m,
    detection_max_range_m,
    plane_ransac_iterations,
    depth_scale_to_m=1.0,
    max_frames=0,
    image_directory="images",
    start_index=0,
):
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not model_path.is_file():
        raise FileNotFoundError(f"eWaSR ONNX model is missing: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    overlay_dir = output_dir / "fusion_overlays"
    depth_overlay_dir = output_dir / "depth_overlays"
    semantic_dir = output_dir / "semantic_masks"
    confidence_dir = output_dir / "confidence_masks"
    overlay_dir.mkdir()
    depth_overlay_dir.mkdir()
    semantic_dir.mkdir()
    confidence_dir.mkdir()

    pairs, pairing = _paired_inputs(
        root,
        image_directory=image_directory,
    )
    pairs = pairs[start_index:]
    if max_frames:
        pairs = pairs[:max_frames]
    segmenter = EWaSRSegmenter(
        model_path=model_path,
        device="cpu",
        preserve_aspect_ratio=True,
    )
    if not segmenter.ready:
        raise RuntimeError(
            f"Could not initialize the eWaSR model: {segmenter.last_error}"
        )
    depth_detector = USV3DObstacleDetector(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_height_m=camera_height_m,
        detection_max_range_m=detection_max_range_m,
        plane_ransac_iterations=plane_ransac_iterations,
    )
    detector = Task2FusionDetector(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_detector=depth_detector,
        segmenter=segmenter,
        shadow_mode=False,
        segmentation_hz=1000.0,
        segmentation_cache_max_age_sec=0.01,
        max_depth_m=detection_max_range_m,
    )

    frames = []
    detection_rows = []
    detection_counts = Counter()
    processing_times = []
    input_geometry = None

    for index, (stem, image_path, depth_path) in enumerate(pairs):
        rgb = _load_rgb(image_path)
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2:
            raise ValueError(f"Depth must be two-dimensional: {depth_path}")
        working_shape = _working_shape(rgb, depth)
        bgr = _working_bgr(rgb, working_shape)
        working_depth = (
            _working_depth(depth, working_shape)
            * float(depth_scale_to_m)
        )

        started = time.perf_counter()
        detections = detector.detect(
            bgr,
            working_depth,
            imu=None,
            now=float(index),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        segmentation = detector.last_segmentation
        if segmentation is None:
            raise RuntimeError(
                f"Fresh eWaSR inference failed for {stem}: "
                f"{segmenter.last_error}"
            )
        processing_times.append(elapsed_ms)
        diagnostics = dict(detector.last_diagnostics)

        for detection in detections:
            detection_type = str(detection.get("type", "unknown"))
            detection_counts[detection_type] += 1
            detection_rows.append(_detection_csv_row(stem, detection))

        semantic_image = Image.fromarray(
            SEMANTIC_PALETTE[segmentation.label_map],
            mode="RGB",
        )
        semantic_image.save(semantic_dir / f"{stem}.png")
        confidence_image = Image.fromarray(
            np.clip(
                segmentation.confidence_map * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8),
            mode="L",
        )
        confidence_image.save(confidence_dir / f"{stem}.png")
        overlay = _semantic_overlay(
            rgb,
            segmentation,
            detections,
            working_shape,
        )
        overlay.save(
            overlay_dir / f"{stem}.jpg",
            quality=92,
            subsampling=0,
        )
        depth_image = Image.fromarray(
            _depth_visualization(working_depth),
            mode="RGB",
        )
        _draw_detections(depth_image, detections, working_shape)
        depth_image.save(
            depth_overlay_dir / f"{stem}.jpg",
            quality=92,
            subsampling=0,
        )

        frames.append(
            {
                "frame_id": stem,
                "image_file": str(image_path.relative_to(root)),
                "depth_file": str(depth_path.relative_to(root)),
                "processing_time_ms": round(elapsed_ms, 3),
                "detections": detections,
                "diagnostics": diagnostics,
                "geometry_error": depth_detector.last_geometry_error,
            }
        )
        if input_geometry is None:
            input_geometry = {
                "rgb_shape": list(rgb.shape),
                "source_depth_shape": list(depth.shape),
                "working_shape": list(working_shape),
                "new_segmentation_shape": list(segmentation.label_map.shape),
                "depth_dtype": str(depth.dtype),
            }

    with (output_dir / "detections.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=DETECTION_CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(detection_rows)

    with (output_dir / "frames.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "frame_id",
                "processing_time_ms",
                "detection_count",
                "fused_count",
                "depth_only_count",
                "segmentation_depth_count",
                "visual_only_count",
                "geometry_error",
            ),
        )
        writer.writeheader()
        for frame in frames:
            diagnostics = frame["diagnostics"]
            writer.writerow(
                {
                    "frame_id": frame["frame_id"],
                    "processing_time_ms": frame["processing_time_ms"],
                    "detection_count": len(frame["detections"]),
                    "fused_count": diagnostics.get("fused_count", 0),
                    "depth_only_count": diagnostics.get(
                        "depth_only_count",
                        0,
                    ),
                    "segmentation_depth_count": diagnostics.get(
                        "segmentation_depth_count",
                        0,
                    ),
                    "visual_only_count": diagnostics.get(
                        "visual_only_count",
                        0,
                    ),
                    "geometry_error": frame["geometry_error"],
                }
            )

    times = np.asarray(processing_times, dtype=np.float64)
    geometry_errors = Counter(
        frame["geometry_error"]
        for frame in frames
        if frame["geometry_error"]
    )
    summary = {
        "source_root": str(root),
        "source_image_directory": str(image_directory),
        "output_root": str(output_dir),
        "pairing": pairing,
        "processed_frame_count": len(frames),
        "frames_with_detections": sum(
            bool(frame["detections"]) for frame in frames
        ),
        "total_detection_count": len(detection_rows),
        "detection_type_counts": dict(sorted(detection_counts.items())),
        "processing_ms": {
            "mean": round(float(np.mean(times)), 3),
            "median": round(float(np.median(times)), 3),
            "p95": round(float(np.quantile(times, 0.95)), 3),
            "min": round(float(np.min(times)), 3),
            "max": round(float(np.max(times)), 3),
        },
        "input_geometry": input_geometry,
        "approximate_intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "detector_config": {
            "camera_height_m": camera_height_m,
            "depth_scale_to_m": depth_scale_to_m,
            "detection_max_range_m": detection_max_range_m,
            "segmentation_metric_depth_max_m": detection_max_range_m,
            "semantic_metric_requires_above_water_geometry": True,
            "plane_ransac_iterations": plane_ransac_iterations,
            "shadow_mode": False,
        },
        "water_plane_geometry": {
            "valid_frame_count": sum(
                frame["geometry_error"] is None for frame in frames
            ),
            "error_frame_count": sum(
                frame["geometry_error"] is not None for frame in frames
            ),
            "errors": dict(sorted(geometry_errors.items())),
        },
        "segmentation": {
            "source": "fresh_official_ewasr_onnx_inference",
            "model_inference_performed": True,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "class_mapping": {
                "0": "obstacle",
                "1": "water",
                "2": "sky",
            },
            "preprocessing": "aspect_preserving_letterbox",
            "imu_prior": (
                "Not used; the non-IMU model was selected because the "
                "folder dataset has no synchronized IMU."
                if "_imu" not in model_path.stem.lower()
                else (
                    "Level-camera fallback because no synchronized IMU was "
                    "supplied with this folder dataset."
                )
            ),
        },
        "limitations": [
            "No calibration manifest was supplied.",
            (
                "RGB and depth were mapped to an aspect-preserving "
                f"{input_geometry['working_shape'][1]}x"
                f"{input_geometry['working_shape'][0]} working geometry; "
                "exact registration still requires camera calibration."
            ),
            (
                f"Water-plane geometry used the supplied {camera_height_m:.2f} "
                "m camera height."
            ),
            (
                "Raw depth values were converted to metric depth with "
                f"scale {depth_scale_to_m:.9f}."
            ),
            (
                "No synchronized IMU was supplied; the official non-IMU "
                "eWaSR model was used."
                if "_imu" not in model_path.stem.lower()
                else (
                    "No synchronized IMU was supplied; the eWaSR IMU "
                    "channel used a level-camera horizon prior."
                )
            ),
            "Distances and bearings are experimental.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "detections.json").write_text(
        json.dumps(
            {"summary": summary, "frames": frames},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "Njord Task 2 WaSR + metric-depth fusion output",
                "",
                f"Processed frames: {len(frames)}",
                f"Total detections: {len(detection_rows)}",
                f"Detection types: {dict(sorted(detection_counts.items()))}",
                (
                    "Valid water-plane frames: "
                    f"{summary['water_plane_geometry']['valid_frame_count']}"
                ),
                "",
                "fusion_overlays/: semantic colors and fused detections on RGB",
                "depth_overlays/: detections on colorized depth",
                "semantic_masks/: newly inferred obstacle/water/sky masks",
                "confidence_masks/: newly inferred confidence maps",
                "detections.json: full per-frame results and diagnostics",
                "detections.csv: one row per detection",
                "frames.csv: one row per processed frame",
                "summary.json: aggregate counts, configuration and limitations",
                "",
                "IMPORTANT:",
                "- Masks were generated from images by the official eWaSR ONNX model.",
                "- The existing source masks/ directory was not read.",
                "- RGB aspect ratio was preserved during model inference.",
                f"- Camera height was fixed to {camera_height_m:.2f} m.",
                (
                    "- Semantic candidates received metric distance only "
                    "when supported above the estimated water plane."
                ),
                (
                    "- Raw depth-to-metre scale was fixed to "
                    f"{depth_scale_to_m:.9f}."
                ),
                "- Intrinsics remain approximate because calibration is absent.",
                "- Distances/bearings are experimental, not field-calibrated.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh eWaSR inference, fuse it with metric-depth geometry, "
            "and write visual/JSON/CSV outputs"
        )
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--image-dir-name",
        default="images",
        help="Image subdirectory inside root (for example images or left)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fx", type=float, default=280.0)
    parser.add_argument("--fy", type=float, default=280.0)
    parser.add_argument("--cx", type=float, default=256.0)
    parser.add_argument("--cy", type=float, default=144.0)
    parser.add_argument("--camera-height-m", type=float, default=0.25)
    parser.add_argument(
        "--depth-scale-to-m",
        type=float,
        default=1.0,
        help="Multiply raw depth values by this factor before geometry",
    )
    parser.add_argument("--max-range-m", type=float, default=12.0)
    parser.add_argument("--ransac-iterations", type=int, default=350)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N pairs; 0 processes every pair",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many sorted input pairs before processing",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    values = (
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        args.camera_height_m,
        args.depth_scale_to_m,
        args.max_range_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("Intrinsics and geometry values must be finite")
    if args.fx <= 0.0 or args.fy <= 0.0:
        raise SystemExit("Focal lengths must be positive")
    if (
        args.camera_height_m <= 0.0
        or args.depth_scale_to_m <= 0.0
        or args.max_range_m <= 0.0
    ):
        raise SystemExit(
            "Camera height, depth scale and maximum range must be positive"
        )
    if args.ransac_iterations <= 0:
        raise SystemExit("RANSAC iterations must be positive")
    if args.max_frames < 0:
        raise SystemExit("Maximum frame count cannot be negative")
    if args.start_index < 0:
        raise SystemExit("Start index cannot be negative")

    summary = run_folder(
        args.root,
        args.output_dir,
        model_path=args.model,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        camera_height_m=args.camera_height_m,
        detection_max_range_m=args.max_range_m,
        plane_ransac_iterations=args.ransac_iterations,
        depth_scale_to_m=args.depth_scale_to_m,
        max_frames=args.max_frames,
        image_directory=args.image_dir_name,
        start_index=args.start_index,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
