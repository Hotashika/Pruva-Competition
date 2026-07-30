#!/usr/bin/env python3
"""Replay a recorded Njord RGB/depth/IMU run through Task 2 fusion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vision.task2_fusion_detector import Task2FusionDetector


DEFAULT_MODEL_PATH = (
    REPOSITORY_ROOT
    / "models"
    / "ewasr"
    / "ewasr_resnet18_imu.torchscript"
)


def _read_json(path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _camera_intrinsics(run_dir):
    path = run_dir / "calibration.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file is missing: {path}")
    payload = _read_json(path)
    camera = payload.get("camera", payload)
    left = camera.get("left", camera.get("left_cam", camera))
    try:
        return {
            "fx": float(left["fx"]),
            "fy": float(left.get("fy", left["fx"])),
            "cx": float(left["cx"]),
            "cy": float(left["cy"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Calibration must contain left-camera fx, fy, cx and cy"
        ) from exc


def _imu_by_frame(run_dir, manifest):
    imu_name = manifest.get("imu_file")
    if not imu_name:
        return {}
    path = run_dir / imu_name
    if not path.is_file():
        return {}

    samples = {}
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if row.get("source") not in ("", "zed"):
                continue
            frame_id = row.get("frame_id")
            if not frame_id:
                continue
            try:
                samples[int(frame_id)] = (
                    float(row["roll_rad"]),
                    float(row["pitch_rad"]),
                    float(row["yaw_rad"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
    return samples


def _frame_rows(run_dir, manifest):
    metadata_name = manifest.get("metadata_file")
    if not metadata_name:
        metadata_name = (
            "frames.csv"
            if (run_dir / "frames.csv").is_file()
            else "metadata.csv"
        )
    path = run_dir / metadata_name
    if not path.is_file():
        raise FileNotFoundError(f"Frame metadata is missing: {path}")
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def _row_imu(row, separate_imu):
    frame_id = int(row["frame_id"])
    if frame_id in separate_imu:
        return separate_imu[frame_id]
    try:
        return (
            float(row["roll_rad"]),
            float(row["pitch_rad"]),
            float(row["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError):
        return (0.0, 0.0, 0.0)


def _load_bgr(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def run_replay(
    run_dir,
    *,
    detector=None,
    model_path=DEFAULT_MODEL_PATH,
    output_path=None,
    max_frames=0,
):
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Dataset run does not exist: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    intrinsics = _camera_intrinsics(run_dir)
    rows = _frame_rows(run_dir, manifest)
    imu_samples = _imu_by_frame(run_dir, manifest)
    if max_frames:
        rows = rows[: int(max_frames)]

    if detector is None:
        detector = Task2FusionDetector(
            **intrinsics,
            model_path=model_path,
            shadow_mode=False,
        )

    output_path = (
        run_dir / "task2_fusion_replay.json"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    type_counts = Counter()
    processing_times_ms = []
    segmentation_ready_frames = 0

    for row in rows:
        frame_id = int(row["frame_id"])
        camera_timestamp_ms = int(row["camera_timestamp_ms"])
        left_path = run_dir / row["left_file"]
        depth_path = run_dir / row["depth_file"]
        if not left_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f"Frame {frame_id} files are incomplete: "
                f"{left_path}, {depth_path}"
            )

        bgr_image = _load_bgr(left_path)
        depth = np.load(depth_path, allow_pickle=False)
        imu = _row_imu(row, imu_samples)
        started = time.perf_counter()
        detections = detector.detect(
            bgr_image,
            depth,
            imu=imu,
            now=camera_timestamp_ms / 1000.0,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        processing_times_ms.append(elapsed_ms)
        diagnostics = dict(detector.last_diagnostics)
        if diagnostics.get("segmentation_ready"):
            segmentation_ready_frames += 1

        for detection in detections:
            type_counts[str(detection.get("type", "unknown"))] += 1
        records.append(
            {
                "frame_id": frame_id,
                "camera_timestamp_ms": camera_timestamp_ms,
                "processing_time_ms": round(elapsed_ms, 3),
                "imu_rad": {
                    "roll": imu[0],
                    "pitch": imu[1],
                    "yaw": imu[2],
                },
                "detections": detections,
                "diagnostics": diagnostics,
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "model_path": str(Path(model_path)),
        "frame_count": len(records),
        "segmentation_ready_frames": segmentation_ready_frames,
        "detection_type_counts": dict(sorted(type_counts.items())),
        "mean_processing_time_ms": (
            None
            if not processing_times_ms
            else round(float(np.mean(processing_times_ms)), 3)
        ),
        "p95_processing_time_ms": (
            None
            if not processing_times_ms
            else round(float(np.quantile(processing_times_ms, 0.95)), 3)
        ),
    }
    output_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "frames": records,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Replay a Njord dataset run through Task 2 fusion"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary, output_path = run_replay(
        args.run_dir,
        model_path=args.model,
        output_path=args.output,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
