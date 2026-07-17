#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from njord.core.shared_frame_source import (
    close_capture_source,
    open_or_start_capture_source,
)
from njord.vision.horizon_mask import create_horizon_mask, render_horizon_overlay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one synchronized ZED frame and visualize its IMU horizon."
    )
    parser.add_argument("--output-dir", default="output/horizon_debug")
    parser.add_argument("--roll-offset-deg", type=float, default=0.0)
    parser.add_argument("--pitch-offset-deg", type=float, default=0.0)
    parser.add_argument("--flip-roll", action="store_true")
    parser.add_argument("--flip-pitch", action="store_true")
    parser.add_argument("--invert-mask", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = None
    capture_process = None
    stop_event = None
    try:
        source, capture_process, stop_event = open_or_start_capture_source()
        frame_data = source.read(timeout=3.0)
        fx, fy, cx, cy = source.get_camera_intrinsics()
        frame = frame_data["frame_bgr"]
        imu = frame_data["imu"]

        mask = create_horizon_mask(
            width=frame.shape[1],
            height=frame.shape[0],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            roll=imu["roll"],
            pitch=imu["pitch"],
            roll_offset=math.radians(args.roll_offset_deg),
            pitch_offset=math.radians(args.pitch_offset_deg),
            flip_roll=args.flip_roll,
            flip_pitch=args.flip_pitch,
            invert=args.invert_mask,
        )
        overlay = render_horizon_overlay(frame, mask)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "imu_mask.png"
        overlay_path = output_dir / "imu_overlay.jpg"
        cv2.imwrite(str(mask_path), mask * 255)
        cv2.imwrite(str(overlay_path), overlay)

        print(
            "Saved IMU horizon debug: "
            f"roll={math.degrees(imu['roll']):.2f} deg, "
            f"pitch={math.degrees(imu['pitch']):.2f} deg"
        )
        print(f"  mask:    {mask_path.resolve()}")
        print(f"  overlay: {overlay_path.resolve()}")

        if args.show:
            cv2.imshow("Njord IMU horizon", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    finally:
        close_capture_source(source, capture_process, stop_event)


if __name__ == "__main__":
    main()
