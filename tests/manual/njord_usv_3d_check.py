"""GPS gerektirmeyen manuel Njord 3B engel algilama kontrolu.

Bu script ZED goruntusu ve metrik depth verisiyle ``USV3DObstacleDetector``
calistirir. Sonuclari terminale ve guvenli test topic'ine yayinlar:

    /vision/test_detections

Script mission, GPS, MAVLink, hiz veya konum komutu baslatmaz. Aracta acik bir
Njord ana sureci yokken calistirilmasi onerilir.

Kullanim:

    source /opt/ros/kilted/setup.bash
    python3.12 tests/manual/njord_usv_3d_check.py

Sinirli sure calistirmak icin:

    python3.12 tests/manual/njord_usv_3d_check.py --duration-sec 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from njord.core.shared_frame_source import (
    close_capture_source,
    open_or_start_capture_source,
)
from vision.usv_3d_detector import USV3DObstacleDetector


TEST_TOPIC = "/vision/test_detections"


class ManualDetectionPublisher(Node):
    def __init__(self):
        super().__init__("njord_usv_3d_manual_check")
        self.publisher = self.create_publisher(String, TEST_TOPIC, 10)

    def publish_detections(self, payload):
        message = String()
        message.data = json.dumps(payload)
        self.publisher.publish(message)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="GPS olmadan Njord USV 3B engel algilama kontrolu.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="0 verilirse Ctrl+C'ye kadar calisir.",
    )
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=0.25,
        help="Kamera optik merkezinin su yuzeyinden yuksekligi.",
    )
    parser.add_argument(
        "--max-range-m",
        type=float,
        default=12.0,
        help="Algilama icin en uzak mesafe.",
    )
    parser.add_argument(
        "--ransac-iterations",
        type=int,
        default=100,
        help="Su duzlemi RANSAC tekrar sayisi.",
    )
    parser.add_argument(
        "--full-ransac-interval",
        type=int,
        default=5,
        help="Tam RANSAC calistirilacak geometri karesi araligi.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=2,
        help="Depth boyutlarini bu tam sayi katsayisiyla kucultur.",
    )
    parser.add_argument(
        "--geometry-hz",
        type=float,
        default=7.5,
        help="3B geometrinin calisma hizi; ara karelerde son sonuc kullanilir.",
    )
    parser.add_argument(
        "--status-interval-sec",
        type=float,
        default=1.0,
        help="Bos karelerde terminal durum mesaji araligi.",
    )
    return parser


def _validate_args(args):
    if args.duration_sec < 0.0:
        raise ValueError("duration-sec sifir veya pozitif olmali.")
    if args.camera_height_m <= 0.0:
        raise ValueError("camera-height-m pozitif olmali.")
    if args.max_range_m <= 0.0:
        raise ValueError("max-range-m pozitif olmali.")
    if args.ransac_iterations < 1:
        raise ValueError("ransac-iterations en az 1 olmali.")
    if args.full_ransac_interval < 1:
        raise ValueError("full-ransac-interval en az 1 olmali.")
    if args.downsample_factor < 1:
        raise ValueError("downsample-factor en az 1 olmali.")
    if args.geometry_hz <= 0.0:
        raise ValueError("geometry-hz pozitif olmali.")
    if args.status_interval_sec <= 0.0:
        raise ValueError("status-interval-sec pozitif olmali.")


def _print_frame_status(frame_id, detector, detections):
    result = detector.last_result
    if result is None:
        print(
            f"[USV3D] frame={frame_id} detector_error="
            f"{detector.last_geometry_error}"
        )
        return

    print(
        f"[USV3D] frame={frame_id} "
        f"plane_confidence={result.plane.confidence:.3f} "
        f"obstacle_count={len(detections)}"
    )
    for detection in detections:
        print(
            "  obstacle "
            f"id={detection['geometry_id']} "
            f"distance={detection['distance']:.2f}m "
            f"angle={detection['angle']:.1f}deg "
            f"side={detection['side']} "
            f"confidence={detection['confidence']:.3f}"
        )


def run_check(args):
    _validate_args(args)
    frame_source = None
    capture_process = None
    capture_stop_event = None
    node = None
    ros_initialized = False

    try:
        frame_source, capture_process, capture_stop_event = (
            open_or_start_capture_source()
        )
        fx, fy, cx, cy = frame_source.get_camera_intrinsics()
        detector = USV3DObstacleDetector(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_height_m=args.camera_height_m,
            detection_max_range_m=args.max_range_m,
            plane_ransac_iterations=args.ransac_iterations,
            plane_full_ransac_interval=args.full_ransac_interval,
            depth_downsample_factor=args.downsample_factor,
            geometry_hz=args.geometry_hz,
        )

        rclpy.init(args=[])
        ros_initialized = True
        node = ManualDetectionPublisher()

        source_mode = (
            "existing capture"
            if capture_process is None
            else "local test capture"
        )
        print(
            "[USV3D] Test started. "
            f"source={source_mode}, topic={TEST_TOPIC}, "
            f"intrinsics=({fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}), "
            f"geometry_hz={args.geometry_hz:.1f}, "
            f"downsample={args.downsample_factor}x, "
            f"ransac={args.ransac_iterations}/"
            f"{args.full_ransac_interval}-frame"
        )
        print("[USV3D] GPS/MAVLink/vehicle movement commands are disabled.")

        started_at = time.monotonic()
        last_status_at = 0.0
        timeout_count = 0

        while rclpy.ok():
            if (
                args.duration_sec > 0.0
                and time.monotonic() - started_at >= args.duration_sec
            ):
                break

            try:
                frame = frame_source.read(timeout=3.0)
            except TimeoutError:
                timeout_count += 1
                print(
                    f"[USV3D] New camera frame was not received "
                    f"(timeout {timeout_count}/3)."
                )
                if timeout_count >= 3:
                    return 2
                continue

            timeout_count = 0
            detections = detector.detect(
                frame["frame_bgr"],
                frame["depth"],
            )
            payload = {
                "frame_id": frame["frame_id"],
                "camera_timestamp_ms": frame["timestamp_ms"],
                "test_mode": True,
                "detections": detections,
            }
            node.publish_detections(payload)
            rclpy.spin_once(node, timeout_sec=0.0)

            now = time.monotonic()
            if detections or now - last_status_at >= args.status_interval_sec:
                _print_frame_status(
                    frame["frame_id"],
                    detector,
                    detections,
                )
                last_status_at = now

        return 0
    except KeyboardInterrupt:
        print("\n[USV3D] Test stopped by user.")
        return 0
    except Exception as exc:
        print(f"[USV3D] Test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if ros_initialized and rclpy.ok():
            rclpy.shutdown()
        close_capture_source(
            frame_source,
            capture_process,
            capture_stop_event,
        )


def main():
    args = _build_parser().parse_args()
    raise SystemExit(run_check(args))


if __name__ == "__main__":
    main()
