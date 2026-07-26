"""Manually collect synchronized ZED stereo, depth and IMU training data."""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from multiprocessing import get_context
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
COMPETITION_ROOT = PROJECT_ROOT.parent
if str(COMPETITION_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPETITION_ROOT))

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "datasets"


def collection_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if (
        not normalized
        or normalized in (".", "..")
        or Path(normalized).name != normalized
        or "/" in normalized
        or "\\" in normalized
    ):
        raise argparse.ArgumentTypeError(
            "collection name must be a single non-empty directory name"
        )
    return normalized


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("value must be zero or a positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect synchronized left/right JPEG, metric depth NPY and IMU "
            "metadata without starting a mission."
        ),
        epilog=(
            "Default mode opens the ZED camera and requires njord/main.py to be "
            "stopped. --attach connects to a running Njord system instead."
        ),
    )
    parser.add_argument(
        "--attach",
        action="store_true",
        help=(
            "attach to a running njord/main.py and write separate frames, GPS, "
            "IMU and Task 2 kinematics files"
        ),
    )
    parser.add_argument(
        "--name",
        type=collection_name,
        default="manual",
        help="collection folder name under the output directory (default: manual)",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=5.0,
        help="dataset sampling rate (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=non_negative_float,
        default=0.0,
        help="recording duration in seconds; zero records until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"dataset root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    return parser


def _ros_timestamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _quaternion_to_euler(x: float, y: float, z: float, w: float):
    """Convert a ROS quaternion to roll, pitch and yaw in radians."""

    x = float(x)
    y = float(y)
    z = float(z)
    w = float(w)

    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = (
        math.copysign(math.pi / 2.0, sin_pitch)
        if abs(sin_pitch) >= 1.0
        else math.asin(sin_pitch)
    )

    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


def collect(*, output_dir: Path, name: str, fps: float, duration: float) -> Path:
    # Import lazily so ``--help`` and unit tests do not require the ZED SDK.
    from njord.core import capture_proc

    normalized_name = collection_name(name)
    validated_fps = positive_float(str(fps))
    validated_duration = non_negative_float(str(duration))

    context = get_context("spawn")
    stop_event = context.Event()
    ready_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=capture_proc.run_capture,
        kwargs={
            "stop_event": stop_event,
            "ready_queue": ready_queue,
            "dataset_output_root": str(Path(output_dir).expanduser().resolve()),
            "dataset_name": normalized_name,
            "dataset_record_fps": validated_fps,
            "publish_shared_memory": False,
        },
        daemon=False,
    )

    process.start()
    ready_message = None
    try:
        try:
            ready_message = ready_queue.get(timeout=20.0)
        except queue.Empty as exc:
            raise RuntimeError("ZED camera did not become ready within 20 seconds") from exc

        if "error" in ready_message:
            raise RuntimeError(str(ready_message["error"]))
        if "dataset_error" in ready_message:
            raise RuntimeError(str(ready_message["dataset_error"]))
        if "dataset_run_dir" not in ready_message:
            raise RuntimeError("dataset recorder did not return an output directory")

        run_dir = Path(ready_message["dataset_run_dir"])
        print(f"[DATASET] Recording -> {run_dir}")
        if validated_duration > 0.0:
            print(f"[DATASET] Duration: {validated_duration:.1f} seconds")
        else:
            print("[DATASET] Press Ctrl+C to finish recording.")

        deadline = (
            None
            if validated_duration == 0.0
            else time.monotonic() + validated_duration
        )
        while process.is_alive():
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[DATASET] Stop requested.")
    finally:
        stop_event.set()
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)

    if process.exitcode not in (0, None):
        raise RuntimeError(f"capture process exited with code {process.exitcode}")
    if ready_message is None:
        raise RuntimeError("collection stopped before the ZED camera became ready")
    run_dir = Path(ready_message["dataset_run_dir"])
    print(f"[DATASET] Finalized -> {run_dir}")
    return run_dir


def collect_attached(
    *,
    output_dir: Path,
    name: str,
    fps: float,
    duration: float,
) -> Path:
    """Record a manually controlled Task 2 test from a running Njord stack."""

    # Keep --help and the default camera-owning collector importable without
    # ROS, OpenCV or a running ZED installation.
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu, NavSatFix
    from std_msgs.msg import String

    from njord.core.dataset_recorder import Task2TestRecorder
    from njord.core.shared_frame_source import SharedFrameSource
    from njord.missions.task2_collision_avoidance import KINEMATICS_TOPIC

    normalized_name = collection_name(name)
    validated_fps = positive_float(str(fps))
    validated_duration = non_negative_float(str(duration))
    output_root = Path(output_dir).expanduser().resolve() / normalized_name
    sample_interval_ms = max(1, int(round(1000.0 / validated_fps)))

    recorder = None
    source = None
    node = None
    spin_thread = None
    owns_rclpy_context = False
    run_dir = None

    class LiveSensorNode(Node):
        def __init__(self, active_recorder):
            super().__init__("njord_manual_test_recorder")
            self.recorder = active_recorder
            self.failure = None
            self._frame_reference_lock = threading.Lock()
            self._latest_frame_reference = (None, None)
            self.create_subscription(NavSatFix, "/cube/gps", self._gps_callback, 50)
            self.create_subscription(Imu, "/cube/imu", self._imu_callback, 100)
            self.create_subscription(
                String,
                KINEMATICS_TOPIC,
                self._kinematics_callback,
                50,
            )

        def update_frame_reference(self, frame_id, camera_timestamp_ms):
            with self._frame_reference_lock:
                self._latest_frame_reference = (
                    int(frame_id),
                    int(camera_timestamp_ms),
                )

        def _frame_reference(self):
            with self._frame_reference_lock:
                return self._latest_frame_reference

        def _fail(self, stream_name, exc):
            if self.failure is None:
                self.failure = RuntimeError(
                    f"{stream_name} recording failed: {exc}"
                )
                self.get_logger().error(str(self.failure))

        def _gps_callback(self, message):
            if self.failure is not None:
                return
            try:
                frame_id, camera_timestamp_ms = self._frame_reference()
                self.recorder.record_gps(
                    latitude_deg=message.latitude,
                    longitude_deg=message.longitude,
                    altitude_m=message.altitude,
                    ros_timestamp_ns=_ros_timestamp_ns(message),
                    frame_id=frame_id,
                    camera_timestamp_ms=camera_timestamp_ms,
                    position_covariance_type=message.position_covariance_type,
                )
            except Exception as exc:  # noqa: BLE001 - callback reports disk errors
                self._fail("GPS", exc)

        def _imu_callback(self, message):
            if self.failure is not None:
                return
            try:
                orientation = message.orientation
                roll, pitch, yaw = _quaternion_to_euler(
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                )
                angular_velocity = message.angular_velocity
                linear_acceleration = message.linear_acceleration
                frame_id, camera_timestamp_ms = self._frame_reference()
                self.recorder.record_imu_sample(
                    source="pixhawk",
                    ros_timestamp_ns=_ros_timestamp_ns(message),
                    frame_id=frame_id,
                    camera_timestamp_ms=camera_timestamp_ms,
                    roll_rad=roll,
                    pitch_rad=pitch,
                    yaw_rad=yaw,
                    orientation_x=orientation.x,
                    orientation_y=orientation.y,
                    orientation_z=orientation.z,
                    orientation_w=orientation.w,
                    angular_velocity_x_rad_s=angular_velocity.x,
                    angular_velocity_y_rad_s=angular_velocity.y,
                    angular_velocity_z_rad_s=angular_velocity.z,
                    linear_acceleration_x_m_s2=linear_acceleration.x,
                    linear_acceleration_y_m_s2=linear_acceleration.y,
                    linear_acceleration_z_m_s2=linear_acceleration.z,
                )
            except Exception as exc:  # noqa: BLE001 - callback reports disk errors
                self._fail("IMU", exc)

        def _kinematics_callback(self, message):
            if self.failure is not None:
                return
            try:
                payload = json.loads(message.data)
                if not isinstance(payload, dict):
                    raise ValueError("kinematics payload must be a JSON object")
                self.recorder.record_kinematics(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self.get_logger().warning(
                    f"Invalid {KINEMATICS_TOPIC} sample ignored: {exc}"
                )
            except Exception as exc:  # noqa: BLE001 - callback reports disk errors
                self._fail("kinematics", exc)

    try:
        try:
            source = SharedFrameSource(retries=50, delay=0.1)
        except RuntimeError as exc:
            raise RuntimeError(
                "--attach requires a running njord/main.py shared-memory capture"
            ) from exc

        fx, fy, cx, cy = source.get_camera_intrinsics()
        calibration = {
            "left": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "live_attach": {
                "record_fps": validated_fps,
                "records_rgb": True,
                "records_depth": True,
                "records_right": False,
                "separate_files": [
                    "frames.csv",
                    "gps.csv",
                    "imu.csv",
                    "kinematics.csv",
                ],
            },
        }
        recorder = Task2TestRecorder(
            output_root,
            calibration=calibration,
        )
        run_dir = recorder.run_dir

        if not rclpy.ok():
            rclpy.init()
            owns_rclpy_context = True
        node = LiveSensorNode(recorder)
        spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(node,),
            name="njord-test-recorder-ros",
            daemon=True,
        )
        spin_thread.start()

        print(f"[DATASET] Attached recording -> {run_dir}")
        print(
            "[DATASET] Separate outputs: frames.csv, gps.csv, imu.csv, "
            "kinematics.csv, depth/*.npy"
        )
        if validated_duration > 0.0:
            print(f"[DATASET] Duration: {validated_duration:.1f} seconds")
        else:
            print("[DATASET] Press Ctrl+C to finish recording.")

        deadline = (
            None
            if validated_duration == 0.0
            else time.monotonic() + validated_duration
        )
        last_record_timestamp_ms = None
        while rclpy.ok():
            if node.failure is not None:
                raise node.failure
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                frame = source.read(timeout=0.5)
            except TimeoutError:
                continue

            timestamp_ms = int(frame["timestamp_ms"])
            node.update_frame_reference(frame["frame_id"], timestamp_ms)
            if (
                last_record_timestamp_ms is not None
                and timestamp_ms - last_record_timestamp_ms < sample_interval_ms
            ):
                continue
            last_record_timestamp_ms = timestamp_ms
            imu = frame["imu"]
            recorder.record_frame(
                frame_id=frame["frame_id"],
                camera_timestamp_ms=timestamp_ms,
                left_image=frame["frame_bgr"],
                right_image=None,
                depth_map=frame["depth"],
                roll=imu["roll"],
                pitch=imu["pitch"],
                yaw=imu["yaw"],
            )
    except KeyboardInterrupt:
        print("\n[DATASET] Stop requested.")
    finally:
        if node is not None:
            node.destroy_node()
        if owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if source is not None:
            source.close()
        if recorder is not None:
            recorder.close()

    if run_dir is None:
        raise RuntimeError("attached collection stopped before recording started")
    print(f"[DATASET] Finalized -> {run_dir}")
    return run_dir


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        collector = collect_attached if arguments.attach else collect
        collector(
            output_dir=arguments.output_dir,
            name=arguments.name,
            fps=arguments.fps,
            duration=arguments.duration,
        )
    except (argparse.ArgumentTypeError, OSError, RuntimeError, ValueError) as exc:
        print(f"[DATASET] Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
