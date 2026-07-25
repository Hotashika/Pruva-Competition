"""Manually collect synchronized ZED stereo, depth and IMU training data."""

from __future__ import annotations

import argparse
import math
import queue
import sys
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
            "Stop njord/main.py before running this module; "
            "the ZED camera can only be opened once."
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


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        collect(
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
