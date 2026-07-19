from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
from PIL import Image


SCHEMA_VERSION = 2
METADATA_FIELDS = (
    "frame_id",
    "camera_timestamp_ms",
    "system_timestamp_utc",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "left_file",
    "right_file",
    "depth_file",
)


class DatasetRecorderError(RuntimeError):
    """Raised when the asynchronous dataset writer cannot persist a frame."""


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    camera_timestamp_ms: int
    system_timestamp_utc: str
    roll: float
    pitch: float
    yaw: float
    left_image: np.ndarray
    right_image: Optional[np.ndarray]
    depth_map: np.ndarray


ImageWriter = Callable[[Path, np.ndarray, int], None]
DepthWriter = Callable[[Path, np.ndarray], None]
_STOP = object()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"run_{timestamp}_{os.getpid()}"


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"Value is not JSON/YAML compatible: {type(value).__name__}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            _json_compatible(payload),
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def _write_jpeg_atomic(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    temporary_path = path.with_name(f"{path.stem}.part{path.suffix}")
    if image.dtype != np.uint8:
        raise ValueError("JPEG image arrays must use uint8 dtype")
    if image.ndim == 2:
        output_image = Image.fromarray(image)
    elif image.ndim == 3 and image.shape[2] == 1:
        output_image = Image.fromarray(image[:, :, 0])
    elif image.ndim == 3 and image.shape[2] == 3:
        # Capture and vision buffers use OpenCV's BGR channel order.
        output_image = Image.fromarray(image[:, :, ::-1])
    elif image.ndim == 3 and image.shape[2] == 4:
        # ZED capture buffers are BGRA; JPEG does not store alpha.
        output_image = Image.fromarray(image[:, :, [2, 1, 0]])
    else:
        raise ValueError(
            "JPEG image arrays must be grayscale, BGR, or BGRA uint8 arrays"
        )
    output_image.save(
        temporary_path,
        format="JPEG",
        quality=int(jpeg_quality),
    )
    os.replace(temporary_path, path)


def _write_depth_atomic(path: Path, depth_map: np.ndarray) -> None:
    temporary_path = path.with_name(f"{path.stem}.part{path.suffix}")
    with temporary_path.open("wb") as output_file:
        np.save(output_file, depth_map, allow_pickle=False)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


class DatasetRecorder:
    """Asynchronously persist frame-synchronised stereo, depth and IMU data.

    The recorder deliberately owns only the disk-writing layer. Camera capture
    and object detection can feed it independently using ``frame_id`` as their
    shared key. Images are copied before they enter the queue so a capture
    buffer may be reused immediately after ``record_frame`` returns.

    ``calibration.yaml`` is written as JSON-compatible YAML 1.2. This avoids a
    runtime PyYAML dependency while keeping the file consumable by YAML tools.
    """

    def __init__(
        self,
        output_root: os.PathLike[str] | str,
        *,
        run_name: Optional[str] = None,
        calibration: Optional[Mapping[str, Any]] = None,
        record_right: bool = True,
        jpeg_quality: int = 90,
        queue_size: int = 64,
        manifest_interval_frames: int = 30,
        image_writer: Optional[ImageWriter] = None,
        depth_writer: Optional[DepthWriter] = None,
    ):
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in the range [1, 100]")
        if int(queue_size) < 1:
            raise ValueError("queue_size must be at least 1")
        if int(manifest_interval_frames) < 1:
            raise ValueError("manifest_interval_frames must be at least 1")

        selected_run_name = _default_run_name() if run_name is None else str(run_name)
        if (
            not selected_run_name
            or selected_run_name in (".", "..")
            or Path(selected_run_name).name != selected_run_name
            or "/" in selected_run_name
            or "\\" in selected_run_name
        ):
            raise ValueError("run_name must be a single non-empty directory name")

        self.output_root = Path(output_root)
        self.run_name = selected_run_name
        self.run_dir = self.output_root / self.run_name
        self.left_dir = self.run_dir / "left"
        self.right_dir = self.run_dir / "right"
        self.depth_dir = self.run_dir / "depth"
        self.metadata_path = self.run_dir / "metadata.csv"
        self.manifest_path = self.run_dir / "manifest.json"
        self.calibration_path = self.run_dir / "calibration.yaml"
        self.record_right = bool(record_right)
        self.jpeg_quality = int(jpeg_quality)
        self.manifest_interval_frames = int(manifest_interval_frames)
        self._image_writer = image_writer or _write_jpeg_atomic
        self._depth_writer = depth_writer or _write_depth_atomic

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(exist_ok=False)
        self.left_dir.mkdir()
        if self.record_right:
            self.right_dir.mkdir()
        self.depth_dir.mkdir()

        calibration_payload = {
            "schema_version": SCHEMA_VERSION,
            "provided": calibration is not None,
            "camera": {} if calibration is None else calibration,
        }
        # JSON is valid YAML 1.2 and can be parsed without adding PyYAML to the
        # runtime dependencies.
        _write_json_atomic(self.calibration_path, calibration_payload)

        self._metadata_file = self.metadata_path.open(
            "x",
            newline="",
            encoding="utf-8",
            buffering=1,
        )
        self._metadata_writer = csv.DictWriter(
            self._metadata_file,
            fieldnames=METADATA_FIELDS,
        )
        self._metadata_writer.writeheader()
        self._metadata_file.flush()

        self._queue: queue.Queue[FramePacket | object] = queue.Queue(
            maxsize=int(queue_size)
        )
        self._state_lock = threading.Lock()
        self._manifest_write_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._accepting = True
        self._shutdown_started = False
        self._closed = False
        self._last_seen_frame_id: Optional[int] = None
        self._last_seen_camera_timestamp_ms: Optional[int] = None
        self._frames_seen = 0
        self._frames_accepted = 0
        self._frames_written = 0
        self._frames_dropped = 0
        self._frames_failed = 0
        self._writer_errors: list[str] = []
        self._finalization_error: Optional[str] = None
        self._last_close_timeout_utc: Optional[str] = None

        created_utc = _utc_now_iso()
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.run_name,
            "status": "recording",
            "created_utc": created_utc,
            "updated_utc": created_utc,
            "closed_utc": None,
            "last_close_timeout_utc": None,
            "record_right": self.record_right,
            "jpeg_quality": self.jpeg_quality,
            "queue_size": int(queue_size),
            "manifest_interval_frames": self.manifest_interval_frames,
            "metadata_file": self.metadata_path.name,
            "calibration_file": self.calibration_path.name,
            "left_directory": self.left_dir.name,
            "right_directory": self.right_dir.name if self.record_right else None,
            "depth_directory": self.depth_dir.name,
            "frames_seen": 0,
            "frames_accepted": 0,
            "frames_written": 0,
            "frames_dropped": 0,
            "frames_failed": 0,
            "writer_error_count": 0,
            "writer_errors": [],
        }
        _write_json_atomic(self.manifest_path, self._manifest)

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"dataset-writer-{self.run_name}",
            daemon=False,
        )
        self._writer_thread.start()

    @property
    def frames_seen(self) -> int:
        with self._state_lock:
            return self._frames_seen

    @property
    def frames_accepted(self) -> int:
        with self._state_lock:
            return self._frames_accepted

    @property
    def frames_written(self) -> int:
        with self._state_lock:
            return self._frames_written

    @property
    def frames_dropped(self) -> int:
        with self._state_lock:
            return self._frames_dropped

    @property
    def frames_failed(self) -> int:
        with self._state_lock:
            return self._frames_failed

    def record_frame(
        self,
        *,
        frame_id: int,
        camera_timestamp_ms: int,
        left_image: np.ndarray,
        right_image: Optional[np.ndarray],
        depth_map: np.ndarray,
        roll: float,
        pitch: float,
        yaw: float,
        system_timestamp_utc: Optional[str] = None,
    ) -> bool:
        """Queue one frame without blocking the capture loop.

        Returns ``True`` when the frame was accepted and ``False`` when the
        bounded queue was full. Frame IDs and camera timestamps must be
        strictly increasing even when an intermediate frame is dropped.
        """

        frame_id = int(frame_id)
        camera_timestamp_ms = int(camera_timestamp_ms)
        if frame_id <= 0:
            raise ValueError("frame_id must be positive")
        if camera_timestamp_ms < 0:
            raise ValueError("camera_timestamp_ms cannot be negative")
        orientation = (float(roll), float(pitch), float(yaw))
        if not all(math.isfinite(item) for item in orientation):
            raise ValueError("roll, pitch and yaw must be finite")
        if not isinstance(left_image, np.ndarray) or left_image.size == 0:
            raise ValueError("left_image must be a non-empty numpy array")
        if self.record_right and (
            not isinstance(right_image, np.ndarray) or right_image.size == 0
        ):
            raise ValueError(
                "right_image must be a non-empty numpy array when record_right=True"
            )
        if not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
            raise ValueError("depth_map must be a non-empty numpy array")
        if depth_map.ndim != 2 or not np.issubdtype(depth_map.dtype, np.number):
            raise ValueError("depth_map must be a two-dimensional numeric array")
        if left_image.shape[:2] != depth_map.shape:
            raise ValueError("depth_map dimensions must match the camera image")

        packet = FramePacket(
            frame_id=frame_id,
            camera_timestamp_ms=camera_timestamp_ms,
            system_timestamp_utc=system_timestamp_utc or _utc_now_iso(),
            roll=orientation[0],
            pitch=orientation[1],
            yaw=orientation[2],
            left_image=np.ascontiguousarray(left_image).copy(),
            right_image=None
            if not self.record_right or right_image is None
            else np.ascontiguousarray(right_image).copy(),
            depth_map=np.ascontiguousarray(depth_map).copy(),
        )

        # Queue acceptance and close's sentinel insertion share this lock. This
        # prevents a concurrent capture callback from placing a frame behind
        # the sentinel while the recorder is shutting down.
        with self._state_lock:
            if self._closed or not self._accepting:
                raise RuntimeError("DatasetRecorder is closed")
            if self._writer_errors:
                raise DatasetRecorderError(self._writer_errors[0])
            if (
                self._last_seen_frame_id is not None
                and frame_id <= self._last_seen_frame_id
            ):
                raise ValueError("frame_id must be strictly increasing")
            if (
                self._last_seen_camera_timestamp_ms is not None
                and camera_timestamp_ms <= self._last_seen_camera_timestamp_ms
            ):
                raise ValueError("camera_timestamp_ms must be strictly increasing")
            self._last_seen_frame_id = frame_id
            self._last_seen_camera_timestamp_ms = camera_timestamp_ms
            self._frames_seen += 1
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                self._frames_dropped += 1
                return False
            self._frames_accepted += 1
        return True

    def _writer_loop(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._stop_requested.is_set():
                        return
                    continue

                try:
                    if item is _STOP:
                        return
                    assert isinstance(item, FramePacket)
                    self._write_packet(item)
                    with self._state_lock:
                        self._frames_written += 1
                except Exception as exc:  # noqa: BLE001 - persisted and surfaced on close
                    error_text = f"frame={getattr(item, 'frame_id', '?')}: {exc}"
                    with self._state_lock:
                        self._frames_failed += 1
                        self._writer_errors.append(error_text)
                finally:
                    self._queue.task_done()

                self._checkpoint_manifest_if_due()
                if self._stop_requested.is_set() and self._queue.empty():
                    return
        finally:
            self._finalize_writer()

    def _write_packet(self, packet: FramePacket) -> None:
        filename = f"{packet.frame_id:08d}.jpg"
        depth_filename = f"{packet.frame_id:08d}.npy"
        left_path = self.left_dir / filename
        right_path = self.right_dir / filename if self.record_right else None
        depth_path = self.depth_dir / depth_filename
        created_paths: list[Path] = []

        try:
            self._image_writer(left_path, packet.left_image, self.jpeg_quality)
            created_paths.append(left_path)
            if right_path is not None:
                assert packet.right_image is not None
                self._image_writer(
                    right_path,
                    packet.right_image,
                    self.jpeg_quality,
                )
                created_paths.append(right_path)
            self._depth_writer(depth_path, packet.depth_map)
            created_paths.append(depth_path)

            self._metadata_writer.writerow(
                {
                    "frame_id": packet.frame_id,
                    "camera_timestamp_ms": packet.camera_timestamp_ms,
                    "system_timestamp_utc": packet.system_timestamp_utc,
                    "roll_rad": f"{packet.roll:.9f}",
                    "pitch_rad": f"{packet.pitch:.9f}",
                    "yaw_rad": f"{packet.yaw:.9f}",
                    "left_file": left_path.relative_to(self.run_dir).as_posix(),
                    "right_file": ""
                    if right_path is None
                    else right_path.relative_to(self.run_dir).as_posix(),
                    "depth_file": depth_path.relative_to(self.run_dir).as_posix(),
                }
            )
            self._metadata_file.flush()
        except Exception:
            for created_path in created_paths:
                try:
                    created_path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def _snapshot_manifest(
        self,
        status: str,
        *,
        terminal: bool = False,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._state_lock:
            snapshot = dict(self._manifest)
            snapshot.update(
                {
                    "status": status,
                    "updated_utc": now,
                    "closed_utc": now if terminal else snapshot.get("closed_utc"),
                    "last_close_timeout_utc": self._last_close_timeout_utc,
                    "frames_seen": self._frames_seen,
                    "frames_accepted": self._frames_accepted,
                    "frames_written": self._frames_written,
                    "frames_dropped": self._frames_dropped,
                    "frames_failed": self._frames_failed,
                    "writer_error_count": len(self._writer_errors),
                    "writer_errors": list(self._writer_errors[:20]),
                }
            )
        return snapshot

    def _write_manifest(self, status: str, *, terminal: bool = False) -> None:
        with self._manifest_write_lock:
            snapshot = self._snapshot_manifest(status, terminal=terminal)
            _write_json_atomic(self.manifest_path, snapshot)
            self._manifest = snapshot

    def _checkpoint_manifest_if_due(self) -> None:
        with self._state_lock:
            completed_frames = self._frames_written + self._frames_failed
            should_update = (
                completed_frames > 0
                and completed_frames % self.manifest_interval_frames == 0
            )
            status = "recording" if self._accepting else "closing"
        if not should_update:
            return

        try:
            self._write_manifest(status)
        except Exception as exc:  # noqa: BLE001 - surfaced by close and final manifest
            with self._state_lock:
                self._writer_errors.append(f"manifest checkpoint: {exc}")

    def _finalize_writer(self) -> None:
        cleanup_errors: list[str] = []
        try:
            self._metadata_file.flush()
        except Exception as exc:  # noqa: BLE001 - persisted and surfaced by close
            cleanup_errors.append(f"metadata flush: {exc}")
        try:
            self._metadata_file.close()
        except Exception as exc:  # noqa: BLE001 - persisted and surfaced by close
            cleanup_errors.append(f"metadata close: {exc}")

        if cleanup_errors:
            with self._state_lock:
                self._writer_errors.extend(cleanup_errors)

        with self._state_lock:
            has_errors = bool(self._writer_errors)

        try:
            self._write_manifest("error" if has_errors else "closed", terminal=True)
        except Exception as exc:  # noqa: BLE001 - no manifest is available to persist this
            with self._state_lock:
                self._finalization_error = f"manifest finalization: {exc}"
        finally:
            with self._state_lock:
                self._closed = True

    def close(self, timeout: float = 30.0) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._accepting = False
            enqueue_stop = not self._shutdown_started
            self._shutdown_started = True

        # Wake an idle writer immediately. If the bounded queue is full, the
        # stop event makes the writer exit after draining all accepted frames.
        if enqueue_stop:
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                pass
        self._stop_requested.set()

        timeout_seconds = max(0.0, float(timeout))
        self._writer_thread.join(timeout=timeout_seconds)
        if self._writer_thread.is_alive():
            timeout_utc = _utc_now_iso()
            with self._state_lock:
                self._last_close_timeout_utc = timeout_utc
            try:
                self._write_manifest("timeout")
            except Exception as exc:  # noqa: BLE001 - retain the timeout as primary failure
                raise TimeoutError(
                    "DatasetRecorder writer thread did not stop before timeout; "
                    f"manifest update also failed: {exc}"
                ) from exc
            raise TimeoutError("DatasetRecorder writer thread did not stop before timeout")

        with self._state_lock:
            has_errors = bool(self._writer_errors)
            finalization_error = self._finalization_error

        if finalization_error is not None:
            raise DatasetRecorderError(finalization_error)

        if has_errors:
            raise DatasetRecorderError(self._manifest["writer_errors"][0])

    def __enter__(self) -> "DatasetRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
