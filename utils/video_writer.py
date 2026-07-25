"""Crash-aware asynchronous video recording shared by both profiles."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Iterable

import cv2


DEFAULT_CODECS = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
)

_STOP = object()


@dataclass(frozen=True)
class VideoWriterResult:
    """Terminal state of one video recording."""

    requested_path: str
    output_path: str | None
    partial_path: str | None
    written_frames: int
    finalized: bool
    error: str | None = None


def _partial_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")


def _available_output_path(requested_path: Path) -> Path:
    """Avoid replacing a completed recording from a same-second restart."""

    candidate = requested_path
    suffix_number = 1
    while candidate.exists() or _partial_path(candidate).exists():
        candidate = requested_path.with_name(
            f"{requested_path.stem}_{suffix_number}{requested_path.suffix}"
        )
        suffix_number += 1
    return candidate


@contextmanager
def _protect_default_termination_handlers():
    """
    Keep a second terminal signal from interrupting a writer drain.

    Launchers install their own idempotent handlers, which must stay active so
    they can start the vehicle-safe shutdown. Only Python/OS default handlers
    are temporarily ignored here.
    """

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handlers = {}
    candidates = (
        (signal.SIGINT, signal.default_int_handler),
        (signal.SIGTERM, signal.SIG_DFL),
    )
    try:
        for signum, interrupting_handler in candidates:
            current_handler = signal.getsignal(signum)
            if current_handler == interrupting_handler:
                previous_handlers[signum] = current_handler
                signal.signal(signum, signal.SIG_IGN)
        yield
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


class QueuedVideoWriter:
    """
    Write frames on a bounded background queue and publish only finalized files.

    The active recording uses ``*.partial.mp4`` (or ``*.partial.avi``). After
    ``VideoWriter.release()`` succeeds and OpenCV can read the first frame, the
    file is atomically renamed to its final name.
    """

    def __init__(
        self,
        video_path,
        frame_size,
        fps,
        *,
        queue_size=100,
        codecs=DEFAULT_CODECS,
        logger=None,
        label="video",
        cv2_module=None,
    ):
        self.requested_path = str(video_path)
        self.frame_size = tuple(frame_size)
        self.fps = float(fps)
        self.codecs = tuple(codecs)
        self.logger = logger or logging.getLogger(__name__)
        self.label = str(label)
        self._cv2 = cv2 if cv2_module is None else cv2_module

        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_requested = threading.Event()
        self._finished = threading.Event()
        self._state_lock = threading.Lock()
        self._accepting = True
        self._close_requested = False
        self._opened_output_path = None
        self._opened_partial_path = None
        self._written_frames = 0
        self._result = None

        # A bounded close timeout can still let the process finish if a codec
        # blocks inside native code. Successful normal shutdown always joins
        # this thread before returning.
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.label}-disk-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def is_alive(self):
        return self._thread.is_alive()

    def enqueue(self, frame):
        """Queue a frame without blocking capture. Return False when rejected."""

        with self._state_lock:
            if not self._accepting or self._finished.is_set():
                return False
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                return False
        return True

    def request_close(self):
        """Stop accepting frames and ask the worker to drain accepted frames."""

        with self._state_lock:
            self._accepting = False
            if self._close_requested:
                return
            self._close_requested = True

        self._stop_requested.set()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            # The worker also observes _stop_requested after it drains the full
            # queue, so shutdown cannot block while trying to enqueue a sentinel.
            pass

    def wait_closed(self, timeout):
        """Wait for finalization and return a result, including timeout state."""

        self._thread.join(timeout=max(0.0, float(timeout)))
        if self._thread.is_alive():
            with self._state_lock:
                output_path = self._opened_output_path
                partial_path = self._opened_partial_path
                written_frames = self._written_frames
            return VideoWriterResult(
                requested_path=self.requested_path,
                output_path=output_path,
                partial_path=partial_path,
                written_frames=written_frames,
                finalized=False,
                error="video writer did not stop before the close timeout",
            )

        with self._state_lock:
            return self._result

    def close(self, timeout=30.0):
        self.request_close()
        with _protect_default_termination_handlers():
            return self.wait_closed(timeout)

    def _open_writer(self):
        requested_path = Path(self.requested_path)
        requested_path.parent.mkdir(parents=True, exist_ok=True)
        open_errors = []

        for fourcc_name, extension in self.codecs:
            codec_requested_path = requested_path.with_suffix(extension)
            output_path = _available_output_path(codec_requested_path)
            partial_path = _partial_path(output_path)
            writer = None

            try:
                writer = self._cv2.VideoWriter(
                    str(partial_path),
                    self._cv2.VideoWriter_fourcc(*fourcc_name),
                    self.fps,
                    self.frame_size,
                )
                opened = writer.isOpened()
            except Exception as exc:  # noqa: BLE001 - codec/backend boundary
                open_errors.append(f"{fourcc_name}: {exc}")
                if writer is not None:
                    try:
                        writer.release()
                    except Exception:  # noqa: BLE001 - best-effort failed open cleanup
                        pass
                continue

            if opened:
                with self._state_lock:
                    self._opened_output_path = str(output_path)
                    self._opened_partial_path = str(partial_path)
                return writer, output_path, partial_path, fourcc_name

            try:
                writer.release()
            except Exception as exc:  # noqa: BLE001 - codec/backend boundary
                open_errors.append(f"{fourcc_name} release: {exc}")
            else:
                open_errors.append(f"{fourcc_name}: codec could not be opened")

            # Failed OpenCV backends sometimes create an empty placeholder.
            try:
                if partial_path.exists() and partial_path.stat().st_size == 0:
                    partial_path.unlink()
            except OSError:
                pass

        details = "; ".join(open_errors) if open_errors else "no codecs configured"
        raise RuntimeError(f"VideoWriter could not be opened ({details})")

    def _next_item(self):
        while True:
            try:
                return self._queue.get(timeout=0.1), True
            except queue.Empty:
                if self._stop_requested.is_set():
                    return _STOP, False

    def _validate_partial_file(self, partial_path):
        try:
            if not partial_path.is_file() or partial_path.stat().st_size <= 0:
                return False, "finalized video file is empty or missing"
        except OSError as exc:
            return False, f"could not inspect finalized video: {exc}"

        capture = None
        try:
            capture = self._cv2.VideoCapture(str(partial_path))
            if not capture.isOpened():
                return False, "OpenCV could not reopen the finalized video"
            frame_ok, _frame = capture.read()
            if not frame_ok:
                return False, "finalized video does not contain a readable frame"
        except Exception as exc:  # noqa: BLE001 - codec/backend boundary
            return False, f"video validation failed: {exc}"
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:  # noqa: BLE001 - best-effort validation cleanup
                    pass
        return True, None

    def _run(self):
        writer = None
        output_path = None
        partial_path = None
        codec_name = None
        written_frames = 0
        error = None
        finalized = False

        try:
            writer, output_path, partial_path, codec_name = self._open_writer()
            self.logger.info(
                "%s recording started: %s | codec=%s | fps=%s | size=%s",
                self.label,
                partial_path,
                codec_name,
                self.fps,
                self.frame_size,
            )

            while True:
                item, item_came_from_queue = self._next_item()
                if item is _STOP:
                    if item_came_from_queue:
                        self._queue.task_done()
                    break

                try:
                    writer.write(item)
                    written_frames += 1
                    with self._state_lock:
                        self._written_frames = written_frames
                finally:
                    self._queue.task_done()

        except Exception as exc:  # noqa: BLE001 - worker must report all failures
            error = f"{type(exc).__name__}: {exc}"
            self.logger.exception("%s recording failed.", self.label)
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:  # noqa: BLE001 - codec/backend boundary
                    release_error = f"VideoWriter.release failed: {exc}"
                    error = f"{error}; {release_error}" if error else release_error
                    self.logger.exception("%s release failed.", self.label)

            if error is None and written_frames <= 0:
                error = "recording stopped before any frame was written"

            if error is None and partial_path is not None:
                valid, validation_error = self._validate_partial_file(partial_path)
                if not valid:
                    error = validation_error

            if error is None and partial_path is not None and output_path is not None:
                try:
                    os.replace(partial_path, output_path)
                    finalized = True
                except OSError as exc:
                    error = f"could not publish finalized video: {exc}"

            result = VideoWriterResult(
                requested_path=self.requested_path,
                output_path=str(output_path) if output_path is not None else None,
                partial_path=str(partial_path) if partial_path is not None else None,
                written_frames=written_frames,
                finalized=finalized,
                error=error,
            )
            with self._state_lock:
                self._accepting = False
                self._result = result
            self._finished.set()

            if finalized:
                self.logger.info(
                    "%s recording finalized: %s | frames=%d",
                    self.label,
                    output_path,
                    written_frames,
                )
            else:
                self.logger.error(
                    "%s recording was not finalized: partial=%s | frames=%d | error=%s",
                    self.label,
                    partial_path,
                    written_frames,
                    error,
                )


def close_video_writers(
    writers: Iterable[QueuedVideoWriter],
    timeout=30.0,
):
    """Request all writers together, then wait for them under one deadline."""

    active_writers = [writer for writer in writers if writer is not None]
    for writer in active_writers:
        writer.request_close()

    deadline = time.monotonic() + max(0.0, float(timeout))
    results = []
    with _protect_default_termination_handlers():
        for writer in active_writers:
            remaining = max(0.0, deadline - time.monotonic())
            results.append(writer.wait_closed(remaining))
    return results
