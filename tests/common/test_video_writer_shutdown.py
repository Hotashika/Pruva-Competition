import os
from pathlib import Path
import signal
import threading
import time

from utils.shutdown_signals import GracefulShutdown
from utils.video_writer import QueuedVideoWriter


class _FakeCapture:
    def __init__(self, path):
        self.path = Path(path)

    def isOpened(self):
        if not self.path.is_file():
            return False
        payload = self.path.read_bytes()
        return b"HEADER" in payload and b"FRAME" in payload and b"MOOV" in payload

    def read(self):
        return self.isOpened(), object()

    def release(self):
        return None


class _FakeWriter:
    def __init__(self, backend, path, opened):
        self.backend = backend
        self.path = Path(path)
        self.opened = opened
        self.released = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"HEADER" if opened else b"")

    def isOpened(self):
        return self.opened

    def write(self, _frame):
        if self.backend.fail_write:
            raise RuntimeError("synthetic write failure")
        with self.backend.lock:
            with self.path.open("ab") as output_file:
                output_file.write(b"FRAME")

    def release(self):
        if self.released:
            return
        self.released = True
        self.backend.writer_release_count += 1
        if self.opened:
            with self.path.open("ab") as output_file:
                output_file.write(b"MOOV")


class _FakeCv2:
    def __init__(self, *, opened_codecs=None, fail_write=False, open_gate=None):
        self.opened_codecs = list(opened_codecs or [True])
        self.fail_write = fail_write
        self.open_gate = open_gate
        self.open_started = threading.Event()
        self.writer_release_count = 0
        self.lock = threading.Lock()
        self.opened_paths = []

    @staticmethod
    def VideoWriter_fourcc(*_name):
        return 1

    def VideoWriter(self, path, _fourcc, _fps, _frame_size):
        self.open_started.set()
        if self.open_gate is not None:
            self.open_gate.wait(timeout=2.0)
        opened = self.opened_codecs.pop(0) if self.opened_codecs else False
        self.opened_paths.append(path)
        return _FakeWriter(self, path, opened)

    @staticmethod
    def VideoCapture(path):
        return _FakeCapture(path)


def test_writer_atomically_publishes_only_after_release_and_validation(tmp_path):
    backend = _FakeCv2()
    final_path = tmp_path / "run.mp4"
    recorder = QueuedVideoWriter(
        final_path,
        (64, 48),
        5,
        cv2_module=backend,
        label="test",
    )

    assert recorder.enqueue(object()) is True
    assert recorder.enqueue(object()) is True
    result = recorder.close(timeout=2.0)

    assert result.finalized is True
    assert result.error is None
    assert result.written_frames == 2
    assert result.output_path == str(final_path)
    assert final_path.is_file()
    assert not (tmp_path / "run.partial.mp4").exists()
    assert backend.writer_release_count == 1
    assert final_path.read_bytes().endswith(b"MOOV")


def test_write_failure_releases_writer_but_keeps_partial_file(tmp_path):
    backend = _FakeCv2(fail_write=True)
    final_path = tmp_path / "run.mp4"
    recorder = QueuedVideoWriter(
        final_path,
        (64, 48),
        5,
        cv2_module=backend,
        label="test",
    )

    assert recorder.enqueue(object()) is True
    result = recorder.close(timeout=2.0)

    assert result.finalized is False
    assert "synthetic write failure" in result.error
    assert result.partial_path == str(tmp_path / "run.partial.mp4")
    assert not final_path.exists()
    assert Path(result.partial_path).is_file()
    assert backend.writer_release_count == 1


def test_full_queue_does_not_block_close_request(tmp_path):
    open_gate = threading.Event()
    backend = _FakeCv2(open_gate=open_gate)
    recorder = QueuedVideoWriter(
        tmp_path / "run.mp4",
        (64, 48),
        5,
        queue_size=1,
        cv2_module=backend,
        label="test",
    )
    assert backend.open_started.wait(timeout=1.0)
    assert recorder.enqueue(object()) is True

    started = time.monotonic()
    recorder.request_close()
    elapsed = time.monotonic() - started
    open_gate.set()
    result = recorder.wait_closed(timeout=2.0)

    assert elapsed < 0.2
    assert result.finalized is True
    assert result.written_frames == 1


def test_codec_fallback_uses_partial_avi_then_publishes_avi(tmp_path):
    backend = _FakeCv2(opened_codecs=[False, False, True])
    recorder = QueuedVideoWriter(
        tmp_path / "run.mp4",
        (64, 48),
        5,
        cv2_module=backend,
        label="test",
    )

    assert recorder.enqueue(object()) is True
    result = recorder.close(timeout=2.0)

    assert result.finalized is True
    assert result.output_path == str(tmp_path / "run.avi")
    assert (tmp_path / "run.avi").is_file()
    assert not (tmp_path / "run.partial.avi").exists()


class _CountingStopEvent:
    def __init__(self, calls):
        self.calls = calls

    def set(self):
        self.calls.append("stop_event")


def test_repeated_shutdown_signal_is_idempotent_and_stops_mission_first():
    calls = []
    messages = []
    controller = GracefulShutdown(
        stop_event=_CountingStopEvent(calls),
        on_first_signal=lambda _signum: calls.append("mission"),
        notify=messages.append,
    )

    controller.handle_signal(signal.SIGINT, None)
    controller.handle_signal(signal.SIGINT, None)

    assert controller.is_requested is True
    assert controller.signal_number == signal.SIGINT
    assert calls == ["mission", "stop_event"]
    assert sum("safe shutdown started" in message for message in messages) == 1
    assert sum("already in progress" in message for message in messages) == 1


def test_installed_handler_absorbs_repeated_real_sigint():
    calls = []
    controller = GracefulShutdown(
        on_first_signal=lambda signum: calls.append(signum),
        notify=lambda _message: None,
    )
    controller.install()
    try:
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
    finally:
        controller.restore()

    assert calls == [signal.SIGINT]


def test_shutdown_still_sets_stop_event_when_mission_notification_fails():
    calls = []

    def fail_mission_stop(_signum):
        calls.append("mission")
        raise OSError("synthetic signal failure")

    controller = GracefulShutdown(
        stop_event=_CountingStopEvent(calls),
        on_first_signal=fail_mission_stop,
        notify=lambda _message: None,
    )

    controller.handle_signal(signal.SIGTERM, None)

    assert calls == ["mission", "stop_event"]
