"""Idempotent launcher signal handling for safe vehicle and recorder shutdown."""

from __future__ import annotations

import signal
import threading


class GracefulShutdown:
    """Convert SIGINT/SIGTERM into a one-shot cooperative shutdown request."""

    def __init__(
        self,
        *,
        stop_event=None,
        on_first_signal=None,
        notify=print,
    ):
        self._stop_event = stop_event
        self._on_first_signal = on_first_signal
        self._notify = notify
        self._requested = threading.Event()
        self._installed_handlers = {}
        self._repeated_notice_printed = False
        self.signal_number = None

    def _emit(self, message):
        try:
            self._notify(message)
        except Exception:
            # Signal handling must still stop the mission/capture if terminal
            # output is interrupted or already unavailable.
            pass

    @property
    def is_requested(self):
        return self._requested.is_set()

    def wait(self, timeout=None):
        return self._requested.wait(timeout)

    def install(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers can only be installed by the main thread")
        if self._installed_handlers:
            return

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._installed_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self.handle_signal)

    def restore(self):
        if not self._installed_handlers:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers can only be restored by the main thread")

        for signum, previous_handler in self._installed_handlers.items():
            signal.signal(signum, previous_handler)
        self._installed_handlers.clear()

    def handle_signal(self, signum, _frame):
        if self._requested.is_set():
            if not self._repeated_notice_printed:
                self._emit(
                    "[SYSTEM] Shutdown is already in progress; "
                    "waiting for video finalization."
                )
                self._repeated_notice_printed = True
            return

        self.signal_number = signum
        self._requested.set()

        signal_name = signal.Signals(signum).name
        self._emit(f"\n[SYSTEM] {signal_name} received; safe shutdown started.")
        try:
            if self._on_first_signal is not None:
                self._on_first_signal(signum)
        except Exception as exc:  # noqa: BLE001 - signal path must remain idempotent
            self._emit(
                "[SYSTEM] Could not notify the mission process during shutdown: "
                f"{exc}"
            )
        finally:
            if self._stop_event is not None:
                self._stop_event.set()
