import math


class FrameCadence:
    """Select source frames at a stable target rate without building backlog."""

    def __init__(self, fps):
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be a positive finite number")
        self.period_ms = 1000.0 / fps
        self.next_due_ms = None

    def due(self, now_ms):
        now_ms = float(now_ms)
        if not math.isfinite(now_ms):
            raise ValueError("now_ms must be finite")

        if self.next_due_ms is None:
            self.next_due_ms = now_ms + self.period_ms
            return True

        if now_ms + 1e-9 < self.next_due_ms:
            return False

        missed_periods = max(
            1,
            int(math.floor((now_ms - self.next_due_ms) / self.period_ms)) + 1,
        )
        self.next_due_ms += missed_periods * self.period_ms
        return True
