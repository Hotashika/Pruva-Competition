import math


def exponential_backoff_delay(
    failure_count,
    *,
    initial_sec=0.01,
    maximum_sec=0.25,
):
    """Return a bounded exponential delay for consecutive transient failures."""
    failure_count = max(1, int(failure_count))
    initial_sec = float(initial_sec)
    maximum_sec = float(maximum_sec)
    if (
        not math.isfinite(initial_sec)
        or not math.isfinite(maximum_sec)
        or initial_sec <= 0.0
        or maximum_sec < initial_sec
    ):
        raise ValueError("backoff bounds must be finite and positive")

    exponent = min(failure_count - 1, 30)
    return min(initial_sec * (2 ** exponent), maximum_sec)
