import pytest
import numpy as np

from utils.frame_cadence import FrameCadence
from utils.retry_backoff import exponential_backoff_delay
from vision.render import draw_detections


def test_ten_fps_cadence_selects_ten_frames_from_fifteen_fps_source():
    cadence = FrameCadence(10)
    source_times_ms = [index * (1000.0 / 15.0) for index in range(15)]

    selected = [
        index
        for index, timestamp_ms in enumerate(source_times_ms)
        if cadence.due(timestamp_ms)
    ]

    assert selected == [0, 2, 3, 5, 6, 8, 9, 11, 12, 14]


def test_cadence_skips_missed_periods_without_replaying_backlog():
    cadence = FrameCadence(10)

    assert cadence.due(0.0)
    assert cadence.due(550.0)
    assert not cadence.due(551.0)
    assert cadence.due(600.0)


def test_exponential_backoff_is_bounded():
    delays = [
        exponential_backoff_delay(failure_count)
        for failure_count in range(1, 10)
    ]

    assert delays[:5] == [0.01, 0.02, 0.04, 0.08, 0.16]
    assert delays[5:] == [0.25, 0.25, 0.25, 0.25]


def test_cached_detection_renderer_draws_without_loading_a_model():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    rendered = draw_detections(
        frame,
        [{
            "class": "buoy",
            "confidence": 0.91,
            "distance": 3.4,
            "bbox": [20, 20, 70, 65],
            "track_id": None,
        }],
    )

    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(frame) == 0


@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_cadence_rejects_invalid_rates(fps):
    with pytest.raises(ValueError):
        FrameCadence(fps)
