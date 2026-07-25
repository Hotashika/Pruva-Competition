import importlib
import sys
import types

import numpy as np


class _FakeFlask:
    def __init__(self, _name):
        pass

    def route(self, _path):
        return lambda function: function

    def run(self, **_kwargs):
        pass


def _load_with_fake_flask(module_name):
    fake_flask = types.ModuleType("flask")
    fake_flask.Flask = _FakeFlask
    fake_flask.Response = lambda *args, **kwargs: (args, kwargs)
    previous = sys.modules.get("flask")
    sys.modules["flask"] = fake_flask
    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        if previous is None:
            sys.modules.pop("flask", None)
        else:
            sys.modules["flask"] = previous


def test_video_source_frame_is_delivered_only_once_per_frame_id():
    from njord.core import shared_state

    video_server = _load_with_fake_flask("njord.servers.video_server")
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    with shared_state.frame_condition:
        shared_state.latest_frame = frame
        shared_state.latest_frame_id = 41
        shared_state.frame_condition.notify_all()

    delivered = video_server._wait_for_source_frame(40, timeout=0.01)

    assert delivered is not None
    assert delivered[0] == 41
    assert video_server._wait_for_source_frame(41, timeout=0.01) is None


def test_data_snapshot_is_delivered_only_once_per_data_id():
    from njord.core import shared_state

    data_server = _load_with_fake_flask("njord.servers.data_server")
    with shared_state.data_condition:
        shared_state.latest_data_id = 73
        shared_state.latest_timestamp = 123456
        shared_state.latest_imu = {"roll": 0.1, "pitch": 0.2, "yaw": 0.3}
        shared_state.latest_center_depth = 4.25
        shared_state.data_condition.notify_all()

    snapshot = data_server._wait_for_data(72, timeout=0.01)

    assert snapshot == {
        "data_id": 73,
        "timestamp": 123456,
        "imu": {"roll": 0.1, "pitch": 0.2, "yaw": 0.3},
        "center_depth": 4.25,
    }
    assert data_server._wait_for_data(73, timeout=0.01) is None
