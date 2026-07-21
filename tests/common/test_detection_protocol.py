import math

from bridge.detection_protocol import (
    DETECTION_STRUCT,
    decode_detection_payload,
    encode_detection_payload,
)


def test_detection_payload_round_trip():
    payload = encode_detection_payload(
        sequence=17,
        frame_id=902,
        confidence=0.913,
        depth=4.72,
        angle=-8.4,
        latitude=37.9521234,
        longitude=32.5012345,
        class_name="red_buoy",
    )

    assert len(payload) == DETECTION_STRUCT.size
    decoded = decode_detection_payload(payload.ljust(128, b"\x00"))
    assert decoded["sequence"] == 17
    assert decoded["frame_id"] == 902
    assert decoded["class_name"] == "red_buoy"
    assert math.isclose(decoded["confidence"], 0.913, rel_tol=1e-5)
    assert math.isclose(decoded["depth"], 4.72, rel_tol=1e-5)
    assert math.isclose(decoded["angle"], -8.4, rel_tol=1e-5)
    assert decoded["lat"] == 37.9521234
    assert decoded["lon"] == 32.5012345


def test_detection_class_name_is_bounded():
    payload = encode_detection_payload(0, 0, 1, 1, 0, 1, 1, "x" * 100)
    assert decode_detection_payload(payload)["class_name"] == "x" * 31
