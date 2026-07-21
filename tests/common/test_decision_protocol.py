from bridge.decision_protocol import (
    DECISION_STRUCT,
    decode_decision_payload,
    encode_decision_payload,
)


def test_decision_payload_round_trip():
    payload = encode_decision_payload(
        sequence=42,
        active_mission="task2",
        stage="AVOIDING",
        action="Turn to starboard",
        reason="Collision risk requires give-way manoeuvre",
        colreg_rule="RULE 15/16",
        collision_risk=True,
        current_target=2,
        target_count=6,
        progress_percent=33.3,
    )

    assert len(payload) == DECISION_STRUCT.size
    assert len(payload) <= 128
    decoded = decode_decision_payload(payload)
    assert decoded["sequence"] == 42
    assert decoded["active_mission"] == "task2"
    assert decoded["stage"] == "AVOIDING"
    assert decoded["action"] == "Turn to starboard"
    assert decoded["collision_risk"] is True
    assert decoded["current_target"] == 2
    assert decoded["target_count"] == 6
    assert abs(decoded["progress_percent"] - 33.3) < 0.001


def test_unknown_collision_risk_round_trip():
    payload = encode_decision_payload(
        sequence=0,
        active_mission="task1",
        stage="WAITING",
        action="Wait",
        reason="Checking sensors",
        collision_risk=None,
    )
    assert decode_decision_payload(payload)["collision_risk"] is None
