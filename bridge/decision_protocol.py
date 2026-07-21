import math
import struct


DECISION_PAYLOAD_TYPE = 201
DECISION_PROTOCOL_VERSION = 1
DECISION_STRUCT = struct.Struct("<BIBbHHf20s30s44s16s")
MISSION_NAME_TO_ID = {"task1": 1, "task2": 2, "task3": 3, "task4": 4}
MISSION_ID_TO_NAME = {value: key for key, value in MISSION_NAME_TO_ID.items()}


def _fixed_text(value, size):
    return str(value or "").encode("utf-8")[:size].ljust(size, b"\x00")


def encode_decision_payload(
        sequence,
        active_mission,
        stage,
        action,
        reason,
        colreg_rule="",
        collision_risk=None,
        current_target=0,
        target_count=0,
        progress_percent=0.0,
):
    mission_id = MISSION_NAME_TO_ID.get(str(active_mission).strip().lower(), 0)
    risk_value = -1 if collision_risk is None else int(bool(collision_risk))
    progress = max(0.0, min(float(progress_percent), 100.0))
    if not math.isfinite(progress):
        progress = 0.0
    return DECISION_STRUCT.pack(
        DECISION_PROTOCOL_VERSION,
        int(sequence) & 0xFFFFFFFF,
        mission_id,
        risk_value,
        max(0, min(int(current_target), 65535)),
        max(0, min(int(target_count), 65535)),
        progress,
        _fixed_text(stage, 20),
        _fixed_text(action, 30),
        _fixed_text(reason, 44),
        _fixed_text(colreg_rule, 16),
    )


def decode_decision_payload(payload):
    raw = bytes(payload)
    if len(raw) < DECISION_STRUCT.size:
        raise ValueError("Decision payload is shorter than expected")
    (
        version,
        sequence,
        mission_id,
        risk_value,
        current_target,
        target_count,
        progress_percent,
        stage,
        action,
        reason,
        colreg_rule,
    ) = DECISION_STRUCT.unpack(raw[:DECISION_STRUCT.size])
    if version != DECISION_PROTOCOL_VERSION:
        raise ValueError(f"Unsupported decision protocol version: {version}")
    if risk_value not in (-1, 0, 1):
        raise ValueError("Invalid collision-risk value")
    if not math.isfinite(progress_percent) or not 0.0 <= progress_percent <= 100.0:
        raise ValueError("Invalid mission progress value")

    def text(raw_value):
        return raw_value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    return {
        "sequence": sequence,
        "active_mission": MISSION_ID_TO_NAME.get(mission_id, "unknown"),
        "stage": text(stage),
        "action": text(action),
        "reason": text(reason),
        "colreg_rule": text(colreg_rule),
        "collision_risk": None if risk_value == -1 else bool(risk_value),
        "current_target": current_target,
        "target_count": target_count,
        "progress_percent": progress_percent,
    }
