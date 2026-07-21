import struct


DETECTION_PAYLOAD_TYPE = 200
DETECTION_PROTOCOL_VERSION = 1
DETECTION_CLASS_NAME_SIZE = 32
DETECTION_STRUCT = struct.Struct("<BIIfffii32s")


def encode_detection_payload(
        sequence,
        frame_id,
        confidence,
        depth,
        angle,
        latitude,
        longitude,
        class_name,
):
    class_bytes = str(class_name).encode("utf-8")[:DETECTION_CLASS_NAME_SIZE - 1]
    return DETECTION_STRUCT.pack(
        DETECTION_PROTOCOL_VERSION,
        int(sequence) & 0xFFFFFFFF,
        int(frame_id) & 0xFFFFFFFF,
        float(confidence),
        float(depth),
        float(angle),
        int(round(float(latitude) * 1e7)),
        int(round(float(longitude) * 1e7)),
        class_bytes.ljust(DETECTION_CLASS_NAME_SIZE, b"\x00"),
    )


def decode_detection_payload(payload):
    raw = bytes(payload)
    if len(raw) < DETECTION_STRUCT.size:
        raise ValueError("Detection payload is shorter than expected")

    (
        version,
        sequence,
        frame_id,
        confidence,
        depth,
        angle,
        latitude_e7,
        longitude_e7,
        class_bytes,
    ) = DETECTION_STRUCT.unpack(raw[:DETECTION_STRUCT.size])
    if version != DETECTION_PROTOCOL_VERSION:
        raise ValueError(f"Unsupported detection protocol version: {version}")

    return {
        "sequence": sequence,
        "frame_id": frame_id,
        "confidence": confidence,
        "depth": depth,
        "angle": angle,
        "lat": latitude_e7 / 1e7,
        "lon": longitude_e7 / 1e7,
        "class_name": class_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace"),
    }
