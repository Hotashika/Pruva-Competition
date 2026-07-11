import time

from pymavlink import mavutil

DEFAULT_CONNECTION_STRING = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_HEARTBEAT_TIMEOUT = 15


def _close_mavlink(master):
    try:
        master.close()
    except Exception:
        pass


def _get_source_ids(message, master):
    source_system = (
        message.get_srcSystem()
        if hasattr(message, "get_srcSystem")
        else master.target_system
    )
    source_component = (
        message.get_srcComponent()
        if hasattr(message, "get_srcComponent")
        else master.target_component
    )
    return source_system, source_component


def _is_vehicle_heartbeat(message):
    mavlink = mavutil.mavlink
    autopilot = getattr(message, "autopilot", None)
    vehicle_type = getattr(message, "type", None)

    if vehicle_type == getattr(mavlink, "MAV_TYPE_GCS", None):
        return False
    if autopilot == getattr(mavlink, "MAV_AUTOPILOT_INVALID", None):
        return False

    return True


def connect_mavlink(
        connection_string=DEFAULT_CONNECTION_STRING,
        baud=DEFAULT_BAUD,
        heartbeat_timeout=DEFAULT_HEARTBEAT_TIMEOUT,
        logger=None,
):
    """MAVLink baglantisi kurar ve heartbeat bekledikten sonra master nesnesini dondurur.

    Ornek connection_string degerleri:
        - Orange Cube USB: "/dev/ttyACM0"
        - TELEM / USB-TTL: "/dev/ttyUSB0"
        - SITL: "udpin:127.0.0.1:14550"
    """
    if logger is not None:
        logger.info(f"MAVLink baglaniyor: {connection_string}, baud={baud}")

    master = mavutil.mavlink_connection(connection_string, baud=baud)

    if logger is not None:
        logger.info("Heartbeat bekleniyor...")

    deadline = time.monotonic() + float(heartbeat_timeout)
    heartbeat = None
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        candidate = master.recv_match(
            type="HEARTBEAT",
            blocking=True,
            timeout=min(1.0, remaining),
        )
        if candidate is None:
            continue

        source_system, source_component = _get_source_ids(candidate, master)
        if source_system in (None, 0) or source_component in (None, 0):
            continue
        if not _is_vehicle_heartbeat(candidate):
            if logger is not None:
                logger.warn(
                    "Arac olmayan MAVLink heartbeat yok sayildi: "
                    f"system={source_system}, component={source_component}"
                )
            continue

        heartbeat = candidate
        master.target_system = source_system
        master.target_component = source_component
        break

    if heartbeat is None:
        _close_mavlink(master)
        raise TimeoutError(
            f"{heartbeat_timeout} saniye icinde MAVLink heartbeat alinamadi."
        )

    source_system, source_component = _get_source_ids(heartbeat, master)

    if source_system in (None, 0) or source_component in (None, 0):
        _close_mavlink(master)
        raise ConnectionError(
            "Gecersiz MAVLink heartbeat kaynagi: "
            f"system={source_system}, component={source_component}"
        )

    if master.target_system in (None, 0):
        master.target_system = source_system
    if master.target_component in (None, 0):
        master.target_component = source_component

    if logger is not None:
        logger.info(
            f"MAVLink baglandi. system={master.target_system}, "
            f"component={master.target_component}"
        )

    return master


__all__ = [
    "DEFAULT_CONNECTION_STRING",
    "DEFAULT_BAUD",
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "connect_mavlink",
]
