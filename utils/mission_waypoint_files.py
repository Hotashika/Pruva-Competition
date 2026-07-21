"""MAVLink mission komutlarını waypoint dosyalarına eşleme yardımcıları."""

import os
from pathlib import Path


VALID_MISSION_COMMANDS = (1, 2, 3, 4)


def format_mission_waypoint_files(mapping):
    return ",".join(
        f"{command}:{filename}"
        for command, filename in sorted(mapping.items())
    )


def parse_mission_waypoint_files(value):
    """``command:filename`` listesini güvenli bir eşlemeye dönüştürür."""
    mapping = {}
    for raw_entry in str(value or "").split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            command_text, filename = entry.split(":", 1)
            command = int(command_text.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Gecersiz MAVLINK_MISSION_WAYPOINT_FILES girdisi: {entry!r}"
            ) from exc

        filename = filename.strip()
        if command not in VALID_MISSION_COMMANDS:
            raise ValueError(f"Waypoint senkron komutu 1..4 olmali: {command}")
        if os.path.basename(filename) != filename or not filename.lower().endswith(
            ".waypoints"
        ):
            raise ValueError(f"Gecersiz waypoint dosya adi: {filename!r}")
        mapping[command] = filename
    return mapping


def resolve_mission_waypoint_directory(mapping, value):
    """Require an explicit destination whenever waypoint sync is enabled."""
    directory = str(value or "").strip()
    if mapping and not directory:
        raise ValueError(
            "mission_waypoint_directory is required when mission waypoint "
            "synchronization is configured"
        )
    return Path(directory) if directory else None
