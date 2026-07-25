#!/usr/bin/env python3
"""TEKNOFEST terminal monitoru baslaticisi."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.system_monitor import run_monitor


if __name__ == "__main__":
    run_monitor(
        title="PRUVA TEKNOFEST",
        node_name="teknofest_system_monitor",
        subscribe_mission_status=True,
    )
