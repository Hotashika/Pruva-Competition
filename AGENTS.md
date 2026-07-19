# Repository Guidelines

## Project Layout

This repository contains two parallel competition applications:

- `njord/`: Njord competition runtime, missions, vision, camera, servers, config, waypoints, scripts, and models.
- `teknofest/`: TEKNOFEST competition runtime with the same package layout.
- `bridge/`: ROS 2 and MAVLink bridge code.
- `utils/`: Shared MAVLink, waypoint, battery, task selection, and JSON helpers.
- `tests/`: Pytest suite grouped by `common/`, `njord/`, and `teknofest/`.

Keep competition-specific logic inside its own package. Move code to `utils/` or `bridge/` only when it is genuinely shared.

## Environment Setup

Use Python 3.12, especially for the TEKNOFEST/ZED integration.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ZED Python bindings are not installed from `requirements.txt`. Install the provided wheel when available:

```bash
pip install pyzed-5.1-cp312-cp312-linux_x86_64.whl
```

Alternatively, use the SDK installer helper:

```bash
python /usr/local/zed/get_python_api.py
```

Mission launchers may require ROS 2, a MAVLink endpoint, ZED SDK/camera access, and the expected model files.

## Development Commands

Run the full test suite:

```bash
python -m pytest tests
```

Run focused tests while developing:

```bash
python -m pytest tests/njord/test_task4_surprise.py -q
python -m pytest tests/teknofest/test_task2_obstacle_avoidance.py -q
```

Run mission entry points:

```bash
python njord/main.py --task-1
python teknofest/main.py --task-2
```

Hardware-dependent tests under `tests/common/` should be run explicitly and only with the required devices connected.

## Code Style

- Use four-space indentation.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for constants.
- Preserve the existing `taskN_...py` mission naming pattern.
- Keep imports grouped and avoid unrelated formatting churn.
- Prefer existing local helpers and patterns over introducing new abstractions.

There is no configured formatter or linter, so keep edits small, readable, and consistent with nearby code.

## Testing Expectations

Use pytest for new and existing tests. Test files should be named `test_*.py`, and test functions or methods should start with `test_`.

Add regression tests in the matching area:

- Shared helpers: `tests/common/`
- Njord behavior: `tests/njord/`
- TEKNOFEST behavior: `tests/teknofest/`

Mock or isolate ROS, MAVLink, camera, network, shared memory, and vehicle dependencies where practical. Prioritize mission state transitions, timeout handling, geometry, waypoint behavior, and safety guards.

## Assets And Runtime Files

Model weights belong under the relevant package's `models/` directory. Do not commit generated caches, temporary logs, shared-memory dumps, local environment folders, or runtime output.

## Git And Pull Requests

Use short imperative Conventional Commit subjects, for example:

```text
feat: add docking timeout guard
fix: update waypoint parsing
test: cover orange boundary guard
```

Keep each commit focused. Pull requests should include:

- Affected competition package and task.
- Operational impact.
- Configuration, model, or dependency changes.
- Tests run.
- Any validation that still requires vessel hardware.

Include logs, screenshots, or recorded detection output when changing vision, telemetry, mission execution, or UI behavior.
