import json
import os
import tempfile
import time


TASK_STATUS_IDLE = "idle"
TASK_STATUS_SELECTED = "selected"
TASK_STATUS_STOP = "stop"
TASK_STATUS_EMERGENCY = "emergency"


def default_task_selection_file():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "runtime", "mission_state.json")


def build_task_selection_state(command):
    command = int(command)
    state = {
        "selected_task": None,
        "status": TASK_STATUS_IDLE,
        "command": command,
        "updated_at": time.time(),
    }

    if command in (1, 2, 3, 4):
        state["selected_task"] = command
        state["status"] = TASK_STATUS_SELECTED
    elif command == 90:
        state["status"] = TASK_STATUS_STOP
    elif command == 99:
        state["status"] = TASK_STATUS_EMERGENCY
    elif command != 0:
        state["status"] = "invalid"

    return state


def write_task_selection(path, command):
    state = build_task_selection_state(command)
    write_task_selection_state(path, state)
    return state


def write_task_selection_state(path, state):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=".mission_state.",
        suffix=".json",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=True, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def read_task_selection(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return build_task_selection_state(0)


def clear_task_selection(path):
    state = build_task_selection_state(0)
    write_task_selection_state(path, state)
    return state
