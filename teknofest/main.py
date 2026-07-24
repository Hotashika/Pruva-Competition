import argparse
import os
import queue
import shlex
import signal
import subprocess

import sys
import threading
import time
from multiprocessing import get_context

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ROOT = os.path.dirname(PROJECT_ROOT)
while COMPETITION_ROOT in sys.path:
    sys.path.remove(COMPETITION_ROOT)
sys.path.insert(0, COMPETITION_ROOT)

from teknofest.core import capture_proc
from teknofest.core import data_writer
from teknofest.config.mission_config import (
    MAVLINK_BRIDGE_DEFAULTS,
    MAVLINK_BRIDGE_OVERRIDES,
    MISSION_SPECS,
    WAYPOINT_DIRECTORY,
)
from teknofest.servers import data_server
from utils import waypoint_server


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Start the selected TEKNOFEST mission.")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--competition",
        dest="task",
        action="store_const",
        const="competition",
        help="Run Missions 1, 2 and 3 continuously using GN transitions.",
    )
    for task_number in range(1, 4):
        task_group.add_argument(
            f"--task-{task_number}",
            f"--task{task_number}",
            dest="task",
            action="store_const",
            const=f"task{task_number}",
            help=(
                f"Start TEKNOFEST Mission {task_number} with its dedicated "
                f"teknofest_task{task_number}.waypoints route where applicable."
            ),
        )
    parser.epilog = (
        "Bir gorev secenegi verilmezse arayuz modu acilir: "
        "1=task1->task2->task3, "
        "2=yalniz task1, 3=yalniz task2, 4=yalniz task3."
    )
    return parser.parse_args(argv)


def build_mission_launch_command(
        ros2_setup,
        python_path_setup,
        python_executable,
        mission_filename,
):
    """Build a package-module launch command for a TEKNOFEST mission."""
    mission_module = (
        f"teknofest.missions.{os.path.splitext(mission_filename)[0]}"
    )
    return (
        f"{ros2_setup} && {python_path_setup} && "
        f"{shlex.quote(python_executable)} -m {shlex.quote(mission_module)}"
    )


def launch_child_process(command):
    return subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        start_new_session=True,
    )


def signal_child_process(process, sig):
    if process is None or process.poll() is not None:
        return

    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except AttributeError:
        process.send_signal(sig)


def stop_child_process(name, process, timeout_sec=5.0, sig=signal.SIGINT):
    if process is None or process.poll() is not None:
        return

    print(f"[SYSTEM] Stopping {name}...")
    signal_child_process(process, sig)

    try:
        process.wait(timeout=timeout_sec)
        return
    except subprocess.TimeoutExpired:
        print(f"[SYSTEM] {name} did not stop in time, sending SIGTERM...")

    signal_child_process(process, signal.SIGTERM)

    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        print(f"[SYSTEM] {name} did not stop after SIGTERM, sending SIGKILL...")

    signal_child_process(process, getattr(signal, "SIGKILL", signal.SIGTERM))
    process.wait(timeout=2)


def run_startup_cleanup():
    cleanup_script = os.path.join(PROJECT_ROOT, "scripts", "cleanup_shm.sh")
    if not os.path.isfile(cleanup_script):
        raise FileNotFoundError(f"Shared memory cleanup script not found: {cleanup_script}")

    print(f"[SYSTEM] Startup shared memory cleanup running: {cleanup_script}")
    subprocess.run(["/bin/bash", cleanup_script], check=True)


def configure_mavlink_bridge_environment():
    for key, value in MAVLINK_BRIDGE_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.update(MAVLINK_BRIDGE_OVERRIDES)

    print(
        "[SYSTEM] TEKNOFEST mission interface: "
        "1=task1->task2->task3, "
        "2=yalniz task1, 3=yalniz task2, 4=yalniz task3; "
        f"topic={os.environ['MAVLINK_MISSION_START_TOPIC']}, "
        f"waypoint_directory={os.environ['MAVLINK_MISSION_WAYPOINT_DIRECTORY']}, "
        f"waypoints={os.environ['MAVLINK_MISSION_WAYPOINT_FILES']}"
    )


def start_capture_process():
    mp_context = get_context("spawn")
    frame_lock = mp_context.Lock()
    frame_ready_event = mp_context.Event()
    stop_event = mp_context.Event()
    ready_queue = mp_context.Queue(maxsize=1)

    process = mp_context.Process(
        target=capture_proc.run_capture,
        kwargs={
            "lock": frame_lock,
            "frame_ready_event": frame_ready_event,
            "stop_event": stop_event,
            "ready_queue": ready_queue,
        },
        daemon=False,
    )

    print("[SYSTEM] ZED capture process is starting with spawn context...")
    process.start()

    try:
        ready_msg = ready_queue.get(timeout=20)
    except queue.Empty as exc:
        stop_event.set()
        process.terminate()
        process.join(timeout=2)
        raise RuntimeError("ZED capture process did not become ready in time.") from exc

    if "error" in ready_msg:
        stop_event.set()
        process.join(timeout=2)
        raise RuntimeError(f"ZED capture process failed: {ready_msg['error']}")

    fx = ready_msg["fx"]
    cx = ready_msg["cx"]
    print(f"[SYSTEM] ZED calibration loaded: fx={fx:.2f}, cx={cx:.2f}")

    return process, frame_lock, frame_ready_event, stop_event, fx, cx


if __name__ == "__main__":
    args = parse_args()
    interface_mode = args.task is None
    mission_name = None
    mission_filename = None
    if not interface_mode:
        mission_name, mission_filename = MISSION_SPECS[args.task]

    fx = None
    cx = None
    capture_process = None
    capture_stop_event = None
    frame_lock = None
    frame_ready_event = None
    p_bridge = None
    p_vision = None
    p_teknofest_mission = None
    p_mission_manager = None

    try:
        run_startup_cleanup()
        threading.Thread(
            target=waypoint_server.start,
            args=(8000, WAYPOINT_DIRECTORY),
            daemon=True,
        ).start()
        print("[SYSTEM] Waypoint upload -> http://0.0.0.0:8000/api/mission/upload_txt")

        (
            capture_process,
            frame_lock,
            frame_ready_event,
            capture_stop_event,
            fx,
            cx,
        ) = start_capture_process()

        # Flask
        threading.Thread(target=data_server.start, args=(5001,), daemon=True).start()

        print("[SYSTEM] ZED capture was launched with success.")
        print("[SYSTEM] Data stream   -> http://0.0.0.0:5001/data/stream")

        print("\n[SYSTEM] Vision and bridge node launch in ROS2...")
        time.sleep(1)
        configure_mavlink_bridge_environment()

        if os.path.isfile("/opt/ros/kilted/setup.bash"):
            ros2_setup = "source /opt/ros/kilted/setup.bash"
        else:
            ros2_setup = "source /opt/ros/foxy/setup.bash"

        python_path_setup = (
            f"export PYTHONPATH={shlex.quote(COMPETITION_ROOT)}:"
            "${PYTHONPATH:-}"
        )

        vision_path = os.path.join(PROJECT_ROOT, "vision", "vision_node.py")
        bridge_path = os.path.join(COMPETITION_ROOT, "bridge", "bridge_node.py")
        mission_manager_path = os.path.join(PROJECT_ROOT, "mission_manager.py")

        vision_args_setup = f"--fx {shlex.quote(str(fx))} --cx {shlex.quote(str(cx))}"

        cmd_vision = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(vision_path)} {vision_args_setup}"
        )
        cmd_bridge = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(bridge_path)}"
        )
        cmd_mission_manager = (
            f"{ros2_setup} && {python_path_setup} && "
            f"{shlex.quote(sys.executable)} {shlex.quote(mission_manager_path)}"
        )

        cmd_teknofest_mission = None
        if not interface_mode:
            cmd_teknofest_mission = build_mission_launch_command(
                ros2_setup,
                python_path_setup,
                sys.executable,
                mission_filename,
            )

        p_bridge = launch_child_process(cmd_bridge)
        print(f" -> Bridge Node launched (PID: {p_bridge.pid})")

        p_vision = launch_child_process(cmd_vision)
        print(f" -> Vision Node launched (PID: {p_vision.pid})")

        time.sleep(2)

        if interface_mode:
            p_mission_manager = launch_child_process(cmd_mission_manager)
            print(f" -> TEKNOFEST Mission Manager launched (PID: {p_mission_manager.pid})")
            print(
                " -> Mission Planner secimi bekleniyor: "
                "1=task1->task2->task3; 2/3/4=tek gorev\n"
            )
        else:
            p_teknofest_mission = launch_child_process(cmd_teknofest_mission)
            print(
                f" -> TEKNOFEST {mission_name} Node launched "
                f"(PID: {p_teknofest_mission.pid})\n"
            )

        print("[SYSTEM] System active. Ctrl+C at the terminal to close.")

        data_writer.run(frame_lock, frame_ready_event, capture_stop_event)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by the user (Ctrl+C)...")
    except Exception as exc:
        print(f"[SYSTEM] Hata olustu: {exc}")
        raise
    finally:
        print("[SYSTEM] Cleaning process was started...")

        # Mission once kapanir; SIGINT handler'i bridge hâlâ ayaktayken araci
        # durdurup DISARM eder. Ardindan vision, en son bridge kapatilir.
        mission_process_name = (
            "TEKNOFEST Mission Manager"
            if interface_mode
            else f"TEKNOFEST {mission_name} Node"
        )
        mission_process = p_mission_manager if interface_mode else p_teknofest_mission
        subprocesses = (
            (mission_process_name, mission_process, 7.0),
            ("Vision Node", p_vision, 3.0),
            ("Bridge Node", p_bridge, 5.0),
        )
        for process_name, process, timeout_sec in subprocesses:
            try:
                stop_child_process(process_name, process, timeout_sec=timeout_sec)
            except Exception as exc:
                print(f"[SYSTEM] Error while stopping {process_name}: {exc}")

        print("[SYSTEM] Sub-processes closed.")

        if capture_stop_event is not None:
            capture_stop_event.set()

        if capture_process is not None:
            capture_process.join(timeout=3)
            if capture_process.is_alive():
                capture_process.terminate()
                capture_process.join(timeout=2)
            print("[SYSTEM] ZED capture process closed.")

        print("[SYSTEM] The entire system was safely stopped.")
