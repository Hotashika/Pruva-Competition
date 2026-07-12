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
if COMPETITION_ROOT not in sys.path:
    sys.path.insert(0, COMPETITION_ROOT)

from teknofest.core import capture_proc
from teknofest.core import data_writer
from teknofest.servers import data_server


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
    fx = None
    cx = None
    capture_process = None
    capture_stop_event = None
    frame_lock = None
    frame_ready_event = None
    p_bridge = None
    p_vision = None

    try:
        run_startup_cleanup()

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

        if os.path.isfile("/opt/ros/kilted/setup.bash"):
            ros2_setup = "source /opt/ros/kilted/setup.bash"
        else:
            ros2_setup = "source /opt/ros/foxy/setup.bash"

        python_path_setup = (
            f"export PYTHONPATH={shlex.quote(PROJECT_ROOT)}:"
            f"{shlex.quote(COMPETITION_ROOT)}:${{PYTHONPATH:-}}"
        )

        vision_path = os.path.join(PROJECT_ROOT, "vision", "vision_node.py")
        bridge_path = os.path.join(COMPETITION_ROOT, "bridge", "bridge_node.py")

        vision_args_setup = f"--fx {shlex.quote(str(fx))} --cx {shlex.quote(str(cx))}"

        ################################################################################################################
        # SETUP TEKNOFEST MISSION PATHS
        ################################################################################################################
        teknofest_task1_path = os.path.join(PROJECT_ROOT, "missions", "task1_point_tracking.py")
        teknofest_task2_path = os.path.join(PROJECT_ROOT, "missions", "task2_point_tracking_task_in_an_environment_with_obstacle.py")
        teknofest_task3_path = os.path.join(PROJECT_ROOT, "missions", "task3_kamikaze_engagement.py")
        ################################################################################################################
        mission_env_setup = (
            "export MAVLINK_MISSION_LAUNCH_ENABLED=1 && "
            "export MAVLINK_MISSION_SEQUENCE=1,2,3 && "
            f"export MAVLINK_MISSION_1_PATH={shlex.quote(teknofest_task1_path)} && "
            f"export MAVLINK_MISSION_2_PATH={shlex.quote(teknofest_task2_path)} && "
            f"export MAVLINK_MISSION_3_PATH={shlex.quote(teknofest_task3_path)}"
        )

        cmd_vision = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(vision_path)} {vision_args_setup}"
        )
        cmd_bridge = (
            f"{ros2_setup} && {python_path_setup} && {mission_env_setup} && {shlex.quote(sys.executable)} {shlex.quote(bridge_path)}"
        )

        p_bridge = launch_child_process(cmd_bridge)
        print(f" -> Bridge Node launched (PID: {p_bridge.pid})")

        p_vision = launch_child_process(cmd_vision)
        print(f" -> Vision Node launched (PID: {p_vision.pid})")

        time.sleep(2)

        ################################################################################################################
        #   TEKNOFEST MISSION START
        #   Mission nodes are started by bridge when Pixhawk sends MAV_CMD_USER_1.
        #   TEKNOFEST uses its existing waypoint files and continues M1 -> M2 -> M3 automatically.
        ################################################################################################################
        print(" -> TEKNOFEST missions are waiting for MAVLink start command (M1 -> M2 -> M3).\n")
        ################################################################################################################

        print("[SYSTEM] System active. Ctrl+C at the terminal to close.")

        data_writer.run(frame_lock, frame_ready_event, capture_stop_event)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by the user (Ctrl+C)...")
    except Exception as exc:
        print(f"[SYSTEM] Hata olustu: {exc}")
        raise
    finally:
        print("[SYSTEM] Cleaning process was started...")

        try:
            stop_child_process("Vision Node", p_vision, timeout_sec=3.0)
            stop_child_process("Bridge Node", p_bridge, timeout_sec=5.0)
        except Exception as exc:
            print(f"[SYSTEM] Error while sub-process shut down: {exc}")

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
