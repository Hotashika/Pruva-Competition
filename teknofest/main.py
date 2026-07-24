import os
import queue
import shlex
import signal
import subprocess

import sys
import threading
import time
from multiprocessing import get_context
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ROOT = os.path.dirname(PROJECT_ROOT)
ZED_STARTUP_TIMEOUT_SEC = float(os.getenv("ZED_STARTUP_TIMEOUT_SEC", "60.0"))
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


def resolve_ros2_setup():
    """Kurulu ROS 2 dağıtımını ortamdan veya /opt/ros altından bulur."""
    requested = os.getenv("ROS_DISTRO", "").strip()
    candidates = [requested] if requested else []
    candidates.extend(["jazzy", "humble", "kilted", "foxy"])
    seen = set()
    for distro in candidates:
        if not distro or distro in seen:
            continue
        seen.add(distro)
        setup_path = Path("/opt/ros") / distro / "setup.bash"
        if setup_path.is_file():
            return f"source {shlex.quote(str(setup_path))}", distro
    raise FileNotFoundError(
        "ROS 2 setup.bash bulunamadı. ROS_DISTRO ayarını ve /opt/ros kurulumunu kontrol edin."
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


def monitor_child_processes(
        processes,
        monitor_stop_event,
        system_stop_event,
        poll_interval_sec=0.25,
):
    """Kritik bir ROS alt prosesi kapanırsa ana veri döngüsünü durdurur."""
    while monitor_stop_event is not None and not monitor_stop_event.is_set():
        for name, process in processes:
            if process is None:
                continue
            return_code = process.poll()
            if return_code is not None:
                print(
                    f"[SYSTEM] CRITICAL: {name} beklenmedik şekilde kapandı "
                    f"(exit={return_code}). Sistem güvenli kapatılıyor."
                )
                if system_stop_event is not None:
                    system_stop_event.set()
                monitor_stop_event.set()
                return
        time.sleep(poll_interval_sec)


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
        ready_msg = ready_queue.get(timeout=ZED_STARTUP_TIMEOUT_SEC)
    except queue.Empty as exc:
        stop_event.set()
        process.terminate()
        process.join(timeout=2)
        raise RuntimeError(
            "ZED capture process did not become ready in time "
            f"({ZED_STARTUP_TIMEOUT_SEC:.0f}s)."
        ) from exc

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
    p_mission_manager = None
    child_monitor_stop_event = None

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

        ros2_setup, ros_distro = resolve_ros2_setup()
        print(f"[SYSTEM] ROS 2 distribution: {ros_distro}")

        python_path_setup = (
            # Depo kokunu once ekle. Aksi halde Jetson'da PROJECT_ROOT veya
            # eski kurulumlardan gelen baska bir ``utils`` paketi, ortak
            # /utils paketini golgeleyebilir ve alt prosesler baslatilamaz.
            f"export PYTHONPATH={shlex.quote(COMPETITION_ROOT)}:"
            f"{shlex.quote(PROJECT_ROOT)}:${{PYTHONPATH:-}}"
        )

        vision_path = os.path.join(PROJECT_ROOT, "vision", "vision_node.py")
        bridge_path = os.path.join(COMPETITION_ROOT, "bridge", "bridge_node.py")
        mission_manager_path = os.path.join(
            PROJECT_ROOT,
            "missions",
            "mission_manager.py",
        )

        vision_args_setup = f"--fx {shlex.quote(str(fx))} --cx {shlex.quote(str(cx))}"
        bridge_teknofest_setup = (
            "export MAVLINK_MISSION_PARAM_NAME=SCR_USER2 && "
            "export MAVLINK_WAYPOINT_PREFIX=teknofest"
        )

        cmd_vision = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(vision_path)} {vision_args_setup}"
        )
        cmd_bridge = (
            f"{ros2_setup} && {python_path_setup} && {bridge_teknofest_setup} && "
            f"{shlex.quote(sys.executable)} {shlex.quote(bridge_path)}"
        )
        cmd_mission_manager = (
            f"{ros2_setup} && {python_path_setup} && "
            f"{shlex.quote(sys.executable)} {shlex.quote(mission_manager_path)}"
        )

        p_bridge = launch_child_process(cmd_bridge)
        print(f" -> Bridge Node launched (PID: {p_bridge.pid})")

        p_vision = launch_child_process(cmd_vision)
        print(f" -> Vision Node launched (PID: {p_vision.pid})")

        time.sleep(2)

        # Tek manager SCR_USER2=1/2/3 komutuna gore yalniz secilen
        # algoritmanin araca hareket komutu gondermesine izin verir.
        p_mission_manager = launch_child_process(cmd_mission_manager)
        print(
            f" -> TEKNOFEST Mission Manager launched "
            f"(PID: {p_mission_manager.pid})\n"
        )
        child_monitor_stop_event = threading.Event()
        threading.Thread(
            target=monitor_child_processes,
            args=(
                (
                    ("Bridge Node", p_bridge),
                    ("Vision Node", p_vision),
                    ("TEKNOFEST Mission Manager", p_mission_manager),
                ),
                child_monitor_stop_event,
                capture_stop_event,
            ),
            daemon=True,
        ).start()
        print("[SYSTEM] System active. Ctrl+C at the terminal to close.")

        data_writer.run(
            frame_lock,
            frame_ready_event,
            capture_stop_event,
            fx=fx,
            cx=cx,
        )

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by the user (Ctrl+C)...")
    except Exception as exc:
        print(f"[SYSTEM] Hata olustu: {exc}")
        raise
    finally:
        print("[SYSTEM] Cleaning process was started...")

        # Kontrollü Ctrl+C/kapanış sırasında mission prosesinin exit=0 ile
        # bitmesi çökme değildir. Monitörü önce durdur; gerçek çalışma
        # sırasında beklenmedik kapanmaları izlemeye devam eder.
        if child_monitor_stop_event is not None:
            child_monitor_stop_event.set()

        # Mission once kapanir; SIGINT handler'i bridge hâlâ ayaktayken araci
        # durdurup DISARM eder. Ardindan vision, en son bridge kapatilir.
        subprocesses = (
            ("TEKNOFEST Mission Manager", p_mission_manager, 7.0),
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
