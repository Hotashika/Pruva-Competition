#!/usr/bin/env python3

import os
import signal
import subprocess
import sys
import threading
import time

from pymavlink import mavutil


MISSION_PARAM_NAME = "SCR_USER1"
MISSION_IDLE = 0
MISSION_1 = 1
MISSION_2 = 2
MISSION_3 = 3
MISSION_4 = 4
MISSION_STOP = 90
MISSION_EMERGENCY = 99

VALID_MISSION_COMMANDS = {
    MISSION_IDLE,
    MISSION_1,
    MISSION_2,
    MISSION_3,
    MISSION_4,
    MISSION_STOP,
    MISSION_EMERGENCY,
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MISSION_PATHS = {
    MISSION_1: os.path.join(PROJECT_ROOT, "missions", "task1_maneuvering_and_path_finding.py"),
    MISSION_2: os.path.join(PROJECT_ROOT, "missions", "task2_collision_avoidance.py"),
    MISSION_3: os.path.join(PROJECT_ROOT, "missions", "task3_docking.py"),
    MISSION_4: os.path.join(PROJECT_ROOT, "missions", "task4_surprise.py"),
}


class JetsonMissionListener:
    """
    Standalone SCR_USER1 mission listener.

    Run this only when another process is not already reading the same MAVLink
    serial stream. If bridge_node.py is active, use its integrated listener.
    """

    def __init__(self):
        self.connection_string = os.getenv("MAVLINK_CONNECTION_STRING", "/dev/ttyACM0")
        self.baud_rate = int(os.getenv("MAVLINK_BAUD", "921600"))
        self.heartbeat_timeout = float(os.getenv("MAVLINK_HEARTBEAT_TIMEOUT", "15"))
        self.request_interval = float(os.getenv("MISSION_PARAM_POLL_SEC", "0.5"))

        self.master = None
        self.running = False
        self.listener_thread = None
        self.send_lock = threading.Lock()

        self.initialized = False
        self.last_command = MISSION_IDLE
        self.current_mode = "UNKNOWN"
        self.pixhawk_link_ok = False

        self.active_mission_name = None
        self.active_mission_process = None
        self.mission_paths = self._load_mission_paths()

    def _load_mission_paths(self):
        paths = {}
        for mission_number, default_path in DEFAULT_MISSION_PATHS.items():
            path = os.getenv(f"MAVLINK_MISSION_{mission_number}_PATH", default_path)
            if path:
                paths[mission_number] = path
        return paths

    def connect(self):
        print(
            f"[MISSION] MAVLink connecting: {self.connection_string}, "
            f"baud={self.baud_rate}"
        )
        self.master = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baud_rate,
            source_system=int(os.getenv("MAVLINK_SOURCE_SYSTEM", "1")),
            source_component=int(os.getenv("MAVLINK_SOURCE_COMPONENT", "191")),
        )

        heartbeat = self.master.wait_heartbeat(timeout=self.heartbeat_timeout)
        if heartbeat is None:
            raise ConnectionError("Pixhawk HEARTBEAT could not be received.")

        self.pixhawk_link_ok = True
        self.current_mode = mavutil.mode_string_v10(heartbeat)
        print(
            "[MISSION] Pixhawk connected. "
            f"system={self.master.target_system}, "
            f"component={self.master.target_component}, mode={self.current_mode}"
        )

    def start(self):
        if self.master is None:
            raise RuntimeError("connect() must be called before start().")

        if self.listener_thread and self.listener_thread.is_alive():
            print("[MISSION] Listener is already running.")
            return

        self.running = True
        self.listener_thread = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name="scr-user1-mission-listener",
        )
        self.listener_thread.start()
        print(f"[MISSION] {MISSION_PARAM_NAME} listener started.")

    def stop(self):
        self.running = False
        self.stop_current_mission()

        if self.listener_thread:
            self.listener_thread.join(timeout=2)

        if self.master is not None:
            try:
                self.master.close()
            except Exception:
                pass

        print("[MISSION] Listener stopped.")

    def _listener_loop(self):
        last_request_time = 0.0

        while self.running:
            try:
                now = time.monotonic()
                if now - last_request_time >= self.request_interval:
                    self._request_mission_parameter()
                    last_request_time = now

                message = self.master.recv_match(blocking=True, timeout=0.1)
                if message is None or message.get_type() == "BAD_DATA":
                    continue

                message_type = message.get_type()
                if message_type == "HEARTBEAT":
                    self._handle_heartbeat(message)
                elif message_type == "PARAM_VALUE":
                    self._handle_parameter_value(message)

            except Exception as exc:
                self.pixhawk_link_ok = False
                print(f"[MISSION] MAVLink listener error: {exc}")
                time.sleep(1)

    def _handle_heartbeat(self, message):
        self.pixhawk_link_ok = True
        self.current_mode = mavutil.mode_string_v10(message)

    def _request_mission_parameter(self):
        with self.send_lock:
            self.master.mav.param_request_read_send(
                self.master.target_system,
                self.master.target_component,
                MISSION_PARAM_NAME.encode("ascii"),
                -1,
            )

    def _handle_parameter_value(self, message):
        param_name = getattr(message, "param_id", "")
        if isinstance(param_name, bytes):
            param_name = param_name.decode("utf-8", errors="ignore")
        param_name = str(param_name).rstrip("\x00")
        if param_name != MISSION_PARAM_NAME:
            return

        command = int(round(float(getattr(message, "param_value", MISSION_IDLE))))

        if not self.initialized:
            self.initialized = True
            self.last_command = command
            if command != MISSION_IDLE:
                print(f"[MISSION] Stale startup command found: {command}. Resetting.")
                self._acknowledge_command()
            return

        if command == self.last_command:
            return

        print(f"[MISSION] {MISSION_PARAM_NAME} changed: {self.last_command} -> {command}")
        self.last_command = command

        if command == MISSION_IDLE:
            return

        self._process_command(command)

    def _process_command(self, command):
        if command == MISSION_STOP:
            print("[MISSION] Stop command received.")
            self.stop_current_mission()
            self._acknowledge_command()
            return

        if command == MISSION_EMERGENCY:
            print("[MISSION] Emergency mission cancel received.")
            self.emergency_stop()
            self._acknowledge_command()
            return

        if command not in VALID_MISSION_COMMANDS or command == MISSION_IDLE:
            print(f"[MISSION] Unknown command: {command}")
            self._acknowledge_command()
            return

        if not self._mission_start_conditions_ok():
            print("[MISSION] Mission start conditions are not OK.")
            self._acknowledge_command()
            return

        if self._start_mission_process(command):
            print(f"[MISSION] M{command} started.")

        self._acknowledge_command()

    def _mission_start_conditions_ok(self):
        if not self.pixhawk_link_ok:
            print("[MISSION] Pixhawk link is not healthy.")
            return False

        if self.current_mode != "GUIDED":
            print(f"[MISSION] Vehicle is not GUIDED. Current mode: {self.current_mode}")
            return False

        return True

    def _start_mission_process(self, mission_number):
        mission_name = f"M{mission_number}"
        mission_path = self.mission_paths.get(mission_number)
        if not mission_path:
            print(f"[MISSION] No path configured for {mission_name}.")
            return False

        if not os.path.isfile(mission_path):
            print(f"[MISSION] Script not found for {mission_name}: {mission_path}")
            return False

        if self.active_mission_process and self.active_mission_process.poll() is None:
            print(f"[MISSION] Another mission is active: {self.active_mission_name}")
            return False

        try:
            self.active_mission_process = subprocess.Popen(
                [sys.executable, mission_path],
                start_new_session=True,
            )
            self.active_mission_name = mission_name
            return True
        except Exception as exc:
            self.active_mission_process = None
            self.active_mission_name = None
            print(f"[MISSION] {mission_name} could not be started: {exc}")
            return False

    def stop_current_mission(self):
        process = self.active_mission_process
        if process is None:
            return

        if process.poll() is not None:
            self.active_mission_process = None
            self.active_mission_name = None
            return

        print(f"[MISSION] Stopping {self.active_mission_name}...")
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except AttributeError:
            process.send_signal(signal.SIGINT)

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=2)

        self.active_mission_process = None
        self.active_mission_name = None

    def emergency_stop(self):
        self.stop_current_mission()

    def _acknowledge_command(self):
        self._set_mission_parameter(MISSION_IDLE)
        self.last_command = MISSION_IDLE

    def _set_mission_parameter(self, value):
        with self.send_lock:
            self.master.mav.param_set_send(
                self.master.target_system,
                self.master.target_component,
                MISSION_PARAM_NAME.encode("ascii"),
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
        print(f"[MISSION] {MISSION_PARAM_NAME}={value} sent.")


def main():
    listener = JetsonMissionListener()
    listener.connect()
    listener.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[MISSION] Interrupted.")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
