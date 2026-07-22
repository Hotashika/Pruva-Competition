#!/usr/bin/env python3

"""Arayüzden gelen TEKNOFEST görev komutlarını process olarak yönetir."""

import os
import signal
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ROOT = os.path.dirname(PROJECT_ROOT)
if COMPETITION_ROOT not in sys.path:
    sys.path.insert(0, COMPETITION_ROOT)

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int32, String

from teknofest.config.mission_config import MAVLINK_BRIDGE_DEFAULTS, MISSION_COMMANDS
from utils.mavlink_utilities import parse_bridge_state
from utils.task_selection_state import (
    clear_task_selection,
    read_task_selection,
    write_task_selection,
)


MISSION_PATHS = {
    command: os.path.join(PROJECT_ROOT, "missions", mission_filename)
    for command, (_, _, mission_filename) in MISSION_COMMANDS.items()
}
MISSION_MODULES = {
    command: f"teknofest.missions.{os.path.splitext(mission_filename)[0]}"
    for command, (_, _, mission_filename) in MISSION_COMMANDS.items()
}
MISSION_NAMES = {
    command: display_name
    for command, (_, display_name, _) in MISSION_COMMANDS.items()
}
MISSION_TASK_KEYS = {
    command: task_key
    for command, (task_key, _, _) in MISSION_COMMANDS.items()
}


class MissionManager(Node):
    """``/mission_start`` komutuyla seçilen TEKNOFEST görevini çalıştırır."""

    def __init__(self):
        super().__init__("teknofest_mission_manager")

        self.declare_parameter(
            "mission_start_topic",
            os.getenv(
                "MAVLINK_MISSION_START_TOPIC",
                MAVLINK_BRIDGE_DEFAULTS["MAVLINK_MISSION_START_TOPIC"],
            ),
        )
        self.declare_parameter(
            "mission_start_ack_topic",
            os.getenv(
                "MAVLINK_MISSION_START_ACK_TOPIC",
                MAVLINK_BRIDGE_DEFAULTS["MAVLINK_MISSION_START_ACK_TOPIC"],
            ),
        )
        self.declare_parameter(
            "task_selection_file",
            os.getenv(
                "MISSION_SELECTION_FILE",
                MAVLINK_BRIDGE_DEFAULTS["MISSION_SELECTION_FILE"],
            ),
        )

        self.mission_start_topic = str(self.get_parameter("mission_start_topic").value)
        self.mission_start_ack_topic = str(
            self.get_parameter("mission_start_ack_topic").value
        )
        self.task_selection_file = str(self.get_parameter("task_selection_file").value)

        clear_task_selection(self.task_selection_file)
        self.bridge_state = {}
        self.last_reported_state = None
        self.active_command = None
        self.active_task_key = None
        self.active_mission_process = None

        self.status_pub = self.create_publisher(String, "/mission_manager/status", 10)
        self.mission_start_ack_pub = self.create_publisher(
            Int32, self.mission_start_ack_topic, 10
        )
        self.create_subscription(
            Int32, self.mission_start_topic, self._mission_start_callback, 10
        )
        self.create_subscription(String, "/cube/state", self._bridge_state_callback, 10)
        self.create_timer(1.0, self._status_loop)

        choices = ", ".join(
            f"{command}={MISSION_NAMES[command]}"
            for command in MISSION_PATHS
        )
        self._publish_status(
            f"TEKNOFEST Mission Manager aktif. {self.mission_start_topic} "
            f"dinleniyor; secenekler: {choices}"
        )

    def _bridge_state_callback(self, msg):
        self.bridge_state = parse_bridge_state(msg.data)

    def _mission_start_callback(self, msg):
        try:
            command = int(msg.data)
        except (TypeError, ValueError):
            self._publish_status(f"Gecersiz mission_start verisi: {msg.data}")
            return

        started_new_mission = False
        if command in MISSION_PATHS:
            if self._mission_is_running(command):
                self._publish_status(
                    f"{MISSION_NAMES[command]} zaten aktif; process yeniden "
                    "baslatilmadan ACK yenilenecek."
                )
            else:
                if not self._start_mission(command):
                    return
                started_new_mission = True
        elif command in (90, 99):
            self._stop_active_mission()

        try:
            state = write_task_selection(self.task_selection_file, command)
        except Exception as exc:
            self._publish_status(f"Gorev secim JSON dosyasi yazilamadi: {exc}")
            if started_new_mission:
                self._stop_active_mission()
            return

        ack = Int32()
        ack.data = command
        self.mission_start_ack_pub.publish(ack)
        self._publish_status(
            f"command={command}, selected_task={state.get('selected_task')}, "
            f"status={state.get('status')}; ack yayinlandi"
        )

    def _mission_is_running(self, command):
        process = self.active_mission_process
        return (
            self.active_command == command
            and process is not None
            and process.poll() is None
        )

    def _start_mission(self, command):
        mission_path = MISSION_PATHS[command]
        display_name = MISSION_NAMES[command]
        task_key = MISSION_TASK_KEYS[command]
        if not os.path.isfile(mission_path):
            self._publish_status(f"Mission script bulunamadi: {mission_path}")
            return False

        self._stop_active_mission()
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", MISSION_MODULES[command]],
                cwd=COMPETITION_ROOT,
                start_new_session=True,
            )
        except Exception as exc:
            self._publish_status(f"{display_name} baslatilamadi: {exc}")
            return False

        self.active_command = command
        self.active_task_key = task_key
        self.active_mission_process = process
        self._publish_status(
            f"{display_name} baslatildi: pid={process.pid}, script={mission_path}"
        )
        return True

    def _stop_active_mission(self):
        process = self.active_mission_process
        task_key = self.active_task_key
        self.active_mission_process = None
        self.active_command = None
        self.active_task_key = None
        if process is None or process.poll() is not None:
            return

        self._publish_status(f"{task_key} durduruluyor: pid={process.pid}")
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        except AttributeError:
            process.send_signal(signal.SIGINT)

        try:
            process.wait(timeout=7)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self._publish_status(f"{task_key} durduruldu")

    def _status_loop(self):
        self._reap_finished_mission()
        selection = read_task_selection(self.task_selection_file)
        current_state = {
            "selected_task": selection.get("selected_task"),
            "task_status": selection.get("status"),
            "active_task": self.active_task_key,
            "connected": self.bridge_state.get("connected"),
            "armed": self.bridge_state.get("armed"),
            "mode": self.bridge_state.get("mode"),
        }
        if current_state == self.last_reported_state:
            return

        self.last_reported_state = current_state
        self._publish_status(
            "System status: "
            f"selected_task={current_state['selected_task']}, "
            f"active_task={current_state['active_task']}, "
            f"task_status={current_state['task_status']}, "
            f"connected={current_state['connected']}, "
            f"armed={current_state['armed']}, mode={current_state['mode']}"
        )

    def _reap_finished_mission(self):
        process = self.active_mission_process
        if process is None:
            return False

        return_code = process.poll()
        if return_code is None:
            return False

        task_key = self.active_task_key
        self.active_mission_process = None
        self.active_command = None
        self.active_task_key = None
        try:
            clear_task_selection(self.task_selection_file)
        except Exception as exc:
            self._publish_status(
                f"Tamamlanan gorev sonrasi durum dosyasi sifirlanamadi: {exc}"
            )
        self._publish_status(
            f"{task_key} process sonlandi: pid={process.pid}, "
            f"return_code={return_code}; aktif gorev temizlendi"
        )
        return True

    def _publish_status(self, text):
        self.get_logger().info(text)
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def destroy_node(self):
        self._stop_active_mission()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("TEKNOFEST Mission Manager kapatiliyor...")
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
