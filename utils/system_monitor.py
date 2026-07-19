#!/usr/bin/env python3
"""Yarisma profilleri icin ortak, salt-okunur curses sistem monitoru."""

import curses
import math
import sys
import time
from collections import deque
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, NavSatFix
from std_msgs.msg import Float32, String

from utils.mavlink_utilities import parse_bridge_state


STALE_AFTER_SEC = 3.0
REFRESH_INTERVAL_SEC = 0.5


class SystemMonitorNode(Node):
    def __init__(self, node_name, subscribe_mission_status=True):
        super().__init__(node_name)
        self.started_at = time.monotonic()
        self.bridge_state = {}
        self.bridge_state_time = None
        self.gps = None
        self.gps_time = None
        self.heading = None
        self.heading_time = None
        self.battery = None
        self.battery_time = None
        self.active_task = None
        self.task_time = None
        self.mission_status = (
            "Waiting for Mission Manager"
            if subscribe_mission_status
            else "Mission Manager not used by this profile"
        )
        self.last_error = "None"
        self.events = deque(maxlen=5)

        self.create_subscription(String, "/cube/state", self._state_callback, 10)
        self.create_subscription(NavSatFix, "/cube/gps", self._gps_callback, 10)
        self.create_subscription(Float32, "/cube/gps/heading", self._heading_callback, 10)
        self.create_subscription(BatteryState, "/cube/battery", self._battery_callback, 10)
        self.create_subscription(String, "/mission/active_task", self._task_callback, 10)
        if subscribe_mission_status:
            self.create_subscription(
                String, "/mission_manager/status", self._status_callback, 10
            )
        self.create_subscription(String, "/cube/error", self._error_callback, 10)
        self.create_subscription(String, "/cube/diagnostics", self._diagnostic_callback, 10)

    def _event(self, text):
        self.events.appendleft((time.strftime("%H:%M:%S"), str(text)))

    def _state_callback(self, message):
        new_state = parse_bridge_state(message.data)
        if new_state != self.bridge_state:
            self._event(f"Vehicle state: {message.data}")
        self.bridge_state = new_state
        self.bridge_state_time = time.monotonic()

    def _gps_callback(self, message):
        self.gps = (float(message.latitude), float(message.longitude))
        self.gps_time = time.monotonic()

    def _heading_callback(self, message):
        self.heading = float(message.data)
        self.heading_time = time.monotonic()

    def _battery_callback(self, message):
        self.battery = (float(message.voltage), float(message.percentage))
        self.battery_time = time.monotonic()

    def _task_callback(self, message):
        task = message.data.strip() or None
        if task != self.active_task:
            self._event(f"Active task: {task or 'None'}")
        self.active_task = task
        self.task_time = time.monotonic()

    def _status_callback(self, message):
        self.mission_status = message.data.strip()
        self._event(self.mission_status)

    def _error_callback(self, message):
        self.last_error = message.data.strip() or "Unknown error"
        self._event(f"ERROR: {self.last_error}")

    def _diagnostic_callback(self, message):
        text = message.data.strip()
        if "mission dosyaya yazildi" in text.lower():
            self._event(f"WAYPOINT SAVED: {text}")

    @staticmethod
    def _fresh(timestamp):
        return timestamp is not None and time.monotonic() - timestamp <= STALE_AFTER_SEC


def _safe_add(window, row, column, text, attributes=0):
    height, width = window.getmaxyx()
    if row < 0 or row >= height or column >= width:
        return
    try:
        window.addnstr(row, column, str(text), max(0, width - column - 1), attributes)
    except curses.error:
        pass


def _status_text(value, fresh, waiting="WAITING"):
    return str(value) if fresh else waiting


def _draw(screen, node, title):
    screen.erase()
    height, width = screen.getmaxyx()
    if height < 20 or width < 72:
        _safe_add(screen, 0, 0, "Terminal en az 72x20 olmali. Pencereyi buyutun.", curses.A_BOLD)
        screen.refresh()
        return

    display_title = f" {title} - SYSTEM MONITOR "
    _safe_add(screen, 0, 0, display_title, curses.A_BOLD | curses.color_pair(1))
    _safe_add(screen, 0, max(1, width - 10), time.strftime("%H:%M:%S"), curses.A_BOLD)
    separator = "─" * (width - 1)
    _safe_add(screen, 1, 0, separator)

    state_fresh = node._fresh(node.bridge_state_time)
    connected = state_fresh and node.bridge_state.get("connected") is True
    armed = state_fresh and node.bridge_state.get("armed") is True
    mode = node.bridge_state.get("mode", "WAITING") if state_fresh else "STALE"
    connection_text = "CONNECTED" if connected else ("DISCONNECTED" if state_fresh else "WAITING")
    connection_color = curses.color_pair(2 if connected else 3) | curses.A_BOLD

    _safe_add(screen, 2, 1, "MAVLink")
    _safe_add(screen, 2, 18, connection_text, connection_color)
    _safe_add(screen, 2, 42, "Uptime")
    _safe_add(screen, 2, 58, f"{time.monotonic() - node.started_at:7.0f} s")
    _safe_add(screen, 3, 1, "Vehicle mode")
    _safe_add(screen, 3, 18, mode, curses.A_BOLD)
    _safe_add(screen, 3, 42, "Armed")
    _safe_add(screen, 3, 58, "YES" if armed else "NO", curses.color_pair(3 if armed else 2))

    gps_fresh = node._fresh(node.gps_time)
    gps_text = "WAITING"
    if gps_fresh and node.gps is not None:
        gps_text = f"{node.gps[0]:.7f}, {node.gps[1]:.7f}"
    heading_text = _status_text(
        f"{node.heading:.1f} deg" if node.heading is not None and math.isfinite(node.heading) else "INVALID",
        node._fresh(node.heading_time),
    )
    _safe_add(screen, 5, 1, "GPS")
    _safe_add(screen, 5, 18, gps_text)
    _safe_add(screen, 6, 1, "Heading")
    _safe_add(screen, 6, 18, heading_text)

    battery_text = "WAITING"
    battery_percent = "WAITING"
    if node._fresh(node.battery_time) and node.battery is not None:
        battery_text = f"{node.battery[0]:.2f} V"
        percentage = node.battery[1]
        battery_percent = f"{percentage * 100:.0f}%" if math.isfinite(percentage) and percentage >= 0 else "N/A"
    _safe_add(screen, 5, 42, "Battery")
    _safe_add(screen, 5, 58, battery_text)
    _safe_add(screen, 6, 42, "Remaining")
    _safe_add(screen, 6, 58, battery_percent)

    _safe_add(screen, 8, 1, "Active task")
    _safe_add(screen, 8, 18, node.active_task or "NONE", curses.A_BOLD)
    _safe_add(screen, 9, 1, "Mission status")
    _safe_add(screen, 9, 18, node.mission_status)
    _safe_add(screen, 10, 1, "Last error")
    _safe_add(screen, 10, 18, node.last_error, curses.color_pair(3) if node.last_error != "None" else 0)

    _safe_add(screen, 12, 0, separator)
    _safe_add(screen, 13, 1, "RECENT EVENTS", curses.A_BOLD)
    for index, (event_time, text) in enumerate(node.events):
        _safe_add(screen, 14 + index, 1, f"{event_time}  {text}")

    _safe_add(screen, height - 1, 1, "q: quit   r: redraw", curses.A_DIM)
    screen.refresh()


def _run(screen, node, title):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    curses.start_color()
    try:
        curses.use_default_colors()
        background = -1
    except curses.error:
        background = curses.COLOR_BLACK
    curses.init_pair(1, curses.COLOR_CYAN, background)
    curses.init_pair(2, curses.COLOR_GREEN, background)
    curses.init_pair(3, curses.COLOR_RED, background)
    screen.nodelay(True)
    screen.timeout(50)

    last_draw = 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        key = screen.getch()
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R"), curses.KEY_RESIZE):
            last_draw = 0.0
        now = time.monotonic()
        if now - last_draw >= REFRESH_INTERVAL_SEC:
            _draw(screen, node, title)
            last_draw = now


def run_monitor(title, node_name, subscribe_mission_status=True, args=None):
    rclpy.init(args=args)
    node = SystemMonitorNode(node_name, subscribe_mission_status)
    try:
        curses.wrapper(_run, node, title)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
