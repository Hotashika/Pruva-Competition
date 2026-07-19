"""NJORD Task 4: optimize unordered GPS points and avoid detected buoys."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from std_msgs.msg import String

from utils.mavlink_utilities import (
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    create_mission_topics,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    wait_for_mission_services,
)
from utils.read_waypoints import parse_qgc_waypoints


BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = Path(
    os.getenv(
        "NJORD_TASK4_WAYPOINT_PATH",
        BASE_DIR.parent / "waypoints" / "njord_task4.waypoints",
    )
)
ACTIVE_TASK_NAME = "task4"
HOLD_MODE_NAME = "HOLD"

EARTH_RADIUS_M = 6_378_137.0
MIN_VALID_ABS_COORD = 1e-6
EXACT_ROUTE_LIMIT = 12
WAYPOINT_TOLERANCE_M = float(os.getenv("TASK4_WAYPOINT_TOLERANCE_M", "1.5"))
GEOFENCE_RADIUS_M = float(os.getenv("TASK4_GEOFENCE_RADIUS_M", "250.0"))

GPS_TIMEOUT_SEC = 2.0
HEADING_TIMEOUT_SEC = 2.0
BRIDGE_STATE_TIMEOUT_SEC = 2.0
VISION_DETECTION_TIMEOUT_SEC = 1.0

AVOID_ENTER_DISTANCE_M = 4.5
AVOID_EXIT_DISTANCE_M = 5.5
EMERGENCY_DISTANCE_M = 2.0
AVOID_LINEAR_X = 0.65
AVOID_TURN_Z = 0.65
AVOID_MIN_DURATION_SEC = 0.6
AVOID_CLEAR_DURATION_SEC = 0.5
AVOID_MAX_DURATION_SEC = 6.0

BUOY_CLASSES = {
    "red_buoy",
    "green_buoy",
    "east_cardinal",
    "west_cardinal",
    "buoy",
}
ANGLE_KEYS = (
    "Buoy angle: ",
    "Buoy angle",
    "bearing",
    "angle_deg",
    "angle",
)


class MissionState(Enum):
    INIT = auto()
    NAVIGATING = auto()
    AVOIDING = auto()
    FINISHED = auto()
    FAILSAFE = auto()


@dataclass(frozen=True)
class RouteSolution:
    waypoints: list[dict]
    distance_m: float
    method: str


def gps_distance_m(point_a, point_b):
    """Great-circle distance for two ``(latitude, longitude)`` points."""
    lat1, lon1 = map(math.radians, point_a)
    lat2, lon2 = map(math.radians, point_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    hav = min(1.0, max(0.0, hav))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(hav))


def route_distance_m(start, waypoints):
    """Distance of start -> all waypoints; the route does not return home."""
    previous = (float(start[0]), float(start[1]))
    total = 0.0
    for waypoint in waypoints:
        current = (float(waypoint["lat"]), float(waypoint["lon"]))
        total += gps_distance_m(previous, current)
        previous = current
    return total


def _exact_open_route(start, waypoints):
    """Find the exact shortest open route with Held-Karp dynamic programming."""
    count = len(waypoints)
    if count <= 1:
        return list(waypoints)

    points = [(float(item["lat"]), float(item["lon"])) for item in waypoints]
    start_distances = [gps_distance_m(start, point) for point in points]
    between = [
        [gps_distance_m(points[i], points[j]) for j in range(count)]
        for i in range(count)
    ]

    # (visited mask, last point) -> (distance, previous point)
    dp = {
        (1 << index, index): (start_distances[index], None)
        for index in range(count)
    }
    for mask in range(1, 1 << count):
        for last in range(count):
            entry = dp.get((mask, last))
            if entry is None:
                continue
            for nxt in range(count):
                bit = 1 << nxt
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_cost = entry[0] + between[last][nxt]
                current = dp.get((next_mask, nxt))
                if current is None or next_cost < current[0] - 1e-9:
                    dp[(next_mask, nxt)] = (next_cost, last)

    mask = (1 << count) - 1
    last = min(range(count), key=lambda index: (dp[(mask, index)][0], index))
    order = []
    while last is not None:
        order.append(last)
        previous = dp[(mask, last)][1]
        mask ^= 1 << last
        last = previous
    order.reverse()
    return [waypoints[index] for index in order]


def _nearest_neighbour_route(start, waypoints):
    remaining = list(range(len(waypoints)))
    route = []
    current = (float(start[0]), float(start[1]))
    while remaining:
        best = min(
            remaining,
            key=lambda index: (
                gps_distance_m(
                    current,
                    (float(waypoints[index]["lat"]), float(waypoints[index]["lon"])),
                ),
                index,
            ),
        )
        route.append(waypoints[best])
        current = (float(waypoints[best]["lat"]), float(waypoints[best]["lon"]))
        remaining.remove(best)
    return route


def _two_opt_open_route(start, route, max_passes=50):
    """Improve a large open route while keeping the vehicle start fixed."""
    best = list(route)
    best_distance = route_distance_m(start, best)
    for _ in range(max_passes):
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 1, len(best)):
                candidate = (
                    best[:left]
                    + list(reversed(best[left : right + 1]))
                    + best[right + 1 :]
                )
                candidate_distance = route_distance_m(start, candidate)
                if candidate_distance < best_distance - 1e-6:
                    best = candidate
                    best_distance = candidate_distance
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def optimize_waypoint_order(start, waypoints, exact_limit=EXACT_ROUTE_LIMIT):
    """Return the fastest geometric visit order starting at current GPS."""
    clean = [dict(item) for item in waypoints]
    if len(clean) <= exact_limit:
        ordered = _exact_open_route(start, clean)
        method = "held-karp-exact"
    else:
        seed = _nearest_neighbour_route(start, clean)
        ordered = _two_opt_open_route(start, seed)
        method = "nearest-neighbour+2-opt"
    return RouteSolution(ordered, route_distance_m(start, ordered), method)


def load_task4_waypoints(path=WAYPOINT_PATH):
    """Load QGC route points, excluding the HOME item at sequence zero."""
    return [
        item
        for item in parse_qgc_waypoints(path)
        if int(item.get("seq", -1)) != 0
    ]


class Task4FastRoute:
    def __init__(self, node, mission_topics, mission_clients, waypoints):
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.clients = mission_clients

        self.unordered_waypoints = list(waypoints)
        self.waypoints = []
        self.route_optimized = False
        self.current_target_index = 0

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self.home_lat = None
        self.home_lon = None
        self.last_gps_time = None
        self.last_heading_time = None
        self.last_bridge_state_time = None
        self.bridge_connected = False

        self.state = MissionState.INIT
        self.finished = False
        self.hold_mode_requested = False
        self.hold_mode_future = None

        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.avoid_turn_direction = -1.0

    def update_gps(self, lat, lon, now=None):
        now = time.monotonic() if now is None else float(now)
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.last_gps_time = now
        if self.home_lat is None:
            self.home_lat = self.current_lat
            self.home_lon = self.current_lon
            self._optimize_route()

    def update_heading(self, heading, now=None):
        self.current_heading = float(heading) % 360.0
        self.last_heading_time = time.monotonic() if now is None else float(now)

    def update_bridge_state(self, connected, now=None):
        self.bridge_connected = bool(connected)
        self.last_bridge_state_time = time.monotonic() if now is None else float(now)

    def _optimize_route(self):
        if self.route_optimized or self.current_lat is None:
            return
        if not self.unordered_waypoints:
            self.enter_failsafe("Task 4 waypoint list is empty")
            return

        start = (self.current_lat, self.current_lon)
        solution = optimize_waypoint_order(start, self.unordered_waypoints)
        original_distance = route_distance_m(start, self.unordered_waypoints)
        self.waypoints = solution.waypoints
        self.route_optimized = True
        sequence = [item.get("seq", "?") for item in self.waypoints]
        self.logger.info(
            "Task 4 route optimized: method=%s distance=%.1fm saved=%.1fm order=%s"
            % (
                solution.method,
                solution.distance_m,
                max(0.0, original_distance - solution.distance_m),
                sequence,
            )
        )

    def _request_hold_mode(self):
        if self.hold_mode_requested or self.clients is None:
            return
        self.hold_mode_requested = True
        request = SetMode.Request()
        request.base_mode = 0
        request.custom_mode = HOLD_MODE_NAME
        try:
            self.hold_mode_future = self.clients.set_mode_client.call_async(request)
            self.logger.warn("Task 4 failsafe: HOLD mode requested.")
        except Exception as exc:
            self.logger.error(f"Task 4 HOLD request failed: {exc}")

    def enter_failsafe(self, reason):
        """Stop the mission permanently and request autopilot HOLD mode."""
        if self.state != MissionState.FAILSAFE:
            self.logger.error(reason)
        self.state = MissionState.FAILSAFE
        stop_vehicle(self.topics.cmd_vel_pub)
        self._request_hold_mode()

    def _sensors_ready(self, now):
        checks = (
            (self.last_gps_time, GPS_TIMEOUT_SEC, "GPS data timeout"),
            (self.last_heading_time, HEADING_TIMEOUT_SEC, "heading data timeout"),
            (self.last_bridge_state_time, BRIDGE_STATE_TIMEOUT_SEC, "bridge state timeout"),
        )
        for timestamp, timeout, reason in checks:
            if timestamp is None or now - timestamp > timeout:
                self.enter_failsafe(reason)
                return False
        if not self.bridge_connected:
            self.enter_failsafe("Bridge disconnected")
            return False
        return True

    def _inside_geofence(self):
        if self.home_lat is None or self.current_lat is None:
            return True
        distance = gps_distance_m(
            (self.home_lat, self.home_lon),
            (self.current_lat, self.current_lon),
        )
        if distance > GEOFENCE_RADIUS_M:
            self.enter_failsafe(
                f"Task 4 geofence violation: {distance:.1f}m > {GEOFENCE_RADIUS_M:.1f}m"
            )
            return False
        return True

    @staticmethod
    def _angle_deg(detection):
        for key in ANGLE_KEYS:
            try:
                value = float(detection.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    @classmethod
    def _normalized_buoy(cls, detection):
        if not isinstance(detection, dict):
            return None
        object_class = str(detection.get("class", "")).strip().lower()
        object_type = str(detection.get("type", "")).strip().lower()
        if object_type != "buoy" and object_class not in BUOY_CLASSES:
            return None
        try:
            distance = float(detection.get("distance"))
        except (TypeError, ValueError):
            return None
        angle = cls._angle_deg(detection)
        if not math.isfinite(distance) or distance <= 0.0 or angle is None:
            return None
        return {
            "class": object_class or "buoy",
            "distance": distance,
            "angle": angle,
        }

    @classmethod
    def _nearest_buoy(cls, detections):
        buoys = []
        for detection in detections or []:
            buoy = cls._normalized_buoy(detection)
            if buoy is not None and buoy["distance"] <= AVOID_EXIT_DISTANCE_M:
                buoys.append(buoy)
        return min(buoys, key=lambda item: item["distance"]) if buoys else None

    def _target_bearing_error(self):
        if self.current_target_index >= len(self.waypoints):
            return 0.0
        target = self.waypoints[self.current_target_index]
        lat_scale = math.cos(math.radians(self.current_lat))
        north = math.radians(float(target["lat"]) - self.current_lat)
        east = math.radians(float(target["lon"]) - self.current_lon) * lat_scale
        target_bearing = math.degrees(math.atan2(east, north)) % 360.0
        return ((target_bearing - self.current_heading + 180.0) % 360.0) - 180.0

    def _choose_avoid_turn(self, buoy):
        # Positive camera angle is starboard/right; turn away from the buoy.
        if buoy["angle"] > 3.0:
            return 1.0  # port/left
        if buoy["angle"] < -3.0:
            return -1.0  # starboard/right
        # For a centred buoy, use the side nearest to the optimized route.
        return -1.0 if self._target_bearing_error() >= 0.0 else 1.0

    def _publish_avoidance(self, buoy):
        linear_x = (
            0.0
            if buoy is not None and buoy["distance"] <= EMERGENCY_DISTANCE_M
            else AVOID_LINEAR_X
        )
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=linear_x,
            angular_z=self.avoid_turn_direction * AVOID_TURN_Z,
        )

    def _update_avoidance(self, buoy, now):
        elapsed = now - self.avoid_started_time
        if elapsed >= AVOID_MAX_DURATION_SEC:
            self.enter_failsafe(
                f"Task 4 obstacle did not clear within {AVOID_MAX_DURATION_SEC:.1f}s"
            )
            return

        clear = buoy is None or buoy["distance"] >= AVOID_EXIT_DISTANCE_M
        if elapsed >= AVOID_MIN_DURATION_SEC and clear:
            if self.avoid_clear_started_time is None:
                self.avoid_clear_started_time = now
            elif now - self.avoid_clear_started_time >= AVOID_CLEAR_DURATION_SEC:
                stop_vehicle(self.topics.cmd_vel_pub)
                self.state = MissionState.NAVIGATING
                self.avoid_started_time = None
                self.avoid_clear_started_time = None
                self.logger.info("Task 4 obstacle cleared; optimized route resumed.")
                return
        else:
            self.avoid_clear_started_time = None
        self._publish_avoidance(buoy)

    def update(self, detections, now=None):
        now = time.monotonic() if now is None else float(now)
        if self.state in (MissionState.FINISHED, MissionState.FAILSAFE):
            return
        if not self._sensors_ready(now) or not self._inside_geofence():
            return
        if not self.route_optimized:
            self._optimize_route()
        if not self.waypoints:
            self.enter_failsafe("Task 4 has no optimized waypoints")
            return

        if self.current_target_index >= len(self.waypoints):
            stop_vehicle(self.topics.cmd_vel_pub)
            self.finished = True
            self.state = MissionState.FINISHED
            self.logger.info("TASK 4 COMPLETED: all optimized route points reached.")
            return

        buoy = self._nearest_buoy(detections)
        if self.state == MissionState.AVOIDING:
            self._update_avoidance(buoy, now)
            return
        if buoy is not None and buoy["distance"] <= AVOID_ENTER_DISTANCE_M:
            self.state = MissionState.AVOIDING
            self.avoid_started_time = now
            self.avoid_clear_started_time = None
            self.avoid_turn_direction = self._choose_avoid_turn(buoy)
            direction = (
                "starboard/right" if self.avoid_turn_direction < 0 else "port/left"
            )
            self.logger.warn(
                "Task 4 buoy avoidance: class=%s distance=%.1fm angle=%.1fdeg turn=%s"
                % (buoy["class"], buoy["distance"], buoy["angle"], direction)
            )
            self._publish_avoidance(buoy)
            return

        target = self.waypoints[self.current_target_index]
        distance = gps_distance_m(
            (self.current_lat, self.current_lon),
            (float(target["lat"]), float(target["lon"])),
        )
        if distance <= WAYPOINT_TOLERANCE_M:
            self.logger.info(
                f"Task 4 WP{target.get('seq', self.current_target_index)} reached "
                f"({distance:.2f}m remaining)."
            )
            self.current_target_index += 1
            return

        publish_set_position(
            self.topics.position_target_pub,
            float(target["lat"]),
            float(target["lon"]),
            float(target.get("alt", 0.0)),
        )
        self.logger.info(
            "Task 4 target %d/%d seq=%s distance=%.1fm"
            % (
                self.current_target_index + 1,
                len(self.waypoints),
                target.get("seq", "?"),
                distance,
            ),
            throttle_duration_sec=1.0,
        )


class Task4Node(Node):
    def __init__(self):
        super().__init__("task4_fast_route_node")
        self.get_logger().info("Task 4 (Fast Optimized Route) Node starting...")

        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback,
        )

        self.latest_detections = []
        self.last_detection_time = None
        self.bridge_connected = False
        self.valid_gps_received = False
        self.valid_heading_received = False
        self.mission_active = False

        self.vision_sub = self.create_subscription(
            String, "/vision/detections", self.vision_callback, 10
        )
        self.active_task_pub = self.create_publisher(
            String, "/mission/active_task", 10
        )

        waypoints = load_task4_waypoints()
        self.get_logger().info(
            f"Task 4 unordered waypoint path={WAYPOINT_PATH.resolve()} count={len(waypoints)}"
        )
        self.task = Task4FastRoute(
            self, self.mission_topics, self.mission_clients, waypoints
        )

        self.control_timer = self.create_timer(0.1, self.timer_callback)
        self.active_task_timer = self.create_timer(1.0, self.publish_active_task)

    def publish_active_task(self):
        if not self.mission_active:
            return
        message = String()
        message.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(message)

    def vision_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().warn(f"Invalid Task 4 vision JSON ignored: {exc}")
            return
        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            self.get_logger().warn("Task 4 vision detections is not a list")
            return
        self.latest_detections = detections
        self.last_detection_time = time.monotonic()

    def _current_detections(self):
        if self.last_detection_time is None:
            return []
        if time.monotonic() - self.last_detection_time > VISION_DETECTION_TIMEOUT_SEC:
            return []
        return self.latest_detections

    def gps_callback(self, message):
        if (
            abs(message.latitude) < MIN_VALID_ABS_COORD
            and abs(message.longitude) < MIN_VALID_ABS_COORD
        ):
            self.get_logger().warn("Invalid Task 4 GPS (0,0) ignored.")
            return
        self.valid_gps_received = True
        self.task.update_gps(message.latitude, message.longitude)

    def heading_callback(self, message):
        self.valid_heading_received = True
        self.task.update_heading(message.data)

    def state_callback(self, message):
        self.bridge_connected = "connected=True" in message.data
        self.task.update_bridge_state(self.bridge_connected)

    def wait_until_ready(self, timeout_sec=30.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.task.state == MissionState.FAILSAFE:
                self.get_logger().error(
                    "Task 4 entered FAILSAFE while waiting for readiness."
                )
                return False
            if (
                self.bridge_connected
                and self.valid_gps_received
                and self.valid_heading_received
                and self.task.route_optimized
            ):
                return True
            self.get_logger().info(
                "Task 4 waiting for Bridge, GPS and heading...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def timer_callback(self):
        if not self.mission_active:
            return
        try:
            self.task.update(self._current_detections())
        except Exception as exc:
            self.get_logger().error(f"Unexpected Task 4 control error: {exc}")
            self.task.enter_failsafe(str(exc))


def main(args=None):
    rclpy.init(args=args)
    node = Task4Node()
    try:
        if not node.wait_until_ready(timeout_sec=30.0):
            node.get_logger().error(
                "Task 4 prerequisites not ready; mission not starting."
            )
            return

        if call_set_mode(
            node, node.mission_clients.set_mode_client, "GUIDED", timeout_sec=5.0
        ) is False:
            node.get_logger().error("Task 4 GUIDED mode failed.")
            return
        if call_trigger_service(
            node,
            node.mission_clients.force_arm_client,
            "FORCE ARM",
            timeout_sec=5.0,
        ) is False:
            node.get_logger().error("Task 4 FORCE ARM failed.")
            return

        node.mission_active = True
        node.task.state = MissionState.NAVIGATING
        node.publish_active_task()
        node.get_logger().info("Task 4 optimized mission loop started.")

        while (
            rclpy.ok()
            and not node.task.finished
            and node.task.state != MissionState.FAILSAFE
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        node.mission_active = False
        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error(
                "Task 4 stopped by FAILSAFE; vehicle will remain in HOLD."
            )
            stop_vehicle(node.mission_topics.cmd_vel_pub)
            if node.task.hold_mode_future is not None:
                rclpy.spin_until_future_complete(
                    node, node.task.hold_mode_future, timeout_sec=2.0
                )
            else:
                call_set_mode(
                    node,
                    node.mission_clients.set_mode_client,
                    HOLD_MODE_NAME,
                    timeout_sec=2.0,
                )
            return

        stop_vehicle(node.mission_topics.cmd_vel_pub)
        node.get_logger().info("Task 4 completed; disarming vehicle.")
        call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
    except KeyboardInterrupt:
        node.mission_active = False
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        try:
            call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
        except Exception:
            pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
