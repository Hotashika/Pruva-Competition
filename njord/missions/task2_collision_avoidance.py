#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from std_msgs.msg import String

from utils.mavlink_utilities import (
    align_heading_to_gps_target,
    calculate_gps_distance,
    call_set_mode,
    call_trigger_service,
    create_mission_clients,
    create_mission_topics,
    parse_bridge_state,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    wait_for_mission_services,
)
from utils.read_waypoints import parse_qgc_waypoints


BASE_DIR = Path(__file__).resolve().parent.parent
WAYPOINT_PATH = BASE_DIR.parent / "waypoints" / "njord_task2.waypoints"
KINEMATICS_OUTPUT_DIR = BASE_DIR / "logs" / "task2_vessel_kinematics"
ACTIVE_TASK_NAME = "task2"
HOLD_MODE_NAME = "HOLD"
EARTH_RADIUS_M = 6_371_000.0

KINEMATICS_CSV_FIELDS = (
    "system_timestamp_utc",
    "camera_timestamp_ms",
    "frame_id",
    "detected",
    "track_id",
    "distance_m",
    "bearing_deg",
    "relative_course_deg",
    "relative_speed_mps",
    "true_course_deg",
    "true_speed_mps",
    "closing_rate_mps",
    "tcpa_sec",
    "dcpa_m",
)
# Existing movement commands are intentionally preserved. In this project,
# negative angular_z means starboard/right.
AVOID_LINEAR_X = 0.5
AVOID_TURN_Z = -0.6

WAYPOINT_TOLERANCE_M = 1.0
WAYPOINT_SETTLE_SEC = 0.75
WAYPOINT_HEADING_TOLERANCE_DEG = 15.0
GPS_TIMEOUT_SEC = 2.0
HEADING_TIMEOUT_SEC = 2.0
BRIDGE_STATE_TIMEOUT_SEC = 10.0
MIN_VALID_ABS_COORD = 1e-6

# Vessel monitoring and collision-risk thresholds. These are competition
# defaults, not fixed COLREG distances, and should be tuned during water tests.
MONITOR_DISTANCE_M = 12.0
AVOID_ENTER_DISTANCE_M = 4.5
AVOID_EXIT_DISTANCE_M = 5.5
EMERGENCY_DISTANCE_M = 2.5
SAFE_DCPA_M = 2.5
MAX_TCPA_SEC = 15.0
EMERGENCY_TCPA_SEC = 4.0
MIN_TRACK_SPAN_SEC = 0.4
MIN_TRACK_SAMPLES = 3
MIN_CLOSING_RATE_MPS = 0.05
CONSTANT_BEARING_SPAN_DEG = 8.0

HEAD_ON_HALF_ANGLE_DEG = 15.0
STAND_ON_GRACE_SEC = 2.5
AVOID_MIN_DURATION_SEC = 0.8
AVOID_CLEAR_DURATION_SEC = 1.0
AVOID_MAX_DURATION_SEC = 10.0
VISION_DETECTION_TIMEOUT_SEC = 1.0

VESSEL_TYPES = {"vessel", "boat", "ship"}
BUOY_MODEL_TYPES = {
    "green_buoys",
    "red_buoys",
    "north_buoys",
    "east_buoys",
    "south_buoys",
    "west_buoys",
}
COLLISION_TARGET_ANGLE_KEYS = (
    "Vessel angle: ",
    "Vessel angle",
    "Buoy angle: ",
    "Buoy angle",
    "bearing",
    "angle_deg",
    "angle",
)


class MissionState(Enum):
    INIT = auto()
    NAVIGATING = auto()
    STAND_ON = auto()
    AVOIDING = auto()
    FINISHED = auto()
    FAILSAFE = auto()


@dataclass(frozen=True)
class VesselObservation:
    timestamp: float
    camera_timestamp_ms: int | None
    frame_id: int | None
    track_id: int | None
    distance_m: float
    angle_deg: float
    forward_m: float
    starboard_m: float
    latitude: float
    longitude: float
    heading_deg: float


@dataclass(frozen=True)
class CollisionAssessment:
    risk: bool
    reason: str
    closing_rate_mps: float = 0.0
    tcpa_sec: float | None = None
    dcpa_m: float | None = None


@dataclass(frozen=True)
class VesselKinematics:
    relative_course_deg: float
    relative_speed_mps: float
    true_course_deg: float
    true_speed_mps: float


class VesselKinematicsCsvRecorder:
    """Write timestamped Task 2 vessel-motion estimates to a line-buffered CSV."""

    def __init__(self, output_dir=KINEMATICS_OUTPUT_DIR, *, run_name=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if run_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            run_name = f"vessel_kinematics_{timestamp}.csv"
        run_name = str(run_name)
        if Path(run_name).name != run_name or not run_name.lower().endswith(".csv"):
            raise ValueError("run_name must be a single CSV filename")

        self.path = self.output_dir / run_name
        self._file = self.path.open("x", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=KINEMATICS_CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._closed = False

    @staticmethod
    def _number(value, digits=6):
        if value is None:
            return ""
        return f"{float(value):.{digits}f}"

    def record(
        self,
        observation,
        kinematics,
        assessment,
        *,
        frame_id=None,
        camera_timestamp_ms=None,
    ):
        if self._closed:
            raise RuntimeError("VesselKinematicsCsvRecorder is closed")
        detected = observation is not None
        if detected:
            frame_id = observation.frame_id
            camera_timestamp_ms = observation.camera_timestamp_ms
        frame_id = 0 if frame_id is None else int(frame_id)
        camera_timestamp_ms = (
            0 if camera_timestamp_ms is None else int(camera_timestamp_ms)
        )
        self._writer.writerow(
            {
                "system_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "camera_timestamp_ms": camera_timestamp_ms,
                "frame_id": frame_id,
                "detected": int(detected),
                "track_id": 0
                if not detected or observation.track_id is None
                else observation.track_id,
                "distance_m": self._number(
                    0.0 if not detected else observation.distance_m
                ),
                "bearing_deg": self._number(
                    0.0 if not detected else observation.angle_deg
                ),
                "relative_course_deg": self._number(
                    0.0
                    if kinematics is None
                    else kinematics.relative_course_deg
                ),
                "relative_speed_mps": self._number(
                    0.0
                    if kinematics is None
                    else kinematics.relative_speed_mps
                ),
                "true_course_deg": self._number(
                    0.0 if kinematics is None else kinematics.true_course_deg
                ),
                "true_speed_mps": self._number(
                    0.0 if kinematics is None else kinematics.true_speed_mps
                ),
                "closing_rate_mps": self._number(assessment.closing_rate_mps),
                "tcpa_sec": self._number(
                    0.0 if assessment.tcpa_sec is None else assessment.tcpa_sec
                ),
                "dcpa_m": self._number(
                    0.0 if assessment.dcpa_m is None else assessment.dcpa_m
                ),
            }
        )
        self._file.flush()

    def close(self):
        if self._closed:
            return
        self._file.flush()
        self._file.close()
        self._closed = True


def load_task2_waypoints(path=WAYPOINT_PATH):
    """Load route points and discard the QGC HOME item at sequence zero."""
    waypoints = parse_qgc_waypoints(path)
    route = [wp for wp in waypoints if int(wp.get("seq", -1)) != 0]
    return route or waypoints


class Task2CollisionAvoidance:
    def __init__(self, node, mission_topics, mission_clients, waypoints):
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.clients = mission_clients

        self.waypoints = list(waypoints)
        self.current_target_index = 0
        self.waypoint_tolerance = WAYPOINT_TOLERANCE_M

        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self.last_gps_time = None
        self.last_heading_time = None
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self.last_bridge_state_time = None

        self.finished = False
        self.state = MissionState.INIT
        self.track = deque(maxlen=12)
        self.stand_on_risk_since = None
        self.avoid_started_time = None
        self.avoid_clear_started_time = None
        self.aligned_target_key = None
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None
        self.hold_mode_requested = False
        self.hold_mode_future = None
        self.kinematics_callback = None
        self.latest_kinematics = None

    def update_gps(self, lat, lon, now=None):
        self.current_lat = float(lat)
        self.current_lon = float(lon)
        self.last_gps_time = time.monotonic() if now is None else float(now)

    def update_heading(self, heading_deg, now=None):
        self.current_heading = float(heading_deg)
        self.last_heading_time = time.monotonic() if now is None else float(now)

    def update_bridge_state(self, connected, armed, mode, now=None):
        self.bridge_connected = bool(connected)
        self.bridge_armed = bool(armed)
        self.bridge_mode = str(mode or "UNKNOWN").strip().upper()
        self.last_bridge_state_time = time.monotonic() if now is None else float(now)

    @staticmethod
    def _finite_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _optional_int(value):
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_vessel(cls, detection):
        detector_type = str(detection.get("type", "")).strip().lower()
        model_class = str(detection.get("class", "")).strip().lower()
        is_vessel = detector_type == "vessel" or model_class in VESSEL_TYPES

        # Task 2 water tests use the buoy detector as the collision target.
        # These names mirror the classes embedded in the current buoy model.
        is_buoy = detector_type == "buoy" and model_class in BUOY_MODEL_TYPES
        return is_vessel or is_buoy

    @classmethod
    def _detection_angle_deg(cls, detection):
        for key in COLLISION_TARGET_ANGLE_KEYS:
            if key not in detection:
                continue
            value = cls._finite_float(detection.get(key))
            if value is not None:
                return value
        return None

    @classmethod
    def _normalized_vessel(cls, detection):
        if not isinstance(detection, dict) or not cls._is_vessel(detection):
            return None
        distance_m = cls._finite_float(detection.get("distance"))
        angle_deg = cls._detection_angle_deg(detection)
        if distance_m is None or distance_m <= 0.0 or angle_deg is None:
            return None
        return {
            "distance": distance_m,
            "angle": angle_deg,
            "track_id": cls._optional_int(detection.get("track_id")),
            "raw": detection,
        }

    @classmethod
    def _nearest_vessel(cls, detections):
        vessels = []
        for detection in detections or []:
            vessel = cls._normalized_vessel(detection)
            if vessel is not None and vessel["distance"] <= MONITOR_DISTANCE_M:
                vessels.append(vessel)
        return min(vessels, key=lambda item: item["distance"]) if vessels else None

    def _record_observation(
        self,
        vessel,
        sample_timestamp,
        *,
        frame_id=None,
        camera_timestamp_ms=None,
    ):
        angle_rad = math.radians(vessel["angle"])
        observation = VesselObservation(
            timestamp=float(sample_timestamp),
            camera_timestamp_ms=self._optional_int(camera_timestamp_ms),
            frame_id=self._optional_int(frame_id),
            track_id=vessel.get("track_id"),
            distance_m=vessel["distance"],
            angle_deg=vessel["angle"],
            forward_m=vessel["distance"] * math.cos(angle_rad),
            starboard_m=vessel["distance"] * math.sin(angle_rad),
            latitude=float(self.current_lat),
            longitude=float(self.current_lon),
            heading_deg=float(self.current_heading),
        )

        # Without a tracker id, a large jump is treated as another vessel so
        # observations from different targets are not mixed in one CPA track.
        if self.track:
            previous = self.track[-1]
            track_id_changed = (
                previous.track_id is not None
                and observation.track_id is not None
                and previous.track_id != observation.track_id
            )
            if (
                track_id_changed
                or abs(previous.angle_deg - observation.angle_deg) > 30.0
                or abs(previous.distance_m - observation.distance_m) > 5.0
                or observation.timestamp - previous.timestamp > 1.5
            ):
                self.track.clear()

        self.track.append(observation)
        return observation

    @staticmethod
    def _gps_displacement_m(first, latest):
        first_lat_rad = math.radians(first.latitude)
        latest_lat_rad = math.radians(latest.latitude)
        mean_lat_rad = (first_lat_rad + latest_lat_rad) / 2.0
        north_m = EARTH_RADIUS_M * (latest_lat_rad - first_lat_rad)
        east_m = (
            EARTH_RADIUS_M
            * math.cos(mean_lat_rad)
            * math.radians(latest.longitude - first.longitude)
        )
        return north_m, east_m

    @staticmethod
    def _relative_offset_world(observation):
        heading_rad = math.radians(observation.heading_deg)
        north_m = (
            observation.forward_m * math.cos(heading_rad)
            - observation.starboard_m * math.sin(heading_rad)
        )
        east_m = (
            observation.forward_m * math.sin(heading_rad)
            + observation.starboard_m * math.cos(heading_rad)
        )
        return north_m, east_m

    def _estimate_kinematics(self):
        if len(self.track) < MIN_TRACK_SAMPLES:
            return None

        first = self.track[0]
        latest = self.track[-1]
        elapsed = latest.timestamp - first.timestamp
        if elapsed < MIN_TRACK_SPAN_SEC:
            return None

        own_north_m, own_east_m = self._gps_displacement_m(first, latest)
        first_rel_north_m, first_rel_east_m = self._relative_offset_world(first)
        latest_rel_north_m, latest_rel_east_m = self._relative_offset_world(latest)

        own_velocity_north = own_north_m / elapsed
        own_velocity_east = own_east_m / elapsed
        relative_velocity_north = (
            latest_rel_north_m - first_rel_north_m
        ) / elapsed
        relative_velocity_east = (
            latest_rel_east_m - first_rel_east_m
        ) / elapsed
        true_velocity_north = own_velocity_north + relative_velocity_north
        true_velocity_east = own_velocity_east + relative_velocity_east

        latest_heading_rad = math.radians(latest.heading_deg)
        relative_velocity_forward = (
            relative_velocity_north * math.cos(latest_heading_rad)
            + relative_velocity_east * math.sin(latest_heading_rad)
        )
        relative_velocity_starboard = (
            -relative_velocity_north * math.sin(latest_heading_rad)
            + relative_velocity_east * math.cos(latest_heading_rad)
        )

        relative_speed = math.hypot(
            relative_velocity_forward,
            relative_velocity_starboard,
        )
        true_speed = math.hypot(true_velocity_north, true_velocity_east)
        relative_course = math.degrees(
            math.atan2(relative_velocity_starboard, relative_velocity_forward)
        )
        true_course = (
            math.degrees(math.atan2(true_velocity_east, true_velocity_north))
            + 360.0
        ) % 360.0

        return VesselKinematics(
            relative_course_deg=relative_course,
            relative_speed_mps=relative_speed,
            true_course_deg=true_course,
            true_speed_mps=true_speed,
        )

    def _assess_collision_risk(self):
        if not self.track:
            return CollisionAssessment(False, "no_track")

        latest = self.track[-1]
        if latest.distance_m <= EMERGENCY_DISTANCE_M:
            return CollisionAssessment(True, "emergency_distance")

        if len(self.track) < MIN_TRACK_SAMPLES:
            return CollisionAssessment(False, "collecting_track")

        first = self.track[0]
        elapsed = latest.timestamp - first.timestamp
        if elapsed < MIN_TRACK_SPAN_SEC:
            return CollisionAssessment(False, "collecting_track")

        closing_rate = (first.distance_m - latest.distance_m) / elapsed
        if closing_rate < MIN_CLOSING_RATE_MPS:
            return CollisionAssessment(False, "not_closing", closing_rate_mps=closing_rate)

        velocity_forward = (latest.forward_m - first.forward_m) / elapsed
        velocity_starboard = (latest.starboard_m - first.starboard_m) / elapsed
        velocity_sq = velocity_forward ** 2 + velocity_starboard ** 2

        tcpa = None
        dcpa = None
        if velocity_sq > 1e-6:
            tcpa = -(
                latest.forward_m * velocity_forward
                + latest.starboard_m * velocity_starboard
            ) / velocity_sq
            if tcpa >= 0.0:
                cpa_forward = latest.forward_m + velocity_forward * tcpa
                cpa_starboard = latest.starboard_m + velocity_starboard * tcpa
                dcpa = math.hypot(cpa_forward, cpa_starboard)

        if (
            tcpa is not None
            and dcpa is not None
            and 0.0 <= tcpa <= MAX_TCPA_SEC
            and dcpa <= SAFE_DCPA_M
        ):
            return CollisionAssessment(
                True,
                "unsafe_cpa",
                closing_rate_mps=closing_rate,
                tcpa_sec=tcpa,
                dcpa_m=dcpa,
            )

        angle_span = max(item.angle_deg for item in self.track) - min(
            item.angle_deg for item in self.track
        )
        if (
            latest.distance_m <= AVOID_ENTER_DISTANCE_M
            and angle_span <= CONSTANT_BEARING_SPAN_DEG
        ):
            return CollisionAssessment(
                True,
                "constant_bearing_closing_range",
                closing_rate_mps=closing_rate,
                tcpa_sec=tcpa,
                dcpa_m=dcpa,
            )

        return CollisionAssessment(
            False,
            "safe_cpa",
            closing_rate_mps=closing_rate,
            tcpa_sec=tcpa,
            dcpa_m=dcpa,
        )

    @staticmethod
    def _encounter_role(angle_deg):
        """Return encounter and COLREG role using camera-relative bearing.

        A full overtaking classification will require target course/velocity.
        Until then, a target ahead is handled conservatively as head-on.
        """
        if abs(angle_deg) <= HEAD_ON_HALF_ANGLE_DEG:
            return "head_on", "give_way"
        if angle_deg > HEAD_ON_HALF_ANGLE_DEG:
            return "crossing_starboard", "give_way"
        return "crossing_port", "stand_on"

    def _request_hold_mode(self):
        if self.hold_mode_requested or self.clients is None:
            return
        self.hold_mode_requested = True
        request = SetMode.Request()
        request.base_mode = 0
        request.custom_mode = HOLD_MODE_NAME
        try:
            self.hold_mode_future = self.clients.set_mode_client.call_async(request)
            self.logger.warn("Task 2 failsafe: HOLD mode requested.")
        except Exception as exc:
            self.logger.error(f"Task 2 HOLD request failed: {exc}")

    def _enter_failsafe(self, reason):
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
                self._enter_failsafe(reason)
                return False
        if not self.bridge_connected:
            self._enter_failsafe("MAVLink bridge disconnected")
            return False
        if self.bridge_mode != "GUIDED":
            self._enter_failsafe(
                f"Orange Cube left GUIDED mode (mode={self.bridge_mode})"
            )
            return False
        if not self.bridge_armed:
            self._enter_failsafe("Orange Cube is no longer armed")
            return False
        return True

    def _begin_waypoint_hold(self, waypoint_name, now):
        """Stop briefly at a waypoint before aligning with the next leg."""
        stop_vehicle(self.topics.cmd_vel_pub)
        self.waypoint_hold_until = float(now) + WAYPOINT_SETTLE_SEC
        self.waypoint_hold_name = waypoint_name
        self.aligned_target_key = None
        self.logger.info(
            f"{waypoint_name} reached; vehicle stopped for "
            f"{WAYPOINT_SETTLE_SEC:.2f}s before next heading alignment."
        )

    def _waypoint_hold_active(self, now):
        """Keep the vehicle stopped until the waypoint settle time expires."""
        if self.waypoint_hold_until is None:
            return False

        remaining = self.waypoint_hold_until - float(now)
        if remaining > 0.0:
            publish_cmd_vel(
                self.topics.cmd_vel_pub,
                linear_x=0.0,
                angular_z=0.0,
            )
            self.logger.info(
                f"Holding at {self.waypoint_hold_name}: {remaining:.2f}s remaining.",
                throttle_duration_sec=0.5,
            )
            return True

        completed_name = self.waypoint_hold_name
        self.waypoint_hold_until = None
        self.waypoint_hold_name = None
        self.logger.info(
            f"{completed_name} stop stabilized; aligning with the next waypoint."
        )
        return False

    def _publish_waypoint_target(self, target):
        if self.current_lat is None or self.current_lon is None:
            return False
        distance = calculate_gps_distance(
            self.current_lat,
            self.current_lon,
            target["lat"],
            target["lon"],
        )
        if distance <= self.waypoint_tolerance:
            self.logger.info(
                f"WP{self.current_target_index} reached. Remaining={distance:.2f}m"
            )
            return True

        target_name = f"WP{self.current_target_index}"
        target_key = (
            target_name,
            round(float(target["lat"]), 7),
            round(float(target["lon"]), 7),
        )
        if self.aligned_target_key != target_key:
            if not align_heading_to_gps_target(
                self.topics.cmd_vel_pub,
                self.current_lat,
                self.current_lon,
                self.current_heading,
                target["lat"],
                target["lon"],
                logger=self.logger,
                target_name=target_name,
                tolerance_deg=WAYPOINT_HEADING_TOLERANCE_DEG,
            ):
                return False
            self.aligned_target_key = target_key

        publish_set_position(
            self.topics.position_target_pub,
            target["lat"],
            target["lon"],
            target.get("alt", 20.0),
        )
        self.logger.info(
            f"WP{self.current_target_index}: distance={distance:.2f}m | set_position sent",
            throttle_duration_sec=1.0,
        )
        return False

    def _start_starboard_avoidance(self, now, encounter, assessment):
        self.state = MissionState.AVOIDING
        self.avoid_started_time = now
        self.avoid_clear_started_time = None
        self.stand_on_risk_since = None
        # Avoidance changes the heading, so re-align before resuming this leg.
        self.aligned_target_key = None
        tcpa_text = "unknown" if assessment.tcpa_sec is None else f"{assessment.tcpa_sec:.1f}s"
        dcpa_text = "unknown" if assessment.dcpa_m is None else f"{assessment.dcpa_m:.1f}m"
        self.logger.warn(
            "Collision risk: encounter=%s reason=%s TCPA=%s DCPA=%s; "
            "starting starboard avoidance."
            % (encounter, assessment.reason, tcpa_text, dcpa_text)
        )
        self._publish_starboard_command()

    def _publish_starboard_command(self):
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=AVOID_LINEAR_X,
            angular_z=AVOID_TURN_Z,
        )

    def _update_avoidance(self, vessel, now):
        if self.avoid_started_time is None:
            self.avoid_started_time = now

        elapsed = now - self.avoid_started_time
        if elapsed >= AVOID_MAX_DURATION_SEC:
            self._enter_failsafe(
                f"Starboard avoidance exceeded {AVOID_MAX_DURATION_SEC:.1f}s"
            )
            return

        vessel_clear = vessel is None or vessel["distance"] >= AVOID_EXIT_DISTANCE_M
        if elapsed >= AVOID_MIN_DURATION_SEC and vessel_clear:
            if self.avoid_clear_started_time is None:
                self.avoid_clear_started_time = now
            elif now - self.avoid_clear_started_time >= AVOID_CLEAR_DURATION_SEC:
                self.logger.info("Vessel is past and clear; resuming the same waypoint.")
                stop_vehicle(self.topics.cmd_vel_pub)
                self.state = MissionState.NAVIGATING
                self.avoid_started_time = None
                self.avoid_clear_started_time = None
                self.track.clear()
                return
        else:
            self.avoid_clear_started_time = None

        self._publish_starboard_command()

    def update(
        self,
        detections,
        now=None,
        record_observation=True,
        *,
        frame_id=None,
        camera_timestamp_ms=None,
    ):
        now = time.monotonic() if now is None else float(now)

        if self.state in (MissionState.FINISHED, MissionState.FAILSAFE):
            return
        if not self._sensors_ready(now):
            return
        if not self.waypoints:
            self._enter_failsafe("Task 2 waypoint list is empty")
            return

        if self._waypoint_hold_active(now):
            return

        if self.current_target_index >= len(self.waypoints):
            stop_vehicle(self.topics.cmd_vel_pub)
            self.finished = True
            self.state = MissionState.FINISHED
            self.logger.info("TASK 2 COMPLETED")
            return

        vessel = self._nearest_vessel(detections)
        observation = None
        if vessel is not None and record_observation:
            camera_timestamp_ms = self._optional_int(camera_timestamp_ms)
            sample_timestamp = (
                now
                if camera_timestamp_ms is None
                else camera_timestamp_ms / 1000.0
            )
            observation = self._record_observation(
                vessel,
                sample_timestamp,
                frame_id=frame_id,
                camera_timestamp_ms=camera_timestamp_ms,
            )

        assessment = (
            self._assess_collision_risk()
            if vessel is not None
            else CollisionAssessment(False, "no_vessel")
        )
        if observation is not None:
            self.latest_kinematics = self._estimate_kinematics()
        elif vessel is None:
            self.latest_kinematics = None
        if record_observation and self.kinematics_callback is not None:
            self.kinematics_callback(
                observation,
                self.latest_kinematics,
                assessment,
                frame_id,
                camera_timestamp_ms,
            )

        if self.state == MissionState.AVOIDING:
            self._update_avoidance(vessel, now)
            return

        if vessel is None:
            self.track.clear()
            self.stand_on_risk_since = None
            if self.state == MissionState.STAND_ON:
                self.state = MissionState.NAVIGATING
        elif assessment.risk:
            encounter, role = self._encounter_role(vessel["angle"])
            emergency = (
                vessel["distance"] <= EMERGENCY_DISTANCE_M
                or (
                    assessment.tcpa_sec is not None
                    and assessment.tcpa_sec <= EMERGENCY_TCPA_SEC
                )
            )
            if role == "give_way":
                self._start_starboard_avoidance(now, encounter, assessment)
                return

            if self.stand_on_risk_since is None:
                self.stand_on_risk_since = now
                self.logger.warn(
                    f"Collision risk from port side; standing on while monitoring ({assessment.reason})."
                )
            self.state = MissionState.STAND_ON
            if emergency or now - self.stand_on_risk_since >= STAND_ON_GRACE_SEC:
                self._start_starboard_avoidance(now, encounter, assessment)
                return
        else:
            self.stand_on_risk_since = None
            self.state = MissionState.NAVIGATING

        target = self.waypoints[self.current_target_index]
        if self._publish_waypoint_target(target):
            self._begin_waypoint_hold(
                f"WP{self.current_target_index}",
                now,
            )
            self.current_target_index += 1


class Task2Node(Node):
    def __init__(self):
        super().__init__("task2_collision_avoidance_node")
        self.get_logger().info("Task 2 (Collision Avoidance) Node starting...")

        self.mission_clients = create_mission_clients(self)
        wait_for_mission_services(self, self.mission_clients)
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback,
        )

        self.latest_detections = []
        self.latest_frame_id = None
        self.latest_camera_timestamp_ms = None
        self.last_detection_time = None
        self.last_consumed_detection_time = None
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self._last_logged_bridge_state = None
        self.valid_gps_received = False
        self.valid_heading_received = False
        self.mission_active = False
        self.kinematics_recorder = None

        self.vision_sub = self.create_subscription(
            String,
            "/vision/detections",
            self.vision_callback,
            10,
        )
        self.active_task_pub = self.create_publisher(String, "/mission/active_task", 10)

        waypoints = load_task2_waypoints()
        self.get_logger().info(
            f"Task 2 waypoint path={WAYPOINT_PATH.resolve()} count={len(waypoints)}"
        )
        self.task = Task2CollisionAvoidance(
            self,
            self.mission_topics,
            self.mission_clients,
            waypoints,
        )

        self.control_timer = self.create_timer(0.1, self.timer_callback)
        self.active_task_timer = self.create_timer(1.0, self.publish_active_task)

    def publish_active_task(self):
        message = String()
        message.data = ACTIVE_TASK_NAME
        self.active_task_pub.publish(message)

    def vision_callback(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f"Invalid vision JSON ignored: {exc}",
                throttle_duration_sec=2.0,
            )
            return
        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            self.get_logger().warn(
                "Vision detections is not a list; message ignored.",
                throttle_duration_sec=2.0,
            )
            return
        self.latest_detections = detections
        self.latest_frame_id = self.task._optional_int(payload.get("frame_id"))
        self.latest_camera_timestamp_ms = self.task._optional_int(
            payload.get("camera_timestamp_ms")
        )
        self.last_detection_time = time.monotonic()

    def _current_detection_sample(self):
        if self.last_detection_time is None:
            return [], False, None, None
        if time.monotonic() - self.last_detection_time > VISION_DETECTION_TIMEOUT_SEC:
            return [], False, None, None
        is_new = self.last_consumed_detection_time != self.last_detection_time
        if is_new:
            self.last_consumed_detection_time = self.last_detection_time
        return (
            self.latest_detections,
            is_new,
            self.latest_frame_id,
            self.latest_camera_timestamp_ms,
        )

    def start_kinematics_recording(self):
        if self.kinematics_recorder is not None:
            return self.kinematics_recorder.path
        output_dir = os.getenv(
            "NJORD_TASK2_KINEMATICS_DIR",
            str(KINEMATICS_OUTPUT_DIR),
        )
        try:
            self.kinematics_recorder = VesselKinematicsCsvRecorder(output_dir)
        except Exception as exc:
            self.get_logger().error(
                f"Task 2 vessel kinematics CSV could not be started: {exc}"
            )
            return None
        self.task.kinematics_callback = self._record_kinematics
        self.get_logger().info(
            f"Task 2 vessel kinematics CSV -> {self.kinematics_recorder.path}"
        )
        return self.kinematics_recorder.path

    def _record_kinematics(
        self,
        observation,
        kinematics,
        assessment,
        frame_id,
        camera_timestamp_ms,
    ):
        if self.kinematics_recorder is None:
            return
        try:
            self.kinematics_recorder.record(
                observation,
                kinematics,
                assessment,
                frame_id=frame_id,
                camera_timestamp_ms=camera_timestamp_ms,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Task 2 vessel kinematics logging disabled: {exc}"
            )
            try:
                self.kinematics_recorder.close()
            except Exception:
                pass
            self.kinematics_recorder = None
            self.task.kinematics_callback = None

    def gps_callback(self, message):
        if abs(message.latitude) < MIN_VALID_ABS_COORD and abs(message.longitude) < MIN_VALID_ABS_COORD:
            self.get_logger().warn("Invalid GPS (0,0) ignored.", throttle_duration_sec=2.0)
            return
        self.valid_gps_received = True
        self.task.update_gps(message.latitude, message.longitude)

    def heading_callback(self, message):
        self.valid_heading_received = True
        self.task.update_heading(message.data)

    def state_callback(self, message):
        state = parse_bridge_state(message.data)
        required_keys = {"connected", "armed", "mode"}
        if not required_keys.issubset(state):
            self.get_logger().warn(
                f"Incomplete /cube/state ignored: {message.data}",
                throttle_duration_sec=2.0,
            )
            return

        self.bridge_connected = state["connected"] is True
        self.bridge_armed = state["armed"] is True
        self.bridge_mode = str(state["mode"] or "UNKNOWN").strip().upper()
        current_state = (
            self.bridge_connected,
            self.bridge_armed,
            self.bridge_mode,
        )
        if current_state != self._last_logged_bridge_state:
            self.get_logger().info(
                "Task2 bridge state: "
                f"connected={self.bridge_connected}, "
                f"armed={self.bridge_armed}, mode={self.bridge_mode}"
            )
            self._last_logged_bridge_state = current_state

        self.task.update_bridge_state(
            self.bridge_connected,
            self.bridge_armed,
            self.bridge_mode,
        )

    def wait_until_ready(self, timeout_sec=30.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            gps_fresh = (
                self.task.last_gps_time is not None
                and now - self.task.last_gps_time <= GPS_TIMEOUT_SEC
            )
            heading_fresh = (
                self.task.last_heading_time is not None
                and now - self.task.last_heading_time <= HEADING_TIMEOUT_SEC
            )
            state_fresh = (
                self.task.last_bridge_state_time is not None
                and now - self.task.last_bridge_state_time <= BRIDGE_STATE_TIMEOUT_SEC
            )
            if (
                self.bridge_connected
                and self.valid_gps_received
                and self.valid_heading_received
                and gps_fresh
                and heading_fresh
                and state_fresh
            ):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def wait_for_vehicle_state(
            self,
            expected_mode=None,
            expected_armed=None,
            timeout_sec=6.0,
    ):
        expected_mode = (
            None
            if expected_mode is None
            else str(expected_mode).strip().upper()
        )
        deadline = time.monotonic() + float(timeout_sec)
        expected_parts = ["connected=True"]
        if expected_mode is not None:
            expected_parts.append(f"mode={expected_mode}")
        if expected_armed is not None:
            expected_parts.append(f"armed={bool(expected_armed)}")
        expected_text = ", ".join(expected_parts)
        self.get_logger().info(
            f"Task2 waiting for confirmed vehicle state: {expected_text}"
        )

        while rclpy.ok() and time.monotonic() < deadline:
            mode_ok = expected_mode is None or self.bridge_mode == expected_mode
            armed_ok = (
                expected_armed is None
                or self.bridge_armed == bool(expected_armed)
            )
            if self.bridge_connected and mode_ok and armed_ok:
                self.get_logger().info(
                    f"Task2 vehicle state confirmed: {expected_text}"
                )
                return True
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().error(
            "Task2 vehicle-state confirmation timeout: "
            f"expected=({expected_text}), actual=(connected={self.bridge_connected}, "
            f"armed={self.bridge_armed}, mode={self.bridge_mode})"
        )
        return False

    def timer_callback(self):
        if not self.mission_active:
            return
        try:
            (
                detections,
                is_new,
                frame_id,
                camera_timestamp_ms,
            ) = self._current_detection_sample()
            self.task.update(
                detections,
                record_observation=is_new,
                frame_id=frame_id,
                camera_timestamp_ms=camera_timestamp_ms,
            )
        except Exception as exc:
            self.get_logger().error(f"Unexpected Task 2 control error: {exc}")
            self.task._enter_failsafe(str(exc))

    def destroy_node(self):
        self.task.kinematics_callback = None
        if self.kinematics_recorder is not None:
            try:
                self.kinematics_recorder.close()
            except Exception as exc:
                self.get_logger().error(
                    f"Task 2 vessel kinematics CSV close failed: {exc}"
                )
            self.kinematics_recorder = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Task2Node()
    try:
        if not node.task.waypoints:
            node.get_logger().error("Task 2 has no waypoint; mission not starting.")
            return
        if not node.wait_until_ready(timeout_sec=30.0):
            node.get_logger().error(
                "Bridge/GPS/heading not ready within 30 seconds; mission not starting."
            )
            return

        if call_set_mode(node, node.mission_clients.set_mode_client, "GUIDED") is False:
            node.get_logger().error("Failed to switch to GUIDED mode.")
            return
        if not node.wait_for_vehicle_state(
            expected_mode="GUIDED",
            timeout_sec=6.0,
        ):
            node.get_logger().error(
                "GUIDED was not confirmed on /cube/state; mission not starting."
            )
            return
        if call_trigger_service(
            node,
            node.mission_clients.force_arm_client,
            "FORCE ARM",
        ) is False:
            node.get_logger().error("FORCE ARM failed.")
            return
        if not node.wait_for_vehicle_state(
            expected_mode="GUIDED",
            expected_armed=True,
            timeout_sec=6.0,
        ):
            node.get_logger().error(
                "armed=True and mode=GUIDED were not confirmed; mission not starting."
            )
            return
        if not node.wait_until_ready(timeout_sec=3.0):
            node.get_logger().error(
                "Fresh GPS/heading/bridge data was not restored after arming; "
                "mission not starting."
            )
            return

        node.start_kinematics_recording()
        node.mission_active = True
        node.task.state = MissionState.NAVIGATING
        node.publish_active_task()
        node.get_logger().info(
            "Task 2 mission loop started with confirmed vehicle state: "
            f"connected={node.bridge_connected}, armed={node.bridge_armed}, "
            f"mode={node.bridge_mode}"
        )

        while (
            rclpy.ok()
            and not node.task.finished
            and node.task.state != MissionState.FAILSAFE
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        node.mission_active = False
        if node.task.state == MissionState.FAILSAFE:
            node.get_logger().error(
                "Task 2 terminated due to FAILSAFE. Vehicle will stay in HOLD "
                "if mode change succeeds."
            )
            stop_vehicle(node.mission_topics.cmd_vel_pub)
            if node.task.hold_mode_future is not None:
                rclpy.spin_until_future_complete(
                    node,
                    node.task.hold_mode_future,
                    timeout_sec=2.0,
                )
                if not node.task.hold_mode_future.done():
                    node.get_logger().error(
                        "HOLD mode request did not complete before shutdown."
                    )
            else:
                call_set_mode(
                    node,
                    node.mission_clients.set_mode_client,
                    HOLD_MODE_NAME,
                    timeout_sec=2.0,
                )
            return

        node.get_logger().info("Task 2 finished. Stopping vehicle.")
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        node.get_logger().info("Disarming vehicle...")
        call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
    except KeyboardInterrupt:
        node.get_logger().info("Task 2 interrupted manually.")
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
