"""TEKNOFEST parkurlarını tek ARM/GUIDED yaşam döngüsünde sırayla çalıştırır."""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_TEXT = str(REPO_ROOT)
# Dosya-yolu ile başlatmada missions/utils, ortak utils paketini gölgelemesin.
while REPO_ROOT_TEXT in sys.path:
    sys.path.remove(REPO_ROOT_TEXT)
sys.path.insert(0, REPO_ROOT_TEXT)

import rclpy
from std_msgs.msg import String

from teknofest.missions.task1_point_tracking import (
    DETECTION_STALE_SEC,
    MissionState as Task1State,
    Task1Node,
)
from teknofest.missions.task2_point_tracking_task_in_an_environment_with_obstacle import (
    MissionState as Task2State,
    Task2PointTrackingWithObstacleAvoidance,
)
from teknofest.missions.task3_kamikaze_engagement import (
    MissionState as Task3State,
    Task3KamikazeEngagement,
)
from teknofest.missions.utils.competition_waypoints import (
    build_competition_routes,
    load_competition_points,
)
from utils.mavlink_utilities import (
    call_set_mode,
    call_trigger_service,
    parse_bridge_state,
    stop_vehicle,
)
from utils.telemetry_csv_logger import TelemetryCsvLogger, TelemetrySample


class CompetitionState(Enum):
    PARKUR_1 = auto()
    PARKUR_2 = auto()
    PARKUR_3 = auto()
    FINISHED = auto()
    FAILSAFE = auto()


class CompetitionNode(Task1Node):
    """Ortak sensör akışıyla aynı anda yalnızca bir parkur davranışını çalıştırır."""

    def __init__(self, competition_points):
        self.active_task_name = "task1"
        self.current_lat = None
        self.current_lon = None
        self.latest_telemetry_sample = None
        self.telemetry_logger = None
        self.telemetry_csv_path = None
        super().__init__()

        self.competition_points = competition_points
        self.task1 = self.task
        self.task2 = Task2PointTrackingWithObstacleAvoidance(
            self, self.mission_topics, self.mission_clients
        )
        self.task3 = Task3KamikazeEngagement(
            self, self.mission_topics, self.mission_clients
        )

        routes = build_competition_routes(competition_points)
        self.task1.waypoints = routes["task1"]
        self.task2.waypoints = routes["task2"]
        self.competition_state = CompetitionState.PARKUR_1
        self.telemetry_sub = self.create_subscription(
            String,
            "/cube/telemetry",
            self.telemetry_callback,
            10,
        )

        self.get_logger().info(
            "Competition mode hazır: PARKUR_1 GN1->GN2->GN3->GN4, "
            "PARKUR_2 GN4->GN5, PARKUR_3 GN5 sonrası."
        )

    def _publish_active_task(self):
        msg = String()
        msg.data = self.active_task_name
        self.active_task_pub.publish(msg)

    def gps_callback(self, msg):
        if not super().gps_callback(msg):
            return
        if not hasattr(self, "task2"):
            return
        self.task2.update_gps(self.current_lat, self.current_lon)
        self.task3.update_gps(self.current_lat, self.current_lon)

    def heading_callback(self, msg):
        super().heading_callback(msg)
        if not hasattr(self, "task2") or not self.valid_heading_received:
            return
        self.task2.update_heading(self.current_heading)
        self.task3.update_heading(self.current_heading)

    def state_callback(self, msg):
        super().state_callback(msg)
        if not hasattr(self, "task2"):
            return
        state = parse_bridge_state(msg.data)
        if {"connected", "armed", "mode"}.issubset(state):
            self.task2.update_bridge_state(
                state["connected"], state["armed"], state["mode"]
            )
        self.task3.update_bridge_state(msg.data)

    def telemetry_callback(self, msg):
        """Cache the latest complete bridge sample for the 1 Hz CSV writer."""

        try:
            payload = json.loads(msg.data)
            sample = TelemetrySample(
                latitude_deg=float(payload["latitude_deg"]),
                longitude_deg=float(payload["longitude_deg"]),
                ground_speed_m_s=float(payload["ground_speed_m_s"]),
                roll_deg=float(payload["roll_deg"]),
                pitch_deg=float(payload["pitch_deg"]),
                heading_deg=float(payload["heading_deg"]),
                speed_setpoint_m_s=float(payload["speed_setpoint_m_s"]),
                heading_setpoint_deg=float(payload["heading_setpoint_deg"]),
            )
            values = (
                sample.latitude_deg,
                sample.longitude_deg,
                sample.ground_speed_m_s,
                sample.roll_deg,
                sample.pitch_deg,
                sample.heading_deg,
                sample.speed_setpoint_m_s,
                sample.heading_setpoint_deg,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("telemetry contains a non-finite value")
            self.latest_telemetry_sample = sample
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(
                f"Geçersiz /cube/telemetry mesajı yok sayıldı: {exc}",
                throttle_duration_sec=2.0,
            )

    def wait_for_complete_telemetry(self, timeout_sec=10.0):
        """Wait until every mandatory CSV field has arrived from the bridge."""

        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            if self.latest_telemetry_sample is not None:
                return True
            self.get_logger().info(
                "CSV kaydı için hız ve yönelim telemetrisi bekleniyor...",
                throttle_duration_sec=2.0,
            )
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def start_telemetry_recording(self):
        """Start a new 1 Hz CSV when the Task 1 competition chain starts."""

        if self.telemetry_logger is not None:
            return
        if self.latest_telemetry_sample is None:
            raise RuntimeError("tam telemetri alınmadan CSV kaydı başlatılamaz")

        output_directory = Path(
            os.getenv(
                "TEKNOFEST_TELEMETRY_DIRECTORY",
                str(REPO_ROOT / "teknofest" / "logs" / "telemetry"),
            )
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.telemetry_csv_path = (
            output_directory / f"vehicle_telemetry_{timestamp}.csv"
        )
        logger = TelemetryCsvLogger(
            self.telemetry_csv_path,
            sample_rate_hz=1.0,
            append=False,
        )
        logger.start(lambda: self.latest_telemetry_sample)
        self.telemetry_logger = logger
        self.get_logger().info(
            f"1 Hz araç telemetri CSV kaydı başladı: {self.telemetry_csv_path}"
        )

    def stop_telemetry_recording(self):
        """Write the final sample and close the CSV; safe to call repeatedly."""

        logger = self.telemetry_logger
        if logger is None:
            return
        self.telemetry_logger = None

        close_error = None
        try:
            if self.latest_telemetry_sample is not None:
                logger.write(self.latest_telemetry_sample)
        except Exception as exc:  # noqa: BLE001 - kapanış yine devam etmeli
            close_error = exc

        try:
            logger.close()
        except Exception as exc:  # noqa: BLE001 - kapanış güvenliği
            if close_error is None:
                close_error = exc

        if close_error is None:
            self.get_logger().info(
                f"Araç telemetri CSV kaydı kapatıldı: {self.telemetry_csv_path}"
            )
        else:
            self.get_logger().error(
                f"Telemetri CSV kapatma hatası: {close_error}"
            )

    def _transition_to(self, state, task_name):
        completed_task_name = self.active_task_name
        stop_vehicle(self.mission_topics.cmd_vel_pub)

        if task_name == "task2":
            if self.current_lat is None or self.current_lon is None:
                self._enter_competition_failsafe(
                    "Task 2 geçişinde geçerli GPS yok."
                )
                return

            self.task2.reset_geofence_origin(
                self.current_lat,
                self.current_lon,
            )
        elif task_name == "task3":
            if (
                    self.current_lat is None
                    or self.current_lon is None
                    or self.current_heading is None
            ):
                self._enter_competition_failsafe(
                    "Task 3 geçişinde geçerli GPS/heading yok."
                )
                return
            self.task3.reset_for_entry(
                self.current_lat,
                self.current_lon,
                self.current_heading,
            )

        self.competition_state = state
        self.active_task_name = task_name
        self._publish_active_task()
        self.get_logger().info(
            f"{completed_task_name} tamamlandı; {task_name} otomatik başlatıldı."
        )

    def _enter_competition_failsafe(self, reason):
        self.get_logger().error(reason)
        self.competition_state = CompetitionState.FAILSAFE
        stop_vehicle(self.mission_topics.cmd_vel_pub)
        self.task1._request_hold_mode()

    def _finish_competition(self):
        stop_vehicle(self.mission_topics.cmd_vel_pub)
        self.competition_state = CompetitionState.FINISHED
        self.active_task_name = "finished"
        self._publish_active_task()
        self.stop_telemetry_recording()
        self.get_logger().info(
            "Task 3 tamamlandı; yarışma zinciri ve telemetri kaydı sonlandırıldı."
        )

    # noinspection D
    def timer_callback(self):
        if not self.mission_active or not hasattr(self, "task2"):
            return

        detections = self._get_fresh_detections()
        vision_age = (
            None
            if self.last_detection_message_time is None
            else time.monotonic() - self.last_detection_message_time
        )
        vision_stale_limit = DETECTION_STALE_SEC
        if self.competition_state == CompetitionState.PARKUR_3:
            task3_config = getattr(self.task3, "config", None)
            vision_stale_limit = min(
                DETECTION_STALE_SEC,
                float(
                    getattr(
                        task3_config,
                        "vision_stale_sec",
                        DETECTION_STALE_SEC,
                    )
                ),
            )
        if vision_age is None or vision_age > vision_stale_limit:
            self._enter_competition_failsafe("Vision heartbeat kaybı. FAILSAFE + HOLD.")
            return

        try:
            if self.competition_state == CompetitionState.PARKUR_1:
                self.task1.update(detections)
                if self.task1.state == Task1State.FAILSAFE:
                    self._enter_competition_failsafe("Task 1 FAILSAFE.")
                elif self.task1.finished:
                    self._transition_to(CompetitionState.PARKUR_2, "task2")

            elif self.competition_state == CompetitionState.PARKUR_2:
                self.task2.update(detections)
                if self.task2.state == Task2State.FAILSAFE:
                    self._enter_competition_failsafe("Task 2 FAILSAFE.")
                elif self.task2.finished:
                    self._transition_to(CompetitionState.PARKUR_3, "task3")

            elif self.competition_state == CompetitionState.PARKUR_3:
                self.task3.update(detections)
                if self.task3.state == Task3State.FAILSAFE:
                    self._enter_competition_failsafe("Task 3 FAILSAFE.")
                elif self.task3.finished:
                    self._finish_competition()
        except Exception as exc:  # noqa: BLE001
            self._enter_competition_failsafe(f"Competition timer hatası: {exc}")


def main(args=None):
    try:
        competition_points = load_competition_points()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[COMPETITION] GN waypoint doğrulaması başarısız: {exc}")
        return

    rclpy.init(args=args)
    node = CompetitionNode(competition_points)
    try:
        if not node.wait_for_bridge_connection(timeout_sec=30.0):
            node.get_logger().error("Bridge hazır değil; araç ARM edilmedi.")
            return
        if not node.wait_for_valid_navigation_data(timeout_sec=30.0):
            node.get_logger().error("GPS/heading hazır değil; araç ARM edilmedi.")
            return
        if not node.wait_for_complete_telemetry(timeout_sec=10.0):
            node.get_logger().error(
                "Yer hızı/roll/pitch telemetrisi hazır değil; görev başlatılmadı."
            )
            return
        if not node.wait_for_vision(timeout_sec=30.0):
            node.get_logger().error("Vision hazır değil; araç ARM edilmedi.")
            return
        if call_set_mode(node, node.mission_clients.set_mode_client, "GUIDED") is False:
            return
        if call_trigger_service(
                node, node.mission_clients.force_arm_client, "FORCE ARM"
        ) is False:
            return
        if not node.wait_for_operational_vehicle_state(timeout_sec=6.0):
            return

        node.mission_active = True
        node.start_telemetry_recording()
        node.get_logger().info(
            "Mission Planner Görev 1 zinciri başladı: task1 -> task2 -> task3."
        )
        while (
                rclpy.ok()
                and node.competition_state
                not in (
                    CompetitionState.FINISHED,
                    CompetitionState.FAILSAFE,
                )
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.competition_state == CompetitionState.FINISHED:
            node.get_logger().info(
                "Competition tamamlandı: task1 -> task2 -> task3."
            )
    except KeyboardInterrupt:
        node.get_logger().info("Competition görevi kullanıcı tarafından durduruldu.")
    finally:
        node.mission_active = False
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        node.stop_telemetry_recording()
        call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
