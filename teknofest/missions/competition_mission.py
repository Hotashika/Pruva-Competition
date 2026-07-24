"""TEKNOFEST parkurlarını tek ARM/GUIDED yaşam döngüsünde sırayla çalıştırır."""

import sys
import time
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


class CompetitionState(Enum):
    PARKUR_1 = auto()
    PARKUR_2 = auto()
    PARKUR_3 = auto()
    FAILSAFE = auto()


class CompetitionNode(Task1Node):
    """Ortak sensör akışıyla aynı anda yalnızca bir parkur davranışını çalıştırır."""

    def __init__(self, competition_points):
        self.active_task_name = "task1"
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

        self.get_logger().info(
            "Competition mode hazır: PARKUR_1 GN1->GN2->GN3->GN4, "
            "PARKUR_2 GN4->GN5, PARKUR_3 GN5 sonrası."
        )

    def _publish_active_task(self):
        msg = String()
        msg.data = self.active_task_name
        self.active_task_pub.publish(msg)

    def gps_callback(self, msg):
        super().gps_callback(msg)
        if not hasattr(self, "task2") or not self.valid_gps_received:
            return
        self.task2.update_gps(msg.latitude, msg.longitude)
        self.task3.update_gps(msg.latitude, msg.longitude, self.current_heading or 0.0)

    def heading_callback(self, msg):
        super().heading_callback(msg)
        if not hasattr(self, "task2") or not self.valid_heading_received:
            return
        self.task2.update_heading(self.current_heading)
        self.task3.current_heading = self.current_heading
        self.task3.last_heading_time = self.task1.last_heading_time

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
        if vision_age is None or vision_age > DETECTION_STALE_SEC:
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
        node.get_logger().info(
            "Mission Planner Görev 1 zinciri başladı: task1 -> task2 -> task3."
        )
        while (
                rclpy.ok()
                and node.competition_state != CompetitionState.FAILSAFE
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Competition görevi kullanıcı tarafından durduruldu.")
    finally:
        node.mission_active = False
        stop_vehicle(node.mission_topics.cmd_vel_pub)
        call_trigger_service(node, node.mission_clients.disarm_client, "DISARM")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
