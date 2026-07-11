# task2_collision_avoidance
from utils.mavlink_utilities import (
    calculate_gps_distance,
    call_trigger_service,
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
)


class Task2CollisionAvoidance:
    def __init__(self, node, mission_topics, mission_clients, waypoints):
        self.node = node
        self.logger = node.get_logger()

        self.topics = mission_topics
        self.clients = mission_clients

        self.waypoints = waypoints
        self.current_target_index = 0
        self.waypoint_tolerance = 1.0

        self.current_lat = None
        self.current_lon = None
        self.finished = False

    def update_gps(self, lat, lon):
        self.current_lat = float(lat)
        self.current_lon = float(lon)

    @staticmethod
    def _normalize_bearing_error(value):
        return (float(value) + 180.0) % 360.0 - 180.0

    def _classify_side(self, bearing):
        try:
            bearing_error = self._normalize_bearing_error(bearing)
        except (TypeError, ValueError):
            return "head_on"
        if abs(bearing_error) <= 15.0:
            return "head_on"
        if bearing_error < 0.0:
            return "left"
        return "right"

    def _set_position_to_gps_target(self, target_gps, target_name):
        if self.current_lat is None or self.current_lon is None:
            self.logger.info("GPS konumu bekleniyor...", throttle_duration_sec=2.0)
            stop_vehicle(self.topics.cmd_vel_pub)
            return False

        distance = calculate_gps_distance(
            self.current_lat,
            self.current_lon,
            target_gps["lat"],
            target_gps["lon"],
        )

        if distance <= self.waypoint_tolerance:
            self.logger.info(f"{target_name} ulaşıldı. Kalan mesafe: {distance:.2f} m")
            return True

        publish_set_position(
            self.topics.position_target_pub,
            target_gps["lat"],
            target_gps["lon"],
        )
        self.logger.info(
            f"{target_name}: mesafe={distance:.2f} m | set_position gönderildi",
            throttle_duration_sec=1.0,
        )
        return False

    def update(self, detections):
        if not self.waypoints:
            self.logger.warn("Görev listesi boş! Lütfen YKİ'den GPS rotası yükleyin.")
            stop_vehicle(self.topics.cmd_vel_pub)
            return

        if self.current_target_index >= len(self.waypoints):
            if not self.finished:
                self.logger.info("GÖREV TAMAMLANDI!")
                stop_vehicle(self.topics.cmd_vel_pub)
                call_trigger_service(self.node, self.clients.disarm_client, "DISARM")
                self.finished = True
            return

        target_gps = self.waypoints[self.current_target_index]

        for obj in detections or []:
            if obj.get("class") != "vessel" or not (0 < obj.get("distance", 0) < 4.5):
                continue

            side = self._classify_side(obj.get("bearing", 0.0))

            match side:
                case "head_on":
                    self.logger.info("Vessel comes from across! Avoiding from right side.")
                    publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.5, angular_z=-0.6)
                case "left":
                    self.logger.info("Vessel comes from left side! Avoiding from right side.")
                    publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.5, angular_z=0.6)
                case "right":
                    self.logger.info("Vessel comes from right side! Avoiding from right side.")
                    publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.5, angular_z=0.6)
            return

        if self._set_position_to_gps_target(target_gps, f"WP{self.current_target_index}"):
            self.current_target_index += 1
