#!/usr/bin/env python3
"""Mission Planner komutuyla TEKNOFEST Task 1/2/3 secen guvenli yonetici.

Tek bir ROS dugumu /cube/cmd_vel yayincisina sahip olur. Boylece farkli
gorev dugumlerinin ayni anda motor komutu gondermesi engellenir.
"""

import json
import math
import os
import threading
import time
from enum import Enum, auto

import rclpy
from mavros_msgs.srv import SetMode
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

from teknofest.missions.task1_point_tracking import (
    DETECTION_STALE_SEC,
    GPS_TIMEOUT_SEC as TASK1_GPS_TIMEOUT_SEC,
    MissionState as Task1State,
    Task1Maneuvering,
    WAYPOINT_PATH as TASK1_WAYPOINT_PATH,
)
from teknofest.missions.task2_point_tracking_task_in_an_environment_with_obstacle import (
    HEADING_TIMEOUT_SEC as TASK2_HEADING_TIMEOUT_SEC,
    MissionState as Task2State,
    Task2PointTrackingWithObstacleAvoidance,
    WAYPOINT_PATH as TASK2_WAYPOINT_PATH,
)
from teknofest.missions.task3_kamikaze_engagement import (
    ACTIVE_TARGET_COLOR,
    DRIVE_MODE,
    GPS_TIMEOUT_SEC as TASK3_GPS_TIMEOUT_SEC,
    HEADING_TIMEOUT_SEC as TASK3_HEADING_TIMEOUT_SEC,
    IMU_TIMEOUT_SEC,
    MissionState as Task3State,
    SUPPORTED_TARGET_COLORS,
    Task3KamikazeEngagement,
    VISION_TIMEOUT_SEC,
)
from utils.mavlink_utilities import (
    create_mission_clients,
    create_mission_topics,
    parse_bridge_state,
    stop_vehicle,
)


class ManagerState(Enum):
    IDLE = auto()
    STARTING = auto()
    START_FAILED = auto()
    RUNNING = auto()
    STOPPING = auto()
    COMPLETE = auto()
    FAILSAFE = auto()


class TeknofestMissionManager(Node):
    """Exactly one selected TEKNOFEST mission may command the vehicle."""

    STARTUP_SENSOR_TIMEOUT_SEC = 30.0
    BRIDGE_STATE_TIMEOUT_SEC = 3.0
    WAYPOINT_DOWNLOAD_MAX_AGE_SEC = 15.0
    VALID_COMMANDS = (1, 2, 3)

    def __init__(self):
        super().__init__("teknofest_mission_manager")

        self.mission_clients = create_mission_clients(self)
        requested_target_color = os.getenv(
            "TASK3_TARGET_COLOR",
            ACTIVE_TARGET_COLOR,
        ).strip().lower()
        if requested_target_color not in SUPPORTED_TARGET_COLORS:
            raise ValueError(
                "TASK3_TARGET_COLOR desteklenmiyor: "
                f"{requested_target_color!r}; "
                f"izin verilen={sorted(SUPPORTED_TARGET_COLORS)}"
            )
        self.task3_target_class = f"{requested_target_color}_buoy"
        self.mission_topics = create_mission_topics(
            self,
            gps_callback=self.gps_callback,
            heading_callback=self.heading_callback,
            state_callback=self.state_callback,
        )

        detection_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            String,
            "/vision/detections",
            self.vision_callback,
            detection_qos,
        )
        self.create_subscription(Imu, "/cube/imu", self.imu_callback, 10)
        self.create_subscription(
            Int32,
            "/mission_start",
            self.mission_start_callback,
            10,
        )
        self.mission_ack_pub = self.create_publisher(
            Int32,
            "/mission_start_ack",
            10,
        )
        self.active_task_pub = self.create_publisher(
            String,
            "/mission/active_task",
            10,
        )

        self.state = ManagerState.IDLE
        self.active_task_number = None
        self.task = None
        self.start_cancel_requested = False
        self.action_in_progress = False
        self.action_lock = threading.Lock()

        self.current_lat = None
        self.current_lon = None
        self.last_gps_time = None
        self.current_heading = None
        self.last_heading_time = None
        self.bridge_state_text = None
        self.bridge_connected = False
        self.bridge_armed = False
        self.bridge_mode = "UNKNOWN"
        self.last_bridge_state_time = None
        self.latest_imu = None
        self.last_imu_time = None
        self.latest_detections = []
        self.latest_frame_id = None
        self.vision_sequence = 0
        self.last_vision_time = None

        self.create_timer(0.1, self.control_timer_callback)
        self.create_timer(1.0, self.publish_active_task)
        self.get_logger().info(
            "TEKNOFEST mission manager hazir. SCR_USER2=1, 2 veya 3 bekleniyor; "
            "90=STOP, 99=ACIL STOP. "
            f"Task 3 hedefi={self.task3_target_class}."
        )

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------
    def gps_callback(self, msg):
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        status = getattr(getattr(msg, "status", None), "status", 0)
        if (
                status < 0
                or not math.isfinite(lat)
                or not math.isfinite(lon)
                or not (-90.0 <= lat <= 90.0)
                or not (-180.0 <= lon <= 180.0)
                or (abs(lat) < 1e-6 and abs(lon) < 1e-6)
        ):
            self.get_logger().warning(
                "Gecersiz GPS fix yok sayildi.",
                throttle_duration_sec=2.0,
            )
            return

        self.current_lat = lat
        self.current_lon = lon
        self.last_gps_time = time.monotonic()
        if self.task is not None:
            self.task.update_gps(lat, lon)

    def heading_callback(self, msg):
        heading = float(msg.data)
        if not math.isfinite(heading):
            self.get_logger().warning(
                "Gecersiz heading yok sayildi.",
                throttle_duration_sec=2.0,
            )
            return

        self.current_heading = heading % 360.0
        self.last_heading_time = time.monotonic()
        if self.task is not None:
            self.task.update_heading(self.current_heading)

    def state_callback(self, msg):
        parsed = parse_bridge_state(msg.data)
        if not {"connected", "armed", "mode"}.issubset(parsed):
            self.get_logger().warning(
                "Eksik /cube/state mesaji yok sayildi.",
                throttle_duration_sec=2.0,
            )
            return

        self.bridge_state_text = msg.data
        self.bridge_connected = parsed["connected"] is True
        self.bridge_armed = parsed["armed"] is True
        self.bridge_mode = str(parsed["mode"] or "UNKNOWN").strip().upper()
        self.last_bridge_state_time = time.monotonic()
        self._feed_bridge_state_to_task()

    def imu_callback(self, msg):
        values = (
            float(msg.angular_velocity.z),
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        )
        covariance = getattr(msg, "linear_acceleration_covariance", None)
        if (
                not all(math.isfinite(value) for value in values)
                or (
                    covariance is not None
                    and len(covariance) > 0
                    and covariance[0] < 0.0
                )
        ):
            self.get_logger().warning(
                "Gecersiz IMU paketi yok sayildi.",
                throttle_duration_sec=2.0,
            )
            return

        self.latest_imu = values
        self.last_imu_time = time.monotonic()
        if self.active_task_number == 3 and self.task is not None:
            self.task.update_imu(*values)

    @staticmethod
    def _extract_detections(payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            detections = payload.get("detections", payload.get("objects", []))
            return detections if isinstance(detections, list) else None
        return None

    def vision_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().warning(
                f"Vision JSON ayrıştırılamadi: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        detections = self._extract_detections(payload)
        if detections is None:
            self.get_logger().warning(
                "Vision detections bir liste degil; paket yok sayildi.",
                throttle_duration_sec=2.0,
            )
            return

        source_frame_id = payload.get("frame_id", payload.get("timestamp")) \
            if isinstance(payload, dict) else None
        if source_frame_id is None:
            self.vision_sequence += 1
            self.latest_frame_id = ("callback", self.vision_sequence)
        else:
            self.latest_frame_id = ("camera", str(source_frame_id))
        self.latest_detections = detections
        self.last_vision_time = time.monotonic()
        if self.active_task_number == 3 and self.task is not None:
            self.task.update_vision_timestamp()

    # ------------------------------------------------------------------
    # Mission command and lifecycle
    # ------------------------------------------------------------------
    def publish_active_task(self):
        if self.active_task_number not in self.VALID_COMMANDS:
            return
        msg = String()
        msg.data = f"task{self.active_task_number}"
        self.active_task_pub.publish(msg)

    def _publish_ack(self, command):
        msg = Int32()
        msg.data = int(command)
        self.mission_ack_pub.publish(msg)

    def mission_start_callback(self, msg):
        command = int(msg.data)
        if command == 0:
            if self.state == ManagerState.START_FAILED:
                self.state = ManagerState.IDLE
                self.get_logger().info(
                    "Basarisiz baslatma komutu Pixhawk tarafinda sifirlandi; "
                    "yeni bir gorev secimi bekleniyor."
                )
            return

        if command in self.VALID_COMMANDS:
            if (
                    command == self.active_task_number
                    and self.state == ManagerState.RUNNING
            ):
                self._publish_ack(command)
                return
            if self.state not in (ManagerState.IDLE,):
                self.get_logger().warning(
                    f"Task {command} reddedildi: mevcut durum={self.state.name}, "
                    f"aktif task={self.active_task_number}."
                )
                return
            self._launch_start(command)
            return

        if command in (90, 99):
            self.start_cancel_requested = True
            self._launch_stop(command)
            return

        if command == 4:
            self.get_logger().error(
                "SCR_USER2=4 bu TEKNOFEST paketinde tanimli degil; arac hareket "
                "ettirilmeden komut temizleniyor."
            )
            self._publish_ack(command)
            return

        self.get_logger().warning(f"Bilinmeyen mission komutu: {command}")

    def _launch_start(self, command):
        with self.action_lock:
            if self.action_in_progress:
                return
            self.action_in_progress = True
            self.start_cancel_requested = False
            self.active_task_number = int(command)
            self.task = None
            self.state = ManagerState.STARTING
            self.latest_detections = []
            self.latest_frame_id = None
            self.last_vision_time = None
            self.publish_active_task()
        threading.Thread(
            target=self._start_worker,
            args=(int(command),),
            daemon=True,
        ).start()

    def _launch_stop(self, command):
        with self.action_lock:
            if self.state == ManagerState.STOPPING:
                return
            self.state = ManagerState.STOPPING
        try:
            stop_vehicle(self.mission_topics.cmd_vel_pub, repeat_count=3)
            if self.active_task_number == 3 and self.task is not None:
                self.task.stop_mission("Mission Planner STOP")
        finally:
            threading.Thread(
                target=self._stop_worker,
                args=(int(command), "ACIL STOP" if command == 99 else "STOP"),
                daemon=True,
            ).start()

    def _start_worker(self, command):
        armed_during_start = False
        try:
            if not self._wait_for_services(self.STARTUP_SENSOR_TIMEOUT_SEC):
                raise RuntimeError("Bridge servisleri hazir olmadi.")
            ready, reason = self._wait_for_required_real_data(
                command,
                self.STARTUP_SENSOR_TIMEOUT_SEC,
            )
            if not ready:
                raise RuntimeError(reason)
            if self.start_cancel_requested:
                raise RuntimeError("Baslatma STOP komutuyla iptal edildi.")

            self._validate_fresh_waypoint_download(command)
            self.task = self._build_task(command)
            self._feed_cached_data_to_task()

            if not self._set_mode_and_wait(DRIVE_MODE):
                raise RuntimeError(f"{DRIVE_MODE} moduna gecilemedi.")
            if self.start_cancel_requested:
                raise RuntimeError("Baslatma ARM oncesinde iptal edildi.")
            if command == 3:
                arm_client = self.mission_clients.arm_client
                arm_label = "NORMAL ARM"
            else:
                arm_client = self.mission_clients.force_arm_client
                arm_label = "FORCE ARM"
            if not self._trigger_and_wait(arm_client, arm_label):
                raise RuntimeError(f"{arm_label} basarisiz.")
            armed_during_start = True
            if not self._wait_for_operational_state(8.0):
                raise RuntimeError("GUIDED + ARM heartbeat ile dogrulanamadi.")
            if self.start_cancel_requested:
                raise RuntimeError("Baslatma ARM sonrasinda iptal edildi.")

            if command == 3:
                ok, reason = self.task.start_mission()
                if not ok:
                    raise RuntimeError(reason)

            self.state = ManagerState.RUNNING
            self._publish_ack(command)
            self.get_logger().info(
                f"Task {command} gercek sensorlerle baslatildi; "
                f"{arm_label} kullaniliyor."
            )
        except Exception as exc:  # noqa: BLE001 - safety boundary
            self.get_logger().error(f"Task {command} baslatilamadi: {exc}")
            stop_vehicle(self.mission_topics.cmd_vel_pub, repeat_count=3)
            disarm_ok = True
            if armed_during_start or self.bridge_armed:
                disarm_ok = self._disarm_with_retries(
                    "BASLATMA IPTAL DISARM"
                )
            if disarm_ok:
                self.task = None
                self.active_task_number = None
                self.state = ManagerState.START_FAILED
            else:
                self.state = ManagerState.FAILSAFE
                self.get_logger().error(
                    "Baslatma iptal edildi fakat DISARM dogrulanamadi; "
                    "FAILSAFE kilidi korunuyor."
                )
            # State once kilitlenir, sonra ACK gonderilir. Bridge'in hemen
            # yayinlayacagi /mission_start=0 ancak START_FAILED durumunu acar.
            self._publish_ack(command)
        finally:
            with self.action_lock:
                self.action_in_progress = False

    def _stop_worker(self, command, label):
        disarm_ok = False
        try:
            disarm_ok = (
                self._wait_for_services(5.0)
                and self._disarm_with_retries(label)
            )
            if disarm_ok:
                self._publish_ack(command)
                self.get_logger().info(f"{label}: arac DISARM edildi.")
            else:
                self.get_logger().error(
                    f"{label}: DISARM dogrulanamadi; ACK gonderilmedi."
                )
        finally:
            if disarm_ok:
                self.task = None
                self.active_task_number = None
                self.state = ManagerState.IDLE
            else:
                self.state = ManagerState.FAILSAFE
            with self.action_lock:
                self.action_in_progress = False

    def _build_task(self, command):
        if command == 1:
            return Task1Maneuvering(
                self,
                self.mission_topics,
                self.mission_clients,
            )
        if command == 2:
            return Task2PointTrackingWithObstacleAvoidance(
                self,
                self.mission_topics,
                self.mission_clients,
            )
        return Task3KamikazeEngagement(
            self,
            self.mission_topics,
            self.mission_clients,
            target_class=self.task3_target_class,
        )

    def _validate_fresh_waypoint_download(self, command):
        if command == 3:
            return
        path = TASK1_WAYPOINT_PATH if command == 1 else TASK2_WAYPOINT_PATH
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Task {command} waypoint dosyasi bulunamadi: {path}"
            ) from exc
        age_sec = time.time() - stat.st_mtime
        if stat.st_size <= 0:
            raise RuntimeError(f"Task {command} waypoint dosyasi bos: {path}")
        if age_sec < -5.0 or age_sec > self.WAYPOINT_DOWNLOAD_MAX_AGE_SEC:
            raise RuntimeError(
                f"Task {command} waypoint dosyasi taze degil "
                f"(yas={age_sec:.1f}s, limit={self.WAYPOINT_DOWNLOAD_MAX_AGE_SEC:.0f}s). "
                "Gorevi ROS topic'inden elle baslatmayin; Mission Planner "
                "SCR_USER2 komutunu kullanin."
            )

    def _feed_bridge_state_to_task(self):
        if self.task is None or self.bridge_state_text is None:
            return
        if self.active_task_number == 3:
            self.task.update_bridge_state(self.bridge_state_text)
        else:
            self.task.update_bridge_state(
                self.bridge_connected,
                self.bridge_armed,
                self.bridge_mode,
            )

    def _feed_cached_data_to_task(self):
        if self.current_lat is not None and self.current_lon is not None:
            self.task.update_gps(self.current_lat, self.current_lon)
        if self.current_heading is not None:
            self.task.update_heading(self.current_heading)
        self._feed_bridge_state_to_task()
        if self.active_task_number == 3:
            if self.latest_imu is not None:
                self.task.update_imu(*self.latest_imu)
            if self.last_vision_time is not None:
                self.task.update_vision_timestamp()

    def _fresh(self, timestamp, maximum_age):
        return timestamp is not None and time.monotonic() - timestamp <= maximum_age

    def _required_data_ready(self, command):
        gps_timeout = (
            TASK3_GPS_TIMEOUT_SEC if command == 3 else TASK1_GPS_TIMEOUT_SEC
        )
        heading_timeout = (
            TASK3_HEADING_TIMEOUT_SEC
            if command == 3
            else TASK2_HEADING_TIMEOUT_SEC
        )
        if not self._fresh(self.last_gps_time, gps_timeout):
            return False, "Gecerli ve guncel GPS bekleniyor."
        if not self._fresh(self.last_heading_time, heading_timeout):
            return False, "Gecerli ve guncel heading bekleniyor."
        if (
                not self._fresh(
                    self.last_bridge_state_time,
                    self.BRIDGE_STATE_TIMEOUT_SEC,
                )
                or not self.bridge_connected
        ):
            return False, "Pixhawk/MAVLink bridge baglantisi bekleniyor."
        if not self._fresh(
                self.last_vision_time,
                VISION_TIMEOUT_SEC if command == 3 else DETECTION_STALE_SEC,
        ):
            return False, "Gercek kamera/vision akisi bekleniyor."
        if command == 3 and not self._fresh(self.last_imu_time, IMU_TIMEOUT_SEC):
            return False, "Task 3 icin gecerli IMU verisi bekleniyor."
        return True, "Hazir."

    def _wait_for_required_real_data(self, command, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        last_reason = "Sensorler bekleniyor."
        while rclpy.ok() and time.monotonic() < deadline:
            if self.start_cancel_requested:
                return False, "Baslatma STOP komutuyla iptal edildi."
            ready, last_reason = self._required_data_ready(command)
            if ready:
                return True, last_reason
            self.get_logger().info(last_reason, throttle_duration_sec=2.0)
            time.sleep(0.1)
        return False, f"Sensor hazirlik zaman asimi: {last_reason}"

    def _wait_for_services(self, timeout_sec):
        clients = (
            self.mission_clients.set_mode_client,
            self.mission_clients.arm_client,
            self.mission_clients.force_arm_client,
            self.mission_clients.disarm_client,
        )
        deadline = time.monotonic() + timeout_sec
        for client in clients:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not client.wait_for_service(timeout_sec=remaining):
                return False
        return True

    @staticmethod
    def _wait_future(future, timeout_sec):
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(timeout_sec):
            return None
        if future.exception() is not None:
            return None
        return future.result()

    def _set_mode_and_wait(self, mode_name, timeout_sec=5.0):
        request = SetMode.Request()
        request.base_mode = 0
        request.custom_mode = str(mode_name)
        response = self._wait_future(
            self.mission_clients.set_mode_client.call_async(request),
            timeout_sec,
        )
        return bool(response is not None and response.mode_sent)

    def _trigger_and_wait(self, client, label, timeout_sec=5.0):
        response = self._wait_future(
            client.call_async(Trigger.Request()),
            timeout_sec,
        )
        success = bool(response is not None and response.success)
        message = "<cevap yok>" if response is None else response.message
        log = self.get_logger().info if success else self.get_logger().error
        log(f"{label}: success={success}, message={message!r}")
        return success

    def _wait_for_operational_state(self, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if (
                    self.bridge_connected
                    and self.bridge_armed
                    and self.bridge_mode == DRIVE_MODE
                    and self._fresh(
                        self.last_bridge_state_time,
                        self.BRIDGE_STATE_TIMEOUT_SEC,
                    )
            ):
                return True
            time.sleep(0.1)
        return False

    def _disarm_with_retries(self, label, attempts=3):
        stop_vehicle(self.mission_topics.cmd_vel_pub, repeat_count=3)
        for attempt in range(1, attempts + 1):
            if self._trigger_and_wait(
                    self.mission_clients.disarm_client,
                    f"{label} ({attempt}/{attempts})",
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    def control_timer_callback(self):
        if self.state != ManagerState.RUNNING or self.task is None:
            return
        try:
            if not self._fresh(
                    self.last_vision_time,
                    VISION_TIMEOUT_SEC
                    if self.active_task_number == 3
                    else DETECTION_STALE_SEC,
            ):
                raise RuntimeError("Vision heartbeat zaman asimi.")

            if self.active_task_number == 3:
                self.task.update(
                    self.latest_detections,
                    frame_id=self.latest_frame_id,
                )
                failed = self.task.state == Task3State.FAILSAFE
                completed = self.task.state == Task3State.DONE
            elif self.active_task_number == 2:
                self.task.update(self.latest_detections)
                failed = self.task.state == Task2State.FAILSAFE
                completed = self.task.finished
            else:
                self.task.update(self.latest_detections)
                failed = self.task.state == Task1State.FAILSAFE
                completed = self.task.finished

            if failed:
                self._finish_active_task(success=False)
            elif completed:
                self._finish_active_task(success=True)
        except Exception as exc:  # noqa: BLE001 - motor safety boundary
            self.get_logger().error(f"Aktif gorev kontrol hatasi: {exc}")
            self._finish_active_task(success=False)

    def _finish_active_task(self, success):
        if self.state != ManagerState.RUNNING:
            return
        self.state = ManagerState.COMPLETE if success else ManagerState.FAILSAFE
        stop_vehicle(self.mission_topics.cmd_vel_pub, repeat_count=3)
        label = "GOREV SONU DISARM" if success else "FAILSAFE DISARM"
        threading.Thread(
            target=self._completion_worker,
            args=(label,),
            daemon=True,
        ).start()

    def _completion_worker(self, label):
        disarm_ok = self._disarm_with_retries(label)
        if disarm_ok:
            self.get_logger().info(f"{label} tamamlandi.")
            self.task = None
            self.active_task_number = None
            self.state = ManagerState.IDLE
        else:
            self.get_logger().error(
                f"{label} dogrulanamadi; FAILSAFE kilidi korunuyor. "
                "Mission Planner'dan DISARM uygulayin."
            )
            self.state = ManagerState.FAILSAFE

    def shutdown_and_disarm(self):
        self.start_cancel_requested = True
        stop_vehicle(self.mission_topics.cmd_vel_pub, repeat_count=3)
        if not self.bridge_armed:
            return True
        try:
            future = self.mission_clients.disarm_client.call_async(
                Trigger.Request()
            )
            # Ana rclpy.spin() artik durdugu icin servis cevabini bu thread'de
            # islemek gerekir; Event ile beklemek callback'i calistiramaz.
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            response = future.result() if future.done() else None
            success = bool(response is not None and response.success)
            if not success:
                message = "<cevap yok>" if response is None else response.message
                self.get_logger().error(
                    f"KAPANIS DISARM basarisiz: {message!r}"
                )
            return success
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"KAPANIS DISARM hatasi: {exc}")
            return False


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestMissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Mission manager kullanici tarafindan durduruldu.")
    finally:
        node.shutdown_and_disarm()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
