import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from mavros_msgs.srv import SetMode
from sensor_msgs.msg import Imu, NavSatFix, BatteryState
from std_msgs.msg import String, Float32
from std_srvs.srv import Trigger


# ============================================================
# TOPIC STRUCTURES
# ============================================================

@dataclass
class BridgeTopics:
    imu_pub: object
    gps_pub: object
    gps_heading_pub: object
    relative_alt_pub: object
    battery_pub: object
    state_pub: object
    error_pub: object
    cmd_vel_sub: object
    position_target_sub: object


@dataclass
class MissionTopics:
    gps_sub: object
    heading_sub: object
    state_sub: object
    cmd_vel_pub: object
    position_target_pub: object


def create_bridge_topics(node, cmd_vel_callback, set_position_callback=None):
    """
    orange_cube_bridge.py icin topic yapilarini olusturur.

    Bridge tarafinin gorevi:
        - MAVLink'ten gelen verileri ROS topic olarak yayinlamak
        - /cube/cmd_vel topic'ini dinleyip araca hareket komutu gondermek
        - /cube/set_position topic'ini dinleyip global position target gondermek

    Olusan topicler:
        Publishers:
            /cube/imu
            /cube/gps
            /cube/gps/heading
            /cube/gps/relative_altitude
            /cube/battery
            /cube/state
            /cube/error

        Subscriber:
            /cube/cmd_vel
            /cube/set_position
    """

    imu_pub = node.create_publisher(
        Imu,
        '/cube/imu',
        10
    )

    gps_pub = node.create_publisher(
        NavSatFix,
        '/cube/gps',
        10
    )

    gps_heading_pub = node.create_publisher(
        Float32,
        '/cube/gps/heading',
        10
    )

    relative_alt_pub = node.create_publisher(
        Float32,
        '/cube/gps/relative_altitude',
        10
    )

    battery_pub = node.create_publisher(
        BatteryState,
        '/cube/battery',
        10
    )

    state_pub = node.create_publisher(
        String,
        '/cube/state',
        10
    )

    error_pub = node.create_publisher(
        String,
        '/cube/error',
        10
    )

    cmd_vel_sub = node.create_subscription(
        Twist,
        '/cube/cmd_vel',
        cmd_vel_callback,
        10
    )

    position_target_sub = None
    if set_position_callback is not None:
        position_target_sub = node.create_subscription(
            NavSatFix,
            '/cube/set_position',
            set_position_callback,
            10
        )

    return BridgeTopics(
        imu_pub=imu_pub,
        gps_pub=gps_pub,
        gps_heading_pub=gps_heading_pub,
        relative_alt_pub=relative_alt_pub,
        battery_pub=battery_pub,
        state_pub=state_pub,
        error_pub=error_pub,
        cmd_vel_sub=cmd_vel_sub,
        position_target_sub=position_target_sub,
    )


def create_mission_topics(node, gps_callback, heading_callback, state_callback):
    """
    mission_node.py veya diger gorev node'lari icin topic yapilarini olusturur.

    Mission tarafinin gorevi:
        - GPS bilgisini dinlemek
        - Heading bilgisini dinlemek
        - Bridge state bilgisini dinlemek
        - /cube/cmd_vel uzerinden hareket komutu yayinlamak
        - /cube/set_position uzerinden GPS hedefi yayinlamak

    Olusan topicler:
        Subscribers:
            /cube/gps
            /cube/gps/heading
            /cube/state

        Publisher:
            /cube/cmd_vel
            /cube/set_position
    """

    gps_sub = node.create_subscription(
        NavSatFix,
        '/cube/gps',
        gps_callback,
        10
    )

    heading_sub = node.create_subscription(
        Float32,
        '/cube/gps/heading',
        heading_callback,
        10
    )

    state_sub = node.create_subscription(
        String,
        '/cube/state',
        state_callback,
        10
    )

    cmd_vel_pub = node.create_publisher(
        Twist,
        '/cube/cmd_vel',
        10
    )

    position_target_pub = node.create_publisher(
        NavSatFix,
        '/cube/set_position',
        10
    )

    return MissionTopics(
        gps_sub=gps_sub,
        heading_sub=heading_sub,
        state_sub=state_sub,
        cmd_vel_pub=cmd_vel_pub,
        position_target_pub=position_target_pub,
    )


# ============================================================
# SERVICE STRUCTURES
# ============================================================

@dataclass
class BridgeServices:
    set_mode_srv: object
    arm_srv: object
    force_arm_srv: object
    disarm_srv: object


@dataclass
class MissionServiceClients:
    set_mode_client: object
    arm_client: object
    force_arm_client: object
    disarm_client: object


def create_bridge_services(node, set_mode_callback, arm_callback, force_arm_callback, disarm_callback):
    """
    orange_cube_bridge.py icin service server yapilarini olusturur.

    Bridge tarafinda olusan servisler:
        /cube/set_mode_service
        /cube/arm
        /cube/force_arm
        /cube/disarm

    Bu servisleri mission_node veya diger gorev node'lari cagirir.
    Bridge bu istekleri MAVLink komutuna cevirir.
    """

    set_mode_srv = node.create_service(
        SetMode,
        '/cube/set_mode_service',
        set_mode_callback
    )

    arm_srv = node.create_service(
        Trigger,
        '/cube/arm',
        arm_callback
    )

    force_arm_srv = node.create_service(
        Trigger,
        '/cube/force_arm',
        force_arm_callback
    )

    disarm_srv = node.create_service(
        Trigger,
        '/cube/disarm',
        disarm_callback
    )

    return BridgeServices(
        set_mode_srv=set_mode_srv,
        arm_srv=arm_srv,
        force_arm_srv=force_arm_srv,
        disarm_srv=disarm_srv,
    )


def create_mission_clients(node):
    """
    mission_node.py veya diger gorev node'lari icin service client yapilarini olusturur.

    Mission tarafinin cagiracagi servisler:
        /cube/set_mode_service
        /cube/arm
        /cube/force_arm
        /cube/disarm
    """

    set_mode_client = node.create_client(
        SetMode,
        '/cube/set_mode_service'
    )

    arm_client = node.create_client(
        Trigger,
        '/cube/arm'
    )

    force_arm_client = node.create_client(
        Trigger,
        '/cube/force_arm'
    )

    disarm_client = node.create_client(
        Trigger,
        '/cube/disarm'
    )

    return MissionServiceClients(
        set_mode_client=set_mode_client,
        arm_client=arm_client,
        force_arm_client=force_arm_client,
        disarm_client=disarm_client,
    )


# ============================================================
# SERVICE HELPER FUNCTIONS FOR MISSION NODES
# ============================================================

def wait_for_mission_services(node, clients):
    """
    Mission node baslamadan once bridge servislerini bekler.
    """

    node.get_logger().info('Servisler bekleniyor...')

    while not clients.set_mode_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().warn('/cube/set_mode_service bekleniyor...')

    while not clients.arm_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().warn('/cube/arm bekleniyor...')

    while not clients.force_arm_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().warn('/cube/force_arm bekleniyor...')

    while not clients.disarm_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().warn('/cube/disarm bekleniyor...')

    node.get_logger().info('Servisler hazir.')


def call_set_mode(node, set_mode_client, mode_name, timeout_sec=5.0):
    """
    Mission node icinden mod degistirme servisini cagirir.

    Ornek:
        call_set_mode(self, self.set_mode_client, 'GUIDED')
        call_set_mode(self, self.set_mode_client, 'HOLD')
    """

    req = SetMode.Request()
    req.base_mode = 0
    req.custom_mode = str(mode_name)

    future = set_mode_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)

    if not future.done():
        node.get_logger().error(f'Mod degistirme zaman asimi: {mode_name}')
        return False

    res = future.result()

    if res is not None and res.mode_sent:
        node.get_logger().info(f'Mod degistirildi: {mode_name}')
        return True

    node.get_logger().error(f'Mod degistirilemedi: {mode_name}')
    return False


def call_trigger_service(node, client, name, timeout_sec=5.0):
    """
    ARM / DISARM gibi Trigger servislerini cagirir.

    Ornek:
        call_trigger_service(self, self.arm_client, 'ARM')
        call_trigger_service(self, self.disarm_client, 'DISARM')
    """

    req = Trigger.Request()

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)

    if not future.done():
        node.get_logger().error(f'{name} servis zaman asimi.')
        return False

    res = future.result()

    if res is None:
        node.get_logger().error(f'{name} cevabi gelmedi.')
        return False

    node.get_logger().info(
        f'{name}: success={res.success}, message={res.message}'
    )

    return bool(res.success)


def call_force_arm(node, clients, timeout_sec=5.0):
    """
    Mission node icinden force arm servisini cagirir.

    Bridge tarafindaki force_arm_callback MAVLink'te
    MAV_CMD_COMPONENT_ARM_DISARM komutunu param1=1 ve param2=21196
    ile gondermelidir.
    """

    return call_trigger_service(
        node,
        clients.force_arm_client,
        'FORCE ARM',
        timeout_sec=timeout_sec
    )


# ============================================================
# COMMON CMD_VEL HELPER
# ============================================================

def publish_cmd_vel(cmd_vel_pub, linear_x, angular_z):
    """
    /cube/cmd_vel topic'ine hareket komutu yayinlar.

    linear_x:
        ileri/geri hiz komutu
        Açıklama: Pozitif degerler ileri, negatif degerler geri hareket anlamina gelir.

    angular_z:
        sag/sol donus komutu
        Açıklama: Pozitif degerler saat yönünün tersine (sola), negatif degerler saat yönünde (sağa) donus anlamina gelir.
    """

    msg = Twist()
    msg.linear.x = float(linear_x)
    msg.angular.z = float(angular_z)

    cmd_vel_pub.publish(msg)


def publish_set_position(position_target_pub, lat, lon, altitude=20.0):
    """
    /cube/set_position topic'ine GPS hedefi yayinlar.
    Bridge bu hedefi MAVLink SET_POSITION_TARGET_GLOBAL_INT komutuna cevirir.
    """

    msg = NavSatFix()
    msg.latitude = float(lat)
    msg.longitude = float(lon)
    msg.altitude = float(altitude)

    position_target_pub.publish(msg)


def stop_vehicle(cmd_vel_pub, repeat_count=10):
    """
    Araci durdurmak icin birkac kez sifir cmd_vel yayinlar.
    """

    for _ in range(repeat_count):
        publish_cmd_vel(
            cmd_vel_pub,
            0.0,
            0.0
        )


def calculate_gps_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Dünya yarıçapı (metre)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)

    y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))

    bearing_rad = math.atan2(y, x)

    bearing_deg = (math.degrees(bearing_rad) + 360) % 360

    return bearing_deg
