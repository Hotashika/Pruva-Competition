"""Task 3 boyunca kullanilan ortak skid-steer donus komutu."""

try:
    from utils.mavlink_utilities import (
        DEFAULT_SKID_STEER_MAX_YAW_OFFSET,
        DEFAULT_SKID_STEER_TURN_THRUST,
        publish_skid_steer_turn,
    )
except ImportError:
    # Sensor-flow birim testleri mavlink_utilities icin kucuk bir fake modul
    # kurar. Uretim kodunda her zaman yukaridaki ortak yardimci kullanilir.
    from utils.mavlink_utilities import publish_cmd_vel

    DEFAULT_SKID_STEER_TURN_THRUST = 0.18
    DEFAULT_SKID_STEER_MAX_YAW_OFFSET = 0.18

    def publish_skid_steer_turn(
            cmd_vel_pub,
            angular_z,
            base_thrust=DEFAULT_SKID_STEER_TURN_THRUST,
            max_yaw_offset=DEFAULT_SKID_STEER_MAX_YAW_OFFSET,
    ):
        limit = abs(float(max_yaw_offset))
        limited = max(-limit, min(limit, float(angular_z)))
        publish_cmd_vel(cmd_vel_pub, float(base_thrust), limited)
        return float(base_thrust), limited


# Bu degerler sahadaki Pixhawk skid-steer mikseri icin dogrulanan komuttur.
# Tum gorevlerin ayni arac donus sozlesmesini kullanmasi gerekir.
TASK3_TURN_BASE_THRUST = DEFAULT_SKID_STEER_TURN_THRUST
TASK3_MAX_YAW_OFFSET_RAD = DEFAULT_SKID_STEER_MAX_YAW_OFFSET


def publish_task3_turn(cmd_vel_pub, angular_z):
    """Task 3 asamalarinda ortak, sinirli donus komutunu uygula."""

    return publish_skid_steer_turn(
        cmd_vel_pub,
        angular_z=angular_z,
        base_thrust=TASK3_TURN_BASE_THRUST,
        max_yaw_offset=TASK3_MAX_YAW_OFFSET_RAD,
    )
