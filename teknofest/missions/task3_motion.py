"""Task 3 boyunca kullanılan ortak skid-steer dönüş komutu."""

from utils.mavlink_utilities import publish_cmd_vel


# Bu ikili sahadaki Pixhawk skid-steer mikseri için arama aşamasında
# doğrulanan komuttur. Her aşamanın farklı bir "saf yaw" davranışı
# kullanması, aynı araçta farklı motor sonuçları üretmemelidir.
TASK3_TURN_BASE_THRUST = 0.18
TASK3_MAX_YAW_OFFSET_RAD = 0.18


def publish_task3_turn(cmd_vel_pub, angular_z):
    """Bütün Task 3 aşamalarında aynı dönüş sözleşmesini uygula.

    Motorların gerçek sol/sağ karışımı Pixhawk tarafından yapılır. Bu
    fonksiyon doğrudan motor çıkışı üretmez; sahada doğrulanan taban itki ve
    yaw-offset sınırlarını tek yerde tutar.
    """

    limited_angular = max(
        -TASK3_MAX_YAW_OFFSET_RAD,
        min(TASK3_MAX_YAW_OFFSET_RAD, float(angular_z)),
    )
    publish_cmd_vel(
        cmd_vel_pub,
        linear_x=TASK3_TURN_BASE_THRUST,
        angular_z=limited_angular,
    )
    return TASK3_TURN_BASE_THRUST, limited_angular
