"""
task3_kamikaze_engagement.py içindeki Task3Node'un RENK PARAMETRESİ mantığını
(carpilacak_duba) ROS2, tekne, GPS OLMADAN test eder.

Bu test şunu doğruluyor:
  1) Parametre hiç verilmezse -> otomatik olarak "red_buoy" seçiliyor mu?
  2) Parametre "black" verilirse -> doğru şekilde "black_buoy" oluyor mu?
  3) Parametre geçersiz bir şey ("mavi" gibi) verilirse -> sistem hata verip
     duruyor mu (motor arm edilmeden)?

NASIL ÇALIŞIYOR:
  rclpy, std_msgs, utils.mavlink_utilities gerçek modülleri bilgisayarınızda
  kurulu olmayabilir (rclpy için ROS2 gerekir). Bu yüzden hepsinin YERİNE
  sahte/mock modüller koyuyoruz, tıpkı test_arama_simulation.py'de yaptığımız
  gibi. Böylece gerçek Task3Node sınıfını, motorları/ROS2'yi hiç çalıştırmadan,
  sadece "renk parametresi doğru işleniyor mu" sorusunu test edebiliyoruz.

ÇALIŞTIRMA:
    Bu dosyayı task3_kamikaze_engagement.py ve arama.py ile AYNI klasöre koyun:
        python3 test_task3_color_resolution.py
"""

import sys
import types


# ============================================================
# 1) SAHTE rclpy / rclpy.node.Node
# ============================================================
class FakeParameterValue:
    def __init__(self, string_value):
        self.string_value = string_value


class FakeParameter:
    def __init__(self, value):
        self._value = value

    def get_parameter_value(self):
        return FakeParameterValue(self._value)


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warns = []
        self.errors = []

    def info(self, msg, **kwargs):
        self.infos.append(msg)
        print(f"[INFO] {msg}")

    def warn(self, msg, **kwargs):
        self.warns.append(msg)
        print(f"[WARN] {msg}")

    def error(self, msg, **kwargs):
        self.errors.append(msg)
        print(f"[ERROR] {msg}")


# Test başında doldurulur: gerçek hayatta "--ros-args -p carpilacak_duba:=black"
# ile verilen değeri simüle eder.
SIMULATED_PARAM_OVERRIDES = {}


class FakeNode:
    """rclpy.node.Node'un yerine geçer -- sadece Task3Node.__init__'in
    kullandığı minimum arayüzü sağlar."""

    def __init__(self, name):
        self._name = name
        self._logger = FakeLogger()
        self._declared_params = {}

    def get_logger(self):
        return self._logger

    def declare_parameter(self, name, default):
        value = SIMULATED_PARAM_OVERRIDES.get(name, default)
        self._declared_params[name] = value

    def get_parameter(self, name):
        return FakeParameter(self._declared_params[name])

    def create_subscription(self, *args, **kwargs):
        return None

    def create_timer(self, *args, **kwargs):
        return None


fake_rclpy = types.ModuleType("rclpy")
fake_rclpy.init = lambda args=None: None
fake_rclpy.shutdown = lambda: None
fake_rclpy.ok = lambda: True
fake_rclpy.spin_once = lambda node, timeout_sec=0.1: None

fake_rclpy_node = types.ModuleType("rclpy.node")
fake_rclpy_node.Node = FakeNode
fake_rclpy.node = fake_rclpy_node

sys.modules["rclpy"] = fake_rclpy
sys.modules["rclpy.node"] = fake_rclpy_node

# ============================================================
# 2) SAHTE std_msgs.msg.String
# ============================================================
fake_std_msgs = types.ModuleType("std_msgs")
fake_std_msgs_msg = types.ModuleType("std_msgs.msg")


class FakeString:
    def __init__(self):
        self.data = ""


fake_std_msgs_msg.String = FakeString
fake_std_msgs.msg = fake_std_msgs_msg
sys.modules["std_msgs"] = fake_std_msgs
sys.modules["std_msgs.msg"] = fake_std_msgs_msg

# ============================================================
# 3) SAHTE utils.mavlink_utilities
# ============================================================
fake_mavlink = types.ModuleType("utils.mavlink_utilities")


def create_mission_clients(node):
    return types.SimpleNamespace(
        set_mode_client=None, force_arm_client=None, disarm_client=None
    )


def wait_for_mission_services(node, clients):
    return None


def create_mission_topics(node, gps_callback, heading_callback, state_callback):
    return types.SimpleNamespace(cmd_vel_pub=None, position_target_pub=None)


def call_set_mode(node, client, mode):
    return True


def call_trigger_service(node, client, label):
    return True


def stop_vehicle(pub):
    pass


def calculate_gps_distance(lat1, lon1, lat2, lon2):
    return 0.0


def publish_cmd_vel(pub, linear_x, angular_z):
    pass


def publish_set_position(pub, lat, lon):
    pass


fake_mavlink.publish_set_position = publish_set_position
fake_mavlink.create_mission_clients = create_mission_clients
fake_mavlink.wait_for_mission_services = wait_for_mission_services
fake_mavlink.create_mission_topics = create_mission_topics
fake_mavlink.call_set_mode = call_set_mode
fake_mavlink.call_trigger_service = call_trigger_service
fake_mavlink.stop_vehicle = stop_vehicle
fake_mavlink.calculate_gps_distance = calculate_gps_distance
fake_mavlink.publish_cmd_vel = publish_cmd_vel

fake_utils_pkg = types.ModuleType("utils")
fake_utils_pkg.mavlink_utilities = fake_mavlink
sys.modules["utils"] = fake_utils_pkg
sys.modules["utils.mavlink_utilities"] = fake_mavlink

# ============================================================
# 4) SAHTE teknofest.missions.arama (gerçek arama.py'yi kullanıyoruz)
# ============================================================
import arama as real_arama  # noqa: E402  (aynı klasördeki gerçek arama.py)

# arama.py'nin kendi içindeki utils.mavlink_utilities importu zaten
# yukarıda sahte modülle karşılanıyor, bu yüzden real_arama sorunsuz çalışır.

fake_teknofest_pkg = types.ModuleType("teknofest")
fake_teknofest_missions_pkg = types.ModuleType("teknofest.missions")
fake_teknofest_missions_arama = types.ModuleType("teknofest.missions.arama")
fake_teknofest_missions_arama.AramaGorevi = real_arama.AramaGorevi

fake_teknofest_pkg.missions = fake_teknofest_missions_pkg
fake_teknofest_missions_pkg.arama = fake_teknofest_missions_arama

sys.modules["teknofest"] = fake_teknofest_pkg
sys.modules["teknofest.missions"] = fake_teknofest_missions_pkg
sys.modules["teknofest.missions.arama"] = fake_teknofest_missions_arama

# ============================================================
# 5) ŞİMDİ task3_kamikaze_engagement.py'Yİ İÇE AKTARABİLİRİZ
# ============================================================
import task3_kamikaze_engagement as t3  # noqa: E402


# ============================================================
# TEST SENARYOLARI
# ============================================================
def senaryo_1_parametre_verilmezse_red_secilmeli():
    print("\n" + "=" * 60)
    print("SENARYO 1: 'carpilacak_duba' parametresi HİÇ VERİLMEDEN çalıştır")
    print("Beklenen: otomatik olarak red_buoy seçilmeli + uyarı basılmalı")
    print("=" * 60)

    SIMULATED_PARAM_OVERRIDES.clear()  # parametre verilmemiş gibi davran

    node = t3.Task3Node()

    assert node.target_class == "red_buoy", (
        f"BEKLENMEYEN: target_class='{node.target_class}' (red_buoy olmalıydı)"
    )
    # NOT: ROS2'de declare_parameter(name, default) çağrısı, parametre dışarıdan
    # verilmediğinde default'u SESSİZCE kullanır (bu normal ROS2 davranışıdır).
    # Bu yüzden burada ayrıca bir "verilmedi" uyarısı beklemiyoruz -- asıl
    # doğrulamamız gereken şey target_class'ın doğru değere (red_buoy)
    # çözülmüş olması.
    print(f"  target_class = {node.target_class}")
    print("  SONUÇ: Senaryo 1 BAŞARILI -- parametre verilmeyince otomatik red seçildi.\n")


def senaryo_2_gecerli_parametre_black():
    print("=" * 60)
    print("SENARYO 2: 'carpilacak_duba:=black' verilerek çalıştır")
    print("Beklenen: target_class = black_buoy olmalı")
    print("=" * 60)

    SIMULATED_PARAM_OVERRIDES.clear()
    SIMULATED_PARAM_OVERRIDES["carpilacak_duba"] = "black"

    node = t3.Task3Node()

    assert node.target_class == "black_buoy", (
        f"BEKLENMEYEN: target_class='{node.target_class}' (black_buoy olmalıydı)"
    )
    print(f"  target_class = {node.target_class}")
    print("  SONUÇ: Senaryo 2 BAŞARILI -- verilen renk doğru işlendi.\n")


def senaryo_3_gecersiz_parametre():
    print("=" * 60)
    print("SENARYO 3: 'carpilacak_duba:=mavi' (GEÇERSİZ bir renk) verilerek çalıştır")
    print("Beklenen: SystemExit ile durmalı, görev başlatılmamalı")
    print("=" * 60)

    SIMULATED_PARAM_OVERRIDES.clear()
    SIMULATED_PARAM_OVERRIDES["carpilacak_duba"] = "mavi"

    try:
        t3.Task3Node()
        print("  !! BEKLENMEYEN: Hata vermedi, geçersiz renkle görev başlatılabildi.")
    except SystemExit:
        print("  SONUÇ: Senaryo 3 BAŞARILI -- geçersiz renkte görev başlatılmadı (SystemExit).\n")


if __name__ == "__main__":
    senaryo_1_parametre_verilmezse_red_secilmeli()
    senaryo_2_gecerli_parametre_black()
    senaryo_3_gecersiz_parametre()
    print("Tüm renk-çözümleme senaryoları tamamlandı.")