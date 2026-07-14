"""
carpma.py'yi tekne/ROS2/IMU donanimi olmadan test etmek icin masaustu simulasyonu.

Kullanim:
    1) Bu dosyayi carpma.py ile AYNI klasore koyun (teknofest/missions/).
    2) Terminalde o klasore gidip calistirin:
         python test_carpma_simulation.py
"""

import math
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ------------------------------------------------------------------
# KRITIK: carpma.py, BACKING_OFF/COOLDOWN/TIMEOUT surelerini GERCEK saatle
# (time.monotonic()) olcuyor. Bu test ise zamani "t += 0.1" ile simule
# ediyor ve gercekte neredeyse hic beklemeden (milisaniyeler icinde) tum
# donguyu geciyor -- yani gercek zaman asla 2 saniyeye ulasmiyor ve kod
# sonsuza kadar BACKING_OFF'ta takili kaliyor gibi GORUNUYOR (aslinda kod
# dogru calisiyor, sadece test onu gercek zamanda hicbir zaman ilerletmiyor).
#
# Cozum: time.monotonic() fonksiyonunu, bizim kontrol ettigimiz simule
# edilmis saatimizi donduren sahte bir fonksiyonla degistiriyoruz. Boylece
# carpma.py icindeki TUM sure kontrolleri (BACKOFF_DURATION_SEC, COOLDOWN_SEC,
# PER_ATTEMPT_TIMEOUT_SEC, TOTAL_CARPMA_TIMEOUT_SEC) simule ettigimiz zamana
# gore dogru sekilde tetiklenir.
# ------------------------------------------------------------------
_sim_clock = [0.0]


def _fake_monotonic():
    return _sim_clock[0]


time.monotonic = _fake_monotonic

# ------------------------------------------------------------------
# utils.mavlink_utilities'i sahte fonksiyonlarla olustur
# ------------------------------------------------------------------
fake_utils_pkg = types.ModuleType("utils")
fake_mavlink = types.ModuleType("utils.mavlink_utilities")


def publish_cmd_vel(pub, linear_x, angular_z):
    print(f"    cmd_vel  -> linear_x={linear_x:+.2f}  angular_z={angular_z:+.2f}")


def stop_vehicle(pub):
    print("    STOP")


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def calculate_gps_distance(lat1, lon1, lat2, lon2):
    R = 6378137.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


fake_mavlink.publish_cmd_vel = publish_cmd_vel
fake_mavlink.stop_vehicle = stop_vehicle
fake_mavlink.calculate_bearing = calculate_bearing
fake_mavlink.calculate_gps_distance = calculate_gps_distance

sys.modules["utils"] = fake_utils_pkg
sys.modules["utils.mavlink_utilities"] = fake_mavlink

# ------------------------------------------------------------------
from carpma import CarpmaGorevi, IMPACT_CONSECUTIVE_SAMPLES, TOTAL_CARPMA_TIMEOUT_SEC  # noqa: E402


class FakeLogger:
    def info(self, msg, **kwargs): print(f"[INFO] {msg}")
    def warn(self, msg, **kwargs): print(f"[WARN] {msg}")
    def error(self, msg, **kwargs): print(f"[ERROR] {msg}")


class FakeNode:
    def get_logger(self):
        return FakeLogger()


class FakeTopics:
    cmd_vel_pub = None


BASELINE_ACCEL = 9.81  # sakin sudaki tipik ivme buyuklugu (yerçekimi)
NOISE_AMPLITUDE = 0.3  # normal titresim/dalga gurultusu


def run_scenario(name, hits_at_seconds, inject_single_noise_spike=False, max_time=60.0):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    _sim_clock[0] = 0.0  # her senaryo t=0'dan baslasin diye saati sifirla

    node = FakeNode()
    topics = FakeTopics()
    carpma = CarpmaGorevi(node, topics, target_class="red_buoy")

    lat, lon = 41.000000, 29.000000
    heading = 90.0
    dt = 0.1
    t = 0.0
    hits_remaining = list(hits_at_seconds)

    # Ilk birkac saniye baseline IMU dolsun diye normal (gurultulu) veri besliyoruz
    while t < max_time:
        _sim_clock[0] = t  # carpma.py'nin gordugu "saat" bu satirla ilerliyor
        carpma.update_gps(lat, lon, heading)

        detections = [{
            "class": "red_buoy",
            "confidence": 0.9,
            "distance": 2.0,
            "bbox": [300, 200, 340, 240],
            "track_id": 1,
            "Buoy angle: ": 2.0,
            "Buoy side: ": "across",
        }]

        finished = carpma.update(detections)

        # IMU besleme: normal zamanlarda gurultu, planlanan aninda spike
        is_impact_moment = hits_remaining and abs(t - hits_remaining[0]) < 0.05
        if is_impact_moment:
            for _ in range(IMPACT_CONSECUTIVE_SAMPLES):
                carpma.update_imu(BASELINE_ACCEL + 12.0, 0.0, 0.0)
            hits_remaining.pop(0)
        elif inject_single_noise_spike and abs(t - 5.0) < 0.05:
            # tek ornekli gurultu sicramasi -> YOK SAYILMALI (art arda degil)
            carpma.update_imu(BASELINE_ACCEL + 12.0, 0.0, 0.0)
        else:
            noise = NOISE_AMPLITUDE * math.sin(t * 7.0)
            carpma.update_imu(BASELINE_ACCEL + noise, 0.0, 0.0)

        if finished:
            print(f"\n>>> BITTI: t={t:.1f}s  hit_count={carpma.hit_count}  "
                  f"success={carpma.success}  state={carpma.state.name}")
            return

        t += dt

    print(f"\n>>> {max_time:.0f}sn icinde tamamlanmadi (test suresi yetersiz olabilir). "
          f"hit_count={carpma.hit_count}")


if __name__ == "__main__":
    # Senaryo 1: 3 carpis da zamaninda gerceklesiyor -> BASARILI beklenir
    run_scenario(
        "SENARYO 1: Normal 3 carpis (10s, 20s, 30s)",
        hits_at_seconds=[10.0, 20.0, 30.0],
    )

    # Senaryo 2: hic carpma olmuyor -> TOTAL_CARPMA_TIMEOUT_SEC sonra ISKALANDI beklenir
    run_scenario(
        "SENARYO 2: Hic temas yok -> zaman asimi (ISKALANDI beklenir)",
        hits_at_seconds=[],
        max_time=TOTAL_CARPMA_TIMEOUT_SEC + 5.0,
    )

    # Senaryo 3: tek ornekli gurultu sicramasi + gercek 3 carpis
    #            -> gurultu sicramasi hit saymamali, sadece gercek 3 tanesi sayilmali
    run_scenario(
        "SENARYO 3: Tek ornekli gurultu sicramasi (5s) yanlislikla sayilmamali",
        hits_at_seconds=[10.0, 20.0, 30.0],
        inject_single_noise_spike=True,
    )