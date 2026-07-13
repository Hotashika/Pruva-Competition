"""
arama.py (AramaGorevi) için ROS2'siz, teknesiz, GPS'siz masaüstü simülasyonu.

Bu dosya, gerçek utils.mavlink_utilities modülünü (rclpy'ye bağımlı olduğu için)
İÇE AKTARMAZ. Onun yerine sys.modules içine SAHTE bir "utils.mavlink_utilities"
modülü yerleştirir. arama.py bu sahte modülü gerçek modülmüş gibi kullanır.

ÖNEMLİ GÜNCELLEME:
  - Şimdilik ARANACAK/ÇARPILACAK DUBA = KIRMIZI (red_buoy) olarak test ediliyor
    (yarışma günü bu, task3_kamikaze_engagement.py'deki 'carpilacak_duba'
    parametresiyle değişecek, bu test dosyasında bir şey değiştirmeye gerek yok).
  - Gerçek saat (time.sleep) yerine SAHTE/HIZLANDIRILMIŞ bir saat kullanılıyor.
    Önceki versiyonda STATION_TIMEOUT_SEC=18sn gerçek zamanda bekleniyordu,
    bu yüzden Senaryo 2 gerçekten ~45 saniye sürüyordu. Şimdi zamanı kendimiz
    ilerletiyoruz, test birkaç saniyede bitiyor.
  - Konsol spam'i azaltıldı: aynı "stop_vehicle -> DUR" satırı art arda
    onlarca kez basılmıyor, sadece durum değiştiğinde özet basılıyor.

ÇALIŞTIRMA:
    Bu dosyayı arama.py ile AYNI klasöre koyun, sonra:
        python3 test_arama_simulation.py
"""

import math
import sys
import types

# ============================================================
# 1) SAHTE SAAT (arama.py içindeki time.monotonic() çağrılarını devralır)
# ============================================================
class FakeClock:
    def __init__(self):
        self._now = 0.0

    def monotonic(self):
        return self._now

    def advance(self, dt):
        self._now += dt


fake_clock = FakeClock()

fake_time_module = types.ModuleType("time")
fake_time_module.monotonic = fake_clock.monotonic
fake_time_module.sleep = lambda s: None  # gerçekte hiç bekleme

# ============================================================
# 2) SAHTE utils.mavlink_utilities MODÜLÜ
# ============================================================
fake_mavlink = types.ModuleType("utils.mavlink_utilities")

_last_action = {"text": None, "count": 0}


def _log_action(text):
    """Aynı komut art arda tekrar ediyorsa spam yapmadan sayaç gösterir."""
    if text == _last_action["text"]:
        _last_action["count"] += 1
        return
    if _last_action["text"] is not None and _last_action["count"] > 1:
        print(f"    ... (yukarıdaki satır {_last_action['count']} kez tekrarlandı)")
    print(text)
    _last_action["text"] = text
    _last_action["count"] = 1


def publish_cmd_vel(pub, linear_x, angular_z):
    _log_action(f"    cmd_vel -> linear_x={linear_x:+.2f}  angular_z={angular_z:+.2f}")


def stop_vehicle(pub):
    _log_action("    stop_vehicle -> DUR")


def publish_set_position(pub, lat, lon):
    _log_action(f"    set_pos -> git: {lat:.6f}, {lon:.6f}")


def calculate_gps_distance(lat1, lon1, lat2, lon2):
    """Basit haversine mesafe hesabı (metre)."""
    R = 6378137.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


fake_mavlink.publish_cmd_vel = publish_cmd_vel
fake_mavlink.stop_vehicle = stop_vehicle
fake_mavlink.publish_set_position = publish_set_position
fake_mavlink.calculate_gps_distance = calculate_gps_distance

fake_utils_pkg = types.ModuleType("utils")
fake_utils_pkg.mavlink_utilities = fake_mavlink
sys.modules["utils"] = fake_utils_pkg
sys.modules["utils.mavlink_utilities"] = fake_mavlink

# ============================================================
# 3) arama.py'Yİ İÇE AKTAR, SONRA İÇİNDEKİ time MODÜLÜNÜ SAHTESİYLE DEĞİŞTİR
# ============================================================
import arama  # noqa: E402

arama.time = fake_time_module  # arama.py artık gerçek saat yerine bizim saatimizi kullanıyor

from arama import AramaGorevi  # noqa: E402


# ============================================================
# 4) SAHTE NODE VE TOPICS
# ============================================================
class FakeLogger:
    def info(self, msg, **kwargs):
        print(f"[INFO] {msg}")

    def warn(self, msg, **kwargs):
        print(f"[WARN] {msg}")

    def error(self, msg, **kwargs):
        print(f"[ERROR] {msg}")


class FakeNode:
    def get_logger(self):
        return FakeLogger()


class FakeTopics:
    cmd_vel_pub = "cmd_vel_pub"
    position_target_pub = "position_target_pub"


def advance_heading_towards(current, speed_deg_per_tick):
    return (current + speed_deg_per_tick) % 360.0


TICK_DT = 0.1  # gerçek timer_callback'teki 0.1 sn'ye karşılık gelir (SAHTE zamanda ilerler)


# ============================================================
# SENARYO 1: Kırmızı duba bir süre sonra beliriyor
# ============================================================
def senaryo_1_duba_beliriyor():
    print("\n" + "=" * 60)
    print("SENARYO 1: Kırmızı duba (red_buoy) bir süre sonra beliriyor")
    print("=" * 60)

    fake_clock._now = 0.0
    _last_action["text"] = None

    node = FakeNode()
    topics = FakeTopics()

    # ŞİMDİLİK: aranacak/çarpılacak duba = kırmızı
    arama_gorevi = AramaGorevi(node, topics, target_class="red_buoy")

    lat, lon = 41.015137, 28.979530
    heading = 0.0
    arama_gorevi.update_gps(lat, lon, heading)

    duba_belirdi = False
    max_ticks = int(40.0 / TICK_DT)

    for tick in range(max_ticks):
        fake_clock.advance(TICK_DT)
        heading = advance_heading_towards(heading, 4.0)
        arama_gorevi.update_gps(lat, lon, heading)

        elapsed = fake_clock.monotonic()
        if elapsed > 3.0 and not duba_belirdi:
            duba_belirdi = True
            print(f"  >>> [t={elapsed:.1f}s] Kırmızı duba görüş alanına girdi.")

        detections = []
        if duba_belirdi:
            detections = [{
                "class": "red_buoy",
                "distance": 6.5,
                "Buoy angle: ": 3.2,
            }]

        found = arama_gorevi.update(detections)
        if found:
            print(f"  >>> Hedef bulundu, arama tamamlandı (tick={tick}, sahte_t={elapsed:.1f}s).")
            break

    if not arama_gorevi.finished:
        print("  !! Beklenenden farklı: hedef bulunamadı.")
    else:
        print("  SONUÇ: Senaryo 1 BAŞARILI -- arama, kırmızı dubayı buldu.\n")


# ============================================================
# SENARYO 2: Kırmızı duba hiç belirmiyor -> istasyon değiştirmeli
# ============================================================
def senaryo_2_duba_hic_yok():
    print("\n" + "=" * 60)
    print("SENARYO 2: Kırmızı duba hiç belirmiyor -> istasyon değişmeli")
    print("=" * 60)

    fake_clock._now = 0.0
    _last_action["text"] = None

    node = FakeNode()
    topics = FakeTopics()

    arama_gorevi = AramaGorevi(node, topics, target_class="red_buoy")

    lat, lon = 41.015137, 28.979530
    heading = 0.0
    arama_gorevi.update_gps(lat, lon, heading)

    relocations_seen = 0
    last_state = None
    max_ticks = int(90.0 / TICK_DT)  # sahte zamanda 90 sn'ye kadar (gerçekte anlık biter)

    for tick in range(max_ticks):
        fake_clock.advance(TICK_DT)
        heading = advance_heading_towards(heading, 4.0)

        if arama_gorevi.relocation_target is not None and arama_gorevi.state.name == "RELOCATING":
            target_lat, target_lon = arama_gorevi.relocation_target
            lat, lon = target_lat, target_lon  # test amaçlı: hedefe anında ulaşmış say

        arama_gorevi.update_gps(lat, lon, heading)

        if arama_gorevi.state.name != last_state:
            print(f"  -> Durum değişti: {last_state} => {arama_gorevi.state.name}")
            if arama_gorevi.state.name == "SCANNING" and last_state == "RELOCATING":
                relocations_seen += 1
            last_state = arama_gorevi.state.name

        arama_gorevi.update(detections=[])  # kırmızı duba asla görünmüyor

        if relocations_seen >= 2:
            print("  >>> En az 2 istasyon değişimi gözlendi, senaryo yeterince test edildi.")
            break

    print(f"\n  Ziyaret edilen konum sayısı: {len(arama_gorevi.visited_positions)}")
    for i, (vlat, vlon) in enumerate(arama_gorevi.visited_positions):
        print(f"    {i + 1}) {vlat:.6f}, {vlon:.6f}")

    if relocations_seen >= 1 and len(arama_gorevi.visited_positions) >= 2:
        print("  SONUÇ: Senaryo 2 BAŞARILI -- duba bulunamayınca istasyon değişti,")
        print("         her istasyon farklı bir konumdaydı (aynı yere dönmedi).\n")
    else:
        print("  !! Beklenenden farklı: istasyon değişimi yeterince gözlenmedi.")


# ============================================================
# SENARYO 3: target_class VERİLMEDEN çağırmayı denersek (hata beklenmeli)
# ============================================================
def senaryo_3_renksiz_deneme():
    print("\n" + "=" * 60)
    print("SENARYO 3: target_class VERMEDEN AramaGorevi oluşturmayı deniyoruz")
    print("(Bunun HATA vermesi bekleniyor -- renk artık zorunlu)")
    print("=" * 60)

    node = FakeNode()
    topics = FakeTopics()

    try:
        AramaGorevi(node, topics)  # target_class kasıtlı olarak verilmedi
        print("  !! BEKLENMEYEN: Hata vermedi -- renk hâlâ opsiyonel olabilir, kontrol et.")
    except TypeError as exc:
        print(f"  SONUÇ: Senaryo 3 BAŞARILI -- beklenen hata alındı: {exc}")


if __name__ == "__main__":
    senaryo_1_duba_beliriyor()
    senaryo_2_duba_hic_yok()
    senaryo_3_renksiz_deneme()
    print("\nTüm senaryolar tamamlandı.")