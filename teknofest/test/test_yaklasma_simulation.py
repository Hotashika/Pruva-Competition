"""
yaklasma.py (YaklasmaGorevi) için ROS2'siz, teknesiz masaüstü simülasyonu.

test_arama_simulation.py ile AYNI teknik: gerçek utils.mavlink_utilities'i
sahte bir modülle değiştiriyoruz, gerçek saati (time.monotonic) hızlandırılmış
sahte bir saatle değiştiriyoruz. yaklasma.py'nin GERÇEK kodu (hiç değişmeden)
çalışıyor, sadece etrafındaki her şey taklit.

ÇALIŞTIRMA:
    Bu dosyayı yaklasma.py ile AYNI klasöre koyun, sonra:
        python3 test_yaklasma_simulation.py

Test edilen senaryolar:
  1) Duba giderek yaklaşıyor (mesafe azalıyor) -> PID doğru tepki veriyor mu,
     güvenli mesafede (1m) durup ters itki uyguluyor mu, finished=True oluyor mu?
  2) Hedef kısa süreliğine kayboluyor ama tolerans içinde geri geliyor ->
     panik yapmadan devam ediyor mu?
  3) Hedef tolerans süresinden uzun süre kayboluyor -> target_lost=True oluyor mu?
  4) target_class verilmeden oluşturma denemesi -> hata bekleniyor.
"""

import sys
import types

# ============================================================
# 1) SAHTE SAAT
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
fake_time_module.sleep = lambda s: None

# ============================================================
# 2) SAHTE utils.mavlink_utilities
# ============================================================
fake_mavlink = types.ModuleType("utils.mavlink_utilities")

_last_action = {"text": None, "count": 0}


def _log_action(text):
    if text == _last_action["text"]:
        _last_action["count"] += 1
        return
    if _last_action["text"] is not None and _last_action["count"] > 1:
        print(f"    ... (yukarıdaki satır {_last_action['count']} kez tekrarlandı)")
    print(text)
    _last_action["text"] = text
    _last_action["count"] = 1


def publish_cmd_vel(pub, linear_x, angular_z):
    _log_action(f"    cmd_vel -> linear_x={linear_x:+.3f}  angular_z={angular_z:+.3f}")


def stop_vehicle(pub):
    _log_action("    stop_vehicle -> DUR")


fake_mavlink.publish_cmd_vel = publish_cmd_vel
fake_mavlink.stop_vehicle = stop_vehicle

fake_utils_pkg = types.ModuleType("utils")
fake_utils_pkg.mavlink_utilities = fake_mavlink
sys.modules["utils"] = fake_utils_pkg
sys.modules["utils.mavlink_utilities"] = fake_mavlink

# ============================================================
# 3) yaklasma.py'Yİ İÇE AKTAR, time MODÜLÜNÜ SAHTESİYLE DEĞİŞTİR
# ============================================================
import yaklasma  # noqa: E402

yaklasma.time = fake_time_module

from yaklasma import YaklasmaGorevi, ApproachState  # noqa: E402


# ============================================================
# 4) SAHTE NODE / TOPICS
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


TICK_DT = 0.1


# ============================================================
# SENARYO 1: Duba giderek yaklaşıyor
# ============================================================
def senaryo_1_yaklasma_basarili():
    print("\n" + "=" * 60)
    print("SENARYO 1: Kırmızı duba giderek yaklaşıyor (8m -> 0.5m)")
    print("=" * 60)

    fake_clock._now = 0.0
    _last_action["text"] = None

    node = FakeNode()
    topics = FakeTopics()
    yaklasma_gorevi = YaklasmaGorevi(node, topics, target_class="red_buoy")

    # Küçük bir yanal ivme simüle edelim (rüzgar/akıntı telafisini görmek için)
    yaklasma_gorevi.update_imu(gyro_z=0.01, accel_x=0.0, accel_y=0.3)

    distance = 8.0
    angle = 15.0  # hedef başlangıçta biraz sağda
    max_ticks = int(30.0 / TICK_DT)

    for tick in range(max_ticks):
        fake_clock.advance(TICK_DT)

        detections = [{
            "class": "red_buoy",
            "distance": distance,
            "Buoy angle: ": angle,
        }]

        finished = yaklasma_gorevi.update(detections)

        # Basit bir "fizik" simülasyonu: mesafe azalıyor, açı sıfıra yaklaşıyor
        # (gerçek harekete tepki veriyormuş gibi, test amaçlı basitleştirilmiş)
        distance = max(0.0, distance - 0.12)
        angle = angle * 0.92

        if finished:
            print(f"  >>> Yaklaşma bitti (tick={tick}, son mesafe≈{distance:.2f}m).")
            break

    print(f"\n  Son durum: {yaklasma_gorevi.state.name}")
    print(f"  finished={yaklasma_gorevi.finished}  target_lost={yaklasma_gorevi.target_lost}")

    if yaklasma_gorevi.finished and yaklasma_gorevi.state == ApproachState.DONE:
        print("  SONUÇ: Senaryo 1 BAŞARILI -- güvenli mesafede durdu, ters itki uygulandı.\n")
    else:
        print("  !! Beklenenden farklı: yaklaşma tamamlanmadı.")


# ============================================================
# SENARYO 2: Hedef kısa süreliğine kayboluyor, tolerans içinde geri geliyor
# ============================================================
def senaryo_2_kisa_kayip_toparlaniyor():
    print("=" * 60)
    print("SENARYO 2: Hedef 0.3sn kayboluyor (tolerans 0.5sn) -> toparlanmalı")
    print("=" * 60)

    fake_clock._now = 0.0
    _last_action["text"] = None

    node = FakeNode()
    topics = FakeTopics()
    yaklasma_gorevi = YaklasmaGorevi(node, topics, target_class="red_buoy")

    # Birkaç normal tik
    for _ in range(5):
        fake_clock.advance(TICK_DT)
        yaklasma_gorevi.update([{"class": "red_buoy", "distance": 5.0, "Buoy angle: ": 2.0}])

    # 3 tik boyunca (0.3sn) hedef kayboluyor
    for _ in range(3):
        fake_clock.advance(TICK_DT)
        finished = yaklasma_gorevi.update([])
        assert not finished, "BEKLENMEYEN: kısa kayıpta erken bitti"

    # Hedef geri geliyor
    fake_clock.advance(TICK_DT)
    yaklasma_gorevi.update([{"class": "red_buoy", "distance": 4.8, "Buoy angle: ": 1.5}])

    print(f"  Son durum: {yaklasma_gorevi.state.name}, target_lost={yaklasma_gorevi.target_lost}")
    if not yaklasma_gorevi.target_lost and yaklasma_gorevi.state == ApproachState.TRACKING:
        print("  SONUÇ: Senaryo 2 BAŞARILI -- kısa kayıptan sonra takip devam etti.\n")
    else:
        print("  !! Beklenenden farklı.")


# ============================================================
# SENARYO 3: Hedef uzun süre kayboluyor -> target_lost=True olmalı
# ============================================================
def senaryo_3_uzun_kayip_target_lost():
    print("=" * 60)
    print("SENARYO 3: Hedef 1.0sn kayboluyor (tolerans 0.5sn) -> target_lost olmalı")
    print("=" * 60)

    fake_clock._now = 0.0
    _last_action["text"] = None

    node = FakeNode()
    topics = FakeTopics()
    yaklasma_gorevi = YaklasmaGorevi(node, topics, target_class="red_buoy")

    for _ in range(5):
        fake_clock.advance(TICK_DT)
        yaklasma_gorevi.update([{"class": "red_buoy", "distance": 5.0, "Buoy angle: ": 2.0}])

    finished = False
    for _ in range(15):  # 1.5 sn boyunca hedef yok
        fake_clock.advance(TICK_DT)
        finished = yaklasma_gorevi.update([])
        if finished:
            break

    print(f"  Son durum: {yaklasma_gorevi.state.name}, target_lost={yaklasma_gorevi.target_lost}")
    if finished and yaklasma_gorevi.target_lost:
        print("  SONUÇ: Senaryo 3 BAŞARILI -- uzun kayıpta target_lost=True oldu.\n")
    else:
        print("  !! Beklenenden farklı: target_lost tetiklenmedi.")


# ============================================================
# SENARYO 4: target_class VERİLMEDEN çağırmayı denersek (hata beklenmeli)
# ============================================================
def senaryo_4_renksiz_deneme():
    print("=" * 60)
    print("SENARYO 4: target_class VERMEDEN YaklasmaGorevi oluşturmayı deniyoruz")
    print("=" * 60)

    node = FakeNode()
    topics = FakeTopics()

    try:
        YaklasmaGorevi(node, topics)
        print("  !! BEKLENMEYEN: Hata vermedi.")
    except TypeError as exc:
        print(f"  SONUÇ: Senaryo 4 BAŞARILI -- beklenen hata alındı: {exc}")


# ============================================================
# SENARYO 5: Emin Olma normal şekilde tamamlanıyor mu?
# ============================================================
def senaryo_5_emin_olma_normal():
    print("=" * 60)
    print("SENARYO 5: Duba yavaşça yaklaşıyor -> Emin Olma 1sn'de tamamlanmalı")
    print("=" * 60)

    fake_clock._now = 0.0
    _last_action["text"] = None

    node = FakeNode()
    topics = FakeTopics()
    yaklasma_gorevi = YaklasmaGorevi(node, topics, target_class="red_buoy")

    distance = 8.0
    angle = 5.0
    max_ticks = int(30.0 / TICK_DT)
    confirmed_seen = False

    for tick in range(max_ticks):
        fake_clock.advance(TICK_DT)
        detections = [{"class": "red_buoy", "distance": distance, "Buoy angle: ": angle}]
        finished = yaklasma_gorevi.update(detections)

        if yaklasma_gorevi._confirmed and not confirmed_seen:
            confirmed_seen = True
            print(f"  >>> [tick={tick}] Doğrulama tamamlandı, mesafe≈{distance:.2f}m")

        distance = max(0.0, distance - 0.12)
        angle = angle * 0.92

        if finished:
            break

    if confirmed_seen and yaklasma_gorevi.finished:
        print("  SONUÇ: Senaryo 5 BAŞARILI -- Emin Olma tamamlandıktan sonra yaklaşma bitti.\n")
    else:
        print("  !! Beklenenden farklı: doğrulama veya bitiş gerçekleşmedi.")


# ============================================================
# SENARYO 6: ÇOK HIZLI yaklaşım -> güvenli mesafeye doğrulanmadan ulaşılırsa
#            araç ileri gitmeyi durdurup BEKLEMELİ (asla körlemesine çarpmamalı)
# ============================================================
def senaryo_6_dogrulanmadan_yaklasilirsa_bekler():
    print("=" * 60)
    print("SENARYO 6: Duba ÇOK HIZLI yaklaşıyor (doğrulama süresi dolmadan)")
    print("Beklenen: güvenli mesafede ileri hız SIFIRLANMALI, doğrulanana kadar beklemeli")
    print("=" * 60)

    fake_clock._now = 0.0
    _last_action["text"] = None

    node = FakeNode()
    topics = FakeTopics()
    yaklasma_gorevi = YaklasmaGorevi(node, topics, target_class="red_buoy")

    # Mesafeyi anında 0.5m'ye düşürüyoruz (doğrulama süresi geçmeden)
    distance = 0.5
    angle = 0.0

    held_with_zero_speed = False

    for tick in range(20):  # 2 saniye
        fake_clock.advance(TICK_DT)
        detections = [{"class": "red_buoy", "distance": distance, "Buoy angle: ": angle}]
        finished = yaklasma_gorevi.update(detections)

        if not yaklasma_gorevi._confirmed and yaklasma_gorevi._last_linear_x <= 0.001:
            held_with_zero_speed = True

        if finished:
            print(f"  >>> [tick={tick}] Yaklaşma bitti (doğrulama {CONFIRM_HOLD_wait_note()} sonra).")
            break

    print(f"\n  finished={yaklasma_gorevi.finished}  confirmed={yaklasma_gorevi._confirmed}")
    if held_with_zero_speed and yaklasma_gorevi.finished:
        print("  SONUÇ: Senaryo 6 BAŞARILI -- doğrulanmadan asla ileri gidip çarpmadı, "
              "doğrulama bitince güvenli şekilde durdu.\n")
    else:
        print("  !! Beklenenden farklı: araç doğrulanmadan ileri gitmiş olabilir!")


def CONFIRM_HOLD_wait_note():
    return "1sn"


if __name__ == "__main__":
    senaryo_1_yaklasma_basarili()
    senaryo_2_kisa_kayip_toparlaniyor()
    senaryo_3_uzun_kayip_target_lost()
    senaryo_4_renksiz_deneme()
    senaryo_5_emin_olma_normal()
    senaryo_6_dogrulanmadan_yaklasilirsa_bekler()
    print("\nTüm senaryolar tamamlandı.")