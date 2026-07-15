#!/usr/bin/env python3
"""
ARAMA MODÜLÜ TEST SİMÜLASYONU v2 — GERÇEK KODU TEST EDER
ROS2 BAĞIMSIZ, HIZLI (SAHTE SAAT), SENARYO + ASSERT TABANLI

Bu dosya bir önceki versiyondan farklı olarak:
  1. AramaGorevi'nin kendi elle kopyalanmış bir sürümünü DEĞİL, gerçek
     `teknofest.missions.arama.AramaGorevi` sınıfını import edip test eder.
     (utils.mavlink_utilities sahte modül olarak enjekte edilir.)
  2. time.monotonic() sahte bir saatle değiştirilir -> 180 saniyelik bir
     senaryo gerçek zamanda milisaniyeler içinde koşar (time.sleep yok).
  3. Açı/birim hatası düzeltildi: angular_z (rad/s) artık math.degrees() ile
     doğru şekilde derece heading'e çevriliyor.
  4. RELOCATING durumunda publish_set_position ile gönderilen waypoint
     artık gerçekten simüle ediliyor (eskiden tekne hiç hareket etmiyordu).
  5. Kamera/vision'ın kontrol döngüsünden daha YAVAŞ gelmesi (vision_hz)
     ve ara sıra tespiti kaçırması (dropout_rate) simüle edilebiliyor.
  6. Pusula donması (heading freeze) enjekte edilebiliyor.
  7. Her senaryo somut assert'lerle PASS/FAIL sonucu üretiyor, sonda özet
     tablo ve CI için anlamlı exit code veriliyor.

Kullanım:
    python3 test_arama_simulation.py --repo-root /path/to/proje --all
    python3 test_arama_simulation.py --repo-root /path/to/proje --scenario flicker_detection
    python3 test_arama_simulation.py --repo-root /path/to/proje --list
"""

import sys
import time
import math
import random
import argparse
import importlib
import types
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any


# ============================================================
# 1) utils.mavlink_utilities SAHTESİ
#    (gerçek arama.py'yi ROS2/mavlink olmadan import edebilmek için)
# ============================================================

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _project_gps(lat, lon, bearing_deg, distance_m):
    R = 6378137.0
    bearing_rad = math.radians(bearing_deg)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ang_dist = distance_m / R
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(ang_dist)
        + math.cos(lat_rad) * math.sin(ang_dist) * math.cos(bearing_rad)
    )
    new_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(ang_dist) * math.cos(lat_rad),
        math.cos(ang_dist) - math.sin(lat_rad) * math.sin(new_lat_rad),
    )
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)


def _bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _install_fake_mavlink_utilities():
    """utils.mavlink_utilities'i gerçek ROS2/mavlink olmadan sağlar."""
    if 'utils.mavlink_utilities' in sys.modules:
        return sys.modules['utils.mavlink_utilities']

    utils_pkg = types.ModuleType('utils')
    utils_pkg.__path__ = []
    mav = types.ModuleType('utils.mavlink_utilities')

    def publish_cmd_vel(pub, linear_x=0.0, angular_z=0.0):
        pub.publish({'linear_x': linear_x, 'angular_z': angular_z})

    def stop_vehicle(pub, repeat_count=10):
        for _ in range(repeat_count):
            pub.publish({'linear_x': 0.0, 'angular_z': 0.0})

    def publish_set_position(pub, lat, lon, altitude=20.0):
        pub.publish({'lat': lat, 'lon': lon, 'altitude': altitude})

    def calculate_angle_error_deg(target_deg, current_deg):
        return (float(target_deg) - float(current_deg) + 180.0) % 360.0 - 180.0

    def calculate_bearing(lat1, lon1, lat2, lon2):
        return _bearing(lat1, lon1, lat2, lon2)

    mav.publish_cmd_vel = publish_cmd_vel
    mav.stop_vehicle = stop_vehicle
    mav.publish_set_position = publish_set_position
    mav.calculate_gps_distance = _haversine
    mav.calculate_angle_error_deg = calculate_angle_error_deg
    mav.calculate_bearing = calculate_bearing
    # main.py da aynı sahte modülü kullanmak isterse diye (arama.py bunları
    # kullanmıyor ama import zincirini kırmamak için ekliyoruz)
    mav.create_mission_topics = lambda *a, **k: None
    mav.create_mission_clients = lambda *a, **k: None
    mav.wait_for_mission_services = lambda *a, **k: None
    mav.call_set_mode = lambda *a, **k: True
    mav.call_trigger_service = lambda *a, **k: True

    utils_pkg.mavlink_utilities = mav
    sys.modules['utils'] = utils_pkg
    sys.modules['utils.mavlink_utilities'] = mav
    return mav


def _find_repo_root(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if (p / 'teknofest').is_dir():
            return p
        raise SystemExit(f"'--repo-root {explicit}' altında 'teknofest' paketi bulunamadı.")

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / 'teknofest').is_dir():
            return parent
    raise SystemExit(
        "'teknofest' paketi otomatik bulunamadı.\n"
        "Bu dosyayı proje içine koyun ya da:\n"
        "    python3 test_arama_simulation.py --repo-root /proje/kok/dizini --all"
    )


def _import_real_arama(repo_root: Path):
    """GERÇEK arama.py'yi import eder (kopya değil)."""
    _install_fake_mavlink_utilities()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module = importlib.import_module('teknofest.missions.arama')
    importlib.reload(module)  # olası eski import cache'ini temizle
    return module


# ============================================================
# 2) SAHTE SAAT — time.monotonic()'i değiştirip simülasyonu
#    gerçek zamandan koparır, testler anında koşar.
# ============================================================

class SimClock:
    def __init__(self, start: float = 10_000.0):
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, dt: float):
        self.t += dt


class ClockPatcher:
    def __init__(self, clock: SimClock):
        self.clock = clock
        self._orig = None

    def __enter__(self):
        self._orig = time.monotonic
        time.monotonic = self.clock.now
        return self.clock

    def __exit__(self, *exc):
        time.monotonic = self._orig


# ============================================================
# 3) ROS2 MOCK SINIFLARI
# ============================================================

class MockLogger:
    def info(self, msg, *a, **k):
        pass

    def warn(self, msg, *a, **k):
        pass

    def warning(self, msg, *a, **k):
        pass

    def error(self, msg, *a, **k):
        pass

    def debug(self, msg, *a, **k):
        pass


class VerboseMockLogger(MockLogger):
    def info(self, msg, *a, **k):
        print(f"[INFO] {msg}")

    def warn(self, msg, *a, **k):
        print(f"[WARN] {msg}")

    warning = warn

    def error(self, msg, *a, **k):
        print(f"[ERROR] {msg}")


class MockNode:
    def __init__(self, verbose=False):
        self._logger = VerboseMockLogger() if verbose else MockLogger()

    def get_logger(self):
        return self._logger


class MockPublisher:
    def __init__(self):
        self.last_msg = None
        self.count = 0

    def publish(self, msg):
        self.last_msg = msg
        self.count += 1


class MockTopics:
    def __init__(self):
        self.cmd_vel_pub = MockPublisher()
        self.position_target_pub = MockPublisher()


# ============================================================
# 4) ARAÇ FİZİĞİ SİMÜLASYONU
#    DÜZELTME 1: angular_z (rad/s) artık math.degrees() ile doğru
#                şekilde derece heading'e çevriliyor.
#    DÜZELTME 2: position_target_pub'a (waypoint) gönderilen komutlar
#                da simüle ediliyor -> RELOCATING artık gerçekten
#                ilerliyor.
# ============================================================

class VehicleSimulator:
    MAX_RELOCATE_SPEED_MPS = 1.2  # GUIDED modda otopilotun tahmini ilerleme hızı

    def __init__(self, home_lat, home_lon, noise_level=0.0, freeze_heading_after=None):
        self.lat = home_lat
        self.lon = home_lon
        self.heading = 0.0
        self.noise_level = noise_level
        self.freeze_heading_after = freeze_heading_after  # sim saniyesi
        self.total_distance = 0.0
        self.total_rotation_deg = 0.0

    def step(self, cmd_vel_pub, position_pub, dt, sim_time):
        heading_frozen = (
            self.freeze_heading_after is not None and sim_time >= self.freeze_heading_after
        )

        wp = position_pub.last_msg
        if wp is not None:
            target_lat, target_lon = wp['lat'], wp['lon']
            dist = _haversine(self.lat, self.lon, target_lat, target_lon)
            if dist > 0.3:
                bearing = _bearing(self.lat, self.lon, target_lat, target_lon)
                step_dist = min(dist, self.MAX_RELOCATE_SPEED_MPS * dt)
                self.lat, self.lon = _project_gps(self.lat, self.lon, bearing, step_dist)
                if not heading_frozen:
                    self.heading = bearing
                self.total_distance += step_dist
            return

        cmd = cmd_vel_pub.last_msg or {'linear_x': 0.0, 'angular_z': 0.0}
        linear_x = cmd.get('linear_x', 0.0)
        angular_z = cmd.get('angular_z', 0.0)  # rad/s

        if not heading_frozen:
            delta_deg = math.degrees(angular_z * dt)
            self.heading = (self.heading + delta_deg) % 360.0
            self.total_rotation_deg += abs(delta_deg)

        if abs(linear_x) > 1e-3:
            step_dist = linear_x * dt
            self.lat, self.lon = _project_gps(self.lat, self.lon, self.heading, step_dist)
            self.total_distance += abs(step_dist)

        if self.noise_level > 0:
            self.lat += random.uniform(-self.noise_level, self.noise_level) * 1e-6
            self.lon += random.uniform(-self.noise_level, self.noise_level) * 1e-6

    def gps(self):
        return self.lat, self.lon, self.heading


# ============================================================
# 5) VISION SİMÜLASYONU
#    DÜZELTME: vision_hz artık gerçekten kullanılıyor (kontrol
#    döngüsünden daha yavaş kare gelmesini simüle eder) ve
#    dropout_rate ile ara sıra tespiti kaçırma (flicker) simüle
#    edilebiliyor.
# ============================================================

class VisionSimulator:
    def __init__(self, home_lat, home_lon, target_bearing, target_distance,
                 target_class="red_buoy", detection_range=40.0, field_of_view=70.0,
                 vision_hz=10.0, dropout_rate=0.0, never_visible=False, noise_level=0.1):
        self.target_class = target_class
        self.detection_range = detection_range
        self.field_of_view = field_of_view
        self.vision_hz = vision_hz
        self.dropout_rate = dropout_rate
        self.never_visible = never_visible
        self.noise_level = noise_level

        self.target_lat, self.target_lon = _project_gps(home_lat, home_lon, target_bearing, target_distance)
        self._next_frame_time = 0.0
        self.frames_sent = 0
        self.frames_with_detection = 0

    def update(self, lat, lon, heading, sim_time) -> List[Dict]:
        """Her tick çağrılır ama sadece vision_hz periyodunda 'yeni kare'
        üretir; aradaki ticklerde gerçek bir vision node'un o an bir şey
        yayınlamamasını simüle etmek için boş liste döner (bayat veri
        TEKRARLANMAZ)."""
        if sim_time < self._next_frame_time:
            return []
        self._next_frame_time = sim_time + (1.0 / self.vision_hz)
        self.frames_sent += 1

        if self.never_visible:
            return []

        distance = _haversine(lat, lon, self.target_lat, self.target_lon)
        if distance > self.detection_range:
            return []

        bearing = _bearing(lat, lon, self.target_lat, self.target_lon)
        angle = (bearing - heading + 180) % 360 - 180
        if abs(angle) > self.field_of_view / 2:
            return []

        if self.dropout_rate > 0 and random.random() < self.dropout_rate:
            return []  # gerçek hayattaki tekli kare kaçırma (dalga/parıltı vb.)

        if self.noise_level > 0:
            distance += random.uniform(-self.noise_level, self.noise_level)
            angle += random.uniform(-2.0, 2.0)
        distance = max(0.5, distance)

        self.frames_with_detection += 1
        return [{
            'class': self.target_class,
            'distance': distance,
            'Buoy angle: ': angle,
            'confidence': random.uniform(0.7, 1.0),
        }]


# ============================================================
# 6) SENARYO TANIMI VE ÇALIŞTIRICI
# ============================================================

@dataclass
class Scenario:
    name: str
    description: str
    target_bearing: float = 45.0
    target_distance: float = 25.0
    detection_range: float = 40.0
    dropout_rate: float = 0.0
    vision_hz: float = 10.0
    never_visible: bool = False
    freeze_heading_after: Optional[float] = None
    max_sim_seconds: float = 180.0
    check: Callable[[Dict[str, Any]], List[str]] = None


def run_scenario(scenario: Scenario, repo_root: Path, seed: int, verbose: bool = False) -> Dict[str, Any]:
    random.seed(seed)
    arama_module = _import_real_arama(repo_root)

    node = MockNode(verbose=verbose)
    topics = MockTopics()
    home_lat, home_lon = 40.0, 30.0

    vehicle = VehicleSimulator(home_lat, home_lon, noise_level=0.05,
                                freeze_heading_after=scenario.freeze_heading_after)
    vision = VisionSimulator(
        home_lat, home_lon,
        target_bearing=scenario.target_bearing,
        target_distance=scenario.target_distance,
        detection_range=scenario.detection_range,
        vision_hz=scenario.vision_hz,
        dropout_rate=scenario.dropout_rate,
        never_visible=scenario.never_visible,
    )

    clock = SimClock()
    error_raised = None

    with ClockPatcher(clock):
        arama = arama_module.AramaGorevi(node, topics, target_class="red_buoy", test_mode=verbose)
        lat, lon, heading = vehicle.gps()
        arama.update_gps(lat, lon, heading)

        dt = 0.1
        ticks = int(scenario.max_sim_seconds / dt)
        found_at = None

        try:
            for i in range(ticks):
                lat, lon, heading = vehicle.gps()
                detections = vision.update(lat, lon, heading, clock.now())

                arama.update(detections)

                vehicle.step(topics.cmd_vel_pub, topics.position_target_pub, dt, clock.now())

                lat, lon, heading = vehicle.gps()
                arama.update_gps(lat, lon, heading)

                if arama.finished and found_at is None:
                    found_at = clock.now() - 10_000.0

                clock.advance(dt)

                if arama.finished:
                    break
        except Exception as exc:  # simülasyon sırasında algoritma çökmesin
            error_raised = f"{type(exc).__name__}: {exc}"

    status = arama.get_search_status() if error_raised is None else {}
    result = {
        "scenario": scenario.name,
        "finished": bool(getattr(arama, "finished", False)),
        "found_at_sec": found_at,
        "visited_positions": status.get("visited_positions"),
        "search_retry_count": status.get("search_retry_count"),
        "rotated_deg": status.get("rotated_deg"),
        "vehicle_total_rotation_deg": vehicle.total_rotation_deg,
        "vehicle_total_distance_m": vehicle.total_distance,
        "frames_sent": vision.frames_sent,
        "frames_with_detection": vision.frames_with_detection,
        "error": error_raised,
        "sim_seconds_used": clock.now() - 10_000.0,
    }

    errors = []
    if error_raised:
        errors.append(f"Simülasyon sırasında istisna fırlatıldı: {error_raised}")
    elif scenario.check:
        errors.extend(scenario.check(result))

    result["pass"] = len(errors) == 0
    result["errors"] = errors
    return result


# ============================================================
# 7) SENARYOLAR
# ============================================================

def _scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="baseline_visible",
            description="Hedef başından beri görünür ve menzilde -> hızlı onaylanmalı",
            target_bearing=30.0, target_distance=20.0,
            max_sim_seconds=60.0,
            check=lambda r: (
                [] if r["finished"] and r["found_at_sec"] is not None and r["found_at_sec"] < 40.0
                else [f"Hedef beklenen sürede bulunamadı (found_at={r['found_at_sec']}, finished={r['finished']})"]
            ),
        ),
        Scenario(
            name="relocation_physics",
            description="Birkaç istasyon değişimi gerektiren orta mesafeli hedef. "
                        "Bu senaryonun asıl amacı RELOCATING sırasında geminin "
                        "gerçekten fiziksel olarak hareket ettiğini doğrulamak "
                        "(eski testte bu hiç simüle edilmiyordu, tekne hiç yer "
                        "değiştirmiyordu) — hedefin bulunması ikincil kriterdir.",
            target_bearing=140.0, target_distance=30.0, detection_range=20.0,
            max_sim_seconds=240.0,
            check=lambda r: (
                (["Hiç istasyon değişimi olmadı (relocation simüle edilmiyor olabilir)"]
                 if (r["visited_positions"] or 0) < 2 else [])
                + (["Araç hiç yer değiştirmemiş (relocation fiziksel olarak simüle edilmiyor)"]
                   if r["vehicle_total_distance_m"] < 5.0 else [])
                + (["Hedef bulunamadı (istasyon fiziği çalışıyor ama arama başarısız)"]
                   if not r["finished"] else [])
            ),
        ),
        Scenario(
            name="wide_area_slow_coverage",
            description="Hedef, arama alanının uzak bir köşesinde (55m). "
                        "STATION_MOVE_DISTANCE_M büyüme formülü kademeli olduğundan "
                        "(idx arttıkça yavaş yavaş genişler), buraya ulaşmak çok "
                        "sayıda istasyon gerektirir — bu BİLGİLENDİRME amaçlı bir "
                        "senaryo: geniş alanı hızlı taramak isteniyorsa büyüme "
                        "katsayısının/limitin ayarlanması gerektiğini gösterir.",
            target_bearing=200.0, target_distance=55.0, detection_range=25.0,
            max_sim_seconds=900.0,
            check=lambda r: (
                [] if r["finished"]
                else [f"Bilgi: {r['sim_seconds_used']:.0f}sn / {r['visited_positions']} istasyonda "
                      f"hâlâ bulunamadı — geniş alan taraması istenen kadar hızlı değilse "
                      f"STATION_MOVE_DISTANCE_M büyüme katsayısını artırmayı düşünün."]
            ),
        ),
        Scenario(
            name="flicker_detection",
            description="Hedef görünür ama %35 ihtimalle her karede kaçırılıyor "
                        "(dalga/parıltı benzeri gerçek hayat gürültüsü). Pencere "
                        "tabanlı onay mantığını test eder.",
            target_bearing=10.0, target_distance=15.0,
            dropout_rate=0.35,
            max_sim_seconds=90.0,
            check=lambda r: (
                [] if r["finished"]
                else ["Titrek tespit altında hedef hiç onaylanamadı — onay mantığı "
                      "tek karelik kaçırmalara karşı çok kırılgan olabilir."]
            ),
        ),
        Scenario(
            name="never_visible",
            description="Hedef simülasyon boyunca hiç görünmüyor. Algoritma "
                        "çökmemeli, sonsuz döngüye girmemeli ve arama alanını "
                        "sistemli şekilde taramaya devam etmeli.",
            never_visible=True,
            max_sim_seconds=150.0,
            check=lambda r: (
                (["Hedef yokken 'finished=True' oldu (yanlış pozitif!)"] if r["finished"] else [])
                + (["Araç hiç istasyon değiştirmedi, sıkışmış olabilir"]
                   if (r["visited_positions"] or 0) < 2 else [])
            ),
        ),
        Scenario(
            name="low_vision_rate",
            description="Kamera kontrol döngüsünden çok daha yavaş (3 Hz) veri "
                        "üretiyor. Gerçek donanımda sık karşılaşılan bir durum.",
            target_bearing=60.0, target_distance=18.0,
            vision_hz=3.0,
            max_sim_seconds=90.0,
            check=lambda r: (
                [] if r["finished"]
                else ["Düşük vision hızında hedef onaylanamadı."]
            ),
        ),
        Scenario(
            name="heading_freeze",
            description="30. saniyeden sonra pusula/heading verisi donuyor. "
                        "STEP_MAX_DURATION_SEC gibi bir koruma yoksa arama bir "
                        "dönüş adımında sonsuza kadar takılabilir.",
            target_bearing=300.0, target_distance=30.0, detection_range=20.0,
            freeze_heading_after=30.0,
            max_sim_seconds=120.0,
            check=lambda r: (
                [] if (r["visited_positions"] or 0) >= 1 or r["finished"]
                else ["Heading dondu ve arama hiçbir ilerleme kaydetmedi "
                      "(muhtemelen dönüş adımı için üst süre sınırı yok)."]
            ),
        ),
    ]


# ============================================================
# 8) ANA PROGRAM
# ============================================================

def _print_result(res: Dict[str, Any], description: str):
    status = "✅ PASS" if res["pass"] else "❌ FAIL"
    print(f"\n{status}  {res['scenario']}")
    print(f"    {description}")
    print(f"    finished={res['finished']}  bulunma_süresi={res['found_at_sec']}  "
          f"istasyon={res['visited_positions']}  retry={res['search_retry_count']}")
    print(f"    araç_dönüş={res['vehicle_total_rotation_deg']:.1f}°  "
          f"araç_mesafe={res['vehicle_total_distance_m']:.1f}m  "
          f"kare(gönderilen/tespit)={res['frames_sent']}/{res['frames_with_detection']}  "
          f"sim_süre={res['sim_seconds_used']:.1f}s")
    for e in res["errors"]:
        print(f"    ⚠️  {e}")


def main():
    parser = argparse.ArgumentParser(description="Arama modülü senaryo testleri (gerçek kodu test eder)")
    parser.add_argument('--repo-root', type=str, default=None,
                        help="'teknofest' paketini içeren proje kök dizini")
    parser.add_argument('--scenario', type=str, default=None, help="Tek bir senaryo çalıştır")
    parser.add_argument('--all', action='store_true', help="Tüm senaryoları çalıştır")
    parser.add_argument('--list', action='store_true', help="Senaryoları listele")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true', help="Algoritmanın kendi loglarını da bas")
    args = parser.parse_args()

    scenarios = _scenarios()

    if args.list:
        for s in scenarios:
            print(f"- {s.name}: {s.description}")
        return

    repo_root = _find_repo_root(args.repo_root)
    print(f"Proje kökü: {repo_root}")

    if args.scenario:
        chosen = [s for s in scenarios if s.name == args.scenario]
        if not chosen:
            names = ", ".join(s.name for s in scenarios)
            raise SystemExit(f"'{args.scenario}' bulunamadı. Mevcut: {names}")
    elif args.all:
        chosen = scenarios
    else:
        chosen = scenarios
        print("(--scenario/--all verilmedi, tüm senaryolar çalıştırılıyor)")

    results = []
    for s in chosen:
        res = run_scenario(s, repo_root, seed=args.seed, verbose=args.verbose)
        _print_result(res, s.description)
        results.append(res)

    passed = sum(1 for r in results if r["pass"])
    print(f"\n{'=' * 60}")
    print(f"SONUÇ: {passed}/{len(results)} senaryo geçti")
    print(f"{'=' * 60}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()