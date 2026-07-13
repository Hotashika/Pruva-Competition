"""
Task-3 Kamikaze Angajman Görevi — Aşama 1: ARAMA

Bu modül tek başına çalışan bir ROS2 node DEĞİLDİR. task3_kamikaze_engagement.py
içindeki Task3KamikazeEngagement sınıfı tarafından kullanılan bir yardımcı
sınıf (AramaGorevi) içerir. Entegrasyon için dosyanın en altındaki NOT kısmına bak.

Mantık:
  1) Araç bulunduğu konumda durur, 20 derecelik adımlarla döner (gerçek pusula
     heading'ine göre, zaman tahminine göre değil -> rüzgar/akıntı etkisine
     karşı daha sağlam).
  2) Her 20 derecelik adımdan sonra STEP_SETTLE_SEC kadar durup vision_node'un
     o karede tespit yapmasına fırsat tanır.
  3) Bir konumda STATION_TIMEOUT_SEC saniyeden fazla kalınırsa (veya tam tur
     tamamlanırsa) hedef bulunamamış demektir -> yeni, daha önce taranmamış
     bir konuma ilerler.
  4) Ziyaret edilen konumlar (visited_positions) hafızada tutulur; yeni durak
     seçilirken bu konumlardan yeterince uzak olması şartı aranır.
  5) Hedef (duba) net bir şekilde tespit edilince (class eşleşiyor, mesafe ve
     açı geçerli) state=TARGET_FOUND olur, finished=True döner ve üst katman
     (Task3KamikazeEngagement) buradan YAKLAŞMA aşamasına geçer.
"""

import math
import time
from enum import Enum, auto

from utils.mavlink_utilities import (
    publish_cmd_vel,
    publish_set_position,
    stop_vehicle,
    calculate_gps_distance,
)

# ============================================================
# ARAMA PARAMETRELERİ (sahaya göre ayarlayın)
# ============================================================
# NOT: Aranacak/çarpılacak duba rengi ARTIK BURADA SABİT DEĞİL.
# Renk, task3_kamikaze_engagement.py'de 'carpilacak_duba' ROS2 parametresi
# ile dışarıdan verilir ve AramaGorevi'ye target_class argümanı olarak geçirilir.
# Bu dosyada TARGET_CLASS sabiti yalnızca bu modülü tek başına (örn. test
# script'i içinde) çalıştırmak isteyenler için bir GERİYE DÖNÜK VARSAYILANDIR.

SEARCH_STEP_DEG = 20.0             # Her adımda dönülecek açı
SEARCH_ANGULAR_SPEED = 0.3         # Dönüş sırasında angular_z (rad/s). İşareti ters
                                    # gelirse -0.3 yapın (dönüş yönü tercih meselesi).
STEP_SETTLE_SEC = 1.2              # Her 20°'lik adımdan sonra sabit bekleme süresi
STATION_TIMEOUT_SEC = 18.0         # Bir konumda en fazla bu kadar kal (15-20 sn aralığı)
MAX_SEARCH_ROTATION_DEG = 360.0    # Bir konumda toplam taranacak açı (360 = tam tur)

STATION_MOVE_DISTANCE_M = 8.0      # Yeni konuma geçerken kat edilecek temel mesafe
STATION_MIN_SEPARATION_M = 5.0     # Yeni konum, ziyaret edilenlerden en az bu kadar uzak olmalı
RELOCATE_TOLERANCE_M = 2.0         # Yeni konuma "ulaşıldı" sayılacak mesafe

# ÖNEMLİ: Bu değer, task3_kamikaze_engagement.py'deki GEOFENCE_RADIUS_M (150m)
# değerinden KÜÇÜK olmalı, yoksa arama sırasında FAILSAFE tetiklenebilir.
SEARCH_AREA_RADIUS_M = 60.0

GOLDEN_ANGLE_DEG = 137.5           # Yeni istasyon yönleri arasında iyi dağılım sağlar


class SearchState(Enum):
    SCANNING = auto()      # Bulunduğu konumda 20°'lik adımlarla dönerek tarıyor
    STEP_PAUSE = auto()    # Bir adım döndükten sonra kısa bekleme (tespit için)
    RELOCATING = auto()    # Yeni bir arama konumuna ilerliyor
    TARGET_FOUND = auto()  # Hedef bulundu, arama bitti


class AramaGorevi:
    def __init__(self, node, mission_topics, target_class):
        """
        target_class: ZORUNLU. "red_buoy", "green_buoy" veya "black_buoy" gibi
        tam sınıf adı. Bu değer task3_kamikaze_engagement.py tarafından,
        'carpilacak_duba' ROS2 parametresinden üretilip buraya geçirilir.
        Burada varsayılan bir renk VERİLMEZ; çünkü renk yarışma günü hakem
        tarafından belirlenene kadar bilinmiyor (None olabilir) ve AramaGorevi
        rastgele/varsayılan bir renkle "sessizce" aramaya başlamamalı.
        """
        self.node = node
        self.logger = node.get_logger()
        self.topics = mission_topics
        self.target_class = target_class

        self.state = SearchState.SCANNING
        self.finished = False
        self.found_target = None

        # Konum verileri
        self.home_lat = None
        self.home_lon = None
        self.current_lat = None
        self.current_lon = None
        self.current_heading = 0.0
        self.visited_positions = []  # [(lat, lon), ...]
        self.station_index = 0

        # Tarama (dönme) takibi
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None
        self.station_start_time = None
        self.step_pause_until = None

        # Yer değiştirme (relocation) hedefi
        self.relocation_target = None

    # --------------------------------------------------------
    def update_gps(self, lat, lon, heading):
        """Task3KamikazeEngagement.update_gps() içinden çağrılmalı."""
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading

        if self.home_lat is None:
            self.home_lat = lat
            self.home_lon = lon
            self.visited_positions.append((lat, lon))
            self.station_start_time = time.monotonic()
            self.logger.info(f"[ARAMA] İlk arama konumu kaydedildi: {lat:.6f}, {lon:.6f}")

    # --------------------------------------------------------
    def _select_target(self, detections):
        """Tespitler arasından geçerli, en yakın duba adayını seçer."""
        candidates = [
            d for d in detections
            if d.get("class") == self.target_class
            and d.get("distance") is not None
            and d.get("distance", -1) > 0
            and d.get("Buoy angle: ") is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda d: d["distance"])

    # --------------------------------------------------------
    @staticmethod
    def _heading_diff(a, b):
        """b - a farkını [-180, 180] aralığına sararak döner (derece)."""
        return (b - a + 180.0) % 360.0 - 180.0

    @staticmethod
    def _project_gps(lat, lon, bearing_deg, distance_m):
        """Verilen konumdan, bearing (kuzeyden saat yönünde derece) ve mesafeye (m)
        göre yeni bir GPS noktası hesaplar (düz-dünya/haversine yaklaşık formülü)."""
        R = 6378137.0  # Dünya yarıçapı (m)
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

    def _is_far_enough_from_visited(self, lat, lon):
        for vlat, vlon in self.visited_positions:
            if calculate_gps_distance(lat, lon, vlat, vlon) < STATION_MIN_SEPARATION_M:
                return False
        return True

    def _next_station_target(self):
        """Home merkezli, ziyaret edilmemiş, alan sınırı içinde yeni bir durak üretir."""
        bearing, distance, target_lat, target_lon = None, None, None, None

        for attempt in range(10):
            idx = self.station_index + attempt
            bearing = (idx * GOLDEN_ANGLE_DEG) % 360.0
            distance = min(
                STATION_MOVE_DISTANCE_M * (1 + idx * 0.4),
                SEARCH_AREA_RADIUS_M * 0.9,
            )
            target_lat, target_lon = self._project_gps(
                self.home_lat, self.home_lon, bearing, distance
            )
            if self._is_far_enough_from_visited(target_lat, target_lon):
                break

        self.station_index += 1
        return target_lat, target_lon

    # --------------------------------------------------------
    def _start_relocation(self):
        self.state = SearchState.RELOCATING
        self.rotated_deg_this_station = 0.0
        self.step_start_heading = None

        target_lat, target_lon = self._next_station_target()
        self.relocation_target = (target_lat, target_lon)

        self.logger.info(
            f"[ARAMA] {STATION_TIMEOUT_SEC:.0f}sn'de hedef bulunamadı, "
            f"yeni konuma geçiliyor: {target_lat:.6f}, {target_lon:.6f}"
        )

    def _do_relocation(self):
        target_lat, target_lon = self.relocation_target
        distance = calculate_gps_distance(
            self.current_lat, self.current_lon, target_lat, target_lon
        )

        if distance < RELOCATE_TOLERANCE_M:
            self.logger.info("[ARAMA] Yeni konuma ulaşıldı, tarama yeniden başlıyor.")
            self.visited_positions.append((self.current_lat, self.current_lon))
            self.station_start_time = time.monotonic()
            self.state = SearchState.SCANNING
            return

        publish_set_position(self.topics.position_target_pub, target_lat, target_lon)

    # --------------------------------------------------------
    def update(self, detections):
        """Her tick'te (0.1 sn) çağrılır. Hedef bulunduysa True döner."""
        if self.current_lat is None:
            return False  # GPS henüz gelmedi

        # 1) Öncelik: her zaman önce hedef var mı diye bak
        target = self._select_target(detections)
        if target is not None:
            self.state = SearchState.TARGET_FOUND
            self.found_target = target
            self.finished = True
            stop_vehicle(self.topics.cmd_vel_pub)
            self.logger.info(
                f"[ARAMA] Hedef bulundu! mesafe={target['distance']:.2f}m "
                f"açı={target['Buoy angle: ']:.1f}° -> YAKLAŞMA'ya geçiliyor."
            )
            return True

        now = time.monotonic()
        if self.station_start_time is None:
            self.station_start_time = now

        # 2) İstasyon zaman aşımı kontrolü (15-20 sn)
        elapsed = now - self.station_start_time
        if self.state != SearchState.RELOCATING and elapsed > STATION_TIMEOUT_SEC:
            self._start_relocation()

        # 3) Yer değiştiriliyorsa, sadece ona devam et
        if self.state == SearchState.RELOCATING:
            self._do_relocation()
            return False

        # 4) Adımlar arası bekleme
        if self.state == SearchState.STEP_PAUSE:
            if self.step_pause_until is not None and now < self.step_pause_until:
                stop_vehicle(self.topics.cmd_vel_pub)
                return False
            self.state = SearchState.SCANNING

        if self.step_start_heading is None:
            self.step_start_heading = self.current_heading

        rotated_now = abs(self._heading_diff(self.step_start_heading, self.current_heading))

        if rotated_now >= SEARCH_STEP_DEG:
            # 20 derecelik adım tamamlandı -> dur, bekle, tespit şansı ver
            self.rotated_deg_this_station += rotated_now
            self.step_start_heading = None
            stop_vehicle(self.topics.cmd_vel_pub)
            self.state = SearchState.STEP_PAUSE
            self.step_pause_until = now + STEP_SETTLE_SEC

            if self.rotated_deg_this_station >= MAX_SEARCH_ROTATION_DEG:
                self._start_relocation()
            return False

        # 5) Dönmeye devam et
        publish_cmd_vel(self.topics.cmd_vel_pub, linear_x=0.0, angular_z=SEARCH_ANGULAR_SPEED)
        return False


# ============================================================
# ENTEGRASYON NOTU (task3_kamikaze_engagement.py'ye eklenecekler)
# ============================================================
# 1) Dosyanın başına:
#        from teknofest.missions.arama import AramaGorevi
#
# 2) Task3KamikazeEngagement.__init__ içine:
#        self.arama = AramaGorevi(node, mission_topics, target_class=self.target_class)
#
# 3) Task3KamikazeEngagement.update_gps() içine, en sona:
#        self.arama.update_gps(lat, lon, heading)
#
# 4) Task3KamikazeEngagement.update() içinde, en sondaki
#        self.handle_vision_detections(detections)
#    satırını şununla değiştirin:
#
#        if not self.arama.finished:
#            self.arama.update(detections)
#            return
#
#        self.handle_vision_detections(detections)
#
# Bu sayede: hedef bulunana kadar arama çalışır; bulunduğu an
# otomatik olarak mevcut yaklaşma/çarpma mantığına (handle_vision_detections)
# devrediliyor. Ayrıca handle_vision_detections içindeki
#        if not detections:
#            # TODO: ARAMA ALGORİTMASI
#            return
# satırındaki TODO artık gereksiz -- arama zaten üst katmanda hallediliyor,
# o satırı silebilirsiniz.