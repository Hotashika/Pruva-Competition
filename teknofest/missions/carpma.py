#!/usr/bin/env python3
"""Task 3 çarpma: kamera doğrulaması, sınırlı saldırı ve IMU ile 3 ayrı temas.

v3 değişikliği:
  - Saldırı (STRIKING) sırasında hedef açısı STRIKE_MAX_ANGLE_DEG'i aşarsa
    eskiden araç sadece durup STRIKE_TIMEOUT_SEC dolana kadar pasif
    bekliyor, sonra doğrudan MISS oluyordu. Artık bu durumda ileri hız
    kesilip (linear_x=0.0) saf yaw (angular_z) ile açı toparlanmaya
    çalışılıyor; toparlanırsa saldırıya kaldığı yerden devam ediliyor.
    Üst sınır olarak STRIKE_TIMEOUT_SEC / STRIKE_MAX_GPS_M kontrolleri
    aynen korunuyor, yani sonsuz beklemeye girmiyor.
"""
import math
import time
from collections import deque
from enum import Enum, auto
from utils.mavlink_utilities import (
    calculate_bearing,
    calculate_gps_distance,
    publish_cmd_vel,
    stop_vehicle,
)

REQUIRED_HITS=3
CAMERA_CONFIRM_FRAMES=5
TARGET_LOST_TIMEOUT_SEC=1.0
STRIKE_SPEED=0.22
STRIKE_TIMEOUT_SEC=6.0
STRIKE_MAX_GPS_M=4.0
STRIKE_MAX_ANGLE_DEG=30.0
MAX_ANGULAR_Z=0.30
ANGLE_KP=0.025
BACKOFF_SPEED=-0.18
BACKOFF_TIMEOUT_SEC=5.0
BACKOFF_REQUIRED_GPS_M=0.60
BACKOFF_REQUIRED_CAMERA_INCREASE_M=0.40
BACKOFF_MAX_HEADING_CORRECTION_RAD=0.18
COOLDOWN_SEC=0.8
IMPACT_DELTA_THRESHOLD=4.0
IMPACT_CONSECUTIVE_SAMPLES=2
IMPACT_MIN_FORWARD_SPEED=0.10
IMPACT_MAX_CAMERA_DISTANCE_M=2.0
IMPACT_TARGET_FRESH_SEC=0.60
BASELINE_WINDOW=20
CONF_DISTANCE_RATIO=0.30
CONF_ANGLE_SPREAD_DEG=18.0
MAX_TRACK_ANGLE_JUMP_DEG=30.0
MAX_TRACK_DISTANCE_RATIO=0.60
BACKOFF_MAX_LATERAL_M=1.0
DEFAULT_MIN_TARGET_CONFIDENCE=0.65

class CarpmaState(Enum):
    CAMERA_CONFIRM=auto(); STRIKING=auto(); BACKING_OFF=auto(); COOLDOWN=auto(); COMPLETE=auto(); MISSED=auto()

class CarpmaGorevi:
    def __init__(self,node,mission_topics,target_class,
                 min_target_confidence=DEFAULT_MIN_TARGET_CONFIDENCE,
                 impact_delta_threshold=IMPACT_DELTA_THRESHOLD):
        self.node=node; self.logger=node.get_logger(); self.topics=mission_topics; self.target_class=target_class
        self.min_target_confidence=float(min_target_confidence)
        self.impact_delta_threshold=float(impact_delta_threshold)
        self.state=CarpmaState.CAMERA_CONFIRM; self.finished=False; self.success=False; self.hit_count=0
        self.current_lat=self.current_lon=self.current_heading=None
        self.latest_target=None; self.last_seen_time=None; self.last_processed_frame_id=None
        self.confirm_frame_ids=set(); self.confirm_distances=[]; self.confirm_angles=[]
        self.camera_confirm_start_time=None
        self.current_speed=0.0; self.accel_baseline=deque(maxlen=BASELINE_WINDOW); self.spike_count=0
        self.last_impact_time=None; self.strike_start_time=None; self.strike_start_lat=None; self.strike_start_lon=None
        self.backoff_start_time=None; self.backoff_start_lat=None; self.backoff_start_lon=None; self.backoff_start_distance=None
        self.backoff_heading_deg=None
        self.cooldown_start_time=None

    def update_gps(self,lat,lon,heading=None):
        self.current_lat,self.current_lon=float(lat),float(lon)
        if heading is not None: self.update_heading(heading)
    def update_heading(self,heading): self.current_heading=float(heading)%360.0

    def update_imu(self,ax,ay,az):
        mag=math.sqrt(ax*ax+ay*ay+az*az)
        if self.state!=CarpmaState.STRIKING or self.current_speed<IMPACT_MIN_FORWARD_SPEED:
            self.accel_baseline.append(mag); self.spike_count=0; return
        target_is_fresh=(
            self.last_seen_time is not None
            and time.monotonic()-self.last_seen_time<=IMPACT_TARGET_FRESH_SEC
        )
        if (
            self.latest_target is None
            or float(self.latest_target.get('distance',999))>IMPACT_MAX_CAMERA_DISTANCE_M
            or not target_is_fresh
        ):
            # Saldırı başlamışken bayat/uzak hedef bağlamında gelen sıçramayı
            # ne temas say ne de sakin-su baseline'ını kirlet.
            self.spike_count=0; return
        if len(self.accel_baseline)<max(5,BASELINE_WINDOW//2): self.accel_baseline.append(mag); return
        baseline=sum(self.accel_baseline)/len(self.accel_baseline); delta=abs(mag-baseline)
        self.spike_count=self.spike_count+1 if delta>=self.impact_delta_threshold else 0
        if delta<self.impact_delta_threshold: self.accel_baseline.append(mag)
        if self.spike_count>=IMPACT_CONSECUTIVE_SAMPLES: self._register_hit(delta)

    def _select_target(self,detections):
        valid=[]; reference=self.latest_target
        for det in detections or []:
            try:
                if det.get('class')!=self.target_class: continue
                d=float(det['distance']); a=float(det['Buoy angle: ']); c=float(det.get('confidence',0))
                if c<self.min_target_confidence: continue
                if not (math.isfinite(d) and d>0 and math.isfinite(a)): continue
                if reference is not None:
                    old_d=float(reference['distance']); old_a=float(reference['Buoy angle: '])
                    if abs(a-old_a)>MAX_TRACK_ANGLE_JUMP_DEG: continue
                    if abs(d-old_d)>max(0.8,old_d*MAX_TRACK_DISTANCE_RATIO): continue
                valid.append((c,det))
            except (KeyError,TypeError,ValueError): continue
        return max(valid,key=lambda x:x[0])[1] if valid else None

    def _process_frame(self,detections,frame_id,now):
        if frame_id is None or frame_id==self.last_processed_frame_id: return None
        self.last_processed_frame_id=frame_id; target=self._select_target(detections)
        if target is None:
            # Onay kareleri arka arkaya olmalı; aradaki gerçek bir kaçırma
            # önceki aday dizisini geçersiz kılar.
            if self.state==CarpmaState.CAMERA_CONFIRM:
                self.confirm_frame_ids.clear(); self.confirm_distances.clear(); self.confirm_angles.clear()
            return False
        self.latest_target=target; self.last_seen_time=now
        if self.state==CarpmaState.CAMERA_CONFIRM:
            self.confirm_frame_ids.add(frame_id); self.confirm_distances.append(float(target['distance'])); self.confirm_angles.append(float(target['Buoy angle: ']))
            if len(self.confirm_frame_ids)>=CAMERA_CONFIRM_FRAMES: self._finish_camera_confirmation(now)
        return True

    def _finish_camera_confirmation(self,now):
        ds=self.confirm_distances[-CAMERA_CONFIRM_FRAMES:]; ang=self.confirm_angles[-CAMERA_CONFIRM_FRAMES:]
        mean=sum(ds)/len(ds)
        distance_ok=max(ds)-min(ds)<=max(0.25,mean*CONF_DISTANCE_RATIO)
        angle_ok=max(ang)-min(ang)<=CONF_ANGLE_SPREAD_DEG and sum(abs(a) for a in ang)/len(ang)<=20.0
        if not(distance_ok and angle_ok): self._miss('5 kamera karesi tutarlı değil'); return
        self.state=CarpmaState.STRIKING; self.strike_start_time=now
        self.strike_start_lat,self.strike_start_lon=self.current_lat,self.current_lon
        # CAMERA_CONFIRM boyunca gerçek IMU baseline'ı zaten toplandı.
        # Temas sürüşü başlarken bunu silmek, ilk saniyelerde fiziksel
        # teması algılayamamaya neden olurdu.
        self.spike_count=0
        self.logger.info('[ÇARPMA] Kamera doğrulandı; sınırlı fiziksel temas sürüşü başladı.')

    def _register_hit(self,delta):
        now=time.monotonic()
        if self.last_impact_time is not None and now-self.last_impact_time<1.0: return
        self.last_impact_time=now; self.hit_count+=1; self.current_speed=0.0; self.spike_count=0
        stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1)
        self.logger.info(f'[ÇARPMA] Gerçek IMU teması {self.hit_count}/{REQUIRED_HITS}; Δ={delta:.2f}')
        if self.hit_count>=REQUIRED_HITS:
            self.state=CarpmaState.COMPLETE; self.finished=True; self.success=True; return
        self.state=CarpmaState.BACKING_OFF; self.backoff_start_time=now
        self.backoff_start_lat,self.backoff_start_lon=self.current_lat,self.current_lon
        self.backoff_start_distance=None if self.latest_target is None else float(self.latest_target['distance'])
        self.backoff_heading_deg=self.current_heading

    def _strike(self,now):
        if self.latest_target is None: stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); return
        if now-self.strike_start_time>STRIKE_TIMEOUT_SEC: self._miss('temas zaman aşımı'); return
        if None not in (self.strike_start_lat,self.strike_start_lon,self.current_lat,self.current_lon):
            moved=calculate_gps_distance(self.strike_start_lat,self.strike_start_lon,self.current_lat,self.current_lon)
            if moved>STRIKE_MAX_GPS_M: self._miss('temas olmadan azami ilerleme aşıldı'); return
        angle=float(self.latest_target['Buoy angle: '])
        angular=max(-MAX_ANGULAR_Z,min(MAX_ANGULAR_Z,ANGLE_KP*angle))
        if abs(angle)>STRIKE_MAX_ANGLE_DEG:
            # Açı çok büyükse pes etmek yerine ileri hızı kesip saf yaw ile
            # toparlamaya çalış; STRIKE_TIMEOUT_SEC/STRIKE_MAX_GPS_M üst
            # sınırları hâlâ geçerli, yani sonsuz beklemeye girmez.
            self.current_speed=0.0
            publish_cmd_vel(self.topics.cmd_vel_pub,linear_x=0.0,angular_z=angular)
            return
        self.current_speed=STRIKE_SPEED
        publish_cmd_vel(self.topics.cmd_vel_pub,linear_x=STRIKE_SPEED,angular_z=angular)

    def _backoff(self,now):
        gps_far=False; camera_far=False
        if None not in (self.backoff_start_lat,self.backoff_start_lon,self.current_lat,self.current_lon,self.backoff_heading_deg):
            moved=calculate_gps_distance(self.backoff_start_lat,self.backoff_start_lon,self.current_lat,self.current_lon)
            bearing=calculate_bearing(self.backoff_start_lat,self.backoff_start_lon,self.current_lat,self.current_lon)
            angle=math.radians((bearing-self.backoff_heading_deg+180.0)%360.0-180.0)
            backward_progress=-moved*math.cos(angle)
            lateral=abs(moved*math.sin(angle))
            gps_far=(backward_progress>=BACKOFF_REQUIRED_GPS_M and lateral<=BACKOFF_MAX_LATERAL_M)
        if self.backoff_start_distance is not None and self.latest_target is not None:
            camera_far=float(self.latest_target['distance'])-self.backoff_start_distance>=BACKOFF_REQUIRED_CAMERA_INCREASE_M
        # İkinci vuruşu gerçekten ayrı bir fiziksel temas sayabilmek için hem
        # Pixhawk GPS'i geri hareketi hem de kamera mesafesi artışı doğrulasın.
        if gps_far and camera_far:
            stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); self.current_speed=0.0
            self.state=CarpmaState.COOLDOWN; self.cooldown_start_time=now; return
        if now-self.backoff_start_time>BACKOFF_TIMEOUT_SEC: self._miss('temastan sonra gerçek uzaklaşma doğrulanamadı'); return
        heading_error=0.0
        if self.current_heading is not None and self.backoff_heading_deg is not None:
            heading_error=(self.backoff_heading_deg-self.current_heading+180.0)%360.0-180.0
        heading_correction=max(
            -BACKOFF_MAX_HEADING_CORRECTION_RAD,
            min(BACKOFF_MAX_HEADING_CORRECTION_RAD,math.radians(heading_error)),
        )
        self.current_speed=BACKOFF_SPEED
        publish_cmd_vel(
            self.topics.cmd_vel_pub,
            linear_x=BACKOFF_SPEED,
            angular_z=heading_correction,
        )

    def _miss(self,reason):
        stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); self.current_speed=0.0
        self.state=CarpmaState.MISSED; self.finished=True; self.success=False
        self.logger.warning(f'[ÇARPMA] {reason}; aramaya dönülecek.')


    def update(self,detections,frame_id=None):
        now=time.monotonic()
        if self.state==CarpmaState.CAMERA_CONFIRM and self.camera_confirm_start_time is None:
            self.camera_confirm_start_time=now
        frame_status=self._process_frame(detections,frame_id,now)
        if self.state in (CarpmaState.COMPLETE,CarpmaState.MISSED):
            stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); return True
        if self.state==CarpmaState.STRIKING and frame_status is False:
            self._miss('fiziksel temas sürüşünde hedef yeni kamera karesinde kayboldu'); return True
        if self.state in (CarpmaState.CAMERA_CONFIRM,CarpmaState.STRIKING):
            if self.last_seen_time is None:
                # Yaklaşmanın son karesini çarpma onayında tekrar kullanma;
                # ilk yeni gerçek kamera karesini kısa süre bekle.
                if (
                    self.state==CarpmaState.CAMERA_CONFIRM
                    and now-self.camera_confirm_start_time<=TARGET_LOST_TIMEOUT_SEC
                ):
                    stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1)
                    return False
                self._miss('hedef kamerada kayboldu'); return True
            if now-self.last_seen_time>TARGET_LOST_TIMEOUT_SEC:
                self._miss('hedef kamerada kayboldu'); return True
        if self.state==CarpmaState.CAMERA_CONFIRM: stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1)
        elif self.state==CarpmaState.STRIKING: self._strike(now)
        elif self.state==CarpmaState.BACKING_OFF: self._backoff(now)
        elif self.state==CarpmaState.COOLDOWN:
            stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1)
            if now-self.cooldown_start_time>=COOLDOWN_SEC:
                self.state=CarpmaState.CAMERA_CONFIRM; self.confirm_frame_ids.clear(); self.confirm_distances.clear(); self.confirm_angles.clear(); self.last_seen_time=None; self.camera_confirm_start_time=now
        return self.state==CarpmaState.COMPLETE

    def should_retry_search(self): return self.state==CarpmaState.MISSED
    def reset_carpma(self):
        stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1)
        target=self.target_class; node=self.node; topics=self.topics
        self.__init__(
            node, topics, target,
            min_target_confidence=self.min_target_confidence,
            impact_delta_threshold=self.impact_delta_threshold,
        )
    def get_status(self): return {'state':self.state.name,'finished':self.finished,'success':self.success,'hit_count':self.hit_count,'required_hits':REQUIRED_HITS}
