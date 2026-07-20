#!/usr/bin/env python3
"""Task 3 yaklaşma: kamera mesafesinin 1/3'ü, GPS ilerleme ve 5 kare doğrulama.

v3 değişikliği:
  - Her segment kendi içinde SEGMENT_TIMEOUT_SEC/STALL_TIMEOUT_SEC ile
    korunuyordu ama toplam yaklaşma fazı (art arda 1/3'lük segmentler
    zinciri) için bir üst sınır yoktu. Artık MAX_APPROACH_SEGMENTS ve
    APPROACH_TOTAL_TIMEOUT_SEC ile toplam segment sayısı ve toplam süre de
    sınırlanıyor; aşılırsa güvenli biçimde LOST'a düşüp aramaya dönülüyor.
"""
import math
import time
from enum import Enum, auto
from utils.mavlink_utilities import calculate_gps_distance, publish_cmd_vel, stop_vehicle

REQUIRED_DISTINCT_FRAMES = 5
TARGET_LOST_TIMEOUT_SEC = 1.0
SEGMENT_FRACTION = 1.0 / 3.0
MIN_SEGMENT_M = 0.40
MAX_SEGMENT_M = 6.0
IMPACT_ENTRY_DISTANCE_M = 1.5
DISTANCE_CONSISTENCY_RATIO = 0.30
MIN_CAMERA_PROGRESS_RATIO = 0.35
ANGLE_KP = 0.02
MAX_ANGULAR_Z = 0.30
APPROACH_SPEED = 0.30
SEGMENT_TIMEOUT_SEC = 12.0
STALL_TIMEOUT_SEC = 4.0
MIN_GPS_PROGRESS_M = 0.25
MAX_TRACK_ANGLE_JUMP_DEG = 30.0
MAX_TRACK_DISTANCE_RATIO = 0.60
MAX_APPROACH_SEGMENTS = 8
APPROACH_TOTAL_TIMEOUT_SEC = 60.0

class ApproachState(Enum):
    WAITING_TARGET = auto(); MOVING_SEGMENT = auto(); CONFIRMING = auto(); DONE = auto(); LOST = auto()

class YaklasmaGorevi:
    def __init__(self, node, mission_topics, target_class, safe_stop_distance=None):
        self.node = node; self.logger = node.get_logger(); self.topics = mission_topics; self.target_class = target_class
        self.impact_entry_distance = float(safe_stop_distance or IMPACT_ENTRY_DISTANCE_M)
        self.state = ApproachState.WAITING_TARGET; self.finished = False; self.target_lost = False
        self.current_lat = self.current_lon = self.current_heading = None
        self.last_seen_time = self.last_processed_frame_id = None; self.latest_target = None
        self.segment_start_lat = self.segment_start_lon = self.segment_goal_m = self.segment_start_distance = None
        self.segment_start_time = self.last_progress_time = None; self.best_travelled = 0.0
        self.confirm_frame_ids = set(); self.confirm_distances = []; self.confirm_angles = []
        self.approach_start_time = None; self.segment_count = 0

    def update_gps(self, lat, lon, heading=None):
        self.current_lat, self.current_lon = float(lat), float(lon)
        if heading is not None: self.update_heading(heading)
    def update_heading(self, heading): self.current_heading = float(heading) % 360.0
    def update_imu(self, gyro_z, accel_x, accel_y): pass

    def _select_target(self, detections):
        candidates=[]
        for det in detections or []:
            try:
                if det.get("class") != self.target_class: continue
                d=float(det["distance"]); a=float(det["Buoy angle: "]); c=float(det.get("confidence",0))
                if not (math.isfinite(d) and d>0 and math.isfinite(a)): continue
                if self.latest_target is not None:
                    pd=float(self.latest_target["distance"]); pa=float(self.latest_target["Buoy angle: "])
                    if abs(a-pa)>MAX_TRACK_ANGLE_JUMP_DEG or abs(d-pd)>max(0.8,pd*MAX_TRACK_DISTANCE_RATIO): continue
                candidates.append((c,det))
            except (KeyError,TypeError,ValueError): continue
        return max(candidates,key=lambda x:x[0])[1] if candidates else None

    def _new_frame(self,detections,frame_id,now):
        if frame_id is None or frame_id==self.last_processed_frame_id: return
        self.last_processed_frame_id=frame_id; target=self._select_target(detections)
        if target is None: return
        self.latest_target=target; self.last_seen_time=now
        if self.state==ApproachState.WAITING_TARGET: self._start_segment(target,now)
        elif self.state==ApproachState.CONFIRMING:
            self.confirm_frame_ids.add(frame_id); self.confirm_distances.append(float(target["distance"])); self.confirm_angles.append(float(target["Buoy angle: "]))
            if len(self.confirm_frame_ids)>=REQUIRED_DISTINCT_FRAMES: self._finish_confirmation(now)

    def _start_segment(self,target,now):
        if None in (self.current_lat,self.current_lon): return
        if self.approach_start_time is None:
            self.approach_start_time = now
        self.segment_count += 1
        d=float(target["distance"]); self.segment_start_distance=d
        self.segment_goal_m=max(MIN_SEGMENT_M,min(MAX_SEGMENT_M,d*SEGMENT_FRACTION))
        self.segment_start_lat,self.segment_start_lon=self.current_lat,self.current_lon
        self.segment_start_time=self.last_progress_time=now; self.best_travelled=0.0; self.state=ApproachState.MOVING_SEGMENT
        self.logger.info(f"[YAKLAŞMA] Kamera={d:.2f}m; hedef ilerleme={self.segment_goal_m:.2f}m (segment {self.segment_count}/{MAX_APPROACH_SEGMENTS}).")

    def _move_segment(self,now):
        if self.latest_target is None or None in (self.current_lat,self.current_lon,self.segment_start_lat,self.segment_start_lon):
            stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); return
        travelled=calculate_gps_distance(self.segment_start_lat,self.segment_start_lon,self.current_lat,self.current_lon)
        if travelled>self.best_travelled+MIN_GPS_PROGRESS_M:
            self.best_travelled=travelled; self.last_progress_time=now
        camera_distance=float(self.latest_target["distance"])
        camera_reduction=max(0.0,self.segment_start_distance-camera_distance)
        gps_ok=travelled>=self.segment_goal_m
        camera_ok=camera_reduction>=self.segment_goal_m*MIN_CAMERA_PROGRESS_RATIO
        if gps_ok and camera_ok:
            stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); self.state=ApproachState.CONFIRMING
            self.confirm_frame_ids.clear(); self.confirm_distances.clear(); self.confirm_angles.clear(); return
        if now-self.segment_start_time>SEGMENT_TIMEOUT_SEC or now-self.last_progress_time>STALL_TIMEOUT_SEC:
            self._lose_target("yaklaşma segmentinde ilerleme doğrulanamadı"); return
        angle=float(self.latest_target["Buoy angle: "])
        angular=max(-MAX_ANGULAR_Z,min(MAX_ANGULAR_Z,ANGLE_KP*angle))
        publish_cmd_vel(self.topics.cmd_vel_pub,linear_x=APPROACH_SPEED,angular_z=angular)

    def _finish_confirmation(self,now):
        ds=self.confirm_distances[-REQUIRED_DISTINCT_FRAMES:]; ang=self.confirm_angles[-REQUIRED_DISTINCT_FRAMES:]
        mean=sum(ds)/len(ds); consistent=max(ds)-min(ds)<=max(0.30,mean*DISTANCE_CONSISTENCY_RATIO)
        centered=sum(abs(a) for a in ang)/len(ang)<=25.0
        if not (consistent and centered): self._lose_target("5 kamera karesi tutarlı değil"); return
        if mean<=self.impact_entry_distance:
            stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); self.state=ApproachState.DONE; self.finished=True; return
        if self.segment_count>=MAX_APPROACH_SEGMENTS:
            self._lose_target("azami segment sayısına ulaşıldı, hedefe güvenli biçimde yaklaşılamadı"); return
        target=dict(self.latest_target); target["distance"]=mean; self._start_segment(target,now)

    def _lose_target(self,reason):
        stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); self.state=ApproachState.LOST
        self.target_lost=True; self.finished=False; self.logger.warning(f"[YAKLAŞMA] {reason}; aramaya dönülecek.")

    def update(self,detections,frame_id=None):
        now=time.monotonic(); self._new_frame(detections,frame_id,now)
        if self.state in (ApproachState.DONE,ApproachState.LOST):
            stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1); return self.state==ApproachState.DONE
        if self.approach_start_time is not None and now-self.approach_start_time>APPROACH_TOTAL_TIMEOUT_SEC:
            self._lose_target("yaklaşma fazı toplam süre sınırını aştı"); return False
        if self.last_seen_time is None or now-self.last_seen_time>TARGET_LOST_TIMEOUT_SEC:
            self._lose_target("hedef kamerada kayboldu"); return False
        if self.state==ApproachState.MOVING_SEGMENT: self._move_segment(now)
        else: stop_vehicle(self.topics.cmd_vel_pub,repeat_count=1)
        return self.finished

    def should_return_to_search(self): return self.state==ApproachState.LOST
    def reset_approach(self):
        stop_vehicle(self.topics.cmd_vel_pub, repeat_count=1)
        self.state = ApproachState.WAITING_TARGET
        self.finished = False
        self.target_lost = False
        self.last_seen_time = None
        self.last_processed_frame_id = None
        self.latest_target = None
        self.segment_start_lat = self.segment_start_lon = None
        self.segment_goal_m = self.segment_start_distance = None
        self.segment_start_time = self.last_progress_time = None
        self.best_travelled = 0.0
        self.confirm_frame_ids.clear()
        self.confirm_distances.clear()
        self.confirm_angles.clear()
        self.approach_start_time = None
        self.segment_count = 0
    def get_status(self):
        return {
            "state": self.state.name,
            "finished": self.finished,
            "distance": None if self.latest_target is None else self.latest_target.get("distance"),
            "segment_count": self.segment_count,
        }