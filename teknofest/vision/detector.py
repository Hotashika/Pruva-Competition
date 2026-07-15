import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from teknofest.config.camera_config import CAMERA_WIDTH
from teknofest.config.vision_config import DEVICE, BUOY_MODEL_PATH, TOLERANCE_RATIO, TOLARANCE_DEG
from teknofest.vision.depth_utils import get_distance_from_bbox

_logger = logging.getLogger("teknofest.vision")


class BaseYOLODetector:
    def __init__(self, model_path, device=DEVICE, use_tracking=False, tracker=None):
        model_p = Path(model_path)

        if not model_p.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            model_p = project_root / model_path

        self.model = YOLO(str(model_p))
        self.device = device
        self.class_names = self.model.names

        # DÜZELTME: Modelin gerçek sınıf isimlerini sahaya çıkmadan önce
        # görünür kılıyoruz. arama.py / task3_kamikaze.py "red_buoy",
        # "green_buoy", "black_buoy" bekliyor — model bu isimleri farklı
        # üretiyorsa (örn. sadece "buoy", ya da "red"), d.get("class") ==
        # target_class eşleşmesi sessizce hiç tutmaz ve hiçbir tespit
        # aday sayılmaz. Bu log, sorunu ilk çalıştırmada anında gösterir.
        _logger.info(f"{self.__class__.__name__} model sınıfları yüklendi: {self.class_names}")

        # Takip istenirse True yapılabilir.
        # Ultralytics track() kullanır ve bbox yanında track_id üretmeye çalışır.
        self.use_tracking = use_tracking
        self.tracker = tracker

    # noinspection D
    def detect(self, bgr_image, depth_array):
        """
        YOLO ile nesne tespiti yapar.

        Çıktı formatı:
        [
            {
                "class": "buoy",
                "confidence": 0.91,
                "distance": 4.72,
                "bbox": [x1, y1, x2, y2],
                "track_id": 3 veya None
            }
        ]
        """
        t0 = time.time()

        if self.use_tracking:
            if self.tracker is None:
                results = self.model.track(
                    bgr_image,
                    device=self.device,
                    persist=True,
                    verbose=False,
                )
            else:
                results = self.model.track(
                    bgr_image,
                    device=self.device,
                    tracker=self.tracker,
                    persist=True,
                    verbose=False,
                )
        else:
            results = self.model(
                bgr_image,
                device=self.device,
                verbose=False,
            )

        t1 = time.time()

        detections = []

        if not results:
            return detections

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return detections

        image_h, image_w = bgr_image.shape[:2]

        xyxy_all = boxes.xyxy.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()

        track_ids = None

        if hasattr(boxes, "id") and boxes.id is not None:
            track_ids = boxes.id.int().cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, xyxy_all[i])

            # Görüntü sınırları dışına taşmayı engelle
            x1 = max(0, min(x1, image_w - 1))
            y1 = max(0, min(y1, image_h - 1))
            x2 = max(0, min(x2, image_w - 1))
            y2 = max(0, min(y2, image_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(cls_all[i])
            conf = float(conf_all[i])
            class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")

            bbox = [x1, y1, x2, y2]

            try:
                distance = get_distance_from_bbox(
                    depth_array,
                    bbox,
                    method="median",
                )
            except Exception:
                distance = float("nan")

            # DÜZELTME: distance NaN geldiğinde (derinlik verisi
            # okunamadığında) arama.py'deki filtre
            # (d.get("distance", -1) > 0.5) NaN karşılaştırmasında
            # sessizce False döner (Python'da NaN > x her zaman False'tur)
            # ve tespit hiçbir uyarı vermeden elenir. Sahada "hedefi
            # görüyor ama bulmuyor" belirtisinin bir diğer olası kaynağı
            # budur — derinlik kamerası menzil dışı/gürültülüyse bu satır
            # tetiklenir.
            if not np.isfinite(distance):
                _logger.debug(
                    f"{class_name} tespiti için mesafe NaN/sonsuz geldi "
                    f"(bbox={bbox}); bu tespit arama filtresinde elenecek."
                )

            track_id = None

            if track_ids is not None:
                track_id = int(track_ids[i])

            detections.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "distance": round(float(distance), 2) if np.isfinite(distance) else float("nan"),
                "bbox": bbox,
                "track_id": track_id,
            })

        t2 = time.time()

        return detections

    def draw_detections(self, bgr_image, detections):
        """
        Tespit kutularını, sınıf adını, güven oranını, mesafeyi,
        açı/yön bilgisini ve varsa takip ID bilgisini görüntü üzerine çizer.
        """
        output_frame = bgr_image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            class_name = detection.get("class", "unknown")
            confidence = detection.get("confidence", 0.0)
            distance = detection.get("distance")
            track_id = detection.get("track_id")

            label_parts = [
                class_name,
                f"{confidence:.2f}",
            ]

            if distance is not None and np.isfinite(distance):
                label_parts.append(f"{distance:.2f} m")

            angle = detection.get("Buoy angle: ")

            if angle is not None:
                label_parts.append(f"{angle:.1f} deg")

            side = detection.get("Buoy side: ")

            if side is not None:
                label_parts.append(side)

            if track_id is not None:
                label_parts.append(f"ID:{track_id}")

            label = " | ".join(label_parts)

            # Obje kutusu
            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )

            text_y = max(y1 - 8, text_height + 8)

            # Yazının okunabilmesi için siyah arka plan
            cv2.rectangle(
                output_frame,
                (x1, text_y - text_height - 6),
                (x1 + text_width + 6, text_y + baseline),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                output_frame,
                label,
                (x1 + 3, text_y - 3),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return output_frame


class BuoyDetector(BaseYOLODetector):
    def __init__(
            self,
            model_path=BUOY_MODEL_PATH,
            device=DEVICE,
            fx=None,
            cx=None,
            use_tracking=False,
            tracker=None,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            use_tracking=use_tracking,
            tracker=tracker,
        )

        self.fx = fx
        self.cx = cx if cx is not None else CAMERA_WIDTH / 2

        # DÜZELTME: fx=None sessizce geçilmesin. fx olmadan
        # _compute_angle() HER ZAMAN None döner, bu da arama.py'deki
        # d.get("Buoy angle: ") is not None filtresinin TÜM tespitleri
        # elemesine yol açar — YOLO dubayı doğru tespit etse bile arama
        # hiçbir aday bulamaz. Bu, sahada en sık karşılaşılan "arama
        # gerekli bilgiyi almıyor" nedenlerinden biridir.
        if self.fx is None:
            _logger.warning(
                "BuoyDetector fx=None ile başlatıldı! 'Buoy angle: ' alanı "
                "HER ZAMAN None dönecek ve arama/task3 tarafındaki "
                "d.get('Buoy angle: ') is not None filtresi TÜM tespitleri "
                "eleyecek. vision_node'u --fx <kameranin_piksel_odak_uzakligi> "
                "ile başlatın (--cx opsiyonel, verilmezse CAMERA_WIDTH/2 "
                "kullanılıyor)."
            )

    def detect(self, bgr_image, depth_array):
        detections = super().detect(bgr_image, depth_array)

        for det in detections:
            det["Buoy angle: "] = self._compute_angle(det)
            det["Buoy side: "] = self._compute_side(det)

        return detections

    def _compute_angle(self, detection):
        bbox = detection["bbox"]
        bbox_center_x = (bbox[0] + bbox[2]) / 2

        if self.fx is None:
            return None

        angle_rad = np.arctan2(bbox_center_x - self.cx, self.fx)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _compute_side(self, detection):
        angle_deg = detection.get("Buoy angle: ")

        if angle_deg is None:
            angle_deg = self._compute_angle(detection)

        if angle_deg is not None:
            if abs(angle_deg) <= TOLARANCE_DEG:
                return "across"
            elif angle_deg > 0:
                return "right"
            else:
                return "left"

        # Fallback: fx yoksa piksel merkezine göre sağ/sol/karşı belirle
        bbox = detection["bbox"]
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        image_center_x = CAMERA_WIDTH / 2

        tolerance_px = CAMERA_WIDTH * TOLERANCE_RATIO
        diff = bbox_center_x - image_center_x

        if abs(diff) <= tolerance_px:
            return "across"
        elif diff > 0:
            return "right"
        else:
            return "left"