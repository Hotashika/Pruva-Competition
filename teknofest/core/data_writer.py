# import csv  # IMU CSV logging is disabled for now.
import os
import time
import queue
import threading
import logging
from multiprocessing import shared_memory

import cv2
import numpy as np

from teknofest.config.camera_config import DEPTH_SHAPE, RGB_SHAPE
from teknofest.core import shared_state
from teknofest.vision.detector import BuoyDetector

OUTPUT_DIR = "logs"
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_frames")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")

# CSV_PATH = os.path.join(OUTPUT_DIR, "imu_log.csv")  # IMU CSV logging is disabled for now.
DEPTH_BIN_PATH = os.path.join(OUTPUT_DIR, "depth_stream.bin")  # single append-only file (disabled for now)

VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "run_{ts}.mp4")

# Şartname için en az 1 Hz yeterli.
VIDEO_FPS = 5

# Takip ID istersen True yap.
# True yapılırsa YOLO track() kullanılır ve mümkünse ID bilgisi videoda görünür.
ENABLE_TRACKING = False

logger = logging.getLogger("zed_capture")


def setup_output_dirs():
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)


def attach_shared_memory(name, retries=200, delay=0.1):
    """
    ZED kamerasının açılması (özellikle depth mode ile) birkaç saniye
    sürebilir. Eskiden bu fonksiyon sadece 5 saniye bekleyip pes ediyordu,
    bu yüzden kamera henüz hazır olmadan RuntimeError fırlatıp process'i
    çökertiyordu ("kamera açılmıyor" gibi görünen asıl sebeplerden biri
    buydu). Şimdi ~20 saniyeye kadar bekliyor ve ilerlemeyi logluyor.
    """
    last_error = None

    for attempt in range(retries):
        try:
            shm = shared_memory.SharedMemory(name=name)
            if attempt > 0:
                logger.info("%s shared memory %d. denemede bulundu.", name, attempt + 1)
            return shm
        except FileNotFoundError as exc:
            last_error = exc

            if attempt > 0 and attempt % 20 == 0:
                logger.warning(
                    "%s shared memory hala bulunamadı (%d. deneme). "
                    "Kamera/üretici process açık mı ve önce başlatıldı mı kontrol et.",
                    name,
                    attempt,
                )

            time.sleep(delay)

    raise RuntimeError(
        f"{name} shared memory not found after {retries * delay:.1f}s. "
        f"Kamerayı/üretici process'i (ZED capture) bu script'ten önce başlattığından emin ol."
    ) from last_error


def open_video_writer(video_path, frame_size):
    """
    mp4v codec'i bazı OpenCV kurulumlarında (özellikle pip'ten kurulan
    opencv-python, FFMPEG/H264 desteği olmadan derlendiyse) sessizce
    açılamayabiliyor. Bu durumda video_writer.isOpened() False dönüyor
    ve önceki kodda video hiç yazılmadan process "çalışıyor" gibi
    devam ediyordu. Burada birkaç codec deniyoruz ve hangisinin
    çalıştığını logluyoruz.
    """
    candidates = [
        ("mp4v", video_path),
        ("avc1", video_path),
        ("XVID", os.path.splitext(video_path)[0] + ".avi"),
    ]

    for fourcc_name, path in candidates:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*fourcc_name),
            VIDEO_FPS,
            frame_size,
        )

        if writer.isOpened():
            if path != video_path:
                logger.warning(
                    "mp4v/avc1 codec açılamadı, %s codec ile %s dosyasına yazılıyor.",
                    fourcc_name,
                    path,
                )
            return writer, path

        writer.release()

    return None, None


def disk_writer_worker(q, video_path, frame_size):
    """
    İşlenmiş BGR frame'leri MP4 olarak kaydeder.
    """

    video_writer, opened_path = open_video_writer(video_path, frame_size)

    if video_writer is None:
        logger.error(
            "VideoWriter hiçbir codec ile açılamadı (mp4v/avc1/XVID hepsi başarısız): %s. "
            "OpenCV kurulumunun FFMPEG desteğiyle derlendiğinden emin ol "
            "(örn. 'pip install opencv-python' yerine ffmpeg destekli bir build, "
            "ya da sistemde ffmpeg kurulu olmalı).",
            video_path,
        )

        # Thread kilitlenmesin diye kuyruk kapanana kadar boşalt.
        while True:
            item = q.get()

            if item is None:
                q.task_done()
                break

            q.task_done()

        return

    logger.info(
        "Video kaydı başladı: %s | FPS: %s | Boyut: %s",
        opened_path,
        VIDEO_FPS,
        frame_size,
    )

    written_frames = 0

    try:
        while True:
            item = q.get()

            if item is None:
                q.task_done()
                break

            video_writer.write(item)
            written_frames += 1
            q.task_done()

    except Exception:
        logger.exception("Video yazma sırasında hata oluştu.")

    finally:
        video_writer.release()

        logger.info(
            "Video kaydı tamamlandı: %s | Yazılan frame: %d",
            opened_path,
            written_frames,
        )


def draw_frame_timestamp(frame, timestamp_ms, frame_index):
    """
    Frame üzerine zaman etiketi ve frame numarası yazar.
    """
    timestamp_seconds = timestamp_ms / 1000.0

    timestamp_text = (
        f"Timestamp: {timestamp_ms} ms | "
        f"Time: {timestamp_seconds:.3f} s | "
        f"Frame: {frame_index}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        timestamp_text,
        font,
        font_scale,
        thickness,
    )

    x = 10
    y = 10 + text_height

    # Yazının okunabilmesi için siyah arka plan
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        timestamp_text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame


# noinspection D
def run(frame_lock=None, frame_ready_event=None, stop_event=None):
    setup_output_dirs()

    frame_index = 0
    dropped_frames = 0

    write_queue = queue.Queue(maxsize=100)
    writer_thread = None

    rgb_shm = None
    depth_shm = None
    meta_shm = None
    imu_shm = None

    # Preallocated reusable buffers -> avoids per-frame np/cv2 allocation churn.
    bgra_buf = np.empty(RGB_SHAPE, dtype=np.uint8)
    depth_buf = np.empty(DEPTH_SHAPE, dtype=np.float32)

    h, w = RGB_SHAPE[:2]

    frame_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)

    dh, dw = h // 2, w // 2
    downsampled_depth_buf = np.empty((dh, dw), dtype=np.float32)

    last_drop_log = 0.0
    last_frame_id = 0

    # Video gerçek zamanlı aksın diye sadece VIDEO_FPS kadar frame kaydediyoruz.
    # Örneğin kamera 15 FPS, VIDEO_FPS 5 ise her frame değil, yaklaşık 5 Hz kayıt yapılır.
    record_interval_ms = max(1, int(1000 / VIDEO_FPS))
    last_record_time_ms = None

    # YOLO detector burada bir kere oluşturulur.
    # Döngünün içinde oluşturulmaz; yoksa her frame'de model tekrar yüklenir.
    try:
        buoy_detector = BuoyDetector(
            use_tracking=ENABLE_TRACKING,
        )
    except Exception:
        # Model yüklenemezse (yanlış model yolu, CUDA bulunamadı, vb.)
        # önceki kodda bu hata try/except olmadan yukarı fırlıyordu ve
        # log basılmadan process çöküyordu -> "kamera açılmıyor" gibi
        # yanlış anlaşılabiliyordu. Artık sebep açıkça loglanıyor.
        logger.exception(
            "BuoyDetector (YOLO modeli) yüklenemedi. BUOY_MODEL_PATH doğru mu, "
            "DEVICE ('cuda'/'cpu') sistemde mevcut mu kontrol et."
        )
        raise

    try:
        rgb_shm = attach_shared_memory(shared_state.RGB_SHM_NAME)
        depth_shm = attach_shared_memory(shared_state.DEPTH_SHM_NAME)
        meta_shm = attach_shared_memory(shared_state.META_SHM_NAME)
        imu_shm = attach_shared_memory(shared_state.IMU_SHM_NAME)

        shm_rgb = np.ndarray(
            RGB_SHAPE,
            dtype=np.uint8,
            buffer=rgb_shm.buf,
        )

        shm_depth = np.ndarray(
            DEPTH_SHAPE,
            dtype=np.float32,
            buffer=depth_shm.buf,
        )

        shm_meta = np.ndarray(
            shared_state.META_SHAPE,
            dtype=np.int64,
            buffer=meta_shm.buf,
        )

        shm_imu = np.ndarray(
            shared_state.IMU_SHAPE,
            dtype=np.float64,
            buffer=imu_shm.buf,
        )

        video_path = VIDEO_PATH_TEMPLATE.format(ts=int(time.time()))

        writer_thread = threading.Thread(
            target=disk_writer_worker,
            args=(write_queue, video_path, (w, h)),
            daemon=True,
        )

        writer_thread.start()

        while stop_event is None or not stop_event.is_set():
            if frame_ready_event is not None:
                frame_ready_event.wait(timeout=0.1)
                frame_ready_event.clear()

            if frame_lock is None:
                current_frame_id = int(shm_meta[0])
                timestamp_ms = int(shm_meta[1])
                pitch, yaw, roll = shm_imu.tolist()

                np.copyto(bgra_buf, shm_rgb)
                np.copyto(depth_buf, shm_depth)
            else:
                with frame_lock:
                    current_frame_id = int(shm_meta[0])
                    timestamp_ms = int(shm_meta[1])
                    pitch, yaw, roll = shm_imu.tolist()

                    np.copyto(bgra_buf, shm_rgb)
                    np.copyto(depth_buf, shm_depth)

            if current_frame_id == 0 or current_frame_id == last_frame_id:
                continue

            last_frame_id = current_frame_id

            # BGRA -> BGR
            cv2.cvtColor(
                bgra_buf,
                cv2.COLOR_BGRA2BGR,
                dst=frame_bgr_buf,
            )

            # Depth verisini sistemin diğer tarafları için küçültüyoruz.
            # Tespit için aşağıda full depth_buf kullanıyoruz.
            cv2.resize(
                depth_buf,
                (0, 0),
                dst=downsampled_depth_buf,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_AREA,
            )

            # ------------------------------------------------------------
            # MP4 KAYDI: İşlenmiş kamera verisi
            # - En az 1 Hz
            # - MP4 formatı
            # - Her frame zaman etiketli
            # - Obje bbox + sınıf + güven + mesafe + varsa takip ID
            # ------------------------------------------------------------
            now_record_time_ms = int(time.monotonic() * 1000)

            should_record = (
                    last_record_time_ms is None
                    or now_record_time_ms - last_record_time_ms >= record_interval_ms
            )

            if should_record:
                try:
                    # 1) Tespit / takip yap
                    detections = buoy_detector.detect(
                        frame_bgr_buf,
                        depth_buf,
                    )

                    # 2) Bbox, sınıf, güven, mesafe, ID bilgilerini çiz
                    processed_frame = buoy_detector.draw_detections(
                        frame_bgr_buf,
                        detections,
                    )

                except Exception:
                    logger.exception("Tespit/çizim sırasında hata oluştu. Ham frame kaydedilecek.")
                    processed_frame = frame_bgr_buf.copy()

                # 3) Her kaydedilen frame üzerine zaman etiketi ekle
                draw_frame_timestamp(
                    processed_frame,
                    timestamp_ms=timestamp_ms,
                    frame_index=current_frame_id,
                )

                # 4) MP4'e işlenmiş frame'i yaz
                try:
                    write_queue.put_nowait(processed_frame)
                    last_record_time_ms = now_record_time_ms

                except queue.Full:
                    dropped_frames += 1
                    now = time.monotonic()

                    if now - last_drop_log > 1.0:
                        logger.warning(
                            "Disk write speed is lagging, number of dropped frames: %d",
                            dropped_frames,
                        )
                        last_drop_log = now

            # ------------------------------------------------------------
            # Paylaşılan state güncelleme
            # Diğer modüllerin bozulmaması için latest_frame ham frame olarak bırakıldı.
            # GUI'de de kutulu görüntü görmek istersen:
            # shared_state.latest_frame = processed_frame.copy()
            # yapabilirsin; ama processed_frame sadece kayıt yapılan frame'lerde oluşur.
            # ------------------------------------------------------------
            with shared_state.data_lock:
                shared_state.latest_depth_array = downsampled_depth_buf.copy()
                shared_state.latest_imu = {
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                }
                shared_state.latest_timestamp = timestamp_ms

            with shared_state.frame_lock:
                shared_state.latest_frame = frame_bgr_buf.copy()

            shared_state.data_event.set()
            shared_state.frame_event.set()

            frame_index += 1

    finally:
        print("System shutting down, writing remaining data to disk...")

        if writer_thread is not None:
            write_queue.put(None)
            writer_thread.join()

        for shm in (rgb_shm, depth_shm, meta_shm, imu_shm):
            if shm is not None:
                shm.close()
