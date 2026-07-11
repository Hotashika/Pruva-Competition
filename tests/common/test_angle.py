"""
    python3 tests/test_angle.py --device cuda
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import psutil

COMPETITION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_NAME = os.environ.get("COMPETITION_PROJECT", "../njord").lower()
PROJECT_ROOT = os.path.join(COMPETITION_ROOT, PROJECT_NAME)

for path in (COMPETITION_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from config.camera_config import *
from config.vision_config import *
from core.shared_frame_source import close_capture_source, open_or_start_capture_source
from vision.detector import BuoyDetector

# ---------------- Settings ----------------
LOG_INTERVAL_SEC = 2.0  # Terminal reporting interval (seconds)
MAX_DEPTH_M = 40.0

# JSON log settings
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "angle")
SESSION_START_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_PATH = os.path.join(LOG_DIR, f"angle_log_{SESSION_START_TS}.json")

# In-memory log accumulator (not written to disk during loop)
session_log_entries = []

# Box color (BGR) based on Side label - "across" state should be a different color
SIDE_COLORS = {
    "left": (255, 128, 0),
    "right": (0, 128, 255),
    "across": (0, 255, 0),
}

# ---- Global Loop Control ----
is_running = True

# ---- GPU Monitoring (background thread, does NOT BLOCK main loop) ----
_gpu_usage_lock = threading.Lock()
_gpu_usage_value = "0% (no data yet)"
_tegrastats_proc = None


def signal_handler(sig, frame):
    """Safely shuts down the program when Ctrl+C (SIGINT) is sent."""
    global is_running
    print("\n[INFO] Shutdown signal received. Exiting...")
    is_running = False


signal.signal(signal.SIGINT, signal_handler)


def _gpu_monitor_thread_nvidia_smi():
    """For desktop/nvidia-smi systems: reads GPU usage periodically."""
    global _gpu_usage_value
    while is_running:
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                encoding='utf-8', stderr=subprocess.DEVNULL, timeout=1.0
            )
            with _gpu_usage_lock:
                _gpu_usage_value = f"{result.strip()}%"
        except Exception:
            pass
        time.sleep(1.0)


def _gpu_monitor_thread_tegrastats():
    """For Jetson: starts tegrastats ONCE as a continuous process and reads
    its output in the background, writing the last value to _gpu_usage_value.
    The main loop never spawns processes / blocks with readline."""
    global _gpu_usage_value, _tegrastats_proc
    try:
        _tegrastats_proc = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return

    for line in _tegrastats_proc.stdout:
        if not is_running:
            break
        match = re.search(r'GR3D(?:_FREQ)?\s*(\d+)%', line)
        if match:
            with _gpu_usage_lock:
                _gpu_usage_value = f"{match.group(1)}% (Tegrastats)"


def _gpu_monitor_thread_sysfs():
    """If nvidia-smi and tegrastats are absent: periodic sysfs read (fast, non-blocking)."""
    global _gpu_usage_value
    sysfs_paths = [
        "/sys/class/devfreq/17000000.gv11b/device/load",
        "/sys/class/devfreq/17000000.ga10b/device/load",
        "/sys/devices/gpu.0/load"
    ]
    while is_running:
        for path in sysfs_paths:
            try:
                with open(path, 'r') as f:
                    val = float(f.read().strip())
                    text = f"{val / 10.0:.1f}% (sysfs)" if val > 100 else f"{val:.1f}% (sysfs)"
                    with _gpu_usage_lock:
                        _gpu_usage_value = text
                    break
            except Exception:
                continue
        time.sleep(1.0)


def start_gpu_monitor():
    """Selects the appropriate GPU monitoring method once and starts it in the background (daemon thread)."""
    try:
        subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                encoding='utf-8', stderr=subprocess.DEVNULL, timeout=1.0)
        t = threading.Thread(target=_gpu_monitor_thread_nvidia_smi, daemon=True)
        t.start()
        return
    except Exception:
        pass

    try:
        subprocess.check_output(['which', 'tegrastats'], stderr=subprocess.DEVNULL, timeout=1.0)
        t = threading.Thread(target=_gpu_monitor_thread_tegrastats, daemon=True)
        t.start()
        return
    except Exception:
        pass

    t = threading.Thread(target=_gpu_monitor_thread_sysfs, daemon=True)
    t.start()


def get_gpu_usage() -> str:
    """Instantly returns the last value written by the background thread (non-blocking)."""
    with _gpu_usage_lock:
        return _gpu_usage_value


def save_session_log():
    """Writes all accumulated log entries AT ONCE to a JSON file.
    This function is only called when the program closes (in the finally block),
    NOT inside the loop; thus avoiding frequent disk write overhead."""
    if not session_log_entries:
        print("[INFO] No log entries to save, JSON file not created.")
        return
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        payload = {
            "session_start": SESSION_START_TS,
            "session_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "entry_count": len(session_log_entries),
            "entries": session_log_entries,
        }
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Log saved: {LOG_FILE_PATH} ({len(session_log_entries)} entries)")
    except Exception as e:
        print(f"[ERROR] Failed to write log to JSON file: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Angle Detector live ZED2i test")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the model on (default: cpu)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without opening a cv2 window, using terminal logs only")
    parser.add_argument("--no-log", action="store_true",
                        help="Disables JSON logging and memory accumulation for lightweight testing")
    return parser.parse_args()


# noinspection D
def main():
    global is_running
    args = parse_args()
    frame_source = None
    capture_process = None
    capture_stop_event = None

    start_gpu_monitor()

    try:
        print(f"[INFO] Connecting to capture_proc shared frames... ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
        frame_source, capture_process, capture_stop_event = open_or_start_capture_source()

        # Camera settings belong in capture_proc; tests only consume shared frames.

        FX, CX = frame_source.get_calibration()
        print(f"[INFO] Calibration read from capture_proc: fx={FX:.2f}, cx={CX:.2f}")

        print(f"[INFO] Loading BuoyDetector: {BUOY_MODEL_PATH} (device={args.device})")
        detector = BuoyDetector(model_path=BUOY_MODEL_PATH, device=args.device, fx=FX, cx=CX)
        if detector.model is None:
            raise RuntimeError("Failed to load BuoyDetector model.")

        print(f"[INFO] Classes: {detector.class_names}")

        # Vertical reference line at image center (to visually verify right/left/across distinction)
        image_center_x = CAMERA_WIDTH // 2

        last_log_time = time.time()
        frames_since_last_log = 0

        print(f"\n[INFO] Process started. Status will be reported every {LOG_INTERVAL_SEC} seconds.")
        print(f"[INFO] Logs will be kept in memory only, and written at once to '{LOG_FILE_PATH}' on exit.")
        print("[INFO] Press Ctrl+C in the terminal to stop.\n")

        if not args.headless:
            cv2.namedWindow("Angle Detector - Live Test", cv2.WINDOW_NORMAL)

        while is_running:
            t_loop_start = time.time()

            try:
                frame_data = frame_source.read(timeout=1.0)
            except TimeoutError:
                continue
            t_grab = time.time()

            current_time = time.time()
            frames_since_last_log += 1

            t_retrieve = time.time()

            frame = frame_data["frame_bgr"]
            depth_map = frame_data["depth"]
            t_convert = time.time()

            detections = detector.detect(frame, depth_map)
            t_detect = time.time()

            # Reference center line (to visually track the across tolerance)
            if not args.headless:
                cv2.line(frame, (image_center_x, 0), (image_center_x, CAMERA_HEIGHT), (200, 200, 200), 1)

            detected_objects = []
            detected_objects_struct = []

            for det in detections:
                label = det.get("class", "?")
                conf = det.get("confidence", 0.0)
                distance = det.get("distance", None)
                bbox = det.get("bbox", None)
                side = det.get("Buoy side: ", "?")
                angle = det.get("Buoy angle: ", None)

                distance_text = f"{distance:.2f}m" if distance is not None else "N/A"
                angle_text = f"{angle:.1f}°" if angle is not None else "N/A"
                detected_objects.append(f"{label} ({distance_text}, side={side}, angle={angle_text})")
                detected_objects_struct.append({
                    "class": label,
                    "confidence": conf,
                    "distance_m": distance,
                    "side": side,
                    "angle_deg": angle,
                    "bbox": list(map(float, bbox)) if bbox is not None else None,
                })

                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    box_color = SIDE_COLORS.get(side, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Class name + confidence score: top left of box
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Depth info: top right of box (measure width to align right)
                    (text_w, _), _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    depth_x = max(0, x2 - text_w)
                    cv2.putText(frame, distance_text, (depth_x, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

                    # Side / angle info: bottom of box
                    side_angle_text = f"{side} | {angle_text}"
                    cv2.putText(frame, side_angle_text, (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            t_draw = time.time()

            elapsed_time = current_time - last_log_time
            if elapsed_time >= LOG_INTERVAL_SEC:
                fps = frames_since_last_log / elapsed_time
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                gpu_info = get_gpu_usage()

                print(f"--- [ {time.strftime('%H:%M:%S')} System Status ] ---")
                print(f"FPS  : {fps:.1f}")
                print(f"CPU  : {cpu_usage:.1f}%")
                print(f"RAM  : {ram_usage:.1f}%")
                print(f"GPU  : {gpu_info}")
                print(f"Obj  : {', '.join(detected_objects) if detected_objects else 'None'}")
                print("-" * 35)

                # Only added to memory -- NOT WRITTEN to disk (no CPU/disk load).
                if not args.no_log:
                    session_log_entries.append({
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "fps": round(fps, 2),
                        "cpu_percent": cpu_usage,
                        "ram_percent": ram_usage,
                        "gpu": gpu_info,
                        "detections": detected_objects_struct,
                    })

                last_log_time = current_time
                frames_since_last_log = 0

            if not args.headless:
                cv2.imshow("Angle Detector - Live Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    is_running = False
            t_show = time.time()

            # --- TIMING DIAGNOSTICS: Catch slow frames ---
            step_grab = t_grab - t_loop_start
            step_retrieve = t_retrieve - t_grab
            step_convert = t_convert - t_retrieve
            step_detect = t_detect - t_convert
            step_draw = t_draw - t_detect
            step_show = t_show - t_draw
            step_total = t_show - t_loop_start

            if step_total > 0.15:  # Over 150ms = noticeable stutter
                print(f"[SLOW FRAME] TOTAL={step_total * 1000:.0f}ms | "
                      f"grab={step_grab * 1000:.0f}ms retrieve={step_retrieve * 1000:.0f}ms "
                      f"convert={step_convert * 1000:.0f}ms detect={step_detect * 1000:.0f}ms "
                      f"draw={step_draw * 1000:.0f}ms show={step_show * 1000:.0f}ms")

    finally:
        print("[INFO] Releasing resources...")
        is_running = False
        if _tegrastats_proc is not None:
            try:
                _tegrastats_proc.terminate()
            except Exception:
                pass
        close_capture_source(frame_source, capture_process, capture_stop_event)
        if not args.headless:
            cv2.destroyAllWindows()
        save_session_log()
        print("[INFO] Shutdown successful.")


if __name__ == "__main__":
    main()
