import threading

import cv2
from flask import Flask, Response

from njord.core import shared_state

app = Flask(__name__)

_encoded_condition = threading.Condition()
_encoder_start_lock = threading.Lock()
_encoder_thread = None
_latest_encoded_frame_id = 0
_latest_jpeg = None


def _wait_for_source_frame(last_frame_id, timeout=1.0):
    with shared_state.frame_condition:
        ready = shared_state.frame_condition.wait_for(
            lambda: (
                shared_state.latest_frame is not None
                and shared_state.latest_frame_id != last_frame_id
            ),
            timeout=timeout,
        )
        if not ready:
            return None
        return (
            shared_state.latest_frame_id,
            shared_state.latest_frame.copy(),
        )


def _encode_frames():
    global _latest_encoded_frame_id, _latest_jpeg

    last_frame_id = 0
    while True:
        item = _wait_for_source_frame(last_frame_id)
        if item is None:
            continue
        frame_id, frame = item
        last_frame_id = frame_id

        success, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70],
        )
        if not success:
            continue

        with _encoded_condition:
            _latest_encoded_frame_id = frame_id
            _latest_jpeg = jpeg.tobytes()
            _encoded_condition.notify_all()


def _ensure_encoder_started():
    global _encoder_thread

    with _encoder_start_lock:
        if _encoder_thread is not None and _encoder_thread.is_alive():
            return
        _encoder_thread = threading.Thread(
            target=_encode_frames,
            name="njord-jpeg-encoder",
            daemon=True,
        )
        _encoder_thread.start()


def _wait_for_encoded_frame(last_frame_id, timeout=1.0):
    with _encoded_condition:
        ready = _encoded_condition.wait_for(
            lambda: (
                _latest_jpeg is not None
                and _latest_encoded_frame_id != last_frame_id
            ),
            timeout=timeout,
        )
        if not ready:
            return None
        return _latest_encoded_frame_id, _latest_jpeg


def generate():
    _ensure_encoder_started()
    last_frame_id = 0
    while True:
        item = _wait_for_encoded_frame(last_frame_id)
        if item is None:
            continue
        frame_id, jpeg = item
        last_frame_id = frame_id

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return 'OK', 200


def start(port=5000):
    _ensure_encoder_started()
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
