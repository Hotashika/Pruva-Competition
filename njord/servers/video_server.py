import cv2
from flask import Flask, Response

from njord.core import shared_state

app = Flask(__name__)


def generate():
    while True:
        shared_state.frame_event.wait(timeout=1.0)
        with shared_state.frame_lock:
            if shared_state.latest_frame is None:
                continue
            frame = shared_state.latest_frame.copy()

        success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return 'OK', 200


def start(port=5000):
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
