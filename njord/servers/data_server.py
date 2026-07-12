import json

import numpy as np
from flask import Flask, Response

from njord.core import shared_state

app = Flask(__name__)


def generate():
    while True:
        shared_state.data_event.wait(timeout=1.0)
        with shared_state.data_lock:
            if shared_state.latest_imu is None:
                continue
            imu = shared_state.latest_imu.copy()
            timestamp = shared_state.latest_timestamp
            depth_array = shared_state.latest_depth_array

        if depth_array is not None:
            h, w = depth_array.shape[:2]
            center_depth = float(depth_array[h // 2, w // 2])
            if not np.isfinite(center_depth):
                center_depth = None
        else:
            center_depth = None

        payload = {
            "timestamp": timestamp,
            "imu": imu,
            "center_depth": center_depth
        }

        yield f"data: {json.dumps(payload)}\n\n"


@app.route('/data/stream')
def data_stream():
    return Response(generate(), mimetype='text/event-stream')


@app.route('/health')
def health():
    return 'OK', 200


def start(port=5001):
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
