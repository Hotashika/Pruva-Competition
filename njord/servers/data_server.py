import json

from flask import Flask, Response

from njord.core import shared_state

app = Flask(__name__)


def _wait_for_data(last_data_id, timeout=1.0):
    with shared_state.data_condition:
        ready = shared_state.data_condition.wait_for(
            lambda: (
                shared_state.latest_imu is not None
                and shared_state.latest_data_id != last_data_id
            ),
            timeout=timeout,
        )
        if not ready:
            return None
        return {
            "data_id": shared_state.latest_data_id,
            "timestamp": shared_state.latest_timestamp,
            "imu": shared_state.latest_imu.copy(),
            "center_depth": shared_state.latest_center_depth,
        }


def generate():
    last_data_id = 0
    while True:
        snapshot = _wait_for_data(last_data_id)
        if snapshot is None:
            continue
        last_data_id = snapshot.pop("data_id")

        yield f"data: {json.dumps(snapshot)}\n\n"


@app.route('/data/stream')
def data_stream():
    return Response(generate(), mimetype='text/event-stream')


@app.route('/health')
def health():
    return 'OK', 200


def start(port=5001):
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
