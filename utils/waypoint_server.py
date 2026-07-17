"""HTTP receiver for waypoint files sent by Pruva-GUI."""

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAYPOINT_DIRECTORY = REPOSITORY_ROOT / "waypoints"


def _validated_filename(value):
    """Return a safe QGC waypoint filename without accepting path traversal."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("filename is required")

    filename = value.strip()
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("filename must not contain a directory")
    if Path(filename).suffix.lower() != ".waypoints":
        raise ValueError("filename must have a .waypoints extension")
    return filename


def _validated_content(value):
    if not isinstance(value, str):
        raise ValueError("content must be text")
    if not value.lstrip("\ufeff").startswith("QGC WPL"):
        raise ValueError("content must be in QGroundControl WPL format")
    return value


def overwrite_waypoint_file(directory, filename, content):
    """Atomically replace a waypoint file, creating its directory if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _validated_filename(filename)
    content = _validated_content(content)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=directory,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination


def create_app(waypoint_directory=DEFAULT_WAYPOINT_DIRECTORY):
    app = Flask(__name__)
    app.config["WAYPOINT_DIRECTORY"] = str(waypoint_directory)

    @app.post("/api/mission/upload_txt")
    def upload_waypoints():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(ok=False, success=False, error="JSON body is required"), 400
        if payload.get("type") not in (None, "mission_waypoints_upload"):
            return jsonify(ok=False, success=False, error="unsupported message type"), 400

        try:
            filename = _validated_filename(payload.get("filename"))
            destination_before_write = Path(app.config["WAYPOINT_DIRECTORY"]) / filename
            existed_before_write = destination_before_write.exists()
            destination = overwrite_waypoint_file(
                app.config["WAYPOINT_DIRECTORY"],
                filename,
                payload.get("content"),
            )
        except (OSError, ValueError) as exc:
            app.logger.warning("Waypoint upload rejected: %s", exc)
            return jsonify(ok=False, success=False, error=str(exc)), 400

        app.logger.info(
            "Waypoint upload saved: path=%s bytes=%d overwritten=%s",
            destination,
            len(payload["content"].encode("utf-8")),
            existed_before_write,
        )

        return jsonify(
            ok=True,
            success=True,
            mission_id=payload.get("mission_name") or destination.stem,
            mission_name=payload.get("mission_name") or destination.stem,
            filename=destination.name,
            path=str(destination),
            overwritten=existed_before_write,
            message="Waypoint file saved.",
        )

    @app.get("/health")
    def health():
        return jsonify(ok=True, service="waypoint-upload")

    return app


app = create_app()


def start(port=8000):
    print(
        "[WAYPOINT] Receiver ready: "
        f"http://0.0.0.0:{port}/api/mission/upload_txt -> "
        f"{DEFAULT_WAYPOINT_DIRECTORY.resolve()}",
        flush=True,
    )
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
