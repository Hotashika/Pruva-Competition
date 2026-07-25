import pytest

from utils.waypoint_server import create_app, overwrite_waypoint_file, start


FIRST_MISSION = "QGC WPL 110\n0\t1\t0\t16\t0\t0\t0\t0\t1\t2\t3\t1\n"
SECOND_MISSION = "QGC WPL 110\n0\t1\t0\t16\t0\t0\t0\t0\t4\t5\t6\t1\n"


def test_server_requires_profile_waypoint_directory():
    with pytest.raises(ValueError, match="waypoint_directory"):
        create_app()
    with pytest.raises(ValueError, match="waypoint_directory"):
        start()


def test_overwrite_waypoint_file_replaces_existing_content(tmp_path):
    destination = overwrite_waypoint_file(
        tmp_path, "njord_task1.waypoints", FIRST_MISSION
    )
    overwrite_waypoint_file(tmp_path, "njord_task1.waypoints", SECOND_MISSION)

    assert destination.read_text(encoding="utf-8") == SECOND_MISSION


def test_upload_endpoint_uses_gui_payload_and_overwrites(tmp_path):
    client = create_app(tmp_path).test_client()
    payload = {
        "type": "mission_waypoints_upload",
        "mission_name": "njord_task1",
        "filename": "njord_task1.waypoints",
        "content": FIRST_MISSION,
    }

    assert client.post("/api/mission/upload_txt", json=payload).status_code == 200
    payload["content"] = SECOND_MISSION
    response = client.post("/api/mission/upload_txt", json=payload)

    assert response.status_code == 200
    assert response.get_json()["overwritten"] is True
    assert (tmp_path / "njord_task1.waypoints").read_text() == SECOND_MISSION


def test_upload_endpoint_rejects_traversal_and_invalid_content(tmp_path):
    client = create_app(tmp_path).test_client()

    traversal = client.post(
        "/api/mission/upload_txt",
        json={"filename": "../outside.waypoints", "content": FIRST_MISSION},
    )
    invalid = client.post(
        "/api/mission/upload_txt",
        json={"filename": "route.waypoints", "content": "not a mission"},
    )

    assert traversal.status_code == 400
    assert invalid.status_code == 400
    assert not (tmp_path.parent / "outside.waypoints").exists()
