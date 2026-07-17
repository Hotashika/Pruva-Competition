from types import SimpleNamespace

from utils.pixhawk_waypoints import mission_items_to_qgc


def mission_item(seq, message_type="MISSION_ITEM_INT"):
    item = SimpleNamespace(
        seq=seq,
        current=1 if seq == 0 else 0,
        frame=0 if seq == 0 else 6,
        command=16,
        param1=0.0,
        param2=0.0,
        param3=0.0,
        param4=0.0,
        x=378729055,
        y=324856979,
        z=3.0,
        autocontinue=1,
    )
    item.get_type = lambda: message_type
    return item


def test_mission_items_to_qgc_sorts_and_scales_int_coordinates():
    content = mission_items_to_qgc([mission_item(1), mission_item(0)])
    lines = content.splitlines()

    assert lines[0] == "QGC WPL 110"
    assert lines[1].startswith("0\t1\t0\t16")
    assert "\t37.8729055\t32.4856979\t3.0\t1" in lines[1]
    assert lines[2].startswith("1\t0\t6\t16")
