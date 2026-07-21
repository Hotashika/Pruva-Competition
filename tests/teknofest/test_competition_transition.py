import importlib
import sys
import types


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_competition_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", _module("rclpy"))

    class String:
        def __init__(self):
            self.data = ""

    std_msgs = _module("std_msgs")
    std_msgs.__path__ = []
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", _module("std_msgs.msg", String=String))

    state = types.SimpleNamespace(FAILSAFE="failsafe")
    monkeypatch.setitem(
        sys.modules,
        "teknofest.missions.task1_point_tracking",
        _module(
            "teknofest.missions.task1_point_tracking",
            DETECTION_STALE_SEC=3.0,
            MissionState=state,
            Task1Node=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "teknofest.missions.task2_point_tracking_task_in_an_environment_with_obstacle",
        _module(
            "teknofest.missions.task2_point_tracking_task_in_an_environment_with_obstacle",
            MissionState=state,
            Task2PointTrackingWithObstacleAvoidance=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "teknofest.missions.task3_kamikaze_engagement",
        _module(
            "teknofest.missions.task3_kamikaze_engagement",
            MissionState=state,
            Task3KamikazeEngagement=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.mavlink_utilities",
        _module(
            "utils.mavlink_utilities",
            call_set_mode=lambda *_args, **_kwargs: True,
            call_trigger_service=lambda *_args, **_kwargs: True,
            parse_bridge_state=lambda value: value,
            stop_vehicle=lambda _publisher: None,
        ),
    )
    monkeypatch.delitem(
        sys.modules,
        "teknofest.missions.competition_mission",
        raising=False,
    )
    return importlib.import_module("teknofest.missions.competition_mission")


def test_task2_transition_resets_geofence_origin_to_current_gn4(monkeypatch):
    competition = _load_competition_module(monkeypatch)
    stopped = []
    monkeypatch.setattr(competition, "stop_vehicle", stopped.append)

    reset_origins = []
    node = competition.CompetitionNode.__new__(competition.CompetitionNode)
    node.current_lat = 37.9513201
    node.current_lon = 32.500845
    node.task2 = types.SimpleNamespace(
        reset_geofence_origin=lambda lat, lon: reset_origins.append((lat, lon))
    )
    node.mission_topics = types.SimpleNamespace(cmd_vel_pub="cmd_vel")
    node._publish_active_task = lambda: None
    node._enter_competition_failsafe = lambda reason: (_ for _ in ()).throw(
        AssertionError(reason)
    )

    node._transition_to(competition.CompetitionState.PARKUR_2, "task2")

    assert stopped == ["cmd_vel"]
    assert reset_origins == [(37.9513201, 32.500845)]
    assert node.competition_state == competition.CompetitionState.PARKUR_2
    assert node.active_task_name == "task2"
