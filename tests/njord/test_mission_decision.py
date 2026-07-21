from njord.core.mission_decision import build_mission_decision


def test_task2_avoidance_explains_action_and_colreg():
    decision = build_mission_decision(
        2,
        "AVOIDING",
        current_target=1,
        target_count=4,
    )

    assert decision["stage"] == "AVOIDING"
    assert decision["action"] == "Turn to starboard"
    assert decision["collision_risk"] is True
    assert decision["colreg_rule"] == "RULE 15/16"
    assert decision["current_target"] == 2
    assert decision["progress_percent"] == 25.0


def test_task3_search_explains_ar_tag_stage():
    decision = build_mission_decision(3, "SEARCH_DOCK", target_count=3)

    assert decision["stage"] == "SEARCHING"
    assert "AR tag" in decision["action"]
    assert "correct dock" in decision["reason"]


def test_finished_mission_reports_full_progress():
    decision = build_mission_decision(4, "FINISHED", current_target=2, target_count=5)

    assert decision["current_target"] == 5
    assert decision["progress_percent"] == 100.0
