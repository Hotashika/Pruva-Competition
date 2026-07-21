import json


DECISION_TOPIC = "/mission/decision"


_DECISIONS = {
    1: {
        "INIT": ("WAITING", "Waiting for mission readiness", "Sensors and vehicle state are being checked", "", None),
        "NAVIGATING": ("NAVIGATING", "Navigate to next waypoint", "Following the planned buoy-course route", "", False),
        "AVOIDING": ("AVOIDING", "Pass the detected course marker", "Detected buoy/cardinal mark affects safe path", "", True),
        "FINISHED": ("FINISHED", "Stop and hold position", "Task 1 route is complete", "", False),
        "FAILSAFE": ("FAILSAFE", "Stop and request HOLD", "A navigation safety prerequisite failed", "", None),
    },
    2: {
        "INIT": ("WAITING", "Waiting for mission readiness", "Sensors and vehicle state are being checked", "", None),
        "NAVIGATING": ("NAVIGATING", "Navigate to next waypoint", "No current collision risk", "", False),
        "STAND_ON": ("STAND-ON", "Maintain course and speed", "Stand-on vessel while monitoring collision risk", "RULE 17", True),
        "AVOIDING": ("AVOIDING", "Turn to starboard", "Collision risk requires give-way manoeuvre", "RULE 15/16", True),
        "FINISHED": ("FINISHED", "Stop and hold position", "Task 2 route is complete", "", False),
        "FAILSAFE": ("FAILSAFE", "Stop and request HOLD", "A collision-safety prerequisite failed", "", None),
    },
    3: {
        "WAIT_START": ("WAITING", "Wait for docking start", "Docking sequence has not started", "", None),
        "GO_TO_APPROACH_POINT": ("APPROACH", "Navigate to dock approach point", "Using the configured GNSS approach point", "", False),
        "SEARCH_DOCK": ("SEARCHING", "Search for target AR tag", "Scanning camera feed for the correct dock", "", False),
        "ALIGN_TO_TAG": ("ALIGNING", "Align vessel with target AR tag", "Confirmed tag is outside alignment tolerance", "", False),
        "FINAL_APPROACH": ("DOCKING", "Move into the selected dock", "Target AR tag is confirmed and aligned", "", False),
        "HOLD_POSITION": ("HOLDING", "Hold position in dock", "Required berth hold time is active", "", False),
        "REVERSE_EXIT": ("EXITING", "Reverse out of dock", "Docking hold is complete", "", False),
        "GO_TO_EXIT_POINT": ("EXITING", "Navigate to dock exit point", "Clearing the docking area safely", "", False),
        "MODE_FINISHED": ("MODE COMPLETE", "Prepare the next docking mode", "Current docking mode is complete", "", False),
        "FINISHED": ("FINISHED", "Stop and hold position", "Task 3 docking sequence is complete", "", False),
        "FAILSAFE": ("FAILSAFE", "Stop docking manoeuvre", "A docking safety prerequisite failed", "", None),
    },
    4: {
        "INIT": ("PLANNING", "Optimize waypoint visit order", "Computing the shortest safe open route", "", None),
        "NAVIGATING": ("NAVIGATING", "Navigate optimized route", "Following the shortest planned waypoint order", "", False),
        "AVOIDING": ("AVOIDING", "Pass detected obstacle safely", "A buoy blocks the optimized route", "", True),
        "FINISHED": ("FINISHED", "Stop and hold position", "All Task 4 waypoints were visited", "", False),
        "FAILSAFE": ("FAILSAFE", "Stop and request HOLD", "A route-safety prerequisite failed", "", None),
    },
}


def build_mission_decision(
        mission_number,
        state,
        current_target=0,
        target_count=0,
        action=None,
        reason=None,
):
    mission_number = int(mission_number)
    state_name = getattr(state, "name", state)
    state_name = str(state_name or "UNKNOWN").strip().upper()
    stage, default_action, default_reason, colreg_rule, collision_risk = _DECISIONS.get(
        mission_number, {}
    ).get(
        state_name,
        (state_name, "Monitor mission", "Mission state is active", "", None),
    )
    target_count = max(0, int(target_count or 0))
    current_target = max(0, int(current_target or 0))
    completed_targets = min(current_target, target_count) if target_count else 0
    displayed_target = min(current_target + 1, target_count) if target_count else 0
    progress = 100.0 * completed_targets / target_count if target_count else 0.0
    if state_name == "FINISHED":
        progress = 100.0
        completed_targets = target_count
        displayed_target = target_count
    return {
        "active_mission": f"task{mission_number}",
        "stage": stage,
        "action": action or default_action,
        "reason": reason or default_reason,
        "colreg_rule": colreg_rule,
        "collision_risk": collision_risk,
        "current_target": displayed_target,
        "target_count": target_count,
        "progress_percent": progress,
    }


def mission_decision_json(*args, **kwargs):
    return json.dumps(build_mission_decision(*args, **kwargs), ensure_ascii=True)
