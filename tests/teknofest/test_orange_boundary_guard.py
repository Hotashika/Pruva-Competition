import math
import unittest

from teknofest.missions.utils.orange_boundary_guard import (
    OrangeBoundaryGuard,
    is_orange_boundary_detection,
)


CURRENT_LAT = 37.95125
CURRENT_LON = 32.50090
NORTH_TARGET_LAT = CURRENT_LAT + 0.001
NORTH_TARGET_LON = CURRENT_LON


def orange(angle_deg, distance_m, confidence=0.9):
    return {
        "class": "orange_buoy",
        "confidence": confidence,
        "distance": distance_m,
        "Buoy angle: ": angle_deg,
    }


def orange_xy(x_right_m, y_forward_m):
    return orange(
        math.degrees(math.atan2(x_right_m, y_forward_m)),
        math.hypot(x_right_m, y_forward_m),
    )


class OrangeBoundaryGuardTests(unittest.TestCase):
    def test_orange_class_aliases_are_separated_from_obstacles(self):
        self.assertTrue(is_orange_boundary_detection({"class": "orange_buoy"}))
        self.assertTrue(is_orange_boundary_detection({"label": "Turuncu Duba"}))
        self.assertFalse(is_orange_boundary_detection({"class": "yellow_buoy"}))

    def test_symmetric_corridor_keeps_northbound_target_centered(self):
        guard = OrangeBoundaryGuard()
        decision = guard.compute(
            [orange(-30.0, 8.0), orange(30.0, 8.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )

        self.assertEqual("live", decision.status)
        self.assertEqual("both_boundaries", decision.reason)
        self.assertAlmostEqual(0.0, decision.relative_bearing_deg, delta=0.5)
        self.assertAlmostEqual(8.0, decision.corridor_width_m, delta=0.1)
        self.assertGreater(decision.target_lat, CURRENT_LAT)

    def test_gps_target_outside_corridor_is_clipped_inside_safe_gate(self):
        guard = OrangeBoundaryGuard()
        decision = guard.compute(
            [orange(-30.0, 8.0), orange(30.0, 8.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            CURRENT_LAT,
            CURRENT_LON - 0.001,
            now=10.0,
        )

        self.assertEqual("live", decision.status)
        self.assertGreater(decision.relative_bearing_deg, -30.0)
        self.assertLess(decision.relative_bearing_deg, 0.0)

    def test_close_left_boundary_steers_vehicle_right(self):
        guard = OrangeBoundaryGuard()
        decision = guard.compute(
            [orange(-20.0, 2.0), orange(30.0, 8.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )

        self.assertEqual("live", decision.status)
        self.assertGreater(decision.relative_bearing_deg, 0.0)

    def test_bending_boundary_rows_create_a_turning_safe_target(self):
        guard = OrangeBoundaryGuard()
        decision = guard.compute(
            [
                orange_xy(-3.0, 2.0),
                orange_xy(-1.0, 8.0),
                orange_xy(3.0, 2.0),
                orange_xy(5.0, 8.0),
            ],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )

        self.assertEqual("live", decision.status)
        self.assertGreater(decision.relative_bearing_deg, 2.0)

    def test_single_visible_side_uses_learned_corridor_width(self):
        guard = OrangeBoundaryGuard()
        guard.compute(
            [orange(-30.0, 8.0), orange(30.0, 8.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )
        decision = guard.compute(
            [orange(-25.0, 8.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.1,
        )

        self.assertEqual("live", decision.status)
        self.assertTrue(decision.inferred_boundary)
        self.assertEqual(1, decision.left_count)
        self.assertEqual(0, decision.right_count)

    def test_short_dropout_uses_memory_then_fails_closed(self):
        guard = OrangeBoundaryGuard()
        guard.compute(
            [orange(-30.0, 8.0), orange(30.0, 8.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )

        memory = guard.compute(
            [],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.5,
        )
        blocked = guard.compute(
            [],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=11.1,
        )

        self.assertEqual("memory", memory.status)
        self.assertTrue(memory.has_target)
        self.assertEqual("blocked", blocked.status)
        self.assertTrue(blocked.should_stop)

    def test_implausibly_narrow_corridor_is_rejected(self):
        guard = OrangeBoundaryGuard()
        decision = guard.compute(
            [orange(-5.0, 4.0), orange(5.0, 4.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )

        self.assertEqual("blocked", decision.status)
        self.assertEqual("invalid_corridor_width", decision.reason)

    def test_corridor_requiring_unsafe_turn_is_rejected(self):
        guard = OrangeBoundaryGuard()
        decision = guard.compute(
            [orange(70.0, 20.0)],
            CURRENT_LAT,
            CURRENT_LON,
            0.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )

        self.assertEqual("blocked", decision.status)
        self.assertEqual("corridor_requires_excessive_turn", decision.reason)

    def test_all_numeric_outputs_are_finite(self):
        guard = OrangeBoundaryGuard()
        decision = guard.compute(
            [orange(-35.0, 10.0), orange(20.0, 9.0)],
            CURRENT_LAT,
            CURRENT_LON,
            15.0,
            NORTH_TARGET_LAT,
            NORTH_TARGET_LON,
            now=10.0,
        )

        self.assertTrue(decision.has_target)
        self.assertTrue(math.isfinite(decision.target_lat))
        self.assertTrue(math.isfinite(decision.target_lon))


if __name__ == "__main__":
    unittest.main()
