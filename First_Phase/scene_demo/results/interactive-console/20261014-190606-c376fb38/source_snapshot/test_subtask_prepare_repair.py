import unittest
import numpy as np

from test_subtask_assist import (
    make_reading,
    FakeContext,
    rotation_x,
    rotation_z,
    IDENTITY,
)
import subtask_preparation as sp


class TestSubtaskPreparation(unittest.TestCase):
    def test_downward_heading_is_preserved(self):
        home = rotation_x(np.pi)
        R = rotation_z(0.9) @ rotation_x(np.pi)
        reading = make_reading(orientation=R, home=home)
        plan = sp.plan_route(FakeContext(), reading)

        self.assertFalse(
            np.allclose(R, home),
            "Orientation must demonstrate a non-home horizontal heading",
        )
        self.assertEqual(
            plan['orientation_policy'],
            'preserve_downward_current_orientation',
        )
        self.assertIs(plan['reorientation_required'], False)

        for waypoint in plan['waypoints']:
            np.testing.assert_allclose(waypoint['orientation'], R)

        np.testing.assert_allclose(
            plan['clearance_extent'],
            sp._down_extent(reading, R),
        )
        self.assertEqual(
            plan['collision_scope'],
            'gripper_obb_samples_only',
        )

    def test_non_downward_start_retains_checked_reorientation(self):
        home = rotation_x(np.pi)
        reading = make_reading(orientation=IDENTITY, home=home)
        plan = sp.plan_route(FakeContext(), reading)

        self.assertEqual(
            plan['orientation_policy'],
            'home_orientation_for_non_downward_start',
        )
        self.assertIs(plan['reorientation_required'], True)

        names = [waypoint['name'] for waypoint in plan['waypoints']]
        for required in ('align_clear', 'transit', 'ready'):
            self.assertIn(required, names)
        for waypoint in plan['waypoints']:
            if waypoint['name'] in ('align_clear', 'transit', 'ready'):
                np.testing.assert_allclose(waypoint['orientation'], home)

        np.testing.assert_allclose(
            plan['clearance_extent'],
            sp._max_hand_reach(reading, np.asarray(reading['pose']['position'])),
        )
        self.assertEqual(
            plan['collision_scope'],
            'gripper_obb_samples_only',
        )


if __name__ == '__main__':
    unittest.main()
