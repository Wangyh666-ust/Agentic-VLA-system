import unittest
from unittest.mock import patch
import numpy as np
from scene_demo.finish_v1 import feedback_retreat as feedback
from scene_demo.finish_v1.geometry import screen_shapes


class Port:
    reference = {'finger_home': [.04, -.04], 'homeq': [0.] * 7}

    def __init__(self, stalled=False):
        self.q = np.zeros(7)
        self.q[6] = .006  # Deliberate persistent error unrelated to Cartesian progress.
        self.stalled = stalled
        self.requests = []
        self.commands = []

    def read_robot(self):
        return {'arm_qpos': self.q.copy(), 'arm_qvel': np.zeros(7), 'finger_qpos': [.04, -.04]}

    def measured_pose(self):
        return {'robot': self.read_robot(), 'eef_position': self.q[:3].copy(), 'eef_rotation': np.eye(3)}

    def capture(self):
        return dict(self.measured_pose(), frames=[])

    def obstacle_cloud(self, frames):
        return np.array([[0., 0., -.004]])

    def home_screen(self, waypoints):
        return {'ok': True}

    def robot_shapes(self, q):
        return [{'id': 'hand', 'hand': True, 'center': list(q[:3]),
                 'rotation': np.eye(3), 'half_extents': [.005, .005, .005]}]

    def ik(self, xyz, rotation, seed):
        self.requests.append((np.asarray(seed).copy(), np.asarray(xyz).copy()))
        q = np.asarray(seed).copy()
        q[:3] = xyz
        q[6] = 0.
        return q

    def act(self, phase, target, fingers, **kwargs):
        self.commands.append((phase, np.asarray(target).copy(), list(fingers)))
        if not self.stalled:
            self.q[:3] += .5 * (np.asarray(target)[:3] - self.q[:3])
        return self.read_robot()


def plan(port, obs, verdict, directions_override=None, endpoint_eligible=None):
    d = np.asarray(directions_override[0]).tolist()
    return {'ok': True, 'waypoints': [obs['robot']['arm_qpos']],
            'candidate_reports': [{'direction': d, 'distance': .06}], 'sensor_screen': {'ok': True}}


class FeedbackTests(unittest.TestCase):
    def test_progress_does_not_wait_for_joint_error(self):
        p = Port()
        with patch.object(feedback, 'plan_retreat', side_effect=plan):
            result = feedback.run(p, {}, p.act)
        self.assertTrue(result['ok'], result)
        self.assertGreaterEqual(result['travel_m'], .03)
        self.assertGreater(p.q[6], .004)
        self.assertTrue(all(f == [.04, -.04] for _, _, f in p.commands))
        # Every IK goal begins at latest measured pose rather than a stored waypoint.
        for seed, xyz in p.requests:
            self.assertGreater(np.linalg.norm(xyz - seed[:3]), .007)

    def test_stall_replans_from_actual_state_then_stops_without_home(self):
        p = Port(stalled=True)
        with patch.object(feedback, 'plan_retreat', side_effect=plan) as planner:
            result = feedback.run(p, {}, p.act)
        self.assertEqual(result['reason'], 'retreat replans exhausted')
        self.assertEqual(len(result['replans']), 3)
        self.assertEqual(planner.call_count, 4)
        self.assertTrue(all(phase == 'movement' for phase, _, _ in p.commands))
        np.testing.assert_allclose(planner.call_args.args[1]['robot']['arm_qpos'], p.q)

    def test_blocked_command_is_not_executed(self):
        p = Port()
        with patch.object(feedback, 'plan_retreat', side_effect=plan), patch.object(
                feedback, 'screen_shapes', side_effect=lambda *args, **kwargs:
                    {'ok': not (kwargs.get('require_endpoint_clear') is False), 'reason': 'approach_violation'}):
            result = feedback.run(p, {}, p.act)
        self.assertFalse(result['ok'])
        self.assertEqual(p.commands, [])
        self.assertEqual(len(result['replans']), 3)

    def test_short_exit_keeps_collision_rule_without_endpoint_gate(self):
        p = Port()
        points = p.obstacle_cloud([])
        start = p.robot_shapes(np.zeros(7))
        q = np.zeros(7); q[2] = .002
        short = p.robot_shapes(q)
        self.assertFalse(screen_shapes(points, start, [short])['ok'])
        self.assertTrue(screen_shapes(points, start, [short], require_endpoint_clear=False)['ok'])
        q[2] = -.002
        self.assertFalse(screen_shapes(points, start, [p.robot_shapes(q)], require_endpoint_clear=False)['ok'])

    def test_no_clear_endpoint_cannot_complete(self):
        p = Port()
        with patch.object(feedback, 'plan_retreat', side_effect=plan), patch.object(
                feedback, 'screen_shapes', side_effect=lambda *args, **kwargs:
                    {'ok': kwargs.get('require_endpoint_clear') is False, 'reason': 'endpoint_hand_clear_fail'}):
            result = feedback.run(p, {}, p.act, limit=15)
        self.assertEqual(result['reason'], 'movement budget')
        self.assertEqual(len(p.commands), 15)
        self.assertTrue(all(phase == 'movement' for phase, _, _ in p.commands))

    def test_current_clear_but_return_blocked_continues_exiting(self):
        p = Port()
        with patch.object(feedback, 'plan_retreat', side_effect=plan), patch.object(
                p, 'home_screen', side_effect=[{'ok': False}, {'ok': True}]):
            result = feedback.run(p, {}, p.act)
        self.assertTrue(result['ok'], result)
        self.assertEqual([x['home_screen']['ok'] for x in result['home_candidates']], [False, True])
        self.assertGreater(result['home_candidates'][1]['actions'], result['home_candidates'][0]['actions'])
        self.assertGreater(result['travel_m'], .03)


if __name__ == '__main__':
    unittest.main()
