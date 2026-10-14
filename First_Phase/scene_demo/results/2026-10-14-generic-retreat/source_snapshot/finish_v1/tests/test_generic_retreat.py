"""Geometry integration tests: actual screens, no task-specific labels."""
import unittest
import numpy as np
from scene_demo.finish_v1 import geometry


class GeometryPort:
    def __init__(self, points):
        self.points = points

    def obstacle_cloud(self, frames):
        return self.points

    def robot_shapes(self, q):
        return [{'id': 'hand', 'hand': True, 'center': list(q[:3]),
                 'rotation': np.eye(3).tolist(), 'half_extents': [.02]*3}]

    def ik(self, xyz, rotation, seed):
        return np.r_[xyz, np.zeros(4)]


def observation():
    return {'robot': {'arm_qpos': [0.]*7}, 'eef_position': [0.]*3,
            'eef_rotation': np.eye(3).tolist(), 'frames': [], 'timestamp': 12.5}


def wall(axis, sign):
    a, b = np.meshgrid(np.linspace(-.15, .15, 31), np.linspace(-.15, .15, 31))
    points = np.c_[a.ravel(), b.ravel(), np.full(a.size, sign*.018)]
    return points[:, [(i-axis+2) % 3 for i in range(3)]]


class GenericRetreatTests(unittest.TestCase):
    def test_empty_semantic_boxes_exit_four_wall_orientations(self):
        for axis in (0, 1):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    port = GeometryPort(wall(axis, sign))
                    result = geometry.plan_retreat(port, observation(), {'boxes': {}})
                    self.assertTrue(result['ok'], result)
                    self.assertEqual([x['reason'] for x in result['candidate_reports'][:2]],
                                     ['endpoint_hand_clear_fail']*2)
                    chosen = result['candidate_reports'][-1]
                    self.assertEqual(chosen['direction_source'], 'current_geometry')
                    self.assertLess(chosen['direction'][axis]*sign, -.5)
                    self.assertTrue(result['sensor_screen']['endpoint_hand_clear'])
                    self.assertEqual(result['observation_timestamp'], 12.5)

    def test_original_successful_up_route_preserved(self):
        port = GeometryPort(np.array([[1., 1., 1.]]))
        result = geometry.plan_retreat(port, observation(), {'boxes': {}})
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['candidate_reports']), 1)
        self.assertEqual(result['candidate_reports'][0]['direction'], [0., 0., 1.])
        self.assertEqual(result['candidate_reports'][0]['distance'], .04)

    def test_enclosed_hand_still_rejected(self):
        port = GeometryPort(np.vstack([wall(axis, sign) for axis in range(3) for sign in (-1, 1)]))
        result = geometry.plan_retreat(port, observation(), {'boxes': {}})
        self.assertFalse(result['ok'])
        self.assertEqual(result['waypoints'], [])
        self.assertGreater(len(result['candidate_reports']), 2)
        self.assertTrue(all(not r['sensor_screen']['ok'] for r in result['candidate_reports']))

    def test_override_does_not_silently_expand(self):
        result = geometry.plan_retreat(GeometryPort(wall(0, 1)), observation(), {'boxes': {}},
                                       directions_override=[np.array([0., 0., 1.])])
        self.assertFalse(result['ok'])
        self.assertEqual(len(result['candidate_reports']), 2)
        self.assertTrue(all(r['direction_source']=='override' for r in result['candidate_reports']))

    def test_no_depth_points_cannot_be_accepted(self):
        result = geometry.plan_retreat(GeometryPort(np.zeros((0, 3))), observation(), {'boxes': {}})
        self.assertFalse(result['ok'])
        self.assertEqual(result['candidate_reports'], [])


if __name__ == '__main__':
    unittest.main()
