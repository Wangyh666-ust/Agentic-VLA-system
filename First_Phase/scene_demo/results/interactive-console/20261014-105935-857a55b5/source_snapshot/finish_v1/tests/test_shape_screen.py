"""GPT fixed OBB and projection acceptance, worker mechanical implementation."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from scene_demo.finish_v1 import geometry, vision
from scene_demo.finish_v1.tests.test_takeover_vision import _place_context, _verdict


def box(center=(0, 0, 0), half=(.1, .01, .01), angle=0):
    c, s = np.cos(angle), np.sin(angle)
    return {"id": "hand", "center": list(center), "half_extents": list(half), "rotation": [[c, -s, 0], [s, c, 0], [0, 0, 1]], "hand": True}


class ShapeScreenTests(unittest.TestCase):
    def test_retreat_candidates_start_at_four_centimeters(self):
        from scene_demo.finish_v1.tests.test_core import FakePort
        port = FakePort()
        observation = port.capture()
        with patch.object(port, 'ik', create=True, side_effect=lambda xyz, rotation, seed: np.full(7, xyz[2])), patch.object(geometry, 'screen_shapes', return_value={'ok': True, 'reason': 'clear'}):
            plan = geometry.plan_retreat(port, observation, {'boxes': {}})
        self.assertTrue(plan['ok'])
        self.assertEqual(plan['candidate_reports'][0]['distance'], .04)
        self.assertAlmostEqual(plan['waypoints'][-1][0], .04)
        with patch.object(port, 'ik', create=True, side_effect=lambda xyz, rotation, seed: np.full(7, xyz[2])), patch.object(geometry, 'screen_shapes', return_value={'ok': False, 'reason': 'path_margin_violation'}):
            blocked = geometry.plan_retreat(port, observation, {'boxes': {}})
        self.assertFalse(blocked['ok'])
        self.assertEqual([report['distance'] for report in blocked['candidate_reports']], [.04, .06])

    def test_triangle_surface_distance(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2]])
        points = np.array([[.2, .2, .3], [2., 0., 0.], [.5, -.2, 0.], [.2, .2, 0.]])
        np.testing.assert_allclose(geometry.mesh_surface_distance(points, vertices, faces), [.3, 1., .2, 0.], atol=1e-12)

    def test_degenerate_triangle_rejected(self):
        with self.assertRaises(ValueError):
            geometry.mesh_surface_distance(np.zeros((1, 3)), np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), np.array([[0, 1, 2]]))

    def test_bbox_interior_far_from_surface_preserved(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        distance = geometry.mesh_surface_distance(np.array([[.1, .1, .1]]), vertices, np.array([[0, 1, 2]]))[0]
        self.assertAlmostEqual(distance, .1)
        self.assertGreater(distance, .003)

    def test_pixel_center_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'depth.npy'
            np.save(path, np.ones((1, 1)))
            frame = {'depth_path': str(path), 'K': np.eye(3), 'T_world_camera': np.eye(4)}
            np.testing.assert_allclose(geometry.clouds([frame]), [[0., 0., 1.]])
            frame['pixel_center_offset'] = 0.
            np.testing.assert_allclose(geometry.clouds([frame]), [[0., 0., 1.]])
            frame['pixel_center_offset'] = .5
            np.testing.assert_allclose(geometry.clouds([frame]), [[.5, .5, 1.]])
            for offset in (float('nan'), -1., 2.):
                frame['pixel_center_offset'] = offset
                self.assertEqual(geometry.clouds([frame]).shape, (0, 3))

    def test_thin_rotated_box_outside_obb_inside_rbound(self):
        shape = box(angle=np.pi/2)
        points = np.array([[.06, 0, 0]])
        self.assertLess(np.linalg.norm(points[0]), np.linalg.norm(shape["half_extents"]))
        np.testing.assert_allclose(geometry.shape_distance(points, shape), [.05])
        self.assertTrue(geometry.screen_shapes(points, [shape], [[shape]])["ok"])

    def test_path_entering_obstacle_rejected(self):
        start = box(half=(.01, .01, .01))
        end = box(center=(.05, 0, 0), half=(.01, .01, .01))
        report = geometry.screen_shapes(np.array([[.05, 0, 0]]), [start], [[end]])
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "path_margin_violation")

    def test_rotation_changes_clearance(self):
        points = np.array([[.06, 0, 0]])
        self.assertLess(geometry.shape_distance(points, box())[0], 0)
        np.testing.assert_allclose(geometry.shape_distance(points, box(angle=np.pi/2)), [.05])

    def test_inside_obstacle_kept_with_hand_diagnostic(self):
        shape = box()
        report = geometry.screen_shapes(np.zeros((1, 3)), [shape], [[shape]])
        self.assertFalse(report["ok"])
        self.assertFalse(report["endpoint_hand_clear"])
        self.assertEqual(report["diagnostic"]["geom"], "hand")
        self.assertLess(report["diagnostic"]["current_clearance"], 0)
        self.assertIn("no OBB point deletion", report["diagnostic"]["self_mask_policy"])

    def test_invalid_rotation_bounds_and_margin_rejected(self):
        points = np.array([[1., 1., 1.]])
        invalid = box()
        invalid["rotation"] = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(geometry.screen_shapes(points, [invalid], [[invalid]])["reason"], "invalid_start_shape")
        missing = box()
        del missing["half_extents"]
        self.assertEqual(geometry.screen_shapes(points, [box()], [[missing]])["reason"], "invalid_path_shape")
        for margin in (-.1, float("nan")):
            self.assertEqual(geometry.screen_shapes(points, [box()], [[box()]], margin)["reason"], "invalid_margin")

    def test_projection_overlap_preserves_true_and_null(self):
        boxes = {"front": {name: [.1, .1, .2, .2] for name in ("target", "destination", "gripper")}}
        for raw_clear in (True, None):
            result = vision.reconcile_verdict(_place_context(), _verdict(clear_of_target=raw_clear, boxes=boxes), {"finger_qpos": [.04, -.04]}, {"finger_home": [.04, -.04]})
            self.assertIs(result["clear_of_target"], raw_clear)
            self.assertIn("2D projection overlap near target/support; not 3D contact evidence", result["evidence"])


if __name__ == "__main__":
    unittest.main()
