import sys
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation as _ScipyRotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

import subtask_context as sc
import subtask_geometry as sg
import subtask_preparation as sp


class FakeModel:
    def __init__(self, geom_type, geom_contype, geom_conaffinity, geom_size,
                 geom_names, geom_dataid=None, mesh_vertadr=None,
                 mesh_vertnum=None, mesh_vert=None):
        self.geom_type = np.asarray(geom_type)
        self.geom_contype = np.asarray(geom_contype)
        self.geom_conaffinity = np.asarray(geom_conaffinity)
        self.geom_size = np.asarray(geom_size, dtype=float)
        self._names = list(geom_names)
        self.ngeom = len(self._names)
        if geom_dataid is not None:
            self.geom_dataid = np.asarray(geom_dataid)
        if mesh_vertadr is not None:
            self.mesh_vertadr = np.asarray(mesh_vertadr)
        if mesh_vertnum is not None:
            self.mesh_vertnum = np.asarray(mesh_vertnum)
        if mesh_vert is not None:
            self.mesh_vert = np.asarray(mesh_vert, dtype=float)

    def geom_id2name(self, index):
        return self._names[int(index)]


class FakeData:
    def __init__(self, geom_xpos, geom_xmat, ncon=0, contact=None):
        self.geom_xpos = np.asarray(geom_xpos, dtype=float)
        self.geom_xmat = np.asarray(geom_xmat, dtype=float)
        self.ncon = ncon
        self.contact = contact if contact is not None else []


def rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


IDENTITY = np.eye(3, dtype=float)


def make_box_model_data(half=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.0),
                        rot=None, name='box0'):
    rot = IDENTITY if rot is None else np.asarray(rot, dtype=float)
    model = FakeModel(
        geom_type=[6],
        geom_contype=[1],
        geom_conaffinity=[1],
        geom_size=[list(half)],
        geom_names=[name],
    )
    data = FakeData(geom_xpos=[list(pos)], geom_xmat=[rot.reshape(9)])
    return model, data


class GeometryBoundsTests(unittest.TestCase):
    def test_box_bounds_center_half_and_aabb(self):
        half = (0.02, 0.03, 0.04)
        pos = (0.1, -0.2, 1.0)
        rot = rotation_z(math.pi / 4)
        model, data = make_box_model_data(half=half, pos=pos, rot=rot)
        bounds = sg.geom_bounds(model, data, 0)
        self.assertEqual(bounds['type'], 'box')
        np.testing.assert_allclose(bounds['center'], pos, atol=1e-9)
        np.testing.assert_allclose(bounds['half'], half, atol=1e-9)
        abs_r = np.abs(rot)
        extent = abs_r @ np.asarray(half)
        np.testing.assert_allclose(bounds['aabb_min'], np.asarray(pos) - extent, atol=1e-9)
        np.testing.assert_allclose(bounds['aabb_max'], np.asarray(pos) + extent, atol=1e-9)

    def test_capsule_half_extends_by_radius_along_axis(self):
        radius, half_length = 0.02, 0.5
        model = FakeModel(
            geom_type=[3],
            geom_contype=[1],
            geom_conaffinity=[1],
            geom_size=[[radius, half_length]],
            geom_names=['cap0'],
        )
        data = FakeData(geom_xpos=[[0.0, 0.0, 1.2]], geom_xmat=[IDENTITY.reshape(9)])
        bounds = sg.geom_bounds(model, data, 0)
        self.assertEqual(bounds['type'], 'capsule')
        np.testing.assert_allclose(bounds['half'], [radius, radius, half_length + radius], atol=1e-9)

    def test_mesh_bounds_from_vertices(self):
        verts = np.array([
            [-0.1, -0.05, -0.02],
            [0.1, -0.05, -0.02],
            [0.1, 0.05, -0.02],
            [-0.1, 0.05, -0.02],
            [-0.1, -0.05, 0.02],
            [0.1, -0.05, 0.02],
            [0.1, 0.05, 0.02],
            [-0.1, 0.05, 0.02],
        ])
        model = FakeModel(
            geom_type=[7],
            geom_contype=[1],
            geom_conaffinity=[1],
            geom_size=[[0.1, 0.05, 0.02]],
            geom_names=['mesh0'],
            geom_dataid=[0],
            mesh_vertadr=[0],
            mesh_vertnum=[8],
            mesh_vert=verts,
        )
        data = FakeData(geom_xpos=[[0.0, 0.0, 1.0]], geom_xmat=[IDENTITY.reshape(9)])
        bounds = sg.geom_bounds(model, data, 0)
        self.assertEqual(bounds['type'], 'mesh')
        np.testing.assert_allclose(bounds['mesh_bounds']['min'], [-0.1, -0.05, -0.02], atol=1e-9)
        np.testing.assert_allclose(bounds['mesh_bounds']['max'], [0.1, 0.05, 0.02], atol=1e-9)
        np.testing.assert_allclose(bounds['center'], [0.0, 0.0, 1.0], atol=1e-9)

    def test_mesh_missing_vertices_rejected(self):
        model = FakeModel(
            geom_type=[7],
            geom_contype=[1],
            geom_conaffinity=[1],
            geom_size=[[0.1, 0.05, 0.02]],
            geom_names=['mesh0'],
            geom_dataid=[0],
            mesh_vertadr=[0],
            mesh_vertnum=[0],
            mesh_vert=np.zeros((0, 3)),
        )
        data = FakeData(geom_xpos=[[0.0, 0.0, 1.0]], geom_xmat=[IDENTITY.reshape(9)])
        with self.assertRaises(sg.GeometryError):
            sg.geom_bounds(model, data, 0)

    def test_invalid_rotation_rejected(self):
        model, data = make_box_model_data()
        bad = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        data.geom_xmat = np.asarray([bad.reshape(9)])
        with self.assertRaises(sg.GeometryError):
            sg.geom_bounds(model, data, 0)

    def test_noncollidable_geometry_rejected(self):
        model, data = make_box_model_data()
        model.geom_contype = np.asarray([0])
        model.geom_conaffinity = np.asarray([0])
        with self.assertRaises(sg.GeometryError):
            sg.geom_bounds(model, data, 0)


class FingerprintTests(unittest.TestCase):
    def _target_and_geom(self, pos=(0.0, 0.0, 1.0), rot=None,
                         geom_type='box', size=(0.05, 0.05, 0.05),
                         geom_pos=(0.0, 0.0, 1.0)):
        rot = IDENTITY if rot is None else rot
        target = {'position': list(pos), 'orientation': [list(r) for r in rot]}
        geom = {
            'type': geom_type,
            'size': list(size),
            'geom_position': list(geom_pos),
            'geom_rotation': [list(r) for r in rot],
            'mesh_bounds': None,
        }
        return target, [geom]

    def test_fingerprint_invariant_to_geom_rename(self):
        target, geoms = self._target_and_geom()
        first = sg.geometry_fingerprint(target, geoms)
        renamed = [dict(geoms[0], name='renamed_geometry')]
        second = sg.geometry_fingerprint(target, renamed)
        self.assertEqual(first, second)

    def test_fingerprint_invariant_to_global_rigid_transform(self):
        target, geoms = self._target_and_geom()
        first = sg.geometry_fingerprint(target, geoms)
        offset = np.array([0.3, -0.2, 0.15])
        rotation = rotation_z(0.7)
        new_target = {
            'position': [float(v) for v in (np.asarray(target['position']) + offset)],
            'orientation': [list(r) for r in (rotation @ IDENTITY)],
        }
        geom = geoms[0]
        new_geom_pos = rotation @ np.asarray(geom['geom_position']) + offset
        new_geom_rot = rotation @ np.asarray(geom['geom_rotation'])
        new_geoms = [dict(geom, geom_position=[float(v) for v in new_geom_pos],
                          geom_rotation=[list(r) for r in new_geom_rot])]
        second = sg.geometry_fingerprint(new_target, new_geoms)
        self.assertEqual(first, second)

    def test_fingerprint_changes_on_shape_change(self):
        target, geoms = self._target_and_geom()
        first = sg.geometry_fingerprint(target, geoms)
        changed = [dict(geoms[0], size=[0.06, 0.05, 0.05])]
        second = sg.geometry_fingerprint(target, changed)
        self.assertNotEqual(first, second)

    def test_fingerprint_rejects_missing_fields(self):
        target = {'position': [0.0, 0.0, 1.0], 'orientation': [list(r) for r in IDENTITY]}
        with self.assertRaises(sg.GeometryError):
            sg.geometry_fingerprint(target, [{'type': 'box', 'size': [0.05, 0.05, 0.05]}])

    def test_fingerprint_normalizes_negative_zero(self):
        target, geoms = self._target_and_geom()
        first = sg.geometry_fingerprint(target, geoms)
        geom = geoms[0]
        neg_zero = [list(-np.asarray(geom['geom_position']))]
        tweaked = [dict(geom, geom_position=[-0.0, -0.0, 1.0])]
        second = sg.geometry_fingerprint(target, tweaked)
        self.assertEqual(first, second)
        del neg_zero

    def test_fingerprint_repr_has_no_negative_zero(self):
        target, geoms = self._target_and_geom(geom_pos=(-0.0, -0.0, 1.0))
        first = sg.geometry_fingerprint(target, geoms)
        positive = [dict(geoms[0], geom_position=[0.0, 0.0, 1.0])]
        second = sg.geometry_fingerprint(target, positive)
        self.assertEqual(first, second)


class ContextTests(unittest.TestCase):
    def _capability(self, object_id='obj', target_id='table', goals=None):
        return {
            'object_id': object_id,
            'target_id': target_id,
            'goals': goals if goals is not None else [['on', object_id, target_id]],
        }

    def _reading(self, position=(0.0, 0.0, 1.0), orientation=None,
                 dimensions=(0.05, 0.05, 0.2), calibration_key=None,
                 held=(), complete=True):
        orientation = IDENTITY if orientation is None else orientation
        return {
            'target': {
                'position': list(position),
                'orientation': [list(r) for r in orientation],
                'dimensions': list(dimensions),
                'calibration_key': calibration_key,
            },
            'geometry_source': 'mujoco_collision_obbs',
            'snapshot': {
                'held_objects': list(held),
                'grasp_observation_complete': complete,
            },
        }

    def test_build_context_roundtrip(self):
        ctx = sc.build_context(self._capability(), self._reading())
        self.assertEqual(ctx.operation, 'pick_place')
        self.assertEqual(ctx.object_id, 'obj')
        self.assertEqual(ctx.target_id, 'table')
        self.assertEqual(ctx.shape_family, 'upright_elongated')
        self.assertTrue(ctx.observation_complete)
        payload = ctx.to_dict()
        self.assertEqual(payload['object_id'], 'obj')

    def test_geometry_renaming_preserves_selection(self):
        cap_a = self._capability(object_id='wine', target_id='table')
        cap_b = {
            'object_id': 'renamed_object',
            'target_id': 'renamed_surface',
            'goals': [['on', 'renamed_object', 'renamed_surface']],
        }
        ctx_a = sc.build_context(cap_a, self._reading(calibration_key='libero_wine_side_v1'))
        ctx_b = sc.build_context(cap_b, self._reading(calibration_key='libero_wine_side_v1'))
        sel_a = sc.select_strategies(ctx_a)
        sel_b = sc.select_strategies(ctx_b)
        self.assertEqual(sel_a.prepare_strategy, sel_b.prepare_strategy)
        self.assertEqual(sel_a.grasp_strategy, sel_b.grasp_strategy)
        self.assertEqual(sel_a.shape_family, sel_b.shape_family)
        self.assertNotEqual(ctx_a.object_id, ctx_b.object_id)
        self.assertNotEqual(ctx_a.target_id, ctx_b.target_id)
        self.assertEqual(sel_a.grasp_strategy, 'calibrated_side_v1')

    def test_low_wide_never_selects_grasp(self):
        ctx = sc.build_context(
            self._capability(),
            self._reading(dimensions=(0.2, 0.2, 0.05), calibration_key='libero_wine_side_v1'),
        )
        self.assertEqual(ctx.shape_family, 'low_wide')
        selection = sc.select_strategies(ctx)
        self.assertIsNone(selection.grasp_strategy)
        self.assertIsNotNone(selection.prepare_strategy)

    def test_upright_elongated_without_calibration_matches_no_grasp(self):
        ctx = sc.build_context(
            self._capability(),
            self._reading(dimensions=(0.05, 0.05, 0.2)),
        )
        self.assertEqual(ctx.shape_family, 'upright_elongated')
        selection = sc.select_strategies(ctx)
        self.assertIsNone(selection.grasp_strategy)
        self.assertEqual(selection.prepare_strategy, 'clearance_preposition_v1')

    def test_calibrated_selection_requires_shape_and_key(self):
        matching = sc.build_context(
            self._capability(),
            self._reading(dimensions=(0.05, 0.05, 0.2), calibration_key='libero_wine_side_v1'),
        )
        self.assertEqual(matching.shape_family, 'upright_elongated')
        selection = sc.select_strategies(matching)
        self.assertEqual(selection.grasp_strategy, 'calibrated_side_v1')

        wrong_key = sc.build_context(
            self._capability(),
            self._reading(dimensions=(0.05, 0.05, 0.2), calibration_key='other_key'),
        )
        self.assertIsNone(sc.select_strategies(wrong_key).grasp_strategy)

        wrong_shape = sc.build_context(
            self._capability(),
            self._reading(dimensions=(0.2, 0.2, 0.05), calibration_key='libero_wine_side_v1'),
        )
        self.assertEqual(wrong_shape.shape_family, 'low_wide')
        self.assertIsNone(sc.select_strategies(wrong_shape).grasp_strategy)

    def test_tilted_never_selects_side_helper(self):
        tilted = rotation_y(math.pi / 3)
        ctx = sc.build_context(
            self._capability(),
            self._reading(orientation=tilted, dimensions=(0.05, 0.05, 0.2),
                          calibration_key='libero_wine_side_v1'),
        )
        self.assertEqual(ctx.shape_family, 'tilted')
        selection = sc.select_strategies(ctx)
        self.assertNotEqual(selection.grasp_strategy, 'calibrated_side_v1')
        self.assertIsNone(selection.grasp_strategy)

    def test_foreign_held_object_blocked(self):
        ctx = sc.build_context(
            self._capability(),
            self._reading(held=('mug',)),
        )
        selection = sc.select_strategies(ctx)
        self.assertEqual(selection.status, 'blocked')
        self.assertEqual(selection.reason, 'foreign_object_held')

    def test_unknown_grasp_state_blocked(self):
        ctx = sc.build_context(
            self._capability(),
            self._reading(complete=False),
        )
        selection = sc.select_strategies(ctx)
        self.assertEqual(selection.status, 'blocked')
        self.assertEqual(selection.reason, 'grasp_observation_unknown')

    def test_goal_object_mismatch_rejected(self):
        capability = {
            'object_id': 'obj',
            'target_id': 'table',
            'goals': [['on', 'other_obj', 'table']],
        }
        with self.assertRaises(sc.ContextError):
            sc.build_context(capability, self._reading())


def make_reading(position=(0.0, 0.0, 1.2), orientation=None,
                 home=None, hand=None, obstacles=None,
                 target_min=None, target_max=None,
                 objects=None, predicates=None, held=(),
                 complete=True, gap=0.08, robot_contacts=None):
    orientation = IDENTITY if orientation is None else np.asarray(orientation, dtype=float)
    home = rotation_x(math.pi) if home is None else np.asarray(home, dtype=float)
    base_pos = np.asarray(position, dtype=float)
    if hand is None:
        hand_local = [{'name': 'gripper_0', 'center': [0.0, 0.0, 0.0], 'rotation': IDENTITY, 'half': [0.02, 0.02, 0.02]}]
    else:
        hand_local = [dict(h, center=np.asarray(h['center'], dtype=float),
                           rotation=np.asarray(h['rotation'], dtype=float),
                           half=np.asarray(h['half'], dtype=float)) for h in hand]
    hand = [dict(h, center=base_pos + orientation @ np.asarray(h['center'], dtype=float),
                 rotation=orientation @ np.asarray(h['rotation'], dtype=float),
                 half=np.asarray(h['half'], dtype=float)) for h in hand_local]
    obstacles = obstacles or []
    obstacles = [dict(o, center=np.asarray(o['center'], dtype=float),
                      rotation=np.asarray(o['rotation'], dtype=float),
                      half=np.asarray(o['half'], dtype=float)) for o in obstacles]
    if target_min is None:
        target_min = [-0.05, -0.05, 0.9]
    if target_max is None:
        target_max = [0.05, 0.05, 1.0]
    if objects is None:
        objects = {'obj': {'position': [0.0, 0.0, 0.95]}}
    if predicates is None:
        predicates = {}
    return {
        'pose': {'position': list(position), 'orientation_matrix': [list(r) for r in orientation]},
        'home_orientation': [list(r) for r in home],
        'hand': hand,
        'obstacles': obstacles,
        'target': {
            'world_aabb_min': list(target_min),
            'world_aabb_max': list(target_max),
        },
        'snapshot': {
            'held_objects': list(held),
            'grasp_observation_complete': complete,
            'objects': objects,
            'predicates': predicates,
        },
        'gripper_gap_m': gap,
        'robot_contacts': list(robot_contacts) if robot_contacts is not None else [],
    }


class FakeContext:
    def __init__(self, position=(0.0, 0.0, 0.95), object_id='obj'):
        self.position = tuple(position)
        self.object_id = object_id
        self.geometry_source = 'mujoco_collision_obbs'


class ObstructionTests(unittest.TestCase):
    def test_pose_collisions_sat_separation(self):
        reading = make_reading(
            position=(0.0, 0.0, 1.2),
            obstacles=[{'name': 'wall', 'center': [0.5, 0.0, 1.2],
                        'rotation': IDENTITY, 'half': [0.01, 0.5, 0.5]}],
        )
        far = sp.pose_collisions(reading, (0.0, 0.0, 1.2), IDENTITY)
        self.assertEqual(far, [])
        near = sp.pose_collisions(reading, (0.5, 0.0, 1.2), IDENTITY)
        self.assertTrue(any('wall' == entry.split('|')[-1] for entry in near))

    def test_rotated_boundary_contact_detected(self):
        reading = make_reading(
            position=(0.0, 0.0, 1.2),
            obstacles=[{'name': 'tilted', 'center': [0.0, 0.0, 1.2],
                        'rotation': rotation_z(math.pi / 4), 'half': [0.05, 0.05, 0.05]}],
        )
        collisions = sp.pose_collisions(reading, (0.0, 0.0, 1.2), IDENTITY)
        self.assertIn('gripper_0|tilted', collisions)

    def test_transformed_hand_local_world_conversion(self):
        reading = make_reading(
            position=(0.0, 0.0, 1.2),
            hand=[{'name': 'gripper_0', 'center': [0.02, 0.0, 0.0],
                   'rotation': IDENTITY, 'half': [0.01, 0.01, 0.01]}],
        )
        new_pos = np.array([0.1, 0.2, 1.3])
        transformed = sp.transformed_hand(reading, new_pos, IDENTITY)
        self.assertEqual(len(transformed), 1)
        np.testing.assert_allclose(transformed[0]['center'], new_pos + np.array([0.02, 0.0, 0.0]), atol=1e-9)

    def test_transformed_hand_world_conversion_rotated(self):
        reading = make_reading(
            position=(0.0, 0.0, 1.2),
            hand=[{'name': 'gripper_0', 'center': [0.05, 0.0, 0.0],
                   'rotation': IDENTITY, 'half': [0.01, 0.01, 0.01]}],
        )
        rot = rotation_z(math.pi / 2)
        new_pos = np.array([0.0, 0.0, 1.4])
        transformed = sp.transformed_hand(reading, new_pos, rot)
        expected = new_pos + rot @ np.array([0.05, 0.0, 0.0])
        np.testing.assert_allclose(transformed[0]['center'], expected, atol=1e-9)
        np.testing.assert_allclose(transformed[0]['rotation'], rot, atol=1e-9)

    def test_pose_collisions_empty_for_non_overlap(self):
        reading = make_reading(obstacles=[{'name': 'far', 'center': [5.0, 5.0, 1.2],
                                           'rotation': IDENTITY, 'half': [0.01, 0.01, 0.01]}])
        self.assertEqual(sp.pose_collisions(reading, (0.0, 0.0, 1.2), IDENTITY), [])

    def test_interp_positions_spacing_and_endpoints(self):
        start = np.zeros(3)
        end = np.array([0.05, 0.0, 0.0])
        samples = sp._interpolate_positions(start, end)
        self.assertEqual(len(samples), 11)
        for a, b in zip(samples[:-1], samples[1:]):
            self.assertLessEqual(float(np.linalg.norm(b - a)), 0.005 + 1e-9)
        np.testing.assert_allclose(samples[0], start, atol=1e-9)
        np.testing.assert_allclose(samples[-1], end, atol=1e-9)

    def test_interp_orientations_spacing_and_endpoints(self):
        start = IDENTITY
        end = rotation_y(math.pi / 2)
        samples = sp._interpolate_orientations(start, end)
        self.assertGreater(len(samples), 2)
        for a, b in zip(samples[:-1], samples[1:]):
            rel = b @ a.T
            angle = float(np.linalg.norm(_ScipyRotation.from_matrix(rel).as_rotvec()))
            self.assertLessEqual(angle, 0.05 + 1e-6)
        np.testing.assert_allclose(samples[0], start, atol=1e-9)
        np.testing.assert_allclose(samples[-1], end, atol=1e-6)

    def test_large_obstacle_enclosing_xy_remains_relevant(self):
        reading = make_reading(
            obstacles=[{'name': 'table', 'center': [0.0, 0.0, 0.8],
                        'rotation': IDENTITY, 'half': [1.0, 1.0, 0.1]}],
        )
        current_xy = np.array([0.0, 0.0])
        target_xy = np.array([0.1, 0.1])
        relevant = sp._relevant_obstacles(reading, current_xy, target_xy, 0.2)
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0]['name'], 'table')

    def test_unknown_empty_hand_state_rejected(self):
        reading = make_reading(complete=False)
        with self.assertRaises(sp.PreparationError):
            sp._validate_empty_hand_context(FakeContext(), reading)

    def test_nonempty_held_object_rejected(self):
        reading = make_reading(held=('obj',))
        with self.assertRaises(sp.PreparationError):
            sp._validate_empty_hand_context(FakeContext(), reading)

    def test_current_pose_collision_rejected_at_controller(self):
        reading = make_reading(
            position=(0.0, 0.0, 1.2),
            obstacles=[{'name': 'blocker', 'center': [0.0, 0.0, 1.2],
                        'rotation': IDENTITY, 'half': [0.1, 0.1, 0.1]}],
        )
        with self.assertRaises(sp.PreparationError):
            sp.PreparationController(FakeContext(), reading)


class RouteTests(unittest.TestCase):
    def test_route_selection_independent_of_object_name(self):
        ctx_a = FakeContext(object_id='wine')
        ctx_b = FakeContext(object_id='bottle')
        reading = make_reading()
        plan_a = sp.plan_route(ctx_a, reading)
        plan_b = sp.plan_route(ctx_b, reading)
        self.assertEqual(plan_a['chosen_index'], plan_b['chosen_index'])
        self.assertEqual(len(plan_a['waypoints']), len(plan_b['waypoints']))
        for wa, wb in zip(plan_a['waypoints'], plan_b['waypoints']):
            self.assertEqual(wa['name'], wb['name'])
            np.testing.assert_allclose(wa['position'], wb['position'], atol=1e-9)
            np.testing.assert_allclose(wa['orientation'], wb['orientation'], atol=1e-9)

    def test_check_route_rejects_final_endpoint_collision(self):
        reading = make_reading(
            obstacles=[],
            target_min=[-0.05, -0.05, 0.9],
            target_max=[0.05, 0.05, 1.0],
        )
        ctx = FakeContext(position=(0.0, 0.0, 0.95))
        origin_position = np.array([0.0, 0.0, 1.2])
        origin_orientation = IDENTITY
        waypoints = [
            {'name': 'ready', 'position': np.array([0.0, 0.0, 1.4]), 'orientation': rotation_x(math.pi)},
        ]
        original = sp.pose_collisions
        calls = {'count': 0}

        def replace(reading_arg, position_arg, orientation_arg):
            calls['count'] += 1
            if np.allclose(np.asarray(position_arg), np.array([0.0, 0.0, 1.4]), atol=1e-6):
                return ['gripper_0|imaginary_obstacle']
            return original(reading_arg, position_arg, orientation_arg)

        with mock.patch.object(sp, 'pose_collisions', side_effect=replace):
            ok, reason, _samples = sp._check_route(
                reading, origin_position, origin_orientation,
                origin_position, origin_orientation, waypoints,
            )
        self.assertFalse(ok)
        self.assertEqual(reason, 'pose_collision')
        self.assertGreaterEqual(calls['count'], 1)


class ControllerProgressTests(unittest.TestCase):
    def _controller(self, ctx=None, reading=None, max_actions=200):
        ctx = ctx or FakeContext()
        reading = reading or make_reading()
        return sp.PreparationController(ctx, reading, max_actions=max_actions)

    def test_no_collision_fixture_produces_plan(self):
        controller = self._controller()
        self.assertEqual(controller.phase, 'preparing')
        self.assertIsNone(controller.reason)
        summary = controller.summary()
        self.assertGreaterEqual(len(summary['plan']['waypoints']), 1)
        self.assertEqual(summary['plan']['collision_scope'], 'gripper_obb_samples_only')

    def test_single_iteration_uses_active_waypoint(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        waypoint = controller._plan['waypoints'][0]
        with mock.patch('preparation_diagnostics.servo_action',
                               return_value=np.zeros(7, dtype=np.float32)) as servo:
            action = controller.next_action(reading)
        self.assertIsNotNone(action)
        servo.assert_called_once()
        args = servo.call_args.args
        np.testing.assert_allclose(np.asarray(args[0]), np.asarray(waypoint['position']), atol=1e-9)
        np.testing.assert_allclose(np.asarray(args[2]), np.asarray(waypoint['orientation']), atol=1e-6)

    def test_confirmation_sampling_advances_on_fifth(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        at_waypoint = dict(reading)
        waypoint = controller._plan['waypoints'][0]
        at_waypoint['pose'] = {
            'position': [float(v) for v in np.asarray(waypoint['position'])],
            'orientation_matrix': [list(r) for r in np.asarray(waypoint['orientation'])],
        }
        index_before = controller._waypoint_index
        for _ in range(4):
            controller.observe_after(at_waypoint)
            self.assertEqual(controller._waypoint_index, index_before)
            self.assertEqual(controller.phase, 'preparing')
        controller.observe_after(at_waypoint)
        self.assertGreater(controller._waypoint_index, index_before)

    def test_final_waypoint_confirmation_reaches_ready(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        last_index = len(controller._plan['waypoints']) - 1
        controller._waypoint_index = last_index
        controller._consecutive_confirmations = 5
        waypoint = controller._plan['waypoints'][last_index]
        done_reading = dict(reading)
        done_reading['pose'] = {
            'position': [float(v) for v in np.asarray(waypoint['position'])],
            'orientation_matrix': [list(r) for r in np.asarray(waypoint['orientation'])],
        }
        controller.observe_after(done_reading)
        self.assertEqual(controller.phase, 'ready')

    def test_servo_action_returns_legal_float32_7d(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        action = controller.next_action(reading)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.dtype, np.float32)
        self.assertEqual(action.shape, (7,))
        self.assertTrue(np.isfinite(action).all())
        self.assertEqual(action[-1], -1)
        self.assertEqual(controller.summary()['actual_aux_actions'], 0)

    def test_foreign_held_mutation_fails_before_servo(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        mutated = make_reading(held=('foreign',))
        with mock.patch('preparation_diagnostics.servo_action') as servo:
            action = controller.next_action(mutated)
        self.assertIsNone(action)
        self.assertEqual(controller.phase, 'failed')
        servo.assert_not_called()

    def test_unknown_complete_mutation_fails_before_servo(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        mutated = make_reading(complete=False)
        with mock.patch('preparation_diagnostics.servo_action') as servo:
            action = controller.next_action(mutated)
        self.assertIsNone(action)
        self.assertEqual(controller.phase, 'failed')
        servo.assert_not_called()

    def test_closed_gap_mutation_fails_before_servo(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        mutated = make_reading(gap=0.01)
        with mock.patch('preparation_diagnostics.servo_action') as servo:
            action = controller.next_action(mutated)
        self.assertIsNone(action)
        self.assertEqual(controller.phase, 'failed')
        servo.assert_not_called()

    def test_object_displacement_mutation_fails_before_servo(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        moved_objects = {'obj': {'position': [0.0, 0.0, 0.96]}}
        mutated = make_reading(objects=moved_objects)
        with mock.patch('preparation_diagnostics.servo_action') as servo:
            action = controller.next_action(mutated)
        self.assertIsNone(action)
        self.assertEqual(controller.phase, 'failed')
        servo.assert_not_called()

    def test_true_goal_lost_mutation_fails_before_servo(self):
        reading = make_reading(predicates={'on_table': True})
        controller = self._controller(reading=reading)
        mutated = make_reading(predicates={})
        with mock.patch('preparation_diagnostics.servo_action') as servo:
            action = controller.next_action(mutated)
        self.assertIsNone(action)
        self.assertEqual(controller.phase, 'failed')
        servo.assert_not_called()

    def test_robot_contact_mutation_fails_before_servo(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        mutated = make_reading(robot_contacts=[{'robot_geom': 'g', 'other_geom': 'o'}])
        with mock.patch('preparation_diagnostics.servo_action') as servo:
            action = controller.next_action(mutated)
        self.assertIsNone(action)
        self.assertEqual(controller.phase, 'failed')
        servo.assert_not_called()

    def test_observe_after_malformed_reading_fails(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        malformed = {'pose': {'position': [0.0, float('nan'), 1.0],
                              'orientation_matrix': [list(r) for r in IDENTITY]}}
        controller.observe_after(malformed)
        self.assertEqual(controller.phase, 'failed')
        self.assertEqual(controller._actual_aux_actions, 1)

    def test_repeated_next_action_does_not_count_actions(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        with mock.patch('preparation_diagnostics.servo_action',
                               return_value=np.zeros(7, dtype=np.float32)):
            for _ in range(5):
                controller.next_action(reading)
        self.assertEqual(controller._actual_aux_actions, 0)

    def test_total_action_budget_boundary(self):
        reading = make_reading()
        controller = self._controller(reading=reading, max_actions=2)
        controller._actual_aux_actions = 2
        action = controller.next_action(reading)
        self.assertIsNone(action)
        self.assertEqual(controller.phase, 'failed')
        self.assertEqual(controller.reason, 'prepare_budget_exhausted')

    def test_stage_action_budget_boundary(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        waypoint = controller._plan['waypoints'][0]
        far_pos = np.asarray(waypoint['position'], dtype=float).copy()
        far_pos[0] += 0.03
        far_rot = np.asarray(waypoint['orientation'], dtype=float)
        local_hand = reading['hand'][0]
        far_reading = dict(reading)
        far_reading['pose'] = {
            'position': [float(v) for v in far_pos],
            'orientation_matrix': [list(r) for r in far_rot],
        }
        far_reading['hand'] = [dict(local_hand,
                                    center=far_pos + far_rot @ np.asarray(local_hand['center'], dtype=float),
                                    rotation=far_rot @ np.asarray(local_hand['rotation'], dtype=float))]
        controller._stage_counts[waypoint['name']] = 59
        controller._actual_aux_actions = 58
        controller.observe_after(far_reading)
        self.assertEqual(controller.phase, 'failed')
        self.assertEqual(controller.reason, 'stage_timeout')

    def test_preparation_success_is_preparation_only(self):
        reading = make_reading()
        controller = self._controller(reading=reading)
        summary = controller.summary()
        self.assertEqual(summary['confirmation'], 'preparation_only_not_grasp_or_placement')


class ServiceTests(unittest.TestCase):
    def _make_service(self, assist_mode='enabled'):
        import subtask_assist_service as sas
        service = object.__new__(sas.UnifiedAssistService)
        service._subtask_assist_mode = assist_mode
        service._wine_input_task = 'put obj on table'
        service._subtask_calibration_profiles = {}
        service._subtask_home_orientation = IDENTITY
        service._subtask_kind = None
        service._subtask_context = None
        service._subtask_selection = None
        service._subtask_reading = None
        service._subtask_helper = None
        service._subtask_context_error = None
        service._subtask_observer_error = None
        service._subtask_prepare_confirmed = False
        service._subtask_prepare_local_grasp_actions = 0
        service._subtask_prepare_actions = 0
        service._subtask_vla_actions = 0
        service._subtask_step = 0
        service._subtask_queue_reset_done = False
        service._subtask_extra_goals = None
        service._subtask_prepare_execution = None
        service._grasp_assist_capability = None
        service._grasp_assist_job_id = None
        service._grasp_assist_step = 0
        service._grasp_assist_trigger_proposal = None
        service._grasp_assist_local_actions = 0
        service._grasp_assist_vla_actions = 0
        return service, sas

    def test_disabled_mode_skips_geometry(self):
        service, sas = self._make_service(assist_mode='disabled')
        env = mock.MagicMock()
        capability = {'goals': [['on', 'obj', 'table']]}
        with mock.patch.object(sas.subtask_geometry, 'read_geometry') as read_geometry:
            decision = service._decide_execution_kind(env, capability, 'cap', mock.MagicMock())
        self.assertEqual(decision['kind'], sas.KIND_VLA)
        read_geometry.assert_not_called()

    def test_non_pickplace_skips_geometry(self):
        service, sas = self._make_service()
        env = mock.MagicMock()
        capability = {'goals': [['turnon', 'lamp']]}
        with mock.patch.object(sas.subtask_geometry, 'read_geometry') as read_geometry:
            decision = service._decide_execution_kind(env, capability, 'cap', mock.MagicMock())
        self.assertEqual(decision['kind'], sas.KIND_VLA)
        read_geometry.assert_not_called()

    def test_geometry_exception_blocks_without_vla(self):
        import threading
        import subtask_assist_service as sas

        service, sas_mod = self._make_service()
        service._lock = threading.RLock()
        service._total_steps = 0
        service._require_ready = lambda: None
        service._completion_ready = lambda env, goals, preds: True
        service._last_completion_status = None
        service._apply_completion_probe = lambda job, status: None
        service._holding_guard_reason = lambda goals, status: None
        service._grasp_guard_applies = lambda cid, cap: False
        service._wine_semantic_applies = lambda cid, cap: False
        service._safe_render = lambda env: None
        service._save_frame = lambda path, frame: None
        service._reset_policy_queues = lambda: None

        env = mock.MagicMock()
        env.action_space.low = np.full(7, -1.0, dtype=np.float32)
        env.action_space.high = np.full(7, 1.0, dtype=np.float32)
        service._env = env
        service._env_session_id = 's1'

        import service as _base_service
        service.SessionRecord = _base_service.SessionRecord
        service.PlanRecord = _base_service.PlanRecord
        service.JobRecord = _base_service.JobRecord
        import subtask_assist_service as _sas_selfmod
        service.SceneError = _sas_selfmod.service.SceneError

        session = service.SessionRecord(
            session_id='s1', scene_id='sc', scene={}, seed=0,
            init_state_index=0, run_dir=Path(tempfile.mkdtemp()),
        )
        plan = service.PlanRecord(
            request_id='r1', session_id='s1', payload={'capability_ids': ['cap'], 'budget_per_subgoal': 10},
        )
        job_dir = Path(tempfile.mkdtemp())
        job = service.JobRecord(
            job_id='j1', request_id='r1', session_id='s1',
            capability_id='cap', run_dir=job_dir,
        )

        capability_payload = {
            'goals': [['on', 'obj', 'table']],
            'object_id': 'obj',
            'target_id': 'table',
            'instruction': 'put obj on table',
        }

        real_read = sas_mod.subtask_geometry.read_geometry

        def boom(*a, **k):
            raise sas_mod.subtask_geometry.GeometryError('bad geometry')

        wine_calls = {'n': 0}

        def wine_run(self_arg, *a, **k):
            wine_calls['n'] += 1
            return {'ok': True}

        vla_calls = {'n': 0}

        def vla_select(self_arg, batch):
            vla_calls['n'] += 1
            return np.zeros(7, dtype=np.float32)

        def env_step(action):
            raise AssertionError('env.step must not be called')

        env.step = env_step

        with mock.patch.object(sas_mod.catalog, 'CAPABILITIES', {'cap': capability_payload}), \
                mock.patch.object(sas_mod.subtask_geometry, 'read_geometry', side_effect=boom), \
                mock.patch.object(sas_mod.wd.WineDiagnosticService, '_run_capability', new=wine_run), \
                mock.patch.object(sas_mod.wd.WineDiagnosticService, '_select_action', new=vla_select):
            with self.assertRaises(service.SceneError) as ctx:
                service._run_capability(session, plan, job, 'cap')

        self.assertEqual(ctx.exception.reason, 'subtask_assist_blocked')
        self.assertEqual(job.state, 'error')
        self.assertEqual(job.steps, 0)
        self.assertFalse(job.success)
        self.assertEqual(wine_calls['n'], 0)
        self.assertEqual(vla_calls['n'], 0)

        import json as _json
        report_path = job_dir / 'subtask_assist.json'
        self.assertTrue(report_path.exists())
        report = _json.loads(report_path.read_text(encoding='utf-8'))
        self.assertTrue(report.get('blocked'))
        del real_read

    def test_parent_capability_path_used(self):
        import threading
        import json as _json
        import subtask_assist_service as sas

        service, sas_mod = self._make_service(assist_mode=sas.MODE_DISABLED)
        service._lock = threading.RLock()
        service._total_steps = 0
        service._require_ready = lambda: None
        service._completion_ready = lambda env, goals, preds: True
        service._last_completion_status = None
        service._apply_completion_probe = lambda job, status: None
        service._holding_guard_reason = lambda goals, status: None
        service._grasp_guard_applies = lambda cid, cap: False
        service._wine_semantic_applies = lambda cid, cap: False
        service._safe_render = lambda env: None
        service._save_frame = lambda path, frame: None
        service._clear_wine_evidence = lambda: None

        reset_calls = {'n': 0}
        real_reset = service._reset_policy_queues
        service._reset_policy_queues = lambda: reset_calls.__setitem__('n', reset_calls['n'] + 1)

        env_calls = {'n': 0}
        env = mock.MagicMock()
        env.action_space.low = np.full(7, -1.0, dtype=np.float32)
        env.action_space.high = np.full(7, 1.0, dtype=np.float32)

        def original_step(a):
            env_calls['n'] += 1
            return ('obs', 0.0, False, False, {'is_success': False})

        env.step = original_step
        service._env = env
        service._env_session_id = 's1'
        service._observation_batch = lambda obs, instr: {'batch': 1}

        service.SessionRecord = sas_mod.service.SessionRecord
        service.PlanRecord = sas_mod.service.PlanRecord
        service.JobRecord = sas_mod.service.JobRecord
        service.SceneError = sas_mod.service.SceneError

        session = service.SessionRecord(
            session_id='s1', scene_id='sc', scene={}, seed=0,
            init_state_index=0, run_dir=Path(tempfile.mkdtemp()),
        )
        plan = service.PlanRecord(
            request_id='r1', session_id='s1',
            payload={'capability_ids': ['cap'], 'budget_per_subgoal': 10},
        )
        job_dir = Path(tempfile.mkdtemp())
        job = service.JobRecord(
            job_id='j1', request_id='r1', session_id='s1',
            capability_id='cap', run_dir=job_dir,
        )

        capability_payload = {
            'goals': [['on', 'obj', 'table']],
            'object_id': 'obj',
            'target_id': 'table',
            'instruction': 'put obj on table',
        }

        wine_run_calls = {'n': 0}
        vla_select_calls = {'n': 0}

        def wine_select(self_arg, batch):
            vla_select_calls['n'] += 1
            return np.full(7, 0.1, dtype=np.float32)

        def wine_run(self_arg, session_arg, plan_arg, job_arg, cid):
            wine_run_calls['n'] += 1
            for _ in range(2):
                if plan_arg.cancel_event.is_set():
                    break
                batch = self_arg._observation_batch(None, capability_payload['instruction'])
                a = self_arg._select_action(batch)
                self_arg._env.step(a)
            with self_arg._lock:
                job_arg.steps = self_arg._subtask_step
                job_arg.total_steps = self_arg._subtask_step
            return {'ok': True, 'cancelled': False, 'reason': 'success', 'detail': '', 'steps': 2}

        gas_calls = {'n': 0}

        def gas_run(self_arg, *a, **k):
            gas_calls['n'] += 1
            raise AssertionError('gas must be bypassed')

        with mock.patch.object(sas_mod.catalog, 'CAPABILITIES', {'cap': capability_payload}), \
                mock.patch.object(sas_mod.wd.WineDiagnosticService, '_run_capability', new=wine_run), \
                mock.patch.object(sas_mod.wd.WineDiagnosticService, '_select_action', new=wine_select), \
                mock.patch.object(sas_mod.gas.GraspAssistService, '_run_capability', new=gas_run):
            result = service._run_capability(session, plan, job, 'cap')

        self.assertEqual(wine_run_calls['n'], 1)
        self.assertEqual(gas_calls['n'], 0)
        self.assertEqual(vla_select_calls['n'], 2)
        self.assertEqual(env_calls['n'], 2)
        self.assertTrue(result.get('ok'))
        self.assertIs(env.step, original_step)
        self.assertEqual(reset_calls['n'], 0)

        report_path = job_dir / 'subtask_assist.json'
        self.assertTrue(report_path.exists())
        report = _json.loads(report_path.read_text(encoding='utf-8'))
        self.assertEqual(report.get('capability_id'), 'cap')
        self.assertEqual(report.get('execution_kind'), sas_mod.KIND_VLA)
        self.assertEqual(report.get('actual_total_steps'), 2)
        self.assertEqual(report.get('vla_source_actions'), 2)
        self.assertEqual(report.get('prepare_source_actions'), 0)
        self.assertEqual(report.get('local_grasp_source_actions'), 0)

        sources_path = job_dir / 'action_sources.jsonl'
        self.assertTrue(sources_path.exists())
        lines = [ln for ln in sources_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        records = [_json.loads(ln) for ln in lines]
        for idx, rec in enumerate(records, start=1):
            self.assertEqual(rec['source'], sas_mod.SOURCE_VLA)
            self.assertEqual(rec['step'], idx)

        self.assertIsNone(service._subtask_helper)
        self.assertIsNone(service._subtask_prepare_execution)
        self.assertIsNone(service._subtask_extra_goals)
        self.assertEqual(service._subtask_prepare_actions, 0)
        self.assertEqual(service._subtask_vla_actions, 0)
        self.assertEqual(service._subtask_prepare_local_grasp_actions, 0)
        del real_reset

    def test_generic_first_action_uses_prepare_source(self):
        import threading
        import subtask_assist_service as sas

        service, sas_mod = self._make_service()
        service._lock = threading.RLock()
        service._require_ready = lambda: None
        service._clear_wine_evidence = lambda: None

        reading = make_reading()
        service._subtask_reading = reading
        service._subtask_home_orientation = reading['home_orientation']
        service._subtask_calibration_profiles = {}
        service._subtask_extra_goals = None
        service._grasp_assist_capability = 'cap'
        service._grasp_assist_source = sas_mod.SOURCE_VLA
        service._subtask_kind = sas_mod.KIND_GENERIC

        capability_payload = {
            'goals': [['on', 'obj', 'table']],
            'object_id': 'obj',
            'target_id': 'table',
            'instruction': 'put obj on table',
        }

        helper_action = np.zeros(7, dtype=np.float32)
        helper_action[6] = -1.0
        helper = mock.MagicMock()
        helper.phase = 'preparing'
        helper.next_action.return_value = helper_action
        service._subtask_helper = helper

        env = mock.MagicMock()
        env.action_space.low = np.full(7, -1.0, dtype=np.float32)
        env.action_space.high = np.full(7, 1.0, dtype=np.float32)
        service._env = env

        def wd_select(*a, **k):
            raise AssertionError('wd._select_action must not be called in preparing phase')

        with mock.patch.object(sas_mod.catalog, 'CAPABILITIES', {'cap': capability_payload}), \
                mock.patch.object(sas_mod.subtask_geometry, 'read_geometry', return_value=reading), \
                mock.patch.object(sas_mod.wd.WineDiagnosticService, '_select_action', new=wd_select):
            action = service._select_action(mock.MagicMock())

        self.assertIsNotNone(action)
        self.assertEqual(np.asarray(action).shape, (7,))
        self.assertTrue(np.isfinite(np.asarray(action, dtype=np.float64)).all())
        self.assertAlmostEqual(float(np.asarray(action)[6]), -1.0)
        self.assertEqual(service._grasp_assist_source, sas_mod.SOURCE_PREPARE)
        helper.next_action.assert_called_once()
        self.assertIs(service._grasp_assist_helper, helper)

    def test_ready_resets_queues_once(self):
        import threading
        import subtask_assist_service as sas

        service, sas_mod = self._make_service()
        service._lock = threading.RLock()

        helper = mock.MagicMock()
        helper.phase = 'ready'
        helper.next_action = lambda reading: None

        service._subtask_kind = sas_mod.KIND_GENERIC
        service._subtask_helper = helper
        service._subtask_observer_error = None
        service._grasp_assist_source = sas_mod.SOURCE_PREPARE
        service._grasp_assist_helper = helper
        service._subtask_queue_reset_done = False
        service._subtask_home_orientation = None
        service._subtask_calibration_profiles = {}
        service._subtask_extra_goals = None
        service._grasp_assist_capability = 'cap'

        reset_calls = {'n': 0}

        def reset_queues():
            reset_calls['n'] += 1

        service._reset_policy_queues = reset_queues

        received = []

        def wd_select(self_arg, batch):
            received.append(batch)
            return np.full(7, float(len(received)), dtype=np.float32)

        batch_1 = {'token': 'batch-A', 'obs': [1, 2, 3]}
        batch_2 = {'token': 'batch-B', 'obs': [4, 5, 6]}

        with mock.patch.object(sas_mod.wd.WineDiagnosticService, '_select_action', new=wd_select), \
                mock.patch.object(service, '_clear_wine_evidence', new=lambda: None):
            action_1 = service._select_action(batch_1)
            action_2 = service._select_action(batch_2)

        self.assertEqual(reset_calls['n'], 1)
        self.assertEqual(len(received), 2)
        self.assertIs(received[0], batch_1)
        self.assertIs(received[1], batch_2)
        self.assertEqual(np.asarray(action_1).shape, (7,))
        self.assertEqual(np.asarray(action_2).shape, (7,))
        self.assertEqual(service._grasp_assist_source, sas_mod.SOURCE_VLA)

    def test_reset_assist_session_state_clears_helper(self):
        service, sas = self._make_service()
        service._subtask_helper = mock.MagicMock()
        service._subtask_prepare_execution = 'something'
        service._subtask_extra_goals = [['on', 'a', 'b']]
        service._grasp_assist_helper = mock.MagicMock()
        service._reset_assist_session_state()
        self.assertIsNone(service._subtask_helper)
        self.assertIsNone(service._subtask_prepare_execution)
        self.assertIsNone(service._subtask_extra_goals)
        self.assertIsNone(service._grasp_assist_helper)

    def test_run_capability_action_source_audit_and_cleanup(self):
        import threading
        import json as _json
        import subtask_assist_service as sas

        service, sas_mod = self._make_service()
        service._lock = threading.RLock()
        service._total_steps = 0
        service._require_ready = lambda: None
        service._completion_ready = lambda env, goals, preds: True
        service._last_completion_status = None
        service._apply_completion_probe = lambda job, status: None
        service._holding_guard_reason = lambda goals, status: None
        service._grasp_guard_applies = lambda cid, cap: False
        service._wine_semantic_applies = lambda cid, cap: False
        service._safe_render = lambda env: None
        service._save_frame = lambda path, frame: None
        service._clear_wine_evidence = lambda: None

        reset_calls = {'n': 0}
        service._reset_policy_queues = lambda: reset_calls.__setitem__('n', reset_calls['n'] + 1)

        reading = make_reading()
        ready_reading = make_reading()
        service._subtask_home_orientation = reading['home_orientation']
        service._subtask_calibration_profiles = {}
        service._subtask_extra_goals = None
        service._subtask_reading = reading

        env_calls = {'n': 0}
        env = mock.MagicMock()
        env.action_space.low = np.full(7, -1.0, dtype=np.float32)
        env.action_space.high = np.full(7, 1.0, dtype=np.float32)

        def original_step(a):
            env_calls['n'] += 1
            return ('obs', 0.0, False, False, {'is_success': False})

        env.step = original_step
        service._env = env
        service._env_session_id = 's1'
        service._observation_batch = lambda obs, instr: {'batch': 'fresh'}

        service.SessionRecord = sas_mod.service.SessionRecord
        service.PlanRecord = sas_mod.service.PlanRecord
        service.JobRecord = sas_mod.service.JobRecord
        service.SceneError = sas_mod.service.SceneError

        session = service.SessionRecord(
            session_id='s1', scene_id='sc', scene={}, seed=0,
            init_state_index=0, run_dir=Path(tempfile.mkdtemp()),
        )
        plan = service.PlanRecord(
            request_id='r1', session_id='s1',
            payload={'capability_ids': ['cap'], 'budget_per_subgoal': 10},
        )
        job_dir = Path(tempfile.mkdtemp())
        job = service.JobRecord(
            job_id='j1', request_id='r1', session_id='s1',
            capability_id='cap', run_dir=job_dir,
        )

        capability_payload = {
            'goals': [['on', 'obj', 'table']],
            'object_id': 'obj',
            'target_id': 'table',
            'instruction': 'put obj on table',
        }

        prepare_action = np.zeros(7, dtype=np.float32)
        prepare_action[6] = -1.0

        observe_calls = {'n': 0}
        helper = mock.MagicMock()
        helper.phase = 'preparing'
        helper.next_action.return_value = prepare_action

        def observe_after(read_arg):
            observe_calls['n'] += 1
            helper.phase = 'ready'

        helper.observe_after.side_effect = observe_after
        helper.summary.return_value = {'phase': 'ready'}

        vla_select_batches = []

        def wine_select(self_arg, batch):
            vla_select_batches.append(batch)
            return np.full(7, 0.25, dtype=np.float32)

        def wine_run(self_arg, session_arg, plan_arg, job_arg, cid):
            batch1 = self_arg._observation_batch(None, capability_payload['instruction'])
            a1 = self_arg._select_action(batch1)
            self_arg._env.step(a1)
            batch2 = self_arg._observation_batch(None, capability_payload['instruction'])
            a2 = self_arg._select_action(batch2)
            self_arg._env.step(a2)
            return {'ok': True, 'cancelled': False, 'reason': 'success', 'detail': ''}

        gas_calls = {'n': 0}

        def gas_run(self_arg, *a, **k):
            gas_calls['n'] += 1
            raise AssertionError('gas must be bypassed')

        def decide(env_arg, cap_arg, cid, plan_arg):
            service._subtask_kind = sas_mod.KIND_GENERIC
            service._subtask_context = None
            service._subtask_helper = helper
            return {'kind': sas_mod.KIND_GENERIC, 'reason': 'test'}

        with mock.patch.object(sas_mod.catalog, 'CAPABILITIES', {'cap': capability_payload}), \
                mock.patch.object(sas_mod.subtask_geometry, 'read_geometry', side_effect=[reading, ready_reading]), \
                mock.patch.object(sas_mod.wd.WineDiagnosticService, '_run_capability', new=wine_run), \
                mock.patch.object(sas_mod.wd.WineDiagnosticService, '_select_action', new=wine_select), \
                mock.patch.object(sas_mod.gas.GraspAssistService, '_run_capability', new=gas_run), \
                mock.patch.object(service, '_decide_execution_kind', new=decide):
            result = service._run_capability(session, plan, job, 'cap')

        self.assertEqual(env_calls['n'], 2)
        self.assertIs(env.step, original_step)
        self.assertEqual(observe_calls['n'], 1)
        self.assertEqual(reset_calls['n'], 1)
        self.assertEqual(gas_calls['n'], 0)
        self.assertEqual(len(vla_select_batches), 1)
        self.assertEqual(vla_select_batches[0], {'batch': 'fresh'})
        self.assertTrue(result.get('ok'))

        sources_path = job_dir / 'action_sources.jsonl'
        self.assertTrue(sources_path.exists())
        lines = [ln for ln in sources_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        records = [_json.loads(ln) for ln in lines]
        self.assertEqual(records[0]['source'], sas_mod.SOURCE_PREPARE)
        self.assertEqual(records[1]['source'], sas_mod.SOURCE_VLA)
        self.assertEqual(records[0]['step'], 1)
        self.assertEqual(records[1]['step'], 2)
        self.assertEqual(records[0]['phase_before'], 'preparing')
        self.assertEqual(records[0]['phase_after'], 'ready')
        self.assertEqual(records[1]['phase_before'], 'ready')
        np.testing.assert_allclose(records[0]['sent_action'], prepare_action)

        report_path = job_dir / 'subtask_assist.json'
        self.assertTrue(report_path.exists())
        report = _json.loads(report_path.read_text(encoding='utf-8'))
        self.assertEqual(report.get('actual_total_steps'), 2)
        self.assertEqual(report.get('prepare_source_actions'), 1)
        self.assertEqual(report.get('vla_source_actions'), 1)
        self.assertEqual(report.get('local_grasp_source_actions'), 0)
        self.assertTrue(report.get('prepare_confirmed'))

        self.assertIsNone(service._subtask_helper)
        self.assertIsNone(service._subtask_prepare_execution)
        self.assertIsNone(service._subtask_extra_goals)
        self.assertEqual(service._subtask_prepare_actions, 0)
        self.assertEqual(service._subtask_vla_actions, 0)
        self.assertEqual(service._subtask_prepare_local_grasp_actions, 0)
        self.assertEqual(service._subtask_step, 0)

    def test_base_run_capability_cancellation_prevents_step(self):
        import threading
        import service as base_service

        svc = object.__new__(base_service.SceneService)
        svc._lock = threading.RLock()
        svc._total_steps = 0
        svc._completion_ready = lambda env, goals, preds: False
        svc._last_completion_status = None
        svc._apply_completion_probe = lambda job, status: None
        svc._holding_guard_reason = lambda goals, status: None
        svc._grasp_guard_applies = lambda cid, cap: False
        svc._wine_semantic_applies = lambda cid, cap: False
        svc._safe_render = lambda env: None
        svc._save_frame = lambda path, frame: None
        svc._update_images = lambda session, env: None
        svc._reset_policy_queues = lambda: None
        svc._require_ready = lambda: None
        svc._stop = threading.Event()
        svc.completion_mode = 'native'
        svc.grasp_guard_mode = 'off'
        svc._env_instance_counter = 0

        env = mock.MagicMock()
        env.action_space.low = np.full(7, -1.0, dtype=np.float32)
        env.action_space.high = np.full(7, 1.0, dtype=np.float32)
        env_calls = {'n': 0}

        def env_step(action):
            env_calls['n'] += 1
            return ('obs', 0.0, False, False, {'is_success': False})

        env.step = mock.MagicMock(side_effect=env_step)
        svc._env = env
        svc._env_session_id = 's1'
        svc._last_obs = 'obs0'
        svc._action_function = None

        session = base_service.SessionRecord(
            session_id='s1', scene_id='sc', scene={}, seed=0,
            init_state_index=0, run_dir=Path(tempfile.mkdtemp()),
        )
        plan = base_service.PlanRecord(
            request_id='r1', session_id='s1', payload={'capability_ids': ['cap'], 'budget_per_subgoal': 5},
        )
        job_dir = Path(tempfile.mkdtemp())
        job = base_service.JobRecord(
            job_id='j1', request_id='r1', session_id='s1',
            capability_id='cap', run_dir=job_dir,
        )

        capability_payload = {
            'goals': [['on', 'obj', 'table']],
            'instruction': 'put obj on table',
        }

        svc._observation_batch = lambda obs, instr: {'token': 'b'}

        selected = {'n': 0}

        def select(batch):
            selected['n'] += 1
            plan.cancel_event.set()
            return np.zeros(7, dtype=np.float32)

        svc._select_action = select

        with mock.patch.object(base_service.catalog, 'CAPABILITIES', {'cap': capability_payload}), \
                mock.patch.object(base_service, 'eval_goal_predicate', new=lambda env_arg, goal: False), \
                mock.patch.object(base_service, 'state_sha', new=lambda env_arg: 'deadbeef'), \
                mock.patch.object(base_service, '_save_video', new=lambda *a, **k: None):
            result = svc._run_capability(session, plan, job, 'cap')

        self.assertGreaterEqual(selected['n'], 1)
        self.assertEqual(env_calls['n'], 0)
        self.assertEqual(job.state, 'cancelled')
        self.assertEqual(job.steps, 0)
        self.assertEqual(job.total_steps, 0)
        self.assertFalse(job.success)
        self.assertEqual(job.ended_reason, 'cancelled')
        self.assertFalse(result.get('ok'))
        self.assertTrue(result.get('cancelled'))

    def test_native_goals_already_true_short_circuits_preview(self):
        from unittest import mock

        service, sas = self._make_service()
        catalog = sas.catalog
        goals = [['on', 'obj', 'table'], ['in', 'obj', 'table']]
        capability = ContextTests()._capability(object_id='obj', target_id='table', goals=goals)
        reading = ContextTests()._reading(position=(0.0, 0.0, 1.0), complete=True)
        reading['snapshot']['predicates'] = {
            catalog.goal_key(goals[0]): True,
            catalog.goal_key(goals[1]): True,
        }
        plan = mock.Mock()
        plan.completed_capability_ids = []

        with mock.patch.object(
            sas.subtask_geometry,
            'read_geometry',
            return_value=reading,
        ) as read_geometry:
            with mock.patch.object(
                service, '_preview_action'
            ) as preview_action:
                result = service._decide_execution_kind(
                    object(), capability, 'cap0', plan
                )

        self.assertEqual(
            result,
            {
                'kind': sas.KIND_VLA,
                'reason': 'native_goals_already_true_use_original_completion_gate',
            },
        )
        self.assertEqual(service._subtask_kind, sas.KIND_VLA)
        self.assertIsNone(service._subtask_helper)
        self.assertIsNone(service._subtask_prepare_execution)
        read_geometry.assert_called_once()
        preview_action.assert_not_called()

    def test_native_goal_predicate_false_or_none_does_not_short_circuit(self):
        from unittest import mock

        service, sas = self._make_service()
        catalog = sas.catalog
        goals = [['on', 'obj', 'table'], ['in', 'obj', 'table']]
        for predicate_value in (False, None):
            with self.subTest(predicate_value=predicate_value):
                service, sas = self._make_service()
                capability = ContextTests()._capability(
                    object_id='obj', target_id='table', goals=goals
                )
                reading = ContextTests()._reading(
                    position=(0.0, 0.0, 1.0), complete=True
                )
                predicate_key = catalog.goal_key(goals[0])
                reading['snapshot']['predicates'] = {
                    catalog.goal_key(goals[0]): predicate_value,
                    catalog.goal_key(goals[1]): True,
                }
                plan = mock.Mock()
                plan.completed_capability_ids = []
                helper = object()
                preview = {
                    'kind': sas.KIND_GENERIC,
                    'reason': 'preview_selected',
                    'helper': helper,
                    'prepare_execution': 'fake_prepare_execution',
                }

                with mock.patch.object(
                    sas.subtask_geometry,
                    'read_geometry',
                    return_value=reading,
                ):
                    with mock.patch.object(
                        service, '_preview_action', return_value=preview
                    ) as preview_action:
                        result = service._decide_execution_kind(
                            object(), capability, 'cap0', plan
                        )

                self.assertEqual(result['kind'], sas.KIND_GENERIC)
                self.assertEqual(result['reason'], 'preview_selected')
                self.assertEqual(service._subtask_kind, sas.KIND_GENERIC)
                self.assertIs(service._subtask_helper, helper)
                self.assertEqual(
                    service._subtask_prepare_execution, 'fake_prepare_execution'
                )
                preview_action.assert_called_once()
                self.assertEqual(
                    reading['snapshot']['predicates'][predicate_key],
                    predicate_value,
                )

    def test_native_goals_all_true_but_foreign_held_remains_blocked(self):
        from unittest import mock

        service, sas = self._make_service()
        catalog = sas.catalog
        goals = [['on', 'obj', 'table'], ['in', 'obj', 'table']]
        capability = ContextTests()._capability(object_id='obj', target_id='table', goals=goals)
        reading = ContextTests()._reading(
            position=(0.0, 0.0, 1.0),
            held=('foreign',),
            complete=True,
        )
        reading['snapshot']['predicates'] = {
            catalog.goal_key(goals[0]): True,
            catalog.goal_key(goals[1]): True,
        }
        plan = mock.Mock()
        plan.completed_capability_ids = []

        with mock.patch.object(
            sas.subtask_geometry,
            'read_geometry',
            return_value=reading,
        ):
            with mock.patch.object(
                service, '_preview_action'
            ) as preview_action:
                result = service._decide_execution_kind(
                    object(), capability, 'cap0', plan
                )

        self.assertEqual(result['kind'], sas.KIND_VLA)
        self.assertEqual(result['reason'], 'foreign_object_held')
        self.assertTrue(result.get('blocked'))
        self.assertIsNone(service._subtask_helper)
        self.assertIsNone(service._subtask_prepare_execution)
        preview_action.assert_not_called()



class ConvexCertificateTests(unittest.TestCase):
    TETRA = [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]

    def _tetra_convex(self):
        return sg._mesh_convex_local(self.TETRA, localcenter=np.zeros(3))

    def _obb(self, center, half, rotation=None, convex=None):
        if rotation is None:
            rotation = np.eye(3)
        obb = {
            "center": np.asarray(center, dtype=np.float64),
            "rotation": np.asarray(rotation, dtype=np.float64),
            "half": np.asarray(half, dtype=np.float64),
            "name": "test_obb",
        }
        if convex is not None:
            obb["convex"] = convex
        return obb

    def test_real_tetra_outside_box_certificate_clear(self):
        convex = self._tetra_convex()
        a = self._obb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], convex=convex)
        b = self._obb([0.75, 0.75, 0.75], [0.1, 0.1, 0.1])
        self.assertTrue(sp.obb_overlap(a, b))
        cert = sp.support_certificate(a, b)
        self.assertTrue(cert["certified_clear"])
        self.assertGreater(cert["maximum_separation_gap_m"], 0.006)

    def test_inside_box_not_falsely_excluded(self):
        convex = self._tetra_convex()
        a = self._obb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], convex=convex)
        b = self._obb([-0.5, -0.5, -0.5], [0.05, 0.05, 0.05])
        cert = sp.support_certificate(a, b)
        self.assertFalse(cert["certified_clear"])
        self.assertEqual(cert["hand"]["basis"], "actual_mesh_convex_hull")
        self.assertEqual(cert["obstacle"]["basis"], "unchanged_conservative_obb")

    def test_global_rigid_transform_preserves_gap(self):
        convex = self._tetra_convex()
        a0 = self._obb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], convex=convex)
        b0 = self._obb([0.75, 0.75, 0.75], [0.1, 0.1, 0.1])
        base = sp.support_certificate(a0, b0)
        R = rotation_z(0.7) @ rotation_x(0.4)
        t = np.array([0.3, -0.2, 1.1])
        a1 = self._obb(R @ a0["center"] + t, a0["half"], rotation=R @ a0["rotation"], convex=convex)
        b1 = self._obb(R @ b0["center"] + t, b0["half"], rotation=R @ b0["rotation"])
        moved = sp.support_certificate(a1, b1)
        self.assertAlmostEqual(moved["maximum_separation_gap_m"], base["maximum_separation_gap_m"], places=9)
        self.assertEqual(moved["certified_clear"], base["certified_clear"])

    def test_malformed_convex_raises_preparation_error(self):
        a_ok = self._obb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], convex=self._tetra_convex())
        b = self._obb([0.75, 0.75, 0.75], [0.1, 0.1, 0.1])
        convex = self._tetra_convex()
        bad_cases = [
            {"face_axes": convex["face_axes"], "edge_axes": convex["edge_axes"]},
            {"vertices": convex["vertices"], "edge_axes": convex["edge_axes"]},
            {"vertices": convex["vertices"], "face_axes": convex["face_axes"]},
            {"vertices": convex["vertices"], "face_axes": convex["face_axes"],
             "edge_axes": np.full_like(convex["edge_axes"], np.nan)},
        ]
        with self.subTest(bad="malformed"):
            for bad in bad_cases:
                obb = dict(a_ok)
                obb["convex"] = bad
                with self.assertRaises(sp.PreparationError):
                    sp.support_certificate(obb, b)

    def test_distinct_hull_data_not_shared_cache(self):
        a_convex = self._tetra_convex()
        mixed = [[0.0, 0.0, 0.0], [-1, -1, -1], [1.0, -1, -1], [-1, 1, -1], [-1, -1, 1]]
        b_convex = sg._mesh_convex_local(mixed, localcenter=np.zeros(3))
        for convex in (a_convex, b_convex):
            self.assertTrue(np.all(np.isfinite(convex["vertices"])))
            self.assertTrue(np.all(np.isfinite(convex["face_axes"])))
            self.assertTrue(np.all(np.isfinite(convex["edge_axes"])))
        self.assertGreater(np.abs(a_convex["vertices"]).max(), 0.0)
        self.assertGreater(np.abs(b_convex["vertices"]).max(), 0.0)
        again = sg._mesh_convex_local(mixed, localcenter=np.zeros(3))
        np.testing.assert_allclose(again["vertices"], b_convex["vertices"])
        np.testing.assert_allclose(again["face_axes"], b_convex["face_axes"])
        np.testing.assert_allclose(again["edge_axes"], b_convex["edge_axes"])

if __name__ == '__main__':
    unittest.main()