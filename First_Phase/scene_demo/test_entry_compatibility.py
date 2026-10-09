import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

import entry_compatibility as ep
import subtask_preparation as sp

IDENTITY = np.eye(3, dtype=float)


def rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


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


def make_ctx(position=(0.0, 0.0, 0.95), orientation=None, dimensions=(0.1, 0.1, 0.05),
             shape_family='low_wide', object_id='obj', target_id='dest'):
    orientation = IDENTITY if orientation is None else orientation
    return SimpleNamespace(
        operation='pick_place',
        object_id=object_id,
        target_id=target_id,
        goals=(),
        position=tuple(position),
        orientation=np.asarray(orientation, dtype=float),
        dimensions=tuple(dimensions),
        shape_family=shape_family,
        geometry_source='mujoco_collision_obbs',
        held_objects=(),
        grasp_observation_complete=True,
    )


def apply_ctx_target(reading, ctx):
    reading['target']['position'] = [float(v) for v in ctx.position]
    reading['target']['orientation'] = [list(r) for r in np.asarray(ctx.orientation, dtype=float)]
    reading['target']['dimensions'] = [float(v) for v in ctx.dimensions]
    return reading


def default_reading(ctx=None, position=(0.0, 0.0, 1.2), orientation=None, gap=0.08, **kw):
    if ctx is None:
        ctx = make_ctx()
    orientation = rotation_x(math.pi) if orientation is None else orientation
    r = make_reading(position=position, orientation=orientation, gap=gap, **kw)
    r['snapshot']['objects'] = {
        ctx.object_id: {'position': [float(v) for v in ctx.position], 'grasped': False},
        'protected': {'position': [0.3, 0.2, 0.95], 'grasped': False},
    }
    apply_ctx_target(r, ctx)
    return r


def default_ctx():
    return make_ctx()


def default_prov():
    return {'cut_step': 30, 'source_sha': 'abc'}


class TestReferenceTarget(unittest.TestCase):
    def setUp(self):
        self.ctx = default_ctx()
        self.reading = default_reading(self.ctx)
        self.prov = default_prov()

    def test_make_entry_reference_roundtrip(self):
        ref = ep.make_entry_reference(self.ctx, self.reading, provenance=self.prov)
        self.assertEqual(ref['schema_version'], 1)
        self.assertEqual(ref['operation'], 'pick_place')
        self.assertEqual(ref['shape_family'], 'low_wide')
        self.assertEqual(ref['provenance'], self.prov)
        self.assertIsNot(ref['provenance'], self.prov)
        np.testing.assert_allclose(ref['reference_dimensions'], [0.1, 0.1, 0.05])
        tgt = ep.entry_target(ref, self.ctx, self.reading)
        # Object-relative ref position uses object_R; with current object_R=I the target returns the EEF position.
        np.testing.assert_allclose(tgt['position'], [0.0, 0.0, 1.2], atol=1e-9)
        np.testing.assert_allclose(tgt['orientation'], rotation_x(math.pi), atol=1e-9)
        self.assertEqual(tgt['geometry_compatibility'], 'diagnostic_reference_only')

    def test_reference_target_with_translation_and_rotation(self):
        ctx = make_ctx(position=(0.2, -0.1, 0.9))
        eef = np.array([0.2, -0.1, 1.18])
        reading = default_reading(ctx, position=tuple(eef))
        ref = ep.make_entry_reference(ctx, reading, provenance=self.prov)
        np.testing.assert_allclose(ref['position_normalized'], [0.0, 0.0, 5.6], atol=1e-9)
        tgt = ep.entry_target(ref, ctx, reading)
        np.testing.assert_allclose(tgt['position'], eef, atol=1e-9)

    def test_object_translation_rz_dims_scale(self):
        ctx = make_ctx(position=(0.1, 0.2, 0.9), orientation=rotation_z(math.pi / 2),
                       dimensions=(0.1, 0.1, 0.05))
        # EEF at object + Rz @ (0.03, 0, 0.13)
        offset = rotation_z(math.pi / 2) @ np.array([0.03, 0.0, 0.13])
        eef = np.array([0.1, 0.2, 0.9]) + offset
        reading = default_reading(ctx, position=tuple(eef), orientation=rotation_z(math.pi / 2) @ rotation_x(math.pi))
        ref = ep.make_entry_reference(ctx, reading, provenance=self.prov)
        np.testing.assert_allclose(ref['position_normalized'], [0.3, 0.0, 2.6], atol=1e-9)
        # Now scale dims by 1.05
        ctx2 = make_ctx(position=(0.1, 0.2, 0.9), orientation=rotation_z(math.pi / 2),
                        dimensions=(0.105, 0.105, 0.0525))
        eef2 = eef * 1.05
        reading2 = default_reading(ctx2, position=tuple(eef2), orientation=rotation_z(math.pi / 2) @ rotation_x(math.pi))
        tgt = ep.entry_target(ref, ctx2, reading2)
        expected_pos = np.array([0.1, 0.2, 0.9]) + rotation_z(math.pi / 2) @ (np.array([0.03, 0.0, 0.13]) * 1.05)
        np.testing.assert_allclose(tgt['position'], expected_pos, atol=1e-9)
        np.testing.assert_allclose(tgt['orientation'], rotation_z(math.pi / 2) @ rotation_x(math.pi), atol=1e-9)

    def test_rename_object_id_consistently(self):
        ctx = make_ctx(object_id='thing')
        reading = default_reading(ctx)
        ref = ep.make_entry_reference(ctx, reading, provenance=self.prov)
        tgt = ep.entry_target(ref, ctx, reading)
        np.testing.assert_allclose(tgt['position'], [0.0, 0.0, 1.2], atol=1e-9)
        self.assertIn(ctx.object_id, reading['snapshot']['objects'])
        self.assertNotIn(ctx.object_id, ref)

    def test_provenance_is_deep_copied(self):
        nested = {'cut_step': 30, 'source_sha': 'abc', 'meta': {'a': [1, 2]}}
        ref = ep.make_entry_reference(self.ctx, self.reading, provenance=nested)
        nested['meta']['a'].append(3)
        nested['cut_step'] = 99
        self.assertEqual(ref['provenance']['meta']['a'], [1, 2])
        self.assertEqual(ref['provenance']['cut_step'], 30)


class TestValidationRejections(unittest.TestCase):
    def setUp(self):
        self.ctx = default_ctx()
        self.reading = default_reading(self.ctx)
        self.prov = default_prov()

    def _ref(self):
        return ep.make_entry_reference(self.ctx, self.reading, provenance=self.prov)

    def test_reference_invalid_geometry(self):
        cases = {
            'shape_family_wrong': lambda r: r.update(shape_family='tall_thin'),
            'schema_bool': lambda r: r.update(schema_version=True),
            'schema_wrong': lambda r: r.update(schema_version=2),
            'provenance_empty': lambda r: r.update(provenance={}),
            'provenance_missing': lambda r: r.pop('provenance'),
            'gap_low': lambda r: r.update(gripper_gap_m=0.05),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                ref = self._ref()
                mutate(ref)
                with self.assertRaises(ep.EntryError):
                    ep.entry_target(ref, self.ctx, self.reading)

    def test_so3_rejects_reflection(self):
        ref = self._ref()
        bad = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, -1.0]])
        ref['orientation_relative'] = [list(r) for r in bad]
        with self.assertRaises(ep.EntryError):
            ep.entry_target(ref, self.ctx, self.reading)

    def test_so3_rejects_sub_allclose_orthogonality(self):
        ref = self._ref()
        bad = np.diag([1.0 + 2e-6, 1.0 - 2e-6, 1.0])
        # Within np.allclose default rtol=1e-5 -> allclose True, but SO3 needs rtol=0
        self.assertTrue(np.allclose(bad.T @ bad, np.eye(3)))
        ref['orientation_relative'] = [list(r) for r in bad]
        with self.assertRaises(ep.EntryError):
            ep.entry_target(ref, self.ctx, self.reading)

    def test_bool_vector_matrix_rejects(self):
        cases = [
            ('pos_norm_bool', 'position_normalized', [True, False, True]),
            ('pos_norm_partial_bool', 'position_normalized', [1.0, True, 0.0]),
            ('orient_bool', 'orientation_relative', [[True, False, False], [False, True, False], [False, False, True]]),
            ('dims_bool', 'reference_dimensions', [True, 0.1, 0.05]),
        ]
        for name, field, value in cases:
            with self.subTest(case=name):
                ref = self._ref()
                ref[field] = value
                with self.assertRaises(ep.EntryError):
                    ep.entry_target(ref, self.ctx, self.reading)

    def test_numeric_string_and_nan_and_inf_and_complex(self):
        cases = [
            ('nan_pos', 'position_normalized', [float('nan'), 0.0, 0.0]),
            ('inf_pos', 'position_normalized', [float('inf'), 0.0, 0.0]),
            ('str_pos', 'position_normalized', ['1.0', '0.0', '0.0']),
            ('complex_pos', 'position_normalized', [1 + 0j, 0j, 0j]),
            ('ragged_pos', 'position_normalized', [[1.0, 2.0], [3.0]]),
            ('nan_dims', 'reference_dimensions', [0.1, float('nan'), 0.05]),
            ('neg_dims', 'reference_dimensions', [-0.1, 0.1, 0.05]),
            ('zero_dims', 'reference_dimensions', [0.0, 0.1, 0.05]),
            ('str_dims', 'reference_dimensions', ['0.1', '0.1', '0.05']),
        ]
        for name, field, value in cases:
            with self.subTest(case=name):
                ref = self._ref()
                ref[field] = value
                with self.assertRaises(ep.EntryError):
                    ep.entry_target(ref, self.ctx, self.reading)

    def test_dimension_ratio_reject(self):
        ref = self._ref()
        ctx = make_ctx(dimensions=(0.2, 0.1, 0.05))
        reading = default_reading(ctx)
        with self.assertRaises(ep.EntryError):
            ep.entry_target(ref, ctx, reading)

    def test_gap_mismatch_reject(self):
        ref = self._ref()
        reading = default_reading(self.ctx, gap=0.2)
        with self.assertRaises(ep.EntryError):
            ep.entry_target(ref, self.ctx, reading)

    def test_bad_readings_reject(self):
        cases = {}
        def base():
            return default_reading(self.ctx)
        def held(r):
            r['snapshot']['held_objects'] = ['obj']
        def incomplete(r):
            r['snapshot']['grasp_observation_complete'] = False
        def missing_target(r):
            r.pop('target')
        def empty_objects(r):
            r['snapshot']['objects'] = {}
        def target_object_missing(r):
            r['snapshot']['objects'] = {'other': {'position': [0.0, 0.0, 0.95], 'grasped': False}}
        def target_pos_mismatch(r):
            r['snapshot']['objects']['obj']['position'] = [0.5, 0.0, 0.95]
        def unknown_grasp(r):
            r['snapshot']['objects']['obj']['grasped'] = True
        def contact(r):
            r['robot_contacts'] = ['link']
        def closed_gap(r):
            r['gripper_gap_m'] = 0.03
        cases = {
            'held': held, 'incomplete': incomplete, 'missing_target': missing_target,
            'empty_objects': empty_objects, 'target_object_missing': target_object_missing,
            'target_pos_mismatch': target_pos_mismatch, 'unknown_grasp': unknown_grasp,
            'contact': contact, 'closed_gap': closed_gap,
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                r = base()
                mutate(r)
                with self.assertRaises(ep.EntryError):
                    ep.make_entry_reference(self.ctx, r, provenance=self.prov)

    def test_pose_not_downward_reject(self):
        reading = default_reading(self.ctx, orientation=IDENTITY)
        with self.assertRaises(ep.EntryError):
            ep.make_entry_reference(self.ctx, reading, provenance=self.prov)

    def test_provenance_required(self):
        with self.assertRaises(ep.EntryError):
            ep.make_entry_reference(self.ctx, self.reading, provenance={})
        with self.assertRaises(ep.EntryError):
            ep.make_entry_reference(self.ctx, self.reading, provenance=None)


class TestControllerPlan(unittest.TestCase):
    def setUp(self):
        self.ctx = default_ctx()
        self.reading = default_reading(self.ctx)
        self.prov = default_prov()
        self.ref = ep.make_entry_reference(self.ctx, self.reading, provenance=self.prov)

    def test_plan_waypoints(self):
        ctl = ep.EntryPreparationController(self.ctx, self.reading, self.ref)
        summary = ctl.summary()
        names = [w['name'] for w in summary['plan']['waypoints']]
        self.assertIn('entry_align', names)
        self.assertIn('entry_ready', names)
        self.assertEqual(names[-2], 'entry_align')
        self.assertEqual(names[-1], 'entry_ready')
        generic_last = summary['plan']['waypoints'][-3]
        align = summary['plan']['waypoints'][-2]
        ready = summary['plan']['waypoints'][-1]
        np.testing.assert_allclose(align['position'], generic_last['position'], atol=1e-9)
        target = ep.entry_target(self.ref, self.ctx, self.reading)
        np.testing.assert_allclose(align['orientation'], target['orientation'], atol=1e-9)
        np.testing.assert_allclose(ready['position'], target['position'], atol=1e-9)
        np.testing.assert_allclose(ready['orientation'], target['orientation'], atol=1e-9)
        self.assertEqual(summary['generic_action_cap'], 200)
        self.assertEqual(summary['bridge_action_cap'], 100)
        self.assertEqual(summary['total_prepare_cap'], 300)

    def test_caller_mutation_does_not_affect_controller(self):
        ref_copy = copy.deepcopy(self.ref)
        ctl = ep.EntryPreparationController(self.ctx, self.reading, ref_copy)
        ref_copy['shape_family'] = 'mutated'
        ref_copy['position_normalized'] = [99.0, 99.0, 99.0]
        summary = ctl.summary()
        self.assertEqual(summary['entry_reference']['shape_family'], 'low_wide')

    def test_bridge_endpoint_collision_rejected(self):
        # Reference built from a clear reading whose EEF sits at the bridge endpoint.
        eef = (0.12, 0.0, 1.08)
        clear_reading = default_reading(self.ctx, position=eef)
        ref = ep.make_entry_reference(self.ctx, clear_reading, provenance=self.prov)
        # Runtime reading uses the default EEF and places a wall over the bridge endpoint.
        runtime_reading = default_reading(self.ctx)
        obstacle = {'name': 'entry_wall', 'center': [0.12, 0.0, 1.08], 'rotation': IDENTITY, 'half': [0.01, 0.01, 0.01]}
        runtime_reading['obstacles'] = [obstacle]
        # Parent controller must construct from the clear start first.
        sp.PreparationController(self.ctx, runtime_reading, max_actions=200)
        with self.assertRaises(ep.EntryError):
            ep.EntryPreparationController(self.ctx, runtime_reading, ref)


class TestRealObservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(sp, 'PreparationController'):
            raise unittest.SkipTest('parent PreparationController missing')

    def _make(self, position=(0.0, 0.0, 1.2), gap=0.08):
        ctx = default_ctx()
        reading = default_reading(ctx, position=position, gap=gap)
        ref = ep.make_entry_reference(ctx, reading, provenance=default_prov())
        ctl = ep.EntryPreparationController(ctx, reading, ref)
        return ctx, reading, ref, ctl

    def _reading_at(self, ctx, waypoint):
        p = tuple(float(v) for v in waypoint['position'])
        R = np.asarray(waypoint['orientation'], dtype=float)
        return default_reading(ctx, position=p, orientation=R)

    def _generic_count(self, ctl):
        return ctl._original_waypoint_count

    def test_four_confirmations_not_enough_fifth_advances(self):
        ctx, reading, ref, ctl = self._make()
        wp0 = ctl._plan['waypoints'][0]
        exact = self._reading_at(ctx, wp0)
        start_index = ctl._waypoint_index
        for i in range(4):
            ctl.observe_after(exact)
            self.assertEqual(ctl._waypoint_index, start_index)
        ctl.observe_after(exact)
        self.assertEqual(ctl._waypoint_index, start_index + 1)
        self.assertEqual(ctl._actual_aux_actions, 5)

    def test_generic_ready_freezes_generic_actions(self):
        ctx, reading, ref, ctl = self._make()
        n_generic = ctl._original_waypoint_count
        for i in range(n_generic):
            wp = ctl._plan['waypoints'][i]
            exact = self._reading_at(ctx, wp)
            for _ in range(5):
                ctl.observe_after(exact)
        s = ctl.summary()
        self.assertTrue(s['generic_ready'])
        self.assertEqual(s['phase'], 'preparing')
        self.assertEqual(s['generic_actions'], 5 * n_generic)
        self.assertEqual(s['bridge_actions'], 0)
        self.assertEqual(ctl.current_stage_name() if hasattr(ctl, 'current_stage_name') else ctl.summary()['current_stage'], 'entry_align')
        # Now bridge waypoint advancement works.
        wp_align = ctl._plan['waypoints'][n_generic]
        fp = self._reading_at(ctx, wp_align)
        for _ in range(5):
            ctl.observe_after(fp)
        self.assertEqual(ctl._waypoint_index, n_generic + 1)
        self.assertTrue(ctl.summary()['generic_ready'])
        self.assertEqual(ctl.summary()['generic_actions'], 5 * n_generic)

    def test_generic_ready_phase_preparing_for_bridge(self):
        ctx, reading, ref, ctl = self._make()
        n_generic = ctl._original_waypoint_count
        for i in range(n_generic):
            wp = ctl._plan['waypoints'][i]
            exact = self._reading_at(ctx, wp)
            for _ in range(5):
                ctl.observe_after(exact)
        s = ctl.summary()
        self.assertTrue(s['generic_ready'])
        self.assertEqual(s['phase'], 'preparing')
        # Finish bridge waypoints.
        for i in range(n_generic, len(ctl._plan['waypoints'])):
            wp = ctl._plan['waypoints'][i]
            exact = self._reading_at(ctx, wp)
            for _ in range(5):
                ctl.observe_after(exact)
        s = ctl.summary()
        self.assertEqual(s['phase'], 'ready')
        self.assertTrue(s['ready_confirmed'])
        self.assertEqual(s['generic_actions'], 5 * n_generic)
        self.assertEqual(s['bridge_actions'], 5 * (len(ctl._plan['waypoints']) - n_generic))
        self.assertEqual(s['actual_aux_actions'], s['generic_actions'] + s['bridge_actions'])


class TestBoundaries(unittest.TestCase):
    def _make_at(self, position=(0.0, 0.0, 1.2)):
        ctx = default_ctx()
        reading = default_reading(ctx, position=position)
        ref = ep.make_entry_reference(ctx, reading, provenance=default_prov())
        ctl = ep.EntryPreparationController(ctx, reading, ref)
        return ctx, ctl

    def _reading_at(self, ctx, waypoint):
        p = tuple(float(v) for v in waypoint['position'])
        R = np.asarray(waypoint['orientation'], dtype=float)
        return default_reading(ctx, position=p, orientation=R)

    def _reading_unaligned(self, ctx, waypoint, offset_xy=0.03):
        p = np.asarray(waypoint['position'], dtype=float) + np.array([offset_xy, offset_xy, 0.0])
        R = np.asarray(waypoint['orientation'], dtype=float)
        return default_reading(ctx, position=tuple(p), orientation=R)

    def test_generic_cap_exhausted_advances_bridge(self):
        ctx, ctl = self._make_at()
        n_generic = ctl._original_waypoint_count
        ctl._waypoint_index = n_generic - 1
        ctl._actual_aux_actions = 199
        ctl._consecutive_confirmations = 4
        wp = ctl._plan['waypoints'][n_generic - 1]
        ctl._stage_counts[wp['name']] = 4
        # Exact waypoint once: actions=200 -> cap hit; base advances first.
        exact = self._reading_at(ctx, wp)
        ctl.observe_after(exact)
        s = ctl.summary()
        self.assertEqual(s['phase'], 'preparing')
        self.assertIsNone(s['reason'])
        self.assertEqual(s['generic_actions'], 200)
        self.assertEqual(s['bridge_actions'], 0)
        self.assertTrue(s['generic_ready'])
        self.assertEqual(ctl._bridge_start_actions, 200)
        # Continue the same controller through both bridge waypoints.
        for i in range(n_generic, len(ctl._plan['waypoints'])):
            wp_b = ctl._plan['waypoints'][i]
            exact_b = self._reading_at(ctx, wp_b)
            for _ in range(5):
                ctl.observe_after(exact_b)
        s = ctl.summary()
        self.assertEqual(s['phase'], 'ready')
        self.assertEqual(s['actual_aux_actions'], 210)
        self.assertEqual(s['generic_actions'], 200)
        self.assertEqual(s['bridge_actions'], 10)

    def test_generic_budget_exhausted_at_start(self):
        ctx, ctl = self._make_at()
        ctl._waypoint_index = 0
        ctl._actual_aux_actions = 199
        ctl._consecutive_confirmations = 0
        wp = ctl._plan['waypoints'][0]
        safe_unaligned = self._reading_unaligned(ctx, wp, offset_xy=0.03)
        ctl.observe_after(safe_unaligned)
        self.assertEqual(ctl._phase, 'failed')
        self.assertEqual(ctl._reason, 'generic_prepare_budget_exhausted')

    def test_bridge_cap_exhausted(self):
        ctx, ctl = self._make_at()
        n_generic = ctl._original_waypoint_count
        ctl._waypoint_index = n_generic
        ctl._bridge_start_actions = 200
        ctl._actual_aux_actions = 299
        ctl._consecutive_confirmations = 0
        wp = ctl._plan['waypoints'][n_generic]
        safe_unaligned = self._reading_unaligned(ctx, wp, offset_xy=0.03)
        ctl.observe_after(safe_unaligned)
        self.assertEqual(ctl._phase, 'failed')
        self.assertEqual(ctl._reason, 'entry_bridge_budget_exhausted')
        self.assertTrue(ctl.summary()['generic_ready'])
        self.assertEqual(ctl.summary()['bridge_actions'], 100)

    def test_bridge_cap_reached_exactly(self):
        ctx, ctl = self._make_at()
        n_generic = ctl._original_waypoint_count
        ctl._waypoint_index = n_generic + 1
        ctl._bridge_start_actions = 200
        ctl._actual_aux_actions = 299
        ctl._consecutive_confirmations = 4
        wp = ctl._plan['waypoints'][n_generic + 1]
        ctl._stage_counts[wp['name']] = 4
        exact = self._reading_at(ctx, wp)
        ctl.observe_after(exact)
        # base advances past last waypoint -> ready if actions <= max
        self.assertEqual(ctl._phase, 'ready')
        self.assertTrue(ctl.summary()['ready_confirmed'])
        self.assertEqual(ctl.summary()['actual_aux_actions'], 300)
        self.assertEqual(ctl.summary()['bridge_actions'], 100)

    def test_stage_timeout_reason(self):
        ctx, ctl = self._make_at()
        wp = ctl._plan['waypoints'][0]
        ctl._stage_counts[wp['name']] = 59
        ctl._consecutive_confirmations = 0
        safe_unaligned = self._reading_unaligned(ctx, wp, offset_xy=0.03)
        ctl.observe_after(safe_unaligned)
        self.assertEqual(ctl._phase, 'failed')
        self.assertEqual(ctl._reason, 'stage_timeout')

    def test_runtime_held_rejected(self):
        ctx, ctl = self._make_at()
        wp = ctl._plan['waypoints'][0]
        r = self._reading_at(ctx, wp)
        r['snapshot']['held_objects'] = ['obj']
        ctl.next_action(r)
        self.assertEqual(ctl._phase, 'failed')

    def test_runtime_incomplete_rejected(self):
        ctx, ctl = self._make_at()
        wp = ctl._plan['waypoints'][0]
        r = self._reading_at(ctx, wp)
        r['snapshot']['grasp_observation_complete'] = False
        ctl.next_action(r)
        self.assertEqual(ctl._phase, 'failed')

    def test_runtime_contact_rejected(self):
        ctx, ctl = self._make_at()
        wp = ctl._plan['waypoints'][0]
        r = self._reading_at(ctx, wp)
        r['robot_contacts'] = ['link']
        ctl.next_action(r)
        self.assertEqual(ctl._phase, 'failed')

    def test_runtime_protected_move_rejected(self):
        ctx, ctl = self._make_at()
        wp = ctl._plan['waypoints'][0]
        r = self._reading_at(ctx, wp)
        r['snapshot']['objects']['protected']['position'] = [0.31, 0.2, 0.95]
        ctl.next_action(r)
        self.assertEqual(ctl._phase, 'failed')
        self.assertIn('protected', ctl._reason)
        self.assertIn('moved', ctl._reason)


if __name__ == '__main__':
    unittest.main()
