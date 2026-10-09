import copy
import importlib.util
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
RUNNER_PATH = HERE / 'home20_20261009_run.py'


def load_runner():
    name = 'home20_20261009_run'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = load_runner()

BOWL = ['on', 'akita_black_bowl_1', 'plate_1']
WINE = ['on', 'wine_bottle_1', 'wine_rack_1_top_region']
GOALS = [BOWL, WINE]
WINE_KEY = 'on|wine_bottle_1|wine_rack_1_top_region'

GOLD = {
    'bowl_to_plate': ['on', 'akita_black_bowl_1', 'plate_1'],
    'wine_to_rack': ['on', 'wine_bottle_1', 'wine_rack_1_top_region'],
    'stove_on': ['turnon', 'flat_stove_1'],
    'soup_to_basket': ['in', 'alphabet_soup_1', 'basket_1_contain_region'],
    'sauce_to_basket': ['in', 'tomato_sauce_1', 'basket_1_contain_region'],
    'white_mug_left': ['on', 'porcelain_mug_1', 'plate_1'],
    'yellow_mug_right': ['on', 'white_yellow_mug_1', 'plate_2'],
}

SCENES = {
    'goal_table': ('libero_goal', 8),
    'basket_two': ('libero_10', 0),
    'mugs_two': ('libero_10', 4),
}

SEQUENCES = [
    ('goal_table', ['wine_to_rack', 'bowl_to_plate']),
    ('goal_table', ['bowl_to_plate', 'wine_to_rack']),
    ('goal_table', ['bowl_to_plate', 'stove_on']),
    ('goal_table', ['stove_on', 'bowl_to_plate']),
    ('basket_two', ['soup_to_basket', 'sauce_to_basket']),
    ('basket_two', ['sauce_to_basket', 'soup_to_basket']),
    ('mugs_two', ['white_mug_left', 'yellow_mug_right']),
    ('mugs_two', ['yellow_mug_right', 'white_mug_left']),
    ('goal_table', ['bowl_to_plate', 'wine_to_rack', 'stove_on']),
    ('goal_table', ['stove_on', 'wine_to_rack', 'bowl_to_plate']),
]


def key(goal):
    return '|'.join(GOLD[goal])


def valid_snapshot():
    return {
        'predicates': {
            'on|akita_black_bowl_1|plate_1': True,
            'on|wine_bottle_1|wine_rack_1_top_region': True,
        },
        'objects': {
            'akita_black_bowl_1': {
                'grasped': False,
                'linear_velocity': [0.0, 0.0, 0.0],
                'angular_velocity': [0.0, 0.0, 0.0],
            },
            'wine_bottle_1': {
                'grasped': False,
                'linear_velocity': [0.0, 0.0, 0.0],
                'angular_velocity': [0.0, 0.0, 0.0],
            },
            'plate_1': {
                'grasped': False,
                'linear_velocity': [0.0, 0.0, 0.0],
                'angular_velocity': [0.0, 0.0, 0.0],
            },
        },
        'held_objects': [],
        'grasp_observation_complete': True,
    }


def valid_row():
    return {
        'snapshot': valid_snapshot(),
        'current_ready': {'ready': True, 'unknown': []},
        'final_ready': {'ready': True, 'unknown': []},
        'completed_predicates': {WINE_KEY: True},
    }


def valid_case(cid, scene, goals, init_state_index, model_seed):
    suite, task_id = SCENES[scene]
    caps = list(goals)
    return {
        'case_id': cid,
        'scene_id': scene,
        'suite': suite,
        'task_id': task_id,
        'init_state_index': init_state_index,
        'model_seed': model_seed,
        'env_seed': 0,
        'capability_ids': caps,
        'gold_goals': [list(GOLD[c]) for c in caps],
        'vla_cap': 336,
        'home_cap': 360,
    }


def valid_cases():
    cases = []
    for i, (scene, goals) in enumerate(SEQUENCES):
        cases.append(valid_case('case_%02d' % (i * 2 + 1), scene, goals, 0, 2))
        cases.append(valid_case('case_%02d' % (i * 2 + 2), scene, goals, 1, 3))
    return cases


class TestHome20(unittest.TestCase):
    def test_01_released_stable_placement_ready_true(self):
        snap = valid_snapshot()
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIs(out['ready'], True)

    def test_02_predicate_false_gives_ready_false(self):
        snap = valid_snapshot()
        snap['predicates']['on|akita_black_bowl_1|plate_1'] = False
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIs(out['ready'], False)

    def test_03_predicate_none_gives_ready_none_and_unknown(self):
        snap = valid_snapshot()
        snap['predicates']['on|akita_black_bowl_1|plate_1'] = None
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIsNone(out['ready'])
        self.assertTrue(out['unknown'])

    def test_04_foreign_nongold_grasped_held_gives_false(self):
        snap = valid_snapshot()
        snap['objects']['foreign'] = {
            'grasped': True,
            'linear_velocity': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0],
        }
        snap['held_objects'] = ['foreign']
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIs(out['ready'], False)
        self.assertIn('foreign', out['held_objects'])

    def test_05_incomplete_holding_observation_gives_none(self):
        snap = valid_snapshot()
        snap['grasp_observation_complete'] = False
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIsNone(out['ready'])

    def test_06_target_linear_velocity_over_limit_gives_false(self):
        snap = valid_snapshot()
        snap['objects']['akita_black_bowl_1']['linear_velocity'] = [0.021, 0.0, 0.0]
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIs(out['ready'], False)

    def test_07_target_angular_velocity_over_limit_gives_false(self):
        snap = valid_snapshot()
        snap['objects']['akita_black_bowl_1']['angular_velocity'] = [0.201, 0.0, 0.0]
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIs(out['ready'], False)

    def test_08_nonfinite_velocity_gives_none(self):
        snap = valid_snapshot()
        snap['objects']['akita_black_bowl_1']['linear_velocity'] = [float('nan'), 0.0, 0.0]
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIsNone(out['ready'])

    def test_09_boolean_masquerading_as_velocity_gives_none(self):
        snap = valid_snapshot()
        snap['objects']['akita_black_bowl_1']['linear_velocity'] = [True, 0.0, 0.0]
        out = R.snapshot_ready(snap, [BOWL])
        self.assertIsNone(out['ready'])

    def test_10_turnon_true_requires_empty_hand_no_stove_velocity_requirement(self):
        snap = valid_snapshot()
        snap['predicates']['turnon|flat_stove_1'] = True
        out = R.snapshot_ready(snap, [['turnon', 'flat_stove_1']])
        self.assertIs(out['ready'], True)
        self.assertIn('akita_black_bowl_1', snap['objects'])
        self.assertIn('wine_bottle_1', snap['objects'])
        self.assertIn('plate_1', snap['objects'])
        snap['objects']['foreign'] = {
            'grasped': True,
            'linear_velocity': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0],
        }
        snap['held_objects'] = ['foreign']
        out = R.snapshot_ready(snap, [['turnon', 'flat_stove_1']])
        self.assertIs(out['ready'], False)

    def test_11_four_valid_rows_no_success(self):
        rows = [valid_row() for _ in range(4)]
        out = R.score_rows(rows, GOALS, [WINE])
        self.assertFalse(out['physical_gold_success'])

    def test_12_five_valid_rows_success(self):
        rows = [valid_row() for _ in range(5)]
        out = R.score_rows(rows, GOALS, [WINE])
        self.assertTrue(out['physical_gold_success'])

    def test_13_six_valid_rows_earlier_final_ready_false_no_success(self):
        rows = [valid_row() for _ in range(6)]
        rows[-2]['final_ready']['ready'] = False
        out = R.score_rows(rows, GOALS, [WINE])
        self.assertFalse(out['physical_gold_success'])

    def test_14_six_valid_rows_completed_wine_false_no_success(self):
        rows = [valid_row() for _ in range(6)]
        rows[0]['completed_predicates'][WINE_KEY] = False
        out = R.score_rows(rows, GOALS, [WINE])
        self.assertFalse(out['completed_goals_preserved'])
        self.assertFalse(out['physical_gold_success'])

    def test_15_five_valid_rows_completed_wine_none_no_success_unknown(self):
        rows = [valid_row() for _ in range(5)]
        rows[0]['completed_predicates'][WINE_KEY] = None
        out = R.score_rows(rows, GOALS, [WINE])
        self.assertFalse(out['physical_gold_success'])
        self.assertTrue(out['unknown'])

    def test_16_combine_specs_accepts_fixed20_and_rejects_seven_mutations(self):
        cases = valid_cases()
        accepted = R.combine_specs({'cases': cases})
        self.assertEqual(len(accepted), 20)
        mutations = []
        m = copy.deepcopy(cases)
        m.pop()
        mutations.append(m)
        m = copy.deepcopy(cases)
        m[0]['gold_goals'] = [['on', 'wrong', 'target']]
        mutations.append(m)
        m = copy.deepcopy(cases)
        m[0]['scene_id'] = 'patched_scene'
        mutations.append(m)
        m = copy.deepcopy(cases)
        m[0]['vla_cap'] = 337
        mutations.append(m)
        m = copy.deepcopy(cases)
        m[0]['suite'] = 'other_suite'
        m[0]['task_id'] = 99
        mutations.append(m)
        m = copy.deepcopy(cases)
        m[0]['model_seed'] = 99
        mutations.append(m)
        m = copy.deepcopy(cases)
        m[1]['case_id'] = m[0]['case_id']
        mutations.append(m)
        for i, mut in enumerate(mutations):
            with self.subTest(mutation=i):
                with self.assertRaises(ValueError):
                    R.combine_specs({'cases': mut})


if __name__ == '__main__':
    unittest.main()
