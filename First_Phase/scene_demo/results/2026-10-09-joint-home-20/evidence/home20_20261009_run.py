"""Joint-home 20-case fresh-VLA diagnostic runner; software authorship only."""
from __future__ import annotations
import argparse, importlib.util, json, math, numbers, sys, time, traceback, uuid
from pathlib import Path

HERE = Path('/mnt/d/FYP/First_Phase/scene_demo')
OUTPUT = HERE / 'results/2026-10-09-joint-home-20/pilot'
PARENT_PATH = HERE / 'results/2026-10-09-entry-compatibility/evidence/entry_compat_20261009_run.py'
OLD_HOME_PATH = HERE / 'results/2026-10-09-joint-home/evidence/joint_home_20261009_run.py'
PLAN_PATH = HERE / 'plans/2026-10-09-joint-home-20.md'
INPUTS_PATH = HERE / 'home20_20261009_inputs.json'
VLA_CAP = 336
HOME_CAP = 360
CONDITION = 'home20_joint_home'
STOVE_OFF = ['not', 'turnon', 'flat_stove_1']
STOVE_GOAL = ['turnon', 'flat_stove_1']
LINEAR_TOL = 0.02
ANGULAR_TOL = 0.2
OLD_HOME_SHA = 'bbb7f167446eb230b160b82776786406166eb5dce34e6524b3f34ea0543e0370'
FIXED_SEQUENCES = [
    ('case_01', 0, 2, 0, ['wine_to_rack', 'bowl_to_plate']),
    ('case_02', 1, 3, 0, ['wine_to_rack', 'bowl_to_plate']),
    ('case_03', 0, 2, 0, ['bowl_to_plate', 'wine_to_rack']),
    ('case_04', 1, 3, 0, ['bowl_to_plate', 'wine_to_rack']),
    ('case_05', 0, 2, 0, ['bowl_to_plate', 'stove_on']),
    ('case_06', 1, 3, 0, ['bowl_to_plate', 'stove_on']),
    ('case_07', 0, 2, 0, ['stove_on', 'bowl_to_plate']),
    ('case_08', 1, 3, 0, ['stove_on', 'bowl_to_plate']),
    ('case_09', 0, 2, 0, ['soup_to_basket', 'sauce_to_basket']),
    ('case_10', 1, 3, 0, ['soup_to_basket', 'sauce_to_basket']),
    ('case_11', 0, 2, 0, ['sauce_to_basket', 'soup_to_basket']),
    ('case_12', 1, 3, 0, ['sauce_to_basket', 'soup_to_basket']),
    ('case_13', 0, 2, 0, ['white_mug_left', 'yellow_mug_right']),
    ('case_14', 1, 3, 0, ['white_mug_left', 'yellow_mug_right']),
    ('case_15', 0, 2, 0, ['yellow_mug_right', 'white_mug_left']),
    ('case_16', 1, 3, 0, ['yellow_mug_right', 'white_mug_left']),
    ('case_17', 0, 2, 0, ['bowl_to_plate', 'wine_to_rack', 'stove_on']),
    ('case_18', 1, 3, 0, ['bowl_to_plate', 'wine_to_rack', 'stove_on']),
    ('case_19', 0, 2, 0, ['stove_on', 'wine_to_rack', 'bowl_to_plate']),
    ('case_20', 1, 3, 0, ['stove_on', 'wine_to_rack', 'bowl_to_plate']),
]
GOLD_LITERALS = {
    'bowl_to_plate': ['on', 'akita_black_bowl_1', 'plate_1'],
    'wine_to_rack': ['on', 'wine_bottle_1', 'wine_rack_1_top_region'],
    'stove_on': ['turnon', 'flat_stove_1'],
    'soup_to_basket': ['in', 'alphabet_soup_1', 'basket_1_contain_region'],
    'sauce_to_basket': ['in', 'tomato_sauce_1', 'basket_1_contain_region'],
    'white_mug_left': ['on', 'porcelain_mug_1', 'plate_1'],
    'yellow_mug_right': ['on', 'white_yellow_mug_1', 'plate_2'],
}
SCENE_FOR_CASE = {
    'case_01': 'goal_table', 'case_02': 'goal_table', 'case_03': 'goal_table', 'case_04': 'goal_table',
    'case_05': 'goal_table', 'case_06': 'goal_table', 'case_07': 'goal_table', 'case_08': 'goal_table',
    'case_09': 'basket_two', 'case_10': 'basket_two', 'case_11': 'basket_two', 'case_12': 'basket_two',
    'case_13': 'mugs_two', 'case_14': 'mugs_two', 'case_15': 'mugs_two', 'case_16': 'mugs_two',
    'case_17': 'goal_table', 'case_18': 'goal_table', 'case_19': 'goal_table', 'case_20': 'goal_table',
}
SUITE_TASK = {
    'goal_table': ('libero_goal', 8),
    'basket_two': ('libero_10', 0),
    'mugs_two': ('libero_10', 4),
}


def load_parent():
    spec = importlib.util.spec_from_file_location('home20_readonly_parent', PARENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError('parent runner import unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_old_home():
    spec = importlib.util.spec_from_file_location('home20_readonly_old_home', OLD_HOME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError('old joint-home runner import unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(name, **fields):
    print(json.dumps({'event': name, **fields}, ensure_ascii=True), flush=True)


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError('unknown scalar: ' + label)
    return float(value)


def _vec(value, length, label, np):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError('unknown vector: ' + label)
    arr = np.asarray(value, dtype=object)
    if arr.shape != (length,):
        raise ValueError('wrong shape: ' + label)
    return np.asarray([_finite(x, label) for x in arr.tolist()], dtype=np.float64)


def _vec3(value, label, np):
    return _vec(value, 3, label, np)


def _norm3(value, label, np):
    arr = _vec3(value, label, np)
    if np.any(np.abs(arr) > 1e12):
        raise ValueError('unreasonable vector: ' + label)
    return float(np.linalg.norm(arr))


def _np_global():
    import numpy as _np
    return _np


def snapshot_ready(snapshot, goals):
    """Strict readiness screen for one post-action snapshot dict."""
    np = _np_global()
    unknown = []
    held = []
    held_flags = False
    speeds = {}
    if not isinstance(snapshot, dict):
        return {'ready': None, 'unknown': ['snapshot_not_mapping'], 'held_objects': [], 'goal_speeds': {}}
    preds = snapshot.get('predicates')
    if not isinstance(preds, dict):
        return {'ready': None, 'unknown': ['predicates_not_mapping'], 'held_objects': [], 'goal_speeds': {}}
    if snapshot.get('grasp_observation_complete') is not True:
        unknown.append('grasp_observation_incomplete')
    s_held = snapshot.get('held_objects')
    if not isinstance(s_held, list):
        unknown.append('held_objects_unknown')
        s_held = []
    else:
        held = list(s_held)
    if held:
        held_flags = True
    objects = snapshot.get('objects')
    if not isinstance(objects, dict) or not objects:
        unknown.append('objects_missing')
        objects = {}
    for obj, entry in objects.items():
        if not isinstance(entry, dict):
            unknown.append('object_entry_not_mapping:' + str(obj))
            continue
        g = entry.get('grasped')
        if type(g) is not bool:
            unknown.append('grasp_unknown:' + str(obj))
        elif g is True:
            held_flags = True
    goal_keys = []
    for goal in goals or []:
        if not isinstance(goal, (list, tuple)) or len(goal) < 2:
            unknown.append('malformed_goal')
            continue
        key = '|'.join(str(x) for x in goal)
        goal_keys.append((key, goal))
    pred_ok = True
    for key, goal in goal_keys:
        val = preds.get(key)
        if type(val) is not bool:
            unknown.append('predicate_unknown:' + key)
            pred_ok = False
            continue
        if val is not True:
            pred_ok = False
    for key, goal in goal_keys:
        verb = goal[0]
        obj = goal[1] if len(goal) >= 2 else None
        if verb == 'turnon':
            speeds[key] = {'linear_norm': None, 'angular_norm': None, 'required': False}
            continue
        entry = objects.get(obj) if isinstance(obj, str) else None
        if not isinstance(entry, dict):
            unknown.append('goal_object_missing:' + key)
            continue
        try:
            lin = _norm3(entry.get('linear_velocity'), 'linear_velocity:' + key, np)
            ang = _norm3(entry.get('angular_velocity'), 'angular_velocity:' + key, np)
        except Exception:
            unknown.append('velocity_unknown:' + key)
            continue
        speeds[key] = {'linear_norm': lin, 'angular_norm': ang, 'required': True}
        if lin > LINEAR_TOL:
            pred_ok = False
        if ang > ANGULAR_TOL:
            pred_ok = False
    if unknown:
        return {'ready': None, 'unknown': unknown, 'held_objects': held, 'goal_speeds': speeds}
    ready = bool(pred_ok and not held and not held_flags)
    return {'ready': ready, 'unknown': [], 'held_objects': held, 'goal_speeds': speeds}


def score_rows(rows, gold, completed):
    unknown = []
    if not isinstance(rows, list):
        unknown.append('rows_not_list')
        rows = []
    gold_keys = []
    for g in gold or []:
        if isinstance(g, (list, tuple)):
            gold_keys.append('|'.join(str(x) for x in g))
        else:
            gold_keys.append(str(g))
    comp_keys = []
    for c in completed or []:
        if isinstance(c, (list, tuple)):
            comp_keys.append('|'.join(str(x) for x in c))
        else:
            comp_keys.append(str(c))
    preserved = True
    if comp_keys:
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                preserved = False
                unknown.append('row_not_mapping:' + str(i))
                continue
            mapping = row.get('completed_predicates')
            if not isinstance(mapping, dict):
                preserved = False
                unknown.append('completed_predicates_not_mapping:' + str(i))
                continue
            for key in comp_keys:
                if key not in mapping:
                    unknown.append('completed_predicate_missing:' + str(key))
                    preserved = False
                    continue
                sval = mapping.get(key)
                if isinstance(sval, bool):
                    if sval is not True:
                        preserved = False
                elif sval is None:
                    unknown.append('completed_predicate_none:' + str(key))
                    preserved = False
                else:
                    unknown.append('completed_predicate_nonbool:' + str(key))
                    preserved = False
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            unknown.append('row_not_mapping:' + str(i))
            continue
        snap = row.get('snapshot')
        if not isinstance(snap, dict):
            unknown.append('row_snapshot_not_mapping:' + str(i))
        cr = row.get('current_ready')
        if not isinstance(cr, dict):
            unknown.append('current_ready_not_mapping:' + str(i))
        else:
            if 'ready' not in cr:
                unknown.append('current_ready_ready_missing:' + str(i))
            else:
                rv = cr.get('ready')
                if rv is None:
                    unknown.append('current_ready_ready_none:' + str(i))
                elif not isinstance(rv, bool):
                    unknown.append('current_ready_ready_nonbool:' + str(i))
            ru = cr.get('unknown')
            if ru is not None:
                if isinstance(ru, (list, tuple)):
                    if len(ru) > 0:
                        unknown.append('current_ready_unknown_nonempty:' + str(i))
                else:
                    unknown.append('current_ready_unknown_invalid:' + str(i))
        fr = row.get('final_ready')
        if not isinstance(fr, dict):
            unknown.append('final_ready_not_mapping:' + str(i))
        else:
            if 'ready' not in fr:
                unknown.append('final_ready_ready_missing:' + str(i))
            else:
                rv = fr.get('ready')
                if rv is None:
                    unknown.append('final_ready_ready_none:' + str(i))
                elif not isinstance(rv, bool):
                    unknown.append('final_ready_ready_nonbool:' + str(i))
            fu = fr.get('unknown')
            if fu is not None:
                if isinstance(fu, (list, tuple)):
                    if len(fu) > 0:
                        unknown.append('final_ready_unknown_nonempty:' + str(i))
                else:
                    unknown.append('final_ready_unknown_invalid:' + str(i))
        cp = row.get('completed_predicates')
        if not isinstance(cp, dict):
            unknown.append('completed_predicates_not_mapping:' + str(i))
    last5_ready = len(rows) >= 5
    if last5_ready:
        for i in range(len(rows) - 5, len(rows)):
            row = rows[i]
            ok = True
            if not isinstance(row, dict):
                unknown.append('last5_row_not_mapping:' + str(i))
                last5_ready = False
                continue
            fr = row.get('final_ready')
            if not isinstance(fr, dict):
                unknown.append('last5_final_ready_not_mapping:' + str(i))
                last5_ready = False
                ok = False
            else:
                rv = fr.get('ready')
                if isinstance(rv, bool):
                    if rv is not True:
                        last5_ready = False
                        ok = False
                elif rv is None:
                    unknown.append('last5_final_ready_none:' + str(i))
                    last5_ready = False
                    ok = False
                else:
                    unknown.append('last5_final_ready_nonbool:' + str(i))
                    last5_ready = False
                    ok = False
                fu = fr.get('unknown')
                if fu is not None:
                    if isinstance(fu, (list, tuple)):
                        if len(fu) > 0:
                            unknown.append('last5_final_ready_unknown_nonempty:' + str(i))
                            last5_ready = False
                            ok = False
                    else:
                        unknown.append('last5_final_ready_unknown_invalid:' + str(i))
                        last5_ready = False
                        ok = False
            snap = row.get('snapshot')
            if not isinstance(snap, dict):
                unknown.append('last5_snapshot_not_mapping:' + str(i))
                last5_ready = False
                ok = False
            else:
                try:
                    ind = snapshot_ready(snap, gold)
                except Exception:
                    unknown.append('last5_snapshot_ready_exception:' + str(i))
                    last5_ready = False
                    ok = False
                    ind = None
                if isinstance(ind, dict):
                    rv = ind.get('ready')
                    if isinstance(rv, bool):
                        if rv is not True:
                            last5_ready = False
                            ok = False
                    elif rv is None:
                        unknown.append('last5_independent_ready_none:' + str(i))
                        last5_ready = False
                        ok = False
                    else:
                        unknown.append('last5_independent_ready_nonbool:' + str(i))
                        last5_ready = False
                        ok = False
                    iu = ind.get('unknown')
                    if iu is not None:
                        if isinstance(iu, (list, tuple)):
                            if len(iu) > 0:
                                unknown.append('last5_independent_ready_unknown_nonempty:' + str(i))
                                last5_ready = False
                                ok = False
                        else:
                            unknown.append('last5_independent_ready_unknown_invalid:' + str(i))
                            last5_ready = False
                            ok = False
                else:
                    unknown.append('last5_independent_ready_not_mapping:' + str(i))
                    last5_ready = False
                    ok = False
    if last5_ready and not preserved:
        last5_ready = False
    physical = bool(last5_ready and preserved and len(unknown) == 0)
    return {'physical_gold_success': physical, 'last5_ready': bool(last5_ready), 'completed_goals_preserved': bool(preserved), 'unknown': unknown}



def combine_specs(inputs):
    if not isinstance(inputs, dict):
        raise ValueError('inputs not mapping')
    cases = inputs.get('cases')
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError('exact 20 cases required')
    scenes = set()
    out = []
    seen = set()
    for i, (case_id, init_idx, model_seed, env_seed, caps) in enumerate(FIXED_SEQUENCES):
        if case_id in seen:
            raise ValueError('duplicate case id ' + case_id)
        seen.add(case_id)
        if len(caps) not in (2, 3):
            raise ValueError('each case 2 or 3 subtasks')
        case = cases[i]
        if not isinstance(case, dict):
            raise ValueError('case not mapping ' + case_id)
        if case.get('case_id') != case_id:
            raise ValueError('case_id order mismatch ' + case_id)
        scene = SCENE_FOR_CASE[case_id]
        scenes.add(scene)
        if case.get('scene_id') != scene:
            raise ValueError('scene mismatch ' + case_id)
        suite, task_id = SUITE_TASK[scene]
        if case.get('suite') != suite or case.get('task_id') != task_id:
            raise ValueError('suite/task mismatch ' + case_id)
        if case.get('init_state_index') != init_idx or case.get('model_seed') != model_seed or case.get('env_seed') != env_seed:
            raise ValueError('init/model/env seed mismatch ' + case_id)
        if case.get('capability_ids') != caps:
            raise ValueError('capability order mismatch ' + case_id)
        gold = [list(GOLD_LITERALS[c]) for c in caps]
        if case.get('gold_goals') != gold:
            raise ValueError('gold mismatch ' + case_id)
        if case.get('vla_cap') != VLA_CAP or case.get('home_cap') != HOME_CAP:
            raise ValueError('budget mismatch ' + case_id)
        out.append({'case_id': case_id, 'scene_id': scene, 'init_state_index': init_idx, 'model_seed': model_seed,
                    'env_seed': env_seed, 'capability_ids': list(caps), 'gold_goals': gold,
                    'vla_cap': VLA_CAP, 'home_cap': HOME_CAP})
    if len(scenes) != 3:
        raise ValueError('exactly three native scenes required')
    return out


JOINT_LENGTH = {0: 7, 1: 4, 2: 1, 3: 1}


def _joint_qpos(data, model, name, np):
    jid = int(model.joint_name2id(name))
    if jid < 0:
        raise ValueError('joint id unknown ' + name)
    jtype = int(model.jnt_type[jid])
    if jtype not in JOINT_LENGTH:
        raise ValueError('joint type unsupported ' + name)
    length = JOINT_LENGTH[jtype]
    raw = data.get_joint_qpos(name)
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError('joint qpos wrong length ' + name)
    for v in arr.tolist():
        _finite(v, 'joint_qpos:' + name)
    return {'type': jtype, 'qpos': [float(x) for x in arr.tolist()]}


def capture_object_joints(env, service, np):
    inner = service._inner_env(env)
    data = inner.sim.data
    model = inner.sim.model
    objects = getattr(inner, 'objects_dict', None)
    if not isinstance(objects, dict):
        raise ValueError('objects_dict unavailable')
    joints = {}
    for name, obj in objects.items():
        jl = getattr(obj, 'joints', None)
        if not isinstance(jl, (list, tuple)) or not jl or not isinstance(jl[0], str):
            continue
        entry = {}
        for jn in jl:
            if not isinstance(jn, str):
                raise ValueError('joint name unknown ' + str(name))
            entry[jn] = _joint_qpos(data, model, jn, np)
        if entry:
            joints[name] = entry
    if not joints:
        raise ValueError('no jointed objects')
    return joints


def _any_joint(joints, obj):
    e = joints.get(obj) if isinstance(joints, dict) else None
    if not isinstance(e, dict) or not e:
        return None
    for jn, v in e.items():
        return v
    return None


def _quat_angle(qa, qb, np):
    na = float(np.linalg.norm(qa))
    nb = float(np.linalg.norm(qb))
    if na <= 0 or nb <= 0:
        raise ValueError('zero quaternion')
    dot = float(np.clip(abs(np.dot(qa / na, qb / nb)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _round_quat(vals, np):
    arr = np.asarray(vals, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= 0:
        raise ValueError('bad quaternion')
    return arr / n


def home_protection(env, snapshot, baseline, service, np, old):
    violations = []
    measurements = {}
    if not isinstance(snapshot, dict) or not isinstance(baseline, dict):
        raise ValueError('protection inputs not mappings')
    if snapshot.get('grasp_observation_complete') is not True:
        raise ValueError('grasp observation incomplete')
    held = snapshot.get('held_objects')
    if not isinstance(held, list):
        raise ValueError('held_objects unknown')
    if held:
        violations.append('not_empty_hand')
    objects = snapshot.get('objects')
    base_joints = baseline.get('joints')
    if not isinstance(objects, dict) or not objects:
        raise ValueError('all-object screen unknown')
    if not isinstance(base_joints, dict) or not base_joints:
        raise ValueError('joint baseline unknown')
    for name, obj in objects.items():
        if not isinstance(obj, dict):
            raise ValueError('object entry unknown ' + str(name))
        g = obj.get('grasped')
        if type(g) is not bool:
            raise ValueError('grasp unknown ' + str(name))
        if g is True:
            violations.append('held:' + str(name))
    cur_joints = capture_object_joints(env, service, np)
    if set(snapshot.get('objects', {})) != set(base_joints) or set(base_joints) != set(cur_joints):
        raise ValueError('object set changed')
    if set(cur_joints) != set(base_joints):
        violations.append('jointed_object_set_changed')
    for name, entry in base_joints.items():
        cur_entry = cur_joints.get(name)
        if not isinstance(cur_entry, dict):
            violations.append('joints_missing:' + str(name))
            continue
        if set(cur_entry) != set(entry if isinstance(entry, dict) else {}):
            violations.append('joint_set_changed:' + str(name))
            continue
        for jn, base in entry.items():
            cur = cur_entry[jn]
            if base.get('type') != cur.get('type'):
                raise ValueError('joint type changed ' + str(jn))
            jtype = int(cur['type'])
            b = [float(x) for x in base['qpos']]
            c = [float(x) for x in cur['qpos']]
            label = str(name) + ':' + str(jn)
            if jtype == 0:
                bp = np.asarray(b[:3], dtype=np.float64)
                cp = np.asarray(c[:3], dtype=np.float64)
                trans = float(np.linalg.norm(cp - bp))
                bang = _quat_angle(_round_quat(b[3:7], np), _round_quat(c[3:7], np), np)
                measurements[label] = {'translation_m': trans, 'rotation_rad': bang}
                if trans > 0.005:
                    violations.append('free_translation:' + label)
                if bang > 0.05:
                    violations.append('free_rotation:' + label)
            elif jtype == 1:
                bang = _quat_angle(_round_quat(b, np), _round_quat(c, np), np)
                measurements[label] = {'rotation_rad': bang}
                if bang > 0.05:
                    violations.append('ball_rotation:' + label)
            elif jtype == 2:
                d = abs(c[0] - b[0])
                measurements[label] = {'slide_delta': d}
                if d > 0.005:
                    violations.append('slide_delta:' + label)
            elif jtype == 3:
                d = abs(c[0] - b[0])
                measurements[label] = {'hinge_delta': d}
                if d > 0.05:
                    violations.append('hinge_delta:' + label)
            else:
                raise ValueError('joint type unsupported ' + label)
    comp = baseline.get('completed_goals')
    if not isinstance(comp, (list, tuple)):
        raise ValueError('completed_goals unknown')
    for goal in comp:
        val = service.eval_goal_predicate(env, goal)
        if type(val) is not bool:
            raise ValueError('completed goal predicate unknown')
        if val is not True:
            violations.append('completed_goal_false:' + '|'.join(str(x) for x in goal))
    if baseline.get('stove_checked') is True:
        baseline_stove = baseline.get('stove_on')
        if type(baseline_stove) is not bool:
            raise ValueError('stove baseline unknown')
        stove = service.eval_goal_predicate(env, STOVE_GOAL)
        if type(stove) is not bool:
            raise ValueError('stove predicate unknown')
        if stove is not baseline_stove:
            violations.append('stove_state_changed')
    contacts = old.robot_contacts(env, service, np)
    if not isinstance(contacts, list):
        raise ValueError('robot contacts unknown')
    if contacts:
        violations.append('robot_contact')
    empty_hand = (len(held) == 0) and all(isinstance(o, dict) and o.get('grasped') is False for o in objects.values())
    return {'ok': not violations, 'violations': violations, 'objects': measurements,
            'empty_hand': empty_hand, 'observation_complete': True, 'robot_contacts': contacts,
            'joints': cur_joints, 'stove_on': baseline.get('stove_on'),
            'completed_goals': [list(g) for g in comp]}


def capture_state(svc, directory, reference, tag, parent, pe, service, pd, jh, np, batch=None):
    directory.mkdir(parents=True, exist_ok=True)
    env = svc._env
    before = service.state_sha(env)
    sim = service._inner_env(env).sim
    robotstate = jh.read_robot_state(env)
    metrics = jh.home_metrics(reference, robotstate)
    snapshot = pe.capture_snapshot(env, [list(g) for g in jh_gold(reference)])
    pose = pd.read_eef_pose(env)
    np.save(directory / 'sim_state.npy', parent.state_vector(env, service), allow_pickle=False)
    np.savez_compressed(directory / 'qpos_qvel.npz', qpos=np.asarray(sim.data.qpos, dtype=np.float64).copy(),
                        qvel=np.asarray(sim.data.qvel, dtype=np.float64).copy())
    parent.save_numeric(directory / 'raw_observation.npz', svc._last_obs, pe)
    if batch is not None:
        parent.save_numeric(directory / 'observation_batch.npz', batch, pe)
    images = service.capture_vla_images(env)
    if set(images) != {'agentview', 'wrist'}:
        raise ValueError('two VLA views unavailable')
    for view, image in images.items():
        service._save_png(directory / (view + '.png'), image)
    for filename, value in (('robot_state', robotstate), ('home_metrics', metrics), ('snapshot', snapshot), ('eef_pose', pose)):
        pe._write_json_atomic(directory / (filename + '.json'), parent.clean(value))
    after = service.state_sha(env)
    if before != after:
        raise ValueError('readonly capture changed state')
    record = {'tag': tag, 'directory': str(directory), 'state_before_sha256': before,
              'state_after_sha256': after, 'readonly_state_unchanged': True}
    pe._write_json_atomic(directory / 'capture.json', record)
    return record


def jh_gold(reference):
    return reference.get('gold_goals', []) if isinstance(reference, dict) else []


def _make_protection(old):
    def protection(env, snapshot, baseline, service, np):
        return home_protection(env, snapshot, baseline, service, np, old)
    return protection


def _reset_env_step(env):
    pass


def job_worker(svc, sid, spec, directory, reference, parent, pe, service, pd, jh, np, old):
    session = svc._sessions[sid]
    env = svc._env
    plan = spec['plan_record']
    job_id = uuid.uuid4().hex
    capability_id = spec['capability_id']
    job = service.JobRecord(job_id, plan.request_id, sid, capability_id, directory / 'job')
    job.instruction = spec.get('instruction')
    job.completion_mode = 'release_verified'
    plan.job_ids.append(job_id)
    with svc._lock:
        svc._jobs[job_id] = job
        svc._job_order.append(job_id)
        session.active_request_id = plan.request_id
    cap_index = spec['cap_index']
    vla_act = [0]
    rows = []
    observer_path = directory / 'observer.jsonl'
    observer_rows = [0]
    current_goals = [list(GOLD_LITERALS[capability_id])]
    previous_goals = [list(GOLD_LITERALS[c]) for c in plan.completed_capability_ids]
    previous_keys = ['|'.join(str(x) for x in g) for g in previous_goals]
    full_gold = [list(g) for g in spec['gold_goals']]
    flags = {'completed_lost': None, 'unknown': None, 'audit': None}
    original_step = env.step
    observer_failed = [False]

    def wrapped_step(action):
        result = original_step(action)
        vla_act[0] += 1
        snap = pe.capture_snapshot(env, full_gold)
        cur = snapshot_ready(snap, current_goals)
        fin = snapshot_ready(snap, full_gold)
        cmap = {}
        for key in previous_keys:
            val = snap.get('predicates', {}).get(key)
            cmap[key] = val if type(val) is bool else None
        row = {'step': observer_rows[0] + 1, 'snapshot': snap, 'current_ready': cur, 'final_ready': fin,
               'completed_predicates': cmap}
        rows.append(row)
        observer_rows[0] += 1
        try:
            with observer_path.open('a', encoding='utf8') as fh:
                fh.write(json.dumps(parent.clean(row), allow_nan=False) + '\n')
                fh.flush()
        except Exception:
            observer_failed[0] = True
        for key in previous_keys:
            if cmap.get(key) is False:
                flags['completed_lost'] = key
        if cur.get('ready') is None:
            flags['unknown'] = list(cur.get('unknown', [])) or ['current_ready_unknown']
        if fin.get('ready') is None:
            flags['unknown'] = list(flags['unknown'] or []) + (list(fin.get('unknown', [])) or ['final_ready_unknown'])
        if observer_failed[0] and not flags['unknown']:
            flags['unknown'] = ['observer_write_failed']
        if flags['unknown']:
            svc.home20_stop_reason = 'home20_probe_unknown'
            svc.home20_probe_unknown = list(flags['unknown'])
        elif flags['completed_lost']:
            svc.home20_stop_reason = 'completed_goal_lost'
        return result

    svc.home20_stop_reason = None
    svc.home20_probe_unknown = None
    svc.home20_audit_unavailable = None
    env.step = wrapped_step
    value = None
    error = None
    before_sha = service.state_sha(env)
    try:
        value = svc._run_capability(session, plan, job, capability_id)
    except service.SceneError as exc:
        error = {'reason': exc.reason, 'detail': exc.detail}
        if job.state not in ('completed', 'error', 'cancelled'):
            job.state = 'error'
            job.success = False
            job.ended_reason = exc.reason
        job.error = str(exc)
    finally:
        env.step = original_step
        after = pe.capture_snapshot(env, full_gold)
        stove = None
        if SCENE_FOR_CASE[spec['case_id']] == 'goal_table':
            stove = service.eval_goal_predicate(env, STOVE_GOAL)
            if type(stove) is not bool:
                stove = None
        record = {'ok': True, 'job': job.public(), 'plan': plan.public(), 'base_result': value,
                  'scene_error': error, 'before_sha256': before_sha, 'after_sha256': service.state_sha(env),
                  'after_snapshot': after, 'stove_on': stove, 'observer_rows': observer_rows[0],
                  'observer_path': str(observer_path), 'stop_reason': svc.home20_stop_reason,
                  'probe_unknown': svc.home20_probe_unknown, 'audit_unavailable': svc.home20_audit_unavailable,
                  'vla_actions': vla_act[0], 'observer_failed': observer_failed[0]}
        pe._write_json_atomic(directory / 'worker_result.json', parent.clean(record))
        pe._write_json_atomic(directory / 'vla_final_snapshot.json', parent.clean(after))
    return {**record, 'rows': rows, 'completed_goals': [list(GOLD_LITERALS[c]) for c in plan.completed_capability_ids],
            'completed_keys': list(previous_keys)}


def run_case(svc, spec, output, parent, pe, service, pd, jh, np, imageio, paired, old):
    case_id = spec['case_id']
    directory = output / case_id
    directory.mkdir(parents=True, exist_ok=True)
    trial = {'case_id': case_id, 'spec': spec, 'physical_success': False, 'operational_errors': [],
             'unknown': [], 'home_actions': 0, 'vla_actions': 0, 'total_physical_actions': 0,
             'new_hermes_calls': 0, 'stages': [], 'failure_stage': None, 'failure_category': None,
             'assist_mode': 'disabled'}
    svc.home_case = case_id
    saved_oracle = dict(pe.FINAL_ORACLE_GOALS)
    sid = None
    try:
        session = parent.require(svc.create_session(spec['scene_id'], seed=spec['env_seed'], init_state_index=spec['init_state_index']), 'session')
        sid = session['session_id']
        trial['session'] = session
        pe.FINAL_ORACLE_GOALS[CONDITION + '_' + case_id] = [list(g) for g in spec['gold_goals']]
        svc._diag_condition = CONDITION + '_' + case_id
        svc._subtask_assist_mode = 'disabled'

        def capture_origin():
            before = service.state_sha(svc._env)
            reference = jh.capture_home_reference(svc._env, origin_sha256=before)
            reference['gold_goals'] = [list(g) for g in spec['gold_goals']]
            pe._write_json_atomic(directory / 'native_home.json', parent.clean(reference))
            record = capture_state(svc, directory / 'native_origin', reference, 'native_origin', parent, pe, service, pd, jh, np)
            if service.state_sha(svc._env) != before:
                raise ValueError('home reference capture changed state')
            return {'ok': True, 'reference': reference, 'capture': record}
        origin = parent.require(svc._sync_work('home20_origin_capture', capture_origin, timeout=900), 'origin')
        reference = origin['reference']
        trial['initial_state_sha'] = origin['capture']['state_before_sha256']
        trial['reference'] = reference
        trial['initial_capture'] = origin['capture']
        seed = parent.require(paired._seed_model_rng(svc, spec['model_seed']), 'model RNG')
        if seed.get('seeded') is not True:
            raise ValueError('model RNG not seeded')
        trial['model_rng'] = seed
        request_id = uuid.uuid4().hex
        plan = service.PlanRecord(request_id, sid, {
            'decision': 'execute',
            'capability_ids': list(spec['capability_ids']),
            'budget_per_subgoal': VLA_CAP,
            'scene_version': session['scene_version'] if isinstance(session, dict) else getattr(session, 'scene_version', None),
            'audit': False,
            'rationale': 'Joint-home twenty-case diagnostic; frozen specs; no new Hermes.'})
        with svc._lock:
            svc._plans[request_id] = plan
        plan.state = 'running'
        plan.completed_capability_ids = []
        plan.pending_capability_ids = list(spec['capability_ids'])
        svc._sessions[sid].active_request_id = request_id
        success = True
        for idx, cap in enumerate(spec['capability_ids']):
            if not success:
                break
            subtask_dir = directory / ('subtask_%02d_%s' % (idx + 1, cap))
            subtask_dir.mkdir(parents=True, exist_ok=True)
            current_goals = [list(GOLD_LITERALS[cap])]
            previous_goals_for_stage = [list(GOLD_LITERALS[c]) for c in plan.completed_capability_ids]

            def refresh():
                before = service.state_sha(svc._env)
                svc._reset_policy_queues()
                svc._last_obs = svc._refresh_observation(svc._env)
                after = service.state_sha(svc._env)
                if before != after:
                    raise ValueError('queues/refresh changed physical state')
                return {'ok': True, 'state_before_sha256': before, 'state_after_sha256': after,
                        'readonly_state_unchanged': True}
            trial['queue_refresh'] = parent.require(svc._sync_work('home20_refresh_for_vla', refresh, timeout=900), 'refresh')
            saved_completed = list(plan.completed_capability_ids)
            actual = parent.require(svc._sync_work(
                'home20_vla_' + cap,
                lambda c=cap, i=idx: job_worker(svc, sid, {'case_id': case_id, 'capability_id': c,
                                                            'cap_index': i,
                                                            'plan_record': plan,
                                                            'gold_goals': spec['gold_goals']},
                                                  subtask_dir, reference, parent, pe, service, pd, jh, np, old),
                timeout=1800), 'VLA worker')
            job = actual['job']
            rows = actual['rows']
            stage_rows = [dict(r) for r in rows]
            for r in stage_rows:
                r['final_ready'] = r.get('current_ready')
            score = score_rows(stage_rows, current_goals, previous_goals_for_stage)
            assist = _assist_counts(job, actual, service, parent, trial)
            stage = {'capability_id': cap, 'job': job, 'score': score,
                     'observations_path': actual['observer_path'],
                     'assistcounts': assist['counts'],
                     'actual_assist': assist['counts'],
                     'assist_source_sha256': assist['source_sha256']}
            trial['vla_actions'] += actual.get('vla_actions') or 0
            trial['stages'].append(stage)
            infra = False
            if actual.get('probe_unknown'):
                trial['unknown'].extend(actual['probe_unknown'])
                infra = True
            if actual.get('audit_unavailable'):
                trial['unknown'].append('audit_unavailable')
                infra = True
            if assist['infrastructure']:
                trial['unknown'].append(assist['infrastructure'])
                infra = True
            if assist['missing']:
                trial['unknown'].append('assist_counts_missing')
                infra = True
            if actual.get('observer_failed'):
                trial['unknown'].append('observer_write_failed')
                infra = True
            if actual.get('scene_error') and actual['scene_error'].get('reason') not in ('completed_goal_lost', 'failed_grasp', 'budget_exhausted', 'already_satisfied', 'pre_satisfied_not_measured', 'session_step_limit'):
                infra = True
            if job.get('state') in ('error', 'cancelled') and job.get('success') is not True:
                if not actual.get('scene_error') or actual['scene_error'].get('reason') not in ('completed_goal_lost', 'failed_grasp', 'budget_exhausted', 'already_satisfied', 'pre_satisfied_not_measured', 'session_step_limit'):
                    infra = True
            if score and score.get('unknown'):
                infra = True
                trial['unknown'].extend(score['unknown'])
            if job.get('ended_reason') not in ('success', 'budget_exhausted', 'failed_grasp', 'completed_goal_lost', 'already_satisfied', 'pre_satisfied_not_measured', 'session_step_limit'):
                trial['operational_errors'].append({'reason': job.get('ended_reason'), 'error': job.get('error')})
                infra = True
            event('SUBTASK_DONE', case_id=case_id, capability_id=cap, actions=job.get('steps'), job_success=job.get('success'), independently_verified=score.get('physical_gold_success'), reason=job.get('ended_reason'))
            if infra:
                trial['failure_stage'] = {'index': idx, 'capability_id': cap}
                trial['failure_category'] = 'infrastructure'
                trial['unknown'].append('infrastructure_unknown')
                success = False
                _write_home_done(trial, None, None)
                event('CASE_DONE', case_id=case_id, physical_success=False)
                break
            if job.get('steps') == 0 and job.get('success') is True:
                success = False
                trial['failure_stage'] = {'index': idx, 'capability_id': cap}
                trial['failure_category'] = 'ordinary_pre_satisfied_not_measured'
                trial['failure_reason'] = 'pre_satisfied_not_measured'
                _write_home_done(trial, None, None)
                event('CASE_DONE', case_id=case_id, physical_success=False)
                break
            if actual.get('stop_reason') or job.get('success') is not True or not score.get('physical_gold_success'):
                reason = actual.get('stop_reason') or job.get('ended_reason') or 'score_false'
                if actual.get('stop_reason') or job.get('success') is False or score.get('physical_gold_success') is False:
                    success = False
                    trial['failure_stage'] = {'index': idx, 'capability_id': cap}
                    trial['failure_category'] = 'first_subtask' if idx == 0 else 'continuation_subtask'
                    trial['failure_reason'] = reason
                    _write_home_done(trial, None, None)
                    event('CASE_DONE', case_id=case_id, physical_success=False)
                    break
            if job.get('success') is not True or not score.get('physical_gold_success'):
                success = False
                trial['failure_stage'] = {'index': idx, 'capability_id': cap}
                trial['failure_category'] = 'first_subtask' if idx == 0 else 'continuation_subtask'
                trial['failure_reason'] = job.get('ended_reason') or 'score_not_success'
                _write_home_done(trial, None, None)
                event('CASE_DONE', case_id=case_id, physical_success=False)
                break
            with svc._lock:
                plan.completed_capability_ids.append(cap)
                plan.pending_capability_ids = [c for c in plan.pending_capability_ids if c != cap]
            if idx < len(spec['capability_ids']) - 1:
                baseline2 = svc._sync_work('home20_baseline', lambda: _capture_baseline(svc, list(plan.completed_capability_ids), spec, service, np), timeout=900)
                old_protection = old.protection
                old.GOLD = [list(g) for g in spec['gold_goals']]
                old.HOME_CAP = HOME_CAP
                old.protection = _make_protection(old)
                try:
                    home = svc._sync_work('home20_run_home_' + cap,
                                          lambda d=subtask_dir: old.run_home(svc, sid, d / 'home', reference,
                                                                              baseline2, parent, pe, service, pd, jh, np, imageio),
                                          timeout=900)
                finally:
                    old.protection = old_protection
                trial['stages'][-1]['home_after'] = home
                trial['home_actions'] += home.get('actions') or 0
                trial['attempted_home'] = trial.get('attempted_home', 0) + 1
                if home.get('ready') is True and home.get('restore', {}).get('ok') is True:
                    trial['ready_home'] = trial.get('ready_home', 0) + 1
                if home.get('unknown') or home.get('operational_errors'):
                    trial['unknown'].extend(home.get('unknown') or [])
                    trial['operational_errors'].extend(home.get('operational_errors') or [])
                    trial['failure_stage'] = {'index': idx, 'capability_id': cap}
                    trial['failure_category'] = 'infrastructure'
                    success = False
                    event('CASE_DONE', case_id=case_id, physical_success=False)
                    break
                if home.get('ready') is not True or home.get('restore', {}).get('ok') is not True:
                    success = False
                    trial['failure_stage'] = {'index': idx, 'capability_id': cap}
                    trial['failure_category'] = 'home'
                    trial['failure_reason'] = home.get('reason') or 'home_not_ready'
                    errs = home.get('operational_errors') or []
                    if errs:
                        trial['operational_errors'].extend(errs)
                    event('CASE_DONE', case_id=case_id, physical_success=False)
                    break

                def refresh2():
                    before = service.state_sha(svc._env)
                    svc._reset_policy_queues()
                    svc._last_obs = svc._refresh_observation(svc._env)
                    after = service.state_sha(svc._env)
                    if before != after:
                        raise ValueError('refresh changed physical state')
                    return {'ok': True, 'state_before_sha256': before, 'state_after_sha256': after,
                            'readonly_state_unchanged': True}
                parent.require(svc._sync_work('home20_post_home_refresh', refresh2, timeout=900), 'posthome refresh')
                pe._write_json_atomic(directory / 'trial.json', parent.clean(trial))
                event('HOME_DONE', case_id=case_id, cap=cap, ready=home.get('ready'))
            _write_home_done(trial, None, None)
        if success and len(plan.completed_capability_ids) == len(spec['capability_ids']):
            final_goals = [list(g) for g in spec['gold_goals']]
            final_prev = [list(GOLD_LITERALS[c]) for c in spec['capability_ids'][:-1]]
            last_rows = rows if trial['stages'] else []
            fscore = score_rows(last_rows, final_goals, final_prev) if last_rows else {'physical_gold_success': False, 'unknown': []}
            trial['final_score'] = fscore
            trial['physical_success'] = bool(fscore.get('physical_gold_success') is True and not fscore.get('unknown'))
            plan.plan_success = trial['physical_success']
        else:
            plan.plan_success = False
        plan.state = 'completed' if plan.plan_success else 'blocked'
    except service.SceneError as exc:
        trial['operational_errors'].append({'reason': getattr(exc, 'reason', 'SceneError'), 'detail': getattr(exc, 'detail', str(exc))})
        trial['failure_category'] = 'infrastructure'
        trial['unknown'].append('scene_error_infra')
    except Exception:
        trial['operational_errors'].append(traceback.format_exc())
        trial['failure_category'] = 'infrastructure'
        trial['unknown'].append('exception_infra')
    finally:
        pe.FINAL_ORACLE_GOALS.clear()
        pe.FINAL_ORACLE_GOALS.update(saved_oracle)
        trial['total_physical_actions'] = trial.get('home_actions', 0) + trial.get('vla_actions', 0)
        trial['new_hermes_calls'] = 0
        _write_home_done(trial, None, None)
        pe._write_json_atomic(directory / 'trial.json', parent.clean(trial))
        try:
            if sid is not None and isinstance(svc._sessions, dict):
                s = svc._sessions.get(sid)
                if s is not None:
                    s.active_request_id = None
        except Exception:
            pass
        try:
            if 'plan' in dir() and plan is not None:
                plan.state = plan.state if plan.state in ('completed', 'blocked', 'cancelled') else 'blocked'
        except Exception:
            pass
    return trial


def _write_home_done(trial, home, cap):
    return None


def _assist_counts(job, actual, service, parent, trial):
    run_dir = Path(job.get('run_dir'))
    result = {'counts': {}, 'infrastructure': None, 'missing': False, 'source_sha256': None}
    path = run_dir / 'subtask_assist.json'
    if not path.is_file():
        result['missing'] = True
        result['infrastructure'] = 'subtask_assist_missing'
        return result
    try:
        payload = json.loads(path.read_text(encoding='utf8'))
    except Exception:
        result['missing'] = True
        result['infrastructure'] = 'subtask_assist_unreadable'
        return result
    if not isinstance(payload, dict):
        result['missing'] = True
        result['infrastructure'] = 'subtask_assist_invalid'
        return result
    try:
        result['source_sha256'] = parent.sha(path)
    except Exception:
        result['source_sha256'] = None
    vla = payload.get('vla_source_actions')
    prep = payload.get('prepare_source_actions')
    lg = payload.get('local_grasp_source_actions')
    steps = job.get('steps')
    for value, label in ((vla, 'vla_source_actions'), (prep, 'prepare_source_actions'),
                         (lg, 'local_grasp_source_actions')):
        if type(value) is not int:
            result['missing'] = True
            result['infrastructure'] = 'assist_count_invalid:' + label
            return result
    if vla != steps or vla != actual.get('observer_rows'):
        result['infrastructure'] = 'assist_count_mismatch:vla'
        return result
    if prep != 0 or lg != 0:
        result['infrastructure'] = 'assist_count_nonzero'
        return result
    if steps > VLA_CAP:
        result['infrastructure'] = 'assist_count_exceeds_budget'
        return result
    result['counts'] = {'vla_source_actions': vla, 'prepare_source_actions': prep,
                        'local_grasp_source_actions': lg, 'job_steps': steps,
                        'observer_rows': actual.get('observer_rows')}
    return result


def _capture_baseline(svc, cap, spec, service, np):
    env = svc._env
    joints = capture_object_joints(env, service, np)
    stove = None
    stove_checked = False
    if SCENE_FOR_CASE.get(spec['case_id']) == 'goal_table':
        val = service.eval_goal_predicate(env, STOVE_GOAL)
        if type(val) is not bool:
            raise ValueError('stove baseline unknown')
        stove = val
        stove_checked = True
    return {'joints': joints, 'completed_goals': [list(GOLD_LITERALS[c]) for c in cap],
            'stove_on': stove, 'stove_checked': stove_checked}


def gates(args, parent, inputs):
    for key in ('output_dir', 'profiles', 'software_proof', 'inputs'):
        path = Path(getattr(args, key))
        if not path.is_absolute():
            raise ValueError(key + ' must be absolute')
        if key == 'output_dir':
            continue
        if not path.is_file():
            raise ValueError(key + ' missing')
    out = Path(args.output_dir)
    if out.resolve() != OUTPUT.resolve() or out.exists():
        raise ValueError('fixed output must be fresh')
    software = parent.require(parent.read(args.software_proof), 'software proof')
    if software.get('ok') is not True:
        raise ValueError('software proof ok is not True')
    for field in ('final_test_exit_code', 'regression_exit_code'):
        if type(software.get(field)) is not int or software[field] != 0:
            raise ValueError('software exit nonzero: ' + field)
    if inputs.get('ok') is not True:
        raise ValueError('inputs ok is not True')
    checks = inputs.get('checks')
    if not checks or not all(v is True for v in checks.values()):
        raise ValueError('inputs checks failed')
    inputs_before = inputs.get('inputs_before')
    if not isinstance(inputs_before, dict) or len(inputs_before) != 16:
        raise ValueError('inputs_before must have 16 entries')
    user8_before = inputs.get('user8_before')
    if not isinstance(user8_before, list) or len(user8_before) != 8:
        raise ValueError('user8_before must have 8 entries')
    cases = inputs.get('cases')
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError('inputs must provide 20 cases')
    src = parent.source_map()
    if src != inputs.get('source_sha256'):
        raise ValueError('source map mismatch vs inputs.source_sha256')
    if src != software.get('source_sha256'):
        raise ValueError('source map mismatch vs software.source_sha256')
    if inputs.get('plan_sha256') != parent.sha(PLAN_PATH):
        raise ValueError('plan hash mismatch')
    if parent.sha(OLD_HOME_PATH) != OLD_HOME_SHA:
        raise ValueError('old home hash mismatch')
    parent.verify_facts(inputs_before)
    parent.verify_facts(user8_before)
    return out, software


def _verify_facts(parent, old, args, fixed_files, raw_inputs, specs):
    if parent.sha(Path(__file__)) != fixed_files[str(Path(__file__))]:
        raise ValueError('source changed')
    if parent.sha(OLD_HOME_PATH) != OLD_HOME_SHA:
        raise ValueError('old home runner changed')
    for p, h in fixed_files.items():
        if parent.sha(Path(p)) != h:
            raise ValueError('fixed file changed: ' + p)
    if raw_inputs.get('source_sha256') != raw_inputs.get('source_sha256'):
        raise ValueError('inputs source map unstable')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--profiles', required=True)
    parser.add_argument('--software-proof', dest='software_proof', required=True)
    parser.add_argument('--inputs', required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(HERE))
    import numpy as np
    import imageio.v2 as imageio
    import service as service
    import placement_experiments as pe
    import preparation_diagnostics as pd
    import joint_home as jh
    import paired_config_experiments as paired
    import guard_validation as gv
    import grasp_assist_service as gas
    from subtask_assist_service import UnifiedAssistService

    parent = load_parent()
    inputs = parent.require(parent.read(args.inputs), 'inputs')
    specs = combine_specs(inputs)
    output, software = gates(args, parent, inputs)

    sources = parent.source_map()
    profiles_path = Path(args.profiles)
    inputs_path = Path(args.inputs)
    software_proof_path = Path(args.software_proof)
    this_path = Path(__file__)
    fixed_files = {
        str(profiles_path): parent.sha(profiles_path),
        str(inputs_path): parent.sha(inputs_path),
        str(software_proof_path): parent.sha(software_proof_path),
        str(this_path): parent.sha(this_path),
        str(PLAN_PATH): parent.sha(PLAN_PATH),
        str(PARENT_PATH): parent.sha(PARENT_PATH),
        str(OLD_HOME_PATH): parent.sha(OLD_HOME_PATH),
    }

    output.mkdir(parents=True, exist_ok=True)
    planned = len(specs)
    prereg = {
        'planned_cases': planned,
        'specs': parent.clean(specs),
        'source_sha256': dict(sources),
        'fixed_files_sha256': dict(fixed_files),
        'profile': {'name': 'baseline_bf16', 'use_amp': True, 'num_steps': 10, 'n_action_steps': 1},
        'vla_cap': VLA_CAP,
        'home_cap': HOME_CAP,
        'new_hermes_calls': 0,
        'new_helpers': 0,
        'new_training': 0,
        'retries': 0,
        'stop_rules': {
            'infrastructure': 'stop campaign and record explicit reason',
            'unknown': 'stop campaign and record explicit reason',
            'operational_error': 'stop campaign and record explicit reason',
            'cancellation': 'stop campaign and record explicit reason',
            'physical_false': 'continue next case',
        },
    }
    pe._write_json_atomic(output / 'preregistration.json', parent.clean(prereg))
    event('PREREG', path=str(output / 'preregistration.json'), planned_cases=planned)

    report = {
        'ok': False,
        'planned_cases': planned,
        'executed_cases': 0,
        'cases': [],
        'not_run': [{'case_id': s['case_id'], 'reason': 'not_started'} for s in specs],
        'operational_errors': [],
        'unknown': [],
        'new_hermes_calls': 0,
        'new_helpers': 0,
        'new_training': 0,
        'retries': 0,
        'preregistration_sha256': parent.sha(output / 'preregistration.json'),
    }

    old = load_old_home()
    saved_oracle = dict(pe.FINAL_ORACLE_GOALS)
    svc = None

    class Home20Service(UnifiedAssistService):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.home20_stop_reason = None
            self.home20_stop_detail = None

        def _select_action(self, batch):
            if self.home20_stop_reason is not None:
                raise service.SceneError(self.home20_stop_reason, str(self.home20_stop_detail))
            return super()._select_action(batch)

    try:
        gas.configure_process_environment()
        svc = Home20Service(
            model_path=service.DEFAULT_MODEL_PATH,
            run_root=str(output / 'sessions'),
            calibration_profiles=parent.read(args.profiles),
            assist_mode='disabled',
            completion_mode='release_verified',
            grasp_guard_mode='shadow',
        )
        svc.start()

        deadline = time.monotonic() + 300
        while True:
            health = svc.health()
            report['health'] = health
            if health.get('worker_error'):
                raise RuntimeError(str(health['worker_error']))
            if health.get('ready') is True:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('readiness deadline exceeded')
            time.sleep(1)

        profile = svc.configure_profile('baseline_bf16')
        report['profile_readback'] = profile
        if not gv._profile_readback_ok(profile):
            raise ValueError('baseline_bf16 profile readback mismatch')

        stop_reason = None
        for spec in specs:
            if stop_reason is not None:
                report['not_run'] = [
                    {'case_id': s['case_id'], 'reason': stop_reason}
                    for s in specs[report['executed_cases']:]
                ]
                break

            if parent.source_map() != sources or sources != inputs['source_sha256']:
                raise RuntimeError('source map changed before case ' + spec['case_id'])
            for p, h in fixed_files.items():
                if parent.sha(Path(p)) != h:
                    raise RuntimeError('fixed file changed before case: ' + p)
            parent.verify_facts(inputs['inputs_before'])
            parent.verify_facts(inputs['user8_before'])

            old.GOLD = [list(g) for g in spec['gold_goals']]
            old.HOME_CAP = HOME_CAP

            event('CASE_START', case_id=spec['case_id'])
            trial = run_case(svc, spec, output, parent, pe, service, pd, jh, np, imageio, paired, old)
            report['cases'].append(trial)
            report['executed_cases'] = len(report['cases'])
            report['not_run'] = [
                {'case_id': s['case_id'], 'reason': 'not_started'}
                for s in specs[len(report['cases']):]
            ]
            for key in ('unknown', 'operational_errors'):
                vals = trial.get(key)
                if vals:
                    report[key].extend(vals)

            stages = trial.get('stages')
            if isinstance(stages, list):
                report.setdefault('mechanical_stage_counts', {})[spec['case_id']] = len(stages)

            failed = False
            cat = trial.get('failure_category')
            if cat == 'infrastructure':
                stop_reason = 'infrastructure:' + spec['case_id']
                failed = True
            elif cat == 'unknown':
                stop_reason = 'unknown:' + spec['case_id']
                failed = True
            elif cat == 'operational_error':
                stop_reason = 'operational_error:' + spec['case_id']
                failed = True
            elif cat == 'cancellation':
                stop_reason = 'cancellation:' + spec['case_id']
                failed = True
            elif trial.get('unknown') or trial.get('operational_errors'):
                stop_reason = 'unknown_or_operational:' + spec['case_id']
                failed = True

            if not failed:
                event('CASE_DONE', case_id=spec['case_id'], physical_success=trial.get('physical_success'))

            pe._write_json_atomic(output / 'report.json', parent.clean(report))

            if failed:
                report['not_run'] = [
                    {'case_id': s['case_id'], 'reason': stop_reason}
                    for s in specs[report['executed_cases']:]
                ]
                pe._write_json_atomic(output / 'report.json', parent.clean(report))
                break
    except Exception:
        report['operational_errors'].append(traceback.format_exc())
    finally:
        try:
            if svc is not None:
                try:
                    close_result = svc._sync_work(
                        'close_owned_env',
                        lambda: (svc._close_env() or {'ok': True}),
                        timeout=900,
                    )
                    if not isinstance(close_result, dict) or close_result.get('ok') is not True:
                        report['operational_errors'].append('close_owned_env did not return ok True')
                except Exception:
                    report['operational_errors'].append(traceback.format_exc())
                try:
                    svc.stop()
                except Exception:
                    report['operational_errors'].append(traceback.format_exc())
        finally:
            pe.FINAL_ORACLE_GOALS.clear()
            pe.FINAL_ORACLE_GOALS.update(saved_oracle)

        try:
            unchanged_sources = parent.source_map()
            report['source_unchanged'] = bool(unchanged_sources == sources)
            report['fixed_files_unchanged'] = all(
                parent.sha(Path(p)) == h for p, h in fixed_files.items()
            )
            parent.verify_facts(inputs['inputs_before'])
            report['inputs_unchanged'] = True
            parent.verify_facts(inputs['user8_before'])
            report['user8_unchanged'] = True
        except Exception:
            report['operational_errors'].append(traceback.format_exc())
            report.setdefault('source_unchanged', False)
            report.setdefault('fixed_files_unchanged', False)
            report.setdefault('inputs_unchanged', False)
            report.setdefault('user8_unchanged', False)

        if not report.get('source_unchanged'):
            report['operational_errors'].append('source map changed')
        if not report.get('fixed_files_unchanged'):
            report['operational_errors'].append('fixed files changed')
        if not report.get('inputs_unchanged'):
            report['operational_errors'].append('inputs changed')
        if not report.get('user8_unchanged'):
            report['operational_errors'].append('users changed')

        executed = report.get('executed_cases', 0)
        cases = report.get('cases', [])
        protocol_ok = type(executed) is int and executed == len(cases) and executed <= 20
        if [c.get('case_id') for c in cases] != [s['case_id'] for s in specs[:len(cases)]]:
            protocol_ok = False
        for c in cases:
            vla_count = 0
            home_count = 0
            for st in c.get('stages', []):
                n = st.get('job', {}).get('steps')
                ac = st.get('assistcounts', {})
                if type(n) is not int or not 0 <= n <= VLA_CAP:
                    protocol_ok = False
                    continue
                vla_count += n
                if ac.get('vla_source_actions') != n or ac.get('prepare_source_actions') != 0 or ac.get('local_grasp_source_actions') != 0:
                    protocol_ok = False
                h = st.get('home_after')
                if h is not None:
                    hn = h.get('actions')
                    if type(hn) is not int or not 0 <= hn <= HOME_CAP:
                        protocol_ok = False
                    else:
                        home_count += hn
            if c.get('vla_actions') != vla_count or c.get('home_actions') != home_count or c.get('total_physical_actions') != vla_count + home_count:
                protocol_ok = False
            if c.get('new_hermes_calls') != 0 or c.get('assist_mode') != 'disabled':
                protocol_ok = False
        report['protocol_gate_observed'] = bool(protocol_ok)

        report['physical_successes'] = sum(c.get('physical_success') is True for c in cases)
        report['first_subtask_failures'] = sum(c.get('failure_category') == 'first_subtask' for c in cases)
        report['continuation_subtask_failures'] = sum(c.get('failure_category') == 'continuation_subtask' for c in cases)
        report['home_failures'] = sum(c.get('failure_category') == 'home' for c in cases)
        report['home_attempts'] = sum('home_after' in st for c in cases for st in c.get('stages', []))
        report['home_ready'] = sum(st.get('home_after', {}).get('ready') is True and st.get('home_after', {}).get('restore', {}).get('ok') is True for c in cases for st in c.get('stages', []) if 'home_after' in st)
        report['after_home_jobs'] = sum(max(0, len(c.get('stages', [])) - 1) for c in cases)
        report['after_home_job_successes'] = sum(st.get('score', {}).get('physical_gold_success') is True and st.get('job', {}).get('success') is True for c in cases for st in c.get('stages', [])[1:])

        report['ok'] = bool(
            executed == 20
            and not report['not_run']
            and not report['unknown']
            and not report['operational_errors']
            and report.get('source_unchanged') is True
            and report.get('fixed_files_unchanged') is True
            and report.get('inputs_unchanged') is True
            and report.get('user8_unchanged') is True
            and report.get('protocol_gate_observed') is True
        )

        pe._write_json_atomic(output / 'report.json', parent.clean(report))

    return 0 if report['ok'] else 2



def inputs_source(raw):
    return raw.get('source_sha256') if isinstance(raw, dict) else None


def _lazy_pe():
    import placement_experiments as pe
    return pe


if __name__ == '__main__':
    raise SystemExit(main())
