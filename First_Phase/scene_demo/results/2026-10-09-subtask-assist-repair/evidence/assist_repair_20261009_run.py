"""Native worker-call script: bounded repair, two enabled branches."""
import argparse
import gzip
import hashlib
import json
import sys
import time
import traceback
import uuid
from pathlib import Path

HERE = Path('/mnt/d/FYP/First_Phase/scene_demo')
PLAN = ((2, 'enabled'), (3, 'enabled'))
ORIGIN = '8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd'
HANDOFF = {2: '2a4521602e384127e079782f4ff0b9ba8537548aabce2ba348a9c7320129c857',
           3: '69b98ee1794f350d6f64f5c574bfe8e76a745c684cd65f641152a529e759d20b'}
COUNTS = {2: 346, 3: 348}
GOLD = [['on', 'akita_black_bowl_1', 'plate_1'], ['on', 'wine_bottle_1', 'wine_rack_1_top_region']]
STOVE_OFF = ['not', 'turnon', 'flat_stove_1']
CONDITION = 'subtask_assist_pilot'
SOURCE_NAMES = ('subtask_context.py', 'subtask_geometry.py', 'subtask_preparation.py',
                'subtask_assist_service.py', 'test_subtask_assist.py', 'test_subtask_prepare_repair.py')
PHYSICAL_ERRORS = {'budget_exhausted', 'subtask_prepare_failed', 'subtask_assist_blocked'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if hasattr(value, 'tolist'): return clean(value.tolist())
    if hasattr(value, 'item'): return value.item()
    return value


def require(result, label):
    if not isinstance(result, dict) or result.get('ok') is not True:
        raise RuntimeError('%s: %r' % (label, result))
    return result


def gates(args):
    for name in ('output_dir', 'profiles', 'preflight', 'software_proof'):
        path = Path(getattr(args, name))
        if not path.is_absolute(): raise ValueError('%s must be absolute' % name)
        if name != 'output_dir' and not path.is_file(): raise ValueError('%s missing' % name)
    output = Path(args.output_dir)
    if output.resolve() != (HERE / 'results/2026-10-09-subtask-assist-repair/pilot').resolve():
        raise ValueError('output must be the fixed new subtask-assist-repair/pilot directory')
    if output.exists(): raise ValueError('output exists')
    profiles = read(args.profiles)
    if not isinstance(profiles, dict): raise ValueError('profiles must be a dict')
    preflight = require(read(args.preflight), 'preflight')
    cases = preflight.get('cases')
    if not isinstance(cases, list) or len(cases) != 2: raise ValueError('preflight needs two cases')
    seen, usable = set(), False
    for case in cases:
        index = case.get('case_index')
        if type(index) is not int or index not in HANDOFF or index in seen:
            raise ValueError('preflight case identity invalid')
        seen.add(index)
        if case.get('handoff_sha') != HANDOFF[index]: raise ValueError('preflight handoff SHA mismatch')
        if type(case.get('bowl_route_usable')) is not bool: raise ValueError('unknown actual route result')
        usable |= case['bowl_route_usable']
    if seen != {2, 3} or not usable: raise ValueError('no actual module route usable')
    software = require(read(args.software_proof), 'software proof')
    for key in ('final_test_exit_code', 'regression_exit_code'):
        if type(software.get(key)) is not int or software[key] != 0: raise ValueError(key + ' not zero')
    mapping = software.get('source_sha256')
    if not isinstance(mapping, dict) or not set(SOURCE_NAMES).issubset(mapping):
        raise ValueError('all six source hashes required')
    for name, expected in mapping.items():
        source = (HERE / name).resolve()
        if HERE.resolve() not in source.parents or not source.is_file() or sha(source) != expected:
            raise ValueError('software source mismatch: ' + name)
    return output, profiles, preflight, software


def load_prefix(index):
    directory = HERE / 'results/2026-10-09-collision-chain/pilot_v2' / ('case_%02d' % index) / 'job_01'
    paths = (directory / 'action_sources.jsonl.gz', directory / 'result.json')
    result = read(paths[1])
    if result['state_before_sha'] != ORIGIN or result['state_after_sha'] != HANDOFF[index]:
        raise ValueError('recorded prefix result SHA mismatch')
    with gzip.open(paths[0], 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != COUNTS[index] or [r.get('step') for r in rows] != list(range(1, COUNTS[index] + 1)):
        raise ValueError('prefix count/step continuity mismatch')
    for row in rows:
        action = row.get('sent_action')
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError('prefix sent_action must have seven values')
        import math
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in action):
            raise ValueError('prefix sent_action must be finite')
    return rows, {str(path): sha(path) for path in paths}


def run_trial(svc, index, mode, output, rows, input_hashes, pe, pilot, paired, service):
    import numpy as np
    entry = {'case_index': index, 'mode': mode, 'model_seed': index, 'operational_errors': [],
             'reused_recorded_hermes_plan': True, 'new_hermes_calls': 0, 'original_request': pilot.CASES[index]}
    directory = output / ('case_%02d_%s' % (index, mode))
    directory.mkdir()
    entry['trial_dir'] = str(directory)
    try:
        if any(sha(path) != expected for path, expected in input_hashes.items()): raise ValueError('input changed')
        created = require(svc.create_session('goal_table', seed=0, init_state_index=0), 'create session')
        sid = created['session_id']; entry['session'] = created

        def prefix_work():
            session, env = svc._sessions[sid], svc._env
            actual_origin = service.state_sha(env)
            if actual_origin != ORIGIN: raise ValueError('actual origin SHA mismatch: ' + actual_origin)
            before = pe.capture_snapshot(env, GOLD)
            count = 0
            for row in rows:
                sent = np.asarray(row['sent_action'], dtype=np.float32)
                if sent.shape != (7,) or not np.isfinite(sent).all(): raise ValueError('invalid float32 action')
                result = env.step(sent)
                svc._last_obs = result[0]; svc._total_steps += 1; session.total_steps += 1; count += 1
            actual_handoff = service.state_sha(env)
            if actual_handoff != HANDOFF[index]: raise ValueError('actual handoff SHA mismatch: ' + actual_handoff)
            snapshot = pe.capture_snapshot(env, GOLD)
            if snapshot.get('held_objects') != [] or snapshot.get('grasp_observation_complete') is not True:
                raise ValueError('handoff hand not known empty')
            if snapshot.get('predicates', {}).get('on|wine_bottle_1|wine_rack_1_top_region') is not True:
                raise ValueError('handoff wine predicate not true')
            return {'ok': True, 'origin_sha': actual_origin, 'handoff_sha': actual_handoff,
                    'replay_actions': count, 'before_snapshot': before, 'handoff_snapshot': snapshot}

        entry['prefix'] = require(svc._sync_work('pilot_prefix', prefix_work, timeout=900), 'prefix worker')
        seeded = require(paired._seed_model_rng(svc, index), 'model seed')
        if seeded.get('seeded') is not True: raise RuntimeError('model RNG not seeded')
        entry['model_rng'] = seeded

        def bowl_work():
            session = svc._sessions[sid]
            svc._subtask_assist_mode = mode; svc._diag_condition = CONDITION
            request_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
            plan = service.PlanRecord(request_id, sid, {'decision': 'execute',
                'capability_ids': ['wine_to_rack', 'bowl_to_plate'], 'budget_per_subgoal': 500,
                'scene_version': session.scene_version, 'audit': False,
                'rationale': 'Mechanical reconstruction of recorded wine-to-bowl diagnostic plan; no new Hermes call.'})
            plan.completed_capability_ids = ['wine_to_rack']; plan.pending_capability_ids = ['bowl_to_plate']
            job = service.JobRecord(job_id, request_id, sid, 'bowl_to_plate', directory / 'job')
            plan.job_ids = [job_id]; plan.state = 'running'
            with svc._lock:
                svc._plans[request_id] = plan; svc._jobs[job_id] = job
                svc._job_order.append(job_id); session.active_request_id = request_id
            before_sha = service.state_sha(svc._env); value = None; scene_error = None
            try:
                value = svc._run_capability(session, plan, job, 'bowl_to_plate')
            except service.SceneError as exc:
                scene_error = {'reason': exc.reason, 'detail': exc.detail}
                if job.state not in ('completed', 'error', 'cancelled'):
                    job.state = 'error'; job.success = False; job.ended_reason = exc.reason
                job.error = str(exc)
            finally:
                after_sha = service.state_sha(svc._env)
                after = pe.capture_snapshot(svc._env, GOLD)
                stove_off = service.eval_goal_predicate(svc._env, STOVE_OFF)
                plan.state = 'completed' if job.success is True else 'blocked'
                if job.success is True:
                    plan.completed_capability_ids.append('bowl_to_plate'); plan.pending_capability_ids = []
                session.active_request_id = None
                persisted = {'ok': True, 'job': job.public(), 'plan': plan.public(), 'base_result': value,
                    'scene_error': scene_error, 'before_sha': before_sha, 'after_sha': after_sha,
                    'after_snapshot': after, 'stove_off': stove_off}
                pe._write_json_atomic(directory / 'worker_result.json', clean(persisted))
            return persisted

        actual = require(svc._sync_work('pilot_bowl_job', bowl_work, timeout=900), 'bowl worker')
        entry.update(actual)
        oracle_rows, paths = pilot.read_oracle_rows([actual['job']])
        entry['score'] = pilot.score_gold(oracle_rows, actual['after_snapshot'],
                                          entry['prefix']['before_snapshot'], actual['stove_off'])
        entry['telemetry_paths'] = paths
        run_dir = Path(actual['job']['run_dir'])
        entry['subtask_assist'] = read(run_dir / 'subtask_assist.json')
        entry['diagnostic'] = read(run_dir / 'diagnostic.json') if (run_dir / 'diagnostic.json').is_file() else None
        entry['artifacts'] = {'run_dir': str(run_dir), 'rollout': actual['job'].get('rollout_path'),
                              'action_sources': str(run_dir / 'action_sources.jsonl'), 'telemetry': paths}
        assist = entry['subtask_assist']
        entry['conclusion'] = 'prepare_failure' if mode == 'enabled' and assist.get('prepare_confirmed') is not True else 'policy_result'
        error = actual.get('scene_error'); reason = error['reason'] if error else actual['job'].get('ended_reason')
        if actual['job'].get('state') == 'cancelled' or (actual['job'].get('error') and reason not in PHYSICAL_ERRORS):
            entry['operational_errors'].append({'reason': reason, 'error': actual['job'].get('error')})
    except Exception:
        entry['operational_errors'].append(traceback.format_exc())
    finally:
        pe._write_json_atomic(directory / 'trial.json', clean(entry))
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output-dir', 'profiles', 'preflight', 'software-proof'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output, profiles, preflight, software = gates(args)
    loaded = {index: load_prefix(index) for index in (2, 3)}
    sys.path.insert(0, str(HERE))
    import grasp_assist_service as gas
    import guard_validation as gv
    import paired_config_experiments as paired
    import placement_experiments as pe
    import service
    import side_grasp_pilot as pilot
    from subtask_assist_service import UnifiedAssistService
    source_hashes = {name: sha(HERE / name) for name in SOURCE_NAMES}
    input_hashes = {path: digest for _, mapping in loaded.values() for path, digest in mapping.items()}
    for baseline in ('/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-09-subtask-assist/pilot/case_02_disabled/trial.json', '/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-09-subtask-assist/pilot/case_03_disabled/trial.json'):
        input_hashes[baseline] = sha(baseline)
    prereg = {'plan': PLAN, 'gold': GOLD, 'protected_stove_goal': STOVE_OFF, 'cheese_tolerance_m': .005,
        'strict_post_action_samples': 5, 'scene': 'goal_table', 'env_seed': 0, 'init_state_index': 0,
        'model_seeds': {2: 2, 3: 3}, 'budget_per_subgoal': 500, 'prepare_max_actions': 200,
        'profile': 'baseline_bf16', 'profile_config': {'use_amp': True, 'num_steps': 10, 'n_action_steps': 1},
        'checkpoint_revision': '6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
        'completion_mode': 'release_verified', 'grasp_guard_mode': 'shadow', 'source_sha256': source_hashes,
        'input_sha256': input_hashes, 'script_sha256': sha(Path(__file__)),
        'gate_inputs': {name: {'path': getattr(args, name), 'sha256': sha(getattr(args, name))}
                        for name in ('profiles', 'preflight', 'software_proof')},
        'previous_baselines': ['/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-09-subtask-assist/pilot/case_02_disabled/trial.json', '/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-09-subtask-assist/pilot/case_03_disabled/trial.json'],
        'previous_baseline_commit': '4fde4780f398a51019b903a42ef03420fb681324',
        'new_baseline_jobs': 0, 'fresh_hermes_calls': 0,
        'new_hermes_calls': 0, 'reused_old_hermes_plan': True, 'max_new_jobs': 2, 'retries': 0,
        'limitations': 'Two recorded handoff states; descriptive same-state comparisons, no training or general success rate.'}
    output.mkdir(parents=True)
    pe._write_json_atomic(output / 'preregistration.json', clean(prereg))
    report = {'ok': False, 'planned_trials': 2, 'executed_trials': 0, 'trials': [], 'not_run': list(PLAN),
              'operational_errors': [], 'new_hermes_calls': 0, 'preregistration_sha256': sha(output / 'preregistration.json')}
    gas.configure_process_environment()
    saved_oracle = dict(pe.FINAL_ORACLE_GOALS); svc = None
    try:
        pe.FINAL_ORACLE_GOALS[CONDITION] = [list(goal) for goal in GOLD]
        svc = UnifiedAssistService(model_path=service.DEFAULT_MODEL_PATH, run_root=str(output / 'sessions'),
             calibration_profiles=profiles, assist_mode='disabled', completion_mode='release_verified', grasp_guard_mode='shadow')
        svc.start(); deadline = time.monotonic() + 180
        while True:
            health = svc.health(); report['health'] = health
            if health.get('worker_error'): raise RuntimeError('worker error: ' + str(health['worker_error']))
            if health.get('ready') is True: break
            if time.monotonic() >= deadline: raise TimeoutError('readiness timeout180')
            time.sleep(1)
        profile = svc.configure_profile('baseline_bf16'); report['profile_readback'] = profile
        if not gv._profile_readback_ok(profile): raise RuntimeError('profile readback mismatch')
        for index, mode in PLAN:
            if any(sha(HERE / name) != expected for name, expected in source_hashes.items()): raise ValueError('source changed')
            print(json.dumps({'event': 'START', 'case_index': index, 'mode': mode}), flush=True)
            trial = run_trial(svc, index, mode, output, *loaded[index], pe, pilot, paired, service)
            report['trials'].append(trial); report['executed_trials'] = len(report['trials'])
            report['not_run'] = list(PLAN[len(report['trials']):])
            print(json.dumps(clean({'event': 'DONE', 'case_index': index, 'mode': mode,
                'steps': trial.get('job', {}).get('steps'), 'source_counts': trial.get('subtask_assist'),
                'score': trial.get('score'), 'operational_errors': trial['operational_errors']})), flush=True)
            report['operational_errors'].extend(trial['operational_errors'])
            pe._write_json_atomic(output / 'report.json', clean(report))
            if trial['operational_errors']: break
    except Exception:
        report['operational_errors'].append(traceback.format_exc())
    finally:
        if svc is not None:
            try:
                report['close_env'] = svc._sync_work('close_env', lambda: (svc._close_env() or {'ok': True}))
                require(report['close_env'], 'close owned environment')
            except Exception: report['operational_errors'].append(traceback.format_exc())
            finally: svc.stop()
        pe.FINAL_ORACLE_GOALS.clear(); pe.FINAL_ORACLE_GOALS.update(saved_oracle)
        report['source_unchanged'] = all(sha(HERE / name) == digest for name, digest in source_hashes.items())
        report['inputs_unchanged'] = all(sha(path) == digest for path, digest in input_hashes.items())
        report['ok'] = report['executed_trials'] == 2 and not report['operational_errors'] and report['source_unchanged'] and report['inputs_unchanged']
        pe._write_json_atomic(output / 'report.json', clean(report))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
