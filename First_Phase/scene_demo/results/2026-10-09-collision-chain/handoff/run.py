import gzip, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np

CODE = Path('/mnt/d/FYP/First_Phase/scene_demo')
sys.path.insert(0, str(CODE))
import placement_experiments as pe
import service, collision_grasp, local_grasp, preparation_diagnostics as pd
import catalog, guard_validation as gv, paired_config_experiments as paired, side_grasp_pilot as pilot

OUT = Path('/mnt/d/FYP/First_Phase/tmp/collision_handoff_v1')
HISTORY = CODE / 'results/2026-10-09-side-grasp/pilot/case_01'
BASELINE = Path('/home/yhwang/fyp/scene_demo/grasp_assist/2026-10-09-collision-probe-v1/probe.json')
ORIGIN = '8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd'
BOWL = 'ed679730451dee4fee5a384e9eda27d1419f34ca9c98d78d03b7da49a1309b82'
TRIGGER = '5edc127658b88ee42ad028637723d453378f71b0257f0ea1102a50cabed76162'
OLD_FINAL = '3f25a29eaa8180b7c402c72f3e1930178717f9a283a8dd5b5a9fbd6c677219ee'
WINE = 'wine_bottle_1'
SOURCE_NAMES = ('collision_grasp.py', 'collision_geometry.py', 'side_grasp.py', 'local_grasp.py', 'preparation_diagnostics.py', 'service.py', 'placement_experiments.py', 'guard_validation.py', 'paired_config_experiments.py', 'side_grasp_pilot.py', 'catalog.py')
GOLD = [['on', 'akita_black_bowl_1', 'plate_1'], ['on', 'wine_bottle_1', 'wine_rack_1_top_region']]
HANDOFF = '8c331a68332675507f3cdb676d04ece432bcfc75fa473584910b209ece77d014'
INPUTS = (HISTORY / 'job_01/action_sources.jsonl.gz', HISTORY / 'job_02/action_sources.jsonl.gz', HISTORY / 'job_02/wine_telemetry.jsonl.gz')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=default), encoding='utf-8')

def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]

def action_array(value):
    action = np.asarray(value, dtype=np.float32)
    assert action.shape == (7,) and np.all(np.isfinite(action)), 'invalid 7D action'
    return action

def run_worker(svc, sid, bowl, wine, reference, report):
    env = svc._env
    record = svc._sessions[sid]
    frames = []
    helper = None
    reading = None
    origin_snapshot = pe.capture_snapshot(env, GOLD)
    report['origin_snapshot'] = origin_snapshot
    gold_rows = []
    def step(action):
        result = env.step(action_array(action))
        report['actual_env_steps'] += 1
        if isinstance(result, tuple) and result: svc._last_obs = result[0]
        svc._total_steps = record.total_steps = report['actual_env_steps']
        return result
    def frame():
        result = svc._safe_render(env)
        assert result is not None, 'render returned None'
        frames.append(np.array(result, copy=True))
        return result
    try:
        report['origin_state_sha'] = service.state_sha(env)
        assert report['origin_state_sha'] == ORIGIN, 'origin SHA mismatch'
        for row in bowl: step(row['sent_action'])
        report['after_bowl_state_sha'] = service.state_sha(env)
        assert report['after_bowl_state_sha'] == BOWL, '104-step SHA mismatch'
        assert all(row['source'] == 'vla' for row in wine[:120])
        for row in wine[:120]: step(row['sent_action'])
        report['physical_prefix_steps'] = report['actual_env_steps']
        report['trigger_state_sha'] = service.state_sha(env)
        report['equal_trigger_state'] = report['trigger_state_sha'] == TRIGGER
        assert report['equal_trigger_state'], '224-step SHA mismatch'
        reading = collision_grasp.read_geometry(env)
        report['trigger_reading'] = reading
        report['trigger_descriptor'] = reading['collision_geometry']
        measured = reading['snapshot']['objects'][WINE]
        expected = reference['before_snapshot']['objects'][WINE]
        report['trigger_wine_max_errors'] = {
            key: float(np.max(np.abs(np.asarray(measured[key], dtype=np.float64) - np.asarray(expected[key], dtype=np.float64))))
            for key in ('position', 'quaternion')}
        assert all(np.isfinite(v) and v <= 1e-9 for v in report['trigger_wine_max_errors'].values()), 'trigger wine mismatch'
        initial_position = np.asarray(measured['position'], dtype=np.float64)
        initial_rotation = np.asarray(reading['wine_rotation'], dtype=np.float64)
        helper = collision_grasp.LocalGraspController()
        svc._save_frame(OUT / 'first.png', frame())
        assert 'proposed_vla_action' in wine[120], 'missing original proposed VLA action'
        action = helper.next_action(reading, action_array(wine[120]['proposed_vla_action']))
        with (OUT / 'steps.jsonl').open('w', encoding='utf-8') as log:
            while action is not None and helper.phase not in (local_grasp.CONFIRMED, local_grasp.FAILED) and report['aux_actions'] < 200:
                before = helper.summary()
                phase_before = helper.phase
                step(action)
                report['aux_actions'] += 1
                reading = collision_grasp.read_geometry(env)
                helper.observe_after(reading)
                snapshot = reading['snapshot']
                position = np.asarray(snapshot['objects'][WINE]['position'], dtype=np.float64)
                rotation_deg = float(np.degrees(pd.orientation_error_rad(np.asarray(reading['wine_rotation']), initial_rotation)))
                contacts = reading['collision_geometry']['wine_robot_contacts']
                row = {'step': report['aux_actions'], 'actual_env_step': report['actual_env_steps'], 'action': np.asarray(action).tolist(),
                       'phase_before': phase_before, 'phase_after': helper.phase,
                       'route_before': before.get('route_stage'), 'route_after': helper.summary().get('route_stage'),
                       'helper_summary': helper.summary(), 'snapshot': snapshot, 'state_sha': service.state_sha(env),
                       'wine_relative_rotation_deg': rotation_deg, 'wine_translation_m': float(np.linalg.norm(position - initial_position)),
                       'lift_m': float(position[2] - initial_position[2]), 'wine_robot_contacts': contacts}
                log.write(json.dumps(row, default=default) + '\n'); log.flush()
                frame()
                if phase_before == local_grasp.ABOVE:
                    report['max_approach_wine_rotation_deg'] = max(report['max_approach_wine_rotation_deg'], rotation_deg)
                    report['approach_contact_samples'] += int(bool(contacts))
                assert report['aux_actions'] == helper.total_actions, 'helper action accounting mismatch'
                if helper.phase in (local_grasp.CONFIRMED, local_grasp.FAILED) or report['aux_actions'] == 200: break
                action = helper.next_action(reading)
            if report['aux_actions'] == 200 and helper.phase not in (local_grasp.CONFIRMED, local_grasp.FAILED):
                helper.next_action(reading)  # Budget terminal check only; no physical step.
        report['phase'] = helper.phase; report['reason'] = helper.reason
        report['summary'] = helper.summary(); report['route_counts'] = report['summary'].get('route_counts')
        report['final_snapshot'] = reading['snapshot']
        report['final_state_sha'] = service.state_sha(env)
        report['lift_m'] = float(np.asarray(reading['snapshot']['objects'][WINE]['position'])[2] - initial_position[2])
        report['lift_five_sample_gate'] = helper._stage_streak[local_grasp.LIFT] >= local_grasp.LIFT_STREAK
        report['final_bowl_predicate'] = service.eval_goal_predicate(env, ['on', 'akita_black_bowl_1', 'plate_1'])
        report['candidate_confirmed'] = (helper.phase == local_grasp.CONFIRMED and reading['snapshot']['objects'][WINE]['grasped'] is True
                                         and report['lift_m'] >= .02 and report['final_bowl_predicate'] is True and report['lift_five_sample_gate'])
        assert report['aux_actions'] == helper.total_actions
        report['handoff_state_sha'] = service.state_sha(env)
        if not report['candidate_confirmed']:
            report['reason'] = helper.reason or 'local_confirmation_failed'
            report['placement_success'] = False
            return {'ok': True}
        assert report['handoff_state_sha'] == HANDOFF, 'confirmed handoff SHA mismatch'
        report['wine_ever_held'] = True
        svc._reset_policy_queues()  # One handoff queue reset; no simulator reset.
        instruction = catalog.CAPABILITIES['wine_to_rack']['instruction']
        remaining = 500 - 120 - helper.total_actions
        report['post_handoff_budget'] = remaining
        assert remaining >= 0
        with (OUT / 'continuation.jsonl').open('w', encoding='utf-8') as log:
            for index in range(remaining):
                before = pe.capture_snapshot(env, GOLD)
                before_sha = service.state_sha(env)
                batch = svc._observation_batch(svc._last_obs, instruction)
                # The actual normal policy input batch; no diagnostic/oracle data is injected.
                input_record = {}
                first_numeric = {}
                for key, value in batch.items():
                    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                        input_record[key] = list(value); continue
                    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
                    array = np.asarray(value)
                    if array.dtype.kind not in 'biuf':
                        input_record[key] = str(value); continue
                    input_record[key] = {'sha256': hashlib.sha256(array.tobytes()).hexdigest(), 'shape': list(array.shape), 'dtype': str(array.dtype)}
                    if index == 0: first_numeric[str(key)] = array
                if index == 0:
                    np.savez_compressed(OUT / 'first_policy_inputs.npz', **first_numeric)
                report['actual_policy_calls'] += 1
                proposed = np.asarray(svc._select_action(batch), dtype=np.float64).reshape(-1)
                assert proposed.shape == (7,) and np.all(np.isfinite(proposed)), 'invalid continuation action'
                sent = np.clip(proposed, env.action_space.low, env.action_space.high).astype(np.float32)
                step(sent)
                report['post_handoff_vla_actions'] += 1
                after = pe.capture_snapshot(env, GOLD)
                stove_off = service.eval_goal_predicate(env, ['not', 'turnon', 'flat_stove_1'])
                gold_rows.append(after)
                score = pilot.score_gold(gold_rows, after, origin_snapshot, stove_off)
                held = after.get('held_objects')
                if after.get('objects', {}).get(WINE, {}).get('grasped') is False and after.get('grasp_observation_complete') is True and report['first_release_step'] is None:
                    report['first_release_step'] = report['post_handoff_vla_actions']; report['wine_released'] = True
                wine_key = catalog.goal_key(GOLD[1])
                if after.get('predicates', {}).get(wine_key) is True:
                    report['wine_target_ever_true'] = True
                    if report['first_target_step'] is None: report['first_target_step'] = report['post_handoff_vla_actions']
                phase = after.get('phases', {}).get(wine_key)
                if phase == 'released_stable' and report['first_released_stable_step'] is None:
                    report['first_released_stable_step'] = report['post_handoff_vla_actions']
                row = {'step': index + 1, 'actual_env_step': report['actual_env_steps'], 'source': 'vla_continuation',
                       'sent_action': sent.tolist(), 'proposed_vla_action': proposed.tolist(), 'gripper_command': float(sent[-1]),
                       'before_snapshot': before, 'after_snapshot': after, 'before_state_sha': before_sha,
                       'after_state_sha': service.state_sha(env), 'policy_input': input_record, 'final_score': score}
                log.write(json.dumps(row, default=default) + '\n'); log.flush(); frame()
                report['final_score'] = score
                report['placement_success'] = score['combined_success'] is True
                if report['placement_success']:
                    report['reason'] = 'independent_combined_success'; break
        if not report['placement_success']: report['reason'] = 'budget_exhausted'
        report['wine_budget_used'] = 120 + helper.total_actions + report['post_handoff_vla_actions']
        assert report['wine_budget_used'] <= 500
        report['ok'] = not report['operational_errors']
        return {'ok': True}
    finally:
        report['final_snapshot'] = pe.capture_snapshot(env, [['on', WINE, 'wine_rack_1_top_region'], ['on', 'akita_black_bowl_1', 'plate_1']])
        report['final_state_sha'] = service.state_sha(env)
        if helper is not None:
            report['phase'] = helper.phase; report['grasp_phase'] = helper.phase; report['helper_reason'] = helper.reason; report['summary'] = helper.summary()
            report['route_counts'] = report['summary'].get('route_counts')
        if frames:
            try:
                svc._save_frame(OUT / 'last.png', frames[-1])
                service._save_video(OUT / 'rollout.mp4', frames, 20)
                assert (OUT / 'rollout.mp4').is_file() and (OUT / 'rollout.mp4').stat().st_size > 0
                report['video_frames'] = len(frames)
            except Exception as exc:
                report['operational_errors'].append('video: ' + type(exc).__name__ + ': ' + str(exc)); report['ok'] = False
        write(OUT / 'report.json', report)

def main():
    assert not OUT.exists(), 'candidate output exists; no retry'
    sources = {name: sha(CODE / name) for name in SOURCE_NAMES}
    audit_inputs = INPUTS + (BASELINE, HISTORY / 'job_02/grasp_assist.json')
    input_hashes = {str(path): sha(path) for path in audit_inputs}
    baseline_raw = BASELINE.read_bytes(); baseline = json.loads(baseline_raw)
    assert baseline.get('ok') is True and not baseline['errors'] and baseline['actual_replay_steps'] == 235
    assert baseline['trigger_state_sha'] == TRIGGER and baseline['final_state_sha'] == OLD_FINAL
    bowl, wine, telemetry = (load(path) for path in INPUTS)
    for rows, count in ((bowl, 104), (wine, 131)):
        assert len(rows) == count and [row['step'] for row in rows] == list(range(1, count + 1))
    reference = [row for row in telemetry if row.get('step') == 121]
    assert len(reference) == 1
    old_helper = json.loads((HISTORY / 'job_02/grasp_assist.json').read_bytes())
    OUT.mkdir(parents=True)
    prereg = {'candidate_attempts': 1, 'source_sha256': sources, 'input_sha256': input_hashes,
              'baseline_probe_sha256': hashlib.sha256(baseline_raw).hexdigest(), 'baseline_failure_helper': old_helper, 'old_failure_stop_reason': old_helper['reason'],
              'total_aux_limit': 200, 'route_limits': [60, 60, 60], 'above_limit': 150, 'same_prefix_steps': 224,
              'wine_rotation_drift_deg': 20, 'wine_translation_drift_m': .015, 'protection_tolerance_m': .005,
              'local_grasp_only': True, 'model_calls': 0, 'hermes_calls': 0, 'seed': 0, 'init_state_index': 0}
    prereg.update({'diagnostic': 'collision_handoff_v1', 'fixed_gold': GOLD, 'preregistered_budget': 500,
                   'physical_prefix_steps': 224, 'bowl_prefix_steps': 104, 'wine_prefix_steps': 120,
                   'model_seed': 1, 'profile': 'baseline_bf16', 'retries': 0, 'hermes_calls': 0,
                   'strict_streak': 5, 'cheese_tolerance_m': .005, 'stove_must_be_off': True,
                   'continuation_service': 'pe.DiagnosticService', 'grasp_guard_mode': 'shadow',
                   'distinct_five_case_pilot_service': 'GraspAssistService enforce; not this continuation diagnostic',
                   'control_sources': ['recorded_vla_prefix', 'local_grasp', 'vla_continuation'],
                   'continuation_rng': 'model seed1 newly initialized, not reproduction of old policy noise sequence',
                   'expected_state_sha': {'origin': ORIGIN, 'bowl104': BOWL, 'trigger224': TRIGGER, 'handoff': HANDOFF},
                   'model_path': service.DEFAULT_MODEL_PATH, 'model_revision': service.MODEL_REVISION_DEFAULT})
    prereg['local_grasp_only'] = False; prereg['model_calls'] = 'post-handoff actual policy calls recorded'
    write(OUT / 'preregistration.json', prereg)
    report = {'ok': False, 'equal_trigger_state': False, 'baseline_reproduced': True,
              'baseline_reference': {'path': str(BASELINE), 'sha256': prereg['baseline_probe_sha256'], 'trigger_state_sha': TRIGGER, 'final_state_sha': OLD_FINAL, 'aux_actions': 11},
              'candidate_confirmed': False, 'phase': None, 'reason': None, 'aux_actions': 0, 'actual_env_steps': 0, 'physical_prefix_steps': 0,
              'max_approach_wine_rotation_deg': 0., 'approach_contact_samples': 0, 'route_counts': None, 'final_snapshot': None,
              'local_grasp_only': True, 'placement_task_success': None, 'model_calls': 0, 'vla_calls': 0, 'hermes_calls': 0, 'operational_errors': []}
    report.update({'diagnostic': 'collision_handoff_v1', 'post_handoff_vla_actions': 0, 'actual_policy_calls': 0, 'placement_success': False,
                   'wine_released': False, 'first_release_step': None, 'wine_target_ever_true': False,
                   'first_target_step': None, 'first_released_stable_step': None, 'final_score': None, 'grasp_phase': None, 'helper_reason': None,
                   'preregistered_budget': 500, 'model_seed': 1, 'origin_snapshot': None})
    report['local_grasp_only'] = False
    svc = None
    try:
        svc = pe.DiagnosticService(model_path=service.DEFAULT_MODEL_PATH, run_root=str(OUT / 'sessions'), completion_mode='release_verified', grasp_guard_mode='shadow')
        svc.start(); deadline = time.monotonic() + 900
        while True:
            health = svc.health()
            assert not health.get('worker_error'), health.get('worker_error')
            if health.get('ready'): break
            assert time.monotonic() < deadline, 'readiness timeout'
            time.sleep(.1)
        profile = svc.configure_profile('baseline_bf16'); report['profile_readback'] = profile
        assert gv._profile_readback_ok(profile), 'baseline profile readback failed'
        session = svc.create_session('goal_table', seed=0, init_state_index=0)
        assert session.get('ok') is True, session
        report['session_id'] = session['session_id']
        seeded = paired._seed_model_rng(svc, 1); report['model_rng'] = seeded
        assert seeded.get('ok') is True and seeded.get('seeded') is True, 'model RNG seeding failed'
        result = svc._sync_work('collision_handoff_v1', lambda: run_worker(svc, session['session_id'], bowl, wine, reference[0], report), timeout=900)
        assert result.get('ok') is True, result
    except Exception as exc:
        report['operational_errors'].append(type(exc).__name__ + ': ' + str(exc)); report['ok'] = False
        traceback.print_exc()
    finally:
        if svc is not None:
            try:
                closed = svc._sync_work('close_env', lambda: (svc._close_env() or {'ok': True}), timeout=30)
                assert closed.get('ok') is True, closed
            except Exception as exc:
                report['operational_errors'].append('close_env: ' + str(exc)); report['ok'] = False
            finally: svc.stop()
        report['source_sha256_after'] = {name: sha(CODE / name) for name in SOURCE_NAMES}
        report['sources_unchanged'] = report['source_sha256_after'] == sources
        report['inputs_unchanged'] = all(sha(path) == input_hashes[str(path)] for path in audit_inputs)
        if not report['sources_unchanged'] or not report['inputs_unchanged']:
            report['operational_errors'].append('source/input SHA changed'); report['ok'] = False
        report['model_calls'] = report['vla_calls'] = report['actual_policy_calls']
        report['placement_task_success'] = report['placement_success']
        report['baseline_probe_unchanged'] = BASELINE.read_bytes() == baseline_raw
        if not report['baseline_probe_unchanged']: report['operational_errors'].append('baseline probe changed')
        report['ok'] = not report['operational_errors']
        report['physics_calls'] = report['actual_env_steps']; write(OUT / 'report.json', report)
    print(json.dumps({key: report.get(key) for key in ('ok', 'candidate_confirmed', 'phase', 'reason', 'aux_actions', 'physical_prefix_steps', 'post_handoff_vla_actions', 'placement_success', 'operational_errors')}), flush=True)
    return 2 if report['operational_errors'] else 0

if __name__ == '__main__': sys.exit(main())
