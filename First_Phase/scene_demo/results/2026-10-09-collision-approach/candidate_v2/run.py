import gzip, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np

CODE = Path('/mnt/c/Users/Admin1/.codex/worktrees/wine-grasp-assist/FYP/First_Phase/scene_demo')
sys.path.insert(0, str(CODE))
import placement_experiments as pe
import service, collision_grasp, local_grasp, preparation_diagnostics as pd

OUT = Path('/home/yhwang/fyp/scene_demo/grasp_assist/2026-10-09-collision-candidate-v2')
HISTORY = CODE / 'results/2026-10-09-side-grasp/pilot/case_01'
BASELINE = Path('/home/yhwang/fyp/scene_demo/grasp_assist/2026-10-09-collision-probe-v1/probe.json')
ORIGIN = '8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd'
BOWL = 'ed679730451dee4fee5a384e9eda27d1419f34ca9c98d78d03b7da49a1309b82'
TRIGGER = '5edc127658b88ee42ad028637723d453378f71b0257f0ea1102a50cabed76162'
OLD_FINAL = '3f25a29eaa8180b7c402c72f3e1930178717f9a283a8dd5b5a9fbd6c677219ee'
WINE = 'wine_bottle_1'
SOURCE_NAMES = ('collision_grasp.py', 'collision_geometry.py', 'side_grasp.py', 'local_grasp.py', 'preparation_diagnostics.py', 'service.py', 'placement_experiments.py')
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
        report['ok'] = report['candidate_confirmed']
        assert report['aux_actions'] == helper.total_actions
        return {'ok': True}
    finally:
        report['final_snapshot'] = pe.capture_snapshot(env, [['on', WINE, 'wine_rack_1_top_region'], ['on', 'akita_black_bowl_1', 'plate_1']])
        report['final_state_sha'] = service.state_sha(env)
        if helper is not None:
            report['phase'] = helper.phase; report['reason'] = helper.reason; report['summary'] = helper.summary()
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
    input_hashes = {str(path): sha(path) for path in INPUTS}
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
    write(OUT / 'preregistration.json', prereg)
    report = {'ok': False, 'equal_trigger_state': False, 'baseline_reproduced': True,
              'baseline_reference': {'path': str(BASELINE), 'sha256': prereg['baseline_probe_sha256'], 'trigger_state_sha': TRIGGER, 'final_state_sha': OLD_FINAL, 'aux_actions': 11},
              'candidate_confirmed': False, 'phase': None, 'reason': None, 'aux_actions': 0, 'actual_env_steps': 0, 'physical_prefix_steps': 0,
              'max_approach_wine_rotation_deg': 0., 'approach_contact_samples': 0, 'route_counts': None, 'final_snapshot': None,
              'local_grasp_only': True, 'placement_task_success': None, 'model_calls': 0, 'vla_calls': 0, 'hermes_calls': 0, 'operational_errors': []}
    svc = None
    try:
        svc = pe.DiagnosticService(run_root=str(OUT / 'sessions'), policy_loader=lambda _svc: None, completion_mode='release_verified', grasp_guard_mode='shadow')
        svc.start(); deadline = time.monotonic() + 60
        while True:
            health = svc.health()
            assert not health.get('worker_error'), health.get('worker_error')
            if health.get('ready'): break
            assert time.monotonic() < deadline, 'readiness timeout'
            time.sleep(.1)
        session = svc.create_session('goal_table', seed=0, init_state_index=0)
        assert session.get('ok') is True, session
        report['session_id'] = session['session_id']
        result = svc._sync_work('collision_candidate', lambda: run_worker(svc, session['session_id'], bowl, wine, reference[0], report), timeout=180)
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
        report['inputs_unchanged'] = all(sha(path) == input_hashes[str(path)] for path in INPUTS)
        if not report['sources_unchanged'] or not report['inputs_unchanged']:
            report['operational_errors'].append('source/input SHA changed'); report['ok'] = False
        report['physics_calls'] = report['actual_env_steps']; write(OUT / 'report.json', report)
    print(json.dumps({key: report.get(key) for key in ('ok', 'candidate_confirmed', 'phase', 'reason', 'aux_actions', 'physical_prefix_steps', 'operational_errors')}), flush=True)
    return 2 if report['operational_errors'] else 0

if __name__ == '__main__': sys.exit(main())
