import argparse
import gzip
import json
import os
import sys
import traceback
from types import SimpleNamespace

try:
    from . import safe_exit_pilot_io as io
except Exception:
    import safe_exit_pilot_io as io


def run_case(case, out):
    import numpy as np
    import imageio.v2 as imageio

    for p in (io.HERE, io.HERE.parent):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    import scene_demo.service as service
    import scene_demo.placement_experiments as pe
    import scene_demo.preparation_diagnostics as pd
    import scene_demo.joint_home as jh
    import scene_demo.safe_exit as sx

    out = io.path(out)
    os.makedirs(str(out), exist_ok=False)

    cid = case['case_id']
    if '02' in cid:
        gold = [['on', 'wine_bottle_1', 'wine_rack_1_top_region']]
    else:
        gold = [['turnon', 'flat_stove_1']]

    record = {
        'case_id': cid,
        'status': 'unknown',
        'reason': None,
        'origin_sha256': None,
        'replayed_terminal_sha256': None,
        'replay_actions': 0,
        'exit_actions': 0,
        'home_actions': 0,
        'exit': None,
        'home': None,
        'errors': [],
    }

    state = {'phase': None, 'counts': {'replay': 0, 'exit': 0, 'home': 0}}
    ex = None
    home_res = None
    env = None
    writer = None
    jsonl_f = None
    original_step = None
    prev_globals = {}
    _MISSING = object()

    try:
        old = io.load_module(
            'safe_exit_old_home',
            io.HERE / 'results' / '2026-10-09-joint-home' / 'evidence' / 'joint_home_20261009_run.py',
        )
        h20 = io.load_module(
            'safe_exit_old20',
            io.HERE / 'results' / '2026-10-09-joint-home-20' / 'evidence' / 'home20_20261009_run.py',
        )
        parent = h20.load_parent()

        casedir = io.path(case['case_directory'])
        trace = io.path(case['trace_sources']['action_sources']['path'])

        expected_trace = casedir / ('subtask_01_' + case['capability_id']) / 'job' / 'action_sources.jsonl.gz'
        if trace != expected_trace:
            raise RuntimeError('trace path mismatch')

        reference = case['reference']
        nh = io.read(casedir / 'native_home.json')
        if reference != case['native_home_reference']:
            raise RuntimeError('reference != native_home_reference')
        if reference != nh:
            raise RuntimeError('reference != native_home.json')

        rows = []
        with gzip.open(str(trace), 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        N = len(rows)
        expected_N = case['worker']['vla_actions']
        if N != expected_N:
            raise RuntimeError('trace row count mismatch')

        actions = []
        for i, row in enumerate(rows):
            if row.get('source') != 'vla':
                raise RuntimeError('row source not vla')
            if row.get('step') != i + 1:
                raise RuntimeError('row step not sequential')
            sa = row.get('sent_action')
            if not isinstance(sa, (list, tuple)) or len(sa) != 7:
                raise RuntimeError('sent_action shape')
            arr = []
            for x in sa:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    raise RuntimeError('sent_action element type')
                if not (-1.0 <= float(x) <= 1.0):
                    raise RuntimeError('sent_action range')
                arr.append(float(x))
            a = np.asarray(arr, dtype=np.float32)
            if not np.all(np.isfinite(a)):
                raise RuntimeError('sent_action nonfinite')
            actions.append(a)

        builder = object.__new__(service.SceneService)
        builder._env_factory = None
        env = service.SceneService._build_env(builder, 'libero_goal', 8, 0, case['spec']['init_state_index'])
        obs, info = env.reset(seed=0)

        sha0 = service.state_sha(env)
        if sha0 != case['initial_state_sha']:
            raise RuntimeError('initial state sha mismatch')
        if sha0 != case['worker']['before_sha256']:
            raise RuntimeError('worker before sha mismatch')
        if sha0 != reference['origin_sha256']:
            raise RuntimeError('reference origin sha mismatch')
        record['origin_sha256'] = sha0

        sha_pre = service.state_sha(env)
        origin_vec = parent.state_vector(env, service)
        np.save(str(out / 'native_origin.npy'), origin_vec)
        inner = service._inner_env(env)
        qpos = np.array(inner.sim.data.qpos, copy=True)
        qvel = np.array(inner.sim.data.qvel, copy=True)
        np.savez(str(out / 'native_origin_qpos_qvel.npz'), qpos=qpos, qvel=qvel)
        sha_post = service.state_sha(env)
        if sha_pre != sha_post:
            raise RuntimeError('state changed during read-only snapshot')

        _svc = {'svc': None}
        original_step = env.step

        def wrapped_step(action):
            ph = state['phase']
            if ph is None:
                raise RuntimeError('step outside phase')
            cap = {'replay': N, 'exit': 80, 'home': 360}[ph]
            if state['counts'][ph] >= cap:
                raise RuntimeError('cap exceeded for ' + ph)
            res = original_step(action)
            state['counts'][ph] += 1
            if ph == 'exit':
                svc = _svc['svc']
                if svc is not None:
                    svc._last_obs = res[0]
                    svc._total_steps += 1
                    sess = svc._sessions.get(cid)
                    if sess is not None:
                        sess.total_steps += 1
            return res

        env.step = wrapped_step

        state['phase'] = 'replay'
        replay_log = open(str(out / 'replay.jsonl'), 'x')
        try:
            for i, a in enumerate(actions):
                obs, reward, done, truncated, info = env.step(a)
                st_sha = service.state_sha(env)
                replay_log.write(json.dumps({
                    'step': i + 1,
                    'sent_action': [float(x) for x in a],
                    'state_sha256': st_sha,
                    'original_step_state_sha256': 'unknown',
                }) + '\n')
                replay_log.flush()
        finally:
            replay_log.close()
        state['phase'] = None

        replay_count = state['counts']['replay']
        if replay_count != N:
            raise RuntimeError('replay count mismatch')
        sha_after = service.state_sha(env)
        if sha_after != case['worker']['after_sha256']:
            raise RuntimeError('worker after sha mismatch')
        if sha_after != case['home']['final_state_sha256']:
            raise RuntimeError('home final sha mismatch')
        record['replayed_terminal_sha256'] = sha_after

        sha_pre = service.state_sha(env)
        term_vec = parent.state_vector(env, service)
        np.save(str(out / 'replayed_terminal.npy'), term_vec)
        inner = service._inner_env(env)
        qpos = np.array(inner.sim.data.qpos, copy=True)
        qvel = np.array(inner.sim.data.qvel, copy=True)
        np.savez(str(out / 'replayed_terminal_qpos_qvel.npz'), qpos=qpos, qvel=qvel)
        sha_post = service.state_sha(env)
        if sha_pre != sha_post:
            raise RuntimeError('state changed during terminal snapshot')

        svc = SimpleNamespace(
            _env=env,
            _last_obs=obs,
            _total_steps=replay_count,
            home_case=cid,
            _sessions={cid: SimpleNamespace(total_steps=replay_count)},
        )
        _svc['svc'] = svc

        stove_on = service.eval_goal_predicate(env, ['turnon', 'flat_stove_1'])
        if type(stove_on) is not bool:
            raise TypeError('stove_on not bool')
        baseline = {
            'joints': h20.capture_object_joints(env, service, np),
            'completed_goals': gold,
            'stove_on': stove_on,
            'stove_checked': True,
        }

        def _check_prot(d):
            if type(d) is not dict:
                raise TypeError('protection not dict')
            if type(d.get('ok')) is not bool:
                raise TypeError('protection.ok not bool')
            v = d.get('violations')
            if type(v) is not list:
                raise TypeError('violations not list')
            for x in v:
                if type(x) is not str:
                    raise TypeError('violation not str')
            rc = d.get('robot_contacts')
            if type(rc) is not list:
                raise TypeError('robot_contacts not list')
            if type(d.get('empty_hand')) is not bool:
                raise TypeError('empty_hand not bool')
            if type(d.get('observation_complete')) is not bool:
                raise TypeError('observation_complete not bool')
            if type(d.get('completed_goals')) is not list:
                raise TypeError('completed_goals not list')

        def fullguard():
            snap = pe.capture_snapshot(env, gold)
            d = h20.home_protection(env, snap, baseline, service, np, old)
            _check_prot(d)
            return d

        def retreatguard():
            full = fullguard()
            viol = full['violations']
            nv = [v for v in viol if v != 'robot_contact']
            ok = (len(nv) == 0) and (full['empty_hand'] is True) and (full['observation_complete'] is True)
            out = dict(full)
            out['violations'] = nv
            out['ok'] = ok
            out['base_protection'] = ok
            out['full_contact_record'] = full
            _check_prot(out)
            return out

        rg0 = retreatguard()
        if rg0['ok'] is not True:
            raise RuntimeError('initial retreatguard not ok')
        if rg0['empty_hand'] is not True:
            raise RuntimeError('initial empty_hand not true')
        if rg0['observation_complete'] is not True:
            raise RuntimeError('initial observation_complete not true')

        prev_globals = {k: getattr(old, k, _MISSING) for k in ('GOLD', 'protection', 'HOME_CAP')}
        old.GOLD = gold
        old.protection = h20._make_protection(old)
        old.HOME_CAP = 360

        ret_dir = out / 'retreat'
        os.makedirs(str(ret_dir), exist_ok=False)
        writer = imageio.get_writer(str(ret_dir / 'safe_exit.mp4'), fps=20)
        imgs0 = service.capture_vla_images(env)
        frame0 = imgs0['agentview']
        service._save_png(str(ret_dir / 'first.png'), frame0)
        writer.append_data(frame0)

        jsonl_f = open(str(out / 'safe_exit.jsonl'), 'x')

        def emit(row):
            snap = pe.capture_snapshot(env, gold)
            sha = service.state_sha(env)
            rec = dict(row) if isinstance(row, dict) else {'row': row}
            rec['snapshot'] = parent.clean(snap)
            rec['state_sha256'] = sha
            try:
                torques = getattr(env, 'torques', None)
                if torques is None:
                    inner = service._inner_env(env)
                    torques = inner.sim.data.ctrl
                rec['actuator_ctrl'] = parent.clean(np.asarray(torques))
            except Exception:
                pass
            jsonl_f.write(json.dumps(parent.clean(rec), allow_nan=False) + '\n')
            jsonl_f.flush()
            imgs = service.capture_vla_images(env)
            writer.append_data(imgs['agentview'])

        state['phase'] = 'exit'
        ex = sx.run_safe_exit(env, reference, retreatguard, emit, None)
        state['phase'] = None
        exit_count = state['counts']['exit']

        if type(ex) is not dict:
            raise TypeError('sx result not dict')
        if type(ex.get('ok')) is not bool:
            raise TypeError('sx.ok not bool')
        if type(ex.get('ready')) is not bool:
            raise TypeError('sx.ready not bool')
        if type(ex.get('actions')) is not int:
            raise TypeError('sx.actions not int')
        if ex['actions'] != exit_count:
            raise RuntimeError('sx actions count mismatch')
        if exit_count > 80:
            raise RuntimeError('sx exit cap exceeded')
        errs = ex.get('errors')
        if type(errs) is not list:
            raise TypeError('sx.errors not list')
        if type(ex.get('restored')) is not bool:
            raise TypeError('sx.restored not bool')
        if len(errs) != 0:
            raise RuntimeError('sx errors present')
        if ex['ok'] is not True:
            raise RuntimeError('sx not ok')

        record['exit'] = ex

        if ex['ready'] is not True:
            record['status'] = 'blocked'
            record['reason'] = ex.get('reason') or 'exit_not_ready'
        else:
            if ex['restored'] is not True:
                raise RuntimeError('sx restored false')
            ex_reason = ex.get('reason')
            cs = ex.get('confirmed_samples')
            already_clear = (
                ex_reason == 'already_clear'
                and ex.get('ready') is True
                and ex.get('actions') == 0
                and cs == 0
            )
            if not already_clear:
                if type(cs) is not int or cs < 5:
                    raise RuntimeError('sx confirmed_samples invalid')

            state['phase'] = 'home'
            home_res = old.run_home(
                svc, cid, out / 'home', reference, baseline,
                parent, pe, service, pd, jh, np, imageio,
            )
            state['phase'] = None
            home_count = state['counts']['home']

            if type(home_res) is not dict:
                raise TypeError('home result not dict')
            if type(home_res.get('actions')) is not int:
                raise TypeError('home.actions not int')
            if home_res['actions'] != home_count:
                raise RuntimeError('home actions count mismatch')
            if home_count > 360:
                raise RuntimeError('home cap exceeded')

            op_errs = home_res.get('operational_errors')
            unk = home_res.get('unknown')
            if type(op_errs) is not list or type(unk) is not list:
                raise TypeError('home error lists malformed')
            if len(op_errs) > 0 or len(unk) > 0:
                raise RuntimeError('home unknown or operational errors')

            record['home'] = home_res

            home_reason = home_res.get('reason')

            if home_reason == 'joint_route_blocked' and home_count == 0:
                restore = home_res.get('restore')
                if (
                    type(restore) is dict
                    and restore.get('ok') is True
                    and restore.get('original_identity') is True
                    and home_res.get('ready') is False
                ):
                    record['status'] = 'blocked'
                    record['reason'] = 'joint_route_blocked'
                else:
                    record['status'] = 'unknown'
                    record['reason'] = 'joint_route_blocked_restore_invalid'
                    record['errors'].append('joint_route_blocked restore invalid')
            else:
                ok = home_res.get('ok')
                ready = home_res.get('ready')
                restore = home_res.get('restore')
                pa_streak = home_res.get('postaction_ready_streak')
                last_met = home_res.get('last_metrics')
                last_prot = home_res.get('last_protection')

                if type(ok) is not bool or type(ready) is not bool:
                    raise TypeError('home ok/ready malformed')
                if type(restore) is not dict:
                    raise TypeError('home restore malformed')
                if type(pa_streak) is not int:
                    raise TypeError('home postaction streak malformed')

                final_fg = fullguard()
                success = (
                    ok is True
                    and ready is True
                    and restore.get('ok') is True
                    and pa_streak >= 5
                    and type(last_met) is dict
                    and last_met.get('ready') is True
                    and type(last_prot) is dict
                    and last_prot.get('ok') is True
                    and type(restore.get('readonly_metrics')) is dict
                    and restore['readonly_metrics'].get('ready') is True
                    and type(restore.get('readonly_protection')) is dict
                    and restore['readonly_protection'].get('ok') is True
                    and final_fg.get('ok') is True
                    and len(final_fg.get('robot_contacts', [])) == 0
                )

                if success:
                    record['status'] = 'success'
                else:
                    record['status'] = 'blocked'
                    record['reason'] = home_reason or 'physical_gate'

    except KeyboardInterrupt:
        record['status'] = 'unknown'
        record['reason'] = 'cancel'
        record['errors'].append(traceback.format_exc())
    except Exception:
        record['status'] = 'unknown'
        record['reason'] = record.get('reason') or 'operational'
        record['errors'].append(traceback.format_exc())

    try:
        if env is not None and original_step is not None:
            env.step = original_step
    except Exception:
        record['errors'].append('restore step: ' + traceback.format_exc())

    try:
        for k, v in prev_globals.items():
            if v is _MISSING:
                if hasattr(old, k):
                    delattr(old, k)
            else:
                setattr(old, k, v)
    except Exception:
        record['errors'].append('restore globals: ' + traceback.format_exc())

    try:
        if env is not None:
            sha_pre = service.state_sha(env)
            record['final_snapshot'] = pe.capture_snapshot(env, gold)
            sha_post = service.state_sha(env)
            if sha_pre != sha_post:
                raise RuntimeError('final snapshot changed state')
            record['final_state_sha256'] = sha_post
    except Exception:
        record['status'] = 'unknown'
        record['errors'].append('final snapshot: ' + traceback.format_exc())

    try:
        if writer is not None:
            writer.close()
    except Exception:
        record['errors'].append('writer close: ' + traceback.format_exc())

    try:
        if jsonl_f is not None:
            jsonl_f.close()
    except Exception:
        record['errors'].append('jsonl close: ' + traceback.format_exc())

    try:
        if env is not None:
            env.close()
    except Exception:
        record['errors'].append('env close: ' + traceback.format_exc())

    record['replay_actions'] = state['counts']['replay']
    record['exit_actions'] = state['counts']['exit']
    record['home_actions'] = state['counts']['home']
    record['exit'] = ex
    record['home'] = home_res

    if len(record['errors']) > 0:
        record['status'] = 'unknown'
        record['prior_reason'] = record.get('reason')
        record['reason'] = 'operational_cleanup'

    return record


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--software-proof', required=True)
    p.add_argument('--proposal', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)

    try:
        inputs, software, proposal = io.gates(args)
    except Exception:
        print(json.dumps({'error': traceback.format_exc()}))
        return 2

    outdir = io.path(io.OUTPUT)
    os.makedirs(str(outdir), exist_ok=False)

    prereg = {
        'budget': {
            'cases': 3,
            'replay_actions': 334,
            'exit_cap_per_case': 80,
            'exit_total_cap': 240,
            'home_cap_per_case': 360,
            'home_total_cap': 1080,
            'new_vla_calls': 0,
            'new_hermes_calls': 0,
            'retries': 0,
        },
        'source_sha256': io.sha(io.path(__file__)),
        'inputs_sha256': io.sha(io.path(args.inputs)),
        'proposal_sha256': io.sha(io.path(args.proposal)),
    }
    io.write_json(outdir / 'preregistration.json', prereg)

    facts = {
        'frozen_files': inputs.get('frozen_files'),
        'registry_evidence': inputs.get('registry_evidence'),
        'user8_before': inputs.get('user8_before'),
    }

    def facts_verify_all():
        for key in ('frozen_files', 'registry_evidence', 'user8_before'):
            coll = facts.get(key)
            if not coll:
                return False
            if not io.verify_facts(coll):
                return False
        return True

    src_map = software.get('source_sha256')
    if not isinstance(src_map, dict) or not src_map:
        src_map = None

    cases = inputs['cases']
    results = []
    stopped = False
    total_replay = 0
    total_exit = 0
    total_home = 0
    counts_unknown = False
    campaign_errors = []

    run_case_raise_count = 0

    for case in cases:
        cid = case['case_id']
        if stopped:
            results.append({'case_id': cid, 'status': 'not_run'})
            continue

        try:
            if not facts_verify_all():
                raise RuntimeError('facts verify returned false')
        except Exception:
            results.append({'case_id': cid, 'status': 'unknown', 'reason': 'facts_before'})
            stopped = True
            continue

        try:
            if src_map is None:
                raise RuntimeError('software source_sha256 missing')
            current = {}
            for pth in io.HERE.glob('*.py'):
                current[pth.name] = io.sha(pth)
            if current != src_map:
                raise RuntimeError('source drift')
        except Exception:
            results.append({'case_id': cid, 'status': 'unknown', 'reason': 'software_drift'})
            stopped = True
            continue

        try:
            rec = run_case(case, outdir / cid)
        except Exception:
            run_case_raise_count += 1
            rec = {
                'case_id': cid,
                'status': 'unknown',
                'reason': 'run_case_raise',
                'errors': [traceback.format_exc()],
                'replay_actions': None,
                'exit_actions': None,
                'home_actions': None,
            }
            counts_unknown = True

        results.append(rec)

        for k_total, k_rec in (('replay', 'replay_actions'), ('exit', 'exit_actions'), ('home', 'home_actions')):
            v = rec.get(k_rec)
            if v is None:
                counts_unknown = True
            elif k_total == 'replay':
                total_replay += v
            elif k_total == 'exit':
                total_exit += v
            else:
                total_home += v

        print(json.dumps({
            'case_id': cid,
            'status': rec.get('status'),
            'replay_actions': rec.get('replay_actions'),
            'exit_actions': rec.get('exit_actions'),
            'home_actions': rec.get('home_actions'),
        }, sort_keys=True))

        try:
            if not facts_verify_all():
                if rec.get('status') != 'unknown':
                    rec['status'] = 'unknown'
                    rec['reason'] = 'facts_after'
                stopped = True
        except Exception:
            rec['status'] = 'unknown'
            rec['reason'] = 'facts_after'
            stopped = True

        if rec.get('status') == 'unknown':
            if 'reason' not in rec or rec.get('reason') is None:
                rec['reason'] = 'unknown'
            stopped = True

    try:
        if not facts_verify_all():
            campaign_errors.append('final_facts_verification_failed')
            stopped = True
    except Exception:
        campaign_errors.append('final_facts_verification_failed')
        stopped = True

    for i, case in enumerate(cases):
        cid = case['case_id']
        if i < len(results):
            rec = results[i]
        else:
            rec = {'case_id': cid, 'status': 'not_run'}
        try:
            os.makedirs(str(outdir / cid), exist_ok=True)
            io.write_json(outdir / cid / 'result.json', rec)
        except Exception:
            rec['status'] = 'unknown'
            rec['reason'] = 'result_write_failed'
            errs = rec.get('errors')
            if not isinstance(errs, list):
                errs = []
                rec['errors'] = errs
            errs.append('result write: ' + traceback.format_exc())
            stopped = True
            campaign_errors.append('result_write_failed:' + cid)

    summary = {
        'cases': results,
        'actual_replay_actions': None if counts_unknown else total_replay,
        'total_exit_actions': None if counts_unknown else total_exit,
        'total_home_actions': None if counts_unknown else total_home,
        'new_vla_calls': 0,
        'new_hermes_calls': 0,
        'model_loads': 0,
        'retries': 0,
        'run_case_raise_count': run_case_raise_count,
        'execution_complete': not stopped,
        'physical_success_count': sum(1 for r in results if r.get('status') == 'success'),
        'physical_failures': sum(1 for r in results if r.get('status') == 'blocked'),
        'infrastructure_errors': sum(1 for r in results if r.get('status') == 'unknown'),
        'counts_unknown': counts_unknown,
        'campaign_errors': campaign_errors,
    }
    summary['ok'] = (not stopped) and summary['infrastructure_errors'] == 0 and len(campaign_errors) == 0

    io.write_json(outdir / 'summary.json', summary)
    print(json.dumps({'summary': summary}, sort_keys=True))

    if summary['ok']:
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
