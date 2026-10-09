"""Native mechanical diagnostic runner; five ordered gates, never production."""
import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import time
import traceback
import uuid
from pathlib import Path

HERE = Path('/mnt/d/FYP/First_Phase/scene_demo')
OUTPUT = HERE / 'results/2026-10-09-entry-compatibility/pilot'
PLAN_PATH = HERE / 'plans/2026-10-09-entry-compatibility.md'
CASES = ('case_01_reference_replay', 'case_02_native_mid_resume',
         'case_03_helper_native_entry', 'case_04_helper_after_wine',
         'case_05_helper_native_holdout')
BOWL_GOLD = [['on', 'akita_black_bowl_1', 'plate_1']]
COMBINED_GOLD = [['on', 'akita_black_bowl_1', 'plate_1'],
                 ['on', 'wine_bottle_1', 'wine_rack_1_top_region']]
STOVE_OFF = ['not', 'turnon', 'flat_stove_1']
ORIGIN = '8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd'
WINE_HANDOFF = '2a4521602e384127e079782f4ff0b9ba8537548aabce2ba348a9c7320129c857'
CONDITION = 'entry_compat_probe'
PHYSICAL_ERRORS = {'budget_exhausted', 'subtask_prepare_failed', 'subtask_assist_blocked',
                   'entry_bridge_budget_exhausted', 'generic_prepare_budget_exhausted'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if hasattr(value, 'tolist'): return clean(value.tolist())
    if hasattr(value, 'item'): return value.item()
    return value


def require(result, label):
    if not isinstance(result, dict) or result.get('ok') is not True:
        raise RuntimeError('%s: %r' % (label, result))
    return result


def linux_path(path):
    if not isinstance(path, str): raise ValueError('path must be string')
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', path)
    if match: return Path('/mnt/' + match[1].lower() + '/' + match[2].replace('\\', '/'))
    result = Path(path)
    if not result.is_absolute(): raise ValueError('frozen path must be absolute')
    return result


def verify_facts(mapping):
    facts = list(mapping.values()) if isinstance(mapping, dict) else mapping
    if not isinstance(facts, list) or not facts: raise ValueError('empty frozen facts')
    for fact in facts:
        p = linux_path(fact['path'])
        if p.stat().st_size != fact['bytes'] or sha(p) != fact['sha256']:
            raise ValueError('frozen file changed: ' + str(p))
    return True


def source_map():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py')) if p.is_file()}


def gates(args):
    for name in ('output_dir', 'profiles', 'software_proof', 'inputs'):
        p = Path(getattr(args, name))
        if not p.is_absolute(): raise ValueError(name + ' must be absolute')
        if name != 'output_dir' and not p.is_file(): raise ValueError(name + ' missing')
    output = Path(args.output_dir)
    if output.resolve() != OUTPUT.resolve() or output.exists():
        raise ValueError('fixed output must be fresh')
    software = require(read(args.software_proof), 'software proof')
    for key in ('final_test_exit_code', 'regression_exit_code'):
        if type(software.get(key)) is not int or software[key] != 0:
            raise ValueError('software exit is not exact zero: ' + key)
    frozen = software.get('source_sha256')
    actual = source_map()
    if not isinstance(frozen, dict) or frozen != actual:
        raise ValueError('software proof must freeze every current top-level Python source')
    inputs = require(read(args.inputs), 'inputs')
    checks = inputs.get('checks')
    if not isinstance(checks, dict) or not checks or not all(v is True for v in checks.values()):
        raise ValueError('all input checks must be true')
    verify_facts(inputs['inputs_before']); verify_facts(inputs['user8_before'])
    if len(inputs['user8_before']) != 8: raise ValueError('eight user files required')
    profiles = read(args.profiles)
    if not isinstance(profiles, dict): raise ValueError('profiles must be mapping')
    return output, profiles, software, inputs, actual


def frozen_path(inputs, suffix):
    matches = [linux_path(p) for p in inputs['inputs_before'] if p.replace('\\', '/').endswith(suffix)]
    if len(matches) != 1: raise ValueError('exact historical file identity required: ' + suffix)
    return matches[0]


def json_rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def checked_actions(rows, key, count):
    if len(rows) != count or [r.get('step') for r in rows] != list(range(1, count + 1)):
        raise ValueError('historical action continuity/count mismatch')
    for row in rows:
        a = row.get(key)
        if not isinstance(a, list) or len(a) != 7 or any(isinstance(x, bool) or
            not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > 1 for x in a):
            raise ValueError('historical action must be finite nonbool7 within[-1,1]')
    return rows


def load_inputs(inputs):
    bowlp = frozen_path(inputs, 'release_verified_bowls/baseline_bf16/bowl_control/job_01/telemetry.jsonl.gz')
    diagp = frozen_path(inputs, 'release_verified_bowls/baseline_bf16/bowl_control/job_01/diagnostic.json.gz')
    winep = frozen_path(inputs, '2026-10-09-collision-chain/pilot_v2/case_02/job_01/action_sources.jsonl.gz')
    resultp = frozen_path(inputs, '2026-10-09-collision-chain/pilot_v2/case_02/job_01/result.json')
    bowl = checked_actions(json_rows(bowlp), 'action', 102)
    wine = checked_actions(json_rows(winep), 'sent_action', 346)
    with gzip.open(diagp, 'rt', encoding='utf8') as f: diagnostic = json.load(f)
    result = read(resultp)
    if result['state_before_sha'] != ORIGIN or result['state_after_sha'] != WINE_HANDOFF:
        raise ValueError('wine history origin/handoff mismatch')
    return bowl, wine, diagnostic, {str(p): sha(p) for p in (bowlp, diagp, winep, resultp)}


def gold_for(case):
    return [list(g) for g in (COMBINED_GOLD if case == CASES[3] else BOWL_GOLD)]


def score_literal(rows, final, before, stove_off, goals):
    import numpy as np
    unknown = []
    predicates = final.get('predicates', {})
    goal_flags = {}
    for goal in goals:
        k = '|'.join(goal); v = predicates.get(k)
        if type(v) is not bool: unknown.append('predicate:' + k)
        goal_flags[k] = v is True
    last5 = len(rows) >= 5 and all(r.get('strict_candidate') is True for r in rows[-5:])
    held = final.get('held_objects'); complete = final.get('grasp_observation_complete')
    if not isinstance(held, list) or complete is not True: unknown.append('complete_holding_screen')
    empty = isinstance(held, list) and held == [] and complete is True
    objects = final.get('objects')
    if not isinstance(objects, dict) or not objects: objects = {}; unknown.append('physical_objects')
    all_unheld = bool(objects)
    for name, obj in objects.items():
        value = obj.get('grasped') if isinstance(obj, dict) else None
        if type(value) is not bool: unknown.append('grasp:' + name)
        all_unheld = all_unheld and value is False
    displacement = {}; protection = True
    for name in ('wine_bottle_1', 'cream_cheese_1'):
        try:
            a = np.asarray(before['objects'][name]['position'], dtype=np.float64)
            b = np.asarray(objects[name]['position'], dtype=np.float64)
            if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('invalid position')
            d = float(np.linalg.norm(b-a)); displacement[name] = d; protection &= d <= .005
        except Exception:
            displacement[name] = None; protection = False; unknown.append('position:' + name)
    if type(stove_off) is not bool: unknown.append('stove_off')
    strict_final = final.get('strict_candidate') is True
    success = bool(goal_flags and all(goal_flags.values()) and last5 and strict_final and empty
                   and all_unheld and stove_off is True and protection and not unknown)
    return {'gold': goals, 'final_predicates': goal_flags, 'last5_strict_samples': last5,
            'final_strict_candidate': final.get('strict_candidate'), 'empty_hand': empty,
            'all_physical_objects_unheld': all_unheld, 'stove_off': stove_off,
            'protected_displacement_m': displacement, 'protection_ok': protection,
            'unknown': unknown, 'physical_gold_success': success}


def make_service_class(UnifiedAssistService, ep, capture):
    class EntryProbeService(UnifiedAssistService):
        def _build_generic_helper(self, ctx, reading):
            if not getattr(self, 'entry_reference', None):
                raise ep.EntryError('entry reference unavailable')
            return ep.EntryPreparationController(ctx, reading, self.entry_reference)

        def _select_action(self, batch):
            helper = getattr(self, '_subtask_helper', None)
            if (getattr(self, 'entry_case', None) in CASES[2:]
                and getattr(helper, 'phase', None) == 'ready'
                and getattr(self, 'entry_handoff_capture', None) is None):
                self.entry_handoff_capture = capture(self, batch)
            return super()._select_action(batch)
    return EntryProbeService


def flatten_numeric(value, parts, arrays, omitted):
    import numpy as np
    if isinstance(value, dict):
        for key, child in value.items(): flatten_numeric(child, parts+[str(key)], arrays, omitted)
        return
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    try: array = np.asarray(value)
    except Exception: array = None
    if array is not None and array.dtype.kind in 'biufc' and array.dtype.hasobject is False:
        arrays[json.dumps(parts, ensure_ascii=True)] = array
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value): flatten_numeric(child, parts+[i], arrays, omitted)
    else:
        omitted.append({'path':parts,'value':value if isinstance(value,(str,bool,int,float,type(None))) else None,
                        'type':type(value).__name__})


def state_vector(env, service):
    import numpy as np
    sim = service._inner_env(env).sim
    try: flat = sim.get_state().flatten()
    except Exception: flat = np.concatenate([np.asarray(sim.data.qpos,dtype=np.float64),np.asarray(sim.data.qvel,dtype=np.float64)])
    state = np.ascontiguousarray(np.asarray(flat,dtype=np.float64).reshape(-1))
    if hashlib.sha256(state.tobytes()).hexdigest() != service.state_sha(env):
        raise ValueError('native state vector SHA mismatch')
    return state


def save_numeric(path, value, pe):
    import numpy as np
    arrays, nonnumeric = {}, []
    flatten_numeric(value, [], arrays, nonnumeric)
    if not arrays: raise ValueError('numeric observation/input unavailable')
    np.savez_compressed(path, **arrays)
    pe._write_json_atomic(path.with_suffix('.keys.json'), {'npz_keys':list(arrays),'nonnumeric':nonnumeric})


def save_handoff(svc, batch, profiles, pe, service, pd, sg, catalog):
    import numpy as np
    directory = svc.entry_case_dir/'handoff'
    if directory.exists() or svc.entry_handoff_count != 0: raise ValueError('handoff must be captured once')
    directory.mkdir()
    env = svc._env; before_sha = service.state_sha(env)
    pose = pd.read_eef_pose(env)
    extra = [g for g in svc.entry_case_gold if g not in BOWL_GOLD]
    reading = sg.read_geometry(env, catalog.CAPABILITIES['bowl_to_plate'],
        svc._subtask_home_orientation, profiles, extra or None)
    snapshot = pe.capture_snapshot(env, svc.entry_case_gold)
    sim = service._inner_env(env).sim
    qpos = np.asarray(sim.data.qpos,dtype=np.float64).copy()
    qvel = np.asarray(sim.data.qvel,dtype=np.float64).copy()
    np.save(directory/'sim_state.npy',state_vector(env,service),allow_pickle=False)
    np.savez_compressed(directory/'qpos_qvel.npz',qpos=qpos,qvel=qvel)
    save_numeric(directory/'observation.npz',svc._last_obs,pe)
    save_numeric(directory/'policy_batch.npz',batch,pe)
    images = service.capture_vla_images(env)
    if set(images) != {'agentview','wrist'}: raise ValueError('both native VLA camera frames required')
    for view, image in images.items(): service._save_png(directory/(view+'.png'),image)
    cut_dir = Path(svc.entry_reference_cut['directory'])
    cut_pose = read(cut_dir/'pose.json'); cut_reading = read(cut_dir/'reading.json')
    with np.load(cut_dir/'qpos_qvel.npz',allow_pickle=False) as reference:
        if qpos.shape!=reference['qpos'].shape or qvel.shape!=reference['qvel'].shape:
            raise ValueError('reference robot state dimensions differ')
        qpos_l2=float(np.linalg.norm(qpos-reference['qpos']))
        qvel_l2=float(np.linalg.norm(qvel-reference['qvel']))
    p=np.asarray(pose['position'],dtype=np.float64); p0=np.asarray(cut_pose['position'],dtype=np.float64)
    R=np.asarray(pose['orientation_matrix'],dtype=np.float64); R0=np.asarray(cut_pose['orientation_matrix'],dtype=np.float64)
    gap=reading.get('gripper_gap_m'); gap0=cut_reading.get('gripper_gap_m')
    gap_difference=float(gap-gap0) if type(gap) in (int,float) and type(gap0) in (int,float) else None
    comparison={'reference_cut_step':30,'eef_position_error_m':float(np.linalg.norm(p-p0)),
        'eef_orientation_error_rad':pd.orientation_error_rad(R,R0),
        'gripper_gap_difference_m':gap_difference,'qpos_l2':qpos_l2,'qvel_l2':qvel_l2,
        'reference_state_sha256':svc.entry_reference_cut['state_sha256'],
        'actual_state_sha256':before_sha,'complete_state_equality_required':False}
    after_sha=service.state_sha(env)
    if before_sha!=after_sha: raise ValueError('read-only handoff capture changed physical state')
    for name,value in (('pose',pose),('reading',reading),('snapshot',snapshot),('comparison',comparison)):
        pe._write_json_atomic(directory/(name+'.json'),clean(value))
    svc.entry_handoff_count+=1
    record={'directory':str(directory),'state_before_sha256':before_sha,'state_after_sha256':after_sha,
        'readonly_state_unchanged':True,'capture_count':svc.entry_handoff_count,
        'before_first_vla':True,'current_batch_saved':True,'comparison':comparison,
        'session_id':svc.entry_case_session_id,'job_id':getattr(svc,'entry_case_job_id',None),
        'case':svc.entry_case}
    pe._write_json_atomic(directory/'capture.json',clean(record))
    return record


def save_cut(svc, session, directory, old_row, historical_hashes, pe, service, pd, sg, sc, catalog, ep, profiles):
    import numpy as np
    env = svc._env; cut = directory/'cut'; cut.mkdir()
    state_hash = service.state_sha(env); snapshot = pe.capture_snapshot(env, BOWL_GOLD)
    old = old_row['snapshot']
    np.testing.assert_allclose(snapshot['eef_position'], old['eef_position'], atol=1e-6, rtol=0)
    if set(snapshot['objects']) != set(old['objects']): raise ValueError('cut physical object set mismatch')
    for name, obj in old['objects'].items():
        np.testing.assert_allclose(snapshot['objects'][name]['position'], obj['position'], atol=1e-6, rtol=0)
    pose = pd.read_eef_pose(env)
    reading = sg.read_geometry(env, catalog.CAPABILITIES['bowl_to_plate'], pose['orientation_matrix'], profiles)
    ctx = sc.build_context(catalog.CAPABILITIES['bowl_to_plate'], reading)
    reference = ep.make_entry_reference(ctx, reading, provenance={
        'cut_step':30, 'cut_state_sha256':state_hash, 'historical_files_sha256':historical_hashes,
        'origin_sha256':ORIGIN, 'source':'reference_replay_actual_worker_observation'})
    images = service.capture_vla_images(env)
    if set(images) != {'agentview','wrist'}: raise ValueError('both native VLA camera frames required')
    for view, image in images.items(): service._save_png(cut/(view+'.png'), image)
    inner = service._inner_env(env); sim = inner.sim
    np.save(cut/'sim_state.npy',state_vector(env,service),allow_pickle=False)
    np.savez_compressed(cut/'qpos_qvel.npz',qpos=np.asarray(sim.data.qpos),qvel=np.asarray(sim.data.qvel))
    save_numeric(cut/'observation.npz',svc._last_obs,pe)
    pe._write_json_atomic(cut/'reading.json',clean(reading))
    pe._write_json_atomic(cut/'snapshot.json',clean(snapshot))
    pe._write_json_atomic(cut/'pose.json',clean(pose))
    pe._write_json_atomic(directory.parent/'entry_reference.json',clean(reference))
    if service.state_sha(env)!=state_hash: raise ValueError('read-only cut capture changed physical state')
    return {'state_sha256':state_hash,'directory':str(cut),'reference':reference,'old_row30_position_comparison':True,
            'old_orientation_not_available_no_equality_claim':True,'images':{v:str(cut/(v+'.png')) for v in images}}


def replay(svc, sid, rows, action_key, goals, directory, pe, service, cut_callback=None, save_video=False):
    import numpy as np
    import imageio.v2 as imageio
    session, env = svc._sessions[sid], svc._env
    before = pe.capture_snapshot(env, goals); origin = service.state_sha(env)
    snapshots, cut, writer = [], None, None
    if save_video:
        initial = service.capture_vla_images(env)['agentview']; service._save_png(directory/'first.png',initial)
        writer = imageio.get_writer(str(directory/'rollout.mp4'),fps=20); writer.append_data(initial)
    try:
        with (directory/'replay.jsonl').open('w',encoding='utf8') as f:
            for count, row in enumerate(rows,1):
                action = np.asarray(row[action_key],dtype=np.float32)
                if action.shape!=(7,) or not np.isfinite(action).all(): raise ValueError('invalid replay action')
                result = env.step(action); svc._last_obs = result[0]
                svc._total_steps+=1; session.total_steps+=1
                snapshot = pe.capture_snapshot(env,goals); snapshots.append(snapshot)
                f.write(json.dumps(clean({'step':count,'source':'historical_replay','sent_action':action,
                    'state_sha256':service.state_sha(env),'oracle_snapshot':snapshot}))+'\n'); f.flush()
                frames = service.capture_vla_images(env)
                if save_video: writer.append_data(frames['agentview']); service._save_png(directory/'last.png',frames['agentview'])
                if count==30 and cut_callback:cut=cut_callback()
    finally:
        if writer is not None: writer.close()
    after = pe.capture_snapshot(env,goals); stove_off = service.eval_goal_predicate(env,STOVE_OFF)
    pe._write_json_atomic(directory/'replay_before.json',clean(before))
    pe._write_json_atomic(directory/'replay_after.json',clean(after))
    return {'ok':True,'before':before,'after':after,'origin_sha256':origin,
            'after_sha256':service.state_sha(env),'replay_actions':len(rows),'cut':cut,
            'score':score_literal(snapshots,after,before,stove_off,goals),'stove_off':stove_off}


def job_worker(svc, sid, case, directory, gold, pe, service):
    session = svc._sessions[sid]; env = svc._env
    before = pe.capture_snapshot(env,gold); before_sha = service.state_sha(env)
    request_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    capids = ['wine_to_rack','bowl_to_plate'] if case==CASES[3] else ['bowl_to_plate']
    plan = service.PlanRecord(request_id,sid,{'decision':'execute','capability_ids':capids,
        'budget_per_subgoal':500,'scene_version':session.scene_version,'audit':False,
        'rationale':'Five-gate entry compatibility diagnostic; no new Hermes request.'})
    if case==CASES[3]:plan.completed_capability_ids=['wine_to_rack']
    plan.pending_capability_ids=['bowl_to_plate']; plan.state='running'
    job = service.JobRecord(job_id,request_id,sid,'bowl_to_plate',directory/'job');plan.job_ids=[job_id]
    for record in (session,job):
        record.entry_case_dir=directory;record.entry_case_gold=gold
        record.entry_reference_cut=getattr(svc,'entry_reference_cut',None)
    svc.entry_case_job_id=job_id
    with svc._lock:
        svc._plans[request_id]=plan;svc._jobs[job_id]=job;svc._job_order.append(job_id);session.active_request_id=request_id
    value, error = None, None
    try: value=svc._run_capability(session,plan,job,'bowl_to_plate')
    except service.SceneError as exc:
        error={'reason':exc.reason,'detail':exc.detail}
        if job.state not in ('completed','error','cancelled'):job.state='error';job.success=False;job.ended_reason=exc.reason
        job.error=str(exc)
    finally:
        after=pe.capture_snapshot(env,gold);stove=service.eval_goal_predicate(env,STOVE_OFF)
        plan.state='cancelled' if job.state=='cancelled' else ('completed' if job.success is True else 'blocked')
        if job.success is True:plan.completed_capability_ids.append('bowl_to_plate');plan.pending_capability_ids=[]
        session.active_request_id=None
        pe._write_json_atomic(directory/'before_snapshot.json',clean(before))
        pe._write_json_atomic(directory/'after_snapshot.json',clean(after))
        pe._write_json_atomic(directory/'raw_job.json',clean(job.public()))
        pe._write_json_atomic(directory/'raw_plan.json',clean(plan.public()))
    return {'ok':True,'job':job.public(),'plan':plan.public(),'base_result':value,'scene_error':error,
            'before':before,'after':after,'stove_off':stove,'before_sha256':before_sha,'after_sha256':service.state_sha(env)}


def file_records(directory):
    return [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.rglob('*')) if p.is_file()]


def run_case(svc, case, output, bowl, wine, historical_hashes, profiles, pe, service, pd, sg, sc, catalog, ep, pilot, paired):
    directory=output/case;directory.mkdir(); gold=gold_for(case)
    enabled=case in CASES[2:]; init_index=1 if case==CASES[4] else 0
    entry={'case':case,'operational_errors':[],'unknown':[],'physical_success':False,'new_hermes_calls':0,
           'model_seed':0,'init_state_index':init_index,'gold':gold,'assist_enabled':enabled,'replay_actions':0}
    try:
        pe.FINAL_ORACLE_GOALS[CONDITION]=[list(g) for g in gold]
        svc._diag_condition=CONDITION;svc._subtask_assist_mode='enabled' if enabled else 'disabled'
        session=require(svc.create_session('goal_table',seed=0,init_state_index=init_index),'session');sid=session['session_id']
        entry['session']=session
        svc.entry_case=case;svc.entry_case_dir=directory;svc.entry_case_gold=gold
        svc.entry_case_session_id=sid;svc.entry_case_job_id=None
        svc.entry_handoff_capture=None;svc.entry_handoff_count=0
        def origin_read():return {'ok':True,'state_sha256':service.state_sha(svc._env)}
        origin=require(svc._sync_work('entry_origin',origin_read,timeout=900),'origin');entry['origin_sha256']=origin['state_sha256']
        if init_index==0 and origin['state_sha256']!=ORIGIN:raise ValueError('native origin mismatch')
        if case==CASES[0]:
            def replay_reference():
                def cut():return save_cut(svc,svc._sessions[sid],directory,bowl[29],historical_hashes,pe,service,pd,sg,sc,catalog,ep,profiles)
                return replay(svc,sid,bowl,'action',gold,directory,pe,service,cut_callback=cut,save_video=True)
            value=require(svc._sync_work('reference_replay',replay_reference,timeout=900),'reference replay')
            svc.entry_reference=value['cut']['reference'];svc.entry_cut_sha=value['cut']['state_sha256']
            svc.entry_reference_cut=value['cut']
            entry.update(replay_actions=102,new_vla_actions=0,prepare_actions=0,score=value['score'],
                new_motion_actions=0,physical_actions=102,new_model_calls=0,
                physical_success=value['score']['physical_gold_success'],cut_sha256=svc.entry_cut_sha,
                after_sha256=value['after_sha256'],replay_before_sha256=value['origin_sha256'])
        else:
            if case==CASES[1]:
                value=require(svc._sync_work('native_cut_replay',lambda:replay(svc,sid,bowl[:30],'action',gold,directory,pe,service),timeout=900),'cut replay')
                if value['after_sha256']!=svc.entry_cut_sha:raise ValueError('replayed cut state differs from case01')
                entry.update(replay_actions=30,prefix_after_sha256=value['after_sha256'])
                def capture_resumed_cut():
                    batch=svc._observation_batch(svc._last_obs,catalog.CAPABILITIES['bowl_to_plate']['instruction'])
                    return {'ok':True,'capture':save_handoff(svc,batch,profiles,pe,service,pd,sg,catalog)}
                captured=require(svc._sync_work('resumed_cut_handoff_capture',capture_resumed_cut,timeout=900),'case02 handoff capture')
                svc.entry_handoff_capture=captured['capture']
            if case==CASES[3]:
                value=require(svc._sync_work('wine_prefix_replay',lambda:replay(svc,sid,wine,'sent_action',gold,directory,pe,service),timeout=900),'wine replay')
                if value['origin_sha256']!=ORIGIN or value['after_sha256']!=WINE_HANDOFF:raise ValueError('wine exact state mismatch')
                if value['after'].get('held_objects')!=[] or value['after'].get('grasp_observation_complete') is not True or value['after']['predicates'].get('on|wine_bottle_1|wine_rack_1_top_region') is not True:
                    raise ValueError('wine handoff not known empty and goal true')
                entry.update(replay_actions=346,prefix_after_sha256=value['after_sha256'])
            seeded=require(paired._seed_model_rng(svc,0),'fresh model seed')
            if seeded.get('seeded') is not True:raise RuntimeError('unseeded model RNG')
            entry['model_rng']=seeded
            actual=require(svc._sync_work('entry_bowl_job',lambda:job_worker(svc,sid,case,directory,gold,pe,service),timeout=900),'bowl worker')
            job=actual['job'];run_dir=Path(job['run_dir']);rows,paths=pilot.read_oracle_rows([job])
            score=score_literal(rows,actual['after'],actual['before'],actual['stove_off'],gold)
            assist=read(run_dir/'subtask_assist.json')
            entry.update(job=job,plan=actual['plan'],scene_error=actual['scene_error'],score=score,
                prepare_confirmed=assist.get('prepare_confirmed'),prepare_actions=assist.get('prepare_source_actions'),
                new_vla_actions=assist.get('vla_source_actions'),source_counts={
                    'prepare':assist.get('prepare_source_actions'),'local_grasp':assist.get('local_grasp_source_actions'),
                    'vla':assist.get('vla_source_actions')},
                new_motion_actions=job.get('steps'),physical_actions=entry['replay_actions']+job['steps'],
                before_sha256=actual['before_sha256'],after_sha256=actual['after_sha256'],
                handoff_capture=svc.entry_handoff_capture,handoff_capture_count=svc.entry_handoff_count,
                telemetry_paths=paths,assistance_path=str(run_dir/'subtask_assist.json'),
                diagnostic_path=str(run_dir/'diagnostic.json'),rollout=job.get('rollout_path'))
            entry['physical_success']=job.get('success') is True and score['physical_gold_success'] is True and (not enabled or assist.get('prepare_confirmed') is True)
            reason=actual['scene_error']['reason'] if actual['scene_error'] else job.get('ended_reason')
            if job.get('state')=='cancelled':entry['operational_errors'].append('cancelled')
            elif job.get('error') and reason not in PHYSICAL_ERRORS:entry['operational_errors'].append({'reason':reason,'error':job['error']})
            elif job.get('state')=='error' and reason not in PHYSICAL_ERRORS:entry['operational_errors'].append({'reason':reason,'error':'job_error'})
            if enabled and assist.get('prepare_confirmed') is True and svc.entry_handoff_count!=1:
                entry['operational_errors'].append('confirmed_prepare_without_once_only_actual_handoff_capture')
        entry['unknown']=entry['score']['unknown'];entry['gate_pass']=entry['physical_success'] is True and not entry['unknown'] and not entry['operational_errors']
    except Exception:
        entry['operational_errors'].append(traceback.format_exc());entry['gate_pass']=False
    finally:
        entry['evidence']=file_records(directory)
        pe._write_json_atomic(directory/'trial.json',clean(entry))
    return entry


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output-dir','profiles','software-proof','inputs'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args(argv);output,profiles,software,inputs,sources=gates(args)
    bowl,wine,diagnostic,history=load_inputs(inputs)
    sys.path.insert(0,str(HERE))
    import catalog
    import preparation_diagnostics as pd
    import placement_experiments as pe
    import service
    import subtask_geometry as sg
    import subtask_context as sc
    import entry_compatibility as ep
    import side_grasp_pilot as pilot
    import paired_config_experiments as paired
    import guard_validation as gv
    import grasp_assist_service as gas
    from subtask_assist_service import UnifiedAssistService
    EntryProbeService=make_service_class(UnifiedAssistService,ep,
        lambda svc,batch:save_handoff(svc,batch,profiles,pe,service,pd,sg,catalog))
    fixed_files={str(Path(args.profiles)):sha(args.profiles),str(PLAN_PATH):sha(PLAN_PATH),
                 str(Path(__file__)):sha(__file__),str(Path(args.inputs)):sha(args.inputs),
                 str(Path(args.software_proof)):sha(args.software_proof)}
    prereg={'cases':CASES,'max_physical_cases':5,'case01_reference_actions':102,'cut_step':30,
        'case02_replay_actions':30,'case04_wine_replay_actions':346,'case05_init_state_index':1,
        'cut_origin_sha256':ORIGIN,'wine_handoff_sha256':WINE_HANDOFF,'gold_by_case':{c:gold_for(c) for c in CASES},
        'stop_rule':'Stop remaining cases unless previous physical_success is strictly True and no operation/unknown/cancel errors; no retry.',
        'enabled_cases':CASES[2:],'generic_prepare_cap':200,'entry_bridge_cap':100,'total_action_budget':500,
        'prepare_position_tolerance_m':.005,'prepare_orientation_tolerance_rad':.05,'post_samples':5,
        'shape_dimension_profile_tolerance':.10,'protected_objects':['wine_bottle_1','cream_cheese_1'],
        'protected_displacement_m':.005,'stove_goal':STOVE_OFF,'env_seed':0,'model_seed':0,
        'model_seed_limit':'Fresh continuation seed0 is not recreation of old random sampling stream.',
        'profile':'baseline_bf16','profile_config':{'use_amp':True,'num_steps':10,'n_action_steps':1},
        'instruction':catalog.CAPABILITIES['bowl_to_plate']['instruction'],'completion_mode':'release_verified',
        'grasp_guard_mode':'shadow','checkpoint_revision':'6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
        'new_hermes_calls':0,'new_baseline_jobs':0,'retries':0,'extra_physical_preflight':0,
        'source_sha256':sources,'historical_sha256':history,'fixed_files_sha256':fixed_files,
        'inputs_before':inputs['inputs_before'],'user8_before':inputs['user8_before'],
        'entry_profile':'object-local normalized position, relative orientation/gap at preregistered step30; no retuning',
        'no_production_changes':True,'full_arm_ik_or_collision_guarantee':False}
    output.mkdir(parents=True);pe._write_json_atomic(output/'preregistration.json',clean(prereg))
    print(json.dumps({'event':'PREREG','planned_cases':5,'path':str(output/'preregistration.json')}),flush=True)
    report={'ok':False,'planned_cases':5,'executed_cases':0,'cases':[],'not_run':[{'case':c,'reason':'not_started'} for c in CASES],
            'operational_errors':[],'new_hermes_calls':0,'model_seed':0,'preregistration_sha256':sha(output/'preregistration.json')}
    svc=None;saved_oracle=dict(pe.FINAL_ORACLE_GOALS)
    try:
        gas.configure_process_environment()
        svc=EntryProbeService(model_path=service.DEFAULT_MODEL_PATH,run_root=str(output/'sessions'),
            calibration_profiles=profiles,assist_mode='disabled',completion_mode='release_verified',grasp_guard_mode='shadow')
        svc.start();deadline=time.monotonic()+180
        while True:
            health=svc.health();report['health']=health
            if health.get('worker_error'):raise RuntimeError(str(health['worker_error']))
            if health.get('ready') is True:break
            if time.monotonic()>=deadline:raise TimeoutError('readiness timeout180')
            time.sleep(1)
        profile=svc.configure_profile('baseline_bf16');report['profile_readback']=profile
        if not gv._profile_readback_ok(profile):raise RuntimeError('profile readback mismatch')
        for case in CASES:
            if source_map()!=sources:raise ValueError('current sources changed')
            verify_facts(inputs['inputs_before']);verify_facts(inputs['user8_before'])
            print(json.dumps({'event':'START','case':case}),flush=True)
            value=run_case(svc,case,output,bowl,wine,history,profiles,pe,service,pd,sg,sc,catalog,ep,pilot,paired)
            report['cases'].append(value);report['executed_cases']=len(report['cases'])
            report['operational_errors'].extend(value['operational_errors'])
            report['not_run']=[{'case':c,'reason':'not_started'} for c in CASES[len(report['cases']):]]
            print(json.dumps({'event':'DONE','case':case,'physical_success':value['physical_success'],'gate_pass':value['gate_pass'],
                'replay_actions':value['replay_actions'],'prepare_actions':value.get('prepare_actions'),'vla_actions':value.get('new_vla_actions')}),flush=True)
            if value['gate_pass'] is not True:
                report['not_run']=[{'case':c,'reason':'previous_case_gate_failed:'+case} for c in CASES[len(report['cases']):]]
                report['gate_stopped_at']=case
            pe._write_json_atomic(output/'report.json',clean(report))
            if value['gate_pass'] is not True:break
    except Exception:report['operational_errors'].append(traceback.format_exc())
    finally:
        if svc is not None:
            try:report['close_env']=require(svc._sync_work('close_owned_env',lambda:(svc._close_env() or {'ok':True}),timeout=900),'close env')
            except Exception:report['operational_errors'].append(traceback.format_exc())
            finally:svc.stop()
        pe.FINAL_ORACLE_GOALS.clear();pe.FINAL_ORACLE_GOALS.update(saved_oracle)
        report['source_unchanged']=source_map()==sources
        report['fixed_files_unchanged']=all(sha(p)==h for p,h in fixed_files.items())
        try:verify_facts(inputs['inputs_before']);report['inputs_unchanged']=True
        except Exception:report['inputs_unchanged']=False;report['operational_errors'].append(traceback.format_exc())
        try:verify_facts(inputs['user8_before']);report['user8_unchanged']=True
        except Exception:report['user8_unchanged']=False;report['operational_errors'].append(traceback.format_exc())
        report['protocol_gate_observed']=all(v.get('gate_pass') is True for v in report['cases'][:-1]) and report['executed_cases']<=5
        report['ok']=not report['operational_errors'] and report['source_unchanged'] and report['fixed_files_unchanged'] and report['inputs_unchanged'] and report['user8_unchanged'] and report['protocol_gate_observed']
        pe._write_json_atomic(output/'report.json',clean(report))
    return 0 if report['ok'] else 2


if __name__=='__main__':
    raise SystemExit(main())
