"""Read original local evidence; derive comparisons and exact-byte archives."""
import gzip, hashlib, json, shutil
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parents[3]
old = out.parent/'2026-10-14-sim-fast'
manifest = []
missing = []
PHASES = ['policy','handoff_and_release','retreat_plan_motion_confirm','home_entry_and_home','finalization']
def local(p):
    s=str(p).replace('\\','/')
    for prefix in ['/mnt/d/FYP/','D:/FYP/']:
        if s.lower().startswith(prefix.lower()): return root/s[len(prefix):]
    return Path(s)
def read(p):
    p=local(p)
    if not p.is_file(): return None
    return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):
    p=local(p)
    if not p.is_file(): return []
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(local(p).read_bytes()).hexdigest() if local(p).is_file() else None
def dig(x,*keys):
    for k in keys:
        if isinstance(x,dict): x=x.get(k)
        elif isinstance(x,list) and isinstance(k,int) and len(x)>k: x=x[k]
        else:return None
    return x
def archive(p):
    p=local(p)
    if not p.is_file(): missing.append(str(p));return
    rel=p.relative_to(root)
    dest=out/'evidence'/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest)
    manifest.append({'source':str(p),'relative_copy':dest.relative_to(out).as_posix(),'sha256':sha(p),'copy_sha256':sha(dest)})
def refs(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in ('rgb_path','depth_path','path') and isinstance(v,str): yield v
            else:yield from refs(v)
    elif isinstance(x,list):
        for v in x:yield from refs(v)
def release_check(event,archive_new):
    capture=event.get('capture') or {}
    images=event.get('images') or []
    records=[]
    for im in images:
        p=im.get('path'); actual=sha(p) if p else None
        records.append({'view':im.get('view'),'path':p,'exists':actual is not None,'sha256':actual,'declared_sha256':im.get('sha256'),'sha_matches':actual==im.get('sha256') if actual and im.get('sha256') else None})
    frames=capture.get('frames',[])+capture.get('verification_frames',[])+capture.get('checker_frames',[])
    timestamps=[f['timestamp'] for f in frames if 'timestamp' in f]
    time_unknown=not timestamps
    if archive_new:
        for p in set(refs(event)):archive(p)
    return {'event_id':event.get('event_id'),'capture_timestamp':capture.get('timestamp'),'frame_timestamps':timestamps,'timestamps_matched':all(t==capture.get('timestamp') for t in timestamps) if not time_unknown else None,'timestamp_scope_note':'Frames without individual timestamps inherit capture timestamp only; no inferred image timestamp.','images':records,'three_views_present':{'wrist','agentview','sideview'}.issubset({i['view'] for i in records}),'all_images_exist':bool(records) and all(i['exists'] for i in records)}
def job_info(job,archive_new):
    run=local(job['run_dir']); pilot=read(run/'pilot_result.json')
    if pilot is None:return {'capability':job.get('capability_id'),'status':'unknown','missing_pilot':str(run/'pilot_result.json')}
    C=dig(pilot,'finish','plan','geometry_frames',0,'rgb_path')
    D=dig(pilot,'finish','home_entry','clearance_screen','frames',0,'rgb_path')
    paths=[run/'first.png',run/'finish/simulator_samples.jsonl',local(C) if C else None,local(D) if D else None,run/'finish_actions.jsonl',run/'pilot_result.json']
    boundaries={k:{'source':str(p) if p else None,'st_mtime_ns':p.stat().st_mtime_ns if p and p.is_file() else None} for k,p in zip('ABCDEF',paths)}
    times=[v['st_mtime_ns'] for v in boundaries.values()]
    valid=all(t is not None for t in times) and all(b>=a for a,b in zip(times,times[1:]))
    phases={k:(times[i+1]-times[i])/1e9 if valid else None for i,k in enumerate(PHASES)}
    sensors=list((run/'finish/sensors').rglob('*')); sensors=[p for p in sensors if p.is_file()]
    sensor_stats={'all_files':len(sensors),'bytes':sum(p.stat().st_size for p in sensors),'PNG':sum(p.suffix.lower()=='.png' for p in sensors),'NPY':sum(p.suffix.lower()=='.npy' for p in sensors),'groups':sum('checker' not in p.stem for p in (run/'finish/sensors').glob('cap*_agentview.png'))}
    requests=sorted((run/'xvla_calls').glob('request_*.json.gz')); http=[]
    payloads={}
    for p in requests:
        response=p.with_name(p.name.replace('request_','response_').removesuffix('.gz'))
        delta=(response.stat().st_mtime_ns-p.stat().st_mtime_ns)/1e9 if response.is_file() else None
        http.append({'request':str(p),'response':str(response),'seconds':delta if delta is not None and delta>=0 else None})
        payloads[p.name]=hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()
    events=lines(run/'events.jsonl'); action_present=bool(events) and 'action' in events[0]
    actions=[e.get('action') for e in events] if action_present else None
    process=lines(run/'finish/process_events.jsonl')
    release=[release_check(e,archive_new) for e in process if str(e.get('event_id','')).startswith('release')]
    simulator=lines(run/'finish/simulator_samples.jsonl'); independent=lines(run/'independent_finish.jsonl')
    if archive_new:
        for name in ['pilot_result.json','result.json','events.jsonl','finish_actions.jsonl','independent_finish.jsonl','handoff_reference.json','first.png','last.png','finish/checks.jsonl','finish/simulator_samples.jsonl','finish/process_events.jsonl']:archive(run/name)
    finish=pilot.get('finish') or {}; status=pilot.get('finish_status') or {}
    return {'capability':pilot.get('capability'),'run_dir':str(run),'status':'available','raw_boundaries':boundaries,'phase_status':'available' if valid else 'unknown_missing_or_nonmonotonic','phase_seconds':phases,'source_counts':pilot.get('source_counts'),'finish_counts':finish.get('counts'),'release_open_actions':finish.get('release_open_actions'),'trigger_steps':[r.get('step') for r in simulator if r.get('triggered') or r.get('trigger')],'finish_status':status,'independent_streak':status.get('independent_streak'),'handoff_evidence':pilot.get('handoff_evidence'),'final_metrics':pilot.get('actual_final_metrics'),'handoff_reference':pilot.get('handoff_reference'),'next_capability':dig(read(run/'handoff_reference.json'),'next_capability'),'final_independent_predicates':independent[-1] if independent else None,'optimization_counters':{k:status.get(k) for k in ['skipped_policy_captures','skipped_helper_captures','observation_cache_hits','observation_cache_misses','native_step_seconds','native_step_count']},'sensor_files':sensor_stats,'VLM_request_file_count':sum(p.is_file() for p in (run/'finish/model_requests').rglob('*')),'VLA_HTTP_proxy':{'note':'Response mtime minus request mtime includes inference and HTTP overhead; not pure GPU.','count':len(http),'sum_seconds':sum(i['seconds'] for i in http) if all(i['seconds'] is not None for i in http) else None,'pairs':http},'first_png_sha256':sha(run/'first.png'),'request_payload_sha256':payloads,'action_key_in_first_event':action_present,'action_sequence':actions,'release_events':release}
def case(folder,kind,new):
    prefix='bowl2' if kind=='bowl' and not new else kind
    timingname='bowl2_valid_timing.json' if kind=='bowl' and not new else kind+'_timing.json'
    final=read(folder/(prefix+'_final.json')); timing=read(folder/timingname)
    if final is None or timing is None:return {'status':'pending','timing':timing}
    jobs=[job_info(j,new) for j in final.get('jobs',[])]
    phases={k:sum(j['phase_seconds'][k] for j in jobs) if jobs and all(j.get('phase_seconds',{}).get(k) is not None for j in jobs) else None for k in PHASES}
    planpath=local(final['jobs'][0]['run_dir']).parent/'planning_grounding' if final.get('jobs') else None
    planning=read(planpath/'joint_planning_timing.json') if planpath else None
    if new and planpath:
        for name in ['joint_planning_timing.json','joint_draft.json','selected_targets_review.json','inventory_confirmed.json','planning_three_views.png']:archive(planpath/name)
    cold=None; changes=timing.get('stage_changes',[])
    for i,e in enumerate(changes):
        if e.get('stage')=='loading' and '正在加载 X-VLA' in e.get('message',''):
            nextplan=next((s for s in changes[i+1:] if s.get('stage')=='planning'),None)
            if nextplan:cold={'seconds':nextplan['monotonic']-e['monotonic'],'from':e,'to':nextplan,'note':'Polling stage approximation includes model/build overhead; not paired warm-model time.'};break
    return {'status':'available','completed':timing.get('completed'),'timing':timing,'phase_seconds':phases,'other_execution_seconds':timing.get('execution_seconds')-sum(phases.values()) if timing.get('execution_seconds') is not None and all(v is not None for v in phases.values()) else None,'planning':planning,'cold_loading_to_planning_proxy':cold,'jobs':jobs,'sensor_files':{k:sum(j['sensor_files'][k] for j in jobs) for k in ['all_files','bytes','PNG','NPY','groups']}}
def delta(a,b):return {'old':a,'new':b,'delta':b-a,'percent':100*(b-a)/a if a else None} if isinstance(a,(float,int)) and isinstance(b,(float,int)) else {'old':a,'new':b,'delta':None,'percent':None}
def compare(a,b):
    if b['status']=='pending':return {'status':'pending'}
    result={'status':'available','timings':{k:delta(a['timing'].get(k),b['timing'].get(k)) for k in ['load_scene_seconds','prepare_and_plan_seconds','execution_seconds','total_seconds']},'phases':{k:delta(a['phase_seconds'][k],b['phase_seconds'][k]) for k in PHASES},'sensor_files':{k:delta(a['sensor_files'][k],b['sensor_files'][k]) for k in a['sensor_files']},'jobs':[],'pairing_note':'Model RNG is not paired. Cached environment reuses unmodified native step-returned observations and avoids extra force refreshes; old/new observation sampling is not guaranteed byte-equivalent. Installed robosuite.step updates observables before returning observations with periodic sensor sampling. Same initial seed or code hashes do not prove equal observations, inputs, actions, or pure causal speed; use explicit input/action comparisons.'}
    for oldjob in a['jobs']:
        newjob=next((j for j in b['jobs'] if j['capability']==oldjob['capability']),None)
        if not newjob:result['jobs'].append({'capability':oldjob['capability'],'status':'unmatched'});continue
        req=[]
        for k in sorted(set(oldjob['request_payload_sha256'])|set(newjob['request_payload_sha256'])):
            x=oldjob['request_payload_sha256'].get(k);y=newjob['request_payload_sha256'].get(k)
            req.append({'request':k,'old_sha256':x,'new_sha256':y,'equal':x==y if x and y else None})
        result['jobs'].append({'capability':oldjob['capability'],'source_counts_equal':oldjob['source_counts']==newjob['source_counts'],'old_source_counts':oldjob['source_counts'],'new_source_counts':newjob['source_counts'],'finish_counts_equal':oldjob['finish_counts']==newjob['finish_counts'],'old_finish_counts':oldjob['finish_counts'],'new_finish_counts':newjob['finish_counts'],'first_png_equal':oldjob['first_png_sha256']==newjob['first_png_sha256'],'request_payload_comparison':req,'request_equal_count':sum(r['equal'] is True for r in req),'request_unequal_count':sum(r['equal'] is False for r in req),'action_comparison':{'status':'compared' if oldjob['action_key_in_first_event'] and newjob['action_key_in_first_event'] else 'not_compared','equal':oldjob['action_sequence']==newjob['action_sequence'] if oldjob['action_key_in_first_event'] and newjob['action_key_in_first_event'] else None,'old_count':len(oldjob['action_sequence']) if oldjob['action_sequence'] is not None else None,'new_count':len(newjob['action_sequence']) if newjob['action_sequence'] is not None else None}})
    return result
expected={'bowl':[61.842,.814,47.168,37.917,1.958],'chain':[108.044,14.083,105.409,150.696,4.637]}
old_protected=read(old/'protected_controls.json') or {}
new_protected=read(out/'protected_controls.json') or {}
result={'note':'Original file mtime phase proxies: capture/render/write overlap; neither pure physics nor pure planning. No scientific success claims.','protected_source_hashes':{'recorded_sets_equal':old_protected==new_protected,'current_hash_checks':{k:{'recorded_sha256':v,'current_sha256':sha(root/'First_Phase/scene_demo'/k),'equal':sha(root/'First_Phase/scene_demo'/k)==v} for k,v in new_protected.items()},'note':'Native policy/control source hashes establish code preservation; they do not establish identical model inputs or RNG.'},'cases':{}}
for kind in ['bowl','chain']:
    a=case(old,kind,False);b=case(out,kind,True)
    check={k:{'computed':a['phase_seconds'][k],'expected':v,'within_1_second':a['phase_seconds'][k] is not None and abs(a['phase_seconds'][k]-v)<=1} for k,v in zip(PHASES,expected[kind])}
    result['cases'][kind]={'baseline':a,'new':b,'comparison':compare(a,b),'baseline_phase_check':check}
    print(json.dumps({'case':kind,'baseline_phases':a['phase_seconds'],'baseline_within_1s':all(v['within_1_second'] for v in check.values()),'baseline_sensors':a['sensor_files'],'new_status':b['status']},ensure_ascii=False))
    for x in a['jobs']+(b.get('jobs') or []):x.pop('action_sequence',None)
(out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
unique={m['source']:m for m in manifest}
(out/'evidence_manifest.json').write_text(json.dumps({'files':list(unique.values()),'missing_sources':sorted(set(missing))},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
