"""Fresh real Qwen on old synchronous captures; never execute environment actions."""
import copy,hashlib,json,shutil,sys
from pathlib import Path
from scene_demo.xvla_evidence_memory import XvlaEvidenceMemoryClient,PROTOCOL,is_placement
from scene_demo.xvla_process_evidence import capture_frames
ROOT=Path('/mnt/d/FYP');OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'2026-10-13-xvla-timeline-trial'
def load(p):return json.loads(Path(p).read_bytes())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cap(ctx,paths):
    return {'timestamp':ctx['actuator_measurements']['timestamp'],
            'robot':{'finger_qpos':ctx['actuator_measurements']['finger_qpos']},
            'frames':[{'view':f['view'],'rgb_path':p} for f,p in zip(ctx['sensor_frames'],paths)]}
def dataset(case):
    folder=OLD/('microwave_mug' if case=='microwave_mug' else 'basket_independent/basket_two')
    finish=next(folder.rglob('process_events.jsonl')).parent
    checks=rows(finish/'checks.jsonl');releases=rows(finish/'process_checks.jsonl')
    traces=rows(next(folder.rglob('native_steps.jsonl')));by_step={r['step']:r for r in rows(finish.parent/'events.jsonl')}
    def truth(time):
        matches=[r for r in traces if r['source']=='vla' and abs(r['sim_time']-time)<1e-8]
        assert len(matches)==1
        step=matches[0]['counts']['vla'];state=by_step[step]
        return {'vla_step':step,'native_success':state['native_success'],'predicates':state['declared_predicates']}
    return finish,checks,releases,truth
def init(finish,ctx,directory,budget):
    directory.mkdir(exist_ok=False)
    client=XvlaEvidenceMemoryClient(max_calls=budget);client.set_check_contract(ctx);client.audit_dir=directory/'model_requests'
    client.remember_identity_images([finish/'sensors'/('cap%06d_%s.png'%(i,v))
        for i,v in [(2,'wrist'),(1,'agentview'),(3,'sideview')]],client.required_views)
    return client
def retain_check(meta):
    packet=meta['event_packet'];now=meta['input_context']['actuator_measurements']['timestamp']
    archive=[r for r in meta['release_archive'] if r['timestamp']<=now]
    assert [r['event_id'] for r in packet['release_history']]==[r['event_id'] for r in archive]
    paths=meta['request_image_paths']
    for event,group in zip(archive,packet['release_history']):
        for index,digest in zip(group['frames'],event['source']['sha256']):
            assert sha(paths[index])==digest
    assert all(group['time']<=now for group in packet['groups'])
    return len(archive)
def save(directory,records,kind,meta,truth,client):
    path=directory/('%02d_%s.json'%(len(records)+1,kind));dump(path,meta)
    now=meta['input_context']['actuator_measurements']['timestamp']
    proofs=meta['process_evidence']
    late=[p for p in proofs if p['confirmation_kind']=='current_observation' and p['timestamp']==now]
    value={'kind':kind,'time':now,'metadata':str(path.relative_to(ROOT)),
       'goal':meta['visual_goal_verdict']['goal_achieved'],'checks':meta['effective_checks'],
       'memory':[{k:p[k] for k in ('id','evidence_id','event_id','confirmation_kind','timestamp','status','object_key')} for p in proofs],
       'current_new_proofs':len(late),'all_release_originals_verified':True,
       'retained_release_events':retain_check(meta),'request_images':len(meta['request_image_paths']),
       'actual_Qwen_calls':client._calls,'same_capture_reused':meta.get('reused_same_capture',False),
       'evaluation_only_truth_not_sent':truth(now)}
    records.append(value);dump(directory/'progress.json',records)
    print(json.dumps({'case':directory.name,**{k:value[k] for k in ('kind','time','goal','current_new_proofs','retained_release_events','request_images','actual_Qwen_calls')},
        'checks':[{k:c.get(k) for k in ('id','achieved','history_event_id')} for c in value['checks']]},ensure_ascii=False),flush=True)
    return value
def failure(directory,client,exc):
    value={'type':type(exc).__name__,'message':str(exc),'actual_Qwen_calls':client._calls}
    dump(directory/'error.json',value);print(json.dumps({'case':directory.name,'error':value}),flush=True)
    return value

mode=sys.argv[1]
if mode=='freeze':
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=False);manifest={}
    files=[ROOT/'First_Phase/scene_demo'/n for n in ['xvla_evidence_memory.py','xvla_finish.py','xvla_monitor.py','xvla_service.py','xvla_pilot.py',
        'tests/test_xvla_evidence_memory.py','tests/test_xvla_evidence_recovery.py']]+[Path(__file__).resolve()]
    for path in files:
        dest=snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        manifest[path.relative_to(ROOT).as_posix()]=sha(path)
    dump(OUT/'source_manifest.json',manifest);print(json.dumps({'frozen':len(files),'protocol':PROTOCOL}));sys.exit(0)
for path,digest in load(OUT/'source_manifest.json').items():assert sha(ROOT/path)==digest

if mode=='probe':
    finish,checks,releases,truth=dataset('basket_two');directory=OUT/'recovery_probe'
    client=init(finish,checks[0]['ctx'],directory,2);records=[];error=None
    for event in releases:
        client.remember_release(event['capture'],event['context'],event['event_id'])
        client.note_release_status(event['event_id'],'unclassified_recovery_probe')
    triggered=False
    try:
        first=checks[1];client.observe_capture(cap(first['ctx'],first['image_paths']))
        meta=client.analyze(first['ctx'],first['image_paths']);record=save(directory,records,'current',meta,truth,client)
        triggered=record['current_new_proofs']>0
        if triggered:
            last=checks[-1];client.observe_capture(cap(last['ctx'],last['image_paths']))
            meta=client.analyze(last['ctx'],last['image_paths']);save(directory,records,'occlusion_followup',meta,truth,client)
    except Exception as exc:error=failure(directory,client,exc)
    dump(directory/'summary.json',{'triggered':triggered,'actual_Qwen_calls':client._calls,'error':error,'records':records,
        'classification_missing_control':True,'old_answers_imported':0,'new_environment_actions':0})
elif mode in ('microwave_mug','basket_two'):
    finish,checks,releases,truth=dataset(mode);directory=OUT/mode
    client=init(finish,checks[0]['ctx'],directory,8);records=[];pending=[];error=None
    stream=[(r['ctx']['actuator_measurements']['timestamp'],'check',r) for r in checks]
    stream += [(r['capture']['timestamp'],'release',r) for r in releases]
    try:
        for time,kind,record in sorted(stream,key=lambda r:r[0]):
            if kind=='release':
                client.remember_release(record['capture'],record['context'],record['event_id']);pending.append(record);continue
            ctx=record['ctx'];paths=record['image_paths'];capture=cap(ctx,paths)
            client.remember_observation(capture,ctx)
            for event in pending:
                client.observe_capture(event['capture']);epaths=[str(f['rgb_path']) for f in capture_frames(event['capture'])]
                meta=client.analyze_release(event['context'],epaths,event['event_id']);save(directory,records,'release',meta,truth,client)
            pending.clear();client.observe_capture(capture)
            meta=client.analyze(ctx,paths);save(directory,records,'current',meta,truth,client)
    except Exception as exc:error=failure(directory,client,exc)
    finals=[r for r in records if r['kind']=='current'];final=finals[-1] if finals else None
    false_positives=[r['time'] for r in records if r['goal'] is True and not r['evaluation_only_truth_not_sent']['native_success']]
    required={g['id'] for g in checks[-1]['ctx']['criteria']['success_checks'] if is_placement(g)}
    confirmed={c['id'] for c in final['checks'] if c.get('history_event_id') and c['achieved'] is True} if final else set()
    full=error is None and len(finals)==len(checks)
    dump(directory/'summary.json',{'complete':full,'passed':full and final['goal'] is True and required<=confirmed and not false_positives,
        'final_goal':None if final is None else final['goal'],'false_positives':false_positives,
        'actual_Qwen_calls':client._calls,'error':error,'records':records,'old_answers_imported':0,'new_environment_actions':0})
elif mode=='no_history':
    finish,checks,_,truth=dataset('microwave_mug');last=checks[-1];directory=OUT/'microwave_no_history'
    client=init(finish,last['ctx'],directory,1);records=[];error=None
    try:
        client.observe_capture(cap(last['ctx'],last['image_paths']))
        meta=client.analyze(last['ctx'],last['image_paths']);save(directory,records,'no_history',meta,truth,client)
    except Exception as exc:error=failure(directory,client,exc)
    dump(directory/'summary.json',{'passed':error is None and records[-1]['goal'] is not True,
        'actual_Qwen_calls':client._calls,'records':records,'error':error})
else:raise ValueError('unknown replay mode')
