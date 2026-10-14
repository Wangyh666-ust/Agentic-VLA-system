"""Chronological replay of recorded captures, real Qwen, zero environment actions."""
import copy,hashlib,json,shutil,sys
from pathlib import Path
from scene_demo.xvla_evidence_memory import XvlaEvidenceMemoryClient,PROTOCOL

ROOT=Path('/mnt/d/FYP');OUT=Path(__file__).resolve().parent
OLD=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial'
def load(p):return json.loads(Path(p).read_bytes())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cap(ctx,paths):
    assert len(paths)==3
    return {'timestamp':ctx['actuator_measurements']['timestamp'],
            'robot':{'finger_qpos':ctx['actuator_measurements']['finger_qpos']},
            'frames':[{'view':f['view'],'rgb_path':p} for f,p in zip(ctx['sensor_frames'],paths)]}
def init(ctx,finish,budget,directory):
    client=XvlaEvidenceMemoryClient(max_calls=budget);client.set_check_contract(ctx);client.audit_dir=directory/'model_requests'
    refs=[finish/'sensors'/('cap%06d_%s.png'%(i,v)) for i,v in [(2,'wrist'),(1,'agentview'),(3,'sideview')]]
    client.remember_identity_images(refs,client.required_views)
    return client

attempt=sys.argv[1] if len(sys.argv)>1 else 'attempt_01'
DEST=OUT/attempt;DEST.mkdir(exist_ok=False)
manifest={}
for name in ['xvla_evidence_memory.py','xvla_finish.py','xvla_monitor.py','xvla_service.py','xvla_pilot.py']:
    p=ROOT/'First_Phase/scene_demo'/name;target=DEST/'source'/name;target.parent.mkdir(exist_ok=True)
    shutil.copyfile(p,target);manifest[p.relative_to(ROOT).as_posix()]=sha(p)
shutil.copyfile(__file__,DEST/'source/replay.py');dump(DEST/'source_manifest.json',manifest)
results=[];case_results={}
for case,relative in [('microwave_mug','microwave_mug'),('basket_two','basket_independent/basket_two')]:
    directory=DEST/case;directory.mkdir();folder=OLD/relative
    finish=next(folder.rglob('process_events.jsonl')).parent
    checks=rows(finish/'checks.jsonl');releases=rows(finish/'process_checks.jsonl')
    traces=rows(next(folder.rglob('native_steps.jsonl')));steps=rows(finish.parent/'events.jsonl')
    by_step={r['step']:r for r in steps}
    def truth(time):
        matches=[r for r in traces if r['source']=='vla' and abs(r['sim_time']-time)<1e-8]
        assert len(matches)==1
        step=matches[0]['counts']['vla'];state=by_step[step]
        return {'vla_step':step,'native_success':state['native_success'],'predicates':state['declared_predicates']}
    client=init(checks[0]['ctx'],finish,8,directory)
    stream=[(r['ctx']['actuator_measurements']['timestamp'],'check',r) for r in checks]
    stream += [(r['capture']['timestamp'],'release',r) for r in releases]
    pending=[];records=[];negative=None;error=None
    def save(kind,ctx,meta):
        index=len(records)+1;path=directory/('%02d_%s.json'%(index,kind));dump(path,meta)
        value={'kind':kind,'time':ctx['actuator_measurements']['timestamp'],'metadata':str(path.relative_to(ROOT)),
            'goal':meta['visual_goal_verdict']['goal_achieved'],'checks':meta['effective_checks'],
            'memory':[{k:p[k] for k in ('id','event_id','timestamp','status','object_key')} for p in meta['process_evidence']],
            'call_index':meta.get('call_index'),'actual_Qwen_calls':client._calls,
            'same_capture_reused':meta.get('reused_same_capture',False),
            'evaluation_only_truth_not_sent':truth(ctx['actuator_measurements']['timestamp'])}
        records.append(value);dump(directory/'progress.json',records)
        print(json.dumps({'case':case,**value},ensure_ascii=False),flush=True)
        return value
    try:
        for time,kind,record in sorted(stream,key=lambda r:r[0]):
            if kind=='release':
                client.remember_release(record['capture'],record['context'],record['event_id']);pending.append(record)
                continue
            ctx=record['ctx'];paths=record['image_paths'];capture=cap(ctx,paths)
            client.remember_observation(capture,ctx)
            for event in pending:
                ec=event['capture'];econtext=event['context'];client.observe_capture(ec)
                from scene_demo.xvla_process_evidence import capture_frames
                epaths=[str(f['rgb_path']) for f in capture_frames(ec)]
                meta=client.analyze_release(econtext,epaths,event['event_id'])
                event_ctx=meta['input_context'];actual_time=event_ctx['actuator_measurements']['timestamp']
                value=save('release',event_ctx,meta)
                if negative is None:negative=value
                assert actual_time<=time
            pending.clear();client.observe_capture(capture)
            meta=client.analyze(ctx,paths);save('current',ctx,meta)
        assert not pending
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc),'actual_Qwen_calls':client._calls}
        dump(directory/'error.json',error);print(json.dumps({'case':case,'error':error}),flush=True)
    positives=[r for r in records if r['kind']=='current']
    final=positives[-1] if positives else None
    false_positives=[r for r in records if r['goal'] is True and not r['evaluation_only_truth_not_sent']['native_success']]
    required=[g['id'] for g in checks[-1]['ctx']['criteria']['success_checks'] if g['relation'] in ('in','on','right_of') and not g.get('preserved')]
    confirmed=[] if final is None else [r['id'] for r in final['checks'] if r.get('history_event_id') and r['achieved'] is True]
    positive_pass=error is None and len(positives)==len(checks) and final['goal'] is True and set(required)<=set(confirmed) and not false_positives
    negative_pass=negative is not None and negative['goal'] is not True and not negative['evaluation_only_truth_not_sent']['native_success']
    for mode,passed,record in [('positive',positive_pass,final),('negative',negative_pass,negative)]:
        results.append({'case':case,'mode':mode,'passed':passed,'goal':None if record is None else record['goal'],
                        'checks':[] if record is None else record['checks'],'error':error})
    case_results[case]={'complete':error is None,'checkpoints':len(positives),'release_checks':sum(r['kind']=='release' for r in records),
        'actual_Qwen_calls':client._calls,'false_positives':false_positives,'confirmed_goals':confirmed,
        'final_memory':client.process_evidence_snapshot(),'records':records}
    dump(directory/'summary.json',case_results[case])
    dump(DEST/'summary.json',{'protocol':PROTOCOL,'complete':False,'passed':False,'results':results,'cases':case_results,'new_environment_actions':0})

# Independent negative: same final mug photos, no release evidence of any kind.
folder=OLD/'microwave_mug';finish=next(folder.rglob('checks.jsonl')).parent
last=rows(finish/'checks.jsonl')[-1];ctx=last['ctx'];paths=last['image_paths']
directory=DEST/'microwave_no_history';directory.mkdir();client=init(ctx,finish,1,directory);client.observe_capture(cap(ctx,paths))
try:
    value=client.analyze(ctx,paths);dump(directory/'result.json',value)
    no_history={'case':'microwave_mug','mode':'no_history','passed':value['visual_goal_verdict']['goal_achieved'] is not True,
        'goal':value['visual_goal_verdict']['goal_achieved'],'checks':value['effective_checks']}
except Exception as exc:
    no_history={'case':'microwave_mug','mode':'no_history','passed':False,'error':str(exc)}
results.append(no_history)
requests=len(list(DEST.rglob('call_*_response.json')))
summary={'protocol':PROTOCOL,'complete':all(c['complete'] for c in case_results.values()) and len(results)==5,
    'passed':all(r['passed'] for r in results),'results':results,'cases':case_results,
    'actual_Qwen_requests':requests,'new_environment_actions':0,'old_model_verdicts_imported':0,'subagents':0}
dump(DEST/'summary.json',summary)
print(json.dumps({'complete':summary['complete'],'passed':summary['passed'],'requests':requests,'results':results},ensure_ascii=False),flush=True)
