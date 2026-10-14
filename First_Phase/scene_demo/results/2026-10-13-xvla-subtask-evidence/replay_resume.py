"""Actual Qwen on retained original images; no simulator or VLA execution."""
import hashlib,json,shutil,sys
from pathlib import Path
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
from scene_demo.xvla_subtask_evidence import SubtaskEvidence,PROTOCOL
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
        matches=[r for r in traces if r['source']=='vla' and abs(r['sim_time']-time)<1e-8];assert len(matches)==1
        step=matches[0]['counts']['vla'];state=by_step[step]
        return {'vla_step':step,'native_success':state['native_success'],'predicates':state['declared_predicates']}
    return finish,checks,releases,truth
def init(finish,ctx,directory,budget):
    directory.mkdir(exist_ok=False)
    store=SubtaskEvidence({'session_id':'old_photo_replay','logical_subtask_id':mode,'attempt_id':'1'},directory/'evidence')
    client=XvlaSubtaskReviewClient(max_calls=budget,evidence_store=store)
    client.set_check_contract(ctx);client.audit_dir=directory/'model_requests'
    client.remember_identity_images([finish/'sensors'/('cap%06d_%s.png'%(i,v)) for i,v in [(2,'wrist'),(1,'agentview'),(3,'sideview')]],client.required_views)
    original=client._request
    cached=sorted((OUT/mode/'model_requests').glob('call_*_request.json')) if directory.name!=mode else []
    client.reused_prefix=0
    def bounded(*args,**kwargs):
        if client.reused_prefix<len(cached):
            request_path=cached[client.reused_prefix];request=load(request_path)
            context,paths,prompt=args
            assert request['prompt']==prompt and request['context']==context
            assert [str(p) for p in paths]==[i['path'] for i in request['images']]
            for item in request['images']:assert sha(item['path'])==item['sha256']
            response_path=request_path.with_name(request_path.name.replace('_request','_response'));response=load(response_path)
            client._calls+=1;client.reused_prefix+=1
            return {'verdict':client.parse_verdict(response['choices'][0]['message']['content']),
                'usage':response['usage'],'model':response['model'],'http_status':request['http_status'],
                'request_sha256':request['request_sha256'],'response_sha256':sha(response_path),
                'call_index':client._calls}

        if len(list(OUT.glob('*/model_requests/call_*_request.json')))>=26:raise RuntimeError('global request budget 26 exhausted')
        return original(*args,**kwargs)
    client._request=bounded
    return client
def save(directory,records,kind,meta,truth,client):
    path=directory/('%02d_%s.json'%(len(records)+1,kind));dump(path,meta)
    cid=meta['current_capture_id'];now=client.store.observations[cid]['captured_at']
    value={'kind':kind,'time':now,'metadata':str(path.relative_to(ROOT)),
        'goal':meta['visual_goal_verdict']['goal_achieved'],'checks':meta['effective_checks'],
        'proofs':[{k:p[k] for k in ('proof_id','target_id','capture_id','event_id','observed_at','reviewed_at','status','kind','request_sha256')} for p in meta['process_evidence']],
        'actual_Qwen_calls':client._calls,'reused':meta.get('reused_same_capture',False),
        'evaluation_only_truth_not_sent':truth(now)}
    records.append(value);dump(directory/'progress.json',records)
    print(json.dumps({'case':directory.name,**value},ensure_ascii=False),flush=True)
    return value
def soup_recovered(client,first_release,late=False):
    expected=client.store.events[first_release]['capture_id']
    return any('alphabet_soup' in p['target_id'] and p['capture_id']==expected and p['event_id']==first_release
        and p['status']=='confirmed' and (not late or p['kind']=='historical_review' and p['reviewed_at']>p['observed_at'])
        for p in client.process_evidence_snapshot())
def failure(directory,client,exc):
    value={'type':type(exc).__name__,'message':str(exc),'actual_Qwen_calls':client._calls}
    dump(directory/'error.json',value);print(json.dumps({'case':directory.name,'error':value}),flush=True);return value

mode=sys.argv[1]
if mode=='freeze':
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=False);manifest={}
    names=['xvla_subtask_evidence.py','xvla_subtask_review.py','xvla_finish.py','xvla_monitor.py','xvla_service.py','xvla_pilot.py',
        'tests/test_xvla_subtask_review.py','tests/test_xvla_integration.py','tests/test_xvla_event_checker.py','plans/20261013-xvla-subtask-evidence-execution.json']
    for path in [ROOT/'First_Phase/scene_demo'/n for n in names]+[Path(__file__).resolve()]:
        dest=snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        manifest[path.relative_to(ROOT).as_posix()]=sha(path)
    dump(OUT/'source_manifest.json',manifest);print(json.dumps({'frozen':len(manifest),'protocol':PROTOCOL}));sys.exit(0)
for path,digest in load(OUT/'source_manifest.json').items():assert sha(ROOT/path)==digest,path

case='microwave_mug' if mode in ('microwave_mug','no_history') else 'basket_two'
finish,checks,releases,truth=dataset(case);directory=OUT/(mode+'_resume' if mode in ('microwave_mug','basket_two') else mode)
for event in releases:
    ordered=capture_frames(event['capture'])
    assert [f['view'] for f in ordered]==list(XvlaSubtaskReviewClient.required_views)
    assert len(ordered)==3
for path,digest in load(OUT/'resume_manifest.json').items():assert sha(ROOT/path)==digest,path
budget={'probe':4,'microwave_mug':10,'basket_two':10,'no_history':2}[mode]
client=init(finish,checks[0]['ctx'],directory,budget);records=[];error=None;triggered=False;stopped=False
try:
    if mode=='probe':
        # Explicit unclassified-event control: no imported old verdicts and no invented Unknown answers.
        for event in releases:
            client.remember_release(event['capture'],event['context'],event['event_id'])
            client.note_release_status(event['event_id'],'unclassified_control')
        record=checks[1];client.observe_capture(cap(record['ctx'],record['image_paths']))
        meta=client.analyze(record['ctx'],record['image_paths']);save(directory,records,'historical_probe',meta,truth,client)
        triggered=soup_recovered(client,releases[0]['event_id'],late=True)
        if triggered:
            record=checks[-1];client.observe_capture(cap(record['ctx'],record['image_paths']))
            meta=client.analyze(record['ctx'],record['image_paths']);save(directory,records,'occlusion_followup',meta,truth,client)
        else:stopped=True
    elif mode in ('microwave_mug','basket_two'):
        stream=[(r['ctx']['actuator_measurements']['timestamp'],'check',r) for r in checks]
        stream += [(r['capture']['timestamp'],'release',r) for r in releases];pending=[]
        for time,kind,record in sorted(stream,key=lambda r:r[0]):
            if kind=='release':
                client.remember_release(record['capture'],record['context'],record['event_id']);pending.append(record);continue
            ctx=record['ctx'];paths=record['image_paths'];capture=cap(ctx,paths);client.remember_observation(capture,ctx)
            for event in pending:
                client.observe_capture(event['capture']);epaths=[str(f['rgb_path']) for f in capture_frames(event['capture'])]
                meta=client.analyze_release(event['context'],epaths,event['event_id']);save(directory,records,'release',meta,truth,client)
            pending.clear();client.observe_capture(capture)
            meta=client.analyze(ctx,paths);value=save(directory,records,'current',meta,truth,client)
            if mode=='basket_two' and value['evaluation_only_truth_not_sent']['vla_step']==300:
                triggered=soup_recovered(client,releases[0]['event_id'])
                if not triggered:stopped=True;break
    else:
        record=checks[-1];client.observe_capture(cap(record['ctx'],record['image_paths']))
        meta=client.analyze(record['ctx'],record['image_paths']);save(directory,records,'no_history',meta,truth,client)
except Exception as exc:error=failure(directory,client,exc)
finals=[r for r in records if r['kind']=='current'];final=records[-1] if records else None
false_positives=[r['time'] for r in records if r['goal'] is True and not r['evaluation_only_truth_not_sent']['native_success']]
complete=error is None and not stopped and (len(finals)==len(checks) if mode in ('microwave_mug','basket_two') else True)
passed=complete and final is not None and (final['goal'] is not True if mode=='no_history' else final['goal'] is True) and not false_positives
dump(directory/'summary.json',{'complete':complete,'passed':passed,'historical_soup_triggered':triggered,'stopped_by_rule':stopped,
    'final_goal':None if final is None else final['goal'],'actual_Qwen_calls':client._calls-client.reused_prefix,'reused_same_campaign_requests':client.reused_prefix,'logical_checks_including_reuse':client._calls,'false_positives':false_positives,
    'error':error,'records':records,'old_answers_imported':0,'new_environment_actions':0,
    'classification_missing_control':mode=='probe'})
