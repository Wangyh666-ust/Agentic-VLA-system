"""Bounded v7.1 protocol correction checks, distinct from complete v7 timeline."""
import hashlib,json,sys
from pathlib import Path
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
from scene_demo.xvla_subtask_evidence import SubtaskEvidence,PROTOCOL
ROOT=Path('/mnt/d/FYP');OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'2026-10-13-xvla-timeline-trial'
def load(p):return json.loads(Path(p).read_bytes())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,digest in load(OUT/'final_source_manifest.json').items():assert sha(ROOT/path)==digest,path
mode=sys.argv[1];case='basket_independent/basket_two' if mode=='basket' else 'microwave_mug'
finish=next((OLD/case).rglob('process_events.jsonl')).parent
checks=rows(finish/'checks.jsonl');releases=rows(finish/'process_checks.jsonl')
directory=OUT/('v7_1_'+mode);directory.mkdir(exist_ok=False)
store=SubtaskEvidence({'session_id':'old_photo_followup','logical_subtask_id':mode,'attempt_id':'protocol_fix_1'},directory/'evidence')
client=XvlaSubtaskReviewClient(max_calls=2 if mode=='negative' else 4,evidence_store=store)
client.set_check_contract(checks[0]['ctx']);client.audit_dir=directory/'model_requests'
client.remember_identity_images([finish/'sensors'/('cap%06d_%s.png'%(i,v)) for i,v in [(2,'wrist'),(1,'agentview'),(3,'sideview')]],client.required_views)
if mode!='negative':
    for event in releases:
        client.remember_release(event['capture'],event['context'],event['event_id']);client.note_release_status(event['event_id'],'unclassified_protocol_followup')
records=[];triggered=False;error=None
def inspect(record,label):
    ctx=record['ctx'];paths=record['image_paths'];capture={'timestamp':ctx['actuator_measurements']['timestamp'],
        'robot':{'finger_qpos':ctx['actuator_measurements']['finger_qpos']},
        'frames':[{'view':f['view'],'rgb_path':p} for f,p in zip(ctx['sensor_frames'],paths)]}
    client.observe_capture(capture);meta=client.analyze(ctx,paths);dump(directory/(label+'.json'),meta)
    value={'label':label,'time':capture['timestamp'],'goal':meta['visual_goal_verdict']['goal_achieved'],
        'checks':meta['effective_checks'],'proofs':meta['process_evidence'],'actual_Qwen_calls':client._calls}
    records.append(value);dump(directory/'progress.json',records);print(json.dumps(value,ensure_ascii=False),flush=True)
    return value
try:
    if mode=='basket':
        inspect(checks[1],'historical_probe')
        event_id=releases[0]['event_id'];cid=store.events[event_id]['capture_id']
        triggered=any('alphabet_soup' in p['target_id'] and p['capture_id']==cid and p['kind']=='historical_review'
            and p['observed_at']<p['reviewed_at'] and p['status']=='confirmed' for p in client.process_evidence_snapshot())
        if triggered:inspect(checks[-1],'occlusion_followup')
    else:inspect(checks[-1],'final_comparison' if mode=='microwave' else 'no_history')
except Exception as exc:
    error={'type':type(exc).__name__,'message':str(exc)};dump(directory/'error.json',error)
    print(json.dumps(error),flush=True)
dump(directory/'summary.json',{'protocol':PROTOCOL,'actual_Qwen_calls':client._calls,'error':error,'records':records,
    'historical_soup_triggered':triggered,'stopped_by_rule':mode=='basket' and not triggered,
    'passed':error is None and bool(records) and (records[-1]['goal'] is not True if mode=='negative' else records[-1]['goal'] is True and (mode!='basket' or triggered)),
    'complete_timeline_replay':False,'new_environment_actions':0,'old_answers_imported':0})
