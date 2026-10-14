"""Replay identical v5 responses in sequence; request only the missing last frame."""
import base64,copy,hashlib,json,shutil
from pathlib import Path
from scene_demo.xvla_evidence_memory import XvlaEvidenceMemoryClient
from scene_demo.finish_v1 import vision
from scene_demo.xvla_process_evidence import capture_frames
ROOT=Path('/mnt/d/FYP');OUT=Path(__file__).resolve().parent;DEST=OUT/'continuation';DEST.mkdir(exist_ok=False)
def load(p):return json.loads(Path(p).read_bytes())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={}
for name in ['xvla_evidence_memory.py','xvla_finish.py','xvla_monitor.py','xvla_service.py','xvla_pilot.py']:
    p=ROOT/'First_Phase/scene_demo'/name;d=DEST/'source'/name;d.parent.mkdir(exist_ok=True);shutil.copyfile(p,d);manifest[p.relative_to(ROOT).as_posix()]=sha(p)
shutil.copyfile(__file__,DEST/'source/complete_replay.py');dump(DEST/'source_manifest.json',manifest)
OLD=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/microwave_mug'
finish=next(OLD.rglob('process_checks.jsonl')).parent
checks=rows(finish/'checks.jsonl');releases=rows(finish/'process_checks.jsonl')
client=XvlaEvidenceMemoryClient(max_calls=8);client.set_check_contract(checks[0]['ctx']);client.audit_dir=DEST/'model_requests'
client.remember_identity_images([finish/'sensors'/('cap%06d_%s.png'%(i,v)) for i,v in [(2,'wrist'),(1,'agentview'),(3,'sideview')]],client.required_views)
original_request=client._request;reused=[];actual=[]
def request(ctx,paths,prompt):
    index=client._calls+1
    if index==8:
        assert len(reused)==7 and not actual
        result=original_request(ctx,paths,prompt);actual.append(index);return result
    assert 1<=index<=7
    source=OUT/'attempt_02/microwave_mug/model_requests'/('call_%02d_request.json'%index)
    saved=load(source);response=source.with_name(source.name.replace('_request','_response'))
    content=[{'type':'text','text':prompt}]
    for frame,url in zip(ctx['sensor_frames'],client._load_images(paths)):
        content.extend([{'type':'text','text':f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"},
            {'type':'image_url','image_url':{'url':url}}])
    body={'model':client.model,'temperature':0,'max_tokens':1600,'messages':[{'role':'user','content':content}]};body.update(client.request_options)
    digest=hashlib.sha256(json.dumps(body).encode()).hexdigest()
    assert digest==saved['request_sha256'],'Replay prompt/images/history diverged from original request'
    raw=load(response);verdict=client.parse_verdict(raw['choices'][0]['message']['content']);client._calls+=1
    reused.append({'call_index':index,'request_source':str(source.relative_to(ROOT)),'response_source':str(response.relative_to(ROOT)),
        'request_sha256':digest,'response_sha256':sha(response),'identical_request_verified':True})
    return {'verdict':verdict,'call_index':index,'request_sha256':digest,'response_sha256':sha(response),
        'usage':raw['usage'],'replayed_existing_v5_response':True}
client._request=request
def cap(ctx,paths):return {'timestamp':ctx['actuator_measurements']['timestamp'],
    'robot':{'finger_qpos':ctx['actuator_measurements']['finger_qpos']},
    'frames':[{'view':f['view'],'rgb_path':p} for f,p in zip(ctx['sensor_frames'],paths)]}
pending=[];records=[]
stream=[(r['ctx']['actuator_measurements']['timestamp'],'check',r) for r in checks]
stream += [(r['capture']['timestamp'],'release',r) for r in releases]
for time,kind,r in sorted(stream,key=lambda r:r[0]):
    if kind=='release':client.remember_release(r['capture'],r['context'],r['event_id']);pending.append(r);continue
    ctx=r['ctx'];paths=r['image_paths'];capture=cap(ctx,paths);client.remember_observation(capture,ctx)
    def save(kind,meta):
        dump(DEST/('%02d_%s.json'%(len(records)+1,kind)),meta)
        records.append({'kind':kind,'time':meta['input_context']['actuator_measurements']['timestamp'],
            'goal':meta['visual_goal_verdict']['goal_achieved'],'checks':meta['effective_checks'],
            'call_index':meta['call_index'],'replayed_existing_v5_response':meta.get('replayed_existing_v5_response',False)})
    for e in pending:
        client.observe_capture(e['capture']);epaths=[str(f['rgb_path']) for f in capture_frames(e['capture'])]
        save('release',client.analyze_release(e['context'],epaths,e['event_id']))
    pending.clear();client.observe_capture(capture);save('current',client.analyze(ctx,paths))
assert len(actual)==1 and len(reused)==7 and len(records)==8
dump(DEST/'summary.json',{'complete':True,'final':records[-1],'records':records,'actual_Qwen_requests':1,
    'replayed_existing_v5_responses':7,'reused_sources':reused,'new_environment_actions':0,'old_v3_model_verdicts_imported':0,
    'qualification':False,'qualification_reason':'Existing open-door negative still incorrectly accepted; basket process proof incomplete.',
    'final_memory':client.process_evidence_snapshot()})
print(json.dumps({'complete':True,'actual_Qwen_requests':1,'replayed_existing_v5_responses':7,'final':records[-1]},ensure_ascii=False))
