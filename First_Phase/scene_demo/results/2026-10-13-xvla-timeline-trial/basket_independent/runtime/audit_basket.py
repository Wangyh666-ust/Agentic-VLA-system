"""Read-only packet/trace audit. No model request or environment action."""
import base64
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path('/mnt/d/FYP');OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial';OUT=OUT/'basket_independent';CASE=OUT/'basket_two'
def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
job=next(CASE.rglob('pilot_result.json')).parent;finish=job/'finish';pilot=load(job/'pilot_result.json');case=load(CASE/'result.json')
native=rows(next(CASE.rglob('native_steps.jsonl')));events=rows(job/'events.jsonl')
releases=rows(finish/'process_events.jsonl');classified=rows(finish/'process_checks.jsonl');checks=rows(finish/'checks.jsonl')
metadata={r['metadata']['call_index']:r['metadata'] for r in classified if r['status']=='classified'}
metadata.update({r['initial_check']['metadata']['call_index']:r['initial_check']['metadata'] for r in checks})
packets=[]
for path in sorted((finish/'model_requests').glob('call_*_request.json')):
    index=int(path.stem.split('_')[1]);request=load(path);meta=metadata[index]
    frames=request['context']['sensor_frames'];images=request['images'];assert len(frames)==len(images)
    content=[{'type':'text','text':request['prompt']}];image_rows=[]
    for frame,image in zip(frames,images):
        data=Path(image['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==image['sha256']
        size=struct.unpack('>II',data[16:24]);assert size==((512,512) if frame['view']=='sideview' else (256,256))
        assert data[:8]==b'\x89PNG\r\n\x1a\n' and data[24:26]==bytes([8,2])
        content.append({'type':'text','text':f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"})
        content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode('ascii')}})
        image_rows.append({**frame,**image,'size':size})
    body={'model':'qwen3-vl-plus','temperature':0,'max_tokens':1600,'messages':[{'role':'user','content':content}]}
    body.update(request['request_options']);assert hashlib.sha256(json.dumps(body).encode()).hexdigest()==request['request_sha256']==meta['request_sha256']
    response_path=finish/'model_requests'/('call_%02d_response.json'%index);assert sha(response_path)==meta['response_sha256']
    response=load(response_path);raw=json.loads(response['choices'][0]['message']['content'])
    assert raw['checks']==meta['raw_visual_goal_verdict']['checks'] and request['http_status']==meta['http_status']==200
    timeline=request['context'].get('visual_timeline',{})
    if timeline:
        current=request['context']['actuator_measurements'];indices=timeline['current_frame_indices']
        assert len(indices)==3 and all(frames[i]['timestamp']==current['timestamp'] for i in indices)
        for old in timeline['process_history']:
            event=next(e for e in releases if e['event_id']==old['event_id']);pc=next(c for c in classified if c['event_id']==old['event_id'])
            assert old['timestamp']==event['capture']['timestamp']==old['actuator_measurements']['timestamp']
            assert old['actuator_measurements']['finger_qpos']==event['capture']['robot']['finger_qpos']
            assert old['raw_checks']==pc['metadata']['raw_visual_goal_verdict']['checks']
            assert [images[i]['path'] for i in old['frame_indices']]==[im['path'] for im in event['images']]
            assert all(frames[i]['timestamp']==old['timestamp'] for i in old['frame_indices'])
        assert all(old['timestamp']<=current['timestamp'] for old in timeline['previous_check_states'])
    packets.append({'call_index':index,'http_status':200,'body_sha_matches':True,'response_sha_matches':True,
        'image_count':len(images),'images':image_rows,'process_groups':len(timeline.get('process_history',[])),
        'previous_state_count':len(timeline.get('previous_check_states',[])),
        'role':meta.get('observation_role','current_and_process'),'raw_checks':raw['checks'],
        'effective_checks':meta['effective_checks'],'usage':response.get('usage')})
release_audits=[]
for event in releases:
    pc=next(c for c in classified if c['event_id']==event['event_id']);time=event['capture']['timestamp']
    trace=[r for r in native if r['source']=='vla' and r['sim_time']<=time];assert trace[-1]['sim_time']==time
    assert trace[-1]['finger_qpos']==event['capture']['robot']['finger_qpos']
    reference=pc['context']['actuator_measurements']['standard_open_finger_qpos']
    errors=[max(abs(a-b) for a,b in zip(r['finger_qpos'],reference)) for r in trace]
    assert any(e>=.004 for e in errors[:-2]) and all(e<=.002 for e in errors[-2:])
    step=trace[-1]['counts']['vla'];declared=next(r['declared_predicates'] for r in events if r['step']==step)
    release_audits.append({'event_id':event['event_id'],'vla_step':step,'timestamp':time,
        'declared_predicates':declared,'last_two_open_errors_m':errors[-2:],
        'raw_checks':pc['metadata']['raw_visual_goal_verdict']['checks'],'effective_checks':pc['metadata']['effective_checks'],
        'images':event['images'],'protocol_passed':True})
predicates=events[0]['declared_predicates'].keys()
sources={kind:sum(r['source']==kind for r in native) for kind in ('vla','prepare','settling','finish')}
audit={'case':'basket_two','complete':case['complete'],'plan_state':case['plan']['state'],'ended_reason':pilot['outcome']['reason'],
    'source_actions':sources,'vla_calls':pilot['vla_calls'],'Hermes_invocations':case['agent_result']['hermes_invocations'],
    'Hermes_API_usage':case['agent_result']['usage'],'Qwen_requests':len(packets),'current_checks':len(checks),'release_classifications':len(classified),
    'release_audits':release_audits,'packets':packets,
    'predicate_first_true':{key:next((r['step'] for r in events if r['declared_predicates'][key] is True),None) for key in predicates},
    'predicate_true_counts':{key:sum(r['declared_predicates'][key] is True for r in events) for key in predicates},
    'native_success_first_true':next((r['step'] for r in events if r['native_success']),None),
    'native_success_true_steps':sum(r['native_success'] for r in events),'last_native_step':events[-1],
    'current_checks_vs_native':[{'step':next(r['counts']['vla'] for r in native if r['sim_time']==c['ctx']['actuator_measurements']['timestamp']),
        'native_success':next(r['native_success'] for r in native if r['sim_time']==c['ctx']['actuator_measurements']['timestamp']),
        'model_checks':c['initial_check']['metadata']['effective_checks']} for c in checks],
    'history_references':sum(bool(row.get('evidence_event_ids')) for c in checks for row in c['initial_check']['metadata']['effective_checks']),
    'last_effective_checks':checks[-1]['initial_check']['metadata']['effective_checks'],
    'last_current_images':checks[-1]['image_paths'],'finish_status':pilot.get('finish_status'),'independent_streak':pilot.get('independent_streak'),'handoff_evidence':case.get('handoff_evidence'),'agent_result':case.get('agent_result'),'gate_ok':case['complete'],
    'diagnostic':'输入协议、视觉判断、仿真目标和原收尾分开记录；成功不能单凭当前可见就称历史复用效果已验证。',
    'audit_model_calls':0,'audit_environment_actions':0}
dump(OUT/'audit.json',audit)
print(json.dumps({k:audit[k] for k in ('complete','ended_reason','source_actions','vla_calls','Qwen_requests','current_checks','release_classifications','native_success_first_true','native_success_true_steps','history_references','gate_ok')},ensure_ascii=False))
