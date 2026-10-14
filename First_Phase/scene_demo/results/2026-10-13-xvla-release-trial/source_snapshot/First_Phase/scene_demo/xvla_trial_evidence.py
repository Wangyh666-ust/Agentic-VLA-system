"""Read-only post-run gate for the explicitly authorized microwave-first trial."""
from __future__ import annotations
import hashlib
import json
import re
import struct
from pathlib import Path

NATIVE = 'xvla_microwave_place_close'
IN = 'in|white_yellow_mug_1|microwave_1_heating_region'
CLOSE = 'close|microwave_1'
VIEWS = ('wrist','agentview','sideview')

def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def _lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _goals(verdict):
    return {item['id']:item for item in verdict['checks']}

def _png(path, dimensions):
    raw=Path(path).read_bytes()
    return (raw[:8]==b'\x89PNG\r\n\x1a\n' and raw[12:16]==b'IHDR'
            and struct.unpack('>II',raw[16:24])==dimensions and raw[24]==8 and raw[25]==2)

def _response(path):
    content=_json(path)['choices'][0]['message']['content']
    if isinstance(content,list):
        content=''.join(x.get('text','') for x in content)
    match=re.search(r'\{.*\}',content,re.S)
    return json.loads(match.group(0) if match else content)

def _image_contract(capture, context, images):
    timestamp=capture['timestamp']
    frames=capture.get('frames',[])+capture.get('verification_frames',[])
    selected=[]
    for view in VIEWS:
        matching=[f for f in frames if f.get('view')==view]
        assert matching
        assert all(f.get('timestamp',timestamp)==timestamp for f in matching)
        selected.append(matching[0])
    paths=[str(f['rgb_path']) for f in selected]
    assert len(set(paths))==3
    assert context['sensor_frames']==[{'frame':i,'view':view,'timestamp':timestamp} for i,view in enumerate(VIEWS)]
    encoder=context['actuator_measurements']
    assert encoder['timestamp']==timestamp
    assert encoder['finger_qpos']==capture['robot']['finger_qpos']
    assert len(images)==3
    for i,(frame,image) in enumerate(zip(selected,images)):
        assert str(image.get('path',image.get('rgb_path')))==paths[i]
        assert image['sha256']==_sha(paths[i])
        assert _png(paths[i],(512,512) if i==2 else (256,256))
    return paths

def _request_contract(directory, metadata, context, paths, raw_visual):
    index=metadata['call_index']
    assert type(index) is int and index>0 and metadata['http_status']==200
    request_path=directory/('call_%02d_request.json'%index)
    response_path=directory/('call_%02d_response.json'%index)
    request=_json(request_path)
    assert request['http_status']==200 and request['context']==context
    assert re.fullmatch('[0-9a-f]{64}',metadata['request_sha256'])
    assert request['request_sha256']==metadata['request_sha256']
    assert metadata['response_sha256']==_sha(response_path)
    assert [image['path'] for image in request['images']]==paths
    assert all(image['sha256']==_sha(image['path']) for image in request['images'])
    assert _goals(_response(response_path))==_goals(raw_visual)
    return {'request':str(request_path),'response':str(response_path),'call_index':index,
            'request_sha256':metadata['request_sha256'],'response_sha256':metadata['response_sha256']}

def assess_microwave_reuse(case_dir: Path) -> dict:
    result={'gate_ok':False,'checks':{},'counts':{},'source_paths':{},
            'selected_event':None,'reuse_check':None,'reason':'unverified: required actual evidence missing'}
    try:
        case_dir=Path(case_dir)
        case_result=_json(case_dir/'result.json')
        pilots=list(case_dir.rglob('pilot_result.json'))
        assert len(pilots)==1, 'expected one native job'
        pilot_path=pilots[0];pilot=_json(pilot_path);job_dir=pilot_path.parent;finish=job_dir/'finish'
        native_paths=list(case_dir.rglob('native_steps.jsonl'))
        assert len(native_paths)==1, 'expected one native session trace'
        native=_lines(native_paths[0]);events=_lines(finish/'process_events.jsonl')
        classified=_lines(finish/'process_checks.jsonl');current=_lines(finish/'checks.jsonl');job_events=_lines(job_dir/'events.jsonl')
        result['source_paths']={'case_result':str(case_dir/'result.json'),'pilot':str(pilot_path),
            'native_steps':str(native_paths[0]),'job_events':str(job_dir/'events.jsonl'),
            'process_events':str(finish/'process_events.jsonl'),'process_checks':str(finish/'process_checks.jsonl'),
            'checks':str(finish/'checks.jsonl'),'model_requests':str(finish/'model_requests')}
        result['checks'].update(unique_native_skill=pilot.get('capability')==NATIVE and case_result.get('plan',{}).get('capability_ids')==[NATIVE],
            actual_vla=pilot.get('source_counts',{}).get('vla',0)>0 and pilot.get('vla_calls',0)>0,
            complete=case_result.get('complete') is True,
            original_handoff=pilot.get('handoff_evidence',{}).get('ok') is True,
            independent_five=pilot.get('finish_status',{}).get('independent_streak',0)>=5)
        result['counts']={'vla_actions':pilot.get('source_counts',{}).get('vla',0),'vla_calls':pilot.get('vla_calls',0),
            'release_events':len(events),'classified_events':sum(r.get('status')=='classified' for r in classified),'reuse_checks':0}
        errors=[]
        for event in events:
            try:
                event_id=event['event_id'];capture=event['capture'];time=capture['timestamp']
                trace=[r for r in native if r.get('source')=='vla' and r['sim_time']<=time]
                assert trace and trace[-1]['sim_time']==time
                step=trace[-1]['counts']['vla']
                declared=next(r['declared_predicates'] for r in job_events if r['step']==step)
                assert declared[IN] is True and declared[CLOSE] is False
                assert trace[-1]['finger_qpos']==capture['robot']['finger_qpos']
                pc=next(r for r in classified if r.get('event_id')==event_id and r.get('status')=='classified')
                assert pc['capture']==capture
                context=pc['context'];meta=pc['metadata'];raw=meta['raw_visual_goal_verdict']
                assert meta['observation_role']=='release_snapshot' and meta['event_id']==event_id
                assert _goals(raw)['goal_0']['achieved'] is True
                home=context['actuator_measurements']['standard_open_finger_qpos']
                errors_from_open=[max(abs(a-b) for a,b in zip(r['finger_qpos'],home)) for r in trace]
                assert len(errors_from_open)>=3 and all(e<=.002 for e in errors_from_open[-2:])
                assert any(e>=.004 for e in errors_from_open[:-2])
                paths=_image_contract(capture,context,event['images'])
                release_request=_request_contract(finish/'model_requests',meta,context,paths,raw)
                ledger=next(e for e in meta['process_evidence'] if e['id']=='goal_0')
                assert ledger['event_id']==event_id and ledger['timestamp']==time
                assert ledger['provenance']['images']==[{'path':p,'sha256':_sha(p)} for p in paths]
                assert ledger['provenance']['request_sha256']==meta['request_sha256']
                assert ledger['provenance']['response_sha256']==meta['response_sha256']
                assert ledger['provenance']['call_index']==meta['call_index']
                for check in current:
                    try:
                        ctx=check['ctx'];later=ctx['actuator_measurements']['timestamp'];assert later>time
                        cm=check['initial_check']['metadata'];rv=cm['raw_visual_goal_verdict']
                        assert _goals(rv)['goal_0']['achieved'] is None and _goals(rv)['goal_1']['achieved'] is True
                        effective={e['id']:e for e in cm['effective_checks']}
                        assert effective['goal_0']['achieved'] is True and effective['goal_0']['history_event_id']==event_id
                        assert effective['goal_1']['achieved'] is True and cm['visual_goal_verdict']['goal_achieved'] is True
                        assert _goals(cm['visual_goal_verdict'])==effective
                        assert any(r['sim_time']==later and r.get('native_success') is True for r in native)
                        late_images=check['image_paths']
                        request=_json(finish/'model_requests'/('call_%02d_request.json'%cm['call_index']))
                        synthetic_capture={'timestamp':later,'robot':check['robot'],
                            'frames':[{'view':view,'rgb_path':path,'timestamp':later} for view,path in zip(VIEWS,late_images)]}
                        paths_later=_image_contract(synthetic_capture,ctx,request['images'])
                        current_request=_request_contract(finish/'model_requests',cm,ctx,paths_later,rv)
                        assert current_request['call_index']>release_request['call_index']
                        remembered=next(e for e in cm['process_evidence'] if e['id']=='goal_0')
                        assert remembered==ledger
                        result['counts']['reuse_checks']+=1
                        result['checks'].update(vla_release_with_in_true_door_open=True,measured_transition=True,
                            synchronized_three_images=True,actual_release_request_response=True,release_classified=True,
                            matching_placement_ledger=True,current_raw_unknown_door_closed=True,actual_current_request_response=True,
                            history_reused=True,independent_native_success=True)
                        result['selected_event']={'event_id':event_id,'timestamp':time,'vla_step':step,
                            'trace_finger_errors':errors_from_open,'event':event,'request':release_request}
                        result['reuse_check']={'timestamp':later,'check':check,'request':current_request}
                        result['gate_ok']=all(result['checks'].values())
                        result['reason']='passed actual release/classification/reuse/full-task gate' if result['gate_ok'] else 'unverified: full-task/handoff prerequisites failed'
                        return result
                    except (KeyError,AssertionError,ValueError,TypeError,StopIteration,OSError) as error:
                        errors.append('current evidence: '+str(error or type(error).__name__))
            except (KeyError,AssertionError,ValueError,TypeError,StopIteration,OSError) as error:
                errors.append('release evidence: '+str(error or type(error).__name__))
        result['checks']['history_reused']=False
        result['reason']='unverified: no qualifying actual release and history reuse; '+ '; '.join(errors[:6])
    except Exception as error:
        result['reason']='unverified: '+type(error).__name__+': '+str(error)
    return result
