"""File-only evidence fixtures: no fake fixture is a physical experiment."""
import copy
import hashlib
import json
import struct
import zlib
from pathlib import Path
import pytest
from scene_demo.xvla_trial_evidence import assess_microwave_reuse, IN, CLOSE
from scene_demo.xvla_pilot import verify_case_prerequisite

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')

def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

def png(path,size):
    def chunk(tag,data):
        return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data)&0xffffffff)
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',size,size,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\x80'*(size*3))*size))+chunk(b'IEND',b'')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)

def checks(a,b):
    return {'checks':[{'id':'goal_0','achieved':a,'reason':'fixture placement'}, {'id':'goal_1','achieved':b,'reason':'fixture door'}],'goal_achieved':a is True and b is True,'reason':'fixture'}

@pytest.fixture
def actual_structure(tmp_path):
    case=tmp_path/'microwave_mug';job=case/'runs/session/jobs/j';finish=job/'finish';request_dir=finish/'model_requests'
    native_path=case/'runs/session/native/native_steps.jsonl'
    def observation(time):
        frames=[];images=[]
        for i,view in enumerate(('wrist','agentview','sideview')):
            p=finish/('t%s_%s.png'%(time,view));png(p,512 if i==2 else 256)
            frames.append({'view':view,'rgb_path':str(p),'timestamp':time});images.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        capture={'timestamp':time,'frames':frames,'robot':{'finger_qpos':[.04,.04],'arm_qvel':[0.]*7}}
        ctx={'sensor_frames':[{'frame':i,'view':view,'timestamp':time} for i,view in enumerate(('wrist','agentview','sideview'))],
             'actuator_measurements':{'timestamp':time,'finger_qpos':[.04,.04],'standard_open_finger_qpos':[.04,.04]}}
        return capture,ctx,images
    release,rc,ri=observation(3.)
    later,cc,ci=observation(4.)
    def request(index,ctx,images,raw):
        response={'choices':[{'message':{'content':json.dumps(raw)}}]}
        dump(request_dir/('call_%02d_response.json'%index),response)
        digest=hashlib.sha256((request_dir/('call_%02d_response.json'%index)).read_bytes()).hexdigest()
        dump(request_dir/('call_%02d_request.json'%index),{'context':ctx,'images':images,'http_status':200,'request_sha256':'a'*64})
        return {'call_index':index,'http_status':200,'request_sha256':'a'*64,'response_sha256':digest,'raw_visual_goal_verdict':raw}
    rm=request(1,rc,ri,checks(True,False))
    ledger={'id':'goal_0','event_id':'release_0001','timestamp':3.,'provenance':{'images':ri,'request_sha256':rm['request_sha256'],'response_sha256':rm['response_sha256'],'call_index':1}}
    rm.update(observation_role='release_snapshot',event_id='release_0001',process_evidence=[ledger])
    cm=request(2,cc,ci,checks(None,True));effective=checks(True,True)['checks'];effective[0]['history_event_id']='release_0001'
    cm.update(effective_checks=effective,visual_goal_verdict={'checks':effective,'goal_achieved':True},process_evidence=[ledger])
    dump(case/'result.json',{'complete':True,'plan':{'capability_ids':['xvla_microwave_place_close']}})
    dump(job/'pilot_result.json',{'capability':'xvla_microwave_place_close','source_counts':{'vla':4},'vla_calls':1,'handoff_evidence':{'ok':True},'finish_status':{'independent_streak':5}})
    lines(native_path,[{'source':'vla','sim_time':float(i),'counts':{'vla':i},'finger_qpos':([.01,.01] if i==1 else [.04,.04]),'native_success':i==4} for i in range(1,5)])
    lines(job/'events.jsonl',[{'step':i,'declared_predicates':{IN:i>=3,CLOSE:i==4}} for i in range(1,5)])
    event={'event_id':'release_0001','capture':release,'images':[dict(img,view=view) for img,view in zip(ri,('wrist','agentview','sideview'))]}
    lines(finish/'process_events.jsonl',[event]);lines(finish/'process_checks.jsonl',[{'event_id':'release_0001','status':'classified','capture':release,'context':rc,'metadata':rm}])
    current={'ctx':cc,'robot':later['robot'],'image_paths':[i['path'] for i in ci],'initial_check':{'metadata':cm}}
    lines(finish/'checks.jsonl',[current])
    return case


def test_all_real_source_structures_required(actual_structure):
    value=assess_microwave_reuse(actual_structure)
    assert value['gate_ok'] is True, value
    assert all(value['checks'].values())
    assert value['counts']['reuse_checks']==1
    scope={'configuration':{'case_order':['microwave_mug','basket_two'],'case_prerequisites':{'microwave_mug':None,'basket_two':'microwave_mug'}}}
    verify_case_prerequisite('basket_two',actual_structure.parent,scope)

@pytest.mark.parametrize('failure',['source','time','finger','sha','png','no_reuse','no_classification','no_response','not_complete','native_false'])
def test_refuse_bad_or_missing_actual_provenance(actual_structure,failure):
    case=actual_structure;job=next(case.rglob('pilot_result.json')).parent;finish=job/'finish'
    if failure in ('source','time','finger','native_false'):
        path=next(case.rglob('native_steps.jsonl'));rows=[json.loads(l) for l in path.read_text().splitlines()]
        if failure=='source':rows[2]['source']='finish'
        elif failure=='time':rows[2]['sim_time']=3.1
        elif failure=='finger':rows[2]['finger_qpos']=[.01,.01]
        else:rows[-1]['native_success']=False
        lines(path,rows)
    elif failure in ('sha','png'):
        path=finish/'process_events.jsonl';event=json.loads(path.read_text())
        if failure=='sha':event['images'][0]['sha256']='0'*64
        else:Path(event['images'][0]['path']).write_bytes(b'not png')
        lines(path,[event])
    elif failure=='no_reuse':
        path=finish/'checks.jsonl';row=json.loads(path.read_text());row['initial_check']['metadata']['effective_checks'][0].pop('history_event_id');lines(path,[row])
    elif failure=='no_classification':(finish/'process_checks.jsonl').write_text('')
    elif failure=='no_response':(finish/'model_requests/call_01_response.json').unlink()
    else:dump(case/'result.json',{'complete':False})
    assert assess_microwave_reuse(case)['gate_ok'] is False


def test_scope_first_micro_only_and_original_gate_preserved(tmp_path):
    scope={'configuration':{'case_order':['microwave_mug','basket_two'],'case_prerequisites':{'microwave_mug':None,'basket_two':'microwave_mug'}}}
    verify_case_prerequisite('microwave_mug',tmp_path,scope)
    with pytest.raises(RuntimeError,match='microwave_process_evidence_not_passed'):
        verify_case_prerequisite('basket_two',tmp_path,scope)
    with pytest.raises(RuntimeError,match='case_outside_explicit_trial_scope'):
        verify_case_prerequisite('wine_bowl_stove',tmp_path,scope)
    with pytest.raises(RuntimeError,match='invalid_explicit_trial_scope'):
        verify_case_prerequisite('microwave_mug',tmp_path,{'configuration':{'case_order':['basket_two','microwave_mug']}})
    verify_case_prerequisite('wine_bowl_stove',tmp_path,{})
    dump(tmp_path/'wine_bowl_stove/result.json',{'complete':False})
    with pytest.raises(RuntimeError,match='first_chain_not_complete'):
        verify_case_prerequisite('basket_two',tmp_path,{})


@pytest.fixture
def joint_structure(actual_structure):
    case=actual_structure;job=next(case.rglob('pilot_result.json')).parent;finish=job/'finish';directory=finish/'model_requests'
    process=json.loads((finish/'process_checks.jsonl').read_text());current=json.loads((finish/'checks.jsonl').read_text())
    release_request=json.loads((directory/'call_01_request.json').read_text())
    current_request=json.loads((directory/'call_02_request.json').read_text())
    def judgement(a,b):
        raw=checks(a,b)
        for row in raw['checks']:row.update(target_matches_definition=True,current_evidence=True,evidence_event_ids=[])
        return raw
    release_raw=judgement(False,False)
    dump(directory/'call_01_response.json',{'choices':[{'message':{'content':json.dumps(release_raw)}}]})
    process['metadata'].update(checker='xvla_timeline_skill_v3',raw_visual_goal_verdict=release_raw,
        effective_checks=release_raw['checks'],process_evidence=[],
        response_sha256=hashlib.sha256((directory/'call_01_response.json').read_bytes()).hexdigest())
    context=copy.deepcopy(current['ctx']);context['sensor_frames']=copy.deepcopy(process['context']['sensor_frames'])
    context['sensor_frames'] += [{**f,'frame':f['frame']+3} for f in current['ctx']['sensor_frames']]
    context['visual_timeline']={'current_frame_indices':[3,4,5],
        'process_history':[{'event_id':'release_0001','timestamp':3.,'classification_status':'classified',
            'raw_checks':release_raw['checks'],'effective_checks':release_raw['checks'],
            'actuator_measurements':process['context']['actuator_measurements'],'frame_indices':[0,1,2]}],
        'previous_check_states':[],'history_complete':True,'omitted_release_event_ids':[]}
    current_request.update(context=context,images=release_request['images']+current_request['images'])
    dump(directory/'call_02_request.json',current_request)
    raw=judgement(True,True);raw['checks'][0].update(current_evidence=False,evidence_event_ids=['release_0001'])
    dump(directory/'call_02_response.json',{'choices':[{'message':{'content':json.dumps(raw)}}]})
    effective=copy.deepcopy(raw['checks']);effective[0].update(history_event_id='release_0001',history_event_ids=['release_0001'])
    current['initial_check']['metadata'].update(checker='xvla_timeline_skill_v3',input_context=context,
        raw_visual_goal_verdict=raw,effective_checks=effective,visual_goal_verdict={'checks':effective,'goal_achieved':True},
        process_evidence=[],response_sha256=hashlib.sha256((directory/'call_02_response.json').read_bytes()).hexdigest())
    lines(finish/'process_checks.jsonl',[process]);lines(finish/'checks.jsonl',[current])
    return case


def test_joint_gate_rechecks_negative_process_with_original_images(joint_structure):
    value=assess_microwave_reuse(joint_structure)
    assert value['gate_ok'] is True,value
    assert value['checks']['joint_process_images_and_states_received'] is True
    assert 'matching_placement_ledger' not in value['checks']


@pytest.mark.parametrize('failure',['no_process_images','old_status_omitted','wrong_event','old_door_only','identity_unknown'])
def test_joint_gate_rejects_missing_or_unqualified_sources(joint_structure,failure):
    case=joint_structure;finish=next(case.rglob('pilot_result.json')).parent/'finish'
    path=finish/'checks.jsonl';row=json.loads(path.read_text());meta=row['initial_check']['metadata']
    request_path=finish/'model_requests/call_02_request.json';request=json.loads(request_path.read_text())
    if failure=='no_process_images':request['images']=request['images'][-3:]
    elif failure=='old_status_omitted':meta['input_context']['visual_timeline']['process_history'][0].pop('raw_checks');request['context']=meta['input_context']
    elif failure=='wrong_event':meta['raw_visual_goal_verdict']['checks'][0]['evidence_event_ids']=['wrong']
    elif failure=='old_door_only':meta['raw_visual_goal_verdict']['checks'][1]['current_evidence']=False
    else:meta['raw_visual_goal_verdict']['checks'][0]['target_matches_definition']=None
    dump(request_path,request);lines(path,[row])
    assert assess_microwave_reuse(case)['gate_ok'] is False
