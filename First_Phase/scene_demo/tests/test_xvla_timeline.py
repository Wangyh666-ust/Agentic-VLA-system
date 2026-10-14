"""Temporal input and identity safeguards; no real HTTP or environment actions."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from unittest import mock
import pytest
from scene_demo import catalog
from scene_demo.xvla_finish import XvlaQwenClient, build_prompt, build_skill_context
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_timeline import MAX_INPUT_IMAGES
from scene_demo.tests.test_xvla_process_evidence import capture, context, checks, respond


def sample(tmp_path, cid, time):
    folder=tmp_path/str(time);folder.mkdir(exist_ok=True)
    cap=capture(folder,time=time)
    return cap,context(cid,cap),[f['rgb_path'] for f in cap['frames']]


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_all_skills_receive_failed_process_photos_and_previous_states(tmp_path,cid):
    client=XvlaQwenClient(env_path='/nonexistent')
    cap,ctx,paths=sample(tmp_path,cid,1.)
    client.observe_capture(cap)
    count=len(ctx['criteria']['success_checks'])
    with mock.patch.object(client,'_request',side_effect=respond(client,[False]*count)):
        client.analyze_release(ctx,paths,'release_0001')
    cap,ctx,paths=sample(tmp_path,cid,10.);client.observe_capture(cap)
    with mock.patch.object(client,'_request',side_effect=respond(client,[None]*count)):
        client.analyze(ctx,paths)
    cap,ctx,paths=sample(tmp_path,cid,20.);client.observe_capture(cap)
    index=next(i for i,c in enumerate(ctx['criteria']['success_checks']) if c['relation'] in ('on','in','right_of'))
    seen={}
    def request(current,images,prompt):
        seen.update(context=current,images=images,prompt=prompt)
        return respond(client,[True]*count,{index:['release_0001']})(current,images,prompt)
    with mock.patch.object(client,'_request',side_effect=request):
        result=client.analyze(ctx,paths)
    assert result['verdict']['operation_achieved'] is True
    assert len(seen['images'])==6
    safe=json.loads(seen['prompt'].splitlines()[-1]);timeline=safe['visual_timeline']
    assert [f['timestamp'] for f in safe['sensor_frames']]==[1.]*3+[20.]*3
    assert timeline['process_history'][0]['raw_checks'][0]['achieved'] is False
    assert timeline['previous_check_states'][0]['raw_checks'][0]['achieved'] is None
    assert timeline['current_frame_indices']==[3,4,5]
    assert timeline['process_history'][0]['actuator_measurements']['timestamp']==1.
    assert safe['actuator_measurements']['timestamp']==20.
    assert result['effective_checks'][index]['history_event_id']=='release_0001'
    assert 'target_id' in safe['success_checks'][index]
    for token in ('private request','current_subtask','predicates','native_success'):
        assert token not in seen['prompt']


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
@pytest.mark.parametrize('identity',[False,None])
def test_distractor_cannot_prove_goal_failure_or_success(tmp_path,cid,identity):
    client=XvlaQwenClient(env_path='/nonexistent')
    _,ctx,_=sample(tmp_path,cid,1.);client.set_check_contract(ctx)
    for value in (False,True):
        rows=checks(ctx,[True]*len(ctx['criteria']['success_checks']))
        rows[0].update(achieved=value,target_matches_definition=identity)
        assert client.parse_verdict(json.dumps({'checks':rows}))['operation_achieved'] is None
        assert client.last_skill_verdict['checks'][0]['achieved'] is value
        assert client.last_effective_checks[0]['achieved'] is None


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_latest_states_and_preserved_conditions_cannot_use_history_alone(tmp_path,cid):
    _,ctx,_=sample(tmp_path,cid,1.);client=XvlaQwenClient(env_path='/nonexistent');client.set_check_contract(ctx)
    client._active_event_ids={'release_0001'}
    for index,condition in enumerate(ctx['criteria']['success_checks']):
        if condition['relation'] in ('on','in','right_of') and not condition.get('preserved'):
            continue
        for value in (True,False):
            rows=checks(ctx,[True]*len(ctx['criteria']['success_checks']))
            rows[index].update(achieved=value,current_evidence=False,evidence_event_ids=['release_0001'])
            assert client.parse_verdict(json.dumps({'checks':rows}))['operation_achieved'] is None
    assert '不要求正面' in build_prompt(ctx)


def test_two_different_placements_keep_both_event_triplets(tmp_path):
    client=XvlaQwenClient(env_path='/nonexistent')
    for time,event_id in [(1.,'release_0001'),(2.,'release_0002')]:
        cap,ctx,paths=sample(tmp_path,'xvla_basket_both',time);client.observe_capture(cap)
        with mock.patch.object(client,'_request',side_effect=respond(client,[False,None])):
            client.analyze_release(ctx,paths,event_id)
    cap,ctx,paths=sample(tmp_path,'xvla_basket_both',3.);client.observe_capture(cap)
    with mock.patch.object(client,'_request',side_effect=respond(client,[True,True],{0:['release_0001'],1:['release_0002']})):
        result=client.analyze(ctx,paths)
    assert len(result['request_image_paths'])==9
    assert [c['history_event_id'] for c in result['effective_checks']]==['release_0001','release_0002']
    assert [h['raw_checks'][0]['achieved'] for h in result['input_context']['visual_timeline']['process_history']]==[False,False]


def test_unknown_not_promoted_from_old_boolean_and_history_is_reset(tmp_path):
    client=XvlaQwenClient(env_path='/nonexistent')
    cap,ctx,paths=sample(tmp_path,'xvla_basket_both',1.);client.observe_capture(cap)
    with mock.patch.object(client,'_request',side_effect=respond(client,[True,True])):
        client.analyze_release(ctx,paths,'release_0001')
    cap,ctx,paths=sample(tmp_path,'xvla_basket_both',2.);client.observe_capture(cap)
    with mock.patch.object(client,'_request',side_effect=respond(client,[None,True])):
        assert client.analyze(ctx,paths)['verdict']['operation_achieved'] is None
    _,other,_=sample(tmp_path,'xvla_mugs_both',3.);client.set_check_contract(other)
    assert client._timeline.releases=={} and client._timeline.check_states==[]


@pytest.mark.parametrize('bad',['future','changed_bytes','same_time_different_feedback'])
def test_stale_or_inconsistent_history_fails_before_request(tmp_path,bad):
    client=XvlaQwenClient(env_path='/nonexistent')
    cap,ctx,paths=sample(tmp_path,'xvla_basket_both',2.)
    client.remember_release(cap,ctx,'release_0001')
    if bad=='changed_bytes':Path(paths[0]).write_bytes(b'changed source')
    time=1. if bad=='future' else 2. if bad=='same_time_different_feedback' else 3.
    current,ctx,paths=sample(tmp_path,'xvla_basket_both',time)
    if bad=='same_time_different_feedback':
        current['robot']['finger_qpos']=[.02,-.02]
        ctx['actuator_measurements']['finger_qpos']=[.02,-.02]
    client.observe_capture(current)
    with mock.patch.object(client,'_request') as request:
        with pytest.raises(ValueError):client.analyze(ctx,paths)
    request.assert_not_called()


def test_release_at_polling_boundary_keeps_sources_without_fake_time_gap(tmp_path):
    client=XvlaQwenClient(env_path='/nonexistent')
    cap,ctx,paths=sample(tmp_path,'xvla_basket_both',1.);client.remember_release(cap,ctx,'release_0001')
    current=copy.deepcopy(cap);current_paths=[]
    for frame,path in zip(current['frames'],paths):
        copied=tmp_path/('current_'+Path(path).name);copied.write_bytes(Path(path).read_bytes())
        frame['rgb_path']=str(copied);current_paths.append(str(copied))
    client.observe_capture(current)
    with mock.patch.object(client,'_request',side_effect=respond(client,[None,None])):
        result=client.analyze(ctx,current_paths)
    assert len(result['request_image_paths'])==6
    assert {frame['timestamp'] for frame in result['input_context']['sensor_frames']}=={1.}


def test_strict_identity_fields_and_event_references(tmp_path):
    _,ctx,_=sample(tmp_path,'xvla_basket_both',1.);client=XvlaQwenClient(env_path='/nonexistent');client.set_check_contract(ctx)
    for field,value in [('target_matches_definition',1),('current_evidence','true'),('evidence_event_ids',['ghost'])]:
        rows=checks(ctx,[True,True]);rows[0][field]=value
        with pytest.raises(ValueError):client.parse_verdict(json.dumps({'checks':rows}))
    rows=checks(ctx,[True,True]);del rows[0]['target_matches_definition']
    with pytest.raises(ValueError):client.parse_verdict(json.dumps({'checks':rows}))


def test_bounded_history_does_not_silently_hide_overflow(tmp_path):
    client=XvlaQwenClient(env_path='/nonexistent')
    for time in range(1,6):
        cap,ctx,_=sample(tmp_path,'xvla_basket_both',float(time))
        client.remember_release(cap,ctx,'release_%04d'%time)
    cap,ctx,paths=sample(tmp_path,'xvla_basket_both',6.);client.observe_capture(cap)
    with mock.patch.object(client,'_request',side_effect=respond(client,[None,None])):
        result=client.analyze(ctx,paths)
    assert len(result['request_image_paths'])==MAX_INPUT_IMAGES==15
    assert len(client._load_images(result['request_image_paths']))==15
    timeline=result['input_context']['visual_timeline']
    assert timeline['history_complete'] is False
    assert timeline['omitted_release_event_ids']==['release_0005']


def test_original_failed_trial_images_and_states_enter_actual_serializer(tmp_path):
    root=Path(__file__).resolve().parents[1]
    job=root/'results/2026-10-13-xvla-release-trial/microwave_mug/runs/20261013T051908Z_microwave_mug_88287feb/cap_01_xvla_microwave_place_close_02ebbbf4'
    process=[json.loads(line) for line in (job/'finish/process_checks.jsonl').read_text().splitlines()]
    current=[json.loads(line) for line in (job/'finish/checks.jsonl').read_text().splitlines()]
    client=XvlaQwenClient(env_path='/nonexistent');client.audit_dir=tmp_path/'audit'
    def upgraded(original):
        ctx=build_skill_context(catalog.CAPABILITIES['xvla_microwave_place_close'],'saved original task')
        for field in ('sensor_frames','actuator_measurements'):ctx[field]=copy.deepcopy(original[field])
        return ctx
    for row in process:
        ctx=upgraded(row['context']);client.remember_release(row['capture'],ctx,row['event_id'])
        client._timeline.classify_release(row['event_id'],row['metadata']['raw_visual_goal_verdict']['checks'],row['metadata']['effective_checks'] if 'effective_checks' in row['metadata'] else row['metadata']['raw_visual_goal_verdict']['checks'])
    prior=current[-2]
    client._timeline.remember_check(upgraded(prior['ctx']),prior['initial_check']['metadata']['raw_visual_goal_verdict']['checks'],prior['initial_check']['metadata']['effective_checks'])
    row=current[-1];ctx=upgraded(row['ctx'])
    cap={'timestamp':ctx['actuator_measurements']['timestamp'],
         'robot':row['robot'],'frames':[{'view':v,'rgb_path':p} for v,p in zip(client.required_views,row['image_paths'])]}
    client.observe_capture(cap)
    replies=checks(ctx,[True,True]);replies[0].update(current_evidence=False,evidence_event_ids=['release_0001'])
    response=mock.Mock();response.getcode.return_value=200
    response.read.return_value=json.dumps({'choices':[{'message':{'content':json.dumps({'checks':replies,'reason':'software fake joint judgement'})}}]}).encode()
    with mock.patch.object(client,'_resolve_key',return_value='software-test-only'),mock.patch('urllib.request.urlopen',return_value=response) as http:
        result=client.analyze(ctx,row['image_paths'])
    assert http.call_count==1
    body=json.loads(http.call_args.args[0].data);items=body['messages'][0]['content']
    urls=[item['image_url']['url'] for item in items if item['type']=='image_url']
    assert len(urls)==9
    image_records=[]
    for url,path in zip(urls,result['request_image_paths']):
        raw=base64.b64decode(url.split(',',1)[1]);assert raw==Path(path).read_bytes()
        image_records.append({'path':path,'sha256':hashlib.sha256(raw).hexdigest()})
    safe=json.loads(items[0]['text'].splitlines()[-1]);timeline=safe['visual_timeline']
    assert [h['raw_checks'][0]['achieved'] for h in timeline['process_history']]==[False,False]
    assert timeline['previous_check_states'][0]['raw_checks'][0]['achieved'] is False
    assert len({f['timestamp'] for f in safe['sensor_frames']})==3
    assert result['verdict']['operation_achieved'] is True  # fake response, not physical success
    (tmp_path/'proof.json').write_text(json.dumps({'fake_http_response':True,'actual_http_calls':0,'environment_actions':0,
        'images':image_records,'safe_prompt':safe,'request_sha256':result['request_sha256']},ensure_ascii=False,indent=2),encoding='utf-8')
