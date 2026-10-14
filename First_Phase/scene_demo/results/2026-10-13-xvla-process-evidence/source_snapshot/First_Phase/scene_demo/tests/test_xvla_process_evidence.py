"""Release events and task-local evidence contracts; all responses are fake."""
import copy
import json
import math
from pathlib import Path
from unittest import mock
import pytest
from scene_demo import catalog
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_process_evidence import ReleaseSampler, PlacementEvidence, VIEWS
from scene_demo.xvla_finish import build_skill_context, build_prompt, XvlaQwenClient

HOME=[.04,-.04]
CLOSED=[.02,-.02]

def capture(tmp_path, fingers=HOME, time=1.0):
    frames=[]
    for v in VIEWS:
        path=tmp_path/(v+'.png');path.write_bytes(b'software only image')
        frames.append({'view':v,'rgb_path':str(path)})
    return {'timestamp':time,'frames':frames,'robot':{'finger_qpos':list(fingers),'arm_qvel':[0.]*7}}

def context(cid, cap):
    ctx=build_skill_context(catalog.CAPABILITIES[cid],'private request')
    ctx['sensor_frames']=[{'frame':i,'view':v,'timestamp':cap['timestamp']} for i,v in enumerate(VIEWS)]
    ctx['actuator_measurements']={'timestamp':cap['timestamp'],'finger_qpos':cap['robot']['finger_qpos'],'standard_open_finger_qpos':HOME}
    return ctx

def checks(ctx, values):
    return [{'id':c['id'],'achieved':v,'reason':'fake visual evidence'} for c,v in zip(ctx['criteria']['success_checks'],values)]

def respond(client, values):
    def request(ctx, paths, prompt):
        raw={'checks':checks(ctx,values),'reason':'fake visual classification'}
        return {'verdict':client.parse_verdict(json.dumps(raw)), 'call_index':None,
                'request_sha256':None,'response_sha256':None}
    return request

def test_sampler_transitions(tmp_path):
    sampler=ReleaseSampler(HOME);take=mock.Mock(side_effect=lambda:capture(tmp_path))
    for _ in range(4):assert sampler.observe(HOME,take) is None
    assert sampler.observe(CLOSED,take) is None
    assert sampler.observe(HOME,take) is None
    assert sampler.observe([.037,-.037],take) is None
    assert sampler.observe(HOME,take) is None
    event=sampler.observe(HOME,take);assert event['event_id']=='release_0001';take.assert_called_once()
    assert event['detector']=='measured_open_transition'
    for _ in range(4):assert sampler.observe(HOME,take) is None
    sampler.observe(CLOSED,take);sampler.observe(HOME,take)
    assert sampler.observe(HOME,take)['event_id']=='release_0002';assert take.call_count==2

@pytest.mark.parametrize('fingers',[[math.nan,0.],[True,0.],[0.],[0.,0.,0.]])
def test_sampler_invalid(fingers):
    with pytest.raises(ValueError):ReleaseSampler(fingers)
    with pytest.raises(ValueError):ReleaseSampler(HOME).observe(fingers,mock.Mock())

def test_sampler_capture_validation(tmp_path):
    for field in ('notopen','time','missing','duplicate'):
        sampler=ReleaseSampler(HOME);sampler.observe(CLOSED,lambda:None);sampler.observe(HOME,lambda:None)
        cap=capture(tmp_path)
        if field=='notopen':cap['robot']['finger_qpos']=CLOSED
        if field=='time':cap['timestamp']=math.nan
        if field=='missing':cap['frames'].pop()
        if field=='duplicate':cap['frames'][1]['rgb_path']=cap['frames'][0]['rgb_path']
        with pytest.raises(ValueError):sampler.observe(HOME,lambda:cap)

@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_all_ten_release_current_and_retraction(tmp_path,cid):
    cap=capture(tmp_path);ctx=context(cid,cap);contract=ctx['criteria']['success_checks']
    index=next(i for i,c in enumerate(contract) if c['relation'] in ('on','in','right_of'))
    values=[False]*len(contract);values[index]=True
    client=XvlaQwenClient(env_path='/nonexistent',max_calls=14);client.observe_capture(cap)
    with mock.patch.object(client,'_request',side_effect=respond(client,values)):
        result=client.analyze_release(ctx,[f['rgb_path'] for f in cap['frames']],'release_0001')
    assert len(client.process_evidence_snapshot())==1
    assert result['visual_goal_verdict']['goal_achieved'] is (True if len(contract)==1 else False)
    assert json.loads(result['prompt_text'].splitlines()[-1])['observation_role']=='release_snapshot'
    assert '得到支撑' in result['prompt_text']
    current=[True]*len(contract);current[index]=None
    with mock.patch.object(client,'_request',side_effect=respond(client,current)):
        result=client.analyze(ctx,[f['rgb_path'] for f in cap['frames']])
    assert result['verdict']['operation_achieved'] is True
    assert result['raw_visual_goal_verdict']['checks'][index]['achieved'] is None
    assert result['effective_checks'][index]['history_event_id']=='release_0001'
    assert result['raw_visual_goal_verdict']['checks'][index]['reason']=='fake visual evidence'
    assert result['effective_checks'][index]['current_reason']=='fake visual evidence'
    assert 'release_0001' in result['effective_checks'][index]['reason']
    assert 'release_0001' in result['visual_goal_verdict']['reason']
    assert 'release_0001' in result['verdict']['evidence'][0]
    assert set(json.loads(result['prompt_text'].splitlines()[-1]))=={'current_skill','success_checks','sensor_frames','actuator_measurements'}
    current[index]=False
    with mock.patch.object(client,'_request',side_effect=respond(client,current)):
        result=client.analyze(ctx,[f['rgb_path'] for f in cap['frames']])
    assert result['verdict']['operation_achieved'] is False
    assert 'release_0001' not in result['visual_goal_verdict']['reason']
    assert client.process_evidence_snapshot()==[]
    current[index]=None
    with mock.patch.object(client,'_request',side_effect=respond(client,current)):
        assert client.analyze(ctx,[f['rgb_path'] for f in cap['frames']])['verdict']['operation_achieved'] is None

@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_state_never_cached_and_empty_hand_never_cached(tmp_path,cid):
    cap=capture(tmp_path);ctx=context(cid,cap);ledger=PlacementEvidence(ctx['criteria']['success_checks'])
    ledger.record_release('release',ctx,checks(ctx,[False]*len(ledger.ids)),{})
    assert ledger.snapshot()==[]
    ledger.record_release('release',ctx,checks(ctx,[True]*len(ledger.ids)),{})
    expected=[c['id'] for c in ledger.contract if c['relation'] in ('on','in','right_of') and not c.get('preserved')]
    assert [c['id'] for c in ledger.snapshot()]==expected
    result=ledger.combine(checks(ctx,[None]*len(ledger.ids)))
    for c in result:assert c['achieved'] is (True if c['id'] in expected else None)
    copied=ledger.snapshot();copied.clear();assert len(ledger.snapshot())==len(expected)

def test_contract_new_client_and_id_isolation(tmp_path):
    cap=capture(tmp_path);ctx=context('xvla_moka_both',cap);client=XvlaQwenClient(env_path='/nonexistent')
    client.set_check_contract(ctx);client._placement_evidence.record_release('release',ctx,checks(ctx,[True,False,True]),{})
    assert len(client.process_evidence_snapshot())==1
    fused=client._placement_evidence.combine(checks(ctx,[None,None,True]));assert fused[0]['achieved'] is True;assert fused[1]['achieved'] is None
    client.set_check_contract(ctx);assert len(client.process_evidence_snapshot())==1
    client.set_check_contract(context('xvla_mugs_both',cap));assert client.process_evidence_snapshot()==[]
    assert XvlaQwenClient(env_path='/nonexistent').process_evidence_snapshot()==[]
    ledger=PlacementEvidence(ctx['criteria']['success_checks'])
    for raw in (checks(ctx,[True]),checks(ctx,[True,True,1])):
        with pytest.raises(ValueError):ledger.record_release('r',ctx,raw,{})
    bad=checks(ctx,[True]*3);bad[1]['id']=bad[0]['id']
    with pytest.raises(ValueError):ledger.combine(bad)

def test_current_held_target_does_not_require_prior_release(tmp_path):
    cap=capture(tmp_path,CLOSED);ctx=context('xvla_mugs_both',cap);client=XvlaQwenClient(env_path='/nonexistent');client.observe_capture(cap)
    with mock.patch.object(client,'_request',side_effect=respond(client,[True,True])):
        assert client.analyze(ctx,[f['rgb_path'] for f in cap['frames']])['verdict']['operation_achieved'] is True
    assert client.process_evidence_snapshot()==[]


def test_release_false_retracts_previous_true(tmp_path):
    cap=capture(tmp_path);ctx=context('xvla_mugs_both',cap);ledger=PlacementEvidence(ctx['criteria']['success_checks'])
    ledger.record_release('release1',ctx,checks(ctx,[True,False]),{})
    assert len(ledger.snapshot())==1
    ledger.record_release('release2',ctx,checks(ctx,[False,None]),{})
    assert ledger.snapshot()==[]
    assert ledger.combine(checks(ctx,[None,True]))[0]['achieved'] is None
    release_prompt=build_prompt(ctx,observation_role='release_snapshot')
    assert '仍夹持不否定' not in release_prompt
    assert '仍夹持不否定' in build_prompt(ctx)
