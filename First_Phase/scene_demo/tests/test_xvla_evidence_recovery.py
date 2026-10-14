"""First release missed -> later proof -> occlusion, through the real adapter."""
import copy,json
from unittest.mock import patch
import pytest
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_evidence_memory import XvlaEvidenceMemoryClient,is_placement,MAX_IMAGES
from scene_demo.tests.test_xvla_evidence_memory import row
from scene_demo.tests.test_xvla_process_evidence import capture,context

def take(tmp_path,t,cid='xvla_basket_both',closed=False):
    folder=tmp_path/str(t);folder.mkdir()
    cap=capture(folder,time=t)
    if closed:cap['robot']['finger_qpos']=[.015,-.015]
    ctx=context(cid,cap)
    return cap,ctx,[f['rgb_path'] for f in cap['frames']]

@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_all_ten_missed_release_late_confirmation_and_occlusion(tmp_path,cid):
    client=XvlaEvidenceMemoryClient(env_path='/none');packets=[]
    def request(ctx,paths,prompt):
        packet=copy.deepcopy(client._active_packet);packets.append(packet)
        values=[row(g) for g in packet['goals']]
        for goal,value in zip(packet['goals'],values):
            value['frames']=packet['current_frames'][:1]
            value['identity_frames']=packet['current_frames'][:1]
            if is_placement(goal) and len(packets)!=2:
                value.update(identity='uncertain',object_key=None,relation='unknown',
                             supported=None,frames=[],identity_frames=[])
        return {'verdict':client.parse_verdict(json.dumps({'observations':values,'reason':'test'})),
                'call_index':len(packets)}
    cap,ctx,paths=take(tmp_path,1.,cid);client.observe_capture(cap)
    with patch.object(client,'_request',side_effect=request):
        initial=client.analyze_release(ctx,paths,'release_0001')
        assert initial['process_evidence']==[]
        later,lctx,lpaths=take(tmp_path,2.,cid);client.observe_capture(later)
        confirmed=client.analyze(lctx,lpaths)
        expected={g['id'] for g in packets[-1]['goals'] if is_placement(g)}
        assert {p['id'] for p in confirmed['process_evidence']}==expected
        assert packets[-1]['release_history'][0]['event_id']=='release_0001'
        assert set(paths)<=set(confirmed['request_image_paths'])
        for proof in confirmed['process_evidence']:
            assert proof['event_id'] is None and proof['timestamp']==2.
            assert proof['evidence_id']=='observation_0002'
            assert proof['confirmation_kind']=='current_observation'
            assert proof['provenance']['sources'][0]['paths']==lpaths
        frozen=copy.deepcopy(confirmed['process_evidence'])
        occluded,octx,opaths=take(tmp_path,3.,cid);client.observe_capture(occluded)
        reused=client.analyze(octx,opaths)
        assert reused['verdict']['operation_achieved'] is True
        assert reused['process_evidence']==frozen
        for value in reused['effective_checks']:
            if value['id'] in expected:
                assert value['history_release_event_id'] is None
                assert value['history_event_id']=='observation_0002'
        assert set(paths+lpaths)<=set(reused['request_image_paths'])

@pytest.mark.parametrize('defect',['closed','unsupported','uncertain','duplicate','missing_current'])
def test_late_proof_requires_open_support_identity_and_current_fact(tmp_path,defect):
    cap,ctx,paths=take(tmp_path,1.,closed=defect=='closed')
    client=XvlaEvidenceMemoryClient(env_path='/none');client.observe_capture(cap)
    def request(ctx,paths,prompt):
        packet=client._active_packet;values=[row(g) for g in packet['goals']]
        for value in values:
            if defect=='unsupported':value['supported']=False
            if defect=='uncertain':value['identity']='uncertain'
            if defect=='duplicate':value['object_key']='one_can'
            if defect=='missing_current':value['frames']=[]
        return {'verdict':client.parse_verdict(json.dumps({'observations':values,'reason':'test'}))}
    with patch.object(client,'_request',side_effect=request):
        if defect=='missing_current':
            with pytest.raises(ValueError,match='fact source missing'):client.analyze(ctx,paths)
        else:
            result=client.analyze(ctx,paths)
            assert result['process_evidence']==[]
            if defect=='closed':assert result['verdict']['operation_achieved'] is True

def test_unclassified_events_past_legacy_cap_preserve_bytes_and_filter_future(tmp_path):
    client=XvlaEvidenceMemoryClient(env_path='/none');all_paths=[];captures=[]
    for t in range(1,7):
        cap,ctx,paths=take(tmp_path,float(t));captures.append((cap,ctx,paths));all_paths+=paths
        client.remember_release(cap,ctx,'release_%04d'%t)
        client.note_release_status('release_%04d'%t,'unchecked_due_budget')
    assert len(client.release_archive_snapshot())==6
    packet,_,images,_=client._packet(captures[-1][1],captures[-1][2])
    assert len(packet['release_history'])==6 and set(images)==set(all_paths)
    earlier=client._packet(captures[1][1],captures[1][2])[0]
    assert len(earlier['release_history'])==2
    assert max(g['time'] for g in earlier['groups'])==2.
    # The fifth source is checked even though the base timeline omitted it.
    cap,ctx,paths=captures[4]
    from pathlib import Path
    Path(paths[0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):client.remember_release(cap,ctx,'release_0005')
    with pytest.raises(ValueError,match='bytes changed'):client._packet(captures[-1][1],captures[-1][2])

def test_budget_fails_explicitly_without_discarding_any_archived_event(tmp_path):
    client=XvlaEvidenceMemoryClient(env_path='/none')
    for t in range(1,MAX_IMAGES//3+2):
        cap,ctx,paths=take(tmp_path,float(t));client.remember_release(cap,ctx,'r%d'%t)
    with pytest.raises(ValueError,match='budget exceeded'):client._packet(ctx,paths)
    assert len(client.release_archive_snapshot())==MAX_IMAGES//3+1
    other=context('xvla_microwave_place_close',cap);client.set_check_contract(other)
    assert client.release_archive_snapshot()==[] and client.process_evidence_snapshot()==[]
