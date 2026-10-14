"""Temporal evidence invariants; fake model responses, no environment actions."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import pytest
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_evidence_memory import is_placement,STATES
from scene_demo.xvla_subtask_evidence import SubtaskEvidence,goal_key
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
from scene_demo.tests.test_xvla_evidence_recovery import take

def box(index):return [20+index*170,20,130+index*170,140]

def response(packet,positive_history=False,current_positive=False):
    first=packet['observations'][0]['frames'][0]
    current=next(o for o in packet['observations'] if o['capture_id']==packet['current_capture_id'])['frames'][0]
    bindings=[];rows=[];history=[]
    for i,g in enumerate(packet['goals']):
        anchors=[{'frame':first,'box':box(i)}]
        bindings.append({'target_id':g['target_id'],'matched':True,'features':'distinct object '+str(i),'anchors':anchors,'distractors':[]})
        def row(frame,positive):
            state=STATES[g['relation']] if g['relation'] in STATES else 'at_destination' if positive else 'unknown'
            return {'id':g['id'],'state':state,'same_target':True if state!='unknown' else None,
                'evidence':[{'frame':frame,'box':box(i)}] if state!='unknown' else [],
                'identity_refs':anchors if state!='unknown' else [],'supported':True if positive else None,
                'change':'none','comparison':'Compared initial target and the specified capture.'}
        if packet['mode']=='current':rows.append(row(current,current_positive))
        for request in packet['review_requests']:
            if request['id']==g['id']:
                frame=next(o for o in packet['observations'] if o['capture_id']==request['capture_id'])['frames'][0]
                history.append({**row(frame,positive_history),'capture_id':request['capture_id']})
    return {'bindings':bindings if packet['mode']=='initial' else [],'current':rows,'history':history}

def install(client,policy):
    def request(ctx,paths,prompt):
        client._calls+=1
        raw=policy(client._active_packet,client._calls)
        return {'verdict':client.parse_verdict(json.dumps(raw)),'request_sha256':'request'+str(client._calls),'response_sha256':'response'+str(client._calls)}
    return patch.object(client,'_request',side_effect=request)

@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_all_ten_unknown_release_re_review_old_capture_then_occlusion(tmp_path,cid):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.,cid);client.remember_release(cap,ctx,'release_0001');client.observe_capture(cap)
    with install(client,lambda p,n:response(p,positive_history=p['reviewed_at']>=2)):
        initial=client.analyze(ctx,paths)
        assert initial['process_evidence']==[]
        cap2,ctx2,paths2=take(tmp_path,2.,cid);client.observe_capture(cap2)
        later=client.analyze(ctx2,paths2)
        required=[g for g in ctx['criteria']['success_checks'] if is_placement(g)]
        assert len(later['process_evidence'])==len(required)
        assert later['verdict']['operation_achieved'] is True
        for proof in later['process_evidence']:
            assert proof['observed_at']==1. and proof['reviewed_at']==2.
            assert proof['kind']=='historical_review' and proof['event_id']=='release_0001'
            assert proof['supersedes_review_id'] and proof['request_sha256']=='request3'
        assert later['pages'][0]['image_paths'][:3]==paths
        cap3,ctx3,paths3=take(tmp_path,3.,cid);client.observe_capture(cap3)
        occluded=client.analyze(ctx3,paths3)
        assert occluded['process_evidence']==later['process_evidence']
        assert occluded['verdict']['operation_achieved'] is True
        assert all(c['current_evidence'] is False for c in occluded['effective_checks'] if c.get('history_event_id'))
        count=client._calls;again=client.analyze(ctx3,paths3)
        assert again['reused_same_capture'] and client._calls==count

def test_old_feedback_not_current_feedback_and_state_not_inherited(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.,'xvla_microwave_place_close',closed=True)
    client.remember_observation(cap,ctx)
    with pytest.raises(ValueError,match='measured opening'):client.remember_release(cap,ctx,'r1')
    # Request a historical review of a closed-gripper capture, never borrow later open feedback.
    item=next(iter(client.store.observations.values()))
    client.store.events['test']={'event_id':'test','capture_id':item['capture_id'],'status':'test_fixture'}
    cap2,ctx2,paths2=take(tmp_path,2.,'xvla_microwave_place_close');client.observe_capture(cap2)
    with install(client,lambda p,n:response(p,positive_history=True)):
        meta=client.analyze(ctx2,paths2)
    assert meta['process_evidence']==[] and meta['verdict']['operation_achieved'] is None
    assert all(r['id']=='goal_0' for r in meta['pages'][0]['packet']['review_requests'])

def test_repair_is_local_bounded_and_proofs_point_to_actual_response(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.);client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    def policy(p,n):
        raw=response(p,positive_history=True)
        if n==2:
            raw['history'][0]['evidence']=[]
        if n==3:
            # A valid other target from the first answer must survive a bad repair answer.
            raw['history'][1]['evidence']=[]
        return raw
    with install(client,policy):meta=client.analyze(ctx,paths)
    proofs=meta['process_evidence'];assert len(proofs)==2
    assert {p['request_sha256'] for p in proofs}=={'request2','request3'}
    assert client.store.repair_used and client._calls==3
    cap2,ctx2,paths2=take(tmp_path,2.);client.observe_capture(cap2)
    def missing(p,n):
        raw=response(p);raw['current'][0]['evidence']=[];raw['current'][0].update(same_target=True,state='at_destination');return raw
    with install(client,missing):meta=client.analyze(ctx2,paths2)
    assert client._calls==4 and meta['effective_checks'][0]['achieved'] is None
    assert meta['effective_checks'][1]['achieved'] is True

@pytest.mark.parametrize('defect',['duplicate','distractor','no_support'])
def test_reject_false_identity_and_unsupported_proof(tmp_path,defect):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.);client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    def policy(p,n):
        raw=response(p,positive_history=True)
        for r in raw['history']:
            if defect=='duplicate':r['evidence'][0]['box']=box(0)
            if defect=='distractor':r['identity_refs'][0]['box']=[800,800,900,900]
            if defect=='no_support':r['supported']=False
        return raw
    with install(client,policy):meta=client.analyze(ctx,paths)
    assert meta['process_evidence']==[]

def test_all_events_paged_no_future_no_tampering(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    takes=[]
    for i in range(1,22):
        cap,ctx,paths=take(tmp_path,float(i)*2);takes.append((cap,ctx,paths));client.remember_release(cap,ctx,'r%d'%i)
    current=client.store.observations[client.store.events['r21']['capture_id']]
    pages=client._pages(current,'current');assert len(pages)==2
    assert {r['event_id'] for p,_,_ in pages for r in p['review_requests']}==set(client.store.events)
    earlier=client.store.observations[client.store.events['r2']['capture_id']]
    assert set(client._pages(earlier,'current')[0][0]['coverage']['all_event_ids'])=={'r1','r2'}
    Path(takes[5][2][0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):client._pages(current,'current')
    assert len(client.store.observations)==21 and len(client.store.events)==21

def test_new_scope_and_changed_contract_do_not_inherit_proofs(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.);client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    with install(client,lambda p,n:response(p,True)):client.analyze(ctx,paths)
    old=client.store;assert old.proofs
    cap2,ctx2,_=take(tmp_path,2.,'xvla_microwave_place_close');client.remember_observation(cap2,ctx2)
    assert client.store is not old and not client.store.proofs
    shared=XvlaSubtaskReviewClient(env_path='/none',evidence_store=old)
    shared.set_check_contract(ctx);shared.set_check_contract(ctx2)
    assert shared.store is old
    assert goal_key(ctx['criteria']['success_checks'][0])!=goal_key(ctx2['criteria']['success_checks'][0])

def test_confirmed_removal_cannot_be_undone_by_old_release(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.);client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    with install(client,lambda p,n:response(p,True)):client.analyze(ctx,paths)
    cap2,ctx2,paths2=take(tmp_path,2.);client.observe_capture(cap2)
    def removed(p,n):
        raw=response(p,True,True)
        for r in raw['current']:r.update(state='outside',change='removed')
        return raw
    with install(client,removed):meta=client.analyze(ctx2,paths2)
    assert all(p['status']=='revoked' for p in meta['process_evidence'])
    cap3,ctx3,paths3=take(tmp_path,3.);client.observe_capture(cap3)
    with install(client,lambda p,n:response(p,True)):meta=client.analyze(ctx3,paths3)
    assert all(p['status']=='revoked' for p in meta['process_evidence'])
    assert meta['verdict']['operation_achieved'] is None
    client.observe_capture(cap)
    with pytest.raises(ValueError,match='backwards'):client.analyze(ctx,paths)

def test_component_event_numbers_do_not_collide_in_same_logical_task(tmp_path):
    store=SubtaskEvidence()
    clients=[XvlaSubtaskReviewClient(env_path='/none',evidence_store=store,event_namespace=n) for n in ('stage1','stage2')]
    for i,client in enumerate(clients):
        cap,ctx,paths=take(tmp_path,float(i+1));client.remember_release(cap,ctx,'release_0001')
    assert set(store.events)=={'stage1/release_0001','stage2/release_0001'}
    assert len(store.observations)==2
    assert len({v['capture_id'] for v in store.events.values()})==2

def test_corrected_binding_requires_reconfirmation_and_door_requires_current(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.,'xvla_microwave_place_close');client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    with install(client,lambda p,n:response(p,True)):client.analyze(ctx,paths)
    assert client.process_evidence_snapshot()[0]['status']=='confirmed'
    cap2,ctx2,paths2=take(tmp_path,2.,'xvla_microwave_place_close');client.observe_capture(cap2)
    def correction(p,n):
        raw=response(p)
        b=response({**p,'mode':'initial'})['bindings'][0]
        b.update(corrects_previous=True,correction_reason='Old anchor was the distractor, new region has the specified handle.')
        b['anchors'][0]['box']=[800,800,900,900];raw['bindings']=[b]
        raw['current'][1].update(state='unknown',same_target=None,evidence=[],identity_refs=[])
        return raw
    with install(client,correction):meta=client.analyze(ctx2,paths2)
    assert meta['process_evidence'][0]['status']=='identity_recheck_required'
    assert meta['verdict']['operation_achieved'] is None
    assert meta['effective_checks'][1]['achieved'] is None

def test_known_state_null_identity_is_repaired_without_inventing_identity(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.);client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    def policy(p,n):
        raw=response(p,True)
        if n==2:
            for row in raw['history']:row['same_target']=None
        return raw
    with install(client,policy):meta=client.analyze(ctx,paths)
    assert client._calls==3
    assert all(p['request_sha256']=='request3' for p in meta['process_evidence'])
    assert len(meta['process_evidence'])==2
    assert all('same_target=true' in i['error'] for i in meta['pages'][0]['original_review']['issues'])

def test_stable_capture_view_references_resolve_without_guessing(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.);client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    def policy(p,n):
        raw=response(p,True)
        for row in raw['history']:
            row['evidence']=[{'capture_id':row['capture_id'],'view':'wrist','box':row['evidence'][0]['box']}]
        return raw
    with install(client,policy):meta=client.analyze(ctx,paths)
    assert not client.store.repair_used and len(meta['process_evidence'])==2

def test_initial_missing_device_binding_is_added_by_repair(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.,'xvla_microwave_place_close');client.remember_observation(cap,ctx)
    def policy(p,n):
        raw=response(p)
        if n==1:raw['bindings']=raw['bindings'][:1]
        return raw
    with install(client,policy):client.analyze_initial()
    assert client._calls==2 and client.store.repair_used
    assert set(client.store.bindings)=={g['target_id'] for g in ctx['criteria']['success_checks']}

def test_container_and_contents_overlap_is_not_two_objects_counted_twice(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.,'xvla_microwave_place_close');client.remember_observation(cap,ctx);client.observe_capture(cap)
    def policy(p,n):
        raw=response(p,current_positive=True)
        for row in raw['current']:row['evidence'][0]['box']=box(0)
        return raw
    with install(client,policy):meta=client.analyze(ctx,paths)
    assert meta['verdict']['operation_achieved'] is True
    assert not meta['pages'][0]['applied_review']['issues']
