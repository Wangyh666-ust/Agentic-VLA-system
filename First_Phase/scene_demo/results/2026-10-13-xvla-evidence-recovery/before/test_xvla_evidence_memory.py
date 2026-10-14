"""Evidence lifetime, capture provenance and complete production adapter tests."""
import copy,json
from unittest.mock import patch
import pytest
from scene_demo import catalog
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_finish import build_skill_context
from scene_demo.xvla_evidence_memory import XvlaEvidenceMemoryClient,update_memory,evaluate,STATES
from scene_demo.tests.test_xvla_process_evidence import capture,context


def row(goal,relation='at_destination',state=None):
    return {'id':goal['id'],'identity':'matched','object_key':goal['target_id'],
        'relation':'unknown' if goal['relation'] in STATES else relation,
        'observed_state':state or STATES.get(goal['relation'],'unknown'),
        'supported':True,'frames':[0],'identity_frames':[0],
        'change':{'kind':'none','same_object':None,'frames':[],'reason':''},'reason':'test visual fact'}


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_all_ten_memory_lifetime_and_current_states(cid):
    goals=build_skill_context(catalog.CAPABILITIES[cid],'test')['criteria']['success_checks']
    initial=[row(g) for g in goals]
    memory=update_memory(goals,{},initial,1.,'release_0001',{'sources':[]})
    expected={g['id'] for g in goals if g['relation'] in ('on','in','right_of') and not g.get('preserved')}
    assert set(memory)==expected
    unknown=copy.deepcopy(initial)
    for g,r in zip(goals,unknown):
        if g['id'] in expected:r.update(identity='uncertain',object_key=None,relation='unknown',supported=None,frames=[],identity_frames=[])
    unchanged=update_memory(goals,memory,unknown,2.,'release_0002',{})
    assert unchanged==memory
    assert all(r['achieved'] is True for r in evaluate(goals,unchanged,unknown))
    for g,r in zip(goals,unknown):
        if g['id'] not in expected:
            r.update(observed_state='unknown',relation='unknown')
    effective=evaluate(goals,unchanged,unknown)
    assert all(r['achieved'] is (True if g['id'] in expected else None) for g,r in zip(goals,effective))
    index=next(i for i,g in enumerate(goals) if g['id'] in expected)
    removal=copy.deepcopy(unknown);removal[index]=row(goals[index],'outside')
    removal[index]['change']={'kind':'removed','same_object':True,'frames':[0],'reason':'same target now outside'}
    revoked=update_memory(goals,memory,removal,3.,provenance={'image':'current'})
    assert revoked[goals[index]['id']]['status']=='revoked'
    assert evaluate(goals,revoked,removal)[index]['achieved'] is False
    assert evaluate(goals,revoked,unknown)[index]['achieved'] is None
    assert update_memory(goals,revoked,initial,4.,'release_0003',{})[goals[index]['id']]['status']=='confirmed'


def test_wrong_object_bare_outside_or_unrelated_release_not_revoke():
    goals=build_skill_context(catalog.CAPABILITIES['xvla_basket_both'],'test')['criteria']['success_checks']
    rows=[row(g) for g in goals];memory=update_memory(goals,{},rows,1.,'r1',{})
    current=copy.deepcopy(rows);current[0].update(object_key='other_can',relation='outside')
    current[0]['change']={'kind':'removed','same_object':True,'frames':[0],'reason':'other can on table'}
    assert update_memory(goals,memory,current,2.)==memory
    assert evaluate(goals,memory,current)[0]['achieved'] is True
    current[0]=row(goals[0],'outside')
    assert update_memory(goals,memory,current,2.)==memory
    assert evaluate(goals,memory,current)[0]['achieved'] is None
    for r in rows:r['object_key']='one_physical_can'
    assert update_memory(goals,{},rows,1.,'r1',{})=={}
    assert all(r['achieved'] is None for r in evaluate(goals,{},rows))


def test_sparse_windows_never_mislabeled(tmp_path):
    client=XvlaEvidenceMemoryClient(env_path='/none')
    def take(t):
        p=tmp_path/str(t);p.mkdir();cap=capture(p,time=t);return cap,context('xvla_microwave_place_close',cap)
    first,fc=take(8.25);client.remember_observation(first,fc)
    rel,rc=take(26.25);client.remember_release(rel,rc,'release_0001')
    far,farctx=take(30.75);client.remember_observation(far,farctx)
    assert client._events['release_0001']['before'] is None
    assert client._events['release_0001']['after'] is None
    close,cc=take(31.);client.remember_release(close,cc,'release_0002')
    after,ac=take(31.25);client.remember_observation(after,ac)
    assert client._events['release_0002']['before']['context']['actuator_measurements']['timestamp']==30.75
    assert client._events['release_0002']['after']['context']['actuator_measurements']['timestamp']==31.25


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_real_adapter_classifies_once_then_retains_uncertain(tmp_path,cid):
    client=XvlaEvidenceMemoryClient(env_path='/none')
    def take(t):
        p=tmp_path/str(t);p.mkdir();cap=capture(p,time=t);return cap,context(cid,cap)
    cap,ctx=take(1.);client.remember_observation(cap,ctx);client.observe_capture(cap)
    paths=[f['rgb_path'] for f in cap['frames']]
    packets=[]
    def request(c,paths,prompt):
        packet=copy.deepcopy(client._active_packet);packets.append(packet)
        values=[row(g) for g in packet['goals']]
        for g,r in zip(packet['goals'],values):
            r['frames']=packet['current_frames'][:1];r['identity_frames']=packet['current_frames'][:1]
            if packet['role']=='current_review' and g['relation'] not in STATES and not g.get('preserved'):
                r.update(identity='uncertain',object_key=None,relation='unknown',frames=[],identity_frames=[])
        return {'verdict':client.parse_verdict(json.dumps({'observations':values,'reason':'test'})),'call_index':len(packets)}
    with patch.object(client,'_request',side_effect=request):
        result=client.analyze_release(ctx,paths,'release_0001')
        same=client.analyze(ctx,paths)
        assert same['reused_same_capture'] and len(packets)==1
        original=copy.deepcopy(client._memory)
        nextcap,nextctx=take(2.);client.observe_capture(nextcap)
        result=client.analyze(nextctx,[f['rgb_path'] for f in nextcap['frames']])
        assert len(packets)==2 and result['verdict']['operation_achieved'] is True
        assert client._memory==original
        assert all(g['time']<=2. for g in packets[-1]['groups'])
        assert packets[-1]['previous_checks'][0]['role']=='release_review'
        assert len(packets[-1]['memory'])==len(original)
        # Duplicate release uses its immutable original result, no reclassification.
        client.observe_capture(cap);client.analyze_release(ctx,paths,'release_0001')
        assert len(packets)==2


def test_wrong_domain_and_forged_counterexample_frames_rejected(tmp_path):
    c=capture(tmp_path,time=1.);ctx=context('xvla_microwave_place_close',c)
    client=XvlaEvidenceMemoryClient(env_path='/none');client.set_check_contract(ctx)
    packet,*_=client._packet(ctx,[f['rgb_path'] for f in c['frames']]);client._active_packet=packet
    rows=[row(g) for g in packet['goals']]
    raw={'observations':rows,'reason':'test'}
    client.parse_verdict(json.dumps(raw))
    rows[1]['observed_state']='on'
    with pytest.raises(ValueError,match='domain'):client.parse_verdict(json.dumps(raw))
    rows[1]['observed_state']='closed'
    rows[0]['change']={'kind':'removed','same_object':True,'frames':[999],'reason':'forged'}
    with pytest.raises(ValueError,match='counterexample'):client.parse_verdict(json.dumps(raw))


def test_type_specific_fields_ignore_only_irrelevant_legacy_fields(tmp_path):
    c=capture(tmp_path,time=1.);ctx=context('xvla_microwave_place_close',c)
    client=XvlaEvidenceMemoryClient(env_path='/none');client.set_check_contract(ctx)
    packet,*_=client._packet(ctx,[f['rgb_path'] for f in c['frames']]);client._active_packet=packet
    rows=[row(g) for g in packet['goals']]
    # A state never uses placement relation to pass; contradictory-looking
    # legacy placement text cannot change an observed open door into closed.
    rows[1].update(relation='at_destination',observed_state='open')
    raw={'observations':rows,'reason':'visible open door'}
    assert client.parse_verdict(json.dumps(raw))['operation_achieved'] is False
    assert client.last_raw_observations['observations'][1]['relation']=='at_destination'
    assert client.last_observations['observations'][1]['relation']=='unknown'
    del rows[0]['observed_state'];del rows[1]['relation'];del rows[1]['supported']
    assert client.parse_verdict(json.dumps(raw))['operation_achieved'] is False
    rows[1]['observed_state']='unknown'
    assert client.parse_verdict(json.dumps(raw))['operation_achieved'] is None


def test_uncertain_outside_without_facts_cannot_abort_or_revoke(tmp_path):
    c=capture(tmp_path,time=1.);ctx=context('xvla_microwave_place_close',c)
    client=XvlaEvidenceMemoryClient(env_path='/none');client.set_check_contract(ctx)
    packet,*_=client._packet(ctx,[f['rgb_path'] for f in c['frames']]);client._active_packet=packet
    rows=[row(g) for g in packet['goals']]
    client._memory=update_memory(packet['goals'],{},rows,0.,'earlier',{})
    rows[0].update(identity='uncertain',relation='outside',frames=[],identity_frames=[])
    raw={'observations':rows,'reason':'identity uncertain, no valid outside evidence'}
    assert client.parse_verdict(json.dumps(raw))['operation_achieved'] is True
    assert update_memory(packet['goals'],client._memory,rows,1.)==client._memory
    rows[0]['identity']='matched';rows[0]['identity_frames']=[0]
    with pytest.raises(ValueError,match='fact source missing'):client.parse_verdict(json.dumps(raw))
