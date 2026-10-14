"""Generic observation reducer and real packet construction (no model calls)."""
import copy
import json
from unittest import mock
import pytest
from scene_demo import catalog
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_event_checker import XvlaEventQwenClient, reduce_observations
from scene_demo.xvla_finish import build_skill_context
from scene_demo.tests.test_xvla_process_evidence import capture, context


def fact(goal, relation='satisfied', obj=None, frames=None):
    return {'id':goal['id'],'identity':'matched','object_key':obj or goal['target_id'],
            'relation':relation,'frames':frames or [0],'identity_frames':[0],
            'supported':True,'continuity':'intact','reason':'test observed geometry'}


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_all_native_temporal_rules(cid):
    defs=build_skill_context(catalog.CAPABILITIES[cid],'test')['criteria']['success_checks']
    old=[fact(g) for g in defs]
    now=[fact(g,'occluded' if g['relation'] in ('in','on','right_of') and not g.get('preserved') else 'satisfied') for g in defs]
    events=[{'event_id':'release_1','observations':old}]
    assert all(r['achieved'] is True for r in reduce_observations(defs,events,now))
    for i,g in enumerate(defs):
        candidate=copy.deepcopy(now);candidate[i]['identity']='uncertain'
        assert reduce_observations(defs,events,candidate)[i]['achieved'] is None
        candidate=copy.deepcopy(now);candidate[i]['relation']='violated'
        assert reduce_observations(defs,events,candidate)[i]['achieved'] is False
        candidate=copy.deepcopy(now);candidate[i]['relation']='occluded';candidate[i]['continuity']='broken'
        assert reduce_observations(defs,events,candidate)[i]['achieved'] is None
        if g['relation'] not in ('in','on','right_of') or g.get('preserved'):
            candidate=copy.deepcopy(now);candidate[i]['relation']='occluded'
            assert reduce_observations(defs,events,candidate)[i]['achieved'] is None


def test_one_object_cannot_count_twice_or_opening_prove_support():
    defs=build_skill_context(catalog.CAPABILITIES['xvla_basket_both'],'test')['criteria']['success_checks']
    rows=[fact(g,obj='one_can') for g in defs]
    assert all(r['achieved'] is None for r in reduce_observations(defs,[],rows))
    old=[fact(g) for g in defs];now=[fact(g,'occluded') for g in defs]
    for r in old:r['supported']=None
    assert all(r['achieved'] is None for r in reduce_observations(defs,[{'event_id':'r1','observations':old}],now))
    for r in old:r['supported']=True
    later=[fact(g,'violated') for g in defs]
    assert all(r['achieved'] is None for r in reduce_observations(defs,[{'event_id':'r1','observations':old},{'event_id':'r2','observations':later}],now))


def test_real_packet_window_and_strict_event_frames(tmp_path):
    client=XvlaEventQwenClient(env_path='/none')
    def sample(t):
        p=tmp_path/str(t);p.mkdir();cap=capture(p,time=float(t));return cap,context('xvla_microwave_place_close',cap)
    start,sc=sample(1);client.remember_observation(start,sc)
    pre,pc=sample(2);client.remember_observation(pre,pc)
    rel,rc=sample(3);client.remember_release(rel,rc,'release_0001')
    after,ac=sample(4);client.remember_observation(after,ac)
    cur,cc=sample(5);client.observe_capture(cur)
    paths=[f['rgb_path'] for f in cur['frames']]
    packet,request_context,files=client._packet(cc,paths,'current_review')
    assert len(files)==12  # reference and oldest before are the same original triplet
    assert [g['kind'] for g in packet['groups']]==['reference','before','release','after','current']
    assert all(len(g['frames'])==3 for g in packet['groups'])
    client._active_packet=packet
    defs=cc['criteria']['success_checks']
    old=[fact(g,frames=packet['events'][0]['relation_frames'][:1]) for g in defs]
    now=[fact(g,'occluded' if i==0 else 'satisfied',frames=packet['current_frames'][:1]) for i,g in enumerate(defs)]
    value={'events':[{'event_id':'release_0001','observations':old}],'current':now,'reason':'test'}
    assert client.parse_verdict(json.dumps(value))['operation_achieved'] is True
    assert client.last_effective_checks[0]['history_event_id']=='release_0001'
    bad=copy.deepcopy(value);bad['events']=[]
    with pytest.raises(ValueError,match='events'):client.parse_verdict(json.dumps(bad))
    bad=copy.deepcopy(value);bad['events'][0]['observations'][0]['frames']=packet['current_frames'][:1]
    with pytest.raises(ValueError,match='frames'):client.parse_verdict(json.dumps(bad))
    bad=copy.deepcopy(value);bad['events'][0]['observations'][0]['identity_frames']=packet['current_frames'][:1]
    # Retrospective identity matching may compare later images already in this request.
    assert client.parse_verdict(json.dumps(bad))['operation_achieved'] is True
    bad['events'][0]['observations'][0]['identity_frames']=[999]
    with pytest.raises(ValueError,match='identity_frames'):client.parse_verdict(json.dumps(bad))
    # Same exact PNGs are loaded; source tampering is rejected before HTTP.
    from pathlib import Path
    Path(files[0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes'):client._packet(cc,paths,'current_review')


def test_online_monitor_uses_new_protocol():
    import inspect
    from scene_demo.xvla_service import XvlaSceneService
    assert 'xvla_evidence_memory' in inspect.getsource(XvlaSceneService._make_finish_monitor)


def test_broken_identity_and_previous_removal_cannot_resurrect_old_placement():
    defs=build_skill_context(catalog.CAPABILITIES['xvla_basket_both'],'test')['criteria']['success_checks']
    old=[fact(g) for g in defs];now=[fact(g,'occluded') for g in defs]
    events=[{'event_id':'r1','observed_until':1,'observations':old}]
    broken=[fact(g,'unknown') for g in defs]
    for r in broken:r.update(identity='uncertain',continuity='broken')
    assert all(r['achieved'] is None for r in reduce_observations(defs,events+[{'event_id':'r2','observations':broken}],now))
    revoked={g['id']:2 for g in defs}
    assert all(r['achieved'] is None for r in reduce_observations(defs,events,now,revoked))
    events.append({'event_id':'r3','observed_until':3,'observations':old})
    assert all(r['achieved'] is True for r in reduce_observations(defs,events,now,revoked))


def test_independent_cases_require_frozen_offline_qualification(tmp_path,monkeypatch):
    from scene_demo import xvla_pilot as pilot
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    sources={}
    for name in ('xvla_event_checker.py','xvla_evidence_memory.py','xvla_monitor.py','xvla_service.py'):
        p=tmp_path/'First_Phase/scene_demo'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
        sources[p.relative_to(tmp_path).as_posix()]=pilot.sha(p)
    cases=[('microwave_mug','positive'),('microwave_mug','negative'),('basket_two','positive'),('basket_two','negative'),('microwave_mug','no_history')]
    report={'passed':True,'complete':True,'results':[{'case':c,'mode':m,'passed':True} for c,m in cases]}
    p=tmp_path/'gate.json';p.write_text(json.dumps(report))
    proof={'path':'gate.json','sha256':pilot.sha(p),'source_sha256':sources}
    config={'configuration':{'independent_cases':['microwave_mug','basket_two'],'offline_validation':proof}}
    for case in ('microwave_mug','basket_two'):pilot.verify_case_prerequisite(case,tmp_path,config)
    with pytest.raises(RuntimeError,match='scope'):pilot.verify_case_prerequisite('mugs_two',tmp_path,config)
    report['results'][0]['passed']=False;p.write_text(json.dumps(report));proof['sha256']=pilot.sha(p)
    with pytest.raises(RuntimeError,match='offline_validation'):pilot.verify_case_prerequisite('basket_two',tmp_path,config)


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_production_composition_all_ten_with_isolated_current_states(tmp_path,cid):
    client=XvlaEventQwenClient(env_path='/none')
    initial=tmp_path/'initial';initial.mkdir();start=capture(initial,time=1.)
    ctx=context(cid,start);client.remember_observation(start,ctx)
    client.remember_release(start,ctx,'release_0001')
    latest=tmp_path/'latest';latest.mkdir();current=capture(latest,time=2.)
    cc=context(cid,current);client.observe_capture(current)
    packets=[]
    def request(c,paths,prompt):
        packet=copy.deepcopy(client._active_packet);packets.append(packet)
        events=[{'event_id':e['event_id'],'observations':[fact(g,frames=e['relation_frames'][:1]) for g in packet['goals']]} for e in packet['events']]
        values=[fact(g,'occluded' if packet['role']=='current_review' else 'satisfied',frames=packet['current_frames'][:1]) for g in packet['goals']]
        value={'events':events,'current':values,'reason':'test visual facts'}
        return {'verdict':client.parse_verdict(json.dumps(value)),'call_index':len(packets)}
    with mock.patch.object(client,'_request',side_effect=request):
        result=client.analyze(cc,[f['rgb_path'] for f in current['frames']])
    assert result['verdict']['operation_achieved'] is True
    definitions=cc['criteria']['success_checks']
    states=[g for g in definitions if g['relation'] not in ('in','on','right_of') or g.get('preserved')]
    assert len(packets)==1+bool(states)
    assert len(result['effective_checks'])==len(definitions)
    assert all(r['history_mode']=='occlusion_support' for r in result['effective_checks'] if r.get('history_event_id'))
    if states:
        assert packets[1]['role']=='current_state_only' and not packets[1]['events']
        assert [g['kind'] for g in packets[1]['groups']]==['reference','current']
        assert packets[1]['previous_checks']==[]
