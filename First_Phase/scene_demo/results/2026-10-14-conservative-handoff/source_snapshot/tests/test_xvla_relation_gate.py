"""False handoff prevention and true handoff reachability, across native skills."""
import copy
import pytest
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
from scene_demo.tests.test_xvla_subtask_review import response, install
from scene_demo.tests.test_xvla_evidence_recovery import take


@pytest.mark.parametrize('cid', list(NATIVE_LONG_SKILLS))
@pytest.mark.parametrize('defect', ['none','needs_motion','contradiction','no_support','region_unknown','missing_detail','view_contradiction','inferred_support'])
def test_all_native_handoff_gate(tmp_path,cid,defect):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.,cid,closed=True)
    client.remember_observation(cap,ctx);client.observe_capture(cap)
    def policy(p,n):
        raw=response(p,current_positive=True)
        if defect=='region_unknown':
            for region in raw['regions']:region['confirmed']=False
        for row in raw['current']:
            if row['state']!='at_destination':continue
            if defect=='no_support':row['supported']=None
            if defect in ('needs_motion','contradiction'):row['placement'][defect]=True
            if defect=='missing_detail':row.pop('placement')
            if defect=='view_contradiction':row['placement']['views'][1]['relation']='violated'
            if defect=='inferred_support':row['placement']['support_contact'].update(visible=False,evidence=[])
        return raw
    with install(client,policy):result=client.analyze(ctx,paths)
    has_placement=any(g['relation'] in ('in','on','right_of') for g in ctx['criteria']['success_checks'])
    expected=defect=='none' or not has_placement
    assert (result['verdict']['operation_achieved'] is True)==expected
    assert (result['handoff_review'] is not None)==expected


def test_candidate_recheck_can_veto_and_recent_frames_are_sent(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    for stamp in (1.,1.4,1.8,2.):
        cap,ctx,paths=take(tmp_path,stamp,'xvla_book_caddy',closed=True)
        client.remember_observation(cap,ctx)
    client.observe_capture(cap)
    def policy(p,n):
        raw=response(p,current_positive=p.get('decision_stage')!='handoff_review')
        if p.get('decision_stage')=='handoff_review':
            assert p['previous_reviews']==[]
            assert [o['captured_at'] for o in p['observations']]==[1.,1.4,1.8,2.]
        return raw
    with install(client,policy):result=client.analyze(ctx,paths)
    assert result['handoff_review'] is not None
    assert result['verdict']['operation_achieved'] is None


def test_region_correction_revokes_old_proof(tmp_path):
    client=XvlaSubtaskReviewClient(env_path='/none')
    cap,ctx,paths=take(tmp_path,1.,'xvla_book_caddy')
    client.remember_release(cap,ctx,'r1');client.observe_capture(cap)
    with install(client,lambda p,n:response(p,positive_history=True)):client.analyze(ctx,paths)
    assert len(client.store.proofs)==1
    cap,ctx,paths=take(tmp_path,2.,'xvla_book_caddy');client.observe_capture(cap)
    def corrected(p,n):
        raw=response(p)
        region=copy.deepcopy(p['regions'][0])
        region.update(corrects_previous=True,correction_reason='different actual compartment',confirmed=False)
        raw['regions']=[region]
        return raw
    with install(client,corrected):result=client.analyze(ctx,paths)
    assert result['verdict']['operation_achieved'] is None
    assert next(iter(client.store.proofs.values()))['status']=='region_recheck_required'
