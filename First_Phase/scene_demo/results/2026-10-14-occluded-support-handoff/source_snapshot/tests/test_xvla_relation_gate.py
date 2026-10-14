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


@pytest.mark.parametrize('cid', list(NATIVE_LONG_SKILLS))
@pytest.mark.parametrize('defect', ['none', 'support_unknown', 'depth_unknown', 'needs_motion', 'no_support',
                                  'view_contradiction', 'current_only', 'initial_only',
                                  'invalid_frame', 'forged_validation', 'moving', 'motion_missing',
                                  'motion_unsynced', 'extra_prior_relation', 'prior_relation_only',
                                  'extra_invalid_relation'])
def test_occluded_contact_requires_geometry_and_actual_process_refs(tmp_path, cid, defect):
    client = XvlaSubtaskReviewClient(env_path='/none')
    for stamp in (1., 1.4, 1.8, 2.):
        cap, ctx, paths = take(tmp_path, stamp, cid, closed=True)
        ctx['actuator_measurements']['max_abs_arm_qvel'] = .05
        if defect == 'moving': ctx['actuator_measurements']['max_abs_arm_qvel'] = .4
        if defect == 'motion_missing': ctx['actuator_measurements'].pop('max_abs_arm_qvel')
        client.remember_observation(cap, ctx)
        if defect == 'motion_unsynced':
            # Tamper only the saved motion timestamp; no mismatched live capture
            # can enter through the synchronized sensor validation path.
            last = list(client.store.observations.values())[-1]
            last['context']['actuator_measurements']['timestamp'] = stamp + .1
    client.observe_capture(cap)

    def policy(packet, _):
        raw = response(packet, current_positive=True)
        if not raw['current']:
            return raw
        current = next(o for o in packet['observations'] if o['capture_id'] == packet['current_capture_id'])
        prior = packet['observations'][-2]
        for row in raw['current']:
            if row['state'] != 'at_destination':
                continue
            detail = row['placement']
            detail['support_contact'].update(visible=False, observation='contact occluded', evidence=[])
            detail['support_inference'] = {
                'depth_sufficient': True, 'spatial_observation': 'body seated below boundary without a visible gap',
                'temporal_observation': 'insertion completed relative to the fixed rim in both samples',
                'evidence': [{'frame': f, 'box': [20, 20, 130, 140]}
                             for f in (prior['frames'][0], current['frames'][0])],
            }
            if defect == 'depth_unknown': detail['support_inference']['depth_sufficient'] = None
            if defect == 'needs_motion': detail['needs_motion'] = True
            if defect == 'no_support': row['supported'] = False
            if defect == 'support_unknown': row['supported'] = None
            if defect == 'view_contradiction': detail['views'][1]['relation'] = 'violated'
            if defect == 'current_only': detail['support_inference']['evidence'].pop(0)
            if defect == 'initial_only': detail['support_inference']['evidence'][0]['frame'] = packet['observations'][0]['frames'][0]
            if defect == 'invalid_frame': detail['support_inference']['evidence'][0]['frame'] = 999
            if defect in ('extra_prior_relation', 'prior_relation_only', 'extra_invalid_relation'):
                old_ref = {'frame': prior['frames'][0] if defect != 'extra_invalid_relation' else 999,
                           'box': [20, 20, 130, 140]}
                for field in ('destination_evidence', 'relation_evidence'):
                    detail[field] = ([old_ref] if defect == 'prior_relation_only' else detail[field] + [old_ref])
            if defect == 'forged_validation':
                detail.pop('support_inference')
                detail['_geometry_sequence_validated'] = True
        return raw

    with install(client, policy):
        result = client.analyze(ctx, paths)
    has_placement = any(g['relation'] in ('in', 'on', 'right_of') for g in ctx['criteria']['success_checks'])
    expected = defect in ('none', 'support_unknown', 'extra_prior_relation') or not has_placement
    assert (result['verdict']['operation_achieved'] is True) == expected
    assert (result['handoff_review'] is not None) == expected
    # Pre-release acceptance must never become a released historical proof.
    assert client.store.proofs == {}


def test_handoff_packet_keeps_approach_and_neighbours_without_future(tmp_path):
    client = XvlaSubtaskReviewClient(env_path='/none')
    observations = []
    for stamp in (.5, 1., 2., 4., 8., 12., 15.25, 15.5, 15.75, 20.):
        cap, ctx, paths = take(tmp_path, stamp, 'xvla_book_caddy', closed=True)
        observations.append(client.remember_observation(cap, ctx))
    packet, _, _ = client._pages(observations[-2], 'current')[0]
    times = [o['captured_at'] for o in packet['observations']]
    assert {2., 4., 8., 12., 15.25, 15.5, 15.75} <= set(times)
    assert 20. not in times
    assert len(times) * 3 <= 54
