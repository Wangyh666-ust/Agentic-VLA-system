"""Planning-to-checker identity handoff and invalidation, no model/motion calls."""
import copy
import pytest
from scene_demo import catalog, run_agent
from scene_demo.xvla_grounding import validate_inventory, validate_selection, seed_checker
from scene_demo.xvla_process_evidence import capture_frames
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.tests.test_xvla_evidence_recovery import take
from scene_demo.tests.test_xvla_subtask_review import install, response


def inventory_for(capability):
    keys = sorted({v for goal in capability['goals'] for v in goal[1:]})
    objects = [{'object_id': f'obj_{i}', 'description': 'visible distinct object '+str(i), 'confirmed': True,
                'anchors': [{'frame': 1, 'box': [i*70, 100, i*70+50, 200]}],
                'view_observations': [{'frame': f, 'visibility': 'visible' if f == 1 else 'occluded', 'reason': 'image'} for f in range(3)]}
               for i in range(len(keys))]
    return {'objects': objects}, {key: obj['object_id'] for key, obj in zip(keys, objects)}


@pytest.mark.parametrize('cid', list(NATIVE_LONG_SKILLS))
def test_all_native_share_initial_identity_and_source_images(tmp_path, cid):
    cap, ctx, paths = take(tmp_path, 1., cid)
    inventory, selection = inventory_for(catalog.CAPABILITIES[cid])
    assert validate_inventory(inventory) == inventory
    assert validate_selection(inventory, selection, [catalog.CAPABILITIES[cid]]) == selection
    client = XvlaSubtaskReviewClient(env_path='/none')
    seed_checker(client, inventory, selection, cap, ctx)
    assert not client.store.initial_review_done  # Parent inventory is not region grounding.
    assert client._calls == 0
    assert client.store.observations[client.store.initial_id]['paths'] == paths
    assert set(client.store.bindings) == set(selection)
    packet = client._pages(client._reference, 'current')[0][0]
    assert {b['target_id'] for b in packet['bindings']} == set(selection)
    assert all(b['features'].startswith('visible distinct') for b in packet['bindings'])


def test_reject_unconfirmed_duplicate_targets_and_missing_destination():
    cap = catalog.CAPABILITIES['xvla_basket_both']
    inventory, mapping = inventory_for(cap)
    moved = [g[1] for g in cap['goals']]
    bad = dict(mapping); bad[moved[1]] = bad[moved[0]]
    with pytest.raises(ValueError, match='distinct manipulated'): validate_selection(inventory, bad, [cap])
    bad = dict(mapping); bad.pop(cap['goals'][0][2])
    with pytest.raises(ValueError, match='every goal'): validate_selection(inventory, bad, [cap])
    inventory['objects'][0]['confirmed'] = False
    with pytest.raises(ValueError, match='not visually confirmed'): validate_selection(inventory, mapping, [cap])


def test_three_view_assessment_mandatory_but_occlusion_allowed():
    inventory, _ = inventory_for(catalog.CAPABILITIES['xvla_basket_both'])
    validate_inventory(inventory)
    inventory['objects'][0]['view_observations'].pop()
    with pytest.raises(ValueError, match='three views'): validate_inventory(inventory)


def test_checker_only_frames_do_not_replace_geometry(tmp_path):
    cap, _, _ = take(tmp_path, 1.)
    original = copy.deepcopy(cap['frames'])
    cap['checker_frames'] = copy.deepcopy(cap['frames'])
    for f in cap['checker_frames']: f['rgb_path'] = 'checker_' + f['view'] + '.png'
    assert all(f['rgb_path'].startswith('checker_') for f in capture_frames(cap))
    assert cap['frames'] == original


def test_planner_receives_inventory_and_three_view_image():
    inventory, _ = inventory_for(catalog.CAPABILITIES['xvla_basket_both'])
    session = {'scene_inventory': inventory, 'planning_image': '/three.png', 'images': [{'view': 'agentview', 'image_path': '/native.png'}]}
    prompt = run_agent.build_initial_prompt(session, 'put both cans in basket', 'r', ['/native.png'])
    assert 'target_bindings' in prompt and 'obj_0' in prompt
    assert run_agent.Runner._pick_agentview(session) == '/three.png'


def test_destination_correction_invalidates_both_old_placements(tmp_path):
    client = XvlaSubtaskReviewClient(env_path='/none')
    cap, ctx, paths = take(tmp_path, 1.)
    inventory, mapping = inventory_for(catalog.CAPABILITIES['xvla_basket_both'])
    seed_checker(client, inventory, mapping, cap, ctx)
    client.remember_release(cap, ctx, 'r1'); client.observe_capture(cap)
    def matching(p, n):
        raw = response(p, positive_history=True)
        for row in raw['history']:
            target = next(g['target_id'] for g in p['goals'] if g['id'] == row['id'])
            row['identity_refs'] = next(b['anchors'] for b in p['bindings'] if b['target_id'] == target)
        return raw
    with install(client, matching): first = client.analyze(ctx, paths)
    assert len(first['process_evidence']) == 2
    later, later_ctx, later_paths = take(tmp_path, 2.)
    client.observe_capture(later)
    def correct(p, n):
        raw = response(p)
        destination = p['goals'][0]['destination_id']
        raw['bindings'] = [{'target_id': destination, 'matched': True, 'features': 'different actual basket',
            'anchors': [{'frame': 4, 'box': [700, 600, 950, 900]}], 'distractors': [],
            'corrects_previous': True, 'correction_reason': 'Initial region belonged to another container'}]
        return raw
    with install(client, correct): last = client.analyze(later_ctx, later_paths)
    assert last['verdict']['operation_achieved'] is None
    assert all(p['status'] == 'identity_recheck_required' for p in last['process_evidence'])
