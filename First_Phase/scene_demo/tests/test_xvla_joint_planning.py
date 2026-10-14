"""Native capability boundary and review reuse must not weaken execution gates."""
import copy
from types import SimpleNamespace

from scene_demo import catalog, service
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS, NATIVE_ATOMIC_CAPABILITIES
from scene_demo.xvla_service import XvlaSceneService
from scene_demo.xvla_planning import signature, reusable_review, build_prompt
from scene_demo.xvla_subtask_evidence import digest


def test_every_scene_exposes_only_trained_skills_and_all_native_goals():
    seen = set()
    for scene in catalog.SCENES:
        caps = catalog.scene_capabilities(scene, include_audit=True, backend='xvla')
        for cap in caps:
            cid = cap['capability_id']; seen.add(cid)
            assert cid in catalog.XVLA_NATIVE_IDS
            assert cid not in NATIVE_ATOMIC_CAPABILITIES
            if cid in NATIVE_LONG_SKILLS:
                assert cap['instruction'] == NATIVE_LONG_SKILLS[cid]['instruction']
                assert cap['goals'] == NATIVE_LONG_SKILLS[cid]['goals']
    assert seen == catalog.XVLA_NATIVE_IDS
    caps = catalog.scene_capabilities('bowl_drawer', backend='xvla')
    assert [c['capability_id'] for c in caps] == ['xvla_bowl_drawer_close']
    assert {g[0] for g in caps[0]['goals']} == {'in','close'}


def test_raw_service_cannot_route_split_candidates_even_without_grounding():
    svc = XvlaSceneService.__new__(XvlaSceneService)
    svc._sessions = {}
    for cid in NATIVE_ATOMIC_CAPABILITIES:
        result = svc.submit_plan({'decision':'execute','session_id':'unknown','capability_ids':[cid]})
        assert result['reason'] == 'non_native_capability'
    for cid in ['soup_to_basket','mug_to_microwave','microwave_close']:
        assert svc.submit_plan({'decision':'execute','capability_ids':[cid]})['ok'] is False


def test_public_validation_also_rejects_legacy_atomic_bypass():
    scene = {'scene_id':'bowl_drawer','scene_version':0,'state':'ready','capability_backend':'xvla'}
    payload = {'request_id':'r','session_id':'s','scene_version':0,'decision':'execute',
               'capability_ids':['xvla_t03_bowl_to_drawer']}
    assert service.validate_plan_request(scene,None,payload)['reason'] == 'non_native_capability'
    # Resume and worker dispatch use the same lower-level validator.
    assert service.validate_capability_list('bowl_drawer',payload['capability_ids'],True,backend='xvla')['reason'] == 'non_native_capability'


def test_review_reuse_requires_exact_plan_inventory_and_image_bytes(tmp_path):
    photo = tmp_path/'original.png';photo.write_bytes(b'original bytes')
    inventory = {'source_images':[{'path':str(photo),'sha256':digest(photo)}],'objects':[{'id':'bowl'}]}
    payload = {'request_id':'r','session_id':'s','scene_version':0,'decision':'execute',
               'capability_ids':['xvla_bowl_drawer_close'],'target_bindings':{'bowl':'obj1'},
               'request_scope':{'coverage':'partial'}}
    session = SimpleNamespace(scene_inventory=inventory,prepared_joint_plan={
        'signature':signature(payload,inventory),'review':{'passed':True}})
    assert reusable_review(session,SimpleNamespace(**payload)) == {'passed':True}
    for key,value in [('scene_version',1),('capability_ids',['xvla_book_caddy']),
                      ('target_bindings',{'bowl':'obj2'}),('request_scope',{'coverage':'full'})]:
        changed = {**payload,key:value}
        assert reusable_review(session,SimpleNamespace(**changed)) is None
    session.scene_inventory = {**inventory,'objects':[{'id':'other'}]}
    assert reusable_review(session,SimpleNamespace(**payload)) is None
    session.scene_inventory = inventory;photo.write_bytes(b'changed')
    assert reusable_review(session,SimpleNamespace(**payload)) is None


def test_prompt_supplies_full_native_goals_without_half_skill_candidates():
    prompt = build_prompt({'scene_id':'bowl_drawer','capabilities':catalog.scene_capabilities('bowl_drawer',backend='xvla')},'收拾碗和瓶子')
    assert 'xvla_bowl_drawer_close' in prompt
    assert 'xvla_t03_bowl_to_drawer' not in prompt
    assert '"close"' in prompt and '保持打开' in prompt
