"""Acceleration must preserve identity, source-time and confirmation boundaries."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scene_demo import catalog
from scene_demo.xvla_check_schedule import check_profile, profiles
from scene_demo.xvla_console import shared_planner_home
from scene_demo.xvla_grounding import apply_inventory_review, seed_checker, verify_plan
from scene_demo.xvla_monitor import XvlaFinishMonitor
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
from scene_demo.tests.test_xvla_grounding import inventory_for
from scene_demo.tests.test_xvla_evidence_recovery import take


def test_frozen_profiles_cover_actual_success_rows_and_all_native_skills():
    data=profiles()
    root=Path(__file__).resolve().parents[3]
    source=root/data['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==data['source_sha256']
    rows=list(csv.DictReader(source.open()))
    assert len(data['profiles'])==40
    for p in data['profiles']:
        measured=[int(r['policy_actions']) for r in rows if r['suite']==p['suite']
                  and int(r['task'])==p['task_id'] and r['success']=='True']
        assert p['earliest_success_step']==min(measured)
        assert min(measured)<=p['first_check_step']<min(measured)+30
    for cap in NATIVE_LONG_SKILLS.values():
        p=check_profile(cap)
        assert p['suite']==cap['native_task']['suite'] and p['task_id']==cap['native_task']['task_id']
    assert check_profile({'instruction':'untrained partial task'})['first_check_step']==150


def test_schedule_floor_motion_filter_and_final_budget_are_separate_from_success(tmp_path):
    cap,ctx,_=take(tmp_path,1.,'xvla_book_caddy',closed=True)
    cap['robot']['arm_qvel']=[.4]*7
    monitor=object.__new__(XvlaFinishMonitor)
    monitor.context=ctx;monitor.reference={'finger_home':[.04,-.04]}
    monitor.history=[cap];monitor.budget=900;monitor.schedule_skips=0
    monitor.configure_checks(catalog.CAPABILITIES['xvla_book_caddy'])
    assert monitor.check_schedule['first_check_step']==180
    assert not monitor.visual_check_due(150)
    assert not monitor.visual_check_due(180)  # Still moving; no success inferred.
    assert monitor.schedule_skips==1
    cap['robot']['arm_qvel']=[.01]*7
    assert monitor.visual_check_due(210)
    assert not monitor.visual_check_due(240)
    assert monitor.visual_check_due(270)
    cap['robot']['arm_qvel']=[.4]*7
    assert monitor.visual_check_due(900)


def test_shared_runtime_survives_distinct_requests_but_rejects_external_home(tmp_path):
    manifest=tmp_path/'manifest.json'
    first=shared_planner_home(tmp_path,manifest)
    (first/'installs').mkdir()
    (first/'runtime_marker').write_text('ready')
    second=shared_planner_home(tmp_path,manifest)
    assert second==first and (second/'runtime_marker').read_text()=='ready'
    manifest.write_text(json.dumps({'path':'/outside/workspace'}))
    with pytest.raises(ValueError,match='inside console_profiles'):shared_planner_home(tmp_path,manifest)


def test_inventory_delta_preserves_unchanged_views_and_requires_full_audit():
    first,_=inventory_for(catalog.CAPABILITIES['xvla_book_caddy'])
    ids=[o['object_id'] for o in first['objects']]
    changed=copy.deepcopy(first['objects'][0]);changed['description']='corrected visible target'
    delta={'reviewed_ids':ids,'removed_ids':[],'reviewed_views':[0,1,2],'updates':[changed],
           'coverage':{'targets_identifiable':True}}
    final=apply_inventory_review(first,delta)
    assert final['objects'][0]==changed and final['objects'][1]==first['objects'][1]
    assert first['objects'][0]['description']!=changed['description']
    bad=copy.deepcopy(delta);bad['reviewed_ids'].pop()
    with pytest.raises(ValueError,match='every original'):apply_inventory_review(first,bad)
    bad=copy.deepcopy(delta);bad['updates'][0]['view_observations'].pop()
    with pytest.raises(ValueError,match='three views'):apply_inventory_review(first,bad)


@pytest.mark.parametrize('cid',list(NATIVE_LONG_SKILLS))
def test_combined_audit_seeds_identity_and_regions_without_second_model_call(tmp_path,cid):
    cap,ctx,paths=take(tmp_path,1.,cid)
    capability=catalog.CAPABILITIES[cid]
    inventory,selection=inventory_for(capability)
    inventory.update(captured_at=1.,source_images=[{'path':p} for p in paths])
    goals=ctx['criteria']['success_checks']
    region_ids={g['destination_id'] for g in goals if g['relation'] in ('in','on','right_of')}
    raw={'checks':[{'target_id':k,'matches_request':True,'features':'independently checked '+k,
                    'anchors':[{'frame':1,'box':[20,20,120,120]}],'reason':'original images compared'} for k in selection],
         'regions':[{'destination_id':d,'confirmed':True,'description':'specific compartment',
                     'boundary':'visible walls','anchors':[{'frame':1,'box':[700,300,900,700]}]} for d in region_ids]}
    request_client=SimpleNamespace(raw_inventory=raw,_request=lambda *a:{'request_sha256':'request','response_sha256':'response'})
    checked=verify_plan(inventory,selection,'explicit task',[capability],tmp_path,client=request_client)
    assert checked['passed'] is True
    checker=XvlaSubtaskReviewClient(env_path='/none')
    seed_checker(checker,inventory,selection,cap,ctx,verified_review=checked)
    assert checker.store.initial_review_done and checker.analyze_initial() is None and checker._calls==0
    assert region_ids<=set(checker.store.regions)
    changed=copy.deepcopy(checked);changed['image_sha256'][0]='wrong pixels'
    with pytest.raises(ValueError,match='original images'):
        seed_checker(XvlaSubtaskReviewClient(env_path='/none'),inventory,selection,cap,ctx,verified_review=changed)


def test_combined_audit_rejects_unlocalized_region(tmp_path):
    cap,ctx,paths=take(tmp_path,1.,'xvla_book_caddy')
    capability=catalog.CAPABILITIES['xvla_book_caddy'];inventory,selection=inventory_for(capability)
    inventory.update(captured_at=1.,source_images=[{'path':p} for p in paths])
    raw={'checks':[{'target_id':k,'matches_request':True,'features':'target','anchors':[{'frame':1,'box':[20,20,120,120]}],'reason':'visible'} for k in selection], 'regions':[]}
    client=SimpleNamespace(raw_inventory=raw,_request=lambda *a:{'request_sha256':'r','response_sha256':'s'})
    assert verify_plan(inventory,selection,'book',[capability],tmp_path,client=client)['passed'] is False


def test_combined_audit_keeps_relevant_distractor_but_omits_unrelated_objects(tmp_path):
    _,_,paths=take(tmp_path,1.,'xvla_book_caddy')
    capability=catalog.CAPABILITIES['xvla_book_caddy'];inventory,selection=inventory_for(capability)
    inventory.update(captured_at=1.,source_images=[{'path':p} for p in paths])
    inventory['objects'][0]['confusable_with']=['distractor']
    inventory['objects'] += [{'object_id':'distractor','description':'similar black mat','confusable_with':['unrelated']},
                             {'object_id':'unrelated','description':'desk lamp'}]
    packets=[]
    def request(ctx,images,prompt):
        packets.append(json.loads(prompt.splitlines()[-1]))
        return {'request_sha256':'r','response_sha256':'s'}
    client=SimpleNamespace(raw_inventory={'checks':[],'regions':[]},_request=request)
    verify_plan(inventory,selection,'book',[capability],tmp_path,client=client)
    ids={o['object_id'] for o in packets[0]['inventory']['objects']}
    assert ids==set(selection.values())|{'distractor'}
