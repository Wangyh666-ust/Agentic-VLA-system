import copy
import json
import inspect
import importlib.util
import sys
from pathlib import Path
import pytest
from scene_demo.xvla_checker import GroundedQwenClient, build_prompt
from scene_demo.finish_v1 import vision, simple_checker


@pytest.fixture
def sample():
    context={'user_request':'secret whole chain','current_subtask':{'target':'wine_bottle_1','destination':'wine_rack_1_top_region','operation':'place','phase':'secretphase'},'criteria':{'task_description':'ignored'},'sensor_frames':[{'frame':0,'timestamp':1.,'view':'wrist'},{'frame':1,'timestamp':1.,'view':'sideview'}],'actuator_measurements':{'timestamp':1.,'finger_qpos':[.04,-.04],'standard_open_finger_qpos':[.04,-.04],'stage':'secretstage'},'predicates':{'secrettruth':True}}
    capture={'timestamp':1.,'frames':[{'view':'wrist','rgb_path':'w'},{'view':'agentview','rgb_path':'a'}],'verification_frames':[{'view':'sideview','rgb_path':'s'}],'robot':{'finger_qpos':[.04,-.04]}}
    return context,capture


def test_real_request_accepts_three_images():
    source=inspect.getsource(vision.QwenClient._load_images)
    assert '1 <= len(image_paths) <= 6' in source
    assert GroundedQwenClient._request is vision.QwenClient._request
    assert GroundedQwenClient.request_options is simple_checker.SimpleQwenClient.request_options


def test_safe_prompt_and_expansion(sample):
    context,capture=sample;c=GroundedQwenClient(max_calls=0);c.observe_capture(capture)
    seen={}
    def request(ctx,paths,prompt):
        seen.update(context=ctx,paths=paths,prompt=prompt)
        return {'verdict':c.parse_verdict('{"target_matches_definition":true,"goal_achieved":true,"reason":"visual"}')}
    c._request=request
    result=c.analyze(context,['w','s'])
    assert seen['paths']==['w','a','s']
    for forbidden in ['secret whole chain','secretphase','secretstage','secrettruth','user_request','current_subtask','predicates']:
        assert forbidden not in seen['prompt']
    assert '"goal_relation": "on"' in seen['prompt']
    assert result['visual_goal_verdict']['goal_achieved'] is True
    assert context['sensor_frames'][1]['view']=='sideview'


@pytest.mark.parametrize('change',['time','finger','path','capture'])
def test_capture_mismatch_rejected(sample,change):
    context,capture=sample;c=GroundedQwenClient(max_calls=0);c.observe_capture(capture)
    paths=['w','s']
    if change=='time':context['actuator_measurements']['timestamp']=2.
    if change=='finger':context['actuator_measurements']['finger_qpos']=[.03,-.04]
    if change=='path':paths=['w','unknown']
    if change=='capture':
        other=copy.deepcopy(capture);other['frames'][0]['rgb_path']='w2';c.observe_capture(other)
    with pytest.raises(ValueError):c.analyze(context,paths)


@pytest.mark.parametrize('identity',[False,None])
def test_unknown_identity_normalization(identity):
    c=GroundedQwenClient();raw={'target_matches_definition':identity,'goal_achieved':False,'reason':'unknown identity'}
    result=c.parse_verdict(json.dumps(raw))
    assert result['operation_achieved'] is None and c.last_two_question_verdict==raw


@pytest.mark.parametrize('key',['target_matches_definition','goal_achieved'])
def test_strict_boolean(key):
    raw={'target_matches_definition':True,'goal_achieved':True,'reason':'x'};raw[key]=1
    with pytest.raises(ValueError):GroundedQwenClient().parse_verdict(json.dumps(raw))


def test_in_semantics_and_encoder(sample):
    context,_=sample;context['current_subtask'].update(target='white_yellow_mug_1',destination='microwave_1_heating_region')
    assert '"goal_relation": "in"' in build_prompt(context)
    c=GroundedQwenClient();v=c.parse_verdict('{"target_matches_definition":true,"goal_achieved":true,"reason":"x"}')
    fused=simple_checker.reconcile(context,v,{'finger_qpos':[.04,-.04]},{'finger_home':[.04,-.04]})
    assert fused['released'] is True and fused['operation_achieved'] is True


def test_original_camera_label_mismatch(sample):
    context,capture=sample;c=GroundedQwenClient(max_calls=0);c.observe_capture(capture)
    context['sensor_frames'][1]['view']='agentview'
    with pytest.raises(ValueError,match='original camera label/path mismatch'):
        c.analyze(context,['w','s'])


def test_turnon_static_device(sample):
    context,_=sample
    context['current_subtask'].update(target='flat_stove_1',destination=None,operation='turn')
    context['criteria']['desired_state']='on'
    prompt=build_prompt(context)
    assert '方形单头炉面与圆形加热圈，旋钮用于开关。' in prompt
    assert '红色加热圈' in prompt and '"goal_relation": "turnon"' in prompt
    assert '"target_category": "flat_stove"' in prompt


def audit_module(monkeypatch):
    directory=Path(__file__).resolve().parents[1]/'results/2026-10-12-xvla-checker-grounding'
    monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('grounding_audit_test',directory/'audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_audit_refuses_existing_evidence(monkeypatch,tmp_path):
    audit=audit_module(monkeypatch);monkeypatch.setattr(audit,'HERE',tmp_path)
    (tmp_path/'calls').mkdir()
    with pytest.raises(FileExistsError,match='refuse to overwrite audit evidence'):
        audit.main()


def test_audit_failure_preserves_actual_count(monkeypatch,tmp_path):
    audit=audit_module(monkeypatch);monkeypatch.setattr(audit,'HERE',tmp_path)
    (tmp_path/'source_manifest.json').write_text('{}')
    (tmp_path/'cases.json').write_text(json.dumps([{'name':str(i),'capture':{},'source_context':{},'source_image_paths':[],'expected_allowed':[True]} for i in range(9)]))
    class FailedClient:
        def __init__(self,**kwargs):self._calls=0
        def observe_capture(self,c):pass
        def analyze(self,c,p):
            self._calls+=1
            raise ValueError('fake parse failure')
    monkeypatch.setattr(audit,'GroundedQwenClient',FailedClient)
    with pytest.raises(ValueError,match='fake parse failure'):audit.main()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['qwen_calls']==1 and result['cases']==[] and result['complete'] is False
    assert result['all_allowed_match'] is False and result['error']['current_case']=='0'
