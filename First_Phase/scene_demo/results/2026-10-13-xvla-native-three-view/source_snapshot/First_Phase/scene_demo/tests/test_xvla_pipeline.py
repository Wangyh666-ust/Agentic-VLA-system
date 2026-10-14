"""Pure boundary tests: no simulator, transport, or planner is constructed."""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from scene_demo import xvla_service as x
from scene_demo import xvla_pilot as pilot


def test_next_reference_order():
    ids=pilot.CASES['wine_bowl_stove']['capabilities']
    assert [x.next_capability(ids,c) for c in ids]==['bowl_to_plate','stove_on','global_home']


def test_strict_plan_order(monkeypatch):
    svc=object.__new__(x.XvlaSceneService)
    svc.expected_capabilities=['wine_to_rack','bowl_to_plate','stove_on']
    parent=Mock(return_value={'ok':True})
    monkeypatch.setattr(x.service.SceneService,'submit_plan',parent)
    bad=svc.submit_plan({'decision':'execute','capability_ids':['stove_on','wine_to_rack','bowl_to_plate']})
    assert bad['reason']=='pilot_plan_mismatch'
    parent.assert_not_called()
    assert svc.submit_plan({'decision':'execute','capability_ids':svc.expected_capabilities})=={'ok':True}


def test_cache_reset_does_not_reset_scene(monkeypatch):
    from scene_demo import joint_home, safe_exit
    svc=object.__new__(x.XvlaSceneService)
    client=SimpleNamespace(action_plan=[1],proprio=np.ones(20))
    def reset():
        client.action_plan=[]
        client.proprio=None
    client.reset=Mock(side_effect=reset)
    svc._xvla_client=client
    objects=np.array([7.,8.])
    svc._env=SimpleNamespace(objects=objects,reset=Mock(),step=Mock())
    ctrl=SimpleNamespace(update=Mock(),reset_goal=Mock(),use_delta=True)
    robot=SimpleNamespace(controller=ctrl,gripper=SimpleNamespace(current_action=np.array([.9,.8])))
    svc._robot=lambda:robot
    measured_hold=np.array([.2,-.3],dtype=np.float32)
    monkeypatch.setattr(joint_home,'read_robot_state',lambda env:{'finger_qpos':[.01,.02]})
    monkeypatch.setattr(safe_exit,'_finger_hold',lambda env,state:measured_hold)
    svc._controller=lambda:ctrl
    svc._refresh_observation=Mock(return_value={'actual':1})
    svc._actual_state=lambda:{'queue_size':len(client.action_plan),'proprio_is_none':client.proprio is None}
    svc._cache_evidence=[]
    svc._reset_policy_queues()
    assert svc._cache_evidence==[{'queue_size':0,'proprio_is_none':True}]
    assert ctrl.use_delta is False
    assert np.array_equal(objects,[7.,8.])
    svc._env.reset.assert_not_called()
    svc._env.step.assert_not_called()
    assert np.array_equal(robot.gripper.current_action,measured_hold)


def test_independent_preserves_prior_goals(monkeypatch,tmp_path):
    from scene_demo import safe_exit_contacts
    from scene_demo import xvla_finish as simple_checker
    svc=object.__new__(x.XvlaSceneService)
    monitor=SimpleNamespace(port=SimpleNamespace(start=Mock(),close=Mock()))
    monkeypatch.setattr(x.service.SceneService,'_make_finish_monitor',lambda *a:monitor)
    monkeypatch.setattr(simple_checker,'XvlaQwenClient',lambda **kw:SimpleNamespace())
    monkeypatch.setattr(safe_exit_contacts,'contacts',lambda env:[])
    monkeypatch.setattr(x,'reference_metrics',lambda *a:{'ready':True})
    monkeypatch.setattr(x.service,'eval_goal_predicate',lambda env,g:g[1]!='wine_bottle_1')
    svc._actual_state=lambda:{}
    svc._completion_ready=lambda env,g,p:all(p.values())
    session=SimpleNamespace(home_reference={},xvla_completed=['wine_to_rack'],scene_id='goal_table')
    job=SimpleNamespace(capability_id='bowl_to_plate',run_dir=tmp_path)
    result=svc._make_finish_monitor(session,None,job,None,{},800,[],None)
    assert result.independent_ready() is False
    assert 'wine_to_rack' in (tmp_path/'independent_finish.jsonl').read_text()


def assignment_from_source(path,name,env):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    import os
    previous=dict(os.environ)
    try:
        for key in ('SCENE_BUDGET_PER_SUBGOAL','SCENE_EXPECTED_WORKFLOW','SCENE_EXPECTED_MODEL_REVISION'):
            os.environ.pop(key,None)
        os.environ.update(env)
        return eval(compile(ast.Expression(node.value),str(path),'eval'),{'os':os,'int':int})
    finally:
        os.environ.clear()
        os.environ.update(previous)


def test_opt_in_budget_and_identity():
    here=Path(x.__file__).parent
    assert assignment_from_source(here/'mcp_server.py','DEFAULT_BUDGET',{})==600
    assert assignment_from_source(here/'mcp_server.py','DEFAULT_BUDGET',{'SCENE_BUDGET_PER_SUBGOAL':'800'})==800
    assert assignment_from_source(here/'run_agent.py','EXPECTED_WORKFLOW',{})=='persistent_scene_v2'
    assert assignment_from_source(here/'run_agent.py','EXPECTED_MODEL_REVISION',{})=='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
    assert assignment_from_source(here/'run_agent.py','EXPECTED_WORKFLOW',{'SCENE_EXPECTED_WORKFLOW':'persistent_scene_xvla_v2'})=='persistent_scene_xvla_v2'
    assert assignment_from_source(here/'run_agent.py','EXPECTED_MODEL_REVISION',{'SCENE_EXPECTED_MODEL_REVISION':pilot.REVISION})==pilot.REVISION


def test_help_pure(capsys):
    with pytest.raises(SystemExit) as error:
        pilot.main(['--help'])
    assert error.value.code==0
    assert '--prepare' in capsys.readouterr().out


def recorded_handoff():
    # Data contract fixture only; no controller or simulator runs in this test.
    return {'source_counts':{'vla':1}, 'vla_calls':1,
        'finish_status':{'prechecks':1,'terminal':{'ok':True},'independent_streak':5},
        'finish':{'release_open_actions':1,'counts':{'movement':1,'home':1}},
        'actual_final_metrics':{'ready':True},
        'cache_reset':[{'queue_size':0,'proprio_is_none':True,'controller_use_delta':False,
                        'goal_pos_error':0.,'goal_ori_error':0.,'gripper_hold_error':0.}]}


def test_handoff_evidence_contract():
    result=x.handoff_evidence(recorded_handoff())
    assert result['ok'] is True
    assert all(result['checks'].values())
    assert x.handoff_evidence({})['ok'] is False


@pytest.mark.parametrize('path,value',[
    (('finish','release_open_actions'),None),
    (('finish','counts','movement'),0),
    (('finish','counts','home'),0),
    (('finish_status','prechecks'),0),
    (('finish_status','terminal','ok'),None),
    (('finish_status','independent_streak'),4),
    (('actual_final_metrics','ready'),False),
    (('source_counts','vla'),0),
    (('vla_calls',),0),
    (('cache_reset',0,'queue_size'),1),
    (('cache_reset',0,'proprio_is_none'),False),
    (('cache_reset',0,'controller_use_delta'),True),
    (('cache_reset',0,'goal_pos_error'),1e-7),
    (('cache_reset',0,'goal_ori_error'),1e-7),
    (('cache_reset',0,'gripper_hold_error'),None),
    (('cache_reset',0,'gripper_hold_error'),1e-7),
])
def test_handoff_evidence_rejects_missing_or_failed_measurement(path,value):
    recorded=recorded_handoff()
    target=recorded
    for key in path[:-1]:
        target=target[key]
    if value is None:
        del target[path[-1]]
    else:
        target[path[-1]]=value
    assert x.handoff_evidence(recorded)['ok'] is False


def test_load_policy_sets_identity_before_worker_ready(monkeypatch):
    svc=object.__new__(x.XvlaSceneService)
    svc._model_revision='old_detected_revision'
    client=object()
    monkeypatch.setattr(x,'make_client',lambda:client)
    svc._load_policy()
    assert svc._xvla_client is client
    assert svc._model_revision==pilot.REVISION


def bridge_fixture(tmp_path,fail=False):
    controller=SimpleNamespace(use_delta=False,update=Mock(),reset_goal=Mock())
    robot=SimpleNamespace(controller=controller,sim=SimpleNamespace(data=SimpleNamespace(time=12.5)))
    seen=[]
    port=SimpleNamespace(_started=False,output_dir=tmp_path,_robot_=lambda:robot,
                         step=Mock(),capture=Mock(),robot_shapes=Mock(),guarded_step=Mock())
    def start():
        if port._started:
            seen.append('idempotent')
            return
        seen.append(controller.use_delta)
        if fail:
            raise RuntimeError('original_start_failure')
        port._started=True
        robot.controller=SimpleNamespace(adapter=True)
    def close():
        seen.append('close')
        if port._started:
            robot.controller=controller
            port._started=False
    port.start,port.close=start,close
    return port,controller,seen


def test_mode_bridge_preserves_port_methods_and_restores(tmp_path):
    port,controller,seen=bridge_fixture(tmp_path)
    methods={k:getattr(port,k) for k in ('step','capture','robot_shapes','guarded_step')}
    assert x.bridge_port(port) is port
    port.start()
    assert seen==[True]
    assert controller.use_delta is True
    port.start()
    assert seen==[True,'idempotent']
    port.close()
    assert controller.use_delta is False
    assert controller.update.call_args.kwargs=={'force':True}
    controller.reset_goal.assert_called_once()
    assert all(getattr(port,k) is method for k,method in methods.items())
    import json
    rows=[json.loads(line) for line in (tmp_path/'controller_modes.jsonl').read_text().splitlines()]
    assert rows[0]=={'event':'start','before':False,'after':True,'timestamp':12.5}
    assert rows[1]=={'event':'close','before':True,'after':False,'timestamp':12.5}


def test_mode_bridge_start_exception_restores_and_rethrows(tmp_path):
    port,controller,seen=bridge_fixture(tmp_path,fail=True)
    x.bridge_port(port)
    with pytest.raises(RuntimeError,match='original_start_failure'):
        port.start()
    assert seen==[True]
    assert controller.use_delta is False
    controller.reset_goal.assert_called_once()
    assert 'start_error' in (tmp_path/'controller_modes.jsonl').read_text()


def test_mode_bridge_close_without_start_does_not_switch(tmp_path):
    port,controller,seen=bridge_fixture(tmp_path)
    x.bridge_port(port).close()
    assert controller.use_delta is False
    controller.update.assert_not_called()
    assert not (tmp_path/'controller_modes.jsonl').exists()


def test_vla_requires_absolute_before_request():
    svc=object.__new__(x.XvlaSceneService)
    svc._controller=lambda:SimpleNamespace(use_delta=True)
    svc._xvla_client=SimpleNamespace(step=Mock())
    svc._env=SimpleNamespace(action_source='finish')
    with pytest.raises(RuntimeError,match='xvla_requires_absolute_controller'):
        svc._select_action(({},'goal'))
    svc._xvla_client.step.assert_not_called()
    assert svc._env.action_source=='finish'
    svc._controller=lambda:SimpleNamespace(use_delta=False)
    svc._xvla_client.step.return_value=np.array([2.,3.,4.,0.,0.,0.,1.])
    assert svc._select_action(({},'goal'))[0]==2.
    assert svc._env.action_source=='vla'


def independent_prerequisite(tmp_path,monkeypatch):
    import hashlib,json
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    data=recorded_handoff()
    data['finish_status']['terminal']['ok']=False
    data['outcome']={'ok':False}
    data['job']={'success':False,'ended_reason':'final visual failed'}
    path=tmp_path/'pilot.json'
    def write():
        raw=json.dumps(data).encode()
        path.write_bytes(raw)
        return {'configuration':{'independent_case_prerequisite':{'path':'pilot.json','sha256':hashlib.sha256(raw).hexdigest()}}}
    return data,path,write


def test_default_independent_case_requires_complete(tmp_path):
    import json
    first=tmp_path/'wine_bowl_stove'
    first.mkdir()
    (first/'result.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(RuntimeError,match='first_chain_not_complete'):
        pilot.verify_case_prerequisite('basket_two',tmp_path,{})
    pilot.verify_case_prerequisite('wine_bowl_stove',tmp_path,{})


def test_independent_evidence_allows_new_session_without_changing_failure(tmp_path,monkeypatch):
    import json
    data,path,write=independent_prerequisite(tmp_path,monkeypatch)
    value=write()
    original=path.read_bytes()
    pilot.verify_case_prerequisite('basket_two',tmp_path,value)
    assert path.read_bytes()==original
    preserved=json.loads(original)
    assert preserved['outcome']['ok'] is False
    assert preserved['job']['success'] is False
    assert preserved['finish_status']['terminal']['ok'] is False


@pytest.mark.parametrize('field,value',[
    (('finish_status','prechecks'),None),(('source_counts','vla'),0),
    (('finish','release_open_actions'),None),(('finish_status','independent_streak'),4),
    (('actual_final_metrics','ready'),False),(('cache_reset',0,'queue_size'),1),
    (('cache_reset',0,'controller_use_delta'),True),(('job','ended_reason'),'other failure'),
    (('outcome','ok'),True),(('finish_status','terminal','ok'),True),
])
def test_independent_prerequisite_rejects_wrong_evidence(tmp_path,monkeypatch,field,value):
    data,path,write=independent_prerequisite(tmp_path,monkeypatch)
    target=data
    for key in field[:-1]:target=target[key]
    if value is None:del target[field[-1]]
    else:target[field[-1]]=value
    with pytest.raises(RuntimeError,match='independent_case_prerequisite_failed'):
        pilot.verify_case_prerequisite('mugs_two',tmp_path,write())


def test_independent_prerequisite_rejects_sha_and_external_path(tmp_path,monkeypatch):
    data,path,write=independent_prerequisite(tmp_path,monkeypatch)
    value=write()
    value['configuration']['independent_case_prerequisite']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='independent_case_prerequisite_failed'):
        pilot.verify_case_prerequisite('basket_two',tmp_path,value)
    value=write()
    value['configuration']['independent_case_prerequisite']['path']='../outside.json'
    with pytest.raises(RuntimeError,match='independent_case_prerequisite_failed'):
        pilot.verify_case_prerequisite('basket_two',tmp_path,value)
