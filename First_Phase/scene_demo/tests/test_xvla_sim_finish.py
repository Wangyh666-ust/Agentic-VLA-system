"""Software-only native gate and inherited controller boundary tests."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scene_demo import xvla_sim_finish as sim
from scene_demo.finish_v1 import control, vision


@pytest.fixture
def monitor(monkeypatch,tmp_path):
    from scene_demo.finish_v1 import sensors
    port=SimpleNamespace(step=Mock(),close=Mock(),phase='observing')
    robot={'finger_qpos':[.04,-.04],'arm_qvel':[0.]*7}
    port.capture=Mock(return_value={'timestamp':1.,'robot':robot,'frames':[{'view':'wrist','rgb_path':'w'}]})
    monkeypatch.setattr(sensors,'SensorPort',lambda *a,**kw:port)
    m=sim.SimFinishMonitor(object(),{'finger_home':[.04,-.04]},tmp_path,{},lambda f:f(),None,lambda:True,20)
    m.configure_simulation([['on','current','target']],[['on','prior','target']])
    from scene_demo import service
    monkeypatch.setattr(service,'eval_goal_predicate',lambda env,g:True)
    monkeypatch.setattr(vision.QwenClient,'_request',Mock(side_effect=AssertionError('no Qwen')))
    return m


def test_five_unique_and_duplicate(monitor):
    assert [monitor.visual_check_due(i) for i in range(1,5)]==[False]*4
    assert monitor.visual_check_due(4) is False and monitor.candidate_streak==4
    assert monitor.visual_check_due(5) is True
    assert monitor.trigger_count==1 and monitor.sample_count==5


@pytest.mark.parametrize('value',[False])
def test_prior_false_or_unknown_blocks(monitor,monkeypatch,value):
    from scene_demo import service
    monkeypatch.setattr(service,'eval_goal_predicate',lambda env,g:True if g[1]=='current' else value)
    assert not any(monitor.visual_check_due(i) for i in range(1,8))
    assert monitor.candidate_streak==0


def test_exception_false_and_reset(monitor,monkeypatch):
    from scene_demo import service
    assert monitor.visual_check_due(1) is False
    monkeypatch.setattr(service,'eval_goal_predicate',Mock(side_effect=RuntimeError('unknown')))
    with pytest.raises(RuntimeError,match='unknown'):monitor.poll(2)


@pytest.mark.parametrize('value',[None,'true'])
def test_unknown_invalid_stops_before_controller(monitor,monkeypatch,value):
    from scene_demo import service
    monkeypatch.setattr(service,'eval_goal_predicate',lambda env,g:value)
    run=Mock();monkeypatch.setattr(control,'run_finish',run)
    with pytest.raises(ValueError,match='invalid simulator predicate'):monitor.poll(1)
    run.assert_not_called()


def test_poll_original_controller_after_gate_and_final_changed(monitor,monkeypatch):
    from scene_demo import service
    run=Mock(return_value={'ok':True});monkeypatch.setattr(control,'run_finish',run)
    for i in range(1,5):assert monitor.poll(i) is None
    run.assert_not_called()
    def finish(port,ctx,verdict,decide,check,**kw):
        monkeypatch.setattr(service,'eval_goal_predicate',lambda env,g:False)
        assert check(ctx,[])['operation_achieved'] is False
        return {'ok':False,'reason':'final visual failed'}
    run.side_effect=finish
    result=monitor.poll(5)
    assert result['ok'] is False and result['reason']=='final_simulator_check_failed'
    assert monitor._finish['handoff_mode']=='simulator_predicates'
    assert monitor.status()['calls']==0 and monitor.status()['actual_visual_requests']==0
    assert monitor.status()['simulator_checks']==2
    vision.QwenClient._request.assert_not_called()


def test_budget_final_check_does_not_bypass_five(monitor,monkeypatch):
    run=Mock();monkeypatch.setattr(control,'run_finish',run)
    assert monitor.poll(20) is None
    run.assert_not_called()


def test_goal_configuration_copied(monitor):
    goals=[['on','new','target']];monitor.configure_simulation(goals,[]);goals[0][1]='changed'
    assert monitor.goals[0][1]=='new'
    with pytest.raises(ValueError):monitor.configure_simulation([],[])
