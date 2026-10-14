"""Software-only native gate and inherited controller boundary tests."""
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json
from pathlib import Path
import pytest
from scene_demo import xvla_sim_finish as sim
from scene_demo.finish_v1 import control, vision


@pytest.fixture
def monitor(monkeypatch,tmp_path):
    from scene_demo.finish_v1 import sensors
    port=SimpleNamespace(step=Mock(),close=Mock(),phase='observing')
    robot={'finger_qpos':[.04,-.04],'arm_qvel':[0.]*7}
    port.capture=Mock(return_value={'timestamp':1.,'robot':robot,'frames':[{'view':'wrist','rgb_path':'w'}]})
    port.read_robot=Mock(return_value=robot)
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


def test_no_periodic_history_but_every_action_independent(monitor):
    monitor.independent_ready = Mock(side_effect=[True, True, False] + [True]*5)
    monitor.on_action = Mock()
    for i in range(8):
        monitor._on_action({'phase':'home'})
        if i == 2:
            assert monitor.independent_streak == 0
    assert monitor.independent_streak == 5
    assert monitor.independent_ready.call_count == monitor.on_action.call_count == 8
    assert monitor.helper_counts['home'] == 8
    monitor.record(5); monitor.record(5); monitor.record(10)
    monitor.port.capture.assert_not_called()
    assert len(monitor.history) == 0
    status = monitor.status()
    assert status['skipped_policy_captures'] == 2
    assert status['skipped_helper_captures'] == 1


def test_original_visual_history_capture_at_five(monitor):
    from scene_demo.finish_v1.runtime import FinishMonitor
    FinishMonitor.record(monitor, 5)
    monitor.port.capture.assert_called_once()
    assert len(monitor.history) == 1


def test_release_triplet_and_original_hashes(monitor, monkeypatch, tmp_path):
    from scene_demo import xvla_grounding
    frames = []
    for view in ('wrist', 'agentview', 'sideview'):
        path = tmp_path / (view+'.png')
        path.write_bytes(view.encode())
        frames.append({'view':view, 'rgb_path':str(path)})
    cap = {'timestamp':1., 'robot':{'finger_qpos':[.04,-.04]}, 'frames':[frames[0]]}
    monitor.port.capture.return_value = cap
    monitor.port.env, monitor.port.output_dir = monitor.env, tmp_path
    render = Mock(return_value=frames[1:])
    monkeypatch.setattr(xvla_grounding, 'render_external', render)
    monitor.release_external_views = True
    monitor.port.read_robot.side_effect = [ {'finger_qpos':[0.,0.]},
        {'finger_qpos':[.04,-.04]}, {'finger_qpos':[.04,-.04]} ]
    for step in (1,2,3):
        monitor.record(step)
    assert monitor.release_events == 1
    monitor.port.capture.assert_called_once()
    render.assert_called_once_with(monitor.env, tmp_path, 'wrist_checker')
    event = json.loads((monitor.output_dir/'process_events.jsonl').read_text())
    assert len(event['images']) == 3
    for frame, row in zip(frames, event['images']):
        assert row['sha256'] == hashlib.sha256(Path(frame['rgb_path']).read_bytes()).hexdigest()
