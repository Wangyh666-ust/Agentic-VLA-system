"""Real service wrappers with inert fixtures: no simulator or model calls."""
import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scene_demo import catalog, service, xvla_service as x, xvla_finish, xvla_pilot
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS

class FakePort:
    def __init__(self, *args, **kwargs):
        self.original_start = Mock()
        self.start = self.original_start
        self.close = Mock()
        self.step = Mock()

@pytest.mark.parametrize('cid', list(NATIVE_LONG_SKILLS) + ['soup_to_basket'])
def test_real_monitor_constructs_whole_context(monkeypatch, tmp_path, cid):
    from scene_demo.finish_v1 import sensors, vision
    from scene_demo import safe_exit_contacts
    monkeypatch.setattr(sensors, 'SensorPort', FakePort)
    monkeypatch.setattr(vision, 'QwenClient', lambda **kw: SimpleNamespace())
    monkeypatch.setattr(x, 'reference_metrics', lambda *a: {'ready': True})
    monkeypatch.setattr(safe_exit_contacts, 'contacts', lambda env: [])
    states = {catalog.goal_key(g): True for g in catalog.CAPABILITIES[cid]['goals']}
    states['on|wine_bottle_1|wine_rack_1_top_region'] = True
    monkeypatch.setattr(service, 'eval_goal_predicate', lambda env, goal: states[catalog.goal_key(goal)])
    svc = object.__new__(x.XvlaSceneService)
    svc._actual_state = lambda: {'mock': True}
    svc._completion_ready = lambda env, goals, predicates: all(predicates.values())
    session = SimpleNamespace(home_reference={'finger_home':[.04,-.04]}, xvla_completed=['wine_to_rack'], scene_id=catalog.CAPABILITIES[cid]['scene_ids'][0])
    plan = SimpleNamespace(user_request='Explicit full requested skill')
    job = SimpleNamespace(capability_id=cid, run_dir=tmp_path)
    monitor = svc._make_finish_monitor(session, plan, job, object(), catalog.CAPABILITIES[cid], 800, [], None)
    from scene_demo.xvla_monitor import XvlaFinishMonitor
    assert isinstance(monitor,XvlaFinishMonitor)
    assert isinstance(monitor.call_client, xvla_finish.XvlaQwenClient)
    assert monitor.call_client.max_calls == 14
    assert monitor.port.checker_views == ('wrist', 'agentview', 'sideview')
    context = monitor.context
    assert context['criteria']['skill_description'] == catalog.CAPABILITIES[cid]['instruction']
    assert len(context['criteria']['success_checks']) == len(catalog.CAPABILITIES[cid]['goals'])
    last_goal = catalog.CAPABILITIES[catalog.CAPABILITIES[cid]['component_capability_ids'][-1]]['goals'][0] if cid in NATIVE_LONG_SKILLS else catalog.CAPABILITIES[cid]['goals'][-1]
    assert context['current_subtask']['target'] == last_goal[1]
    assert monitor.independent_ready() is True
    for key in list(states):
        states[key] = False
        assert monitor.independent_ready() is False
        states[key] = True
    monitor._orig_step.assert_not_called()
    monitor.port.original_start.assert_not_called()

@pytest.mark.parametrize('cid', list(NATIVE_LONG_SKILLS))
@pytest.mark.parametrize('next_id', [None, 'corresponding_atomic'])
def test_native_boundary_one_parent_call(monkeypatch, tmp_path, cid, next_id):
    if next_id:
        next_id = NATIVE_LONG_SKILLS[cid]['component_capability_ids'][0]
    svc = object.__new__(x.XvlaSceneService)
    client = SimpleNamespace(job_calls=0)
    svc._xvla_client = client
    svc._env = SimpleNamespace(counts={'vla':0})
    svc._cache_evidence = []
    svc._actual_state = lambda: {'mock':True}
    svc._prepare_first = Mock()
    svc._reset_policy_queues = Mock()
    reference_calls = []
    def reference(env, identifier, global_reference):
        reference_calls.append(identifier)
        return {'reference_kind': identifier}
    monkeypatch.setattr(x, 'reference_for', reference)
    monkeypatch.setattr(x, 'reference_metrics', lambda *a: {'ready':True})
    monkeypatch.setattr(x, 'handoff_evidence', lambda pilot: {'ok':True})
    monkeypatch.setattr(service, 'eval_goal_predicate', lambda env,goal: True)
    calls = []
    def parent(self, session, plan, job, identifier):
        calls.append({'id':identifier, 'instruction':catalog.CAPABILITIES[identifier]['instruction'],
                      'goals':copy.deepcopy(catalog.CAPABILITIES[identifier]['goals'])})
        job.success = True
        return {'ok':True}
    monkeypatch.setattr(service.SceneService, '_run_capability', parent)
    session = SimpleNamespace(xvla_global_reference={'home':'global'}, xvla_completed=[], home_reference={})
    plan = SimpleNamespace(capability_ids=[cid] + ([next_id] if next_id else []))
    job = service.JobRecord('j', 'r', 's', cid, tmp_path)
    assert svc._run_capability(session, plan, job, cid)['ok'] is True
    assert calls == [{'id':cid,'instruction':catalog.CAPABILITIES[cid]['instruction'],'goals':catalog.CAPABILITIES[cid]['goals']}]
    assert reference_calls == [cid] + ([next_id] if next_id else [])
    assert session.home_reference == ({'reference_kind':next_id} if next_id else {'home':'global','reference_kind':'global_home'})
    svc._prepare_first.assert_called_once()
    svc._reset_policy_queues.assert_called_once()
    assert session.xvla_completed == [cid]


def test_partial_wine_and_invalid_bodies(monkeypatch, tmp_path):
    svc = object.__new__(x.XvlaSceneService)
    svc.expected_capabilities = None
    rec = service.SessionRecord('s','basket_two',catalog.SCENES['basket_two'],0,0,tmp_path)
    rec.capability_backend = 'xvla'
    svc._sessions = {'s':rec}
    parent = Mock(return_value={'ok':True})
    monkeypatch.setattr(service.SceneService,'submit_plan',parent)
    for body in (None, [], 'bad'):
        svc.submit_plan(body)
        assert parent.call_args.args[0] is body
    base = {'session_id':'s','decision':'execute','capability_ids':['soup_to_basket']}
    svc.submit_plan(base)
    assert parent.call_args.args[0]['capability_ids'] == ['soup_to_basket']
    rec.scene_id = 'goal_table'
    rec.scene = catalog.SCENES['goal_table']
    wine = ['wine_to_rack','bowl_to_plate','stove_on']
    svc.submit_plan(dict(base,capability_ids=wine))
    assert parent.call_args.args[0]['capability_ids'] == wine
    assert parent.call_args.args[0]['_xvla_native_route']['replacements'] == []
    for ids in ([],['wine_to_rack']*2,['wine_to_rack']*7):
        payload = dict(base,capability_ids=ids)
        svc.submit_plan(payload)
        assert parent.call_args.args[0] is payload
        assert service.validate_capability_list('goal_table', ids, False, backend='xvla')['reason'] == 'invalid_capabilities'


def test_default_context_and_pilot_contract():
    svc = object.__new__(service.SceneService)
    assert svc._build_finish_context(catalog.CAPABILITIES['soup_to_basket'], 'Put soup in basket')['current_subtask']['target'] == 'alphabet_soup_1'
    with pytest.raises(ValueError, match='exactly one goal'):
        svc._build_finish_context(catalog.CAPABILITIES['xvla_basket_both'], 'Put both in basket')
    assert xvla_pilot.CASES['wine_bowl_stove']['capabilities'] == ['wine_to_rack','bowl_to_plate','stove_on']
    for case, cid in [('basket_two','xvla_basket_both'),('mugs_two','xvla_mugs_both'),('microwave_mug','xvla_microwave_place_close')]:
        assert xvla_pilot.CASES[case]['capabilities'] == [cid]
    assert '2026-10-13-xvla-process-evidence/pilot' in str(xvla_pilot.DEFAULT_OUTPUT)
    sources = {p.name for p in xvla_pilot.frozen_sources()}
    assert {'xvla_skills.py','xvla_finish.py','xvla_process_evidence.py','xvla_monitor.py'} <= sources


@pytest.mark.parametrize('body', [None, [], 'bad'])
def test_nonobject_body_real_parent_rejection(body):
    svc = object.__new__(x.XvlaSceneService)
    assert svc.submit_plan(body)['reason'] == 'invalid_body'


@pytest.mark.parametrize('condition', [False, None, 'error', True])
def test_moka_precondition_no_automatic_ignition(monkeypatch, tmp_path, condition):
    cid = 'xvla_moka_both'
    svc = object.__new__(x.XvlaSceneService)
    svc._xvla_client = SimpleNamespace(job_calls=0)
    svc._env = SimpleNamespace(counts={'vla':0})
    svc._cache_evidence = []
    svc._actual_state = lambda: {'mock':True}
    svc._prepare_first = Mock()
    svc._reset_policy_queues = Mock()
    parent = Mock(return_value={'ok':False})
    monkeypatch.setattr(service.SceneService,'_run_capability',parent)
    def evaluate(env, goal):
        if condition == 'error':
            raise ValueError('unknown fixture state')
        return condition
    monkeypatch.setattr(service,'eval_goal_predicate',evaluate)
    monkeypatch.setattr(x,'reference_for',lambda *a:{})
    monkeypatch.setattr(x,'reference_metrics',lambda *a:{'ready':True})
    session = SimpleNamespace(xvla_completed=[],xvla_global_reference={},home_reference={})
    plan = SimpleNamespace(capability_ids=[cid])
    job = service.JobRecord('j','r','s',cid,tmp_path)
    result = svc._run_capability(session,plan,job,cid)
    if condition is True:
        parent.assert_called_once()
        svc._prepare_first.assert_called_once()
    else:
        assert result['reason'] == 'native_precondition_failed'
        parent.assert_not_called()
        svc._prepare_first.assert_not_called()
        svc._reset_policy_queues.assert_not_called()
    assert svc._env.counts['vla'] == 0
