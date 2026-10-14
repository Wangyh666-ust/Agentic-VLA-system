"""Software interface checks; no simulation or model requests."""
import copy
import re
from pathlib import Path
from unittest.mock import Mock
import pytest
from scene_demo import catalog, service, xvla_service, xvla_backend
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS, NATIVE_SKILL_GUIDANCE, route_capabilities

@pytest.mark.parametrize('identifier', list(NATIVE_LONG_SKILLS))
def test_routes(identifier):
    skill = NATIVE_LONG_SKILLS[identifier]
    scene = skill['scene_ids'][0]
    components = skill['component_capability_ids']
    original = copy.deepcopy(components)
    route = route_capabilities(scene, components)
    assert route['requested_capability_ids'] == original
    assert route['executed_capability_ids'] == [identifier]
    assert components == original
    assert route['replacements'][0]['start_index'] == 0
    assert route_capabilities(scene, [identifier])['executed_capability_ids'] == [identifier]
    for ids in (components[:1], list(reversed(components)), [components[0], 'other', components[1]]):
        assert route_capabilities(scene, ids)['executed_capability_ids'] == ids
    assert route_capabilities(scene, components, [])['executed_capability_ids'] == components
    assert route_capabilities('goal_table', components)['executed_capability_ids'] == components
    assert catalog.deduplicate_goals(components) == skill['goals']
    route['requested_capability_ids'].clear()
    assert components == original

@pytest.mark.parametrize('identifier', list(NATIVE_LONG_SKILLS))
def test_catalog_backend_and_bddl(identifier):
    skill = NATIVE_LONG_SKILLS[identifier]
    scene = skill['scene_ids'][0]
    assert identifier not in [c['capability_id'] for c in catalog.scene_capabilities(scene)]
    assert identifier in [c['capability_id'] for c in catalog.scene_capabilities(scene, backend='xvla')]
    assert service.validate_capability_list(scene, [identifier], False)['reason'] == 'unsupported_backend'
    assert service.validate_capability_list(scene, [identifier], False, backend='xvla') is None
    assert xvla_backend.REFERENCE_SLOTS[identifier] == ('libero_10', skill['native_task']['task_id'])
    root = Path('/home/yhwang/fyp/xvla_native/LIBERO/libero/libero/bddl_files/libero_10')
    files = list(root.glob('*' + skill['instruction'].replace(' ', '_') + '.bddl'))
    assert len(files) == 1
    content = files[0].read_text()
    language = re.search(r'\(:language\s+(.*?)\)', content, re.S).group(1)
    assert ' '.join(language.split()) == skill['instruction']
    goal = content.split('(:goal', 1)[1].lower()
    actual = re.findall(r'\((in|on|close)\s+([^()]+)\)', goal)
    assert [[predicate] + args.split() for predicate, args in actual] == skill['goals']


def test_public_and_route_snapshot(tmp_path):
    record = service.SessionRecord('s', 'basket_two', catalog.SCENES['basket_two'], 0, 0, tmp_path)
    assert record.public()['capability_backend'] == 'smolvla'
    assert 'xvla_basket_both' not in record.capability_ids()
    record.capability_backend = 'xvla'
    assert 'xvla_basket_both' in record.capability_ids()
    route = route_capabilities('basket_two', ['soup_to_basket', 'sauce_to_basket'])
    plan = service.PlanRecord('r', 's', {'decision':'execute', 'capability_ids':route['executed_capability_ids'], '_xvla_native_route':route})
    assert plan.public()['native_skill_route'] == route
    route['requested_capability_ids'].clear()
    assert plan.public()['native_skill_route']['requested_capability_ids'] == ['soup_to_basket', 'sauce_to_basket']
    assert 'native_skill_route' not in service.PlanRecord('r', 's', {}).public()


def test_submit_routing(monkeypatch, tmp_path):
    svc = object.__new__(xvla_service.XvlaSceneService)
    record = service.SessionRecord('s', 'basket_two', catalog.SCENES['basket_two'], 0, 0, tmp_path)
    record.capability_backend = 'xvla'
    svc._sessions = {'s': record}
    svc.expected_capabilities = ['xvla_basket_both']
    parent = Mock(return_value={'ok':True})
    monkeypatch.setattr(service.SceneService, 'submit_plan', parent)
    payload = {'session_id':'s', 'decision':'execute', 'capability_ids':['soup_to_basket','sauce_to_basket']}
    original = copy.deepcopy(payload)
    assert svc.submit_plan(payload) == {'ok':True}
    assert payload == original
    forwarded = parent.call_args.args[0]
    assert forwarded['capability_ids'] == ['xvla_basket_both']
    assert forwarded['_xvla_native_route']['requested_capability_ids'] == original['capability_ids']
    assert svc.submit_plan(dict(payload, capability_ids=['sauce_to_basket','soup_to_basket']))['reason'] == 'pilot_plan_mismatch'
    svc.expected_capabilities = None
    for ids in (['soup_to_basket'] * 2, ['soup_to_basket'] * 7, 'bad'):
        bad = dict(payload, capability_ids=ids)
        svc.submit_plan(bad)
        assert parent.call_args.args[0] is bad


def test_shared_guidance():
    from scene_demo import run_agent, setup_profile
    assert NATIVE_SKILL_GUIDANCE in run_agent.INITIAL_PROMPT_HEAD
    assert NATIVE_SKILL_GUIDANCE in run_agent.REPAIR_PROMPT_HEAD
    assert NATIVE_SKILL_GUIDANCE in setup_profile.SOUL_MD


def test_mcp_filter():
    import ast
    from types import SimpleNamespace
    from typing import Any
    source = Path(catalog.__file__).with_name('mcp_server.py')
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_atomic_capabilities')
    namespace = {'Any':Any, 'NATIVE_LONG_SKILLS':NATIVE_LONG_SKILLS}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    mcp_server = SimpleNamespace(_atomic_capabilities=namespace['_atomic_capabilities'])
    entries = catalog.scene_capabilities('basket_two', include_audit=True, backend='xvla')
    filtered = mcp_server._atomic_capabilities(entries)
    assert 'xvla_basket_both' in [c['capability_id'] for c in filtered]
    assert 'basket_both' not in [c['capability_id'] for c in filtered]
    forged = dict(next(c for c in entries if c['capability_id']=='xvla_basket_both'), capability_id='unregistered')
    assert mcp_server._atomic_capabilities([forged]) == []
