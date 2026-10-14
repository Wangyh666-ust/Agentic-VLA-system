"""Independent installed LIBERO source checks; no simulator construction."""
import ast
import builtins
import copy
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scene_demo import catalog, service, xvla_backend, xvla_service as x, xvla_pilot
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS, NATIVE_ATOMIC_CAPABILITIES, NATIVE_SCENES

LIBERO = Path('/home/yhwang/fyp/xvla_native/LIBERO/libero/libero')

def mapping():
    tree = ast.parse((LIBERO/'benchmark/libero_suite_task_map.py').read_text())
    node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets))
    return ast.literal_eval(node.value)['libero_10']

@pytest.mark.parametrize('cid', list(NATIVE_LONG_SKILLS))
def test_installed_task_and_reference(cid):
    skill = NATIVE_LONG_SKILLS[cid]
    task = skill['native_task']['task_id']
    assert mapping()[task].lower().endswith(skill['instruction'].replace(' ', '_'))
    files = list((LIBERO/'bddl_files/libero_10').glob('*'+skill['instruction'].replace(' ','_')+'.bddl'))
    assert len(files) == 1
    content = files[0].read_text()
    assert ' '.join(re.search(r'\(:language\s+(.*?)\)',content,re.S).group(1).split()) == skill['instruction']
    parsed = re.findall(r'\((in|on|close|turnon)\s+([^()]+)\)',content.split('(:goal',1)[1].lower())
    assert Counter((op,*args.split()) for op,args in parsed) == Counter(map(tuple,skill['goals']))
    scene = catalog.SCENES[skill['scene_ids'][0]]
    assert (scene['suite'],scene['task_id']) == ('libero_10',task)
    preserved = [g for check in skill.get('preserved_checks',[]) for g in check['goals']]
    component_goals = [g for component in skill['component_capability_ids'] for g in catalog.CAPABILITIES[component]['goals']]
    assert Counter(map(tuple,component_goals+preserved)) == Counter(map(tuple,skill['goals']))
    assert xvla_backend.REFERENCE_SLOTS[cid] == ('libero_10',task)
    for component in skill['component_capability_ids']:
        assert xvla_backend.REFERENCE_SLOTS[component] == ('libero_10',task)
        if component in NATIVE_ATOMIC_CAPABILITIES:
            assert catalog.CAPABILITIES[component]['scene_ids'] == skill['scene_ids']
            assert catalog.CAPABILITIES[component]['backends'] == ['xvla']
    directory = xvla_backend.BENCH_ROOT/'libero_10'/('task_%02d'%task)/'episode_0/attempt_001'
    assert (directory/'initial_state.npy').is_file()
    assert (directory/'initial_audit.json').is_file()
    if task == 8:
        initial = content.split('(:init',1)[1].split('(:goal',1)[0].lower()
        assert '(turnon flat_stove_1)' in initial
        assert skill['required_initial_goals'] == [['turnon','flat_stove_1']]
        assert all(catalog.CAPABILITIES[c]['goals'][0][0]=='on' for c in skill['component_capability_ids'])


def test_all_coverage_and_baseline_values():
    assert {s['native_task']['task_id'] for s in NATIVE_LONG_SKILLS.values()} == set(range(10))
    assert len(NATIVE_SCENES) == 7 and len(NATIVE_ATOMIC_CAPABILITIES) == 13
    root = Path(catalog.__file__).parent/'results/2026-10-13-xvla-native-three-view/source_snapshot/First_Phase/scene_demo'
    old_skills = {}
    exec(compile((root/'xvla_skills.py').read_text(),str(root/'xvla_skills.py'),'exec'),old_skills)
    def baseline_import(name,*args,**kwargs):
        if name == 'scene_demo.xvla_skills':
            return SimpleNamespace(NATIVE_LONG_SKILLS=old_skills['NATIVE_LONG_SKILLS'])
        return builtins.__import__(name,*args,**kwargs)
    old = {'__builtins__':dict(vars(builtins),__import__=baseline_import)}
    exec(compile((root/'catalog.py').read_text(),str(root/'catalog.py'),'exec'),old)
    assert len(old['SCENES']) == 8
    assert len(old['CAPABILITIES']) == 16
    for scene,value in old['SCENES'].items():
        assert catalog.SCENES[scene] == value
    for cid,value in old['CAPABILITIES'].items():
        assert catalog.CAPABILITIES[cid] == value
    assert len(xvla_pilot.CASES) == 11
    assert all(c['budget_per_subgoal']==(800 if name=='wine_bowl_stove' else 900) for name,c in xvla_pilot.CASES.items())


def test_scene_isolation_and_mcp_reachability():
    smol = object.__new__(service.SceneService)
    smol._build_env = Mock()
    smol._sync_work = Mock()
    assert not set(NATIVE_SCENES) & {s['scene_id'] for s in smol.scenes()['scenes']}
    for scene in NATIVE_SCENES:
        assert catalog.scene_capabilities(scene) == []
        assert smol.create_session(scene)['reason'] == 'unsupported_backend'
    smol._build_env.assert_not_called()
    smol._sync_work.assert_not_called()
    xvla = object.__new__(x.XvlaSceneService)
    xvla.capability_backend = 'xvla'
    public_scenes = {s['scene_id'] for s in xvla.scenes()['scenes']}
    tree = ast.parse(Path(catalog.__file__).with_name('mcp_server.py').read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_atomic_capabilities')
    from typing import Any
    namespace = {'Any':Any,'NATIVE_LONG_SKILLS':NATIVE_LONG_SKILLS}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'mcp_filter','exec'),namespace)
    for cid,skill in NATIVE_LONG_SKILLS.items():
        scene = skill['scene_ids'][0]
        assert scene in public_scenes
        public = catalog.scene_capabilities(scene,backend='xvla')
        assert cid in [c['capability_id'] for c in namespace['_atomic_capabilities'](public)]
