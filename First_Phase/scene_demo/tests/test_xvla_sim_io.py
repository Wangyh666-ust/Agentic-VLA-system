"""Native boundary mocked; no simulator or model initialization."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from scene_demo.xvla_backend import NativeXvlaEnv
from scene_demo.xvla_sim_io import CachedNativeXvlaEnv


@pytest.fixture
def env(monkeypatch, tmp_path):
    env = CachedNativeXvlaEnv('fixture', 0, 0, 0, tmp_path)
    data = SimpleNamespace(time=0., **{n: np.zeros(2) for n in
        ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')})
    model = SimpleNamespace(**{n: np.zeros(2) for n in ('cam_pos', 'cam_quat', 'cam_fovy')})
    env._env = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data, model=model)))
    getter = Mock(side_effect=lambda: {'agentview_image': np.arange(12).reshape(2, 2, 3).copy()})
    monkeypatch.setattr(NativeXvlaEnv, '_get_observation', getter)
    return env, getter


def test_same_state_defensive_copies_and_render(env):
    e, getter = env
    first = e.get_current_observation()
    first['agentview_image'][:] = -1
    second = e.get_current_observation()
    assert second['agentview_image'][0, 0, 0] == 0
    assert getter.call_count == 1
    np.testing.assert_array_equal(e.render(), np.flip(second['agentview_image'], (0, 1)))
    assert e.observation_cache_hits == 2 and e.observation_cache_misses == 1


@pytest.mark.parametrize('group,name', [('data', n) for n in
    ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')] +
    [('model', n) for n in ('cam_pos', 'cam_quat', 'cam_fovy')])
def test_same_time_state_change_invalidates(env, group, name):
    e, getter = env
    e.get_current_observation()
    getattr(getattr(e._env.env.sim, group), name)[0] = 1
    e.get_current_observation()
    assert getter.call_count == 2


def test_time_step_reset_and_failed_step_invalidation(env, monkeypatch):
    e, getter = env
    e.get_current_observation()
    e._env.env.sim.data.time = 1
    e.get_current_observation()
    assert getter.call_count == 2
    result = ({'agentview_image': np.ones((2, 2, 3))}, 1., False, False, {})
    step = Mock(return_value=result)
    monkeypatch.setattr(NativeXvlaEnv, 'step', step)
    assert e.step('action') is result
    result[0]['agentview_image'][:] = 9
    assert np.all(e.get_current_observation()['agentview_image'] == 1)
    step.assert_called_once_with('action')
    assert e.native_step_count == 1 and e.native_step_seconds >= 0
    step.side_effect = ValueError('failed')
    with pytest.raises(ValueError):
        e.step('bad')
    e.get_current_observation()
    assert getter.call_count == 3
    monkeypatch.setattr(NativeXvlaEnv, 'reset', Mock(return_value=({}, {})))
    e.reset(seed=0)
    e.get_current_observation()
    assert getter.call_count == 4


def test_uninitialized_falls_back(env):
    e, getter = env
    e._env = None
    e.get_current_observation(); e.get_current_observation()
    assert getter.call_count == 2


def test_visual_selection_and_refresh_unchanged(monkeypatch, tmp_path):
    from scene_demo import xvla_service as x, service
    svc = object.__new__(x.XvlaSceneService)
    svc._session_run_dir = tmp_path
    native, cached = Mock(), Mock()
    monkeypatch.setattr(x, 'NativeXvlaEnv', native)
    monkeypatch.setattr(x, 'CachedNativeXvlaEnv', cached)
    svc._build_env('s', 0, 0, 0)
    native.assert_called_once()
    cached.assert_not_called()
    svc.checker_mode = 'simulator'
    svc._build_env('s', 0, 0, 0)
    cached.assert_called_once()
    svc.checker_mode = 'visual'
    refresh = Mock(return_value={'native': True})
    monkeypatch.setattr(service.SceneService, '_refresh_observation', refresh)
    assert svc._refresh_observation(object()) == {'native': True}
    refresh.assert_called_once()


def test_protected_control_hashes():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / 'results/2026-10-14-sim-fast/protected_controls.json').read_text())
    assert len(expected) == 10
    for relative, digest in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
