"""Pure adapter contracts: no simulator construction or network transport."""
import collections
import json
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

from First_Phase.scene_demo import xvla_backend as xb


@pytest.fixture
def client(monkeypatch):
    class Author:
        def __init__(self, host, port):
            self.url = f'http://{host}:{port}/act'
            self.processor = SimpleNamespace(Rotate6D_to_AxisAngle=lambda a: a[:, :3])
            self.reset()

        def reset(self):
            self.proprio = None
            self.action_plan = collections.deque()

        def _format_query(self, obs, goal):
            if self.proprio is None:
                self.proprio = np.zeros(20)
            return {'domain_id': 3, 'steps': 10, 'language_instruction': goal}

        def step(self, obs, goal):
            if not self.action_plan:
                action = self._post(self._format_query(obs, goal))
                self.proprio[:9] = action[-1, :9].copy()
                final = np.concatenate([action[:, :3], self.processor.Rotate6D_to_AxisAngle(action[:, 3:9]), action[:, 9:10]], axis=-1)
                self.action_plan.extend(final.tolist())
            result = np.array(self.action_plan.popleft(), dtype=np.float32)
            result[-1] = 1. if result[-1] > .5 else -1.
            return result

    author = ModuleType('evaluation.libero.libero_client')
    author.ClientModel = Author
    for name in ('evaluation', 'evaluation.libero'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, author.__name__, author)
    monkeypatch.setitem(sys.modules, 'json_numpy', SimpleNamespace(loads=lambda b: json.loads(b)))
    return xb.make_client()


def response(client, action):
    content = json.dumps({'action': np.asarray(action).tolist()}).encode()
    client.session.post = lambda *a, **kw: SimpleNamespace(content=content, status_code=200, raise_for_status=lambda: None)


def test_absolute_action_space():
    env = xb.NativeXvlaEnv('libero_goal', 9, 142, 0, 'unused')
    action = np.array([2, -3, 4, 2, 3, -4, -1.])
    np.testing.assert_array_equal(np.clip(action, env.action_space.low, env.action_space.high), action)
    raw = {'agentview_image': object()}
    assert env._format_raw_obs(raw) is raw


def test_vla_budget_before_uninitialized_environment():
    env = xb.NativeXvlaEnv('libero_goal', 9, 142, 0, 'unused')
    env.action_source = 'vla'
    env.counts['vla'] = 2400
    with pytest.raises(RuntimeError, match='^vla_action_budget_exhausted$'):
        env.step(np.zeros(7))
    assert env._env is None and env.counts['vla'] == 2400


@pytest.mark.parametrize('source', ['finish', 'prepare', 'settling'])
def test_helper_budget_before_uninitialized_environment(source):
    env = xb.NativeXvlaEnv('libero_goal', 9, 142, 0, 'unused')
    env.action_source = source
    env.counts.update(settling=10, prepare=90, finish=2900)
    with pytest.raises(RuntimeError, match='^helper_action_budget_exhausted$'):
        env.step(np.zeros(7))
    assert env._env is None
    assert sum(env.counts[key] for key in ('settling', 'prepare', 'finish')) == 3000


def test_inherited_conversion_and_reset(client, tmp_path):
    client.directory = tmp_path
    action = np.ones((30, 20))
    action[:, 9] = .6
    action[1, 9] = .5
    response(client, action)
    first = client.step({}, 'goal')
    assert first.shape == (7,) and first.dtype == np.float32 and first[-1] == 1
    assert client.step({}, 'goal')[-1] == -1
    assert client.total_calls == client.job_calls == 1
    np.testing.assert_array_equal(client.proprio[:9], action[-1, :9])
    assert len(client.action_plan) == 28
    assert (tmp_path / 'request_001.json.gz').is_file()
    assert (tmp_path / 'response_001.json').is_file()
    client.reset()
    assert client.proprio is None and not client.action_plan


@pytest.mark.parametrize('shape,nan', [((29, 20), False), ((30, 10), False), ((30, 20), True)])
def test_response_contract(client, shape, nan):
    action = np.zeros(shape)
    if nan:
        action[0, 0] = np.nan
    response(client, action)
    with pytest.raises(ValueError, match='finite'):
        client.step({}, 'goal')
    assert not client.action_plan


def test_reference_private_fk(monkeypatch, tmp_path):
    from First_Phase.scene_demo import joint_home as jh
    monkeypatch.setattr(xb, 'BENCH_ROOT', tmp_path)
    directory = tmp_path / 'libero_goal/task_09/episode_0/attempt_001'
    directory.mkdir(parents=True)
    np.save(directory / 'initial_state.npy', np.arange(30, dtype=float))
    (directory / 'initial_audit.json').write_text('{"robot_model":"Panda"}')
    data = SimpleNamespace(qpos=np.zeros(29), qvel=np.ones(29), time=12.)
    sim = SimpleNamespace(data=data, model=SimpleNamespace(_model=object()))
    robot = SimpleNamespace(robot_model=type('Panda', (), {})(), sim=sim, eef_site_id=0)
    env = SimpleNamespace(_env=SimpleNamespace(env=SimpleNamespace(robots=[robot])))
    monkeypatch.setattr(jh, 'read_robot_state', lambda e: {'qpos_indexes': list(range(7)), 'finger_qpos_indexes': [7, 8]})
    def scratch(raw):
        return SimpleNamespace(qpos=np.zeros(29), qvel=np.zeros(29), time=0., site_xpos=np.zeros((1, 3)), site_xmat=np.eye(3).reshape(1, 9))
    def forward(raw, d):
        d.site_xpos[0] = d.qpos[:3]
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(MjData=scratch, mj_forward=forward))
    before = (data.qpos.tobytes(), data.qvel.tobytes(), data.time)
    original = {'homeq': [0.]*7, 'finger_home': [0.]*2}
    reference = xb.reference_for(env, 'wine_to_rack', original)
    assert reference['homeq'] == list(range(1, 8))
    assert reference['finger_home'] == [8., 9.]
    assert reference['eef_pose']['position'] == [1., 2., 3.]
    assert reference['home_joint_speed'] == [0.]*7
    assert reference['reference_kind'] == 'next_subtask_native_initial'
    assert original['homeq'] == [0.]*7
    assert before == (data.qpos.tobytes(), data.qvel.tobytes(), data.time)
