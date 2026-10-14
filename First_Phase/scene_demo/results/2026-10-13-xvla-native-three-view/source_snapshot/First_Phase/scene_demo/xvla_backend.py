"""Isolated native X-VLA transport and persistent scene adapter."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

MODEL_REVISION = '129e71460678b7236cee6fc9707f09d9fa0c3590'
MODEL_PATH = '/home/yhwang/fyp/xvla_native/model'
SERVER_URL = 'http://127.0.0.1:8794/act'
BENCH_ROOT = Path(__file__).resolve().parent / 'results/2026-10-12-xvla-native-benchmark'
REFERENCE_SLOTS = {
    "xvla_basket_both": ("libero_10", 0),
    "xvla_mugs_both": ("libero_10", 4),
    "xvla_microwave_place_close": ("libero_10", 9),
    'wine_to_rack': ('libero_goal', 9), 'bowl_to_plate': ('libero_goal', 8),
    'stove_on': ('libero_goal', 7), 'soup_to_basket': ('libero_10', 0),
    'sauce_to_basket': ('libero_10', 0), 'white_mug_left': ('libero_10', 4),
    'white_mug_right': ('libero_10', 4), 'yellow_mug_right': ('libero_10', 4),
    'mug_to_microwave': ('libero_10', 9), 'microwave_close': ('libero_10', 9),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _append(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value) + '\n')
        handle.flush()


class NativeXvlaEnv:
    def __init__(self, suite_name: str, task_id: int, seed: int,
                 init_state_index: int, log_dir: Path, session_limit: int = 8000):
        self.suite_name, self.task_id = suite_name, task_id
        self.seed_value, self.init_state_index = seed, init_state_index
        self.log_dir, self.session_limit = Path(log_dir), session_limit
        self._env = None
        self._reset_count = 0
        self.action_source = 'finish'
        self.counts = dict.fromkeys(('settling', 'vla', 'finish', 'prepare'), 0)
        self.action_space = SimpleNamespace(low=np.array([-np.inf]*6+[-1.]),
                                           high=np.array([np.inf]*6+[1.]), shape=(7,))

    def reset(self, *, seed: int | None = None) -> tuple[dict, dict]:
        if self._reset_count:
            raise RuntimeError('persistent X-VLA session cannot reset')
        import evaluation.libero.libero_client as author
        suite = author.benchmark_dict[self.suite_name]()
        task = suite.get_task(self.task_id)
        bddl = Path(author.get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        initial = np.asarray(suite.get_task_init_states(self.task_id)[self.init_state_index])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._reset_count = 1
        self._env = author.OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                             camera_widths=256, control_freq=20,
                                             ignore_done=True, horizon=self.session_limit)
        self._env.seed(self.seed_value if seed is None else seed)
        self._env.reset()
        self._env.set_init_state(initial)
        metadata = {'reset_count': self._reset_count, 'suite': self.suite_name,
                    'task_id': self.task_id, 'init_state_index': self.init_state_index,
                    'seed': self.seed_value if seed is None else seed,
                    'bddl_path': str(bddl), 'bddl_sha256': _sha(bddl),
                    'init_sha256': hashlib.sha256(initial.tobytes()).hexdigest()}
        (self.log_dir / 'native_reset.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        for _ in range(10):
            self.action_source = 'settling'
            self.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
        self._env.env.robots[0].controller.use_delta = False
        return self._get_observation(), metadata

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        from . import joint_home as jh
        from . import preparation_diagnostics as pd
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('action must be finite (7,)')
        source = self.action_source
        if source not in self.counts:
            raise ValueError('unknown action source')
        if source == 'vla' and self.counts['vla'] >= 2400:
            raise RuntimeError('vla_action_budget_exhausted')
        if source in ('settling', 'prepare', 'finish') and sum(
                self.counts[key] for key in ('settling', 'prepare', 'finish')) >= 3000:
            raise RuntimeError('helper_action_budget_exhausted')
        before = pd.read_eef_pose(self)
        obs, reward, done, info = self._env.step(action)
        state = jh.read_robot_state(self)
        success = bool(self._env.check_success())
        self.counts[source] += 1
        _append(self.log_dir / 'native_steps.jsonl', {
            'action': action.tolist(), 'source': source, 'counts': dict(self.counts),
            'sim_time': float(self._env.env.sim.data.time), 'native_done': bool(done),
            'native_success': success, 'arm_qpos': state['arm_qpos'],
            'arm_qvel': state['arm_qvel'], 'finger_qpos': state['finger_qpos'],
            'finger_qvel': state['finger_qvel'], 'eef_before': before['position'],
            'eef_after': pd.read_eef_pose(self)['position']})
        self.action_source = 'finish'
        return obs, float(reward), False, False, {**info, 'is_success': success}

    def _get_observation(self) -> dict:
        return self._env.env._get_observations(force_update=True)

    def _format_raw_obs(self, obs: dict) -> dict:
        return obs

    def render(self) -> np.ndarray:
        return np.flip(self._get_observation()['agentview_image'], (0, 1))

    def close(self) -> None:
        if self._env is not None:
            self._env.close()


def make_client(host: str = '127.0.0.1', port: int = 8794) -> Any:
    import evaluation.libero.libero_client as author
    import requests
    import json_numpy

    class AuditedClient(author.ClientModel):
        def __init__(self):
            super().__init__(host, port)
            self.directory = None
            self.total_calls = self.job_calls = 0
            self.session = requests.Session()
            self.session.trust_env = False

        def _post(self, payload):
            self.total_calls += 1
            self.job_calls += 1
            directory = Path(self.directory) if self.directory is not None else None
            stem = f'{self.job_calls:03d}'
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
                with gzip.open(directory / f'request_{stem}.json.gz', 'wt', encoding='utf-8') as handle:
                    json.dump(payload, handle)
            response = self.session.post(self.url, json=payload, timeout=(10, 180))
            if directory is not None:
                (directory / f'response_{stem}.json').write_bytes(response.content)
                _append(directory / 'calls.jsonl', {'total_call': self.total_calls,
                        'job_call': self.job_calls, 'status_code': response.status_code})
            response.raise_for_status()
            data = json_numpy.loads(response.content)
            action = np.asarray(data['action'])
            if action.shape != (30, 20) or not np.isfinite(action).all():
                raise ValueError('response action must be finite (30,20)')
            return action

    return AuditedClient()


def reference_for(env: NativeXvlaEnv, capability_id: str, global_reference: dict) -> dict:
    from . import joint_home as jh
    import mujoco
    suite, task = REFERENCE_SLOTS[capability_id]
    directory = BENCH_ROOT / suite / f'task_{task:02d}' / 'episode_0/attempt_001'
    source, audit_path = directory / 'initial_state.npy', directory / 'initial_audit.json'
    array = np.load(source, allow_pickle=False)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    robot = env._env.env.robots[0]
    robot_model = type(robot.robot_model).__name__
    if audit['robot_model'] != robot_model:
        raise ValueError('reference robot model mismatch')
    state = jh.read_robot_state(env)
    indexes = state['qpos_indexes'] + state['finger_qpos_indexes']
    if indexes != list(range(9)):
        raise ValueError('reference requires ordered arm/finger qpos indexes 0..8')
    target = np.asarray(array[1 + np.asarray(indexes)], float)
    if target.shape != (9,) or not np.isfinite(target).all():
        raise ValueError('invalid reference qpos')
    sim = robot.sim
    before = (sim.data.qpos.tobytes(), sim.data.qvel.tobytes(), sim.data.time)
    raw = sim.model._model
    scratch = mujoco.MjData(raw)
    scratch.qpos[:] = sim.data.qpos[:]
    scratch.qvel[:] = sim.data.qvel[:]
    scratch.time = sim.data.time
    scratch.qpos[indexes] = target
    mujoco.mj_forward(raw, scratch)
    site = robot.eef_site_id
    if not isinstance(site, (int, np.integer)):
        site = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_SITE, robot.controller.eef_name)
    result = copy.deepcopy(global_reference)
    result.update(homeq=target[:7].tolist(), finger_home=target[7:].tolist(),
                  home_joint_speed=[0.]*7, finger_speed=[0.]*2,
                  eef_pose={'position': scratch.site_xpos[site].tolist(),
                            'orientation_matrix': scratch.site_xmat[site].reshape(3, 3).tolist()},
                  reference_kind='next_subtask_native_initial', capability_id=capability_id,
                  source_path=str(source), source_sha256=_sha(source),
                  audit_sha256=_sha(audit_path), source_robot_model=robot_model)
    if before != (sim.data.qpos.tobytes(), sim.data.qvel.tobytes(), sim.data.time):
        raise RuntimeError('live simulator changed during reference FK')
    return result


def reference_metrics(env, reference: dict) -> dict:
    from . import joint_home as jh
    from . import preparation_diagnostics as pd
    metrics = jh.home_metrics(reference, jh.read_robot_state(env))
    pose = pd.read_eef_pose(env)
    position = float(np.linalg.norm(np.asarray(pose['position']) - reference['eef_pose']['position']))
    orientation = pd.orientation_error_rad(pose['orientation_matrix'], reference['eef_pose']['orientation_matrix'])
    metrics.update(eef_position_error_m=position, eef_orientation_error_rad=orientation,
                   ready=bool(metrics['ready'] and position <= .005 and orientation is not None and orientation <= .05))
    return metrics
