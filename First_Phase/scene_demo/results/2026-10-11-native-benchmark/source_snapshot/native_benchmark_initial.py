"""Fixed native LIBERO benchmark; SDK owns rollout and scoring."""
import csv
import json
import time
import traceback
import dataclasses
import importlib.metadata
import subprocess
from pathlib import Path

import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed

OUT = Path('/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark')
MODEL = Path('/home/yhwang/fyp/libero_demo/models/smolvla_libero')
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']


def serial(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=serial), encoding='utf-8')
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if list(OUT.glob('*/eval_info.json')):
        raise RuntimeError('Existing eval_info.json: duplicate launch prohibited')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = SmolVLAConfig.from_pretrained(str(MODEL))
    cfg.pretrained_path = MODEL
    cfg.device = 'cuda'
    cfg.load_vlm_weights = False
    cfg.use_amp = False
    assert (cfg.chunk_size, cfg.n_action_steps, cfg.num_steps) == (50, 1, 10)
    save(OUT / 'run_metadata.json', {'policy_config': dataclasses.asdict(cfg),
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ['lerobot', 'hf-libero', 'mujoco', 'robosuite', 'torch', 'transformers']},
        'implementation_model': 'gpt-6.1-sol', 'implementation_reasoning_effort': 'low'})
    initial_env = LiberoEnv(task=SUITES[0], task_ids=[0], observation_height=256,
                            observation_width=256, control_mode='relative', init_states=True, hard_reset=True)
    policy = make_policy(cfg=cfg, env_cfg=initial_env, rename_map={})
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                               'rename_observations_processor': {'rename_map': {}}})
    original_select = policy.select_action
    progress = {'status': 'running', 'policy_select_calls': 0, 'completed_tasks': 0,
                'implementation_model': 'gpt-6.1-sol', 'implementation_reasoning_effort': 'low'}
    rows = []
    current = {}

    def select(batch, *args, **kwargs):
        progress['policy_select_calls'] += 1
        current['policy_select_calls'] += 1
        if current['policy_select_calls'] == 1:
            current['first_input'] = {k: (list(v.shape) if hasattr(v, 'shape') else v)
                                      for k, v in batch.items()}
            save(current['taskdir'] / 'first_input.json', current['first_input'])
        result = original_select(batch, *args, **kwargs)
        if progress['policy_select_calls'] % 10 == 0:
            progress['scored_steps'] = [s['scored_steps'] for s in current['states']]
            save(OUT / 'progress.json', progress)
        return result

    policy.select_action = select
    for suite in SUITES:
        for task_id in range(10):
            if (OUT / 'STOP').exists():
                progress['status'] = 'stopped'
                save(OUT / 'progress.json', progress)
                return
            taskdir = OUT / f'{suite}_{task_id:02d}'
            task_started = time.monotonic()
            taskdir.mkdir(exist_ok=True)
            env_cfg = LiberoEnv(task=suite, task_ids=[task_id], observation_height=256,
                observation_width=256, control_mode='relative', init_states=True, hard_reset=True)
            set_seed(1000)
            envs = make_env(env_cfg, n_envs=5, use_async_envs=False)
            vec = envs[suite][task_id]
            states = [{'scored_steps': 0, 'padding_steps': 0, 'success': False,
                       'terminated': False, 'truncated': False, 'done': False} for _ in range(5)]
            current.clear()
            current.update(taskdir=taskdir, states=states, policy_select_calls=0)
            progress.update(suite=suite, task_id=task_id, status='running', scored_steps=[0]*5)
            save(OUT / 'progress.json', progress)
            language = vec.call('task_description')
            with (taskdir / 'actions.jsonl').open('w', encoding='utf-8') as stream:
                for i, env in enumerate(vec.envs):
                    original_step = env.step
                    def step(action, _i=i, _original=original_step):
                        result = _original(action)
                        _, reward, terminated, truncated, info = result
                        state = states[_i]
                        padding = state['done']
                        state['padding_steps' if padding else 'scored_steps'] += 1
                        if not padding:
                            state['success'] = state['success'] or bool(info.get('is_success', False))
                            state['terminated'] = bool(terminated)
                            state['truncated'] = bool(truncated)
                            state['done'] = bool(terminated or truncated)
                        stream.write(json.dumps({'env': _i, 'action': action, 'padding': padding,
                            'scored_steps': state['scored_steps'], 'is_success': bool(info.get('is_success', False)),
                            'terminated': bool(terminated), 'truncated': bool(truncated)}, default=serial) + '\n')
                        stream.flush()
                        return result
                    env.step = step
                env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
                with torch.no_grad():
                    info = eval_policy_all(envs=envs, policy=policy, env_preprocessor=env_pre,
                        env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                        n_episodes=5, max_episodes_rendered=5, videos_dir=taskdir/'videos',
                        return_episode_data=False, start_seed=1000, max_parallel_tasks=1)
            save(taskdir / 'eval_info.json', info)
            successes = info['per_task'][0]['metrics']['successes']
            limit = {'libero_spatial': 280, 'libero_object': 280, 'libero_goal': 300, 'libero_10': 520}[suite]
            for state in states:
                state['end_condition'] = ('native_success' if state['success'] else
                    'terminated' if state['terminated'] else
                    'budget_exhausted' if state['scored_steps'] == limit else 'unknown')
            audit = {'language': language, 'episodes': states,
                     'elapsed_seconds': time.monotonic() - task_started,
                     'policy_select_calls': current['policy_select_calls'],
                     'native_successes': successes, 'valid': False}
            valid = len(successes) == 5 and all(s['scored_steps'] > 0 for s in states) and \
                list(map(bool, successes)) == [s['success'] for s in states] and current['policy_select_calls'] > 0 and \
                all(s['end_condition'] != 'unknown' for s in states)
            audit['valid'] = valid
            save(taskdir / 'audit.json', audit)
            if not valid:
                raise RuntimeError(f'Native scoring audit mismatch at {suite}/{task_id}: {audit}')
            rows.append({'suite': suite, 'task_id': task_id, 'episodes': 5,
                         'successes': sum(map(bool, successes)), 'scored_steps': sum(s['scored_steps'] for s in states)})
            with (OUT / 'tasks.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            total = sum(r['successes'] for r in rows)
            save(OUT / 'summary.json', {'completed_tasks': len(rows), 'valid_episodes': len(rows)*5,
                'successes': total, 'success_rate': total/(len(rows)*5), 'tasks': rows})
            progress.update(completed_tasks=len(rows), scored_steps=[s['scored_steps'] for s in states],
                            last_successes=successes)
            if len(rows) == 1:
                progress['status'] = 'awaiting_first_task_review'
                save(OUT / 'progress.json', progress)
                while not (OUT / 'CONTINUE').exists():
                    if (OUT / 'STOP').exists():
                        progress['status'] = 'stopped'
                        save(OUT / 'progress.json', progress)
                        return
                    time.sleep(1)
            save(OUT / 'progress.json', progress)
    progress['status'] = 'complete'
    save(OUT / 'progress.json', progress)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        save(OUT / 'error.json', {'status': 'unknown', 'error': traceback.format_exc()})
        traceback.print_exc()
        raise
