"""One-shot scene rendering. No policy, planner, service, or env.step calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace


def render(scene_id, output, seed=142, init_state_index=0):
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from scene_demo import catalog
    from scene_demo.finish_v1.sensors import SensorPort
    from scene_demo.xvla_grounding import attach_checker_cameras
    from scene_demo.xvla_process_evidence import capture_frames

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = catalog.SCENES[scene_id]
    random.seed(seed)
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()[spec['suite']]()
    task = suite.get_task(spec['task_id'])
    initial = np.asarray(suite.get_task_init_states(spec['task_id'])[init_state_index])
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
        camera_heights=256, camera_widths=256, control_freq=20, ignore_done=True, horizon=8000)
    step_attempts = []
    def forbid_step(*args, **kwargs):
        step_attempts.append(True)
        raise RuntimeError('scene preview must not execute robot actions')
    try:
        # Fail if any downstream preview helper tries to issue a robot action.
        env.step = env.env.step = forbid_step
        env.seed(seed)
        env.reset()
        env.set_init_state(initial)
        inner = env.env
        # Catalog variants are static initial placements, without settling actions.
        for patch in spec.get('patches', []):
            joint = inner.objects_dict[patch['object_id']].joints[0]
            qpos = np.asarray(inner.sim.data.get_joint_qpos(joint)).copy()
            if patch['kind'] == 'offset':
                qpos[:3] += np.asarray(patch['xyz'])
            elif patch['kind'] == 'place_on':
                qpos[:3] = inner.sim.data.body_xpos[inner.obj_body_id[patch['target_id']]] + [0., 0., .05]
            else:
                raise ValueError('unsupported scene variant')
            inner.sim.data.set_joint_qpos(joint, qpos)
            inner.sim.forward()
        wrapper = SimpleNamespace(_env=env)
        def signature():
            return hashlib.sha256(inner.sim.data.qpos.tobytes() + inner.sim.data.qvel.tobytes()
                                  + np.asarray([inner.sim.data.time]).tobytes()).hexdigest()
        before = signature()
        capture = attach_checker_cameras(SensorPort(wrapper, {}, output)).capture()
        pictures = capture_frames(capture)
        after = signature()
        if before != after:
            raise RuntimeError('rendering changed simulator state')
        result = {'ok': True, 'scene': scene_id, 'seed': seed, 'init_state_index': init_state_index,
                  'frames': [{'view': f['view'], 'path': f['rgb_path'], 'timestamp': capture['timestamp']}
                             for f in pictures], 'action_steps': 0, 'step_attempts': len(step_attempts),
                  'models_started': [], 'state_before': before, 'state_after': after,
                  'initial_state_sha256': hashlib.sha256(initial.tobytes()).hexdigest()}
        (output / 'capture.json').write_text(json.dumps(capture, ensure_ascii=False, indent=2))
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.scene, args.output), ensure_ascii=False), flush=True)
