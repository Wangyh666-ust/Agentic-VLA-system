"""Render two existing native initial states; zero step/policy/model calls."""
import json
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from PIL import Image
import evaluation.libero.libero_client as author
from scene_demo.xvla_grounding import render_external, triptych

out = Path(__file__).resolve().parent / 'camera_probe'
out.mkdir(exist_ok=True)
for task_id, name in [(9, 'microwave_mug'), (0, 'basket_two')]:
    directory = out / name; directory.mkdir(exist_ok=True)
    suite = author.benchmark_dict['libero_10']()
    task = suite.get_task(task_id)
    env = author.OffScreenRenderEnv(bddl_file_name=str(Path(author.get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file), camera_heights=256, camera_widths=256)
    try:
        env.seed(142); env.reset(); env.set_init_state(suite.get_task_init_states(task_id)[0])
        sim = env.env.sim
        cameras = {n: {'position': sim.data.cam_xpos[sim.model.camera_name2id(n)].tolist(), 'rotation': sim.data.cam_xmat[sim.model.camera_name2id(n)].tolist()} for n in sim.model.camera_names}
        (directory/'native_cameras.json').write_text(json.dumps(cameras, indent=2))
        policy_before = np.array(sim.render(width=256, height=256, camera_name='agentview', depth=False))
        external = render_external(SimpleNamespace(_env=env), directory, 'initial')
        policy_after = np.array(sim.render(width=256, height=256, camera_name='agentview', depth=False))
        assert np.array_equal(policy_before, policy_after), 'policy camera changed'
        wrist = np.asarray(sim.render(width=256, height=256, camera_name='robot0_eye_in_hand', depth=False))[::-1]
        path = directory/'wrist.png'; Image.fromarray(wrist).save(path)
        triptych([{'view':'wrist','rgb_path':str(path)}]+external, directory/'three_views.png')
        print(json.dumps({'case': name, 'policy_pixels_unchanged': True, 'physical_steps': 0, 'cameras': external}), flush=True)
    finally:
        env.close()
