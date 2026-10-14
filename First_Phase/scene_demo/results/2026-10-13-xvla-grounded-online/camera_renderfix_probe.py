"""Reproduce forward normalization on saved states; verify camera-only update."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import evaluation.libero.libero_client as author
from scene_demo.xvla_grounding import render_external, triptych
OUT=Path(__file__).resolve().parent
rows=[]
for task_id,name in [(9,'microwave_mug'),(0,'basket_two')]:
 directory=OUT/'camera_renderfix_probe'/name;directory.mkdir(parents=True,exist_ok=True)
 suite=author.benchmark_dict['libero_10']();task=suite.get_task(task_id)
 env=author.OffScreenRenderEnv(bddl_file_name=str(Path(author.get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
 try:
  env.seed(142);env.reset()
  source=next((OUT/name/'runs').rglob('initial_state.npy'))
  env.set_init_state(np.load(source));sim=env.env.sim
  # Restore the exact recorded numeric state after initializing all transforms.
  sim.set_state_from_flattened(np.load(source))
  q=sim.data.qpos.copy();v=sim.data.qvel.copy();t=sim.data.time
  sim.forward()
  delta=float(np.max(np.abs(sim.data.qpos-q)))
  sim.data.qpos[:]=q;sim.data.qvel[:]=v;sim.data.time=t
  before=np.asarray(sim.render(width=256,height=256,camera_name='agentview',depth=False)).copy()
  frames=render_external(SimpleNamespace(_env=env),directory,'initial')
  after=np.asarray(sim.render(width=256,height=256,camera_name='agentview',depth=False))
  assert np.array_equal(before,after)
  wrist=np.asarray(sim.render(width=256,height=256,camera_name='robot0_eye_in_hand',depth=False))[::-1]
  p=directory/'wrist.png';Image.fromarray(wrist).save(p)
  triptych([{'view':'wrist','rgb_path':str(p)}]+frames,directory/'three_views.png')
  rows.append({'case':name,'saved_state':str(source),'full_forward_max_qpos_change':delta,'camera_only_qpos_exact':np.array_equal(sim.data.qpos,q),'camera_only_qvel_exact':np.array_equal(sim.data.qvel,v),'camera_only_time_exact':sim.data.time==t,'native_policy_pixels_exact':True,'new_environment_steps':0})
 finally:env.close()
(OUT/'camera_renderfix_probe/result.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows),flush=True)
