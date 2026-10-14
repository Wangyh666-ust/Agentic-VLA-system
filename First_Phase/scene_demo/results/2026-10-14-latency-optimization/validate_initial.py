"""Original planning triplet, task-only inventory and merged audit; no simulator."""
import json
import sys
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
sys.path.insert(0,str(root/'First_Phase'))
from scene_demo import catalog
from scene_demo.xvla_grounding import inspect_scene, verify_plan, seed_checker, InventoryClient
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
from scene_demo.xvla_finish import build_skill_context

directory=out/'book_initial'
assert not directory.exists()
directory.mkdir()
source=root/'First_Phase/scene_demo/results/interactive-console/runs/20261014T085743Z_book_caddy_cdfcb45d/planning_grounding/capture.json'
capture=json.loads(source.read_text())
request='将书本整理好'
client=InventoryClient(max_calls=2)
inventory,image=inspect_scene(capture,request,directory/'grounding',client)
targets=[o for o in inventory['objects'] if o.get('role')=='target']
destinations=[o for o in inventory['objects'] if o.get('role')=='destination']
assert len(targets)==len(destinations)==1,'Expected one book and one caddy; inspect actual original-image identity'
selection={'black_book_1':targets[0]['object_id'],'desk_caddy_1_back_contain_region':destinations[0]['object_id']}
capability=catalog.CAPABILITIES['xvla_book_caddy']
audit_client=InventoryClient(max_calls=1)
review=verify_plan(inventory,selection,request,[capability],directory,client=audit_client)
assert review['passed'],'Merged three-view grounding not confirmed'
context=build_skill_context(capability,request)
context.update(sensor_frames=[{'frame':i,'view':v,'timestamp':capture['timestamp']} for i,v in enumerate(('wrist','agentview','sideview'))],
    actuator_measurements={'timestamp':capture['timestamp'],'finger_qpos':capture['robot']['finger_qpos'],
        'standard_open_finger_qpos':capture['robot']['finger_qpos'],'max_abs_arm_qvel':max(map(abs,capture['robot']['arm_qvel']))})
checker=XvlaSubtaskReviewClient(env_path='/none')
seed_checker(checker,inventory,selection,capture,context,verified_review=review)
assert checker.analyze_initial() is None and checker._calls==0
summary={'object_count':len(inventory['objects']),'objects':inventory['objects'],'inventory_calls':client._calls,
    'merged_review_calls':audit_client._calls,'checker_initial_calls':checker._calls,'regions':checker.store.regions,
    'new_robot_actions':0,'source':str(source),'model':client.model}
(directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('objects','regions')},ensure_ascii=False))
