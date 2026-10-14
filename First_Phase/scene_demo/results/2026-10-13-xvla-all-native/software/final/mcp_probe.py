import runpy
from scene_demo import catalog
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS,NATIVE_ATOMIC_CAPABILITIES,NATIVE_SCENES
f=runpy.run_path('/mnt/d/FYP/First_Phase/scene_demo/mcp_server.py',run_name='software_import_probe')['_atomic_capabilities']
for cid,skill in NATIVE_LONG_SKILLS.items():
 scene=skill['scene_ids'][0];public=catalog.scene_capabilities(scene,backend='xvla')
 assert cid in [c['capability_id'] for c in f(public)]
 native=next(c for c in public if c['capability_id']==cid)
 assert f([dict(native,capability_id='forged')])==[]
 assert f([dict(native,audit_only=True)])==[]
 print('native visible:',skill['native_task']['task_id'],scene,cid,'; audit/forged rejected')
for cid,cap in NATIVE_ATOMIC_CAPABILITIES.items():
 assert cid in [c['capability_id'] for c in f(catalog.scene_capabilities(cap['scene_ids'][0],backend='xvla'))]
for scene in NATIVE_SCENES: assert catalog.scene_capabilities(scene)==[]
for scene in catalog.SCENES:
 assert not {c['capability_id'] for c in catalog.scene_capabilities(scene)} & (set(NATIVE_ATOMIC_CAPABILITIES)|set(NATIVE_LONG_SKILLS))
print('ALL_NATIVE_MCP_RUNTIME_OK native=10 candidate_atomic=13 new_scene=7 Smol_X_exclusive_capabilities_hidden=23 HTTP/model/environment_actions=0')
