import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scene_demo import xvla_backend
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
root=Path('/home/yhwang/fyp/xvla_native/LIBERO/libero/libero/assets/scenes')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cameras=[]
for filename in ('libero_kitchen_tabletop_base_style.xml','libero_living_room_tabletop_base_style.xml','libero_study_base_style.xml'):
 p=root/filename;tree=ET.parse(p);attrs={c.attrib['name']:dict(c.attrib) for c in tree.iter('camera') if c.attrib.get('name') in ('agentview','sideview')}
 assert set(attrs)=={'agentview','sideview'}
 assert {k:v for k,v in attrs['agentview'].items() if k!='name'} != {k:v for k,v in attrs['sideview'].items() if k!='name'}
 cameras.append({'path':str(p),'sha256':sha(p),'cameras':attrs})
references=[]
for task in range(10):
 d=xvla_backend.BENCH_ROOT/'libero_10'/('task_%02d'%task)/'episode_0/attempt_001';p=d/'initial_state.npy';audit=d/'initial_audit.json';a=np.load(p,allow_pickle=False);j=json.loads(audit.read_text());assert np.isfinite(a).all()
 references.append({'task_id':task,'array_path':str(p),'array_sha256':sha(p),'shape':list(a.shape),'dtype':str(a.dtype),'all_finite':True,'audit_path':str(audit),'audit_sha256':sha(audit),'audit_keys':sorted(j)})
slots=[]
for cid,s in NATIVE_LONG_SKILLS.items():
 expected=('libero_10',s['native_task']['task_id'])
 for identifier in [cid]+s['component_capability_ids']:
  assert xvla_backend.REFERENCE_SLOTS[identifier]==expected
  slots.append({'capability':identifier,'slot':list(expected)})
print(json.dumps({'camera_assets':cameras,'native_references':references,'reference_slots':slots,'simulator_constructed':False},ensure_ascii=False,indent=2))
