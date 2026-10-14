import ast,json,re
from pathlib import Path
from collections import Counter
from scene_demo import catalog,xvla_backend
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
root=Path('/home/yhwang/fyp/xvla_native/LIBERO/libero/libero')
tree=ast.parse((root/'benchmark/libero_suite_task_map.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets))
mapping=ast.literal_eval(node.value)['libero_10']
rows=[]
for cid,s in sorted(NATIVE_LONG_SKILLS.items(),key=lambda x:x[1]['native_task']['task_id']):
 task=s['native_task']['task_id'];source=next((root/'bddl_files/libero_10').glob('*'+s['instruction'].replace(' ','_')+'.bddl'));text=source.read_text()
 language=' '.join(re.search(r'\(:language\s+(.*?)\)',text,re.S).group(1).split())
 goals=[[op]+args.split() for op,args in re.findall(r'\((in|on|close|turnon)\s+([^()]+)\)',text.split('(:goal',1)[1].lower())]
 row={'task_id':task,'native_id':cid,'scene_id':s['scene_ids'][0],'sdk_task':mapping[task],'bddl_source':str(source),'language':language,'goals':goals,'language_match':language==s['instruction'],'goals_match':Counter(map(tuple,goals))==Counter(map(tuple,s['goals'])),'components':s['component_capability_ids'],'reference_slot':xvla_backend.REFERENCE_SLOTS[cid]}
 assert row['language_match'] and row['goals_match'];rows.append(row)
print(json.dumps({'coverage':rows,'VLA_Qwen_Hermes_calls':0,'simulator_actions':0,'physical_effect':'unverified'},ensure_ascii=False,indent=2))
