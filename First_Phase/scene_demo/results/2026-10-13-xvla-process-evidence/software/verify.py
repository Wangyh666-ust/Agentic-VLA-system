import json,hashlib,subprocess,sys
from pathlib import Path
root=Path('D:/FYP');scene=root/'First_Phase/scene_demo';out=scene/'results/2026-10-13-xvla-process-evidence/software'
files=['xvla_process_evidence.py','xvla_monitor.py','xvla_finish.py','service.py','xvla_service.py','xvla_pilot.py','tests/test_xvla_finish.py','tests/test_xvla_integration.py','tests/test_xvla_process_evidence.py','tests/test_xvla_process_monitor.py']
syntax=[]
for name in files:
 p=scene/name;compile(p.read_bytes(),str(p),'exec');syntax.append(name)
whitespace=[]
for name in files:
 p=scene/name
 for i,line in enumerate(p.read_bytes().splitlines(),1):
  if line.rstrip(b' \t') != line:whitespace.append({'file':name,'line':i})
assert not whitespace,whitespace
regpath=scene/'experiment_registry.json';idx=scene/'EXPERIMENT_INDEX.md';registry=json.loads(regpath.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert len(registry['entries'])==47
old_good=all(sha(root/e['path'])==e['sha256'] for item in registry['entries'] for e in item['evidence'])
snapshot=scene/'results/2026-10-13-xvla-all-native/source_snapshot/First_Phase/scene_demo'
protected=['finish_v1/control.py','finish_v1/sensors.py','finish_v1/simple_checker.py','finish_v1/vision.py','finish_v1/feedback_retreat.py','joint_home.py','safe_exit.py','safe_exit_contacts.py','placement_completion.py','finish_v1/runtime.py']
protected_checks=[{'file':name,'current_sha':sha(scene/name),'snapshot_sha':sha(snapshot/name),'equal':(scene/name).read_bytes()==(snapshot/name).read_bytes()} for name in protected]
result={'syntax_compiled':syntax,'trailing_whitespace':whitespace,'old_entry_count':len(registry['entries']),'old_evidence_count':sum(len(e['evidence']) for e in registry['entries']),'old_evidence_sha_match':old_good,'registry_original_bytes_equal':regpath.read_bytes()==(out/'experiment_registry.json.before').read_bytes(),'index_original_bytes_equal':idx.read_bytes()==(out/'EXPERIMENT_INDEX.md.before').read_bytes(),'protected_sources':protected_checks,'experimental_model_calls':{'VLA':0,'Qwen':0,'Hermes':0},'environment_actions':0,'implementation_executor':{'model':'gpt-6.1-sol','reasoning_effort':'low'}}
assert old_good and result['registry_original_bytes_equal'] and result['index_original_bytes_equal'] and all(c['equal'] for c in protected_checks)
(out/'source_qualification.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(result,ensure_ascii=False))
