import json,subprocess,hashlib,difflib,re
from pathlib import Path
root=Path('D:/FYP');scene=root/'First_Phase/scene_demo';soft=scene/'results/2026-10-13-xvla-process-evidence/software';out=soft/'final';snapshot=scene/'results/2026-10-13-xvla-all-native/source_snapshot/First_Phase/scene_demo'
env=['PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream','LIBERO_CONFIG_PATH=/home/yhwang/fyp/xvla_native/libero_config','MUJOCO_GL=egl','PYOPENGL_PLATFORM=egl','LD_LIBRARY_PATH=/usr/lib/wsl/lib','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1']
base=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env']+env+['/home/yhwang/fyp/xvla_native/sim_venv/bin/python']
results={}
def run(name,argv):
 (out/(name+'.argv.json')).write_bytes((json.dumps({'cwd':str(root),'argv':argv},indent=2)+'\n').encode());r=subprocess.run(argv,cwd=root,capture_output=True);(out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr);(out/(name+'.exitcode')).write_text(str(r.returncode)+'\n');results[name]={'exitcode':r.returncode,'stdout':name+'.stdout','argv':name+'.argv.json'};print(name,r.returncode,r.stdout.decode(errors='replace')[-250:],flush=True);assert r.returncode==0,name+' failed; stop without repair';return r
r=run('template',base+['-m','pytest','-q','-p','no:cacheprovider','First_Phase/scene_demo/tests/test_xvla_process_monitor.py'])
results['template']['count']=re.findall(r'\d+ passed[^\r\n]*',r.stdout.decode())[-1]
r=run('regression',base+['-m','pytest','-q','-p','no:cacheprovider','First_Phase/scene_demo/tests','First_Phase/scene_demo/finish_v1/tests'])
results['regression']['count']=re.findall(r'\d+ passed[^\r\n]*',r.stdout.decode())[-1]
run('help',base+['-m','scene_demo.xvla_pilot','--help'])
run('git_whitespace',['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check','--','First_Phase/scene_demo/service.py','First_Phase/scene_demo/tests/test_xvla_finish.py','First_Phase/scene_demo/tests/test_xvla_integration.py'])
run('registry_validate',['D:/miniconda/python.exe',str(root/'.agents/skills/fyp-experiment-review/scripts/review.py'),'validate'])
changed=['xvla_process_evidence.py','xvla_monitor.py','xvla_finish.py','service.py','xvla_service.py','xvla_pilot.py','tests/test_xvla_finish.py','tests/test_xvla_integration.py','tests/test_xvla_process_evidence.py','tests/test_xvla_process_monitor.py']
for name in changed:
 p=scene/name;compile(p.read_bytes(),str(p),'exec');assert all(line.rstrip(b' \t')==line for line in p.read_bytes().splitlines())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=['finish_v1/control.py','finish_v1/sensors.py','finish_v1/simple_checker.py','finish_v1/vision.py','finish_v1/feedback_retreat.py','joint_home.py','safe_exit.py','safe_exit_contacts.py','placement_completion.py','finish_v1/runtime.py','mcp_server.py','catalog.py','xvla_skills.py','xvla_backend.py']
source_check=[{'file':n,'sha256':sha(scene/n),'frozen_sha256':sha(snapshot/n),'same':(scene/n).read_bytes()==(snapshot/n).read_bytes()} for n in protected];assert all(s['same'] for s in source_check)
registry=scene/'experiment_registry.json';index=scene/'EXPERIMENT_INDEX.md';reg=json.loads(registry.read_bytes());old=json.loads((soft/'experiment_registry.json.before').read_bytes())
assert reg==old and len(reg['entries'])==47 and sum(len(e['evidence']) for e in reg['entries'])==7608
assert registry.read_bytes()==(soft/'experiment_registry.json.before').read_bytes();assert index.read_bytes()==(soft/'EXPERIMENT_INDEX.md.before').read_bytes()
assert all(sha(root/e['path'])==e['sha256'] for item in reg['entries'] for e in item['evidence'])
previous=(snapshot/'xvla_pilot.py').read_text(encoding='utf-8-sig').splitlines(True);current=(scene/'xvla_pilot.py').read_text(encoding='utf-8-sig').splitlines(True);(out/'pilot_frozen_to_current.diff').write_bytes(''.join(difflib.unified_diff(previous,current,fromfile='all-native/xvla_pilot.py',tofile='process-evidence/xvla_pilot.py')).encode())
q={'qualified':True,'commands':results,'compiled_sources':changed,'space_tab_trailing_whitespace':[],'protected_sources':source_check,'registry_entries':47,'evidence_count':7608,'old_entries_deep_equal':True,'registry_bytes_unchanged':True,'index_bytes_unchanged':True,'all_old_evidence_sha_match':True,'ten_native_same_monitor_class':True,'no_capability_name_branches_in_monitor':True,'fixtures':'initial open/jitter0; measured close then2open capture1; repeat-open0; rearm second event; all10 placement ledger verified; state never caches; client/contract resets; current/release False retracts','mock_serializer':'2 independent 3PNG HTTP serialization requests from same saved capture; does not prove true before/after placement or occlusion sequence','release_budget':4,'current_budget':10,'total_budget':14,'MCP_qualification':'reuse previous actual all10native/13candidate SDK qualification, mcp/catalog/skills/backend SHA unchanged','experimental_model_calls':{'VLA':0,'Qwen':0,'Hermes':0},'environment_actions':0,'root_model':'unknown','root_effort':'unknown','implementation_executor':{'model':'gpt-6.1-sol','effort':'low','configuration_source':'prior explicit spawn for existing executor_finish thread'},'physical_visual_effect':'unverified'}
(out/'qualification.json').write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print('qualified=true',flush=True)
