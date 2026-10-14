from pathlib import Path
import subprocess,json,hashlib,re
ROOT=Path('D:/FYP');HERE=ROOT/'First_Phase/scene_demo';OUT=HERE/'results/2026-10-13-xvla-all-native/software/final'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
registry=HERE/'experiment_registry.json';index=HERE/'EXPERIMENT_INDEX.md'
(OUT/'registry.before.json').write_bytes(registry.read_bytes());(OUT/'index.before.md').write_bytes(index.read_bytes())
def evidence_check(name):
 data=json.loads(registry.read_text(encoding='utf-8'));rows=[]
 for entry in data['entries']:
  for e in entry['evidence']:
   p=ROOT/e['path'];actual=sha(p) if p.exists() else None
   rows.append({'entry':entry['id'],'path':e['path'],'expected':e['sha256'],'actual':actual,'matches':actual==e['sha256']})
 report={'entries':len(data['entries']),'evidence':len(rows),'all_match':all(r['matches'] for r in rows),'rows':rows}
 (OUT/(name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(name,report['entries'],report['evidence'],report['all_match'],flush=True);return report
before=evidence_check('evidence.before')
results={}
def run(name,argv):
 (OUT/(name+'.argv.json')).write_text(json.dumps(argv,ensure_ascii=False,indent=2),encoding='utf-8')
 r=subprocess.run(argv,cwd=ROOT,capture_output=True)
 (OUT/(name+'.stdout')).write_bytes(r.stdout);(OUT/(name+'.stderr')).write_bytes(r.stderr);(OUT/(name+'.exitcode')).write_text(str(r.returncode)+'\n')
 results[name]={'argv':name+'.argv.json','stdout':name+'.stdout','stderr':name+'.stderr','exitcode':r.returncode}
 print(name,'exit',r.returncode,r.stdout[-350:].decode('utf-8',errors='replace'),flush=True);return r
wsl=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream','LIBERO_CONFIG_PATH=/home/yhwang/fyp/xvla_native/libero_config','MUJOCO_GL=egl','PYOPENGL_PLATFORM=egl','LD_LIBRARY_PATH=/usr/lib/wsl/lib','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','/home/yhwang/fyp/xvla_native/sim_venv/bin/python']
reg=run('regression',wsl+['-m','pytest','-q','-p','no:cacheprovider','First_Phase/scene_demo/tests','First_Phase/scene_demo/finish_v1/tests'])
probe_prefix=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase','/usr/bin/python3']
relative='First_Phase/scene_demo/results/2026-10-13-xvla-all-native/software/final/'
run('mcp',probe_prefix+[relative+'mcp_probe.py'])
help_result=run('help',wsl+['-m','scene_demo.xvla_pilot','--help'])
from_env={'wine_bowl_stove','basket_two','mugs_two','microwave_mug','basket_cheese_butter','stove_moka','bowl_drawer','book_caddy','mug_pudding','basket_soup_cheese','moka_two'}
help_checks={case:case.encode() in help_result.stdout for case in sorted(from_env)}
(OUT/'help.cases.json').write_text(json.dumps(help_checks,indent=2))
run('static',wsl+[relative+'static_probe.py'])
files=['catalog.py','service.py','mcp_server.py','run_agent.py','setup_profile.py','xvla_service.py','xvla_backend.py','xvla_pilot.py','xvla_skills.py','xvla_finish.py','finish_v1/runtime.py','tests/test_xvla_skills.py','tests/test_xvla_finish.py','tests/test_xvla_integration.py','tests/test_xvla_pipeline.py','tests/test_oracle.py','tests/test_xvla_full_catalog.py','tests/test_xvla_full_checks.py','finish_v1/tests/test_sensors.py']
syntax=[];spaces=[]
for f in files:
 p=HERE/f
 try:compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');syntax.append({'file':f,'passed':True})
 except Exception as error:syntax.append({'file':f,'passed':False,'error':str(error)})
 for i,line in enumerate(p.read_bytes().splitlines(),1):
  if line.endswith((b' ',b'\t')):spaces.append({'file':f,'line':i})
(OUT/'source_checks.json').write_text(json.dumps({'syntax':syntax,'trailing_whitespace':spaces,'passed':not spaces and all(s['passed'] for s in syntax)},indent=2))
tracked=['catalog.py','service.py','tests/test_oracle.py','run_agent.py','mcp_server.py','setup_profile.py','finish_v1/runtime.py','finish_v1/tests/test_sensors.py']
run('git_whitespace',['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check','--']+['First_Phase/scene_demo/'+f for f in tracked])
old=HERE/'results/2026-10-12-xvla-persistent-pilot/v5/source_snapshot/First_Phase/scene_demo'
protected=['finish_v1/control.py','finish_v1/sensors.py','finish_v1/simple_checker.py','finish_v1/vision.py','finish_v1/feedback_retreat.py','joint_home.py','safe_exit.py','safe_exit_contacts.py','placement_completion.py']
protected_rows=[{'file':f,'old_sha256':sha(old/f),'current_sha256':sha(HERE/f),'same':sha(old/f)==sha(HERE/f)} for f in protected]
(OUT/'downstream_unchanged.json').write_text(json.dumps(protected_rows,indent=2))
run('registry_validate',['D:/miniconda/python.exe','.agents/skills/fyp-experiment-review/scripts/review.py','validate'])
after=evidence_check('evidence.after')
summary={'commands':results,'test_summary':re.findall(r'[^\n]*(?:passed|failed)[^\n]*',reg.stdout.decode('utf-8',errors='replace'))[-1:],'native_interfaces':'10/10' if results['mcp']['exitcode']==0 else 'failed','help_11_cases':all(help_checks.values()),'source_syntax_whitespace_passed':not spaces and all(s['passed'] for s in syntax),'downstream_9_byte_identical':all(r['same'] for r in protected_rows),'registry_entries':after['entries'],'evidence_count':after['evidence'],'all_old_evidence_sha_match':after['all_match'],'registry_bytes_unchanged':registry.read_bytes()==(OUT/'registry.before.json').read_bytes(),'index_bytes_unchanged':index.read_bytes()==(OUT/'index.before.md').read_bytes(),'implementation_executor':{'model':'gpt-6.1-sol','effort':'low','configuration_source':'previous explicit spawn for this executor thread'},'root_model':None,'root_effort':None,'new_calls':{'VLA':0,'Qwen':0,'Hermes':0},'environment_actions':0,'physical_visual_effect':'unknown/unverified'}
summary['qualified']=all(r['exitcode']==0 for r in results.values()) and all(summary[k] for k in ['help_11_cases','source_syntax_whitespace_passed','downstream_9_byte_identical','all_old_evidence_sha_match','registry_bytes_unchanged','index_bytes_unchanged'])
(OUT/'qualification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)
