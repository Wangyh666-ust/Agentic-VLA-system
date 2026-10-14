"""Reuse the already initialized Hermes home; help command only, no LLM/task."""
import json,os,subprocess,sys,time
from pathlib import Path
out=Path(__file__).resolve().parent;root=out.parents[3]
sys.path.insert(0,str(root/'First_Phase'))
from scene_demo.xvla_console import shared_planner_home
runtime=Path('/home/yhwang/fyp/xvla_native')
ready=runtime/'console_profiles/20261014-165713-cfefe80d'
assert (ready/'installs').is_dir() and (ready/'logs/agent.log').is_file()
manifest=root/'First_Phase/scene_demo/results/interactive-console/shared_hermes_profile.json'
if not manifest.exists():
    manifest.write_text(json.dumps({'path':str(ready),'migration':'reuse completed console runtime; no copied credentials'}))
home=shared_planner_home(runtime,manifest)
assert shared_planner_home(runtime,manifest)==home
environment=dict(os.environ,HERMES_HOME=str(home))
start=time.monotonic()
run=subprocess.run(['/home/yhwang/.local/bin/hermes','--help'],env=environment,capture_output=True,timeout=120)
elapsed=time.monotonic()-start
(out/'runtime_help.stdout').write_bytes(run.stdout);(out/'runtime_help.stderr').write_bytes(run.stderr)
assert run.returncode==0
summary={'home':str(home),'reused_initialized_home':True,'help_wall_seconds':elapsed,
    'new_model_calls':0,'new_robot_actions':0,'help_command_only':True}
(out/'runtime_validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
