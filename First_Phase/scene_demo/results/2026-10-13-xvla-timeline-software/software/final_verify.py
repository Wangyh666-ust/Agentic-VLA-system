import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
SOFTWARE=OUT/'software'
LABEL=sys.argv[1] if len(sys.argv)>1 else 'final'
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(name,argv):
    name=LABEL+'_'+name
    if (SOFTWARE/(name+'.stdout')).exists():raise FileExistsError('refuse to overwrite command output')
    dump(SOFTWARE/(name+'.argv.json'),argv)
    result=subprocess.run(argv,cwd=ROOT,capture_output=True)
    (SOFTWARE/(name+'.stdout')).write_bytes(result.stdout)
    (SOFTWARE/(name+'.stderr')).write_bytes(result.stderr)
    (SOFTWARE/(name+'.exitcode')).write_text(str(result.returncode)+'\n')
    return result.returncode

old_sources=json.loads((ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/source_manifest.json').read_text())
changed={'xvla_finish.py','xvla_monitor.py','xvla_process_evidence.py','xvla_pilot.py','xvla_trial_evidence.py'}
protected={p:sha(ROOT/p)==value for p,value in old_sources.items() if Path(p).name not in changed}
assert all(protected.values()),'unexpected change to protected sources'
dump(SOFTWARE/(LABEL+'_protected_sources.json'),protected)
argv=json.loads((SOFTWARE/'targeted.argv.json').read_text())
argv[-1]='--basetemp=/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-software/software/pytest_'+LABEL
codes={'final_targeted':run('final_targeted',argv)}
if codes['final_targeted']:dump(SOFTWARE/(LABEL+'_commands.json'),codes);sys.exit(2)
compile_argv=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1',
 '/home/yhwang/fyp/xvla_native/sim_venv/bin/python','-c',
 "from pathlib import Path; files=['xvla_finish.py','xvla_monitor.py','xvla_process_evidence.py','xvla_timeline.py','xvla_trial_evidence.py','xvla_pilot.py']; [compile((Path('First_Phase/scene_demo')/p).read_bytes(),p,'exec') for p in files]; print('compiled',len(files),'files')"]
codes['compile']=run('compile',compile_argv)
codes['help']=run('help',['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1',
 'PYTHONPATH=/mnt/d/FYP/First_Phase','/home/yhwang/fyp/xvla_native/sim_venv/bin/python','-m','scene_demo.xvla_pilot','--help'])
codes['whitespace']=run('whitespace',['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check','--',
 'First_Phase/scene_demo/xvla_finish.py','First_Phase/scene_demo/xvla_monitor.py','First_Phase/scene_demo/xvla_process_evidence.py',
 'First_Phase/scene_demo/xvla_timeline.py','First_Phase/scene_demo/xvla_trial_evidence.py','First_Phase/scene_demo/xvla_pilot.py',
 'First_Phase/scene_demo/tests/test_xvla_timeline.py','First_Phase/scene_demo/tests/test_xvla_finish.py',
 'First_Phase/scene_demo/tests/test_xvla_full_checks.py','First_Phase/scene_demo/tests/test_xvla_process_evidence.py',
 'First_Phase/scene_demo/tests/test_xvla_process_monitor.py','First_Phase/scene_demo/tests/test_xvla_trial_evidence.py'])
dump(SOFTWARE/(LABEL+'_commands.json'),codes)
print(json.dumps(codes))
sys.exit(0 if all(v==0 for v in codes.values()) else 2)
