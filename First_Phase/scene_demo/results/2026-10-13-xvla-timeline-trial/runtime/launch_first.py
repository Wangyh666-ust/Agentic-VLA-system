"""Execute the frozen first case once; no helper, retry, or basket launch."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
RUNTIME=OUT/'runtime'
OLD=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/runtime/run_microwave.py'
source=OLD.read_text(encoding='utf-8').replace('2026-10-13-xvla-release-trial','2026-10-13-xvla-timeline-trial')
source=source.replace("vla_requests=list(CASE.rglob('xvla_calls/call_*.json')) if CASE.exists() else []",
 "vla_requests=[r for p in CASE.rglob('calls.jsonl') for r in rows(p)] if CASE.exists() else []")
source=source.replace("'qwen_request_files':len(requests),'case_exitcode':None", 
 "'qwen_request_files':len(requests),'request_images':len(current.get('request_image_paths',[])),'previous_check_states':len(current.get('input_context',{}).get('visual_timeline',{}).get('previous_check_states',[])),'process_triplets_in_current':len(current.get('input_context',{}).get('visual_timeline',{}).get('process_history',[])),'case_exitcode':None")
target=RUNTIME/'run_microwave.py'
assert not target.exists() and not (RUNTIME/'worker.stdout').exists(),'refuse first case rerun'
compile(source,str(target),'exec');target.write_text(source,encoding='utf-8')
argv=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1',
 'PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream',
 '/home/yhwang/fyp/xvla_native/sim_venv/bin/python','-u',
 '/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/runtime/run_microwave.py']
(RUNTIME/'worker.argv.json').write_text(json.dumps(argv,indent=2)+'\n')
with (RUNTIME/'worker.stdout').open('wb') as stdout,(RUNTIME/'worker.stderr').open('wb') as stderr:
    result=subprocess.run(argv,cwd=ROOT,stdout=stdout,stderr=stderr)
(RUNTIME/'worker.exitcode').write_text(str(result.returncode)+'\n')
print(json.dumps({'worker_exitcode':result.returncode,'output':str(OUT),'case':'microwave_mug','basket_launches':0}))
sys.exit(result.returncode)
