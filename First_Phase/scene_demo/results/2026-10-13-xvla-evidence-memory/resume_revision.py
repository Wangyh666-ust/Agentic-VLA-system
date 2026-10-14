"""Use native Windows for old-file hashing; resume same authorized replay."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
(OUT/'review-v2-wsl-interruption.json').write_text(json.dumps({
 'reason':'Read-only review hash validation stalled in WSL p9_client_rpc for 4 minutes; terminate only review pid 469 and rerun same proposal via native Windows.',
 'Qwen_requests_before_interruption':0,'environment_actions':0,'proposal_changed':False})+'\n')
r=subprocess.run([sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-evidence-memory-v2.json')],capture_output=True)
(OUT/'review-v2-windows.stdout').write_bytes(r.stdout);(OUT/'review-v2-windows.stderr').write_bytes(r.stderr)
print(r.stdout.decode('utf-8'),flush=True)
if r.returncode:sys.exit(r.returncode)
assert not (OUT/'attempt_02').exists()
cmd=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP/First_Phase','/home/yhwang/fyp/xvla_native/venv/bin/python','-u','First_Phase/scene_demo/results/2026-10-13-xvla-evidence-memory/replay.py','attempt_02']
with (OUT/'replay-v2.stdout').open('xb') as out,(OUT/'replay-v2.stderr').open('xb') as err:
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err)
    for line in p.stdout:out.write(line);out.flush();print(line.decode('utf-8').rstrip(),flush=True)
    code=p.wait()
(OUT/'revision-exit.json').write_text(json.dumps({'replay_returncode':code})+'\n')
sys.exit(code)
