from pathlib import Path
import json,subprocess
ROOT=Path('D:/FYP');OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/runtime'
argv=['D:/miniconda/python.exe','.agents/skills/fyp-experiment-review/scripts/review.py','check','--proposal','First_Phase/scene_demo/plans/20261013-xvla-release-trial.json']
(OUT/'preflight.argv.json').write_text(json.dumps(argv,indent=2));r=subprocess.run(argv,cwd=ROOT,capture_output=True)
(OUT/'preflight.stdout').write_bytes(r.stdout);(OUT/'preflight.stderr').write_bytes(r.stderr);(OUT/'preflight.exitcode').write_text(str(r.returncode)+'\n');print('PREFLIGHT',r.returncode,r.stdout.decode('utf-8',errors='replace'),flush=True)
if r.returncode:raise SystemExit(r.returncode)
argv=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream','/home/yhwang/fyp/xvla_native/sim_venv/bin/python','-u','First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/runtime/run_microwave.py']
(OUT/'worker.argv.json').write_text(json.dumps(argv,indent=2))
with (OUT/'worker.stdout').open('wb') as log,(OUT/'worker.stderr').open('wb') as err:
 p=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
 for line in iter(p.stdout.readline,b''):
  log.write(line);log.flush();print(line.decode('utf-8',errors='replace').rstrip(),flush=True)
 p.wait()
(OUT/'worker.exitcode').write_text(str(p.returncode)+'\n');print('WORKER_EXIT',p.returncode,flush=True)
