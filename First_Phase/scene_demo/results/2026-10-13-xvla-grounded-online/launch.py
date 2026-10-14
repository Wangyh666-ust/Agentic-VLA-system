"""Two independently authorized cases, once each; retain stdout and stop own server."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path('/mnt/d/FYP');OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-grounded-online'
PROPOSAL=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-grounded-online.json'
RUNTIME=OUT/'runtime';RUNTIME.mkdir(exist_ok=True)
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def check():
 r=subprocess.run([sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(PROPOSAL)],capture_output=True)
 (RUNTIME/'review.stdout').write_bytes(r.stdout);(RUNTIME/'review.stderr').write_bytes(r.stderr)
 if r.returncode:raise RuntimeError('experiment review failed')
def rows(path):
 try:return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
 except (OSError,json.JSONDecodeError):return []
def progress(name):
 d=OUT/name; traces=[r for p in d.rglob('native_steps.jsonl') for r in rows(p)]
 events=[r for p in d.rglob('process_events.jsonl') for r in rows(p)]
 item={'case':name,'source_counts':{s:sum(r.get('source')==s for r in traces) for s in ('settling','prepare','vla','finish')},'release_events':len(events),
 'model_responses':len(list(d.rglob('*_response.json'))),'native_success_latest':traces[-1].get('native_success') if traces else None,'result_exists':(d/'result.json').exists()}
 dump(RUNTIME/(name+'_progress.json'),item);print(json.dumps(item),flush=True)
check()
for port in (8769,8794):
 if subprocess.run(['ss','-H','-ltn','sport = :'+str(port)],capture_output=True).stdout.strip():raise RuntimeError('port occupied '+str(port))
assert not any((OUT/n).exists() for n in ('microwave_mug','basket_two')), 'refuse retry'
spec=json.loads((ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/ready.json').read_text())['future_server']
spec['argv'][-1]=str(RUNTIME/'server_output')
env=dict(os.environ);env.update(spec['env'])
with (RUNTIME/'server.stdout').open('wb') as out,(RUNTIME/'server.stderr').open('wb') as err:
 server=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=env,stdout=out,stderr=err)
dump(RUNTIME/'server.json',{'pid':server.pid,'spec':spec,'owned':True})
try:
 ready=False
 for i in range(60):
  if server.poll() is not None:raise RuntimeError('model server exited')
  try:
   with urllib.request.urlopen('http://127.0.0.1:8794/openapi.json',timeout=3) as res:
    ready='post' in json.load(res)['paths']['/act']
   if ready:break
  except Exception:pass
  if i%6==0:print('model loading',flush=True)
  time.sleep(5)
 if not ready:raise RuntimeError('model server not ready')
 for name in ('microwave_mug','basket_two'):
  check()
  argv=[sys.executable,'-u','-m','scene_demo.xvla_pilot','--case',name,'--output',str(OUT),'--proposal',str(PROPOSAL)]
  dump(RUNTIME/(name+'_argv.json'),argv)
  with (RUNTIME/(name+'.stdout')).open('wb') as out,(RUNTIME/(name+'.stderr')).open('wb') as err:
   case=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err)
  while case.poll() is None:
   time.sleep(25);progress(name)
  dump(RUNTIME/(name+'_exit.json'),{'exitcode':case.returncode,'launch_count':1})
  progress(name)
finally:
 server.terminate()
 try:server.wait(timeout=20)
 except subprocess.TimeoutExpired:
  server.kill();server.wait()
 dump(RUNTIME/'server_cleanup.json',{'owned_pid':server.pid,'exitcode':server.returncode})
