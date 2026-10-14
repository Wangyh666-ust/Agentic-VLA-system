"""One-shot authorized microwave launch. Does not launch the basket case."""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path('/mnt/d/FYP');OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial';RUNTIME=OUT/'runtime';READY=json.loads((OUT/'ready.json').read_text());CASE=OUT/'microwave_mug'
def write(path,value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def process(pid):
 try:
  text=Path('/proc/%d/stat'%pid).read_text();rest=text[text.rfind(')')+2:].split();argv=Path('/proc/%d/cmdline'%pid).read_bytes().decode().strip('\0').split('\0')
  return {'pid':pid,'startticks':rest[19],'state':rest[0],'argv':argv,'cwd':os.readlink('/proc/%d/cwd'%pid),'alive':rest[0]!='Z'}
 except (OSError,ValueError):return {'pid':pid,'alive':False}
def listeners(port):
 import re
 r=subprocess.run(['ss','-H','-ltnp','sport = :'+str(port)],capture_output=True)
 return {'port':port,'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'pids':sorted(set(int(p) for p in re.findall(r'pid=(\d+)',r.stdout.decode()))),'listening':bool(r.stdout.strip())}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def freeze():
 hashes=json.loads((OUT/'source_manifest.json').read_text());rows=[{'path':relative,'matches':sha(ROOT/relative)==sha(OUT/'source_snapshot'/relative)==digest} for relative,digest in hashes.items()]
 write(RUNTIME/'freeze.before.json',rows);return all(r['matches'] for r in rows)
def rows(path):
 try:return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
 except (OSError,json.JSONDecodeError):return []
def progress(stage):
 pilots=list(CASE.rglob('pilot_result.json')) if CASE.exists() else []
 native_paths=list(CASE.rglob('native_steps.jsonl')) if CASE.exists() else []
 traces=[r for p in native_paths for r in rows(p)]
 events=[r for p in CASE.rglob('process_events.jsonl') for r in rows(p)] if CASE.exists() else []
 classified=[r for p in CASE.rglob('process_checks.jsonl') for r in rows(p)] if CASE.exists() else []
 checks=[r for p in CASE.rglob('checks.jsonl') for r in rows(p)] if CASE.exists() else []
 current=checks[-1].get('initial_check',{}).get('metadata',{}) if checks else {}
 requests=list(CASE.rglob('call_*_request.json')) if CASE.exists() else []
 vla_requests=[r for p in CASE.rglob('calls.jsonl') for r in rows(p)] if CASE.exists() else []
 value={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,'vla_actions':sum(r.get('source')=='vla' for r in traces),
 'vla_call_files':len(vla_requests),'release_events':len(events),'classified':sum(r.get('status')=='classified' for r in classified),
 'raw_goals':current.get('raw_visual_goal_verdict',{}).get('checks','pending'),'effective_goals':current.get('effective_checks','pending'),
 'qwen_request_files':len(requests),'request_images':len(current.get('request_image_paths',[])),'previous_check_states':len(current.get('input_context',{}).get('visual_timeline',{}).get('previous_check_states',[])),'process_triplets_in_current':len(current.get('input_context',{}).get('visual_timeline',{}).get('process_history',[])),'case_exitcode':None,'result_present':(CASE/'result.json').exists()}
 write(OUT/'microwave_progress.json',value);print('PROGRESS '+json.dumps(value,ensure_ascii=False),flush=True);return value
if not freeze():
 write(RUNTIME/'launch_status.json',{'started':False,'reason':'frozen_source_mismatch','actions':0});raise SystemExit(2)
ports=[listeners(8769),listeners(8794)];write(RUNTIME/'ports.before.json',ports)
if ports[0]['listening']:
 write(RUNTIME/'launch_status.json',{'started':False,'reason':'8769_occupied','actions':0});raise SystemExit(2)
server_spec=READY['future_server'];server=None
if ports[1]['listening']:
 records=[process(pid) for pid in ports[1]['pids']]
 exact=[r for r in records if r.get('alive') and r.get('argv')==server_spec['argv'] and r.get('cwd')==server_spec['cwd'] and r.get('startticks')]
 if len(exact)!=1:
  write(RUNTIME/'launch_status.json',{'started':False,'reason':'8794_identity_unverified','processes':records,'actions':0});raise SystemExit(2)
 server_record={**exact[0],'owned':False,'reused':True}
else:
 output=Path(server_spec['argv'][-1]);output.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ);env.update(server_spec['env'])
 stdout=(RUNTIME/'server.stdout').open('wb');stderr=(RUNTIME/'server.stderr').open('wb')
 server=subprocess.Popen(server_spec['argv'],cwd=server_spec['cwd'],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 stdout.close();stderr.close();server_record={**process(server.pid),'owned':True,'reused':False,'stdout':str(RUNTIME/'server.stdout'),'stderr':str(RUNTIME/'server.stderr')}
write(RUNTIME/'server_state.json',server_record);write(RUNTIME/'server.argv.json',server_spec)
print('SERVER '+json.dumps(server_record),flush=True)
def cleanup_failed_start():
 current=process(server_record['pid'])
 if server_record['owned'] and current.get('alive') and current.get('startticks')==server_record.get('startticks') and current.get('argv')==server_record.get('argv'):
  os.kill(current['pid'],15)
  if server is not None:
   try:server.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  write(RUNTIME/'failed_start_cleanup.json',{'matched_owned_process':True,'signal':'SIGTERM','after':process(current['pid'])})
readiness=[];deadline=time.monotonic()+300;is_ready=False
while time.monotonic()<deadline:
 state=process(server_record['pid'])
 if not state.get('alive'):break
 try:
  request=urllib.request.Request('http://127.0.0.1:8794/openapi.json',method='GET')
  with urllib.request.urlopen(request,timeout=5) as response:
   body=response.read();status=response.status
  schema=json.loads(body);is_ready=status==200 and 'post' in schema.get('paths',{}).get('/act',{})
  readiness.append({'time':time.time(),'status':status,'act_post_present':is_ready})
  if is_ready:
   (RUNTIME/'openapi.response.json').write_bytes(body);break
 except Exception as error:readiness.append({'time':time.time(),'error':str(error)})
 write(RUNTIME/'readiness.json',readiness)
 if len(readiness)%7==0:progress('model_startup')
 time.sleep(5)
write(RUNTIME/'readiness.json',readiness)
if not is_ready:
 cleanup_failed_start();write(RUNTIME/'launch_status.json',{'started':False,'reason':'model_readiness_failed','actions':0,'server_after':process(server_record['pid'])});raise SystemExit(2)
wrapped=READY['future_case_argv']['microwave_mug'];assert wrapped[:7]==['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env']
argv=wrapped[6:];write(RUNTIME/'case.argv.json',{'windows_prepared_argv':wrapped,'actual_argv':argv,'cwd':str(ROOT),'case_launch_count':1})
assert not CASE.exists(),'refuse case overwrite or retry'
stdout=(RUNTIME/'microwave.stdout').open('wb');stderr=(RUNTIME/'microwave.stderr').open('wb')
case=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
stdout.close();stderr.close();case_record={**process(case.pid),'owned':True,'stdout':str(RUNTIME/'microwave.stdout'),'stderr':str(RUNTIME/'microwave.stderr')};write(RUNTIME/'case_state.json',case_record)
progress('case_running')
while case.poll() is None:
 time.sleep(35);progress('case_running' if case.poll() is None else 'case_exited')
write(RUNTIME/'case.exitcode.json',{'exitcode':case.returncode,'case_launch_count':1,'basket_launch_count':0,'case':case_record})
from scene_demo.xvla_trial_evidence import assess_microwave_reuse
assessment=assess_microwave_reuse(CASE)
check_rows=[r for p in CASE.rglob('checks.jsonl') for r in rows(p)]
all_effective=[r.get('initial_check',{}).get('metadata',{}).get('effective_checks',[]) for r in check_rows]
assessment['actual_total_reuse_checks']=sum(any('history_event_id' in c for c in group) for group in all_effective)
assessment['actual_total_reused_items']=sum('history_event_id' in c for group in all_effective for c in group)
trace=[r for p in CASE.rglob('native_steps.jsonl') for r in rows(p)]
source_counts={source:sum(r.get('source')==source for r in trace) for source in ('vla','prepare','finish','settling')}
pilots=[]
for p in CASE.rglob('pilot_result.json'):
 try:pilots.append(json.loads(p.read_text()))
 except (OSError,json.JSONDecodeError):pass
usage_files=list(CASE.rglob('usage_*.json'))
usage=[]
for p in usage_files:
 try:usage.append({'path':str(p),'value':json.loads(p.read_text())})
 except (OSError,json.JSONDecodeError):usage.append({'path':str(p),'value':'unavailable'})
assessment['actual_execution']={'case_exitcode':case.returncode,'case_launches':1,'basket_launches':0,'source_counts':source_counts,'vla_calls':sum(p.get('vla_calls',0) for p in pilots),
 'qwen_recorded_requests':len(list(CASE.rglob('call_*_request.json'))),'qwen_recorded_responses':len(list(CASE.rglob('call_*_response.json'))),'Hermes_original_usage':usage}
write(OUT/'microwave_assessment.json',assessment)
state=process(server_record['pid']);write(RUNTIME/'server.after.json',{**state,'owned':server_record['owned']})
final=progress('case_finished');final['case_exitcode']=case.returncode;final['gate_ok']=assessment['gate_ok'];final['reason']=assessment['reason'];write(OUT/'microwave_progress.json',final)
write(RUNTIME/'freeze.after.json',[{'path':relative,'matches':sha(ROOT/relative)==sha(OUT/'source_snapshot'/relative)==digest} for relative,digest in json.loads((OUT/'source_manifest.json').read_text()).items()])
print('ASSESSMENT '+json.dumps({'gate_ok':assessment['gate_ok'],'reason':assessment['reason'],'counts':assessment['counts'],'execution':assessment['actual_execution'],'total_reuse_checks':assessment['actual_total_reuse_checks'],'server_after':state,'server_owned':server_record['owned']},ensure_ascii=False),flush=True)
