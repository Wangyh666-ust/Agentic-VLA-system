"""One authorized full console run, no human-wait time in the measured total."""
import hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
scene=root/'First_Phase/scene_demo'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base='http://127.0.0.1:8083'
def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def get(path):
    with opener.open(base+path,timeout=10) as r:return json.load(r)
def post(path,body):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode(),method='POST',
        headers={'Content-Type':'application/json','X-Console-Token':token})
    with opener.open(req,timeout=15) as r:return json.load(r)
def wait(target,limit):
    deadline=time.monotonic()+limit; previous=None
    while time.monotonic()<deadline:
        state=get('/api/state');save('online_latest.json',state)
        key=(state.get('stage'),state.get('message'))
        if key!=previous:
            print(json.dumps({'elapsed':round(time.monotonic()-started,2),'stage':key[0],'message':key[1]},ensure_ascii=False),flush=True)
            previous=key
        if not state['busy']:
            if state['stage'] in target:return state
            if state['stage'] in ('error','blocked','cancelled'):raise RuntimeError(str(state.get('error_detail') or key))
        time.sleep(2)
    raise TimeoutError('Stage deadline exceeded; inspect state before any further action')

assert json.loads((out/'replay_v2/summary.json').read_text())['success'] is True,'retreat replay did not pass'
assert json.loads((out/'online_review.stdout').read_text())['ok'] is True
assert not (out/'online_timing.json').exists(),'one online case only'
state=get('/api/state')
assert not state['busy'] and state['stage'] in ('idle','history','blocked','error','completed'),'do not interrupt user work'
save('console_before_refresh.json',state)
token=get('/api/bootstrap')['token']
old=Path(state['run_path'])/'console_state.json' if state.get('run_path') else None
oldhash=hashlib.sha256(old.read_bytes()).hexdigest() if old else None
boot=get('/api/bootstrap');post('/api/history',{'id':boot['examples'][0]['id']})
process_file=scene/'results/interactive-console/web_process.json'
pid=json.loads(process_file.read_text())['pid']
script=scene/'xvla_console.py'
assert str(script) in Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+45
while Path(f'/proc/{pid}').exists():
    if ') Z ' in Path(f'/proc/{pid}/stat').read_text():break
    if time.monotonic()>deadline:raise RuntimeError('old UI did not stop')
    time.sleep(.25)
proc=subprocess.run([sys.executable,str(script),'--launch'],cwd=root,capture_output=True,timeout=30)
(out/'ui_launch.stdout').write_bytes(proc.stdout);(out/'ui_launch.stderr').write_bytes(proc.stderr)
assert proc.returncode==0
assert not old or hashlib.sha256(old.read_bytes()).hexdigest()==oldhash
token=get('/api/bootstrap')['token']
save('ui_refresh.json',{'old_pid':pid,'new_pid':json.loads(process_file.read_text())['pid'],'old_record_unchanged':True})
started=time.monotonic(); wall_started=time.time()
timing={'started_unix':wall_started,'request':'收拾桌面，将碗和瓶子归位','scene':'bowl_drawer',
        'human_wait_seconds':0,'helpers':False,'finish_streak':5,'completed':False}
save('online_timing.json',timing)
try:
    post('/api/load',{'scene':'bowl_drawer'})
    ready=wait({'ready'},180)
    loaded=time.monotonic();timing['load_scene_seconds']=loaded-started
    post('/api/plan',{'request':timing['request'],'request_id':ready['request_id']})
    state=wait({'draft'},600);save('online_draft_state.json',state)
    planned=time.monotonic();timing['prepare_and_plan_seconds']=planned-loaded
    draft=state['draft']
    assert draft['capability_ids']==['xvla_bowl_drawer_close']
    assert draft['request_scope']['coverage']=='partial'
    assert draft['request_scope']['unmet_outcomes']
    save('online_timing.json',timing)
    post('/api/execute',{'request_id':state['request_id']})
    state=wait({'completed','blocked','error','cancelled'},2500)
    ended=time.monotonic();save('online_final_state.json',state)
    timing.update(execution_seconds=ended-planned,total_seconds=ended-started,
                  stage=state['stage'],completed=state['stage']=='completed',run_path=state.get('run_path'))
    print(json.dumps(timing,ensure_ascii=False),flush=True)
except BaseException as exc:
    timing.update(error=str(exc),total_seconds=time.monotonic()-started)
    raise
finally:
    timing['ended_unix']=time.time();save('online_timing.json',timing)
