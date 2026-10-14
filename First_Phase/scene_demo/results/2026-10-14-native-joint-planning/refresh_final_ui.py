"""Refresh idle UI only; no scene load, VLA, model call or execution."""
import hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
out=out/'final_ui';out.mkdir(exist_ok=False)
script=root/'First_Phase/scene_demo/xvla_console.py'
process_file=root/'First_Phase/scene_demo/results/interactive-console/web_process.json'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base='http://127.0.0.1:8083'
def get(route):
    with opener.open(base+route,timeout=5) as response:return json.load(response)
state=get('/api/state')
assert not state['busy'] and state['stage'] in ('error','blocked','done','result','history'), 'Do not interrupt live user work'
(out/'console_before_refresh.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
old=Path(state['run_path'])/'console_state.json' if state.get('run_path') else None
oldhash=hashlib.sha256(old.read_bytes()).hexdigest() if old else None
pid=json.loads(process_file.read_text())['pid']
assert str(script) in Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
boot=get('/api/bootstrap')
request=urllib.request.Request(base+'/api/history',data=json.dumps({'id':boot['examples'][0]['id']}).encode(),headers={'Content-Type':'application/json','X-Console-Token':boot['token']},method='POST')
with opener.open(request,timeout=5) as response:assert json.load(response)['ok']
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+45
while Path(f'/proc/{pid}').exists():
    if ') Z ' in Path(f'/proc/{pid}/stat').read_text():break
    if time.monotonic()>deadline:raise RuntimeError('Old UI still running')
    time.sleep(.25)
started=subprocess.run([sys.executable,str(script),'--launch'],cwd=root,capture_output=True,timeout=30)
(out/'ui_launch.stdout').write_bytes(started.stdout);(out/'ui_launch.stderr').write_bytes(started.stderr)
assert started.returncode==0
current=get('/api/state')
assert not current['busy']
assert not old or hashlib.sha256(old.read_bytes()).hexdigest()==oldhash
result={'before_pid':pid,'after_pid':json.loads(process_file.read_text())['pid'],'stage':current['stage'],'old_record_unchanged':True,'scene_loads':0,'new_robot_actions':0}
(out/'ui_refresh.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result))
