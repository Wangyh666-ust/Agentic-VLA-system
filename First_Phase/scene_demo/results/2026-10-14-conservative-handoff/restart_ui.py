"""Restart the idle local UI without launching a scene or modifying old run evidence."""
import hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent;root=out.parents[3]
console=root/'First_Phase/scene_demo/xvla_console.py'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base='http://127.0.0.1:8083'
def get(path):
    with opener.open(base+path,timeout=5) as response:return json.load(response)
state=get('/api/state')
if state.get('busy') or state.get('stage') not in ('blocked','complete','idle','cancelled'):
    raise RuntimeError('UI is not idle; leave it untouched')
(out/'console_before_restart.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
old_file=Path(state['run_path'])/'console_state.json' if state.get('run_path') else None
old_hash=hashlib.sha256(old_file.read_bytes()).hexdigest() if old_file else None
process=root/'First_Phase/scene_demo/results/interactive-console/web_process.json'
pid=json.loads(process.read_text())['pid']
args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
if str(console) not in args:raise RuntimeError('Saved PID is not this UI')
# Leaving the finished live view clears the in-memory directory. The existing
# shutdown handler would otherwise rewrite a frozen terminal run as cancelled.
boot=get('/api/bootstrap')
payload=json.dumps({'id':boot['examples'][0]['id']}).encode()
request=urllib.request.Request(base+'/api/history',data=payload,
    headers={'Content-Type':'application/json','X-Console-Token':boot['token']},method='POST')
with opener.open(request,timeout=5) as response:
    if response.status!=200:raise RuntimeError('Could not detach completed view')
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+45
while Path(f'/proc/{pid}').exists():
    if ') Z ' in Path(f'/proc/{pid}/stat').read_text():break
    if time.monotonic()>deadline:raise RuntimeError('Old UI has not exited')
    time.sleep(.25)
started=subprocess.run([sys.executable,str(console),'--launch'],cwd=root,capture_output=True,timeout=30)
(out/'ui_launch.stdout').write_bytes(started.stdout);(out/'ui_launch.stderr').write_bytes(started.stderr)
if started.returncode:raise RuntimeError('UI restart failed; see ui_launch.stderr')
current=get('/api/state')
assert not current['busy'] and current['stage']=='idle'
if old_file:assert hashlib.sha256(old_file.read_bytes()).hexdigest()==old_hash
result={'before_pid':pid,'after_pid':json.loads(process.read_text())['pid'],'stage':current['stage'],
    'new_model_or_simulator_started':False,'old_console_evidence_unchanged':True}
(out/'ui_restart.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
