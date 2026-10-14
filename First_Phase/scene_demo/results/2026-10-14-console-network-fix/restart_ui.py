"""Update the idle UI and restore the failed scene preview without execution."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

out = Path(__file__).resolve().parent
root = out.parents[3]
script = root / 'First_Phase/scene_demo/xvla_console.py'
process_file = root / 'First_Phase/scene_demo/results/interactive-console/web_process.json'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = 'http://127.0.0.1:8083'
def get(route):
    with opener.open(base + route, timeout=5) as response:
        return json.load(response)
def post(route, body, token):
    request = urllib.request.Request(base + route, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'X-Console-Token': token}, method='POST')
    with opener.open(request, timeout=5) as response:
        return json.load(response)

state = get('/api/state')
assert not state['busy'] and state['stage'] == 'error'
assert state['run_path'].endswith('20261014-145221-ee32577c')
(out / 'console_before_restart.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
old_state = Path(state['run_path']) / 'console_state.json'
old_hash = hashlib.sha256(old_state.read_bytes()).hexdigest()
pid = json.loads(process_file.read_text())['pid']
assert str(script) in Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
boot = get('/api/bootstrap')
# Detach the frozen error record so shutdown cannot rewrite it as cancelled.
assert post('/api/history', {'id': boot['examples'][0]['id']}, boot['token'])['ok']
os.kill(pid, signal.SIGTERM)
deadline = time.monotonic() + 45
while Path(f'/proc/{pid}').exists():
    if ') Z ' in Path(f'/proc/{pid}/stat').read_text():
        break
    if time.monotonic() > deadline:
        raise RuntimeError('Old UI has not exited')
    time.sleep(.25)
started = subprocess.run([sys.executable, str(script), '--launch'], cwd=root,
                         capture_output=True, timeout=30)
(out / 'ui_launch.stdout').write_bytes(started.stdout)
(out / 'ui_launch.stderr').write_bytes(started.stderr)
assert started.returncode == 0
boot = get('/api/bootstrap')
assert post('/api/load', {'scene': state['scene']}, boot['token'])['ok']
deadline = time.monotonic() + 60
while True:
    current = get('/api/state')
    if not current['busy']:
        break
    if time.monotonic() > deadline:
        raise RuntimeError('Scene preview has not finished')
    time.sleep(.5)
assert current['stage'] == 'ready' and current['preview_only']
assert [f['view'] for f in current['frames']] == ['wrist', 'agentview', 'sideview']
assert all(Path(f['path']).is_file() for f in current['frames'])
assert hashlib.sha256(old_state.read_bytes()).hexdigest() == old_hash
result = {'before_pid': pid, 'after_pid': json.loads(process_file.read_text())['pid'],
          'stage': current['stage'], 'scene': current['scene'],
          'frames': current['frames'], 'previous_request': state['request'],
          'new_run_path': current['run_path'], 'old_failure_unchanged': True,
          'preview_only': True, 'new_robot_actions': 0}
(out / 'ui_restart.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'frames'}, ensure_ascii=False))
