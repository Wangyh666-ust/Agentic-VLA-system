"""Exercise the real UI only as far as its user-confirmation boundary."""
import json
from pathlib import Path
import time
import urllib.request

out = Path(__file__).resolve().parent
root = out.parents[3]
assert json.loads((out / 'online_review.stdout').read_text())['ok']
assert json.loads((out / 'inventory_replay/summary.json').read_text())['ok']
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

restored = json.loads((out / 'ui_restart.json').read_text())
state = get('/api/state')
assert not state['busy'] and state['stage'] == 'ready' and state['preview_only']
assert state['run_path'] == restored['new_run_path']
directory = Path(state['run_path'])
rid = state['request_id']
token = get('/api/bootstrap')['token']
request = restored['previous_request']
assert request == '帮我整理一下桌子，把书本归位'
started = time.monotonic()
assert post('/api/plan', {'request_id': rid, 'request': request}, token)['ok']
previous = None
checkpoints = []
while True:
    state = get('/api/state')
    assert state['request_id'] == rid, 'User changed request; stop observer'
    if state['stage'] != previous:
        row = {'seconds': round(time.monotonic() - started, 2), 'stage': state['stage'], 'message': state['message']}
        checkpoints.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (out / 'online_checkpoints.json').write_text(json.dumps(checkpoints, ensure_ascii=False, indent=2))
        previous = state['stage']
    if not state['busy']:
        break
    if time.monotonic() - started > 900:
        post('/api/stop', {'request_id': rid}, token)
        raise RuntimeError('Planning exceeded test deadline; stop requested')
    time.sleep(1)

(out / 'online_final_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
assert state['stage'] == 'draft', state.get('error_detail', state.get('message'))
assert state['draft']['decision'] == 'execute'
assert state['draft']['capability_ids'] == ['xvla_book_caddy']
setup = json.loads((directory / 'run_setup.json').read_text())
assert setup['history_audit'] == 'not_run_in_interactive_path'
assert not (directory / 'review_check.json').exists()
assert not (directory / 'user_execution_confirmation.json').exists()
assert not state.get('jobs') and not state.get('plan')
assert 'POST /act' not in Path(state['model_log']).read_text()
session_id = state['session_id']
runs = list((root / 'First_Phase/scene_demo/results/interactive-console/runs').glob('*' + session_id[:8]))
assert len(runs) == 1
steps = [json.loads(line) for line in (runs[0] / 'native/native_steps.jsonl').read_text().splitlines()]
assert len(steps) == 10 and all(row['source'] == 'settling' for row in steps)
model_files = sorted((runs[0] / 'planning_grounding/model_requests').glob('*_response.json'))
assert len(model_files) == 2
assert all(json.loads(path.read_text())['model'] == 'qwen3.8-max' for path in model_files)
result = {'ok': True, 'run_path': str(directory), 'session_run_path': str(runs[0]),
          'stage': state['stage'], 'capability_ids': state['draft']['capability_ids'],
          'checkpoints': checkpoints, 'history_scan': False,
          'settling_actions': len(steps), 'vla_actions': 0, 'vla_inference_requests': 0,
          'execution_submitted': False, 'inventory_requests': len(model_files)}
(out / 'online_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False), flush=True)
