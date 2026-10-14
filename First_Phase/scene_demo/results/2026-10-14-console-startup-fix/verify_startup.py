"""Zero-action check of the real console model start/stop/restart path."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

out = Path(__file__).resolve().parent
root = out.parents[3]
env_file = root / 'First_Phase/scene_demo/results/2026-10-14-qwen38-three-requests/runtime/table_tidy_stove_environment.json'
os.environ.update(json.loads(env_file.read_text()))
for path in os.environ['PYTHONPATH'].split(':'):
    if path not in sys.path:
        sys.path.insert(0, path)
from scene_demo.xvla_console import Console

shared = root / 'First_Phase/scene_demo/results/interactive-console'
old_files = [shared / 'model.log', shared / 'model_output/info.json',
             shared / '20261014-142953-3e090f52/console_state.json']
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in old_files}
console = Console(out / 'probe_console')
console.directory = out / 'zero_action_probe'
console.directory.mkdir(exist_ok=False)
runs = []
try:
    for index in range(2):
        started = time.monotonic()
        try:
            console.ensure_runtime()
            health = console.svc.health()
            assert health['ready'] and health['active_request_id'] is None
            assert console.svc._env is None
            assert not console.svc._sessions and not console.svc._plans
            assert console.model.poll() is None
            runtime = Path(console.state['model_runtime'])
            assert (runtime / 'output/info.json').is_file()
            runs.append({'index': index + 1, 'pid': console.model.pid,
                         'ready': True, 'runtime': str(runtime),
                         'startup_seconds': round(time.monotonic() - started, 2),
                         'scene_count': 0, 'plan_count': 0})
            print(json.dumps(runs[-1]), flush=True)
        finally:
            console.close_runtime()
        assert console.model is None and console.svc is None
    assert runs[0]['runtime'] != runs[1]['runtime']
    logs = [Path(row['runtime']) / 'model.log' for row in runs]
    assert all('Application startup complete' in p.read_text() for p in logs)
    assert all('POST /act' not in p.read_text() for p in logs)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)] for p in old_files)
    result = {'ok': True, 'real_model_starts': len(runs), 'runs': runs,
              'new_vla_inference_requests': 0, 'new_vlm_requests': 0,
              'new_robot_actions': 0, 'original_failure_unchanged': True}
    (out / 'startup_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
finally:
    console.close_runtime()
