"""Local command worker; preserve actual argv, environment and raw output."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
case = sys.argv[1]
if case not in ('bowl_wine', 'bowl_stove'):
    raise ValueError(case)
record_path = HERE / f'launch_{case}.json'
log_path = HERE / f'launch_{case}.log'
if record_path.exists() or log_path.exists():
    raise RuntimeError('Do not overwrite a launch')
environment = {
    'LIBERO_CONFIG_PATH': '/home/yhwang/fyp/libero_demo/libero_config',
    'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONPATH': '/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo',
}
argv = [sys.executable, '-B', '-m', 'scene_demo.finish_v1.takeover_repair_pilot', '--case', case]
record = {'argv': argv, 'environment': environment,
          'start_utc': datetime.now(timezone.utc).isoformat(),
          'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
          'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'subagents': 0}
record_path.write_text(json.dumps(record, indent=2) + '\n')
with log_path.open('wb') as log:
    child = subprocess.Popen(argv, cwd=ROOT, env={**os.environ, **environment}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in child.stdout:
        log.write(line)
        log.flush()
        if line.startswith(b'{"event":'):
            print(line.decode().rstrip(), flush=True)
    record['exit_code'] = child.wait()
record['end_utc'] = datetime.now(timezone.utc).isoformat()
record_path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'case': case, 'exit_code': record['exit_code']}), flush=True)
sys.exit(record['exit_code'])
