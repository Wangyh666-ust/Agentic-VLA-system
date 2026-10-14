"""Launch one independent case with original bytes logged, no model delegation."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'First_Phase/scene_demo/results/2026-10-10-simple-checker'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['wine', 'bowl', 'sauce', 'stove'])
    parser.add_argument('--revision', default='')
    parser.add_argument('--campaign', default='2026-10-10-simple-checker')
    args = parser.parse_args()
    base = ROOT / 'First_Phase/scene_demo/results' / args.campaign
    indices = {'wine': 1, 'bowl': 0, 'sauce': 4, 'stove': 2}
    evidence = base / 'evidence'
    proposal = ROOT / 'First_Phase/scene_demo/plans' / (args.campaign + '.json')
    inputs = evidence / 'inputs.json'
    run_name = args.case + args.revision
    cmd = [sys.executable, '-B', '-m', 'scene_demo.finish_v1.handoff_pilot', '--proposal', str(proposal),
           '--inputs', str(inputs), '--output', str(base / ('pilot_' + run_name)),
           '--start-index', str(indices[args.case]), '--limit', '1', '--max-qwen-calls', '6', '--simple-checker']
    env = dict(os.environ, LIBERO_CONFIG_PATH='/home/yhwang/fyp/libero_demo/libero_config', MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'First_Phase'))
    record = {'argv': cmd, 'environment': {k: env[k] for k in
              ('LIBERO_CONFIG_PATH', 'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH')},
              'start_utc': datetime.now(timezone.utc).isoformat(),
              'proposal_sha256': hashlib.sha256(proposal.read_bytes()).hexdigest(),
              'inputs_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(), 'subagents': 0}
    with (evidence / ('launch_' + run_name + '.log')).open('xb') as log:
        print('running independent case: ' + args.case, flush=True)
        run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=run.returncode, end_utc=datetime.now(timezone.utc).isoformat())
    (evidence / ('launch_' + run_name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    summary = base / ('pilot_' + run_name) / 'summary.json'
    if summary.exists(): print(summary.read_text(), flush=True)
    return run.returncode


if __name__ == '__main__': sys.exit(main())
