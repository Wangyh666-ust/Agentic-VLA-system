"""Run the independently authorized basket case once and preserve raw streams."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('D:/FYP')
OUT = ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/basket_independent'
RUNTIME = OUT/'runtime'
assert not (RUNTIME/'worker.stdout').exists(), 'refuse launch overwrite or retry'
argv = [
    'wsl.exe', '-d', 'Ubuntu', '--cd', '/mnt/d/FYP', '--exec', 'env',
    'PYTHONDONTWRITEBYTECODE=1',
    'PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream',
    '/home/yhwang/fyp/xvla_native/sim_venv/bin/python', '-u',
    '/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/basket_independent/runtime/run_basket.py',
]
(RUNTIME/'worker.argv.json').write_text(json.dumps(argv, indent=2)+'\n', encoding='utf-8')
with (RUNTIME/'worker.stdout').open('wb') as stdout, (RUNTIME/'worker.stderr').open('wb') as stderr:
    result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr)
(RUNTIME/'worker.exitcode').write_text(str(result.returncode)+'\n', encoding='utf-8')
print(json.dumps({'worker_exitcode': result.returncode, 'output': str(OUT), 'case': 'basket_two', 'launches': 1}))
sys.exit(result.returncode)
