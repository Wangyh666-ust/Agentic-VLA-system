"""Capture the read-only basket audit, then identity-checked server cleanup."""
import json
from pathlib import Path
import subprocess

ROOT = Path('D:/FYP')
OUT = ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial'
assert (OUT/'basket_independent/runtime/worker.exitcode').exists()
tasks = [
    ('basket_independent/runtime', 'audit', '/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/basket_independent/runtime/audit_basket.py'),
    ('runtime', 'cleanup', '/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/runtime/cleanup.py'),
]
results = {}
for directory, name, script in tasks:
    destination = OUT/directory
    assert not (destination/(name+'.stdout')).exists()
    argv = ['wsl.exe', '-d', 'Ubuntu', '--cd', '/mnt/d/FYP', '--exec', 'env',
            'PYTHONDONTWRITEBYTECODE=1', '/home/yhwang/fyp/xvla_native/venv/bin/python', '-u', script]
    (destination/(name+'.argv.json')).write_text(json.dumps(argv, indent=2)+'\n', encoding='utf-8')
    result = subprocess.run(argv, cwd=ROOT, capture_output=True)
    (destination/(name+'.stdout')).write_bytes(result.stdout)
    (destination/(name+'.stderr')).write_bytes(result.stderr)
    (destination/(name+'.exitcode')).write_text(str(result.returncode)+'\n', encoding='utf-8')
    results[name] = result.returncode
    print(name+': '+result.stdout.decode('utf-8', errors='replace'), flush=True)
    if result.returncode:
        print(result.stderr.decode('utf-8', errors='replace'), flush=True)
print(json.dumps(results))
assert all(value == 0 for value in results.values())
