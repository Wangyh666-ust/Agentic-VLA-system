"""Replay only the exact archived triplet; never create an environment."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

out = Path(__file__).resolve().parent
root = out.parents[3]
os.environ.update(json.loads((root / 'First_Phase/scene_demo/results/2026-10-14-qwen38-three-requests/runtime/table_tidy_stove_environment.json').read_text()))
for path in os.environ['PYTHONPATH'].split(':'):
    if path not in sys.path:
        sys.path.insert(0, path)
from scene_demo.xvla_grounding import InventoryClient, build_inventory_prompt, validate_inventory

source = root / 'First_Phase/scene_demo/results/interactive-console/runs/20261014T065708Z_book_caddy_c4025ecb/planning_grounding'
files = ['cap000002_wrist.png', 'cap000002_wrist_checker_agentview.png', 'cap000002_wrist_checker_sideview.png']
views = ['wrist', 'agentview', 'sideview']
paths = [str(source / name) for name in files]
context = {'sensor_frames': [{'frame': i, 'view': view, 'timestamp': -1} for i, view in enumerate(views)]}
request = '帮我整理一下桌子，把书本归位'
client = InventoryClient(max_calls=2)
client.audit_dir = out / 'inventory_replay/model_requests'
client.audit_dir.mkdir(parents=True, exist_ok=False)
manifest = {'original_attempt': '20261014-145221-ee32577c',
            'timestamp_note': 'Original capture metadata was not saved before failure; -1 explicitly denotes unknown capture time. No robot feedback synthesized.',
            'context': context, 'request': request,
            'images': [{'path': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]}
(out / 'inventory_replay/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
first = None
results = []
try:
    for label in ['initial', 'review']:
        started = time.monotonic()
        meta = client._request(context, paths, build_inventory_prompt(request, first))
        inventory = validate_inventory(client.raw_inventory)
        result = {'stage': label, 'seconds': round(time.monotonic() - started, 2),
                  'metadata': meta, 'inventory': inventory}
        (out / f'inventory_replay/{label}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        results.append({'stage': label, 'seconds': result['seconds'], 'object_count': len(inventory['objects'])})
        print(json.dumps(results[-1]), flush=True)
        first = inventory
    summary = {'ok': True, 'requests': client._calls, 'results': results, 'robot_actions': 0}
except Exception as exc:
    summary = {'ok': False, 'requests': client._calls, 'results': results,
               'error': str(exc), 'error_type': type(exc).__name__, 'robot_actions': 0}
finally:
    (out / 'inventory_replay/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary), flush=True)
if not summary['ok']:
    sys.exit(1)
