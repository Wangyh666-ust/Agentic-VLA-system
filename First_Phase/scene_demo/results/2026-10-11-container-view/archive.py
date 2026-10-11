"""Archive referenced evidence and append the given registry entry without rewriting history."""
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REGISTRY = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
INDEX = ROOT / 'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
IDENTIFIER = HERE.name.replace('2026-10-11-', '20261011-')
PLAN = ROOT / 'First_Phase/scene_demo/plans' / (HERE.name + '.json')


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def evidence(path):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


selected = {PLAN}
for path in HERE.rglob('*'):
    if path.is_file() and 'sensors' not in path.parts and path.name != 'manifest.json':
        selected.add(path)
for path in list(selected):
    if path.suffix not in ('.json', '.jsonl'):
        continue
    values = ([json.loads(line) for line in path.read_text().splitlines()]
              if path.suffix == '.jsonl' else [json.loads(path.read_text())])
    for value in values:
        for name in strings(value):
            if not name.startswith(str(HERE)):
                continue
            candidate = Path(name).resolve()
            if candidate.is_relative_to(HERE) and candidate.is_file():
                selected.add(candidate)
manifest_path = HERE / 'manifest.json'
manifest_path.write_text(json.dumps({'tracked_files': [evidence(p) for p in sorted(selected)]}, ensure_ascii=False, indent=2) + '\n')
entry = json.loads((HERE / 'registry_entry.json').read_text())
assert entry['id'] == IDENTIFIER
entry['configuration'] = json.loads(PLAN.read_text())['configuration']
entry['configuration_complete'] = True
entry['evidence'] = [evidence(p) for p in sorted(selected)] + [evidence(manifest_path)]
original = REGISTRY.read_bytes().decode('utf-8')
registry = json.loads(original)
assert not any(e['id'] == IDENTIFIER for e in registry['entries']), 'Do not replace old evidence'
offset = original.rfind('\n  ]')
assert offset >= 0
updated = original[:offset].rstrip() + ',\n' + textwrap.indent(json.dumps(entry, ensure_ascii=False, indent=2), '    ') + original[offset:]
assert json.loads(updated)['entries'][:-1] == registry['entries']
REGISTRY.write_bytes(updated.encode('utf-8'))
with INDEX.open('a') as index:
    index.write('\n- `' + IDENTIFIER + '`：' + entry['title'] + '。[报告、视频与明确结论](results/' + HERE.name + '/README.md)。\n')
print(json.dumps({'files': len(selected), 'registry_entries': len(registry['entries']) + 1, 'old_entries_unchanged': True}))
