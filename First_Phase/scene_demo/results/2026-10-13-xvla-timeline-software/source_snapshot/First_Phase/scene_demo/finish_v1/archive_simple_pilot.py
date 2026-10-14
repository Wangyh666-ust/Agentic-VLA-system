"""Preserve legacy registry bytes; stage only reviewed source/evidence paths."""
import json
import re
import subprocess
from .prepare_simple_pilot import ROOT, OUT, PLAN, sha


def main():
    registry_path = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
    data = json.loads(registry_path.read_text(encoding='utf-8'))
    entry = data['entries'][-1]
    assert entry['id'] == '20261010-simple-checker'
    manifest = json.loads((OUT / 'evidence/manifest.json').read_text())
    entry['evidence'] = [{'path': str(PLAN.relative_to(ROOT)), 'sha256': sha(PLAN)}]
    entry['evidence'].extend({'path': r['path'].replace('\\', '/'), 'sha256': r['sha256']} for r in manifest['tracked_files'])
    entry['evidence'].append({'path': str((OUT / 'evidence/manifest.json').relative_to(ROOT)), 'sha256': sha(OUT / 'evidence/manifest.json')})
    original = subprocess.check_output(['git', 'show', 'HEAD:First_Phase/scene_demo/experiment_registry.json'], cwd=ROOT).decode('utf-8')
    assert json.loads(original)['entries'] == data['entries'][:-1]
    newline = '\r\n' if '\r\n' in original else '\n'
    at = original.rfind(newline + '  ]')
    assert at > 0
    chunk = '\n'.join('    ' + line for line in json.dumps(entry, ensure_ascii=False, indent=2).splitlines())
    text = original[:at] + ',' + newline + chunk.replace('\n', newline) + original[at:]
    text = re.sub(r'("root_note": )"[^"\r\n]*"', lambda m: m[1] + json.dumps(data['root_note']), text, count=1)
    text = text.replace('"updated_at": "2026-10-10"', '"updated_at": "2026-10-11"', 1)
    assert json.loads(text)['entries'][:-1] == json.loads(original)['entries']
    registry_path.write_bytes(text.encode('utf-8'))
    paths = ['.gitignore', '.gitattributes', 'First_Phase/scene_demo/EXPERIMENT_INDEX.md',
             'First_Phase/scene_demo/experiment_registry.json', str(PLAN.relative_to(ROOT))]
    names = ['control.py', 'vision.py', 'runtime.py', 'handoff_pilot.py', 'simple_checker.py',
             'prepare_simple_pilot.py', 'refresh_simple_pilot.py', 'run_simple_case.py', 'report_simple_pilot.py',
             'register_simple_pilot.py', 'archive_simple_pilot.py', 'tests/test_simple_checker.py']
    paths.extend('First_Phase/scene_demo/finish_v1/' + name for name in names)
    paths.extend(r['path'].replace('\\', '/') for r in manifest['tracked_files'])
    paths.append(str((OUT / 'evidence/manifest.json').relative_to(ROOT)))
    unique = sorted(set(paths))
    subprocess.run(['git', 'add', '-f', '--', *unique], cwd=ROOT, check=True)
    print(json.dumps({'staged_paths': len(unique), 'legacy_entries_byte_preserved': True}))


if __name__ == '__main__': main()
