"""Offline, byte-preserving provenance correction bookkeeping (standard library)."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
POST = BASE / 'postprocess'
REGISTRY = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
ENTRY_ID = '20261012-xvla-native-benchmark'
REPORTS = ('README.md', 'CONCLUSIONS.md', 'final_verification.json')
ALLOWED = (*REPORTS, 'manifest.json')
BASELINE = POST / 'provenance_wording_baseline.json'
PROOF = POST / 'provenance_wording_preservation.json'
VALIDATION = POST / 'provenance_wording_validation.json'
EXECUTION = ('GPT 6 Astra tool-worker subagent executing offline commands; '
             'no additional DeepSeek, Qwen or VLA inference in postprocessing; '
             'GPT worker usage is not included in DeepSeek HTTP usage totals')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode('utf-8'))


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def info(path):
    data = path.read_bytes()
    return {'sha256': digest(data), 'size': len(data)}


def inventory():
    return {p.relative_to(BASE).as_posix(): info(p)
            for p in sorted(BASE.rglob('*')) if p.is_file()}


def masked_registry(data):
    obj = json.loads(data)
    entry = next(e for e in obj['entries'] if e['id'] == ENTRY_ID)
    marker = ('"id": "' + ENTRY_ID + '"').encode()
    prefix, suffix = data.split(marker)
    for ev in entry['evidence']:
        if ev['path'] in {(BASE / p).relative_to(ROOT).as_posix() for p in ALLOWED}:
            old = ev['sha256'].encode()
            assert suffix.count(old) == 1
            suffix = suffix.replace(old, b'<allowed-evidence-sha256>')
    return prefix + marker + suffix


def snapshot():
    assert not BASELINE.exists()
    before = POST / 'provenance_wording_before'
    assert not before.exists()
    manifest = load(BASE / 'manifest.json')
    current = inventory()
    for name, value in manifest['files'].items():
        assert current[name] == value, name
    registry_bytes = REGISTRY.read_bytes()
    registry = json.loads(registry_bytes)
    old = [e for e in registry['entries'] if e['id'] != ENTRY_ID]
    assert len(old) == 41 and len(registry['entries']) == 42
    baseline = {
        'registry_sha256_before': digest(registry_bytes),
        'registry_masked_sha256': digest(masked_registry(registry_bytes)),
        'old_entries_sha256': canonical(old),
        'per_old_entry_sha256': {e['id']: canonical(e) for e in old},
        'current_entry_before': next(e for e in registry['entries'] if e['id'] == ENTRY_ID),
        'manifest_before': manifest,
        'files_before': current,
    }
    before.mkdir()
    for name in REPORTS:
        (before / name).write_bytes((BASE / name).read_bytes())
        assert info(before / name) == current[name]
    save(BASELINE, baseline)
    print(json.dumps({'snapshot_ok': True, 'reports': list(REPORTS),
                      'original_manifest_files_verified': len(manifest['files']),
                      'old_entry_count': len(old)}))


def preservation():
    baseline = load(BASELINE)
    before = POST / 'provenance_wording_before'
    for name in REPORTS:
        assert info(before / name) == baseline['files_before'][name]
    original = load(before / 'final_verification.json')
    current = load(BASE / 'final_verification.json')
    assert current['execution'] == EXECUTION
    changed = [k for k in original.keys() | current.keys() if original.get(k) != current.get(k)]
    assert changed == ['execution'], changed
    old_value = original.pop('execution')
    current.pop('execution')
    assert original == current
    # Also prove byte identity outside the single metadata string replacement.
    old_bytes = (before / 'final_verification.json').read_bytes()
    expected = old_bytes.replace(json.dumps(old_value).encode(), json.dumps(EXECUTION).encode())
    assert expected == (BASE / 'final_verification.json').read_bytes()
    unchanged = {}
    for name, value in baseline['files_before'].items():
        if name not in ALLOWED:
            assert info(BASE / name) == value, name
            unchanged[name] = value
    registry_bytes = REGISTRY.read_bytes()
    registry = json.loads(registry_bytes)
    old = [e for e in registry['entries'] if e['id'] != ENTRY_ID]
    assert len(old) == 41
    assert canonical(old) == baseline['old_entries_sha256']
    assert digest(masked_registry(registry_bytes)) == baseline['registry_masked_sha256']
    groups = {
        'slot_results': [p for p in unchanged if Path(p).match('libero_*/task_*/episode_*/result.json')],
        'responses': [p for p in unchanged if Path(p).name.startswith('response_') and p.endswith('.json')],
        'steps': [p for p in unchanged if Path(p).name == 'steps.jsonl'],
        'source_snapshot': [p for p in unchanged if p.startswith('source_snapshot/')],
    }
    assert len(groups['slot_results']) == 200
    assert len(groups['responses']) == 1160
    assert len(groups['steps']) == 200
    usage = [load(BASE / 'authoring' / phase / 'proof.json')
             for phase in ('author', 'author_fix', 'author_final')]
    totals = {'requests': sum(p['call_count'] for p in usage),
              'input_tokens': sum(p['usage']['input_tokens'] for p in usage),
              'output_tokens': sum(p['usage']['output_tokens'] for p in usage)}
    assert totals == {'requests': 3, 'input_tokens': 19029, 'output_tokens': 17844}
    return {
        'ok': True,
        'executor_model': 'openai/gpt-6-astra',
        'reasoning_effort': 'not exposed',
        'execution': 'GPT tool-worker subagent; offline provenance wording correction',
        'final_verification_changed_fields': changed,
        'json_except_execution_equal': True,
        'bytes_except_execution_equal': True,
        'json_except_execution_before_sha256': canonical(original),
        'json_except_execution_after_sha256': canonical(current),
        'old_41_entries_unchanged': True,
        'old_entries_before_sha256': baseline['old_entries_sha256'],
        'old_entries_after_sha256': canonical(old),
        'registry_bytes_except_four_evidence_hashes_unchanged': True,
        'unchanged_preexisting_files': len(unchanged),
        'unchanged_files_inventory_sha256': canonical(unchanged),
        'raw_and_source_checks': {k: {'count': len(v), 'unchanged': True,
            'inventory_sha256': canonical({p: unchanged[p] for p in v})} for k, v in groups.items()},
        'historical_deepseek_http_usage': totals,
        'scientific_totals_unchanged': current['totals'],
        'additional_deepseek_qwen_vla_calls': 0,
        'additional_environment_actions': 0,
        'pytest_or_video_decode_rerun': False,
    }


def refresh_chain():
    baseline = load(BASELINE)
    manifest = copy.deepcopy(baseline['manifest_before'])
    files = inventory()
    manifest['files'] = {k: v for k, v in files.items() if k not in manifest['exclusions']}
    manifest['file_count'] = len(manifest['files'])
    manifest['total_bytes'] = sum(v['size'] for v in manifest['files'].values())
    save(BASE / 'manifest.json', manifest)
    data = REGISTRY.read_bytes()
    obj = json.loads(data)
    entry = next(e for e in obj['entries'] if e['id'] == ENTRY_ID)
    marker = ('"id": "' + ENTRY_ID + '"').encode()
    prefix, suffix = data.split(marker)
    for ev in entry['evidence']:
        if ev['path'] in {(BASE / p).relative_to(ROOT).as_posix() for p in ALLOWED}:
            old = ev['sha256'].encode()
            new = info(ROOT / ev['path'])['sha256'].encode()
            assert suffix.count(old) == 1
            suffix = suffix.replace(old, new)
    REGISTRY.write_bytes(prefix + marker + suffix)


def validate():
    command = [sys.executable, str(ROOT / '.agents/skills/fyp-experiment-review/scripts/review.py'), 'validate']
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert not result.stderr, result.stderr
    assert json.loads(result.stdout)['ok'] is True
    return result.stdout


def verify():
    proof = preservation()
    assert load(PROOF) == proof
    manifest = load(BASE / 'manifest.json')
    actual = {k: v for k, v in inventory().items() if k not in manifest['exclusions']}
    assert actual == manifest['files']
    assert manifest['file_count'] == len(actual)
    assert manifest['total_bytes'] == sum(v['size'] for v in actual.values())
    output = validate()
    assert output == VALIDATION.read_bytes()
    print(json.dumps({'ok': True, 'manifest_chain_ok': True,
        'manifest_file_count': len(actual), 'registry_validate': json.loads(output),
        'json_except_execution_equal': True, 'old_41_entries_unchanged': True,
        'raw_and_source_checks': proof['raw_and_source_checks']}))


def finalize():
    assert not PROOF.exists() and not VALIDATION.exists()
    save(PROOF, preservation())
    refresh_chain()
    VALIDATION.write_bytes(validate())
    # Close the validation receipt, include it in the manifest, then validate the final chain.
    refresh_chain()
    verify()


if __name__ == '__main__':
    {'snapshot': snapshot, 'finalize': finalize, 'verify': verify}[sys.argv[1]]()
