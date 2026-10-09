import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scene_demo.safe_exit_pilot_io as io
import scene_demo.safe_exit_pilot as p


# ---------------------------------------------------------------------------
# A: io.path normalization + sha/read on real stdlib file IO + verify_facts
# ---------------------------------------------------------------------------

def test_path_windows_and_posix_normalization():
    a = io.path('D:\\FYP\\First_Phase\\scene_demo\\x')
    b = io.path('D:/FYP/First_Phase/scene_demo/x')
    expected = io.HERE / 'x'
    assert a == expected
    assert b == expected
    assert a == b


def test_sha_and_read_on_real_file(tmp_path):
    target = tmp_path / 'sample.json'
    target.write_text('{"hello": 1}\n', encoding='utf-8')
    import hashlib as _hashlib
    expected_sha = _hashlib.sha256(target.read_bytes()).hexdigest()
    assert io.sha(target) == expected_sha
    assert io.read(target) == {'hello': 1}


def test_verify_facts_mapping_form_true_and_mismatch_raises(tmp_path):
    target = tmp_path / 'fact.txt'
    target.write_text('payload', encoding='utf-8')
    size = target.stat().st_size
    digest = io.sha(target)
    key = str(target)
    mapping = {key: {'bytes': size, 'sha256': digest}}
    assert io.verify_facts(mapping) is True

    # corrupt hash -> must raise
    bad = {key: {'bytes': size, 'sha256': '0' * 64}}
    with pytest.raises(Exception):
        io.verify_facts(bad)


def test_verify_facts_list_form_true_and_mismatch_raises(tmp_path):
    target = tmp_path / 'fact2.txt'
    target.write_text('payload2', encoding='utf-8')
    size = target.stat().st_size
    digest = io.sha(target)
    records = [{'path': str(target), 'bytes': size, 'sha256': digest}]
    assert io.verify_facts(records) is True

    bad_records = [{'path': str(target), 'bytes': size, 'sha256': '0' * 64}]
    with pytest.raises(Exception):
        io.verify_facts(bad_records)


# ---------------------------------------------------------------------------
# B: real frozen inputs file, must yield case_02 / case_07 / case_08
# ---------------------------------------------------------------------------

def test_verify_inputs_reads_frozen_original_data(monkeypatch):
    snapshot = io.HERE / 'results/2026-10-09-safe-exit/evidence/registry_before.json'
    assert io.sha(snapshot) == 'd86e68ebc154f61f2de04b62959fb6fe76ff78ef28d84ebf7654a727d5cad858'
    assert len(io.read(snapshot)['entries']) == 15

    original_path = io.path

    def historical_path(value):
        p = original_path(value)
        if p.resolve() == (io.HERE / 'experiment_registry.json').resolve():
            return snapshot
        return p

    monkeypatch.setattr(io, 'path', historical_path)

    frozen_inputs = io.HERE / 'results/2026-10-09-safe-exit/evidence/frozen_inputs.json'
    args = SimpleNamespace(inputs=str(frozen_inputs))
    inputs = io._verify_inputs(args)
    case_ids = [c['case_id'] for c in inputs['cases']]
    assert case_ids == ['case_02', 'case_07', 'case_08']


# ---------------------------------------------------------------------------
# C: _verify_software unit fixture (tmp proof, real HERE/.py hashes)
# ---------------------------------------------------------------------------

def _real_source_sha_map():
    return {
        entry.name: io.sha(entry)
        for entry in io.HERE.iterdir()
        if entry.is_file() and entry.suffix == '.py'
    }


def _collect_skill_files():
    """Return {relative_key: sha} for the three required skill files, if present."""
    root_resolved = io.ROOT.resolve()
    keys = {}
    for key in (
        '.agents/skills/fyp-experiment-review/scripts/review.py',
        '.agents/skills/fyp-experiment-review/tests/test_review.py',
        '.agents/skills/fyp-experiment-review/references/schema.md',
    ):
        target = (io.ROOT / key).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            raise AssertionError('skill path escapes ROOT: %s' % key)
        assert target.is_file(), 'missing skill file: %s' % (target,)
        keys[key] = io.sha(target)
    return keys


def _find_existing_proposal():
    """Locate a real proposal JSON under HERE with intent=new_experiment."""
    candidates = []
    for p in io.HERE.rglob('*.json'):
        if not p.is_file():
            continue
        try:
            data = io.read(p)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get('intent') == 'new_experiment' and data.get('question_key') == 'joint-home-handoff':
            candidates.append((p, data))
    assert candidates, 'no existing proposal found under HERE'
    # prefer one whose configuration validates
    for p, data in candidates:
        try:
            io._verify_proposal(SimpleNamespace(proposal=str(p)))
            return p
        except Exception:
            continue
    raise AssertionError('found proposals but none validate via _verify_proposal')


def _build_software_proof(proposal_path):
    proof = {
        'ok': True,
        'test_exit_code': 0,
        'source_sha256': _real_source_sha_map(),
        'proposal_sha256': io.sha(proposal_path),
        'skill_sha256': _collect_skill_files(),
    }
    return proof


def test_verify_software_valid_fixture(tmp_path):
    proposal_path = _find_existing_proposal()
    proof = _build_software_proof(proposal_path)
    proof_path = tmp_path / 'software_proof.json'
    proof_path.write_text(json.dumps(proof), encoding='utf-8')

    args = SimpleNamespace(
        software_proof=str(proof_path),
        proposal=str(proposal_path),
    )
    result = io._verify_software(args)
    assert result['ok'] is True
    assert result['test_exit_code'] == 0
    assert set(result['source_sha256'].keys()) == set(_real_source_sha_map().keys())


def test_verify_software_corrupt_safe_exit_sha_raises(tmp_path):
    proposal_path = _find_existing_proposal()
    proof = _build_software_proof(proposal_path)
    # corrupt one of the source hashes
    key = 'safe_exit.py'
    assert key in proof['source_sha256'], 'expected safe_exit.py among HERE/*.py'
    proof['source_sha256'][key] = '0' * 64
    proof_path = tmp_path / 'software_proof_bad.json'
    proof_path.write_text(json.dumps(proof), encoding='utf-8')

    args = SimpleNamespace(
        software_proof=str(proof_path),
        proposal=str(proposal_path),
    )
    with pytest.raises(Exception):
        io._verify_software(args)


# ---------------------------------------------------------------------------
# D: main orchestration pure tests, run_case monkeypatched
# ---------------------------------------------------------------------------

def _real_core_source_map():
    """Real HERE/*.py hashes used both for software proof and main source-freeze."""
    return _real_source_sha_map()


def _make_fake_inputs():
    record = {'path': 'dummy', 'bytes': 1, 'sha256': '0' * 64}
    return {
        'cases': [
            {'case_id': 'case_02'},
            {'case_id': 'case_07'},
            {'case_id': 'case_08'},
        ],
        'frozen_files': [dict(record)],
        'registry_evidence': [dict(record)],
        'user8_before': [dict(record)],
    }


def _make_fake_software(proposal_path):
    _write_dummy_abs_paths(proposal_path.parent)
    return {
        'ok': True,
        'test_exit_code': 0,
        'source_sha256': _real_core_source_map(),
        'proposal_sha256': io.sha(proposal_path),
        'skill_sha256': _collect_skill_files(),
    }


def _write_dummy_abs_paths(tmp_path):
    inputs = tmp_path / 'inputs.json'
    inputs.write_text('{}\n', encoding='utf-8')
    software_proof = tmp_path / 'software_proof.json'
    software_proof.write_text('{}\n', encoding='utf-8')
    proposal = tmp_path / 'proposal.json'
    proposal.write_text('{}\n', encoding='utf-8')
    return inputs, software_proof, proposal


def _main_args(tmp_path, output_dir):
    inputs, software_proof, proposal = _write_dummy_abs_paths(tmp_path)
    return [
        '--inputs', str(inputs),
        '--software-proof', str(software_proof),
        '--proposal', str(proposal),
        '--output-dir', str(output_dir),
    ]


def _install_gates_stub(monkeypatch, tmp_path, fake_inputs, fake_software, fake_proposal,
                        captured_args):
    def fake_gates(args):
        captured_args.append(args)
        return fake_inputs, fake_software, fake_proposal
    monkeypatch.setattr(io, 'gates', fake_gates)


def _install_verify_facts_stub(monkeypatch, recorded):
    def fake_verify_facts(facts):
        recorded.append(facts)
        return True
    monkeypatch.setattr(io, 'verify_facts', fake_verify_facts)


def _setup_main(
    monkeypatch,
    tmp_path,
    outputs,
    fake_inputs,
    fake_software,
    fake_proposal,
    record_arg,
    gates_args_captured,
):
    monkeypatch.setattr(io, 'OUTPUT', tmp_path / 'pilot')
    _install_gates_stub(monkeypatch, tmp_path, fake_inputs, fake_software, fake_proposal,
                        gates_args_captured)
    _install_verify_facts_stub(monkeypatch, record_arg)

    calls = []

    def fake_run_case(case, out):
        calls.append((case, out))
        item = outputs[len(calls) - 1]
        return dict(item)

    monkeypatch.setattr(p, 'run_case', fake_run_case)
    return calls


def test_main_success_and_blocked_mix(tmp_path, monkeypatch):
    fake_inputs = _make_fake_inputs()
    fake_software = _make_fake_software(tmp_path / 'proposal.json')
    fake_proposal = {'intent': 'new_experiment'}

    outputs = [
        {
            'case_id': 'case_02',
            'status': 'blocked',
            'reason': 'joint_route_blocked',
            'replay_actions': 180,
            'exit_actions': 0,
            'home_actions': 0,
            'errors': [],
        },
        {
            'case_id': 'case_07',
            'status': 'success',
            'reason': None,
            'replay_actions': 78,
            'exit_actions': 5,
            'home_actions': 100,
            'errors': [],
        },
        {
            'case_id': 'case_08',
            'status': 'blocked',
            'reason': 'physical_gate',
            'replay_actions': 76,
            'exit_actions': 80,
            'home_actions': 0,
            'errors': [],
        },
    ]

    recorded = []
    gates_args = []
    calls = _setup_main(
        monkeypatch, tmp_path, outputs,
        fake_inputs, fake_software, fake_proposal,
        recorded, gates_args,
    )

    argv = _main_args(tmp_path, tmp_path / 'pilot')
    rc = p.main(argv)

    assert rc == 0
    assert len(calls) == 3
    assert len(gates_args) == 1

    # verify_facts was called per collection; the whole group dict was not
    # passed down as a single object
    assert len(recorded) >= 9
    for entry in recorded:
        assert not (isinstance(entry, dict) and set(entry.keys()) >= {
            'frozen_files', 'registry_evidence', 'user8_before'
        }), 'group dict leaked into verify_facts'
        # each call must be a non-empty collection (list or mapping)
        assert entry, 'verify_facts must not receive empty collection'
        assert isinstance(entry, (list, tuple, dict))

    summary_path = (tmp_path / 'pilot' / 'summary.json')
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    assert summary['actual_replay_actions'] == 334
    assert summary['total_exit_actions'] == 85
    assert summary['total_home_actions'] == 100
    assert summary['new_vla_calls'] == 0
    assert summary['new_hermes_calls'] == 0

    statuses = [c.get('status') for c in summary['cases']]
    assert statuses == ['blocked', 'success', 'blocked']

    run_case_pairs = [c[0]['case_id'] for c in calls]
    assert run_case_pairs == ['case_02', 'case_07', 'case_08']


def test_main_unknown_stops_remaining_cases(tmp_path, monkeypatch):
    fake_inputs = _make_fake_inputs()
    fake_software = _make_fake_software(tmp_path / 'proposal.json')
    fake_proposal = {'intent': 'new_experiment'}

    outputs = [
        {
            'case_id': 'case_02',
            'status': 'unknown',
            'reason': 'fake',
            'replay_actions': 180,
            'exit_actions': 1,
            'home_actions': 0,
            'errors': ['fake'],
        },
    ]

    recorded = []
    gates_args = []
    calls = _setup_main(
        monkeypatch, tmp_path, outputs,
        fake_inputs, fake_software, fake_proposal,
        recorded, gates_args,
    )

    argv = _main_args(tmp_path, tmp_path / 'pilot')
    rc = p.main(argv)

    assert rc == 2
    assert len(calls) == 1
    assert calls[0][0]['case_id'] == 'case_02'

    summary = json.loads((tmp_path / 'pilot' / 'summary.json').read_text(encoding='utf-8'))
    cases = summary['cases']
    assert cases[0]['status'] == 'unknown'
    assert cases[1]['status'] == 'not_run'
    assert cases[2]['status'] == 'not_run'

    assert summary['actual_replay_actions'] == 180
    assert summary['total_exit_actions'] == 1
    assert summary['total_home_actions'] == 0


def test_main_source_drift_blocks_before_run_case(tmp_path, monkeypatch):
    _write_dummy_abs_paths(tmp_path)
    fake_inputs = _make_fake_inputs()
    # corrupt one source hash so main's source-freeze check fails before run_case
    bad_sources = dict(_real_core_source_map())
    first_key = sorted(bad_sources.keys())[0]
    bad_sources[first_key] = '0' * 64
    fake_software = {
        'ok': True,
        'test_exit_code': 0,
        'source_sha256': bad_sources,
        'proposal_sha256': io.sha(tmp_path / 'proposal.json'),
        'skill_sha256': _collect_skill_files(),
    }
    fake_proposal = {'intent': 'new_experiment'}

    recorded = []
    gates_args = []
    outputs = []
    calls = _setup_main(
        monkeypatch, tmp_path, outputs,
        fake_inputs, fake_software, fake_proposal,
        recorded, gates_args,
    )

    argv = _main_args(tmp_path, tmp_path / 'pilot')
    rc = p.main(argv)

    assert rc == 2
    assert calls == []
    summary = json.loads((tmp_path / 'pilot' / 'summary.json').read_text(encoding='utf-8'))
    cases = summary['cases']
    assert cases[0]['status'] == 'unknown'
    assert cases[0].get('reason') == 'software_drift'
    assert cases[1]['status'] == 'not_run'
    assert cases[2]['status'] == 'not_run'
    assert summary['ok'] is False


def test_main_gate_raises_returns_two_without_creating_output(tmp_path, monkeypatch):
    fake_inputs = _make_fake_inputs()
    fake_software = _make_fake_software(tmp_path / 'proposal.json')
    fake_proposal = {'intent': 'new_experiment'}

    def bad_gates(args):
        raise ValueError('gate boom')

    monkeypatch.setattr(io, 'gates', bad_gates)

    run_case_called = []

    def fake_run_case(case, out):
        run_case_called.append((case, out))
        return {'case_id': case['case_id'], 'status': 'success',
                'replay_actions': 0, 'exit_actions': 0, 'home_actions': 0,
                'errors': []}

    monkeypatch.setattr(p, 'run_case', fake_run_case)

    output_dir = tmp_path / 'pilot'
    argv = _main_args(tmp_path, output_dir)
    rc = p.main(argv)

    assert rc == 2
    assert run_case_called == []
    assert not output_dir.exists()


def test_cli_module_calls_main_and_rejects_invalid_gate(tmp_path):
    import subprocess

    output_existed = io.OUTPUT.exists()
    argv = [
        sys.executable,
        '-B',
        '-m',
        'scene_demo.safe_exit_pilot',
        '--inputs',
        str((tmp_path / 'missing_inputs.json').resolve()),
        '--software-proof',
        str((tmp_path / 'missing_software.json').resolve()),
        '--proposal',
        str((tmp_path / 'missing_proposal.json').resolve()),
        '--output-dir',
        str((tmp_path / 'invalid_output').resolve()),
    ]
    result = subprocess.run(
        argv,
        cwd=str(io.HERE.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert payload.get('error')
    assert not (tmp_path / 'invalid_output').exists()
    assert io.OUTPUT.exists() == output_existed
