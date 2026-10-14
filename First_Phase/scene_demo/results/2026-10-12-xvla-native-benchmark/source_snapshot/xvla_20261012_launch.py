"""One-shot mechanical archive/launcher. No evaluation or retry logic."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path('D:/FYP')
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-12-xvla-native-benchmark'
BASE = '/home/yhwang/fyp/xvla_native'
UNC = Path('//wsl.localhost/Ubuntu')
PROPOSAL = 'First_Phase/scene_demo/plans/20261012-xvla-native-benchmark.json'
REVIEW = '.agents/skills/fyp-experiment-review/scripts/review.py'
SIM = BASE + '/sim_venv/bin/python'
MODEL = BASE + '/venv/bin/python'
WOUT = '/mnt/d/FYP/' + OUT.relative_to(ROOT).as_posix()
STATE = {'planner': 'openai/gpt-6-astra', 'reasoning_effort': 'not exposed',
         'scope': 'first native episode only; no automatic continuation',
         'server': None, 'client': None}
PROCESSES = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    assert path.parent.is_dir(), str(path.parent)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def save_state():
    STATE['updated_utc'] = utc()
    write_json(OUT / 'launch_first.json', STATE)


def run_record(name, cmd, source=None, timeout=60):
    start = utc()
    result = subprocess.run(cmd, input=source.encode() if source else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=ROOT, timeout=timeout)
    (OUT / (name + '.stdout.log')).write_bytes(result.stdout)
    (OUT / (name + '.stderr.log')).write_bytes(result.stderr)
    (OUT / (name + '.log')).write_bytes(result.stdout + result.stderr)
    record = {'command': cmd, 'stdin_source': source, 'started_utc': start,
              'finished_utc': utc(), 'exit_code': result.returncode,
              'stdout': result.stdout.decode('utf-8', errors='replace'),
              'stderr': result.stderr.decode('utf-8', errors='replace')}
    write_json(OUT / (name + '.record.json'), record)
    if result.returncode:
        raise RuntimeError(f'{name} exit={result.returncode}; see raw stdout/stderr logs')
    return result.stdout


def wsl_python(name, source, python=SIM, timeout=60):
    return run_record(name, ['wsl', '-d', 'Ubuntu', '--cd', '/mnt/d/FYP', '--exec',
                            'env', 'PYTHONDONTWRITEBYTECODE=1', python, '-'], source, timeout)


def freeze():
    records = []

    def copy(src, rel):
        dst = OUT / rel
        assert src.is_file(), str(src)
        assert dst.parent.is_dir(), str(dst.parent)
        assert not dst.exists(), f'will not overwrite {dst}'
        before = digest(src)
        shutil.copyfile(src, dst)
        assert digest(dst) == before == digest(src)
        records.append({'source': str(src), 'snapshot': rel, 'sha256': before,
                        'bytes': dst.stat().st_size})

    for directory in ('source_snapshot', 'authoring', 'install'):
        assert OUT.is_dir()
        (OUT / directory).mkdir(exist_ok=True)
    for rel in ('First_Phase/libero_demo/xvla_native_benchmark.py',
                'First_Phase/libero_demo/test_xvla_native_benchmark.py', PROPOSAL,
                'First_Phase/scene_demo/plans/2026-10-12-xvla-native.md'):
        copy(ROOT / rel, 'source_snapshot/' + Path(rel).name)
    for rel in ('evaluation/libero/libero_client.py', 'deploy.py'):
        copy(UNC / (BASE + '/upstream/' + rel).lstrip('/'), 'source_snapshot/' + Path(rel).name)
    for name in ('install_verification.json', 'RESUME.md'):
        copy(ROOT / 'First_Phase/tmp/xvla_20261012_setup' / name, 'install/' + name)
    calls = []
    for phase in ('author', 'author_fix', 'author_final'):
        folder = ROOT / ('First_Phase/tmp/xvla_20261012_' + phase)
        (OUT / 'authoring' / phase).mkdir(exist_ok=True)
        for name in ('request.json', 'response.json', 'proof.json', 'patch.txt'):
            copy(folder / name, 'authoring/' + phase + '/' + name)
        proof = json.loads((folder / 'proof.json').read_text(encoding='utf-8'))
        assert proof['call_count'] == 1 and proof['http_status'] == 200
        assert digest(folder / 'request.json') == proof['request_sha256']
        assert digest(folder / 'response.json') == proof['response_sha256']
        calls.append({'phase': phase, 'original_proof': proof})
    final = ROOT / 'First_Phase/tmp/xvla_20261012_author_final'
    for name in ('context_repair.patch', 'context_repair_proof.json'):
        copy(final / name, 'authoring/author_final/' + name)
    repair = json.loads((final / 'context_repair_proof.json').read_text(encoding='utf-8'))
    for rel, sha in repair['final_sha256'].items():
        assert digest(ROOT / rel) == sha, 'runner/test changed after context repair'
    copy(Path(__file__), 'source_snapshot/xvla_20261012_launch.py')
    result = {'created_utc': utc(), 'planner': 'openai/gpt-6-astra',
              'reasoning_effort': 'not exposed', 'files': records,
              'actual_historical_deepseek_calls': 3, 'new_deepseek_calls': 0,
              'calls': calls, 'gpt_context_mechanical_repair': repair,
              'original_candidate_applied_verbatim': False,
              'corrected_fields': ['weight_sha256 -> model_sha256',
                  'weight_revision -> model_revision', 'model_file -> weight.path; is_file',
                  'controller_name/controller_use_delta/controller_control_dim -> initial_audit',
                  'author_score and policy_actions/setup_actions/model_calls/horizon integer checks',
                  '_read_slot_result propagates invalid JSON', 'paused_after_limit progress',
                  'test end_reason/native_success, first attempt row, action identity, done=False',
                  'real installed freeze_metadata contract test'],
              'rejected_candidate_test': repair['rejected'],
              'verification_note': 'Original repair proof records 10 passed/1 failed. User reports subsequent setuptools65.5.1 main-agent verification 11 passed/21 warnings. No pytest rerun in this launch task.',
              'historical_failure_record_preserved': 'provenance.json'}
    write_json(OUT / 'freeze.json', result)
    return result


def prepare():
    assert OUT.is_dir() and ROOT.is_dir()
    assert not (OUT / 'launch_first.json').exists(), 'prior launch record exists; no restart'
    assert not (OUT / 'freeze.json').exists(), 'prior freeze exists; no overwrite'
    STATE['started_utc'] = utc()
    STATE['phase'] = 'zero_action_verification'
    save_state()
    raw = wsl_python('verification_65_5_1', """import json, importlib.metadata, torch
from pkg_resources import packaging
version = importlib.metadata.version('setuptools')
equal = torch.__version__ == '1.11.0+cpu'
print(json.dumps({'setuptools': version, 'torch': str(torch.__version__), 'torch_version_equal': equal, 'pkg_resources_packaging': packaging.__name__, 'inference_calls': 0, 'simulator_actions': 0}))
assert version == '65.5.1'
assert equal
""")
    write_json(OUT / 'verification_65_5_1.json', json.loads(raw))
    frozen = freeze()
    STATE['source_sha256'] = {row['snapshot']: row['sha256'] for row in frozen['files']}
    STATE['freeze_sha256'] = digest(OUT / 'freeze.json')
    run_record('review_history_query', [sys.executable, '-B', REVIEW, 'query', '--text', 'benchmark-success-rate'])
    raw = run_record('review_first_ready', [sys.executable, '-B', REVIEW, 'check', '--proposal', PROPOSAL])
    gate = json.loads(raw)
    (OUT / 'review_first_ready.json').write_bytes(raw)
    assert gate['ok'] is True, 'review gate not ok=true'
    STATE['review_ok'] = True
    raw = wsl_python('source_port_preflight', """import json, subprocess, socket
from pathlib import Path
base = Path('/home/yhwang/fyp/xvla_native')
expected = {'upstream': '6bc2513f5f1cbec715cc668b414392a6cae5c671', 'LIBERO': '8f1084e3132a39270c3a13ebe37270a43ece2a01'}
rows = {}
for name, sha in expected.items():
    head = subprocess.check_output(['git','-C',str(base/name),'rev-parse','HEAD']).decode().strip()
    status = subprocess.check_output(['git','-C',str(base/name),'status','--porcelain','--untracked-files=all']).decode()
    rows[name] = {'head':head,'status':status,'expected':sha}
print(json.dumps(rows), flush=True)
assert all(r['head']==r['expected'] and not r['status'] for r in rows.values()), 'source SHA/clean mismatch'
s = socket.socket()
s.bind(('127.0.0.1',8794))
s.close()
print('WSL 127.0.0.1:8794 bind available', flush=True)
""")
    win = socket.socket()
    try:
        win.bind(('127.0.0.1', 8794))
    finally:
        win.close()
    for name in ('libero_10', 'metadata.json', 'STOP', 'server.log', 'first_episode.log', 'server_runtime'):
        assert not (OUT / name).exists(), f'prior runtime artifact exists: {name}'
    STATE['phase'] = 'ready_to_launch'
    save_state()
    print(json.dumps({'phase': STATE['phase'], 'review_ok': True, 'frozen_files': len(frozen['files'])}))


def command(cwd, env, tail):
    return ['wsl', '-d', 'Ubuntu', '--cd', cwd, '--exec', 'env'] + env + tail


def process_snapshot(label, expected):
    source = """import json, os, time
from pathlib import Path
expected = EXPECTED
found = []
deadline = time.monotonic() + 4
while time.monotonic() < deadline:
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            argv = p.joinpath('cmdline').read_bytes().decode().rstrip('\\0').split('\\0')
            if argv == expected:
                found.append({'pid': int(p.name), 'argv': argv, 'cwd': os.readlink(p/'cwd'), 'stat': (p/'stat').read_text()})
        except (OSError, UnicodeError): pass
    if found: break
    time.sleep(0.1)
print(json.dumps(found))
""".replace('EXPECTED', repr(expected))
    rows = json.loads(wsl_python(label + '_linux_process', source, timeout=15))
    assert len(rows) == 1, f'{label}: expected one exact Linux process, found {len(rows)}'
    return rows[0]


def spawn(label, cmd, tail, logfile):
    assert OUT.is_dir()
    with (OUT / logfile).open('xb') as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    PROCESSES[label] = p
    STATE[label] = {'windows_popen_pid': p.pid, 'command': cmd, 'started_utc': utc(),
                    'log': str(OUT / logfile), 'windows_poll': p.poll()}
    save_state()
    STATE[label]['linux_process'] = process_snapshot(label, tail)
    STATE[label]['linux_pid'] = STATE[label]['linux_process']['pid']
    STATE[label]['windows_poll'] = p.poll()
    save_state()
    assert p.poll() is None, f'{label} exited {p.returncode}'
    return p


def launch():
    STATE.update(json.loads((OUT / 'launch_first.json').read_text(encoding='utf-8')))
    assert STATE['phase'] == 'ready_to_launch'
    assert digest(OUT / 'freeze.json') == STATE['freeze_sha256']
    frozen = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    for row in frozen['files']:
        assert digest(Path(row['source'])) == row['sha256'] == digest(OUT / row['snapshot'])
    assert OUT.is_dir()
    (OUT / 'server_runtime').mkdir()
    tail = [MODEL, '-u', '-m', 'deploy', '--model_path', BASE + '/model', '--port', '8794',
            '--host', '127.0.0.1', '--disable_slurm', '--output_dir', WOUT + '/server_runtime']
    cmd = command(BASE + '/upstream', [
        'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=' + BASE + '/upstream',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN=1',
        'TOKENIZERS_PARALLELISM=false', 'HF_HOME=' + BASE + '/cache/huggingface',
        'XDG_CACHE_HOME=' + BASE + '/cache'], tail)
    STATE['phase'] = 'starting_server'
    server = spawn('server', cmd, tail, 'server.log')
    pid = STATE['server']['linux_pid']
    raw = wsl_python('server_route_ready', """import json, time, urllib.request
from pathlib import Path
pid = PID
deadline = time.monotonic() + 115
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
while time.monotonic() < deadline:
    assert Path('/proc',str(pid)).exists(), 'server Linux process exited'
    try:
        response = opener.open('http://127.0.0.1:8794/openapi.json', timeout=2)
    except (OSError, urllib.error.URLError) as exc:
        print(repr(exc), flush=True)
        time.sleep(0.5)
        continue
    body = response.read()
    Path(OUTPATH, 'server_openapi.json').write_bytes(body)
    schema = json.loads(body)
    assert 'post' in schema.get('paths', {}).get('/act', {}), 'POST /act absent'
    print(json.dumps({'ready':True,'method':'GET','url':response.url,'status':response.status,'post_act_present':True,'server_linux_pid':pid,'inference_probe_calls':0}), flush=True)
    break
else:
    raise TimeoutError('GET /openapi.json not ready within readiness budget')
""".replace('PID', str(pid)).replace('OUTPATH', repr(WOUT)), timeout=120)
    assert server.poll() is None, f'server exited {server.returncode}'
    STATE['service'] = json.loads(raw.decode().splitlines()[-1])
    STATE['service']['runtime_metadata'] = json.loads((OUT / 'server_runtime/info.json').read_text())
    STATE['service']['dtype'] = 'float32; fixed upstream deploy.py lines77-81; no dtype override'
    STATE['phase'] = 'starting_first_episode'
    save_state()
    tail = [SIM, '-u', '/mnt/d/FYP/First_Phase/libero_demo/xvla_native_benchmark.py', '--max-new-episodes', '1']
    cmd = command('/mnt/d/FYP', [
        'PYTHONPATH=/mnt/d/FYP/First_Phase/libero_demo:' + BASE + '/LIBERO:' + BASE + '/upstream',
        'LIBERO_CONFIG_PATH=' + BASE + '/libero_config', 'MUJOCO_GL=egl', 'PYOPENGL_PLATFORM=egl',
        'LD_LIBRARY_PATH=/usr/lib/wsl/lib', 'PYTHONDONTWRITEBYTECODE=1',
        'TOKENIZERS_PARALLELISM=false', 'XDG_CACHE_HOME=' + BASE + '/cache',
        'MPLCONFIGDIR=' + BASE + '/cache/matplotlib', 'NUMBA_CACHE_DIR=' + BASE + '/cache/numba'], tail)
    client = spawn('client', cmd, tail, 'first_episode.log')
    STATE['phase'] = 'first_episode_started_not_awaited'
    STATE['slot'] = 'libero_10/task_00/episode_0'
    STATE['max_new_episodes'] = 1
    STATE['physical_success'] = 'not evaluated by launcher'
    STATE['server']['windows_poll'] = server.poll()
    STATE['client']['windows_poll'] = client.poll()
    STATE['status_observed_utc'] = utc()
    save_state()
    print(json.dumps(STATE, ensure_ascii=False))


if __name__ == '__main__':
    try:
        if sys.argv[1:] == ['prepare']:
            prepare()
        elif sys.argv[1:] == ['launch']:
            launch()
        else:
            raise ValueError('expected prepare or launch')
    except Exception:
        error = traceback.format_exc()
        if OUT.is_dir():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            (OUT / ('launch_failure_' + stamp + '.log')).write_bytes(error.encode('utf-8'))
            for label, proc in PROCESSES.items():
                STATE[label]['windows_poll'] = proc.poll()
            STATE['phase'] = 'stopped_on_failure_no_retry'
            STATE['error'] = error
            save_state()
        print(error, file=sys.stderr)
        sys.exit(1)
