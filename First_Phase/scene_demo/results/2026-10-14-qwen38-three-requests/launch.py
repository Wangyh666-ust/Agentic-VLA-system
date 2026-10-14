"""Run the three user-request simulation cases once; retain outputs and own-process cleanup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/mnt/d/FYP')
OUT = Path(__file__).resolve().parent
PROPOSAL = ROOT / 'First_Phase/scene_demo/plans/20261014-qwen38-three-requests.json'
RUNTIME = OUT / 'runtime'
RUNTIME.mkdir(exist_ok=True)
CASES = ('table_tidy_stove', 'heat_yellow_mug', 'discard_two_cans')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def progress(name):
    directory = OUT / name
    traces = [r for p in directory.rglob('native_steps.jsonl') for r in rows(p)]
    events = [r for p in directory.rglob('process_events.jsonl') for r in rows(p)]
    item = {'case': name,
            'source_counts': {s: sum(r.get('source') == s for r in traces) for s in ('settling', 'prepare', 'vla', 'finish')},
            'release_events': len(events), 'model_responses': len(list(directory.rglob('*_response.json'))),
            'native_success_latest': traces[-1].get('native_success') if traces else None,
            'result_exists': (directory / 'result.json').exists()}
    dump(RUNTIME / (name + '_progress.json'), item)
    print(json.dumps(item), flush=True)


def main():
    assert json.loads((OUT / 'review_check.json').read_text(encoding='utf-8-sig'))['ok'] is True
    assert not any((OUT / name).exists() for name in CASES), 'refuse retry'
    for port in (8769, 8794):
        assert not subprocess.run(['ss', '-H', '-ltn', 'sport = :' + str(port)], capture_output=True).stdout.strip(), 'port occupied'
    ready = json.loads((ROOT / 'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/ready.json').read_text())
    spec = ready['future_server']
    spec['argv'][-1] = str(RUNTIME / 'server_output')
    env = dict(os.environ)
    env.update(spec['env'])
    with (RUNTIME / 'server.stdout').open('wb') as stdout, (RUNTIME / 'server.stderr').open('wb') as stderr:
        server = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    dump(RUNTIME / 'server.json', {'pid': server.pid, 'spec': spec, 'owned': True})
    case = None
    try:
        server_ready = False
        for i in range(60):
            if server.poll() is not None:
                raise RuntimeError('model server exited')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8794/openapi.json', timeout=3) as response:
                    server_ready = 'post' in json.load(response)['paths']['/act']
                if server_ready:
                    break
            except Exception:
                pass
            if i % 6 == 0:
                print('model loading', flush=True)
            time.sleep(5)
        if not server_ready:
            raise RuntimeError('model server not ready')
        for name in CASES:
            # Reuse the recorded successful launch environment exactly, with a pinned planner model.
            previous = ready['future_case_argv']['microwave_mug']
            case_env = dict(os.environ)
            for item in previous[previous.index('env') + 1:]:
                if '=' not in item:
                    break
                key, value = item.split('=', 1)
                case_env[key] = value
            case_env['SCENE_PLANNER_MODEL'] = 'qwen3.8-max'
            argv = ['/home/yhwang/fyp/xvla_native/sim_venv/bin/python', '-u', str(OUT / 'run_case.py'), '--case', name]
            dump(RUNTIME / (name + '_argv.json'), argv)
            dump(RUNTIME / (name + '_environment.json'), {k: case_env[k] for k in (
                'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH', 'LIBERO_CONFIG_PATH', 'MUJOCO_GL', 'PYOPENGL_PLATFORM',
                'LD_LIBRARY_PATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'SCENE_PLANNER_MODEL')})
            with (RUNTIME / (name + '.stdout')).open('wb') as stdout, (RUNTIME / (name + '.stderr')).open('wb') as stderr:
                case = subprocess.Popen(argv, cwd=ROOT, env=case_env, stdout=stdout, stderr=stderr, start_new_session=True)
            dump(RUNTIME / (name + '_pid.json'), {'pid': case.pid, 'owned': True})
            deadline = time.monotonic() + 2400
            while case.poll() is None:
                if time.monotonic() > deadline:
                    os.killpg(case.pid, signal.SIGTERM)
                    case.wait(timeout=20)
                    dump(RUNTIME / (name + '_watchdog.json'), {'reason': 'outer_wall_deadline', 'seconds': 2400})
                    break
                time.sleep(25)
                progress(name)
            dump(RUNTIME / (name + '_exit.json'), {'exitcode': case.returncode, 'launch_count': 1})
            progress(name)
            case = None
    finally:
        if case is not None and case.poll() is None:
            os.killpg(case.pid, signal.SIGTERM)
            try:
                case.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(case.pid, signal.SIGKILL)
                case.wait()
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        dump(RUNTIME / 'server_cleanup.json', {'owned_pid': server.pid, 'exitcode': server.returncode})


if __name__ == '__main__':
    main()
