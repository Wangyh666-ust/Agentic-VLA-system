"""Offline, fixed-criteria archive worker. Never constructs an environment or policy."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/mnt/d/FYP') if os.name != 'nt' else Path('D:/FYP')
BASE = ROOT / 'First_Phase/scene_demo/results/2026-10-12-xvla-native-benchmark'
POST = BASE / 'postprocess'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
EXPECTED = dict(zip(SUITES, (48, 50, 50, 49)))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def require(ok, label):
    if not ok:
        raise AssertionError(label)


def slots():
    expected = {f'{s}/task_{t:02d}/episode_{e}' for s in SUITES for t in range(10) for e in range(5)}
    actual = {str(p.parent.relative_to(BASE)).replace('\\', '/') for s in SUITES for p in (BASE / s).glob('task_*/episode_*/result.json')}
    require(actual == expected, '200 fixed unique slots')
    summary = load(BASE / 'summary.json')
    require(summary['complete'] is True and summary['unknown_paths'] == [], 'summary complete/unknown')
    rows = []
    for slot in sorted(expected):
        folder = BASE / slot
        require(sorted(p.name for p in folder.glob('attempt*')) == ['attempt_001'], slot + ' unique attempt')
        r = load(folder / 'result.json')
        s, t, e = slot.split('/')
        require((r['suite'], r['task_id'], r['episode']) == (s, int(t[5:]), int(e[8:])), slot + ' identity')
        require(r['status'] == 'completed' and r['valid'] is True, slot + ' completed valid')
        require(r == load(folder / 'attempt_001/result.json'), slot + ' slot/attempt result equal')
        rows.append((slot, r))
    return rows


def proc(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return {'exists': False}
    stat = (p / 'stat').read_text()
    fields = stat.rsplit(')', 1)[1].split()
    return {'exists': True, 'argv': [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x],
            'start_ticks': fields[19], 'state': fields[0], 'stat': stat}


def gpu():
    exe = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    commands = [[exe], [exe, '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv'],
                [exe, '--query-compute-apps=pid,process_name,used_memory', '--format=csv']]
    out = []
    for cmd in commands:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out.append({'utc': now(), 'command': cmd, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
        require(p.returncode == 0, 'nvidia-smi capture')
    return {'measurement': 'current snapshot, NOT peak memory', 'commands': out}


def stop_server():
    require(os.name != 'nt', 'server identity requires original Ubuntu /proc')
    rows = slots()
    launch = load(BASE / 'launch_remaining.json')
    client, server = proc(878), proc(423)
    record = {'utc': now(), 'unique_completed_slots': len(rows), 'client_before': client, 'server_before': server,
              'new_model_calls': 0, 'new_physical_actions': 0, 'signals': []}
    save(POST / 'gpu_before_stop.json', gpu())
    require(not client['exists'], 'client878 must have exited')
    expected = launch['final_preflight']['server']
    require(server['exists'] and server['argv'] == expected['argv'] and
            server['start_ticks'] == launch['final_preflight']['server_start_ticks'], 'server423 full identity')
    record['full_argv_and_starttime_match'] = True
    # Recheck immediately before sending the sole permitted signal.
    again = proc(423)
    require(again['argv'] == server['argv'] and again['start_ticks'] == server['start_ticks'], 'identity immediately before SIGTERM')
    os.kill(423, signal.SIGTERM)
    record['signals'].append({'pid': 423, 'signal': 'SIGTERM', 'utc': now()})
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and proc(423)['exists']:
        time.sleep(0.5)
    record['server_after'] = proc(423)
    record['client_after'] = proc(878)
    record['gpu_after'] = gpu()
    record['ok'] = not record['server_after']['exists'] and not record['client_after']['exists']
    record['finished_utc'] = now()
    save(POST / 'server_stop.json', record)
    require(record['ok'], 'server did not exit; no escalation allowed')
    print(json.dumps(record, ensure_ascii=False, indent=2))


def native_path(value):
    if value.startswith('D:\\FYP\\'):
        return ROOT / value[len('D:\\FYP\\'):].replace('\\', '/')
    prefix = '\\\\wsl.localhost\\Ubuntu'
    if value.startswith(prefix):
        return Path(value[len(prefix):].replace('\\', '/'))
    return Path(value)


def inventory(paths):
    return {p.relative_to(BASE).as_posix(): {'sha256': digest(p), 'size': p.stat().st_size}
            for p in sorted(paths) if p.is_file()}


def run_record(command, name, cwd=ROOT):
    p = subprocess.run(command, cwd=cwd, capture_output=True, timeout=600)
    (POST / (name + '.stdout.log')).write_bytes(p.stdout)
    (POST / (name + '.stderr.log')).write_bytes(p.stderr)
    save(POST / (name + '.json'), {'command': command, 'cwd': str(cwd), 'returncode': p.returncode,
                                'utc': now(), 'stdout': p.stdout.decode(errors='replace'), 'stderr': p.stderr.decode(errors='replace')})
    require(p.returncode == 0, name + ' exit code')
    return p


def source_check():
    freeze = load(BASE / 'freeze.json')
    launch = load(BASE / 'launch_remaining.json')
    require(digest(BASE / 'freeze.json') == launch['freeze_sha256'], 'freeze itself unchanged')
    checked = []
    for f in freeze['files']:
        source, snapshot = native_path(f['source']), BASE / f['snapshot']
        require(digest(source) == digest(snapshot) == f['sha256'], 'frozen source/snapshot: ' + f['source'])
        checked.append({'source': str(source), 'snapshot': f['snapshot'], 'sha256': f['sha256'], 'ok': True})
    for name, sha in launch['first_episode_sha256'].items():
        require(digest(BASE / name) == sha, 'first episode preserved: ' + name)
    require(digest(BASE / 'first_acceptance.json') == launch['first_acceptance_sha256'], 'first acceptance unchanged')
    return checked


def audit():
    import socket
    def blocked(*args, **kwargs):
        raise RuntimeError('Offline archive: network connections prohibited')
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    import numpy as np
    import imageio_ffmpeg
    from evaluation.libero.libero_client import LiberoAbsActionProcessor
    processor = LiberoAbsActionProcessor()
    require(load(POST / 'server_stop.json')['ok'], 'server stopped before audit')
    rows = slots()
    initial_files = [p for p in BASE.rglob('*') if p.is_file() and POST not in p.parents]
    baseline = inventory(initial_files)
    if (POST / 'original_inventory.json').exists():
        require(load(POST / 'original_inventory.json') == baseline, 'original inventory preserved across offline tool retry')
    else:
        save(POST / 'original_inventory.json', baseline)
    registry = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
    old = load(registry)
    require(len(old['entries']) == 41, 'original 41 registry entries')
    if (POST / 'registry_before.json').exists():
        require((POST / 'registry_before.json').read_bytes() == registry.read_bytes(), 'registry baseline unchanged')
    else:
        shutil.copyfile(registry, POST / 'registry_before.json')
    index = ROOT / 'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
    if (POST / 'index_before.snapshot').exists():
        require((POST / 'index_before.snapshot').read_bytes() == index.read_bytes(), 'index baseline unchanged')
    else:
        shutil.copyfile(index, POST / 'index_before.snapshot')
    source_rows = source_check()
    metadata = load(BASE / 'metadata.json')
    config = metadata['configuration']
    require(config == load(BASE / 'source_snapshot/20261012-xvla-native-benchmark.json')['configuration'], 'actual proposal configuration')
    fp = hashlib.sha256(json.dumps({'config': config, 'runner': metadata['runner_sha'],
                                   'author': metadata['author_sha']}, sort_keys=True).encode()).hexdigest()
    require(fp == metadata['fingerprint'], 'recomputed metadata fingerprint')
    history = [e for e in old['entries'] if 'benchmark-success-rate' in e['question_keys']]
    save(POST / 'history_review.json', history)
    proposal = {'schema_version': 1, 'id': '20261012-xvla-native-offline-archive', 'intent': 'analyze_existing',
                'question_key': 'benchmark-success-rate', 'purpose': '按用户固定判据离线核验既有X-VLA 200次结果并归档，零新增模型与环境动作。',
                'prior_evidence': [e['id'] for e in history],
                'history_review': [{'experiment_id': e['id'], 'disposition': 'reuse',
                                    'reason': '复用旧SmolVLA原生benchmark作非配对历史背景；本次核验已完成X-VLA原始记录，不新跑实验。'} for e in history],
                'new_physical_actions': 0,
                'analysis_steps': ['解析200唯一slot及全部steps/calls/response', '原作者动作数值转换与float32发送字节比对',
                                   '完整解码200原视频到null统计帧数', '冻结来源与报告证据哈希归档']}
    save(POST / 'analysis_proposal.json', proposal)
    gate = run_record([sys.executable, str(ROOT / '.agents/skills/fyp-experiment-review/scripts/review.py'),
                       'check', '--proposal', str(POST / 'analysis_proposal.json')], 'review_analysis')
    require(json.loads(gate.stdout)['ok'] is True, 'offline analysis review ok')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    episodes, audits, videos, failures = [], [], [], []
    with (POST / 'video_decode.log').open('xb') as video_log:
        for slot, r in rows:
            folder = BASE / slot / 'attempt_001'
            steps = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
            calls = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines()]
            policy = [s for s in steps if s['phase'] == 'policy']
            settle = [s for s in steps if s['phase'] == 'settling']
            n = r['policy_actions']
            checks = {}
            def check(name, ok):
                checks[name] = bool(ok)
                require(ok, slot + ': ' + name)
            check('result_bytes_equal', digest(BASE / slot / 'result.json') == digest(folder / 'result.json'))
            check('fingerprint_matches_metadata', r['fingerprint'] == fp)
            check('row_counts_and_order', len(policy) == n and len(settle) == r['setup_actions'] == 10 and
                  [s['phase'] for s in steps] == ['settling'] * 10 + ['policy'] * n)
            check('policy_done_equals_independent', all(type(s['done']) is bool and s['done'] is s['independent_success'] for s in policy))
            true_indices = [i + 1 for i, s in enumerate(policy) if s['done']]
            check('first_success_last_or_all_false', true_indices == ([n] if r['success'] else []))
            check('result_score_consistency', type(r['success']) is bool and r['success'] is r['independent_success'] and
                  r['author_score'] == int(r['success']) and policy[-1]['done'] is r['success'])
            check('horizon_and_end_reason', r['horizon'] == (900 if r['suite'] == 'libero_10' else 800) and
                  r['end_reason'] == ('native_success' if r['success'] else 'budget_exhausted') and
                  (0 < n <= r['horizon']) and (r['success'] or n == r['horizon']))
            c = math.ceil(n / 30)
            check('call_counts', c == len(calls) == r['model_calls'] == r['model_requests'])
            check('numbered_request_response_files', sorted(p.name for p in folder.glob('request_*.json.gz')) ==
                  [f'request_{i:03d}.json.gz' for i in range(1, c + 1)] and
                  sorted(p.name for p in folder.glob('response_*.json')) == [f'response_{i:03d}.json' for i in range(1, c + 1)])
            with gzip.open(folder / 'request_001.json.gz', 'rt') as f:
                request = json.load(f)
            language = request['language_instruction']
            check('first_request_domain_and_denoise', request['domain_id'] == 3 and request['steps'] == 10 and isinstance(language, str) and bool(language))
            author_score = load(folder / 'results.json')
            check('author_results_score', author_score == {f'sim/{r["suite"]}/{language}': float(r['success'])})
            converted = []
            for i, call in enumerate(calls, 1):
                raw = np.array(load(folder / f'response_{i:03d}.json')['action'])
                check(f'response_{i:03d}_finite_30x20', raw.shape == (30, 20) and np.isfinite(raw).all())
                check(f'call_{i:03d}_identity', call['call'] == i and call['shape'] == [30, 20] and math.isfinite(call['seconds']) and call['seconds'] >= 0)
                target_axis = processor.Rotate6D_to_AxisAngle(raw[:, 3:9])
                chunk = np.concatenate([raw[:, :3], target_axis, raw[:, 9:10]], axis=-1).astype(np.float32)
                chunk[:, -1] = np.where(chunk[:, -1] > 0.5, 1.0, -1.0)
                converted.append(chunk)
            expected = np.concatenate(converted)[:n]
            sent = np.array([s['action'] for s in policy], dtype=np.float32)
            all_sent = np.array([s['action'] for s in steps], dtype=np.float32)
            check('all_sent_finite_7d_gripper', all_sent.shape == (n + 10, 7) and np.isfinite(all_sent).all() and np.isin(all_sent[:, -1], [-1, 1]).all())
            check('author_conversion_float32_element_equal', np.array_equal(expected, sent))
            check('author_conversion_float32_byte_equal', expected.tobytes() == sent.tobytes())
            check('native_control', r['controller_name'] == 'OSC_POSE' and r['controller_use_delta'] is False and
                  r['controller_control_dim'] == 6 and r['control_freq'] == 20 and r['robot_model'] in ('MountedPanda', 'OnTheGroundPanda'))
            check('init_and_seed', r['init_index'] == r['episode'] and r['seed'] == 142 + r['episode'])
            check('wall_finite', math.isfinite(r['wall_seconds']) and r['wall_seconds'] >= 0)
            check('bddl_and_init_hash', digest(Path(r['bddl_path'])) == r['bddl_sha'] and digest(Path(r['init_path'])) == r['init_sha'])
            mp4s = list(folder.glob('*.mp4'))
            check('one_original_video', len(mp4s) == 1)
            video = mp4s[0]
            command = [ffmpeg, '-hide_banner', '-nostdin', '-v', 'info', '-xerror', '-i', str(video),
                       '-map', '0:v:0', '-an', '-vsync', '0', '-progress', 'pipe:1', '-nostats', '-f', 'null', '-']
            decoded = subprocess.run(command, capture_output=True, timeout=120)
            video_log.write(json.dumps({'slot': slot, 'command': command, 'returncode': decoded.returncode}).encode() + b'\n')
            video_log.write(decoded.stdout + decoded.stderr + b'\n')
            check('video_full_decode_exit_zero', decoded.returncode == 0)
            frames = re.findall(rb'^frame=(\d+)\s*$', decoded.stdout, re.M)
            fps = re.findall(rb', ([0-9.]+) fps[,\s]', decoded.stderr)
            check('video_decode_count_present', bool(frames) and bool(fps) and b'progress=end' in decoded.stdout)
            count, rate = int(frames[-1]), float(fps[0])
            check('video_frames_equal_policy', count == n)
            check('author_video_fps30', rate == 30)
            video_path = video.relative_to(BASE).as_posix()
            videos.append({'slot': slot, 'path': video_path, 'frames': count, 'fps': rate, 'method': 'full ffmpeg decode to null, vsync0', 'ok': True})
            episode = {'suite': r['suite'], 'task': r['task_id'], 'episode': r['episode'], 'language': language,
                       'success': r['success'], 'policy_actions': n, 'setup_actions': 10, 'model_calls': c,
                       'wall_seconds': r['wall_seconds'], 'result_path': slot + '/result.json', 'video_path': video_path}
            episodes.append(episode)
            audits.append({'slot': slot, 'checks': checks, 'true_policy_indices_1based': true_indices,
                           'sent_float32_sha256': hashlib.sha256(sent.tobytes()).hexdigest(),
                           'converted_float32_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                           'action_elements': sent.size, 'action_bytes': sent.nbytes})
            if not r['success']:
                failures.append(dict(episode, end_reason=r['end_reason']))
            if len(episodes) % 25 == 0:
                print(f'audited {len(episodes)}/200, full video decode included', flush=True)
    totals = {'completed': len(episodes), 'success': sum(e['success'] for e in episodes), 'failed': len(failures),
              'model_calls': sum(e['model_calls'] for e in episodes), 'policy_actions': sum(e['policy_actions'] for e in episodes),
              'setup_actions': sum(e['setup_actions'] for e in episodes), 'videos': len(videos), 'video_frames': sum(v['frames'] for v in videos),
              'episodewall_sum': sum(e['wall_seconds'] for e in episodes), 'episodewall_min': min(e['wall_seconds'] for e in episodes),
              'episodewall_max': max(e['wall_seconds'] for e in episodes)}
    require((totals['completed'], totals['success'], totals['failed'], totals['policy_actions'], totals['setup_actions'], totals['videos']) ==
            (200, 197, 3, 32184, 2000, 200), 'fixed aggregate counts')
    suites = {s: {'completed': sum(e['suite'] == s for e in episodes), 'success': sum(e['success'] for e in episodes if e['suite'] == s)} for s in SUITES}
    require(all(v == {'completed': 50, 'success': EXPECTED[s]} for s, v in suites.items()), 'four suite counts')
    require({e['result_path'] for e in failures} == {'libero_10/task_03/episode_2/result.json', 'libero_spatial/task_04/episode_1/result.json', 'libero_spatial/task_07/episode_2/result.json'}, 'three failure identities')
    with (BASE / 'tasks.csv').open(newline='') as f:
        tasks = list(csv.DictReader(f))
    require(len(tasks) == 40, '40 original tasks.csv rows')
    for t in tasks:
        group = [e for e in episodes if e['suite'] == t['suite'] and e['task'] == int(t['task_id'])]
        require(len(group) == int(t['valid_episodes']) == 5 and sum(e['success'] for e in group) == int(t['successes']), 'tasks.csv independently matches')
    summary = load(BASE / 'summary.json')
    require(summary['valid_episodes'] == 200 and summary['successes'] == 197 and summary['rate'] == .985 and
            summary['policy_actions_total'] == 32184 and summary['setup_actions_total'] == 2000, 'original summary matches')
    proofs = []
    for name, usage in [('author', (7553, 9166)), ('author_fix', (10107, 7374)), ('author_final', (1369, 1304))]:
        proof = load(BASE / 'authoring' / name / 'proof.json')
        require((proof['usage']['input_tokens'], proof['usage']['output_tokens']) == usage and proof['call_count'] == 1 and proof['http_status'] == 200, name + ' actual usage')
        require(digest(BASE / 'authoring' / name / 'request.json') == proof['request_sha256'] and
                digest(BASE / 'authoring' / name / 'response.json') == proof['response_sha256'], name + ' proof bytes')
        proofs.append({'phase': name, 'model': proof['model'], 'usage': proof['usage'], 'proof': f'authoring/{name}/proof.json'})
    usage_totals = {key: sum(p['usage'][key] for p in proofs) for key in ['input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens']}
    usage_totals['total_tokens'] = usage_totals['input_tokens'] + usage_totals['output_tokens']
    require(inventory(initial_files) == baseline, 'all original campaign files unchanged after audit')
    source_check()
    result = {'ok': True, 'utc': now(), 'executor_model': 'openai/gpt-6-astra', 'reasoning_effort': 'not exposed',
              'execution': 'local offline worker; no subagent or model-generation call',
              'totals': totals, 'per_suite': suites, 'fingerprint': fp, 'configuration': config,
              'sourcehash_checks': source_rows, 'original_files_unchanged': True, 'original_file_count': len(baseline),
              'assertions': {'fixed_200_unique_slots': True, 'each_task_five': True, 'summary_complete_unknown_empty': True,
                             'independent_score_consistency_all_policy_rows': True, 'author_conversion_element_and_byte_equal_all_policy_actions': True,
                             'raw_response_finite_30x20_all_calls': True, 'calls_ceil_policy_div30': True, 'all_200_videos_full_decode_frames_match_fps30': True,
                             'all_400_result_fingerprints_match_metadata': True, 'freeze_and_sources_unchanged': True, 'original_tasks_csv_preserved': True},
              'scoring': '独立读取原生谓词；same native goal as done, not external vision or extra real-world physical annotation',
              'robot_models': {name: sum(r['robot_model'] == name for _, r in rows) for name in sorted({r['robot_model'] for _, r in rows})},
              'failures': failures, 'episodes': episodes, 'episode_assertions': audits, 'videos': videos,
              'deepseek_actual_calls': 3, 'deepseek_calls': proofs, 'deepseek_usage_totals': usage_totals,
              'context_repair': 'GPT诊断与worker机械定位；保留早期失败和修正记录，不能称3次实现均一次通过。',
              'postprocess_new_model_calls': 0, 'postprocess_new_physical_actions': 0, 'env_constructed': False, 'env_step_calls': 0,
              'network_connections_blocked': True, 'auxiliary_actions': 0, 'Hermes_Qwen_calls': 0, 'training_runs': 0,
              'reset_internal_physics_steps': 'not separately observed', 'control_hz': 20, 'author_video_fps': 30,
              'wall_time_note': 'sum of episode execution times, excludes development/setup/first-episode pause; video duration is not control time',
              'server_stop': 'postprocess/server_stop.json'}
    save(POST / 'audit.json', result)
    with (BASE / 'episodes.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    print(json.dumps({'totals': totals, 'per_suite': suites, 'failures': failures, 'deepseek_usage': usage_totals,
                      'robot_models': result['robot_models']}, ensure_ascii=False, indent=2))


def prepare():
    result = load(POST / 'audit.json')
    require(result['ok'], 'full offline audit passed')
    baseline = load(POST / 'original_inventory.json')
    require(inventory(BASE / name for name in baseline) == baseline, 'original inventory unchanged before prepare')
    selected = [('libero_10', 0, 0, 'long00_ep0_two_cans'), ('libero_10', 4, 0, 'long04_ep0_two_mugs'),
                ('libero_10', 8, 0, 'long08_ep0_two_moka_pots'), ('libero_10', 9, 0, 'long09_ep0_microwave'),
                ('libero_goal', 8, 0, 'goal08_ep0_bowl'), ('libero_goal', 9, 0, 'goal09_ep0_wine'),
                ('libero_10', 3, 2, 'failed_long03_ep2'), ('libero_spatial', 4, 1, 'failed_spatial04_ep1'),
                ('libero_spatial', 7, 2, 'failed_spatial07_ep2')]
    (BASE / 'videos').mkdir(exist_ok=True)
    copies = []
    for suite, task, episode, name in selected:
        row = next(e for e in result['episodes'] if (e['suite'], e['task'], e['episode']) == (suite, task, episode))
        src, dst = BASE / row['video_path'], BASE / 'videos' / (name + '.mp4')
        require(not dst.exists(), 'do not overwrite representative video')
        shutil.copyfile(src, dst)
        require(digest(src) == digest(dst), 'representative video byte-identical')
        copies.append({'source': row['video_path'], 'copy': dst.relative_to(BASE).as_posix(), 'sha256': digest(dst),
                       'size': dst.stat().st_size, 'fps': 30, 'transcoded': False})
    save(POST / 'representative_videos.json', copies)
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_xvla_native_benchmark.py']
    started = now()
    p = subprocess.run(command, cwd=ROOT / 'First_Phase/libero_demo', stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=120)
    with (BASE / 'software/final_pytest.log').open('xb') as f:
        f.write(p.stdout)
    text = p.stdout.decode(errors='replace')
    save(POST / 'final_pytest.json', {'command': command, 'cwd': str(ROOT / 'First_Phase/libero_demo'),
                                     'started_utc': started, 'finished_utc': now(), 'returncode': p.returncode,
                                     'raw_log': 'software/final_pytest.log', 'raw_log_sha256': digest(BASE / 'software/final_pytest.log'),
                                     'environment': {k: os.environ.get(k) for k in ('PYTHONPATH', 'LIBERO_CONFIG_PATH', 'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'PYTHONDONTWRITEBYTECODE')},
                                     'new_model_calls': 0, 'new_physical_actions': 0})
    print(text, flush=True)
    require(p.returncode == 0 and re.search(r'\b11 passed\b', text) is not None, 'exact existing 11 pytest passed')
    require(inventory(BASE / name for name in baseline) == baseline, 'original files unchanged after pytest')
    source_check()
    result['representative_videos'] = copies
    result['software_test'] = load(POST / 'final_pytest.json')
    result['assertions']['existing_11_pytest_passed'] = True
    result['assertions']['nine_representative_original_video_copies_sha_equal'] = True
    result['assertions']['original_files_unchanged_after_pytest'] = True
    result['assertions']['server423_stopped_gpu_compute_process_absent'] = load(POST / 'server_stop.json')['ok']
    result['archive_status_note'] = 'Dataset verification passed. Registry validation outcome is in postprocess/finalize.log; manifest excludes that active log and itself.'
    save(BASE / 'final_verification.json', result)
    for suite in SUITES:
        for task in range(10):
            group = [e for e in result['episodes'] if e['suite'] == suite and e['task'] == task]
            print(json.dumps({'suite': suite, 'task': task, 'language': group[0]['language'],
                              'episodes_0_4': [e['success'] for e in group], 'successes': sum(e['success'] for e in group)}, ensure_ascii=False))


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_manifest():
    excluded = {'manifest.json': 'self hash excluded', 'postprocess/finalize.log': 'actively written finalization/validate receipt; verify separately after close'}
    files = inventory(p for p in BASE.rglob('*') if p.is_file() and p.relative_to(BASE).as_posix() not in excluded)
    obj = {'schema_version': 1, 'created_utc': now(), 'algorithm': 'SHA256', 'path_base': '.',
           'file_count': len(files), 'total_bytes': sum(v['size'] for v in files.values()), 'exclusions': excluded, 'files': files}
    (BASE / 'manifest.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return obj


def verify_manifest():
    manifest = load(BASE / 'manifest.json')
    actual = inventory(p for p in BASE.rglob('*') if p.is_file() and p.relative_to(BASE).as_posix() not in manifest['exclusions'])
    require(manifest['files'] == actual, 'all stable campaign files covered and SHA/size equal manifest')
    return {'ok': True, 'file_count': len(actual), 'total_bytes': sum(v['size'] for v in actual.values())}


def finalize():
    require(not (POST / 'finalize.log').exists(), 'finalize only once; preserve failed receipt if any')
    with (POST / 'finalize.log').open('xb') as log:
        def record(obj):
            raw = (json.dumps(obj, ensure_ascii=False) + '\n').encode()
            log.write(raw)
            log.flush()
            print(raw.decode(), end='', flush=True)
        record({'started_utc': now(), 'new_model_calls': 0, 'new_physical_actions': 0})
        data = load(BASE / 'final_verification.json')
        require(data['ok'] and len(data['episodes']) == 200, 'final verification')
        baseline = load(POST / 'original_inventory.json')
        require(inventory(BASE / name for name in baseline) == baseline, 'all original evidence preserved')
        source_check()
        with (BASE / 'episodes.csv').open(encoding='utf-8', newline='') as f:
            csv_rows = list(csv.DictReader(f))
        require(len(csv_rows) == 200 and len({(r['suite'], r['task'], r['episode']) for r in csv_rows}) == 200, 'CSV 200 unique rows')
        for r, e in zip(csv_rows, data['episodes']):
            require(all(r[k] == str(e[k]) for k in r), 'CSV matches independently audited episode')
        for copy in data['representative_videos']:
            require(digest(BASE / copy['source']) == digest(BASE / copy['copy']) == copy['sha256'], 'representative SHA')
        index = ROOT / 'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
        old_index = (POST / 'index_before.snapshot').read_bytes()
        require(index.read_bytes().startswith(old_index), 'historical index bytes unchanged')
        require(index.read_bytes()[len(old_index):].count(b'20261012-xvla-native-benchmark') == 1, 'one appended campaign index row')
        registry = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
        before = load(POST / 'registry_before.json')
        current = load(registry)
        require(current == before and len(current['entries']) == 41, 'registry still baseline before append')
        shutil.copyfile(Path(__file__), POST / 'xvla_20261012_archive.py')
        write_manifest()
        stable = ['README.md', 'CONCLUSIONS.md', 'summary.json', 'tasks.csv', 'episodes.csv', 'final_verification.json',
                  'manifest.json', 'freeze.json', 'first_acceptance.json', 'metadata.json']
        stable += [p.relative_to(BASE).as_posix() for p in sorted((BASE / 'source_snapshot').rglob('*')) if p.is_file()]
        stable += [e['result_path'] for e in data['episodes']]
        evidence = [{'path': (BASE / name).relative_to(ROOT).as_posix(), 'sha256': digest(BASE / name)} for name in stable]
        entry = {'id': '20261012-xvla-native-benchmark', 'title': 'X-VLA原生40任务×5次缩小样本复现与离线核验',
                 'date': '2026-10-12', 'question_keys': ['benchmark-success-rate'],
                 'tags': ['X-VLA', 'LIBERO', 'native', 'benchmark', 'offline-verification'],
                 'conditions': ['作者公开权重与客户端协议；FP32在RTX5070Laptop8GB执行，40原生任务每task5，init0..4/seed142..146。',
                                '原作者30动作chunk/10denoise/domain3/absolute OSC_POSE；800/900预算；原语言、原init、两路图像；模型RNG沿server默认未固定。',
                                '成功以done与独立读取原生谓词一致为准，无附加释放、退离或home条件；MountedPanda125例、OnTheGroundPanda75例。',
                                '实际配置与原proposal和metadata一致；Python3.10客户端、GPUtorch2.11cu128、MuJoCo2.3.7。'],
                 'findings': ['200/200完整且唯一，197/200=98.5%；Spatial48/50、Object50/50、Goal50/50、Long49/50。',
                              '实际1160次X-VLA调用，32184policy+2000settle；全部policy计分一致且原作者数值转换后发送float32字节精确相同。',
                              '200原视频完整解码32184帧，均30fps且逐episode等于policy_actions；9原视频副本SHA一致。',
                              '3失败Long03ep2/Spatial04ep1/Spatial07ep2全部budget_exhausted，原始记录保留，未诊断原因。',
                              'episodewall合计6142.17249584198秒；辅助0、Hermes/Qwen0、训练0。本次离线核验新增模型/物理动作0，指定server已退出。',
                              '作者公开权重/客户端协议下的缩小样本复现，取得相近量级表现。'],
                 'limits': ['不是严格复现论文或超越作者；作者项目公布平均98.1%，本轮每task5而当前CLI默认50，HF格式转换有性能变化提示。',
                            'Python客户端3.10 vs示例3.8；GPU/torch不同；MuJoCo2.3.7为本地固定选择，非作者锁文件；模型RNG未固定。',
                            '独立读取同一原生谓词，不是外部视觉或额外真实物理标注；内部reset物理子步未单独观测。',
                            '20Hz为物理控制频率，30fps为原视频播放速率；episodewall之和不含开发准备及首例暂停。',
                            '旧SmolVLA139/200仅非配对历史背景：模型、控制、预算、MuJoCo同时不同，不能单因素归因。',
                            '持续场景拆分任务与任务间归位接续尚未验证，不能把原生长任务成功等同任意组合已修复。',
                            '3次实际DeepSeek开发请求并非均一次通过；contextrepair为GPT诊断与worker机械定位，历史失败记录保留。'],
                 'configuration': load(BASE / 'source_snapshot/20261012-xvla-native-benchmark.json')['configuration'],
                 'configuration_complete': True, 'evidence': evidence}
        current['entries'].append(entry)
        registry.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        after = load(registry)
        require(after['entries'][:-1] == before['entries'] and len(after['entries']) == 42, 'old41 data fully preserved')
        require({k: v for k, v in after.items() if k != 'entries'} == {k: v for k, v in before.items() if k != 'entries'}, 'registry old top-level fields preserved')
        save(POST / 'registry_preservation.json', {'ok': True, 'old_entry_count': 41, 'new_entry_count': 42,
                 'old_entries_sha256': canonical_hash(before['entries']), 'preserved_entries_sha256': canonical_hash(after['entries'][:-1]),
                 'old_top_level_fields_unchanged': True, 'old_index_prefix_sha256': hashlib.sha256(old_index).hexdigest(),
                 'per_old_entry_sha256': [{'id': e['id'], 'before': canonical_hash(e), 'after': canonical_hash(after['entries'][i])} for i, e in enumerate(before['entries'])]})
        # Include the completed preservation proof, then refresh only this campaign's manifest evidence hash.
        manifest = write_manifest()
        manifest_rel = (BASE / 'manifest.json').relative_to(ROOT).as_posix()
        next(e for e in current['entries'][-1]['evidence'] if e['path'] == manifest_rel)['sha256'] = digest(BASE / 'manifest.json')
        registry.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        require(load(registry)['entries'][:-1] == before['entries'], 'old41 preserved after own manifest hash refresh')
        readme = (BASE / 'README.md').read_text(encoding='utf-8')
        suite_labels = {'Spatial': 'libero_spatial', 'Object': 'libero_object', 'Goal': 'libero_goal', 'Long': 'libero_10'}
        task_lines = re.findall(r'^\| (Spatial|Object|Goal|Long)(\d{2}) \| (.*?) \| (.*?) \| (\d)/5 \|$', readme, re.M)
        require(len(task_lines) == 40, 'README has exactly 40 task result rows')
        for label, task, language, marks, successes in task_lines:
            group = [e for e in data['episodes'] if e['suite'] == suite_labels[label] and e['task'] == int(task)]
            require(len(group) == 5 and all(e['language'] == language for e in group), 'README original language matches requests')
            require(marks.split() == [('✓' if e['success'] else '✗') for e in group] and int(successes) == sum(e['success'] for e in group), 'README per-episode scores match')
        for name in ('README.md', 'CONCLUSIONS.md'):
            text = (BASE / name).read_text(encoding='utf-8')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if not target.startswith(('https://', 'http://', '#')):
                    require((BASE / target.split('#')[0]).exists(), name + ' local evidence link exists: ' + target)
        require(not proc(423)['exists'] and not proc(878)['exists'], 'server and client remain exited')
        record({'registry_old41_unchanged': True, 'registry_entries': 42, 'new_evidence_count': len(evidence),
                'manifest_file_count': manifest['file_count'], 'manifest_total_bytes': manifest['total_bytes'],
                'readme_all40_languages_and_scores_match_raw': True, 'report_local_links_resolve': True})
        command = [sys.executable, str(ROOT / '.agents/skills/fyp-experiment-review/scripts/review.py'), 'validate']
        record({'command': command, 'cwd': str(ROOT), 'raw_output_follows': True})
        p = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        log.write(p.stdout)
        log.write(p.stderr)
        log.flush()
        print(p.stdout.decode(errors='replace'), end='', flush=True)
        require(p.returncode == 0 and json.loads(p.stdout)['ok'] is True, 'review validate ok true')
        require(json.loads(p.stdout)['entry_count'] == 42, 'review validates all42')
        require(inventory(BASE / name for name in baseline) == baseline, 'original inventory final')
        record({'manifest_verification': verify_manifest(), 'review_validate_returncode': p.returncode,
                'old41_unchanged': load(registry)['entries'][:-1] == before['entries'],
                'server423': proc(423), 'client878': proc(878), 'finished_utc': now(), 'ok': True})
    # Receipt hash is printed after closing it; no file includes its own hash.
    print(json.dumps({'closed_finalize_log_sha256': digest(POST / 'finalize.log'),
                      'closed_finalize_log_size': (POST / 'finalize.log').stat().st_size}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['stop', 'audit', 'prepare', 'finalize', 'verify'])
    args = parser.parse_args()
    POST.mkdir(parents=True, exist_ok=True)
    if args.phase == 'stop':
        stop_server()
    elif args.phase == 'audit':
        audit()
    elif args.phase == 'prepare':
        prepare()
    elif args.phase == 'finalize':
        finalize()
    elif args.phase == 'verify':
        print(json.dumps(verify_manifest()))
        print(json.dumps({'closed_finalize_log_sha256': digest(POST / 'finalize.log'), 'size': (POST / 'finalize.log').stat().st_size}))


if __name__ == '__main__':
    main()
