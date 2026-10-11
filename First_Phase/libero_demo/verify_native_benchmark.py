"""Read-only verification of completed native benchmark evidence."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-native-benchmark'
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
LIMITS = dict(zip(SUITES, [280, 280, 300, 520]))


def require(condition, path, field, detail='mismatch'):
    if not condition:
        raise ValueError(f'{path}: {field}: {detail}')


def read(path):
    require(path.is_file(), path, 'existence', 'required evidence missing')
    return json.loads(path.read_text(encoding='utf-8'))


def verify():
    progress = read(OUT / 'progress.json')
    require(progress.get('status') == 'complete', OUT / 'progress.json', 'status',
            f"expected complete, got {progress.get('status')!r}")
    rows, per_suite = [], {}
    scored_total = padding_total = calls_total = 0
    for suite in SUITES:
        suite_successes = 0
        for task_id in range(10):
            folder = OUT / f'{suite}_{task_id:02d}'
            audit_path = folder / 'audit.json'
            info_path = folder / 'eval_info.json'
            input_path = folder / 'first_input.json'
            action_path = folder / 'actions.jsonl'
            audit, info, first = read(audit_path), read(info_path), read(input_path)
            require(action_path.is_file(), action_path, 'existence')
            require(audit.get('valid') is True, audit_path, 'valid')
            flags, episodes = audit['native_successes'], audit['episodes']
            require(len(flags) == 5 and all(type(v) is bool for v in flags), audit_path, 'native_successes')
            require(len(episodes) == 5 and all(type(e['success']) is bool for e in episodes), audit_path, 'episodes.success')
            require([e['success'] for e in episodes] == flags, audit_path, 'episodes.success/native_successes')
            require(len(info['per_task']) == 1, info_path, 'per_task')
            task = info['per_task'][0]
            require(task['task_group'] == suite and type(task['task_id']) is int and task['task_id'] == task_id,
                    info_path, 'task_group/task_id')
            sdk_flags = task['metrics']['successes']
            require(len(sdk_flags) == 5 and all(type(v) is bool for v in sdk_flags) and sdk_flags == flags,
                    info_path, 'metrics.successes')
            require(task['n_episodes'] == 5 and task['n_success'] == sum(flags), info_path, 'n_episodes/n_success')
            calls = audit['policy_select_calls']
            require(type(calls) is int and calls > 0, audit_path, 'policy_select_calls')
            language = audit['language']
            require(len(language) == 5 and all(type(v) is str for v in language) and
                    all(v == language[0] for v in language), audit_path, 'language')
            require(len(first['task']) == 5 and all(type(v) is str for v in first['task']) and
                    [v.rstrip('\r\n') for v in first['task']] == language, input_path, 'task')
            for key, expected in [('observation.images.image', [5, 3, 256, 256]),
                                  ('observation.images.image2', [5, 3, 256, 256]),
                                  ('observation.state', [5, 8])]:
                require(first[key] == expected, input_path, key)
            for index, episode in enumerate(episodes):
                steps = episode['scored_steps']
                expected = 'native_success' if flags[index] else 'budget_exhausted'
                require(type(steps) is int and 0 < steps <= LIMITS[suite] and
                        (flags[index] or steps == LIMITS[suite]), audit_path, f'episodes[{index}].scored_steps')
                require(episode['end_condition'] == expected, audit_path, f'episodes[{index}].end_condition')
                require(type(episode['padding_steps']) is int and episode['padding_steps'] >= 0,
                        audit_path, f'episodes[{index}].padding_steps')
            scored, padding, observed, last_success = [0]*5, [0]*5, [False]*5, [False]*5
            with action_path.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    action = json.loads(line)
                    field = f'line {line_number}'
                    env = action['env']
                    require(type(env) is int and 0 <= env < 5, action_path, field + '.env')
                    values = action['action']
                    require(type(values) is list and len(values) == 7 and
                            all(type(v) in (int, float) and math.isfinite(v) for v in values),
                            action_path, field + '.action')
                    require(type(action['padding']) is bool and type(action['is_success']) is bool,
                            action_path, field + '.padding/is_success')
                    require(type(action['scored_steps']) is int, action_path, field + '.scored_steps')
                    if action['padding']:
                        padding[env] += 1
                    else:
                        scored[env] += 1
                        require(action['scored_steps'] == scored[env], action_path, field + '.scored_steps',
                                f"expected {scored[env]}, got {action['scored_steps']}")
                        observed[env] |= action['is_success']
                        last_success[env] = action['is_success']
            require(scored == [e['scored_steps'] for e in episodes], action_path, 'scored action counts')
            require(padding == [e['padding_steps'] for e in episodes], action_path, 'padding action counts')
            require(observed == flags, action_path, 'nonpadding native success OR')
            require(all(not flag or last_success[i] for i, flag in enumerate(flags)), action_path, 'last scored success')
            rows.append(dict(suite=suite, task_id=task_id, episodes=5, successes=sum(flags), scored_steps=sum(scored)))
            suite_successes += sum(flags)
            scored_total += sum(scored)
            padding_total += sum(padding)
            calls_total += calls
        per_suite[suite] = dict(completed_tasks=10, valid_episodes=50, successes=suite_successes,
                                success_rate=suite_successes / 50)
    successes = sum(row['successes'] for row in rows)
    summary_path = OUT / 'summary.json'
    summary = read(summary_path)
    require(summary['tasks'] == rows, summary_path, 'tasks')
    require(len(rows) == 40 and sum(row['episodes'] for row in rows) == 200 and
            summary['completed_tasks'] == 40 and summary['valid_episodes'] == 200 and
            summary['successes'] == successes, summary_path, 'totals')
    require(type(summary['success_rate']) in (int, float) and
            abs(summary['success_rate'] - successes / 200) <= 1e-12, summary_path, 'success_rate')
    manifest_path = OUT / 'completed_before_resume_sha256.json'
    manifest = read(manifest_path)
    require(len(manifest['files']) == 88, manifest_path, 'files count')
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), manifest_path, relative, 'path outside repository')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        require(digest.hexdigest() == expected, path, 'sha256', 'original evidence changed')
    interrupted = OUT / 'interrupted_attempts'
    excluded = sorted(str(path.relative_to(OUT)) for path in interrupted.glob('*/*') if path.is_dir())
    return dict(ok=True, completed_tasks=40, valid_episodes=200, successes=successes,
                success_rate=successes/200, per_suite=per_suite,
                scored_vla_env_actions=scored_total, padding_vla_env_actions=padding_total,
                policy_select_calls_for_valid_tasks=calls_total, original_88_files_unchanged=True,
                excluded_interrupted_groups=excluded, zero_new_physical_actions=True,
                zero_new_vlm_calls=True, implementation_model='gpt-6.1-sol',
                implementation_reasoning_effort='low')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(json.dumps(verify(), separators=(',', ':')))
    except Exception as exc:
        print(json.dumps(dict(ok=False, error=str(exc)), separators=(',', ':')), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
