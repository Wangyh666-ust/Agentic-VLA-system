"""Audit existing evidence only; never invokes models, policy or simulator."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CASES = ('table_tidy_stove', 'heat_yellow_mug', 'discard_two_cans')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def local_path(value):
    return ROOT / value.split('/FYP/', 1)[1]


def check_request(path):
    record = read(path)
    response = read(path.with_name(path.name.replace('_request.json', '_response.json')))
    frames = record['context']['sensor_frames']
    assert len(frames) == len(record['images'])
    content = [{'type': 'text', 'text': record['prompt']}]
    for frame, image in zip(frames, record['images']):
        data = local_path(image['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == image['sha256']
        content.extend([{'type': 'text', 'text': f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')}}])
    body = {'model': response['model'], 'temperature': 0, 'max_tokens': 1600, 'messages': [{'role': 'user', 'content': content}]}
    body.update(record['request_options'])
    assert hashlib.sha256(json.dumps(body).encode('utf-8')).hexdigest() == record['request_sha256']
    return {'path': path.relative_to(ROOT).as_posix(), 'model': response['model'],
            'image_count': len(record['images']), 'hashes_verified': True,
            'usage': response.get('usage'), 'finish_reason': response['choices'][0].get('finish_reason')}


def main():
    result = {'cases': {}, 'all_cases_finished': True, 'protected_source_unchanged': {}, 'subagents': 0}
    baseline = read(ROOT / 'First_Phase/scene_demo/results/2026-10-14-qwen38-initial-recognition/configuration_verification.json')
    frozen = read(OUT / 'source_manifest.json')
    for name in baseline['protected_control_hashes_match_previous_experiment']:
        relative = 'First_Phase/scene_demo/' + name
        result['protected_source_unchanged'][name] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == frozen[relative]
    assert all(result['protected_source_unchanged'].values())
    for name in CASES:
        directory = OUT / name
        case = {'finished': (directory / 'result.json').exists()}
        result['all_cases_finished'] &= case['finished']
        traces = [row for p in directory.rglob('native_steps.jsonl') for row in rows(p)]
        case['source_counts'] = {s: sum(r['source'] == s for r in traces) for s in ('settling', 'prepare', 'vla', 'finish')}
        case['actual_qwen_responses'] = [check_request(p) for p in sorted(directory.rglob('call_*_request.json'))]
        case['direct_qwen_tokens'] = sum((r['usage'] or {}).get('total_tokens', 0) for r in case['actual_qwen_responses'])
        events = [r for p in directory.rglob('process_events.jsonl') for r in rows(p)]
        case['release_events'] = len(events)
        for event in events:
            assert len(event['images']) == 3
            for image in event['images']:
                assert hashlib.sha256(local_path(image['path']).read_bytes()).hexdigest() == image['sha256']
        if case['finished']:
            record = read(directory / 'result.json')
            plan = record.get('plan') or {}
            case.update(status=record['status'], decision=record.get('decision'), error=record.get('error'),
                        pipeline_terminal=record.get('pipeline_terminal'), requested_capabilities=plan.get('capability_ids'),
                        rationale=plan.get('rationale'), physical_user_goal_success=record.get('physical_user_goal_success'),
                        independent_user_goals=record.get('independent_user_goals'),
                        hermes_output=(record.get('agent_result') or {}).get('hermes_output'))
            case['jobs'] = [{'capability': j['capability'], 'outcome': j['outcome'], 'source_counts': j['source_counts'],
                             'vla_calls': j['vla_calls'], 'handoff_ok': (j.get('handoff_evidence') or {}).get('ok'),
                             'reference_kind': (j.get('handoff_reference') or {}).get('reference_kind'),
                             'reference_capability': (j.get('handoff_reference') or {}).get('capability_id'),
                             'independent_streak': (j.get('finish_status') or {}).get('independent_streak'),
                             'helper_counts': (j.get('finish_status') or {}).get('helper_counts')}
                            for j in record.get('jobs', [])]
        result['cases'][name] = case
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'all_cases_finished': result['all_cases_finished'],
                      'cases': {k: {f: v.get(f) for f in ('finished', 'decision', 'source_counts', 'release_events', 'physical_user_goal_success')}
                                for k, v in result['cases'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
