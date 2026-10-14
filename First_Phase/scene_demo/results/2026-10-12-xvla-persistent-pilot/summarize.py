"""Read preserved campaign evidence; no model or simulator operations."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def states(rows, key):
    values = [(r.get('step'), r.get(key)) for r in rows]
    return {'first_true_step': next((step for step, value in values if value is True), None),
            'true_count': sum(v is True for _, v in values),
            'last': values[-1][1] if values else None, 'sample_count': len(values)}


def trial(version, name, category):
    directory = BASE / version / name
    result = load(directory / 'result.json')
    native = [r for p in directory.rglob('native_steps.jsonl') for r in lines(p)]
    jobs = []
    for pilot in directory.rglob('pilot_result.json'):
        p = load(pilot)
        events = lines(pilot.parent / 'events.jsonl') if (pilot.parent / 'events.jsonl').exists() else []
        goal_keys = sorted({k for r in events for k in r.get('declared_predicates', {})})
        goals = {k: states([{'step': r.get('step'), 'value': r.get('declared_predicates', {}).get(k)} for r in events], 'value') for k in goal_keys}
        checks = []
        check_file = pilot.parent / 'finish/checks.jsonl'
        for c in lines(check_file) if check_file.exists() else []:
            measurements = c.get('ctx', {}).get('actuator_measurements', {})
            timestamp = measurements.get('timestamp')
            matches = [r for r in native if timestamp is not None and abs(r['sim_time'] - timestamp) <= 1e-9]
            row = matches[-1] if matches else None
            step = row['counts']['vla'] if row else None
            event = next((e for e in events if e.get('step') == step), None)
            visual = c.get('initial_check', {}).get('metadata', {}).get('visual_goal_verdict')
            checks.append({'call_index': c.get('call_index'), 'timestamp': timestamp,
                           'matched_native_rows': len(matches), 'cumulative_vla': step,
                           'declared_predicates': event.get('declared_predicates') if event else None,
                           'completion_ready': event.get('completion_ready') if event else None,
                           'visual_goal_verdict': visual, 'verdict': c.get('verdict'),
                           'encoder': measurements, 'sensor_frames': c.get('sensor_frames'),
                           'usage': c.get('usage'), 'image_paths': c.get('image_paths')})
        independent_path = pilot.parent / 'independent_finish.jsonl'
        independent = lines(independent_path) if independent_path.exists() else []
        independent_keys = sorted({k for r in independent for k in r.get('predicates', {})})
        independent_goals = {k: {'true_count': sum(r.get('predicates', {}).get(k) is True for r in independent),
                                'sample_count': len(independent),
                                'last': independent[-1].get('predicates', {}).get(k) if independent else None} for k in independent_keys}
        from scene_demo.xvla_service import handoff_evidence
        try:
            handoff = handoff_evidence(p)
        except Exception as exc:
            handoff = {'error': repr(exc)}
        jobs.append({'pilot_path': str(pilot), 'capability': p.get('capability'), 'job': p.get('job'),
                     'outcome': p.get('outcome'), 'source_counts': p.get('source_counts'),
                     'vla_calls': p.get('vla_calls'), 'finish': p.get('finish'),
                     'finish_status': p.get('finish_status'), 'actual_final_metrics': p.get('actual_final_metrics'),
                     'after': p.get('after'), 'last_cache_reset': (p.get('cache_reset') or [None])[-1],
                     'handoff_reference': p.get('handoff_reference'), 'handoff_audit': handoff,
                     'declared_goals': goals, 'completion_ready': states(events, 'completion_ready'),
                     'checks': checks, 'independent_goals': independent_goals})
    return {'version': version, 'case': name, 'classification': category,
            'result_path': str(directory / 'result.json'), 'complete': result.get('complete'),
            'plan': result.get('plan'), 'hermes_invocations': result.get('agent_result', {}).get('hermes_invocations'),
            'hermes_usage': result.get('agent_result', {}).get('usage'),
            'session_counts': native[-1]['counts'] if native else None,
            'vla_calls': sum(j['vla_calls'] for j in jobs if j.get('vla_calls') is not None),
            'jobs': jobs}


def main():
    main_trials = [trial(v, n, 'main_trial') for v, n in [('v4', 'wine_bowl_stove'), ('v5', 'basket_two'), ('v5', 'mugs_two'), ('v5', 'microwave_mug')]]
    startup = [trial(v, 'wine_bowl_stove', 'integration_startup_failure') for v in ['v2', 'v3']]
    offline = [{'directory': str(BASE / n), 'result': load(BASE / n / 'result.json'),
                'physical_actions': 0, 'xvla_calls': 0, 'qwen_calls': 3} for n in ['checker_agentview', 'checker_rack_description']]
    all_trials = main_trials + startup
    counts = {k: sum(t['session_counts'][k] for t in all_trials if t['session_counts'] is not None) for k in ['settling', 'vla', 'finish', 'prepare']}
    online_checks = [c for t in all_trials for j in t['jobs'] for c in j['checks']]
    out = {'main_case_count': 4, 'main_complete_count': sum(t['complete'] is True for t in main_trials),
           'main_trials': main_trials, 'startup_failures': startup, 'offline_checker_comparisons': offline,
           'totals': {'new_actions': counts, 'physical_actions_total': sum(counts.values()),
                      'xvla_calls': sum(t['vla_calls'] for t in all_trials),
                      'hermes_invocations': sum(t['hermes_invocations'] or 0 for t in all_trials),
                      'online_qwen_calls': len(online_checks),
                      'online_qwen_tokens': sum((c.get('usage') or {}).get('total_tokens', 0) for c in online_checks),
                      'offline_qwen_calls': 6, 'hermes_token_usage': None},
           'agents': {'implementation_execution': {'model': 'gpt-6.1-sol', 'reasoning_effort': 'low', 'evidence': 'actual executor spawn/dispatch configuration'},
                      'main_planning': {'model': None, 'reasoning_effort': None, 'reason': 'host did not expose actual configuration'}}}
    out['totals']['offline_qwen_tokens'] = sum(c.get('usage', {}).get('total_tokens', 0) for t in offline for c in t['result']['cases'])
    out['totals']['all_qwen_tokens'] = out['totals']['online_qwen_tokens'] + out['totals']['offline_qwen_tokens']
    (BASE / 'summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'main_case_count': 4, 'main_complete_count': out['main_complete_count'], 'totals': out['totals'],
                      'goals': [{'case': t['case'], 'goals': [j['declared_goals'] for j in t['jobs']], 'ready': [j['completion_ready'] for j in t['jobs']]} for t in main_trials]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
