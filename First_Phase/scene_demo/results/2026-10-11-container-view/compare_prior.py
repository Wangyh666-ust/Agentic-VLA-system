"""Compare recorded prefixes only; no models, actions, or pairing assumption."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '2026-10-11-container-pilot'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


comparisons = []
for case in ('sauce_basket', 'mug_microwave'):
    old = read(PRIOR / case / 'result.json')
    new = read(HERE / case / 'result.json')
    for first, second in zip(old['subtasks'], new['subtasks']):
        assert first['capability'] == second['capability']
        a, b = Path(first['job']['run_dir']), Path(second['job']['run_dir'])
        events_a, events_b = rows(a / 'events.jsonl'), rows(b / 'events.jsonl')
        n = min(len(events_a), len(events_b))
        checks_a, checks_b = rows(a / 'finish/checks.jsonl'), rows(b / 'finish/checks.jsonl')
        paired_checks = []
        # Runtime checks only at150/300; any later call follows finishing.
        for i in range(min(len(events_a)//150, len(events_b)//150)):
            qa = read(a / f'finish/model_requests/call_{i+1:02d}_request.json')
            qb = read(b / f'finish/model_requests/call_{i+1:02d}_request.json')
            paired_checks.append({'vla_step': (i+1)*150,
                'same_wrist_pixels': qa['images'][0]['sha256'] == qb['images'][0]['sha256'],
                'same_robot_feedback': checks_a[i]['robot'] == checks_b[i]['robot'],
                'old_verdict': checks_a[i]['raw_verdict'], 'new_verdict': checks_b[i]['raw_verdict']})
        comparisons.append({'case': case, 'capability': first['capability'], 'prefix_length': n,
            'same_initial_state_sha': first['state_before'] == second['state_before'],
            'same_sent_action_prefix': all(x['action'] == y['action'] for x, y in zip(events_a[:n], events_b[:n])),
            'same_native_predicate_prefix': all(x['declared_predicates'] == y['declared_predicates'] for x,y in zip(events_a[:n], events_b[:n])),
            'checks': paired_checks,
            'limitation': 'Observed equality only; not a preregistered isolated-camera control or proof of full RNG/policy state pairing'})
(HERE / 'prior_action_comparison.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + '\n')
for c in comparisons:
    print(json.dumps({k:c[k] for k in ('case','prefix_length','same_initial_state_sha','same_sent_action_prefix','same_native_predicate_prefix')}))
