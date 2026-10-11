"""Offline fusion regression: no HTTP requests, replay, or robot actions."""
import copy
import hashlib
import json
from pathlib import Path

from scene_demo.finish_v1 import simple_checker, vision

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-final-input-audit/audit.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    sources = {item['path']: item['sha256'] for item in audit['source_evidence']}
    used = {AUDIT.relative_to(ROOT).as_posix(): sha(AUDIT)}
    rows = []
    for original in audit['rows']:
        request_path = ROOT / original['request_path']
        response_path = request_path.with_name(request_path.name.replace('_request', '_response'))
        checks_path = request_path.parent.parent / 'checks.jsonl'
        for path in (request_path, response_path, checks_path):
            rel = path.relative_to(ROOT).as_posix()
            assert sha(path) == sources[rel], rel
            used[rel] = sources[rel]
        request = json.loads(request_path.read_text(encoding='utf-8'))
        response = json.loads(response_path.read_text(encoding='utf-8'))
        records = [json.loads(line) for line in checks_path.read_text(encoding='utf-8').splitlines()]
        sampled = next(r for r in records if r['request_sha256'] == request['request_sha256'])
        context = copy.deepcopy(request['context'])
        assert sampled['ctx'] == context
        for item in request['images']:
            path = Path(item['path'])
            assert sha(path) == item['sha256']
            used[path.relative_to(ROOT).as_posix()] = item['sha256']
        client = simple_checker.SimpleQwenClient(max_calls=0)
        raw = client.parse_verdict(vision._extract_content(response))
        assert client.last_two_question_verdict == original['answer']
        reference = {'finger_home': context['actuator_measurements']['standard_open_finger_qpos']}
        fused = simple_checker.reconcile(context, raw, sampled['robot'], reference)
        decision = simple_checker.decide(context, fused)
        encoder = simple_checker.encoder_observation(context)
        assert fused['released'] is encoder['at_standard_open']
        assert fused['operation_achieved'] is original['answer']['goal_achieved']
        assert fused['clear_of_target'] is None
        prompt = simple_checker.build_prompt(context)
        assert 'post_execution' not in prompt and 'holding_or_contact' not in prompt
        (OUT / (original['name'] + '_new_prompt.txt')).write_text(prompt + '\n', encoding='utf-8')
        rows.append({'name': original['name'], 'source_request': original['request_path'],
                     'historical_answer': original['answer'], 'encoder': encoder,
                     'fused_verdict': fused, 'new_decision': decision,
                     'historical_physical_counts': original['finish_counts'],
                     'historical_independent_ready': original['last_independent'].get('ready'),
                     'limitation': 'old visual goal reused; this is not a new Qwen response or physical run'})
    by_name = {row['name']: row for row in rows}
    assert by_name['bowl_final_pass']['new_decision'] == by_name['bowl_final_veto']['new_decision']
    assert by_name['bowl_final_veto']['new_decision']['state'] == 'complete'
    assert by_name['wine_final']['new_decision']['state'] == 'incomplete'
    assert by_name['wine_final']['fused_verdict']['released'] is True
    for name in ('wine_trigger', 'stove_trigger'):
        assert by_name[name]['new_decision']['state'] == 'needs_release'
    result = {'checker': simple_checker.CHECKER_NAME, 'physical_actions': 0,
              'real_model_calls': 0, 'replayed_actions': 0, 'rows': rows,
              'source_evidence': [{'path': path, 'sha256': digest} for path, digest in sorted(used.items())],
              'limits': ['New neutral prompt has not yet been tested with a live Qwen request.',
                         'No new combination or stove retreat/home result is claimed.']}
    (OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'saved_checks': len(rows), 'real_model_calls': 0,
                      'physical_actions': 0, 'decisions': {row['name']: row['new_decision']['state'] for row in rows}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
