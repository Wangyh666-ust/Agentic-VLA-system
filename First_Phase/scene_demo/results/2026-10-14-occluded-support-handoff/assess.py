"""Check saved replay outputs at the unchanged release-decision boundary."""
import hashlib
import json
from pathlib import Path
import sys

out = Path(__file__).resolve().parent
root = out.parents[3]
sys.path.insert(0, str(root / 'First_Phase'))
from scene_demo.finish_v1 import simple_checker

name, expected = sys.argv[1:3]
directory = out / name
manifest = json.loads((directory / 'manifest.json').read_text())
result = json.loads((directory / 'result.json').read_text())
source = Path(manifest['request']).parent.parent / 'subtask_evidence/observations.jsonl'
target = next(json.loads(line) for line in source.read_text().splitlines()
              if json.loads(line)['captured_at'] is not None
              and abs(json.loads(line)['captured_at'] - manifest['target_time']) < 1e-6)
ctx = target['context']
measurements = ctx['actuator_measurements']
fused = simple_checker.reconcile(ctx, result['verdict'],
    {'finger_qpos': measurements['finger_qpos']},
    {'finger_home': measurements['standard_open_finger_qpos']})
decision = simple_checker.decide(ctx, fused)
review = result['handoff_review'] or result['pages'][0]
rows = review['applied_review']['normalized']['current']
payload = {'case': name, 'expected': expected, 'observed_at': manifest['target_time'],
    'goal_achieved': result['verdict']['operation_achieved'],
    'handoff_review_triggered': result['handoff_review'] is not None,
    'decision': decision, 'released_encoder': fused['released'],
    'current_rows': rows, 'actual_model_calls': result['actual_Qwen_calls'],
    'new_robot_actions': 0, 'historical_proofs': result['process_evidence']}
if expected == 'allow':
    payload['passed'] = decision['allow_release'] is True and result['handoff_review'] is not None
elif expected == 'complete':
    payload['passed'] = decision['state'] == 'complete' and fused['released'] is True
else:
    assert expected == 'block'
    payload['passed'] = decision['allow_release'] is False
assert all(json.loads(p.read_text())['model'] == 'qwen3.8-max'
           for p in (directory / 'model_requests').glob('*_response.json'))
protected = json.loads((out / 'protected_controls.json').read_text())
assert all(hashlib.sha256((root / 'First_Phase/scene_demo' / n).read_bytes()).hexdigest() == h
           for n, h in protected.items())
(directory / 'assessment.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in payload.items() if k not in ('current_rows', 'historical_proofs')}, ensure_ascii=False))
assert payload['passed'], 'Expected release decision not reached; stop dependent verification'
