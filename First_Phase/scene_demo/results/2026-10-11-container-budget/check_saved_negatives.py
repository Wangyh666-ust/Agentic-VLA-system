"""Three changed-prompt checks of saved synchronized samples; zero robot actions."""
import hashlib
import json
from pathlib import Path
from scene_demo.finish_v1 import simple_checker

HERE = Path(__file__).resolve().parent
rows = []
for sample in json.loads((HERE / 'offline_inputs.json').read_text()):
    source = Path(sample['request_path'])
    old = json.loads(source.read_text())
    images = [image['path'] for image in old['images']]
    assert all(hashlib.sha256(Path(image['path']).read_bytes()).hexdigest() == image['sha256']
               for image in old['images'])
    client = simple_checker.SimpleQwenClient(max_calls=1)
    client.audit_dir = HERE / 'saved_negatives' / sample['id']
    result = client.analyze(old['context'], images)
    answer = result['visual_goal_verdict']
    rows.append({'id': sample['id'], 'source_request': str(source),
                 'source_request_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                 'answer': answer, 'pass': answer['goal_achieved'] is sample['expected_goal'],
                 'usage': result.get('usage'), 'robot_actions': 0})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
(HERE / 'saved_negative_summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
if not all(row['pass'] for row in rows):
    raise SystemExit('Saved negatives failed: do not launch online cases')
