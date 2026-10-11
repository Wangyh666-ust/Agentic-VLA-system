"""Inspect historical HTTP inputs. No robot construction or real HTTP calls."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'First_Phase'))
from scene_demo.finish_v1.simple_checker import SimpleQwenClient, build_prompt
from scene_demo.finish_v1 import vision

SCENE = ROOT / 'First_Phase/scene_demo'
OUT = Path(__file__).resolve().parent
CASES = [
    ('wine_trigger', '2026-10-11-feedback-retreat', 'bowl_wine', 'wine_to_rack', 1),
    ('wine_final', '2026-10-11-feedback-retreat', 'bowl_wine', 'wine_to_rack', 2),
    ('stove_trigger', '2026-10-11-feedback-retreat', 'bowl_stove', 'stove_on', 1),
    ('bowl_final_pass', '2026-10-11-stove-exit-repair', 'bowl_stove', 'bowl_to_plate', 2),
    ('bowl_final_veto', '2026-10-11-stove-candidate-start', 'bowl_stove', 'bowl_to_plate', 2),
]


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


rows = []
sources = []
for name, folder, case, capability, index in CASES:
    campaign = SCENE / 'results' / folder
    result_path = campaign / case / 'result.json'
    result = read(result_path)
    sub = next(s for s in result['subtasks'] if s['capability'] == capability)
    run = Path(sub['job']['run_dir'])
    request_path = run / f'finish/model_requests/call_{index:02d}_request.json'
    response_path = request_path.with_name(request_path.name.replace('_request', '_response'))
    record = read(request_path)
    response_bytes = response_path.read_bytes()
    response = json.loads(response_bytes)
    with zipfile.ZipFile(campaign / 'source_snapshot.zip') as archive:
        frozen_match = {
            f: archive.read(f) == (ROOT / f).read_bytes()
            for f in ['First_Phase/scene_demo/finish_v1/vision.py',
                      'First_Phase/scene_demo/finish_v1/simple_checker.py',
                      'First_Phase/scene_demo/finish_v1/runtime.py']
        }
    if not all(frozen_match.values()):
        raise RuntimeError('Frozen serializer mismatch; do not infer historical request bytes')

    client = SimpleQwenClient(model=response['model'], max_calls=1)
    client._resolve_key = lambda: 'offline-placeholder-never-sent'
    client.record_response = lambda *args, **kwargs: None
    captured = []

    class MockResponse:
        def getcode(self):
            return record['http_status']

        def read(self):
            return response_bytes

    def intercept(req, timeout):
        captured.append(req.data)
        return MockResponse()

    paths = [image['path'] for image in record['images']]
    with patch.object(vision.urllib.request, 'urlopen', side_effect=intercept):
        client.analyze(record['context'], paths)
    assert len(captured) == 1
    body = json.loads(captured[0])
    checks_path = run / 'finish/checks.jsonl'
    check = next(json.loads(line) for line in checks_path.read_text().splitlines()
                 if json.loads(line)['call_index'] == index)
    measurement = record['context']['actuator_measurements']
    fingers = measurement['finger_qpos']
    standard = measurement['standard_open_finger_qpos']
    independent = json.loads((run / 'independent_finish.jsonl').read_text().splitlines()[-1])
    images = [{**image, 'file_hash_matches': sha(Path(image['path']).read_bytes()) == image['sha256']}
              for image in record['images']]
    prompt_path = OUT / f'{name}_actual_prompt.txt'
    prompt_path.write_text(record['prompt'] + '\n', encoding='utf-8')
    content = body['messages'][0]['content']
    row = {
        'name': name, 'request_path': request_path.relative_to(ROOT).as_posix(),
        'frozen_serializer_matches': frozen_match,
        'full_http_body_hash_matches': sha(captured[0]) == record['request_sha256'],
        'request_sha256': record['request_sha256'],
        'prompt_rebuild_identical': record['prompt'] == build_prompt(record['context']),
        'image_labels': [x['text'] for x in content[1:] if x['type'] == 'text'],
        'actual_images': images,
        'image_and_encoder_timestamps_match': all(f['timestamp'] == measurement['timestamp']
                                                 for f in record['context']['sensor_frames']),
        'encoder_matches_saved_robot': fingers == check['robot']['finger_qpos'],
        'speed_matches_saved_robot': measurement['max_abs_arm_qvel'] == max(abs(q) for q in check['robot']['arm_qvel']),
        'finger_qpos_m': fingers, 'standard_open_finger_qpos_m': standard,
        'max_standard_open_error_mm': 1000 * max(abs(a - b) for a, b in zip(fingers, standard)),
        'calibrated_open': max(abs(a - b) for a, b in zip(fingers, standard)) <= .002,
        'model': body['model'], 'temperature': body['temperature'],
        'finish_reason': response['choices'][0]['finish_reason'],
        'phase_or_stage_labels_in_sent_text': any(token in record['prompt'] for token in ['"stage"', '"phase"', 'post_execution']),
        'message_count': len(body['messages']), 'prior_responses_sent': False,
        'answer': json.loads(response['choices'][0]['message']['content']),
        'finish_counts': sub['finish']['counts'],
        'job_end_reason': sub['job']['ended_reason'],
        'last_independent': {k: independent[k] for k in ['timestamp', 'predicates', 'contacts', 'release_status', 'ready']},
        'actual_prompt_path': prompt_path.relative_to(ROOT).as_posix(),
    }
    rows.append(row)
    sources.extend([result_path, request_path, response_path, checks_path, run/'independent_finish.jsonl',
                    run/'finish_actions.jsonl', run/'action_sources.jsonl', campaign/'source_snapshot.zip'])
    sources.extend(Path(p) for p in paths)

pair = [next(row for row in rows if row['name'] == name) for name in ['bowl_final_pass', 'bowl_final_veto']]
output = {
    'physical_actions': 0, 'real_model_calls': 0, 'mock_http_serializations': len(rows),
    'rows': rows,
    'bowl_identical_full_http_request': pair[0]['request_sha256'] == pair[1]['request_sha256'],
    'bowl_answers_disagree': pair[0]['answer']['holding_or_contact'] != pair[1]['answer']['holding_or_contact'],
    'limits': ['Matching local request bytes establishes input provenance, not provider-internal image processing or an exact cause of model error.',
               'Open encoders alone do not prove absence of contact; independent simulator truth is diagnosis evidence, not model input.'],
    'source_evidence': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p.read_bytes())}
                        for p in sorted(set(sources))],
}
(OUT/'audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'physical_actions':0, 'real_model_calls':0,
                 'checked_requests':len(rows),
                 'all_http_bytes_match':all(row['full_http_body_hash_matches'] for row in rows),
                 'all_images_encoder_sync_match':all(row['image_and_encoder_timestamps_match'] and row['encoder_matches_saved_robot'] and all(im['file_hash_matches'] for im in row['actual_images']) for row in rows),
                 'bowl_identical_input_disagreement': output['bowl_identical_full_http_request'] and output['bowl_answers_disagree']}, ensure_ascii=False))
