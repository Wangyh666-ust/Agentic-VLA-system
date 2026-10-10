"""One saved-image Qwen check; no simulator, rendering or robot controls."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "First_Phase/scene_demo/experiment_registry.json").is_file())
sys.path.insert(0, str(ROOT / 'First_Phase'))
from scene_demo.finish_v1 import vision

FIXTURES = ROOT / 'First_Phase/scene_demo/results/2026-10-10-vision-grounding/evidence/fixtures.json'
CAMPAIGN = FIXTURES.parent.parent
CONTEXT_KEYS = {'user_request', 'current_subtask', 'criteria', 'sensor_frames', 'actuator_measurements'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def equal_value(actual, expected):
    return type(actual) is type(expected) and actual == expected

def score(context, verdict, expected):
    decision = vision.decide(context, verdict)
    fields = {}
    for key, wanted in expected.get('core', {}).items():
        allowed = wanted if isinstance(wanted, list) else [wanted]
        actual = verdict.get(key)
        fields[key] = {'actual': actual, 'expected': wanted,
                       'ok': any(equal_value(actual, x) for x in allowed)}
    states = expected['allowed_decisions']
    fields['decision.state'] = {'actual': decision['state'], 'expected': states,
                                'ok': decision['state'] in states and decision['state'] not in expected.get('forbidden_decisions', [])}
    for key, wanted in expected.get('decision_fields', {}).items():
        fields['decision.' + key] = {'actual': decision.get(key), 'expected': wanted,
                                     'ok': equal_value(decision.get(key), wanted)}
    grounding = {}
    for view, roi in expected.get('grounding_roi', {}).items():
        grounding[view] = {}
        for key in expected.get('grounding_required_boxes', []):
            box = verdict.get('boxes', {}).get(view, {}).get(key)
            center = None
            ok = False
            if isinstance(box, list) and len(box) == 4:
                center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                ok = all(math.isfinite(x) for x in center) and roi[0] <= center[0] <= roi[2] and roi[1] <= center[1] <= roi[3]
            grounding[view][key] = {'box': box, 'center': center, 'independent_roi': roi, 'ok': ok}
    conservative = decision['state'] == 'unknown'
    acceptable = all(v['ok'] for v in fields.values()) and all(v['ok'] for boxes in grounding.values() for v in boxes.values())
    return {'decision': decision, 'field_checks': fields, 'grounding_checks': grounding,
            'criteria_satisfied': acceptable, 'pass': acceptable and not conservative,
            'conservative_unknown': conservative,
            'safe_rejection': conservative and decision['state'] in states}

def main():
    parser = argparse.ArgumentParser(description='Check one frozen image fixture with production QwenClient; zero physical actions. API is called only after validated arguments and image hashes.')
    parser.add_argument('--case', required=True, help='Frozen fixture id')
    parser.add_argument('--output', required=True, type=Path, help='New result JSON inside this vision-grounding campaign')
    parser.add_argument('--max-calls', type=int, choices=[1], default=1, help='Exactly one production Qwen call; no retry')
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(CAMPAIGN.resolve())
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing result')
    data = json.loads(FIXTURES.read_text(encoding='utf-8'))
    case = next(c for c in data['cases'] if c['id'] == args.case)
    context = copy.deepcopy(case['context'])
    if not set(context) <= CONTEXT_KEYS:
        raise ValueError('Fixture context contains non-whitelisted top-level fields')
    def no_hidden(value):
        if isinstance(value, dict):
            if set(value) & {'gold', 'native', 'predicates', 'sim_truth', 'old_verdict', 'expected', 'independent_snapshot', 'grounding_roi'}:
                raise ValueError('Offline scoring evidence must not enter model context')
            for v in value.values():
                no_hidden(v)
        elif isinstance(value, list):
            for v in value:
                no_hidden(v)
    no_hidden(context)
    images = []
    proof = []
    for relative, frozen in zip(case['image_paths'], case['image_SHA']):
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        actual = sha(path)
        if actual != frozen['sha256'] or path.stat().st_size != frozen['bytes']:
            raise ValueError('Original image SHA/bytes mismatch: ' + relative)
        images.append(str(path))
        proof.append({'path': relative, 'sha256': actual, 'bytes': path.stat().st_size})
    if len(images) != len(case['image_paths']) or len(images) != len(case['sensor_frames']):
        raise ValueError('Original image and frame list lengths differ')
    if case.get('actuator_measurements') is not None and case.get('reference_finger_home') is not None:
        ref_source = case['reference_source']
        if sha(ROOT / ref_source['path']) != ref_source['sha256']:
            raise ValueError('Original encoder-reference source SHA mismatch')
        context['actuator_measurements']['standard_open_finger_qpos'] = copy.deepcopy(case['reference_finger_home'])
    prompt = vision.build_prompt(context)
    result = {'case_id': args.case, 'source_sha': sha(FIXTURES),
              'labels_sha256': data['labels_sha256'], 'vision_source_sha256': sha(Path(vision.__file__)),
              'context': context, 'build_prompt': prompt, 'image_SHA': proof,
              'raw_verdict': None, 'verdict': None, 'decision': None, 'field_checks': {},
              'grounding_checks': {}, 'pass': False, 'conservative_unknown': False,
              'usage': {}, 'request_sha256': None, 'response_sha256': None,
              'physical_actions': 0, 'model_calls': 0, 'error': None}
    client = vision.QwenClient(max_calls=args.max_calls)
    try:
        meta = client.analyze(context, images)
        result['model_calls'] = client._calls
        raw = meta['verdict']
        verdict = raw
        measures = case.get('actuator_measurements')
        if measures is not None and case.get('reference_finger_home') is not None:
            ref_source = case['reference_source']
            if sha(ROOT / ref_source['path']) != ref_source['sha256']:
                raise ValueError('Original encoder-reference source SHA mismatch')
            verdict = vision.reconcile_verdict(context, raw,
                {'finger_qpos': measures['finger_qpos']},
                {'finger_home': case['reference_finger_home']})
        raw_scores = score(context, raw, case['expected'])
        fused_scores = score(context, verdict, case['expected'])
        result.update(fused_scores)
        result.update({'raw_verdict': raw, 'verdict': verdict,
                       'raw_scores': raw_scores, 'fused_scores': fused_scores,
                       'usage': meta.get('usage', {}), 'model': meta.get('model'),
                       'http_status': meta.get('http_status'), 'call_index': meta.get('call_index'),
                       'request_sha256': meta.get('request_sha256'),
                       'response_sha256': meta.get('response_sha256')})
    except Exception as exc:
        result['model_calls'] = client._calls
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'case_id': args.case, 'output': str(output), 'pass': result['pass'],
                      'conservative_unknown': result['conservative_unknown'],
                      'model_calls': result['model_calls'], 'physical_actions': 0,
                      'error': result['error']}, ensure_ascii=False))
    return 0 if result['error'] is None else 2

if __name__ == '__main__':
    raise SystemExit(main())
