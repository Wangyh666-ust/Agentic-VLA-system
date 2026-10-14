"""Replay only saved image-recognition requests. No robot/simulator imports."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'First_Phase/scene_demo/results/2026-10-13-xvla-grounded-online'
MODEL = 'qwen3.8-max'
ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
SOURCES = {
    'basket_two': OLD / 'basket_renderfix/basket_two/runs/20261013T155633Z_basket_two_594bcb3b/planning_grounding/model_requests/call_01_request.json',
    'microwave_mug': OLD / 'microwave_mug/runs/20261013T153850Z_microwave_mug_83806dba/planning_grounding/model_requests/call_01_request.json',
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def local_path(value):
    suffix = value.split('/FYP/', 1)[1]
    return ROOT / suffix


def build(source):
    record = json.loads(source.read_text(encoding='utf-8'))
    content = [{'type': 'text', 'text': record['prompt']}]
    for frame, image in zip(record['context']['sensor_frames'], record['images']):
        data = local_path(image['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == image['sha256']
        content.extend([
            {'type': 'text', 'text': f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')}},
        ])
    assert len(record['images']) == len(record['context']['sensor_frames']) == 3
    body = {'model': 'qwen3-vl-plus', 'temperature': 0, 'max_tokens': 1600,
            'messages': [{'role': 'user', 'content': content}]}
    body.update(record['request_options'])
    assert hashlib.sha256(json.dumps(body).encode('utf-8')).hexdigest() == record['request_sha256']
    body['model'] = MODEL
    return record, json.dumps(body).encode('utf-8')


def key():
    value = os.environ.get('DASHSCOPE_API_KEY')
    if value:
        return value.strip()
    path = Path('/home/yhwang/fyp/xvla_native/hermes_home/.env')
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip().removeprefix('export ')
        if '=' in line:
            name, value = line.split('=', 1)
            if name.strip() == 'DASHSCOPE_API_KEY':
                return value.strip().strip('\"\'')
    raise RuntimeError('project API key missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    checked = []
    for case, source in SOURCES.items():
        record, data = build(source)
        checked.append({'case': case, 'source': str(source.relative_to(ROOT)),
                        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'original_request_hash_verified': True, 'images_verified': 3,
                        'only_changed_body_field': 'model', 'model': MODEL,
                        'request_sha256': hashlib.sha256(data).hexdigest()})
    if args.prepare:
        dump(OUT / 'input_validation.json', checked)
        print(json.dumps(checked, ensure_ascii=False), flush=True)
        return
    gate = json.loads((OUT / 'review_check.json').read_text(encoding='utf-8-sig'))
    assert gate.get('ok') is True, 'proposal gate missing or not passed'
    api_key = key()
    for case, source in SOURCES.items():
        directory = OUT / case
        assert not directory.exists(), 'do not overwrite a previous attempt'
        record, data = build(source)
        record.update(model=MODEL, endpoint=ENDPOINT, source_request=str(source.relative_to(ROOT)),
                      source_request_sha256=record['request_sha256'],
                      request_sha256=hashlib.sha256(data).hexdigest())
        dump(directory / 'request.json', record)
        start = time.time()
        request = urllib.request.Request(ENDPOINT, data=data, method='POST', headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key})
        print(json.dumps({'case': case, 'status': 'request_started', 'model': MODEL}), flush=True)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            (directory / 'response.json').write_bytes(raw)
            dump(directory / 'result.json', {'http_status': exc.code, 'status': 'rejected',
                                            'elapsed_seconds': time.time() - start})
            print(json.dumps({'case': case, 'http_status': exc.code, 'status': 'rejected'}), flush=True)
            return
        except Exception as exc:
            dump(directory / 'result.json', {'status': 'transport_error', 'error_type': type(exc).__name__,
                                            'elapsed_seconds': time.time() - start})
            print(json.dumps({'case': case, 'status': 'transport_error', 'error_type': type(exc).__name__}), flush=True)
            return
        (directory / 'response.json').write_bytes(raw)
        payload = json.loads(raw)
        message = payload['choices'][0]['message']
        text = message.get('content') or ''
        result = {'http_status': status, 'status': 'response_received', 'requested_model': MODEL,
                  'returned_model': payload.get('model'), 'usage': payload.get('usage'),
                  'finish_reason': payload['choices'][0].get('finish_reason'),
                  'elapsed_seconds': time.time() - start,
                  'response_sha256': hashlib.sha256(raw).hexdigest()}
        try:
            clean = text.strip()
            if clean.startswith('```') and clean.endswith('```'):
                clean = '\n'.join(clean.splitlines()[1:-1])
            inventory = json.loads(clean)
            dump(directory / 'inventory.json', inventory)
            result['inventory_parsed'] = True
        except (ValueError, TypeError):
            result['inventory_parsed'] = False
        dump(directory / 'result.json', result)
        print(json.dumps({'case': case, **result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
