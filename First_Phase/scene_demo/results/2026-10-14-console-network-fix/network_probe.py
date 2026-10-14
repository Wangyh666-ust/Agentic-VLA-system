"""Compare the inherited proxy and direct API route without any model call."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

out = Path(__file__).resolve().parent
root = out.parents[3]
env_path = root / 'First_Phase/scene_demo/results/2026-10-14-qwen38-three-requests/runtime/table_tidy_stove_environment.json'
os.environ.update(json.loads(env_path.read_text()))
for path in os.environ['PYTHONPATH'].split(':'):
    if path not in sys.path:
        sys.path.insert(0, path)
from scene_demo.finish_v1.vision import QwenClient

client = QwenClient()
key = client._resolve_key()
assert key
proxies = urllib.request.getproxies()
safe_proxies = {name: {'host': urllib.parse.urlsplit(value).hostname,
                       'port': urllib.parse.urlsplit(value).port}
                for name, value in proxies.items() if name != 'no'}
def probe(mode):
    started = time.monotonic()
    opener = urllib.request.build_opener() if mode == 'inherited_proxy' else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(client.base_url + '/models',
                                     headers={'Authorization': 'Bearer ' + key})
    result = {'mode': mode}
    try:
        with opener.open(request, timeout=12) as response:
            data = json.loads(response.read())
            result.update(status=response.status, model_count=len(data.get('data', [])),
                          qwen38_listed=any(row.get('id') == 'qwen3.8-max' for row in data.get('data', [])))
    except Exception as exc:
        reason = getattr(exc, 'reason', exc)
        result.update(error_type=type(exc).__name__, reason_type=type(reason).__name__,
                      status=getattr(exc, 'code', None), errno=getattr(reason, 'errno', None))
    result['seconds'] = round(time.monotonic() - started, 2)
    return result
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(probe, ['inherited_proxy', 'direct']))
record = {'proxies': safe_proxies, 'endpoint': client.base_url,
          'probes': results, 'vlm_requests': 0, 'robot_actions': 0}
(out / 'network_probe.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
