"""Small frozen timing lookup; no experiment-history scan or goal oracle."""
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def profiles():
    return json.loads(Path(__file__).with_name('xvla_check_profiles.json').read_text(encoding='utf-8'))


def check_profile(capability):
    native = capability.get('native_task')
    for row in profiles()['profiles']:
        exact = capability.get('instruction') == row['instruction']
        slot = not native or (native.get('suite') == row['suite'] and native.get('task_id') == row['task_id'])
        if exact and slot:
            return {**row, 'source_sha256': profiles()['source_sha256']}
    return {'first_check_step': 150, 'earliest_success_step': None, 'source': 'unmatched_conservative_default'}
