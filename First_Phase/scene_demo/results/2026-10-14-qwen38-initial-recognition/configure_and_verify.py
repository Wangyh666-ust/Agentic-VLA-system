"""Switch only the X-VLA Hermes profile; verify initial vs terminal model scope."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil

import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PROFILE = Path('/home/yhwang/fyp/xvla_native/hermes_home/config.yaml')
MODEL = 'qwen3.8-max'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    first = json.loads((OUT / 'basket_two/result.json').read_text())
    assert first['http_status'] == 200 and first['returned_model'] == MODEL
    before_bytes = PROFILE.read_bytes()
    before = yaml.safe_load(before_bytes)
    assert before['model']['default'] == 'qwen3-vl-plus'
    after_text, changes = re.subn(r'(?m)^(  default: )qwen3-vl-plus\s*$', r'\g<1>qwen3.8-max', before_bytes.decode('utf-8'))
    assert changes == 1
    expected = copy.deepcopy(before)
    expected['model']['default'] = MODEL
    assert yaml.safe_load(after_text) == expected
    backup = PROFILE.with_name('config.before-qwen38-20261014.yaml')
    assert not backup.exists(), 'do not overwrite configuration backup'
    shutil.copy2(PROFILE, backup)
    PROFILE.write_text(after_text, encoding='utf-8')
    assert yaml.safe_load(PROFILE.read_text()) == expected

    from scene_demo import setup_profile
    from scene_demo.xvla_grounding import InventoryClient
    from scene_demo.finish_v1.vision import QwenClient
    from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
    assert yaml.safe_load(setup_profile.CONFIG_YAML)['model']['default'] == MODEL
    assert InventoryClient(env_path='/none').model == MODEL
    assert InventoryClient(env_path='/none', model='qwen3-vl-plus').model == 'qwen3-vl-plus'
    assert QwenClient(env_path='/none').model == 'qwen3-vl-plus'
    assert XvlaSubtaskReviewClient(env_path='/none').model == 'qwen3-vl-plus'
    old_override = os.environ.get('SCENE_PLANNER_MODEL')
    os.environ['SCENE_PLANNER_MODEL'] = 'qwen3-vl-plus'
    try:
        importlib.reload(setup_profile)
        assert yaml.safe_load(setup_profile.CONFIG_YAML)['model']['default'] == 'qwen3-vl-plus'
        assert InventoryClient(env_path='/none').model == 'qwen3-vl-plus'
    finally:
        if old_override is None:
            del os.environ['SCENE_PLANNER_MODEL']
        else:
            os.environ['SCENE_PLANNER_MODEL'] = old_override
        importlib.reload(setup_profile)
    original = json.loads((ROOT / 'First_Phase/scene_demo/results/2026-10-13-xvla-grounded-online/basket_renderfix/source_manifest.json').read_text())
    names = ['joint_home.py', 'safe_exit.py', 'safe_exit_contacts.py', 'xvla_finish.py', 'xvla_backend.py',
             'finish_v1/runtime.py', 'finish_v1/control.py', 'finish_v1/vision.py', 'xvla_subtask_review.py']
    protected = {n: sha(ROOT / 'First_Phase/scene_demo' / n) == original['First_Phase/scene_demo/' + n] for n in names}
    assert all(protected.values())
    result = {'hermes_profile': str(PROFILE), 'backup': str(backup),
              'before_model': before['model'], 'after_model': expected['model'],
              'before_sha256': hashlib.sha256(before_bytes).hexdigest(), 'after_sha256': sha(PROFILE),
              'only_profile_semantic_change': 'model.default',
              'initial_recognition_default': InventoryClient(env_path='/none').model,
              'terminal_checker_default': XvlaSubtaskReviewClient(env_path='/none').model,
              'explicit_old_model_and_shared_env_override_verified': True,
              'protected_control_hashes_match_previous_experiment': protected,
              'hermes_inference_calls': 0, 'physical_actions': 0}
    (OUT / 'configuration_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
