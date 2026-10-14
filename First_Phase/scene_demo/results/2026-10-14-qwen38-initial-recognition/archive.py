"""Archive the bounded first-recognition comparison and its limitations."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cases = {name: read(OUT / name / 'result.json') for name in ('basket_two', 'microwave_mug')}
    assert all(c['http_status'] == 200 and c['returned_model'] == 'qwen3.8-max'
               and c['inventory_parsed'] and c['finish_reason'] == 'stop' for c in cases.values())
    assert read(OUT / 'pytest_result.json')['returncode'] == 0
    config = read(OUT / 'configuration_verification.json')
    assert all(config['protected_control_hashes_match_previous_experiment'].values())
    files = set()
    diff = []
    baseline = ROOT / 'First_Phase/scene_demo/results/2026-10-13-xvla-grounded-online/basket_renderfix/source_snapshot'
    for name in ('setup_profile.py', 'xvla_grounding.py'):
        relative = Path('First_Phase/scene_demo') / name
        source = ROOT / relative
        destination = OUT / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        diff.extend(difflib.unified_diff((baseline / relative).read_text(encoding='utf-8').splitlines(True),
                                        source.read_text(encoding='utf-8').splitlines(True),
                                        fromfile='previous/' + relative.as_posix(), tofile='current/' + relative.as_posix()))
    (OUT / 'source_changes.diff').write_text(''.join(diff), encoding='utf-8')
    for case in cases:
        request = read(OUT / case / 'request.json')
        assert request['model'] == 'qwen3.8-max'
        for image in request['images']:
            path = ROOT / image['path'].split('/FYP/', 1)[1]
            assert sha(path) == image['sha256']
            files.add(path)
        source = ROOT / request['source_request']
        files.add(source)
        files.add(source.with_name(source.name.replace('_request', '_response')))
    audit = {
        'cases': cases, 'actual_qwen_requests': 2, 'actual_hermes_inferences': 0,
        'actual_physical_actions': 0, 'subagents': 0,
        'total_tokens': sum(c['usage']['total_tokens'] for c in cases.values()),
        'manual_visual_assessment': {
            'basket_two': {'external_target_boxes_correct': True, 'full_three_view_consistency_passed': False,
                           'issues': ['wrist tomato described at right-bottom instead of left-bottom',
                                      'wrist soup described at lower-middle instead of right-bottom',
                                      'wrist orange flat pack described as juice carton',
                                      'simultaneous Milk carton assigned inconsistent upright/lying pose']},
            'microwave_mug': {'target_mug_three_view_identity_passed': True, 'gray_mug_distinguished': True,
                             'online_closing_and_history_tested': False}},
        'configuration_verified': True, 'existing_tests_passed': 15,
        'limitations': ['one first-recognition response per scene; no repeated measures',
                        'no new inventory review, plan selection or terminal-check model calls',
                        'Hermes configuration changed but Hermes inference not launched',
                        'terminal checker remains qwen3-vl-plus'],
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files.update(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'registry_validation.json')
    files.add(ROOT / 'First_Phase/scene_demo/plans/20261014-qwen38-initial-recognition.json')
    registry_path = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
    registry = read(registry_path)
    entry = {
        'id': '20261014-qwen38-initial-recognition',
        'title': 'Qwen3.8初识别对照：双罐外部识别改善但腕部对应仍错；黄杯三视角正确',
        'date': '2026-10-14', 'question_keys': ['qwen38-initial-grounding'],
        'tags': ['Qwen3.8', 'initial-grounding', 'three-view', 'existing-images', 'zero-physical-actions'],
        'conditions': ['复用20261013-xvla-grounded-online两个场景各三张初态原图，旧请求原文与哈希核对通过，body仅model不同。',
                       '真实qwen3.8-max调用2次，每场景首次识别1次，无复核/重试；原temperature0、thinking_budget2048、max_tokens6500。',
                       '用户明确授权两组图片发送现有阿里云百炼；无Hermes推理、VLA或仿真动作、辅助器、子agent。'],
        'findings': ['双罐两外部视角的汤罐/番茄酱罐框正确，纸盒和扁包装已分开；腕部目标对应及部分干扰物对应仍错，完整三视角一致性未通过。',
                     '黄白杯三个视角目标框正确，与灰杯区分正确；本轮仅初态身份，不验证执行中门关闭。',
                     'X-VLA专用Hermes profile及初识别默认改为qwen3.8-max，旧profile已备份；末端检查仍qwen3-vl-plus。',
                     '已有身份接口15 tests通过，九项受保护控制/检查源码哈希与前轮一致；两次总用量10945tokens。'],
        'limits': audit['limitations'] + ['每场景n=1，旧模型未同期重测，不估计成功率。', '当前主agent模型/推理档位没有工具证据，不虚构设置。'],
        'configuration_complete': False,
        'evidence': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sorted(files)],
    }
    assert not any(e['id'] == entry['id'] for e in registry['entries'])
    registry['entries'].append(entry)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (ROOT / 'First_Phase/scene_demo/EXPERIMENT_INDEX.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 2026-10-14：Qwen3.8 初始识别对照\n\n- 实验 `20261014-qwen38-initial-recognition`：[报告](results/2026-10-14-qwen38-initial-recognition/summary.md)。\n- 同图同提示词仅换模型，两次真实识别、零机器人动作。双罐两外部正确但腕部对应仍错；黄杯三视角目标身份正确。\n- X-VLA Hermes与初识别默认切到qwen3.8-max；末端检查、收尾和复位保留。15项已有接口测试通过。\n')
    print(json.dumps({'registered': entry['id'], 'evidence_count': len(files), 'total_tokens': audit['total_tokens']}))


if __name__ == '__main__':
    main()
