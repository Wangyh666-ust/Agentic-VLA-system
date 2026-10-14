"""Audit completed offline plans, update the active role, and register evidence."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

from replay import ROOT, HERE, OUT, OLD, SOURCES, PROTECTED, PROPOSAL, read, dump, digest


def main():
    from scene_demo import setup_profile
    cases = {}
    for name in read(OUT / 'inputs.json'):
        directory = OUT / name
        result = read(directory / 'result.json')
        assert result['status'] == 'captured' and len(result['plans']) == 1
        assert not result['protocol_errors'] and not result['forwarded']
        assert result['physical_actions'] == result['vla_calls'] == result['simulator_instances'] == 0
        plan = result['plans'][0]
        if name == 'table_tidy_stove':
            assert plan['decision'] == 'execute' and plan['request_scope']['coverage'] == 'full'
            assert plan['capability_ids'] == ['bowl_to_plate', 'wine_to_rack', 'stove_on']
            assert result['selection_review']['passed'] is True
        elif name == 'heat_yellow_mug':
            assert plan['decision'] == 'execute' and plan['request_scope']['coverage'] == 'partial'
            assert plan['capability_ids'] == ['xvla_microwave_place_close']
            assert '启动' in ''.join(plan['request_scope']['unmet_outcomes'])
            assert result['selection_review']['passed'] is True
        else:
            assert plan['decision'] == 'unsupported' and plan['capability_ids'] == []
            assert plan['request_scope']['coverage'] == 'none'
        direct = []
        for body_path in sorted(directory.rglob('call_*_body.json')):
            prefix = body_path.name.removesuffix('_body.json')
            record = read(body_path.with_name(prefix + '_request.json'))
            meta = read(body_path.with_name(prefix + '_meta.json'))
            assert digest(body_path) == record['request_sha256'] == meta['request_sha256']
            assert digest(body_path.with_name(prefix + '_response.json')) == meta['response_sha256']
            images = [i for i in read(body_path)['messages'][0]['content'] if i['type'] == 'image_url']
            assert len(images) == len(record['images']) == 3
            for image, reference in zip(images, record['images']):
                actual = hashlib.sha256(base64.b64decode(image['image_url']['url'].split(',', 1)[1])).hexdigest()
                assert actual == reference['sha256'] == digest(Path(reference['path']))
            direct.append(meta)
        log = (directory / 'hermes_runtime_agent.log').read_text()
        usage = [dict(model=m, input_tokens=int(i), output_tokens=int(o), total_tokens=int(t))
                 for m, i, o, t in re.findall(r'API call #\d+: model=(\S+).*? in=(\d+) out=(\d+) total=(\d+)', log)]
        assert usage and all(u['model'] == 'qwen3.8-max' for u in usage)
        cases[name] = dict(request=result['request'], plan=plan, planning_pass=True,
            selection_review_pass=result.get('selection_review', {}).get('passed'),
            direct_qwen=direct, hermes_main_api=usage, physical_actions=0)

    frozen = read(OUT / 'source_manifest.json')
    assert all(digest(HERE / f) == sha for f, sha in frozen.items())
    protected = {f: digest(HERE / f) == read(OLD / 'source_manifest.json')['First_Phase/scene_demo/' + f] for f in PROTECTED}
    assert all(protected.values())
    port_status = {}
    for port in (8778, 8779, 8780, 8769, 8794):
        with socket.socket() as s:
            s.settimeout(.2)
            port_status[str(port)] = s.connect_ex(('127.0.0.1', port)) == 0
    assert not any(port_status.values())

    # Publish only the tested planner role. Keep credentials and all other
    # profile configuration untouched, with a recoverable local role backup.
    home = Path('/home/yhwang/fyp/xvla_native/hermes_home')
    config_sha = digest(home / 'config.yaml')
    before_sha = digest(home / 'SOUL.md')
    backup = home / 'SOUL.before-partial-scope-20261014.md'
    assert not backup.exists()
    shutil.copyfile(home / 'SOUL.md', backup)
    (home / 'SOUL.md').write_text(setup_profile.SOUL_MD, encoding='utf-8')
    assert digest(home / 'config.yaml') == config_sha
    deployment = dict(profile=str(home), role_before_sha256=before_sha, role_after_sha256=digest(home / 'SOUL.md'),
        config_unchanged_sha256=config_sha, backup=str(backup))
    dump(OUT / 'profile_update.json', deployment)

    audit = dict(cases=cases, scope_branch_observed=True, initial_original_images=9,
        all_request_image_hashes_matched=True, protected_source_unchanged=protected,
        listening_ports_after_completion=port_status,
        direct_qwen_responses=sum(len(c['direct_qwen']) for c in cases.values()),
        hermes_main_api_calls=sum(len(c['hermes_main_api']) for c in cases.values()),
        known_tokens=sum(sum(m['usage']['total_tokens'] for m in c['direct_qwen']) +
                        sum(m['total_tokens'] for m in c['hermes_main_api']) for c in cases.values()),
        usage_limit='Hermes title-generation activity appears in logs; its separate usage is unavailable, not counted as zero.',
        manual_image_review=dict(heat='Yellow-handle mug separate from grey speckled mug; external anchors match original pixels.',
            table='Bowl/plate, wine/rack and stove distinct in original side image; selected bindings corroborated.',
            cans='Both external-view anchors locate the two cylinders, not cartons. Wrist anchors still swap the two cans. '
                 'No execution/bindings were submitted for unsupported disposal; this does NOT validate can manipulation or full three-view identity consistency.'),
        physical_actions=0, subagents=0, new_grasp_transport_helpers=False, online_started=False,
        authoring_model='Actual root model/reasoning not evidenced; no delegated implementation model.')
    dump(OUT / 'audit.json', audit)
    summary = '''# 三条旧图的离线规划验证

通用规则已改为：执行明确有用的可支持部分，记录完整/部分/不支持、能做到的结果及剩余缺口；部分计划成功不会被报告成完整需求完成。规则共用于初次规划、原生长程选择、角色提示、工具提交、计划保存和结果返回。执行前目标复核也收到该范围。原收尾、退离和下一起点/home复位保持原实现。

复用上轮同三条原文和九张初始原图，每项重新调用Qwen3.8识别、复核，再由真实Hermes提交一次计划。接收服务仅保存计划，仿真实例、VLA调用、物理动作均为0。

| 原请求 | 实际提交 | 本轮判断 |
|---|---|---|
| 整理桌面，包括碗、酒瓶，并且打开灶台 | 碗放盘子→酒瓶放酒架→开灶；full，剩余缺口为空 | 规划及5项目标绑定复核通过 |
| 帮我将黄杯子加热一下 | 原生入炉关门；partial，明确缺少启动微波炉加热 | 关键部分执行分支真实触发，用户回答明确说明限制，3项目标绑定复核通过 |
| 帮我扔一下桌面上的两个罐头 | unsupported，空动作列表；没有将收纳篮擅自当垃圾桶 | 能力边界判断通过；没有验证双罐入篮动作 |

纠正上一轮报告的措辞：“入炉关门”作为加热准备是合理选择，缺陷是没有明确说明只能完成准备步骤，不能笼统称为技能选择错误。原始在线动作、判据与失败日志保留；本轮没有修复或重测其末端检查迟到和最终独立确认失败。

现有限制：

- 这是三条指定请求各一次的规划验证，不是整体机器人成功率，也不能证明所有长任务都已通过。
- 双罐两个外部视角定位到两个圆柱罐，腕图定位框仍有左右身份交叉。本次第三项无执行计划，故不影响“不擅自替换为收纳”的规划判断，但不能宣称双罐操控的三视角一致性已验收。
- 初始物体清单仍含“篮筐可作为投掷目的地”的用途推断；Hermes没有据此将篮筐当作垃圾桶。该描述不作为用途已确认的证据。
- 离线直接Qwen请求沿用180秒等待，生产仍为90秒。本轮通过不证明生产超时问题已解决。Hermes启动日志有自动依赖准备活动，未做严格锁定环境的单变量因果比较。
- 277项软件测试通过（187项unittest、90项pytest）。直接Qwen请求和原图哈希核对一致；原收尾/控制等9项受保护源码哈希未变。没有子agent或新抓取搬运辅助器。

证据：[完整审计](audit.json) · [整理计划](table_tidy_stove/submitted_plans.json) · [黄杯计划](heat_yellow_mug/submitted_plans.json) · [双罐计划](discard_two_cans/submitted_plans.json) · [黄杯真实回答](heat_yellow_mug/hermes.log) · [软件验证](software_validation.json)。

在线测试尚未启动。按用户要求，先报告离线结果并请求在线许可；若沿用第三条原文，预期是能力说明和零动作，而不是双罐入篮。
'''
    (OUT / 'summary.md').write_text(summary, encoding='utf-8')
    files = {PROPOSAL}
    files.update(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('incremental_validation.json', 'entry_registry.json'))
    for item in read(OUT / 'inputs.json').values():
        files.update(ROOT / relative for relative in item['image_sha256'])
    entry = dict(id='20261014-partial-scope-offline', title='部分执行规则：三条旧图真实Hermes规划通过；在线未启动',
        date='2026-10-14', question_keys=['partial-scope-offline-planning','hermes-intent-planning'],
        tags=['Qwen3.8','offline','partial-coverage','no-physical-actions'],
        conditions=['上轮三条原文与九张原始初态照片；每项新识别两次、真实Hermes一次，execute项选择复核一次。',
            '共8次直接Qwen响应，Hermes主会话API次数见audit；仿真/VLA/辅助器/子agent均0。'],
        findings=['整理full且碗→酒瓶→开灶，黄杯partial且原生入炉关门并明确缺少启动加热；两项目标选择复核通过。',
            '丢弃none/unsupported，无动作，不以收纳篮替代垃圾桶。',
            '修正前轮解释：准备技能选择合理，旧缺陷是能力边界说明不足；与末端视觉和独立完成失败分开。',
            '范围元数据经MCP保存并进入结果；部分计划成功不能提升为完整需求成功。受保护控制源码未改，在线未启动。'],
        limits=['三例n=1，只验证规划；不证明物理流程成功或所有长程可靠。',
            '双罐腕图两个身份框仍交换，外部图正确；本次无双罐操作绑定与动作验收。',
            '离线180秒等待不同于生产90秒；Hermes有依赖准备日志，不能严格单因果归因。',
            'Hermes标题生成用量未知；root实际模型/推理档位未获工具确认。'],
        configuration_complete=False, evidence=[dict(path=p.relative_to(ROOT).as_posix(), sha256=digest(p)) for p in sorted(files)])
    registry_path = HERE / 'experiment_registry.json'
    registry = read(registry_path)
    assert not any(e['id'] == entry['id'] for e in registry['entries'])
    baseline_sha = digest(registry_path)
    registry['entries'].append(entry)
    dump(registry_path, registry)
    dump(OUT / 'entry_registry.json', dict(schema_version=1, entries=[entry]))
    validation = subprocess.run([sys.executable, str(ROOT / '.agents/skills/fyp-experiment-review/scripts/review.py'),
        '--registry', str(OUT / 'entry_registry.json'), 'validate'], capture_output=True)
    payload = json.loads(validation.stdout)
    assert validation.returncode == 0 and payload['ok'] is True
    dump(OUT / 'incremental_validation.json', dict(previous_full_validation=dict(entries=58,evidence=15327,ok=True),
        unchanged_previous_registry_sha256=baseline_sha, new_entry_validation=payload,
        total_registry_entries=len(registry['entries']), note='Only appended one entry; prior 58 entries unchanged. Not a second full rehash.'))
    with (HERE / 'EXPERIMENT_INDEX.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 2026-10-14：部分执行与能力边界的旧图规划\n\n'
            '- 实验 `20261014-partial-scope-offline`：[报告](results/2026-10-14-partial-scope-offline/summary.md)。\n'
            '- 三条旧图规划通过：整理full；黄杯入炉关门partial且明确不能启动加热；丢弃unsupported且不擅自入篮。在线未启动。\n'
            '- 更正前项“加热误匹配”的表述：选择加热准备技能合理，旧问题是漏说能力缺口；物理失败记录仍保留。\n'
            '- 双罐腕图身份框仍交叉，不能宣称操控视觉验收通过；本轮第三项仅能力边界判断。\n')
    print(json.dumps(dict(registered=entry['id'], evidence=len(files), direct_qwen=audit['direct_qwen_responses'],
        hermes_main_api=audit['hermes_main_api_calls'], known_tokens=audit['known_tokens'], online_started=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
