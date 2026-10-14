"""Seal the two independent actual cases without changing previous evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('D:/FYP')
SCENE = ROOT/'First_Phase/scene_demo'
OUT = SCENE/'results/2026-10-13-xvla-timeline-trial'
ARCHIVE = OUT/'archive'
IDENTIFIER = '20261013-xvla-timeline-trial'
REGISTRY = SCENE/'experiment_registry.json'
INDEX = SCENE/'EXPERIMENT_INDEX.md'

def load(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def relative(path):
    return Path(path).relative_to(ROOT).as_posix()

before = load(OUT/'preflight/registry.before.json')
registry = load(REGISTRY)
assert registry == before and len(before['entries']) == 50
assert INDEX.read_bytes() == (OUT/'preflight/index.before.md').read_bytes()
assert not any(entry['id'] == IDENTIFIER for entry in registry['entries'])
old_evidence = [item for entry in before['entries'] for item in entry['evidence']]
assert len(old_evidence) == 9456 and all(sha(ROOT/item['path']) == item['sha256'] for item in old_evidence)
cleanup = load(OUT/'runtime/cleanup.json')
assert cleanup['terminated'] and all(not row['listening'] for row in cleanup['ports'])
assert all(row['all_match'] for row in cleanup['source_checks'].values())
assert cleanup['case_launches'] == {'microwave_mug': 1, 'basket_two': 1}
first = load(OUT/'audit.json')
second = load(OUT/'basket_independent/audit.json')
assert first['case'] == 'microwave_mug' and second['case'] == 'basket_two'
assert first['audit_model_calls'] == second['audit_model_calls'] == 0
assert first['audit_environment_actions'] == second['audit_environment_actions'] == 0
plans = [SCENE/'plans/20261013-xvla-timeline-trial.json',
         SCENE/'plans/20261013-xvla-timeline-basket-independent.json']
proposal_values = [load(path) for path in plans]
assert proposal_values[1]['configuration']['independent_scope_authorization'] == '用户明确选择：两项独立测试，首项失败也测入篮'
source_checks = []
for directory in (OUT, OUT/'basket_independent'):
    for path, expected in load(directory/'source_manifest.json').items():
        row = {'path': path, 'sha256': expected, 'current_match': sha(ROOT/path) == expected,
               'snapshot_match': sha(directory/'source_snapshot'/path) == expected}
        assert row['current_match'] and row['snapshot_match']
        source_checks.append(row)

case_summaries = {}
for audit in (first, second):
    directory = OUT/'microwave_mug' if audit['case'] == 'microwave_mug' else OUT/'basket_independent/basket_two'
    result_value = load(directory/'result.json')
    pilot_value = load(next(directory.rglob('pilot_result.json')))
    referenced = sum(bool(row.get('evidence_event_ids')) for packet in audit['packets']
                     if packet['role'] == 'current_and_process' for row in packet['effective_checks'])
    case_summaries[audit['case']] = {key: audit[key] for key in (
        'complete', 'ended_reason', 'source_actions', 'vla_calls', 'Hermes_invocations', 'Hermes_API_usage',
        'Qwen_requests', 'current_checks', 'release_classifications', 'predicate_first_true',
        'predicate_true_counts', 'native_success_first_true', 'native_success_true_steps', 'last_effective_checks')}
    case_summaries[audit['case']].update(
        current_history_references=referenced,
        release_events=[{'event_id': row['event_id'], 'step': row['vla_step'],
                         'predicates': row['declared_predicates'], 'protocol_passed': row['protocol_passed']}
                        for row in audit['release_audits']],
        packet_count=len(audit['packets']), all_packets_body_sha_match=all(row['body_sha_matches'] for row in audit['packets']),
        final_packet_image_count=audit['packets'][-1]['image_count'],
        final_packet_process_groups=audit['packets'][-1]['process_groups'],
        final_packet_previous_state_count=audit['packets'][-1]['previous_state_count'],
        last_native_success=audit['last_native_step']['native_success'],
        finish_status=pilot_value.get('finish_status'), handoff_evidence=result_value.get('handoff_evidence'),
        native_skill_route=result_value['plan'].get('native_skill_route'),
        planner_rationale=result_value['plan'].get('rationale'),
        independent_streak=(pilot_value.get('finish_status') or {}).get('independent_streak'))

limits = [
    '每项仅一次真实LIBERO/MuJoCo仿真，非实体机器人；无重试、无抓取或搬运辅助器。',
    '原生仿真位置/状态谓词与视觉判断分别记录，不把仿真真值传给检查器或用于代替其放行。',
    '输入链路已执行不等于目标身份识别、过程证据复用或完整任务成功。',
    '收尾/退离/回初始参考动作未启动时只能记未验证，不能记为模块失败。',
    '模型RNG未知/未固定，本轮不是严格配对对照，少量样例不证明总体可靠性或唯一根因。',
    'Hermes invocation可计数，但安装版本未导出该路径的API调用数或token使用量；root实际模型与档位未知。',
]
summary = {
    'schema_version': 1, 'id': IDENTIFIER,
    'purpose': '独立实测入炉关门和双罐入篮，核对通用松手三视图/中间状态输入、过程证据使用及原完整收尾效果。',
    'effective_scope': '两项分别一次；用户明确改为独立测试，首项失败仍测入篮。原条件方案和首项失败结果保持原字节。',
    'cases': case_summaries,
    'interpretation': '两项输入链路均已执行，实际模型没有用过程事件确认目标；原生目标达标与完整视觉流程未通过分别记录。',
    'micro_prompt_observation': 'call_04 prompt明确允许历史放置证据、区分目标与干扰物；原回答推理却要求goal_0当前证据。首次松手wrist可见黄色目标部分，外部桌上灰杯仍成为回答焦点。不能据此证明唯一根因。',
    'micro_door_observation': '最后900步模型门True与原生close一致，但150/300/450/600/750步仍有错误判断，不能称关门识别已修复。',
    'basket_prompt_observation': '目标定义区分圆柱汤罐/酱罐与纸盒/挤压瓶，prompt说明细标签不必全部清晰；原回答仍多次因颜色/文字及篮内遮挡返回身份未知。部分sideview未显示篮子。',
    'basket_route_observation': 'Hermes提交soup_to_basket、sauce_to_basket两个原子能力，服务按现有规则合并成xvla_basket_both执行。规划原话要求原子顺序，不能把服务合并称Hermes自己选择了原生长程；原生AND达标本身也不证明内部动作顺序。',
    'downstream': '原收尾、退离、归位和下一参考算法及配置保持；若finish动作0则本轮未验其效果。',
    'runtime_calls': {'X-VLA': first['vla_calls']+second['vla_calls'],
                      'Qwen': first['Qwen_requests']+second['Qwen_requests'],
                      'Hermes_invocations': first['Hermes_invocations']+second['Hermes_invocations'],
                      'Hermes_API_calls': None, 'Hermes_tokens': None},
    'root_model': 'unknown', 'root_effort': 'unknown', 'subagents_used': 0,
    'audit_new_model_calls': 0, 'audit_new_actions': 0,
    'source_snapshot_rows': len(source_checks), 'source_all_match': True,
    'cleanup': {'owned_server_terminated': cleanup['terminated'], 'ports_closed': True},
    'limits': limits,
}
dump(OUT/'summary.json', summary)
lines = [
    '本轮两项独立实测均未通过完整流程。原生仿真目标达标，但视觉检查没有放行，原收尾未启动。',
    '', '| 任务 | 原生仿真完整目标 | 末次视觉检查 | 完整流程 |', '|---|---|---|---|',
    '| 黄白杯入炉并关门 | 达标，首次第255步，末步仍为真 | 杯子未知；门已关闭 | 未通过 |',
    '| 汤罐、番茄酱罐入篮 | '+('达标，首次第'+str(second['native_success_first_true'])+'步，末步仍为真' if second['last_native_step']['native_success'] else '末步未达标')+' | 两罐均未知 | 未通过 |',
    '', '每项900个VLA步，各30次策略调用；每项两次实测松手均拍到wrist、agentview、sideview。最后检查请求各有9张照片：6张过程图＋3张当前图，另有5次先前检查状态。实际HTTP请求正文与原图片SHA离线重构核验相符，原始回答保存。',
    '', '两项模型都未引用过程event。入炉回答继续围绕桌上灰杯判断，虽然首次松手wrist图可见黄色目标部分；最后门判断正确，中间门判断仍有错误。双罐回答仍以身份或篮内位置不清返回未知。由此确认输入链路已执行，但过程照片复用的判定效果未通过；不能简单归因于照片没传，也不能把仿真判据替代视觉放行。',
    '', '具体末次视觉结果：',
]
for audit in (first, second):
    for row in audit['last_effective_checks']:
        lines.append('- '+audit['case']+' '+row['id']+'：'+str(row['achieved'])+'；'+row['reason'])
lines += ['', '未改prompt、阈值、策略预算或已验证控制/收尾配置，未重试，子agent 0。每项原prepare和10步settling保留；所有finish动作0时，退离与回初始参考未启动，只能记为效果未验证。',
          '', summary['basket_route_observation'],
          '', '这是每项一次仿真，不能推出总体成功率。模型RNG未固定；Hermes API次数与token使用量未导出。用户改为独立测试的范围与第二方案已保存；首次失败没有被覆盖。',
          '', '[结构化报告](summary.json) · [入炉逐请求审计](audit.json) · [入篮逐请求审计](basket_independent/audit.json)']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

ARCHIVE.mkdir(exist_ok=False)
files = sorted({path for path in OUT.rglob('*') if path.is_file()
                and ARCHIVE not in path.parents and path.name not in ('registry.before.json', 'index.before.md', 'evidence_manifest.json')}
               | set(plans), key=relative)
manifest = [{'path': relative(path), 'sha256': sha(path)} for path in files]
dump(OUT/'evidence_manifest.json', {'schema_version': 1, 'files': manifest, 'file_count': len(manifest),
                                 'scope': '两case与shared server关闭后封存；排除台账快照、自身及archive验证输出循环。'})
evidence = manifest+[{'path': relative(OUT/'evidence_manifest.json'), 'sha256': sha(OUT/'evidence_manifest.json')}]
configuration = {'effective_scope': summary['effective_scope'],
                 'microwave_registered_configuration': copy.deepcopy(proposal_values[0]['configuration']),
                 'basket_registered_configuration': copy.deepcopy(proposal_values[1]['configuration']),
                 'actual_execution': case_summaries, 'subagents_used': 0}
entry = {'id': IDENTIFIER, 'title': '新通用时间线两项独立实测：原生目标达标，视觉流程仍未通过',
         'date': '2026-10-13', 'question_keys': ['xvla-process-evidence', 'xvla-checker-view', 'xvla-native-skill-interface'],
         'tags': ['X-VLA', 'actual-simulation', 'timeline', 'three-view', 'independent-cases'],
         'conditions': [summary['effective_scope'], '原生技能、900步VLA、无抓取/搬运辅助、无重试；保留原收尾和准备配置。',
                        '每项一次Hermes invocation，实际X-VLA/Qwen调用见原日志；RNG与Hermes API使用量未知。'],
         'findings': [summary['interpretation'], summary['micro_door_observation'],
                      '每项2松手/9图末次输入/5先前状态已逐字节及同步反馈核验；模型过程事件引用0。',
                      '源快照保持；原finish动作0为未启动，不能判模块效果失败。'],
         'limits': limits, 'configuration': configuration, 'configuration_complete': True, 'evidence': evidence}
registry['entries'].append(entry)
assert load(REGISTRY) == before
dump(REGISTRY, registry)
with INDEX.open('ab') as stream:
    stream.write(('\n\n### '+IDENTIFIER+'：新时间线两项独立实测\n\n'
                  '- 入炉关门与双罐入篮各一次，原生完整目标达到，视觉检查未放行；每项两次松手三图、末次9图与5次先前状态已传入，过程引用0。\n'
                  '- 原收尾未启动；无抓取/搬运辅助，无重试，子agent0。\n'
                  '- [报告](results/2026-10-13-xvla-timeline-trial/summary.md) · [独立入篮方案](plans/20261013-xvla-timeline-basket-independent.json)\n').encode('utf-8'))
argv = [sys.executable, str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'), 'validate']
dump(ARCHIVE/'validate.argv.json', argv)
result = subprocess.run(argv, cwd=ROOT, capture_output=True)
(ARCHIVE/'validate.stdout').write_bytes(result.stdout)
(ARCHIVE/'validate.stderr').write_bytes(result.stderr)
(ARCHIVE/'validate.exitcode').write_text(str(result.returncode)+'\n', encoding='utf-8')
after = load(REGISTRY)
invariants = {'validate_exitcode': result.returncode, 'validate': json.loads(result.stdout),
              'old_entries_deep_equal': after['entries'][:-1] == before['entries'],
              'old_9456_evidence_sha_match': all(sha(ROOT/item['path']) == item['sha256'] for item in old_evidence),
              'index_old_prefix_equal': INDEX.read_bytes().startswith((OUT/'preflight/index.before.md').read_bytes()),
              'new_all_evidence_sha_match': all(sha(ROOT/item['path']) == item['sha256'] for item in evidence),
              'source_all_match': all(row['current_match'] and row['snapshot_match'] for row in source_checks),
              'new_evidence_count': len(evidence), 'new_audit_model_calls': 0, 'new_audit_actions': 0, 'subagents_used': 0}
dump(ARCHIVE/'invariants.json', invariants)
assert result.returncode == 0 and invariants['validate']['entry_count'] == 51
assert all(invariants[key] for key in ('old_entries_deep_equal', 'old_9456_evidence_sha_match', 'index_old_prefix_equal', 'new_all_evidence_sha_match', 'source_all_match'))
print(json.dumps(invariants, ensure_ascii=False), flush=True)
