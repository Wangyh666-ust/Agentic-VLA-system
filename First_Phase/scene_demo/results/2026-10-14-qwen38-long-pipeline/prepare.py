"""Freeze two authorized independent online cases without starting inference."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / 'First_Phase/scene_demo'
OUT = Path(__file__).resolve().parent
PROPOSAL = HERE / 'plans/20261014-qwen38-long-pipeline.json'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


assert not PROPOSAL.exists() and not (OUT / 'source_snapshot').exists()
previous = read(HERE / 'plans/20261013-xvla-grounded-online-renderfix.json')
configuration = copy.deepcopy(previous['configuration'])
hashes = {}
for relative in configuration['source_sha256']:
    source = ROOT / relative
    destination = OUT / 'source_snapshot' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
configuration.update(source_sha256=hashes, online_cases=['microwave_mug', 'basket_two'],
                     planner_model='qwen3.8-max', initial_inventory_model='qwen3.8-max',
                     selected_target_review_model='qwen3.8-max', terminal_checker_model='qwen3-vl-plus',
                     current_transport_timeout_s=90, max_attempts_per_case=1,
                     viewpoint_acceptance='clear external target evidence may suffice; wrist occlusion or imperfect local wording alone does not trigger manual cancellation',
                     subagents_used=0)
configuration.pop('failed_start_record', None)
prior = {
    '20261013-xvla-grounded-online': '复用相同seed/初态、原生长程、机位和收尾配置；旧版入炉误判门已关、双罐初识别错并人工停止，不能回答Qwen3.8规划初识别链的在线效果。',
    '20261014-qwen38-initial-recognition': '复用15项软件测试、配置核验及两次同图首次识别；新模型两外部双罐正确、黄杯正确，但尚无规划/动作/过程/收尾在线证据。',
    '20261012-xvla-native-benchmark': '复用预训练X-VLA原生输入、已保存初态与基础执行结果，不重复训练或纯原生基线。',
}
stop = ('两项独立各一次，首项失败仍测试第二项。单项按900 VLA动作及1800秒既有上限停止，max_repairs=0、不自动重跑。'
        '清晰外部视角能区分真实目标即可继续；腕部遮挡/未知/局部措辞不一致本身不人工停止。'
        '明确主目标身份误认且无可靠清晰参照、保护触发或程序/模型错误时终止该项，未执行阶段记未验证。'
        '不把零下游动作算归位效果失败，不在现场更改提示词、模型、机位、策略或收尾参数；不加入抓取搬运辅助器。')
proposal = {
    'schema_version': 1, 'id': '20261014-qwen38-long-pipeline', 'intent': 'new_experiment',
    'question_key': 'qwen38-long-pipeline',
    'purpose': '验证当前Qwen3.8规划与初识别配置能否让黄白杯入炉关门和双罐入篮两项原生长程独立完成识别、规划、策略动作、松手证据、末端检查、原收尾与末任务home复位。',
    'prior_evidence': list(prior),
    'history_review': [{'experiment_id': k, 'disposition': 'reuse', 'reason': v} for k, v in prior.items()],
    'related_search': '已读取最新两项实验台账、原请求响应、模型配置核验、上轮launch.py与xvla_pilot.run_case；尚无本配置完整长程在线结果。',
    'hypothesis': '新模型清晰外部视角身份识别改善可能使前置识别与规划顺利完成；末端旧模型的视觉误判仍可能阻断，分别归因，不预设完整流程成功。',
    'information_gain': '补足Qwen3.8首次识别之外的复核/规划/实际长程动作与原收尾交接证据，独立测两场景而不以首项成败筛选第二项。',
    'changed_factors': [
        {'name': 'planning_and_initial_identity_model', 'before': 'qwen3-vl-plus', 'after': 'qwen3.8-max', 'reason': '用户已要求替换并完成两次初识别，现在授权完整流程测试。'},
        {'name': 'manual_review_stop_rule', 'before': '初态跨视角定位错误人工停止双罐', 'after': '清晰外部参照可靠时允许局部腕部描述不完善继续，记录为局限', 'reason': '用户接受无需三视角完美一致；仍不把明确的错误实物当成正确目标。'},
    ],
    'fixed_factors': {**previous['fixed_factors'], 'terminal_checker': 'qwen3-vl-plus',
                      'camera_update': 'mj_camlight', 'check_interval': 150,
                      'controller_and_reset_sources': 'frozen hashes; original finish/retreat/home and queue/state reset unchanged',
                      'software_validation': 'previous 15 grounding tests passed; no runtime code changes this turn'},
    'success_criteria': '触发：初识别/清单复核、Hermes目标绑定/原生长程选取及选择复核均实际发生。执行：VLA动作数>0，实测松手事件三视图保存并进入后续审核。效果：原生AND与图像依据分别核对，视觉True必须对应正确目标/状态；原收尾、退离、home真实完成并满足原五连独立判据。当前每场景只有一个原生长程，因此结束后回全局home；下一子任务起始复位算法保持，不宣称本轮测试了跨技能交接。历史证据复用仅在实际发生旧图引用时计为已验证。',
    'failure_criteria': '前置识别/规划未完成为目标触发失败，未启动VLA不计策略失败；动作后未达原生目标为该次执行未达标；视觉误判或收尾失败分别记录；模型True或软件测试通过不代替实物成功。',
    'unknown_criteria': '遮挡且缺可信历史、模型传输失败、证据不足为未知；无自然松手/历史复用时相关机制未验证。两个单次试点不估计成功率。',
    'scope': {'max_cases': 2, 'max_vla_actions_per_case': 900, 'max_helper_actions_per_case': 3000, 'stop_rule': stop},
    'new_physical_actions': 7800, 'configuration': configuration, 'configuration_complete': True,
    'repeat': {'needed': False, 'reason': '已换Qwen3.8规划初识别；现有同图初识别没有新配置的实际长程证据。'},
}
dump(PROPOSAL, proposal)
dump(OUT / 'source_manifest.json', hashes)
print(json.dumps({'proposal': str(PROPOSAL), 'source_count': len(hashes), 'model_calls': 0, 'physical_actions': 0}))
