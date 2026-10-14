"""Freeze the user's three literal requests; no inference or simulation."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / 'First_Phase/scene_demo'
OUT = Path(__file__).resolve().parent
PROPOSAL = HERE / 'plans/20261014-qwen38-three-requests.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


assert not PROPOSAL.exists() and not (OUT / 'source_snapshot').exists()
previous = read(HERE / 'plans/20261014-qwen38-long-pipeline.json')
configuration = copy.deepcopy(previous['configuration'])
hashes = {}
for relative in configuration['source_sha256']:
    source = ROOT / relative
    destination = OUT / 'source_snapshot' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
cases = {
    'table_tidy_stove': {'scene': 'goal_table', 'request': '整理桌面，包括碗、酒瓶，并且打开灶台', 'budget_per_subgoal': 800,
                        'intent_goals': [['on', 'akita_black_bowl_1', 'plate_1'], ['on', 'wine_bottle_1', 'wine_rack_1_top_region'], ['turnon', 'flat_stove_1']]},
    'heat_yellow_mug': {'scene': 'microwave_mug', 'request': '帮我将黄杯子加热一下', 'budget_per_subgoal': 900, 'intent_goals': []},
    'discard_two_cans': {'scene': 'basket_two', 'request': '帮我扔一下桌面上的两个罐头', 'budget_per_subgoal': 900, 'intent_goals': []},
}
configuration.update(source_sha256=hashes, cases=cases, online_cases=list(cases),
                     expected_capabilities_constraint=None, max_repairs=0, max_attempts_per_case=1,
                     acceptance='interpret the literal request; storage cannot silently substitute for discard and placement/door closing cannot constitute heating')
prior = {
    '20261014-qwen38-initial-recognition': '复用新模型配置、15项软件验证和初识别图片结果；尚无本轮三条原文的真实Hermes规划或动作证据。',
    '20261013-xvla-grounded-online': '复用原生长程、三视图、完整原图与收尾实现，并保留已知末端视觉误判风险。',
    '20261012-xvla-persistent-pilot': '复用X-VLA持久场景、多子任务与下一参考复位记录；整体整链未通过，不重做纯基线。',
    '20261008-skill-context': '复用收纳与丢弃意图的旧Hermes规划记录；无垃圾桶旧例正确unsupported，旧模型/旧请求不能回答本轮新模型三条原文。',
}
proposal = {
    'schema_version': 1, 'id': '20261014-qwen38-three-requests', 'intent': 'new_experiment',
    'question_key': 'qwen38-three-user-requests',
    'purpose': '使用当前Qwen3.8初识别/Hermes与X-VLA流水线，独立测试用户三条原文的意图理解、能力匹配和可执行部分完整流程。明确无法加热/丢弃时正确拒绝或澄清也属于规划效果，不能以入炉关门或入篮代替用户目标。',
    'prior_evidence': list(prior),
    'history_review': [{'experiment_id': k, 'disposition': 'reuse', 'reason': v} for k, v in prior.items()],
    'related_search': '已检索catalog.py公开storage_policy、全部动作、run_agent提示词、XvlaSceneService.submit_plan、真实旧规划结果及最新台账。当前没有微波炉启动加热能力、垃圾桶或丢弃能力。',
    'hypothesis': '当前流程应对明确整理与打开灶台选择可用动作，对缺少加热/丢弃能力的请求如实说明限制，而非选择近似收纳动作冒充完成。',
    'information_gain': '测试用户新原文而非预先点名动作任务；此前仅首次识别/旧具体任务，不能证明模型能识别高层意图及缺失能力。',
    'changed_factors': [{'name': 'user_requests', 'before': '显式入炉关门、双罐入篮或酒瓶先碗后灶的指令', 'after': '整理含碗酒瓶并开灶；加热黄杯；扔掉两罐三条原文', 'reason': '用户明确要求三个不同任务，必须保留原意。'},
                        {'name': 'pilot_plan_constraint', 'before': '预先固定capability_ids顺序', 'after': '不强制能力列表或顺序；Hermes自行选择，离线按用户目标评分', 'reason': '测试自然指令理解，不能把答案先强加给模型；生产控制与提示词不改。'}],
    'fixed_factors': {**previous['fixed_factors'], 'current_runtime_models': 'Hermes/初识别/选择复核=qwen3.8-max；末端=qwen3-vl-plus',
                      'negative_cases': '缺少加热与丢弃能力时预测零VLA，不强制运动。'},
    'success_criteria': '触发：每项真实初识别/复核和Hermes实际提交发生。规划：整理目标覆盖碗、酒瓶、开灶，顺序不强加；加热/丢弃缺能力时明确unsupported或clarify，不能声称完成。执行：可执行项记录真实动作、松手三图、末端证据、收尾与下一子任务reference复位；各用户目标独立判据达标并保持前序成果。负例正确零VLA作为正确能力边界，不算VLA失败。',
    'failure_criteria': '没提交计划或接口错误为流程未触发；把收纳替代丢弃、把关门替代加热为意图失败；规划正确但动作/检查/收尾失败分别记录；下游未触发记未验证。',
    'unknown_criteria': '模型传输失败、看不清目标、未取得完整当前/历史证据为未知；每项n=1不估计成功率。',
    'scope': {'max_cases': 3, 'max_vla_actions_per_case': 2400, 'max_helper_actions_per_case': 3000,
              'stop_rule': '三个独立场景各一次，无自动重跑、修复或现场改词；首项失败仍完成后两项的独立规划。单项已有1800秒期限与原保护保持；错误意图不得被重新解释为用户目标成功。助手0、子agent0。'},
    'new_physical_actions': 16200, 'configuration': configuration, 'configuration_complete': True,
    'repeat': {'needed': False, 'reason': '用户新原文的高层语义/缺能力判断未在新模型测试。'},
}
dump(PROPOSAL, proposal)
dump(OUT / 'source_manifest.json', hashes)
dump(OUT / 'cases.json', cases)
print(json.dumps({'proposal': str(PROPOSAL), 'cases': list(cases), 'model_calls': 0, 'physical_actions': 0}))
