import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parents[3]
base = root / 'First_Phase/scene_demo'
source = base / 'results/interactive-console/20261014-145221-ee32577c/proposal.json'
proposal = json.loads(source.read_text(encoding='utf-8'))
purpose = '验证更新后的实际网页从载入书收纳场景到三视图识别、Hermes生成待确认草稿能够通过；不提交动作计划。'
proposal.update(id='20261014-console-prepare-online', question_key='xvla-console-network-and-setup',
    purpose=purpose, intent='new_experiment', new_physical_actions=10,
    hypothesis='移除交互历史扫描、延长初识别请求及外层等待后，原三视图识别可返回并生成待用户确认的原生书收纳草稿。',
    information_gain='先前只有模型启动验证及旧图请求成功，尚未验证新网页实际串联的准备、识别、Hermes与草稿捕获接口。',
    success_criteria='准备不调用review.py且无全台账扫描；真实两次Qwen3.8初识别/复核成功，Hermes草稿为xvla_book_caddy且停在draft；仅10步初始化settling，零VLA推理/动作。',
    failure_criteria='准备扫描历史、识别/规划报错或未生成有效草稿为对应分支失败；不得用X-VLA已启动代替草稿成功。',
    unknown_criteria='用户主动中止或改变请求时记录中止，原错误无法复现不能断言原始网络异常类型。',
    prior_evidence=['20261014-console-startup-fix', '20261014-conservative-handoff-offline'],
    history_review=[{'experiment_id':'20261014-console-startup-fix','disposition':'reuse','reason':'模型两次启动通过，保留独立输出目录。'},
                    {'experiment_id':'20261014-conservative-handoff-offline','disposition':'reuse','reason':'沿用当前Qwen3.8与通用检查模块；本轮不触发动作检查或收尾。'}],
    related_search='原失败20261014-145221-ee32577c历史审计耗时248秒，初识别91秒报通用network failure；本轮旧图请求87.21/94.34秒均成功，但实际UI完整规划路径仍需验证。',
    changed_factors=[{'name':'interactive_preflight','before':'每次全量读取历史证据','after':'仅当前任务准备和源码留档','reason':'用户明确指出交互入口无需每次历史核对。'},
                     {'name':'initial_inventory_timeout','before':'90秒每请求，300秒外层','after':'180秒每请求，390秒外层','reason':'原图真实复核耗时94.34秒，90秒会截断有效请求。'}],
    fixed_factors={'scene':'book_caddy','seed':142,'init_state_index':0,'all_vlm':'qwen3.8-max',
                   'helpers':False,'confirmation_boundary':'只生成草稿，用户确认后才执行','finish_retreat_reset':'本轮不启动，源码不变'},
    scope={'max_cases':1,'max_vla_actions_per_case':0,'max_helper_actions_per_case':10,
           'stop_rule':'仅创建场景时原初始化10步settling；到draft停止，不调用execute。任一步失败则保存错误，停止后续；不自动重跑。'},
    repeat={'needed':False,'reason':'首次验证修改后的实际UI准备到草稿链路。'},
    configuration_complete=True,
    configuration={'scene':'book_caddy','request':'帮我整理一下桌子，把书本归位','seed':142,'init_state_index':0,
                   'all_vlm':'qwen3.8-max','inventory_timeout':180,'grounding_wait':390,'max_vla_actions':0,'max_settling_actions':10,
                   'source_sha256':{}})
snapshot = out / 'online_source_snapshot'
for file in list(base.glob('*.py')) + list((base/'finish_v1').glob('*.py')):
    rel = file.relative_to(base)
    target = snapshot / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = file.read_bytes()
    target.write_bytes(raw)
    proposal['configuration']['source_sha256'][rel.as_posix()] = hashlib.sha256(raw).hexdigest()
(out / 'online_proposal.json').write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding='utf-8')
print('Prepared UI-to-draft verification; no execute call authorized by this script.')
