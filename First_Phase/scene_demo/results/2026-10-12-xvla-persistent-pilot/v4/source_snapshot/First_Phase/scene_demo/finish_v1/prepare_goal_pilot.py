"""Freeze bounded goal-only trigger and path-length-budget repair trials."""
import json
import zipfile
from .prepare_simple_pilot import ROOT, sha, write

NAME = '2026-10-11-goal-finish'
OUT = ROOT / 'First_Phase/scene_demo/results' / NAME
PLAN = ROOT / 'First_Phase/scene_demo/plans' / (NAME + '.json')


def main():
    evidence = OUT / 'evidence'
    evidence.mkdir(parents=True, exist_ok=False)
    data = json.loads((ROOT / 'First_Phase/scene_demo/results/2026-10-10-simple-checker/evidence/inputs.json').read_text())
    data['campaign_id'] = '20261011-goal-finish'
    for case in data['cases']:
        case['context']['criteria']['operation_semantics'] = '根据子任务，指定物品在目的地位置或要求开关状态达到即可交给收尾器。'
    source_paths = list(json.loads((ROOT / 'First_Phase/scene_demo/plans/2026-10-10-simple-checker.json').read_text())['configuration']['source_sha256'])
    source_paths += ['First_Phase/scene_demo/finish_v1/prepare_goal_pilot.py', 'First_Phase/scene_demo/finish_v1/run_simple_case.py']
    for item in data['frozen_files'].values():
        path = ROOT / item['path']
        item.update(bytes=path.stat().st_size, sha256=sha(path))
    for rel in source_paths:
        path = ROOT / rel
        data['frozen_files'][rel] = {'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    inputs = evidence / 'inputs.json'
    write(inputs, data)
    registry = json.loads((ROOT / 'First_Phase/scene_demo/experiment_registry.json').read_text())
    relevant = [e for e in registry['entries'] if 'release-stop' in e['question_keys']]
    plan = {
        'schema_version': 1, 'id': '20261011-goal-finish', 'intent': 'new_experiment', 'question_key': 'release-stop',
        'purpose': '删除任务外悬空/承托条件，修正灶台退离120周期过早结束，让目的判定触发实际松手、退离及原成功归位。不开子agent。',
        'prior_evidence': [e['id'] for e in relevant],
        'history_review': [{'experiment_id': e['id'], 'disposition': 'reuse', 'reason': '复用原末态和成功归位；最近酒瓶/碗未触发，灶台退离持续推进到14/20点却耗尽120周期，旧结果不回答修复效果。'} for e in relevant],
        'new_physical_actions': 4500,
        'hypothesis': '只询问任务目的和手接触可纠正误判；退离跟踪仍推进时按路径分配有界时间可完成原安全路线与归位。',
        'information_gain': '同真实初末态，酒瓶先验触发；灶台独立验证预算修复，不用软件通过代替实际执行。两因素同时改变不能分开归因视觉与控制总体效果。',
        'success_criteria': '正例goal true触发actual_open>0，实测开口；movement>0达到原安全路径终点，原streaming home>0且ready。独立目标/释放/无手接触/归位连续五个actual post样本真；最终两问也通过。',
        'failure_criteria': '正例目的未触发或开爪0停止其依赖，定位触发链；有动作未达效果报告具体阶段，零归位不得称归位失败。负例误触发为错误。',
        'unknown_criteria': '不同SHA、模型null、传输/协议失败记录unknown，不伪造成功。',
        'changed_factors': [
            {'name':'goal_checker','before':'11 model fields; goal AND supported AND at_destination','after':'two booleans goal_achieved/holding_or_contact and reason; goal alone triggers finishing','reason':'落实用户指定条件，无支撑/悬空附加检查'},
            {'name':'retreat_budget','before':'120 control samples for any path','after':'min(480,max(120,16*segments+40)); stalled40 samples stops; same waypoint tracking and .004 tolerance','reason':'旧记录每2mm约8-11周期，120仅覆盖14/20点，持续推进误停'}],
        'fixed_factors': {'fixtures':'exact historical initialization/trajectory/terminal hashes', 'homing':'original joint_home controller, path screen and 360-sample streaming execution unchanged', 'retreat_geometry':'unchanged geometry.py and SensorPort', 'model':'qwen3-vl-plus temperature0 actual RGB+synced encoders, no object oracle inputs'},
        'scope': {'max_cases':5, 'max_vla_actions_per_case':0, 'max_helper_actions_per_case':1500,
                  'stop_rule':'酒瓶、灶台、碗各一次，番茄负例一次；保留一次仅针对明确修复的重试。首例未触发不继续其依赖，独立灶台可检验预算。修复重试须另行冻结来源并过check。不组合、不训练、不子agent。'},
        'configuration': {'inputs_sha256':sha(inputs),'source_sha256':{rel:sha(ROOT/rel) for rel in source_paths},
                          'cases':['wine_exit','stove_exit','bowl_held83','sauce_negative'], 'qwen_max_per_case':6,
                          'release':20,'movement_max':480,'movement_formula':'min(480,max(120,16*segments+40))','no_progress_samples':40,'confirm':20,'home':360,'simple_checker':True},
        'configuration_complete':True,'repeat':{'needed':False,'reason':'changed actual prompt/schema and retreat budget'},
        'execution':{'mode':'single primary agent per user override, no subagents','model_override':'none; no claimed high/low switching'},
    }
    write(PLAN,plan)
    write(evidence/'preflight.json',{'proposal_sha256':sha(PLAN),'inputs_sha256':sha(inputs),'source_sha256':plan['configuration']['source_sha256']})
    with zipfile.ZipFile(evidence/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for rel in source_paths: archive.write(ROOT/rel,rel)
    old = ROOT/'First_Phase/scene_demo/results/2026-10-10-simple-checker/pilot_wine_repair/wine_exit/runtime_checks/model_requests/call_01_request.json'
    (evidence/'actual_old_prompt.txt').write_text(json.loads(old.read_text())['prompt']+'\n',encoding='utf-8')
    print(str(PLAN))


if __name__ == '__main__': main()
