"""Freeze the explicitly authorized four independent two-question trials."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-10-simple-checker'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-10-simple-checker.json'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    evidence = OUT / 'evidence'
    evidence.mkdir(parents=True, exist_ok=False)
    old = ROOT / 'First_Phase/scene_demo/results/2026-10-10-live-handoff/evidence/inputs.json'
    data = json.loads(old.read_text(encoding='utf-8'))
    data['campaign_id'] = '20261010-simple-checker'
    for case in data['cases']:
        if case['case_id'] == 'sauce_negative':
            case['allow_helper'] = True  # Runtime visual verdict, never a fixture whitelist, controls release.
            case['expected_release'] = False
            case['expected_retreat'] = False
            case['context']['user_request'] = '请把番茄酱罐放进篮子。'
        case['context']['criteria']['operation_semantics'] = (
            '当前主要目的达成即停止VLA并移交收尾；不要求VLA自行松手或退离。')
        if case['context']['current_subtask']['operation'] == 'turn':
            case['context']['criteria']['desired_state'] = 'on'
    data['new_api_calls'] = 'at most six per independent case, four cases; no saved-image-only calls'
    source_paths = ['First_Phase/scene_demo/finish_v1/' + name for name in
                    ('handoff_pilot.py', 'control.py', 'runtime.py', 'vision.py', 'simple_checker.py',
                     'sensors.py', 'geometry.py', 'prepare_simple_pilot.py')]
    source_paths.append('First_Phase/scene_demo/joint_home.py')
    for item in data['frozen_files'].values():
        path = ROOT / item['path']
        item.update(bytes=path.stat().st_size, sha256=sha(path))
    for rel in source_paths:
        path = ROOT / rel
        data['frozen_files'][rel] = {'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    inputs = evidence / 'inputs.json'
    write(inputs, data)
    registry = json.loads((ROOT / 'First_Phase/scene_demo/experiment_registry.json').read_text(encoding='utf-8'))
    relevant = [e for e in registry['entries'] if 'release-stop' in e['question_keys']]
    plan = {
        'schema_version': 1, 'id': '20261010-simple-checker', 'intent': 'new_experiment',
        'question_key': 'release-stop',
        'purpose': '采用用户两问检查器，对酒瓶、碗、番茄酱罐、灶台开火四个独立真实仿真状态验证目的达成→实际开爪→退离→原归位。不开子agent、不微调。',
        'prior_evidence': [e['id'] for e in relevant],
        'history_review': [{'experiment_id': e['id'], 'disposition': 'reuse',
                            'reason': '保留已有末态、实际动作与阻断证据；旧视觉判定和开爪前等待不回答当前简化接管规则，原退离/归位复用。'} for e in relevant],
        'new_physical_actions': 3000,
        'hypothesis': '明确目的与夹持独立判断，目的达成直接开爪，可消除旧视觉误判和开爪前停稳阻断；真正未达目的的番茄状态应拒绝收尾。',
        'information_gain': '四项同版真实控制接管，非单纯保存画面检查；番茄原末态未入篮，提供不得误触发的独立样例，不捏造已达目的状态。',
        'success_criteria': '正例视觉目的达成且actual_open>0，实测标准开口，movement>0，旧streaming home>0且ready；独立目标维持、无持物/手部接触、标准归位连续五个实际post样本成立，最终两问视觉也通过。负例目的未达必须零收尾，且不宣称整任务成功。',
        'failure_criteria': '正例未触发或actual_open=0记触发失败；开爪后阻断记对应执行失败，不能替代归位结果。负例误触发记误报。分别报告目的/开爪/退离/归位/最终效果。',
        'unknown_criteria': '模型未知、网络、来源不一致或缺同步反馈记unknown；保存原始输入与响应，不无限重试。',
        'changed_factors': [
            {'name': 'checker', 'before': 'long multi-attribute prompt and automatic relation review', 'after': 'two-question concise prompt; one call per check; no relation review; encoder only controls opening', 'reason': '用户新指定检查器'},
            {'name': 'release_gate', 'before': 'up to ten closed-finger waits and five quiet samples before any open', 'after': 'hold current arm and immediately issue standard-open; two measured-open/arm-position confirmations; no pre-open speed gate', 'reason': '已达目的后实际执行补齐收尾，不等待VLA自行结束'}],
        'fixed_factors': {'fixtures': 'exact five old input fixtures, select wine/bowl/sauce/stove; same full state SHA',
                         'homing': 'unchanged joint_home.py, original streaming loop/config/path screening',
                         'retreat': 'unchanged geometry.py, RGBD path screening and action budgets',
                         'model': 'qwen3-vl-plus temperature=0, actual photos and synchronous encoders; no oracle inputs'},
        'scope': {'max_cases': 4, 'max_vla_actions_per_case': 0, 'max_helper_actions_per_case': 900,
                  'stop_rule': '每个独立任务只运行一次，当前例未触发/失败立即停止本例，不继续任何依赖组合；其余独立任务可继续。零新VLA，历史动作恢复末态。不开新agent，不扩大任务、不训练、不自动开启UI。'},
        'configuration': {'inputs_sha256': sha(inputs), 'source_sha256': {rel: sha(ROOT / rel) for rel in source_paths},
                          'cases': ['wine_exit', 'bowl_held83', 'sauce_negative', 'stove_exit'],
                          'qwen_max_per_case': 6, 'release': 20, 'movement': 120, 'confirm': 20, 'home': 360,
                          'init_steps': 10, 'replay_steps': [180, 83, 336, 78], 'simple_checker': True},
        'configuration_complete': True, 'repeat': {'needed': False, 'reason': 'new checker and handoff gates'},
        'execution': {'mode': 'single primary agent, per latest user instruction; no subagent', 'actual_model_override': 'none; current session model, no invented high/low switching'},
    }
    write(PLAN, plan)
    write(evidence / 'preflight.json', {'proposal_sha256': sha(PLAN), 'inputs_sha256': sha(inputs),
                                      'source_sha256': plan['configuration']['source_sha256']})
    write(evidence / 'registry_at_launch.json', registry)
    with zipfile.ZipFile(evidence / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in source_paths: z.write(ROOT / rel, rel)
    print(json.dumps({'proposal': str(PLAN), 'inputs': str(inputs), 'case_indices': {'wine': 1, 'bowl': 0, 'sauce': 4, 'stove': 2}}))


if __name__ == '__main__': main()
