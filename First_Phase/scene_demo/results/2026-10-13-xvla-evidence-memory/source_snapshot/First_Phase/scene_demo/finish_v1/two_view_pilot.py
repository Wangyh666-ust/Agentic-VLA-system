"""Bounded two-view negatives, then the existing persistent-scene combo pilot."""
import argparse
import copy
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from . import combo_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-two-view-combos'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-two-view-combos.json'
NEGATIVE_PLAN = PLAN.with_name('2026-10-11-two-view-negatives.json')
OLD_RUN = ROOT / ('First_Phase/scene_demo/results/2026-10-11-generic-combos/bowl_wine/runs/'
                  '20261011T044326Z_goal_table_6e6d28d2/cap_01_bowl_to_plate_18c32a1f/finish/model_requests')
SOURCES = pilot.SOURCES + ['First_Phase/scene_demo/finish_v1/two_view_pilot.py']


def negatives():
    cases = []
    for index, name, phase in ((1, 'before_wrong_release', 'executing'), (2, 'beside_plate_after_home', 'home')):
        source = OLD_RUN / f'call_{index:02d}_request.json'
        record = json.loads(source.read_text(encoding='utf-8'))
        context = copy.deepcopy(record['context'])
        timestamp = max(fr['timestamp'] for fr in context['sensor_frames'])
        agent_index = next(i for i, fr in enumerate(context['sensor_frames'])
                           if fr['timestamp'] == timestamp and fr['view'] == 'agentview')
        agent = Path(record['images'][agent_index]['path'])
        count = int(re.fullmatch(r'cap(\d+)_agentview.png', agent.name).group(1))
        # SensorPort.capture saves agent, wrist, side sequentially at one sim time.
        paths = [agent.with_name(f'cap{count+1:06d}_wrist.png'), agent.with_name(f'cap{count+2:06d}_sideview.png')]
        context['sensor_frames'] = [{'frame': i, 'view': view, 'timestamp': timestamp}
                                    for i, view in enumerate(('wrist', 'sideview'))]
        context['current_subtask']['phase'] = phase
        context['actuator_measurements']['stage'] = phase
        assert context['actuator_measurements']['timestamp'] == timestamp
        cases.append({'id': name, 'context': context,
                      'source_request': str(source.relative_to(ROOT)), 'source_sha256': pilot.sha(source),
                      'images': [{'path': str(path), 'sha256': pilot.sha(path)} for path in paths],
                      'expected_goal_achieved': False})
    return cases


def prepare():
    if OUT.exists() or PLAN.exists() or NEGATIVE_PLAN.exists():
        raise RuntimeError('Do not overwrite an existing campaign')
    registry = json.loads((ROOT / 'First_Phase/scene_demo/experiment_registry.json').read_text(encoding='utf-8'))
    ids = [entry['id'] for entry in registry['entries'] if
           set(entry['question_keys']) & {'release-stop', 'visual-completion', 'joint-home-handoff'}]
    review = [{'experiment_id': identity, 'disposition': 'reuse',
               'reason': '复用既有成功与失败证据，不重跑独立归位；旧记录未使用当前同步双视角、无阶段标签和150/300在线检查组合，不能代替本轮验证。'}
              for identity in ids]
    proposal = json.loads((ROOT / 'First_Phase/scene_demo/plans/2026-10-11-generic-combos.json').read_text(encoding='utf-8'))
    proposal.update(id='20261011-two-view-combos', prior_evidence=ids, history_review=review,
        purpose='新配置先拒绝碗在盘旁的保存负例，再验证碗→酒瓶、碗→灶台真实触发松手、退离、原归位与下一子任务，保留前序成果。',
        hypothesis='同步腕部/外部两视角与去阶段标签能拒绝明确未到位负例，并允许已有收尾归位流程在两个固定组合中正确接管。',
        information_gain='与上一轮VLA60误松手相比，检验当前双视角/无阶段标签及150/300介入的组合效果；三个因素同时变化，不能独立归因。',
        success_criteria='两保存负例goal_achieved均为false（不以null通过）。随后每个应收尾子任务实际open>0、movement>0、home>0及归位五连ready；最终视觉与独立literal目标/无持物/无手接触/归位五个actual post通过，第二步读取新图并重置策略队列，两目标保留。',
        failure_criteria='任一保存负例goal=true或null就不启动物理组合。物理应触发而未触发、零收尾动作记触发链失败并停止依赖步骤；误触发、动作后目标不保持、第二技能失败分别记录，不能由归位成功代替。',
        unknown_criteria='模型null、请求/协议异常或初始化未知均停止，不猜测成功；单例通过不证明可靠性。')
    proposal['changed_factors'] = [
        {'name': 'camera_input', 'before': 'two timestamps, up to6 initial images; final main-only',
         'after': 'one current timestamp, wrist + external sideview, synced encoders, exactly2 images for every check', 'reason': '用户指定第一/第三视角同时检查'},
        {'name': 'stage_label', 'before': 'executing VLA incorrectly labeled post_execution in prompt',
         'after': 'no phase/stage labels sent to Qwen; accurate internal audit only', 'reason': '去除误导完成预期'},
        {'name': 'check_schedule', 'before': '60/120/180/240/300', 'after': '150/300 per subtask, candidate/force never early trigger',
         'reason': '按用户要求减少检查次数并检验真实接管'}]
    proposal['scope']['stop_rule'] = ('两保存负例依序各一次，任一不为false停止其余检查及物理组合。每组合最多一次；'
        '碗→酒瓶首碗未通过则停止第二步与碗→灶台；首碗通过但酒瓶失败可继续独立的碗→灶台。'
        '目标应触发却未触发停止依赖步骤，不调归位、不单独测试归位、不扩大、不微调、不开子agent。')
    config = proposal['configuration']
    config['source_sha256'] = {source: pilot.sha(ROOT/source) for source in SOURCES}
    config.update(checker='generic_state_handoff_two_current_views_no_stage',
                  check_schedule='VLA150/300 only; max2 prechecks plus final cleanup check',
                  negative_cases=negatives(), qwen_negative_max=2)
    pilot.write(PLAN, proposal)
    pilot.write(NEGATIVE_PLAN, {'schema_version': 1, 'id': '20261011-two-view-negatives',
        'intent': 'analyze_existing', 'question_key': 'visual-completion',
        'purpose': '当前检查器必须拒绝两组已保存的碗在盘旁负例，随后才允许新组合；零物理动作。',
        'prior_evidence': ids, 'history_review': review, 'new_physical_actions': 0,
        'analysis_steps': ['复用旧VLA60及收尾后同步编码器和同一capture腕部/侧视PNG，核对哈希。',
                           '使用当前原样提示词及Qwen参数，每组最多一次；必须明确false，不以null通过。',
                           '任一true/null停止后续及物理组合，原响应保留，不改词重试。'],
        'configuration': config})
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT/source, source)
    print('Prepared bounded negatives and two combos; zero model/physical actions', flush=True)


def verify_source(proposal):
    for source, digest in proposal['configuration']['source_sha256'].items():
        assert pilot.sha(ROOT/source) == digest, source


def check_negatives():
    from .simple_checker import SimpleQwenClient, reconcile, decide
    proposal = json.loads(NEGATIVE_PLAN.read_text(encoding='utf-8'))
    verify_source(proposal)
    directory = OUT/'negatives'
    directory.mkdir(exist_ok=False)
    command = [sys.executable, '-B', str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),
               'check', '--proposal', str(NEGATIVE_PLAN)]
    gate = subprocess.run(command, capture_output=True)
    (directory/'preflight.log').write_bytes(gate.stdout+gate.stderr)
    if gate.returncode:
        raise RuntimeError('Negative preflight rejected')
    client = SimpleQwenClient(max_calls=2)
    client.audit_dir = directory
    results = []
    for case in proposal['configuration']['negative_cases']:
        for picture in case['images']:
            assert pilot.sha(picture['path']) == picture['sha256']
        assert pilot.sha(ROOT/case['source_request']) == case['source_sha256']
        context = case['context']
        meta = client.analyze(context, [picture['path'] for picture in case['images']])
        measurements = context['actuator_measurements']
        fused = reconcile(context, meta['verdict'], {'finger_qpos': measurements['finger_qpos']},
                          {'finger_home': measurements['standard_open_finger_qpos']})
        row = {'case': case['id'], 'passed': meta['two_question_verdict'].get('goal_achieved') is False,
               'result': meta, 'fused': fused, 'decision': decide(context, fused), 'new_physical_actions': 0}
        results.append(row)
        pilot.write(directory/'result.json', {'passed': len(results)==2 and all(r['passed'] for r in results),
                    'cases': results, 'new_physical_actions': 0, 'qwen_calls': len(results)})
        print(json.dumps({'case': case['id'], 'passed': row['passed'], 'verdict': meta['two_question_verdict']}, ensure_ascii=False), flush=True)
        if not row['passed']:
            break


def run(case):
    gate = json.loads((OUT/'negatives/result.json').read_text(encoding='utf-8'))
    if gate.get('passed') is not True:
        raise RuntimeError('Negative examples did not pass; no physical launch')
    if case == 'bowl_stove':
        first = json.loads((OUT/'bowl_wine/result.json').read_text(encoding='utf-8'))
        if not first['subtasks'] or not first['subtasks'][0]['outcome'].get('ok'):
            raise RuntimeError('Shared first bowl failed; no dependent launch')
    pilot.OUT, pilot.PLAN = OUT, PLAN
    pilot.run(case)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--negative', action='store_true')
    group.add_argument('--case', choices=pilot.CASES)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.negative:
        check_negatives()
    else:
        run(args.case)


if __name__ == '__main__':
    main()
