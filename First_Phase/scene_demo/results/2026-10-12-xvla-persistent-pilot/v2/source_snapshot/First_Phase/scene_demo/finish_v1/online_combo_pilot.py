"""Fresh online 150/300 combo tests; never replay or overwrite prior campaigns."""
import argparse
import json
import zipfile

from . import combo_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-online-150-300-combos'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-online-150-300-combos.json'
SOURCES = pilot.SOURCES + ['First_Phase/scene_demo/finish_v1/online_combo_pilot.py']


def prepare():
    if PLAN.exists() or OUT.exists():
        raise RuntimeError('Do not overwrite an existing campaign')
    proposal = json.loads((ROOT / 'First_Phase/scene_demo/plans/2026-10-11-two-view-combos.json').read_text())
    registry = json.loads((ROOT / 'First_Phase/scene_demo/experiment_registry.json').read_text())
    ids = [e['id'] for e in registry['entries'] if
           set(e['question_keys']) & {'release-stop', 'visual-completion', 'joint-home-handoff'}]
    proposal.update(
        id='20261011-online-150-300-combos',
        purpose='实际在线验证碗→酒瓶、碗→灶台在150/300检查后真正松手、退离、原归位并交接，不用保存图片复核代替运动。',
        prior_evidence=ids,
        history_review=[{'experiment_id': i, 'disposition': 'reuse', 'reason':
            '旧60步在线误松手与双视角保存负例失败均保留；保存图片未进入150/300在线分支，不能回答新在线流程。复用已成功归位，不单测归位。'} for i in ids],
        hypothesis='150/300时刻的真实当前双视角和编码器能正确决定接管；若误判则独立验收阻止后续任务。',
        information_gain='首次实际运行当前150/300在线配置，观察触发、真实收尾与持久场景交接；保存负例失败未解决，不能宣称检查器已可靠。',
        success_criteria='每个已执行子任务仅在VLA150/300在线检查；目的true接管，实际开爪/退离/原归位均发生，最终视觉和独立固定literal目标、无持物、无手接触、归位五连通过；第二任务读取新画面且前序目标保留。',
        failure_criteria='误触发、应触发未触发、收尾零动作、目标未保持、最终视觉错误及第二任务执行失败分别记录。首碗未通过即停止其后续及另一组合，不以归位替代任务完成。',
        unknown_criteria='请求/初始化/模型未知停止，原始错误保留；未启动不计技能失败。',
        changed_factors=[{'name':'execution_scope', 'before':'saved-negative gate failed; new online motion never started',
            'after':'user explicitly requests fresh real online bowl→wine / bowl→stove; no repeated saved-image gate',
            'reason':'2026-10-11用户明确再次要求测试两个实际多步流程，失败负例仍保留且不改为通过'}])
    proposal.pop('startup_correction', None)
    proposal['scope']['stop_rule'] = (
        '每组合最多一次，不重跑保存图、不单测归位。先碗→酒瓶，首碗未完成正确收尾/归位与独立验收，'
        '停止酒瓶及碗→灶台；首碗通过但酒瓶失败可继续碗→灶台。应触发未触发停止依赖步骤，不现场改词重试，不微调、不扩大、不开子agent。')
    config = proposal['configuration']
    config.pop('negative_cases', None)
    config.pop('qwen_negative_max', None)
    config.update(source_sha256={s: pilot.sha(ROOT / s) for s in SOURCES},
        saved_negative_status='failed and unresolved; retained as prior risk, not a pass',
        authorization='user requests actual online combinations after clarification of offline-only prior run')
    proposal['repeat'] = {'needed': False, 'reason': 'first real online execution of this checkpoint configuration'}
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print('Fresh online campaign prepared; zero motion/model calls', flush=True)


def run(case):
    if case == 'bowl_stove':
        first = json.loads((OUT / 'bowl_wine/result.json').read_text())
        if not first['subtasks'] or not first['subtasks'][0]['outcome'].get('ok'):
            raise RuntimeError('First bowl failed; dependent combination stopped')
    pilot.OUT, pilot.PLAN = OUT, PLAN
    pilot.run(case)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', choices=pilot.CASES)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.case)


if __name__ == '__main__':
    main()
