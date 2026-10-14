"""Once-only combinations: bounded takeover commands and static stove cues."""
import argparse
import json
import zipfile
from . import combo_pilot as pilot
from . import feedback_combo_pilot as feedback
from . import neutral_combo_pilot as previous
from .simple_checker import CHECKER_NAME

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-takeover-stove-cue'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-takeover-stove-cue.json'
SOURCES = previous.SOURCES + ['First_Phase/scene_demo/finish_v1/takeover_repair_pilot.py']


def prepare():
    if PLAN.exists() or OUT.exists():
        raise RuntimeError('Do not overwrite existing campaign')
    proposal = json.loads(previous.PLAN.read_text())
    changed = {'First_Phase/scene_demo/safe_exit.py', 'First_Phase/scene_demo/catalog.py',
               'First_Phase/scene_demo/finish_v1/simple_checker.py'}
    for source, digest in proposal['configuration']['source_sha256'].items():
        if source not in changed:
            assert pilot.sha(ROOT / source) == digest, source
    proposal['prior_evidence'].append('20261011-neutral-live-combos')
    proposal['history_review'].append({'experiment_id': '20261011-neutral-live-combos',
        'disposition': 'reuse', 'reason': '首碗两次归位通过；酒瓶目标和开口通过却因实测超ctrlrange约0.019mm而零收尾；灶台亮红圈但Qwen未知。只补这两项变化后的组合效果，不重测独立归位。'})
    proposal.update(id='20261011-takeover-stove-cue',
        purpose='修复微小实测越界的接管阻断，补充静态红圈开启定义；在碗→酒瓶、碗→灶台各一次中确认检查触发后真正开爪、退离和原归位，并保留两项任务成果。',
        hypothesis='接管时仅限幅发出的保持目标，且检查器知晓红色加热圈表示开启，可解除前轮两个入口阻断；后续退离/原归位效果仍需实际验证。',
        information_gain='真实新动作验证此前零执行的酒瓶接管和灶台触发；两项定向修复一同测试，不能分离所有因果贡献，也不估计总体成功率。',
        changed_factors=[
            {'name': 'finger_takeover', 'before': 'reject finite measured qpos strictly outside actuator command range',
             'after': 'preserve measured qpos; clip only generated hold targets and normalized commands to legal command range',
             'reason': '实测与指令范围混用导致约0.019mm越界拒绝接管'},
            {'name': 'static_visual_state_definition', 'before': 'goal-only prompt has no stove red-ring meaning',
             'after': 'catalog flat_stove on/off visual meanings in turn prompt; red heating rings mean on, no current-state claim',
             'reason': '灶台已开但检查器未识别红圈，需交代场景资产标志含义'}],
        repeat={'needed': False, 'reason': 'takeover conversion and static cue definition changed; no live result under these conditions yet'})
    proposal['configuration'].update(
        checker=CHECKER_NAME, base_commit='e390f10',
        source_sha256={source: pilot.sha(ROOT / source) for source in SOURCES},
        finger_takeover='finite measured position preserved; hold command clipped to actuator ctrlrange and [-1,1]',
        visual_state_definition='static catalog flat_stove on=red heating rings; off=clear burner with unlit rings; no oracle state',
        execution='user explicitly requests no subagents; primary session and local tools, no low-model invocation claimed',
        authorization='用户授权修复微小越界及补红圈开启定义，再试两个组合；不用子agent、不重复单独归位、不训练。')
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print('Prepared two once-only combinations; zero actions/model calls', flush=True)


def run(case):
    feedback.OUT, feedback.PLAN = OUT, PLAN
    feedback.run(case)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', choices=pilot.CASES)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.case)


if __name__ == '__main__':
    main()
