"""Two bounded retries, preserving the successful bowl and home implementations."""
import argparse
import json
import zipfile
from . import assisted_combo_pilot as previous
from . import combo_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-feedback-retreat'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-feedback-retreat.json'
SOURCES = previous.SOURCES + [
    'First_Phase/scene_demo/finish_v1/feedback_retreat.py',
    'First_Phase/scene_demo/finish_v1/feedback_combo_pilot.py',
]


def prepare():
    if PLAN.exists() or OUT.exists():
        raise RuntimeError('Do not overwrite an existing campaign')
    proposal = json.loads(previous.PLAN.read_text())
    history = proposal['history_review'] + [
        {'experiment_id': '20261011-assisted-combos', 'disposition': 'reuse',
         'reason': '相同两组合各一次；碗与抓取通过，酒瓶追旧首点停滞、灶台残余旧路径阻断，归位均未启动。'},
        {'experiment_id': '20261011-withdrawal-diagnosis', 'disposition': 'reuse',
         'reason': '零动作静态复核揭示旧目标回追危险及过严逐点门槛；未证明新反馈退离的物理效果。'}]
    proposal.update(id='20261011-feedback-retreat', question_key='withdrawal-control-progress',
        purpose='减少退离逐点到位门槛，用最新实测姿态持续退离并在受阻时换方向，实际完成酒瓶、灶台退离后接原归位；两组合各一次，不开子agent、不微调。',
        prior_evidence=[row['experiment_id'] for row in history], history_review=history,
        hypothesis='每周期从实测位姿生成短退离命令，去除逐点关节到位门槛并允许受筛查的换向，可解决追旧目标造成的停滞。',
        information_gain='新退离实现的两条在线组合试点；旧碗、抓取、检查器与归位固定，区分实际退离动作、换向、最终间隙、归位和任务保留。不作为同随机流配对或总体可靠性估计。',
        success_criteria='首碗仍按原实现通过才执行第二技能。第二技能Qwen达到收尾条件后必须进入measured_pose_feedback并实际movement>0；去除中间0.004rad门槛，实测位移至少3cm、当前CAD手部间隙>=1cm且实测开口/停稳后进入原home；收尾后用户目标、前碗成果、无持物/手接触/原home独立五连通过。',
        failure_criteria='应触发但新退离零动作时停止该例依赖归位并记目标未验证；实际退离后仍无进展、无安全替代路线、目标丢失、或原归位未执行/未到位分阶段保留失败。',
        changed_factors=[
            {'name': 'retreat_controller', 'before': '2mm fixed waypoints with joint error<=0.004 before advance; stalled40 stops',
             'after': '8mm lookahead from measured encoder FK each cycle; actual progress/clearance; max3 screened alternate replans',
             'reason': '解除逐点死锁并避免回追已不适合当前姿态的旧目标'},
            {'name': 'retreat_screen_scope', 'before': 'remaining fixed route rescreen from actual pose back to old first target',
             'after': 'each fresh local command screened with intermediate endpoint requirement deferred until full exit; final hand clearance and original home protection retained',
             'reason': '短退离无需每一步已完全离开；接近障碍的路径规则仍保留'}],
        repeat={'needed': False, 'reason': 'new measured-pose retreat implementation; previously untested'})
    proposal['scope']['stop_rule'] = '两组合各一次；首碗不通过停止第二任务及另一组合。酒瓶检查未触发/新退离零动作停止本例下游，不用归位替代退离验证；已完成首碗时灶台为独立故障试点可继续。最多3次单次收尾内重规划、最多480退离动作；失败不现场重跑，不扩大，不微调，不开子agent。'
    proposal['configuration'].update(source_sha256={s: pilot.sha(ROOT / s) for s in SOURCES},
        retreat='pilot wine/stove only: measured-pose 8mm lookahead; cloud refresh10; short path .003rad CAD sampling; no intermediate endpoint gate; final >=3cm travel +1cm clearance; <=3 alternative replans, max480 movement; final opening/settling then unchanged home',
        execution='primary agent + local tools; explicit user no-subagent instruction, no model effort override claimed')
    proposal['fixed_factors']['bowl'] = 'feedback flag disabled for bowl; original successful release/retreat/home unchanged'
    proposal['fixed_factors']['grasp'] = 'previous validated wine assist and visual association unchanged'
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print('Prepared feedback retreat; zero physical/model calls', flush=True)


def run(case):
    if case == 'bowl_stove':
        first = json.loads((OUT / 'bowl_wine/result.json').read_text())
        if not first['subtasks'] or not first['subtasks'][0]['outcome'].get('ok'):
            raise RuntimeError('First bowl failed; dependent combo stopped')
    from .calibrated_grasp import CalibratedAssistService

    class FeedbackService(CalibratedAssistService):
        def _make_finish_monitor(self, session, plan, job, env, capability, budget, frames, latest_png):
            monitor = super()._make_finish_monitor(session, plan, job, env, capability, budget, frames, latest_png)
            monitor.port.feedback_retreat_enabled = job.capability_id in ('wine_to_rack', 'stove_on')
            return monitor

    pilot.OUT, pilot.PLAN = OUT, PLAN
    pilot.run(case, service_class=FeedbackService, budget_per_subgoal=500)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', choices=pilot.CASES)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.case)


if __name__ == '__main__':
    main()
