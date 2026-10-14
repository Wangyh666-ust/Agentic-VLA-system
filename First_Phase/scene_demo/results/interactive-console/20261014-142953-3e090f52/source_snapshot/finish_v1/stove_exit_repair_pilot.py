"""One revised stove combo after preserving the first feedback batch."""
import argparse
import json
import zipfile
from . import feedback_combo_pilot as previous
from . import combo_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-stove-exit-repair'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-stove-exit-repair.json'
SOURCES = previous.SOURCES + ['First_Phase/scene_demo/finish_v1/stove_exit_repair_pilot.py']


def prepare():
    if PLAN.exists() or OUT.exists():
        raise RuntimeError('Do not overwrite an existing campaign')
    proposal = json.loads(previous.PLAN.read_text())
    proposal['history_review'].append({'experiment_id': '20261011-feedback-retreat', 'disposition': 'reuse',
        'reason': '两例反馈退离真实完成；酒瓶物理归位通过但视觉否决，灶台当前清空却原回程第6-9点碰旋钮。旧试点冻结；仅改退离终点选择，复用原回程筛查。'})
    proposal['prior_evidence'].append('20261011-feedback-retreat')
    proposal.update(id='20261011-stove-exit-repair',
        purpose='仅修订灶台退离终点选择，退出到可安全接原归位的姿态；碗→灶台补一个组合样例，原失败保留，不单测归位、不开子agent、不训练。',
        hypothesis='退离候选终点和实际退出点同时通过原归位路径筛查，可消除当前已清空但回程扫旋钮的交接遗漏。',
        information_gain='上批已执行退离却home0；本例隔离增加退出终点的原回程资格，不改变归位实现，测量真正home>0且开火和前碗目标保留。',
        success_criteria='首碗原流程通过。灶台检查触发反馈退出，actual movement>0；至少一个actual home_candidate筛查通过，实际标准开口/停稳后原home>0、home_streak5、全部物体/开关目标/无接触/无持物独立五连通过，最终视觉认可单列。',
        failure_criteria='反馈/原home应启动而零动作分别标记对应触发或路径资格失败，不用下游替代本目标；存在碰撞预测的候选必须继续退出或换向，未找到安全退出点则停止。',
        changed_factors=[{'name': 'exit_endpoint_eligibility',
            'before': '>=3cm displacement and current hand clearance; return checked only after retreat ends',
            'after': 'candidate terminal and latest actual exit must pass original home_screen; unsafe return continues withdrawal or replans',
            'reason': '上批退出点当前无接触，但原回程会扫旋钮；修退离交接，不改原home'}],
        scope={'max_cases': 1, 'max_vla_actions_per_case': 600, 'max_helper_actions_per_case': 1960,
            'stop_rule': '仅本修订碗→灶台一次；首碗失败停止灶台。灶台退离目标未触发停止其下游；最多480退离、3重规划，原home最多360。保留所有失败，无同版现场重跑，无新任务扩展、无子agent/训练、无单独归位测试。'},
        new_physical_actions=2560,
        repeat={'needed': False, 'reason': 'new exit-to-home eligibility, previous physical failure retained'})
    proposal['configuration'].update(cases={'bowl_stove': pilot.CASES['bowl_stove']},
        source_sha256={source: pilot.sha(ROOT / source) for source in SOURCES},
        exit_endpoint='candidate and actual exit use unchanged original home_screen; blocked actual return continues exiting or replans',
        execution='no subagents; primary agent and local tools; no effort override claimed')
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print('Prepared stove endpoint repair; zero physical/model calls', flush=True)


def run():
    from .calibrated_grasp import CalibratedAssistService

    class FeedbackService(CalibratedAssistService):
        def _make_finish_monitor(self, session, plan, job, env, capability, budget, frames, latest_png):
            monitor = super()._make_finish_monitor(session, plan, job, env, capability, budget, frames, latest_png)
            monitor.port.feedback_retreat_enabled = job.capability_id == 'stove_on'
            return monitor

    pilot.OUT, pilot.PLAN = OUT, PLAN
    pilot.run('bowl_stove', service_class=FeedbackService, budget_per_subgoal=500)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == '__main__':
    main()
