"""Two once-only combinations with the committed visual-goal/encoder checker."""
import argparse
import json
import zipfile
from . import combo_pilot as pilot
from . import feedback_combo_pilot as feedback
from . import stove_candidate_start_pilot as latest

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-neutral-live-combos'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-neutral-live-combos.json'
SOURCES = list(dict.fromkeys(latest.SOURCES + ['First_Phase/scene_demo/finish_v1/neutral_combo_pilot.py']))


def prepare():
    if PLAN.exists() or OUT.exists():
        raise RuntimeError('Do not overwrite existing campaign')
    old = json.loads(latest.PLAN.read_text())
    for source, digest in old['configuration']['source_sha256'].items():
        if source not in ('First_Phase/scene_demo/finish_v1/simple_checker.py',
                          'First_Phase/scene_demo/finish_v1/control.py'):
            assert pilot.sha(ROOT / source) == digest, source
    ids = ['20261011-neutral-encoder-checker', '20261011-final-input-audit',
           '20261011-feedback-retreat', '20261011-stove-exit-repair',
           '20261011-stove-candidate-start', '20261011-goal-finish']
    reasons = [
        '新中性提示词与编码器主导已通过软件和旧记录融合，尚无新版真实模型/运动证据。',
        '原HTTP输入、双视角图像与编码器已核对；不重复保存画面复查。',
        '酒瓶已实际归位但最终视觉否决；灶台原回程碰旋钮，不能合并为视觉失败。',
        '候选退出点资格首修订误用起点导致退离0，失败保留。',
        '候选起点已修；前碗视觉接触否决使灶台未启动，修订路线仍缺实测。',
        '复用原归位与收尾成功证据，不单独重测归位。']
    config = dict(old['configuration'])
    config.update(cases=pilot.CASES, checker='visual_goal_encoder_v3',
                  source_sha256={source: pilot.sha(ROOT / source) for source in SOURCES},
                  base_commit='1364b62',
                  authorization='用户授权按该版本再走碗→酒瓶、碗→灶台；各一次，不开子agent、不训练。')
    proposal = {
        'schema_version': 1, 'id': '20261011-neutral-live-combos', 'intent': 'new_experiment',
        'question_key': 'visual-goal-encoder-authority',
        'purpose': '验证当前中性目标检查＋同步编码器开口判定版本能否在碗→酒瓶、碗→灶台中真正触发并完成松手、退离和原归位，保留前任务成果；各一次。',
        'prior_evidence': ids,
        'history_review': [{'experiment_id': eid, 'disposition': 'reuse', 'reason': reason}
                           for eid, reason in zip(ids, reasons)],
        'hypothesis': '视觉只判当前子任务目标、标准开口由实测编码器决定，可以消除视觉接触幻觉造成的末次否决，并允许已有收尾和归位完成交接。',
        'information_gain': '新版提示词首次真实Qwen＋新VLA在线组合；验证此前未进入的灶台修订路线。固定顺序，不测Hermes规划增益、不作为同随机流配对对照。',
        'success_criteria': '每个目标达到时在实际VLA150/300检查触发；actual release_open>0、movement>0、home>0且原home_streak>=5；末次视觉目标true且同步编码器标准开口，不被视觉接触否决。独立固定已执行全部literal目标、无持物/手接触及原home连续5个actual post样本通过；前碗成果保留，第二技能实际启动。',
        'failure_criteria': '目标应触发却未触发/对应目标动作0时标记该链路未验证，停止依赖后续；真实执行后目标丢失、退离或归位失败分别记录；末次视觉目标false保留，不据开爪改写位置成功。',
        'unknown_criteria': 'Qwen null、传输/协议失败、初始化或反馈不同步则unknown并停止，不自动重试或更换prompt，不用软件通过代替实测。',
        'changed_factors': [{'name': 'completion_checker',
            'before': 'two_question_goal_contact_v2; visual holding/contact may veto measured opening',
            'after': 'visual_goal_encoder_v3; neutral current-goal-only VLM; synchronized calibrated opening authoritative',
            'reason': '用户要求视觉判目标、编码器判开口，旧图遮挡或投影重叠不能否决松手'}],
        'fixed_factors': {'environment': 'same goal_table LIBERO goal8/Panda seed0 init0; one reset per combination',
            'policy': 'same pretrained SmolVLA; reset queue and fresh observations between subtasks; no training',
            'home': 'original streaming .01rad kp50 max360 five ready samples; path checks unchanged',
            'bowl': 'feedback retreat disabled; previous release/retreat/home unchanged',
            'wine_grasp': 'validated collision side-grasp with one dual-view visual association unchanged',
            'retreat': 'current measured-pose wine/stove feedback, candidate home qualification and explicit hypothetical start unchanged from current code; older wine run lacked endpoint qualification, no isolated causal attribution',
            'checks': 'actual VLA150/300; current wrist+sideview and synchronized encoder; no phase/stage in prompt',
            'independent': 'fixed literal executed goals, no holding/hand contacts/home five samples; never sent to Qwen'},
        'scope': {'max_cases': 2, 'max_vla_actions_per_case': 600, 'max_helper_actions_per_case': 1960,
            'stop_rule': '每组合一次，无现场改词/重跑。先碗→酒瓶：首碗失败停止第二技能及另一组合；首碗通过后酒瓶失败时灶台为独立第二技能可继续。应触发但实际目标动作0停止该例依赖验证，不单独测试归位、不微调、不扩大、不开子agent。'},
        'new_physical_actions': 5120, 'configuration': config, 'configuration_complete': True,
        'repeat': {'needed': False, 'reason': 'new committed goal/encoder checker; no live run of this version yet'}}
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print('Prepared two once-only live combinations; zero actions/model calls', flush=True)


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
