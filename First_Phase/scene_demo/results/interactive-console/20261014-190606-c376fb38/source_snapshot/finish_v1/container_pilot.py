"""Once-only tomato basket and mug/microwave sequences with the current checker."""
import argparse
import json
import zipfile
from pathlib import Path
from . import combo_pilot as pilot
from .simple_checker import CHECKER_NAME

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-container-pilot'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-container-pilot.json'
CASES = {
    'sauce_basket': {'scene_id': 'basket_two', 'request': '把番茄酱罐放进篮子。',
                     'capabilities': ['sauce_to_basket']},
    'mug_microwave': {'scene_id': 'microwave_mug', 'request': '把黄白杯放进微波炉，然后关上微波炉门。',
                      'capabilities': ['mug_to_microwave', 'microwave_close']},
}
# Fixed before execution, independently of catalog/Hermes/Qwen outputs.
GOLD = {
    'sauce_to_basket': ['in', 'tomato_sauce_1', 'basket_1_contain_region'],
    'mug_to_microwave': ['in', 'white_yellow_mug_1', 'microwave_1_heating_region'],
    'microwave_close': ['close', 'microwave_1'],
}
SOURCES = pilot.SOURCES + [
    'First_Phase/scene_demo/finish_v1/feedback_retreat.py',
    'First_Phase/scene_demo/finish_v1/container_pilot.py',
]


def prepare():
    if OUT.exists() or PLAN.exists():
        raise RuntimeError('Do not overwrite an existing campaign')
    prior = ['20261007-placement', '20261009-joint-home-12', '20261010-visual-finish',
             '20261010-simple-checker', '20261011-goal-finish', '20261011-takeover-stove-cue']
    config = {
        'cases': CASES, 'gold_goals': GOLD, 'seed': 0, 'init_state_index': 0,
        'scenes': {'basket_two': {'suite': 'libero_10', 'task_id': 0},
                   'microwave_mug': {'suite': 'libero_10', 'task_id': 9}},
        'model_path': '/home/yhwang/fyp/libero_demo/models/smolvla_libero',
        'model_revision': '6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
        'source_sha256': {s: pilot.sha(ROOT / s) for s in SOURCES},
        'checker': CHECKER_NAME, 'qwen_model': 'qwen3.8-max', 'thinking_budget': 1024,
        'check_schedule': [150, 300], 'qwen_max_per_subgoal': 6,
        'vla_budget_per_subgoal': 300, 'grasp_helper': 'none',
        'completion_mode': 'release_verified', 'grasp_guard_mode': 'shadow', 'finish_mode': 'visual',
        'retreat': 'existing measured-pose 8mm feedback; CAD/depth path screens, max480 and max3 replans',
        'release': 'existing 20-cycle opening; fixture operation may open during screened retreat',
        'home': 'unchanged streaming .01rad kp50 max360; five ready post-action samples',
        'execution': 'user requests no subagents; primary session + local workers; no low-model invocation or effort override claimed',
        'initialization': 'one original reset per case; persistent scene between mug and door; fresh observations and policy reset between subgoals',
        'runtime_base_commit': '752b42f',
    }
    proposal = {
        'schema_version': 1, 'id': '20261011-container-pilot', 'intent': 'new_experiment',
        'question_key': 'container-goal-finish-transfer',
        'purpose': '在原生番茄酱入篮、杯子入微波炉→关门任务上，检查现有视觉目标/编码器检查器是否实际触发收尾、退离与原归位，并保留前序目标。',
        'prior_evidence': prior,
        'history_review': [{'experiment_id': x, 'disposition': 'reuse',
            'reason': '复用既有番茄酱未入篮/原生技能失败证据和新版两组合成功流程；旧结果没有回答当前检查器在篮子或微波炉原生场景的完整执行效果，不重复旧基线。'} for x in prior],
        'related_search': '检索registry、EXPERIMENT_INDEX、results README和catalog中的microwave/微波炉，无已归档微波炉动作证据；安装版BDDL核实LIBERO10 task9及LIBERO90 close task。',
        'hypothesis': '通用目标检查和已验证的恢复流程可移用于容器放置与关门；若VLA未到目标，检查器应拒绝收尾而不强制松手。',
        'information_gain': '两个新场景各一次实际执行；微波炉接口原先未支持close，新增明确门状态语义。不是Hermes规划增益实验、同状态配对或可靠性统计。',
        'success_criteria': '预期VLA150/300目标达成时Qwen true触发；记录release_open、movement、home实际动作>0、编码器开口和原home五连。各步独立固定已执行literal目标/无持物/无手接触/原home连续5个post样本通过；杯子完整成功后才关门，关门后杯子入炉成果保留。',
        'failure_criteria': '独立目标达到却检查未触发或目标动作0记触发链路未验证；VLA300未达到目标记底层执行失败，不能称收尾无效；动作已执行后目标丢失或退离/归位未完成分阶段记录。',
        'unknown_criteria': 'Qwen null、接口错误、同步反馈缺失等记录unknown并停止，不补造成功或反复现场修改提示词。',
        'changed_factors': [
            {'name': 'scene_and_subgoal', 'before': 'goal_table bowl/wine/stove', 'after': 'original basket tomato; original LIBERO10 task9 mug then microwave door', 'reason': '用户要求扩大到这两个具体场景'},
            {'name': 'articulated_context', 'before': 'on/in/turnon/turnoff only', 'after': 'explicit open/close door goal, same goal-only checker and recovery', 'reason': '必须准确表示关门目标，不能伪装为灶台开火'},
            {'name': 'retreat_scope', 'before': 'current feedback recovery enabled on tested wine/stove', 'after': 'same feedback recovery enabled for new container/door subgoals', 'reason': '使用已验证的实时退离流程，不回用已停滞的旧逐点退离'}],
        'fixed_factors': {'policy': 'same pretrained SmolVLA; no training or replay',
            'checker': 'current dual-view synchronized encoder goal-only prompt, no stage/phase labels or simulator truth',
            'home': 'original implementation and settings unchanged; no standalone retest',
            'protection': 'same CAD/depth route and original home screens',
            'budget': '150/300 checks, max300 VLA per subgoal; one attempt per case'},
        'scope': {'max_cases': 2, 'max_vla_actions_per_case': 600,
            'max_helper_actions_per_case': 1760,
            'stop_rule': '每例一次，不现场改词重跑；番茄失败时微波炉是独立场景可继续。杯子未完整收尾归位则关门不得启动；应触发而未触发/零目标动作停止依赖验证。不重复单独归位、不微调、不开子agent、不扩大。'},
        'new_physical_actions': 3540, 'configuration': config, 'configuration_complete': True,
        'repeat': {'needed': False, 'reason': 'current recovery and checker have no live results for these new scene/subgoal interfaces'},
    }
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    # Installed SDK/BDDL evidence only: no model or robot actions.
    from libero.libero import benchmark
    import libero.libero.envs.predicates.base_predicates as predicates
    import libero.libero.envs.objects.articulated_objects as objects
    import inspect
    suite = benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(9)
    assert task.name == 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it'
    sdk = Path(inspect.getfile(objects)).parents[1].parent
    bddl = sdk / 'bddl_files' / task.problem_folder / task.bddl_file
    (OUT / 'native_interface.txt').write_text(str(bddl) + '\n' + bddl.read_text() + '\n'
        + inspect.getsource(predicates.Close) + '\n' + inspect.getsource(objects.Microwave))
    print('Prepared two once-only cases; zero policy/helper actions or model calls', flush=True)


def run(case):
    from scene_demo.service import SceneService
    class ContainerService(SceneService):
        def _make_finish_monitor(self, session, plan, job, env, capability, budget, frames, latest_png):
            monitor = super()._make_finish_monitor(session, plan, job, env, capability, budget, frames, latest_png)
            monitor.port.feedback_retreat_enabled = True
            return monitor
    pilot.OUT, pilot.PLAN = OUT, PLAN
    pilot.run(case, service_class=ContainerService, budget_per_subgoal=300,
              case_spec=CASES[case], gold=GOLD)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.case)


if __name__ == '__main__':
    main()
