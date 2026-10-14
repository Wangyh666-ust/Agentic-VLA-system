"""Bounded 600-action container trials after checker status/identity correction."""
import argparse
import json
import zipfile
from . import combo_pilot as pilot
from . import container_pilot as previous
from .simple_checker import CHECKER_NAME

ROOT = pilot.ROOT
OUT = ROOT / "First_Phase/scene_demo/results/2026-10-11-container-budget"
PLAN = ROOT / "First_Phase/scene_demo/plans/2026-10-11-container-budget.json"
SOURCES = previous.SOURCES + ["First_Phase/scene_demo/finish_v1/container_budget_pilot.py", "First_Phase/scene_demo/mcp_server.py", "First_Phase/scene_demo/run_agent.py"]

def prepare():
    if OUT.exists() or PLAN.exists():
        raise RuntimeError('Do not overwrite an existing campaign')
    proposal = json.loads((ROOT / 'First_Phase/scene_demo/plans/2026-10-11-container-view.json').read_text())
    proposal['prior_evidence'].append('20261011-container-view')
    proposal['history_review'].append({'experiment_id': '20261011-container-view', 'disposition': 'reuse',
        'reason': '复用保存双图、同步编码器、实际指令及逐动作记录；已知目标为tomato_sauce罐头而非Milk纸盒/ketchup瓶，未入篮却null提前停止，杯子入口外300步null。未获得600步执行结果，不重测独立归位。'})
    proposal.update(id='20261011-container-budget',
        purpose='纠正指定物品识别与明确未达标却unknown的检查语义；每150个实际VLA动作检查，最多600步，在原先两个容器场景各一次观察目标、实际收尾归位和杯子→关门交接。',
        hypothesis='身份和三值描述明确后，保存的未入篮/入口外应判false；延长预算能避免程序在150未知或300检查时提前终止，但不能保证VLA选对物品或完成放置。',
        information_gain='三个已保存负例只各检查一次用于prompt验收，零机器人动作；各一例新600步执行补充此前没有的300之后能力及当前触发证据。多因素修正，不分离各自因果贡献。',
        success_criteria='保存的番茄150/杯子150/300负例分别返回false且不把Milk当tomato_sauce；在线仅实际VLA150/300/450/600检查，false及预算前null均继续且不松手；true接管后记录actual release_open/movement/home及原home五连。独立固定已执行目标、无持物/手接触和原归位五连均通过；杯子完整成功后才关门，成果保留。',
        failure_criteria='保存负例true或null则停止在线并定位提示词；在线误触发记检查失败，600目标未达到记本次执行未完成。独立已达目标却不触发或零收尾停止依赖验证，不能据此评价收尾器；实际执行后按开爪、退离、归位、成果分别报告。',
        unknown_criteria='有效模型null仅作当前观察未知，600预算前继续至下一150检查点，600仍null停止；传输、协议、相机或编码器同步错误立即停止，不用文字规则把历史null篡改为false。',
        changed_factors=[
            {'name': 'target_identity_and_status', 'before': 'generic goal v4 may substitute manipulated object and let obscured view veto clear negative',
             'after': 'v5 checks named target; asset-verified static appearance for sauce; clear unmet relation false, genuine unobservable relation null',
             'reason': 'Milk偏离目标及未入篮却unknown必须明确区分'},
            {'name': 'action_budget', 'before': 'max300 VLA, only150/300 checkpoints',
             'after': 'max600 VLA, periodic150/300/450/600; finishing unchanged and separately counted',
             'reason': '用户明确长任务需要600步，不把300检查当上限'},
            {'name': 'unknown_continuation', 'before': 'any null immediately ends subtask',
             'after': 'valid null before budget does not release and continues; budget-final null stops',
             'reason': '此前150未知提前结束不能检验后续动作'}],
        fixed_factors={'policy': 'same pretrained weights, atomic instructions, seed0/init0; no training/replay/grasp helper',
            'camera': 'same native wrist+agentview and synchronized measured encoder, no simulator object state sent to Qwen',
            'home_and_finish': 'unchanged control/feedback_retreat/geometry/joint_home settings; no standalone homing',
            'scene': 'original LIBERO10 task0/task9; persistent scene between mug and door; policy queue reset at subtask boundary'},
        scope={'max_cases': 2, 'max_vla_actions_per_case': 1200, 'max_helper_actions_per_case': 1760,
            'stop_rule': '保存负例检查未通过就不启动新运动；各例一次，不现场改词重跑。番茄与杯子为独立场景；杯子未完整成功则不关门。应触发却未触发/零收尾停止依赖验证。保留原2000会话总步保护，必要时如实记录session_limit。不微调、不子agent、不扩大、不单测归位。'},
        new_physical_actions=4160,
        repeat={'needed': False, 'reason': 'new checker and 600-action budget with explicit unknown continuation'})
    proposal['configuration'].update(source_sha256={s: pilot.sha(ROOT / s) for s in SOURCES},
        runtime_base_commit='926b99a', checker=CHECKER_NAME, check_schedule=[150, 300, 450, 600],
        vla_budget_per_subgoal=600, qwen_max_per_subgoal=6, valid_null_before_budget='continue',
        offline_negative_calls=3, wall_budget_seconds_per_case=1800, session_step_limit=2000)
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print('Prepared two bounded 600-action cases and three saved negatives; zero actions/model calls', flush=True)

def run(case):
    from scene_demo import service, joint_home
    import numpy as np
    class ContainerViewService(service.SceneService):
        def _observation_batch(self, obs, instruction):
            batch = super()._observation_batch(obs, instruction)
            assert batch['task'] == [instruction]
            with (OUT / case / 'policy_input_audit.jsonl').open('a') as handle:
                handle.write(json.dumps({'actual_vla_before': getattr(self, '_actual_vla', 0),
                                        'instruction': instruction, 'batch_task': batch['task']}) + '\n')
            return batch

        def _make_finish_monitor(self, session, plan, job, env, capability, budget, frames, latest_png):
            monitor = super()._make_finish_monitor(session, plan, job, env, capability, budget, frames, latest_png)
            monitor.port.feedback_retreat_enabled = True
            monitor.port.checker_external_view = 'agentview'
            return monitor

        def _select_action(self, batch):
            action = super()._select_action(batch)
            self._pending_source = 'vla'
            return action

        def _run_capability(self, session, plan, job, capability_id):
            self._actual_vla = self._local_grasp_actions = 0
            self._last_action_source = 'vla'
            self._pending_source = None
            env = self._env
            original = env.step
            def tracked(action):
                source = self._pending_source or 'finish'
                self._pending_source = None
                result = original(action)
                if source == 'vla':
                    self._actual_vla += 1
                self._last_action_source = source
                with (job.run_dir / 'action_sources.jsonl').open('a') as handle:
                    handle.write(json.dumps({'source': source, 'actual_vla': self._actual_vla,
                        'timestamp': float(joint_home._single_robot(env).sim.data.time),
                        'action': np.asarray(action).tolist()}) + '\n')
                return result
            env.step = tracked
            try:
                return super()._run_capability(session, plan, job, capability_id)
            finally:
                env.step = original
    pilot.OUT, pilot.PLAN = OUT, PLAN
    pilot.run(case, service_class=ContainerViewService, budget_per_subgoal=600,
              case_spec={**previous.CASES[case], "wall_budget_seconds": 1800}, gold=previous.GOLD)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', choices=previous.CASES)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.case)


if __name__ == '__main__':
    main()
