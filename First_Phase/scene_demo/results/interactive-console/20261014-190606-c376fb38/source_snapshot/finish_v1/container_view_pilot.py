"""One corrected-camera trial per container case; preserve the first failures."""
import argparse
import json
import zipfile
from . import combo_pilot as pilot
from . import container_pilot as previous

ROOT = pilot.ROOT
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-container-view'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-container-view.json'
SOURCES = previous.SOURCES + ['First_Phase/scene_demo/finish_v1/container_view_pilot.py']


def prepare():
    if OUT.exists() or PLAN.exists():
        raise RuntimeError('Do not overwrite existing campaign')
    proposal = json.loads(previous.PLAN.read_text())
    proposal['prior_evidence'].append('20261011-container-pilot')
    proposal['history_review'].append({'experiment_id': '20261011-container-pilot', 'disposition': 'reuse',
        'reason': '原始两图证实sideview漏篮子/微波炉背面；番茄150未知，杯子300误判入炉，原归位成功但入炉假、关门未启动。只补修正观察输入后的各一次，不重复独立归位。'})
    proposal.update(id='20261011-container-view',
        purpose='修正新场景外部观察相机，避免把容器旁或入口外误判为入内；在番茄入篮和杯子入炉→关门各一次中验证检查触发、实际收尾归位和目标保留。',
        information_gain='保留首次相机覆盖不足/误判记录，只补当前原生正面外部相机与空间关系提示下的完整流程；不是同随机流对照，不分离两个修正的贡献。',
        changed_factors=[
            {'name': 'checker_external_camera', 'before': 'wrist + native sideview which misses basket and views microwave back',
             'after': 'wrist + current native agentview with scene targets in view; actual camera labels preserved',
             'reason': '同一时刻已存图证实正面相机覆盖任务目标；仅选择检查输入，不改变VLA或路径几何'},
            {'name': 'goal_relation_clarity', 'before': 'goal-only generic prompt permits ambiguous projection interpretation',
             'after': 'explicit required in-relation: beside/entrance-outside/projection overlap does not establish inside',
             'reason': '杯子300步位于入口附近却被判入内；不新增支撑、姿态或松手条件'}],
        repeat={'needed': False, 'reason': 'new checker camera and relation wording; no result under corrected inputs'})
    proposal['configuration'].update(source_sha256={s: pilot.sha(ROOT / s) for s in SOURCES},
        runtime_base_commit='a3c6edd', checker_external_view='agentview',
        action_audit='every actual env.step source, VLA count and simulator timestamp; no simulator goal sent to model',
        initial_failures='20261011-container-pilot preserved unchanged')
    proposal['fixed_factors'].update(policy='same pretrained policy and atomic instructions; no training; new draws, no paired RNG claim',
        home='same control, geometry, joint_home implementations/settings as first trial; only checker input selection and wording change')
    pilot.write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print('Prepared corrected-view cases; zero actions/model calls', flush=True)


def run(case):
    from scene_demo import service, joint_home
    import numpy as np
    class ContainerViewService(service.SceneService):
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
    pilot.run(case, service_class=ContainerViewService, budget_per_subgoal=300,
              case_spec=previous.CASES[case], gold=previous.GOLD)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--case', choices=previous.CASES)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.case)


if __name__ == '__main__':
    main()
