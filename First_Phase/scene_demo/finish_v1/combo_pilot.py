"""Two bounded, real-VLA persistent-scene sequences using the generic checker."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'First_Phase/scene_demo/results/2026-10-11-generic-combos'
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-generic-combos.json'
CASES = {
    'bowl_wine': {'request': '先将碗放到盘子上，再将酒瓶放到架子上。', 'capabilities': ['bowl_to_plate', 'wine_to_rack']},
    'bowl_stove': {'request': '先将碗放到盘子上，再打开灶台。', 'capabilities': ['bowl_to_plate', 'stove_on']},
}
GOLD = {
    'bowl_to_plate': ['on', 'akita_black_bowl_1', 'plate_1'],
    'wine_to_rack': ['on', 'wine_bottle_1', 'wine_rack_1_top_region'],
    'stove_on': ['turnon', 'flat_stove_1'],
}
SOURCES = [f'First_Phase/scene_demo/finish_v1/{s}.py' for s in
           ['combo_pilot','simple_checker','runtime','vision','control','sensors','geometry']]
SOURCES += [f'First_Phase/scene_demo/{s}.py' for s in ['service','joint_home','safe_exit','safe_exit_contacts','catalog','placement_completion']]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist()) + '\n', encoding='utf-8')


def prepare():
    import zipfile
    registry = json.loads((ROOT/'First_Phase/scene_demo/experiment_registry.json').read_text())
    history = [e for e in registry['entries'] if 'release-stop' in e['question_keys']]
    related = ['20261009-joint-home', '20261009-joint-home-12', '20261011-goal-finish']
    ids = list(dict.fromkeys([e['id'] for e in history]+related))
    config = {'scene': 'goal_table', 'suite': 'libero_goal', 'task_id': 8, 'seed': 0, 'init_state_index': 0,
              'cases': CASES, 'model_path': '/home/yhwang/fyp/libero_demo/models/smolvla_libero',
              'model_revision': '6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
              'source_sha256': {s: sha(ROOT/s) for s in SOURCES},
              'vla_budget_per_subgoal': 300, 'qwen_max_per_subgoal': 6,
              'completion_mode': 'release_verified', 'grasp_guard_mode': 'shadow',
              'finish_mode': 'visual', 'checker': 'generic_state_handoff', 'thinking_budget': 1024,
              'check_schedule': 'interval60 VLA actions, max5 prechecks; native candidate ignored by pilot',
              'home': 'original streaming, .01rad waypoints, kp50, max360, five ready samples',
              'execution': 'single primary agent and local worker; no delegation or claimed model override'}
    proposal = {'schema_version': 1, 'id': '20261011-generic-combos', 'intent': 'new_experiment',
                'question_key': 'release-stop',
                'purpose': '用用户确认的通用结束状态提示词，让碗→酒瓶、碗→灶台各子任务实际触发松手/退离/原归位后交接，并保留前序成果。',
                'prior_evidence': ids,
                'history_review': [{'experiment_id': e, 'disposition': 'reuse', 'reason': '复用已成功的归位和末态诊断；旧12组合/上轮单例未使用本轮通用角色提示词及在线两问接管，不能回答这两个新在线组合。'} for e in ids],
                'hypothesis': '明确检查器负责判定收尾触发条件，可在不附加支撑条件时配合已有恢复流程串联两个子任务。',
                'information_gain': '首次在同一持久场景使用本轮通用提示词进行新VLA两步执行，无回放；固定用户指定顺序，不作为Hermes规划增益测量。',
                'success_criteria': '每步Qwen目的true；实际open>0、movement>0、home>0且home ready；最终视觉分离与独立已执行全部literal目标/无持物/无手接触/归位连续5个actual post真，第二步读取新图并重置策略队列；全序列两目标保留。',
                'failure_criteria': '应触发未触发或目标动作0记触发失败，停止依赖交接；动作执行后未达效果记相应阶段失败；第二技能抓取/位置失败与第一步收尾失败分开。',
                'unknown_criteria': '模型null、传输/协议失败或初始化未知则unknown并停止；不将软件通过或归位单独替代组合效果。',
                'changed_factors': [
                    {'name':'checker_prompt','before':'task examples and literal wine-specific correction','after':'user-approved generic end-state inspector role + current subtask natural language','reason':'检验通用语义而不枚举任务'},
                    {'name':'online_dispatch','before':'runtime.poll default old vision decide and simple_handoff false','after':'explicit SimpleQwenClient dispatch to simple_checker.decide and simple_handoff true; initial place includes wrist','reason':'独立末态试点成功路径接入在线入口'},
                    {'name':'new_sequence','before':'historical terminal replay only','after':'real VLA bowl then wine/stove in persistent scene; periodic check without native candidate trigger','reason':'用户授权组合试点，不把回放称作新执行'}],
                'fixed_factors': {'home_controller':'joint_home.py and control.py unchanged from46f197e', 'geometry':'sensors/geometry unchanged', 'policy':'same pretrained SmolVLA; no training', 'scene':'goal_table original, seed0 init0; one reset per whole sequence'},
                'scope': {'max_cases': 2, 'max_vla_actions_per_case': 600, 'max_helper_actions_per_case': 1800,
                          'stop_rule': '每组合最多一次；首例碗若未完成实际收尾/归位则停止第二步及另一组合；碗通过但酒瓶失败时碗→灶台为独立第二技能对照可继续。零触发停止依赖动作，不继续调归位；不微调、不扩大、不开子agent。'},
                'new_physical_actions': 4800, 'configuration': config, 'configuration_complete': True,
                'repeat': {'needed': False, 'reason': 'new generic prompt, online dispatch and real two-step sequence'}}
    if PLAN.exists() or OUT.exists(): raise RuntimeError('Do not overwrite existing campaign')
    write(PLAN, proposal)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(OUT/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for s in SOURCES: z.write(ROOT/s,s)
    print(str(PLAN))


def run(case_id):
    proposal = json.loads(PLAN.read_text())
    for name, digest in proposal['configuration']['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    review = ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'
    gate = subprocess.run([sys.executable,'-B',str(review),'check','--proposal',str(PLAN)], capture_output=True)
    directory = OUT/case_id
    directory.mkdir(exist_ok=False)
    (directory/'preflight.log').write_bytes(gate.stdout+gate.stderr)
    if gate.returncode: raise RuntimeError('Preflight rejected')
    from scene_demo import service, joint_home, safe_exit_contacts
    from scene_demo.finish_v1.simple_checker import SimpleQwenClient
    spec = CASES[case_id]
    timings, subresults = [], []
    def event(name, **fields):
        row = {'event':name,'utc':datetime.now(timezone.utc).isoformat(),**fields}
        with (directory/'events.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    class PilotService(service.SceneService):
        def _make_finish_monitor(self, session, plan, job, env, capability, budget, frames, latest_png):
            monitor = super()._make_finish_monitor(session, plan, job, env, capability, budget, frames, latest_png)
            monitor.call_client = SimpleQwenClient(max_calls=6)
            monitor._client_explicit = True
            index = spec['capabilities'].index(job.capability_id)
            fixed_goals = [GOLD[c] for c in spec['capabilities'][:index+1]]
            def independent():
                values = {service.catalog.goal_key(g): bool(service.eval_goal_predicate(env,g)) for g in fixed_goals}
                state = joint_home.read_robot_state(env)
                home = joint_home.home_metrics(session.home_reference,state)
                contacts = safe_exit_contacts.contacts(env)
                released = self._completion_ready(env,fixed_goals,values)
                ready = bool(released and home['ready'] and contacts==[])
                with (job.run_dir/'independent_finish.jsonl').open('a') as f:
                    f.write(json.dumps({'timestamp':float(joint_home._single_robot(env).sim.data.time),'predicates':values,'home':home,'contacts':contacts,'release_status':self._last_completion_status,'ready':ready},default=lambda x:x.tolist())+'\n')
                return ready
            monitor.independent_ready = independent
            original_check = monitor._check
            def check(*args):
                t=time.perf_counter()
                try:return original_check(*args)
                finally:
                    timings.append({'kind':'qwen_check','capability':job.capability_id,'seconds':time.perf_counter()-t})
                    event('visual_check_done',capability=job.capability_id,seconds=timings[-1]['seconds'],decision=monitor._last_decision)
            monitor._check = check
            original_poll = monitor.poll
            monitor.poll = lambda step, candidate=False, force=False: original_poll(step,candidate=False,force=force)
            job._pilot_monitor = monitor
            return monitor
        def _run_capability(self, session, plan, job, capability_id):
            before = service.state_sha(self._env)
            t=time.perf_counter();event('subtask_start',capability=capability_id,state_sha=before)
            outcome = super()._run_capability(session,plan,job,capability_id)
            monitor = getattr(job,'_pilot_monitor',None)
            row={'capability':capability_id,'outcome':outcome,'job':job.public(),'state_before':before,
                 'state_after':service.state_sha(self._env),'elapsed_seconds':time.perf_counter()-t,
                 'finish':None if monitor is None else monitor._finish,'independent_streak':None if monitor is None else monitor.independent_streak}
            write(job.run_dir/'pilot_result.json',row);subresults.append(row)
            event('subtask_end',capability=capability_id,ok=outcome.get('ok'),steps=job.steps,
                  helper_counts=None if monitor is None else monitor.helper_counts,reason=outcome.get('reason'))
            return outcome
    svc = PilotService(run_root=str(directory/'runs'), completion_mode='release_verified', grasp_guard_mode='shadow',finish_mode='visual')
    start=time.perf_counter();svc.start()
    try:
        deadline=time.monotonic()+180
        while not svc.health()['ready']:
            if svc.health().get('worker_error') or time.monotonic()>deadline:raise RuntimeError(str(svc.health()))
            time.sleep(.2)
        write(directory/'health.json',svc.health())
        session=svc.create_session('goal_table',seed=0,init_state_index=0)
        if not session.get('ok'):raise RuntimeError(str(session))
        sid=session['session_id'];request_id='generic_'+case_id
        bound=svc.bind_request_context(request_id,sid,spec['request'])
        if not bound.get('ok'):raise RuntimeError(str(bound))
        payload={'session_id':sid,'scene_version':session['scene_version'],'request_id':request_id,
                 'decision':'execute','capability_ids':spec['capabilities'],'budget_per_subgoal':300,
                 'rationale':'用户指定顺序；固定执行组合试点，不作为Hermes规划评估'}
        write(directory/'submitted_plan.json',payload)
        submitted=svc.submit_plan(payload)
        if submitted.get('ok') is False:raise RuntimeError(str(submitted))
        deadline=time.monotonic()+900
        while True:
            plan=svc.plan(request_id)
            if plan['state'] in ('completed','blocked','error','cancelled'):break
            if time.monotonic()>deadline:
                svc.cancel(request_id);raise RuntimeError('pilot wall budget exceeded')
            time.sleep(.25)
        write(directory/'result.json',{'case':case_id,'user_request':spec['request'],'plan':plan,'subtasks':subresults,
                                      'timings':timings,'wall_seconds':time.perf_counter()-start,'new_vla':sum(r['job']['steps'] for r in subresults),
                                      'replay_actions':0,'subagents':0,'limitation':'Fixed user-specified sequence, not new Hermes planner run'})
        event('case_done',state=plan['state'],plan_success=plan.get('plan_success'))
    finally:svc.stop()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case',choices=CASES)
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.case:run(a.case)
    else:p.error('choose --prepare or --case')


if __name__=='__main__':main()
