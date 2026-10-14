"""Frozen opt-in X-VLA pilot. Import and --help do not construct a simulator."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / 'results/2026-10-13-xvla-all-native/pilot'
DEFAULT_PROPOSAL = HERE / 'plans/20261013-xvla-all-native.json'
REVISION = '129e71460678b7236cee6fc9707f09d9fa0c3590'
PURPOSE = '验证全部十个LIBERO-10原生长程技能的真实Hermes公开能力选择与完全匹配原子序列融合，在三同步视图逐项整体检查及已验证收尾下保持目标并集，以完整技能边界交接和复位；软件接线尚不代表物理效果。'
CASES = {
 'wine_bowl_stove': {'scene': 'goal_table', 'request': '先把酒瓶放到架子上，再把碗放到盘子上，最后打开灶台。', 'capabilities': ['wine_to_rack','bowl_to_plate','stove_on']},
 'basket_two': {'scene': 'basket_two','request': '先把字母汤罐放进篮子，再把番茄酱罐放进篮子。','capabilities':['xvla_basket_both']},
 'mugs_two': {'scene':'mugs_two','request':'先把白杯放到左边盘子，再把黄白杯放到右边盘子。','capabilities':['xvla_mugs_both']},
 'microwave_mug': {'scene':'microwave_mug','request':'先把黄白杯放进微波炉，再关闭微波炉门。','capabilities':['xvla_microwave_place_close']},
}
try:
    from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
except ModuleNotFoundError:
    from xvla_skills import NATIVE_LONG_SKILLS
for _native, _skill in NATIVE_LONG_SKILLS.items():
    _scene = _skill['scene_ids'][0]
    if _scene not in CASES:
        CASES[_scene] = {'scene':_scene, 'request':_skill['instruction'], 'capabilities':[_native]}
for _name, _case in CASES.items():
    _case['budget_per_subgoal'] = 800 if _name == 'wine_bowl_stove' else 900

HISTORY = {
 '20261013-xvla-all-native-software':'复用本轮十技能软件/真实目录资格证据；未新增模型/物理动作，效果待验证。',
 '20261013-xvla-native-three-view-software':'复用本轮软件/请求字节资格证据；没有新增实际模型/物理动作，物理效果仍待验证。',
 '20261012-xvla-persistent-pilot':'复用旧持久服务、控制器、下一参考及五连判据；本轮改变技能与检查输入。',
 '20261012-xvla-input-audit':'复用原生输入审计；本轮仅检查器增加第三视图。',
 '20261012-xvla-checker-grounding':'复用检查器 grounding 证据；本轮三同步视图与逐项整体检查。',
 '20261012-xvla-native-benchmark':'复用X-VLA原生输入、初态和无辅助动作证据；本轮增加持久场景Hermes与交接。',
 '20261011-native-benchmark':'复用旧模型原生对照；本轮改X-VLA绝对动作与下一任务起始参考。',
 '20261011-takeover-stove-cue':'复用灶台收尾视觉编码器证据和参数；本轮保持实现但更换策略与归位参考。',
 '20261011-container-budget':'复用容器多步失败及预算证据；本轮策略和交接不同。',
 '20261009-joint-home':'复用关节归位保护及独立测量；本轮目标为下一原生任务参考。',
 '20261009-joint-home-12':'复用扩展关节归位证据；本轮加末端姿态判据。',
 '20261008-skill-context':'复用真实Hermes规划与工具上下文；本轮用户顺序与新服务身份。',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2),encoding='utf-8')

def frozen_sources():
    sources = [HERE / p for p in ('xvla_service.py','xvla_backend.py','xvla_pilot.py','xvla_skills.py','xvla_finish.py','service.py',
        'run_agent.py','mcp_server.py','setup_profile.py','joint_home.py','safe_exit.py',
        'safe_exit_contacts.py','placement_completion.py','catalog.py')]
    sources += sorted((HERE / 'finish_v1').glob('*.py'))
    bench = HERE / 'results/2026-10-12-xvla-native-benchmark'
    for suite, task in [('libero_goal',9),('libero_goal',8),('libero_goal',7)] + [('libero_10', i) for i in range(10)]:
        directory = bench / suite / ('task_%02d' % task) / 'episode_0/attempt_001'
        sources += [directory / 'initial_state.npy',directory / 'initial_audit.json']
    return sources

def prepare(output, proposal):
    if proposal.exists() or (output/'source_snapshot').exists() or any((output/k).exists() for k in CASES):
        raise FileExistsError('refuse to overwrite frozen proposal or case records')
    output.mkdir(parents=True,exist_ok=True)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for source in frozen_sources():
        relative = source.relative_to(ROOT).as_posix()
        destination = snapshot / relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        hashes[relative] = sha(source)
    dump(output / 'source_manifest.json',hashes)
    success = [
        '真实Hermes请求计划留存；直接选原生或服务仅融合连续完全匹配的原子顺序，保持目标并集并审计requested/executed/replacements；无匹配保留原子，max_repairs=0。',
        '目标true时150动作三同步相机wrist/agentview/sideview与同步编码器逐项完整技能检查分支实际触发，并记录触发事件；预算尾部不以未经检查成功。',
        '记录真实finger_target开口、编码器开口、退离与归位动作数及前后状态；下游零动作不替代验证。',
        '仅完整技能结束时归位/reset，不在component中途结束；中间归位reference准确来自下一原生初态，关节/开口/速度及末端5mm/0.05rad到位，策略队列清空、实测proprio重建并刷新图像。',
        '所有当前及前序goals原completion gate保留、无持物手接触、独立效果5连通过；末任务回全局home。']
    stop = '首条链任一目标未达、应触发未触发或下游零动作，停止依赖验证和其余十个native case；不用home替代目标。程序错误、不同步或model transport错误停止，无现场改词、重跑或加辅助。'
    config = {'cases':CASES,'model_revision':REVISION,
      'weights_sha256':'260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8',
      'source_xvla':'6bc2513f5f1cbec715cc668b414392a6cae5c671','source_libero':'8f1084e3132a39270c3a13ebe37270a43ece2a01',
      'dtype':'float32','chunk':30,'denoise':10,'domain':3,'control':'absolute','seed':142,
      'init_state_index':0,'vla_budget_per_case':{name:case['budget_per_subgoal'] for name,case in CASES.items()},'session_limit':8000,'wall_deadline_s':1800,
      'checker_views':['wrist','agentview','sideview'],'checker_interval':150,'independent_streak':5,'source_sha256':hashes,'criteria':success,'stop_rule':stop}
    proposal.parent.mkdir(parents=True,exist_ok=True)
    dump(proposal,{'schema_version':1,'id':'20261013-xvla-all-native','intent':'new_experiment',
       'question_key':'xvla-all-native','purpose':PURPOSE,'prior_evidence':list(HISTORY),
       'history_review':[{'experiment_id':k,'disposition':'reuse','reason':v} for k,v in HISTORY.items()],
       'related_search':'主Agent已rg X-VLA、service、finish_v1、原生任务与指令检查；已审查旧持久试点、输入审计及grounding证据，本轮变化仅待实测。',
       'hypothesis':'多因素集成试点可实现原生下一子任务起始状态交接，同时保留前序成果；不作单因素因果或可靠性统计。',
       'information_gain':'旧原子试点不能回答原生长程融合与三同步视图逐项检查的集成效果。',
       'changed_factors':[{'name':n,'before':b,'after':a,'reason':'本轮集成条件变化，不能分开归因。'} for n,b,a in [
         ('native_coverage','3个原生技能','10个原生技能，新增7场景及13候选原子'),
         ('native_budget','三原生case每技能800步','十原生case每完整技能900步，wine原子仍800步'),
         ('new_scenes_and_recipes','旧8scene与原13能力','新增7native场景与13task-scoped候选原子，维持目标只检查不自动点火')]],
       'fixed_factors':{'source_sha256':{k:v for k,v in hashes.items() if any(x in k for x in ('finish_v1/control.py','joint_home.py','safe_exit.py','safe_exit_contacts.py'))},
         'reference_fk':'unchanged native reference FK and original benchmark initial states',
         'controller_algorithms':'unchanged control.run_finish, retreat/home and protection predicates',
         'joint_step_rad':.01,'prepare_limit':360,'checker_interval':150,'independent_streak':5,
         'feedback_retreat':['wine_to_rack','stove_on'],'single_session_preserves_objects':True},
       'success_criteria':'; '.join(success),'failure_criteria':'; '.join(['应触发未触发：触发链路失败、目标效果未验证。','已触发但动作未执行：执行失败。','执行后独立判据不通过：效果失败。','未启动：不计VLA效果分母。']),
       'unknown_criteria':'模型用量不可用时如实标unknown；同步或证据不足不宣称成功。',
       'scope':{'max_cases':len(CASES),'max_vla_actions_per_case':2400,'max_helper_actions_per_case':3000,'stop_rule':stop},
       'new_physical_actions':len(CASES)*(2400+3000),'configuration':config,'configuration_complete':True,
       'repeat':{'needed':False,'reason':'复用旧原生与收尾；本轮新原生技能/三视图集成条件未实测。'}})
    return {'ok':True,'proposal':str(proposal),'source_manifest':str(output/'source_manifest.json')}

def verify_frozen(output, proposal):
    value=json.loads(proposal.read_text(encoding='utf-8'))
    hashes=value['configuration']['source_sha256']
    for relative, expected in hashes.items():
        if sha(ROOT/relative)!=expected or sha(output/'source_snapshot'/relative)!=expected:
            raise RuntimeError('frozen source mismatch: '+relative)
    return value

def verify_case_prerequisite(name: str, output: Path, proposal_value: dict) -> None:
    if name == 'wine_bowl_stove':
        return
    prerequisite = (proposal_value.get('configuration') or {}).get('independent_case_prerequisite')
    if prerequisite is None:
        first = json.loads((output/'wine_bowl_stove/result.json').read_text(encoding='utf-8'))
        if first.get('complete') is not True:
            raise RuntimeError('first_chain_not_complete')
        return
    try:
        relative = Path(prerequisite['path'])
        if relative.is_absolute():
            raise ValueError('absolute prerequisite path')
        source = (ROOT/relative).resolve()
        source.relative_to(ROOT.resolve())
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != prerequisite['sha256']:
            raise ValueError('prerequisite hash mismatch')
        recorded = json.loads(raw)
        from scene_demo.xvla_service import handoff_evidence
        checks = handoff_evidence(recorded)['checks']
        if (not all(value is True for key, value in checks.items() if key != 'finish_terminal')
                or recorded['outcome']['ok'] is not False
                or recorded['job']['ended_reason'] != 'final visual failed'
                or (recorded.get('finish_status') or {}).get('terminal', {}).get('ok') is not False):
            raise ValueError('prerequisite evidence failed')
    except Exception as exc:
        raise RuntimeError('independent_case_prerequisite_failed') from exc


def run_case(name, output, proposal):
    value = verify_frozen(output,proposal)
    verify_case_prerequisite(name, output, value)
    directory=output/name
    directory.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(proposal)]
    check=subprocess.run(command,capture_output=True)
    (directory/'preflight.stdout').write_bytes(check.stdout)
    (directory/'preflight.stderr').write_bytes(check.stderr)
    dump(directory/'preflight.json',{'command':command,'returncode':check.returncode})
    if check.returncode:
        dump(directory/'result.json',{'complete':False,'reason':'preflight_failed','physical_actions':0})
        return 2
    spec=CASES[name]
    budget = str(spec['budget_per_subgoal'])
    os.environ.update(SCENE_SERVICE_URL='http://127.0.0.1:8769',SCENE_BUDGET_PER_SUBGOAL=budget,
        SCENE_EXPECTED_WORKFLOW='persistent_scene_xvla_v3',SCENE_EXPECTED_MODEL_REVISION=REVISION)
    import yaml
    from scene_demo import setup_profile
    home=Path('/home/yhwang/fyp/xvla_native/hermes_home')
    home.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(setup_profile.SRC_ENV,home/'.env')
    (home/'.env').chmod(0o600)
    profile=yaml.safe_load(setup_profile.CONFIG_YAML)
    profile['mcp_servers']['scene_tools']['env'].update(SCENE_SERVICE_URL=os.environ['SCENE_SERVICE_URL'],SCENE_BUDGET_PER_SUBGOAL=budget)
    (home/'config.yaml').write_text(yaml.safe_dump(profile,allow_unicode=True),encoding='utf-8')
    (home/'SOUL.md').write_text(setup_profile.SOUL_MD,encoding='utf-8')
    from scene_demo.xvla_service import XvlaSceneService
    from scene_demo import service
    from scene_demo import run_agent
    run_agent.SERVICE_BASE=os.environ['SCENE_SERVICE_URL']
    run_agent.EXPECTED_WORKFLOW=os.environ['SCENE_EXPECTED_WORKFLOW']
    run_agent.EXPECTED_MODEL_REVISION=REVISION
    svc=XvlaSceneService(str(directory/'runs'),expected_capabilities=spec['capabilities'])
    httpd=None
    try:
        svc.start()
        deadline=time.monotonic()+1800
        startup_deadline=time.monotonic()+60
        while not svc.health().get('ready'):
            if time.monotonic()>=startup_deadline or svc.health().get('worker_error'):
                raise RuntimeError('service_not_ready')
            time.sleep(.25)
        httpd=service._Server(('127.0.0.1',8769),service._Handler)
        httpd.service=svc
        thread=threading.Thread(target=httpd.serve_forever,daemon=True)
        thread.start()
        created=svc.create_session(spec['scene'],seed=142,init_state_index=0)
        if created.get('ok') is False or created.get('state') != 'ready':
            raise RuntimeError('session_create_failed: '+json.dumps(created))
        session_id=created['session']['session_id'] if 'session' in created else created['session_id']
        request_id='xvla-'+name
        svc.bind_request_context(request_id,session_id,spec['request'])
        config=SimpleNamespace(session_id=session_id,request=spec['request'],case_id=None,
            request_id=request_id,timeout=max(1,int(deadline-time.monotonic())),max_repairs=0,cancel_file=None)
        (directory/'agent').mkdir()
        agent=run_agent.Runner(config,run_agent.ServiceClient(),run_agent.HermesRunner(home=str(home),cwd=str(HERE)),
            run_agent.SystemClock(),str(directory/'agent')).run()
        dump(directory/'agent/agent_result.json',agent)
        plan=agent.get('plan') or {}
        pilots=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((directory/'runs').rglob('pilot_result.json'))]
        route = plan.get('native_skill_route') or {}
        requested = route.get('requested_capability_ids', plan.get('capability_ids'))
        executed = route.get('executed_capability_ids', plan.get('capability_ids'))
        replacements = route.get('replacements', [])
        hermes_matching = requested == spec['capabilities']
        matching = executed == spec['capabilities'] and plan.get('capability_ids') == executed
        complete=(plan.get('state')=='completed' and plan.get('plan_success') is True and matching
            and len(pilots)==len(spec['capabilities']) and all(p['job'].get('success') is True
            and (p.get('finish_status') or {}).get('independent_streak',0)>=5
            and (p.get('handoff_evidence') or {}).get('ok') is True for p in pilots))
        dump(directory/'result.json',{'complete':complete,'agent_result':agent,'plan':plan,'jobs':pilots,
            'summary':{'expected_capabilities':spec['capabilities'],'hermes_plan_matches':hermes_matching,
                'requested_plan':requested,'executed_plan':executed,'native_routing_replacements':replacements,
                'effective_plan_matches':matching,
                'model_usage':'See original xvla_calls and Hermes usage logs; unavailable usage remains unknown.'}})
        return 0 if complete else 2
    except Exception as exc:
        dump(directory/'result.json',{'complete':False,'reason':'program_error','error':str(exc)})
        raise
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        svc.stop()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--case',choices=list(CASES))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--proposal',type=Path,default=DEFAULT_PROPOSAL)
    args=parser.parse_args(argv)
    if args.prepare:
        print(json.dumps(prepare(args.output,args.proposal),ensure_ascii=False))
        return 0
    return run_case(args.case,args.output,args.proposal)

if __name__=='__main__':
    raise SystemExit(main())
