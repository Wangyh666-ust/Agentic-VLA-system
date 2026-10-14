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
DEFAULT_OUTPUT = HERE / 'results/2026-10-12-xvla-persistent-pilot'
DEFAULT_PROPOSAL = HERE / 'plans/20261012-xvla-persistent-pilot.json'
REVISION = '129e71460678b7236cee6fc9707f09d9fa0c3590'
PURPOSE = '验证无抓取辅助的X-VLA在酒瓶→碗→灶台及既有失败多步任务中，经真实Hermes提交、视觉/编码器检查与已验证收尾退离后，恢复下一子任务的完整机器人起始状态并保留前序成果，最后回全局home。'
CASES = {
 'wine_bowl_stove': {'scene': 'goal_table', 'request': '先把酒瓶放到架子上，再把碗放到盘子上，最后打开灶台。', 'capabilities': ['wine_to_rack','bowl_to_plate','stove_on']},
 'basket_two': {'scene': 'basket_two','request': '先把字母汤罐放进篮子，再把番茄酱罐放进篮子。','capabilities':['soup_to_basket','sauce_to_basket']},
 'mugs_two': {'scene':'mugs_two','request':'先把白杯放到左边盘子，再把黄白杯放到右边盘子。','capabilities':['white_mug_left','yellow_mug_right']},
 'microwave_mug': {'scene':'microwave_mug','request':'先把黄白杯放进微波炉，再关闭微波炉门。','capabilities':['mug_to_microwave','microwave_close']},
}
HISTORY = {
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
    sources = [HERE / p for p in ('xvla_service.py','xvla_backend.py','xvla_pilot.py','service.py',
        'run_agent.py','mcp_server.py','setup_profile.py','joint_home.py','safe_exit.py',
        'safe_exit_contacts.py','placement_completion.py','catalog.py')]
    sources += sorted((HERE / 'finish_v1').glob('*.py'))
    bench = HERE / 'results/2026-10-12-xvla-native-benchmark'
    for suite, task in [('libero_goal',9),('libero_goal',8),('libero_goal',7),('libero_10',0),('libero_10',4),('libero_10',9)]:
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
        '真实Hermes提交恰为固定用户请求能力顺序，host不替换计划；max_repairs=0。',
        '目标true时150动作视觉/同步编码器检查分支实际触发，并记录触发事件；预算尾部不以未经检查成功。',
        '记录真实finger_target开口、编码器开口、退离与归位动作数及前后状态；下游零动作不替代验证。',
        '中间归位reference准确来自下一原生初态，关节/开口/速度及末端5mm/0.05rad到位，策略队列清空、实测proprio重建并刷新图像。',
        '所有当前及前序goals原completion gate保留、无持物手接触、独立效果5连通过；末任务回全局home。']
    stop = '首条链任一目标未达、应触发未触发或下游零动作，停止依赖验证和后三case；不用home替代目标。程序错误、不同步或model transport错误停止，无现场改词、重跑或加辅助。'
    config = {'cases':CASES,'model_revision':REVISION,
      'weights_sha256':'260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8',
      'source_xvla':'6bc2513f5f1cbec715cc668b414392a6cae5c671','source_libero':'8f1084e3132a39270c3a13ebe37270a43ece2a01',
      'dtype':'float32','chunk':30,'denoise':10,'domain':3,'control':'absolute','seed':142,
      'init_state_index':0,'vla_budget_per_subgoal':800,'session_limit':8000,'wall_deadline_s':1800,
      'checker_interval':150,'independent_streak':5,'source_sha256':hashes,'criteria':success,'stop_rule':stop}
    proposal.parent.mkdir(parents=True,exist_ok=True)
    dump(proposal,{'schema_version':1,'id':'20261012-xvla-persistent-pilot','intent':'new_experiment',
       'question_key':'xvla-persistent-handoff','purpose':PURPOSE,'prior_evidence':list(HISTORY),
       'history_review':[{'experiment_id':k,'disposition':'reuse','reason':v} for k,v in HISTORY.items()],
       'related_search':'主Agent已rg X-VLA、service、finish_v1、原生任务与指令检查；当前没有此接入的历史实验。',
       'hypothesis':'多因素集成试点可实现原生下一子任务起始状态交接，同时保留前序成果；不作单因素因果或可靠性统计。',
       'information_gain':'原生基线不能回答真实Hermes持久多步执行与下一起始参考交接。',
       'changed_factors':[{'name':n,'before':b,'after':a,'reason':'本轮集成条件变化，不能分开归因。'} for n,b,a in [
         ('policy','SmolVLA BF16 relative旧栈','X-VLA FP32 absolute原生栈'),('grasp','抓辅','无抓辅'),
         ('home','固定全局home','下一原生reference，末尾globalhome'),('planner','旧固定手工顺序','真实Hermes提交用户顺序')]],
       'fixed_factors':{'source_sha256':{k:v for k,v in hashes.items() if any(x in k for x in ('finish_v1','joint_home','safe_exit'))},
         'joint_step_rad':.01,'prepare_limit':360,'checker_interval':150,'independent_streak':5,
         'feedback_retreat':['wine_to_rack','stove_on'],'single_session_preserves_objects':True},
       'success_criteria':'; '.join(success),'failure_criteria':'; '.join(['应触发未触发：触发链路失败、目标效果未验证。','已触发但动作未执行：执行失败。','执行后独立判据不通过：效果失败。','未启动：不计VLA效果分母。']),
       'unknown_criteria':'模型用量不可用时如实标unknown；同步或证据不足不宣称成功。',
       'scope':{'max_cases':4,'max_vla_actions_per_case':2400,'max_helper_actions_per_case':3000,'stop_rule':stop},
       'new_physical_actions':21600,'configuration':config,'configuration_complete':True,
       'repeat':{'needed':False,'reason':'已有原生和收尾实验不包含本轮持久X-VLA/Hermes/下一参考多因素集成条件。'}})
    return {'ok':True,'proposal':str(proposal),'source_manifest':str(output/'source_manifest.json')}

def verify_frozen(output, proposal):
    value=json.loads(proposal.read_text(encoding='utf-8'))
    hashes=value['configuration']['source_sha256']
    for relative, expected in hashes.items():
        if sha(ROOT/relative)!=expected or sha(output/'source_snapshot'/relative)!=expected:
            raise RuntimeError('frozen source mismatch: '+relative)
    return value

def run_case(name, output, proposal):
    verify_frozen(output,proposal)
    if name != 'wine_bowl_stove':
        first=json.loads((output/'wine_bowl_stove/result.json').read_text(encoding='utf-8'))
        if first.get('complete') is not True:
            raise RuntimeError('first_chain_not_complete')
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
    os.environ.update(SCENE_SERVICE_URL='http://127.0.0.1:8769',SCENE_BUDGET_PER_SUBGOAL='800',
        SCENE_EXPECTED_WORKFLOW='persistent_scene_xvla_v1',SCENE_EXPECTED_MODEL_REVISION=REVISION)
    import yaml
    from scene_demo import setup_profile
    home=Path('/home/yhwang/fyp/xvla_native/hermes_home')
    home.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(setup_profile.SRC_ENV,home/'.env')
    (home/'.env').chmod(0o600)
    profile=yaml.safe_load(setup_profile.CONFIG_YAML)
    profile['mcp_servers']['scene_tools']['env'].update(SCENE_SERVICE_URL=os.environ['SCENE_SERVICE_URL'],SCENE_BUDGET_PER_SUBGOAL='800')
    (home/'config.yaml').write_text(yaml.safe_dump(profile,allow_unicode=True),encoding='utf-8')
    (home/'SOUL.md').write_text(setup_profile.SOUL_MD,encoding='utf-8')
    from scene_demo.xvla_service import XvlaSceneService
    from scene_demo import service
    from scene_demo import run_agent
    run_agent.SERVICE_BASE=os.environ['SCENE_SERVICE_URL']
    run_agent.EXPECTED_WORKFLOW=os.environ['SCENE_EXPECTED_WORKFLOW']
    run_agent.EXPECTED_MODEL_REVISION=REVISION
    spec=CASES[name]
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
        matching=plan.get('capability_ids')==spec['capabilities']
        complete=(plan.get('state')=='completed' and plan.get('plan_success') is True and matching
            and len(pilots)==len(spec['capabilities']) and all(p['job'].get('success') is True
            and (p.get('finish_status') or {}).get('independent_streak',0)>=5
            and (p.get('handoff_evidence') or {}).get('ok') is True for p in pilots))
        dump(directory/'result.json',{'complete':complete,'agent_result':agent,'plan':plan,'jobs':pilots,
            'summary':{'expected_capabilities':spec['capabilities'],'hermes_plan_matches':matching,
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
