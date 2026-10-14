"""Prepare the authorized two-case test, without model calls or scene actions.
## TASK
Freeze the new timeline checker and register microwave-first validation.
## FILES
Only this new trial directory and plans/20261013-xvla-timeline-trial.json.
## STEPS
Reuse old launch/probe arguments, verify pins, freeze sources, run review check.
## INTERFACE
Original X-VLA FP32 server and scene_demo.xvla_pilot; same two case names.
## DONE WHEN
Pins and software source hashes match; check exits zero; no scene started.
## REPORT
Prepared status, command exits, unresolved items.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
PREFLIGHT=OUT/'preflight'
OLD=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-release-trial'
QUALIFIED=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-software'
PROPOSAL=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-trial.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def run(name,argv):
    dump(PREFLIGHT/(name+'.argv.json'),argv)
    result=subprocess.run(argv,cwd=ROOT,capture_output=True)
    (PREFLIGHT/(name+'.stdout')).write_bytes(result.stdout);(PREFLIGHT/(name+'.stderr')).write_bytes(result.stderr)
    (PREFLIGHT/(name+'.exitcode')).write_text(str(result.returncode)+'\n')
    if result.returncode:raise RuntimeError(name+' failed: '+str(result.returncode))
    return result

assert not PROPOSAL.exists() and not (OUT/'source_snapshot').exists(),'refuse trial overwrite'
registry=load(ROOT/'First_Phase/scene_demo/experiment_registry.json')
(PREFLIGHT/'registry.before.json').write_bytes((ROOT/'First_Phase/scene_demo/experiment_registry.json').read_bytes())
(PREFLIGHT/'index.before.md').write_bytes((ROOT/'First_Phase/scene_demo/EXPERIMENT_INDEX.md').read_bytes())
manifest=load(QUALIFIED/'source_manifest.json')
assert all(sha(ROOT/p)==value and sha(QUALIFIED/'source_snapshot'/p)==value for p,value in manifest.items())
dump(PREFLIGHT/'qualified_source_check.json',{'all_match':True,'source_count':len(manifest),
    'software_tests':'reused 278 passed / 34 subtests, unchanged sources; no test rerun'})
snapshot=OUT/'source_snapshot';snapshot.mkdir()
for relative in manifest:
    target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
dump(OUT/'source_manifest.json',manifest)
probe=(OLD/'preflight/environment_probe.py').read_text(encoding='utf-8').replace('2026-10-13-xvla-release-trial','2026-10-13-xvla-timeline-trial')
(PREFLIGHT/'environment_probe.py').write_text(probe,encoding='utf-8')
argv=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1',
 'PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream',
 '/home/yhwang/fyp/xvla_native/sim_venv/bin/python','/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/preflight/environment_probe.py']
environment=json.loads(run('environment',argv).stdout)
dump(PREFLIGHT/'environment.json',environment)
old_plan=load(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-release-trial.json')
config=copy.deepcopy(old_plan['configuration'])
assert environment['model']['sha256']==config['weights_sha256'] and environment['model']['bytes']==config['weights_bytes']
assert environment['git']['upstream']['stdout'].strip()==config['source_xvla']
assert environment['git']['LIBERO']['stdout'].strip()==config['source_libero']
assert environment['credentials_source_exists']
previous={record['source']:record['sha256'] for record in old_plan['configuration']['external_source_sha256']}
assert all(record['sha256']==previous[record['source']] for record in environment['external_sources'])
assert not environment['ports'][0]['listening'],'8769 occupied'
config.update(checker_protocol='xvla_timeline_skill_v3',source_sha256=manifest,
    external_source_sha256=environment['external_sources'],
    environment_probe='First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/preflight/environment.json',
    root_model='unknown',root_effort='unknown',subagents_used=0)
config['checker'].update(process_capture_limit=4,previous_check_state_limit=10,max_input_images=15,
    process_input='all retained raw True/False/Null judgements plus original triplets',
    latest_states='current evidence only; side/oblique views valid',
    identity='target identity required; distractors do not prove failure')
success='触发：首例至少1个真实VLA松手事件对应独立in=True/close=False，连续两次标准开口及三同期原图。执行：原HTTP请求确实包含该event的三原PNG、中间原判定/反馈与后来当前三图，SHA/帧时刻/响应引用对应。效果：关门后的联合原回答确认目标身份，入炉True引用该event且current_evidence=False，关门True有当前证据；同刻native_success=True，完整case complete及原handoff=True、独立五连。只有assess_microwave_reuse gate_ok=True才开始双罐入篮，第二case以两目标保持及原收尾/归位完整通过验收。'
stop='每case最多1次，无自动重试/改词/改预算。首例采图或时间线输入未触发、模型/来源/同步/程序异常、完整目标或原交接失败，停止且不启动basket；任务完成但没有自然过程引用样本则该机制未验证并停。不用归位或原生谓词代替视觉/过程验证。'
config.update(success_criteria=success,stop_rule=stop)
prior=list(dict.fromkeys(['20261013-xvla-timeline-software','20261013-xvla-release-trial']+old_plan['prior_evidence']))
assert set(entry['id'] for entry in registry['entries'] if 'xvla-process-evidence' in entry['question_keys'])<=set(prior)
plan=copy.deepcopy(old_plan)
plan.update(id='20261013-xvla-timeline-trial',
    purpose='实际检验新通用身份与时间线检查能否完成入炉关门及双罐入篮；先验首例过程图和中间状态真实入末端、正确引用过程证据并完整收尾，通过才运行第二例。无抓取/搬运辅助器，不使用子agent。',
    prior_evidence=prior,history_review=[{'experiment_id':i,'disposition':'reuse',
        'reason':'复用已有原生/控制/采图资格及实际首例误识别证据；新278测试只是软件，真实效果未验证，不重复旧基线。'} for i in prior],
    related_search='已审查旧实际trial两event/原Qwen回答/native逐步目标与新timeline软件原图9张serializer；篮子旧atomic失败，完整原生+新检查尚未实测。',
    hypothesis='原图与中间状态联合重审、逐目标身份确认可使遮挡后的完整技能正确检查并移交原收尾；单例不能证明可靠性或隔离因果。',
    information_gain='旧首例原生入炉/关门已达标但视觉两次False/False，采图成功而无复用；新软件不能证明真实Qwen正确判断及原收尾。',
    changed_factors=[{'name':'checker','before':'过程照片先单独分类，仅True缓存；末端只有当前三图',
        'after':'通用目标身份门禁；过程原图/中间状态及当前三图联合检查，不自动填充Unknown',
        'reason':'与旧实际trial相比改变检查输入和语义约束，不能分开归因；不改变模型权重/相机/控制。'}],
    success_criteria=success,failure_criteria='区分采样/实际输入未执行、模型判断执行后错误或Unknown、原生目标失败、原收尾未启动/未通过；首例任一未过停第二例。',
    unknown_criteria='历史没有自然引用、目标身份/关系看不清或来源缺失均未验证；root与模型RNG未知、非配对对照；未启动不得记VLA失败。',
    configuration=config,configuration_complete=True,repeat={'needed':False,'reason':'同模型/初态复用但检查输入与身份约束有明确变化，验证新的真实效果，不重复基线。'})
plan['scope']['stop_rule']=stop
dump(PROPOSAL,plan)
run('review_check',[sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(PROPOSAL)])
ready=copy.deepcopy(load(OLD/'ready.json'))
ready={key:ready[key] for key in ('future_server','future_case_argv')}
ready['future_server']['argv'][-1]='/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/runtime/server_output'
for name,argv in ready['future_case_argv'].items():
    ready['future_case_argv'][name]=[arg.replace('2026-10-13-xvla-release-trial','2026-10-13-xvla-timeline-trial').replace('20261013-xvla-release-trial.json','20261013-xvla-timeline-trial.json') for arg in argv]
ready.update(prepared_for_main_review=True,model_sha_pin_match=True,subagents_used=0,
    case_order=['microwave_mug','basket_two'],basket_prerequisite='actual joint process evidence and full handoff gate',
    model_calls=0,environment_actions=0)
dump(OUT/'ready.json',ready)
print(json.dumps({'prepared':True,'model_bytes':environment['model']['bytes'],'pin_matches':True,
    'gpu':environment['gpu']['stdout'],'ports':[{k:p[k] for k in ('port','listening')} for p in environment['ports']],
    'source_count':len(manifest),'model_calls':0,'actions':0},ensure_ascii=False))
