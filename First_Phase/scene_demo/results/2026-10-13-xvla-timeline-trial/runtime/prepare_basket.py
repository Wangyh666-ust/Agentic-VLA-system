"""Register the user's independently authorized basket case; no scene actions."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
PARENT=Path(__file__).resolve().parents[1]
OUT=PARENT/'basket_independent'
PROPOSAL=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-basket-independent.json'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not OUT.exists() and not PROPOSAL.exists(),'refuse basket trial overwrite'
OUT.mkdir();runtime=OUT/'runtime';runtime.mkdir();snapshot=OUT/'source_snapshot';snapshot.mkdir()
manifest=load(PARENT/'source_manifest.json')
assert all(sha(ROOT/p)==sha(PARENT/'source_snapshot'/p)==value for p,value in manifest.items())
for relative in manifest:
    target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
dump(OUT/'source_manifest.json',manifest)
plan=copy.deepcopy(load(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-trial.json'))
config=plan['configuration'];config.pop('case_order');config.pop('case_prerequisites')
config['cases']={'basket_two':config['cases']['basket_two']}
# Reuse the existing independent-case eligibility interface. This is historic
# finishing-module qualification, never evidence that the new checker passed.
qualification=load(ROOT/'First_Phase/scene_demo/plans/20261012-xvla-persistent-independent-v5.json')['configuration']['independent_case_prerequisite']
assert sha(ROOT/qualification['path'])==qualification['sha256']
config['independent_case_prerequisite']=qualification
config['independent_scope_authorization']='用户明确选择：两项独立测试，首项失败也测入篮'
config['previous_case_result']='microwave complete=False/gate=False; unchanged, not a prerequisite for this independent comparison'
config['historical_eligibility_note']='旧wine已验证的原收尾/下一参考资格仅满足现有独立case入口；不是本轮时间线或视觉成功判据，不重放旧动作。'
success='真实原生xvla_basket_both一次调用，至少1次实际松手三图/中间状态进入后续当前请求；模型完整两目标True且目标身份已确认；正确引用过程event或当前可见关系依据分别记录。独立两件罐入篮保持、完整case complete/原handoff及五连通过，最后回初始参考且策略状态复位。当前可见而无需历史的成功不能当历史复用效果已验证。'
stop='独立basket只运行1次，900 VLA步、原helper最多3000，不改prompt/阈值/预算，不加抓取/搬运辅助或重试。程序/输入/同步/目标失败保存实际结果后停止，无第三case。'
config.update(success_criteria=success,stop_rule=stop)
plan.update(id='20261013-xvla-timeline-basket-independent',
    purpose='按用户明确选择，将双罐入篮作为独立新流程样例实测；不以入炉失败阻断，不把历史原收尾资格替代本轮过程输入、视觉判断与完整效果。',
    hypothesis='通用时间线与目标身份规则在双罐开放容器任务中可正确判断完整目标并通过原收尾；与入炉不同场景不作单因素因果结论。',
    information_gain='旧篮子atomic流程失败，完整原生双罐加新通用检查尚未实测；该独立任务可分别观察输入执行、视觉判定和完整保持效果。',
    changed_factors=[{'name':'checker','before':'旧atomic篮子试点，单external当前检查',
        'after':'完整原生双罐；三视图过程/当前图及中间状态联合检查、目标身份门禁',
        'reason':'多因素集成实测，与微波炉样例场景不同，不能隔离因果或合并可靠性统计。'}],
    success_criteria=success,failure_criteria='区分采样/输入未执行、模型目标判定失败/Unknown、原生目标失败和原收尾未启动/未通过；只报告实际阶段。',
    unknown_criteria='遮挡/身份/关系或证据缺失标未验证；当前可见不需要过程引用则历史复用效果未验证；RNG未知且非配对。',
    scope={'max_cases':1,'max_vla_actions_per_case':900,'max_helper_actions_per_case':3000,'stop_rule':stop},
    new_physical_actions=3900,repeat={'needed':False,'reason':'完整原生双罐与新检查条件未实测，用户指定独立测试，不重做旧基线。'})
dump(PROPOSAL,plan)
argv=[sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(PROPOSAL)]
dump(runtime/'review_check.argv.json',argv);result=subprocess.run(argv,cwd=ROOT,capture_output=True)
(runtime/'review_check.stdout').write_bytes(result.stdout);(runtime/'review_check.stderr').write_bytes(result.stderr)
(runtime/'review_check.exitcode').write_text(str(result.returncode)+'\n')
assert result.returncode==0,'review refused'
ready=load(PARENT/'ready.json');wrapped=ready['future_case_argv']['basket_two']
wrapped=[arg.replace('/results/2026-10-13-xvla-timeline-trial','/results/2026-10-13-xvla-timeline-trial/basket_independent').replace('20261013-xvla-timeline-trial.json','20261013-xvla-timeline-basket-independent.json') for arg in wrapped]
# Reuse the live verified server argv, including its original log directory.
dump(OUT/'ready.json',{'future_server':ready['future_server'],'future_case_argv':{'basket_two':wrapped},
    'user_independent_authorization':config['independent_scope_authorization'],'model_calls':0,'environment_actions':0})
old=(PARENT/'runtime/run_microwave.py').read_text(encoding='utf-8')
old=old.replace("OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial'", "OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial/basket_independent'")
old=old.replace("CASE=OUT/'microwave_mug'", "CASE=OUT/'basket_two'")
old=old.replace("READY['future_case_argv']['microwave_mug']", "READY['future_case_argv']['basket_two']")
old=old[:old.index('from scene_demo.xvla_trial_evidence import assess_microwave_reuse')]+"\nvalue=json.loads((CASE/'result.json').read_text());final=progress('case_finished');final.update(case_exitcode=case.returncode,complete=value.get('complete',False));write(OUT/'basket_progress.json',final);write(RUNTIME/'server.after.json',{**process(server_record['pid']),'owned':server_record['owned']});write(RUNTIME/'freeze.after.json',[{'path':p,'matches':sha(ROOT/p)==sha(OUT/'source_snapshot'/p)==digest} for p,digest in json.loads((OUT/'source_manifest.json').read_text()).items()]);print('BASKET_COMPLETE '+json.dumps({'complete':value.get('complete',False),'case_exitcode':case.returncode,'server_owned':server_record['owned']}),flush=True)\n"
old=old.replace('microwave_progress.json','basket_progress.json').replace("RUNTIME/'microwave.stdout'","RUNTIME/'basket.stdout'").replace("RUNTIME/'microwave.stderr'","RUNTIME/'basket.stderr'")
old=old.replace("'basket_launch_count':0", "'basket_launch_count':1")
compile(old,'run_basket.py','exec');(runtime/'run_basket.py').write_text(old,encoding='utf-8')
print(json.dumps({'prepared':True,'source_matches_first_trial':True,'new_model_calls':0,'actions':0,'authorization':'independent basket'}))
