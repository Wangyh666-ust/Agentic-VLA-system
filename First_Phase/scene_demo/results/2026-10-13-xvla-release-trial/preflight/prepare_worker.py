from pathlib import Path
import hashlib,json,subprocess,shutil
ROOT=Path('D:/FYP');HERE=ROOT/'First_Phase/scene_demo';OUT=HERE/'results/2026-10-13-xvla-release-trial';PRE=OUT/'preflight';PROPOSAL=HERE/'plans/20261013-xvla-release-trial.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
registry=HERE/'experiment_registry.json';index=HERE/'EXPERIMENT_INDEX.md'
(PRE/'registry.before.json').write_bytes(registry.read_bytes());(PRE/'index.before.md').write_bytes(index.read_bytes())
def evidence_check(name):
 data=json.loads(registry.read_text(encoding='utf-8'));rows=[]
 for e in data['entries']:
  for item in e['evidence']:
   actual=sha(ROOT/item['path']);rows.append({'experiment':e['id'],'path':item['path'],'expected':item['sha256'],'actual':actual,'matches':actual==item['sha256']})
 value={'entries':len(data['entries']),'evidence':len(rows),'all_match':all(r['matches'] for r in rows),'rows':rows};(PRE/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');print(name,value['entries'],value['evidence'],value['all_match'],flush=True);return value
before=evidence_check('evidence.before');commands={}
def run(name,argv):
 (PRE/(name+'.argv.json')).write_text(json.dumps(argv,ensure_ascii=False,indent=2),encoding='utf-8');r=subprocess.run(argv,cwd=ROOT,capture_output=True)
 (PRE/(name+'.stdout')).write_bytes(r.stdout);(PRE/(name+'.stderr')).write_bytes(r.stderr);(PRE/(name+'.exitcode')).write_text(str(r.returncode)+'\n');commands[name]={'argv':str((PRE/(name+'.argv.json')).relative_to(ROOT)),'stdout':str((PRE/(name+'.stdout')).relative_to(ROOT)),'stderr':str((PRE/(name+'.stderr')).relative_to(ROOT)),'exitcode':r.returncode};print(name,r.returncode,r.stdout[-300:].decode('utf-8',errors='replace'),flush=True);return r
review=['D:/miniconda/python.exe','.agents/skills/fyp-experiment-review/scripts/review.py']
run('registry_validate_before',review+['validate'])
prefix=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream','LIBERO_CONFIG_PATH=/home/yhwang/fyp/xvla_native/libero_config','MUJOCO_GL=egl','PYOPENGL_PLATFORM=egl','LD_LIBRARY_PATH=/usr/lib/wsl/lib','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','/home/yhwang/fyp/xvla_native/sim_venv/bin/python']
env_result=run('environment',prefix+['First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/preflight/environment_probe.py'])
if env_result.returncode:raise SystemExit('read-only environment probe failed; original output saved')
environment=json.loads(env_result.stdout);(PRE/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
for name,python in environment['python'].items():run(name+'_python_version',['wsl.exe','-d','Ubuntu','--exec',python,'--version'])
run('tests',prefix+['-m','pytest','-q','-p','no:cacheprovider','First_Phase/scene_demo/tests/test_xvla_trial_evidence.py','First_Phase/scene_demo/tests/test_xvla_pipeline.py'])
run('help',prefix+['-m','scene_demo.xvla_pilot','--help'])
syntax=[]
for name in ('xvla_trial_evidence.py','xvla_pilot.py','tests/test_xvla_trial_evidence.py'):
 p=HERE/name;compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');syntax.append({'path':name,'passed':True})
(PRE/'compile.json').write_text(json.dumps(syntax,indent=2))
sources=list(dict.fromkeys(environment['repo_sources']+['First_Phase/scene_demo/xvla_trial_evidence.py']))
hashes={};manifest=[]
for relative in sources:
 source=ROOT/relative;target=OUT/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 hashes[relative]=sha(source);manifest.append({'path':relative,'current_sha256':hashes[relative],'snapshot_sha256':sha(target),'matches':hashes[relative]==sha(target)})
(OUT/'source_manifest.json').write_text(json.dumps(hashes,indent=2));(PRE/'freeze.before.json').write_text(json.dumps(manifest,indent=2))
history={
 '20261013-xvla-process-evidence-software':'复用请求字节、松手采样与复用软件资格；尚无本机制真实动作效果。',
 '20261013-xvla-all-native-software':'复用十native目录、原生语言/目标和后端隔离资格，不把软件当物理成功。',
 '20261013-xvla-native-three-view-software':'复用三同期视图接口与完整检查资格，不重做基线。',
 '20261012-xvla-persistent-pilot':'复用旧实际atomic持久试点及收尾/参考证据；本轮改变整体技能与检查输入。',
 '20261012-xvla-input-audit':'复用原生输入审计，保持原policy三输入及绝对动作。',
 '20261012-xvla-checker-grounding':'复用原检查器grounding错误证据，本轮静态身份与同期三图不代表效果已验证。',
 '20261012-xvla-native-benchmark':'复用原生基线与初态，不把原生成功率作为当前pipeline成功率。',
 '20261011-takeover-stove-cue':'复用已验证原收尾控制与保护，不改阈值。',
 '20261009-joint-home':'复用原关节归位算法与独立效果判据，不用home替代本轮触发/复用目标。'}
purpose='实际验证原生黄白杯入微波炉再关门：VLA中途松手三同期图确实采集、入炉证据确实分类、关门后raw遮挡未知经该证据复用通过且完整任务/原收尾归位独立验收；仅此gate通过才运行双罐入篮，不加抓取/运输辅助'
stop='每case仅1次。首例目标失败、应触发未触发/零分类、不同步、输入或transport异常、缺历史实际复用一律停，不启动basket；即便任务完成但无自然rawUnknown→effectiveTrue复用也记复用unverified并停。不得用globalhome替代目标，不现场改prompt/阈值/预算，不自动重试。'
success='触发：至少1个VLA来源实测关爪偏差>=.004后连续两次<=.002开口事件，时刻对应原VLA步与in=true/close=false；执行：3同期wrist256/agentview256/side512原PNG、SHA、context/encoder、HTTP200原request/response及call_index匹配，真实goal_0=true分类进入同event/time ledger；复用：关门后原raw goal_0=null且goal_1=true，经该event有效goal_0=true/goal_1=true、完整视觉AND=true；独立效果：该时刻原native_success=true、完整case complete及handoff=true、全in+close及安全机器人参考独立五连。仅assess_microwave_reuse gate_ok=True允许双罐第二case。'
config={'case_order':['microwave_mug','basket_two'],'case_prerequisites':{'microwave_mug':None,'basket_two':'microwave_mug'},
 'cases':{'microwave_mug':{'scene':'microwave_mug','suite':'libero_10','task_id':9,'expected_capabilities':['xvla_microwave_place_close'],'instruction':'put the yellow and white mug in the microwave and close it','budget_per_subgoal':900},'basket_two':{'scene':'basket_two','suite':'libero_10','task_id':0,'expected_capabilities':['xvla_basket_both'],'instruction':'put both the alphabet soup and the tomato sauce in the basket','budget_per_subgoal':900}},
 'workflow':'persistent_scene_xvla_v3','model_revision':'129e71460678b7236cee6fc9707f09d9fa0c3590','weights_sha256':'260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8','weights_bytes':3519068172,
 'source_xvla':'6bc2513f5f1cbec715cc668b414392a6cae5c671','source_libero':'8f1084e3132a39270c3a13ebe37270a43ece2a01',
 'dtype':'float32','chunk':30,'denoise':10,'domain':3,'control':'absolute','seed':142,'init_state_index':0,
 'model_rng':'unknown/unfixed; not a paired stochastic comparison','vla_budget_per_subgoal':900,'session_limit':8000,'helper_limit_including_native_settle':3000,'settle_steps_included':10,'wall_deadline_s':1800,
 'checker':{'current_calls':10,'release_calls':4,'total_calls':14,'interval_vla_steps':150,'views':['wrist','agentview','sideview'],'image_sizes':[[256,256],[256,256],[512,512]],'release_sampler':{'closed_error_min':.004,'open_error_max':.002,'consecutive_open':2}},
 'independent_streak':5,'grasp_guard':'off','max_repairs':0,'transport_or_grasp_assistance':False,'replay':False,'training':False,
 'source_sha256':hashes,'external_source_sha256':environment['external_sources'],'environment_probe':str((PRE/'environment.json').relative_to(ROOT)),'success_criteria':success,'stop_rule':stop}
fixed={'downstream_source_sha256':{k:v for k,v in hashes.items() if any(k.endswith(x) for x in ('finish_v1/control.py','finish_v1/sensors.py','finish_v1/simple_checker.py','finish_v1/vision.py','finish_v1/feedback_retreat.py','joint_home.py','safe_exit.py','safe_exit_contacts.py','placement_completion.py','finish_v1/runtime.py'))},
 'reference_fk_and_cache_guards':'unchanged native reference FK and complete-skill cache/reference protections','joint_step_rad':.01,'prepare_limit':360,'session_limit':8000,'independent_streak':5,'grasp_guard':'off','no_transport_replay_training':True}
proposal={'schema_version':1,'id':'20261013-xvla-release-trial','intent':'new_experiment','question_key':'xvla-process-evidence','purpose':purpose,'prior_evidence':list(history),'history_review':[{'experiment_id':k,'disposition':'reuse','reason':v} for k,v in history.items()],
 'related_search':'主Agent已读旧process-evidence软件、全十原生与旧实际pilot/input audit/grounding原始证据；无本机制真实松手分类与关门后复用效果记录。',
 'hypothesis':'自然VLA松手同期三图的真实入炉证据可在关门后当前遮挡未知时安全复用，完整原生AND任务及原收尾归位通过；多因素集成不作单因素因果判断。',
 'information_gain':'软件mock/原生基线与旧atomic试点不能证明真实触发采图、视觉分类、关门后实际复用及独立完整任务效果。',
 'changed_factors':[{'name':'native_skill','before':'旧实际atomic拆分','after':'完整原生入炉关门/双罐技能各一次调用','reason':'匹配原生完整语言与终态，不证明内部动作顺序。'},{'name':'visual_grounding','before':'旧simple单external检查','after':'wrist/agentview/sideview三同期图与静态身份','reason':'视角与提示变化不可独立归因。'},{'name':'release_evidence','before':'仅当前图','after':'真实松手采样分类ledger与当前raw未知复用','reason':'直接验证本轮预期事件和实际复用，不强迫模型返回unknown。'}],
 'fixed_factors':fixed,'success_criteria':success,'failure_criteria':'触发链路失败/零分类：目标未验证并停；真实分类或复用执行失败：执行失败并停；独立完整任务失败：效果失败并停；不把home或零动作当目标通过。','unknown_criteria':'实际任务完成但无自然复用、视觉身份/关系未知、输入不同步或缺原请求响应证据均unverified并停止后续；实际随机流未知且nonpaired。',
 'scope':{'max_cases':2,'max_vla_actions_per_case':900,'max_helper_actions_per_case':3000,'stop_rule':stop},'new_physical_actions':7800,'configuration':config,'configuration_complete':True,'repeat':{'needed':False,'reason':'旧实验不包含本轮真实VLA中途松手分类与关门后原始unknown复用，不重做旧基线。'}}
assert not PROPOSAL.exists(),'refuse overwrite existing proposal';PROPOSAL.write_text(json.dumps(proposal,ensure_ascii=False,indent=2),encoding='utf-8')
run('proposal_check',review+['check','--proposal',str(PROPOSAL)])
freeze_after=[{'path':relative,'expected':digest,'current':sha(ROOT/relative),'snapshot':sha(OUT/'source_snapshot'/relative),'matches':sha(ROOT/relative)==sha(OUT/'source_snapshot'/relative)==digest} for relative,digest in hashes.items()]
(PRE/'freeze.after.json').write_text(json.dumps(freeze_after,indent=2))
after=evidence_check('evidence.after');run('registry_validate_after',review+['validate'])
case_commands={case:prefix+['-u','-m','scene_demo.xvla_pilot','--case',case,'--output','/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-release-trial','--proposal','/mnt/d/FYP/First_Phase/scene_demo/plans/20261013-xvla-release-trial.json'] for case in ('microwave_mug','basket_two')}
server_argv=['/home/yhwang/fyp/xvla_native/venv/bin/python','-u','-m','deploy','--model_path','/home/yhwang/fyp/xvla_native/model','--port','8794','--host','127.0.0.1','--disable_slurm','--output_dir','/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/runtime/server_output']
server_env={'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/home/yhwang/fyp/xvla_native/upstream','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','TOKENIZERS_PARALLELISM':'false','HF_HOME':'/home/yhwang/fyp/xvla_native/cache/huggingface','XDG_CACHE_HOME':'/home/yhwang/fyp/xvla_native/cache'}
model_ok=environment['model']['sha256']==config['weights_sha256'] and environment['model']['bytes']==config['weights_bytes']
registry_same=registry.read_bytes()==(PRE/'registry.before.json').read_bytes();index_same=index.read_bytes()==(PRE/'index.before.md').read_bytes()
ready={'preflight_software_passed':all(c['exitcode']==0 for c in commands.values()),'commands':commands,'proposal':str(PROPOSAL.relative_to(ROOT)),
 'source_frozen':all(r['matches'] for r in freeze_after),'external_source_frozen':all(e['sha256']==e['snapshot_sha256'] for e in environment['external_sources']),
 'registry_entries':after['entries'],'evidence_count':after['evidence'],'all_old_sha_match':after['all_match'],'registry_bytes_unchanged':registry_same,'index_bytes_unchanged':index_same,
 'model_sha_pin_matches':model_ok,'environment':environment,'case_order':['microwave_mug','basket_two'],'basket_gate':'actual assess_microwave_reuse gate_ok=True only',
 'future_case_argv':case_commands,'future_server':{'argv':server_argv,'cwd':'/home/yhwang/fyp/xvla_native/upstream','env':server_env,'readiness':'GET /openapi.json; schema POST /act; no request performed'},
 'implementation_executor':{'model':'gpt-6.1-sol','effort':'low','source':'existing explicitly spawned executor thread'},'root_model':None,'root_effort':None,
 'new_VLA_Qwen_Hermes_calls':0,'environment_actions':0,'physical_effect':'unknown; not started','launch_performed':False,
 'unresolved':['Existing occupied server identity needs main-agent judgment before using it'] if any(p['listening'] for p in environment['ports']) else []}
ready['prepared_for_main_review']=ready['preflight_software_passed'] and ready['source_frozen'] and ready['external_source_frozen'] and after['all_match'] and model_ok and registry_same and index_same
(OUT/'ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8');print('READY',ready['prepared_for_main_review'],'modelpin',model_ok,'ports',[(p['port'],p['listening']) for p in environment['ports']],flush=True)
