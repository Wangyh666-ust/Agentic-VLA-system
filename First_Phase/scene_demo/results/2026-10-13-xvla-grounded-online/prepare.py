"""Freeze the authorized independent online comparison; no model or motion."""
import hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT/'First_Phase/scene_demo'
OUT = Path(__file__).resolve().parent
PROPOSAL = HERE/'plans/20261013-xvla-grounded-online.json'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not PROPOSAL.exists(), 'do not overwrite frozen proposal'
old = read(HERE/'plans/20261013-xvla-timeline-trial.json')
registry = read(HERE/'experiment_registry.json')['entries']
prior = [r for r in registry if 'xvla-process-evidence' in r['question_keys'] or r['id'] in ('20261011-container-view','20261012-xvla-native-benchmark','20261012-xvla-persistent-pilot')]
sources = [ROOT/p for p in old['configuration']['source_sha256']]
sources += [HERE/p for p in ('xvla_grounding.py','xvla_subtask_review.py','xvla_subtask_evidence.py','xvla_event_checker.py','xvla_evidence_memory.py','tests/test_xvla_grounding.py','tests/test_xvla_subtask_review.py')]
hashes = {}
for p in dict.fromkeys(sources):
    relative=p.relative_to(ROOT).as_posix(); target=OUT/'source_snapshot'/relative
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);hashes[relative]=sha(p)
dump(OUT/'source_manifest.json',hashes)
config = {k:v for k,v in old['configuration'].items() if k not in ('source_sha256','case_order','case_prerequisites','checker','success_criteria','stop_rule','checker_protocol')}
config.update(source_sha256=hashes, online_grounding=True, online_cases=['microwave_mug','basket_two'],
    checker_protocol='xvla_subtask_review_v7_1_with_planning_inventory',
    inventory_calls=2, selected_target_review_calls=1, checker_max_calls=24,
    checker_views=['wrist','agentview','sideview'], checker_sizes=[[256,256],[512,512],[512,512]],
    camera_rig={'relative_to':'static arena workspace_offset','eyes':[[.8,-.65,.85],[.8,.65,.85]],'target':[0,0,.1],'fovy':55},
    policy_camera_change=False, subagents_used=0, root_model='unknown',root_effort='unknown')
stop='每项一次独立在线执行，无重试、辅助抓取或搬运。三视角目标确认/规划绑定未触发或失败时该项零VLA停止，不用原生真值强制放行；首项失败仍测独立第二项。过程中身份/图像/反馈/程序异常按原停止规则停止该项；已有原图全部保留。未自然发生历史纠错则该机制在线效果未验证，不用下游结果代替。'
proposal={'schema_version':1,'id':'20261013-xvla-grounded-online','intent':'new_experiment','question_key':'xvla-process-evidence',
 'purpose':'验证执行前三视角现场物体清单经原图复核、Hermes目标绑定及末端同身份过程检查，改正无效检查机位后，入炉关门与双罐入篮能否独立完成在线全流程；防止初始错认持续沿用，保留原收尾/退离/下一任务完整复位。',
 'prior_evidence':[r['id'] for r in prior],
 'history_review':[{'experiment_id':r['id'],'disposition':'reuse','reason':'复用已保存输入、原生执行、机位或证据恢复结果；既有错误定位及True但依据错误不能证明本轮新三视角初态规划绑定有效。'} for r in prior],
 'hypothesis':'有效双外部取景与共享初态物体身份可改善视觉定位与过程图复核；多因素集成，不能分开归因。',
 'information_gain':'旧离线试验无法回答三视角识别先于规划并共享到在线检查的效果，本轮新策略执行，非动作回放。',
 'changed_factors':[{'name':'planning_identity','before':'Hermes单初图与检查器独立身份','after':'三视图清单初识别/复核，Hermes绑定，执行前审核，检查器共享原图身份并可修订','reason':'消除目标语义到实物对应断点'},
 {'name':'checker_cameras','before':'native agentview256与背面sideview512','after':'静态工作台校准的两侧斜视512，暂时渲染后恢复相机','reason':'让目标/目的地与门状态进入有效视野'}],
 'fixed_factors':{'weights':config['weights_sha256'],'seed':142,'init_state_index':0,'policy_images':'native agentview/wrist unchanged','VLA_instruction':'original registered skill','finish_retreat_reset':'original algorithms/parameters unchanged; only runtime checker image selection prefers checker_frames','native_priority':True,'assistant':False,'max_repairs':0},
 'success_criteria':'触发：每项执行前真实三图识别、复核、Hermes非空目标/目的地绑定及选择复核完成，旧错认可被明确纠正。执行：原图SHA与采样时间保存并实际进入检查器，全部松手事件保留及后续可重审。效果：原生AND目标与视觉目标分别统计；图中目标身份/关系确实正确，完整流程complete、原收尾/退离/归位真实执行及独立五连通过。历史补证以实际旧capture引用计数；未触发不声称通过。',
 'failure_criteria':'初态目标确认或传递失败为目标分支未验证；有VLA动作但未达目标为任务效果失败；True但图片/身份/引用错误为视觉依据失败；下游未触发只记未执行。',
 'unknown_criteria':'不可见且无可信历史证据为unknown；未发生自然身份纠正或遮挡则对应机制在线未验证；仅两个样本不能推出可靠成功率。',
 'scope':{'max_cases':2,'max_vla_actions_per_case':900,'max_helper_actions_per_case':3000,'stop_rule':stop},
 'new_physical_actions':7800,'configuration':config,'configuration_complete':True,
 'repeat':{'needed':False,'reason':'明确改变规划共享身份及检查取景，用户授权修复后两项独立在线测试。'}}
dump(PROPOSAL,proposal)
dump(OUT/'software_validation.json',{'command':'WSL sim_venv python -m pytest test_xvla_grounding.py test_xvla_subtask_review.py test_xvla_integration.py test_xvla_pipeline.py test_service_contract.py finish_v1/tests/test_runtime.py -q', 'result':'284 passed, 4 subtests passed in 24.13s','camera_probe':'both cases zero-step rendering; policy agentview pixels unchanged','limitations':'software/source validation is not model visual quality'})
print(PROPOSAL)
