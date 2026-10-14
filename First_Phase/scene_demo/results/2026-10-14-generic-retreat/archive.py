"""Archive measured outcomes; preserve failures and original source evidence."""
import ast,hashlib,json
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
scene=root/'First_Phase/scene_demo'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def host(path):
    return root/str(path).split('/mnt/d/FYP/',1)[1] if str(path).startswith('/mnt/d/FYP/') else Path(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x] if Path(path).exists() else []
state=read(out/'online_final_state.json')
timing=read(out/'online_timing.json')
console=host(state['run_path'])
run=next((scene/'results/interactive-console/runs').glob('*_'+state['session_id'][:8]))
job=next(run.glob('cap_*'))
pilot=read(job/'pilot_result.json')
finish=pilot['finish'] or {}
native=rows(run/'native/native_steps.jsonl')
events=rows(job/'events.jsonl')
independent=rows(job/'independent_finish.jsonl')
reqdir=job/'finish/model_requests'
requests=sorted(reqdir.glob('*_request.json'))
calls=[]
images=set()
for request in requests:
    data=read(request)
    images.update(host(image['path']) for image in data['images'])
    response=request.with_name(request.name.replace('_request','_response'))
    item={'request':str(request.relative_to(root)),'image_count':len(data['images']),
          'requested_model':data['requested_model'],'http_status':data.get('http_status')}
    if response.exists():
        stat=request.stat()
        start=getattr(stat,'st_birthtime',stat.st_ctime)
        item.update(response_model=read(response).get('model'),
                    wall_seconds_approx=max(0,response.stat().st_mtime-start))
    else:
        transport=request.with_name(request.name.replace('_request','_transport_error'))
        if transport.exists():
            error=read(transport)
            item.update(transport_error=error,wall_seconds_approx=error.get('elapsed_seconds',0))
    calls.append(item)
protected=read(out/'protected_controls.json')
protected_check={name:sha(scene/name)==value for name,value in protected.items()}
assert all(protected_check.values()),protected_check
source=out/'source_snapshot'
for name in ['finish_v1/geometry.py','finish_v1/tests/test_shape_screen.py','finish_v1/tests/test_generic_retreat.py']:
    destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes((scene/name).read_bytes())
executed_source=console/'source_snapshot/finish_v1/geometry.py'
assert ast.dump(ast.parse((scene/'finish_v1/geometry.py').read_text()))==ast.dump(ast.parse(executed_source.read_text()))
vla=[r for r in native if r['source']=='vla']
truth=[r['step'] for r in events if r.get('native_success') is True]
plan=finish.get('plan') or {}
chosen=(plan.get('candidate_reports') or [None])[-1]
summary={'purpose':'修复通用退离接口；在刚才完整关抽屉的例子上确认实际退离通过，再完整重跑记录总耗时。',
 'implementation':{'policy':'legacy_first_then_current_geometry_v1','task_specific_conditions':False,
   'new_auxiliary_grasp_helpers':False,'protected_sources_unchanged':protected_check,
   'current_vs_online_geometry_AST_identical':True,
   'subagents':0,'implementation_agent_model':'not exposed in runtime metadata; no delegated model calls'},
 'software':'348 passed, 53 subtests passed',
 'historical_replay':read(out/'replay_v2/summary.json'),
 'replay_setup_failure':read(out/'replay/summary.json'),
 'online':{'console':str(console.relative_to(root)),'job':str(job.relative_to(root)),
    'stage':state['stage'],'plan_success':state['plan'].get('plan_success'),
    'request_fulfilled':state['plan'].get('request_fulfilled'),'request_scope':state['plan'].get('request_scope'),
    'timing':timing,'planning_timing':read(console/'planning_timing.json'),
    'vla_actions':len(vla),'first_native_success_step':min(truth) if truth else None,
    'last_native_success_step':max(truth) if truth else None,
    'finish_counts':finish.get('counts',{'release':0,'movement':0,'confirm':0,'home':0}),
    'finish_reason':finish.get('reason'),'job_error':pilot['job'].get('error'),
    'home_streak':finish.get('home_streak'),'finish_status':pilot.get('finish_status'),
    'actual_final_metrics':pilot.get('actual_final_metrics'),
    'selected_retreat':chosen,'retreat_replans':finish.get('retreat_replans'),
    'movement_screen_count':len(finish.get('movement_screens',[])),
    'movement_screen_failures':sum(r.get('ok') is not True for r in finish.get('movement_screens',[])),
    'home_entry':finish.get('home_entry'),'home_screen':finish.get('home_screen'),
    'independent_final':independent[-1] if independent else None,
    'model_calls':calls,'visual_wait_seconds_approx':sum(r.get('wall_seconds_approx',0) for r in calls)},
 'limits':['一例完整在线与一例旧动作重放，不代表全部任务成功率；通用实现无任务名称分支。',
   '重放最终视觉未调用，明确unknown；只以真实运动、原净空、独立任务状态及原home判据验退离。',
   '旧首次重放因工具漏还原prepare后的夹爪hold，在第1个历史动作不匹配时停止，未测试退离；修正后完整240步最大编码器误差<1e-9。',
   '总耗时含场景加载和冷模型准备，不含代码修改/重放验证和人为确认等待；轮询误差约2秒。',
   '瓶子归位不在当前场景原生能力库内，计划仍为partial，不以碗任务完成声称整个自然语言需求实现。']}
write(out/'summary.json',summary)
paths=set(out.glob('*.json'))|set(out.glob('*.py'))|set(out.glob('*.stdout'))|set(out.glob('*.stderr'))
paths|=set(source.rglob('*.py'))|set((out/'before').rglob('*.py'))
paths|={out/'replay/summary.json',out/'replay_v2/summary.json',out/'replay_v2/finish_result.json',
         out/'replay_v2/finish_actions.jsonl',out/'replay_v2/old_plan.json',out/'replay_v2/new_plan.json',
         out/'replay_v2/plan_observation.json',out/'replay_v2/before.json',out/'replay_v2/after.json',
         out/'replay_v2/native/native_steps.jsonl',out/'replay_v2/matching.json'}
paths|=set((out/'replay_v2/sensors').glob('*'))
paths|={console/'planning_timing.json',console/'draft.json',console/'run_setup.json',
         job/'pilot_result.json',job/'events.jsonl',job/'finish_actions.jsonl',job/'independent_finish.jsonl',
         job/'handoff_reference.json',job/'finish/checks.jsonl',job/'finish/process_events.jsonl',
         run/'native/native_steps.jsonl',run/'native/native_reset.json'}
paths|=set(reqdir.glob('*.json'))|images
paths={p for p in paths if p.is_file() and p.name not in ('registry_entry.json','registry_validation.stdout','registry_validation.stderr')}
entry={'id':'20261014-generic-retreat','date':'2026-10-14','title':'同刻RGBD通用退离候选与完整碗抽屉在线复测',
 'question_keys':['xvla-drawer-retreat-block','xvla-interactive-latency'],
 'tags':['xvla','retreat','current-geometry','qwen3.8','bowl_drawer'],
 'conditions':['默认退离保留原候选，失败后补充同刻RGBD/机器人几何方向；原收尾控制/深度/净空阈值/home/5/5未改。',
     '一例历史动作重放验证，一例正常交互完整在线；无辅助器，无子agent。'],
 'findings':[f"旧240步重放误差小于1e-9；旧上移失败，新斜向4cm实际退离84步、home135步、5/5，任务全程保持。",
    f"在线状态{state['stage']}；VLA {len(vla)}步，收尾{finish.get('counts')}，结束原因{finish.get('reason')}。",
    f"从场景加载至终态{timing.get('total_seconds',0):.2f}秒；规划{summary['online']['planning_timing']['total_seconds']:.2f}秒；执行{timing.get('execution_seconds',0):.2f}秒。",
    '348项软件测试与53子测试通过；保留首次重放入口不匹配和测试入口缺路径的原失败记录。'],
 'limits':summary['limits'],
 'evidence':[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in sorted(paths)],
 'configuration':read(out/'online_proposal.json')['configuration'],'configuration_complete':True}
write(out/'registry_entry.json',entry)
registry_path=scene/'experiment_registry.json'
registry=read(registry_path)
assert not any(e['id']==entry['id'] for e in registry['entries']),'do not overwrite prior entry'
registry['entries'].append(entry);registry['updated_at']='2026-10-14'
write(registry_path,registry)
index=scene/'EXPERIMENT_INDEX.md'
with index.open('a',encoding='utf-8') as f:
    f.write('\n\n### 20261014-generic-retreat — 通用退离与完整在线复测\n\n')
    f.write('\n'.join('- '+x for x in entry['findings'])+'\n')
    f.write('- 证据：[测量摘要](results/2026-10-14-generic-retreat/summary.json)，[旧动作重放](results/2026-10-14-generic-retreat/replay_v2/summary.json)。仅此样例效果，不代表所有任务成功率。\n')
print(json.dumps({'entry':entry['id'],'evidence_files':len(entry['evidence']),
 'stage':state['stage'],'total_seconds':timing.get('total_seconds'),'finish':finish.get('counts')},ensure_ascii=False))
