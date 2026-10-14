"""Archive selected original bytes, and derive per-skill audit summary."""
import hashlib,json,shutil
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def local(p):return root/str(p).removeprefix('/mnt/d/FYP/')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=[]
def copy(src,dst):
 dst=out/'evidence'/dst
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 manifest.append({'source':src.relative_to(root).as_posix(),'path':dst.relative_to(root).as_posix(),'sha256':sha(dst)})
 return dst
summary={'baseline_commit':'1be31e32a02760a79d81eebea77c0611c3b7063e','implementation_commit':'4e6337c','mode':'simulator','independent_finish_required':5,'cases':[],'limits':['Privileged simulation information, not a real-world result or mathematical upper bound.','Planning still uses Hermes/Qwen plus independent review. No extra grasp/transport helper.']}
for label,final_name,timing_name in [('bowl','bowl2_final.json','bowl2_valid_timing.json'),('chain','chain_final.json','chain_timing.json')]:
 if not (out/final_name).exists():continue
 state=read(out/final_name);case={'case':label,'timing':read(out/timing_name),'skills':[]}
 for job in state.get('jobs',[]):
  run=local(job['run_dir']);d=read(run/'pilot_result.json');cid=job['capability_id']
  for name in ['pilot_result.json','result.json','events.jsonl','finish_actions.jsonl','independent_finish.jsonl','handoff_reference.json','first.png','last.png','finish/checks.jsonl','finish/simulator_samples.jsonl','finish/process_events.jsonl']:
   src=run/name
   if src.exists():copy(src,Path(label)/cid/name)
  samples=[json.loads(x) for x in (run/'finish/simulator_samples.jsonl').read_text().splitlines()]
  independent=[json.loads(x) for x in (run/'independent_finish.jsonl').read_text().splitlines()] if (run/'independent_finish.jsonl').exists() else []
  cache=(d.get('cache_reset') or [{}])[-1]
  case['skills'].append({'capability':cid,'source_counts':d['source_counts'],'first_all_goals_true_step':next((r['step'] for r in samples if r.get('all_true')),None),'trigger_steps':[r['step'] for r in samples if r.get('trigger')],'finish_status':d.get('finish_status'),'finish_counts':(d.get('finish') or {}).get('counts'),'release_open_actions':(d.get('finish') or {}).get('release_open_actions'),'handoff':d.get('handoff_evidence'),'reference_kind':(d.get('handoff_reference') or {}).get('reference_kind'),'next_capability':read(run/'handoff_reference.json')['next_capability'],'final_native_predicates':independent[-1]['predicates'] if independent else {},'final_metrics':d.get('actual_final_metrics'),'cache':{k:cache.get(k) for k in ['queue_size','proprio_is_none','controller_use_delta','goal_pos_error','goal_ori_error','gripper_hold_error']},'execution_model_request_file_count':len(list((run/'finish/model_requests').glob('*')))})
  planning=run.parent/'planning_grounding'
  for name in ['joint_planning_timing.json','joint_draft.json','joint_prompt.txt','selected_targets_review.json','inventory_confirmed.json','planning_three_views.png']:
   if (planning/name).exists() and not any(x['source']==(planning/name).relative_to(root).as_posix() for x in manifest):copy(planning/name,Path(label)/'planning'/name)
 summary['cases'].append(case)
rejected=root/'First_Phase/scene_demo/results/interactive-console/runs/20261014T152045Z_bowl_drawer_c45459a4/planning_grounding/selected_targets_review.json'
copy(rejected,Path('planning_rejected/selected_targets_review.json'))
summary['prior_attempts']=[{'type':'planning_rejected','vla_task_actions':0,'reason':'User test text black bowl versus visible silver patterned bowl; retained independent rejection.'},{'type':'harness_entry_error','vla_task_actions':0,'model_requests':0,'reason':'bowl2_drawer typo from test harness rename; corrected scene ID without production change.'}]
summary['protected_controls_unchanged']=all(sha(root/'First_Phase/scene_demo'/p)==h for p,h in read(out/'protected_controls.json').items())
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'cases':len(summary['cases']),'files':len(manifest),'protected_controls_unchanged':summary['protected_controls_unchanged']}))
