from pathlib import Path
from collections import Counter
import hashlib,json,math,time,traceback
CODE=Path(r'D:\FYP\First_Phase\scene_demo');ROOT=CODE/'results/2026-10-09-subtask-assist-repair';PILOT=ROOT/'pilot';EVIDENCE=ROOT/'evidence'
OUTPUT=EVIDENCE/'trial_metrics.json';PROOF=EVIDENCE/'metric_extraction_proof.json';CASES=['case_02_enabled','case_03_enabled'];BOWL='akita_black_bowl_1';WINE_KEY='on|wine_bottle_1|wine_rack_1_top_region';NAMES=['subtask_context.py','subtask_geometry.py','subtask_preparation.py','subtask_assist_service.py','test_subtask_assist.py','test_subtask_prepare_repair.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def get(d,*keys):
 for k in keys:
  if not isinstance(d,dict):return None
  d=d.get(k)
 return d
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def vec(v):return [float(x) for x in v] if isinstance(v,(list,tuple)) and len(v)==3 and all(finite(x) for x in v) else None
def jsonl(p):
 raw=p.read_bytes();lines=raw.decode('utf-8').splitlines(keepends=True);rows=[]
 for i,line in enumerate(lines,1):
  if not line.endswith(('\n','\r')):raise ValueError(str(p)+': incomplete final line '+str(i))
  if not line.strip():raise ValueError(str(p)+': blank JSONL row '+str(i))
  row=json.loads(line)
  if not isinstance(row,dict):raise ValueError(str(p)+': non-object row '+str(i))
  rows.append(row)
 return rows
def point(row):
 snapshot=row.get('snapshot');eef=vec(get(snapshot,'eef_position'));bowl=vec(get(snapshot,'objects',BOWL,'position'))
 if eef is None or bowl is None:return {'step':row.get('step'),'eef_position':eef,'bowl_position':bowl,'dxy_m':None,'d3_m':None,'dz_m':None}
 delta=[eef[i]-bowl[i] for i in range(3)];return {'step':row.get('step'),'eef_position':eef,'bowl_position':bowl,'dxy_m':math.hypot(*delta[:2]),'d3_m':math.sqrt(sum(x*x for x in delta)),'dz_m':delta[2]}
def minimum(rows,key):
 known=[r for r in rows if finite(r.get(key))]
 return min(known,key=lambda r:r[key]) if known else None
def stepmap(rows):
 ids=[r.get('step') for r in rows];valid=all(isinstance(i,int) and not isinstance(i,bool) and i>0 for i in ids);counts=Counter(i for i in ids if isinstance(i,int) and not isinstance(i,bool));dups=sorted(i for i,n in counts.items() if n>1)
 return {r['step']:r for r in rows if isinstance(r.get('step'),int) and not isinstance(r.get('step'),bool) and counts[r['step']]==1},ids,valid,dups
assert not OUTPUT.exists() and not PROOF.exists(),'metric targets already exist'
source_before={n:sha(CODE/n) for n in NAMES};start=time.monotonic();polls=0;report=None
while time.monotonic()-start<=900:
 polls+=1
 if (PILOT/'report.json').is_file():
  try:report=read(PILOT/'report.json')
  except json.JSONDecodeError:report=None
  if isinstance(report,dict) and report.get('executed_trials')==2 and 'close_env' in report:break
 time.sleep(10)
else:
 print(json.dumps({'status':'METRIC_EXTRACTION_WAIT_DEADLINE','executed_trials':get(report,'executed_trials'),'close_env_present':isinstance(report,dict) and 'close_env'in report,'polls':polls}));raise SystemExit(2)
inputs={};records=[]
def record_input(p):inputs[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
record_input(PILOT/'report.json')
try:
 for name in CASES:
  directory=PILOT/name;paths={'trial':directory/'trial.json','assist':directory/'job/subtask_assist.json','sources':directory/'job/action_sources.jsonl','telemetry':directory/'job/telemetry.jsonl'}
  for p in paths.values():record_input(p)
  trial=read(paths['trial']);assist=read(paths['assist']);sources=jsonl(paths['sources']);telemetry=jsonl(paths['telemetry']);job=trial.get('job') if isinstance(trial.get('job'),dict) else {};prefix=trial.get('prefix') if isinstance(trial.get('prefix'),dict) else {}
  source_map,sids,svalid,sdups=stepmap(sources);tele_map,tids,tvalid,tdups=stepmap(telemetry);missing_source=sorted(set(tele_map)-set(source_map));missing_tele=sorted(set(source_map)-set(tele_map));alignment=svalid and tvalid and not sdups and not tdups and not missing_source and not missing_tele
  counter=Counter(r.get('source') if isinstance(r.get('source'),str) else 'unknown' for r in sources);counts={kind:counter.get(kind,0) for kind in ['prepare','local_grasp','vla','unknown']};counts.update({k:v for k,v in counter.items() if k not in counts})
  assist_counts={kind:assist.get(field) for kind,field in [('prepare','prepare_source_actions'),('local_grasp','local_grasp_source_actions'),('vla','vla_source_actions')]};count_match=all(isinstance(assist_counts[k],int) and not isinstance(assist_counts[k],bool) and assist_counts[k]==counts[k] for k in assist_counts) and counts['unknown']==0 and sum(counts.values())==assist.get('actual_total_steps')
  points=[point(r) for r in telemetry];vla_rows=[r for r in telemetry if r.get('step') in source_map and r.get('step') in tele_map and source_map[r['step']].get('source')=='vla'];vla_points=[point(r) for r in vla_rows]
  baseline=vec(get(prefix,'before_snapshot','objects',BOWL,'position'));initial_z=baseline[2] if baseline is not None else None
  grasp_counts=Counter('true' if get(r,'snapshot','objects',BOWL,'grasped') is True else 'false' if get(r,'snapshot','objects',BOWL,'grasped') is False else 'unknown' for r in telemetry)
  vla_grasp=Counter('true' if get(r,'snapshot','objects',BOWL,'grasped') is True else 'false' if get(r,'snapshot','objects',BOWL,'grasped') is False else 'unknown' for r in vla_rows)
  wine_counts=Counter('true' if get(r,'oracle_snapshot','predicates',WINE_KEY) is True else 'false' if get(r,'oracle_snapshot','predicates',WINE_KEY) is False else 'unknown' for r in telemetry)
  raised=0;z_unknown=0;unknown_grasp_rows=0;object_unknown=Counter();held_unknown=0;complete_not_true=0
  for row in telemetry:
   b=vec(get(row,'snapshot','objects',BOWL,'position'))
   if initial_z is None or b is None:z_unknown+=1
   elif b[2]>=initial_z+.03:raised+=1
   objects=get(row,'snapshot','objects');unknown=False
   if not isinstance(objects,dict) or not objects:unknown=True
   else:
    for oid,ob in objects.items():
     value=get(ob,'grasped')
     if value is not True and value is not False:unknown=True;object_unknown[oid]+=1
   unknown_grasp_rows+=int(unknown)
   held=get(row,'snapshot','held_objects')
   if not isinstance(held,list) or any(not isinstance(h,str) or not h for h in held):held_unknown+=1
   if get(row,'snapshot','grasp_observation_complete') is not True:complete_not_true+=1
  helper=assist.get('helper_summary') if isinstance(assist.get('helper_summary'),dict) else None
  ready_rows=[r.get('step') for r in sources if r.get('phase_after')=='ready' and isinstance(r.get('step'),int) and not isinstance(r.get('step'),bool)]
  known_jobsteps=job.get('steps');lengths_match=isinstance(known_jobsteps,int) and not isinstance(known_jobsteps,bool) and known_jobsteps==len(sources)==len(telemetry)==sum(counts.values())
  rec={'case':name,'case_index':trial.get('case_index'),'mode':trial.get('mode'),'job':{k:job.get(k) for k in ['job_id','success','state','steps','ended_reason','error']},'prefix_handoff_sha':prefix.get('handoff_sha'),'actual_before_sha':trial.get('before_sha'),'job_state_before_sha':job.get('state_before_sha'),'prefix_before_snapshot_bowl_position':baseline,'initial_bowl_z_reference_m':initial_z,'source_counts':counts,'assist_source_counts':assist_counts,'source_counts_assist_match':count_match,'source_telemetry_alignment':alignment,'missing_source_step_ids':missing_source,'missing_telemetry_step_ids':missing_tele,'source_duplicate_step_ids':sdups,'telemetry_duplicate_step_ids':tdups,'source_invalid_step_count':sum(not isinstance(i,int) or isinstance(i,bool) or i<=0 for i in sids),'telemetry_invalid_step_count':sum(not isinstance(i,int) or isinstance(i,bool) or i<=0 for i in tids),'source_step_sequence':sids,'telemetry_step_sequence':tids,'step_count':len(sources),'telemetry_count':len(telemetry),'job_steps_sources_telemetry_match':lengths_match,'prepare_confirmed':assist.get('prepare_confirmed'),'helper':{k:get(helper,k) for k in ['phase','reason','current_stage','stage_counts','last_errors']},'plan_high_z_m':get(helper,'plan','high_z'),'plan_orientation_policy':get(helper,'plan','orientation_policy'),'plan_reorientation_required':get(helper,'plan','reorientation_required'),'plan_clearance_extent_m':get(helper,'plan','clearance_extent'),'first_prepare_ready_step':min(ready_rows) if ready_rows else None,'score':trial.get('score'),'first_post_action':points[0] if points else None,'last_post_action':points[-1] if points else None,'minimum_dxy_all':minimum(points,'dxy_m'),'minimum_d3_all':minimum(points,'d3_m'),'minimum_dxy_vla_aligned':minimum(vla_points,'dxy_m'),'minimum_d3_vla_aligned':minimum(vla_points,'d3_m'),'vla_aligned_samples':len(vla_rows),'distance_unknown_samples':sum(p['d3_m'] is None for p in points),'bowl_grasp_samples':{k:grasp_counts.get(k,0) for k in ['true','false','unknown']},'bowl_grasp_samples_vla_aligned':{k:vla_grasp.get(k,0) for k in ['true','false','unknown']},'bowl_raised_03m_samples':raised if initial_z is not None else None,'bowl_z_unknown_samples':z_unknown,'protected_wine_predicate_samples':{k:wine_counts.get(k,0) for k in ['true','false','unknown']},'any_object_grasp_unknown_samples':unknown_grasp_rows,'object_grasp_unknown_counts':dict(object_unknown),'held_objects_unknown_samples':held_unknown,'grasp_observation_complete_not_true_samples':complete_not_true,'input_paths':{k:str(p) for k,p in paths.items()}}
  records.append(rec)
 source_after={n:sha(CODE/n) for n in NAMES};input_after={p:{'bytes':Path(p).stat().st_size,'sha256':sha(Path(p))} for p in inputs};source_same=source_before==source_after;inputs_same=inputs==input_after
 metrics={'experiment':'two_preregistered_subtask_assist_repair_metrics','units':{'distance':'meters','orientation':'radians'},'definitions':{'dxy':'hypot(snapshot.eef_position.x-bowl.x, snapshot.eef_position.y-bowl.y)','d3':'Euclidean norm of EEF-bowl position difference','bowl_raise_reference':'trial.prefix.before_snapshot.objects.akita_black_bowl_1.position[2]','wine_predicate':'oracle_snapshot.predicates.'+WINE_KEY,'source_classification':'Only same-step, individually unique source and telemetry rows; missing/duplicate/invalid step is never filled','grasp_bool':'is True / is False only; every other value unknown','holding':'Missing/non-list/non-string held_objects is unknown, not empty hand','distances_are_not':'Gripper gap, feasibility or task success threshold'},'report_original':{'ok':report.get('ok'),'executed_trials':report.get('executed_trials'),'close_env':report.get('close_env'),'operational_errors':report.get('operational_errors')},'trials':records,'inputs':inputs,'software_source_sha256':source_after,'physical_steps':0,'policy_model_calls':0,'hermes_api_calls':0}
 OUTPUT.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 all_align=all(r['source_telemetry_alignment'] for r in records);all_counts=all(r['source_counts_assist_match'] and r['job_steps_sources_telemetry_match'] for r in records)
 proof={'status':'TRIAL_METRICS_EXTRACTION_PASS' if source_same and inputs_same and all_align and all_counts else 'TRIAL_METRICS_EXTRACTION_DIFFERENCE_PRESERVED','ok':source_same and inputs_same and all_align and all_counts,'cases_count':len(records),'inputs_before':inputs,'inputs_after':input_after,'inputs_unchanged':inputs_same,'software_source_sha256_before':source_before,'software_source_sha256_after':source_after,'sources_unchanged':source_same,'all_source_telemetry_alignment':all_align,'all_counts_match':all_counts,'metric_path':str(OUTPUT),'metric_sha256':sha(OUTPUT),'metric_bytes':OUTPUT.stat().st_size,'command':r'D:\miniconda\python.exe -u D:\FYP\First_Phase\tmp\assist_repair_20261009_extract_metrics.py','wait_polls':polls,'elapsed_s':time.monotonic()-start,'physical_steps':0,'model_calls':0,'api_calls':0}
 PROOF.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':proof['status'],'ok':proof['ok'],'metric_sha256':proof['metric_sha256'],'source_unchanged':source_same,'inputs_unchanged':inputs_same,'trials':[{'case':r['case'],'steps':r['job']['steps'],'success':r['job']['success'],'ended_reason':r['job']['ended_reason'],'counts':r['source_counts'],'dxy_min':get(r,'minimum_dxy_all','dxy_m'),'d3_min':get(r,'minimum_d3_all','d3_m'),'dxy_vla_min':get(r,'minimum_dxy_vla_aligned','dxy_m'),'bowl_grasp_true':r['bowl_grasp_samples']['true'],'bowl_raised':r['bowl_raised_03m_samples'],'wine_false_unknown':[r['protected_wine_predicate_samples']['false'],r['protected_wine_predicate_samples']['unknown']],'helper':r['helper'],'alignment':r['source_telemetry_alignment']} for r in records]},ensure_ascii=True),flush=True)
 raise SystemExit(0 if proof['ok'] else 2)
except Exception:
 traceback.print_exc();raise
