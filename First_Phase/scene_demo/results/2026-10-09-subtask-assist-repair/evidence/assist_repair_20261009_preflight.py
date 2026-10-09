from pathlib import Path
import copy,hashlib,inspect,json,math,sys,traceback
import numpy as np
CODE=Path('/mnt/d/FYP/First_Phase/scene_demo');sys.path.insert(0,str(CODE))
import service
import preparation_diagnostics as pd
import subtask_geometry as sg
import subtask_preparation as sp
import subtask_context as sc
import catalog
OUT=Path('/mnt/d/FYP/First_Phase/tmp/assist_repair_20261009_preflight')
PRIOR=Path('/mnt/d/FYP/First_Phase/tmp/subtask_assist_preflight/report.json')
PROFILES=CODE/'subtask_assist_profiles.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,float) and not math.isfinite(v):return None
 if v is None or isinstance(v,(str,int,float,bool)):return v
 return None
def facts(paths):return {str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}
assert not OUT.exists(),'fresh output required'
OUT.mkdir();prior=json.loads(PRIOR.read_text(encoding='utf-8'))
paths=[Path(x) for x in prior['source_input_before']]+[Path('/mnt/d/FYP')/x for x in prior['user_before']]+[PRIOR,PROFILES,CODE/'subtask_assist_service.py',Path(__file__)]
before=facts(paths)
report={'ok':False,'status':'ASSIST_REPAIR_PREFLIGHT_RUNNING','source_input_user_before':before,'cases':[],'model_calls':0,'env_steps':0,'new_aux_actions':0,'policy_actions':0,'hermes_calls':0,'source_preparation_sha':sha(CODE/'subtask_preparation.py'),'old_high_z_m':1.4657861680946953}
env=None;exitcode=0
try:
 assert report['source_preparation_sha']=='0c5c64e4be692feb52241aeb01793752783b10ce24a90da2f155c74c349a6ecf'
 svc=service.SceneService(run_root=str(OUT));service._seed_everything(0)
 env=svc._build_env('libero_goal',8,0,0);env.reset(seed=0)
 report['state_before_sha']=service.state_sha(env)
 inner=service._inner_env(env);model=inner.sim.model;data=inner.sim.data
 report['controller_facts']=clean(pd.controller_facts(env))
 controller=inner.robots[0].controller
 report['controller_extra']={k:clean(getattr(controller,k,None)) for k in ('orientation_limits','use_ori','impedance_mode')}
 from robosuite.utils import control_utils
 oscfile=Path(inspect.getsourcefile(control_utils.set_goal_orientation))
 report['set_goal_orientation_source']={'path':str(oscfile),'sha256':sha(oscfile)}
 for case in prior['cases']:
  reading=copy.deepcopy(case['readings']['bowl_to_plate'])
  entry={'case_index':case['case_index'],'source_handoff_sha':case['handoff_sha'],'geometry_rehydration':[]};report['cases'].append(entry)
  for group in ('hand','obstacles'):
   for spec in reading[group]:
    current=sg.geom_bounds(model,data,spec['id'])
    same=current['name']==spec['name'] and current['type']==spec['type'] and bool(np.allclose(current['half'],spec['half'],atol=1e-8,rtol=0))
    assert same,('geom mismatch',spec['id'],spec['name'])
    spec['convex']=current['convex']
    entry['geometry_rehydration'].append({'group':group,'id':spec['id'],'name':spec['name'],'same_id_name_type_half':same,'convex_present':current['convex'] is not None})
  pos=np.asarray(reading['pose']['position'],dtype=float);R=np.asarray(reading['pose']['orientation_matrix'],dtype=float)
  entry['actual_initial_collisions']=sp.pose_collisions(reading,pos,R)
  ctx=sc.build_context(catalog.CAPABILITIES['bowl_to_plate'],reading)
  try:
   route=sp.plan_route(ctx,reading);entry['route']=clean(route)
   entry['controller_summary']=clean(sp.PreparationController(ctx,reading).summary());entry['route_usable']=True
   entry['checks']={'initial_no_overlap':entry['actual_initial_collisions']==[], 'orientation_policy_preserve':route['orientation_policy']=='preserve_downward_current_orientation','reorientation_not_required':route['reorientation_required'] is False,'all_waypoint_R_current':all(bool(np.array_equal(np.asarray(w['orientation']),R)) for w in route['waypoints']),'high_z_lower':route['high_z']<report['old_high_z_m']}
  except Exception as exc:
   entry['route_usable']=False;entry['route_error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};entry['checks']={'route_usable':False}
  print(json.dumps({'case':entry['case_index'],'route_usable':entry['route_usable'],'checks':entry['checks'],'high_z':entry.get('route',{}).get('high_z'),'ready_z':entry.get('route',{}).get('ready_z'),'clearance_extent':entry.get('route',{}).get('clearance_extent'),'max_hand_reach':entry.get('route',{}).get('max_hand_reach'),'chosen_index':entry.get('route',{}).get('chosen_index'),'check_samples':entry.get('route',{}).get('check_samples'),'rejection_reasons':entry.get('route',{}).get('rejection_reasons'),'error':entry.get('route_error',{}).get('message')},ensure_ascii=True),flush=True)
  if not entry['route_usable'] or not all(entry['checks'].values()):raise RuntimeError('case route gate failed; no further case processing')
 report['state_after_sha']=service.state_sha(env);report['state_unchanged']=report['state_after_sha']==report['state_before_sha']
 report['set_goal_orientation_source']['sha256_after']=sha(oscfile)
 report['ok']=len(report['cases'])==2 and report['state_unchanged'] and report['set_goal_orientation_source']['sha256_after']==report['set_goal_orientation_source']['sha256'] and all(e['route_usable'] and all(e['checks'].values()) for e in report['cases'])
except Exception:
 report['error']=traceback.format_exc();exitcode=2;print(report['error'],flush=True)
finally:
 if env is not None:
  report.setdefault('state_after_sha',service.state_sha(env));report['state_unchanged']=report['state_after_sha']==report.get('state_before_sha');env.close();report['own_env_closed']=True
 report['source_input_user_after']=facts(paths);report['source_input_user_unchanged']=before==report['source_input_user_after']
 report['ok']=report['ok'] and report['source_input_user_unchanged']
 if not report['ok']:exitcode=2
 report['status']='ASSIST_REPAIR_ZERO_ACTION_PREFLIGHT_PASS' if report['ok'] else 'ASSIST_REPAIR_ZERO_ACTION_PREFLIGHT_FAIL'
 report['script_sha256']=sha(__file__);report['exit_code']=exitcode
 (OUT/'report.json').write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'status':report['status'],'ok':report['ok'],'state_unchanged':report.get('state_unchanged'),'source_input_user_unchanged':report['source_input_user_unchanged'],'model_calls':0,'env_steps':0,'new_aux_actions':0,'output':str(OUT/'report.json')},ensure_ascii=True),flush=True)
sys.exit(exitcode)
