import gzip,json,sys,time,traceback
from pathlib import Path
import numpy as np
CODE=Path('/mnt/c/Users/Admin1/.codex/worktrees/wine-grasp-assist/FYP/First_Phase/scene_demo')
sys.path.insert(0,str(CODE))
import placement_experiments as pe
import service,side_grasp

OUT=Path('/home/yhwang/fyp/scene_demo/grasp_assist/2026-10-09-collision-probe-v1')
HISTORY=CODE/'results/2026-10-09-side-grasp/pilot/case_01'
ORIGIN='8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd'
BOWL='ed679730451dee4fee5a384e9eda27d1419f34ca9c98d78d03b7da49a1309b82'
FINAL='3f25a29eaa8180b7c402c72f3e1930178717f9a283a8dd5b5a9fbd6c677219ee'
WINE='wine_bottle_1'
def default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
report={'origin_state_sha':None,'after_bowl_state_sha':None,'trigger_state_sha':None,'trigger_reading':None,'geometry':[],'gripper_metadata':{},'aux_rows':[],'final_state_sha':None,'errors':[],'actual_replay_steps':0,'model_calls':0,'hermes_calls':0}
def save():
    (OUT/'probe.json').write_text(json.dumps(report,indent=2,default=default),encoding='utf-8')
def load(job,name):
    with gzip.open(HISTORY/job/name,'rt',encoding='utf-8') as stream:return [json.loads(line) for line in stream]
def optional(fn):
    try:return fn()
    except (AttributeError,KeyError,TypeError,ValueError,IndexError):return None
def name(model,kind,index):return optional(lambda:getattr(model,kind+'_id2name')(int(index)))
def probe_worker(svc,session_id,bowl,wine,reference):
    env=svc._env;record=svc._sessions[session_id]
    def step(row):
        action=np.asarray(row['sent_action'],dtype=np.float32)
        assert action.shape==(7,) and np.all(np.isfinite(action))
        returned=env.step(action)
        report['actual_replay_steps']+=1
        assert report['actual_replay_steps']<=235
        if isinstance(returned,tuple) and returned:svc._last_obs=returned[0]
        svc._total_steps=record.total_steps=report['actual_replay_steps']
    try:
        report['origin_state_sha']=service.state_sha(env);save()
        assert report['origin_state_sha']==ORIGIN,'origin SHA mismatch: '+str(report['origin_state_sha'])
        for row in bowl:step(row)
        report['after_bowl_state_sha']=service.state_sha(env);save()
        assert report['after_bowl_state_sha']==BOWL,'after bowl SHA mismatch: '+str(report['after_bowl_state_sha'])
        assert all(row['source']=='vla' for row in wine[:120])
        for row in wine[:120]:step(row)
        reading=side_grasp.read_geometry(env)
        report['trigger_reading']=reading;report['trigger_state_sha']=service.state_sha(env)
        measured=reading['snapshot']['objects'][WINE]
        expected=reference['before_snapshot']['objects'][WINE]
        diffs={k:float(np.max(np.abs(np.asarray(measured[k],dtype=np.float64)-np.asarray(expected[k],dtype=np.float64)))) for k in ('position','quaternion')}
        report['trigger_wine_max_errors']=diffs;save()
        assert all(np.isfinite(v) and v<=1e-9 for v in diffs.values()),'trigger wine mismatch: '+str(diffs)
        inner=service._inner_env(env);model=inner.sim.model;data=inner.sim.data
        total_collision=0
        for i in range(model.ngeom):
            contype=optional(lambda:int(model.geom_contype[i]));affinity=optional(lambda:int(model.geom_conaffinity[i]))
            if not (contype or affinity):continue
            total_collision+=1;geom_name=name(model,'geom',i)
            if not isinstance(geom_name,str) or not (geom_name.startswith(('gripper0','robot0')) or 'wine_bottle' in geom_name):continue
            body_id=optional(lambda:int(model.geom_bodyid[i]))
            report['geometry'].append({'geom_id':i,'name':geom_name,'body_name':name(model,'body',body_id) if body_id is not None else None,'type':optional(lambda:int(model.geom_type[i])),'size':optional(lambda:np.array(model.geom_size[i],copy=True)),'rbound':optional(lambda:float(model.geom_rbound[i])),'world_position':optional(lambda:np.array(data.geom_xpos[i],copy=True)),'world_matrix':optional(lambda:np.array(data.geom_xmat[i],copy=True).reshape(3,3)),'contype':contype,'conaffinity':affinity})
        report['all_collision_geom_count']=total_collision
        robot=inner.robots[0];gripper=robot.gripper
        report['gripper_metadata']={'important_geoms':optional(lambda:gripper.important_geoms),'important_sites':optional(lambda:gripper.important_sites),'eef_name':optional(lambda:robot.robot_model.eef_name)}
        save()
        assert all(row['source']=='local_grasp' for row in wine[120:]) and len(wine[120:])==11
        for row in wine[120:]:
            step(row);contacts=[]
            for i in range(data.ncon):
                contact=data.contact[i];g1=name(model,'geom',contact.geom1);g2=name(model,'geom',contact.geom2)
                if any(isinstance(n,str) and 'wine_bottle' in n for n in (g1,g2)):
                    contacts.append({'geom1':g1,'geom2':g2,'dist':float(contact.dist),'pos':np.array(contact.pos,copy=True)})
            after=side_grasp.read_geometry(env)
            report['aux_rows'].append({'step':row['step'],'actual_env_step':report['actual_replay_steps'],'sent_action':row['sent_action'],'wine_contacts':contacts,'wine_pose':after['snapshot']['objects'][WINE],'geometry':after,'state_sha':service.state_sha(env)})
            save()
        report['final_state_sha']=service.state_sha(env);save()
        assert report['final_state_sha']==FINAL,'final SHA mismatch: '+str(report['final_state_sha'])
        assert report['actual_replay_steps']==235
        return {'ok':True}
    except Exception as exc:
        report['errors'].append(type(exc).__name__+': '+str(exc));save();raise
def main():
    assert not OUT.exists(),str(OUT)+' exists; no retry'
    OUT.mkdir(parents=True)
    svc=None
    try:
        bowl=load('job_01','action_sources.jsonl.gz');wine=load('job_02','action_sources.jsonl.gz')
        for rows,count in ((bowl,104),(wine,131)):
            assert len(rows)==count and [r['step'] for r in rows]==list(range(1,count+1))
        references=load('job_02','wine_telemetry.jsonl.gz');reference=[r for r in references if r.get('step')==121]
        assert len(reference)==1
        svc=pe.DiagnosticService(run_root=str(OUT/'sessions'),policy_loader=lambda _svc:None,completion_mode='release_verified',grasp_guard_mode='shadow')
        svc.start();deadline=time.monotonic()+60
        while True:
            health=svc.health()
            assert not health.get('worker_error'),health.get('worker_error')
            if health.get('ready'):break
            assert time.monotonic()<deadline,'readiness timeout'
            time.sleep(.1)
        session=svc.create_session('goal_table',seed=0,init_state_index=0)
        assert session.get('ok') is True,session
        report['session_id']=session['session_id']
        outcome=svc._sync_work('collision_probe',lambda:probe_worker(svc,session['session_id'],bowl,wine,reference[0]),timeout=180)
        assert outcome.get('ok') is True,outcome
        report['ok']=True
    except Exception as exc:
        message=type(exc).__name__+': '+str(exc)
        if message not in report['errors']:report['errors'].append(message)
        report['ok']=False
        traceback.print_exc()
    finally:
        if svc is not None:
            try:
                close=svc._sync_work('close_env',lambda:(svc._close_env() or {'ok':True}),timeout=30)
                if close.get('ok') is not True:report['errors'].append('close_env: '+str(close));report['ok']=False
            finally:svc.stop()
        report['physics_calls']=report['actual_replay_steps'];save()
    print(json.dumps({'ok':report.get('ok'),'origin_state_sha':report['origin_state_sha'],'after_bowl_state_sha':report['after_bowl_state_sha'],'trigger_state_sha':report['trigger_state_sha'],'trigger_wine_max_errors':report.get('trigger_wine_max_errors'),'final_state_sha':report['final_state_sha'],'actual_replay_steps':report['actual_replay_steps'],'geometry_count':len(report['geometry']),'contact_counts':[len(r['wine_contacts']) for r in report['aux_rows']],'errors':report['errors'],'output':str(OUT/'probe.json')}),flush=True)
    return 0 if report.get('ok') and not report['errors'] else 2
if __name__=='__main__':sys.exit(main())
