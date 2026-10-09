"""Native mechanical joint-home diagnostic; preparation only until authorized."""
import argparse
import hashlib
import importlib.util
import json
import math
import numbers
import sys
import time
import traceback
import uuid
from pathlib import Path

HERE=Path('/mnt/d/FYP/First_Phase/scene_demo')
OUTPUT=HERE/'results/2026-10-09-joint-home/pilot'
PARENT_PATH=HERE/'results/2026-10-09-entry-compatibility/evidence/entry_compat_20261009_run.py'
PLAN_PATH=HERE/'plans/2026-10-09-joint-home.md'
ORIGIN='8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd'
CASES=((2,'case_02_joint_home',346,'2a4521602e384127e079782f4ff0b9ba8537548aabce2ba348a9c7320129c857'),
       (3,'case_03_joint_home',348,'69b98ee1794f350d6f64f5c574bfe8e76a745c684cd65f641152a529e759d20b'))
GOLD=[['on','akita_black_bowl_1','plate_1'],['on','wine_bottle_1','wine_rack_1_top_region']]
STOVE_OFF=['not','turnon','flat_stove_1']
CONDITION='joint_home_probe'
HOME_CAP=360
VLA_CAP=336


def load_parent():
    spec=importlib.util.spec_from_file_location('joint_home_readonly_parent',PARENT_PATH)
    if spec is None or spec.loader is None:raise RuntimeError('parent runner import unavailable')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def event(name,**fields):
    print(json.dumps({'event':name,**fields},ensure_ascii=True),flush=True)


def gates(args,parent):
    for key in ('output_dir','profiles','software_proof','inputs'):
        path=Path(getattr(args,key))
        if not path.is_absolute():raise ValueError(key+' must be absolute')
        if key!='output_dir' and not path.is_file():raise ValueError(key+' missing')
    output=Path(args.output_dir)
    if output.resolve()!=OUTPUT.resolve() or output.exists():raise ValueError('fixed output must be fresh')
    software=parent.require(parent.read(args.software_proof),'software proof')
    for field in ('final_test_exit_code','regression_exit_code'):
        if type(software.get(field)) is not int or software[field]!=0:raise ValueError('software exit: '+field)
    actual=parent.source_map()
    if software.get('source_sha256')!=actual:raise ValueError('software must freeze every current top-level Python source')
    inputs=parent.require(parent.read(args.inputs),'inputs')
    checks=inputs.get('checks')
    if not isinstance(checks,dict) or not checks or not all(v is True for v in checks.values()):raise ValueError('input checks not strictly true')
    if not isinstance(inputs.get('inputs_before'),dict) or len(inputs['inputs_before'])!=16:raise ValueError('exact16 frozen inputs required')
    if len(inputs.get('user8_before',[]))!=8:raise ValueError('eight user facts required')
    parent.verify_facts(inputs['inputs_before']);parent.verify_facts(inputs['user8_before'])
    if inputs.get('plan_sha256')!=parent.sha(PLAN_PATH):raise ValueError('plan SHA mismatch')
    profiles=parent.read(args.profiles)
    if not isinstance(profiles,dict):raise ValueError('profiles not mapping')
    return output,profiles,software,inputs,actual


def load_wine(inputs,parent):
    loaded={}
    for index,name,count,handoff in CASES:
        item=inputs['wine_cases'][str(index)]
        paths=(parent.linux_path(item['actions_path']),parent.linux_path(item['result_path']))
        for path in paths:
            matches=[v for v in inputs['inputs_before'].values() if parent.linux_path(v['path'])==path]
            if len(matches)!=1 or parent.sha(path)!=matches[0]['sha256']:raise ValueError('wine input not frozen')
        rows=parent.checked_actions(parent.json_rows(paths[0]),'sent_action',count)
        result=parent.read(paths[1])
        if item.get('model_seed')!=index or item.get('action_count')!=count or item.get('origin_sha')!=ORIGIN or item.get('handoff_sha')!=handoff:raise ValueError('wine metadata mismatch')
        if result.get('state_before_sha')!=ORIGIN or result.get('state_after_sha')!=handoff:raise ValueError('wine original result mismatch')
        loaded[index]=rows
    return loaded


def finite_scalar(value,label):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)):
        raise ValueError('unknown finite scalar: '+label)
    return float(value)


def finite_vector(value,length,label,np):
    if not isinstance(value,(list,tuple,np.ndarray)):raise ValueError('unknown vector: '+label)
    a=np.asarray(value,dtype=object)
    if a.shape!=(length,):raise ValueError('wrong vector shape: '+label)
    return np.asarray([finite_scalar(x,label) for x in a.tolist()],dtype=np.float64)


def robot_contacts(env,service,np):
    inner=service._inner_env(env);robots=inner.robots
    if not isinstance(robots,(list,tuple)) or len(robots)!=1:raise ValueError('single robot missing')
    robot=robots[0];sim=inner.sim;model=sim.model;data=sim.data
    names=list(robot.robot_model.contact_geoms)+list(robot.gripper.contact_geoms)
    if not names or any(not isinstance(n,str) or not n for n in names):raise ValueError('robot collision names unknown')
    ids=set()
    for name in names:
        index=model.geom_name2id(name)
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,numbers.Integral) or index<0 or index>=model.ngeom:raise ValueError('robot collision id unknown')
        ids.add(int(index))
    count=data.ncon;contacts=data.contact
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,numbers.Integral) or count<0 or count>len(contacts):raise ValueError('contact count unknown')
    found=[]
    for i in range(int(count)):
        contact=contacts[i];pair=[]
        for raw in (contact.geom1,contact.geom2):
            if isinstance(raw,(bool,np.bool_)) or not isinstance(raw,numbers.Integral) or raw<0 or raw>=model.ngeom:raise ValueError('contact geom index unknown')
            pair.append(int(raw))
        dist=finite_scalar(contact.dist,'contact distance')
        if (pair[0] in ids or pair[1] in ids) and dist<=0:
            pos=finite_vector(contact.pos,3,'contact pos',np)
            names_pair=[model.geom_id2name(g) for g in pair]
            if any(not isinstance(n,str) or not n for n in names_pair):raise ValueError('contact geom name unknown')
            found.append({'geom1':pair[0],'geom2':pair[1],'names':names_pair,'dist':dist,'pos':pos.tolist()})
    return found


def protection(env,snapshot,baseline,service,np):
    violations=[];measurements={}
    objects=snapshot.get('objects');original=baseline.get('objects')
    if not isinstance(objects,dict) or not objects or not isinstance(original,dict) or set(objects)!=set(original):raise ValueError('all-object screen unknown')
    if snapshot.get('grasp_observation_complete') is not True:raise ValueError('holding observation incomplete')
    held=snapshot.get('held_objects')
    if not isinstance(held,list):raise ValueError('held objects unknown')
    if held:violations.append('not_empty_hand')
    for name,obj in objects.items():
        if not isinstance(obj,dict) or type(obj.get('grasped')) is not bool:raise ValueError('object grasp unknown: '+name)
        if obj['grasped'] is not False:violations.append('held:'+name)
        p=finite_vector(obj.get('position'),3,name+' position',np);p0=finite_vector(original[name].get('position'),3,name+' baseline position',np)
        q=finite_vector(obj.get('quaternion'),4,name+' quaternion',np);q0=finite_vector(original[name].get('quaternion'),4,name+' baseline quaternion',np)
        if np.linalg.norm(q)<=0 or np.linalg.norm(q0)<=0:raise ValueError('zero quaternion: '+name)
        displacement=float(np.linalg.norm(p-p0));angle=2*math.acos(float(np.clip(abs(np.dot(q/np.linalg.norm(q),q0/np.linalg.norm(q0))),0,1)))
        measurements[name]={'position_displacement_m':displacement,'rotation_rad':angle}
        if displacement>.005:violations.append('object_displacement:'+name)
        if angle>.05:violations.append('object_rotation:'+name)
    wine=snapshot.get('predicates',{}).get('on|wine_bottle_1|wine_rack_1_top_region')
    if type(wine) is not bool:raise ValueError('wine predicate unknown')
    if wine is not True:violations.append('wine_goal_false')
    stove=service.eval_goal_predicate(env,STOVE_OFF)
    if type(stove) is not bool:raise ValueError('stove predicate unknown')
    if stove is not True:violations.append('stove_not_off')
    contacts=robot_contacts(env,service,np)
    if contacts:violations.append('robot_contact')
    return {'ok':not violations,'violations':violations,'objects':measurements,'wine_goal':wine,'stove_off':stove,'robot_contacts':contacts,'empty_hand':held==[],'observation_complete':True}


def capture_state(svc,directory,reference,tag,parent,pe,service,pd,jh,np,batch=None):
    directory.mkdir();env=svc._env;before=service.state_sha(env);sim=service._inner_env(env).sim
    robotstate=jh.read_robot_state(env);metrics=jh.home_metrics(reference,robotstate)
    snapshot=pe.capture_snapshot(env,GOLD);pose=pd.read_eef_pose(env)
    np.save(directory/'sim_state.npy',parent.state_vector(env,service),allow_pickle=False)
    np.savez_compressed(directory/'qpos_qvel.npz',qpos=np.asarray(sim.data.qpos,dtype=np.float64).copy(),qvel=np.asarray(sim.data.qvel,dtype=np.float64).copy())
    parent.save_numeric(directory/'raw_observation.npz',svc._last_obs,pe)
    if batch is not None:parent.save_numeric(directory/'observation_batch.npz',batch,pe)
    images=service.capture_vla_images(env)
    if set(images)!={'agentview','wrist'}:raise ValueError('two VLA views unavailable')
    for view,image in images.items():service._save_png(directory/(view+'.png'),image)
    for filename,value in (('robot_state',robotstate),('home_metrics',metrics),('snapshot',snapshot),('eef_pose',pose)):
        pe._write_json_atomic(directory/(filename+'.json'),parent.clean(value))
    after=service.state_sha(env)
    if before!=after:raise ValueError('readonly capture changed state')
    record={'tag':tag,'directory':str(directory),'state_before_sha256':before,'state_after_sha256':after,'readonly_state_unchanged':True}
    pe._write_json_atomic(directory/'capture.json',record)
    return record


class RecordingPreprocessor:
    def __init__(self,original,record):self.original=original;self.record=record;self.done=False
    def __call__(self,*args,**kwargs):
        result=self.original(*args,**kwargs)
        if not self.done:
            self.record(result);self.done=True
        return result
    def reset(self,*args,**kwargs):return self.original.reset(*args,**kwargs)
    def __getattr__(self,name):return getattr(self.original,name)


def service_class(UnifiedAssistService,capture):
    class HomeProbeService(UnifiedAssistService):
        def _select_action(self,batch):
            if getattr(self,'home_first_input_pending',False):
                self.home_first_batch_capture=capture(self,batch)
                self.home_first_capture_count+=1;self.home_first_input_pending=False
            return super()._select_action(batch)
    return HomeProbeService


def run_home(svc,sid,directory,reference,baseline,parent,pe,service,pd,jh,np,imageio):
    env=svc._env;session=svc._sessions[sid];robot=service._inner_env(env).robots[0];original=robot.controller
    home={'ok':False,'ready':False,'actions':0,'reason':None,'unknown':[],'operational_errors':[],'restore':None,'last_metrics':None,'last_protection':None,'video':str(directory/'home.mp4'),'initial_image':None}
    directory.mkdir();writer=None;adapter=None;streak=0
    try:
        image_state_before=service.state_sha(env)
        frame=service.capture_vla_images(env)['agentview'];service._save_png(directory/'first.png',frame)
        home['initial_image']=str(directory/'first.png')
        if service.state_sha(env)!=image_state_before:raise ValueError('readonly initial image capture changed physical state')
        original_facts=pd.controller_facts(env)
        mismatch=pd.controller_mismatch(original_facts)
        if mismatch is not None:raise ValueError('original OSC contract mismatch: '+mismatch)
        state=jh.read_robot_state(env);points=jh.joint_waypoints(state['arm_qpos'],reference['homeq'])
        before_screen=service.state_sha(env);screen=jh.screen_joint_path(env,reference,points);after_screen=service.state_sha(env)
        if before_screen!=after_screen or screen.get('live_unchanged') is not True:raise ValueError('path screen changed native state')
        pe._write_json_atomic(directory/'route.json',parent.clean({'waypoints':points,'screen':screen,'state_before_sha256':before_screen,'state_after_sha256':after_screen}))
        home['route_screen']=screen
        if screen.get('ok') is not True:
            home['reason']='joint_route_blocked';return home
        adapter=jh.JointHomeAdapter(original,env,reference)
        if adapter.original is not original:raise ValueError('adapter original identity mismatch')
        robot.controller=adapter
        writer=imageio.get_writer(str(directory/'home.mp4'),fps=20);writer.append_data(frame)
        with (directory/'home.jsonl').open('x',encoding='utf8') as log:
            for i in range(1,HOME_CAP+1):
                before_snapshot=pe.capture_snapshot(env,GOLD);before_state=jh.read_robot_state(env)
                before_metrics=jh.home_metrics(reference,before_state);before_guard=protection(env,before_snapshot,baseline,service,np)
                if before_guard['ok'] is not True:home['reason']='home_protection_failed';home['last_protection']=before_guard;break
                target=points[min(i-1,len(points)-1)];adapter.select_target(target)
                sent=np.asarray([0,0,0,0,0,0,-1],dtype=np.float32)
                obs=env.step(sent);svc._last_obs=obs[0];svc._total_steps+=1;session.total_steps+=1;home['actions']+=1
                after_state=jh.read_robot_state(env);metrics=jh.home_metrics(reference,after_state)
                snapshot=pe.capture_snapshot(env,GOLD);guard=protection(env,snapshot,baseline,service,np);pose=pd.read_eef_pose(env)
                torques=finite_vector(getattr(robot,'torques',None),7,'actual clipped robot torques',np)
                frame=service.capture_vla_images(env)['agentview'];writer.append_data(frame);service._save_png(directory/'last.png',frame)
                home['last_metrics']=metrics;home['last_protection']=guard
                streak=streak+1 if guard['ok'] is True and metrics.get('ready') is True else 0
                row={'step':home['actions'],'source':'joint_home','target':target,'sent_env_action':sent,'robot_actual_clipped_torques':torques,'before_robot_state':before_state,'before_metrics':before_metrics,'before_protection':before_guard,'robot_state':after_state,'metrics':metrics,'eef_pose':pose,'oracle_snapshot':snapshot,'protection':guard,'state_sha256':service.state_sha(env),'confirmed_streak':streak}
                log.write(json.dumps(parent.clean(row),allow_nan=False)+'\n');log.flush()
                if i%20==0:event('HOME',case=svc.home_case,actions=i,metrics=metrics,streak=streak)
                if guard['ok'] is not True:home['reason']='home_protection_failed';break
                if streak>=5:home['ready']=True;home['reason']='joint_home_ready';break
            if home['reason'] is None:home['reason']='joint_home_budget_exhausted'
        home['postaction_ready_streak']=streak;home['ok']=True
    except Exception:
        home['reason']='joint_home_operational_error';home['operational_errors'].append(traceback.format_exc())
    finally:
        try:
            robot.controller=original;original.update(force=True);original.reset_goal()
            facts=pd.controller_facts(env);mismatch=pd.controller_mismatch(facts)
            restored=robot.controller is original and mismatch is None
            state=jh.read_robot_state(env);metrics=jh.home_metrics(reference,state)
            snapshot=pe.capture_snapshot(env,GOLD);guard=protection(env,snapshot,baseline,service,np)
            home['restore']={'original_identity':robot.controller is original,'controller_facts':facts,'controller_mismatch':mismatch,'ok':restored,'readonly_metrics':metrics,'readonly_protection':guard,'not_additional_postaction_confirmation_sample':True}
            if not restored:raise ValueError('OSC restoration did not preserve original contract')
            if home['ready'] and (metrics.get('ready') is not True or guard['ok'] is not True):home['ready']=False;home['reason']='home_restore_confirmation_failed'
        except Exception:
            home['ready']=False;home['operational_errors'].append(traceback.format_exc());home['ok']=False
        if writer is not None:writer.close()
        home['final_state_sha256']=service.state_sha(env)
        pe._write_json_atomic(directory/'home_result.json',parent.clean(home))
    return home


def job_worker(svc,sid,directory,baseline,reference,parent,pe,service,pd,jh,np):
    session=svc._sessions[sid];env=svc._env;request_id=uuid.uuid4().hex;job_id=uuid.uuid4().hex
    plan=service.PlanRecord(request_id,sid,{'decision':'execute','capability_ids':['wine_to_rack','bowl_to_plate'],'budget_per_subgoal':VLA_CAP,'scene_version':session.scene_version,'audit':False,'rationale':'Joint-home diagnostic, reused literal task; no new Hermes.'})
    plan.completed_capability_ids=['wine_to_rack'];plan.pending_capability_ids=['bowl_to_plate'];plan.state='running'
    job=service.JobRecord(job_id,request_id,sid,'bowl_to_plate',directory/'job');plan.job_ids=[job_id]
    with svc._lock:
        svc._plans[request_id]=plan;svc._jobs[job_id]=job;svc._job_order.append(job_id);session.active_request_id=request_id
    svc.home_job_dir=directory;svc.home_reference=reference;svc.home_first_input_pending=True
    svc.home_first_capture_count=0;svc.home_first_batch_capture=None;svc.home_model_input_capture=None
    old_pre=svc._v1._pre
    def record_pre(result):
        before=service.state_sha(env);dest=directory/'first_input';parent.save_numeric(dest/'model_input.npz',result,pe)
        after=service.state_sha(env)
        if before!=after:raise ValueError('preprocessor input capture changed native state')
        svc.home_model_input_capture={'tag':'pre_after_actual_call','state_before_sha256':before,'state_after_sha256':after,'readonly_state_unchanged':True,'path':str(dest/'model_input.npz'),'capture_count':1}
        pe._write_json_atomic(dest/'model_input_capture.json',svc.home_model_input_capture)
    proxy=RecordingPreprocessor(old_pre,record_pre);svc._v1._pre=proxy
    value,error=None,None;before_sha=service.state_sha(env)
    try:value=svc._run_capability(session,plan,job,'bowl_to_plate')
    except service.SceneError as exc:
        error={'reason':exc.reason,'detail':exc.detail}
        if job.state not in ('completed','error','cancelled'):job.state='error';job.success=False;job.ended_reason=exc.reason
        job.error=str(exc)
    finally:
        svc._v1._pre=old_pre;svc.home_first_input_pending=False
        after=pe.capture_snapshot(env,GOLD);stove=service.eval_goal_predicate(env,STOVE_OFF)
        plan.state='cancelled' if job.state=='cancelled' else ('completed' if job.success is True else 'blocked')
        if job.success is True:plan.completed_capability_ids.append('bowl_to_plate');plan.pending_capability_ids=[]
        session.active_request_id=None
        record={'ok':True,'job':job.public(),'plan':plan.public(),'base_result':value,'scene_error':error,'before_sha256':before_sha,'after_sha256':service.state_sha(env),'first_observation_batch':svc.home_first_batch_capture,'first_model_input':svc.home_model_input_capture,'first_batch_capture_count':svc.home_first_capture_count,'preprocessor_original_restored':svc._v1._pre is old_pre}
        pe._write_json_atomic(directory/'worker_result.json',parent.clean(record));pe._write_json_atomic(directory/'vla_final_snapshot.json',parent.clean(after))
    return {**record,'after_snapshot':after,'stove_off':stove}


def combine_video(home_path,job_path,dest,home_actions,vla_actions,imageio):
    records=[];writer=None;output_frames=0;size=None
    try:
        writer=imageio.get_writer(str(dest),fps=20)
        for part,path in enumerate((home_path,job_path)):
            reader=imageio.get_reader(str(path));count=0;fps=float(reader.get_meta_data()['fps'])
            try:
                if abs(fps-20)>1e-6:raise ValueError('video fps not20')
                for i,frame in enumerate(reader):
                    if size is None:size=list(frame.shape[:2])
                    if list(frame.shape[:2])!=size:raise ValueError('video sizes differ')
                    count+=1
                    if part==1 and i==0:continue
                    writer.append_data(frame);output_frames+=1
            finally:reader.close()
            expected=(home_actions if part==0 else vla_actions)+1
            if count!=expected:raise ValueError('source frame count not actual action count+1')
            records.append({'path':str(path),'frame_count':count,'fps':fps,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    finally:
        if writer is not None:writer.close()
    if output_frames!=home_actions+vla_actions+1:raise ValueError('combined frame count mismatch')
    reader=imageio.get_reader(str(dest))
    try:decoded=sum(1 for _ in reader)
    finally:reader.close()
    if decoded!=output_frames:raise ValueError('combined decode frame mismatch')
    return {'path':str(dest),'source_videos':records,'frames':output_frames,'size':size,'fps':20,'dropped_second_segment_first_duplicate_frame':1,'includes_wine_prefix':False,'sha256':hashlib.sha256(Path(dest).read_bytes()).hexdigest()}


def run_case(svc,spec,output,wine_rows,parent,pe,service,pd,jh,np,imageio,paired,pilot):
    index,name,prefix_count,handoff=spec;directory=output/name;directory.mkdir()
    trial={'case':name,'case_index':index,'model_seed':index,'physical_success':False,'gate_pass':False,'operational_errors':[],'unknown':[],'replay_actions':0,'home_actions':0,'vla_actions':0,'first_actual_model_input':None,'new_hermes_calls':0}
    svc.home_case=name
    try:
        session=parent.require(svc.create_session('goal_table',seed=0,init_state_index=0),'session');sid=session['session_id'];trial['session']=session
        pe.FINAL_ORACLE_GOALS[CONDITION]=[list(g) for g in GOLD];svc._diag_condition=CONDITION;svc._subtask_assist_mode='disabled'
        def capture_origin():
            before=service.state_sha(svc._env)
            if before!=ORIGIN:raise ValueError('origin SHA mismatch')
            reference=jh.capture_home_reference(svc._env,origin_sha256=before)
            pe._write_json_atomic(directory/'native_home.json',parent.clean(reference))
            record=capture_state(svc,directory/'native_origin',reference,'native_origin_before_wine',parent,pe,service,pd,jh,np)
            if service.state_sha(svc._env)!=before:raise ValueError('home reference capture changed state')
            return {'ok':True,'reference':reference,'capture':record}
        origin=parent.require(svc._sync_work('joint_home_origin_capture',capture_origin,timeout=900),'origin capture');reference=origin['reference'];trial['origin_capture']=origin['capture']
        prefix=parent.require(svc._sync_work('joint_home_wine_replay',lambda:parent.replay(svc,sid,wine_rows,'sent_action',GOLD,directory,pe,service),timeout=900),'wine replay')
        trial['replay_actions']=prefix['replay_actions']
        if prefix['origin_sha256']!=ORIGIN or prefix['after_sha256']!=handoff:raise ValueError('wine handoff SHA mismatch')
        snapshot=prefix['after']
        if snapshot.get('held_objects')!=[] or snapshot.get('grasp_observation_complete') is not True or snapshot.get('predicates',{}).get('on|wine_bottle_1|wine_rack_1_top_region') is not True:raise ValueError('wine prefix not complete/empty/goaltrue')
        pe._write_json_atomic(directory/'prefix_meta.json',{'origin_sha256':prefix['origin_sha256'],'handoff_sha256':prefix['after_sha256'],'replay_actions':prefix_count})
        event('PREFIX',case=name,replay_actions=prefix_count,handoff_sha256=handoff)
        home=svc._sync_work('joint_home_run',lambda:run_home(svc,sid,directory/'home',reference,snapshot,parent,pe,service,pd,jh,np,imageio),timeout=900)
        trial['home']=home;trial['home_actions']=home['actions'];trial['operational_errors'].extend(home['operational_errors'])
        event('HOME_DONE',case=name,ready=home['ready'],actions=home['actions'],reason=home['reason'])
        if home['ready'] is not True or home['restore'].get('ok') is not True:
            trial['not_run_vla_reason']=home['reason'];return trial
        def refresh():
            before=service.state_sha(svc._env);svc._reset_policy_queues();svc._last_obs=svc._refresh_observation(svc._env);after=service.state_sha(svc._env)
            if before!=after:raise ValueError('queues/refresh changed physical state')
            return {'ok':True,'state_before_sha256':before,'state_after_sha256':after,'readonly_state_unchanged':True}
        trial['queue_refresh']=parent.require(svc._sync_work('joint_home_refresh_for_vla',refresh,timeout=900),'fresh observation')
        seed=parent.require(paired._seed_model_rng(svc,index),'model RNG')
        if seed.get('seeded') is not True:raise ValueError('model RNG not seeded')
        trial['model_rng']=seed
        actual=parent.require(svc._sync_work('joint_home_bowl_vla',lambda:job_worker(svc,sid,directory,snapshot,reference,parent,pe,service,pd,jh,np),timeout=900),'VLA worker')
        job=actual['job'];trial['job']=job;trial['vla_actions']=job['steps'];trial['first_actual_model_input']=actual['first_model_input'];trial['first_observation_batch']=actual['first_observation_batch']
        rows,paths=pilot.read_oracle_rows([job]);score=parent.score_literal(rows,actual['after_snapshot'],snapshot,actual['stove_off'],GOLD)
        trial['score']=score;trial['telemetry_paths']=paths;trial['unknown']=score['unknown']
        assist=parent.read(Path(job['run_dir'])/'subtask_assist.json')
        if assist.get('prepare_source_actions')!=0 or assist.get('local_grasp_source_actions')!=0 or assist.get('vla_source_actions')!=job['steps']:raise ValueError('disabled continuation ran helper or counts differ')
        if job['steps']>VLA_CAP:raise ValueError('VLA336 budget violated')
        if job['steps']>0 and (actual['first_batch_capture_count']!=1 or actual['first_model_input'] is None):raise ValueError('actual model inputs not recorded exactly once')
        reason=actual['scene_error']['reason'] if actual['scene_error'] else job.get('ended_reason')
        if job.get('state')=='cancelled' or (job.get('error') and reason!='budget_exhausted'):trial['operational_errors'].append({'reason':reason,'error':job.get('error')})
        trial['physical_success']=home['ready'] is True and job.get('success') is True and score['physical_gold_success'] is True
        event('VLA_DONE',case=name,actions=job['steps'],physical_success=trial['physical_success'],reason=reason)
        if job.get('rollout_path'):
            trial['combined_video']=combine_video(directory/'home/home.mp4',Path(job['rollout_path']),directory/'combined.mp4',home['actions'],job['steps'],imageio)
        trial['gate_pass']=trial['physical_success'] is True and not trial['unknown'] and not trial['operational_errors']
    except Exception:trial['operational_errors'].append(traceback.format_exc())
    finally:
        trial['new_motion_actions']=trial['home_actions']+trial['vla_actions'];trial['total_physical_actions']=trial['replay_actions']+trial['new_motion_actions']
        pe._write_json_atomic(directory/'trial.json',parent.clean(trial))
    return trial


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output-dir','profiles','software-proof','inputs'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args(argv)
    parent=load_parent();output,profiles,software,inputs,sources=gates(args,parent);wine=load_wine(inputs,parent)
    sys.path.insert(0,str(HERE))
    import numpy as np
    import imageio.v2 as imageio
    import service
    import placement_experiments as pe
    import preparation_diagnostics as pd
    import joint_home as jh
    import paired_config_experiments as paired
    import guard_validation as gv
    import grasp_assist_service as gas
    import side_grasp_pilot as pilot
    import catalog
    from subtask_assist_service import UnifiedAssistService
    fixed_files={str(Path(getattr(args,k))):parent.sha(Path(getattr(args,k))) for k in ('profiles','inputs','software_proof')}
    for p in (Path(__file__),PLAN_PATH,PARENT_PATH):fixed_files[str(p)]=parent.sha(p)
    prereg={'cases':[{'case_index':i,'case':n,'wine_replay_actions':c,'model_seed':i,'handoff_sha256':h} for i,n,c,h in CASES],'max_physical_cases':2,'stop_rule':'Previous physical success strictly True and no unknown/operational/cancel errors, else stop remaining; no retry.','origin_sha256':ORIGIN,'scene':'goal_table','env_seed':0,'init_state_index':0,'gold':GOLD,'stove_goal':STOVE_OFF,'cheese_tolerance_m':.005,'home_all_objects_position_tolerance_m':.005,'home_all_objects_orientation_tolerance_rad':.05,'home_empty_complete_required':True,'robot_contact_dist_le0_rejected':True,'home_cap':HOME_CAP,'joint_step_rad':.01,'joint_error_tolerance_rad':.01,'joint_speed_tolerance_rad_s':.02,'finger_error_tolerance_m':.002,'finger_speed_tolerance_m_s':.02,'home_postaction_samples':5,'vla_budget':VLA_CAP,'profile':'baseline_bf16','profile_config':{'use_amp':True,'num_steps':10,'n_action_steps':1},'instruction':catalog.CAPABILITIES['bowl_to_plate']['instruction'],'completion_mode':'release_verified','grasp_guard_mode':'shadow','assist_mode':'disabled','checkpoint_revision':'6721902bc4d61e50a3bfdb11dfb4cb626f05d102','new_hermes_calls':0,'retries':0,'extra_physical_preflight':0,'cost_disclosure':'Home <=360 separate from VLA336; old disabled500 is reused background, not equal-budget rerun.','source_sha256':sources,'fixed_files_sha256':fixed_files,'inputs_before':inputs['inputs_before'],'user8_before':inputs['user8_before'],'baselines':inputs['baselines'],'scratch_joint_path_forward_only':True,'live_qpos_or_qvel_assignment':False,'full_arm_continuous_collision_guarantee':False}
    output.mkdir(parents=True);pe._write_json_atomic(output/'preregistration.json',parent.clean(prereg));event('PREREG',path=str(output/'preregistration.json'),planned_cases=2)
    report={'ok':False,'planned_cases':2,'executed_cases':0,'cases':[],'not_run':[{'case':x[1],'reason':'not_started'} for x in CASES],'operational_errors':[],'new_hermes_calls':0,'preregistration_sha256':parent.sha(output/'preregistration.json')}
    saved_oracle=dict(pe.FINAL_ORACLE_GOALS);svc=None
    def capture_first(svc,batch):
        return capture_state(svc,svc.home_job_dir/'first_input',svc.home_reference,'before_first_vla',parent,pe,service,pd,jh,np,batch=batch)
    HomeProbeService=service_class(UnifiedAssistService,capture_first)
    try:
        gas.configure_process_environment()
        svc=HomeProbeService(model_path=service.DEFAULT_MODEL_PATH,run_root=str(output/'sessions'),calibration_profiles=profiles,assist_mode='disabled',completion_mode='release_verified',grasp_guard_mode='shadow')
        svc.start();deadline=time.monotonic()+180
        while True:
            health=svc.health();report['health']=health
            if health.get('worker_error'):raise RuntimeError(str(health['worker_error']))
            if health.get('ready') is True:break
            if time.monotonic()>=deadline:raise TimeoutError('readiness180 timeout')
            time.sleep(1)
        profile=svc.configure_profile('baseline_bf16');report['profile_readback']=profile
        if not gv._profile_readback_ok(profile):raise ValueError('baseline profile readback mismatch')
        for spec in CASES:
            if parent.source_map()!=sources:raise ValueError('source changed')
            parent.verify_facts(inputs['inputs_before']);parent.verify_facts(inputs['user8_before'])
            event('START',case=spec[1],model_seed=spec[0])
            trial=run_case(svc,spec,output,wine[spec[0]],parent,pe,service,pd,jh,np,imageio,paired,pilot)
            report['cases'].append(trial);report['executed_cases']=len(report['cases']);report['operational_errors'].extend(trial['operational_errors'])
            report['not_run']=[{'case':s[1],'reason':'not_started'} for s in CASES[len(report['cases']):]]
            if trial['gate_pass'] is not True:report['not_run']=[{'case':s[1],'reason':'previous_case_failed:'+spec[1]} for s in CASES[len(report['cases']):]]
            pe._write_json_atomic(output/'report.json',parent.clean(report))
            if trial['gate_pass'] is not True:break
    except Exception:report['operational_errors'].append(traceback.format_exc())
    finally:
        if svc is not None:
            try:report['close_env']=parent.require(svc._sync_work('close_owned_env',lambda:(svc._close_env() or {'ok':True}),timeout=900),'close environment')
            except Exception:report['operational_errors'].append(traceback.format_exc())
            finally:svc.stop()
        pe.FINAL_ORACLE_GOALS.clear();pe.FINAL_ORACLE_GOALS.update(saved_oracle)
        report['source_unchanged']=parent.source_map()==sources
        report['fixed_files_unchanged']=all(parent.sha(Path(p))==h for p,h in fixed_files.items())
        try:parent.verify_facts(inputs['inputs_before']);report['inputs_unchanged']=True
        except Exception:report['inputs_unchanged']=False;report['operational_errors'].append(traceback.format_exc())
        try:parent.verify_facts(inputs['user8_before']);report['user8_unchanged']=True
        except Exception:report['user8_unchanged']=False;report['operational_errors'].append(traceback.format_exc())
        report['protocol_gate_observed']=report['executed_cases']<=2 and all(t['gate_pass'] is True for t in report['cases'][:-1])
        report['ok']=not report['operational_errors'] and report['source_unchanged'] and report['fixed_files_unchanged'] and report['inputs_unchanged'] and report['user8_unchanged'] and report['protocol_gate_observed']
        pe._write_json_atomic(output/'report.json',parent.clean(report))
    return 0 if report['ok'] else 2


if __name__=='__main__':raise SystemExit(main())
