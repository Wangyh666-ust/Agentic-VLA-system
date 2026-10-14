"""One authorized historical-action replay, then isolated original finish control.

No VLA/Qwen requests. Final visual acceptance is deliberately UNKNOWN; the
independent retreat/native-state/home measurements gate the timed online run.
"""
import importlib.util, json, os, sys, time, traceback
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
scene=root/'First_Phase/scene_demo'
os.environ.update(json.loads((scene/'results/2026-10-14-qwen38-three-requests/runtime/table_tidy_stove_environment.json').read_text()))
for path in os.environ['PYTHONPATH'].split(':'):
    if path not in sys.path:sys.path.insert(0,path)
import numpy as np
from scene_demo.xvla_backend import NativeXvlaEnv, reference_for, reference_metrics
from scene_demo.xvla_service import bridge_port
from scene_demo.finish_v1.sensors import SensorPort
from scene_demo.finish_v1 import control, geometry, simple_checker
from scene_demo.xvla_grounding import attach_checker_cameras
from scene_demo.xvla_finish import build_skill_context
from scene_demo import joint_home, catalog, service, safe_exit_contacts, safe_exit

old=scene/'results/interactive-console/runs/20261014T122115Z_bowl_drawer_647bf4cb'
job=old/'cap_01_xvla_bowl_drawer_close_8e6a690c'
destination=out/(sys.argv[1] if len(sys.argv)>1 else 'replay')
destination.mkdir(exist_ok=False)
def save(name,data):
    (destination/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)))
def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]
native=rows(old/'native/native_steps.jsonl')
oldpilot=json.loads((job/'pilot_result.json').read_text())
reference=oldpilot['handoff_reference']
cid='xvla_bowl_drawer_close'
env=None
started=time.monotonic()
summary={'new_VLA_calls':0,'new_VLM_calls':0,'final_visual_acceptance':'not_run_unknown',
         'replay_source':str(old),'success':False}
try:
    env=NativeXvlaEnv('libero_10',3,142,0,destination/'native',session_limit=1500)
    env.reset()
    sim=env._env.env.sim
    initial=np.asarray(sim.get_state().flatten())
    original=np.load(old/'initial_state.npy')
    summary['initial_state_max_abs_error']=float(np.max(abs(initial-original)))
    save('matching.json',summary)
    assert summary['initial_state_max_abs_error']<1e-10,'initial state mismatch'
    preparation=reference_for(env,cid,reference)
    port=bridge_port(SensorPort(env,preparation,destination/'prepare_sensors'))
    waypoints=joint_home.joint_waypoints(port.read_robot()['arm_qpos'],preparation['homeq'],step_rad=.01)
    expected=[r for r in native if r['source']=='prepare']
    errors=[]
    for index, row in enumerate(expected):
        env.action_source='prepare'
        state=port.step(waypoints[min(index,len(waypoints)-1)],preparation['finger_home'])
        errors.append(float(np.max(abs(np.array(state['arm_qpos'])-row['arm_qpos']))))
    port.close()
    summary['prepare_max_joint_error']=max(errors)
    save('matching.json',summary)
    assert max(errors)<1e-7,'preparation replay mismatch'
    # Match XvlaSceneService._reset_policy_queues after _prepare_first, including
    # the measured gripper hold; restoring the old controller alone is not enough.
    robot=env._env.env.robots[0]
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    robot.controller.use_delta=False
    robot.gripper.current_action=np.asarray(safe_exit._finger_hold(env,joint_home.read_robot_state(env)),dtype=np.float32).copy()
    env._get_observation()
    errors=[]
    for row in [r for r in native if r['source']=='vla']:
        robot.controller.update(force=True)
        env.action_source='vla'
        env.step(row['action'])
        state=joint_home.read_robot_state(env)
        errors.append(max(float(np.max(abs(np.array(state[k])-row[k])))
                          for k in ('arm_qpos','arm_qvel','finger_qpos','finger_qvel')))
        summary['last_replayed_step']=len(errors)
        summary['last_encoder_error']=errors[-1]
        assert errors[-1]<1e-7,'historical action replay mismatch'
    summary['vla_replay_actions']=len(errors)
    summary['vla_max_encoder_error']=max(errors)
    save('matching.json',summary)
    print(json.dumps({'stage':'old_pose_matched',**summary}),flush=True)
    checks=rows(job/'finish/checks.jsonl')
    accepted=checks[0]['verdict']
    port=attach_checker_cameras(bridge_port(SensorPort(env,reference,destination/'sensors')))
    before=port.capture();save('before.json',before)
    context=build_skill_context(catalog.CAPABILITIES[cid],'收拾桌面，将碗和瓶子归位')
    measurements=[]
    def emit(row):
        predicates={ '|'.join(g):bool(service.eval_goal_predicate(env,g)) for g in catalog.CAPABILITIES[cid]['goals']}
        measured={**row,'predicates':predicates,'contacts':safe_exit_contacts.contacts(env),
                  'sim_time':float(sim.data.time)}
        measurements.append(measured)
        with (destination/'finish_actions.jsonl').open('a') as f:f.write(json.dumps(measured,default=lambda v:v.tolist())+'\n')
        if not all(predicates.values()):raise RuntimeError('native task disturbed during finish')
    # Compare old/new candidate generation at the exact capture used by the
    # original controller, after its unchanged release commands.
    spec=importlib.util.spec_from_file_location('old_retreat_geometry',out/'before/finish_v1/geometry.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    original_plan=control.plan_retreat
    def paired_plan(p,observation,verdict):
        oldplan=previous.plan_retreat(p,observation,verdict)
        newplan=original_plan(p,observation,verdict)
        save('old_plan.json',oldplan);save('new_plan.json',newplan);save('plan_observation.json',observation)
        assert not oldplan['ok'],'old failure not reproduced'
        assert [r['reason'] for r in oldplan['candidate_reports']]==['endpoint_hand_clear_fail']*2
        assert newplan['ok'],'new candidate not found'
        assert newplan['candidate_reports'][-1]['direction_source']=='current_geometry','new branch did not trigger'
        print(json.dumps({'stage':'new_exit_screened','chosen':newplan['candidate_reports'][-1]}),flush=True)
        return newplan
    control.plan_retreat=paired_plan
    unknown={**accepted,'operation_achieved':None,'at_destination':None,
             'evidence':['Isolated replay: final VLM not invoked; full acceptance reserved for online run.']}
    try:
        result=control.run_finish(port,context,accepted,simple_checker.decide,
            lambda context, observations:unknown,emit,simple_handoff=True)
    finally:control.plan_retreat=original_plan
    save('finish_result.json',result)
    after=port.capture();save('after.json',after)
    clearance=geometry.screen_shapes(port.obstacle_cloud(after['frames']),port.robot_shapes(after['robot']['arm_qpos']),
                                     [port.robot_shapes(after['robot']['arm_qpos'])])
    metrics=reference_metrics(env,reference)
    summary.update(counts=result['counts'],finish_reason=result['reason'],home_streak=result.get('home_streak'),
        final_metrics=metrics,final_clearance=clearance,native_preserved=all(all(r['predicates'].values()) for r in measurements),
        actual_retreat_actions=result['counts']['movement'],native_success=bool(env._env.check_success()),
        home_contacts=safe_exit_contacts.contacts(env),wall_seconds=time.monotonic()-started)
    summary['success']=bool(summary['actual_retreat_actions']>0 and result.get('home_streak')==5
        and metrics['ready'] and clearance['ok'] and summary['native_preserved'] and summary['native_success']
        and not summary['home_contacts'] and result['reason']=='final visual failed')
    save('summary.json',summary)
    print(json.dumps(summary),flush=True)
    assert summary['success'],'isolated retreat/home failed; do not run online'
except BaseException:
    summary.update(error=traceback.format_exc(),wall_seconds=time.monotonic()-started)
    save('summary.json',summary)
    raise
finally:
    if env is not None:env.close()
