"""Actual Qwen requests on immutable old PNGs; zero simulation/actions."""
import copy, hashlib, json, sys, shutil
from pathlib import Path
from scene_demo.xvla_event_checker import XvlaEventQwenClient

ROOT=Path('/mnt/d/FYP');OUT=Path(__file__).resolve().parent
OLD=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial'
def load(p):return json.loads(Path(p).read_bytes())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def cap(ctx, paths):
    assert len(paths)==3
    return {'timestamp':ctx['actuator_measurements']['timestamp'],
            'robot':{'finger_qpos':ctx['actuator_measurements']['finger_qpos']},
            'frames':[{'view':f['view'],'rgb_path':p} for f,p in zip(ctx['sensor_frames'],paths)]}

def prepare(case,mode):
    folder=OLD/'microwave_mug' if case=='microwave_mug' else OLD/'basket_independent/basket_two'
    finish=next(folder.rglob('process_events.jsonl')).parent
    checks=rows(finish/'checks.jsonl'); releases=rows(finish/'process_checks.jsonl')
    if mode=='negative':
        selected=releases[0];ctx=selected['context'];paths=[r['path'] for r in rows(finish/'process_events.jsonl')[0]['images']]
    else:
        selected=checks[-1];ctx=selected['ctx'];paths=selected['image_paths']
    time=ctx['actuator_measurements']['timestamp']
    client=XvlaEventQwenClient(max_calls=2);client.set_check_contract(ctx)
    refs=[finish/'sensors'/('cap%06d_%s.png'%(i,v)) for i,v in [(2,'wrist'),(1,'agentview'),(3,'sideview')]]
    assert all(p.exists() for p in refs)
    client.remember_identity_images(refs,client.required_views)
    stream=[]
    for row in checks:
        c=row['ctx']; t=c['actuator_measurements']['timestamp']
        if t<time:stream.append((t,'observation',row))
    if mode!='no_history':
        for row in releases:
            t=row['capture']['timestamp']
            if t<=time:stream.append((t,'release',row))
    for t,kind,row in sorted(stream,key=lambda x:x[0]):
        if kind=='release':client.remember_release(row['capture'],row['context'],row['event_id'])
        else:
            client.remember_observation(cap(row['ctx'],row['image_paths']),row['ctx'])
            old=row['initial_check']['metadata']['effective_checks']
            client._states.append({'time':t,'source':'historical_checker_unverified',
                'goals':[{k:r[k] for k in ('id','achieved','evidence_event_ids')} for r in old]})
    # These are sparse, recorded checkpoints, not invented near-release timestamps.
    client.observe_capture(cap(ctx,paths))
    return client,ctx,paths,finish

attempt=sys.argv[1] if len(sys.argv)>1 else 'attempt_01'
dest=OUT/attempt;dest.mkdir(exist_ok=False)
sources={}
for name in ('xvla_event_checker.py','xvla_monitor.py','xvla_service.py','xvla_pilot.py'):
    p=ROOT/'First_Phase/scene_demo'/name;target=dest/'source'/name;target.parent.mkdir(exist_ok=True)
    shutil.copyfile(p,target);sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
dump(dest/'source_manifest.json',sources)
cases=[('microwave_mug','positive'),('microwave_mug','negative'),('basket_two','positive'),
       ('basket_two','negative'),('microwave_mug','no_history')]
results=[]
for case,mode in cases:
    directory=dest/(case+'_'+mode);directory.mkdir()
    client,ctx,paths,source=prepare(case,mode);client.audit_dir=directory
    try:
        result=client.analyze(ctx,paths)
        dump(directory/'result.json',result)
        achieved=result['visual_goal_verdict']['goal_achieved']
        effective=result['effective_checks']
        reuse=[r['id'] for r in effective if r.get('history_event_id')]
        passed=(achieved is True and bool(reuse)) if mode=='positive' else achieved is not True
        if case=='microwave_mug' and mode=='positive':passed=passed and effective[1]['current_evidence'] is True and effective[0]['history_event_id'] is not None
        if case=='basket_two' and mode=='positive':passed=passed and len({r['object_key'] for r in effective})==2
        row={'case':case,'mode':mode,'passed':passed,'goal':achieved,'reuse':reuse,
            'checks':effective,'image_count':len(result['request_image_paths']),
            'event_count':len(result['event_packet']['events']),'usage':result.get('usage')}
    except Exception as exc:
        row={'case':case,'mode':mode,'passed':False,'error':type(exc).__name__+': '+str(exc)}
        dump(directory/'error.json',row)
    results.append(row);dump(dest/'summary.json',{'passed':all(r['passed'] for r in results),'complete':len(results)==len(cases),'results':results,'environment_actions':0})
    print(json.dumps(row,ensure_ascii=False),flush=True)
print(json.dumps({'offline_gate_passed':all(r['passed'] for r in results),'requests':sum(len(list(d.glob('call_*_response.json'))) for d in dest.iterdir() if d.is_dir())}),flush=True)
