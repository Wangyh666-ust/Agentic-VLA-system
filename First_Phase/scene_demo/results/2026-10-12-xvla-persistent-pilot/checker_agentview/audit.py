"""Three saved-image checks, without environment or X-VLA construction."""
import copy
import hashlib
import json
import os
from pathlib import Path
for key in list(os.environ):
    if key.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):
        os.environ.pop(key,None)
from scene_demo.finish_v1.simple_checker import SimpleQwenClient, build_prompt

OUT=Path(__file__).resolve().parent
JOB=Path('/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4')

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    if (OUT/'result.json').exists() or (OUT/'calls').exists():
        raise FileExistsError('refuse to overwrite existing audit results')
    checks=[json.loads(line) for line in (JOB/'finish/checks.jsonl').read_text().splitlines()]
    assert len(checks)==2 and all('ctx' in row for row in checks)
    requests=[next((JOB/'finish').rglob('call_%02d_request.json'%i)) for i in (1,2)]
    original=[json.loads(p.read_text())['context'] for p in requests]
    assert original==[row['ctx'] for row in checks]
    native=JOB.parent/'native/native_steps.jsonl'
    row=json.loads(native.read_text().splitlines()[29])
    assert row['counts']=={'settling':10,'prepare':15,'vla':5,'finish':0}
    assert row['sim_time']==1.500000000000001
    contexts=[copy.deepcopy(original[0]),copy.deepcopy(original[0]),copy.deepcopy(original[1])]
    ctx=contexts[0]
    ctx['actuator_measurements'].update(timestamp=row['sim_time'],finger_qpos=row['finger_qpos'],max_abs_arm_qvel=max(abs(v) for v in row['arm_qvel']))
    ctx['sensor_frames']=[{'timestamp':row['sim_time'],'view':'wrist','frame':0},{'timestamp':row['sim_time'],'view':'agentview','frame':1}]
    for ctx in contexts[1:]:
        ctx['sensor_frames'][1]['view']='agentview'
    image_names=[['cap000002_wrist.png','cap000001_agentview.png'],['cap000092_wrist.png','cap000091_agentview.png'],['cap000263_wrist.png','cap000262_agentview.png']]
    names=['not_placed','just_arrived','after_home']
    images=[[str(JOB/'finish/sensors'/filename) for filename in pair] for pair in image_names]
    sources=[JOB/'finish/checks.jsonl',native,*requests,JOB/'pilot_result.json',JOB/'independent_finish.jsonl']+[Path(p) for pair in images for p in pair]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    save(OUT/'source_contexts.json',{'source_sha256':hashes,'cases':[{'name':name,'context':ctx,'image_paths':paths,'prompt':build_prompt(ctx)} for name,ctx,paths in zip(names,contexts,images)]})
    for ctx in contexts:
        ts=ctx['actuator_measurements']['timestamp']
        assert all(frame['timestamp']==ts for frame in ctx['sensor_frames'])
    client=SimpleQwenClient(max_calls=3)
    client.audit_dir=OUT/'calls'
    result={'new_physical_actions':0,'xvla_calls':0,'qwen_calls':0,'cases':[],'all_match':False}
    try:
        for name,ctx,paths,expected in zip(names,contexts,images,[False,True,True]):
            response=client.analyze(ctx,paths)
            actual=response['visual_goal_verdict'].get('goal_achieved')
            result['cases'].append({'name':name,'expected':expected,'actual':actual,'match':actual is expected,'usage':response.get('usage'),'response':response})
            result['qwen_calls']=client._calls
            save(OUT/'result.json',result)
        result['all_match']=all(case['match'] for case in result['cases'])
    except BaseException as exc:
        result['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        result['qwen_calls']=client._calls
        save(OUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
        for case in result['cases']:
            print(json.dumps({k:case[k] for k in ('name','expected','actual','match','usage')},ensure_ascii=False))

if __name__=='__main__':
    main()
