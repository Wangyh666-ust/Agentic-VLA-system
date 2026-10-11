"""Archive fresh pilot records and videos. No robot or model calls."""
import json
from pathlib import Path
from .assisted_combo_pilot import ROOT, OUT, PLAN
from .combo_pilot import sha, write


def main():
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    cases=[]
    keep=[PLAN,OUT/'source_snapshot.zip']

    def refs(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('rgb_path','depth_path','calibration_path','json_path','path') and isinstance(item,str):
                    p=Path(item)
                    if p.is_file() and p.is_relative_to(OUT):
                        keep.append(p)
                        if p.suffix=='.npy' and p.with_suffix('.json').is_file():keep.append(p.with_suffix('.json'))
                refs(item)
        elif isinstance(value,list):
            for item in value:refs(item)

    for name in ('bowl_wine','bowl_stove'):
        directory=OUT/name
        if not (directory/'result.json').exists():continue
        result=json.loads((directory/'result.json').read_text())
        jobs=[]
        with imageio.get_writer(str(directory/'sequence.mp4'),fps=20) as writer:
            for sub in result['subtasks']:
                job=sub['job'];run=Path(job['run_dir']);finish=sub['finish'] or {}
                def jsonl(p):return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
                actions=jsonl(run/'finish_actions.jsonl')
                independent=jsonl(run/'independent_finish.jsonl')
                sources=jsonl(run/'action_sources.jsonl')
                source_time={round(row['timestamp'],6):row for row in sources}
                finish_sources=[row for row in sources if row['source']=='finish']
                finish_time={round(source['timestamp'],6):action['phase'] for source,action in zip(finish_sources,actions)}
                checks=[]
                for request in sorted(run.glob('**/model_requests/*_request.json')):
                    raw=json.loads(request.read_text())
                    response=json.loads(request.with_name(request.name.replace('_request','_response')).read_text())
                    ctx=raw['context'];t=ctx['actuator_measurements']['timestamp']
                    source=source_time.get(round(t,6),{})
                    phase=ctx['actuator_measurements'].get('stage','pregrasp')
                    checks.append({'kind':'pregrasp' if phase=='pregrasp' else 'online' if phase=='executing' else 'after_cleanup',
                        'actual_vla':source.get('actual_vla',0),'timestamp':t,'content':response['choices'][0]['message']['content'],
                        'usage':response.get('usage',{}),'request_path':request.relative_to(ROOT).as_posix(),
                        'images':raw['images'],'stage_sent':'"stage"' in raw['prompt'] or '"phase"' in raw['prompt']})
                    prompt=request.with_suffix('.txt')
                    prompt.write_text(raw['prompt']+'\n'+'\n'.join(
                        f"Image frame={fr['frame']}; view={fr['view']}; timestamp={fr['timestamp']}; path={img['path']}; sha256={img['sha256']}"
                        for fr,img in zip(ctx['sensor_frames'],raw['images']))+'\n',encoding='utf-8')
                    keep.append(prompt);refs(raw)
                grasp_path=run/'grasp_assist.json'
                grasp=json.loads(grasp_path.read_text()) if grasp_path.exists() else None
                refs(grasp);refs(finish)
                timeline=[{'source':'initial','actual_vla':0,'local_grasp_actions':0}]
                helper_count=0
                for source in sources:
                    if source['source']=='finish':
                        helper_count+=1
                        if helper_count%5:continue
                    timeline.append(dict(source,finish_phase=finish_time.get(round(source.get('timestamp',0),6))))
                count=0
                if job.get('rollout_path'):
                    reader=imageio.get_reader(job['rollout_path'])
                    try:
                        for frame in reader:
                            row=timeline[count] if count<len(timeline) else {}
                            preview=Image.fromarray(frame);draw=ImageDraw.Draw(preview)
                            label='NEW '+sub['capability']+' | '+str(row.get('finish_phase') or row.get('source',''))
                            draw.rectangle((0,0,preview.width,29),fill='black')
                            draw.text((2,1),label,fill='white')
                            draw.text((2,15),f"VLA {row.get('actual_vla','?')} / grasp {row.get('local_grasp_actions','?')}",fill='white')
                            writer.append_data(np.asarray(preview))
                            if count==len(timeline)-1:imageio.imwrite(run/'final_frame.png',frame)
                            count+=1
                    finally:reader.close()
                jobs.append({'capability':sub['capability'],'actual_vla':sub['actual_vla'],
                    'local_grasp_actions':sub['local_grasp_actions'],'success':job['success'],
                    'reason':job['ended_reason'],'error':job.get('error'),
                    'finish_counts':finish.get('counts',{}),
                    'open_and_retreat_actions':sum(a.get('phase')=='movement' and a.get('finger_target') is not None for a in actions),
                    'concurrent_open_retreat':finish.get('concurrent_open_retreat'),
                    'home_streak':finish.get('home_streak',0),'independent_streak':sub['independent_streak'],
                    'final_independent':independent[-1] if independent else None,
                    'grasp':grasp,'qwen':checks,'decoded_frames':count,'timeline_frames':len(timeline)})
        cases.append({'case':name,'state':result['plan']['state'],'plan_success':result['plan']['plan_success'],
            'jobs':jobs,'wall_seconds':result['wall_seconds'],'video':(directory/'sequence.mp4').relative_to(ROOT).as_posix()})
        keep.extend(p for p in directory.rglob('*') if p.is_file() and 'sensors' not in p.parts)
    summary={'id':'20261011-assisted-combos','cases':cases,
        'new_vla':sum(j['actual_vla'] for c in cases for j in c['jobs']),
        'local_grasp_actions':sum(j['local_grasp_actions'] for c in cases for j in c['jobs']),
        'finish_actions':sum(sum(j['finish_counts'].values()) for c in cases for j in c['jobs']),
        'qwen_calls':sum(len(j['qwen']) for c in cases for j in c['jobs']),
        'qwen_tokens':sum(q['usage'].get('total_tokens',0) for c in cases for j in c['jobs'] for q in j['qwen']),
        'replay_actions':0,'subagents':0,'scope':'Fixed user sequences; no Hermes planning evaluation; no training',
        'not_run':[n for n in ('bowl_wine','bowl_stove') if not (OUT/n/'result.json').exists()]}
    write(OUT/'summary.json',summary)
    keep.extend(p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json')
    write(OUT/'manifest.json',{'tracked_files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(set(keep))]})
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False))
    for case in cases:
        print(json.dumps({'case':case['case'],'success':case['plan_success'],'jobs':[
            {k:v for k,v in j.items() if k not in ('grasp','qwen','final_independent')} for j in case['jobs']]},ensure_ascii=False))


if __name__ == '__main__':main()
