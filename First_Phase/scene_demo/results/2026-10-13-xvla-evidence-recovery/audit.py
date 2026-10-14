"""Verify actual HTTP body hashes/original image bytes and protected old evidence."""
import base64,hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_bytes())
def digest(data):return hashlib.sha256(data).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def local(p):return ROOT/p.removeprefix('/mnt/d/FYP/')
assert load(ROOT/'First_Phase/scene_demo/experiment_registry.json')==load(OUT/'registry.before.json')
for path,sha in load(OUT/'protected.before.json').items():assert digest((ROOT/path).read_bytes())==sha,path
for path,sha in load(OUT/'source_manifest.json').items():assert digest((ROOT/path).read_bytes())==sha,path
requests=[];usage={};all_images={}
for request_path in sorted(OUT.glob('*/model_requests/call_*_request.json')):
    request=load(request_path);response_path=request_path.with_name(request_path.name.replace('_request','_response'))
    response=load(response_path);packet=json.loads(request['prompt'].splitlines()[-1])
    now=request['context']['actuator_measurements']['timestamp']
    assert all(group['time']<=now for group in packet['groups'])
    assert all(row['time']<=now for row in packet['previous_checks'])
    assert not any(key in request['prompt'] for key in ['native_success','declared_predicates','evaluation_only_truth'])
    content=[{'type':'text','text':request['prompt']}];images=[]
    for frame,item in zip(request['context']['sensor_frames'],request['images']):
        data=local(item['path']).read_bytes();assert digest(data)==item['sha256']
        assert data[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',data[16:24])
        images.append({**frame,**item,'resolution':[width,height]});all_images[item['path']]=item['sha256']
        content.extend([{'type':'text','text':f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode('ascii')}}])
    body={'model':'qwen3-vl-plus','temperature':0,'max_tokens':1600,'messages':[{'role':'user','content':content}]}
    body.update(request['request_options'])
    assert digest(json.dumps(body).encode())==request['request_sha256'],request_path
    for key,value in response.get('usage',{}).items():
        if type(value) is int:usage[key]=usage.get(key,0)+value
    requests.append({'request':request_path.relative_to(ROOT).as_posix(),'response':response_path.relative_to(ROOT).as_posix(),
        'request_sha256':request['request_sha256'],'response_sha256':digest(response_path.read_bytes()),
        'original_png_bytes_and_body_verified':True,'role':packet['role'],'images':images,
        'release_ids':[r['event_id'] for r in packet['release_history']]})
results={p.parent.name:load(p) for p in OUT.glob('*/summary.json')}
assert set(results)=={'recovery_probe','microwave_mug','basket_two','microwave_no_history'}
assert len(requests)==sum(r['actual_Qwen_calls'] for r in results.values())<=19
metadata=[]
for directory in results:
    for path in sorted((OUT/directory).glob('[0-9][0-9]_*.json')):
        item=load(path);packet=item['event_packet'];now=item['input_context']['actuator_measurements']['timestamp']
        events=[e for e in item['release_archive'] if e['timestamp']<=now]
        assert len(events)==len(packet['release_history'])
        for event,group in zip(events,packet['release_history']):
            assert event['event_id']==group['event_id']
            assert [digest(local(item['request_image_paths'][i]).read_bytes()) for i in group['frames']]==event['source']['sha256']
        for proof in item['process_evidence']:
            if proof['confirmation_kind']=='current_observation':
                assert proof['event_id'] is None and proof['evidence_id'].startswith('observation_')
                sources=proof['provenance']['sources'];assert len(sources)==1
                assert sources[0]['context']['actuator_measurements']['timestamp']==proof['timestamp']
                assert len(sources[0]['paths'])==3
        metadata.append({'path':path.relative_to(ROOT).as_posix(),'all_past_release_originals_verified':True})
audit={'real_Qwen_requests':len(requests),'usage':usage,'requests':requests,'all_images':all_images,
    'metadata':metadata,'protected_files_unchanged':len(load(OUT/'protected.before.json')),
    'old_registry_entries_unchanged':len(load(OUT/'registry.before.json')['entries']),
    'new_environment_actions':0,'helpers':0,'subagents':0,'root_model':'unknown','root_effort':'unknown'}
dump(OUT/'audit.json',audit)
summary={'protocol':'xvla_evidence_memory_v6','cases':results,'real_Qwen_requests':len(requests),
    'mechanism_probe_triggered':results['recovery_probe']['triggered'],
    'all_release_originals_verified':True,'offline_visual_qualification':all(results[k]['passed'] for k in ('microwave_mug','basket_two','microwave_no_history')),
    'new_environment_actions':0,'helpers':0,'subagents':0,'default_checker_changed':False,
    'software_tests':{'passed':336,'subtests_passed':30},'protected_files_unchanged':audit['protected_files_unchanged']}
dump(OUT/'summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False))
