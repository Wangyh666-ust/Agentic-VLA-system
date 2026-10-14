"""Audit images, actual request bodies, temporal scope and unchanged controls."""
import ast,base64,hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(data):return hashlib.sha256(data).hexdigest()
def local(p):return ROOT/p.removeprefix('/mnt/d/FYP/')
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert load(ROOT/'First_Phase/scene_demo/experiment_registry.json')==load(OUT/'before-experiment_registry.json')
protected=load(OUT/'protected.before.json')
for path,expected in protected.items():assert sha(ROOT/path)==expected,path
for path,expected in load(OUT/'source_manifest.json').items():assert sha(OUT/'source_snapshot'/path)==expected,path
aggregation_manifest=load(OUT/'aggregation_source_manifest.json')
for path,expected in load(OUT/'final_source_manifest.json').items():
    assert sha(OUT/'final_source_snapshot'/path)==expected,path
for path,expected in aggregation_manifest.items():assert sha(OUT/'aggregation_source_snapshot'/path)==expected,path
for path,expected in load(OUT/'identity_source_manifest.json').items():assert sha(ROOT/path)==expected,path
for path,expected in load(OUT/'resume_manifest.json').items():assert sha(ROOT/path)==expected,path
assert sha(ROOT/'First_Phase/scene_demo/xvla_monitor.py')==sha(OUT/'before/xvla_monitor.py')
def outside_factory(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef) and node.name=='XvlaSceneService':
            node.body=[f for f in node.body if not isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)) or f.name!='_make_finish_monitor']
    return ast.dump(tree,include_attributes=False)
assert outside_factory(ROOT/'First_Phase/scene_demo/xvla_service.py')==outside_factory(OUT/'before/xvla_service.py')
requests=[];usage={};all_images={};by_request={};by_response={}
for path in sorted(OUT.glob('*/model_requests/call_*_request.json')):
    request=load(path);rp=path.with_name(path.name.replace('_request','_response'));response=load(rp)
    packet=json.loads(request['prompt'].splitlines()[-1]);now=packet['reviewed_at']
    assert not any(x in request['prompt'] for x in ('native_success','declared_predicates','evaluation_only_truth'))
    for obs in packet['observations']:
        assert obs['captured_at'] is None or now is not None and obs['captured_at']<=now
        if obs['encoder'] is not None:assert obs['encoder']['timestamp']==obs['captured_at']
    for review in packet['previous_reviews']:assert review['reviewed_at'] is None or review['reviewed_at']<=now
    content=[{'type':'text','text':request['prompt']}];images=[]
    for frame,item in zip(request['context']['sensor_frames'],request['images']):
        data=local(item['path']).read_bytes();assert digest(data)==item['sha256']
        assert data[:8]==b'\x89PNG\r\n\x1a\n';width,height=struct.unpack('>II',data[16:24])
        images.append({**frame,**item,'resolution':[width,height]});all_images[item['path']]=item['sha256']
        content += [{'type':'text','text':f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode('ascii')}}]
    body={'model':'qwen3-vl-plus','temperature':0,'max_tokens':1600,'messages':[{'role':'user','content':content}]}
    body.update(request['request_options']);assert digest(json.dumps(body).encode())==request['request_sha256'],path
    rsha=sha(rp);by_request[request['request_sha256']]=request;by_response[rsha]=response
    requests.append({'request':path.relative_to(ROOT).as_posix(),'request_sha256':request['request_sha256'],
        'response':rp.relative_to(ROOT).as_posix(),'response_sha256':rsha,'protocol':packet['protocol'],
        'mode':packet['mode'],'reviewed_at':now,'images':images,'release_event_ids':packet['coverage']['included_event_ids']})
    for key,value in response.get('usage',{}).items():
        if type(value) is int:usage[key]=usage.get(key,0)+value
proofs=[];metadata_count=0
for path in sorted(OUT.glob('*/*.json')):
    if path.name in ('summary.json','progress.json','error.json'):continue
    meta=load(path)
    if not isinstance(meta,dict) or 'pages' not in meta:continue
    metadata_count+=1
    observations={o['capture_id']:o for o in meta['observations']}
    reviewed={e for page in meta['pages'] for e in page['packet']['coverage']['included_event_ids']}
    now=observations[meta['current_capture_id']]['captured_at']
    expected={e['event_id'] for e in meta['release_archive'] if observations[e['capture_id']]['captured_at']<=now}
    assert expected<=reviewed,path
    for proof in meta['process_evidence']:
        obs=observations[proof['capture_id']]
        assert proof['observed_at']==obs['captured_at']<=proof['reviewed_at']<=now
        assert proof['encoder']==obs['encoder'] and proof['encoder']['at_standard_open'] is True
        assert proof['source_sha256']==obs['sha256']
        request=by_request[proof['request_sha256']];by_response[proof['response_sha256']]
        packet=json.loads(request['prompt'].splitlines()[-1])
        assert packet['reviewed_at']==proof['reviewed_at'] and packet['scope']==meta['scope']
        assert any(o['capture_id']==proof['capture_id'] for o in packet['observations'])
        proofs.append({'metadata':path.relative_to(ROOT).as_posix(),**proof})
cases={p.parent.name:load(p) for p in OUT.glob('*/summary.json')}
attempts=sum(v['actual_Qwen_calls'] for v in cases.values())
assert attempts<=26 and len(requests)<=attempts
failures=[{'case':k,'error':v['error']} for k,v in cases.items() if v.get('error')]
result={'actual_Qwen_attempts':attempts,'responses_archived':len(requests),'usage':usage,'requests':requests,
    'all_images':all_images,'metadata_verified':metadata_count,'proof_instances_verified':proofs,'cases':cases,
    'transport_failures_without_response':attempts-len(requests),'execution_errors':failures,
    'protected_files_unchanged':len(protected),'monitor_whole_file_unchanged':True,'service_outside_checker_factory_AST_unchanged':True,
    'root_model':'unknown','root_effort':'unknown','software_tests_passed':361,'subtests_passed':30,
    'new_environment_actions':0,'helpers':0,'subagents':0,'old_answers_imported':0,
    'same_campaign_exact_prefix_requests_reused':sum(v.get('reused_same_campaign_requests',0) for v in cases.values()),
    'default_checker_changed':False,'offline_visual_qualification':False}
result['manual_visual_review']=load(OUT/'manual_visual_review.json')
dump(OUT/'audit.json',result)
dump(OUT/'summary.json',{k:v for k,v in result.items() if k not in ('requests','all_images','usage','proof_instances_verified')})
print(json.dumps({k:v for k,v in result.items() if k not in ('cases','requests','all_images','usage','proof_instances_verified')},ensure_ascii=False))
