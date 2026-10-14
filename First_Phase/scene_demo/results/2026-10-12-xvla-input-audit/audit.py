"""Rebuild historical HTTP bodies using mocked transport only."""
import base64
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from scene_demo.finish_v1 import vision
from scene_demo.finish_v1.simple_checker import SimpleQwenClient

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
BASE=OUT.parent
PILOT=BASE/'2026-10-12-xvla-persistent-pilot'
GROUND=BASE/'2026-10-12-xvla-checker-grounding'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(data):return hashlib.sha256(data).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    frozen={}
    for version in ['v4','v5']:
        manifest=read(PILOT/version/'source_manifest.json')
        for f in ['First_Phase/scene_demo/finish_v1/vision.py','First_Phase/scene_demo/finish_v1/simple_checker.py']:
            frozen[version+'/'+f]=sha((ROOT/f).read_bytes())==manifest[f]
    gm=read(GROUND/'source_manifest.json')
    for name in ['xvla_checker.py']:
        f=ROOT/'First_Phase/scene_demo'/name
        frozen[name]=sha(f.read_bytes())==gm[str(f)]
    assert all(frozen.values()),frozen
    paths=[]
    for v,case in [('v4','wine_bowl_stove'),('v5','basket_two'),('v5','mugs_two'),('v5','microwave_mug')]:
        paths += [('online',p) for p in (PILOT/v/case/'runs').glob('*/cap_*/finish/model_requests/call_*_request.json')]
    for name in ['checker_agentview','checker_rack_description']:
        paths += [('offline_old',p) for p in (PILOT/name/'calls').rglob('call_*_request.json')]
    paths += [('offline_grounding',p) for p in (GROUND/'calls').rglob('call_*_request.json')]
    assert len(paths)==33,len(paths)
    (OUT/'actual_prompts').mkdir(exist_ok=True)
    output=[]
    prepared={c['name']:c for c in read(GROUND/'cases.json')}
    for number,(kind,path) in enumerate(sorted(paths,key=lambda x:str(x[1]))):
        record=read(path);response_path=path.with_name(path.name.replace('_request','_response'))
        response_bytes=response_path.read_bytes();response=json.loads(response_bytes)
        client=SimpleQwenClient(model=response['model'],max_calls=1)
        client._resolve_key=lambda:'offline-placeholder'
        client.record_response=lambda *args,**kwargs:None
        client.request_options=record['request_options']
        captured=[]
        class MockResponse:
            def getcode(self):return record['http_status']
            def read(self):return response_bytes
        def intercept(req,timeout):
            captured.append(req.data);return MockResponse()
        image_paths=[i['path'] for i in record['images']]
        with patch.object(vision.urllib.request,'urlopen',side_effect=intercept):
            client._request(record['context'],image_paths,record['prompt'])
        assert len(captured)==1
        body=json.loads(captured[0]);content=body['messages'][0]['content']
        frames=record['context']['sensor_frames'];measurement=record['context']['actuator_measurements']
        images=[]
        urls=[c['image_url']['url'] for c in content if c['type']=='image_url']
        labels=[c['text'] for c in content[1:] if c['type']=='text']
        for i,(url,image,frame) in enumerate(zip(urls,record['images'],frames)):
            actual=base64.b64decode(url.split(',',1)[1]);disk=Path(image['path']).read_bytes()
            expected_label=f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"
            images.append({'path':image['path'],'actual_png_sha256':sha(actual),'record_sha256':image['sha256'],
                           'disk_sha256':sha(disk),'bytes_match':sha(actual)==image['sha256']==sha(disk),
                           'dimensions':list(Image.open(io.BytesIO(actual)).size),'camera_label':labels[i],
                           'label_matches':labels[i]==expected_label,'frame':frame})
        check=None;native=None;finger_match=None;speed_match=None;standard_source=None
        if kind=='online':
            job=path.parents[2];index=int(path.stem.split('_')[1])
            check=next(c for c in rows(job/'finish/checks.jsonl') if c['call_index']==index)
            finger_match=measurement['finger_qpos']==check['robot']['finger_qpos']
            speed_match=measurement['max_abs_arm_qvel']==max(abs(x) for x in check['robot']['arm_qvel'])
            standard_source='original checks.ctx actuator_measurements.standard_open_finger_qpos'
        elif kind=='offline_grounding':
            case=prepared[path.parent.name];capture=case['capture']
            finger_match=measurement['finger_qpos']==capture['robot']['finger_qpos']
            source=case['source_context']['actuator_measurements'];speed_match=measurement.get('max_abs_arm_qvel')==source.get('max_abs_arm_qvel')
            standard_source='prepared source_context from original first check standard_open reference'
        else:
            standard_source=None
        # Parse only the appended input object; prompt instructions may mention forbidden concepts.
        decoder=json.JSONDecoder();payload=None
        for position in range(len(record['prompt'])):
            if record['prompt'][position]!='{':continue
            try:
                obj,end=decoder.raw_decode(record['prompt'][position:])
                if not record['prompt'][position+end:].strip() and isinstance(obj,dict):payload=obj;break
            except json.JSONDecodeError:pass
        def keys(value):
            if isinstance(value,dict):return set(value)|set().union(*(keys(x) for x in value.values()),set())
            if isinstance(value,list):return set().union(*(keys(x) for x in value),set())
            return set()
        fields=keys(payload)
        prompt_path=OUT/'actual_prompts'/f'{number:02d}_{kind}_{path.parent.name}_{path.stem}.txt'
        prompt_path.write_text(record['prompt'],encoding='utf-8')
        matched=sha(captured[0])==record['request_sha256'] and all(i['bytes_match'] and i['label_matches'] for i in images) and content[0]['text']==record['prompt'] and len(body['messages'])==1 and body['messages'][0]['role']=='user' and all(f['timestamp']==measurement['timestamp'] for f in frames)
        output.append({'kind':kind,'request_path':str(path),'response_path':str(response_path),'body_sha_matches':sha(captured[0])==record['request_sha256'],
                       'body_sha256':sha(captured[0]),'prompt_exact':content[0]['text']==record['prompt'],'prompt_path':str(prompt_path),
                       'message_count':len(body['messages']),'prior_assistant_messages':any(m['role']=='assistant' for m in body['messages']),
                       'images':images,'image_encoder_timestamp':measurement['timestamp'],'timestamps_exact':all(f['timestamp']==measurement['timestamp'] for f in frames),
                       'encoder_finger_matches_source':finger_match,'speed_matches_source':speed_match,'standard_open_source':standard_source,
                       'measurements':measurement,'input_json_fields':sorted(fields),'full_chain_user_request_sent':'user_request' in fields,
                       'risk_fields':sorted(fields&{'phase','stage','predicates','declared_predicates','contacts','goal_scores'}),
                       'serialized_input_matches':matched,'feedback_source_unknown':finger_match is None,'context_source':record.get('source_context')})
    summary={kind:{'count':sum(r['kind']==kind for r in output),'matched':sum(r['kind']==kind and r['serialized_input_matches'] for r in output),
                   'feedback_unknown':sum(r['kind']==kind and r['feedback_source_unknown'] for r in output)} for kind in ['online','offline_old','offline_grounding']}
    result={'real_model_calls':0,'physical_actions':0,'mock_http_serializations':33,'frozen_serializer_matches':frozen,'groups':summary,'rows':output,
            'limits':['Historical bytes match establishes local input provenance only; provider internal image processing and model error causes remain unknown.','Old offline source feedback provenance not inferred from filename; missing source audit remains unknown.']}
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'groups':summary,'all_body_matches':all(r['body_sha_matches'] for r in output),'all_image_label_time_matches':all(r['serialized_input_matches'] for r in output),'dimensions':sorted({tuple(i['dimensions']) for r in output for i in r['images']}),'online_user_request_count':sum(r['kind']=='online' and r['full_chain_user_request_sent'] for r in output),'risk_fields':sorted(set().union(*(set(r['risk_fields']) for r in output))),'real_model_calls':0,'physical_actions':0},ensure_ascii=False))


if __name__=='__main__':main()
