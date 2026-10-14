"""Preserve failed replay, fix only source frame extraction, reuse exact prefix replies."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=(OUT/'replay.py').read_text(encoding='utf-8')
def replace(a,b):
    global source
    assert source.count(a)==1,a;source=source.replace(a,b)
replace('from scene_demo.xvla_subtask_evidence import SubtaskEvidence,PROTOCOL','from scene_demo.xvla_subtask_evidence import SubtaskEvidence,PROTOCOL\nfrom scene_demo.xvla_process_evidence import capture_frames')
replace("epaths=[str(f['rgb_path']) for f in event['capture']['frames']]","epaths=[str(f['rgb_path']) for f in capture_frames(event['capture'])]")
replace("'logical_subtask_id':directory.name","'logical_subtask_id':mode")
replace("original=client._request\n    def bounded(*args,**kwargs):","""original=client._request
    cached=sorted((OUT/mode/'model_requests').glob('call_*_request.json')) if directory.name!=mode else []
    client.reused_prefix=0
    def bounded(*args,**kwargs):
        if client.reused_prefix<len(cached):
            request_path=cached[client.reused_prefix];request=load(request_path)
            context,paths,prompt=args
            assert request['prompt']==prompt and request['context']==context
            assert [str(p) for p in paths]==[i['path'] for i in request['images']]
            for item in request['images']:assert sha(item['path'])==item['sha256']
            response_path=request_path.with_name(request_path.name.replace('_request','_response'));response=load(response_path)
            client._calls+=1;client.reused_prefix+=1
            return {'verdict':client.parse_verdict(response['choices'][0]['message']['content']),
                'usage':response['usage'],'model':response['model'],'http_status':request['http_status'],
                'request_sha256':request['request_sha256'],'response_sha256':sha(response_path),
                'call_index':client._calls}
""")
replace("finish,checks,releases,truth=dataset(case);directory=OUT/mode","""finish,checks,releases,truth=dataset(case);directory=OUT/(mode+'_resume' if mode in ('microwave_mug','basket_two') else mode)
for event in releases:
    ordered=capture_frames(event['capture'])
    assert [f['view'] for f in ordered]==list(XvlaSubtaskReviewClient.required_views)
    assert len(ordered)==3
for path,digest in load(OUT/'resume_manifest.json').items():assert sha(ROOT/path)==digest,path""")
replace("'actual_Qwen_calls':client._calls,'false_positives':false_positives,","'actual_Qwen_calls':client._calls-client.reused_prefix,'reused_same_campaign_requests':client.reused_prefix,'logical_checks_including_reuse':client._calls,'false_positives':false_positives,")
target=OUT/'replay_resume.py';assert not target.exists();target.write_text(source,encoding='utf-8')
proposal=json.loads((ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-execution.json').read_bytes())
proposal['id']='20261013-xvla-subtask-evidence-replay-fix'
proposal['analysis_steps'].append('旧日志release capture.frames只含两路，第三路在verification_frames；回放工具统一使用已有capture_frames，先静态核对所有事件三路顺序。冻结生产检查器不变；已完成的本轮相同请求按prompt/context/原图SHA严格核对后复用，不重采样。零新增动作，原入口错误记录保留；总26请求预算不变。')
proposal['scope']['resume_prefix_reuse']='microwave_mug已有3次实际请求严格逐项复用；basket_two入口失败发生在零模型调用。只继续尚未执行的请求。'
path=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-replay-fix.json';dump(path,proposal)
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (target,path)}
dump(OUT/'resume_manifest.json',manifest)
print(json.dumps({'runner_frozen':str(target),'production_changed':False,'same_campaign_prefix_reuse':True}))
