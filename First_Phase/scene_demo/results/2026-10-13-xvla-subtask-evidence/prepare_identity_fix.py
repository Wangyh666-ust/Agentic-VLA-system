import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=(OUT/'followup.py').read_text(encoding='utf-8')
source=source.replace("OUT/'final_source_manifest.json'","OUT/'identity_source_manifest.json'")
source=source.replace("directory=OUT/('v7_1_'+mode)","directory=OUT/('v7_1_'+mode+'_identityfix')")
anchor="client.set_check_contract(checks[0]['ctx']);client.audit_dir=directory/'model_requests'"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'''
assert mode=='microwave'
original=client._request;client.reused_prefix=0
cached=sorted((OUT/'v7_1_microwave/model_requests').glob('call_*_request.json'))
assert len(cached)==2
def bounded(context,paths,prompt):
    if client.reused_prefix<len(cached):
        path=cached[client.reused_prefix];request=load(path)
        assert request['prompt']==prompt and request['context']==context
        assert [str(p) for p in paths]==[i['path'] for i in request['images']]
        for item in request['images']:assert sha(item['path'])==item['sha256']
        rp=path.with_name(path.name.replace('_request','_response'));reply=load(rp)
        client._calls+=1;client.reused_prefix+=1
        return {'verdict':client.parse_verdict(reply['choices'][0]['message']['content']),
            'request_sha256':request['request_sha256'],'response_sha256':sha(rp),
            'call_index':client._calls,'usage':reply['usage'],'model':reply['model'],'http_status':request['http_status']}
    assert client._calls-client.reused_prefix<1,'one existing-response protocol repair only'
    return original(context,paths,prompt)
client._request=bounded
''')
source=source.replace("'actual_Qwen_calls':client._calls,'error':error,'records':records,","'actual_Qwen_calls':client._calls-client.reused_prefix,'reused_same_campaign_requests':client.reused_prefix,'error':error,'records':records,")
driver=OUT/'identity_fix.py';driver.write_text(source,encoding='utf-8')
proposal=load(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-protocol-followup.json')
proposal['id']='20261013-xvla-subtask-evidence-identity-fix'
proposal['analysis_steps'].append('v7.1入炉旧回答给目标换了初态参照视角而未保留旧锚点；原聚合只拒绝，未提供既定的一次协议修复。通用修正允许带旧锚点链接的新视角增补，并把无链接明确事实交给每任务一次的协议修复；保留有效门状态。严格复用已保存两次请求，prompt/context/PNG SHA逐项相等才复用；仅追加最多一次实际缺链修复，不重采样篮子或更改识别阈值。')
proposal['scope']['identity_fix']={'attempts_so_far':21,'additional_requests_max':1,'same_campaign_prefix_reuse':2,'new_environment_actions':0}
path=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-identity-fix.json';dump(path,proposal)
cmd=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP/First_Phase','/home/yhwang/fyp/xvla_native/venv/bin/python','-m','pytest','First_Phase/scene_demo/tests','-q','-k','xvla']
r=subprocess.run(cmd,capture_output=True,cwd=ROOT)
(OUT/'pytest-identity.stdout.txt').write_bytes(r.stdout);(OUT/'pytest-identity.stderr.txt').write_bytes(r.stderr)
print(r.stdout.decode());assert r.returncode==0
files={ROOT/p for p in load(OUT/'final_source_manifest.json')}|{driver,path}
manifest={};snap=OUT/'identity_source_snapshot';snap.mkdir(exist_ok=False)
for file in sorted(files):
    dest=snap/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    manifest[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
dump(OUT/'identity_source_manifest.json',manifest)
cmd=['D:/miniconda/python.exe','-X','utf8','.agents/skills/fyp-experiment-review/scripts/review.py','check','--proposal',str(path)]
r=subprocess.run(cmd,capture_output=True,cwd=ROOT)
(OUT/'gate-identity.stdout.txt').write_bytes(r.stdout);(OUT/'gate-identity.stderr.txt').write_bytes(r.stderr)
print(r.stdout.decode());assert r.returncode==0
