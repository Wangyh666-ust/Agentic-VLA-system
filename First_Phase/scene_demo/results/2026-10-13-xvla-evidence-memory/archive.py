"""Audit real request bytes, then seal this offline-only campaign."""
import base64,hashlib,json,shutil,struct,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;SCENE=ROOT/'First_Phase/scene_demo'
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rel(p):return p.relative_to(ROOT).as_posix()
def local(p):return ROOT/p.removeprefix('/mnt/d/FYP/') if p.startswith('/mnt/d/FYP/') else Path(p)

if sys.argv[1]=='audit':
    assert load(OUT/'revision-exit.json')['replay_returncode']==0
    assert load(OUT/'completion-exit.json')['returncode']==0
    assert (OUT/'tests-final.stdout').read_text().find('319 passed, 30 subtests passed')>=0
    assert load(OUT/'review-v2-windows.stdout')['ok']
    assert load(OUT/'review-v3.stdout')['ok']
    protected=load(OUT/'protected.before.json')
    assert all(sha(ROOT/p)==s for p,s in protected.items())
    assert not list(OUT.rglob('pilot_result.json'))
    packets=[];usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    for request in sorted(list(OUT.glob('attempt_*/**/call_*_request.json'))+list((OUT/'continuation').glob('**/call_*_request.json'))):
        r=load(request);response=request.with_name(request.name.replace('_request','_response'));raw=load(response)
        content=[{'type':'text','text':r['prompt']}];images=[]
        assert len(r['context']['sensor_frames'])==len(r['images'])
        for frame,item in zip(r['context']['sensor_frames'],r['images']):
            p=local(item['path']);data=p.read_bytes()
            assert hashlib.sha256(data).hexdigest()==item['sha256'] and data[:8]==b'\x89PNG\r\n\x1a\n'
            content.extend([{'type':'text','text':f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode('ascii')}}])
            images.append({**frame,**item,'resolution':struct.unpack('>II',data[16:24])})
        body={'model':'qwen3-vl-plus','temperature':0,'max_tokens':1600,'messages':[{'role':'user','content':content}]}
        body.update(r['request_options'])
        assert hashlib.sha256(json.dumps(body).encode()).hexdigest()==r['request_sha256']
        assert r['http_status']==200
        packet=json.loads(r['prompt'].splitlines()[-1]);now=r['context']['actuator_measurements']['timestamp']
        assert packet['protocol']=='xvla_evidence_memory_v5'
        assert packet['sensor_frames']==r['context']['sensor_frames']
        assert all(g['time']<=now for g in packet['groups'])
        assert all(g['time']<=now for g in packet['previous_checks'])
        assert all(g['timestamp']<=now for g in packet['memory'])
        assert not any(x in r['prompt'] for x in ['native_success','declared_predicates','native_steps','evaluation_only_truth'])
        if packet['role']=='release_review':
            release=next(g for g in packet['groups'] if g['kind']=='release')
            for g in packet['groups']:
                if g['kind'] in ('before','after'):
                    assert abs(g['time']-release['time'])<=1.0
        for key in usage:usage[key]+=raw.get('usage',{}).get(key,0)
        packets.append({'request':rel(request),'response':rel(response),'request_sha256':r['request_sha256'],
            'response_sha256':sha(response),'original_png_bytes_and_body_verified':True,
            'images':images,'role':packet['role'],'time':now,'historical_checks':len(packet['previous_checks']),
            'usage':raw.get('usage'),'finish_reason':raw['choices'][0]['finish_reason'],'model':raw.get('model')})
    attempts=[]
    for p in sorted(OUT.glob('attempt_*')):
        if not p.is_dir():continue
        s=load(p/'summary.json');manifest=load(p/'source_manifest.json')
        assert all(sha(p/'source'/Path(name).name)==digest for name,digest in manifest.items())
        s['id']=p.name;attempts.append(s)
    assert len(attempts)==2
    continuation=load(OUT/'continuation/summary.json')
    assert continuation['complete'] and continuation['actual_Qwen_requests']==1 and continuation['replayed_existing_v5_responses']==7
    latest=load(OUT/'continuation/source_manifest.json')
    assert all(sha(ROOT/name)==digest==sha(OUT/'continuation/source'/Path(name).name) for name,digest in latest.items())
    assert sum(a['actual_Qwen_requests'] for a in attempts)+1==len(packets)
    sources=[p for p in load(SCENE/'results/2026-10-13-xvla-event-checker/source_manifest.json') if p.endswith('.py')]
    sources+=['First_Phase/scene_demo/xvla_evidence_memory.py','First_Phase/scene_demo/tests/test_xvla_evidence_memory.py']
    manifest={};snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in sorted(set(sources)):
        p=ROOT/name;target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);manifest[name]=sha(p)
    dump(OUT/'source_manifest.json',manifest)
    audit={'real_Qwen_requests':len(packets),'usage':usage,'requests':packets,
        'protected_source_sha_unchanged':protected,'software':'319 passed, 30 subtests passed',
        'root_model':'unknown','root_effort':'unknown','subagents':0,'new_environment_actions':0,
        'new_VLA_requests':0,'new_Hermes_invocations':0,'old_model_verdicts_imported':0,
        'default_checker_changed':False,'candidate':'event_checker_enabled=True selects v5',
        'qualification':False,'continuation':{'actual_Qwen_requests':1,'exact_prior_v5_responses_replayed':7,'complete':True},
        'attempts':[{'id':a['id'],'complete':a['complete'],'passed':a['passed'],'requests':a['actual_Qwen_requests']} for a in attempts]}
    dump(OUT/'audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('requests','protected_source_sha_unchanged')},ensure_ascii=False))

elif sys.argv[1]=='seal':
    summary=load(OUT/'summary.json');audit=load(OUT/'audit.json');before=load(OUT/'registry.before.json')
    registry=load(SCENE/'experiment_registry.json');assert registry==before and len(before['entries'])==52
    assert (SCENE/'EXPERIMENT_INDEX.md').read_bytes()==(OUT/'index.before.md').read_bytes()
    assert all(sha(ROOT/p)==s for p,s in audit['protected_source_sha_unchanged'].items())
    archive=OUT/'archive';archive.mkdir(exist_ok=False)
    plans=list((SCENE/'plans').glob('20261013-xvla-evidence-memory*.json'))
    files=sorted({p for p in OUT.rglob('*') if p.is_file() and archive not in p.parents
        and p.name not in ('registry.before.json','index.before.md','evidence_manifest.json') and '__pycache__' not in p.parts}|set(plans),key=rel)
    evidence=[{'path':rel(p),'sha256':sha(p)} for p in files]
    dump(OUT/'evidence_manifest.json',{'files':evidence,'file_count':len(evidence)})
    evidence.append({'path':rel(OUT/'evidence_manifest.json'),'sha256':sha(OUT/'evidence_manifest.json')})
    entry={'id':summary['id'],'date':'2026-10-13','title':summary['title'],
        'question_keys':['xvla-process-evidence','xvla-checker-view','xvla-native-skill-interface'],
        'tags':['X-VLA','offline','confirmed-memory','three-view','negative-control'],
        'conditions':['全10长程通用规则；仅旧图按时间顺序真实Qwen回放；新中间状态，不导入旧模型判断。','三视图原始256x256；时间窗最大1秒，缺失窗口不伪造。'],
        'findings':summary['findings'],'limits':summary['limitations'],
        'configuration':{'protocol':'xvla_evidence_memory_v5','source_sha256':load(OUT/'source_manifest.json'),
            'actual_Qwen_requests':audit['real_Qwen_requests'],'default_enabled':False,'max_images':54,
            'separate_device_vote':False,'new_environment_actions':0},'configuration_complete':True,'evidence':evidence}
    registry['entries'].append(entry);dump(SCENE/'experiment_registry.json',registry)
    with (SCENE/'EXPERIMENT_INDEX.md').open('ab') as stream:
        stream.write(('\n\n### '+summary['id']+'：'+summary['title']+'\n\n'+
            '- '+summary['status']+'\n'+
            '- 全10长程共用规则；319 tests / 30 subtests；原收尾、退离、归位未改。\n'+
            '- '+str(audit['real_Qwen_requests'])+'次真实Qwen旧图请求；零新增VLA/环境动作、零子agent，候选未替换默认检查器。\n'+
            '- [报告](results/2026-10-13-xvla-evidence-memory/summary.md) · [审计](results/2026-10-13-xvla-evidence-memory/audit.json)\n').encode('utf-8'))
    r=subprocess.run([sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'validate'],capture_output=True)
    (archive/'validate.stdout').write_bytes(r.stdout);(archive/'validate.stderr').write_bytes(r.stderr)
    validate=json.loads(r.stdout);assert r.returncode==0 and validate['ok']
    invariants={'validate':validate,'old_entry_count':52,
        'old_evidence_count':sum(len(e['evidence']) for e in before['entries']),
        'old_entries_unchanged':load(SCENE/'experiment_registry.json')['entries'][:-1]==before['entries'],
        'old_evidence_hashes_unchanged':True,'protected_code_unchanged':True,'new_evidence_count':len(evidence)}
    dump(archive/'invariants.json',invariants);print(json.dumps(invariants,ensure_ascii=False))
else:raise ValueError('audit or seal required')
