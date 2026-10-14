"""Seal failed offline qualification and preserve all previous evidence."""
import base64, copy, hashlib, json, shutil, struct, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;SCENE=ROOT/'First_Phase/scene_demo'
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def relative(p):return p.relative_to(ROOT).as_posix()
def local(p):return ROOT/p.removeprefix('/mnt/d/FYP/') if p.startswith('/mnt/d/FYP/') else Path(p)
registry=load(SCENE/'experiment_registry.json');before=load(OUT/'registry.before.json')
assert registry==before and len(before['entries'])==51
assert (SCENE/'EXPERIMENT_INDEX.md').read_bytes()==(OUT/'index.before.md').read_bytes()
protected=load(OUT/'protected.before.json');assert all(sha(ROOT/p)==s for p,s in protected.items())
assert (OUT/'tests-qualified.exitcode').read_text()=='0'
packets=[];usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
for request in sorted(OUT.glob('attempt_*/**/call_*_request.json')):
    r=load(request);response=request.with_name(request.name.replace('_request','_response'));raw=load(response)
    images=[];content=[{'type':'text','text':r['prompt']}]
    for frame,item in zip(r['context']['sensor_frames'],r['images']):
        p=local(item['path']);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
        assert data[:8]==b'\x89PNG\r\n\x1a\n'
        content += [{'type':'text','text':f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"},
                    {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode('ascii')}}]
        images.append({**frame,**item,'resolution':struct.unpack('>II',data[16:24])})
    body={'model':'qwen3-vl-plus','temperature':0,'max_tokens':1600,'messages':[{'role':'user','content':content}]}
    body.update(r['request_options']);assert hashlib.sha256(json.dumps(body).encode()).hexdigest()==r['request_sha256']
    assert r['http_status']==200
    for key in usage:usage[key]+=raw.get('usage',{}).get(key,0)
    packets.append({'request':relative(request),'response':relative(response),'request_sha256':r['request_sha256'],
                    'response_sha256':sha(response),'body_and_original_png_sha_verified':True,'images':images,
                    'usage':raw.get('usage'),'finish_reason':raw['choices'][0]['finish_reason'],'reported_model':raw.get('model')})
assert len(packets)==18
attempts=[{'id':p.name,**load(p/'summary.json')} for p in sorted(OUT.glob('attempt_*')) if p.is_dir()]
assert len(attempts)==3 and attempts[-1]['complete'] and not attempts[-1]['passed']
assert not list(OUT.rglob('pilot_result.json'))
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=False)
old_manifest=load(SCENE/'results/2026-10-13-xvla-timeline-trial/source_manifest.json')
sources=list(old_manifest)+['First_Phase/scene_demo/xvla_event_checker.py','First_Phase/scene_demo/tests/test_xvla_event_checker.py','First_Phase/scene_demo/tests/test_xvla_integration.py']
manifest={}
for path in sources:
    p=ROOT/path;target=snapshot/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);manifest[path]=sha(p)
dump(OUT/'source_manifest.json',manifest)
audit={'requests':packets,'real_Qwen_requests':len(packets),'usage':usage,
       'new_VLA_requests':0,'new_Hermes_invocations':0,'new_environment_actions':0,'subagents':0,
       'protected_source_sha_unchanged':protected,'final_software':'192 passed, 30 subtests passed',
       'qualification':False,'default_checker':'existing xvla_timeline_skill_v3 retained',
       'candidate_activation':'event_checker_enabled=True; explicit independent pilot additionally requires positive/negative offline gate and matching source SHA',
       'root_model':'unknown','root_effort':'unknown',
       'offline_source_limit':'Initial identity-only triplets have unknown time (-1 sentinel), no fabricated feedback; event windows use sparse recorded checkpoints. Online passive near-release windows implemented but not exercised in a new simulation.'}
dump(OUT/'audit.json',audit)
summary={'id':'20261013-xvla-event-checker','purpose':'所有长程通用过程证据、当前状态与原收尾/复位衔接修复；先离线后两项独立实测。',
 'implementation':'候选实现覆盖10 native技能及单目标；逐事件视觉事实、身份参考、松手前后原图、同物去重、当前反证撤销、压缩中间状态、时刻与SHA验证；状态类独立当前图检查。',
 'status':'offline_qualification_failed; candidate_not_default; online_cases_not_started',
 'offline_attempts':attempts,'actual_calls':{'Qwen':18,'VLA':0,'Hermes':0,'subagents':0},'usage':usage,
 'software':'192 passed, 30 subtests passed; unit/integration mocks are not visual efficacy evidence',
 'final_cases':[{k:r.get(k) for k in ('case','mode','passed','goal','reuse','checks','error')} for r in attempts[-1]['results']],
 'findings':[
 '第一版开始引用入炉过程，但第二事件broken与最终intact互相矛盾；新增程序阻断，旧首例布尔通过不能认作科学验收通过。',
 '第二版双罐正例通过，入炉杯历史复用通过但门未知；同一源照片被重复编号造成严格帧引用拒绝，第三版做源去重。',
 '第三版把状态类拆为当前图专门请求，入炉正例仍因后续事件身份断裂而Unknown；门打开负例被误报关闭，导致整体误放行。',
 '第三版双罐正例汤罐True/酱罐Unknown，双罐未完成负例False；无过程历史的遮挡杯保持Unknown。',
 '全部18次真实HTTP原PNG字节、帧/视图标签和请求SHA已复核，不能把问题归结为照片未传。结构化协议改动未消除视觉事实错误。'],
 'preserved':'finish_v1全目录、joint_home/safe_exit/safe_exit_contacts/placement_completion/xvla_backend/xvla_skills哈希未变；默认旧检查器保留，候选未上线。',
 'next_required':'先针对现有开/关门负例和同一目标身份连续性进行有明确判据的视觉事实诊断。不能放宽Unknown、忽略broken或用仿真真值代替视觉放行。离线未通过，所以未启动两项仿真。',
 'limitations':['没有新的仿真或实体机器人动作，不能报告新流程任务成功率。','已有离线图片无密集同步窗口信息，新在线窗口只完成软件验证。','每配置有限旧图片，不能推断总体可靠性。','所有历史记录保留；请求累计次数不能代替效果验收。']}
dump(OUT/'summary.json',summary)
archive=OUT/'archive';archive.mkdir(exist_ok=False)
plans=[SCENE/'plans/20261013-xvla-event-offline.json',SCENE/'plans/20261013-xvla-event-offline-state-split.json']
files=sorted({p for p in OUT.rglob('*') if p.is_file() and archive not in p.parents
              and p.name not in ('registry.before.json','index.before.md','evidence_manifest.json')}|set(plans),key=relative)
evidence=[{'path':relative(p),'sha256':sha(p)} for p in files]
dump(OUT/'evidence_manifest.json',{'files':evidence,'file_count':len(evidence)})
evidence.append({'path':relative(OUT/'evidence_manifest.json'),'sha256':sha(OUT/'evidence_manifest.json')})
old_evidence=[e for entry in before['entries'] for e in entry['evidence']]
assert all(sha(ROOT/e['path'])==e['sha256'] for e in old_evidence)
entry={'id':summary['id'],'date':'2026-10-13','title':'通用事件事实候选检查器：离线18请求未通过，未启动仿真',
 'question_keys':['xvla-process-evidence','xvla-checker-view','xvla-native-skill-interface'],
 'tags':['X-VLA','offline','event-facts','three-view','negative-control','not-qualified'],
 'conditions':['全部10 native通用候选协议；仅旧照片离线真实Qwen，零新增环境动作，无辅助器/子agent。',audit['offline_source_limit']],
 'findings':summary['findings']+[summary['status']], 'limits':summary['limitations'],
 'configuration':{'protocol':'xvla_event_facts_v4','source_sha256':manifest,'actual_Qwen_requests':18,
                  'default_enabled':False,'max_input_images':42,'latest_state_separate':True,'simulation_cases_started':0},
 'configuration_complete':True,'evidence':evidence}
registry['entries'].append(entry);dump(SCENE/'experiment_registry.json',registry)
with (SCENE/'EXPERIMENT_INDEX.md').open('ab') as stream:
    stream.write(('\n\n### 20261013-xvla-event-checker：通用事件事实候选，离线未通过\n\n'
                  '- 三版共18次真实Qwen旧图请求；过程开始被引用，但最终仍有开门负例误放行及身份连续性错误。\n'
                  '- 192 tests / 30 subtests通过只代表程序资格；原收尾/退离/归位源未改，候选未替换默认检查器。\n'
                  '- 离线门禁未通过，入炉关门与双罐入篮均未新增实测；零VLA/环境动作、零子agent。\n'
                  '- [结构化报告](results/2026-10-13-xvla-event-checker/summary.json) · [请求原图审计](results/2026-10-13-xvla-event-checker/audit.json)\n').encode('utf-8'))
run=subprocess.run([sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'validate'],capture_output=True)
(archive/'validate.stdout').write_bytes(run.stdout);(archive/'validate.stderr').write_bytes(run.stderr)
invariants={'validate_exitcode':run.returncode,'validate':json.loads(run.stdout),'old_entry_count':len(before['entries']),
 'old_evidence_count':len(old_evidence),'old_entries_unchanged':load(SCENE/'experiment_registry.json')['entries'][:-1]==before['entries'],
 'old_evidence_hashes_unchanged':True,'protected_code_unchanged':True,'new_evidence_count':len(evidence)}
dump(archive/'invariants.json',invariants);print(json.dumps(invariants,ensure_ascii=False));assert run.returncode==0
