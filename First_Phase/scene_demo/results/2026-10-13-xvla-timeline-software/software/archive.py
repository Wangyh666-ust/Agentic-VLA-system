"""Append software qualification; preserve all previous entries and evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
SOFTWARE=OUT/'software'
REGISTRY=ROOT/'First_Phase/scene_demo/experiment_registry.json'
INDEX=ROOT/'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
IDENTIFIER='20261013-xvla-timeline-software'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def relative(path):return Path(path).relative_to(ROOT).as_posix()

before=load(SOFTWARE/'registry.before.json');registry=load(REGISTRY)
assert registry==before and not any(e['id']==IDENTIFIER for e in registry['entries'])
assert INDEX.read_bytes()==(SOFTWARE/'index.before.md').read_bytes()
commands=load(SOFTWARE/'final_2_commands.json');assert all(code==0 for code in commands.values())
stdout=(SOFTWARE/'final_2_final_targeted.stdout').read_text()
assert '278 passed, 34 subtests passed' in stdout
protected=load(SOFTWARE/'final_2_protected_sources.json');assert all(protected.values())
old_evidence=[item for entry in before['entries'] for item in entry['evidence']]
assert len(before['entries'])==49 and len(old_evidence)==9314
assert all(sha(ROOT/item['path'])==item['sha256'] for item in old_evidence)

sys.path.insert(0,str(ROOT/'First_Phase'))
from scene_demo.xvla_pilot import frozen_sources
sources=frozen_sources()
for name in ('test_xvla_finish.py','test_xvla_full_checks.py','test_xvla_process_evidence.py',
             'test_xvla_process_monitor.py','test_xvla_timeline.py','test_xvla_trial_evidence.py','test_xvla_integration.py'):
    sources.append(ROOT/'First_Phase/scene_demo/tests'/name)
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=False)
manifest={}
for source in sources:
    destination=snapshot/relative(source);destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination);manifest[relative(source)]=sha(source)
dump(OUT/'source_manifest.json',manifest)

proofs=list((SOFTWARE/'pytest_final_2').rglob('proof.json'));assert len(proofs)==1
proof=load(proofs[0]);assert proof['actual_http_calls']==0 and proof['environment_actions']==0 and len(proof['images'])==9
serializer=SOFTWARE/'serializer';serializer.mkdir(exist_ok=False)
shutil.copyfile(proofs[0],serializer/'proof.json')
for source in (proofs[0].parent/'audit').glob('*'):shutil.copyfile(source,serializer/source.name)
for record in proof['images']:
    source=ROOT/record['path'].removeprefix('/mnt/d/FYP/')
    assert sha(source)==record['sha256']
source_whitespace={}
for source in sources:
    if source.suffix=='.py' and source.parent.name in ('scene_demo','tests'):
        text=source.read_text(encoding='utf-8')
        source_whitespace[relative(source)]=all(line==line.rstrip(' \t') for line in text.splitlines())
assert all(source_whitespace.values())
dump(SOFTWARE/'source_whitespace.json',source_whitespace)

summary={'id':IDENTIFIER,'purpose':load(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-software.json')['purpose'],
 'implementation':'全部10 native通用同一monitor/client，不按技能名称分支；目标与目的地ID来自原任务goals，统一身份确认后判断关系。',
 'timeline':'过程三图与中间True/False/Null原判定均保留；当前请求含按时间标注的过程组和当前组三图、逐组同步夹爪反馈及先前检查状态；可重审旧误判，无旧True自动填Unknown。',
 'latest_state':'门/抽屉/开关/preserved只能用最新图证明；侧面与斜视有效，不新增正面要求。',
 'bounds':{'process_captures':4,'previous_check_states':10,'max_images':15,
    'overflow':'显式标记history_complete=false与omitted_release_event_ids，不能引用未提供的event图片。',
    'current_checks':10,'release_classifications':4,'total_runtime_model_budget':14},
 'tests':{'passed':278,'subtests_passed':34,'compile':0,'help':0,'whitespace':0},
 'serializer_proof':{'original_trial':'20261013-xvla-release-trial','process_triplets':2,'current_triplets':1,
    'image_count':9,'original_intermediate_placement_values':[False,False],
    'previous_check_value':False,'request':relative(serializer/'call_01_request.json'),
    'proof':relative(serializer/'proof.json'),'response_is_mock':True,'effect_verified':False},
 'protected_sources':{'count':len(protected),'all_match_previous_actual_trial':True},
 'real_runtime_calls':{'Qwen':0,'X-VLA':0,'Hermes':0},'environment_actions':0,'subagents_used':0,
 'implementation_model':'root actual model/effort unknown; direct implementation per user no-subagent instruction',
 'limits':['软件证明了接口与传输构造，不证明真实Qwen能正确识别或完整任务通过。',
    '尚未复跑入炉关门，入篮仍未启动；沿用旧实际原始图和判定，没有新环境或策略动作。',
    '保留原模型调用预算；过程最多4组，超预算图片不足时只能依据实际提供的证据。'],
 'next_validation':'用新冻结方案复测首例是否真正引用过程原图纠正旧识别，并通过完整技能/原收尾；通过前不继续入篮。'}
dump(OUT/'summary.json',summary)
(OUT/'summary.md').write_text(
 '已将全部10个原生长程技能的检查统一改为过程图与中间状态联合复核。\n\n'
 '每个目标先确认指定对象身份，再判断关系；身份不明或误认干扰物的结果归为未知。过程照片与当时True/False/Null判定均保留，末端重新对照原图，不再只缓存成功布尔值。门、抽屉、开关和保持条件按最新图判断，侧面/斜视证据有效。\n\n'
 '沿用原采样、控制、相机和调用预算。每请求最多4组三视图过程照片加当前三图（15张），最多10次先前检查状态；溢出明确标记，不伪造完整历史。\n\n'
 '278测试通过、34 subtests通过；compile/help/源码空白检查通过。复用实际失败trial的两组松手原图与最后三图，离线验证9张原PNG、旧失败判断和同步反馈确实进入真实请求序列化器；响应是mock，不表示真实模型或任务成功。\n\n'
 '原已验证下游与未修改来源共75项SHA一致；新Qwen/X-VLA/Hermes请求0、环境动作0、子agent0。入炉关门尚未复测，入篮未启动。\n',encoding='utf-8')

selected=[ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-software.json',OUT/'source_manifest.json',OUT/'summary.json',OUT/'summary.md']
selected+=sorted(snapshot.rglob('*'))
selected += [p for p in SOFTWARE.iterdir() if p.is_file() and p.name not in ('registry.before.json','index.before.md','archive.py')]
selected+=sorted(serializer.iterdir())
selected += [ROOT/record['path'].removeprefix('/mnt/d/FYP/') for record in proof['images']]
selected += [ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/summary.json',
    ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-release-trial/audit.json']
files=sorted(set(p for p in selected if p.is_file()))
evidence=[{'path':relative(p),'sha256':sha(p)} for p in files]
dump(OUT/'evidence_manifest.json',evidence)
entry={'id':IDENTIFIER,'title':'全部长程技能通用对象身份与过程图/中间状态联合检查软件资格','date':'2026-10-13',
 'question_keys':['xvla-process-evidence','xvla-checker-view','xvla-native-skill-interface'],
 'tags':['software','X-VLA','timeline','general-grounding','mock'],
 'conditions':['全部10 native复用统一monitor/client，真实模型与环境动作0，用户要求不使用子agent。',
    '复用原实际失败trial的2组松手图、最终图与旧False判断，只验证软件与真实serializer构造。',
    '原策略/相机/控制/收尾/退离/复位与75项未修改来源SHA相同。'],
 'findings':['目标/目的地ID明确，身份未知不能被归为明确失败，通用干扰物规则覆盖全部10技能。',
    '过程照片及成功/失败/未知中间状态传入末端，与当前照片联合复核；旧True不自动填新的Unknown。',
    '历史放置可引用原event图重审；门/抽屉/开关/preserved必须有当前证据，侧面可判，无正面要求。',
    '每组图与其编码器时间对应；同时刻照片不伪造先后；历史源字节变化/未来数据/未知event拒绝。',
    '原始失败trial 9图及旧状态已在真实HTTP serializer的mock请求中逐字节核验，非真实模型效果。',
    '278 passed/34 subtests；compile/help/源码空白检查通过；两case后置gate兼容新联合引用且保留旧审计。'],
 'limits':summary['limits'],'configuration':load(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-software.json')['configuration'],
 'configuration_complete':False,'evidence':evidence}
registry['entries'].append(entry);REGISTRY.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with INDEX.open('ab') as handle:
    handle.write(('\n\n### '+IDENTIFIER+'：通用对象身份与过程图/中间状态联合检查\n\n'
        '- 软件资格：全部10 native统一输入时间线与身份门禁；278测试/34 subtests通过，原失败trial 9图与中间状态离线序列化核验。\n'
        '- 真实模型/环境动作0，子agent0；原相机/收尾/退离/归位不变；实际复测及入篮未启动。\n'
        '- [报告](results/2026-10-13-xvla-timeline-software/summary.md) · [软件方案](plans/20261013-xvla-timeline-software.json)\n').encode('utf-8'))
argv=[sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'validate']
dump(SOFTWARE/'archive_validate.argv.json',argv)
result=subprocess.run(argv,cwd=ROOT,capture_output=True)
(SOFTWARE/'archive_validate.stdout').write_bytes(result.stdout);(SOFTWARE/'archive_validate.stderr').write_bytes(result.stderr)
(SOFTWARE/'archive_validate.exitcode').write_text(str(result.returncode)+'\n')
after=load(REGISTRY)
invariants={'validate_exitcode':result.returncode,'validate':json.loads(result.stdout),
 'old_entries_deep_equal':after['entries'][:-1]==before['entries'],
 'old_9314_evidence_sha_match':all(sha(ROOT/i['path'])==i['sha256'] for i in old_evidence),
 'index_old_prefix_equal':INDEX.read_bytes().startswith((SOFTWARE/'index.before.md').read_bytes()),
 'new_all_evidence_sha_match':all(sha(ROOT/i['path'])==i['sha256'] for i in evidence),
 'source_snapshot_sha_match':all(sha(ROOT/p)==value and sha(snapshot/p)==value for p,value in manifest.items()),
 'new_evidence_count':len(evidence),'real_runtime_model_calls':0,'environment_actions':0,'subagents_used':0}
dump(OUT/'invariants.json',invariants)
assert result.returncode==0 and all(invariants[k] is True for k in ('old_entries_deep_equal','old_9314_evidence_sha_match','index_old_prefix_equal','new_all_evidence_sha_match','source_snapshot_sha_match'))
print(json.dumps(invariants))
