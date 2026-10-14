"""Zero-action software validation; original evidence remains read-only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
SOFTWARE=OUT/'software'
REGISTRY=ROOT/'First_Phase/scene_demo/experiment_registry.json'
INDEX=ROOT/'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
PROPOSAL=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-software.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def command(name,argv):
    dump(SOFTWARE/(name+'.argv.json'),argv)
    result=subprocess.run(argv,cwd=ROOT,capture_output=True)
    (SOFTWARE/(name+'.stdout')).write_bytes(result.stdout)
    (SOFTWARE/(name+'.stderr')).write_bytes(result.stderr)
    (SOFTWARE/(name+'.exitcode')).write_text(str(result.returncode)+'\n')
    return result.returncode

registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
for path,target in [(REGISTRY,SOFTWARE/'registry.before.json'),(INDEX,SOFTWARE/'index.before.md')]:
    if target.exists():raise FileExistsError('refuse to overwrite before evidence')
    target.write_bytes(path.read_bytes())
prior=['20261013-xvla-process-evidence-software','20261013-xvla-release-trial',
       '20261013-xvla-all-native-software','20261013-xvla-native-three-view-software']
related=[entry['id'] for entry in registry['entries'] if 'xvla-process-evidence' in entry['question_keys']]
assert set(related)<=set(prior)
proposal={'schema_version':1,'id':'20261013-xvla-timeline-software','intent':'analyze_existing',
 'question_key':'xvla-process-evidence',
 'purpose':'全部原生长程技能统一区分任务对象与干扰，向末端检查发送过程原图及中间True/False/Null判定，按时间重审证据；侧视可判状态，不增加正面条件。仅软件/原图离线序列化，不运行机器人或真实模型。',
 'prior_evidence':prior,'history_review':[{'experiment_id':i,'disposition':'reuse',
    'reason':'复用旧接口与松手机制资格，以及首例三同期图实际分类失败/空ledger证据；不把mock或原生判据当新检查效果。'} for i in prior],
 'new_physical_actions':0,'analysis_steps':[
    '读取两次实际松手及最后一轮原图/原False判定，保留原证据并复用原传输接口。',
    '验证全部10技能同一对象身份门禁、过程图与中间状态时间线、当前状态约束及来源隔离。',
    '使用原失败trial的两组三图加当前三图重构真实serializer，mock响应仅证明字节/标签/时间/状态已传入。',
    '运行针对性测试、核验已验证下游/旧证据SHA，追加软件资格台账；真实模型请求和环境动作0。'],
 'configuration':{'checker':'xvla_timeline_skill_v3','process_capture_limit':4,
    'previous_state_limit':10,'max_input_images':15,'model_call_budget':14,
    'current_check_limit':10,'release_classification_limit':4,
    'all_native_skills':10,'camera_configuration':'unchanged',
    'actual_model_calls':0,'environment_actions':0,'no_subagents':'user instruction',
    'root_model':'unknown','root_effort':'unknown'},'configuration_complete':False}
if PROPOSAL.exists():raise FileExistsError('refuse to overwrite software proposal')
dump(PROPOSAL,proposal)
codes={}
codes['review_check']=command('review_check',[sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(PROPOSAL)])
if codes['review_check']:dump(SOFTWARE/'commands.json',codes);sys.exit(2)
test_files=sorted((ROOT/'First_Phase/scene_demo/tests').glob('test_xvla*.py'))
tests=[str(p.relative_to(ROOT)).replace('\\','/') for p in test_files]
tests.append('First_Phase/scene_demo/finish_v1/tests/test_runtime.py')
argv=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1',
 'PYTHONPATH=/mnt/d/FYP:/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo/finish_v1:/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream',
 'LIBERO_CONFIG_PATH=/home/yhwang/fyp/xvla_native/libero_config',
 '/home/yhwang/fyp/xvla_native/sim_venv/bin/python','-m','pytest',*tests,'-q',
 '--basetemp=/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-13-xvla-timeline-software/software/pytest_tmp']
codes['targeted']=command('targeted',argv)
dump(SOFTWARE/'commands.json',codes)
print(json.dumps(codes))
sys.exit(0 if all(code==0 for code in codes.values()) else 2)
