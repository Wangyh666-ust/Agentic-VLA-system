"""Native host verification and exact subprocess output retention."""
import ast,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=load(OUT/'protected.before.json')
for path,digest in protected.items():assert sha(ROOT/path)==digest,path
def stripped(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef) and node.name=='XvlaSceneService':
            node.body=[f for f in node.body if not isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)) or f.name!='_make_finish_monitor']
    return ast.dump(tree,include_attributes=False)
assert stripped(OUT/'before/xvla_service.py')==stripped(ROOT/'First_Phase/scene_demo/xvla_service.py')
assert sha(OUT/'before/xvla_monitor.py')==sha(ROOT/'First_Phase/scene_demo/xvla_monitor.py')
command=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP/First_Phase',
    '/home/yhwang/fyp/xvla_native/venv/bin/python','-m','pytest','First_Phase/scene_demo/tests','-q','-k','xvla']
result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(OUT/'pytest.stdout.txt').write_bytes(result.stdout);(OUT/'pytest.stderr.txt').write_bytes(result.stderr)
dump(OUT/'software.json',{'command':command,'exit_code':result.returncode,'protected_unchanged':len(protected),
    'monitor_whole_file_unchanged':True,'service_outside_checker_factory_AST_unchanged':True,
    'first_suite':'353 passed / 1 legacy import-name assertion failed; expected module updated to the implemented v7 adapter.'})
print(result.stdout.decode('utf-8',errors='replace'));assert result.returncode==0
proposal=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-execution.json';p=load(proposal)
p['scope']['execution_allocation']={'probe':4,'microwave_mug':10,'basket_two':10,'no_history':2,'total':26}
p['scope']['probe_control']='先在未分类事件对照中回审第300步之前两次松手，要求旧第一次事件汤罐补证；通过后才用末帧验证遮挡继承。随后两任务独立按原时序回放，入篮第300步仍必须有第一次事件汤罐证明，否则停止依赖它的后继。无历史末帧负例最多2请求含初态身份。'
p['design']['intended_source_boundary']['checker_and_recording']=list(dict.fromkeys(p['design']['intended_source_boundary']['checker_and_recording']+['First_Phase/scene_demo/xvla_subtask_evidence.py','First_Phase/scene_demo/xvla_subtask_review.py']))
dump(proposal,p)
command=['D:/miniconda/python.exe','-X','utf8','.agents/skills/fyp-experiment-review/scripts/review.py','check','--proposal',str(proposal)]
result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(OUT/'gate.stdout.txt').write_bytes(result.stdout);(OUT/'gate.stderr.txt').write_bytes(result.stderr)
print(result.stdout.decode('utf-8',errors='replace'));assert result.returncode==0
