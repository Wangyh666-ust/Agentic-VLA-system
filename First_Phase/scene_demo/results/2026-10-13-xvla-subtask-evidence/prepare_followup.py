import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
proposal=load(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-replay-fix.json')
proposal['id']='20261013-xvla-subtask-evidence-protocol-followup'
proposal['analysis_steps']+=['真实v7回答反复出现明确state而same_target=null、初始漏装置绑定、stable capture/view引用被frame字段拒绝。v7.1明确三值含义、known state要求身份确认、初始逐目标覆盖、精确解析capture/view及按采样时间排图；不从文字自动猜身份。',
    '本轮已实际尝试15请求（14响应、1网络失败）。保留v7原始记录与代码快照。v7.1最多10新增请求：汤罐补证/后续遮挡4，入炉初末过程联合检查4，无历史负例2；总尝试不超过25，低于原26预算。不是完整顺序回放或独立新物理样本；不因一次通过宣称整体资格通过。']
proposal['scope']['followup_allocation']={'basket':4,'microwave':4,'negative':2,'baseline_attempts':15,'maximum_total_attempts':25}
proposal['scope']['stop_rule']+=' v7.1汤罐第一次松手证据仍未建立则不执行其末帧依赖检查；不再反复改prompt抽样，记录现存感知/身份问题。两项独立，原完整时间线失败不可被选帧通过抵消。'
path=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-protocol-followup.json';dump(path,proposal)
cmd=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP/First_Phase','/home/yhwang/fyp/xvla_native/venv/bin/python','-m','pytest','First_Phase/scene_demo/tests','-q','-k','xvla']
test=subprocess.run(cmd,capture_output=True,cwd=ROOT)
(OUT/'pytest-final.stdout.txt').write_bytes(test.stdout);(OUT/'pytest-final.stderr.txt').write_bytes(test.stderr)
print(test.stdout.decode());assert test.returncode==0
snapshot=OUT/'final_source_snapshot';snapshot.mkdir(exist_ok=False);manifest={}
files=[ROOT/p for p in load(OUT/'source_manifest.json') if not p.endswith('/replay.py')]
files += [OUT/'followup.py',path]
for file in files:
    target=snapshot/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    manifest[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
dump(OUT/'final_source_manifest.json',manifest)
cmd=['D:/miniconda/python.exe','-X','utf8','.agents/skills/fyp-experiment-review/scripts/review.py','check','--proposal',str(path)]
gate=subprocess.run(cmd,capture_output=True,cwd=ROOT)
(OUT/'gate-final.stdout.txt').write_bytes(gate.stdout);(OUT/'gate-final.stderr.txt').write_bytes(gate.stderr)
print(gate.stdout.decode());assert gate.returncode==0
