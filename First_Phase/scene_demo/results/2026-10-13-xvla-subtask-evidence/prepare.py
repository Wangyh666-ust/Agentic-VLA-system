import ast,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name in ('experiment_registry.json','EXPERIMENT_INDEX.md'):
    shutil.copyfile(ROOT/'First_Phase/scene_demo'/name,OUT/('before-'+name))
previous=OUT.parent/'2026-10-13-xvla-evidence-recovery'
protected=json.loads((previous/'protected.before.json').read_bytes())
dump(OUT/'protected.before.json',{path:sha(ROOT/path) for path in protected})
for name in ('xvla_service.py','xvla_pilot.py','xvla_monitor.py'):
    dest=OUT/'before'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'First_Phase/scene_demo'/name,dest)
source=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-review.json'
shutil.copyfile(source,OUT/'approved-plan.json')
proposal=json.loads(source.read_bytes());proposal['id']='20261013-xvla-subtask-evidence-execution'
proposal['purpose']=proposal['purpose'].replace('当前仅提交方案，不实施或发起Qwen/环境调用。','用户已授权实现及旧图Qwen回放；零新增环境动作。')
proposal['analysis_steps'][0]='保留批准方案及当前源码、旧台账快照；执行实现、软件回归、实际旧图Qwen和原始证据归档。'
proposal['scope']['current_turn']='implementation_and_offline_validation'
proposal['constraints']['implementation_started']=True
proposal['analysis_steps'].append('入篮按时间推进到第300步即检查汤罐第一次松手历史证据是否补建；失败则停依赖它的后继检查，入炉独立继续。关键指标要求相应旧capture_id和实际重审记录，不用另罐证据替代。')
dump(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-subtask-evidence-execution.json',proposal)
print(json.dumps({'protected':len(protected),'baseline_entries':len(json.loads((OUT/'before-experiment_registry.json').read_bytes())['entries'])}))
