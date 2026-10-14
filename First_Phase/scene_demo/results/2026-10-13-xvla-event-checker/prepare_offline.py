"""Freeze read-only photo analysis scope; no simulator/model calls."""
import json, hashlib, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SCENE=ROOT/'First_Phase/scene_demo'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
reg=json.loads((SCENE/'experiment_registry.json').read_bytes())
prior=[e['id'] for e in reg['entries'] if 'xvla-process-evidence' in e['question_keys']]
for eid in ('20261012-xvla-checker-grounding','20261012-xvla-input-audit','20261012-xvla-persistent-pilot'):
    if eid not in prior:prior.append(eid)
proposal={'schema_version':1,'id':'20261013-xvla-event-offline','intent':'analyze_existing',
 'question_key':'xvla-process-evidence','purpose':'以逐事件事实和逐目标时间汇总修复所有十个native长程检查；保留原收尾/退离/复位，先离线正负例通过才可独立启动两项仿真。',
 'prior_evidence':prior,'history_review':[{'experiment_id':eid,'disposition':'reuse',
 'reason':('复用原始两项失败请求、三视图和检查状态；新增事实协议不修改旧记录。' if eid=='20261013-xvla-timeline-trial' else '复用既有协议、实测采图、原生输入或下游资格记录；旧结果不能证明新事实汇总效果。')} for eid in prior],
 'new_physical_actions':0,'analysis_steps':[
 '软件验证全部十技能：逐事件完整输出、身份/同物计数、撤销、当前状态、SHA/时刻；原finish/control及home文件哈希保持。',
 '真实qwen3-vl-plus只重读旧照片，输入初始三图身份参考（旧时刻未知标-1，只用于身份，不填假编码器）、实测release及已记录当前捕获，原先检查状态压缩入请求。',
 '正例各一：微波炉末态必须入炉历史引用且门当前True；双罐末态须两不同目标同时True且至少一项历史复用。负例各一：未关闭门及仅第一罐入篮不得整体True。无历史的遮挡当前不得凭空通过。',
 '首轮最多6次真实请求；失败先保存原始请求/回答诊断，最多两次有因修订，每次最多6次。未通过不启动新仿真。',
 '通过后另冻结两项独立仿真方案并check，各一次900VLA步，无辅助器，无子agent，保留原下游。'],
 'scope_notes':{'max_offline_calls':18,'root_model':'unknown','root_effort':'unknown','subagents_used':0,
 'offline_gate':'micro final history true + latest close true; basket distinct both true + history reference; negative cases never full true; strict source/protocol validation'}}
dest=SCENE/'plans/20261013-xvla-event-offline.json'
assert not dest.exists() and not (OUT/'registry.before.json').exists()
shutil.copyfile(SCENE/'experiment_registry.json',OUT/'registry.before.json')
shutil.copyfile(SCENE/'EXPERIMENT_INDEX.md',OUT/'index.before.md')
dump(dest,proposal)
protected=list((SCENE/'finish_v1').glob('*.py'))+[SCENE/p for p in ('joint_home.py','safe_exit.py','safe_exit_contacts.py','placement_completion.py','xvla_backend.py','xvla_skills.py')]
dump(OUT/'protected.before.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
run=subprocess.run([sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(dest)],capture_output=True,cwd=ROOT)
(OUT/'offline-review.stdout').write_bytes(run.stdout);(OUT/'offline-review.stderr').write_bytes(run.stderr)
print(run.stdout.decode('utf-8'));sys.exit(run.returncode)
