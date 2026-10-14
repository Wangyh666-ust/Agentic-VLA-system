"""Read only the completed user run; archive its no-release diagnosis."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/mnt/d/FYP')
base = root / 'First_Phase/scene_demo'
attempt = base / 'results/interactive-console/20261014-152038-68f76d9d'
state = json.loads((attempt / 'console_state.json').read_text(encoding='utf-8'))
job = state['jobs'][0]
assert job['job_id'] == '72552d6bb140433bb90b3f2928e5952f'
assert job['ended_reason'] == 'visual_unknown' and state['plan']['state'] == 'blocked'
run = Path(job['run_dir'])
out = base / 'results/2026-10-14-book-held-diagnosis'
out.mkdir(exist_ok=True)
events = [json.loads(x) for x in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
checks = [json.loads(x) for x in (run / 'finish/checks.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(events) == 900 and len(checks) == 6
assert all(e['native_success'] and e['action'][-1] == 1.0 for e in events if e['step'] >= 165)
assert all(c['initial_check']['decision']['state'] == 'unknown' for c in checks)
shutil.copyfile(attempt / 'console_state.json', out / 'console_state_at_diagnosis.json')
shutil.copyfile(base / 'xvla_relation_gate.py', out / 'xvla_relation_gate.py')
shutil.copyfile(base / 'xvla_subtask_review.py', out / 'xvla_subtask_review.py')
shutil.copyfile(base / 'finish_v1/runtime.py', out / 'runtime.py')
selected = []
for c in checks:
    m = c['initial_check']['metadata']
    selected.append({'time': c['sensor_frames'][0]['timestamp'],
                     'decision': c['initial_check']['decision'], 'checks': m['effective_checks']})
summary = {
    'run': str(run), 'request': state['request'], 'trigger': '用户确认执行后的书收纳任务，用户询问为何已入格却迟迟不松手。',
    'native_success_continuous_steps': [165,900], 'native_success_samples': 736,
    'vla_closed_command_continuous_steps': [165,900], 'visual_checks': selected,
    'finish_status': job['finish_status'],
    'cause': '松手前接管同时在提示词及代码要求直接可见支撑接触；不透明盒壁挡住底部时一直unknown。',
    'limitations': ['原生位置判据不等于松手完成或真实承接接触已被视觉确认。',
                    '诊断没有改动运行代码、没有发出松手或重跑；不把先前待确认草稿验证当作本次物理成功。'],
    'new_model_calls': 0, 'new_robot_actions': 0, 'subagents': 0,
}
(out / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(out / 'summary.md').write_text('''# 书已入格但没有松手的原因

用户确认执行的这次尝试已跑满900步，以 `visual_unknown` 阻断。6次执行中检查全部Unknown，收尾松手、退离、归位动作均为0。

第165步至900步，仿真原生“书在指定后格内”持续为真；同段VLA持续输出闭爪指令。原生判据只证明其位置要求，不能替代松手或独立完成验收。

实际使用Qwen3.8，初始、过程及末次三视图均已传入。模型确认同一本书出现在后格内，但答复“书下段被盒壁与隔板遮挡，无法直接看见底部落在后格底面”，因此拒绝确认承接。

关键规则来自上次保守接管修改：提示词要求“尚未松手且没有已确认放置证据时，接管要求直接可见的承接接触”；代码也在未标准开口时要求 `support_contact.visible=True`。这对不透明容器造成无法满足的观察条件。模型回答还混入“未松手”，加重了到位与释放状态的混淆。松手历史证据为0是未松手的结果，不能用尚不存在的松手证据解除这次阻断。

三视图：腕图受夹爪和盒壁遮挡；外部A被灯杆和机械臂遮挡；外部B能见书上缘在格内，但底部接触不可见。仅重复相同视角不能解决该证据缺口。

建议通用修正：分别判断目标身份、指定区域内的位置、是否仍需下降/推进、是否可开始释放及松手后完成。直接可见接触可作为强证据，但不应是所有任务唯一入口；针对接触被遮挡的情况，应由多视角边界/插入深度与时间序列共同提供有依据的接管判断，仍有悬空/入口外/深度不明则保留Unknown。允许接管后沿用原释放、退离和独立终检；以旧提前松手负例和本次入格受遮挡样例配对离线验证，不能仅放宽一条规则就直接在线松手。

本轮仅诊断，零新模型调用、零新动作、无子agent；未修改控制或检查代码。
''',encoding='utf-8')
paths = [out/'summary.json',out/'summary.md',out/'console_state_at_diagnosis.json',out/'xvla_relation_gate.py',out/'xvla_subtask_review.py',out/'runtime.py',run/'events.jsonl',run/'result.json',run/'finish/checks.jsonl']
paths += sorted((run/'finish/model_requests').glob('*.json'))
request=json.loads((run/'finish/model_requests/call_07_request.json').read_text(encoding='utf-8'))
paths += [Path(x['path']) for x in request['images']]
reg=base/'experiment_registry.json'
data=json.loads(reg.read_text(encoding='utf-8-sig'))
eid='20261014-book-held-support-gate'
assert not any(e['id']==eid for e in data['entries'])
data['entries'].append({'id':eid,'date':'2026-10-14','title':'书入后格后未松手：直接可见支撑接触硬门槛阻断',
 'question_keys':['xvla-book-handoff-visibility'],'tags':['xvla','book','handoff','qwen3.8','diagnosis'],
 'conditions':['用户点击确认执行；本轮只读取该已完成尝试，不重跑。','book_caddy seed142/state0，原生书技能900步；无辅助器、无子agent。'],
 'findings':['原生位置165至900持续True；对应VLA持续闭爪。','6次检查全Unknown，0次收尾释放；提示词与代码都要求未松手时直接看见被盒壁遮挡的底部接触。'],
 'limits':summary['limitations'],'configuration_complete':False,
 'evidence':[{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))]})
temp=reg.with_suffix('.json.tmp')
temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(reg)
with (base/'EXPERIMENT_INDEX.md').open('a',encoding='utf-8') as f:
 f.write('\n### 20261014-book-held-support-gate\n\n- 用户执行书收纳：原生位置165至900步持续True，VLA持续闭爪；6次视觉Unknown、0次收尾释放。\n- 未松手时“直接可见支撑接触”的提示词与代码硬门槛，遇不透明盒壁遮挡阻断。只诊断，未改动或重跑。\n- [诊断](results/2026-10-14-book-held-diagnosis/summary.md)。\n')
print(json.dumps({'registered':eid,'checks':len(checks),'new_actions':0,'new_model_calls':0}))
