"""Freeze this repair's evidence without modifying the live draft or old failures."""
import hashlib
import json
from pathlib import Path
import shutil

out = Path(__file__).resolve().parent
root = out.parents[3]
base = root / 'First_Phase/scene_demo'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

online = read(out / 'online_validation.json')
replay = read(out / 'inventory_replay/summary.json')
state = read(out / 'online_final_state.json')
assert online['ok'] and replay['ok']
assert online['stage'] == 'draft' and not online['execution_submitted']
assert online['vla_actions'] == online['vla_inference_requests'] == 0
assert online['settling_actions'] == 10
assert '147 passed, 6 subtests passed' in (out / 'software.stdout').read_text(encoding='utf-8')
assert read(out / 'review_check.stdout')['ok'] and read(out / 'online_review.stdout')['ok']
original = base / 'results/interactive-console/20261014-145221-ee32577c'
assert (original / 'console_state.json').read_bytes() == (out / 'before/failed_console_state.json').read_bytes()
assert (original / 'console_error.log').read_bytes() == (out / 'before/failed_console_error.log').read_bytes()
changed = ['xvla_console.py', 'xvla_console.html', 'xvla_grounding.py', 'xvla_service.py',
           'finish_v1/vision.py', 'finish_v1/simple_checker.py', 'XVLA_CONSOLE.md',
           'tests/test_xvla_console.py', 'tests/test_qwen_transport.py', 'finish_v1/tests/test_grounding.py']
for name in changed:
    target = out / 'source_snapshot' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / name, target)
for name in ['xvla_finish.py', 'finish_v1/control.py', 'finish_v1/feedback_retreat.py', 'joint_home.py']:
    assert (base / name).read_bytes() == (out / 'online_source_snapshot' / name).read_bytes()
attempt = Path(online['run_path'])
run = Path(online['session_run_path'])
snapshot = out / 'online_evidence'
snapshot.mkdir(exist_ok=True)
for name in ['run_setup.json', 'initial_session.json', 'planning_session.json', 'prompt.txt',
             'draft.json', 'task_request.json', 'usage.json', 'hermes.log']:
    shutil.copyfile(attempt / name, snapshot / name)
shutil.copyfile(run / 'native/native_steps.jsonl', snapshot / 'native_steps.jsonl')
shutil.copyfile(Path(state['model_log']), snapshot / 'model_at_draft.log')
shutil.copytree(run / 'planning_grounding', snapshot / 'planning_grounding', dirs_exist_ok=True)
profile = Path('/home/yhwang/fyp/xvla_native/console_profiles') / attempt.name
shutil.copyfile(profile / 'logs/agent.log', snapshot / 'hermes_agent.log')
log = (snapshot / 'hermes_agent.log').read_text(encoding='utf-8')
assert 'api_calls=4/12' in log and 'model=qwen3.8-max' in log
summary = {
    'question': '消除X-VLA启动前全历史台账等待，并修复三视图初识别阻断；真实接口验证到用户确认边界。',
    'original_attempt': original.name,
    'diagnosis': {
        'historical_audit_seconds': 248,
        'historical_failure': 'Qwen network failure，底层异常被旧代码丢弃；91秒与90秒超时相符，无法追溯证明原始异常类型。',
        'confirmed_timeout_problem': '同组旧图复核94.34秒正常返回，超过旧90秒上限。',
    },
    'changes': ['交互任务只准备当前配置与源码记录，不读历史台账或启动review.py。',
                '初识别/复核各180秒，外层390秒；执行中检查器仍90秒。',
                '发送前保存提示词/图片引用/哈希；失败保存异常类型、阶段、耗时并显示简明错误。'],
    'changed_files': changed,
    'software': '147 passed, 6 subtests passed',
    'offline_replay': replay,
    'online_validation': online,
    'elapsed_seconds': {'before_xvla': 1.01, 'loading_and_scene_setup': 17.66,
                        'recognition_and_review': 184.59, 'hermes_planning': 211.68, 'total_to_draft': 414.94},
    'inventory_calls': {'offline_successful': 2, 'online_successful': 2, 'model': 'qwen3.8-max'},
    'hermes': {'logged_main_api_calls': 4, 'model': 'qwen3.8-max',
               'auxiliary_title_request_started': True, 'usage_export_available': False},
    'old_failure_preserved': True, 'subagents': 0, 'helpers': False,
    'limits': ['验证至待确认草稿，未点击执行；不代表书本放置、收尾或完整任务成功。',
               '剩余识别和规划约6分54秒，整体体验仍慢；消除的是启动前全台账等待。',
               '旧网络异常具体类型无法恢复；一次新链路成功不证明网络长期稳定。',
               '主agent模型与推理设置未由独立运行元数据确认；未调用executor。']
}
save(out / 'summary.json', summary)
report = '''# 交互启动等待与初始识别修复

原失败发生在初始识别，尚未开始抓取。交互入口每次运行完整台账审查，造成248秒启动前等待；该扫描不参与当前图像识别，现已从交互入口移除。

旧识别请求上限90秒，并把底层异常统称为“Qwen network failure”。旧记录无法恢复精确异常类型。复用同组三视图，两次正常响应分别用了87.21和94.34秒，证实90秒会截断有效复核。现初识别和复核各等待180秒，外层390秒，发送前留存请求，失败记录阶段与异常类型。执行期检查仍沿用90秒配置。

实际网页接口使用“帮我整理一下桌子，把书本归位”，三视图识别、复核及Hermes规划通过，选择原生书本收纳技能，停在待用户确认草稿。

| 阶段 | 本次耗时 |
| --- | ---: |
| 启动X-VLA前准备 | 约1秒（原248秒） |
| 模型加载与场景创建 | 约18秒 |
| 三视图识别与复核 | 约185秒 |
| Hermes规划 | 约212秒 |
| 到待确认草稿总计 | 约415秒 |

因此启动前历史扫描已消除，初识别阻断在本次实测中已越过；模型识别和规划本身仍然慢。147项测试及6项子测试通过。

只发生原初始化10步静置，零VLA推理、零任务动作，未提交执行计划。本轮不验证放书、松手、收尾及归位效果。原失败记录保留，未增加辅助器，未调用子agent。
'''
(out / 'summary.md').write_text(report, encoding='utf-8')
names = ['summary.json', 'summary.md', 'proposal.json', 'review_check.stdout', 'online_proposal.json',
         'online_review.stdout', 'software.stdout', 'software.stderr', 'network_probe.json',
         'online_validation.json', 'online_final_state.json', 'online_checkpoints.json',
         'online.stdout', 'online.stderr', 'ui_restart.json', 'replay.stdout', 'replay.stderr']
evidence = [out / name for name in names]
for folder in ['before', 'source_snapshot', 'inventory_replay', 'online_evidence']:
    evidence += [p for p in (out / folder).rglob('*') if p.is_file()]
registry = base / 'experiment_registry.json'
data = read(registry)
entry_id = '20261014-console-network-and-setup'
assert not any(e['id'] == entry_id for e in data['entries'])
data['entries'].append({
    'id': entry_id, 'date': '2026-10-14', 'title': '交互启动历史扫描移除与Qwen初识别超时修复',
    'question_keys': ['xvla-console-network-and-setup'], 'tags': ['xvla', 'console', 'qwen3.8', 'timeout', 'draft-only'],
    'conditions': ['书收纳场景seed142/state0；当前原始三视图；不加辅助器；不调用子agent。',
                   '交互准备不查旧台账；初始识别180秒/请求，外层390秒；其余控制和收尾复位保留。'],
    'findings': ['准备248秒降至约1秒；旧图两次Qwen请求87.21/94.34秒均通过。',
                 '真实接口到原生书收纳草稿通过，共414.94秒；尚未用户确认执行。',
                 '147项测试及6项子测试通过；只有10步初始化静置，零VLA推理/任务动作。'],
    'limits': summary['limits'], 'configuration_complete': False,
    'evidence': [{'path': p.relative_to(root).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(set(evidence))]
})
data['updated_at'] = '2026-10-14'
pending = registry.with_suffix('.json.tmp')
save(pending, data)
pending.replace(registry)
with (base / 'EXPERIMENT_INDEX.md').open('a', encoding='utf-8') as stream:
    stream.write('\n### 20261014-console-network-and-setup\n\n'
                 '- 交互入口不再扫描全台账，实测X-VLA启动前准备248秒降至约1秒。\n'
                 '- 旧图正常复核94.34秒超过原90秒；初识别180秒/请求、外层390秒，错误细节留档。\n'
                 '- 实际三视图识别与Hermes书收纳草稿通过，总415秒；识别/规划仍慢。\n'
                 '- 147项测试及6项子测试；10步原初始化静置、零VLA任务动作；未执行放书/收尾。\n'
                 '- 证据：[报告](results/2026-10-14-console-network-fix/summary.md)。\n')
print(json.dumps({'registered': entry_id, 'evidence_count': len(set(evidence)), 'stage': online['stage']}))
