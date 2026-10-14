"""Archive the bounded startup repair and add its original evidence to registry."""
import hashlib
import json
from pathlib import Path
import shutil

out = Path(__file__).resolve().parent
root = out.parents[3]
base = root / 'First_Phase/scene_demo'
startup = json.loads((out / 'startup_validation.json').read_text(encoding='utf-8'))
ui = json.loads((out / 'ui_restart.json').read_text(encoding='utf-8'))
assert startup['ok'] and startup['real_model_starts'] == 2
assert startup['new_robot_actions'] == startup['new_vla_inference_requests'] == startup['new_vlm_requests'] == 0
assert ui['stage'] == 'ready' and ui['old_failure_unchanged'] and ui['preview_only']
assert '16 passed' in (out / 'software.stdout').read_text(encoding='utf-8')
assert json.loads((out / 'review_check.stdout').read_text(encoding='utf-8'))['ok']
changed = ['xvla_console.py', 'tests/test_xvla_console.py', 'XVLA_CONSOLE.md']
for name in changed:
    target = out / 'source_snapshot' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base / name, target)
summary = {
    'original_attempt': '20261014-142953-3e090f52',
    'cause': 'Model weights loaded, then deploy.py exited with code 1 because the reused output directory already contained info.json; old process absent and port free.',
    'fix': 'Unique output and log directories per launch, existing port conflict guard retained, startup error reports exit code and exact log path.',
    'changed_files': changed,
    'software': '16 passed', 'startup_validation': startup,
    'ui_ready': {'scene': ui['scene'], 'stage': ui['stage'], 'frame_count': len(ui['frames']), 'new_run_path': ui['new_run_path']},
    'original_failure_preserved': True, 'new_robot_actions': 0,
    'new_vla_inference_requests': 0, 'new_vlm_requests': 0, 'subagents': 0,
    'limits': ['Only model service lifecycle and preview restored; book task and revised checker not exercised online in this repair.',
               'Root model/reasoning setting not independently verified by a runtime metadata tool; no executor model was invoked.']
}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
evidence = [out / name for name in ['summary.json', 'proposal.json', 'review_check.stdout',
                                  'software.stdout', 'startup_validation.json', 'startup.stdout', 'startup.stderr', 'ui_restart.json']]
evidence += list((out / 'original_failure').glob('*'))
evidence += [out / 'source_snapshot' / name for name in changed]
for row in startup['runs']:
    # Startup probe saved WSL paths; archived files are read here on Windows.
    runtime = out / 'zero_action_probe' / Path(row['runtime']).name
    evidence += [runtime / 'model.log', runtime / 'model_process.json', runtime / 'output/info.json']
registry = base / 'experiment_registry.json'
data = json.loads(registry.read_text(encoding='utf-8-sig'))
entry_id = '20261014-console-startup-fix'
assert not any(e['id'] == entry_id for e in data['entries'])
data['entries'].append({
    'id': entry_id, 'date': '2026-10-14', 'title': '交互界面模型重启：旧 info.json 阻断修复与零动作验证',
    'question_keys': ['xvla-console-model-startup'], 'tags': ['xvla', 'console', 'runtime', 'zero-actions'],
    'conditions': ['本次失败发生于任务执行前；原始 model.log 显示权重已加载，随后因残留 info.json 退出。',
                   '独立启动目录；原输入、检查器、收尾退离复位均沿用；无辅助器，无子agent。'],
    'findings': ['16项界面测试通过；真实服务启动及释放后再次启动均ready，两次输出目录不同。',
                 '新后台已恢复书与收纳架三视图预览，原失败状态和共享日志未改写。',
                 '零VLA推理、零VLM请求、零任务动作；不能计入机器人效果样本。'],
    'limits': summary['limits'], 'configuration_complete': False,
    'evidence': [{'path': p.relative_to(root).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in evidence]
})
data['updated_at'] = '2026-10-14'
pending = registry.with_suffix('.json.tmp')
pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
pending.replace(registry)
with (base / 'EXPERIMENT_INDEX.md').open('a', encoding='utf-8') as stream:
    stream.write('\n### 20261014-console-startup-fix\n\n'
        '- 界面复用输出目录残留 info.json，模型权重已加载但服务退出；不是任务检查结论。\n'
        '- 改为每次独立输出和日志；16项软件测试及真实“启动→释放→再次启动”通过。\n'
        '- 后台已更新并恢复书收纳场景三视图；零新推理、零任务动作，原失败记录保留。\n'
        '- 证据：[summary.json](results/2026-10-14-console-startup-fix/summary.json)。\n')
print(json.dumps({'registered': entry_id, 'evidence_count': len(evidence), 'new_robot_actions': 0}))
