"""Archive referenced original evidence without changing historical entries."""
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN = ROOT / 'First_Phase/scene_demo/plans/2026-10-11-takeover-stove-cue.json'
REGISTRY = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
INDEX = ROOT / 'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
IDENTIFIER = '20261011-takeover-stove-cue'


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def evidence(path):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


proposal = json.loads(PLAN.read_text())
summary = json.loads((HERE / 'summary.json').read_text())
assert len(summary['cases']) == 2 and all(case['plan_success'] for case in summary['cases'])
assert sum(sum(job['actual_finish_counts'].values()) for case in summary['cases'] for job in case['jobs']) == 900
selected = {PLAN}
for path in HERE.rglob('*'):
    if path.is_file() and 'sensors' not in path.parts and path.name != 'manifest.json':
        selected.add(path)
for path in list(selected):
    if path.suffix not in ('.json', '.jsonl'):
        continue
    values = ([json.loads(line) for line in path.read_text().splitlines()]
              if path.suffix == '.jsonl' else [json.loads(path.read_text())])
    for value in values:
        for name in strings(value):
            if not name.startswith(str(HERE)):
                continue
            candidate = Path(name).resolve()
            if candidate.is_relative_to(HERE) and candidate.is_file():
                selected.add(candidate)
manifest = {'tracked_files': [evidence(path) for path in sorted(selected)]}
manifest_path = HERE / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
entry = {
    'id': IDENTIFIER, 'title': '微小越界接管与灶台红圈定义修复：两组合完整通过', 'date': '2026-10-11',
    'question_keys': ['visual-goal-encoder-authority', 'controller-takeover-command-bounds'],
    'tags': ['wine', 'stove', 'two-combos', 'takeover', 'red-ring', 'home', 'success'],
    'conditions': [
        'LIBERO goal_table/Panda seed0 init0，碗→酒瓶与碗→灶台各新执行一次，固定用户顺序；非Hermes自主规划测试。',
        '实际执行源码bff1d60；实测值保留而保持命令限幅；visual_goal_encoder_v4按静态红圈定义判灶台，视觉图与编码器同步；VLA150/300检查。',
        '原归位、当前反馈退离、路径保护、酒瓶抓辅与策略均复用；无现场重跑、训练、子agent或独立归位试验。'],
    'findings': [
        '两组合完整2/2、四子任务收尾归位4/4；两组碗均开爪2/退离103/home131，酒瓶开爪2/退离26/home221，灶台开爪20/退离36/home106；原home与独立五连均通过。',
        '酒瓶300步移交前仍为与前轮相同的0.04001874449m实测越界状态，这次实际接管成功；VLA已经松手，收尾器完成退离与归位。',
        '灶台150步Qwen识别红色加热圈而判true，实际启动同时开爪退离分支；回程首候选仍碰撞，继续退出至36步后路径通过并归位，两目标保留。',
        '新VLA750/抓辅182/收尾900，Qwen10次16386tokens含9次完成与1次抓前关联；九份完成请求实际两图哈希、同步编码器与机器人反馈全部匹配，无phase/stage发送。'],
    'limits': [
        '各组合仅一次的小范围能力证据，不是总体成功率或通用组合可靠性；酒瓶150步仍有夹持混入目标判断的风险。',
        '灶台视觉状态定义对应当前LIBERO资产，非所有真机通用；辅助与路径保护仍使用仿真几何，真实RGBD未实现。',
        '未重新评估或修改原归位模块，未训练；两项定向修复一同测试，不分离全部因果贡献。'],
    'configuration': proposal['configuration'], 'configuration_complete': True,
    'evidence': manifest['tracked_files'] + [evidence(manifest_path)],
}
# Keep historical registry bytes, including native line endings. Only append
# this campaign to the committed baseline; the task began with a clean registry.
original = subprocess.check_output(['git', 'show', 'HEAD:First_Phase/scene_demo/experiment_registry.json'], cwd=ROOT).decode('utf-8')
registry = json.loads(original)
matches = [previous for previous in registry['entries'] if previous['id'] == IDENTIFIER]
assert len(matches) <= 1
assert not matches or registry['entries'][-1]['id'] == IDENTIFIER
offset = original.rfind('\n  ]')
assert offset >= 0
if matches:
    start = original.rfind('    {\n      "id": "' + IDENTIFIER + '"')
    assert start >= 0
    updated = original[:start] + textwrap.indent(json.dumps(entry, ensure_ascii=False, indent=2), '    ') + original[offset:]
else:
    updated = original[:offset].rstrip() + ',\n' + textwrap.indent(json.dumps(entry, ensure_ascii=False, indent=2), '    ') + original[offset:]
historical = registry['entries'][:-1] if matches else registry['entries']
assert json.loads(updated)['entries'][:-1] == historical
REGISTRY.write_bytes(updated.encode('utf-8'))
if IDENTIFIER not in INDEX.read_text():
    with INDEX.open('a') as index:
        index.write('\n- `20261011-takeover-stove-cue`：限幅保持命令、补充静态红圈定义后，两组合各一次均完整通过；酒瓶同样0.0187mm实测越界状态实际接管、退离与归位成功；灶台识别红圈并触发同时开爪退离及原归位，前碗成果保留，独立五连均过。新VLA750/抓辅182/收尾900、Qwen10次16386tokens；无子agent、微调、独立归位或现场重跑。小范围2/2不等于总体可靠性。[两段新视频与明确结论](results/2026-10-11-takeover-stove-cue/README.md)。\n')
print(json.dumps({'manifest_files': len(selected), 'registry_entries': len(historical) + 1, 'old_entries_unchanged': True}))
