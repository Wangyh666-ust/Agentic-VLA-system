"""Append actual repair evidence; leave previous entries byte-preserved."""
import json
import re
from .prepare_goal_pilot import ROOT, OUT, PLAN
from .prepare_simple_pilot import sha


def main():
    registry_path = ROOT/'First_Phase/scene_demo/experiment_registry.json'
    original = registry_path.read_bytes()
    registry = json.loads(original)
    summary = json.loads((OUT/'evidence/summary.json').read_text())
    manifest_path = OUT/'evidence/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    final = {row['case']:row for row in summary['runs']}
    findings = []
    for key in ['wine_exit','stove_exit']:
        row = final[key]
        counts = row['counts']
        findings.append(f"{row['title']}最终{row['run']}：actual_open={row['actual_open']}、movement={counts['movement']}、home={counts['home']}；home_ready={row['home_ready']}、physical_streak={row['physical_streak']}、status={row['status']}、reason={row['reason']}。")
    findings += [f"原归位执行后缀核对相同；实际Qwen{summary['qwen_calls']}次/{summary['qwen_tokens']}tokens，新VLA0、子agent0。",'前两次酒瓶纯提示词修正未触发；增加实际腕部视角后触发，实际开爪后又揭示标准全开先于退离的阻断。所有中间版本和失败保留，不能合并为最终同版成功率。']
    findings.append('酒瓶wine_atomic物理五连通过、home ready但在线最后Qwen误判，原blocked保留。终态仅主视图/名称映射/任务文本三次否决保留；最后字面任务目的+1024短推理复核complete，动作0，不冒称新完整在线成功。')
    if summary.get('incomplete_checks'):
        findings.append('补充番茄罐未入篮负例启动一次，进程输出及模型响应未保留，判定/usage未知；不重试、不计作通过。有响应的20次与这次未核实尝试分开计数。')
    entry = {'id':'20261011-goal-finish','title':'目的区域检查、三视角接管与退离预算修复：酒瓶/灶台真实小试',
             'date':'2026-10-11','question_keys':['release-stop','visual-completion','joint-home-handoff'],
             'tags':['LIBERO','Qwen','收尾','归位','两问检查器'],
             'conditions':['same wine init1/replay180 and stove init0/replay78, task8 libero_10 seed0; exact original terminal SHA checked',
                           'goal-only handoff separate from final acceptance; place uses agent/side/wrist, turn preserves agent/side; synchronous encoders',
                           'path-dependent bounded movement budget; near-open stable feedback permits screened withdrawal, not completion; exact home opening gate and original streaming home unchanged',
                           'actual helper commands and independent post-action gold/contacts/home are recorded; no simulator object truth sent to Qwen'],
             'findings':findings,'limits':['仅两个历史末态的小范围接管，非新VLA从头或组合实验，不证明泛化。','灶台在预算修复版通过，酒瓶后续修复版；各有来源快照，不是最终同版联测。','相机、交接语义和收尾门槛多项修正，不能分别估计各项因果效果。','演示自动收尾未在本轮启用，先保留小试证据。'],
             'configuration':{'physical_trial_configuration':json.loads(PLAN.read_text())['configuration'],
                              'final_checker_recheck':json.loads((ROOT/'First_Phase/scene_demo/plans/2026-10-11-home-primary-existing.json').read_text())['configuration'],
                              'launches':[json.loads(p.read_text()) for p in sorted((OUT/'evidence').glob('launch_*.json'))]},
             'configuration_complete':False,
             'evidence':manifest['tracked_files']+[{'path':manifest_path.relative_to(ROOT).as_posix(),'sha256':sha(manifest_path)}]}
    if any(e['id']==entry['id'] for e in registry['entries']):raise RuntimeError('Already registered')
    closing = original.rfind(b'\n  ]')
    if closing<0:raise RuntimeError('Unexpected registry layout')
    block = json.dumps(entry,ensure_ascii=False,indent=2)
    block = '\n'.join('    '+line for line in block.splitlines()).encode()
    updated = original[:closing]+b',\n'+block+original[closing:]
    updated = re.sub(br'("updated_at"\s*:\s*")[^"]+',br'\g<1>2026-10-11',updated,count=1)
    assert json.loads(updated)['entries'][:-1] == registry['entries']
    registry_path.write_bytes(updated)
    index = ROOT/'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
    text = index.read_text(encoding='utf-8')
    addition = '\n## 2026-10-11 最新修复\n\n[酒瓶/灶台收尾与归位实测](results/2026-10-11-goal-finish/README.md)：以逐阶段动作和独立效果记录为准；保留所有先前失败，不合并版本成功率。\n'
    index.write_text(text+addition,encoding='utf-8')
    print(json.dumps({'registry_entries':len(registry['entries'])+1,'old_entries_preserved':True}))


if __name__=='__main__':main()
