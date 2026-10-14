"""Append this campaign without rewriting any old experimental finding."""
import json
import subprocess
from .prepare_simple_pilot import ROOT, OUT, PLAN, sha, write


def main():
    registry_path = ROOT / 'First_Phase/scene_demo/experiment_registry.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    old_entries = json.loads(json.dumps(registry['entries']))
    manifest = json.loads((OUT / 'evidence/manifest.json').read_text())
    evidence = [{'path': str(PLAN.relative_to(ROOT)), 'sha256': sha(PLAN)}]
    evidence.extend({'path': r['path'].replace('\\', '/'), 'sha256': r['sha256']}
                    for r in manifest['tracked_files'])
    evidence.append({'path': str((OUT / 'evidence/manifest.json').relative_to(ROOT)), 'sha256': sha(OUT / 'evidence/manifest.json')})
    entry = {
        'id': '20261010-simple-checker', 'title': '用户两问检查器四项实测：灶台实际开爪，完整归位0/3，番茄未达目标拒绝',
        'date': '2026-10-10', 'question_keys': ['release-stop', 'visual-completion', 'joint-home-handoff'],
        'tags': ['LIBERO', 'SmolVLA', 'Qwen', '两问检查器', '收尾', '归位'],
        'conditions': [
            'wine init1 replay180；bowl init0 replay83；stove init0 replay78，均libero_10 task8/envseed0；sauce task0/init0/replay336。真实恢复旧轨迹末态SHA一致，检查和辅助动作新执行。',
            '第一次wine解析失败后一次协议修正，冻结新源码；最终四项同版各一次，失败停止本例，仅独立样例继续。没有新VLA、Hermes、训练、子agent或组合。',
            'Qwen主视256与侧视512，两个时刻；同步夹爪编码器。目的达成直接开爪，删除新模式开爪前五连停稳；原退离路径和归位streaming实现/参数保留。',
            '独立效果记录不进入Qwen；开爪与保持闭合等待分开计数；原始图片、响应、动作、视频归档。'],
        'findings': [
            '酒瓶/碗目的独立记录均真，Qwen误判悬空；开爪、退离、归位均0。不能据此否定原归位能力。',
            '灶台目的判对，实际开爪17周期、退离120、确认0、归位0。实测标准开口，开火维持、手部接触0；退离预算耗尽，归位未启动。',
            '番茄目标确实未入篮，正确拒绝收尾，但Qwen错误描述仍夹持；不是完整检查器或完整任务成功。',
            '三项应接管正例实际开爪1/3、完整归位0/3；番茄为未达目标负例，不合并入正例成功率。',
            '5仿真启动、4状态；实际Qwen6次，保留usage9602tokens为下限（首次解析失败usage丢失），新VLA/Hermes/子agent0。'],
        'limits': ['四状态小试，不证明泛化；从旧末态恢复，不是新完整VLA或多任务执行。',
                   '简化了两组判断，但仍通过旧多字段接口兼容；模型返回未带视角的boxes被忽略为未知，未臆造视角。',
                   '酒瓶和碗视觉误判未消除；灶台退离跟踪未完成；自动演示收尾保持off，原成功碗归位记录不作废。',
                   '首次解析失败响应/usage在旧客户端被丢弃，不补造。其后请求、原始响应、哈希与实际图像均保存。'],
        'configuration': json.loads(PLAN.read_text())['configuration'], 'configuration_complete': True,
        'evidence': evidence,
    }
    if any(e['id'] == entry['id'] for e in registry['entries']):
        raise RuntimeError('campaign already registered; do not replace historical evidence')
    registry['entries'].append(entry)
    assert registry['entries'][:-1] == old_entries
    registry['updated_at'] = '2026-10-10'
    registry['root_note'] = f"{len(registry['entries'])} registered campaigns; reused states and revisions overlap; no pooled success rate"
    write(registry_path, registry)
    index_path = ROOT / 'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
    text = index_path.read_text(encoding='utf-8').replace('共21个历史campaign', f"共{len(registry['entries'])}个历史campaign")
    addition = '\n'.join([
        '| 20261010-joint-verifier | 同步编码器与视觉保存状态检查 | 不同修订选定四状态判对、一未知；无物理收尾，不等于同版完整流程通过 | [报告](results/2026-10-10-joint-verifier/README.md) |',
        '| 20261010-live-handoff | 当前检查器实际接管 | 酒瓶未触发；灶台允许后闭爪等待10次阻断，实际开爪/归位0 | [报告](results/2026-10-10-live-handoff/README.md) |',
        '| 20261010-simple-checker | 用户两问检查器四项独立实测 | 酒瓶/碗误判未触发；灶台实际开爪17、退离120、归位0；番茄未入篮拒绝。正例完整归位0/3 | [报告与四视频](results/2026-10-10-simple-checker/README.md) |', ''])
    text = text.replace('## 当前结论', addition + '\n## 当前结论', 1)
    index_path.write_text(text, encoding='utf-8')
    old_control = subprocess.check_output(['git', 'show', 'HEAD:First_Phase/scene_demo/finish_v1/control.py'], cwd=ROOT).decode().replace('\r\n', '\n')
    new_control = (ROOT / 'First_Phase/scene_demo/finish_v1/control.py').read_text().replace('\r\n', '\n')
    marker = "        r['home_entry'] = {"
    assert old_control.split(marker, 1)[1] == new_control.split(marker, 1)[1]
    write(OUT / 'evidence/acceptance.json', {'prior_entries_preserved': len(old_entries), 'registry_entries': len(registry['entries']),
          'original_home_loop_byte_text_equal': True, 'source_tests': {'before_protocol_repair': 40, 'after_protocol_repair': 36},
          'subagents': 0, 'full_flow_passed': False})
    print(json.dumps({'registry_entries': len(registry['entries']), 'old_entries_preserved': True, 'home_loop_unchanged': True}))


if __name__ == '__main__': main()
