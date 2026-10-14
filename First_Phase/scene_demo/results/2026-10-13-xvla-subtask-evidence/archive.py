"""Append this campaign without changing any earlier entry or evidence."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=load(OUT/'audit.json');assert audit['actual_Qwen_attempts']==22 and audit['responses_archived']==21
assert audit['offline_visual_qualification'] is False
registry_path=ROOT/'First_Phase/scene_demo/experiment_registry.json';registry=load(registry_path)
assert registry==load(OUT/'before-experiment_registry.json')
for p,digest in load(OUT/'protected.before.json').items():assert sha(ROOT/p)==digest,p
for p,digest in load(OUT/'identity_source_manifest.json').items():assert sha(ROOT/p)==digest,p
report=(OUT/'summary.md').read_text(encoding='utf-8')
for link in re.findall(r'\]\(([^)]+)\)',report):assert (OUT/link).is_file(),link
files={p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
files.update((ROOT/'First_Phase/scene_demo/plans').glob('20261013-xvla-subtask-evidence-*.json'))
files.update(ROOT/p.removeprefix('/mnt/d/FYP/') for p in audit['all_images'])
# Include manually inspected original pictures even when they were not new model inputs.
files.update((OUT/link).resolve() for link in re.findall(r'\]\(([^)]+)\)',report))
entry={'id':'20261013-xvla-subtask-evidence','title':'同段原图与历史重审通用实现；入炉补证触发，汤罐指定补证及视觉验收未通过',
    'date':'2026-10-13','question_keys':['xvla-process-evidence','xvla-checker-view','xvla-native-skill-interface'],
    'tags':['X-VLA','offline','subtask-scope','historical-review','identity-binding','three-view'],
    'conditions':['仅使用既有入炉关门和双罐入篮两条轨迹照片；22次实际Qwen尝试、21响应、1网络失败，5次本轮完全相同请求复用；无新增物理样本。',
        '全10长程共用任务段记录与审核；361 tests/30 subtests，软件不等于10项实测。零环境/VLA/辅助动作，零子agent。',
        '43个控制保护文件哈希不变，monitor全文件不变，service除检查器工厂外AST不变；保留原收尾、退离与下一子任务全变量复位，默认入口未切换。'],
    'findings':['保存所有已采集原图、采样时间、SHA与同步反馈；session/logical_subtask/attempt隔离，旧事件可重审，observed_at与reviewed_at分开；缺失初态时间为null。',
        '通用身份锚点、初末过程对比、历史与当前分离、事件分页及版本化意见已实现；原图引用与实际HTTP body SHA已审计，未传仿真真值。',
        '真实v7结果暴露same_target模板null、缺装置绑定、引用别名和跨视角参照缺链；v7.1明确协议并补充一次受影响目标修复。容器/内容重叠与多物重复计数分开处理。',
        '双罐v7到第300步仍无汤罐历史证明，按规则停止依赖后继。v7.1模型整体True但只建第300步当前证明，第一次松手仍outside；初始汤罐框偏离物体，不能算本轮目标成功。',
        '入炉顺序回放到第300步Unknown，450步网络失败。独立末帧联合过程重审经一次身份关联修复，实际引用release_0001旧图生成proof，observed_at=8.95/reviewed_at=45.75；门关闭True。',
        '入炉模型True仍未通过逐图验收：历史证据框偏离黄色把手、末帧门板遮挡却声称当前杯子可见。无历史末帧杯Unknown/门True，整体未错误放行。',
        '回放入口曾遗漏verification_frames导致三视图校验错误，已修复且错误原记录保留；线上采集和探针本来含三视图。',
        '更正上轮evidence-recovery摘要：450/600/750步有当前到位判断，不等于纯遮挡沿用；独立recovery_probe才出现当前Unknown后沿用。旧原始证据与条目未更改。'],
    'limits':['汤罐第一次松手历史补证仍未发生，依赖遮挡验证未执行；不能用另一罐、当前True或下游结果替代。',
        '入炉程序分支触发与视觉证据可靠性分别判断；有合法引用不证明模型框住了正确物体。',
        'v7.1针对性选帧检查不是完整时序回放；原完整回放失败未被后续模型True抵消，整体离线资格未通过。',
        '身份补充、协议和时间排序共同变化，无法独立归因；同一旧样本多次请求不是多个物理成功样本。',
        '原收尾/退离/复位此次没有执行，未评价新物理效果。root模型/推理档位无法核实，记unknown。'],
    'configuration_complete':False,'evidence':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(files)]}
assert all(e['id']!=entry['id'] for e in registry['entries']);registry['entries'].append(entry)
registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
index=ROOT/'First_Phase/scene_demo/EXPERIMENT_INDEX.md';assert index.read_bytes()==(OUT/'before-EXPERIMENT_INDEX.md').read_bytes()
with index.open('a',encoding='utf-8',newline='\n') as f:
    f.write('\n\n### 20261013-xvla-subtask-evidence：同段原图重审，关键汤罐补证仍未通过\n\n'
        '- 全10长程通用实现：图像时间/稳定编号/同步反馈、身份参照、旧图重审与意见版本；361 tests、30 subtests。\n'
        '- 22次旧图Qwen尝试（21响应、1网络失败），无新动作/辅助器/子agent。入炉旧事件补证触发但框与可见性有误；汤罐仍只得当前证明，未引用第一次松手补证。\n'
        '- 整体离线资格未通过，候选未切换默认；原收尾、退离、下一子任务复位43个保护文件及monitor保持。\n'
        '- [报告与原图](results/2026-10-13-xvla-subtask-evidence/summary.md) · [请求审计](results/2026-10-13-xvla-subtask-evidence/audit.json)\n')
print(json.dumps({'added':entry['id'],'evidence_files':len(files),'previous_entries_preserved':len(registry['entries'])-1},ensure_ascii=False))
