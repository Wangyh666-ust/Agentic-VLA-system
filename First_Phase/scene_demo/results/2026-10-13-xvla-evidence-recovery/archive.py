"""Append only this campaign; leave all earlier entry objects and evidence intact."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
registry_path=ROOT/'First_Phase/scene_demo/experiment_registry.json'
registry=load(registry_path);assert registry==load(OUT/'registry.before.json')
for path,digest in load(OUT/'protected.before.json').items():assert sha(ROOT/path)==digest
for path,digest in load(OUT/'source_manifest.json').items():assert sha(ROOT/path)==digest
audit=load(OUT/'audit.json');summary=load(OUT/'summary.json')
assert audit['real_Qwen_requests']==14 and summary['offline_visual_qualification'] is False
files={p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
files.add(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-evidence-recovery.json')
for path in audit['all_images']:files.add(ROOT/path.removeprefix('/mnt/d/FYP/'))
entry={
 'id':'20261013-xvla-evidence-recovery','title':'所有松手原图保留与后续补建：机制触发，旧图视觉验收仍未通过',
 'date':'2026-10-13','question_keys':['xvla-process-evidence','xvla-checker-view','xvla-native-skill-interface'],
 'tags':['X-VLA','offline','release-history','late-confirmation','three-view'],
 'conditions':['所有10个长程共用规则，仅两类任务旧图真实Qwen回放。原图wrist/agentview256x256，sideview512x512。',
   '原收尾/退离/下一子任务复位等43个保护文件哈希相同；零新环境/VLA动作、辅助器或子agent。'],
 'findings':[
  'v6事件库独立于确认记忆；漏判、未知、未分类及超出旧首四条保留上限的松手原图均保留，后续请求全部包含已发生事件原图。14次实际请求的原图/帧标签/body SHA已核对。',
  '普通当前检查在身份匹配、at_destination、supported=true且同步实测标准开口时补建放置证据，记录实际确认时刻、三视图及独立evidence_id，event_id=null，不倒填松手时间。',
  '漏分类恢复对照实际补建番茄酱罐并在后续遮挡时复用；完整入篮同样在第300步补建、450/600/750步复用。汤罐仍无有效确认，末帧番茄酱出现无充分变化证据的outside冲突，原证据保留但整体Unknown。',
  '入炉150/164步开门负例正确，300步完整True；450步回答明确事实但frames=[]导致协议拒绝并停止，未达到末帧。无历史负例杯子unknown但门matched缺object_key及事实帧，亦协议拒绝，不能记通过。',
  '336 tests/30 subtests；两处机制软件与实际分支已验证，14次Qwen照片整体资格未通过，候选未替换默认。',
  '更正上一条历史摘要的分辨率文字：sideview实际512x512；旧audit.json已正确记录尺寸，旧条目和原始证据未改写。'
 ],
 'limits':['全部十技能仅共用规则/软件覆盖，两类真实Qwen图片不代表十任务实测成功。',
  '两次补建属于同一旧物理样本的独立对照和完整回放，不能当作独立成功样本；新请求数不是任务成功率。',
  '新加入原图与补建规则共同变化，单轮回答差异无法单独归因；身份混淆、事实帧缺失和outside冲突未在本轮扩大修复。',
  '固定旧轨迹回放不模拟300步移交后的新轨迹；未运行收尾/退离/复位，不评价它们本轮效果。',
  '原图库保留全部事件；单次54张图上限超限明确报错，不静默删除；一秒窗口缺图不伪造。',
  '主Agent模型/推理档位调用元数据不可核实，记录unknown。'],
 'configuration_complete':False,
 'evidence':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(files)]
}
registry['entries'].append(entry)
registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
index=ROOT/'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
assert index.read_bytes()==(OUT/'index.before.md').read_bytes()
with index.open('a',encoding='utf-8',newline='\n') as handle:
    handle.write('\n\n### 20261013-xvla-evidence-recovery：全部松手原图与后续补建，视觉资格未通过\n\n'
      '- 漏判/未知/未分类松手原图全部入后续请求；普通检查可按实际时刻补建，真实Qwen已触发并复用。\n'
      '- 336 tests / 30 subtests覆盖全部10长程；原收尾、退离、复位等43个保护文件未改。\n'
      '- 14次真实旧图Qwen：入炉300步通过、450步缺事实帧停止；入篮汤罐未确认，末帧身份冲突Unknown。离线未通过，零新增环境动作、辅助器、子agent。\n'
      '- [报告](results/2026-10-13-xvla-evidence-recovery/summary.md) · [原图与请求审计](results/2026-10-13-xvla-evidence-recovery/audit.json)\n')
print(json.dumps({'added_id':entry['id'],'new_evidence_count':len(entry['evidence']),'preserved_entries':len(registry['entries'])-1},ensure_ascii=False))
