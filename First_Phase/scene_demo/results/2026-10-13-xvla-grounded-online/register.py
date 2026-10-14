"""Register outcomes and original evidence; never relabel a failed run as passed."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
REGISTRY=ROOT/'First_Phase/scene_demo/experiment_registry.json'
INDEX=ROOT/'First_Phase/scene_demo/EXPERIMENT_INDEX.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(OUT/'audit.json')
assert set(audit['cases'])=={'microwave_mug','basket_two_startup','basket_two'}
assert audit['v1_source_snapshot_matches'] and audit['renderfix_source_freeze_matches']
assert all(audit['protected_control_unchanged'].values())
assert all(not c['source_image_hash_errors'] for c in audit['cases'].values())
files={OUT/'summary.md',OUT/'audit.json',OUT/'audit_v1.json'}
for p in OUT.rglob('*'):
 if not p.is_file() or 'source_snapshot' in p.parts:continue
 if p.suffix in ('.json','.jsonl','.stdout','.stderr') or p.name in ('initial_state.npy','hermes_initial.log'):
  files.add(p)
 if p.name.startswith('call_') and p.name.endswith('_request.json'):
  for image in read(p).get('images',[]):
   image_path=Path(image['path'].replace('/mnt/d/FYP',str(ROOT)))
   assert sha(image_path)==image['sha256'];files.add(image_path)
 if p.name=='process_events.jsonl':
  for line in p.read_text(encoding='utf-8').splitlines():
   for image in json.loads(line)['images']:
    image_path=Path(image['path'].replace('/mnt/d/FYP',str(ROOT)));assert sha(image_path)==image['sha256'];files.add(image_path)
 if p.name in ('three_views.png','planning_three_views.png'):files.add(p)
for folder in ('source_snapshot','basket_renderfix/source_snapshot'):
 files.update(p for p in (OUT/folder).rglob('*') if p.is_file())
for name in ('20261013-xvla-grounded-online.json','20261013-xvla-grounded-online-renderfix.json'):
 files.add(ROOT/'First_Phase/scene_demo/plans'/name)
entry={'id':'20261013-xvla-grounded-online','title':'三视角规划身份在线试点：入炉提前误判关门；双罐错认经复核仍沿用，人工停止',
 'date':'2026-10-14','question_keys':['xvla-process-evidence','xvla-checker-view','xvla-native-skill-interface'],
 'tags':['X-VLA','online','three-view','planning-identity','false-positive','initial-grounding','no-helpers'],
 'conditions':['两项独立在线仿真；入炉150 VLA步/5请求，双罐有效启动87步/3请求后按身份错误停止；另有双罐零VLA取图启动错误保留。',
 '直接Qwen请求7次/响应7份，Hermes入口2次；初始识别、三视图复核、规划绑定、选择复核已实际执行。无子agent、训练、回放或抓取搬运辅助器。',
 '最终334 tests/4 subtests；保留原收尾、退离、归位与复位控制。候选显式启用，默认未切换。'],
 'findings':['入炉黄杯初态身份正确，Hermes直接选原生长程；第150步误判关门True，回答将同刻frame4开门证据误称旧状态，提前停止VLA。',
 '入炉原收尾开爪8步，退离因路径保护阻断，movement/home=0；原生AND从未True，不能据此判原生VLA本身无能力。',
 '双罐初态把橙色长方形包装认作汤罐，番茄罐外部框落在Milk纸盒附近；清单复核及执行前选择复核仍True。原图人工审查后协作取消，最终87 VLA步，松手/末端检查/收尾均0。',
 '故三视角及共享描述输入链路已执行，但防止错误初始身份持续沿用的实际效果失败；不能用模型自我确认True代替原图验收。',
 '双罐首次启动由mj_forward引起约1.11e-16姿态归一化，被逐位状态检查拦截；修为mj_camlight，在两条保存初态零步验证qpos/qvel/time及native像素一致，未放宽断言。',
 '原图请求哈希全部匹配；入炉当前三图时间均8.25，非传错图/时间；微波炉整体与内部区域共享装置仍触发去重误报，待修。'],
 'limits':['本轮两项均未完成；双罐因人工身份审查停止，不能记作完整入篮策略失败或自动拒绝成功。',
 '历史松手补证/遮挡沿用本轮未有效验证，不能用初态绑定或收尾动作代替；入炉唯一松手来自原收尾开爪。',
 '两外部机位可见范围改善不保证模型读取正确；还有动态遮挡及错误跨视角对应。所有十长程仅软件接口覆盖，未做十项在线效果验证。',
 '多因素变化、模型RNG未固定；仅两个场景且一项中途停止，不报告总体成功率或单因素因果。',
 'root实际模型/推理档位未知；Hermes入口次数不等于其内部API调用数/用量。'],
 'configuration_complete':False,
 'evidence':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(files)]}
registry=read(REGISTRY);assert not any(e['id']==entry['id'] for e in registry['entries'])
registry['entries'].append(entry)
REGISTRY.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with INDEX.open('a',encoding='utf-8') as f:
 f.write('\n\n## 2026-10-14：三视角规划身份在线试点\n\n- 实验 `20261013-xvla-grounded-online`：[报告与原图](results/2026-10-13-xvla-grounded-online/summary.md)。\n- 入炉黄杯初态识别正确，但150步提前误判关门；双罐初态错认被复核沿用，87步人工停止。两项均未通过完整流程，历史补证未验证。\n- 相机只读接口启动故障已修并零步核验；334 tests/4 subtests。原收尾/退离/复位控制保留，无辅助器/子agent。\n')
print(json.dumps({'registered':entry['id'],'evidence_count':len(files),'entries':len(registry['entries'])}))
