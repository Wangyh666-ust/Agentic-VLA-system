from pathlib import Path
import json,hashlib,shutil
root=Path('/mnt/d/FYP'); here=root/'First_Phase/scene_demo'
p=here/'XVLA_CONSOLE.md'; s=p.read_text(); s=s.replace('17项软件测试通过（12项界面/接口测试、5项已有覆盖协议测试）','19项软件测试通过（14项界面/接口测试、5项已有覆盖协议测试）').replace('包含：草稿零提交','包含：空任务载入、预览先于任务绑定、三视图预览不调用识别、草稿零提交');p.write_text(s)
out=here/'results/2026-10-14-xvla-console-scene-first';out.mkdir(exist_ok=False)
for name in ('xvla_console.py','xvla_console.html','tests/test_xvla_console.py','XVLA_CONSOLE.md'):
    target=out/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(here/name,target)
shutil.copyfile(here/'results/interactive-console/scene-first-preview.png',out/'preview.png')
base=json.loads((here/'results/2026-10-14-partial-scope-offline/source_manifest.json').read_text())
assert all(hashlib.sha256((here/n).read_bytes()).hexdigest()==v for n,v in base.items())
report={'purpose':'先载入场景、展示三视图，再填写任务和规划','tests':'19 passed in 1.70s','browser':{'load_enabled_with_empty_task':True,'task_input_disabled_before_scene':True},'production_sources_unchanged':len(base),'new_model_calls':0,'new_simulation_actions':0,'subagents':0,'limits':'预览相机接入以既有接口和软件替身验证，未新启动仿真。'}
(out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
registry=here/'experiment_registry.json'; data=json.loads(registry.read_text());eid='20261014-xvla-console-scene-first'
assert not any(e['id']==eid for e in data['entries'])
evidence=[{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob('*')) if p.is_file()]
data['entries'].append({'id':eid,'title':'交互顺序修正：先场景三视图，后输入任务','date':'2026-10-14','question_keys':['xvla-console-scene-first-ui'],'tags':['xvla','ui','software-only'],'conditions':'载入仅接收scene；工作线程用原SensorPort与checker相机拍摄三视图；用户看图后提交任务。','findings':['19项软件测试通过，浏览器核对空任务可点击载入且任务框位于三视图之后。','识别/规划延后至任务提交；生产控制20文件哈希未变。'],'limits':['零新模型调用、零新仿真动作、零子agent；真实预览拍摄尚未新运行。'],'configuration_complete':False,'evidence':evidence})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with (here/'EXPERIMENT_INDEX.md').open('a') as f:
    f.write('\n\n## 2026-10-14：界面改为先看场景再填任务\n\n- `20261014-xvla-console-scene-first`：[验证](results/2026-10-14-xvla-console-scene-first/validation.json)。\n- 载入不要求任务，立即拍摄实际三视图；图片下填写任务后才调用识别与Hermes。19项软件测试通过，浏览器已刷新；零新仿真、零新模型调用。\n')
print(json.dumps({'archived':eid,'tests':19,'production_sources_unchanged':len(base)}))
