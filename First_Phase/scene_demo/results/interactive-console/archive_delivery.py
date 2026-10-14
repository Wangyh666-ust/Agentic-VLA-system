from pathlib import Path
import hashlib,json,shutil
root=Path('/mnt/d/FYP'); here=root/'First_Phase/scene_demo'; out=here/'results/2026-10-14-xvla-console'
out.mkdir(exist_ok=False)
files=['xvla_console.py','xvla_console.html','start_xvla_console.ps1','Start X-VLA.cmd','XVLA_CONSOLE.md','tests/test_xvla_console.py']
for name in files:
    target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(here/name,target)
for name in ['validation.json','ui-preview.png']:
    shutil.copyfile(here/'results/interactive-console'/name,out/name)
registry=here/'experiment_registry.json'
data=json.loads(registry.read_text(encoding='utf-8-sig'))
identifier='20261014-xvla-console-software'
assert not any(e['id']==identifier for e in data['entries'])
evidence=[{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob('*')) if p.is_file()]
entry={'id':identifier,'title':'X-VLA交互工作台软件与浏览器验证（零新仿真）','date':'2026-10-14','question_keys':['xvla-console-software'],'tags':['xvla','hermes','interactive-ui','software-only'], 'conditions':'本地工作台，草稿捕获后由用户明确执行；真实浏览器查看三组旧初始三视图；生产流程20文件哈希未变。','findings':['17项软件测试通过；草稿不执行，重复/错归属/旧版本拒绝，取消跟踪后端终态。','三组历史示例、物品定位框、照片放大、完整/部分/不支持范围显示已浏览器核对；一键启动可用。','新增模型调用0、仿真动作0、子agent0。'],'limits':['新网页入口的实际在线执行尚未运行验证。','旧视觉误判、网络超时及最终稳定筛查问题没有因界面新增而解决。'],'configuration_complete':False,'evidence':evidence}
data['entries'].append(entry)
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (here/'EXPERIMENT_INDEX.md').open('a',encoding='utf-8') as stream:
    stream.write('\n\n## 2026-10-14：X-VLA交互工作台（软件验证）\n\n- 记录 `20261014-xvla-console-software`：[使用说明](XVLA_CONSOLE.md)，[验证记录](results/2026-10-14-xvla-console/validation.json)。\n- 真实Hermes草稿与执行隔离，用户确认后交给原X-VLA服务；显示三视图身份、能力缺口、过程照片、检查与收尾/复位。\n- 17项软件测试及三组历史图片浏览器检查通过，生产源码20文件未变；零新模型调用、零仿真动作、零子agent。新入口在线动作尚未验收。\n')
print(json.dumps({'registered':identifier,'evidence_files':len(evidence),'registry_entries':len(data['entries'])}))
