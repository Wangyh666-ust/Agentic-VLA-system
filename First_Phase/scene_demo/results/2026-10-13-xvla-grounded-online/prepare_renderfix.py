"""Freeze the zero-VLA startup correction and finish the pending basket test."""
import json,hashlib,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
NEW=OUT/'basket_renderfix';NEW.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
probe=read(OUT/'camera_renderfix_probe/result.json')
assert all(all(r[k] is True for k in ('camera_only_qpos_exact','camera_only_qvel_exact','camera_only_time_exact','native_policy_pixels_exact')) for r in probe)
source=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-grounded-online.json'
target=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-grounded-online-renderfix.json'
assert not target.exists()
p=read(source);p['id']='20261013-xvla-grounded-online-renderfix'
p['purpose']='修复双罐启动时只读相机渲染触发的状态变更后，完成用户已授权但尚未执行VLA的双罐在线测试；初态身份/规划/过程检查/完整交接目标不变，入炉不重跑。'
p['changed_factors']=[{'name':'camera_update','before':'mj_forward recomputes physical dynamics and can normalize quaternion','after':'mj_camlight updates only camera/light transforms; exact qpos/qvel/time and native pixels verified on saved failed-start state','reason':'remove demonstrated read-only rendering side effect; no tolerance relaxation or policy change'}]
p['information_gain']='首个双罐启动仅10步原生settling，零Qwen/Hermes/VLA；相机断言阻断，原图状态复现确认mj_forward浮点归一化。修复后首次实际执行双罐，原失败记录保留，不增加入炉样本。'
p['scope']['max_cases']=1;p['new_physical_actions']=3900
p['scope']['stop_rule']='仅完成双罐一次实际执行；再有入口/模型/控制/身份错误停止，不自动重跑。入炉不重跑，不改变视觉判据、提示词或原收尾/复位参数，不加抓取搬运辅助器。'
p['configuration']['online_cases']=['basket_two']
p['configuration']['camera_update']='mj_camlight'
p['configuration']['failed_start_record']=(OUT/'basket_two/result.json').relative_to(ROOT).as_posix()
p['configuration']['camera_read_only_probe']=(OUT/'camera_renderfix_probe/result.json').relative_to(ROOT).as_posix()
hashes={}
for name in p['configuration']['source_sha256']:
 path=ROOT/name;dest=NEW/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
p['configuration']['source_sha256']=hashes
p['repeat']={'needed':False,'reason':'第一启动零VLA，修复已证实相机状态副作用后完成未执行的独立双罐，不重复已完成入炉样例。'}
dump(target,p);dump(NEW/'source_manifest.json',hashes)
launch=(OUT/'launch.py').read_text(encoding='utf-8')
launch=launch.replace("OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-grounded-online'", "OUT=ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-grounded-online/basket_renderfix'")
launch=launch.replace('plans/20261013-xvla-grounded-online.json','plans/20261013-xvla-grounded-online-renderfix.json')
launch=launch.replace("('microwave_mug','basket_two')", "('basket_two',)")
# The host performs the full review before launch; run_case repeats it directly
# before environment creation. Avoid a third identical driver-side validation.
launch=launch.replace('check()\nfor port', 'for port').replace('  check()\n  argv=', '  argv=')
(NEW/'launch.py').write_text(launch,encoding='utf-8')
print(target)
