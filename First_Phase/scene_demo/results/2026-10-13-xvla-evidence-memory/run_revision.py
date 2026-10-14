"""Preserve revision rationale, software output, review output, then replay."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path('/mnt/d/FYP');OUT=Path(__file__).resolve().parent
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=json.loads((ROOT/'First_Phase/scene_demo/plans/20261013-xvla-evidence-memory.json').read_bytes())
plan['id']='20261013-xvla-evidence-memory-v2'
plan['analysis_steps'] += [
 'attempt_01实测16次：新放置证据已保留；micro前期门开误报关、最后一条多填无关relation触发解析拒绝；basket第二目标仍错认桌面物品。整轮未通过，保留全部记录。',
 '本修订按关系类型输出模板，状态以observed_state为唯一字段，忽略该类型无意义的冗余字段并保留原始响应；wrong-domain/缺观察/伪造来源仍拒绝。禁止借冗余relation覆盖open/unknown。',
 '提示先观察current再身份对照，历史True/False和旧位置不是当前视觉事实；先形状后配色、查wrist，区分罐与瓶；当前遮挡填unknown由代码继承，不让模型把旧证据改写成当前可见。',
 '最多再完整回放17次请求，上述通用提示与接口同改，不能独立归因。仍按原正负例判据，不挑选两轮最好结果拼接；若真实视觉仍未通过，记录剩余缺口并保持线上候选关闭。']
proposal=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-evidence-memory-v2.json'
assert not proposal.exists();dump(proposal,plan)
tests=sorted((ROOT/'First_Phase/scene_demo/tests').glob('test_xvla*.py'))
r=subprocess.run([sys.executable,'-m','pytest',*[str(p) for p in tests],'-q'],cwd=ROOT,capture_output=True)
(OUT/'tests.stdout').write_bytes(r.stdout);(OUT/'tests.stderr').write_bytes(r.stderr)
print(r.stdout.decode(),flush=True)
if r.returncode:sys.exit(r.returncode)
r=subprocess.run([sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(proposal)],cwd=ROOT,capture_output=True)
(OUT/'review-v2.stdout').write_bytes(r.stdout);(OUT/'review-v2.stderr').write_bytes(r.stderr)
print(r.stdout.decode(),flush=True)
if r.returncode:sys.exit(r.returncode)
with (OUT/'replay-v2.stdout').open('xb') as out,(OUT/'replay-v2.stderr').open('xb') as err:
    p=subprocess.Popen([sys.executable,'-u',str(OUT/'replay.py'),'attempt_02'],cwd=ROOT,stdout=subprocess.PIPE,stderr=err)
    for line in p.stdout:out.write(line);out.flush();print(line.decode().rstrip(),flush=True)
    code=p.wait()
dump(OUT/'revision-exit.json',{'replay_returncode':code})
sys.exit(code)
