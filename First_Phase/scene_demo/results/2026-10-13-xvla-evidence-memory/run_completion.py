import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'First_Phase/scene_demo/plans/20261013-xvla-evidence-memory-v2.json').read_bytes())
plan['id']='20261013-xvla-evidence-memory-v3'
plan['analysis_steps'] += [
 '第二轮入炉第7次返回identity=uncertain且relation=outside、frames=[]，解析器拒绝；规则要求这种身份未知的判断不能反证同一目标，修为只有matched事实必须带frame。非法frame、matched无依据仍拒绝。',
 '只按原顺序重放attempt_02入炉已有7条原始v5响应；每条重新构造请求body必须逐字节SHA相同（含新中间状态），禁止读取v3旧判断。随后补尚未请求的第900步末帧，最多1次新Qwen。不重新采样或挑选旧响应。',
 '入篮及无历史负例复用第二轮真实结果；开门负例错误仍然保留，不能因为补完末帧而宣布通过。若补完有接口问题保留原错误，停止新增Qwen并报告剩余缺口。']
proposal=ROOT/'First_Phase/scene_demo/plans/20261013-xvla-evidence-memory-v3.json'
assert not proposal.exists();proposal.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
base=['wsl.exe','-d','Ubuntu','--cd','/mnt/d/FYP','--exec','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/mnt/d/FYP/First_Phase','/home/yhwang/fyp/xvla_native/venv/bin/python']
tests=[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'First_Phase/scene_demo/tests').glob('test_xvla*.py'))]
r=subprocess.run(base+['-m','pytest',*tests,'-q'],capture_output=True)
(OUT/'tests-final.stdout').write_bytes(r.stdout);(OUT/'tests-final.stderr').write_bytes(r.stderr);print(r.stdout.decode('utf-8'),flush=True)
if r.returncode:sys.exit(r.returncode)
r=subprocess.run([sys.executable,str(ROOT/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(proposal)],capture_output=True)
(OUT/'review-v3.stdout').write_bytes(r.stdout);(OUT/'review-v3.stderr').write_bytes(r.stderr);print(r.stdout.decode('utf-8'),flush=True)
if r.returncode:sys.exit(r.returncode)
r=subprocess.run(base+['-u','First_Phase/scene_demo/results/2026-10-13-xvla-evidence-memory/complete_replay.py'],capture_output=True)
(OUT/'completion.stdout').write_bytes(r.stdout);(OUT/'completion.stderr').write_bytes(r.stderr);print(r.stdout.decode('utf-8'),flush=True)
if r.returncode:print(r.stderr.decode('utf-8'),flush=True)
(OUT/'completion-exit.json').write_text(json.dumps({'returncode':r.returncode})+'\n')
sys.exit(r.returncode)
