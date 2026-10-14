import json,hashlib,subprocess,shutil
from pathlib import Path
root=Path('D:/FYP');scene=root/'First_Phase/scene_demo';out=scene/'results/2026-10-13-xvla-process-evidence';soft=out/'software'
reg=json.loads((scene/'experiment_registry.json').read_bytes());assert len(reg['entries'])==47 and sum(len(e['evidence']) for e in reg['entries'])==7608
assert all(hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256'] for item in reg['entries'] for e in item['evidence'])
for name in ('experiment_registry.json','EXPERIMENT_INDEX.md'):shutil.copyfile(scene/name,soft/(name+'.before'))
prior=['20261013-xvla-all-native-software','20261013-xvla-native-three-view-software','20261012-xvla-persistent-pilot','20261012-xvla-input-audit','20261012-xvla-checker-grounding']
reasons=['复用10接口与原下游软件资格，补历史证据缺口，不作为本轮效果证据。','复用三视图输入来源和边界软件资格，不作为物理效果证据。','复用四链未完成与末段收尾历史来源，不重复实验。','复用真实PNG/labels/time审计来源，不当新模型判断。','复用误识别与unknown历史，不能充当本轮检查器效果证据。']
p={'schema_version':1,'id':'20261013-xvla-process-evidence-software','intent':'analyze_existing','question_key':'xvla-process-evidence','new_physical_actions':0,'purpose':'为全部10长程技能补统一松手三视图过程证据，验证事件可达、照片与编码器同步及历史融合/撤销，实际效果尚未验证','prior_evidence':prior,'history_review':[{'experiment_id':i,'disposition':'reuse','reason':r} for i,r in zip(prior,reasons)],'analysis_steps':['接口来源核对','实测开口事件正负例','三图来源与逐项历史融合/撤销','10技能同一机制','旧下游与证据保留'],'related_search':'主Agent已读实际runtime.record/poll/_check、SensorPort.capture/read_robot、Qwen1..6图和10技能定义，本轮无新物理实验。'}
path=out/'software_review_proposal.json';path.write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode())
argv=['D:/miniconda/python.exe',str(root/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str(path)];(soft/'check.command.json').write_text(json.dumps(argv));r=subprocess.run(argv,capture_output=True);(soft/'check.stdout').write_bytes(r.stdout);(soft/'check.stderr').write_bytes(r.stderr);(soft/'check.exitcode').write_text(str(r.returncode));print(r.stdout.decode());assert r.returncode==0
