"""Adapt the existing read-only packet audit to the independent basket case."""
from pathlib import Path

ROOT = Path('D:/FYP')
OUT = ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-timeline-trial'
destination = OUT/'basket_independent/runtime/audit_basket.py'
assert not destination.exists()
source = (OUT/'runtime/audit_first.py').read_text(encoding='utf-8')
source = source.replace(";CASE=OUT/'microwave_mug'", ";OUT=OUT/'basket_independent';CASE=OUT/'basket_two'")
source = source.replace("'case':'microwave_mug'", "'case':'basket_two'")
source = source.replace("'current_door_vs_native'", "'current_checks_vs_native'")
source = source.replace("row.get('history_event_ids')", "row.get('evidence_event_ids')")
source = source.replace("'gate_ok':load(OUT/'microwave_assessment.json')['gate_ok']", "'gate_ok':case['complete']")
source = source.replace("'diagnostic':'图与中间状态确实进入真实请求；模型仍未引用过程图确认目标杯。call_04原推理把放置也理解为必须当前证据；最终门True但目标Unknown。此为观察，不证明唯一根因或总体可靠性。'", "'diagnostic':'输入协议、视觉判断、仿真目标和原收尾分开记录；成功不能单凭当前可见就称历史复用效果已验证。'")
source = source.replace("'last_current_images':checks[-1]['image_paths']", "'last_current_images':checks[-1]['image_paths'],'finish_status':pilot.get('finish_status'),'independent_streak':pilot.get('independent_streak'),'handoff_evidence':case.get('handoff_evidence'),'agent_result':case.get('agent_result')")
compile(source, str(destination), 'exec')
destination.write_text(source, encoding='utf-8')
print('prepared read-only basket audit; new model calls=0, actions=0')
