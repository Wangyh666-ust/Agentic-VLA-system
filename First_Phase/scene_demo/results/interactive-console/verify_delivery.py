from pathlib import Path
import hashlib,json,socket
here=Path('/mnt/d/FYP/First_Phase/scene_demo')
out=here/'results/interactive-console'
manifest=json.loads((here/'results/2026-10-14-partial-scope-offline/source_manifest.json').read_text())
unchanged={name:hashlib.sha256((here/name).read_bytes()).hexdigest()==value for name,value in manifest.items()}
assert all(unchanged.values()),unchanged
closed={}
for port in (8769,8794):
    with socket.socket() as probe:
        probe.settimeout(.3)
        closed[str(port)]=probe.connect_ex(('127.0.0.1',port))!=0
assert all(closed.values()),closed
report={'software_tests':{'command':'sim_venv/bin/python -m pytest -q First_Phase/scene_demo/tests/test_xvla_console.py First_Phase/scene_demo/tests/test_planning_scope.py','result':'17 passed in 1.64s'},'production_sources_unchanged':unchanged,'model_and_motion_ports_closed':closed,'new_model_calls':0,'new_simulation_actions':0,'subagents':0,'online_ui_execution':'not run; user-controlled next step','implementation_model_evidence':'root model/reasoning not independently exposed; no executor agent invoked'}
(out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'tests':17,'unchanged_production_files':len(unchanged),'model_and_motion_ports_closed':closed}))
