import json
from pathlib import Path
r=Path('First_Phase/scene_demo/results/2026-10-13-xvla-release-trial')
a=json.loads((r/'audit.json').read_bytes());p=json.loads((r/'archive/invariants.json').read_bytes())
print(json.dumps({'counts':a['counts'],'event_steps':[e['job_step'] for e in a['event_audits']],'invariants':p},ensure_ascii=True))
