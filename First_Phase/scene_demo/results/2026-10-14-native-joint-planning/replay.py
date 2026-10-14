"""Original-photo replay: no environment, policy, or physical actions."""
import json
import sys
import traceback
from pathlib import Path
from scene_demo import catalog, service
from scene_demo.xvla_planning import plan_from_capture

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
# Match XvlaSceneService.__init__ without creating any simulator/model.
service.MAX_BUDGET=900
case=sys.argv[1] if len(sys.argv)>1 else 'bowl_bottle'
if case=='bowl_bottle':
    source=ROOT/'results/interactive-console/runs/20261014T110703Z_bowl_drawer_382fcdca/planning_grounding/capture.json'
    scene='bowl_drawer';request='收拾桌面，将碗和瓶子归位'
else:
    specs={'table_tidy_stove':('goal_table','整理桌面，包括碗、酒瓶，并且打开灶台'),
           'heat_yellow_mug':('microwave_mug','帮我将黄杯子加热一下'),
           'discard_two_cans':('basket_two','帮我扔一下桌面上的两个罐头')}
    scene,request=specs[case]
    source=ROOT/'results/2026-10-14-partial-scope-offline'/case/'grounding/capture.json'
session={'session_id':'offline-'+case,'scene_id':scene,'scene_version':0,'state':'ready',
         'capability_backend':'xvla','capabilities':catalog.scene_capabilities(scene,backend='xvla'),
         'storage_policy':catalog.SCENES[scene].get('storage_policy',{})}
capture=json.loads(source.read_text())
home=Path(json.loads((ROOT/'results/interactive-console/shared_hermes_profile.json').read_text())['path'])
directory=OUT/(sys.argv[2] if len(sys.argv)>2 else case)
if directory.exists():raise RuntimeError('Preserve previous replay; choose a new output directory')
directory.mkdir()
try:
    result=plan_from_capture(capture,request,session,'offline-'+case,directory,home)
    (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'draft':result['draft'],'timing':result['timing'],'review_passed':result['review']['passed']},ensure_ascii=False),flush=True)
except Exception:
    (directory/'error.txt').write_text(traceback.format_exc());raise
