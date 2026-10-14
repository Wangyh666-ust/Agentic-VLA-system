"""Summarize original online outputs without modifying decisions or evidence."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'cases':{},'new_subagents':0,'new_grasp_transport_helpers':0}
for name,relative in [('microwave_mug','microwave_mug'),('basket_two_startup','basket_two'),('basket_two','basket_renderfix/basket_two')]:
 d=OUT/relative
 if not (d/'result.json').exists():continue
 raw=read(d/'result.json');native=[r for p in d.rglob('native_steps.jsonl') for r in rows(p)]
 pilots=[read(p) for p in d.rglob('pilot_result.json')]
 inventory=[read(p) for p in d.rglob('inventory_confirmed.json')]
 verified=[read(p) for p in d.rglob('selected_targets_review.json')]
 events=[r for p in d.rglob('process_events.jsonl') for r in rows(p)]
 reviews=[r for p in d.rglob('reviews.jsonl') for r in rows(p)]
 requests=list(d.rglob('call_*_request.json'));bad_sources=[]
 for p in requests:
  for image in read(p)['images']:
   source=image['path']
   path=Path(source.replace('/mnt/d/FYP',str(ROOT)))
   if not path.exists() or sha(path)!=image['sha256']:bad_sources.append({'request':str(p),'image':source})
 checks=[r for p in d.rglob('checks.jsonl') for r in rows(p)]
 check_summary=[]
 for r in checks:
  meta=r.get('initial_check',{}).get('metadata',{})
  check_summary.append({'step':r.get('step'),'effective_checks':meta.get('effective_checks'),
      'process_evidence':meta.get('process_evidence'),'actual_Qwen_calls':meta.get('actual_Qwen_calls')})
 result['cases'][name]={'complete':raw.get('complete'), 'reason':raw.get('reason'), 'error':raw.get('error'),
  'plan':{k:(raw.get('plan') or {}).get(k) for k in ('state','plan_success','target_bindings','native_skill_route','capability_ids','error')},
  'source_counts':{s:sum(r.get('source')==s for r in native) for s in ('settling','prepare','vla','finish')},
  'first_native_success_vla_step':next((r['counts']['vla'] for r in native if r.get('native_success') and r.get('source')=='vla'),None),
  'native_success_final':native[-1].get('native_success') if native else None,
  'inventories':inventory,'selected_targets_review':verified,'release_events':len(events),
  'release_images':[[i['view'] for i in r['images']] for r in events],
  'qwen_requests':len(requests),'qwen_responses':len(list(d.rglob('call_*_response.json'))),
  'historical_proofs':[p for r in reviews for p in r.get('new_proofs',[]) if p.get('kind')=='historical_review'],
  'finish_status':[p.get('finish_status') for p in pilots],
  'handoff_evidence':[p.get('handoff_evidence') for p in pilots],
  'job_reasons':[p.get('job',{}).get('ended_reason') for p in pilots],
  'checks':check_summary,'source_image_hash_errors':bad_sources}
manifest=read(OUT/'source_manifest.json')
result['v1_source_snapshot_matches']=all(sha(OUT/'source_snapshot'/p)==expected for p,expected in manifest.items())
result['current_changes_from_v1']=[p for p,expected in manifest.items() if sha(ROOT/p)!=expected]
fixed=OUT/'basket_renderfix/source_manifest.json'
if fixed.exists():
 result['renderfix_source_freeze_matches']=all(sha(ROOT/p)==expected and sha(OUT/'basket_renderfix/source_snapshot'/p)==expected for p,expected in read(fixed).items())
protected=['xvla_monitor.py','joint_home.py','safe_exit.py','safe_exit_contacts.py','placement_completion.py','finish_v1/control.py']
old=read(ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-subtask-evidence/protected_manifest.json') if (ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-subtask-evidence/protected_manifest.json').exists() else None
baseline=read(ROOT/'First_Phase/scene_demo/plans/20261013-xvla-timeline-trial.json')['configuration']['source_sha256']
baseline.update(read(ROOT/'First_Phase/scene_demo/results/2026-10-13-xvla-subtask-evidence/final_source_manifest.json'))
result['protected_control_unchanged']={p:sha(ROOT/'First_Phase/scene_demo'/p)==baseline['First_Phase/scene_demo/'+p] for p in protected}
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name,value in result['cases'].items():
 print(json.dumps({k:v for k,v in value.items() if k not in ('inventories','checks','selected_targets_review','finish_status')},ensure_ascii=False))
