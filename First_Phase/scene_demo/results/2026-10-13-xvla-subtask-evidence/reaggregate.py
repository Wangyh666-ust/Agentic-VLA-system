"""Reapply final aggregation to the same archived Qwen answers; zero HTTP."""
import copy,json
from pathlib import Path
from scene_demo.xvla_subtask_evidence import SubtaskEvidence
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient
OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
results=[]
for case,name in [('v7_1_basket','historical_probe'),('v7_1_microwave','final_comparison'),('v7_1_negative','no_history')]:
    path=OUT/case/(name+'.json');meta=load(path);store=SubtaskEvidence(meta['scope'])
    for obs in meta['observations']:
        item=store.observe(obs['context'],obs['paths'],obs['sampling_role'],unknown_time=obs['captured_at'] is None)
        assert item['capture_id']==obs['capture_id']
    for event in meta['release_archive']:store.release(event['event_id'],store.observations[event['capture_id']])
    client=XvlaSubtaskReviewClient(env_path='/none',evidence_store=store)
    client.set_check_contract(store.observations[meta['current_capture_id']]['context'])
    records=[]
    for page in [meta['initial_review']]+meta['pages']:
        assert page['repair'] is None
        client._active_packet=page['packet'];client.parse_verdict(json.dumps(page['original_review']['raw']))
        record=client._apply(copy.deepcopy(client.last_review),page['packet'],page['metadata']);records.append(record)
    current=records[-1]['normalized']['current']
    blocked={r['id'] for r in current if r.get('protocol_invalid') or r.get('grounding_invalid')}
    effective=client._effective(current,blocked)
    results.append({'case':case,'source':str(path.relative_to(OUT)),
        'request_sha256':[p['metadata']['request_sha256'] for p in [meta['initial_review']]+meta['pages']],
        'goal':client._aggregate(effective),'checks':effective,'proofs':client.process_evidence_snapshot(),
        'new_Qwen_calls':client._calls,'new_environment_actions':0})
(OUT/'reaggregation.json').write_text(json.dumps({'reason':'Duplicate-object exclusion applies to distinct placement targets; a container and its contents can overlap in a view.',
    'results':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{'case':r['case'],'goal':r['goal'],'checks':[{k:c[k] for k in ('id','achieved')} for c in r['checks']]} for r in results]))
