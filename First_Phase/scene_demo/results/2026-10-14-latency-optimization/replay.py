"""Replay immutable observations through the production checker; zero robot actions."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'First_Phase'))
from scene_demo.xvla_subtask_review import XvlaSubtaskReviewClient

def main():
    request=ROOT/sys.argv[1]
    out=Path(__file__).resolve().parent/sys.argv[2]
    if out.exists():raise RuntimeError('Do not overwrite a replay')
    out.mkdir()
    saved=json.loads(request.read_text())
    packet=json.loads(saved['prompt'].splitlines()[-1])
    original=[json.loads(line) for line in (request.parent.parent/'subtask_evidence/observations.jsonl').read_text().splitlines()]
    wanted=next(o for o in packet['observations'] if o['capture_id']==packet['current_capture_id'])
    target=next(o for o in original if o['capture_id']==wanted['capture_id'])
    if len(sys.argv)>4:
        target=next(o for o in original if o['captured_at'] is not None and abs(o['captured_at']-float(sys.argv[4]))<1e-6)
    client=XvlaSubtaskReviewClient(max_calls=int(sys.argv[3]) if len(sys.argv)>3 else 8)
    client.audit_dir=out/'model_requests'
    client.store.directory=out/'evidence'
    client.set_check_contract(target['context'])
    mapping={}
    for old in original:
        if old['captured_at'] is not None and old['captured_at']>target['captured_at']:continue
        if [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old['paths']]!=old['sha256']:
            raise RuntimeError('Archived pixels changed')
        mapping[old['capture_id']]=client.store.observe(old['context'],old['paths'],old['sampling_role'],unknown_time=old['captured_at'] is None)
    # Reuse initial identity descriptions only, never old verdicts or proofs.
    first=packet['initial_capture_id']
    frame_map={f:(o['capture_id'],i) for o in packet['observations'] for i,f in enumerate(o['frames'])}
    for binding in packet['bindings']:
        anchors=[]
        for ref in binding['anchors']:
            cid,i=frame_map[ref['frame']]
            if cid==first:
                anchors.append({'capture_id':mapping[cid]['capture_id'],'view':client.required_views[i],'box':ref['box']})
        if anchors:client.store.bindings[binding['target_id']]={'target_id':binding['target_id'],'features':binding['features'],
            'anchors':anchors,'distractors':[],'reviewed_at':mapping[first]['captured_at'],'review_id':'reused_initial_identity'}
    event_file=request.parent.parent/'subtask_evidence/events.jsonl'
    if event_file.exists():
        for line in event_file.read_text().splitlines():
            old=json.loads(line)
            if old['capture_id'] in mapping:client.store.release(old['event_id'],mapping[old['capture_id']])
    manifest={'request':str(request),'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),
        'target_time':target['captured_at'],'new_robot_actions':0,'old_observations':len(mapping),
        'saved_releases':len(client.store.events),'model':client.model}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    try:
        result=client._inspect(target['context'],target['paths'])
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        summary={**manifest,'calls':client._calls,'goal_achieved':result['verdict']['operation_achieved'],
            'handoff_review_triggered':result['handoff_review'] is not None,
            'checks':result['effective_checks'],'regions':result['destination_regions'],'proofs':result['process_evidence']}
    except Exception as exc:
        summary={**manifest,'calls':client._calls,'error_type':type(exc).__name__,'error':str(exc)}
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        raise
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
