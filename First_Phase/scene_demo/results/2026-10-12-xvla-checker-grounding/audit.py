"""Explicit inference entry point; preparation never invokes this file."""
import json
from pathlib import Path
from prepare_audit import HERE,sha
from scene_demo.xvla_checker import GroundedQwenClient


def main():
    if (HERE/'result.json').exists() or (HERE/'calls').exists():
        raise FileExistsError('refuse to overwrite audit evidence')
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for p,digest in manifest.items():
        if sha(p)!=digest:raise RuntimeError('source SHA mismatch: '+p)
    cases=json.loads((HERE/'cases.json').read_text());assert len(cases)==9
    client=GroundedQwenClient(max_calls=9);results=[]
    output={'new_physical_actions':0,'xvla_calls':0,'hermes_calls':0,'qwen_calls':0,
            'cases':results,'complete':False,'all_allowed_match':False}
    def save():
        output['qwen_calls']=client._calls
        output['all_allowed_match']=len(results)==9 and all(r['allowed_match'] for r in results)
        (HERE/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    save()
    current=None
    try:
        for case in cases:
            current=case['name']
            client.observe_capture(case['capture'])
            client.audit_dir=HERE/'calls'/case['name']
            response=client.analyze(case['source_context'],case['source_image_paths'])
            goal=response['visual_goal_verdict']['goal_achieved']
            results.append({'name':case['name'],'raw':client.last_two_question_verdict,'goal_achieved':goal,
                            'allowed_match':any(goal is x for x in case['expected_allowed']), 'usage':response.get('usage'),'response':response})
            save()
        output['complete']=True
    except Exception as exc:
        output['error']={'type':type(exc).__name__,'message':str(exc),'current_case':current}
        raise
    finally:
        save()
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
