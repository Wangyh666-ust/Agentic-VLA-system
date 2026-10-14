"""Offline mechanical derivation; never starts an online harness."""
import ast, difflib, hashlib, json, os, py_compile, shutil, subprocess, sys, tempfile
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parents[3]
old = out.parent / '2026-10-14-sim-fast'
shutil.copyfile(old / 'protected_controls.json', out / 'protected_controls.json')
for kind, source in [('bowl', 'bowl2_proposal.json'), ('chain', 'chain_proposal.json')]:
    p = json.loads((old/source).read_text(encoding='utf-8-sig'))
    p['id'] = '20261015-sim-io-' + kind
    p.pop('prior_attempt', None)
    p['prior_evidence'].append('20261014-sim-fast')
    p['history_review'].append({'experiment_id':'20261014-sim-fast','disposition':'reuse','reason':'Existing simulator baseline: bowl execution 152.7734s and chain execution 392.1394s; all four full native skills passed independent 5/5 and handoff. Reuse native completion evidence and original timing; compare only changed observation I/O.'})
    p['purpose'] = 'Measure simulator execution time and files saved by observation I/O optimization while preserving original full-native completion, release, retreat, reset, and independent five-sample acceptance.'
    p['hypothesis'] = 'Eliminating unused periodic captures and repeated same-state captures reduces simulator observation I/O while preserving original acceptance.'
    p['information_gain'] = 'New I/O timing, capture/cache counts and file sizes against existing original evidence; model RNG is not paired, so causal speed cannot be isolated from action-count differences unless those are checked.'
    p['changed_factors'] = [{'name':'execution_observation_io','before':'Periodic unused history captures, repeated force observations, and extra external 512px views on every geometry capture','after':'No unused periodic history; same-state observation cache; extra external 512px views on release only','reason':'Measure redundant simulator observation I/O reduction with original completion and control path preserved.'}]
    p['fixed_factors']['planning'] = 'qwen3.8-max joint three-view planning plus one independent review; two calls'
    p['fixed_factors']['protected_controls'] = json.loads((out/'protected_controls.json').read_text())
    p['configuration'].update(observation_io='simulator_deduplicated_v1', implementation_baseline='ebfff5e')
    p['repeat'] = {'needed':True,'reason':'Compare new I/O against existing original evidence, not establish a new native success-rate baseline.'}
    p['related_search'] = 'Reuse original prior history and registered 20261014-sim-fast matching question; original successful bowl/chain evidence supplies latency and all-four-skills acceptance baseline.'
    p['success_criteria'] = ('Optimized path actually skips captures and cache_hits>0; original synchronized release wrist/agentview/sideview images present; native trigger>0, actual release_open_actions/movement/home>0; independent five-sample acceptance, next reference, queue/proprio/controller resets and all previous goals pass; execution VLM calls=0 and planning calls=2 recorded. Record monotonic stage timings, action/capture/cache counts and file sizes. ' + ('Prerequisite: bowl_timing.completed=true and every bowl pilot handoff_evidence.ok=true. ' if kind=='chain' else ''))
    p['failure_criteria'] = 'Expected branch not triggered or actual target actions zero means trigger-chain failure and target effect unverified; stop dependent validation. Executed actions with acceptance failure are effect failures. Preserve original failures and never automatically rerun.'
    p['unknown_criteria'] = 'Missing evidence, startup/network/interface failure or unpaired model RNG/action-count differences remain unknown; no claim of causal speed or native reliability from one case.'
    p['scope']['stop_rule'] = 'Maximum one case, no automatic rerun. If first bowl target branch is not triggered, actual actions zero or acceptance fails, stop dependent chain. During chain any expected trigger/action/acceptance failure stops later skills; preserve evidence.'
    (out/(kind+'_proposal.json')).write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    script = (old/('online_bowl2_valid.py' if kind=='bowl' else 'online_chain.py')).read_text(encoding='utf-8-sig')
    script = script.replace('bowl2_valid','bowl').replace('bowl2_review','bowl_review').replace('bowl2_draft','bowl_draft').replace('bowl2_final','bowl_final').replace('bowl2_source','bowl_source')
    script = script.replace("['xvla_sim_finish.py'", "['xvla_sim_io.py','xvla_sim_finish.py'")
    script = script.replace("print(json.dumps({'elapsed':", "timing['stage_changes'].append({'monotonic':time.monotonic(),'elapsed':time.monotonic()-started,'stage':key[0],'message':key[1]})\n            print(json.dumps({'elapsed':")
    script = script.replace("'completed':False}", "'completed':False,'stage_changes':[]}")
    if kind == 'chain':
        marker = "assert json.loads((out/'bowl_timing.json').read_text())['completed'] is True"
        script = script.replace(marker, marker + "\nbowl_final=json.loads((out/'bowl_final.json').read_text(encoding='utf-8-sig'))\nassert bowl_final['jobs'], 'bowl jobs missing'\nfor job in bowl_final['jobs']:\n    pilot=json.loads((Path(job['run_dir'])/'pilot_result.json').read_text(encoding='utf-8-sig'))\n    assert pilot['handoff_evidence']['ok'] is True, 'bowl handoff prerequisite failed'")
    (out/('online_'+kind+'.py')).write_text(script,encoding='utf-8')
    ast.parse(script)
    with tempfile.TemporaryDirectory(prefix='sim_io_compile_') as temp:
        py_compile.compile(str(out/('online_'+kind+'.py')),cfile=str(Path(temp)/(kind+'.pyc')),doraise=True)
    result = subprocess.run([sys.executable,str(root/'.agents/skills/fyp-experiment-review/scripts/review.py'),'check','--proposal',str((out/(kind+'_proposal.json')).relative_to(root))],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out/(kind+'_review.json')).write_bytes(result.stdout)
    print(kind, 'review_exit', result.returncode, result.stdout.decode('utf-8'))
    if result.stderr: print(result.stderr.decode('utf-8'))
    print(''.join(difflib.unified_diff((old/('online_bowl2_valid.py' if kind=='bowl' else 'online_chain.py')).read_text(encoding='utf-8-sig').splitlines(True),script.splitlines(True),fromfile='baseline_'+kind,tofile='new_'+kind)))
assert (old/'protected_controls.json').read_bytes()==(out/'protected_controls.json').read_bytes()
protected=json.loads((out/'protected_controls.json').read_text())
assert len(protected)==10
assert all(hashlib.sha256((root/'First_Phase/scene_demo'/p).read_bytes()).hexdigest()==h for p,h in protected.items())
assert not (out/'bowl_timing.json').exists() and not (out/'chain_timing.json').exists()
print('AST/compile passed; protected controls exact bytes and 10 hashes passed; online run counts zero')
