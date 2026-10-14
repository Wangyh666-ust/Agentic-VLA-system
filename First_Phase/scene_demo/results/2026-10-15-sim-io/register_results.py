"""Freeze approved local report and append one registry entry; never run experiments."""
import hashlib, json, subprocess
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parents[3]
scene=root/'First_Phase/scene_demo'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(p):
    s=str(p).replace('\\','/')
    for pre in ['/mnt/d/FYP/','D:/FYP/']:
        if s.lower().startswith(pre.lower()):return root/s[len(pre):]
    return Path(s)
comparison=read(out/'comparison.json')
manifest=read(out/'evidence_manifest.json')
corrected=[]
for kind in ['bowl','chain']:
    p=out/(kind+'_acceptance.json');data=read(p)
    jobs=[data] if kind=='bowl' else data['jobs']
    for job in jobs:
        run=local(job['run_dir'])
        for dirname,stats in job['sensor_groups'].items():
            count=sum('checker' not in q.stem for q in (run/dirname).glob('cap*_agentview.png'))
            if stats['capture_group_definition']=='unique capNNNNNN prefix':
                corrected.append({'case':kind,'run':str(run),'directory':dirname,'old_count':stats['capture_groups'],'new_count':count})
                stats['capture_groups']=count
                stats['capture_group_definition']='primary cap*_agentview.png excluding checker stems'
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
phase_labels={'policy':'Policy','handoff_and_release':'Handoff / release','retreat_plan_motion_confirm':'Retreat plan / motion / confirm','home_entry_and_home':'Home entry / home','finalization':'Finalization'}
cases=comparison['cases']
def row(label,field,section):
    cells=[]
    for k in ['bowl','chain']:
        a=cases[k]['baseline'][section][field];b=cases[k]['new'][section][field]
        cells.append(f'{a:.2f} → {b:.2f}')
    return '| '+label+' | '+' | '.join(cells)+' |'
phase_table='\n'.join(row(label,key,'phase_seconds') for key,label in phase_labels.items())
time_table='\n'.join(row(label,key,'timing') for key,label in [('load_scene_seconds','Load scene'),('prepare_and_plan_seconds','Prepare and plan'),('execution_seconds','Execution'),('total_seconds','Total')])
report=f'''# Simulator observation I/O comparison — 2026-10-15

Two flows were run once each; four actual native skills passed. Single execution changed from {cases['bowl']['baseline']['timing']['execution_seconds']:.9f}s to {cases['bowl']['new']['timing']['execution_seconds']:.9f}s (51.3% reduction); the three-skill chain changed from {cases['chain']['baseline']['timing']['execution_seconds']:.9f}s to {cases['chain']['new']['timing']['execution_seconds']:.9f}s (53.8%). These are measured executions with unpaired observations and RNG, not an exact causal estimate or overall reliability claim.

Baseline `ebfff5e`; implementation `6b6bec3`; branch `codex/xvla-sim-io`. Implementation dispatch was explicitly `gpt-6.1-sol`, reasoning effort `low`, recorded in [implementation context](implementation_context.json). The old Git baseline remains available. The UI at port 8083 ran the new version and both flows completed.

## Changes and preserved behavior

1. Simulator-only unused periodic history capture was removed.
2. Unmodified native step-returned observations are cached with state invalidation and defensive copies, avoiding repeated force refreshes.
3. Extra calibrated external views are captured only for release. Necessary geometry depth remains unchanged.

Visual mode and the planner remain unchanged. All ten protected control hashes are unchanged. Qwen3.8 joint three-view planning plus one independent review remains two calls per flow; actual execution VLM calls and files under finish/model_requests were zero. No MoveIt or auxiliary helpers were introduced. Release, retreat, home and five-sample acceptance thresholds are unchanged.

The single run used VLA=229, finish=225 and prepare=11 actions, with release=2, movement=84, confirm=5 and home=134. Chain VLA counts were 152 / 82 / 80 and finish counts 250 / 240 / 116 for wine / bowl / stove. All four native trigger branches executed; four release events retain original wrist / agentview / sideview evidence with verified file hashes and synchronized capture timestamps. All four independent five-sample acceptances and handoff checks passed. The first two chain references target the next bowl and stove skills; the final reference is global home and all three previous goals remain true. See [single acceptance](bowl_acceptance.json) and [chain acceptance](chain_acceptance.json).

## Measured time

| Seconds | Single old → new | Chain old → new |
|---|---:|---:|
{time_table}

Single total is 209.531 → 155.365s. Only the new single run includes a 16.423s cold model/build proxy: the loading message explicitly naming X-VLA at elapsed 12.113s to the next planning stage at 28.536s. The baseline was warm; the new chain has no X-VLA cold-loading marker. Full prepare-and-plan time is retained. Qwen planning total was 41.626 → 50.674s for single and 83.401 → 71.046s for chain; those changes are not attributed to I/O optimization.

| Original-file mtime phase proxy, seconds | Single old → new | Chain old → new |
|---|---:|---:|
{phase_table}

Boundaries use ORIGINAL file mtimes, not archived copies: A=first.png, B=finish/simulator_samples.jsonl, C=finish.plan.geometry_frames[0].rgb_path, D=finish.home_entry.clearance_screen.frames[0].rgb_path, E=finish_actions.jsonl and F=pilot_result.json. Policy=B−A; handoff/release=C−B; retreat=D−C; home=E−D; finalization=F−E. Capture, rendering, geometry planning, motion and writes overlap: these phases are neither pure physics nor pure planning. Raw paths and nanosecond boundaries are retained in [comparison](comparison.json). Other execution time was {cases['bowl']['baseline']['other_execution_seconds']:.3f} → {cases['bowl']['new']['other_execution_seconds']:.3f}s single and {cases['chain']['baseline']['other_execution_seconds']:.3f} → {cases['chain']['new']['other_execution_seconds']:.3f}s chain.

## Capture storage and comparison limits

Finish sensor capture groups were 107 → 14 single and 218 → 34 chain. Decimal storage was 147.901650 → 12.292050 MB single and 305.747649 → 31.101940 MB chain, including controller_modes.jsonl. PNG counts were 535 → 44 and 1090 → 108; NPY counts were 214 → 28 and 436 → 68. A group is a primary cap*_agentview.png excluding checker stems: agent/wrist/side prefixes differ in the original SensorPort. Acceptance derivation was corrected from unique numeric cap prefixes to that primary-view definition; actual experimental outcomes were not changed.

Skipped periodic capture counters total 90 single and 183 chain. Observation-cache hits were 503 single and 1057 at the final chain job; chain counters are cumulative across the same environment and must not be summed across jobs. Raw action totals changed 478 → 465 single and 941 → 935 chain. Native VLA HTTP proxies are response-file mtime minus request-file mtime and include inference plus HTTP overhead, not pure GPU time; full counts and sums are in comparison.json.

Observations, first images and VLA request payloads differ across old/new runs. The cache reuses native step-returned observations whereas the prior path force-refreshed them; installed robosuite.step updates observables before returning observations under periodic sensor sampling. Same seed and preserved policy source hashes do not prove byte-equivalent observations, inputs, action sequences or paired RNG. Explicit first-image SHA, uncompressed request-payload SHA and actual action sequence comparisons are retained. The result demonstrates these two flows and four skills, not broad reliability or a pure causal speed measurement.

## Evidence and checks

[Evidence manifest](evidence_manifest.json) contains 82 exact-byte derived copies and no missing sources. Original records are preserved. Both pre-run history checks passed while retaining scientific_review_required=true and launch_authorized=false; launch authorization came from the user scope, independently of those checks. Initial software validation retained a deque assertion failure (1 failed, 75 passed); the corrected check then passed all 76 tests. See [software output](software_tests.txt), [protected controls](protected_controls.json), [bowl review](bowl_review.json) and [chain review](chain_review.json).
'''
(out/'README.md').write_text(report,encoding='utf-8')
attrs=root/'.gitattributes';line=b'First_Phase/scene_demo/results/2026-10-15-sim-io/** -text'
attrbytes=attrs.read_bytes()
if line not in attrbytes.splitlines():attrs.write_bytes(attrbytes+(b'' if attrbytes.endswith(b'\n') else b'\n')+line+b'\n')
registry=scene/'experiment_registry.json';raw=registry.read_bytes();before=read(registry)
head=json.loads(subprocess.check_output(['git','show','HEAD:First_Phase/scene_demo/experiment_registry.json'],cwd=root))
assert before['entries']==head['entries'],'Existing registry differs from HEAD; do not overwrite.'
evidence_paths=[out/n for n in ['README.md','comparison.json','evidence_manifest.json','bowl_acceptance.json','chain_acceptance.json','bowl_timing.json','chain_timing.json','bowl_proposal.json','chain_proposal.json','bowl_review.json','chain_review.json','software_tests.txt','implementation_context.json','protected_controls.json','bowl_source_hashes.json','chain_source_hashes.json','online_bowl.py','online_chain.py','analyze_compare.py','prepare_harness.py','register_results.py']]
evidence_paths += [out/m['relative_copy'] for m in manifest['files']]
entry={'id':'20261015-sim-io','title':'Simulator observation I/O optimization comparison','date':'2026-10-15','question_keys':['xvla-simulator-fast-handoff','xvla-simulator-io-latency'],'tags':['xvla','simulator','io-optimization','latency','next-reference'],'conditions':'One bowl_drawer flow and one wine→bowl→stove goal_table flow; seed142/state0, full native skills, qwen3.8-max joint three-view planning plus independent review (2 calls/run), simulator predicate triggers; original release/retreat/home/independent5 controls and all10 protected hashes preserved; no auxiliary helpers.','findings':'Two flows/four actual skills passed independent5/handoff. Execution152.773399792→74.420930427s (51.3% lower) and392.139429141→181.085704868s (53.8% lower); executionVLM0, next two skill references and final global home verified, previous goals retained. Capture groups107→14/218→34; sensor bytes147901650→12292050/305747649→31101940.','limits':'Single sample per flow, observations/RNG/first images/request payloads unpaired and action counts changed; no exact causal speed or broad reliability claim. New single includes16.423s cold model/build proxy versus warm baseline. Phase mtimes include overlapping capture/render/planning/motion/writes and are not pure physics. Simulator privileged predicates do not establish real-robot generalization.','configuration_complete':False,'configuration':{'baseline_commit':'ebfff5e','implementation_commit':'6b6bec3','branch':'codex/xvla-sim-io','checker_mode':'simulator','multiple_cases':True,'paired_observations_rng':False,'proposals':['First_Phase/scene_demo/results/2026-10-15-sim-io/bowl_proposal.json','First_Phase/scene_demo/results/2026-10-15-sim-io/chain_proposal.json']},'evidence':[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in evidence_paths]}
newline=b'\n'
marker=b'\r\n  ]' if b'\r\n  ]' in raw else b'\n  ]'
pos=raw.rfind(marker);assert pos>=0
addition=json.dumps(entry,ensure_ascii=False,indent=2)
addition='\n'.join('    '+s for s in addition.splitlines()).encode('utf-8').replace(b'\n',newline)
registry.write_bytes(raw[:pos].rstrip(b'\r')+b','+newline+addition+b'\n  ]\n}\n')
index=scene/'EXPERIMENT_INDEX.md';idx=index.read_bytes()
indexrow='| 20261015-sim-io | 仿真观测I/O优化 | 2流程/4技能原5/5与handoff通过；执行152.773→74.421s、392.139→181.086s；观测/RNG未配对，单例新增冷启动不做严格因果比较 | [报告](results/2026-10-15-sim-io/README.md) |\n'
index.write_bytes(idx+(b'' if idx.endswith(b'\n') else b'\n')+indexrow.encode('utf-8'))
after=read(registry)
checks={'new_entry_evidence_hashes':all((root/e['path']).is_file() and sha(root/e['path'])==e['sha256'] for e in entry['evidence']),'manifest_source_and_copy_hashes':all(sha(local(m['source']))==m['sha256']==sha(out/m['relative_copy']) for m in manifest['files']),'manifest_missing_empty':not manifest['missing_sources'],'old_registry_objects_unchanged':after['entries'][:-1]==head['entries'],'protected10_unchanged':len(read(out/'protected_controls.json'))==10 and all(sha(scene/p)==h for p,h in read(out/'protected_controls.json').items()),'baseline_phase_consistency':all(v['within_1_second'] for c in cases.values() for v in c['baseline_phase_check'].values()),'both_cases_available':all(c['new']['status']=='available' for c in cases.values()),'byte_preserving_attribute':line in attrs.read_bytes().splitlines()}
validation={'ok':all(checks.values()),'checks':checks,'new_registry_id':entry['id'],'new_entry_evidence_count':len(entry['evidence']),'manifest_copy_count':len(manifest['files']),'acceptance_capture_derivation_corrections':corrected,'validation_scope':'New entry paths/hashes, original source-to-copy manifests, protected controls, and old registry semantic equality to Git HEAD only; no third full historical evidence rescan.'}
(out/'archive_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert validation['ok']
diff=subprocess.run(['git','diff','--check','--','.gitattributes','First_Phase/scene_demo/experiment_registry.json','First_Phase/scene_demo/EXPERIMENT_INDEX.md'],cwd=root,capture_output=True)
assert diff.returncode==0,diff.stdout.decode()+diff.stderr.decode()
print(json.dumps(validation,ensure_ascii=False))
