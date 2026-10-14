# Simulator observation I/O comparison — 2026-10-15

Two flows were run once each; four actual native skills passed. Single execution changed from 152.773399792s to 74.420930427s (51.3% reduction); the three-skill chain changed from 392.139429141s to 181.085704868s (53.8%). These are measured executions with unpaired observations and RNG, not an exact causal estimate or overall reliability claim.

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
| Load scene | 8.07 → 8.07 | 8.06 → 8.06 |
| Prepare and plan | 48.68 → 72.87 | 92.55 → 80.59 |
| Execution | 152.77 → 74.42 | 392.14 → 181.09 |
| Total | 209.53 → 155.37 | 492.75 → 269.73 |

Single total is 209.531 → 155.365s. Only the new single run includes a 16.423s cold model/build proxy: the loading message explicitly naming X-VLA at elapsed 12.113s to the next planning stage at 28.536s. The baseline was warm; the new chain has no X-VLA cold-loading marker. Full prepare-and-plan time is retained. Qwen planning total was 41.626 → 50.674s for single and 83.401 → 71.046s for chain; those changes are not attributed to I/O optimization.

| Original-file mtime phase proxy, seconds | Single old → new | Chain old → new |
|---|---:|---:|
| Policy | 61.84 → 26.60 | 108.04 → 44.65 |
| Handoff / release | 0.81 → 0.42 | 14.08 → 5.86 |
| Retreat plan / motion / confirm | 47.17 → 29.32 | 105.41 → 64.85 |
| Home entry / home | 37.92 → 14.36 | 150.70 → 56.67 |
| Finalization | 1.96 → 1.07 | 4.64 → 2.32 |

Boundaries use ORIGINAL file mtimes, not archived copies: A=first.png, B=finish/simulator_samples.jsonl, C=finish.plan.geometry_frames[0].rgb_path, D=finish.home_entry.clearance_screen.frames[0].rgb_path, E=finish_actions.jsonl and F=pilot_result.json. Policy=B−A; handoff/release=C−B; retreat=D−C; home=E−D; finalization=F−E. Capture, rendering, geometry planning, motion and writes overlap: these phases are neither pure physics nor pure planning. Raw paths and nanosecond boundaries are retained in [comparison](comparison.json). Other execution time was 3.073 → 2.661s single and 9.270 → 6.743s chain.

## Capture storage and comparison limits

Finish sensor capture groups were 107 → 14 single and 218 → 34 chain. Decimal storage was 147.901650 → 12.292050 MB single and 305.747649 → 31.101940 MB chain, including controller_modes.jsonl. PNG counts were 535 → 44 and 1090 → 108; NPY counts were 214 → 28 and 436 → 68. A group is a primary cap*_agentview.png excluding checker stems: agent/wrist/side prefixes differ in the original SensorPort. Acceptance derivation was corrected from unique numeric cap prefixes to that primary-view definition; actual experimental outcomes were not changed.

Skipped periodic capture counters total 90 single and 183 chain. Observation-cache hits were 503 single and 1057 at the final chain job; chain counters are cumulative across the same environment and must not be summed across jobs. Raw action totals changed 478 → 465 single and 941 → 935 chain. Native VLA HTTP proxies are response-file mtime minus request-file mtime and include inference plus HTTP overhead, not pure GPU time; full counts and sums are in comparison.json.

Observations, first images and VLA request payloads differ across old/new runs. The cache reuses native step-returned observations whereas the prior path force-refreshed them; installed robosuite.step updates observables before returning observations under periodic sensor sampling. Same seed and preserved policy source hashes do not prove byte-equivalent observations, inputs, action sequences or paired RNG. Explicit first-image SHA, uncompressed request-payload SHA and actual action sequence comparisons are retained. The result demonstrates these two flows and four skills, not broad reliability or a pure causal speed measurement.

## Evidence and checks

[Evidence manifest](evidence_manifest.json) contains 82 exact-byte derived copies and no missing sources. Original records are preserved. Both pre-run history checks passed while retaining scientific_review_required=true and launch_authorized=false; launch authorization came from the user scope, independently of those checks. Initial software validation retained a deque assertion failure (1 failed, 75 passed); the corrected check then passed all 76 tests. See [software output](software_tests.txt), [protected controls](protected_controls.json), [bowl review](bowl_review.json) and [chain review](chain_review.json).
