# X-VLA Native LIBERO Benchmark Implementation Plan

> **For agentic workers:** Follow this fixed plan and the six-section DeepSeek task. GPT 6 Astra owns diagnosis and acceptance; actual DeepSeek authors implementation; local tools apply/run/archive. Do not infer model identity from a worker label.

**Goal:** Measure the original author's X-VLA-LIBERO checkpoint on 40 native tasks, five episodes per task (200 total), with author preprocessing, absolute control and stopping semantics.

**Architecture:** Reuse the pinned upstream model HTTP server and inherit the pinned upstream ClientModel and LIBEROEval. Add an isolated audit/resume wrapper around their actual calls; do not change model inputs, action conversion, action queue, physics, scoring or existing scene-demo services. Separate model Python 3.12/CUDA and simulation Python 3.10/CPU environments are already installed.

**Tech Stack:** Original X-VLA FP32, torch 2.11.0+cu128 model server; original LIBERO 8f1084, robosuite 1.4.0, MuJoCo 2.3.7, torch 1.11.0+cpu simulation client.

**Spec:** `20261012-xvla-native-benchmark.json` plus `../../../tmp/xvla_20261012_deepseek_task.md` (exact authoring interface).

## Global Constraints

- User approved 40 tasks x 5 episodes; original CLI defaults to 50 episodes/task. Report reduced-sample reproduction, not exact reproduction or a paired SmolVLA comparison.
- Use `2toINF/X-VLA-Libero` revision `129e71460678b7236cee6fc9707f09d9fa0c3590`, weight SHA256 `260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8`.
- X-VLA source `6bc2513f5f1cbec715cc668b414392a6cae5c671`; LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Preserve FP32, 30 actions/chunk, 10 denoising steps, domain_id 3, two original 256px views, author's state/rotation conventions, native language, 10 settling actions, 800/900 policy-action horizons.
- Preserve author CLI suite order: libero_10, libero_spatial, libero_goal, libero_object; task ids 0..9, episode ids 0..4; environment seed 42 + episode + 100; stored initial-state index episode.
- Native success is the author's `done`, independently cross-checked with `env.check_success()` after each policy action. No strict release, visual checker, grasp helper, home or Hermes.
- First run completes only first episode; inspect nonzero real model calls, nonzero actual absolute-control actions, unchanged input/action contracts, scoring consistency and frames before allowing remaining episodes. A correctly recorded task failure does not invalidate a benchmark; infrastructure/contract/zero-action failures do.
- Do not retry or silently change a completed failure. Incomplete/exception episodes are unknown and stop the campaign. Resume only completed valid episodes with matching configuration and preserve incomplete attempts separately.
- User-authorized local setup changes do not authorize git commits/pushes. Existing modified files and historical evidence stay intact.

## Review Focus

- Absolute z/rotation targets must not be clipped as relative actions.
- Author's agentview flip, untouched wrist image, and chunk-end proprio update must remain byte-for-byte upstream behavior.
- HTTP 200 must contain finite (30,20) actions; errors or malformed responses cannot count as task failure.
- `done` versus independent native predicate disagreement is an audit failure, not success.
- STOP/interruption cannot invent a valid episode or overwrite its partial evidence.

## Task 1: Audited native client

Files: create `First_Phase/libero_demo/xvla_native_benchmark.py` and `test_xvla_native_benchmark.py` only. Exact CLI and functions are fixed in the six-section task. DeepSeek emits a patch; worker applies tests first, records the absent-implementation failure, then implementation and tests. Reuse real upstream classes in interface tests; mock only network and physical env stepping.

## Task 2: Provenance and first native episode

Local worker freezes source/configuration/install and authoring evidence in `scene_demo/results/2026-10-12-xvla-native-benchmark`, runs `review.py check` and necessary software verification. Start original upstream server at localhost:8794 in FP32. Run wrapper `--max-new-episodes 1`; inspect actual recorded inputs, predicted chunk, sent actions, state changes, robot/controller identity, native predicate and video. Preserve all startup failures. No synthetic inference is needed before the first real observation request.

## Task 3: Remaining benchmark and reporting

After first-episode acceptance and a fresh gate, run the same configuration for remaining valid slots, with STOP support and progress persisted after each episode and periodically within an episode. Reuse the first valid episode in the 200 denominator. Report per-task and per-suite counts, valid/unknown/incomplete denominators, policy versus settling actions, latency, videos, hardware/software differences and source hashes. Update experiment index/registry after results. A background launch is not completion; inspect final results.

## Known reproduction differences

Python 3.10.22 client versus author's Python 3.8 example; model Python 3.12.3/torch 2.11.0+cu128 for Blackwell versus older author environment examples; CPU torch 1.11 for client; MuJoCo 2.3.7 is an explicit local selection, not a verified paper lock. Author server does not fix a model RNG seed; retain that behavior and record run identities. Current author HF conversion also documents a small performance drop relative to original reported models. No conclusion of exact 98.1% reproduction is pre-assumed.
