#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""side_grasp_pilot.py - fixed five-case real-Hermes side-grasp assisted pilot.

Software only: import and --help load no model, build no environment, open no
socket and never call Hermes.  A real run happens only via an explicit
``python3 side_grasp_pilot.py --output-dir DIR --calibration FILE --hermes-home DIR``
on the GPU host, limited to the five preregistered goal_table cases below.  The
injected ``side_grasp`` module replaces ``local_grasp`` for the assisted wine
capability.  Reused siblings (grasp_assist_service, placement_experiments,
paired_config_experiments, guard_validation, planning_diagnostics, run_agent,
service, side_grasp, catalog) are imported unchanged; the literal gold, the shared
500-action budget (including the helper action cap), the seeds and the zero-retry
cancellation policy are preregistered BEFORE any model/env/Hermes work, never
derived from a Hermes plan.
"""

from __future__ import annotations

import argparse, hashlib, json, math, os, subprocess, sys, threading, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import catalog, grasp_assist_service as gas  # noqa: E402
import guard_validation as gv, run_agent, service, side_grasp  # noqa: E402
import paired_config_experiments as paired, placement_experiments as pe, planning_diagnostics as pd  # noqa: E402

EXPERIMENT_NAME = CONDITION = "side_grasp_pilot"
SCENE_ID = "goal_table"
SEED = INIT_STATE_INDEX = 0
MODEL_SEEDS = (0, 1, 2, 3, 4)
PROFILE = "baseline_bf16"
HOST, PORT = "127.0.0.1", 8779
ENDPOINT = "http://127.0.0.1:8779"
READY_TIMEOUT_S, CASE_TIMEOUT_S, MIN_FREE_MIB = 180.0, 900.0, 2500
BUDGET_PER_SUBGOAL = 500
HELPER_ACTION_CAP = 200
GOLD = [["on", "akita_black_bowl_1", "plate_1"], ["on", "wine_bottle_1", "wine_rack_1_top_region"]]
GOLD_KEYS = tuple(catalog.goal_key(goal) for goal in GOLD)
PROTECTED_GOALS = [["not", "turnon", "flat_stove_1"]]
STOVE_KEY = catalog.goal_key(PROTECTED_GOALS[0])
CHEESE_OBJECT, CHEESE_TOLERANCE_M = "cream_cheese_1", 0.005
STRICT_STREAK = pe.STRICT_STREAK
REPORT_NAME, PREREG_NAME = "report.json", "preregistration.json"
PREREG_SOURCE_FILES = ("local_grasp.py", "side_grasp.py", "side_grasp_calibration.py")
REFERENCE_SHA256 = "3b546bcf49a3b01f31799212ff509a30c8a73c6836373680d07584f2c4800c10"
REFERENCE_STEP = 93
# Explicit request-order indices whose actual job order must be verified.
ORDER_EXPECTATIONS = {1: ("bowl_to_plate", "wine_to_rack"), 3: ("wine_to_rack", "bowl_to_plate")}
# The five literal requests, ordered 0..4 (model seeds 0..4); the independent
# gold is identical for all five and is never derived from the plan Hermes submits.
CASES = (
    "\u8bf7\u5e2e\u6211\u6574\u7406\u684c\u9762\u4e0a\u7684\u7897\u548c\u9152\u74f6\u3002",
    "\u628a\u7897\u653e\u5230\u76d8\u5b50\u4e0a\uff0c\u7136\u540e\u628a\u9152\u74f6\u653e\u5230\u67b6\u5b50\u4e0a\u3002",
    "\u8bf7\u628a\u9152\u74f6\u548c\u7897\u90fd\u6536\u56de\u5404\u81ea\u7684\u4f4d\u7f6e\u3002",
    "\u5148\u628a\u9152\u74f6\u653e\u5230\u67b6\u5b50\u4e0a\uff0c\u518d\u628a\u7897\u653e\u5230\u76d8\u5b50\u4e0a\u3002",
    "\u6574\u7406\u8fd9\u4e24\u4ef6\u4e1c\u897f\uff1a\u7897\u653e\u76d8\u5b50\uff0c\u9152\u74f6\u653e\u67b6\u5b50\u3002",
)


def _utc(): return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path):
    try: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception: return None


def validate_calibration(calibration_path, source_dir=None):
    """Validate the recorded calibration BEFORE any model/env/Hermes work.

    Fail closed: actual ``data['ok'] is True`` AND
    ``data['calibration']['confirmed'] is True``.  The same-directory
    preregistration.json raw bytes must hash to ``data['preregistration_sha256']``;
    ``prereg['experiment']`` must equal ``side_grasp_calibration``; the
    ``prereg['source_sha256']`` mapping MUST carry entries for local_grasp.py,
    side_grasp.py and side_grasp_calibration.py, each matching the CURRENT raw
    source bytes; the fixed reference identity (source_sha256 and integer step 93)
    is also checked.  ``side_grasp.reference_metadata()`` supplies path/
    source_sha256/step/relative pose and its reference identity is compared.  A
    missing or unknown value is rejected.
    """
    path = Path(calibration_path)
    if not path.is_file(): return None, "calibration file missing: %s" % calibration_path
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc: return None, "calibration unreadable: %s" % exc
    if not isinstance(data, dict) or data.get("ok") is not True: return None, "calibration ok is not True"
    calibration = data.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("confirmed") is not True:
        return None, "calibration.calibration.confirmed is not True"
    prereg_path = path.parent / "preregistration.json"
    if not prereg_path.is_file(): return None, "preregistration missing: %s" % prereg_path
    try: prereg_bytes = prereg_path.read_bytes()
    except Exception as exc: return None, "preregistration unreadable: %s" % exc
    recorded_sha = data.get("preregistration_sha256")
    current_sha = hashlib.sha256(prereg_bytes).hexdigest()
    if not isinstance(recorded_sha, str) or recorded_sha != current_sha:
        return None, "preregistration sha mismatch (recorded=%r current=%r)" % (recorded_sha, current_sha)
    try: prereg = json.loads(prereg_bytes.decode("utf-8"))
    except Exception as exc: return None, "preregistration JSON unreadable: %s" % exc
    if not isinstance(prereg, dict): return None, "preregistration is not a mapping"
    if prereg.get("experiment") != "side_grasp_calibration":
        return None, "preregistration experiment is not side_grasp_calibration"
    source_map = prereg.get("source_sha256")
    if not isinstance(source_map, dict): return None, "preregistration source_sha256 missing"
    source_dir = Path(source_dir) if source_dir is not None else _HERE
    for name in PREREG_SOURCE_FILES:
        recorded = source_map.get(name)
        if not isinstance(recorded, str) or not recorded:
            return None, "source_sha256[%s] missing/unknown" % name
        current = _sha256_file(source_dir / name)
        if not isinstance(current, str) or recorded != current:
            return None, "%s SHA mismatch (recorded=%r current=%r)" % (name, recorded, current)
    ref = prereg.get("reference")
    if not isinstance(ref, dict): return None, "preregistration reference missing"
    if ref.get("source_sha256") != REFERENCE_SHA256:
        return None, "reference source_sha256 mismatch"
    if type(ref.get("step")) is not int or ref.get("step") != REFERENCE_STEP:
        return None, "reference step is not integer 93"
    try: metadata = side_grasp.reference_metadata() or {}
    except Exception as exc: return None, "side_grasp.reference_metadata failed: %s" % exc
    if metadata.get("source_sha256") != REFERENCE_SHA256:
        return None, "module reference source_sha256 mismatch"
    if type(metadata.get("step")) is not int or metadata.get("step") != REFERENCE_STEP:
        return None, "module reference step is not integer 93"
    for name in PREREG_SOURCE_FILES:
        current = _sha256_file(source_dir / name)
        if current is not None and source_map.get(name) != current:
            return None, "%s no longer matches preregistration" % name
    return data, None


def gpu_free_mib():
    """Read the free GPU memory (MiB) before any model is loaded."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20)
    except Exception as exc: return None, "nvidia-smi unavailable: %s" % exc
    if done.returncode != 0: return None, "nvidia-smi failed: %s" % (done.stderr or "").strip()
    values = []
    for line in (done.stdout or "").splitlines():
        try: values.append(int(float(line.strip().split()[0])))
        except Exception: continue
    return (max(values), None) if values else (None, "no GPU memory reading")


def prepare_hermes_home(hermes_home):
    """Copy the real profile to a NEW home (outside output) and point it at 8779."""
    config_path, _config = pd.prepare_home(hermes_home)
    import yaml  # lazy: never needed for import/--help
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    model = cfg.get("model") or {}
    assert model.get("default") == pd.MODEL_NAME and model.get("supports_vision") is True
    tools = ((cfg.get("mcp_servers") or {}).get("scene_tools")) or {}
    tools["env"] = dict(tools.get("env") or {}); tools["env"]["SCENE_SERVICE_URL"] = ENDPOINT
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    assert tools["env"]["SCENE_SERVICE_URL"] == ENDPOINT
    return config_path


def _position(snapshot, object_id):
    entry = (snapshot.get("objects") or {}).get(object_id) if isinstance(snapshot, dict) else None
    value = entry.get("position") if isinstance(entry, dict) else None
    if not isinstance(value, list) or len(value) < 3: return None
    try: return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError): return None


def score_gold(rows, final_snapshot, before_snapshot, stove_off):
    """Pure, independent literal-gold scorer (never derived from a Hermes plan)."""
    sequence = [row.get("strict_candidate") if isinstance(row, dict) else None for row in (rows or [])]
    window = sequence[-STRICT_STREAK:]
    final_five = len(window) == STRICT_STREAK and all(value is True for value in window)
    final = final_snapshot if isinstance(final_snapshot, dict) else {}
    predicates = final.get("predicates") if isinstance(final.get("predicates"), dict) else {}
    gold_predicates = {key: predicates.get(key) for key in GOLD_KEYS}
    held = final.get("held_objects")
    before_position = _position(before_snapshot, CHEESE_OBJECT)
    after_position = _position(final_snapshot, CHEESE_OBJECT)
    displacement = None
    if before_position is not None and after_position is not None:
        displacement = math.sqrt(sum((a - b) ** 2 for a, b in zip(before_position, after_position)))
    cheese_ok = displacement is not None and displacement <= CHEESE_TOLERANCE_M
    known = {"gold_predicates": all(gold_predicates.get(k) is not None for k in GOLD_KEYS),
             "held_objects": isinstance(held, list), "stove_off": stove_off is not None,
             "cheese_displacement_m": displacement is not None}
    combined = bool(final_five and all(gold_predicates.get(k) is True for k in GOLD_KEYS)
                    and isinstance(held, list) and not held and stove_off is True and cheese_ok)
    return {"gold_completed": bool(final_five), "combined_success": combined, "n_actual_samples": len(sequence),
            "final_predicates": gold_predicates, "held_objects": held, "stove_off": stove_off,
            "cheese_displacement_m": displacement, "cheese_within_tolerance": bool(cheese_ok),
            "unknown": [k for k, ok in known.items() if not ok]}


def score_with_order(score, order_ok):
    """Copy a gold score, keeping placement success separate from request order."""
    fixed = dict(score or {})
    placement = fixed.get("combined_success")
    fixed["placement_success"] = placement
    fixed["combined_success"] = bool(placement) and bool(order_ok)
    return fixed


def read_oracle_rows(jobs):
    """Concatenate the raw per-action oracle_snapshot rows in actual job order."""
    rows, paths = [], []
    for job in jobs or []:
        run_dir = job.get("run_dir") if isinstance(job, dict) else None
        if not run_dir: continue
        path = Path(run_dir) / "telemetry.jsonl"; paths.append(str(path))
        if not path.is_file(): continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try: snapshot = json.loads(line).get("oracle_snapshot")
                    except Exception: continue
                    if isinstance(snapshot, dict): rows.append(snapshot)
        except Exception: continue
    return rows, paths


_ASSIST_FIELDS = ("job_id", "capability_id", "condition", "profile", "completion_mode", "grasp_guard_mode",
                  "helper", "assisted", "confirmation_is_not_placement_success", "requested_budget",
                  "effective_budget", "actual_vla_actions", "actual_local_actions", "steps", "phase",
                  "reason", "helper_summary")


def _assist_summary(run_dir):
    """Controller phase / confirmation / VLA+local counts of one job (or None)."""
    path = (Path(run_dir) / gas.ASSIST_REPORT_FILENAME) if run_dir else None
    if path is None or not path.is_file(): return None
    try: report = json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None
    return {k: report.get(k) for k in _ASSIST_FIELDS} if isinstance(report, dict) else None


def _actual_order(jobs):
    """The actual executed capability sequence of the reported jobs (never planned)."""
    order = []
    for job in jobs or []:
        if not isinstance(job, dict): continue
        capability = job.get("capability_id")
        if isinstance(capability, str): order.append(capability)
    return order


def classify_case(result, plan, jobs):
    """Operational errors abort the campaign; ordinary physical failures continue.

    Known physical failure reasons are classified BEFORE the generic
    job.state/error test, so an expected assist physical error is never counted
    as a backend crash.
    """
    ops, phys = [], []
    if result.get("execution_timeout"): ops.append("execution_timeout")
    if result.get("cancellation_pending") or result.get("cancelled_by_user"): ops.append("cancelled")
    invocations = result.get("hermes_invocations")
    if not isinstance(invocations, int) or invocations < 1: ops.append("no_hermes_invocation")
    elif result.get("run_ok") is False: ops.append("hermes_nonzero_exit")
    known_physical = ("budget_exhausted", "failed_grasp", "local_grasp_failed",
                      "local_grasp_unknown", "target_occupied", "session_limit")
    job_ops, physical_job = [], False
    for job in jobs or []:
        if not isinstance(job, dict): continue
        ended = job.get("ended_reason")
        if ended in known_physical:
            phys.append("%s: %s" % (ended, job.get("job_id")))
            physical_job = True
            continue
        if job.get("state") == "error" or ended == "error" or job.get("error"):
            job_ops.append("job_error: %s" % (job.get("job_id"),))
    ops.extend(job_ops)
    state = plan.get("state") if isinstance(plan, dict) else None
    if state == "error":
        if physical_job and not job_ops:
            phys.append("plan_error_after_physical: %s" % (plan.get("error"),))
        else:
            ops.append("plan_error: %s" % (plan.get("error"),))
    elif state == "cancelled": ops.append("plan_cancelled")
    elif state == "blocked": phys.append("plan_blocked")
    if not isinstance(plan, dict) or not plan: phys.append("no_plan")
    return ops, phys


def _request_order(jobs, expectation):
    """Whether the ACTUAL job sequence matches the explicit request order index."""
    if expectation is None: return True
    order = _actual_order(jobs)
    return order == list(expectation)


def run_case(svc, hermes_home, output_dir, index):
    """One fresh session / one reset / one seeded model / one real Hermes case."""
    case_dir = output_dir / ("case_%02d" % index); case_dir.mkdir(parents=True, exist_ok=True)
    entry = {"case_index": index, "model_seed": MODEL_SEEDS[index], "request_text": CASES[index],
             "session_id": None, "request_id": None, "seeded": None, "plan": None, "jobs": [],
             "telemetry_paths": [], "agent": None, "score": None, "actual_order": [],
             "expected_order": list(ORDER_EXPECTATIONS.get(index) or ()), "order_ok": True,
             "operational_errors": [], "physical_failures": []}
    session = svc.create_session(SCENE_ID, seed=SEED, init_state_index=INIT_STATE_INDEX)
    if not session.get("ok"):
        entry["operational_errors"].append("create_session: %s" % (session,)); return entry
    entry["session_id"] = session["session_id"]
    seed_result = paired._seed_model_rng(svc, MODEL_SEEDS[index])
    entry["seeded"] = seed_result
    if not (isinstance(seed_result, dict) and seed_result.get("ok") is True
            and seed_result.get("seeded") is True):
        entry["operational_errors"].append("seed_model_rng: %s" % (seed_result,))
        return entry
    before = svc.final_snapshot(GOLD)
    request_id = uuid.uuid4().hex; entry["request_id"] = request_id
    config = SimpleNamespace(session_id=session["session_id"], request=CASES[index], request_id=request_id,
                             case_id=None, max_repairs=0, timeout=CASE_TIMEOUT_S, cancel_file=None)
    runner = run_agent.Runner(config, run_agent.ServiceClient(),
                              run_agent.HermesRunner(home=hermes_home, cwd=str(_HERE)),
                              run_agent.SystemClock(), str(case_dir))
    result = runner.run()
    pe._write_json_atomic(case_dir / "agent_result.json", result)
    plan = svc.plan(request_id) or {}
    jobs = [svc.job(job_id) for job_id in (plan.get("job_ids") or [])]
    rows, paths = read_oracle_rows(jobs); entry["telemetry_paths"] = paths
    after = svc.final_snapshot(GOLD); protected = svc.final_snapshot(PROTECTED_GOALS)
    stove_off = ((protected.get("snapshot") or {}).get("predicates") or {}).get(STOVE_KEY) if protected.get("ok") else None
    gold_score = score_gold(rows, after.get("snapshot") if after.get("ok") else None,
                            before.get("snapshot") if before.get("ok") else None, stove_off)
    entry["score"] = gold_score
    entry["agent"] = {"request_id": result.get("request_id"), "run_ok": result.get("run_ok"),
                      "chain_ok": result.get("chain_ok"), "plan_success": result.get("plan_success"),
                      "decision": result.get("decision"), "hermes_invocations": result.get("hermes_invocations"),
                      "execution_timeout": result.get("execution_timeout"),
                      "cancelled_by_user": result.get("cancelled_by_user"),
                      "cancellation_pending": result.get("cancellation_pending"), "wall_s": result.get("wall_s"),
                      "error": result.get("error"),
                      "plan_state": (result.get("plan") or {}).get("state") if isinstance(result.get("plan"), dict) else None,
                      "agent_result_path": str(case_dir / "agent_result.json"),
                      "hermes_initial_log": str(case_dir / "hermes_initial.log")}
    entry["jobs"] = [{"job_id": j.get("job_id"), "capability_id": j.get("capability_id"), "state": j.get("state"),
                      "ended_reason": j.get("ended_reason"), "error": j.get("error"), "success": j.get("success"),
                      "steps": j.get("steps"), "phase": j.get("phase"), "held_objects": j.get("held_objects"),
                      "run_dir": j.get("run_dir"), "grasp_assist": _assist_summary(j.get("run_dir"))}
                     for j in jobs if isinstance(j, dict)]
    entry["plan"] = {"request_id": plan.get("request_id"), "state": plan.get("state"),
                     "plan_success": plan.get("plan_success"),
                     "completed_capability_ids": plan.get("completed_capability_ids"),
                     "pending_capability_ids": plan.get("pending_capability_ids"), "job_ids": plan.get("job_ids")}
    entry["actual_order"] = _actual_order(entry["jobs"])
    entry["order_ok"] = _request_order(entry["jobs"], ORDER_EXPECTATIONS.get(index))
    entry["score"] = score_with_order(gold_score, entry["order_ok"])
    entry["operational_errors"], entry["physical_failures"] = classify_case(result, plan, jobs)
    return entry


def run_pilot(args):
    """The only real runner: five fresh assisted Hermes cases, then shut down own server."""
    gas.configure_process_environment()
    output_dir = Path(args.output_dir); report_path = output_dir / REPORT_NAME
    report = {"experiment": EXPERIMENT_NAME, "condition": CONDITION, "generated_utc": _utc(), "status": "starting",
              "planned_cases": len(CASES), "executed_cases": 0, "fatal_error": None,
              "preregistration_sha256": None, "calibration_sha256": _sha256_file(args.calibration),
              "stage_counts": {"gold_completed": 0, "combined_success": 0, "operational": 0, "physical": 0},
              "cases": []}
    had_oracle, saved_oracle = CONDITION in pe.FINAL_ORACLE_GOALS, pe.FINAL_ORACLE_GOALS.get(CONDITION)
    saved_base, svc, httpd = run_agent.SERVICE_BASE, None, None

    def _fail(stage, message):
        report["status"], report["fatal_error"] = stage, message
        try: output_dir.mkdir(parents=True, exist_ok=True)
        except Exception: pass
        pe._write_json_atomic(report_path, report)
        return 1

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        calibration, error = validate_calibration(args.calibration, _HERE)
        if error: return _fail("failed_calibration", "calibration: %s" % error)
        audit = paired._audit_checkpoint_files(service.DEFAULT_MODEL_PATH)
        if not audit.get("ok"): return _fail("failed_checkpoint", "checkpoint audit: %s" % audit.get("reason"))
        pe.FINAL_ORACLE_GOALS[CONDITION] = [list(goal) for goal in GOLD]
        prereg = {"experiment": EXPERIMENT_NAME, "condition": CONDITION, "created_utc": _utc(), "scene_id": SCENE_ID,
                  "seed": SEED, "init_state_index": INIT_STATE_INDEX, "model_seeds": list(MODEL_SEEDS),
                  "requests": list(CASES), "profile": PROFILE, "oracle_goals": [list(g) for g in GOLD],
                  "protected_goals": [list(g) for g in PROTECTED_GOALS], "protected_object": CHEESE_OBJECT,
                  "protected_tolerance_m": CHEESE_TOLERANCE_M, "budget_per_subgoal": BUDGET_PER_SUBGOAL,
                  "helper_action_cap": HELPER_ACTION_CAP, "cancellation": True, "retries": 0, "assisted": True,
                  "order_expectations": {"1": ["bowl_to_plate", "wine_to_rack"],
                                        "3": ["wine_to_rack", "bowl_to_plate"]},
                  "model_path": service.DEFAULT_MODEL_PATH, "checkpoint_audit": audit,
                  "checkpoint_revision": service.MODEL_REVISION_DEFAULT,
                  "capture_images": False, "runs_per_case": 1, "max_repairs": 0,
                  "timeout_s": CASE_TIMEOUT_S, "ready_timeout_s": READY_TIMEOUT_S,
                  "min_free_mib": MIN_FREE_MIB, "host": HOST, "port": PORT, "endpoint": ENDPOINT,
                  "grasp_module": "side_grasp", "controller": "side_grasp.SideGraspController",
                  "calibration_path": str(args.calibration),
                  "source_sha256": {"side_grasp_pilot.py": _sha256_file(_HERE / "side_grasp_pilot.py"),
                                    "local_grasp.py": _sha256_file(_HERE / "local_grasp.py"),
                                    "side_grasp.py": _sha256_file(_HERE / "side_grasp.py"),
                                    "side_grasp_calibration.py": _sha256_file(_HERE / "side_grasp_calibration.py"),
                                    "grasp_assist_service.py": _sha256_file(_HERE / "grasp_assist_service.py")},
                  "calibration_sha256": _sha256_file(args.calibration)}
        pe._write_json_atomic(output_dir / PREREG_NAME, prereg)
        report["preregistration_sha256"] = _sha256_file(output_dir / PREREG_NAME)
        report["sessions_root"] = str(output_dir / "sessions")
        report["hermes_home_config_sha256"] = _sha256_file(prepare_hermes_home(args.hermes_home))
        run_agent.SERVICE_BASE = ENDPOINT
        free_mib, gpu_error = gpu_free_mib(); report["gpu_free_mib"] = free_mib
        if free_mib is None or free_mib < MIN_FREE_MIB:
            return _fail("failed_gpu_precondition", "gpu_precondition: %s (free=%r MiB)" % (gpu_error, free_mib))
        try: httpd = service._Server((HOST, PORT), service._Handler)
        except OSError as exc: return _fail("failed_port", "port %d unavailable: %s" % (PORT, exc))
        svc = gas.GraspAssistService(model_path=service.DEFAULT_MODEL_PATH,
                                     run_root=str(output_dir / "sessions"),
                                     grasp_module=side_grasp)
        svc._diag_condition = CONDITION
        httpd.service = svc
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True).start()
        svc.start()
        deadline, ready = time.monotonic() + READY_TIMEOUT_S, False
        while time.monotonic() < deadline:
            health = svc.health()
            if health.get("worker_error"): break
            if health.get("ready"): ready = True; break
            time.sleep(0.5)
        if not ready: return _fail("failed_ready", "service not ready within %.0fs" % READY_TIMEOUT_S)
        profile_result = svc.configure_profile(PROFILE)
        report["profile"], report["profile_ok"] = profile_result, gv._profile_readback_ok(profile_result)
        if not report["profile_ok"]: return _fail("failed_profile", "profile readback: %s" % (profile_result,))
        report["status"] = "running"
        for index in range(len(CASES)):
            print("START case=%d seed=%d" % (index, MODEL_SEEDS[index]), flush=True)
            entry = run_case(svc, args.hermes_home, output_dir, index)
            report["cases"].append(entry); report["executed_cases"] = index + 1
            score = entry.get("score") or {}; counts = report["stage_counts"]
            counts["gold_completed"] += 1 if score.get("gold_completed") is True else 0
            counts["combined_success"] += 1 if score.get("combined_success") is True else 0
            counts["operational"] += len(entry["operational_errors"]); counts["physical"] += len(entry["physical_failures"])
            if entry["operational_errors"]:
                report["fatal_error"] = "case %d operational: %s" % (index, "; ".join(entry["operational_errors"]))
            pe._write_json_atomic(report_path, report)
            print("END case=%d gold=%s combined=%s ops=%d phys=%d" % (index, score.get("gold_completed"),
                  score.get("combined_success"), len(entry["operational_errors"]),
                  len(entry["physical_failures"])), flush=True)
            if report["fatal_error"]: break
        report["completed"] = report["executed_cases"] == len(CASES) and not report["fatal_error"]
        report["status"] = "completed" if report["completed"] else "stopped"
        pe._write_json_atomic(report_path, report)
        return 0 if report["completed"] else 1
    finally:
        run_agent.SERVICE_BASE = saved_base
        if had_oracle: pe.FINAL_ORACLE_GOALS[CONDITION] = saved_oracle
        else: pe.FINAL_ORACLE_GOALS.pop(CONDITION, None)
        if httpd is not None:
            try: httpd.shutdown(); httpd.server_close()
            except Exception: pass
        if svc is not None:
            try: svc._sync_work("close_env", lambda: (svc._close_env() or {"ok": True}))
            except Exception: pass
            try: svc.stop()
            except Exception: pass


def _build_parser():
    parser = argparse.ArgumentParser(description="Fixed five-case real-Hermes side-grasp assisted combined-task pilot (no run on --help).")
    parser.add_argument("--output-dir", required=True, help="absolute output dir (must not exist)")
    parser.add_argument("--calibration", required=True, help="absolute calibration JSON path")
    parser.add_argument("--hermes-home", required=True, help="absolute NEW hermes home (must not exist)")
    return parser


def main(argv=None):
    parser = _build_parser(); args = parser.parse_args(argv)
    for name in ("output_dir", "calibration", "hermes_home"):
        if not os.path.isabs(getattr(args, name)):
            parser.error("--%s must be an absolute path" % name.replace("_", "-"))
    if Path(args.output_dir).exists(): parser.error("--output-dir already exists; refusing to reuse %s" % args.output_dir)
    if not Path(args.calibration).is_file(): parser.error("--calibration must be an existing file")
    if Path(args.hermes_home).exists(): parser.error("--hermes-home already exists; refusing to reuse %s" % args.hermes_home)
    return run_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
