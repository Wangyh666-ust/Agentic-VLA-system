#!/usr/bin/env python3
"""Isolated local-grasp physical calibration (ONE run, no model, no Hermes).

One assisted calibration on the shared ``goal_table`` scene, after the preserved
102-action recorded bowl prefix:

1. replay at most ``A0_APPROACH_CAP`` (150) historical A0 wine approach actions
   (the ACTUAL ``sent_action`` arrays of the fixed preserved wine telemetry)
   through the worker-owned ``env.step``, sending a historical action only while
   ``local_grasp.LocalGraspController`` returns no action AND stays idle;
2. the FIRST auxiliary action it returns stops the historical replay (the
   historical closing action is NEVER sent) and starts the LOCAL execution:
   every local ``env.step`` is followed by ``observe_after`` until the controller
   is confirmed / failed or ``LOCAL_ACTION_CAP`` (200) local actions are reached.

PASS requires a confirmed controller, a true native bowl predicate, an intact
protected-object displacement (<= the fixed 0.005 m gate) and a real >= 0.02 m
wine lift held for 5 consecutive samples; a confirmed grasp is NOT a wine
placement success.  No model / Hermes / assessment / recovery / forced release is
used, no RNG is reseeded, no source is modified, and ``--help`` parses without
creating a worker, an environment or a model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import local_grasp  # noqa: E402
import placement_experiments as pe  # noqa: E402
import preparation_diagnostics as pd  # noqa: E402
import service  # noqa: E402
import skill_context_diagnostics as context  # noqa: E402

EXPERIMENT_NAME = "grasp_calibration"
SCENE_ID = context.CONDITION_SPECS["shared_after_bowl"]["scene_id"]  # "goal_table"
SCENE_SEED = INIT_STATE_INDEX = 0
SESSION_DIRNAME = "session"
# The two fixed preserved read-only inputs (never written, never re-derived).
REPLAY_ACTIONS_PATH = (
    "/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-08-skill-context/inputs/actions.json"
)
WINE_TELEMETRY_PATH = (
    "/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-08-wine-start-pose/paired/raw/"
    "20261008T121844Z_goal_table_cc11bb90/cap_01_wine_to_rack_3aa815e9/wine_telemetry.jsonl.gz"
)
A0_APPROACH_CAP, LOCAL_ACTION_CAP = 150, 200
WORKER_TIMEOUT_S, READY_DEADLINE_S, POLL_INTERVAL_S, VIDEO_FPS = 180.0, 60.0, 0.5, 20
COMPLETION_MODE, GRASP_GUARD_MODE = "release_verified", "enforce"
BOWL_GOAL = list(context.BOWL_GOAL)
WINE_OBJECT_ID = context.wd.WINE_OBJECT_ID
# Every fixed protected object EXCEPT the grasped wine bottle (the wine moves);
# the 0.005 m gate and the position reader are the accepted preparation contracts.
PROTECTED_OBJECTS = tuple(o for o in pd.PROTECTED_OBJECTS if o != WINE_OBJECT_ID)
PROTECTION_TOLERANCE_M = pd.PROTECTION_TOLERANCE_M  # 0.005
CONTROLLER_GRIP_SITE_ERROR_MAX_M = 0.0001
WINE_LIFT_M, WINE_LIFT_STREAK = context.RAW_STABLE_GRASP_LIFT_M, context.RAW_STABLE_GRASP_WINDOW
# The controller phase vocabulary, read from the live interface when it exposes one.
IDLE_P = {str(v).lower() for v in getattr(local_grasp, "IDLE_PHASES", ("idle",))}
CONFIRMED_P = {str(v).lower() for v in getattr(local_grasp, "CONFIRMED_PHASES", ("confirmed",))}
FAIL_P = {str(v).lower() for v in getattr(local_grasp, "FAIL_PHASES", ("bypass", "bypassed", "failed"))}
SOURCE_FILES = ("grasp_calibration.py", "local_grasp.py", "preparation_diagnostics.py",
                "skill_context_diagnostics.py", "placement_experiments.py", "service.py")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _progress(message: str) -> None:
    sys.stderr.write("[grasp-calibration %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _jsonable(value: Any) -> Any:
    """numpy ndarrays via ``.tolist()``; other unsupported objects via ``str`` only."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _source_sha256() -> dict[str, str | None]:
    digests: dict[str, str | None] = {}
    for name in SOURCE_FILES:
        try:
            digests[name] = _sha256_hex((_HERE / name).read_bytes())
        except Exception:  # noqa: BLE001 - absent/unreadable is recorded as unknown
            digests[name] = None
    return digests


def _configure_process_environment() -> None:
    """Strip proxy variables and pin the isolated runtime (this process only)."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "ftp_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["LIBERO_CONFIG_PATH"] = "/home/yhwang/fyp/libero_demo/libero_config"
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib"
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_sent_actions(path: Any) -> list[list[float]]:
    """The ACTUAL sent 7-D ``sent_action`` arrays of the preserved wine telemetry.

    Every nonblank row MUST carry a well-formed 7-D finite ``sent_action``: a
    malformed row raises ``ValueError`` naming its 1-based line number, so the
    recorded action sequence can never be silently shortened.  Blank lines are
    skipped.
    """
    rows: list[list[float]] = []
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:  # noqa: BLE001 - a malformed row is never skipped
                raise ValueError("line %d: invalid JSON (%s)" % (line_number, exc)) from exc
            if not isinstance(record, dict) or "sent_action" not in record:
                raise ValueError("line %d: missing sent_action" % line_number)
            action = record["sent_action"]
            try:
                vector = [float(item) for item in action]
            except (TypeError, ValueError) as exc:
                raise ValueError("line %d: sent_action is not convertible to floats (%s)"
                                 % (line_number, exc)) from exc
            if len(vector) != service.ACTION_DIM:
                raise ValueError("line %d: sent_action has %d values, expected %d"
                                 % (line_number, len(vector), service.ACTION_DIM))
            if not all(math.isfinite(item) for item in vector):
                raise ValueError("line %d: sent_action has nonfinite values" % line_number)
            rows.append(vector)
    return rows


def _helper_phase(helper: Any) -> str | None:
    value = getattr(helper, "phase", None)
    return value if isinstance(value, str) else None


def _helper_status(helper: Any) -> tuple[bool, str | None]:
    """``(confirmed, failure_reason)`` of the live controller (bypass/failed fails)."""
    phase = (_helper_phase(helper) or "").lower()
    reason = getattr(helper, "reason", None) or ""
    if phase in FAIL_P:
        return False, "helper_phase:%s" % phase
    if "bypass" in str(reason).lower():
        return False, "helper_bypass:%s" % reason
    return phase in CONFIRMED_P, None


def _grip_site_error(geometry: Any, facts: Any) -> float | None:
    """The controller<->grip-site error (m): the interface field, else the norm."""
    if not isinstance(geometry, dict):
        return None
    explicit = pd._finite_scalar(geometry.get("controller_grip_site_error_m"))
    if explicit is not None:
        return explicit
    site = pd._finite_vec3(geometry.get("grip_site"))
    controller = geometry.get("controller") if isinstance(geometry.get("controller"), dict) else facts
    ee = pd._finite_vec3(controller.get("ee_pos")) if isinstance(controller, dict) else None
    if site is None or ee is None:
        return None
    distance = math.sqrt(sum((site[i] - ee[i]) ** 2 for i in range(3)))
    return distance if math.isfinite(distance) else None


def _wine_z(snapshot: Any) -> float | None:
    objects = snapshot.get("objects") if isinstance(snapshot, dict) else None
    entry = objects.get(WINE_OBJECT_ID) if isinstance(objects, dict) else None
    position = entry.get("position") if isinstance(entry, dict) else None
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    return pd._finite_scalar(position[2])


def _bowl_true(snapshot: Any) -> bool | None:
    predicates = snapshot.get("predicates") if isinstance(snapshot, dict) else None
    if not isinstance(predicates, dict) or not predicates:
        return None
    if any(value is None for value in predicates.values()):
        return None
    return all(bool(value) for value in predicates.values())


def _protected_violations(baseline: Any, current: Any) -> list[str]:
    """Fail-closed protected-object displacement over the fixed 0.005 m gate."""
    baseline_map = baseline if isinstance(baseline, dict) else {}
    current_map = current if isinstance(current, dict) else {}
    violations: list[str] = []
    for object_id in PROTECTED_OBJECTS:
        base = pd._finite_vec3(baseline_map.get(object_id))
        now = pd._finite_vec3(current_map.get(object_id))
        if base is None or now is None:
            violations.append("protected_object_unknown:%s" % object_id)
            continue
        distance = math.sqrt(sum((base[i] - now[i]) ** 2 for i in range(3)))
        if not math.isfinite(distance) or distance > PROTECTION_TOLERANCE_M:
            violations.append("protected_object_moved:%s:%.4f" % (object_id, distance))
    return violations


def _wait_ready(svc: Any, deadline_s: float = READY_DEADLINE_S) -> dict[str, Any]:
    """Poll ``svc.health()`` (there is NO ``wait_until_ready``) to a bounded deadline."""
    deadline = time.monotonic() + float(deadline_s)
    while time.monotonic() < deadline:
        try:
            health = svc.health()
        except Exception as exc:  # noqa: BLE001 - a raising health is not ready
            health = {"ready": False, "worker_error": _format_exc(exc)}
        if isinstance(health, dict) and health.get("worker_error"):
            return {"ready": False, "worker_error": str(health["worker_error"])}
        if isinstance(health, dict) and health.get("ready"):
            return {"ready": True, "worker_error": None}
        time.sleep(POLL_INTERVAL_S)
    return {"ready": False, "worker_error": None}


# --- the worker-owned calibration --------------------------------------------


def _calibration_work(svc: Any, session_id: Any, historical: Any, output_dir: Any) -> dict[str, Any]:
    """Worker-owned calibration entry: owns ``calibration_steps.jsonl``.

    The per-action trace is opened here (worker-owned) and closed in the
    ``finally`` on EVERY exit -- an early return or an exception both close it.
    """
    steps_handle = open(Path(str(output_dir)) / "calibration_steps.jsonl", "w", encoding="utf-8")
    try:
        return _calibration_steps(svc, session_id, historical, steps_handle)
    finally:
        steps_handle.close()


def _calibration_steps(svc: Any, session_id: Any, historical: Any, steps_handle: Any) -> dict[str, Any]:
    """Replay the historical approach, then run the local controller (worker-owned)."""
    env = getattr(svc, "_env", None)
    sessions = getattr(svc, "_sessions", None)
    record = sessions.get(session_id) if isinstance(sessions, dict) else None
    result: dict[str, Any] = {
        "ok": False, "reason": None, "kind": None, "detail": "", "controller": None,
        "controller_mismatch": None, "historical_action_cap": A0_APPROACH_CAP,
        "historical_sent": 0, "local_action_cap": LOCAL_ACTION_CAP, "local_actions": 0,
        "aux_actions": 0, "first_aux_action": None, "trigger": None,
        "controller_grip_site_error_m": None, "grip_site_error_max_m": CONTROLLER_GRIP_SITE_ERROR_MAX_M,
        "protected_objects": list(PROTECTED_OBJECTS), "protection_tolerance_m": PROTECTION_TOLERANCE_M,
        "protection_baseline": None, "protection_violations": [], "wine_lift_m": None,
        "wine_lift_reference_m": None, "wine_lift_threshold_m": WINE_LIFT_M,
        "wine_lift_streak_required": WINE_LIFT_STREAK, "wine_lift_streak": 0, "confirmed": False,
        "helper_class": None, "helper_phase": None, "helper_reason": None, "helper_summary": None,
        "final_snapshot": None, "final_bowl_predicate": None, "phases": [], "steps": [], "frames": 0,
    }
    frames: list[Any] = []
    helper: Any = None

    def _write_row(row: dict[str, Any]) -> None:
        """One flushed JSONL row per ACTUAL action (worker-owned trace)."""
        steps_handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        steps_handle.flush()

    def _helper_summary() -> Any:
        try:
            return _jsonable(helper.summary())
        except Exception as exc:  # noqa: BLE001 - a failed summary is recorded, never faked
            return {"error": _format_exc(exc)}

    def _done(reason: str | None, kind: str | None, detail: str = "") -> dict[str, Any]:
        result["ok"] = reason is None
        result["reason"], result["kind"] = reason, kind
        if detail:
            result["detail"] = detail
        result["aux_actions"] = result["local_actions"]
        result["frames"] = len(frames)
        result["_frames"] = frames
        if helper is not None:
            result["helper_phase"] = _helper_phase(helper)
            result["helper_reason"] = getattr(helper, "reason", None)
            try:
                result["helper_summary"] = _jsonable(helper.summary())
            except Exception as exc:  # noqa: BLE001 - a failed summary is recorded, never faked
                result["helper_summary_error"] = _format_exc(exc)
        return result

    if env is None or record is None:
        return _done("no_env", "operational", "no live environment / session record")
    facts = pd.controller_facts(env)
    result["controller"] = _jsonable(facts)
    mismatch = pd.controller_mismatch(facts)
    result["controller_mismatch"] = mismatch
    if mismatch is not None:
        return _done("controller_mismatch", "operational", mismatch)
    try:
        helper = local_grasp.LocalGraspController()
    except Exception as exc:  # noqa: BLE001 - an unusable controller is operational
        return _done("helper_init", "operational", _format_exc(exc))
    result["helper_class"] = type(helper).__name__

    counter = [int(getattr(svc, "_total_steps", 0) or 0)]

    def _step(action: Any, kind: str, phase: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        sent = np.asarray(action, dtype=np.float32).reshape(-1)
        step_result = env.step(sent)
        if isinstance(step_result, tuple) and len(step_result) >= 1:
            svc._last_obs = step_result[0]
        counter[0] += 1
        svc._total_steps = record.total_steps = counter[0]
        frame = svc._safe_render(env)
        if frame is not None:
            frames.append(frame)
        snapshot = pe.capture_snapshot(env, [BOWL_GOAL])
        state_sha = service.state_sha(env)
        result["steps"].append({
            "kind": kind, "phase": _helper_phase(helper) if phase is None else phase,
            "action": _jsonable(sent), "snapshot": _jsonable(snapshot),
            "state_sha": state_sha,
        })
        row = {
            "action_source": "historical" if kind == "historical" else "local_grasp",
            "phase_before": _helper_phase(helper) if phase is None else phase,
            "phase_after": None, "action": _jsonable(sent), "geometry": None,
            "helper_summary": None, "snapshot": _jsonable(snapshot), "state_sha": state_sha,
        }
        return snapshot, row

    # One initial stationary frame so first.png / the video open pre-action.
    initial_frame = svc._safe_render(env)
    if initial_frame is not None:
        frames.append(initial_frame)

    # --- historical A0 approach replay (never sends the closing action) ------
    first_action: Any = None
    history_terminal = False
    for proposed in list(historical)[:A0_APPROACH_CAP]:
        action = helper.next_action(local_grasp.read_geometry(env), proposed)
        phase = _helper_phase(helper)
        result["phases"].append(phase)
        confirmed, failure = _helper_status(helper)
        if failure is not None:
            return _done(failure, "physical", "helper failed during the historical approach")
        if action is not None:
            first_action = np.asarray(action, dtype=np.float32).reshape(-1)
            result["first_aux_action"] = _jsonable(first_action)
            break
        if (phase or "").lower() not in IDLE_P:
            if confirmed:
                history_terminal = True
                break
            return _done("helper_not_idle", "physical", "helper phase %r produced no action" % phase)
        snapshot, row = _step(proposed, "historical", phase)
        row["phase_after"] = _helper_phase(helper)
        row["geometry"] = _jsonable(local_grasp.read_geometry(env))
        row["helper_summary"] = _helper_summary()
        _write_row(row)
        result["historical_sent"] += 1
    if first_action is None and not history_terminal:
        return _done("no_local_trigger", "physical",
                     "no auxiliary action within %d historical approach actions" % A0_APPROACH_CAP)

    # --- local execution (first action already computed) ---------------------
    baseline: dict[str, Any] = {}
    wine_z0: float | None = None
    streak = 0
    if not history_terminal:
        for _ in range(LOCAL_ACTION_CAP):
            if result["local_actions"] == 0:  # trigger gate BEFORE the first local action
                geometry = local_grasp.read_geometry(env)
                result["trigger"] = _jsonable(geometry)
                result["trigger_controller"] = _jsonable(facts)
                error = _grip_site_error(geometry, facts)
                result["controller_grip_site_error_m"] = error
                if error is None or error > CONTROLLER_GRIP_SITE_ERROR_MAX_M:
                    return _done("controller_grip_site_error", "physical",
                                 "controller_grip_site_error_m=%r > %g"
                                 % (error, CONTROLLER_GRIP_SITE_ERROR_MAX_M))
                baseline_snapshot = pe.capture_snapshot(env, [BOWL_GOAL])
                baseline = pd.snapshot_positions(baseline_snapshot)
                wine_z0 = _wine_z(baseline_snapshot)
                result["protection_baseline"] = _jsonable(baseline)
                result["wine_lift_reference_m"] = wine_z0
            snapshot, row = _step(first_action, "local", None)
            result["local_actions"] += 1
            violations = _protected_violations(baseline, pd.snapshot_positions(snapshot))
            if violations:
                result["protection_violations"] = violations
                row["phase_after"] = _helper_phase(helper)
                row["geometry"] = _jsonable(local_grasp.read_geometry(env))
                row["helper_summary"] = _helper_summary()
                _write_row(row)
                return _done("protection_violation", "physical", "; ".join(violations))
            z = _wine_z(snapshot)
            streak = streak + 1 if (z is not None and wine_z0 is not None and (z - wine_z0) >= WINE_LIFT_M) else 0
            result["wine_lift_m"] = None if (z is None or wine_z0 is None) else (z - wine_z0)
            result["wine_lift_streak"] = streak
            reading = local_grasp.read_geometry(env)
            helper.observe_after(reading)
            # Post-action trace captured AFTER the single observe_after (local only).
            row["phase_after"] = _helper_phase(helper)
            row["geometry"] = _jsonable(local_grasp.read_geometry(env))
            row["helper_summary"] = _helper_summary()
            _write_row(row)
            result["local_actions_total"] = getattr(helper, "total_actions", None)
            first_action = helper.next_action(reading)
            if first_action is None:
                break
            first_action = np.asarray(first_action, dtype=np.float32).reshape(-1)
        else:
            return _done("local_action_cap", "physical",
                         "no terminal helper state within %d local actions" % LOCAL_ACTION_CAP)

    # --- terminal: read-only final snapshot only (no extra action) -----------
    confirmed, failure = _helper_status(helper)
    if failure is not None:
        return _done(failure, "physical", "helper failed: %r" % getattr(helper, "reason", None))
    result["confirmed"] = confirmed
    final_snapshot = pe.capture_snapshot(env, [BOWL_GOAL])
    result["final_snapshot"] = _jsonable(final_snapshot)
    result["final_bowl_predicate"] = _bowl_true(final_snapshot)
    if not confirmed:
        return _done("not_confirmed", "physical", "helper phase %r" % _helper_phase(helper))
    if result["final_bowl_predicate"] is not True:
        return _done("final_bowl_predicate", "physical",
                     "native bowl predicate %r" % result["final_bowl_predicate"])
    if streak < WINE_LIFT_STREAK:
        return _done("wine_lift", "physical", "wine lift streak %d < %d" % (streak, WINE_LIFT_STREAK))
    return _done(None, None, "")


# --- preregistration / runner -------------------------------------------------


def _build_preregistration(action_sha: str, historical_sha: str, action_count: int,
                           historical_count: int) -> dict[str, Any]:
    """The frozen plan, written BEFORE the service (and any model) starts."""
    return {
        "experiment": EXPERIMENT_NAME, "created_utc": _now_utc(), "assisted": True,
        "scene_id": SCENE_ID, "scene_seed": SCENE_SEED, "init_state_index": INIT_STATE_INDEX,
        "completion_mode": COMPLETION_MODE, "grasp_guard_mode": GRASP_GUARD_MODE,
        "replay_input_path": REPLAY_ACTIONS_PATH, "replay_input_sha256": action_sha,
        "historical_input_sha256": historical_sha,
        "replay_action_count": action_count,
        "replay_expected_action_count": context.REPLAY_ACTION_COUNT,
        "replay_origin_state_sha": context.REPLAY_ORIGIN_SHA,
        "replay_final_state_sha": context.REPLAY_FINAL_SHA,
        "historical_telemetry_path": WINE_TELEMETRY_PATH, "historical_action_count": historical_count,
        "historical_approach_action_cap": A0_APPROACH_CAP, "local_action_cap": LOCAL_ACTION_CAP,
        "worker_timeout_s": WORKER_TIMEOUT_S, "protected_objects": list(PROTECTED_OBJECTS),
        "protection_tolerance_m": PROTECTION_TOLERANCE_M,
        "controller_grip_site_error_max_m": CONTROLLER_GRIP_SITE_ERROR_MAX_M,
        "wine_lift_threshold_m": WINE_LIFT_M, "wine_lift_streak": WINE_LIFT_STREAK,
        "controller_thresholds": {
            name: getattr(local_grasp, name) for name in (
                "WINE_PAD_HEIGHT_M", "ABOVE_CLEARANCE_M", "LIFT_CLEARANCE_M",
                "ALIGN_TOLERANCE_M", "ALIGN_TOLERANCE_RAD", "ALIGN_STREAK", "GRASP_STREAK",
                "LIFT_STREAK", "LIFT_SUCCESS_M", "NEAR_DISTANCE_M", "PROTECTION_TOLERANCE_M",
                "WINE_TRANSLATION_DRIFT_M", "WINE_ROTATION_DRIFT_RAD", "STAGE_BUDGETS",
                "TOTAL_MAX_ACTIONS",
            )
        },
        "target_formula": (
            "pad_target=wine_xyz+Rwine@[0,0,.105]; "
            "R candidates Rwine@diag(1,-1,-1) and that candidate additionally @diag(-1,-1,1); "
            "choose smaller orientation error relative to current pose, first on tie; "
            "eef=pad_target-Rselected@pad_offset_local; above=eef+[0,0,.06]"
        ),
        "helper_phase_vocabulary": {"idle": sorted(IDLE_P), "confirmed": sorted(CONFIRMED_P),
                                    "failed": sorted(FAIL_P)},
        "source_sha256": _source_sha256(), "hermes_calls": 0, "model_loads": 0,
    }


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME, "generated_utc": _now_utc(), "ok": False,
        "fatal_error": None, "assisted": True, "output_dir": str(args.output_dir),
        "run_root": str(Path(str(args.output_dir)) / SESSION_DIRNAME), "setup": {}, "service": {},
        "session": {}, "replay": None, "calibration": None, "media": {},
        "preregistration_sha256": None,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Calibrate once; persist ``calibration.json`` and the approach/local media."""
    output_dir = Path(str(args.output_dir))
    run_root = output_dir / SESSION_DIRNAME
    report = _base_report(args)
    svc: Any = None
    media: dict[str, Any] = {"first_png": None, "last_png": None, "video": None}
    try:
        run_root.mkdir(parents=True, exist_ok=True)

        actions = context.load_actions(REPLAY_ACTIONS_PATH)
        action_sha = _sha256_hex(Path(REPLAY_ACTIONS_PATH).read_bytes())
        historical = _load_sent_actions(WINE_TELEMETRY_PATH)
        historical_sha = _sha256_hex(Path(WINE_TELEMETRY_PATH).read_bytes())
        report["setup"] = {
            "replay_input_sha256": action_sha, "replay_action_count": len(actions),
            "historical_input_sha256": historical_sha,
            "historical_action_count": len(historical),
            "historical_approach_action_cap": A0_APPROACH_CAP, "local_action_cap": LOCAL_ACTION_CAP,
        }
        if not historical:
            raise RuntimeError("no sent_action rows in %s" % WINE_TELEMETRY_PATH)
        pe._write_json_atomic(output_dir / "preregistration.json",
                              _build_preregistration(action_sha, historical_sha, len(actions),
                                                     len(historical)))
        report["preregistration_sha256"] = _sha256_hex((output_dir / "preregistration.json").read_bytes())

        svc = pe.DiagnosticService(run_root=str(run_root), policy_loader=lambda _svc: None,
                                   completion_mode=COMPLETION_MODE, grasp_guard_mode=GRASP_GUARD_MODE)
        svc.start()
        readiness = _wait_ready(svc, READY_DEADLINE_S)
        report["service"] = readiness
        if not readiness.get("ready"):
            raise RuntimeError("service not ready: %s"
                               % (readiness.get("worker_error") or "timeout after %.0fs" % READY_DEADLINE_S))

        session = svc.create_session(SCENE_ID, seed=SCENE_SEED, init_state_index=INIT_STATE_INDEX)
        if not session.get("ok"):
            raise RuntimeError("create_session: %s" % (session,))
        session_id = session["session_id"]
        record = svc._sessions.get(session_id)
        report["session"] = {
            "session_id": session_id, "scene_id": SCENE_ID, "seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "initial_state_sha": getattr(record, "initial_state_hash", None),
            "xml_sha": getattr(record, "xml_sha", None),
        }

        replay = context.replay_prefix(svc, session_id, actions)
        report["replay"] = replay
        if not (isinstance(replay, dict) and replay.get("ok") is True):
            raise RuntimeError("replay_prefix: %s"
                               % (replay.get("reason") if isinstance(replay, dict) else replay))
        report["after_replay_state_sha"] = replay.get("final_state_sha")

        outcome = svc._sync_work("local_calibration",
                                 lambda: _calibration_work(svc, session_id, historical, output_dir),
                                 WORKER_TIMEOUT_S)
        calibration = outcome if isinstance(outcome, dict) else {
            "ok": False, "reason": "worker_error", "kind": "operational", "detail": repr(outcome)}
        frames = calibration.pop("_frames", []) or []
        report["calibration"] = calibration

        if frames:
            svc._save_frame(output_dir / "first.png", frames[0])
            svc._save_frame(output_dir / "last.png", frames[-1])
            media["first_png"], media["last_png"] = str(output_dir / "first.png"), str(output_dir / "last.png")
            if len(frames) >= 2:
                try:
                    service._save_video(output_dir / "calibration.mp4", frames, VIDEO_FPS)
                    media["video"] = str(output_dir / "calibration.mp4")
                except Exception as exc:  # noqa: BLE001 - a failed video is recorded, never faked
                    report["video_error"] = _format_exc(exc)
        report["ok"] = calibration.get("ok") is True
    except BaseException as exc:  # noqa: BLE001 - keep the real exception and its reason
        report["fatal_error"] = _format_exc(exc)
        report["ok"] = False
    finally:
        if svc is not None:
            try:
                close_env = getattr(svc, "_close_env", None)
                if callable(close_env):
                    svc._sync_work("close_env", lambda: (close_env() or {"ok": True}))
            except Exception:  # noqa: BLE001 - the environment close is best effort
                pass
            try:
                svc.stop()
            except Exception:  # noqa: BLE001
                pass
        report["media"], report["generated_utc"] = media, _now_utc()
        try:
            pe._write_json_atomic(output_dir / "calibration.json", report)
        except Exception as exc:  # noqa: BLE001
            report["write_error"] = _format_exc(exc)
    _progress("calibration report written (%s, ok=%s)" % (output_dir / "calibration.json", report.get("ok")))
    return report


# --- CLI ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ONE isolated assisted local-grasp physical calibration on the shared "
            "goal_table scene: after the recorded 102-action bowl prefix, replay at "
            "most %d historical A0 wine approach actions until the LocalGraspController "
            "returns its first auxiliary action, then run at most %d local actions. No "
            "model, no Hermes, no assessment/recovery/forced release."
            % (A0_APPROACH_CAP, LOCAL_ACTION_CAP)
        )
    )
    parser.add_argument("--output-dir", required=True, type=str,
                        help="absolute, absent output directory for the calibration artifacts")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isabs(str(args.output_dir)):
        parser.error("--output-dir must be an absolute path")
    if Path(str(args.output_dir)).exists():
        parser.error("--output-dir already exists; refusing to reuse %s" % args.output_dir)


def main(argv: list[str] | None = None) -> int:
    # ``--help`` parses without creating any worker / environment / model.
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    _configure_process_environment()
    report = _run(args)
    print("GRASP_CALIBRATION_PASS" if report.get("ok") else "GRASP_CALIBRATION_FAIL")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
