#!/usr/bin/env python3
"""Isolated six-trial wine *context* diagnostic runner (read-only host).

This module hosts :class:`ContextDiagnosticService` -- a subclass of the real
``guard_validation.GuardValidationService`` (itself a subclass of
``wine_diagnostics.WineDiagnosticService``, ``placement_experiments.DiagnosticService``
and the actual ``scene_demo/service.py`` ``SceneService``) -- plus a small,
focused campaign runner that drives ONE worker / ONE model load through exactly
six fixed trials and a recorded-policy *prefix replay*.

Nothing here retrains, downloads, resets the environment between subgoals, calls
Hermes, opens a socket, forces a release, teleports an object, runs a repair,
runs an assessment, runs a recovery step or uses a produced wine action.  The
base service lifecycle (one worker, one ``strict=True`` model load, the
release-verified completion gate, the wine-only grasp guard, ``events.jsonl`` /
PNG / MP4, the wine telemetry, the inherited first-input fingerprint capture) is
reused verbatim; the only addition is an independent, ``finally``-restored
shadow observer identical to the paired-config experiment's known pattern.

Fixed campaign
==============

Two model seeds (0, 1) each run three conditions in this exact order:

* ``native_goal9`` -- the native ``libero_goal/9`` wine scene (``wd.WINE_SCENE_ID``);
* ``shared_initial`` -- the shared ``goal_table`` scene (``wd.SHARED_SCENE_ID``),
  fresh;
* ``shared_after_bowl`` -- the same shared scene after a recorded-policy replay of
  the preserved 102 user bowl actions.

Every condition uses a *fresh* session with ``seed == init_state_index == 0``.
One service/model is loaded once per campaign, under
``wd.register_native_wine_scene()``.  The model RNG is seeded with
``paired._seed_model_rng`` AFTER the environment (and any prefix) exists; the
profile is configured and read back; the fixed pre-wine state SHA and the actual
XML/camera facts are armed for the inherited first-input capture; and exactly one
``wine_to_rack`` plan (budget 300, timeout 900) is submitted with no retry.

Recorded-policy replay (not new VLA execution)
==============================================

:func:`replay_prefix` replays the preserved 102 user bowl actions through the
worker-owned ``env.step`` -- exactly once per action -- after requiring the fixed
origin state SHA and before requiring the fixed final state SHA.  It is a
recorded-policy replay, never a newly generated bowl VLA execution, a recovery, a
repair or a forced release; the wine plan that follows is the only VLA execution
in the trial.

Independent raw stable-grasp statistic
======================================

:func:`summarize_raw_stable_grasp` derives one *monitor-blind* trial statistic,
``raw_stable_grasp_first_step`` (``int`` or ``None``), from the shadow observer's
actual raw probe telemetry only: the ``objects['wine_bottle_1'].position`` z and
``objects['wine_bottle_1'].grasped`` of every recorded row.  It is the actual step
of the fifth of five consecutive *known* samples whose wine z is at least ``0.02``
m above the *actual initial probe's* z while the probe reports ``grasped is True``.
A missing/nonfinite position or a non-boolean grasp resets the streak, an unknown
stays ``None`` and no baseline is ever invented.  Because it never consults the
guard monitor's stage, it can confirm a stable grasp even when the monitor has
latched ``failed_grasp`` (and left its own ``first_grasp_confirmed_step`` null);
``first_grasp_confirmed_step`` is therefore labelled only as *monitor
confirmation*, and its null value never means "never grasped".

Only stdlib + numpy are imported at module import time; ``torch`` is imported
lazily by the pinned base ``_select_action``, so ``--help`` and the GPU-free unit
tests never initialise CUDA, load a model or create a live environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import guard_validation as gv  # noqa: E402
import paired_config_experiments as paired  # noqa: E402
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402

EXPERIMENT_NAME = "skill_context_diagnostics"

ACTION_DIM = 7
REPLAY_ACTION_COUNT = 102

# The two fixed physical-state digests of the preserved bowl prefix.
REPLAY_ORIGIN_SHA = "8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd"
REPLAY_FINAL_SHA = "f0a5c0e135015071b7eaee75cffefb5f50f3ac07516240e03795e48d189e405e"

# The literal, independent wine oracle goal set and the literal bowl goal.
WINE_ORACLE_GOALS: list[list[str]] = [["on", "wine_bottle_1", "wine_rack_1_top_region"]]
BOWL_GOAL: list[str] = ["on", "akita_black_bowl_1", "plate_1"]
WINE_CAPABILITY_ID = "wine_to_rack"
WINE_INSTRUCTION = "put the wine bottle on the rack"

PROFILE = wd.BASELINE_PROFILE  # "baseline_bf16"
COMPLETION_MODE = wd.COMPLETION_MODE  # "release_verified"
GRASP_GUARD_MODE = "shadow"
RATIONALE = "wine context diagnostic"

MODEL_SEEDS = (0, 1)
SCENE_SEED = 0
INIT_STATE_INDEX = 0

CONDITION_ORDER = ("native_goal9", "shared_initial", "shared_after_bowl")
CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "native_goal9": {
        "scene_id": wd.WINE_SCENE_ID,
        "task_id": 9,
        "shared": False,
        "use_prefix": False,
    },
    "shared_initial": {
        "scene_id": wd.SHARED_SCENE_ID,
        "task_id": 8,
        "shared": True,
        "use_prefix": False,
    },
    "shared_after_bowl": {
        "scene_id": wd.SHARED_SCENE_ID,
        "task_id": 8,
        "shared": True,
        "use_prefix": True,
    },
}

EXPECTED_TRIAL_COUNT = len(MODEL_SEEDS) * len(CONDITION_ORDER)

BUDGET = 300
TIMEOUT_S = 900.0
READY_TIMEOUT_S = 900.0
REPLAY_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 0.5

HEALTH_URL = "http://%s:%d/health" % (service.DEFAULT_HOST, service.DEFAULT_PORT)

REPLAY_EVIDENCE_DIRNAME = "prefix_replay"

SOURCE_FILES = (
    "skill_context_diagnostics.py",
    "service.py",
    "grasp_guard.py",
    "wine_semantic.py",
    "wine_diagnostics.py",
    "placement_experiments.py",
    "guard_validation.py",
    "paired_config_experiments.py",
    "placement_completion.py",
    "catalog.py",
)

# The independent raw-probe stable-grasp window and threshold.  The lift
# threshold is the ACTUAL ``wine_diagnostics`` threshold (0.02 m), never guessed.
RAW_STABLE_GRASP_WINDOW = 5
RAW_STABLE_GRASP_LIFT_M = wd.LIFT_THRESHOLD_M

RAW_STABLE_GRASP_LIMITATION = (
    "raw_stable_grasp_first_step is an independent statistic computed ONLY from "
    "the shadow observer's actual raw probe telemetry "
    "(objects['wine_bottle_1'].position/grasped): the step of the fifth of "
    "%.0f consecutive known samples whose wine z is at least %.2f m above the "
    "initial probe's z while the probe reports grasped=True. It never consults "
    "the guard monitor's stage or first_grasp_confirmed_step, so it can confirm a "
    "stable grasp even after the monitor has latched failed_grasp; an unknown "
    "value is never a grasp." % (RAW_STABLE_GRASP_WINDOW, RAW_STABLE_GRASP_LIFT_M)
)

FIRST_GRASP_CONFIRMED_STEP_NOTE = (
    "first_grasp_confirmed_step is the shadow guard MONITOR's own confirmation "
    "step only. The monitor latches failed_grasp, so this field may stay null even "
    "when later shadow actions truly lift and grasp the bottle; a null value must "
    "never be read as 'never grasped'. The independent, monitor-blind statistic "
    "is raw_stable_grasp_first_step."
)

SEMANTIC_WINDOW_LIMITATION = (
    "the shadow semantic tracker needs 20 consecutive semantic samples to report "
    "a completed success, so a strict wine-success stop can occur before that "
    "window is reached. final_semantic and ever_semantic are kept separate from "
    "the strict wine success and a descriptive/unknown semantic status alone is "
    "never called a physical failure."
)

LIMITATIONS: dict[str, str] = {
    "native_vs_shared": (
        "native_goal9 and the shared goal_table conditions differ in task and "
        "environment; they are not the same scene."
    ),
    "before_vs_after_bowl": (
        "the before-bowl vs after-bowl conditions change the scene AND the robot "
        "pose; the after-bowl prefix is a recorded-policy replay, not a freshly "
        "generated VLA bowl execution."
    ),
    "reseed_not_reproduction": (
        "re-seeding the wine model RNG does not reproduce the user's original wine "
        "trajectory; it only fixes the model RNG for this campaign."
    ),
    "no_success_rate": "no success rate is inferred from the six recorded trials.",
    "recorded_policy_replay": (
        "the after-bowl prefix replays the preserved 102 user bowl actions through "
        "env.step; it is not a new VLA bowl execution, a recovery, a repair or a "
        "forced release."
    ),
    "no_settle_step": "no settle step is added after the replay; the state is read as-is.",
    "no_assessment": (
        "no assessment actions, repairs, forced release, recovery or Hermes calls "
        "are performed."
    ),
    "monitor_confirmation_null": FIRST_GRASP_CONFIRMED_STEP_NOTE,
    "semantic_window_vs_strict_stop": SEMANTIC_WINDOW_LIMITATION,
    "raw_stable_grasp": RAW_STABLE_GRASP_LIMITATION,
}


# --- small helpers -----------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _progress(message: str) -> None:
    sys.stderr.write("[skill-context %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the exact bytes of every pinned module this runner depends on."""

    digests: dict[str, str | None] = {}
    for name in SOURCE_FILES:
        try:
            digests[name] = _sha256_hex((_HERE / name).read_bytes())
        except Exception:  # noqa: BLE001 - absent/unreadable is recorded as unknown
            digests[name] = None
    return digests


# --- action loading (exactly 102 finite 7-D rows) ----------------------------


def _coerce_row(value: Any) -> list[float]:
    """Coerce one row into exactly ``ACTION_DIM`` finite floats, or raise."""

    if value is None or isinstance(value, (str, bytes)):
        raise ValueError("action row is not a sequence of numbers: %r" % (value,))
    try:
        components = list(value)
    except TypeError as exc:
        raise ValueError("action row is not iterable: %r" % (value,)) from exc
    numbers: list[float] = []
    for component in components:
        if isinstance(component, bool):
            raise ValueError("action row contains a bool component: %r" % (component,))
        try:
            number = float(component)
        except (TypeError, ValueError) as exc:
            raise ValueError("action row contains a non-numeric component: %r" % (component,)) from exc
        if not math.isfinite(number):
            raise ValueError("action row contains a non-finite component: %r" % (component,))
        numbers.append(number)
    if len(numbers) != ACTION_DIM:
        raise ValueError("action row has %d dimensions, expected %d" % (len(numbers), ACTION_DIM))
    return numbers


def _row_from_entry(entry: Any) -> list[float]:
    """One action row from either a bare list or an ``{"action": [...]}`` record."""

    if isinstance(entry, dict):
        if "action" not in entry:
            raise ValueError("record has no 'action' field: %r" % (sorted(entry),))
        return _coerce_row(entry.get("action"))
    return _coerce_row(entry)


def _rows_from_payload(payload: Any) -> list[list[float]]:
    """Rows from a whole-document JSON payload (array, wrapper object or record)."""

    if isinstance(payload, list):
        return [_row_from_entry(entry) for entry in payload]
    if isinstance(payload, dict):
        for key in ("actions", "action_list", "rows", "records"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [_row_from_entry(entry) for entry in inner]
        return [_row_from_entry(payload)]
    raise ValueError("actions payload is neither a list nor an object: %r" % (type(payload).__name__,))


def load_actions(path: Any) -> list[list[float]]:
    """Load exactly ``REPLAY_ACTION_COUNT`` finite 7-D action rows.

    Two preserved formats are accepted: a whole-document JSON array of rows (each
    a list of numbers or an ``{"action": [...]}`` record), and the actual
    ``events.jsonl`` line-delimited records whose ``action`` field carries one
    7-D sent action per line.  A missing/unreadable file, malformed JSON, a
    non-numeric / non-finite / wrong-width row, or any count other than exactly
    102 rows is rejected with ``ValueError``.  Nothing is fabricated and no row is
    ever silently dropped.
    """

    target = Path(path)
    if not target.is_file():
        raise ValueError("actions input is not a readable file: %s" % (path,))
    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("cannot read actions input %s: %s" % (path, exc)) from exc

    rows: list[list[float]] | None = None
    stripped = text.lstrip()
    if stripped[:1] in ("[", "{"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if payload is not None:
            rows = _rows_from_payload(payload)

    if rows is None:
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("line %d is not valid JSON: %s" % (lineno, exc)) from exc
            try:
                rows.append(_row_from_entry(record))
            except ValueError as exc:
                raise ValueError("line %d: %s" % (lineno, exc)) from exc

    if not rows:
        raise ValueError("no action rows found in %s" % (path,))
    if len(rows) != REPLAY_ACTION_COUNT:
        raise ValueError(
            "expected exactly %d action rows, found %d" % (REPLAY_ACTION_COUNT, len(rows))
        )
    return rows


# --- recorded-policy prefix replay -------------------------------------------


def _replay_evidence_dir(svc: Any, session_id: Any) -> Path | None:
    """The directory the replay evidence is written to (best effort)."""

    record = None
    sessions = getattr(svc, "_sessions", None)
    if isinstance(sessions, dict):
        record = sessions.get(session_id)
    run_dir = getattr(record, "run_dir", None) if record is not None else None
    if run_dir:
        return Path(run_dir) / REPLAY_EVIDENCE_DIRNAME
    root = getattr(svc, "run_root", None)
    if root:
        return Path(root) / ("%s_%s" % (REPLAY_EVIDENCE_DIRNAME, session_id))
    return None


def _flatten_sim_state(env: Any) -> tuple[list[float], str]:
    """The actual float64 simulator state flatten and its source label."""

    getter = getattr(env, "get_state", None)
    if callable(getter):
        try:
            flat = getter().flatten()
            array = np.asarray(flat, dtype=np.float64).reshape(-1)
            return [float(v) for v in array.tolist()], "env.get_state"
        except Exception:  # noqa: BLE001 - fall back to the inner sim state
            pass
    inner = service._inner_env(env)
    flat = inner.sim.get_state().flatten()
    array = np.asarray(flat, dtype=np.float64).reshape(-1)
    return [float(v) for v in array.tolist()], "inner.sim.get_state"


def _save_replay_evidence(evidence_dir: Any, result: dict[str, Any]) -> None:
    """Persist the real replay snapshot + float64 state to the trial directory."""

    if not evidence_dir:
        return
    try:
        directory = Path(evidence_dir)
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 - evidence saving never breaks the run
        return
    try:
        (directory / "replay_final_snapshot.json").write_text(
            json.dumps(result.get("final_snapshot"), indent=2, default=str), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        flat = result.get("state_flatten")
        if flat is not None:
            np.save(str(directory / "replay_state.npy"), np.asarray(flat, dtype=np.float64))
    except Exception:  # noqa: BLE001
        pass
    try:
        summary = {key: value for key, value in result.items() if key != "state_flatten"}
        summary["state_flatten_n"] = len(result.get("state_flatten") or [])
        summary["state_flatten_sha256"] = result.get("state_flatten_sha256")
        (directory / "replay_prefix.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass


def _replay_prefix_work(
    svc: Any, session_id: Any, actions: Any, evidence_dir: Any
) -> dict[str, Any]:
    """The worker-owned replay thunk: validate, check origin, step, check final."""

    result: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "detail": "",
        "expected_origin_state_sha": REPLAY_ORIGIN_SHA,
        "expected_final_state_sha": REPLAY_FINAL_SHA,
        "origin_state_sha": None,
        "final_state_sha": None,
        "replay_action_count": 0,
        "vla_action_count": 0,
        "expected_replay_action_count": REPLAY_ACTION_COUNT,
        "bowl_goal": [list(BOWL_GOAL)],
        "bowl_predicate": None,
        "strict_candidate": None,
        "final_snapshot": None,
        "state_flatten": None,
        "state_flatten_source": None,
        "state_flatten_sha256": None,
        "steps": 0,
        "evidence_dir": str(evidence_dir) if evidence_dir else None,
    }

    sessions = getattr(svc, "_sessions", None)
    record = sessions.get(session_id) if isinstance(sessions, dict) else None
    env = getattr(svc, "_env", None)
    if env is None or record is None:
        result["reason"] = "unknown_session"
        result["detail"] = "no live environment / session record for the prefix replay"
        return result

    rows: list[np.ndarray] = []
    for index, action in enumerate(actions or []):
        try:
            array = np.asarray(action, dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            result["reason"] = "invalid_prefix_actions"
            result["detail"] = "action %d is not numeric" % index
            return result
        if array.size != ACTION_DIM or not bool(np.all(np.isfinite(array))):
            result["reason"] = "invalid_prefix_actions"
            result["detail"] = "action %d has %d dims / non-finite components" % (index, array.size)
            return result
        rows.append(array)
    if len(rows) != REPLAY_ACTION_COUNT:
        result["reason"] = "invalid_prefix_actions"
        result["detail"] = "expected exactly %d actions, got %d" % (REPLAY_ACTION_COUNT, len(rows))
        return result

    origin = service.state_sha(env)
    result["origin_state_sha"] = origin
    if origin != REPLAY_ORIGIN_SHA:
        result["reason"] = "origin_state_sha_mismatch"
        result["detail"] = "actual %r != expected %r before any step" % (origin, REPLAY_ORIGIN_SHA)
        # Abort BEFORE any env.step (the origin requirement is checked first).
        _save_replay_evidence(evidence_dir, result)
        return result

    total = int(getattr(svc, "_total_steps", 0) or 0)
    for index, array in enumerate(rows):
        step_result = env.step(np.asarray(array, dtype=np.float32))
        if isinstance(step_result, tuple) and len(step_result) >= 1:
            svc._last_obs = step_result[0]
        total += 1
        svc._total_steps = total
        record.total_steps = total
        result["replay_action_count"] += 1
        result["steps"] = index + 1

    final = service.state_sha(env)
    result["final_state_sha"] = final
    if final != REPLAY_FINAL_SHA:
        result["reason"] = "final_state_sha_mismatch"
        result["detail"] = "actual %r != expected %r" % (final, REPLAY_FINAL_SHA)
        _save_replay_evidence(evidence_dir, result)
        return result

    try:
        bowl = bool(service.eval_goal_predicate(env, list(BOWL_GOAL)))
    except Exception as exc:  # noqa: BLE001
        result["reason"] = "bowl_predicate_error"
        result["detail"] = _format_exc(exc)
        _save_replay_evidence(evidence_dir, result)
        return result
    result["bowl_predicate"] = bowl

    try:
        snapshot = pe.capture_snapshot(env, [list(BOWL_GOAL)])
    except Exception as exc:  # noqa: BLE001
        result["reason"] = "final_snapshot_error"
        result["detail"] = _format_exc(exc)
        _save_replay_evidence(evidence_dir, result)
        return result
    result["final_snapshot"] = snapshot
    strict = snapshot.get("strict_candidate") if isinstance(snapshot, dict) else None
    result["strict_candidate"] = strict
    if bowl is not True or strict is not True:
        result["reason"] = "bowl_predicate_or_strict_candidate_false"
        result["detail"] = "bowl_predicate=%r strict_candidate=%r" % (bowl, strict)
        _save_replay_evidence(evidence_dir, result)
        return result

    try:
        flat, source = _flatten_sim_state(env)
        result["state_flatten"] = flat
        result["state_flatten_source"] = source
        result["state_flatten_sha256"] = _sha256_hex(
            np.ascontiguousarray(np.asarray(flat, dtype=np.float64)).tobytes()
        )
    except Exception as exc:  # noqa: BLE001 - a missing flatten is recorded, never faked
        result["detail"] = _format_exc(exc)

    result["vla_action_count"] = 0
    result["ok"] = True
    _save_replay_evidence(evidence_dir, result)
    return result


def replay_prefix(svc: Any, session_id: Any, actions: Any) -> dict[str, Any]:
    """Replay the preserved bowl prefix on the worker and return its evidence.

    All simulator reads and actions are worker-owned: the whole replay is
    dispatched through ``svc._sync_work``.  The origin state SHA is required
    *before any step*; every action is stepped exactly once through the real
    ``env.step``; the final state SHA, the bowl predicate and the strict bowl
    candidate are then required.  No reset / ``set_state`` / ``forward`` / policy
    call is ever made.
    """

    evidence_dir = _replay_evidence_dir(svc, session_id)
    work = lambda: _replay_prefix_work(svc, session_id, actions, evidence_dir)
    outcome = svc._sync_work("replay_prefix", work, REPLAY_TIMEOUT_S)
    if not isinstance(outcome, dict) or "origin_state_sha" not in outcome:
        detail = outcome.get("detail") if isinstance(outcome, dict) else repr(outcome)
        reason = outcome.get("reason") if isinstance(outcome, dict) else "replay_non_dict_result"
        return {
            "ok": False,
            "reason": reason or "replay_worker_error",
            "detail": detail or "",
            "origin_state_sha": None,
            "final_state_sha": None,
            "replay_action_count": 0,
            "vla_action_count": 0,
        }
    return outcome


# --- the context diagnostic service ------------------------------------------


class ContextDiagnosticService(gv.GuardValidationService):
    """Wine diagnostics + a single shadow grasps/semantic observer per job.

    Everything (one worker, the ``strict=True`` model load, the release-verified
    completion gate, the ``shadow`` wine-only grasp guard, the inherited
    first-input fingerprint capture, wine telemetry, ``events.jsonl``, PNG/MP4) is
    inherited verbatim.  Only :meth:`_run_capability` is overridden: it installs a
    ``finally``-restored :class:`paired_config_experiments.PassiveStepObserver`
    around this job's worker-owned ``env.step`` (which starts the shadow
    :class:`paired_config_experiments.GuardSemanticObserver` before the first
    action), then delegates to the base implementation.  The observer only
    records: it never changes live termination, success or error, and no
    assessment action, repair, forced release or recovery step is ever run.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("completion_mode", COMPLETION_MODE)
        kwargs.setdefault("grasp_guard_mode", GRASP_GUARD_MODE)
        super().__init__(*args, **kwargs)
        self._context_tracker: Any = None
        self._context_observer_summary: dict | None = None

    def _run_capability(
        self,
        session: service.SessionRecord,
        plan: service.PlanRecord,
        job: service.JobRecord,
        capability_id: str,
    ) -> dict[str, Any]:
        """Shadow-observe this capability's ``env.step``, then run the base."""

        env = self._env
        if env is None or self._env_session_id != session.session_id:
            return super()._run_capability(session, plan, job, capability_id)

        try:
            job.run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001 - the base creates it too
            pass

        tracker = self._context_tracker
        if tracker is None:
            tracker = paired.make_semantic_tracker()
            self._context_tracker = tracker

        observer = paired.GuardSemanticObserver(env, job.run_dir, tracker)
        step_observer = paired.PassiveStepObserver(env, observer)
        step_observer.install()
        try:
            result = super()._run_capability(session, plan, job, capability_id)
        finally:
            step_observer.restore()
            observer.close()
            self._context_observer_summary = observer.summary()
        return result


# --- worker-side read helpers ------------------------------------------------


def _worker_state_sha(svc: Any) -> dict[str, Any]:
    """Read the live physical-state SHA on the worker thread (read-only)."""

    def _work() -> dict[str, Any]:
        env = getattr(svc, "_env", None)
        if env is None:
            return {"ok": False, "reason": "no_env", "detail": "no live environment"}
        try:
            return {"ok": True, "state_sha": service.state_sha(env)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_sha_failed", "detail": _format_exc(exc)}

    return svc._sync_work("state_sha", _work)


def _final_snapshot(svc: Any, goals: Any) -> dict[str, Any]:
    """Read one read-only final snapshot (worker-owned); unknown stays unknown."""

    try:
        result = svc.final_snapshot(goals)
    except Exception as exc:  # noqa: BLE001
        return {"error": _format_exc(exc)}
    if isinstance(result, dict) and result.get("ok") and isinstance(result.get("snapshot"), dict):
        return result["snapshot"]
    return {"error": result}


def _predicate_of(snapshot: Any) -> bool | None:
    if not isinstance(snapshot, dict):
        return None
    predicates = snapshot.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        return None
    if any(value is None for value in predicates.values()):
        return None
    return all(bool(value) for value in predicates.values())


def _strict_of(snapshot: Any) -> bool | None:
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get("strict_candidate")
    return value if isinstance(value, bool) else None


def _fingerprint_digest(fingerprint: Any) -> str | None:
    if not isinstance(fingerprint, dict):
        return None
    value = fingerprint.get("combined_sha256")
    return value if isinstance(value, str) else None


def _read_observer_rows(observer_summary: Any) -> list[dict[str, Any]]:
    """Read the shadow observer's ``guard_semantic.jsonl`` rows (best effort)."""

    if not isinstance(observer_summary, dict):
        return []
    path = observer_summary.get("guard_semantic_path")
    if not path:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(record, dict):
                    rows.append(record)
    except Exception:  # noqa: BLE001 - a missing telemetry file is simply empty
        return []
    return rows


def _trial_scores(
    job_evidence: Any, observer_summary: Any, observer_rows: Any
) -> tuple[bool | None, bool | None, bool | None]:
    """Strict (from the actual wine job), final and ever semantic, kept separate.

    The *strict* score is the actual wine ``wine_to_rack`` job's ``success`` flag
    (never inferred from the semantic observer).  The ``final_semantic`` is the
    shadow observer's *last* tracked tri-state status; the ``ever_semantic`` is a
    separate tri-state over the recorded rows.  An unknown value is never
    promoted into a success.
    """

    strict: bool | None = None
    for evidence in job_evidence or []:
        if not isinstance(evidence, dict):
            continue
        job = evidence.get("job") or {}
        if job.get("capability_id") != WINE_CAPABILITY_ID:
            continue
        success = job.get("success")
        strict = success if isinstance(success, bool) else None

    status = observer_summary.get("last_semantic_status") if isinstance(observer_summary, dict) else None
    final_semantic = paired._semantic_flag(status)

    values = [
        row.get("semantic_success") for row in (observer_rows or []) if isinstance(row, dict)
    ]
    if any(value is True for value in values):
        ever_semantic: bool | None = True
    elif any(value is False for value in values):
        ever_semantic = False
    else:
        ever_semantic = None
    return strict, final_semantic, ever_semantic


# --- independent raw stable-grasp summarizer ---------------------------------


def _probe_object_entry(probe: Any) -> dict | None:
    """The ``objects['wine_bottle_1']`` entry of one raw probe, or ``None``.

    Reuses the exact ``wine_diagnostics`` object id and the exact
    ``grasp_guard.read_probe`` / ``placement_experiments.capture_snapshot`` probe
    schema -- never a guessed field name.
    """

    if not isinstance(probe, dict):
        return None
    objects = probe.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(wd.WINE_OBJECT_ID)
    return entry if isinstance(entry, dict) else None


def _probe_wine_z(probe: Any) -> float | None:
    """The finite wine bottle z of one raw probe, or ``None`` (unknown)."""

    entry = _probe_object_entry(probe)
    if entry is None:
        return None
    position = entry.get("position")
    if not wd._is_sequence(position) or len(position) < 3:
        return None
    return wd._finite_scalar(position[2])


def _probe_wine_grasped(probe: Any) -> bool | None:
    """The actual boolean wine grasp of one raw probe, or ``None`` (unknown)."""

    entry = _probe_object_entry(probe)
    if entry is None:
        return None
    value = entry.get("grasped")
    return value if isinstance(value, bool) else None


def _observer_row_probe(row: Any) -> dict | None:
    """The raw probe of one shadow observer row (or the row itself when it IS one)."""

    if not isinstance(row, dict):
        return None
    probe = row.get("probe")
    if isinstance(probe, dict):
        return probe
    # A bare probe mapping (the actual ``grasp_guard.read_probe`` schema).
    if isinstance(row.get("objects"), dict):
        return row
    return None


def summarize_raw_stable_grasp(
    rows: Any,
    window: int = RAW_STABLE_GRASP_WINDOW,
    lift_threshold_m: float = RAW_STABLE_GRASP_LIFT_M,
) -> dict[str, Any]:
    """Independent, monitor-blind stable-grasp statistic from actual probe rows.

    Only the raw probe telemetry is used: each row's
    ``objects['wine_bottle_1'].position`` z and ``objects['wine_bottle_1'].grasped``
    are read directly, and the guard monitor's stage is never consulted.  The
    statistic is the actual step of the fifth of ``window`` consecutive *known*
    samples whose wine z is at least ``lift_threshold_m`` above the z of the
    actual initial probe while the probe reports ``grasped is True``.  A missing
    or nonfinite position, or a non-boolean grasp, resets the streak.  The start z
    is sourced only from the actual initial probe: when it is unknown the whole
    statistic stays ``None`` (no invented baseline), and an unknown outcome stays
    ``None`` -- it is never turned into a grasp.
    """

    threshold = float(lift_threshold_m)
    record: dict[str, Any] = {
        "raw_stable_grasp_first_step": None,
        "raw_stable_grasp_start_z": None,
        "raw_stable_grasp_start_step": None,
        "raw_stable_grasp_window": int(window),
        "raw_stable_grasp_lift_threshold_m": threshold,
        "n_raw_probe_samples": 0,
        "raw_stable_grasp_source": "shadow_observer_probe",
        "raw_stable_grasp_limitation": RAW_STABLE_GRASP_LIMITATION,
    }
    if not isinstance(rows, (list, tuple)):
        return record

    start_probe: dict | None = None
    start_step: int | None = None
    samples: list[tuple[int, dict]] = []
    for row in rows:
        probe = _observer_row_probe(row)
        if probe is None:
            continue
        if isinstance(row, dict) and row.get("phase") == "start":
            if start_probe is None:
                start_probe = probe
                step = row.get("step")
                start_step = (
                    step if isinstance(step, int) and not isinstance(step, bool) else None
                )
            continue
        step = row.get("step") if isinstance(row, dict) else None
        if isinstance(step, int) and not isinstance(step, bool):
            samples.append((step, probe))

    if start_probe is None:
        # No actual initial probe -> the baseline is unknown and never invented.
        return record

    start_z = _probe_wine_z(start_probe)
    record["raw_stable_grasp_start_z"] = start_z
    record["raw_stable_grasp_start_step"] = start_step
    if start_z is None:
        return record

    record["n_raw_probe_samples"] = len(samples)

    streak = 0
    first_step: int | None = None
    for step, probe in samples:
        z = _probe_wine_z(probe)
        grasped = _probe_wine_grasped(probe)
        if z is None or grasped is None:
            # A missing/nonfinite position or a non-boolean grasp resets the streak.
            streak = 0
            continue
        if (z - start_z) >= threshold and grasped is True:
            streak += 1
            if first_step is None and streak >= int(window):
                first_step = step
        else:
            streak = 0
    record["raw_stable_grasp_first_step"] = first_step
    return record


# --- per-trial context reset and capture arming ------------------------------


def _reset_trial_context(svc: Any) -> None:
    """Reset the per-trial inherited capture/tracker context on the service.

    A *fresh* shadow semantic tracker is created per trial -- the previous
    tracker must never persist across sessions.  The inherited first-input
    fingerprint capture fields/counts are cleared, and the fixed wine oracle
    condition is selected so the inherited fixed-oracle telemetry evaluates the
    wine goals (never the empty declared-goal set).
    """

    svc._context_tracker = None
    svc._context_observer_summary = None
    svc._diag_condition = wd.FINAL_ORACLE_KEY
    svc.first_fingerprint = None
    svc.action_selection_count = 0
    svc.first_input_evidence = None
    svc._first_input_captured = False


def _arm_first_input_capture(
    svc: Any, *, session_id: Any, model_seed: int, trial: dict[str, Any]
) -> Any:
    """Arm the inherited first-input capture with the ACTUAL pre-wine state.

    ``initial_state_sha`` is the actual ``before_wine_state_sha`` (the physical
    state the wine policy actually starts from), never the original session SHA:
    for the after-bowl condition the prefix replay has already changed the state,
    so the original session SHA is not the pre-wine input context.
    """

    return svc.arm_input_capture(
        session_id,
        INIT_STATE_INDEX,
        model_seed,
        PROFILE,
        initial_state_sha=trial.get("before_wine_state_sha"),
        xml_sha=trial.get("xml_sha"),
        camera_names=trial.get("camera_names") or [],
        control_frequency_hz=trial.get("control_frequency_hz"),
    )


def _first_input_operational_errors(xml_sha: Any, first_fingerprint: Any) -> list[str]:
    """Operational errors for a missing XML SHA or an incomplete fingerprint.

    A nonempty XML SHA and a *complete* first-input fingerprint
    (:func:`paired_config_experiments.fingerprint_is_complete`) are required
    before a trial may be accepted: ``None == None`` (or an empty/malformed
    digest) is never evidence of matched inputs.
    """

    errors: list[str] = []
    if not (isinstance(xml_sha, str) and xml_sha.strip()):
        errors.append("xml_sha_missing_or_empty: %r" % (xml_sha,))
    if not paired.fingerprint_is_complete(first_fingerprint):
        errors.append("first_fingerprint_incomplete")
    return errors


# --- plan / trial bookkeeping ------------------------------------------------


def _plan_summary(plan_record: Any) -> dict[str, Any]:
    plan_public = (plan_record or {}).get("plan") or {}
    return {
        "request_id": (plan_record or {}).get("request_id"),
        "submitted": (plan_record or {}).get("submitted"),
        "submit_error": (plan_record or {}).get("submit_error"),
        "timed_out": (plan_record or {}).get("timed_out"),
        "cancel_nonterminal": (plan_record or {}).get("cancel_nonterminal"),
        "cancelled": (plan_record or {}).get("cancelled"),
        "wall_s": (plan_record or {}).get("wall_s"),
        "state": plan_public.get("state"),
        "plan_success": plan_public.get("plan_success"),
        "completed_capability_ids": plan_public.get("completed_capability_ids"),
        "pending_capability_ids": plan_public.get("pending_capability_ids"),
        "job_ids": plan_public.get("job_ids"),
    }


def build_campaign_plan() -> list[dict[str, Any]]:
    """The fixed six-trial order: model seed 0 then 1, condition order fixed."""

    plan: list[dict[str, Any]] = []
    for model_seed in MODEL_SEEDS:
        for condition in CONDITION_ORDER:
            spec = CONDITION_SPECS[condition]
            plan.append(
                {
                    "model_seed": int(model_seed),
                    "condition": condition,
                    "scene_id": spec["scene_id"],
                    "task_id": int(spec["task_id"]),
                    "shared": bool(spec["shared"]),
                    "use_prefix": bool(spec["use_prefix"]),
                }
            )
    return plan


def _compare_across_seeds(trial: dict[str, Any], seen: dict[str, Any], op: Any) -> None:
    """Require the physical input context match across model seeds.

    The initial XML SHA, the first-input fingerprint digest and the actual
    pre-wine state SHA must be non-null and identical for every model-seed repeat
    of the same condition (``None == None`` is never evidence of matched inputs);
    the after-bowl final state SHA is required additionally for the after-bowl
    condition.
    """

    condition = trial.get("condition")
    current = {
        "xml_sha": trial.get("xml_sha"),
        "first_fingerprint": _fingerprint_digest(trial.get("first_fingerprint")),
        "before_wine_state_sha": trial.get("before_wine_state_sha"),
        "after_bowl_state_sha": trial.get("after_bowl_state_sha"),
    }
    reference = seen.get(condition)
    if reference is None:
        seen[condition] = current
        return
    for key in ("xml_sha", "first_fingerprint", "before_wine_state_sha"):
        left = reference.get(key)
        right = current.get(key)
        if left is None or right is None or left != right:
            op("cross_seed_%s: condition=%s %r != %r" % (key, condition, left, right))
    if condition == "shared_after_bowl":
        left = reference.get("after_bowl_state_sha")
        right = current.get("after_bowl_state_sha")
        if left is None or right is None or left != right:
            op("cross_seed_after_bowl_sha: condition=%s %r != %r" % (condition, left, right))


def _run_trial(
    svc: Any, entry: dict[str, Any], actions: Any, run_root: Path, seen: dict[str, Any]
) -> dict[str, Any]:
    """Run one fixed trial: fresh session, optional prefix, one wine plan."""

    condition = entry["condition"]
    model_seed = int(entry["model_seed"])
    spec = CONDITION_SPECS[condition]
    scene_id = spec["scene_id"]
    trial_id = "%s_m%d_%s" % (condition, model_seed, uuid.uuid4().hex[:8])
    started = time.monotonic()
    trial: dict[str, Any] = {
        "trial_id": trial_id,
        "condition": condition,
        "model_seed": model_seed,
        "scene_id": scene_id,
        "task_id": int(spec["task_id"]),
        "shared": bool(spec["shared"]),
        "use_prefix": bool(spec["use_prefix"]),
        "scene_seed": SCENE_SEED,
        "init_state_index": INIT_STATE_INDEX,
        "profile": PROFILE,
        "completion_mode": getattr(svc, "completion_mode", COMPLETION_MODE),
        "grasp_guard_mode": getattr(svc, "grasp_guard_mode", GRASP_GUARD_MODE),
        "capability_id": WINE_CAPABILITY_ID,
        "instruction": WINE_INSTRUCTION,
        "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
        "bowl_goal": list(BOWL_GOAL),
        "session_id": None,
        "initial_state_sha": None,
        "xml_sha": None,
        "camera_names": [],
        "control_frequency_hz": None,
        "before_wine_state_sha": None,
        "prefix": None,
        "after_bowl_state_sha": None,
        "model_rng_seed": None,
        "profile_readback": None,
        "first_fingerprint": None,
        "first_input": None,
        "plan": None,
        "jobs": [],
        "observer_summary": None,
        "final_snapshot": None,
        "final_bowl_snapshot": None,
        "final_bowl_predicate": None,
        "final_bowl_strict": None,
        "strict_wine_success": None,
        "final_semantic": None,
        "ever_semantic": None,
        "first_grasp_confirmed_step": None,
        "first_grasp_confirmed_step_is_monitor_confirmation": True,
        "first_grasp_confirmed_step_note": FIRST_GRASP_CONFIRMED_STEP_NOTE,
        "first_failed_grasp_step": None,
        "raw_stable_grasp": None,
        "raw_stable_grasp_first_step": None,
        "physical_failures": [],
        "errors": [],
        "operational_errors": [],
        "timings": {},
    }
    operational = trial["operational_errors"]

    def _op(message: str) -> None:
        operational.append(message)
        trial["errors"].append(message)

    try:
        # Reset the per-trial inherited capture/tracker context BEFORE any action:
        # a fresh shadow tracker per trial, the fixed wine oracle condition, and
        # cleared first-input capture fields/counts.
        _reset_trial_context(svc)

        session = svc.create_session(scene_id, seed=SCENE_SEED, init_state_index=INIT_STATE_INDEX)
        if not session.get("ok"):
            _op("create_session: %s" % (session,))
            return trial
        session_id = session["session_id"]
        trial["session_id"] = session_id
        record = svc._sessions.get(session_id)
        initial_state_sha = getattr(record, "initial_state_hash", None)
        xml_sha = getattr(record, "xml_sha", None)
        camera_names = list(getattr(record, "camera_names", None) or [])
        trial["initial_state_sha"] = initial_state_sha
        trial["xml_sha"] = xml_sha
        trial["camera_names"] = camera_names

        # The shared fresh initial state must equal the fixed replay origin BEFORE
        # any prefix replay or wine action.
        if spec["shared"] and initial_state_sha != REPLAY_ORIGIN_SHA:
            _op(
                "shared_initial_state_sha: %r != fixed origin %r"
                % (initial_state_sha, REPLAY_ORIGIN_SHA)
            )
            return trial

        if spec["use_prefix"]:
            replay = replay_prefix(svc, session_id, actions)
            trial["prefix"] = replay
            if not (isinstance(replay, dict) and replay.get("ok") is True):
                reason = replay.get("reason") if isinstance(replay, dict) else replay
                _op("prefix_replay: %s" % (reason,))
                return trial
            trial["after_bowl_state_sha"] = replay.get("final_state_sha")
            if (
                replay.get("replay_action_count") != REPLAY_ACTION_COUNT
                or replay.get("vla_action_count") != 0
            ):
                _op(
                    "prefix_replay_counts: replay=%r vla=%r"
                    % (replay.get("replay_action_count"), replay.get("vla_action_count"))
                )
                return trial

        # Seed the model RNG only AFTER the env (and any prefix) exists.
        seeded = paired._seed_model_rng(svc, model_seed)
        trial["model_rng_seed"] = {
            "requested": model_seed,
            "result": seeded if isinstance(seeded, dict) else None,
            "seeded": seeded.get("seeded") if isinstance(seeded, dict) else None,
            "reason": seeded.get("reason") if isinstance(seeded, dict) else None,
        }
        if not (
            isinstance(seeded, dict) and seeded.get("ok") is True and seeded.get("seeded") is True
        ):
            _op("model_rng_seed: %s" % (seeded,))
            return trial

        # Configure + read back the fixed profile exactly (before any action).
        profile_result = svc.configure_profile(PROFILE)
        trial["profile_readback"] = profile_result
        if not gv._profile_readback_ok(profile_result):
            _op("profile_config: %s" % (profile_result,))
            return trial

        # Reset the inherited first-input capture fields/counts was done at the
        # top of the trial (``_reset_trial_context``); the profile readback above
        # does not perform any action.

        frequency = paired._read_control_frequency(svc)
        control_frequency_hz = frequency.get("control_frequency_hz") if frequency.get("ok") else None
        trial["control_frequency_hz"] = control_frequency_hz

        state = _worker_state_sha(svc)
        before_wine_state_sha = state.get("state_sha") if isinstance(state, dict) and state.get("ok") else None
        trial["before_wine_state_sha"] = before_wine_state_sha
        if before_wine_state_sha is None:
            _op("pre_wine_state_sha unavailable: %s" % (state,))
            return trial

        # Arm the first-input capture with the ACTUAL pre-wine state SHA (for the
        # after-bowl condition the prefix replay has already changed the state),
        # never the original session SHA.
        _arm_first_input_capture(svc, session_id=session_id, model_seed=model_seed, trial=trial)

        request_id = "%s-plan1" % trial_id
        plan_record = pe._submit_and_wait(
            svc,
            session_id,
            [WINE_CAPABILITY_ID],
            BUDGET,
            False,
            request_id,
            TIMEOUT_S,
            RATIONALE,
        )
        trial["plan"] = _plan_summary(plan_record)
        plan_public = plan_record.get("plan") or {}
        job_ids = plan_public.get("job_ids") or []
        job_evidence = [pe._job_evidence(svc, job_id) for job_id in job_ids]
        trial["jobs"] = job_evidence

        trial["final_snapshot"] = _final_snapshot(svc, WINE_ORACLE_GOALS)
        if spec["shared"]:
            bowl_snapshot = _final_snapshot(svc, [BOWL_GOAL])
            trial["final_bowl_snapshot"] = bowl_snapshot
            trial["final_bowl_predicate"] = _predicate_of(bowl_snapshot)
            trial["final_bowl_strict"] = _strict_of(bowl_snapshot)

        first_input_evidence = getattr(svc, "first_input_evidence", None)
        first_fingerprint = getattr(svc, "first_fingerprint", None)
        trial["first_fingerprint"] = first_fingerprint
        trial["first_input"] = (
            paired._input_record(first_input_evidence) if isinstance(first_input_evidence, dict) else None
        )
        observer_summary = getattr(svc, "_context_observer_summary", None)
        trial["observer_summary"] = observer_summary if isinstance(observer_summary, dict) else None
        observer_rows = _read_observer_rows(observer_summary)
        strict, final_semantic, ever_semantic = _trial_scores(
            job_evidence, observer_summary, observer_rows
        )
        trial["strict_wine_success"] = strict
        trial["final_semantic"] = final_semantic
        trial["ever_semantic"] = ever_semantic
        if isinstance(observer_summary, dict):
            # ``first_grasp_confirmed_step`` is the MONITOR's own confirmation
            # (see the note); a null value never means "never grasped".
            trial["first_grasp_confirmed_step"] = observer_summary.get("first_grasp_confirmed_step")
            trial["first_failed_grasp_step"] = observer_summary.get("first_failed_grasp_step")

        # Independent, monitor-blind statistic from the actual raw probe rows.
        raw_stable_grasp = summarize_raw_stable_grasp(observer_rows)
        trial["raw_stable_grasp"] = raw_stable_grasp
        trial["raw_stable_grasp_first_step"] = raw_stable_grasp.get("raw_stable_grasp_first_step")

        # --- required input provenance and one-wine-job consistency -----------
        # A nonempty XML SHA and a complete first-input fingerprint are required
        # before accepting a trial; ``None == None`` is never matched inputs.
        for message in _first_input_operational_errors(xml_sha, first_fingerprint):
            _op(message)

        available_wine_jobs = [
            evidence
            for evidence in job_evidence
            if isinstance(evidence, dict)
            and evidence.get("available")
            and (evidence.get("job") or {}).get("capability_id") == WINE_CAPABILITY_ID
        ]
        if len(available_wine_jobs) != 1:
            _op(
                "wine_job_count: expected exactly one available %s job, found %d"
                % (WINE_CAPABILITY_ID, len(available_wine_jobs))
            )
        wine_steps: int | None = None
        if len(available_wine_jobs) == 1:
            raw_steps = (available_wine_jobs[0].get("job") or {}).get("steps")
            if isinstance(raw_steps, int) and not isinstance(raw_steps, bool) and raw_steps > 0:
                wine_steps = raw_steps
            else:
                _op("wine_job_steps_not_positive: %r" % (raw_steps,))
        if wine_steps is not None:
            selection_count = getattr(svc, "action_selection_count", None)
            if selection_count != wine_steps:
                _op(
                    "action_selection_count_mismatch: %r != wine steps %r"
                    % (selection_count, wine_steps)
                )
            if isinstance(observer_summary, dict):
                observer_samples = observer_summary.get("n_samples")
                if observer_samples != wine_steps:
                    _op(
                        "observer_n_samples_mismatch: %r != wine steps %r"
                        % (observer_samples, wine_steps)
                    )
        if isinstance(observer_summary, dict) and observer_summary.get("error"):
            _op("observer_error: %s" % (observer_summary.get("error"),))

        for evidence in job_evidence:
            if not isinstance(evidence, dict) or not evidence.get("available"):
                continue
            job = evidence.get("job") or {}
            state_ = job.get("state")
            ended = job.get("ended_reason")
            detail = job.get("error")
            if state_ == "cancelled" or ended == "cancelled":
                _op(
                    "job_cancelled: job=%s state=%s ended_reason=%s"
                    % (evidence.get("job_id"), state_, ended)
                )
            elif state_ == "error" or ended == "error" or detail:
                _op(
                    "job_error: job=%s state=%s ended_reason=%s error=%s"
                    % (evidence.get("job_id"), state_, ended, detail or "")
                )
            elif ended == "budget_exhausted":
                trial["physical_failures"].append("budget_exhausted: job=%s" % evidence.get("job_id"))
            elif ended == "failed_grasp":
                trial["physical_failures"].append("failed_grasp: job=%s" % evidence.get("job_id"))

        if plan_record.get("submit_error"):
            _op("plan_submit: %s" % (plan_record.get("submit_error"),))
        if plan_record.get("timed_out"):
            _op("plan_timeout: %s" % (plan_record.get("request_id"),))
        if plan_record.get("cancel_nonterminal"):
            _op("cancel_nonterminal: %s" % (plan_record.get("request_id"),))
        if plan_record.get("cancelled") or plan_public.get("state") == "cancelled":
            _op(
                "plan_cancelled: request=%s state=%s"
                % (plan_record.get("request_id"), plan_public.get("state"))
            )
        if plan_public.get("state") == "error":
            _op(
                "plan_error: request=%s error=%s"
                % (plan_record.get("request_id"), plan_public.get("error"))
            )

        _compare_across_seeds(trial, seen, _op)
    except BaseException as exc:  # noqa: BLE001 - preserve the record
        _op("trial_exception: %s" % _format_exc(exc))
    finally:
        trial["timings"]["trial_wall_s"] = round(time.monotonic() - started, 3)
    return trial


# --- service construction / readiness / production health --------------------


def _build_service(args: argparse.Namespace) -> ContextDiagnosticService:
    """Construct the service with the exact required keyword arguments."""

    return ContextDiagnosticService(
        model_path=service.DEFAULT_MODEL_PATH,
        run_root=str(args.run_root),
        completion_mode=COMPLETION_MODE,
        grasp_guard_mode=GRASP_GUARD_MODE,
    )


def _wait_ready(svc: Any, timeout_s: float = READY_TIMEOUT_S) -> dict[str, Any]:
    """Poll the existing health method until ready / errored / timed out."""

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        try:
            health = svc.health()
        except Exception as exc:  # noqa: BLE001 - a raising health is not ready
            health = {"ready": False, "worker_error": _format_exc(exc)}
        if isinstance(health, dict) and health.get("worker_error"):
            return {
                "ready": False,
                "worker_error": str(health.get("worker_error")),
                "model_revision": health.get("model_revision"),
            }
        if isinstance(health, dict) and health.get("ready"):
            return {
                "ready": True,
                "worker_error": None,
                "model_revision": health.get("model_revision"),
            }
        time.sleep(POLL_INTERVAL_S)
    return {"ready": False, "worker_error": None, "model_revision": None}


def _production_health(url: str = HEALTH_URL, timeout_s: float = 5.0) -> dict[str, Any]:
    """Read the production ``/health`` endpoint read-only (never stops production)."""

    try:
        from urllib.request import ProxyHandler, Request, build_opener
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "urllib_unavailable", "detail": _format_exc(exc)}
    opener = build_opener(ProxyHandler({}))
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with opener.open(request, timeout=timeout_s) as response:
            body = response.read()
        data = json.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - unreachable is recorded, never fabricated
        return {"ok": False, "reason": "health_unreachable", "detail": _format_exc(exc), "raw": None}
    if not isinstance(data, dict):
        return {"ok": False, "reason": "health_malformed", "detail": repr(data), "raw": None}
    return {
        "ok": True,
        "reason": None,
        "detail": "",
        "active_request_id": data.get("active_request_id"),
        "raw": data,
    }


def _health_record(health: Any) -> dict[str, Any]:
    if not isinstance(health, dict):
        return {"ok": False, "reason": "non_dict", "active_request_id": None}
    record: dict[str, Any] = {
        "ok": bool(health.get("ok")),
        "reason": health.get("reason"),
        "active_request_id": health.get("active_request_id"),
    }
    raw = health.get("raw")
    if isinstance(raw, dict):
        record["ready"] = raw.get("ready")
        record["worker_error"] = raw.get("worker_error")
        record["model_revision"] = raw.get("model_revision")
    return record


# --- preregistration and report ----------------------------------------------


def _build_preregistration(
    args: argparse.Namespace, plan: list[dict[str, Any]], actions: Any
) -> dict[str, Any]:
    """The frozen plan, written BEFORE the model is loaded."""

    return {
        "experiment": EXPERIMENT_NAME,
        "created_utc": _now_utc(),
        "commitment": (
            "written before the model starts; the raw SHA-256 of this file's bytes is "
            "recorded in the campaign report and the fixed order is never re-decided "
            "after data collection begins"
        ),
        "fixed_spec": {
            "condition_order": list(CONDITION_ORDER),
            "conditions": {name: dict(CONDITION_SPECS[name]) for name in CONDITION_ORDER},
            "model_seeds": list(MODEL_SEEDS),
            "scene_seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "budget": BUDGET,
            "timeout_s": TIMEOUT_S,
            "profile": PROFILE,
            "completion_mode": COMPLETION_MODE,
            "grasp_guard_mode": GRASP_GUARD_MODE,
            "capability_id": WINE_CAPABILITY_ID,
            "instruction": WINE_INSTRUCTION,
            "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
            "bowl_goal": list(BOWL_GOAL),
            "replay_action_count": REPLAY_ACTION_COUNT,
            "replay_origin_state_sha": REPLAY_ORIGIN_SHA,
            "replay_final_state_sha": REPLAY_FINAL_SHA,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "target_revision": paired.FIXED_SOURCE_REVISION,
            "assessment_actions": 0,
            "forced_release": False,
            "recovery": False,
            "assisted": False,
            "hermes_calls": 0,
        },
        "plan": plan,
        "inputs": {
            "input_actions_path": str(args.input_actions),
            "input_actions_count": len(actions),
            "input_actions_sha256": _sha256_hex(Path(args.input_actions).read_bytes()),
        },
        "limitations": dict(LIMITATIONS),
        "source_sha256": _source_sha256(),
        "source_git_sha": getattr(args, "source_git_sha", None),
    }


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "ok": False,
        "fatal_error": None,
        "metadata": {
            "output": str(args.output),
            "run_root": str(args.run_root),
            "input_actions_path": str(args.input_actions),
            "condition_order": list(CONDITION_ORDER),
            "model_seeds": list(MODEL_SEEDS),
            "scene_seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "budget": BUDGET,
            "timeout_s": TIMEOUT_S,
            "profile": PROFILE,
            "completion_mode": COMPLETION_MODE,
            "grasp_guard_mode": GRASP_GUARD_MODE,
            "model_path": service.DEFAULT_MODEL_PATH,
            "model_revision": None,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "target_revision": paired.FIXED_SOURCE_REVISION,
            "source_git_sha": None,
            "source_sha256": _source_sha256(),
            "replay_action_count": REPLAY_ACTION_COUNT,
            "replay_origin_state_sha": REPLAY_ORIGIN_SHA,
            "replay_final_state_sha": REPLAY_FINAL_SHA,
            "production_health_url": HEALTH_URL,
            "assisted": False,
            "hermes_calls": 0,
            "campaign_wall_s": None,
        },
        "preregistration": None,
        "service": {},
        "health_checks": [],
        "trials": [],
        "operational_errors": [],
        "limitations": dict(LIMITATIONS),
        "aggregate": {},
    }


def _aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_trials": len(trials),
        "expected_trials": EXPECTED_TRIAL_COUNT,
        "n_operational_errors": sum(1 for trial in trials if trial.get("operational_errors")),
        "n_strict_wine_success": sum(1 for trial in trials if trial.get("strict_wine_success") is True),
        "n_strict_wine_failure": sum(1 for trial in trials if trial.get("strict_wine_success") is False),
        "n_final_semantic_success": sum(1 for trial in trials if trial.get("final_semantic") is True),
        "n_ever_semantic_success": sum(1 for trial in trials if trial.get("ever_semantic") is True),
        "n_raw_stable_grasp": sum(
            1 for trial in trials if trial.get("raw_stable_grasp_first_step") is not None
        ),
        "n_physical_failures": sum(1 for trial in trials if trial.get("physical_failures")),
        "n_budget_exhausted": sum(
            1
            for trial in trials
            if any("budget_exhausted" in item for item in (trial.get("physical_failures") or []))
        ),
        "reliability_note": (
            "explicit counts only; six recorded trials are not a statistical "
            "reliability claim and no success rate is inferred"
        ),
    }


def _aggregate_operational(trials: list[dict[str, Any]], fatal_error: str | None) -> list[str]:
    messages: list[str] = []
    if fatal_error:
        messages.append("fatal: %s" % fatal_error)
    for trial in trials:
        for message in trial.get("operational_errors") or []:
            messages.append("%s_m%s: %s" % (trial.get("condition"), trial.get("model_seed"), message))
    return messages


# --- the campaign ------------------------------------------------------------


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed six-trial campaign and return the persisted report."""

    output_path = Path(args.output)
    run_root = Path(args.run_root)
    started = time.monotonic()
    report = _base_report(args)
    trials: list[dict[str, Any]] = report["trials"]
    health_checks: list[dict[str, Any]] = report["health_checks"]
    persisted_errors: list[str] = []
    seen: dict[str, Any] = {}
    fatal_error: str | None = None
    svc: Any = None

    def _persist() -> None:
        report["fatal_error"] = fatal_error
        report["operational_errors"] = _aggregate_operational(trials, fatal_error)
        report["aggregate"] = _aggregate(trials)
        report["metadata"]["campaign_wall_s"] = round(time.monotonic() - started, 3)
        report["ok"] = bool(
            fatal_error is None
            and not persisted_errors
            and len(trials) == EXPECTED_TRIAL_COUNT
            and all(not trial.get("operational_errors") for trial in trials)
        )
        try:
            pe._write_json_atomic(output_path, report)
        except Exception as exc:  # noqa: BLE001 - a report write failure is fatal
            persisted_errors.append(_format_exc(exc))
            _progress("report write failed: %s" % exc)

    try:
        run_root.mkdir(parents=True, exist_ok=True)

        try:
            actions = load_actions(args.input_actions)
        except Exception as exc:  # noqa: BLE001 - no actions -> no run
            raise RuntimeError("load_actions: %s" % exc) from exc
        report["metadata"]["input_actions_count"] = len(actions)
        report["metadata"]["input_actions_sha256"] = _sha256_hex(Path(args.input_actions).read_bytes())

        plan = build_campaign_plan()
        report["metadata"]["plan"] = plan
        report["metadata"]["source_git_sha"] = args.source_git_sha or pe._git_rev_parse()

        # 1. Freeze the preregistration BEFORE any model / GPU execution.
        prereg_path = run_root / "preregistration.json"
        try:
            pe._write_json_atomic(prereg_path, _build_preregistration(args, plan, actions))
            prereg_sha = _sha256_hex(prereg_path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - no preregistration -> no run
            raise RuntimeError("preregistration write failed: %s" % _format_exc(exc)) from exc
        report["preregistration"] = {
            "path": str(prereg_path),
            "sha256": prereg_sha,
            "frozen_before_model_start": True,
        }
        _persist()

        # 2. One service / one model load for the whole campaign.
        with wd.register_native_wine_scene():
            svc = _build_service(args)
            svc.start()
            readiness = _wait_ready(svc, READY_TIMEOUT_S)
            report["service"] = readiness
            if not readiness.get("ready"):
                raise RuntimeError(
                    "service not ready: %s"
                    % (readiness.get("worker_error") or "timeout after %.0fs" % READY_TIMEOUT_S)
                )
            report["metadata"]["model_revision"] = readiness.get("model_revision")

            total = len(plan)
            for index, entry in enumerate(plan, start=1):
                health = _production_health()
                health_checks.append(
                    {
                        "trial_index": index,
                        "condition": entry["condition"],
                        "model_seed": entry["model_seed"],
                        "result": _health_record(health),
                    }
                )
                if not health.get("ok"):
                    fatal_error = "production health unreachable before trial %d/%d: %s" % (
                        index,
                        total,
                        health.get("reason") or health.get("detail"),
                    )
                    _progress(fatal_error)
                    break
                if health.get("active_request_id") is not None:
                    fatal_error = (
                        "production busy before trial %d/%d: active_request_id=%r; refusing to "
                        "start a trial (production is never stopped)" % (index, total, health["active_request_id"])
                    )
                    _progress(fatal_error)
                    break

                _progress(
                    "trial %d/%d condition=%s model_seed=%s" % (index, total, entry["condition"], entry["model_seed"])
                )
                try:
                    trial = _run_trial(svc, entry, actions, run_root, seen)
                except BaseException as exc:  # noqa: BLE001 - preserve, then stop
                    trial = {
                        "trial_id": "%s_m%s" % (entry["condition"], entry["model_seed"]),
                        "condition": entry["condition"],
                        "model_seed": entry["model_seed"],
                        "scene_id": entry["scene_id"],
                        "errors": ["trial_exception: %s" % _format_exc(exc)],
                        "operational_errors": ["trial_exception: %s" % _format_exc(exc)],
                    }
                    trials.append(trial)
                    fatal_error = "trial_exception: %s" % _format_exc(exc)
                    _persist()
                    break
                trials.append(trial)
                _persist()
                _progress(
                    "trial %d/%d done strict=%s final_semantic=%s operational=%d"
                    % (
                        index,
                        total,
                        trial.get("strict_wine_success"),
                        trial.get("final_semantic"),
                        len(trial.get("operational_errors") or []),
                    )
                )
                if trial.get("operational_errors"):
                    fatal_error = "operational error(s); stopping campaign (no continuation): %s" % (
                        "; ".join(trial["operational_errors"]),
                    )
                    _progress(fatal_error)
                    break
    except BaseException as exc:  # noqa: BLE001 - never lose the partial report
        fatal_error = fatal_error or _format_exc(exc)
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

    _persist()
    if report.get("ok"):
        print("CONTEXT_DIAGNOSTICS_PASS")
    _progress("report written to %s (%d trials)" % (output_path, len(trials)))
    return report


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated six-trial wine context diagnostic runner: native vs shared, "
            "before vs after a recorded-policy bowl prefix.  No training, downloads, "
            "assessment, forced release, recovery or HTTP server."
        )
    )
    parser.add_argument(
        "--input-actions",
        required=True,
        type=str,
        help="absolute path of the preserved 102-action bowl prefix (events.jsonl or JSON list)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="absolute path of the JSON report (must not exist)",
    )
    parser.add_argument(
        "--run-root",
        required=True,
        type=str,
        help="absolute fresh run root for per-trial artifacts (must not exist; created)",
    )
    parser.add_argument(
        "--source-git-sha",
        type=str,
        default=None,
        help="optional fallback source git SHA (used when git is unavailable)",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isabs(str(args.input_actions)):
        parser.error("--input-actions must be an absolute path")
    if not os.path.isabs(str(args.output)):
        parser.error("--output must be an absolute path")
    if not os.path.isabs(str(args.run_root)):
        parser.error("--run-root must be an absolute path")
    if not Path(str(args.input_actions)).is_file():
        parser.error("--input-actions must be an existing readable file: %s" % args.input_actions)
    if Path(str(args.output)).exists():
        parser.error("--output already exists; refusing to overwrite %s" % args.output)
    if Path(str(args.run_root)).exists():
        parser.error("--run-root already exists; refusing to reuse %s" % args.run_root)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    report = run_campaign(args)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
