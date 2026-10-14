#!/usr/bin/env python3
"""Isolated live grasp-guard validation runner (single plan, read-only freeze).

This module hosts :class:`GuardValidationService` -- a subclass of the real
``wine_diagnostics.WineDiagnosticService`` (itself a subclass of
``placement_experiments.DiagnosticService`` and of the actual
``scene_demo/service.py`` ``SceneService``) -- plus a small runner that drives
exactly one fixed plan through one fresh session and then reads (never mutates)
the frozen worker-owned state twice.

Nothing here retrains, downloads, resets the environment, forces a release,
teleports an object, calls Hermes or opens a socket.  The base service lifecycle
is inherited verbatim: one worker, one ``strict=True`` model load, the
release-verified completion gate, the wine-only grasp guard (``off`` or
``enforce``), the native termination / ``events.jsonl`` / PNG / MP4 logic, and
the wine telemetry with its raw/post/sent audits.

Exact, single-run contract
==========================

* the CLI requires a *fresh absolute* ``--output`` JSON path and ``--run-root``
  directory and refuses to overwrite/reuse either;
* exactly one session is created with ``seed == init_state_index ==
  --state-index``; the model RNG is seeded *after* the environment exists (and
  the profile is configured *after* the RNG seed), and the environment is never
  reset and the RNG is never reseeded again;
* exactly one literal plan is submitted (``['wine_to_rack']`` or
  ``['wine_to_rack', 'bowl_to_plate']``) with the fixed budget ``300`` and
  timeout ``900``;
* after the plan is terminal, exactly two read-only worker probes read the
  physical state hash (via the real ``service.state_sha``), the session step
  count, the real action-selection count and the policy action-queue length --
  separated by exactly ``0.25`` seconds in the *runner* thread.  No
  ``env.step`` / ``env.reset`` / ``env.forward``, no policy prediction and no
  assessment action runs during the freeze.

Operational vs physical
=======================

An operational protocol failure (a plan/job ``state=="error"``, an
``ended_reason=="error"``, a truthy ``job.error``, a submission/readiness/profile
failure, or a cancellation/timeout) sets ``ok=False`` and a non-zero CLI exit.
An ordinary ``budget_exhausted`` / ``failed_grasp`` outcome is recorded as a
*physical* failure -- never as an operational error and never as a fabricated
success.

Only stdlib + numpy are imported at module import time; ``torch`` is imported
lazily by the pinned base ``SceneService._select_action``, so ``--help`` and the
GPU-free unit tests never initialise CUDA, load a model or create a live
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
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

import paired_config_experiments as paired  # noqa: E402
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402

EXPERIMENT_NAME = "grasp_guard_validation"

# The single fixed profile (the base ``configure_profile`` readback contract).
PROFILE = wd.BASELINE_PROFILE
COMPLETION_MODE = wd.COMPLETION_MODE  # "release_verified"
RATIONALE = "grasp guard single-plan live validation"

# Fixed budget / deadline (not CLI-tunable).
BUDGET = 300
TIMEOUT_S = 900.0
READY_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 0.5
FREEZE_SLEEP_S = 0.25

# The only guard modes this runner may arm; "off" and "enforce" (no "shadow").
GUARD_MODES = ("off", "enforce")
# The exact catalog scene ids, never an invented alias.
SCENE_CHOICES = (wd.WINE_SCENE_ID, wd.SHARED_SCENE_ID)

CAPABILITIES_WINE = "wine"
CAPABILITIES_WINE_BOWL = "wine_bowl"
CAPABILITY_PLANS = {
    CAPABILITIES_WINE: ("wine_to_rack",),
    CAPABILITIES_WINE_BOWL: ("wine_to_rack", "bowl_to_plate"),
}
CAPABILITY_CHOICES = tuple(CAPABILITY_PLANS)

CHECKPOINT_FILENAME = "model.safetensors"
SOURCE_FILES = (
    "service.py",
    "grasp_guard.py",
    "wine_semantic.py",
    "wine_diagnostics.py",
    "placement_experiments.py",
    "guard_validation.py",
)

# The installed SmolVLA action-queue key convention (see ``ACTION`` in lerobot's
# SmolVLA modeling module); ``action`` is the documented default.
ACTION_QUEUE_FALLBACK_KEY = "action"
_EXPECTED_PROFILE_APPLIED = {"use_amp": True, "num_steps": 10, "n_action_steps": 1}

# A physical-state digest must be exactly 64 lowercase hex characters.  No
# aliasing, casing or length coercion is accepted.
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# --- small helpers -----------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the exact bytes of every pinned module this runner depends on.

    Hashing reads file bytes only; it never loads model weights.
    """

    digests: dict[str, str | None] = {}
    for name in SOURCE_FILES:
        try:
            digests[name] = _sha256_hex((_HERE / name).read_bytes())
        except Exception:  # noqa: BLE001 - absent/unreadable is recorded as unknown
            digests[name] = None
    return digests


def _checkpoint_sha256(model_path: str) -> str | None:
    """SHA-256 of the pinned ``model.safetensors`` bytes (no weight load)."""

    try:
        return _sha256_hex((Path(model_path) / CHECKPOINT_FILENAME).read_bytes())
    except Exception:  # noqa: BLE001 - missing/unreadable checkpoint is unknown
        return None


def _capability_ids(capabilities: str) -> tuple[str, ...]:
    return CAPABILITY_PLANS[capabilities]


def _profile_readback_ok(result: Any) -> bool:
    """Whether the real ``configure_profile`` return shows the exact profile.

    ``result`` must be a dict whose ``ok`` is *exactly* ``True``.  Both
    ``applied['policy.config']`` and ``applied['policy.model.config']`` must be
    dictionaries that read back ``use_amp`` *exactly* ``True`` (a genuine bool),
    ``num_steps`` a genuine ``int`` equal to 10 and ``n_action_steps`` a genuine
    ``int`` equal to 1.  Bool-as-int, strings, floats, missing entries and
    malformed mappings all fail closed -- nothing is coerced.
    """

    if not isinstance(result, dict) or result.get("ok") is not True:
        return False
    applied = result.get("applied")
    if not isinstance(applied, dict):
        return False
    for key in ("policy.config", "policy.model.config"):
        entry = applied.get(key)
        if not isinstance(entry, dict):
            return False
        if entry.get("use_amp") is not True:
            return False
        num_steps = entry.get("num_steps")
        if (
            type(num_steps) is not int
            or num_steps != _EXPECTED_PROFILE_APPLIED["num_steps"]
        ):
            return False
        n_action_steps = entry.get("n_action_steps")
        if (
            type(n_action_steps) is not int
            or n_action_steps != _EXPECTED_PROFILE_APPLIED["n_action_steps"]
        ):
            return False
    return True


def _control_frequency_hz(env: Any) -> int | None:
    """Read the live control frequency (Hz) without touching the simulator state."""

    value: Any = None
    if env is not None:
        value = getattr(env, "control_freq", None)
        if value is None:
            inner = getattr(getattr(env, "_env", None), "env", None)
            if inner is not None:
                value = getattr(inner, "control_freq", None)
                if value is None:
                    model = getattr(getattr(inner, "sim", None), "model", None)
                    timestep = getattr(getattr(model, "opt", None), "timestep", None)
                    if timestep:
                        try:
                            value = 1.0 / float(timestep)
                        except Exception:  # noqa: BLE001
                            value = None
    try:
        return int(round(float(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


# --- action-queue length (read only; never resets a queue) -------------------


def _action_queue_key(policy: Any) -> str:
    """The installed action key, falling back to the documented ``action`` key."""

    queues = getattr(policy, "_queues", None)
    # A policy instance/class bearing the real ``ACTION`` constant wins.
    candidate = getattr(policy, "ACTION", None)
    if not isinstance(candidate, str) or not candidate:
        module = sys.modules.get(type(policy).__module__)
        candidate = getattr(module, "ACTION", None) if module is not None else None
    if isinstance(candidate, str) and candidate:
        return candidate
    if queues is not None and not isinstance(queues, dict):
        try:
            keys = list(queues.keys())
        except Exception:  # noqa: BLE001
            keys = []
        if len(keys) == 1 and isinstance(keys[0], str):
            return keys[0]
    return ACTION_QUEUE_FALLBACK_KEY


def _action_queue_length(policy: Any) -> int | None:
    """``len(policy._queues[ACTION])`` without mutating or resetting any queue."""

    if policy is None:
        return None
    queues = getattr(policy, "_queues", None)
    if queues is None:
        return None
    key = _action_queue_key(policy)
    queue = None
    try:
        queue = queues[key]
    except Exception:  # noqa: BLE001 - support a mapping whose only key IS the action
        try:
            keys = list(queues.keys())
        except Exception:  # noqa: BLE001
            return None
        if len(keys) == 1:
            try:
                queue = queues[keys[0]]
            except Exception:  # noqa: BLE001
                return None
        else:
            return None
    try:
        return len(queue)
    except Exception:  # noqa: BLE001
        return None


# --- read-only physical-state hash -------------------------------------------


def _direct_state_sha(env: Any) -> str:
    """Fallback hash identical to ``service.state_sha`` (float64 flattened bytes)."""

    sim = service._inner_env(env).sim
    try:
        flat = sim.get_state().flatten()
    except Exception:  # noqa: BLE001 - fall back to the raw qpos/qvel vectors
        flat = np.concatenate(
            [
                np.asarray(sim.data.qpos, dtype=np.float64),
                np.asarray(sim.data.qvel, dtype=np.float64),
            ]
        )
    state = np.asarray(flat, dtype=np.float64).reshape(-1)
    return _sha256_hex(np.ascontiguousarray(state, dtype=np.float64).tobytes())


def _physical_hash(service_: Any, env: Any) -> tuple[str | None, str | None]:
    """The physical state digest via the real ``service.state_sha`` (or fallback)."""

    if env is None:
        return None, None
    helper = getattr(service, "state_sha", None)
    if callable(helper):
        try:
            digest = helper(env)
        except Exception:  # noqa: BLE001
            digest = None
        if isinstance(digest, str) and digest:
            return digest, "service.state_sha"
    try:
        return _direct_state_sha(env), "direct_get_state"
    except Exception:  # noqa: BLE001
        return None, None


# --- read-only freeze probes -------------------------------------------------


def _read_total_steps(service_: Any, session_id: Any) -> int | None:
    sessions = getattr(service_, "_sessions", None)
    if isinstance(sessions, dict):
        record = sessions.get(session_id)
        value = getattr(record, "total_steps", None)
        if value is not None:
            return value
    return getattr(service_, "_total_steps", None)


def _freeze_probe(service_: Any, session_id: Any):
    """A worker-thread thunk that only *reads* the frozen worker-owned state."""

    def _work() -> dict[str, Any]:
        env = getattr(service_, "_env", None)
        physical_hash, source = _physical_hash(service_, env)
        policy = getattr(getattr(service_, "_v1", None), "_policy", None)
        return {
            "ok": True,
            "physical_hash": physical_hash,
            "physical_hash_source": source,
            "total_steps": _read_total_steps(service_, session_id),
            "action_selection_count": getattr(service_, "action_selection_count", None),
            "action_queue_length": _action_queue_length(policy),
        }

    return _work


def _probe_equal(first: Any, second: Any) -> bool:
    """Whether two read-only probes agree on every required evidence field.

    Both probes must be dictionaries whose ``ok`` is *exactly* ``True``.  Each
    ``physical_hash`` must be a 64-character lowercase hexadecimal string and
    each of ``total_steps`` / ``action_selection_count`` /
    ``action_queue_length`` must be a genuine non-negative ``int`` (bools,
    strings, floats and missing/null values are rejected).  Only once both
    inputs validate are the four evidence fields compared exactly.
    """

    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if first.get("ok") is not True or second.get("ok") is not True:
        return False
    for probe in (first, second):
        physical_hash = probe.get("physical_hash")
        if not isinstance(physical_hash, str) or _HEX64_RE.fullmatch(physical_hash) is None:
            return False
        for field in ("total_steps", "action_selection_count", "action_queue_length"):
            value = probe.get(field)
            if type(value) is not int or value < 0:
                return False
    for field in (
        "physical_hash",
        "total_steps",
        "action_selection_count",
        "action_queue_length",
    ):
        if first.get(field) != second.get(field):
            return False
    return True


def _run_freeze_probes(service_: Any, session_id: Any) -> dict[str, Any]:
    """Two read-only worker probes separated by exactly ``0.25`` s (runner thread)."""

    first = service_._sync_work("freeze_probe_1", _freeze_probe(service_, session_id))
    time.sleep(FREEZE_SLEEP_S)
    second = service_._sync_work("freeze_probe_2", _freeze_probe(service_, session_id))
    return {
        "sleep_s": FREEZE_SLEEP_S,
        "probe_1": first,
        "probe_2": second,
        "equal": _probe_equal(first, second),
    }


# --- plan result collection --------------------------------------------------


def _collect_plan_result(
    service_: Any,
    plan_record: dict[str, Any],
    declared_capabilities: Any,
) -> dict[str, Any]:
    """Collect the real plan/job evidence and classify operational vs physical.

    A cancellation, a timeout, a job ``state=="error"``, an
    ``ended_reason=="error"`` or a truthy ``job.error`` is an *operational*
    error.  An ordinary ``budget_exhausted`` / ``failed_grasp`` is a *physical*
    failure.  A physical failure never sets ``ok=False`` and is never a success.
    """

    declared = [str(cap) for cap in (declared_capabilities or [])]
    plan_public = plan_record.get("plan") or {}
    if not isinstance(plan_public, dict):
        plan_public = {}

    record = {
        "request_id": plan_record.get("request_id"),
        "submitted": plan_record.get("submitted"),
        "submit_error": plan_record.get("submit_error"),
        "timed_out": plan_record.get("timed_out"),
        "cancel_nonterminal": plan_record.get("cancel_nonterminal"),
        "cancelled": plan_record.get("cancelled"),
        "wall_s": plan_record.get("wall_s"),
        "state": plan_public.get("state"),
        "error": plan_public.get("error"),
        "plan_success": plan_public.get("plan_success"),
        "completed_capability_ids": plan_public.get("completed_capability_ids"),
        "pending_capability_ids": plan_public.get("pending_capability_ids"),
        "job_ids": plan_public.get("job_ids"),
    }

    jobs: list[dict[str, Any]] = []
    operational: list[str] = []
    physical: list[str] = []
    executed: list[str] = []

    for job_id in plan_public.get("job_ids") or []:
        evidence = wd._job_evidence(service_, job_id)
        if not evidence.get("available"):
            jobs.append({"job_id": job_id, "available": False})
            continue
        job = evidence.get("job") or {}
        capability_id = job.get("capability_id")
        if capability_id and capability_id not in executed:
            executed.append(str(capability_id))
        jobs.append(
            {
                "job_id": job_id,
                "available": True,
                "capability_id": capability_id,
                "state": job.get("state"),
                "ended_reason": job.get("ended_reason"),
                "error": job.get("error"),
                "success": job.get("success"),
                "steps": job.get("steps"),
                "total_steps": job.get("total_steps"),
                "instruction": job.get("instruction"),
                "completion_mode": job.get("completion_mode"),
                "phase": job.get("phase"),
                "held_objects": job.get("held_objects"),
                "completion_ready": job.get("completion_ready"),
                "state_before_sha": job.get("state_before_sha"),
                "state_after_sha": job.get("state_after_sha"),
                "guard": {
                    "grasp_guard_mode": job.get("grasp_guard_mode"),
                    "grasp_stage": job.get("grasp_stage"),
                    "grasp_guard_status": job.get("grasp_guard_status"),
                },
                "run_dir": job.get("run_dir"),
                "wine_telemetry_path": evidence.get("wine_telemetry_path"),
                "wine_diagnostic_path": evidence.get("wine_diagnostic_path"),
                "events_path": evidence.get("events_path"),
                "result_path": evidence.get("result_path"),
                "rollout_path": evidence.get("rollout_path"),
                "latest_png": evidence.get("latest_png"),
            }
        )

        state = job.get("state")
        ended = job.get("ended_reason")
        detail = job.get("error")
        if state == "cancelled" or ended == "cancelled":
            operational.append(
                "job_cancelled: job=%s state=%s ended_reason=%s" % (job_id, state, ended)
            )
        elif state == "error" or ended == "error" or detail:
            operational.append(
                "job_error: job=%s state=%s ended_reason=%s error=%s"
                % (job_id, state, ended, detail or "")
            )
        elif ended == "budget_exhausted":
            physical.append(
                "budget_exhausted: job=%s capability=%s" % (job_id, capability_id)
            )
        elif ended == "failed_grasp":
            physical.append(
                "failed_grasp: job=%s capability=%s" % (job_id, capability_id)
            )

    if plan_record.get("submit_error"):
        operational.append("plan_submit: %s" % (plan_record.get("submit_error"),))
    if plan_record.get("timed_out"):
        operational.append("plan_timeout: %s" % (plan_record.get("request_id"),))
    if plan_record.get("cancel_nonterminal"):
        operational.append("cancel_nonterminal: %s" % (plan_record.get("request_id"),))
    if plan_record.get("cancelled") or record["state"] == "cancelled":
        operational.append(
            "plan_cancelled: request=%s state=%s cancelled=%s"
            % (plan_record.get("request_id"), record["state"], plan_record.get("cancelled"))
        )
    if record["state"] == "error":
        operational.append(
            "plan_error: request=%s error=%s"
            % (plan_record.get("request_id"), record.get("error"))
        )

    skipped = [cap for cap in declared if cap not in executed]
    return {
        "plan": record,
        "jobs": jobs,
        "declared_capabilities": declared,
        "executed_capabilities": executed,
        "skipped_capabilities": skipped,
        "operational_errors": operational,
        "physical_failures": physical,
    }


# --- the guarded validation service ------------------------------------------


class GuardValidationService(wd.WineDiagnosticService):
    """Wine diagnostics + a single read-only first-input fingerprint capture.

    Everything (one worker, the ``strict=True`` model load, the release-verified
    completion gate, the wine-only grasp guard, wine telemetry, ``events.jsonl``,
    PNG/MP4) is inherited verbatim.  Only :meth:`_select_action` is overridden: it
    counts every *real* action-selection call, fingerprints the very first
    preprocessed batch exactly once (before delegating), records the first-input
    evidence matching ``paired._input_record``'s schema, and then returns the base
    implementation's action object *unchanged* (the batch and the action are never
    mutated).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.first_fingerprint: dict | None = None
        self.action_selection_count: int = 0
        self.first_input_evidence: dict | None = None
        self._first_input_captured: bool = False
        # Physical facts read from the real session record / live worker.
        self._guard_session_id: str | None = None
        self._guard_state_index: int | None = None
        self._guard_model_seed: int | None = None
        self._guard_profile: str | None = None
        self._guard_initial_state_sha: str | None = None
        self._guard_xml_sha: str | None = None
        self._guard_camera_names: list[str] = []
        self._guard_control_frequency_hz: int | None = None

    # -- real input context (set by the runner, never guessed) ----------------

    def arm_input_capture(
        self,
        session_id: Any,
        state_index: Any,
        model_seed: Any,
        profile: Any,
        initial_state_sha: Any = None,
        xml_sha: Any = None,
        camera_names: Any = None,
        control_frequency_hz: Any = None,
    ) -> None:
        self._guard_session_id = session_id
        self._guard_state_index = state_index
        self._guard_model_seed = model_seed
        self._guard_profile = profile
        if initial_state_sha is not None:
            self._guard_initial_state_sha = initial_state_sha
        if xml_sha is not None:
            self._guard_xml_sha = xml_sha
        if camera_names is not None:
            self._guard_camera_names = [str(name) for name in camera_names]
        if control_frequency_hz is not None:
            self._guard_control_frequency_hz = control_frequency_hz

    def _session_record(self) -> Any:
        session_id = self._guard_session_id
        sessions = getattr(self, "_sessions", None)
        if session_id is None or not isinstance(sessions, dict):
            return None
        lock = getattr(self, "_lock", None)
        if lock is not None:
            try:
                with lock:
                    return sessions.get(session_id)
            except Exception:  # noqa: BLE001
                pass
        try:
            return sessions.get(session_id)
        except Exception:  # noqa: BLE001
            return None

    def _build_input_evidence(self, fingerprint: dict) -> dict:
        """The evidence record matching ``paired._input_record``'s schema.

        The physical session facts come from the real session record and the
        worker-read control frequency; a value that is genuinely unavailable stays
        ``None`` (it is never fabricated).
        """

        record = self._session_record()
        initial_sha = self._guard_initial_state_sha
        if initial_sha is None and record is not None:
            initial_sha = getattr(record, "initial_state_hash", None)
        xml_sha = self._guard_xml_sha
        if xml_sha is None and record is not None:
            xml_sha = getattr(record, "xml_sha", None)
        camera_names = list(self._guard_camera_names or [])
        if not camera_names and record is not None:
            camera_names = list(getattr(record, "camera_names", None) or [])
        frequency = self._guard_control_frequency_hz
        if frequency is None:
            frequency = _control_frequency_hz(getattr(self, "_env", None))
        return {
            "state_index": self._guard_state_index,
            "model_seed": self._guard_model_seed,
            "profile": self._guard_profile,
            "initial_state_sha": initial_sha,
            "xml_sha": xml_sha,
            "camera_names": camera_names,
            "control_frequency_hz": frequency,
            "fingerprint": fingerprint,
        }

    # -- action selection (worker thread only) --------------------------------

    def _select_action(self, batch: Any) -> Any:
        """Count, fingerprint the first batch exactly once, then delegate."""

        self.action_selection_count += 1
        if not self._first_input_captured:
            self._first_input_captured = True
            fingerprint = paired.fingerprint_batch(batch)
            self.first_fingerprint = fingerprint
            self.first_input_evidence = self._build_input_evidence(fingerprint)
        # The batch is never mutated and the returned action object is the real
        # one the inherited implementation produced.
        return super()._select_action(batch)


# --- service construction and readiness --------------------------------------


def _build_service(args: argparse.Namespace) -> GuardValidationService:
    """Construct the service with the exact required keyword arguments."""

    return GuardValidationService(
        model_path=service.DEFAULT_MODEL_PATH,
        run_root=str(args.run_root),
        completion_mode=COMPLETION_MODE,
        grasp_guard_mode=str(args.guard_mode),
    )


def _wait_ready(service_: Any, timeout_s: float = READY_TIMEOUT_S) -> dict[str, Any]:
    """Poll the existing health endpoint until ready / errored / timed out."""

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        try:
            health = service_.health()
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


# --- report skeleton ---------------------------------------------------------


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "ok": False,
        "fatal_error": None,
        "metadata": {
            "output": str(args.output),
            "run_root": str(args.run_root),
            "scene": args.scene,
            "capabilities": args.capabilities,
            "capability_ids": list(_capability_ids(args.capabilities)),
            "guard_mode": args.guard_mode,
            "state_index": int(args.state_index),
            "model_seed": int(args.model_seed),
            "budget": BUDGET,
            "timeout_s": TIMEOUT_S,
            "profile": PROFILE,
            "completion_mode": COMPLETION_MODE,
            "source_git_sha": args.source_git_sha,
            "model_path": service.DEFAULT_MODEL_PATH,
            "model_revision": service.MODEL_REVISION_DEFAULT,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "source_sha256": _source_sha256(),
            "checkpoint_file": CHECKPOINT_FILENAME,
            "checkpoint_sha256": _checkpoint_sha256(service.DEFAULT_MODEL_PATH),
            "assisted": False,
            "hermes_calls": 0,
        },
        "service": {},
        "session": {
            "session_id": None,
            "scene_id": args.scene,
            "seed": int(args.state_index),
            "init_state_index": int(args.state_index),
            "initial_state_sha": None,
            "xml_sha": None,
            "camera_names": [],
            "control_frequency_hz": None,
        },
        "model_rng_seed": None,
        "profile": None,
        "plan": None,
        "jobs": [],
        "declared_capabilities": [],
        "executed_capabilities": [],
        "skipped_capabilities": [],
        "first_fingerprint": None,
        "first_input": None,
        "input_evidence": None,
        "action_selection_count": None,
        "freeze": None,
        "operational_errors": [],
        "physical_failures": [],
        "null_metrics": [],
        "timings": {},
    }


# --- the single-run validation -----------------------------------------------


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Run exactly one plan through one fresh session, then write the report.

    Always returns the report dict (and always attempts the atomic write).  Any
    unexpected failure is preserved verbatim in ``fatal_error`` with ``ok=False``.
    """

    started = time.monotonic()
    output_path = Path(args.output)
    run_root = Path(args.run_root)
    report = _base_report(args)
    capabilities = _capability_ids(args.capabilities)
    report["declared_capabilities"] = list(capabilities)

    fatal_error: str | None = None
    svc: Any = None
    try:
        run_root.mkdir(parents=True, exist_ok=True)

        with wd.register_native_wine_scene():
            svc = _build_service(args)
            svc.start()

            readiness = _wait_ready(svc, READY_TIMEOUT_S)
            report["service"] = {
                "ready": bool(readiness.get("ready")),
                "worker_error": readiness.get("worker_error"),
                "model_revision": readiness.get("model_revision"),
            }
            if not readiness.get("ready"):
                raise RuntimeError(
                    "service not ready: %s"
                    % (readiness.get("worker_error") or "timeout after %.0fs" % READY_TIMEOUT_S)
                )

            # Exactly one session; seed == init_state_index == --state-index.
            session = svc.create_session(
                args.scene, seed=int(args.state_index), init_state_index=int(args.state_index)
            )
            if not session.get("ok"):
                raise RuntimeError("create_session: %s" % (session,))
            session_id = session["session_id"]
            report["session"]["session_id"] = session_id

            record = svc._sessions.get(session_id)
            initial_state_sha = getattr(record, "initial_state_hash", None)
            xml_sha = getattr(record, "xml_sha", None)
            camera_names = list(getattr(record, "camera_names", None) or [])
            report["session"]["initial_state_sha"] = initial_state_sha
            report["session"]["xml_sha"] = xml_sha
            report["session"]["camera_names"] = camera_names

            # Seed the model RNG only AFTER the env exists; the env is never reset
            # and the RNG is never reseeded again.  The complete raw seeding
            # result is preserved verbatim; an unseeded model RNG is a fatal
            # operational failure raised BEFORE the profile is configured or any
            # plan is submitted.
            seeded = paired._seed_model_rng(svc, int(args.model_seed))
            report["model_rng_seed"] = {
                "requested": int(args.model_seed),
                "result": seeded,
                "seeded": seeded.get("seeded") if isinstance(seeded, dict) else None,
                "reason": seeded.get("reason") if isinstance(seeded, dict) else None,
            }
            if (
                not isinstance(seeded, dict)
                or seeded.get("ok") is not True
                or seeded.get("seeded") is not True
            ):
                raise RuntimeError("model_rng_seed: %s" % (seeded,))

            frequency = paired._read_control_frequency(svc)
            control_frequency_hz = (
                frequency.get("control_frequency_hz") if frequency.get("ok") else None
            )
            report["session"]["control_frequency_hz"] = control_frequency_hz

            svc.arm_input_capture(
                session_id,
                int(args.state_index),
                int(args.model_seed),
                PROFILE,
                initial_state_sha=initial_state_sha,
                xml_sha=xml_sha,
                camera_names=camera_names,
                control_frequency_hz=control_frequency_hz,
            )

            # The profile is configured AFTER the RNG seed and read back exactly.
            profile_result = svc.configure_profile(PROFILE)
            report["profile"] = profile_result
            if not _profile_readback_ok(profile_result):
                raise RuntimeError("profile_config: %s" % (profile_result,))

            request_id = "%s-plan1" % uuid.uuid4().hex
            plan_record = pe._submit_and_wait(
                svc,
                session_id,
                list(capabilities),
                BUDGET,
                False,
                request_id,
                TIMEOUT_S,
                RATIONALE,
            )
            collected = _collect_plan_result(svc, plan_record, capabilities)
            report["plan"] = collected["plan"]
            report["jobs"] = collected["jobs"]
            report["executed_capabilities"] = collected["executed_capabilities"]
            report["skipped_capabilities"] = collected["skipped_capabilities"]
            report["operational_errors"].extend(collected["operational_errors"])
            report["physical_failures"].extend(collected["physical_failures"])

            # Two read-only freeze probes (no step/reset/forward/predict/assess).
            # The full freeze evidence is preserved; a non-dict result or a
            # non-True ``equal`` is an *operational* validation failure (never a
            # physical task outcome, and never a fabricated success).
            freeze = _run_freeze_probes(svc, session_id)
            report["freeze"] = freeze
            if not isinstance(freeze, dict) or freeze.get("equal") is not True:
                report["operational_errors"].append("freeze_probe_invalid_or_changed")

            report["first_fingerprint"] = getattr(svc, "first_fingerprint", None)
            report["action_selection_count"] = getattr(svc, "action_selection_count", None)
            evidence = getattr(svc, "first_input_evidence", None)
            report["input_evidence"] = evidence if isinstance(evidence, dict) else None
            report["first_input"] = (
                paired._input_record(evidence) if isinstance(evidence, dict) else None
            )
    except BaseException as exc:  # noqa: BLE001 - preserve the real failure verbatim
        fatal_error = _format_exc(exc)
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

    report["fatal_error"] = fatal_error
    report["ok"] = fatal_error is None and not report["operational_errors"]

    # Null metrics: an unavailable value is recorded as unknown, never fabricated.
    nulls: list[str] = []
    if report["session"].get("initial_state_sha") is None:
        nulls.append("initial_state_sha")
    if report["session"].get("xml_sha") is None:
        nulls.append("xml_sha")
    if not report["session"].get("camera_names"):
        nulls.append("camera_names")
    if report.get("first_input") is None:
        nulls.append("first_input")
    if report.get("action_selection_count") in (None, 0):
        nulls.append("action_selection_count")
    if isinstance(report.get("plan"), dict) and report["plan"].get("plan_success") is None:
        nulls.append("plan_success")
    report["null_metrics"] = nulls

    report["timings"] = {"validation_wall_s": round(time.monotonic() - started, 3)}

    try:
        pe._write_json_atomic(output_path, report)
    except Exception as exc:  # noqa: BLE001 - a report write failure is fatal
        report["fatal_error"] = _format_exc(exc)
        report["ok"] = False
    return report


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated live grasp-guard validation runner: one fresh session, one "
            "fixed plan, then two read-only freeze probes.  No training, downloads, "
            "forced release, teleport or HTTP server."
        )
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
        help="absolute fresh run root for per-job artifacts (must not exist; created)",
    )
    parser.add_argument(
        "--scene",
        required=True,
        type=str,
        choices=list(SCENE_CHOICES),
        help="exact catalog scene id (wine_native_goal9 or goal_table)",
    )
    parser.add_argument("--state-index", required=True, type=int, help="fresh session seed/index")
    parser.add_argument("--model-seed", required=True, type=int, help="worker model RNG seed")
    parser.add_argument(
        "--guard-mode",
        required=True,
        type=str,
        choices=list(GUARD_MODES),
        help="wine-only grasp guard: off or enforce",
    )
    parser.add_argument(
        "--capabilities",
        required=True,
        type=str,
        choices=list(CAPABILITY_CHOICES),
        help="wine (wine_to_rack) or wine_bowl (wine_to_rack + bowl_to_plate)",
    )
    parser.add_argument(
        "--source-git-sha",
        required=True,
        type=str,
        help="the source git SHA to record in the report metadata",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isabs(str(args.output)):
        parser.error("--output must be an absolute path")
    if not os.path.isabs(str(args.run_root)):
        parser.error("--run-root must be an absolute path")
    if Path(str(args.output)).exists():
        parser.error("--output already exists; refusing to overwrite %s" % args.output)
    if Path(str(args.run_root)).exists():
        parser.error("--run-root already exists; refusing to reuse %s" % args.run_root)
    if args.scene not in SCENE_CHOICES:
        parser.error("--scene must be one of %s" % (list(SCENE_CHOICES),))
    if args.guard_mode not in GUARD_MODES:
        parser.error("--guard-mode must be one of %s" % (list(GUARD_MODES),))
    if args.capabilities not in CAPABILITY_PLANS:
        parser.error("--capabilities must be one of %s" % (list(CAPABILITY_CHOICES),))
    if (
        isinstance(args.state_index, bool)
        or not isinstance(args.state_index, int)
        or args.state_index < 0
    ):
        parser.error("--state-index must be a non-negative integer")
    if isinstance(args.model_seed, bool) or not isinstance(args.model_seed, int):
        parser.error("--model-seed must be an integer")
    if args.scene == wd.WINE_SCENE_ID and args.capabilities != CAPABILITIES_WINE:
        parser.error(
            "scene %r requires --capabilities %s (it has no bowl capability)"
            % (args.scene, CAPABILITIES_WINE)
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    report = run_validation(args)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
