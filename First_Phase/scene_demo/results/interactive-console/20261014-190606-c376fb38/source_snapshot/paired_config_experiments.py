#!/usr/bin/env python3
"""Isolated paired-state wine configuration experiment runner (read-only host).

This module hosts :class:`PairedWineDiagnosticService` -- a subclass of the real
``wine_diagnostics.WineDiagnosticService`` (itself a subclass of
``placement_experiments.DiagnosticService`` and of the actual
``scene_demo/service.py`` ``SceneService``) -- plus a small campaign runner that
drives the native LIBERO ``libero_goal/9`` wine scene through a *paired-state*
precision / action-chunk configuration grid.

Nothing here retrains, downloads, resets the environment between subgoals, calls
Hermes, opens a socket, forces a release or teleports an object.  The base service
lifecycle (one worker, one ``strict=True`` model load, the release-verified
completion gate, native termination, ``events.jsonl`` / PNG / MP4, the wine
telemetry and its raw/post/sent audits) is inherited verbatim.

Paired-state design
===================

Every trial creates a *fresh* session whose environment seed AND init-state index
both equal the state index, then seeds the model RNG to the trial's model seed
*after* the environment exists and never resets it again.  Within one state the
first preprocessed policy-input batch is fingerprinted (two camera tensors, the
``observation.state`` tensor and the exact task text -- exact shape, dtype and the
SHA-256 of each raw byte payload, including BF16 bytes hashed without widening)
and validated against the first record for that state: a missing/incomplete input
or any mismatch raises *before* any prediction or ``env.step`` and aborts the
campaign.  A different model RNG may not change the initial environment or the
first input.

Each discovery state also carries one A/A control duplicate of
state 0 / model seed 0 / ``baseline_bf16``; it is compared by the *actual*
seven-dimensional sent actions (lengths, exact-match count, maximum absolute
difference, terminal stop result and first inputs) as **repeatability** evidence,
never as a claim of bitwise CUDA determinism.

Independent shadow observation
==============================

:class:`PassiveStepObserver` installs a ``finally``-restored, read-only wrapper
around this job's worker-owned ``env.step``.  It calls the saved original step
*exactly once* with the unchanged action object and returns the *same* result
object, and only then invokes :class:`GuardSemanticObserver`, which starts a fresh
``grasp_guard.GraspMonitor`` before the first action, reads
``grasp_guard.read_probe`` and ``wine_semantic.read_wine_semantic`` after every
real action and appends one sample to ``guard_semantic.jsonl``.  The observer never
feeds the policy and never changes live termination, success or error; the
production guard stays ``off``.

Post-policy assessment
======================

Only after a *normal terminal* policy job, the worker executes exactly
``ASSESSMENT_ACTION_COUNT`` (20) ``float32`` seven-dimensional zero actions through
the real ``env.step`` (a neutral gripper command; never a forced open/close).  Each
sample's physics snapshot, semantic read and native flag are appended to
``assessment.jsonl`` and fed to the *same* ``SemanticTracker``.  Every row records
the exact nullable tracked status (``None`` stays ``None``); the final
``semantic_after`` / ``semantic_success`` is the *last* sample's status -- never
``any(samples)`` -- so a completed success followed by a failing end becomes
``False``.  The assessment is a real evaluation action (it can affect settling and
is included in the wall time) but its outcome is never promoted into the policy
benchmark success.

Grasp-proxy and semantic limitations
====================================

The grasp signal is the robosuite contact-geom proxy
``inner._check_grasp(gripper, obj.contact_geoms)`` -- a contact/kinematic screening
proxy, not a tactile or force measurement -- and the semantic reader is an
optional, concurrently arriving ``wine_semantic`` module.  A semantic success is
the *exact* tri-state status of the real ``SemanticTracker``: ``_semantic_flag``
reads only a genuine bool or an explicit bool ``semantic_success`` (the
``update`` / ``tracker.semantic_success`` value) and returns ``None`` for anything
else, so a raw ``semantic_candidate`` is never a completed status and an unknown
value is never turned into success: an absent semantic module yields ``None``, and
no success rate is inferred from the small pre-declared grid.

Only stdlib + numpy are imported at module import time; ``torch`` is imported
lazily by the pinned base :meth:`SceneService._select_action`, so ``--help`` and
the GPU-free unit tests never initialise CUDA or touch the network.
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

import catalog  # noqa: E402
import grasp_guard  # noqa: E402
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402

EXPERIMENT_NAME = "paired_config_experiments"
CONTROL_FREQUENCY_HZ = 20
ACTION_DIM = 7
FIXED_SOURCE_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
AUDITED_CHECKPOINT_FILE_COUNT = 7
ASSESSMENT_ACTION_COUNT = 20
READY_TIMEOUT_S = 120.0

# The exact, fixed wine oracle goal set and instruction (written literally here,
# never derived from the submitted/executed capability schedule).
WINE_ORACLE_GOALS: list[list[str]] = [["on", "wine_bottle_1", "wine_rack_1_top_region"]]
WINE_GOAL_KEY = catalog.goal_key(WINE_ORACLE_GOALS[0])
WINE_OBJECT_ID = "wine_bottle_1"
WINE_INSTRUCTION = "put the wine bottle on the rack"
WINE_CAPABILITY_ID = "wine_to_rack"
WINE_SCENE_ID = wd.WINE_SCENE_ID
COMPLETION_MODE = wd.COMPLETION_MODE
GUARD_MODE = "off"

PROFILE_A = "baseline_bf16"
PROFILE_B = "fp32"
PROFILE_C = "fp32_h5"
PROFILE_ORDER = (PROFILE_A, PROFILE_B, PROFILE_C)
SELECTABLE_PROFILES = (PROFILE_A, PROFILE_B, PROFILE_C)
# Predeclared exact-tie order: baseline_bf16 > fp32 > fp32_h5.
PROFILE_TIE_ORDER = {PROFILE_A: 0, PROFILE_B: 1, PROFILE_C: 2}
DISCOVERY_STATES = (0, 1, 2)
HOLDOUT_STATES = (3, 4, 5)
MODEL_SEED_OFFSETS = (0, 1000)

_SELECTION_RULE = {
    "primary": "highest count of strict POLICY successes across the 18 non-control discovery trials",
    "secondary": "highest count of post-assessment semantic successes",
    "tertiary": "lowest median policy wall time (seconds)",
    "exact_tie_order": [PROFILE_A, PROFILE_B, PROFILE_C],
    "unknown_is_success": False,
    "rate_inference": False,
    "note": (
        "explicit counts only; a three-state/six-trial grid is not a statistical "
        "reliability claim and no success rate is inferred"
    ),
}

_ASSESSMENT_PROTOCOL = {
    "action_count": ASSESSMENT_ACTION_COUNT,
    "action_shape": [ACTION_DIM],
    "action_dtype": "float32",
    "action_value": "zeros (neutral gripper command; never forced open/close)",
    "executor": "worker_thread_env_step",
    "ordering": "after a normal terminal policy job only",
    "semantic_tracker": "the same SemanticTracker instance as the policy job",
    "benchmark_effect": (
        "assessment physics is a real evaluation action, not a VLA action and not a "
        "read-only replay; it can affect settling and is included in the total wall "
        "time, but its outcome is NEVER promoted into the policy benchmark success"
    ),
    "operational_or_cancelled": "no assessment; the campaign stops",
}

_SOURCE_FILES = (
    "paired_config_experiments.py",
    "wine_diagnostics.py",
    "placement_experiments.py",
    "placement_completion.py",
    "service.py",
    "catalog.py",
    "grasp_guard.py",
    "wine_semantic.py",
)


# --- small helpers -----------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _progress(message: str) -> None:
    sys.stderr.write("[paired-config %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _median(values: Any) -> float | None:
    vals = sorted(
        float(value)
        for value in (values or [])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the exact bytes of every pinned module this runner depends on."""

    digests: dict[str, str | None] = {}
    for name in _SOURCE_FILES:
        try:
            digests[name] = _sha256_hex((_HERE / name).read_bytes())
        except Exception:  # noqa: BLE001 - absent/unreadable is recorded as unknown
            digests[name] = None
    return digests


def _audit_checkpoint_files(model_path: str) -> dict[str, Any]:
    """Audit the actual checkpoint root against the seven-file audit contract.

    The audit contract is deliberately *measured*, never guessed: it is the set of
    regular (non-hidden) files that actually exist in the checkpoint root.  The
    contract declares that this set must contain exactly
    ``AUDITED_CHECKPOINT_FILE_COUNT`` files; the exact size and SHA-256 of every
    one of those files is recorded.  A missing directory, an empty directory or a
    file count other than seven is an *operational* failure (``ok`` is False) and
    is never success -- it must stop the campaign before any action runs.
    """

    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return {
            "ok": False,
            "reason": "model path is missing or is not a directory: %s" % model_path,
            "model_path": str(path),
            "files": [],
            "expected_file_count": AUDITED_CHECKPOINT_FILE_COUNT,
        }
    try:
        entries = sorted(
            entry
            for entry in path.iterdir()
            if entry.is_file() and not entry.name.startswith(".")
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reason": "cannot list checkpoint root: %s" % exc,
            "model_path": str(path),
            "files": [],
            "expected_file_count": AUDITED_CHECKPOINT_FILE_COUNT,
        }
    files: list[dict[str, Any]] = []
    for entry in entries:
        try:
            size = entry.stat().st_size
            digest: str | None = _sha256_hex(entry.read_bytes())
            error = None
        except Exception as exc:  # noqa: BLE001 - an unreadable file is recorded, not hidden
            size = None
            digest = None
            error = _format_exc(exc)
        files.append({"name": entry.name, "size": size, "sha256": digest, "error": error})
    ok = len(files) == AUDITED_CHECKPOINT_FILE_COUNT and all(
        isinstance(entry.get("sha256"), str) for entry in files
    )
    reason = None
    if not ok:
        reason = "expected exactly %d audited checkpoint files, found %d" % (
            AUDITED_CHECKPOINT_FILE_COUNT,
            len(files),
        )
    return {
        "ok": ok,
        "reason": reason,
        "model_path": str(path),
        "expected_file_count": AUDITED_CHECKPOINT_FILE_COUNT,
        "files": files,
    }


# --- input batch fingerprint (pure, copy-only) -------------------------------


def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:  # noqa: BLE001
        return None


def _dtype_str(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)


def _raw_bytes(value: Any) -> bytes | None:
    """The exact raw bytes of ``value`` without mutating or numerically widening it.

    ``detach``/``cpu`` are applied to a detached copy only.  A NumPy array (and a
    ``.tobytes`` duck-typed tensor, such as a real BF16 tensor viewed as raw
    bytes) is hashed verbatim; a torch tensor is viewed as ``uint8`` so a BF16
    payload keeps its true two-byte-per-element bytes.  Every failure returns
    ``None`` ("unknown"), never a fabricated digest.
    """

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).tobytes()
    obj = value
    for name in ("detach", "cpu"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                obj = fn()
            except Exception:  # noqa: BLE001 - keep the current object
                pass
    # torch raw view (preserves BF16 bytes exactly).
    view = getattr(obj, "view", None)
    if callable(view):
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                raw = view(torch.uint8)
                as_np = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
                return np.ascontiguousarray(np.asarray(as_np)).tobytes()
        except Exception:  # noqa: BLE001 - not a torch tensor / torch unavailable
            pass
    # NumPy-compatible (``.numpy()``) objects.
    np_fn = getattr(obj, "numpy", None)
    if callable(np_fn):
        try:
            arr = np.asarray(np_fn())
            if arr.dtype != object:
                return np.ascontiguousarray(arr).tobytes()
        except Exception:  # noqa: BLE001 - e.g. a BF16 tensor numpy() rejects
            pass
    # Any duck-typed tensor exposing exact bytes.
    tobytes = getattr(obj, "tobytes", None)
    if callable(tobytes):
        try:
            raw = tobytes()
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw)
        except Exception:  # noqa: BLE001
            pass
    try:
        arr = np.asarray(obj)
        if arr.dtype != object:
            return np.ascontiguousarray(arr).tobytes()
    except Exception:  # noqa: BLE001
        pass
    return None


def _describe_tensor(key: str, value: Any) -> dict[str, Any]:
    raw = _raw_bytes(value)
    return {
        "key": key,
        "present": value is not None,
        "shape": _shape_list(value),
        "dtype": _dtype_str(value),
        "raw_sha256": _sha256_hex(raw) if raw is not None else None,
        "raw_nbytes": len(raw) if raw is not None else None,
    }


_CAMERA_PREFIX = "observation.images"


def fingerprint_batch(batch: dict) -> dict:
    """Pure fingerprint of one already-preprocessed policy-input batch.

    The batch is expected to carry two camera tensors/arrays (keys beginning with
    ``observation.images``) and an ``observation.state`` tensor plus the exact
    task text.  For every tensor the exact shape, dtype string and the SHA-256 of
    its *raw bytes* are recorded, together with the exact task text and a
    canonical combined SHA-256 over the whole structure.  The function never
    mutates ``batch`` and never invents a value: a missing or unreadable tensor
    keeps a ``None`` digest.
    """

    result: dict[str, Any] = {
        "is_mapping": isinstance(batch, dict),
        "keys": None,
        "cameras": [],
        "camera_keys": [],
        "state": None,
        "task": None,
        "task_sha256": None,
        "combined_sha256": None,
        "errors": [],
    }
    if not isinstance(batch, dict):
        result["errors"].append("batch is not a mapping")
        return result

    result["keys"] = sorted(str(key) for key in batch.keys())
    camera_keys = sorted(
        key
        for key in batch.keys()
        if isinstance(key, str) and key.startswith(_CAMERA_PREFIX)
    )
    result["camera_keys"] = list(camera_keys)
    result["cameras"] = [_describe_tensor(key, batch.get(key)) for key in camera_keys]

    state = batch.get("observation.state")
    if state is None:
        observation = batch.get("observation")
        if isinstance(observation, dict):
            state = observation.get("state")
    result["state"] = _describe_tensor("observation.state", state)

    task = wd._batch_task_text(batch)
    result["task"] = task
    if task is not None:
        result["task_sha256"] = _sha256_hex(json.dumps(task, sort_keys=True).encode("utf-8"))

    canonical = {
        "cameras": [
            {
                "key": entry["key"],
                "shape": entry["shape"],
                "dtype": entry["dtype"],
                "raw_sha256": entry["raw_sha256"],
            }
            for entry in result["cameras"]
        ],
        "state": {
            "shape": result["state"]["shape"],
            "dtype": result["state"]["dtype"],
            "raw_sha256": result["state"]["raw_sha256"],
        },
        "task": task,
    }
    result["combined_sha256"] = _sha256_hex(
        json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    )

    if len(result["cameras"]) < 2:
        result["errors"].append("fewer than two camera tensors")
    if any(entry["raw_sha256"] is None for entry in result["cameras"]):
        result["errors"].append("a camera tensor has no readable raw bytes")
    if not result["state"]["present"]:
        result["errors"].append("observation.state missing")
    elif result["state"]["raw_sha256"] is None:
        result["errors"].append("observation.state has no readable raw bytes")
    if task is None:
        result["errors"].append("task text missing")
    return result


def fingerprint_is_complete(fingerprint: Any) -> bool:
    """Whether a fingerprint carries every required digest (never guesses)."""

    if not isinstance(fingerprint, dict):
        return False
    if fingerprint.get("errors"):
        return False
    cameras = fingerprint.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 2:
        return False
    if any(not isinstance(entry, dict) or entry.get("raw_sha256") is None for entry in cameras):
        return False
    state = fingerprint.get("state")
    if not isinstance(state, dict) or state.get("raw_sha256") is None:
        return False
    if fingerprint.get("task") is None or fingerprint.get("task_sha256") is None:
        return False
    return fingerprint.get("combined_sha256") is not None


def _input_record(evidence: dict) -> dict[str, Any]:
    """The comparable fields of one captured first-input evidence record."""

    fingerprint = evidence.get("fingerprint") or {}
    cameras = {
        str(entry.get("key")): entry.get("raw_sha256")
        for entry in (fingerprint.get("cameras") or [])
        if isinstance(entry, dict)
    }
    state = fingerprint.get("state") or {}
    return {
        "state_index": evidence.get("state_index"),
        "initial_state_sha": evidence.get("initial_state_sha"),
        "xml_sha": evidence.get("xml_sha"),
        "camera_names": list(evidence.get("camera_names") or []),
        "control_frequency_hz": evidence.get("control_frequency_hz"),
        "camera_raw_sha256": cameras,
        "state_raw_sha256": state.get("raw_sha256") if isinstance(state, dict) else None,
        "state_shape": state.get("shape") if isinstance(state, dict) else None,
        "state_dtype": state.get("dtype") if isinstance(state, dict) else None,
        "task": fingerprint.get("task"),
        "task_sha256": fingerprint.get("task_sha256"),
        "combined_sha256": fingerprint.get("combined_sha256"),
    }


def compare_inputs(reference: dict, observed: dict) -> list[str]:
    """The list of mismatched comparable fields between two input records."""

    mismatches: list[str] = []
    for field in (
        "initial_state_sha",
        "xml_sha",
        "camera_names",
        "control_frequency_hz",
        "state_raw_sha256",
        "state_shape",
        "state_dtype",
        "task",
        "task_sha256",
        "combined_sha256",
    ):
        if reference.get(field) != observed.get(field):
            mismatches.append("%s: %r != %r" % (field, reference.get(field), observed.get(field)))
    ref_cameras = reference.get("camera_raw_sha256") or {}
    obs_cameras = observed.get("camera_raw_sha256") or {}
    if set(ref_cameras) != set(obs_cameras):
        mismatches.append("camera keys differ: %r != %r" % (sorted(ref_cameras), sorted(obs_cameras)))
    else:
        for key in sorted(ref_cameras):
            if ref_cameras.get(key) != obs_cameras.get(key):
                mismatches.append("camera %s raw sha differs" % key)
    return mismatches


def validate_input(registry: dict, evidence: dict) -> None:
    """Validate one first-input evidence record against the same-state registry.

    Missing/incomplete input or any mismatch with the first record for the same
    state raises ``RuntimeError`` *before* any prediction or ``env.step`` for the
    offending trial, so the caller aborts the campaign.
    """

    fingerprint = evidence.get("fingerprint") or {}
    if not fingerprint_is_complete(fingerprint):
        message = "paired_input_incomplete: state=%r errors=%r" % (
            evidence.get("state_index"),
            fingerprint.get("errors"),
        )
        registry.setdefault("errors", []).append(message)
        raise RuntimeError(message)
    state_index = evidence.get("state_index")
    observed = _input_record(evidence)
    reference = registry.get(state_index)
    if reference is None:
        registry[state_index] = observed
        return
    mismatches = compare_inputs(reference, observed)
    if mismatches:
        message = "paired_input_mismatch: state=%r %s" % (state_index, "; ".join(mismatches))
        registry.setdefault("errors", []).append(message)
        raise RuntimeError(message)


# --- semantic helpers (the arriving module may be absent) --------------------


def load_semantic_module() -> Any:
    """Import ``wine_semantic`` on demand; return ``None`` when it is absent."""

    try:
        import importlib

        return importlib.import_module("wine_semantic")
    except Exception:  # noqa: BLE001 - a concurrently arriving module may be missing
        return None


def make_semantic_tracker() -> Any:
    """A fresh ``SemanticTracker`` when available, else ``None``."""

    module = load_semantic_module()
    if module is None:
        return None
    tracker_cls = getattr(module, "SemanticTracker", None)
    if tracker_cls is None:
        return None
    try:
        return tracker_cls()
    except Exception:  # noqa: BLE001
        return None


def read_semantic(env: Any) -> Any:
    """Read one semantic candidate; an absent/unreadable module is ``None``."""

    module = load_semantic_module()
    if module is None:
        return None
    fn = getattr(module, "read_wine_semantic", None)
    if not callable(fn):
        return None
    try:
        return fn(env)
    except Exception:  # noqa: BLE001 - unknown is never success
        return None


def tracker_update(tracker: Any, step: int, sample: Any) -> Any:
    """Feed one sample to the tracker with the exact ``update(sample)`` call.

    The real ``wine_semantic.SemanticTracker.update`` takes exactly one positional
    argument and returns its status mapping (``state`` / ``candidate_streak`` /
    ``semantic_success``).  ``step`` is retained only so this helper's existing
    signature -- and both of its callers -- stay unchanged; it is NEVER forwarded.
    ``update`` is called *exactly once*: no signature guessing, no introspection,
    no retry and no second call.  A missing tracker/``update`` or any raised
    exception yields ``None`` (unknown is never success).
    """

    if tracker is None:
        return None
    fn = getattr(tracker, "update", None)
    if not callable(fn):
        return None
    try:
        return fn(sample)
    except Exception:  # noqa: BLE001 - unknown is never success
        return None


def _semantic_flag(value: Any) -> bool | None:
    """A tri-state semantic flag from the exact SemanticTracker status.

    Returns a genuine ``True``/``False`` for a bool or for a mapping carrying an
    explicit bool ``semantic_success`` (the exact ``SemanticTracker.update`` return
    value and ``tracker.semantic_success``).  A missing key, a non-bool value, a
    different key (e.g. a raw ``semantic_candidate``) or any other object is
    *unknown* (``None``).  A raw semantic candidate is never a completed status.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        candidate = value.get("semantic_success")
        if isinstance(candidate, bool):
            return candidate
        return None
    return None


def semantic_is_success(value: Any) -> bool:
    """Whether a semantic value is an explicit completed success.

    Only an explicit ``True`` flag counts; ``None`` (unknown) and ``False`` are
    both never a success, so an unknown value is never promoted into success and a
    raw semantic candidate is never a completed status.
    """

    return _semantic_flag(value) is True


def _empty_probe(object_id: str, goal_key: str) -> dict:
    return {
        "objects": {object_id: {"position": None, "grasped": None}},
        "eef_position": None,
        "predicates": {goal_key: None},
        "gripper_qpos": None,
        "gap": None,
    }


def safe_probe(env: Any, object_id: str, goal_key: str) -> dict:
    """One read-only wine probe; a failed read stays fully unknown."""

    try:
        probe = grasp_guard.read_probe(env, object_id, goal_key)
    except Exception:  # noqa: BLE001 - an unavailable probe is never a False
        return _empty_probe(object_id, goal_key)
    if not isinstance(probe, dict):
        return _empty_probe(object_id, goal_key)
    return probe


# --- shadow guard/semantic observer ------------------------------------------


class GuardSemanticObserver:
    """Independent, read-only shadow observer (never changes live termination).

    ``start`` creates a fresh :class:`grasp_guard.GraspMonitor` and calls
    ``monitor.start`` on one read-only probe *before* the first action.  After
    every real action, :meth:`on_action` reads ``grasp_guard.read_probe`` (via
    :func:`safe_probe`), updates the monitor with the sent gripper command, reads
    ``wine_semantic.read_wine_semantic`` and feeds the same ``SemanticTracker``.
    Every sample is appended to ``guard_semantic.jsonl``.  The observer's output
    only records: it is never fed back to the policy and never affects the job's
    termination, success or error.
    """

    def __init__(
        self,
        env: Any,
        run_dir: Path,
        tracker: Any = None,
        object_id: str = WINE_OBJECT_ID,
        goal_key: str = WINE_GOAL_KEY,
    ) -> None:
        self.env = env
        self.run_dir = Path(run_dir)
        self.tracker = tracker
        self.object_id = object_id
        self.goal_key = goal_key
        self.monitor: Any = None
        self.path = self.run_dir / "guard_semantic.jsonl"
        self.initial_status: dict | None = None
        self.first_grasp_confirmed_step: int | None = None
        self.first_failed_grasp_step: int | None = None
        self.steps = 0
        self.semantic_known_count = 0
        self.semantic_unknown_count = 0
        self.last_semantic: Any = None
        self.last_semantic_status: Any = None
        self.error: str | None = None
        self._file: Any = None

    def start(self) -> "GuardSemanticObserver":
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - the observer never breaks the run
            self.error = _format_exc(exc)
            self._file = None
        try:
            self.monitor = grasp_guard.GraspMonitor()
            probe = safe_probe(self.env, self.object_id, self.goal_key)
            self.initial_status = self.monitor.start(probe)
            self._write({"phase": "start", "probe": probe, "monitor": self.initial_status})
        except Exception as exc:  # noqa: BLE001
            self.monitor = None
            self.error = self.error or _format_exc(exc)
        return self

    def on_action(self, action: Any, result: Any) -> dict | None:
        if self.monitor is None:
            return None
        self.steps += 1
        flattened = wd._flat_float_list(action)
        command = None
        if flattened and len(flattened) >= 1:
            command = flattened[-1]
        probe = safe_probe(self.env, self.object_id, self.goal_key)
        try:
            status = self.monitor.update(self.steps, probe, command)
        except Exception as exc:  # noqa: BLE001 - keep the real failure as evidence
            status = {"error": _format_exc(exc)}
        semantic = read_semantic(self.env)
        semantic_status = tracker_update(self.tracker, self.steps, semantic)
        # A raw semantic read is "known" if and ONLY if it is a genuine bool or a
        # dict whose ``semantic_candidate`` is *explicitly* a bool -- the actual raw
        # candidate knowledge, never the tracker's ``semantic_success`` completion.
        # A dict with a missing/``None`` ``semantic_candidate``, or any other type,
        # is an UNKNOWN semantic -- not known, not False, not success -- and the raw
        # value with its null fields is still recorded verbatim below.  A raw
        # ``True`` candidate stays a candidate: it is never converted into a tracked
        # completion.
        candidate = (
            semantic.get("semantic_candidate") if isinstance(semantic, dict) else None
        )
        if isinstance(semantic, bool) or isinstance(candidate, bool):
            self.semantic_known_count += 1
        else:
            self.semantic_unknown_count += 1
        self.last_semantic = semantic
        self.last_semantic_status = semantic_status
        if isinstance(status, dict):
            confirmed = status.get("grasp_confirmed_step")
            failed = status.get("failure_step")
            if confirmed is not None and self.first_grasp_confirmed_step is None:
                self.first_grasp_confirmed_step = confirmed
            if failed is not None and self.first_failed_grasp_step is None:
                self.first_failed_grasp_step = failed
        native_success = None
        if isinstance(result, tuple) and len(result) >= 5 and isinstance(result[4], dict):
            native_success = bool(result[4].get("is_success", False))
        sample = {
            "step": self.steps,
            "command": command,
            "probe": probe,
            "monitor": status,
            "semantic": semantic,
            "semantic_status": semantic_status,
            # The row status is the exact tracked tri-state from the adapter: a
            # missing/unknown status stays ``None`` (JSON null) and a valid
            # ``False``/``True`` status stays a bool.  The raw semantic (or a raw
            # ``semantic_candidate``) is never a fallback success.
            "semantic_success": _semantic_flag(semantic_status),
            "native_success": native_success,
        }
        self._write(sample)
        return sample

    def _write(self, record: dict) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        except Exception:  # noqa: BLE001 - telemetry must never break the run
            pass

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:  # noqa: BLE001
                pass
            self._file = None

    def summary(self) -> dict[str, Any]:
        return {
            "guard_semantic_path": str(self.path),
            "observer_only": True,
            "n_samples": self.steps,
            "initial_status": self.initial_status,
            "first_grasp_confirmed_step": self.first_grasp_confirmed_step,
            "first_failed_grasp_step": self.first_failed_grasp_step,
            "semantic_known_count": self.semantic_known_count,
            "semantic_unknown_count": self.semantic_unknown_count,
            "last_semantic": self.last_semantic,
            "last_semantic_status": self.last_semantic_status,
            "error": self.error,
        }


class PassiveStepObserver:
    """A ``finally``-restored, read-only wrapper around a worker-owned ``env.step``.

    ``install`` saves the current ``env.step`` and replaces it with
    :meth:`wrapped_step`; ``restore`` puts the saved bound method back.  The
    wrapped step calls the saved original step *exactly once* with the unchanged
    action object, returns the *same* result object, and only then invokes the
    observer's ``on_action`` hook (whose failure can never change the result).
    """

    def __init__(self, env: Any, observer: Any) -> None:
        self.env = env
        self.observer = observer
        self.original_step: Any = None
        self.installed = False
        self.counter = 0

    def wrapped_step(self, action: Any) -> Any:
        result = self.original_step(action)
        self.counter += 1
        try:
            self.observer.on_action(action, result)
        except Exception:  # noqa: BLE001 - observation never breaks the run
            pass
        return result

    def install(self) -> "PassiveStepObserver":
        self.original_step = self.env.step
        self.env.step = self.wrapped_step
        self.installed = True
        try:
            self.observer.start()
        except Exception:  # noqa: BLE001
            pass
        return self

    def restore(self) -> None:
        if self.installed and self.original_step is not None:
            try:
                self.env.step = self.original_step
            except Exception:  # noqa: BLE001
                pass
        self.installed = False

    def __enter__(self) -> "PassiveStepObserver":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.restore()
        return False


# --- the paired wine service -------------------------------------------------


class PairedWineDiagnosticService(wd.WineDiagnosticService):
    """Wine diagnostics + a paired-state first-input capture + a shadow observer.

    Everything is inherited from :class:`wine_diagnostics.WineDiagnosticService`
    (which itself extends ``placement_experiments.DiagnosticService`` and the real
    ``service.SceneService``): the single worker, the ``strict=True`` model load,
    the release-verified completion gate, the wine telemetry / raw+post+instruction
    capture, ``events.jsonl``, PNG and MP4 logic are untouched.  Only two methods
    are overridden:

    * :meth:`_select_action` -- captures the *first actual preprocessed batch* of
      each trial immediately BEFORE the policy prediction, fingerprints it and
      validates it against the same-state registry (raising before any prediction
      or ``env.step`` on a missing/mismatched input);
    * :meth:`_run_capability` -- installs a ``finally``-restored
      :class:`PassiveStepObserver` (which starts the shadow
      :class:`GuardSemanticObserver` before the first action) around this job's
      worker-owned ``env.step``, then delegates to the inherited implementation
      unchanged.  The observer only records: it never changes live termination,
      success or error.

    The production guard stays ``off`` (enforced by the constructor), so the
    service's own guard path is never armed; the independent observer uses its own
    :class:`grasp_guard.GraspMonitor` in shadow only.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The guard MUST be off for a paired configuration comparison.  Explicitly
        # reject any caller-requested non-off mode (a bare ``setdefault`` is
        # insufficient: a downstream ``enforce`` default could re-arm the guard and
        # change the live action/termination path), then pass the literal "off" to
        # the base so the base can never choose a different mode.
        requested_guard = kwargs.pop("grasp_guard_mode", None)
        if requested_guard is not None and requested_guard != GUARD_MODE:
            raise ValueError(
                "PairedWineDiagnosticService requires grasp_guard_mode=%r for a "
                "paired configuration comparison; refused non-off request %r"
                % (GUARD_MODE, requested_guard)
            )
        kwargs.setdefault("completion_mode", COMPLETION_MODE)
        kwargs["grasp_guard_mode"] = GUARD_MODE
        super().__init__(*args, **kwargs)
        self._paired_state_index: int | None = None
        self._paired_model_seed: int | None = None
        self._paired_profile: str | None = None
        self._paired_registry: dict | None = None
        self._paired_input_evidence: dict | None = None
        self._paired_initial_state_sha: str | None = None
        self._paired_xml_sha: str | None = None
        self._paired_camera_names: list[str] = []
        self._paired_control_frequency_hz: int | None = None
        self._paired_tracker: Any = None
        self._paired_observer_summary: dict | None = None

    # -- first-input capture (worker thread only) ------------------------------

    def _capture_paired_input(self, batch: Any) -> dict:
        """Fingerprint the first preprocessed batch and validate it.

        Called from :meth:`_select_action` immediately before the base prediction.
        Only the FIRST batch of a trial is fingerprinted and validated; a missing
        or incomplete fingerprint -- or any mismatch against the first record for
        the same env state -- raises ``RuntimeError`` here, i.e. *before* any
        prediction or ``env.step`` of the offending trial, so the campaign aborts.
        Once the first input is captured, its evidence is immutable and is returned
        verbatim for every later action without re-fingerprinting or re-validating
        the later batch.
        """

        existing = self._paired_input_evidence
        if existing is not None:
            # Already captured: return the existing worker-owned first-input
            # evidence immediately.  A later action's batch is NEVER fingerprinted
            # or validated again, so it can neither recompute nor overwrite the
            # initial evidence (and the initial worker-owned comparison stands).
            return existing
        fingerprint = fingerprint_batch(batch)
        evidence = {
            "state_index": self._paired_state_index,
            "model_seed": self._paired_model_seed,
            "profile": self._paired_profile,
            "initial_state_sha": self._paired_initial_state_sha,
            "xml_sha": self._paired_xml_sha,
            "camera_names": list(self._paired_camera_names),
            "control_frequency_hz": self._paired_control_frequency_hz,
            "fingerprint": fingerprint,
        }
        # A first mismatching/missing input raises here -- before any prediction or
        # env.step -- and leaves ``_paired_input_evidence`` unset (rejection stands).
        if self._paired_registry is not None:
            validate_input(self._paired_registry, evidence)
        self._paired_input_evidence = evidence
        return evidence

    def _select_action(self, batch: Any) -> np.ndarray:
        """Capture the first input, then run the inherited action selection."""

        self._capture_paired_input(batch)
        return super()._select_action(batch)

    # -- capability execution (worker thread only) ----------------------------

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

        tracker = self._paired_tracker
        if tracker is None:
            tracker = make_semantic_tracker()
            self._paired_tracker = tracker

        # The passive observer captures the TRUE original step first; the inherited
        # implementations then wrap on top of it, so exactly one physical step runs
        # per action and the observer sees the true result object.
        observer = GuardSemanticObserver(env, job.run_dir, tracker)
        step_observer = PassiveStepObserver(env, observer)
        step_observer.install()
        try:
            result = super()._run_capability(session, plan, job, capability_id)
        finally:
            step_observer.restore()
            observer.close()
            self._paired_observer_summary = observer.summary()
        return result


# --- worker-side helpers -----------------------------------------------------


def _seed_model_rng(service_: PairedWineDiagnosticService, seed: int) -> dict[str, Any]:
    """Seed the model RNG on the worker AFTER the env exists, without resetting it.

    The environment is never reset or rebuilt afterwards, so the same env seed
    yields the same initial state; only the *model* RNG differs between the two
    model-seed repeats of a state.
    """

    def _work() -> dict[str, Any]:
        try:
            import torch
        except Exception:  # noqa: BLE001 - no torch in a GPU-free interpreter
            return {"ok": True, "seeded": False, "reason": "torch unavailable"}
        torch.manual_seed(int(seed))
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:  # noqa: BLE001 - a CPU-only build has no CUDA RNG
            pass
        return {"ok": True, "seeded": True, "seed": int(seed)}

    return service_._sync_work("seed_model_rng", _work)


def _read_control_frequency(service_: PairedWineDiagnosticService) -> dict[str, Any]:
    """Read the live control frequency (Hz) without touching the simulator state."""

    def _work() -> dict[str, Any]:
        env = service_._env
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
            value = int(round(float(value))) if value is not None else None
        except (TypeError, ValueError):
            value = None
        return {"ok": True, "control_frequency_hz": value}

    return service_._sync_work("control_frequency", _work)


def _do_assessment_work(
    service_: PairedWineDiagnosticService,
    run_dir: Path,
    tracker: Any,
) -> dict[str, Any]:
    """Run the post-policy assessment on the worker: exactly 20 zero actions.

    Only reached after a normal terminal policy job.  Each action is a
    ``float32`` seven-dimensional zero vector (neutral gripper command; it never
    forces the gripper open or closed) executed through the real ``env.step``.
    Every sample's physics snapshot, semantic read and native flag are appended to
    ``assessment.jsonl`` and fed to the *same* ``SemanticTracker`` as the policy
    job.  The raw pre-/post-assessment samples are retained separately from the
    tracked statuses; the final ``semantic_after`` / ``semantic_success`` is the
    *last* sample's tracked status (never ``any(samples)``).  The assessment is a
    real evaluation action: it can affect settling and is included in the wall
    time, but its outcome is never promoted into the policy benchmark success.
    """

    env = service_._env
    result: dict[str, Any] = {
        "ok": False,
        "error": None,
        "action_count": 0,
        "expected_action_count": ASSESSMENT_ACTION_COUNT,
        "action_dtype": "float32",
        "action_shape": [ACTION_DIM],
        "path": None,
        "semantic_before": None,
        "semantic_after": None,
        "semantic_success": None,
        "semantic_before_sample": None,
        "semantic_after_sample": None,
        "native_success_final": None,
        "native_success_ever": None,
        "final_snapshot": None,
        "strict_final_score": None,
    }
    if env is None:
        result["error"] = "no live environment for the assessment"
        return result

    run_dir = Path(run_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    path = run_dir / "assessment.jsonl"
    result["path"] = str(path)

    handle = None
    try:
        handle = open(path, "w", encoding="utf-8")
    except Exception:  # noqa: BLE001 - a missing log never breaks the assessment
        handle = None

    snapshots: list[Any] = []
    natives: list[Any] = []
    # The raw pre-assessment sample is retained separately; the tracked *before*
    # status is the tracker's current value read WITHOUT any extra update.
    result["semantic_before_sample"] = read_semantic(env)
    result["semantic_before"] = _semantic_flag(getattr(tracker, "semantic_success", None))

    last_status: Any = None
    try:
        for index in range(ASSESSMENT_ACTION_COUNT):
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            step_result = env.step(action)
            result["action_count"] += 1
            native_success = None
            if isinstance(step_result, tuple) and len(step_result) >= 5:
                if step_result[0] is not None:
                    service_._last_obs = step_result[0]
                if isinstance(step_result[4], dict):
                    native_success = bool(step_result[4].get("is_success", False))
            try:
                snapshot = pe.capture_snapshot(env, WINE_ORACLE_GOALS)
            except Exception as exc:  # noqa: BLE001 - keep the real failure
                snapshot = {"error": _format_exc(exc)}
            semantic = read_semantic(env)
            semantic_status = tracker_update(tracker, index + 1, semantic)
            # The row status is the exact tracked tri-state (nullable): a missing
            # or unknown status stays ``None`` and is never coerced to a bool.
            semantic_success = _semantic_flag(semantic_status)
            snapshots.append(snapshot)
            natives.append(native_success)
            last_status = semantic_status
            record = {
                "step": index + 1,
                "action": [float(v) for v in action.tolist()],
                "action_dtype": "float32",
                "native_success": native_success,
                "snapshot": snapshot,
                "semantic": semantic,
                "semantic_status": semantic_status,
                "semantic_success": semantic_success,
            }
            if handle is not None:
                try:
                    handle.write(json.dumps(record, default=str) + "\n")
                    handle.flush()
                except Exception:  # noqa: BLE001 - telemetry must never break the run
                    pass
    except BaseException as exc:  # noqa: BLE001 - preserve the real failure
        result["error"] = _format_exc(exc)
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    # The raw post-assessment sample is retained separately; the final tracked
    # status is the LAST sample's status (never ``any(samples)``): a completed
    # success followed by a failing end becomes False.
    result["semantic_after_sample"] = read_semantic(env)
    result["semantic_after"] = _semantic_flag(
        last_status
        if last_status is not None
        else getattr(tracker, "semantic_success", None)
    )
    result["semantic_success"] = result["semantic_after"]
    try:
        final_snapshot = pe.capture_snapshot(env, WINE_ORACLE_GOALS)
    except Exception as exc:  # noqa: BLE001
        final_snapshot = {"error": _format_exc(exc)}
    result["final_snapshot"] = final_snapshot
    result["strict_final_score"] = pe.strict_final_score(
        final_snapshot if isinstance(final_snapshot, dict) else None,
        snapshots,
    )
    if natives:
        result["native_success_final"] = natives[-1]
        result["native_success_ever"] = any(value is True for value in natives)
    result["ok"] = result["error"] is None
    return result


def run_assessment(
    service_: PairedWineDiagnosticService, run_dir: Path, tracker: Any
) -> dict[str, Any]:
    """Queue the 20-action assessment on the worker and wait for its summary."""

    outcome = service_._sync_work(
        "assessment",
        lambda: _do_assessment_work(service_, run_dir, tracker),
        READY_TIMEOUT_S,
    )
    if not isinstance(outcome, dict):
        return {"ok": False, "error": "assessment returned a non-dict result"}
    if "action_count" not in outcome:
        # ``_sync_work`` returns an error envelope (``ok=False``) on worker failure.
        return {
            "ok": False,
            "error": outcome.get("detail") or outcome.get("reason") or repr(outcome),
            "action_count": 0,
        }
    return outcome


# --- one trial ---------------------------------------------------------------


def _job_is_normal(evidence: dict) -> bool:
    """Whether a job ended normally (not an operational error, not cancelled)."""

    if not evidence.get("available"):
        return False
    job = evidence.get("job") or {}
    state = job.get("state")
    ended = job.get("ended_reason")
    if state == "error" or ended == "error" or job.get("error"):
        return False
    if state == "cancelled" or ended == "cancelled":
        return False
    return state == "completed"


def _compact_job(evidence: dict) -> dict[str, Any]:
    if not evidence.get("available"):
        return {"job_id": evidence.get("job_id"), "available": False}
    job = evidence.get("job") or {}
    return {
        "job_id": evidence.get("job_id"),
        "available": True,
        "state": job.get("state"),
        "ended_reason": job.get("ended_reason"),
        "error": job.get("error"),
        "success": job.get("success"),
        "steps": job.get("steps"),
        "total_steps": job.get("total_steps"),
        "wall_s": job.get("wall_s"),
        "instruction": job.get("instruction"),
        "completion_mode": job.get("completion_mode"),
        "phase": job.get("phase"),
        "held_objects": job.get("held_objects"),
        "completion_ready": job.get("completion_ready"),
        "state_before_sha": job.get("state_before_sha"),
        "state_after_sha": job.get("state_after_sha"),
        "grasp_guard_mode": job.get("grasp_guard_mode"),
        "grasp_stage": job.get("grasp_stage"),
        "run_dir": job.get("run_dir"),
        "rollout_path": job.get("rollout_path"),
        "latest_png": job.get("latest_png"),
        "wine_telemetry_path": evidence.get("wine_telemetry_path"),
        "wine_diagnostic_path": evidence.get("wine_diagnostic_path"),
        "events_path": evidence.get("events_path"),
        "result_path": evidence.get("result_path"),
    }


def run_trial(
    service_: PairedWineDiagnosticService,
    *,
    phase: str,
    state_index: int,
    model_seed: int,
    profile: str,
    timeout: float = 900.0,
    budget: int = 300,
    model_revision: str | None = None,
    git_sha: str | None = None,
    registry: dict | None = None,
    is_control: bool = False,
    control_of: str | None = None,
) -> dict[str, Any]:
    """Run one paired trial: fresh session, one profile, one policy job, assessment.

    Every trial creates a *fresh* session whose env seed and init-state index both
    equal ``state_index``; the model RNG is seeded to ``model_seed`` after the env
    exists and the env is never reset afterwards.  The profile is configured and
    read back, the queues are reset, and only then is the single policy plan
    submitted.  A trial with an operational error or a cancellation receives NO
    assessment and stops the campaign; an ordinary (budget/task) failure does not.
    """

    started = time.monotonic()
    trial_key = "%s_s%d_m%d_%s_%s" % (
        phase,
        state_index,
        model_seed,
        profile,
        uuid.uuid4().hex[:8],
    )
    service_._diag_condition = wd.FINAL_ORACLE_KEY
    service_._paired_state_index = state_index
    service_._paired_model_seed = model_seed
    service_._paired_profile = profile
    service_._paired_registry = registry
    service_._paired_input_evidence = None
    service_._paired_observer_summary = None

    trial: dict[str, Any] = {
        "trial_id": trial_key,
        "phase": phase,
        "state_index": state_index,
        "envseed": state_index,
        "model_seed": model_seed,
        "profile": profile,
        "profile_readback": None,
        "guard_mode": GUARD_MODE,
        "is_control": bool(is_control),
        "control_of": control_of,
        "scene_id": WINE_SCENE_ID,
        "capability_id": WINE_CAPABILITY_ID,
        "instruction": WINE_INSTRUCTION,
        "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
        "completion_mode": getattr(service_, "completion_mode", None),
        "model_revision": model_revision,
        "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
        "source_git_sha": git_sha,
        "source_sha256": _source_sha256(),
        "session_id": None,
        "initial_state_sha": None,
        "xml_sha": None,
        "camera_names": [],
        "control_frequency_hz": None,
        "first_input": None,
        "task_language": None,
        "policy_instructions": [],
        "plan": None,
        "plan_terminal_state": None,
        "jobs": [],
        "policy_action_count": None,
        "policy_wall_s": None,
        "wine_samples_n": None,
        "sent_actions": [],
        "wine_statistics": None,
        "bottle_lift_m": None,
        "native_region_facts": None,
        "final_snapshot": None,
        "strict_policy_success": None,
        "strict_task_success": None,
        "semantic_before_assessment": None,
        "semantic_after_assessment": None,
        "assessment_action_count": 0,
        "assessment_physics": False,
        "assessment": None,
        "observer": None,
        "artifacts": {},
        "assisted_policy": False,
        "hermes_calls": 0,
        "assessment_included_in_walltime": True,
        "errors": [],
        "operational_errors": [],
        "physical_failures": [],
        "null_metrics": [],
        "timings": {},
    }
    operational = trial["operational_errors"]

    def _op(message: str) -> None:
        operational.append(message)
        trial["errors"].append(message)

    try:
        session = service_.create_session(
            WINE_SCENE_ID, seed=state_index, init_state_index=state_index
        )
        if not session.get("ok"):
            _op("create_session: %s" % session)
            trial["null_metrics"].append("session")
            return trial
        session_id = session["session_id"]
        trial["session_id"] = session_id
        record = service_._sessions.get(session_id)
        trial["initial_state_sha"] = getattr(record, "initial_state_hash", None)
        trial["xml_sha"] = getattr(record, "xml_sha", None)
        trial["camera_names"] = list(getattr(record, "camera_names", None) or [])
        service_._paired_initial_state_sha = trial["initial_state_sha"]
        service_._paired_xml_sha = trial["xml_sha"]
        service_._paired_camera_names = list(trial["camera_names"])

        frequency = _read_control_frequency(service_)
        if frequency.get("ok"):
            trial["control_frequency_hz"] = frequency.get("control_frequency_hz")
            service_._paired_control_frequency_hz = trial["control_frequency_hz"]

        language = wd._read_task_language(service_)
        if language.get("ok"):
            trial["task_language"] = language.get("task_language")

        # Seed the model RNG only AFTER the env exists; the env is never reset
        # afterwards, so every profile/model-seed repeat of this state starts from
        # the same initial environment state.
        seeded = _seed_model_rng(service_, model_seed)
        trial["model_rng_seed"] = {
            "requested": int(model_seed),
            "seeded": bool(seeded.get("seeded")),
            "reason": seeded.get("reason"),
        }

        profile_result = service_.configure_profile(profile)
        trial["profile_readback"] = profile_result
        if not profile_result.get("ok"):
            _op("profile_config: %s" % profile_result)
            trial["null_metrics"].append("profile")
            return trial

        plan_record = pe._submit_and_wait(
            service_,
            session_id,
            [WINE_CAPABILITY_ID],
            int(budget),
            False,
            "%s-plan1" % trial_key,
            float(timeout),
            "paired-state wine configuration trial",
        )
        plan_public = plan_record.get("plan") or {}
        trial["plan"] = {
            "request_id": plan_record.get("request_id"),
            "submitted": plan_record.get("submitted"),
            "submit_error": plan_record.get("submit_error"),
            "timed_out": plan_record.get("timed_out"),
            "cancel_nonterminal": plan_record.get("cancel_nonterminal"),
            "cancelled": plan_record.get("cancelled"),
            "wall_s": plan_record.get("wall_s"),
            "state": plan_public.get("state"),
            "plan_success": plan_public.get("plan_success"),
            "completed_capability_ids": plan_public.get("completed_capability_ids"),
            "pending_capability_ids": plan_public.get("pending_capability_ids"),
            "job_ids": plan_public.get("job_ids"),
        }
        trial["plan_terminal_state"] = plan_public.get("state")

        job_ids = plan_public.get("job_ids") or []
        evidence_list = [wd._job_evidence(service_, job_id) for job_id in job_ids]
        trial["jobs"] = [_compact_job(evidence) for evidence in evidence_list]

        instructions: list[Any] = []
        for evidence in evidence_list:
            if evidence.get("available"):
                instruction = (evidence.get("job") or {}).get("instruction")
                if instruction is not None:
                    instructions.append(instruction)
        trial["policy_instructions"] = instructions
        for instruction in instructions:
            if instruction != WINE_INSTRUCTION:
                _op(
                    "instruction_mismatch: submitted policy instruction %r != %r"
                    % (instruction, WINE_INSTRUCTION)
                )

        if evidence_list:
            last = evidence_list[-1]
            trial["artifacts"] = {
                "run_dir": last.get("run_dir"),
                "wine_telemetry": last.get("wine_telemetry_path"),
                "wine_diagnostic": last.get("wine_diagnostic_path"),
                "guard_semantic": (
                    str(Path(last.get("run_dir")) / "guard_semantic.jsonl")
                    if last.get("run_dir")
                    else None
                ),
                "events": last.get("events_path"),
                "result": last.get("result_path"),
                "rollout": last.get("rollout_path"),
                "latest_png": last.get("latest_png"),
            }

        final_job_samples: list[dict[str, Any]] = []
        if evidence_list:
            final_job_samples = evidence_list[-1].get("wine_samples") or []
        trial["wine_statistics"] = wd.summarize_wine_samples(final_job_samples)
        trial["wine_samples_n"] = len(final_job_samples)
        trial["bottle_lift_m"] = trial["wine_statistics"].get("max_lift_m")
        trial["sent_actions"] = [
            sample.get("sent_action") for sample in final_job_samples
        ]
        trial["policy_action_count"] = sum(
            int((evidence.get("job") or {}).get("steps") or 0)
            for evidence in evidence_list
        )
        policy_walls = [
            (evidence.get("job") or {}).get("wall_s")
            for evidence in evidence_list
            if (evidence.get("job") or {}).get("wall_s") is not None
        ]
        trial["policy_wall_s"] = policy_walls[-1] if policy_walls else None

        final_snapshot: dict[str, Any] | None = None
        if service_._active_request_id is None and service_._env is not None:
            final = service_.final_snapshot(WINE_ORACLE_GOALS)
            if final.get("ok"):
                final_snapshot = final.get("snapshot")
            else:
                _op("final_snapshot: %s" % final)
        else:
            _op(
                "final_snapshot_unavailable: active_request=%r env_present=%s"
                % (service_._active_request_id, service_._env is not None)
            )
        trial["final_snapshot"] = final_snapshot
        if isinstance(final_snapshot, dict):
            trial["native_region_facts"] = {
                "predicates": final_snapshot.get("predicates"),
                "phases": final_snapshot.get("phases"),
                "held_objects": final_snapshot.get("held_objects"),
                "strict_candidate": final_snapshot.get("strict_candidate"),
                "goal_objects": final_snapshot.get("goal_objects"),
            }
            predicates = final_snapshot.get("predicates")
            if isinstance(predicates, dict) and predicates and all(
                value is not None for value in predicates.values()
            ):
                trial["strict_task_success"] = all(predicates.values())
        trial["strict_policy_success"] = pe.strict_final_score(
            final_snapshot if isinstance(final_snapshot, dict) else None,
            [
                sample.get("after_snapshot")
                for sample in final_job_samples
                if isinstance(sample, dict)
            ],
        )

        first_input = service_._paired_input_evidence
        trial["first_input"] = _input_record(first_input) if isinstance(first_input, dict) else None
        observer_summary = service_._paired_observer_summary
        if isinstance(observer_summary, dict):
            trial["observer"] = observer_summary
            trial["first_failed_grasp_step"] = observer_summary.get("first_failed_grasp_step")
            trial["grasp_confirmed_step"] = observer_summary.get("first_grasp_confirmed_step")

        # --- operational vs physical classification --------------------------
        # A cancellation is strictly OPERATIONAL, never a physical budget/task
        # failure: a cancelled job (or plan) is never counted as
        # ``budget_exhausted`` and is never reinterpreted.  Its original state and
        # ended_reason are preserved verbatim in the compact job/plan records, and
        # the cancellation is recorded as an operational error that stops the
        # campaign (no assessment, no later trials).
        for evidence in evidence_list:
            if not evidence.get("available"):
                continue
            job_public = evidence.get("job") or {}
            state = job_public.get("state")
            ended = job_public.get("ended_reason")
            detail = job_public.get("error")
            if state == "cancelled" or ended == "cancelled":
                _op(
                    "job_cancelled: job=%s state=%s ended_reason=%s"
                    % (evidence.get("job_id"), state, ended)
                )
            elif state == "error" or ended == "error" or detail:
                _op(
                    "job_error: job=%s state=%s ended_reason=%s error=%s"
                    % (evidence.get("job_id"), state, ended, detail or "")
                )
            elif ended == "budget_exhausted":
                trial["physical_failures"].append("budget_exhausted: job=%s" % evidence.get("job_id"))

        if plan_record.get("submit_error"):
            _op("plan_submit: %s" % plan_record["submit_error"])
        if plan_record.get("timed_out"):
            _op("plan_timeout: %s" % plan_record.get("request_id"))
        if plan_record.get("cancel_nonterminal"):
            _op("cancel_nonterminal: %s" % plan_record.get("request_id"))
        if plan_record.get("cancelled") or trial["plan_terminal_state"] == "cancelled":
            _op(
                "plan_cancelled: request=%s state=%s cancelled=%s"
                % (
                    plan_record.get("request_id"),
                    trial["plan_terminal_state"],
                    plan_record.get("cancelled"),
                )
            )

        # --- assessment (only after a normal terminal policy job) -------------
        # Cancellation is removed from ``plan_normal``: a cancelled plan/job is
        # operational and must never reach the 20-step assessment.
        normal_jobs = bool(evidence_list) and all(_job_is_normal(e) for e in evidence_list)
        plan_normal = (
            trial["plan_terminal_state"] in service.TERMINAL_PLAN_STATES
            and trial["plan_terminal_state"] != "cancelled"
            and not plan_record.get("timed_out")
            and not plan_record.get("cancel_nonterminal")
            and not plan_record.get("cancelled")
        )
        if operational:
            # An operational/cancelled trial receives NO assessment and the caller
            # stops the campaign; the policy record stays exactly as measured.
            trial["assessment"] = {
                "performed": False,
                "reason": "operational_error_or_cancelled; no assessment; campaign stops",
            }
        elif service_._active_request_id is not None or service_._env is None:
            trial["assessment"] = {
                "performed": False,
                "reason": "no idle live environment for the assessment",
            }
            _op("assessment_unavailable: no idle live environment")
        elif not (normal_jobs and plan_normal):
            trial["assessment"] = {
                "performed": False,
                "reason": "policy job did not end normally; no assessment",
            }
        else:
            run_dir = Path(
                (trial["artifacts"] or {}).get("run_dir")
                or (service_.run_root / trial_key)
            )
            outcome = run_assessment(service_, run_dir, service_._paired_tracker)
            trial["assessment"] = {
                "performed": True,
                "path": outcome.get("path"),
                "action_count": outcome.get("action_count"),
                "expected_action_count": outcome.get("expected_action_count"),
                "action_dtype": outcome.get("action_dtype"),
                "action_shape": outcome.get("action_shape"),
                "strict_final_score": outcome.get("strict_final_score"),
                "final_snapshot": outcome.get("final_snapshot"),
                "native_success_final": outcome.get("native_success_final"),
                "native_success_ever": outcome.get("native_success_ever"),
                "error": outcome.get("error"),
            }
            trial["assessment_action_count"] = int(outcome.get("action_count") or 0)
            trial["assessment_physics"] = bool(trial["assessment_action_count"])
            trial["semantic_before_assessment"] = _semantic_flag(outcome.get("semantic_before"))
            trial["semantic_after_assessment"] = _semantic_flag(outcome.get("semantic_after"))
            trial["assessment_semantic_success"] = outcome.get("semantic_success")
            if outcome.get("error"):
                _op("assessment_error: %s" % outcome.get("error"))

        for name in (
            "initial_state_sha",
            "xml_sha",
            "control_frequency_hz",
            "task_language",
        ):
            if trial.get(name) is None:
                trial["null_metrics"].append(name)
        if trial["first_input"] is None:
            trial["null_metrics"].append("first_input")
        if trial["strict_policy_success"] is None:
            trial["null_metrics"].append("strict_policy_success")
        if trial["strict_task_success"] is None:
            trial["null_metrics"].append("strict_task_success")
        if trial.get("bottle_lift_m") is None:
            trial["null_metrics"].append("bottle_lift_m")

        trial["task_success"] = trial["strict_task_success"]
        return trial
    except BaseException as exc:  # noqa: BLE001 - preserve the record
        _op("trial_exception: %s" % _format_exc(exc))
        return trial
    finally:
        trial["timings"]["trial_wall_s"] = round(time.monotonic() - started, 3)
        trial["policy_job_wall_s"] = trial["policy_wall_s"]
        trial["walltime_includes_assessment"] = bool(trial.get("assessment_action_count"))


# --- plan construction -------------------------------------------------------


def build_discovery_plan() -> list[dict[str, Any]]:
    """The fixed discovery grid: 18 non-control trials plus one A/A duplicate.

    States ``0, 1, 2`` in order; the environment seed always equals the state
    index; within each state the model RNG seeds ``[index, index+1000]`` and the
    profiles ``[baseline_bf16, fp32, fp32_h5]`` in that exact order.  One A/A
    duplicate of state 0 / model seed 0 / ``baseline_bf16`` is inserted
    immediately after the first trial, for 19 trials in total.
    """

    plan: list[dict[str, Any]] = []
    for state_index in DISCOVERY_STATES:
        for offset in MODEL_SEED_OFFSETS:
            for profile in PROFILE_ORDER:
                plan.append(
                    {
                        "state_index": int(state_index),
                        "envseed": int(state_index),
                        "model_seed": int(state_index) + int(offset),
                        "profile": profile,
                        "is_control": False,
                        "control_of": None,
                    }
                )
    if plan:
        first = plan[0]
        duplicate = dict(first)
        duplicate["is_control"] = True
        duplicate["control_of"] = "state0_modelseed0_baseline"
        plan.insert(1, duplicate)
    return plan


def build_holdout_plan(selected_profile: str | None) -> list[dict[str, Any]]:
    """The fixed holdout grid: states 3, 4, 5 with env/model seed == index.

    The baseline is always run; the selected profile is added only when it is not
    the baseline (3 trials for a baseline winner, otherwise 6).  The holdout never
    reselects a profile.
    """

    profiles = [PROFILE_A]
    if selected_profile and selected_profile != PROFILE_A:
        profiles.append(selected_profile)
    plan: list[dict[str, Any]] = []
    for state_index in HOLDOUT_STATES:
        for profile in profiles:
            plan.append(
                {
                    "state_index": int(state_index),
                    "envseed": int(state_index),
                    "model_seed": int(state_index),
                    "profile": profile,
                    "is_control": False,
                    "control_of": None,
                }
            )
    return plan


# --- selection and summaries -------------------------------------------------


def select_profile(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-declared, immutable discovery selection rule.

    Primary: highest count of strict POLICY successes across the 18 non-control
    discovery trials (the A/A duplicate is excluded).  Secondary: highest count of
    post-assessment semantic successes.  Tertiary: lowest median policy wall time.
    An exact tie is broken by the pre-declared order baseline_bf16 > fp32 >
    fp32_h5.  An unknown value is never a success, and no success rate is inferred
    from the counts.
    """

    non_control = [trial for trial in trials if not trial.get("is_control")]
    stats: dict[str, dict[str, Any]] = {}
    for profile in SELECTABLE_PROFILES:
        rows = [trial for trial in non_control if trial.get("profile") == profile]
        stats[profile] = {
            "n_trials": len(rows),
            "strict_successes": sum(
                1 for trial in rows if trial.get("strict_policy_success") is True
            ),
            "semantic_successes": sum(
                1 for trial in rows if trial.get("assessment_semantic_success") is True
            ),
            "median_policy_wall_s": _median(
                [trial.get("policy_wall_s") for trial in rows]
            ),
            "trials": [trial.get("trial_id") for trial in rows],
        }

    def _key(profile: str) -> tuple[float, float, float, int]:
        entry = stats[profile]
        wall = entry["median_policy_wall_s"]
        return (
            -float(entry["strict_successes"]),
            -float(entry["semantic_successes"]),
            float(wall) if wall is not None else float("inf"),
            PROFILE_TIE_ORDER[profile],
        )

    ranking = sorted(SELECTABLE_PROFILES, key=_key)
    return {
        "rule": dict(_SELECTION_RULE),
        "n_non_control_trials": len(non_control),
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
        "stats": stats,
        "reselects": False,
    }


def _paired_outcome(first: Any, second: Any) -> str:
    """A strict-success outcome label for one paired comparison."""

    a = (first or {}).get("strict_policy_success")
    b = (second or {}).get("strict_policy_success")
    if a is True and b is True:
        return "both"
    if a is True:
        return "first"
    if b is True:
        return "second"
    return "neither"


def _side_summary(trial: Any) -> dict[str, Any] | None:
    if not isinstance(trial, dict):
        return None
    return {
        "trial_id": trial.get("trial_id"),
        "profile": trial.get("profile"),
        "strict_policy_success": trial.get("strict_policy_success"),
        "strict_task_success": trial.get("strict_task_success"),
        "plan_terminal_state": trial.get("plan_terminal_state"),
        "policy_action_count": trial.get("policy_action_count"),
        "policy_wall_s": trial.get("policy_wall_s"),
        "bottle_lift_m": trial.get("bottle_lift_m"),
        "semantic_after_assessment": trial.get("semantic_after_assessment"),
        "assessment_semantic_success": trial.get("assessment_semantic_success"),
        "n_sent_actions": len([a for a in (trial.get("sent_actions") or []) if a]),
    }


def _discovery_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired A-vs-B and B-vs-C summaries per state/model-seed group."""

    non_control = [trial for trial in trials if not trial.get("is_control")]
    groups: dict[tuple, dict[str, Any]] = {}
    for trial in non_control:
        key = (trial.get("state_index"), trial.get("model_seed"))
        groups.setdefault(key, {})[trial.get("profile")] = trial

    def _sort_key(key: tuple) -> tuple:
        state_index, model_seed = key
        return (
            state_index if isinstance(state_index, int) else -1,
            model_seed if isinstance(model_seed, int) else -1,
        )

    pairs: list[dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        row = groups[key]
        a, b, c = row.get(PROFILE_A), row.get(PROFILE_B), row.get(PROFILE_C)
        pairs.append(
            {
                "state_index": key[0],
                "model_seed": key[1],
                "A_vs_B": {
                    "first_profile": PROFILE_A,
                    "second_profile": PROFILE_B,
                    "first": _side_summary(a),
                    "second": _side_summary(b),
                    "outcome": _paired_outcome(a, b),
                },
                "B_vs_C": {
                    "first_profile": PROFILE_B,
                    "second_profile": PROFILE_C,
                    "first": _side_summary(b),
                    "second": _side_summary(c),
                    "outcome": _paired_outcome(b, c),
                },
            }
        )
    return {
        "n_non_control_trials": len(non_control),
        "n_groups": len(groups),
        "pairs": pairs,
        "note": (
            "explicit paired counts only; three states x two model seeds is not a "
            "statistical reliability claim and no success rate is inferred"
        ),
    }


def compare_aa(control: Any, repeat: Any) -> dict[str, Any]:
    """Compare the A/A duplicate against its control trial's actual sent actions.

    Both trials share the same env state and the same model seed; the comparison
    reports length agreement, an exact-match count and the maximum absolute
    difference over the actual seven-dimensional sent arrays, plus the terminal
    stop result and the captured first inputs.  This is a *repeatability* report,
    never a claim of bitwise CUDA determinism (a GPU kernel is not required to be
    bit-reproducible and is not asserted to be).
    """

    result: dict[str, Any] = {
        "control_trial_id": (control or {}).get("trial_id"),
        "repeat_trial_id": (repeat or {}).get("trial_id"),
        "state_index": (control or {}).get("state_index"),
        "model_seed": (control or {}).get("model_seed"),
        "profile": (control or {}).get("profile"),
        "control_length": None,
        "repeat_length": None,
        "length_match": None,
        "compared_count": None,
        "exact_match_count": None,
        "max_abs_diff": None,
        "length_mismatches": [],
        "control_stop": None,
        "repeat_stop": None,
        "stop_result_match": None,
        "inputs_match": None,
        "first_input_errors": [],
        "note": (
            "repeatability evidence only; bitwise CUDA determinism is not claimed "
            "or asserted"
        ),
    }
    if not isinstance(control, dict) or not isinstance(repeat, dict):
        result["note"] = "control or repeat trial missing"
        return result

    first = [a for a in (control.get("sent_actions") or [])]
    second = [a for a in (repeat.get("sent_actions") or [])]
    result["control_length"] = sum(1 for a in first if a)
    result["repeat_length"] = sum(1 for a in second if a)
    result["length_match"] = result["control_length"] == result["repeat_length"]

    compared = 0
    exact = 0
    max_diff: float | None = None
    mismatches: list[str] = []
    for index, (left, right) in enumerate(zip(first, second)):
        if not isinstance(left, list) or not isinstance(right, list):
            mismatches.append("index %d: a side has no readable sent action" % index)
            continue
        if len(left) != len(right):
            mismatches.append("index %d: dims %d != %d" % (index, len(left), len(right)))
            continue
        compared += 1
        equal = left == right
        if equal:
            exact += 1
        for component, (lval, rval) in enumerate(zip(left, right)):
            try:
                diff = abs(float(lval) - float(rval))
            except (TypeError, ValueError):
                mismatches.append("index %d.%d: non-numeric component" % (index, component))
                continue
            if max_diff is None or diff > max_diff:
                max_diff = diff
    result["compared_count"] = compared
    result["exact_match_count"] = exact
    result["max_abs_diff"] = max_diff
    result["length_mismatches"] = mismatches[:20]

    result["control_stop"] = {
        "plan_terminal_state": control.get("plan_terminal_state"),
        "policy_action_count": control.get("policy_action_count"),
        "strict_policy_success": control.get("strict_policy_success"),
        "assessment_action_count": control.get("assessment_action_count"),
    }
    result["repeat_stop"] = {
        "plan_terminal_state": repeat.get("plan_terminal_state"),
        "policy_action_count": repeat.get("policy_action_count"),
        "strict_policy_success": repeat.get("strict_policy_success"),
        "assessment_action_count": repeat.get("assessment_action_count"),
    }
    result["stop_result_match"] = result["control_stop"] == result["repeat_stop"]

    left_input = control.get("first_input") or {}
    right_input = repeat.get("first_input") or {}
    result["inputs_match"] = (
        left_input.get("combined_sha256") is not None
        and left_input.get("combined_sha256") == right_input.get("combined_sha256")
    )
    if left_input.get("combined_sha256") is None:
        result["first_input_errors"].append("control input digest missing")
    if right_input.get("combined_sha256") is None:
        result["first_input_errors"].append("repeat input digest missing")
    return result


# --- preregistration ---------------------------------------------------------


def _build_preregistration(
    args: argparse.Namespace,
    source_sha256: dict[str, str | None],
    checkpoint_audit: dict[str, Any],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """The frozen experiment plan, written BEFORE the model is loaded."""

    return {
        "experiment": EXPERIMENT_NAME,
        "created_utc": _now_utc(),
        "phase": args.phase,
        "commitment": (
            "written before the model starts; the raw SHA-256 of this file's bytes "
            "is recorded in the campaign report and the plan is never re-decided "
            "after data collection begins"
        ),
        "fixed_spec": {
            "scene_id": WINE_SCENE_ID,
            "capability_id": WINE_CAPABILITY_ID,
            "instruction": WINE_INSTRUCTION,
            "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
            "completion_mode": COMPLETION_MODE,
            "guard_mode": GUARD_MODE,
            "profiles": list(PROFILE_ORDER),
            "discovery_states": list(DISCOVERY_STATES),
            "holdout_states": list(HOLDOUT_STATES),
            "model_seed_offsets": list(MODEL_SEED_OFFSETS),
            "target_revision": FIXED_SOURCE_REVISION,
            "checkpoint_file_count": AUDITED_CHECKPOINT_FILE_COUNT,
            "assisted_policy": False,
            "hermes_calls": 0,
        },
        "parameters": {
            "phase": args.phase,
            "selected_profile": getattr(args, "selected_profile", None),
            "budget_per_subgoal": int(args.budget),
            "timeout_s": float(args.timeout),
            "assessment_action_count": ASSESSMENT_ACTION_COUNT,
            "assessment_action_shape": [ACTION_DIM],
            "assessment_action_dtype": "float32",
        },
        "plan": plan,
        "selection_rule": dict(_SELECTION_RULE),
        "assessment_protocol": dict(_ASSESSMENT_PROTOCOL),
        "source_sha256": source_sha256,
        "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
        "source_git_sha": getattr(args, "source_git_sha", None),
        "checkpoint_audit": checkpoint_audit,
    }


# --- campaign report ---------------------------------------------------------


def _aa_from_trials(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The A/A comparison: the original first trial versus its duplicate."""

    duplicate = next((trial for trial in trials if trial.get("is_control")), None)
    if duplicate is None:
        return None
    target = next((trial for trial in trials if not trial.get("is_control")), None)
    return compare_aa(target, duplicate)


def _build_campaign_report(
    args: argparse.Namespace,
    trials: list[dict[str, Any]],
    *,
    model_revision: str | None,
    git_sha: str | None,
    started: float,
    fatal_error: str | None,
    plan: list[dict[str, Any]],
    preregistration_path: Path,
    preregistration_sha256: str | None,
    selection: dict[str, Any] | None,
    aa_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    """A complete, self-consistent campaign snapshot (metadata + trials + summary)."""

    discovery = args.phase == "discovery"
    return {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "phase": args.phase,
        "metadata": {
            "output": str(args.output),
            "run_root": str(args.run_root),
            "phase": args.phase,
            "selected_profile": getattr(args, "selected_profile", None),
            "budget_per_subgoal": int(args.budget),
            "timeout_s": float(args.timeout),
            "scene_id": WINE_SCENE_ID,
            "capability_id": WINE_CAPABILITY_ID,
            "instruction": WINE_INSTRUCTION,
            "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
            "completion_mode": COMPLETION_MODE,
            "guard_mode": GUARD_MODE,
            "profiles": list(PROFILE_ORDER),
            "model_path": service.DEFAULT_MODEL_PATH,
            "model_revision": model_revision,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "target_revision": FIXED_SOURCE_REVISION,
            "source_git_sha": git_sha,
            "source_sha256": _source_sha256(),
            "assisted_policy": False,
            "hermes_calls": 0,
            "assessment_action_count": ASSESSMENT_ACTION_COUNT,
            "assessment_is_real_physics": True,
            "assessment_promoted_to_benchmark": False,
            "campaign_wall_s": round(time.monotonic() - started, 3),
            "fatal_error": fatal_error,
        },
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": preregistration_sha256,
            "frozen_before_model_start": preregistration_sha256 is not None,
            "plan": plan,
        },
        "selection": selection,
        "aa_control": aa_comparison,
        "discovery_summary": _discovery_summary(trials) if discovery else None,
        "trials": trials,
        "aggregate": {
            "n_trials": len(trials),
            "n_control_trials": sum(1 for trial in trials if trial.get("is_control")),
            "n_operational_errors": sum(1 for trial in trials if trial.get("operational_errors")),
            "n_strict_policy_success": sum(
                1 for trial in trials if trial.get("strict_policy_success") is True
            ),
            "n_strict_policy_failure": sum(
                1 for trial in trials if trial.get("strict_policy_success") is False
            ),
            "n_strict_policy_unknown": sum(
                1 for trial in trials if trial.get("strict_policy_success") is None
            ),
            "n_budget_exhausted": sum(1 for trial in trials if trial.get("physical_failures")),
            "n_assessments": sum(1 for trial in trials if trial.get("assessment_physics")),
            "reliability_note": (
                "explicit counts only; a three-state / six-trial grid is not a "
                "statistical reliability claim and no success rate is inferred"
            ),
        },
    }


# --- entry point -------------------------------------------------------------


PHASES = ("discovery", "holdout")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated paired-state wine configuration experiment runner (no "
            "training, no downloads, no Hermes, no forced release, no HTTP server)."
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
        help="absolute run root for per-trial artifacts (must not exist)",
    )
    parser.add_argument(
        "--phase",
        choices=list(PHASES),
        default="discovery",
        help="discovery (states 0-2) or holdout (states 3-5)",
    )
    parser.add_argument(
        "--selected-profile",
        choices=list(SELECTABLE_PROFILES),
        default=None,
        help="the pre-selected profile (required for --phase holdout; never reselected)",
    )
    parser.add_argument("--budget", type=int, default=300, help="per-subgoal action budget")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-plan deadline in seconds")
    parser.add_argument(
        "--source-git-sha",
        type=str,
        default=None,
        help="optional fallback source git SHA (used when git is unavailable)",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isabs(args.output):
        parser.error("--output must be an absolute path")
    if not os.path.isabs(args.run_root):
        parser.error("--run-root must be an absolute path")
    if Path(args.output).exists():
        parser.error("--output already exists; refusing to overwrite %s" % args.output)
    if Path(args.run_root).exists():
        parser.error("--run-root already exists; refusing to reuse %s" % args.run_root)
    if args.phase == "holdout" and args.selected_profile not in SELECTABLE_PROFILES:
        parser.error("--phase holdout requires --selected-profile")
    if not isinstance(args.budget, int) or isinstance(args.budget, bool) or args.budget < 1:
        parser.error("--budget must be a positive integer")
    if not (args.timeout > 0):
        parser.error("--timeout must be positive")


def _run_campaign(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    discovery = args.phase == "discovery"
    plan = build_discovery_plan() if discovery else build_holdout_plan(args.selected_profile)
    git_sha = args.source_git_sha or pe._git_rev_parse()
    preregistration_path = run_root / "preregistration.json"

    trials: list[dict[str, Any]] = []
    fatal_error: str | None = None
    model_revision: str | None = None
    started = time.monotonic()
    persist_errors: list[str] = []
    preregistration_sha256: str | None = None
    selection: dict[str, Any] | None = None
    aa_comparison: dict[str, Any] | None = None

    def _persist() -> None:
        """Atomically write the current report snapshot; never raises.

        The selection summary is recomputed from the immutable rule on every
        write, but only for the discovery phase: the holdout phase records the
        pre-selected profile and never reselects.
        """

        nonlocal selection, aa_comparison
        if discovery:
            selection = select_profile(trials)
            aa_comparison = _aa_from_trials(trials)
        else:
            selection = {
                "reselects": False,
                "selected_profile": args.selected_profile,
                "note": "the holdout phase never reselects a profile",
            }
            aa_comparison = None
        try:
            pe._write_json_atomic(
                output_path,
                _build_campaign_report(
                    args,
                    trials,
                    model_revision=model_revision,
                    git_sha=git_sha,
                    started=started,
                    fatal_error=fatal_error,
                    plan=plan,
                    preregistration_path=preregistration_path,
                    preregistration_sha256=preregistration_sha256,
                    selection=selection,
                    aa_comparison=aa_comparison,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a report write failure is fatal
            persist_errors.append(_format_exc(exc))
            _progress("report write failed: %s" % exc)

    # 1. Freeze the preregistration BEFORE the model starts.  A missing model
    #    directory / an unexpected checkpoint file set is an operational failure
    #    and no action is ever taken.
    checkpoint_audit = _audit_checkpoint_files(service.DEFAULT_MODEL_PATH)
    try:
        pe._write_json_atomic(
            preregistration_path,
            _build_preregistration(args, _source_sha256(), checkpoint_audit, plan),
        )
        preregistration_sha256 = _sha256_hex(preregistration_path.read_bytes())
    except Exception as exc:  # noqa: BLE001 - no preregistration -> no run
        fatal_error = "preregistration write failed: %s" % _format_exc(exc)
        _progress(fatal_error)
    if fatal_error is None and not checkpoint_audit.get("ok"):
        fatal_error = "checkpoint audit failed: %s" % checkpoint_audit.get("reason")

    diagnostic_service: PairedWineDiagnosticService | None = None

    if fatal_error is None:
        with wd.register_native_wine_scene():
            try:
                diagnostic_service = PairedWineDiagnosticService(
                    service.DEFAULT_MODEL_PATH, str(run_root)
                )
                diagnostic_service.start()
                ready = False
                deadline = time.monotonic() + READY_TIMEOUT_S
                while time.monotonic() < deadline:
                    health = diagnostic_service.health()
                    if health.get("ready"):
                        ready = True
                        break
                    if health.get("worker_error"):
                        fatal_error = str(health["worker_error"])
                        break
                    time.sleep(0.5)
                if not ready and fatal_error is None:
                    fatal_error = "worker did not become ready within %.0fs" % READY_TIMEOUT_S
                model_revision = diagnostic_service.health().get("model_revision")

                if ready and model_revision != FIXED_SOURCE_REVISION:
                    # ZERO actions: the resident model revision must match the
                    # audited revision before any plan is submitted.
                    fatal_error = (
                        "model_revision %r does not match FIXED_SOURCE_REVISION %r"
                        % (model_revision, FIXED_SOURCE_REVISION)
                    )
                    ready = False

                if ready:
                    registry: dict = {}
                    total = len(plan)
                    for index, entry in enumerate(plan, start=1):
                        _progress(
                            "trial %d/%d phase=%s state=%s model_seed=%s profile=%s control=%s"
                            % (
                                index,
                                total,
                                args.phase,
                                entry["state_index"],
                                entry["model_seed"],
                                entry["profile"],
                                entry["is_control"],
                            )
                        )
                        try:
                            trial = run_trial(
                                diagnostic_service,
                                phase=args.phase,
                                state_index=entry["state_index"],
                                model_seed=entry["model_seed"],
                                profile=entry["profile"],
                                timeout=float(args.timeout),
                                budget=int(args.budget),
                                model_revision=model_revision,
                                git_sha=git_sha,
                                registry=registry,
                                is_control=entry["is_control"],
                                control_of=entry["control_of"],
                            )
                        except BaseException as exc:  # noqa: BLE001 - preserve, then stop
                            fatal_error = _format_exc(exc)
                            trial = {
                                "trial_id": "s%s_m%s_%s"
                                % (entry["state_index"], entry["model_seed"], entry["profile"]),
                                "phase": args.phase,
                                "state_index": entry["state_index"],
                                "model_seed": entry["model_seed"],
                                "profile": entry["profile"],
                                "is_control": entry["is_control"],
                                "errors": ["trial_exception: %s" % _format_exc(exc)],
                                "operational_errors": ["trial_exception: %s" % _format_exc(exc)],
                                "null_metrics": ["trial"],
                            }
                            trials.append(trial)
                            _persist()
                            _progress("trial exception; stopping campaign (no continuation)")
                            break
                        trials.append(trial)
                        # The report is atomically persisted after EVERY trial, so a
                        # crash can never lose a finished trial.
                        _persist()
                        _progress(
                            "trial %d/%d done state=%s strict_policy=%s ops=%d"
                            % (
                                index,
                                total,
                                trial.get("plan_terminal_state"),
                                trial.get("strict_policy_success"),
                                len(trial.get("operational_errors") or []),
                            )
                        )
                        if trial.get("operational_errors"):
                            # An operational error (job error, nonterminal cancel,
                            # instruction mismatch, input mismatch/incompleteness,
                            # assessment failure, ...) stops the campaign.  An
                            # ordinary budget/task failure does NOT.
                            fatal_error = (
                                "operational error(s); stopping campaign (no "
                                "continuation): %s"
                                % ("; ".join(trial["operational_errors"]),)
                            )
                            _progress(fatal_error)
                            break
            except BaseException as exc:  # noqa: BLE001 - never lose the partial report
                fatal_error = _format_exc(exc)
            finally:
                if diagnostic_service is not None:
                    try:
                        if (
                            diagnostic_service._ready
                            and diagnostic_service._active_request_id is None
                        ):
                            diagnostic_service._sync_work(
                                "close_env",
                                lambda: (diagnostic_service._close_env() or {"ok": True}),
                            )
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        diagnostic_service.stop()
                    except Exception:  # noqa: BLE001
                        pass

    # Final persistence, regardless of how the run ended.
    _persist()
    _progress("report written to %s (%d trials)" % (output_path, len(trials)))

    operational_failure = any(bool(trial.get("operational_errors")) for trial in trials)
    return 1 if (fatal_error or persist_errors or operational_failure) else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return _run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
