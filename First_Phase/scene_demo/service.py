#!/usr/bin/env python3
"""Persistent-scene execution service (v2).

Unlike ``libero_demo/service.py`` -- which builds one LIBERO scene per job and
throws it away -- this service keeps *one* LIBERO simulator alive for a whole
session and drives several capability subgoals through it.  The first
``LiberoEnv.reset`` loads the native init state exactly once per session; every
capability afterwards only clears the policy/pre/post action queues and the
instruction state, never the environment.

Design (mirrors the v1 process layout on purpose):

* the main thread runs a stdlib ``ThreadingHTTPServer`` and only ever *reads*
  lock-protected public state or *queues* work;
* a single daemon worker thread exclusively owns the CUDA policy, the
  MuJoCo/LIBERO environment, rendering and predicate evaluation.  Every env
  construction, reset, step, render and predicate probe happens on that thread;
* synchronous requests (session creation, idle extra observations, independent
  evaluation) are queued with an ``Event``/``Future`` and awaited for at most
  60 s *without holding a lock*; plans are asynchronous.

Nothing here fabricates a result: capability success comes from the LIBERO
predicate evaluator (five consecutive satisfied steps), ``plan_success`` is the
final AND over every declared goal, and failures are reported as failures.

The v1 module is loaded by file location (never ``import service``) so the two
same-named modules can coexist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import queue
import random
import signal
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

# The sibling modules live next to this file; make sure they import cleanly no
# matter which directory the interpreter was started from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import catalog  # noqa: E402
import oracle  # noqa: E402


# --- v1 bridge ---------------------------------------------------------------


def _load_v1_libero_service():
    """Load ``libero_demo/service.py`` under a private module name.

    ``import service`` would collide with *this* module, so the sibling file is
    loaded by absolute location and registered in ``sys.modules`` *before*
    ``exec_module`` (dataclasses/pickle lookups rely on that).  The v1 file is
    only read, never modified.
    """
    v1_path = _HERE.parent / "libero_demo" / "service.py"
    if not v1_path.is_file():
        raise RuntimeError("v1 service module missing: %s" % v1_path)
    spec = importlib.util.spec_from_file_location("v1_libero_service", str(v1_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot build import spec for %s" % v1_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["v1_libero_service"] = module
    spec.loader.exec_module(module)
    return module


v1_libero_service = _load_v1_libero_service()

MODEL_REVISION_DEFAULT = v1_libero_service.MODEL_REVISION_DEFAULT
DEFAULT_MODEL_PATH = v1_libero_service.DEFAULT_MODEL_PATH
_batch_observation = v1_libero_service._batch_observation
_save_png = v1_libero_service._save_png
_save_video = v1_libero_service._save_video


# --- configuration -----------------------------------------------------------

DEFAULT_RUN_ROOT = "/home/yhwang/fyp/scene_demo/runs"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
BACKEND_NAME = "smolvla"
ROBOT_NAME = "Franka Panda"
WORKFLOW_NAME = "persistent_scene_v2"

OBS_WIDTH = 256
OBS_HEIGHT = 256
CONTROL_MODE = "relative"
ACTION_DIM = 7
VIDEO_FPS = 20
PNG_EVERY = 5
NUM_STEPS_WAIT = 10
SETTLE_STEPS = 20

SESSION_STEP_LIMIT = 2000
MAX_CAPABILITIES = 6
MIN_BUDGET = 1
MAX_BUDGET = 600
DEFAULT_BUDGET = 300
AUDIT_BUDGET = 600
GOAL_CONSECUTIVE_STEPS = 5
WORKER_WAIT_S = 60.0

VIEW_AGENTVIEW = "agentview"
VIEW_WRIST = "wrist"
EXTRA_VIEW_CANDIDATES = ("frontview", "birdview", "sideview")

DECISIONS = ("execute", "clarify", "unsupported")
TERMINAL_PLAN_STATES = ("completed", "blocked", "error", "cancelled")
ACTIVE_PLAN_STATES = ("queued", "running")

ALLOWED_ARTIFACT_SUFFIXES = (".png", ".mp4", ".json")

_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _LOG_LOCK:
        sys.stderr.write("[scene-service %s] %s\n" % (stamp, message))
        sys.stderr.flush()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _seed_everything(seed: int) -> None:
    """Seed the Python/NumPy/Torch RNGs exactly as v1 does.

    Torch is imported lazily (and its absence tolerated) so the GPU-free
    contract tests can create sessions without it.
    """

    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed))
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch in the GPU-free test interpreter
        return
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --- errors and result helpers ----------------------------------------------


class SceneError(Exception):
    """A request-level failure carrying a machine-readable ``reason``.

    The HTTP status is always derived from ``reason`` via
    :func:`status_for_reason`; the constructor accepts exactly ``reason`` and
    ``detail`` (no status code).
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (reason, detail) if detail else reason)
        self.reason = reason
        self.detail = detail


class NotReadyError(RuntimeError):
    """Raised on the worker when the policy has not (yet) loaded."""


def _err(reason: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "reason": reason, "detail": detail}


_REASON_STATUS = {
    # 400
    "invalid_body": 400,
    "invalid_decision": 400,
    "invalid_capabilities": 400,
    "invalid_budget": 400,
    "invalid_request_id": 400,
    "invalid_fixture": 400,
    "invalid_case": 400,
    "predicate_error": 400,
    # 404
    "unknown_scene": 404,
    "unknown_session": 404,
    "unknown_plan": 404,
    "unknown_job": 404,
    "unknown_case": 404,
    "unknown_capability": 404,
    "unknown_artifact": 404,
    # 409
    "busy": 409,
    "stale_scene_version": 409,
    "stale_evaluation": 409,
    "request_conflict": 409,
    "cross_scene": 409,
    "ownership": 409,
    "session_closed": 409,
    "target_occupied": 409,
    "audit_required": 409,
    "precondition_conflict": 409,
    "plan_not_blocked": 409,
    "resume_exhausted": 409,
    "blocked": 409,
    "cancelled": 409,
    # 503
    "not_ready": 503,
    "worker_error": 503,
    # 500
    "internal_error": 500,
}


def status_for_reason(reason: str) -> int:
    return _REASON_STATUS.get(reason, 400)


# --- pure request validators -------------------------------------------------
#
# These are deliberately free of service state so the contract can be unit
# tested without a GPU, a policy or a simulator.


def validate_capability_list(scene_id: str, capability_ids: Any, audit: bool) -> dict[str, Any] | None:
    """Validate a capability id list against the catalog and the scene."""

    if not isinstance(capability_ids, list) or any(not isinstance(c, str) for c in capability_ids):
        return _err("invalid_capabilities", "capability_ids must be a list of strings")
    if not capability_ids:
        return _err("invalid_capabilities", "at least one capability is required")
    if len(capability_ids) > MAX_CAPABILITIES:
        return _err("invalid_capabilities", "at most %d capabilities per plan" % MAX_CAPABILITIES)
    if len(set(capability_ids)) != len(capability_ids):
        return _err("invalid_capabilities", "duplicate capability ids are not allowed")
    for capability_id in capability_ids:
        capability = catalog.CAPABILITIES.get(capability_id)
        if capability is None:
            return _err("unknown_capability", "unknown capability %r" % capability_id)
        if scene_id not in capability["scene_ids"]:
            return _err(
                "cross_scene",
                "capability %r is not supported in scene %r" % (capability_id, scene_id),
            )
        if capability["audit_only"] and not audit:
            return _err(
                "audit_required",
                "capability %r is audit-only and needs audit=true" % capability_id,
            )
    return None


def validate_plan_request(
    session: dict[str, Any] | None,
    active_plan: dict[str, Any] | None,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate one ``POST /plans`` body.  Returns an error dict or ``None``."""

    if session is None:
        return _err("unknown_session", "no such session")
    # A session is "running" precisely while one of its plans is active, so the
    # active-plan test must come first: an in-flight plan is reported as busy,
    # not as a closed session.
    if active_plan is not None:
        return _err("busy", "plan %s is still active" % active_plan.get("request_id"))
    if session.get("state") != "ready":
        return _err("session_closed", "session is %s, not ready" % session.get("state"))

    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return _err("invalid_request_id", "request_id must be a non-empty string")

    decision = payload.get("decision")
    if decision not in DECISIONS:
        return _err("invalid_decision", "decision must be one of %s" % (list(DECISIONS),))

    capability_ids = payload.get("capability_ids")
    if not isinstance(capability_ids, list) or any(not isinstance(c, str) for c in capability_ids):
        return _err("invalid_capabilities", "capability_ids must be a list of strings")

    # The scene version is checked for *every* decision -- including clarify and
    # unsupported -- so a fresh decision can never be attached to a stale
    # observation of the scene.
    scene_version = payload.get("scene_version")
    if isinstance(scene_version, bool) or not isinstance(scene_version, int):
        return _err("stale_scene_version", "scene_version must be an integer")
    if scene_version != session.get("scene_version"):
        return _err(
            "stale_scene_version",
            "plan targets scene_version %s but the session is at %s"
            % (scene_version, session.get("scene_version")),
        )

    if decision != "execute":
        if capability_ids:
            return _err("invalid_capabilities", "a %s decision must carry no capabilities" % decision)
        return None

    budget = payload.get("budget_per_subgoal", DEFAULT_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, int) or not (MIN_BUDGET <= budget <= MAX_BUDGET):
        return _err(
            "invalid_budget",
            "budget_per_subgoal must be an integer in [%d, %d]" % (MIN_BUDGET, MAX_BUDGET),
        )
    if not capability_ids:
        return _err("invalid_capabilities", "an execute decision requires at least one capability")
    if budget * len(capability_ids) > SESSION_STEP_LIMIT:
        return _err(
            "invalid_budget",
            "budget %d x %d capabilities exceeds the %d-step session limit"
            % (budget, len(capability_ids), SESSION_STEP_LIMIT),
        )

    audit = bool(payload.get("audit", False))
    return validate_capability_list(session.get("scene_id"), capability_ids, audit)


def validate_resume_request(
    session: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate one ``POST /plans/<id>/resume`` body."""

    if session is None:
        return _err("unknown_session", "no such session")
    if plan is None:
        return _err("unknown_plan", "no such plan")
    if plan.get("session_id") != session.get("session_id"):
        return _err("ownership", "plan does not belong to this session")
    if plan.get("state") != "blocked":
        return _err("plan_not_blocked", "only a blocked plan may be resumed")
    if len(plan.get("repair_history") or []) >= 1:
        return _err("resume_exhausted", "the single allowed repair has already been used")
    scene_version = payload.get("scene_version")
    if isinstance(scene_version, bool) or not isinstance(scene_version, int):
        return _err("stale_scene_version", "scene_version must be an integer")
    if scene_version != session.get("scene_version"):
        return _err(
            "stale_scene_version",
            "resume targets scene_version %s but the session is at %s"
            % (scene_version, session.get("scene_version")),
        )
    budget = payload.get("budget_per_subgoal", DEFAULT_BUDGET)
    if isinstance(budget, bool) or not isinstance(budget, int) or not (MIN_BUDGET <= budget <= MAX_BUDGET):
        return _err(
            "invalid_budget",
            "budget_per_subgoal must be an integer in [%d, %d]" % (MIN_BUDGET, MAX_BUDGET),
        )
    capability_ids = payload.get("capability_ids")
    if isinstance(capability_ids, list) and all(isinstance(c, str) for c in capability_ids):
        remaining = SESSION_STEP_LIMIT - int(session.get("total_steps") or 0)
        if budget * len(capability_ids) > remaining:
            return _err(
                "invalid_budget",
                "repaired budget %d x %d exceeds the session's remaining %d steps"
                % (budget, len(capability_ids), max(0, remaining)),
            )
    # audit defaults to False: a repair never silently inherits audit rights.
    audit = bool(payload.get("audit", False))
    return validate_capability_list(session.get("scene_id"), capability_ids, audit)


def validate_evaluate_request(
    session: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    case: dict[str, Any] | None,
    request_id: Any,
) -> dict[str, Any] | None:
    """Validate one ``POST /evaluate`` body (ownership + cross-scene rules)."""

    if session is None:
        return _err("unknown_session", "no such session")
    if case is None:
        return _err("unknown_case", "no such case_id")
    if case.get("scene_id") != session.get("scene_id"):
        return _err(
            "cross_scene",
            "case scene %r != session scene %r" % (case.get("scene_id"), session.get("scene_id")),
        )
    # Evaluation reads the *live* scene, so the session must be idle: no plan
    # may be active and the session itself must be ready.  This closes the race
    # where an old completed plan is evaluated while a newer request is already
    # mutating the same scene.
    if session.get("state") != "ready" or session.get("active_request_id") is not None:
        return _err(
            "busy",
            "session is %s (active_request_id=%r); evaluation needs an idle ready session"
            % (session.get("state"), session.get("active_request_id")),
        )
    if not isinstance(request_id, str) or not request_id or plan is None:
        return _err("ownership", "an existing plan for this session owns the evaluation")
    if plan.get("session_id") != session.get("session_id") or plan.get("request_id") != request_id:
        return _err("ownership", "plan does not belong to this session")
    # Only a plan that has actually finished can be evaluated; a queued or
    # running plan is still mutating the scene.
    if plan.get("state") not in TERMINAL_PLAN_STATES:
        return _err("busy", "plan is %s; evaluation needs a terminal plan" % plan.get("state"))
    # The completed plan must describe the *current* scene version; otherwise the
    # scene has moved on since the plan finished and the evaluation is stale.
    if plan.get("scene_version") != session.get("scene_version"):
        return _err(
            "stale_evaluation",
            "plan scene_version %s != session scene_version %s"
            % (plan.get("scene_version"), session.get("scene_version")),
        )
    return None


def resolve_artifact_path(root: Path, relative: str) -> Path | None:
    """Resolve ``relative`` under ``root``; reject traversal and odd suffixes."""

    if not isinstance(relative, str) or not relative.strip():
        return None
    rel = relative.strip().lstrip("/")
    if not rel:
        return None
    candidate = Path(rel)
    if candidate.is_absolute():
        return None
    parts = candidate.parts
    if any(part in ("", ".", "..") for part in parts):
        return None
    # Never serve dot-files (credentials, .env, .git internals, ...).
    if any(part.startswith(".") for part in parts):
        return None
    if candidate.suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES:
        return None
    root_resolved = Path(root).resolve()
    target = (root_resolved / rel).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target


# --- the persistent native environment --------------------------------------


try:  # pragma: no cover - the real import needs the WSL lerobot tree
    from lerobot.envs.libero import LiberoEnv as LiberoEnv
except Exception:  # noqa: BLE001 - the GPU-free contract tests import without lerobot

    class LiberoEnv:  # type: ignore[no-redef]
        """Placeholder base used only when lerobot is unavailable.

        It exists so ``service.py`` (and therefore the contract tests) stays
        importable on a machine without the WSL environment.  Instantiating it
        is an error: the real service always runs where lerobot is installed.
        """

        metadata = {"render_modes": ["rgb_array"], "render_fps": VIDEO_FPS}

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "lerobot.envs.libero.LiberoEnv is unavailable in this interpreter"
            )


class PersistentLiberoEnv(LiberoEnv):
    """One LIBERO simulator that outlives many capability subgoals.

    Only ``_ensure_env`` is overridden: it builds the plain
    ``OffScreenRenderEnv`` with the native 256 px observation cameras, the
    native 20 Hz control rate and the native relative (delta) end-effector
    control, and additionally pins ``ignore_done=True`` / ``horizon=2000`` so
    the simulator never declares the episode finished on its own.  Everything
    else -- loading the native init state on ``reset``, the observation
    assembly and the wrappers -- stays exactly as the lerobot base class does
    it.  No reset happens here: the single logical reset of a session is
    performed by ``LiberoEnv.reset`` when the session is created.
    """

    def _ensure_env(self) -> None:  # noqa: D401 - mirrors the lerobot hook
        """Build the native ``OffScreenRenderEnv`` on first use.

        The official ``LiberoEnv`` exposes the task's BDDL path as the plain
        ``self._task_bddl_file`` field and has no ``self.task_suite``.  We build
        the exact same native environment as the base class -- 256 px cameras,
        20 Hz relative control -- with ``ignore_done=True`` / ``horizon=2000``
        so the simulator never declares the episode finished on its own.  No
        reset happens here: the single logical episode reset is
        ``LiberoEnv.reset``.
        """

        if self._env is not None:
            return
        try:
            from libero.libero.envs import OffScreenRenderEnv

            self._env = OffScreenRenderEnv(
                bddl_file_name=self._task_bddl_file,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
                control_freq=self.control_freq,
                hard_reset=self.hard_reset,
                ignore_done=True,
                horizon=SESSION_STEP_LIMIT,
            )
        except SceneError:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed, never fall back
            # The base ``LiberoEnv`` builder must never be used as a fallback:
            # building the wrong scene silently is worse than failing.
            raise SceneError(
                "invalid_fixture",
                "could not build the LIBERO environment from %r: %s"
                % (getattr(self, "_task_bddl_file", None), exc),
            ) from exc


# --- simulator interaction helpers (worker thread only) ----------------------


def _inner_env(env: Any) -> Any:
    """Return the robosuite environment wrapped by ``LiberoEnv``."""

    inner = getattr(getattr(env, "_env", None), "env", None)
    if inner is None:
        raise SceneError("invalid_fixture", "environment does not expose a robosuite env")
    return inner


def _object_joint(env: Any, object_id: str) -> str:
    """Return the *name* of ``object_id``'s free joint.

    Native LIBERO/robosuite stores ``objects_dict[object_id].joints`` as a list
    of joint-name strings; the object's free joint is ``joints[0]``.
    """

    objects = getattr(_inner_env(env), "objects_dict", None)
    if not isinstance(objects, dict) or object_id not in objects:
        raise SceneError("invalid_fixture", "unknown object %r" % object_id)
    joints = getattr(objects[object_id], "joints", None) or []
    if not joints or not isinstance(joints[0], str):
        raise SceneError("invalid_fixture", "object %r has no named joint" % object_id)
    return joints[0]


def _body_position(env: Any, object_id: str) -> np.ndarray:
    """Return the world centre of ``object_id`` via its native body index.

    The object id is *not* assumed to occur in ``body_names``: the index comes
    from ``inner.obj_body_id[object_id]`` and the centre from
    ``inner.sim.data.body_xpos``.
    """

    inner = _inner_env(env)
    body_map = getattr(inner, "obj_body_id", None)
    if not isinstance(body_map, dict) or object_id not in body_map:
        raise SceneError("invalid_fixture", "unknown body for object %r" % object_id)
    index = body_map[object_id]
    return np.asarray(inner.sim.data.body_xpos[index], dtype=np.float64)


def apply_scene_patch(env: Any, patch: dict[str, Any]) -> None:
    """Apply one catalog patch, then forward and settle the simulator.

    * ``offset`` translates the free joint position by ``xyz``;
    * ``place_on`` moves the object centre to the target object's centre plus
      ``[0, 0, 0.05]``.

    Only the position part ``qpos[:3]`` is written; ``qpos[3:7]`` (the
    orientation quaternion) is preserved.  ``sim.forward`` is called and the
    sim is settled with no-op actions; the caller refreshes the observation.
    """

    inner = _inner_env(env)
    object_id = patch["object_id"]
    joint_name = _object_joint(env, object_id)
    qpos = np.asarray(inner.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    if qpos.shape[0] < 3:
        raise SceneError("invalid_fixture", "joint %r carries no position" % joint_name)

    kind = patch.get("kind")
    if kind == "offset":
        xyz = np.asarray(patch["xyz"], dtype=np.float64).reshape(3)
        qpos[:3] = qpos[:3] + xyz
    elif kind == "place_on":
        target = _body_position(env, patch["target_id"])
        qpos[:3] = target + np.array([0.0, 0.0, 0.05], dtype=np.float64)
        # qpos[3:7] (the orientation quaternion) is deliberately untouched.
    else:
        raise SceneError("invalid_fixture", "unknown patch kind %r" % kind)

    inner.sim.data.set_joint_qpos(joint_name, qpos)
    inner.sim.forward()
    settle = np.zeros(ACTION_DIM, dtype=np.float64)
    for _ in range(SETTLE_STEPS):
        inner.step(settle)


def state_sha(env: Any) -> str:
    """SHA-256 of the raw flattened float64 simulator state.

    No rounding/quantisation is applied: a real change of a single float64
    element (even tiny) must produce a different digest.
    """

    sim = _inner_env(env).sim
    try:
        # A native MjSimState's ``flatten()`` yields the real numeric state; only
        # then is it coerced.  ``np.asarray(MjSimState)`` would silently produce
        # an object array and always fall through to the qpos/qvel fallback.
        flat = sim.get_state().flatten()
    except Exception:  # noqa: BLE001 - fall back to the raw qpos/qvel vectors
        flat = np.concatenate(
            [np.asarray(sim.data.qpos, dtype=np.float64), np.asarray(sim.data.qvel, dtype=np.float64)]
        )
    state = np.asarray(flat, dtype=np.float64).reshape(-1)
    # No rounding/quantisation: the raw float64 bytes are hashed verbatim.
    return _sha256_bytes(np.ascontiguousarray(state, dtype=np.float64).tobytes())


def object_positions(env: Any) -> dict[str, list[float]]:
    """Private {object_id: [x, y, z]} snapshot used by the independent oracle."""

    inner = _inner_env(env)
    positions: dict[str, list[float]] = {}
    objects = getattr(inner, "objects_dict", None) or {}
    for object_id, obj in objects.items():
        joints = getattr(obj, "joints", None) or []
        if not joints or not isinstance(joints[0], str):
            continue
        try:
            qpos = np.asarray(inner.sim.data.get_joint_qpos(joints[0]), dtype=np.float64)
        except Exception:  # noqa: BLE001 - skip objects without a readable joint
            continue
        positions[object_id] = [float(v) for v in qpos[:3]]
    return positions


def _object_states(inner: Any) -> dict[str, Any]:
    """Return the native object-state map (``inner.object_states_dict``)."""

    states = getattr(inner, "object_states_dict", None)
    return states if isinstance(states, dict) else {}


def eval_goal_predicate(env: Any, predicate: Any) -> bool:
    """Evaluate one declared goal predicate on the worker thread.

    The native signature is ``inner._eval_predicate(list_of_strings)``.  A
    predicate that cannot be evaluated raises ``SceneError('predicate_error')``
    -- it is never silently reported as false.  ``['not', ...]`` recursively
    negates only a *successful* boolean, so a failed inner query raises and can
    never be turned into success through negation.
    """

    try:
        predicate = [str(part) for part in predicate]
    except Exception as exc:  # noqa: BLE001
        raise SceneError("predicate_error", "malformed predicate %r: %s" % (predicate, exc)) from exc
    if not predicate:
        raise SceneError("predicate_error", "empty predicate")
    if predicate[0] == "not":
        if len(predicate) < 2:
            raise SceneError("predicate_error", "negation needs an argument")
        return not eval_goal_predicate(env, predicate[1:])

    inner = _inner_env(env)
    fn = getattr(inner, "_eval_predicate", None)
    if not callable(fn):
        raise SceneError("predicate_error", "environment has no _eval_predicate")
    try:
        result = fn(list(predicate))
    except SceneError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail loudly, never as success
        raise SceneError("predicate_error", "predicate %r failed: %s" % (predicate, exc)) from exc
    return bool(result)


def _to_uint8(frame: Any) -> np.ndarray:
    """Normalise a rendered frame to a contiguous HWC uint8 RGB array."""

    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if array.size and float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_uint8_image(frame: Any) -> np.ndarray:
    """Normalise a raw *camera* frame, flipping both H and W axes.

    Raw LIBERO/robosuite camera pixels arrive upside-down; both axes are
    flipped, exactly as the v1 render path did, so saved frames stay upright.
    """

    return _to_uint8(np.flip(np.asarray(frame), axis=(0, 1)))


def capture_vla_images(env: Any) -> dict[str, np.ndarray]:
    """Capture the agentview + wrist frames that the VLA actually consumes."""

    inner = _inner_env(env)
    images: dict[str, np.ndarray] = {}
    obs = None
    try:
        obs = inner._get_observations()
    except Exception:  # noqa: BLE001
        obs = None
    if isinstance(obs, dict):
        for key, view in (
            ("agentview_image", VIEW_AGENTVIEW),
            ("robot0_eye_in_hand_image", VIEW_WRIST),
        ):
            if key in obs:
                images[view] = _as_uint8_image(obs[key])
    return images


def render_extra_views(env: Any) -> list[tuple[str, np.ndarray]]:
    """Render the optional simulation cameras that actually exist."""

    sim = _inner_env(env).sim
    names = set(getattr(sim.model, "camera_names", []) or [])
    rendered: list[tuple[str, np.ndarray]] = []
    for name in EXTRA_VIEW_CANDIDATES:
        if name not in names:
            continue
        try:
            frame = sim.render(
                camera_name=name, width=OBS_WIDTH, height=OBS_HEIGHT, depth=False
            )
        except Exception:  # noqa: BLE001 - skip cameras the renderer rejects
            continue
        if frame is None:
            continue
        rendered.append((name, _as_uint8_image(frame)))
    return rendered


# --- state records -----------------------------------------------------------


class _Work:
    """One unit of worker-thread work (synchronous future or async plan)."""

    __slots__ = ("kind", "fn", "done", "result", "error", "cancelled")

    def __init__(self, kind: str, fn: Any) -> None:
        self.kind = kind
        self.fn = fn
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None
        self.cancelled = False


class SessionRecord:
    """One persistent scene session (public + private halves)."""

    def __init__(self, session_id: str, scene_id: str, scene: dict, seed: int,
                 init_state_index: int, run_dir: Path) -> None:
        self.session_id = session_id
        self.scene_id = scene_id
        self.scene = scene
        self.seed = seed
        self.init_state_index = init_state_index
        self.run_dir = Path(run_dir)
        self.state = "queued"  # queued -> ready -> running -> closed | error
        self.scene_version = 0
        self.env_instance_id: int | None = None
        self.episode_resets = 0
        self.policy_resets = 0
        self.total_steps = 0
        self.active_request_id: str | None = None
        self.error: str | None = None
        self.images: list[dict[str, Any]] = []
        self.latest_png: str | None = None
        # --- private (never returned by any public API) ---
        self.initial_state_path: str | None = None
        self.initial_state_hash: str | None = None
        self.xml_sha: str | None = None
        self.initial_positions: dict[str, list[float]] = {}
        self.initial_object_states: dict[str, Any] = {}
        self.camera_names: list[str] = []

    def capability_ids(self) -> list[str]:
        return [c["capability_id"] for c in catalog.scene_capabilities(self.scene_id)]

    def public(self) -> dict[str, Any]:
        return {
            "ok": True,
            "session_id": self.session_id,
            "scene_id": self.scene_id,
            "state": self.state,
            "scene_version": self.scene_version,
            "env_instance_id": self.env_instance_id,
            "episode_resets": self.episode_resets,
            "policy_resets": self.policy_resets,
            "total_steps": self.total_steps,
            "seed": self.seed,
            "init_state_index": self.init_state_index,
            "description": self.scene.get("description"),
            "storage_policy": dict(self.scene.get("storage_policy") or {}),
            "capabilities": self.capability_ids(),
            "images": [dict(image) for image in self.images],
            "latest_png": self.latest_png,
            "run_dir": str(self.run_dir),
            "active_request_id": self.active_request_id,
            "error": self.error,
        }


class PlanRecord:
    """One plan request.  ``pending`` is what the worker still has to run."""

    def __init__(self, request_id: str, session_id: str, payload: dict[str, Any]) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.decision = payload.get("decision")
        self.capability_ids = list(payload.get("capability_ids") or [])
        # The goals declared by the *original* request are snapshotted here, as
        # an independent copy, and are never mutated by resume, cancellation or
        # execution: a repair may reorder or add recovery capabilities, but it
        # can never erase what the plan initially declared.
        self.original_capability_ids = list(self.capability_ids)
        self.pending_capability_ids = list(self.capability_ids)
        self.completed_capability_ids: list[str] = []
        self.rationale = str(payload.get("rationale") or "")
        self.audit = bool(payload.get("audit", False))
        self.budget_per_subgoal = int(payload.get("budget_per_subgoal", DEFAULT_BUDGET))
        self.scene_version = payload.get("scene_version")
        self.state = "queued"
        self.job_ids: list[str] = []
        self.plan_success: bool | None = None
        self.regressions: list[list[str]] = []
        self.error: str | None = None
        self.repair_history: list[dict[str, Any]] = []
        self.cancel_event = threading.Event()
        self.started_utc: str | None = None
        self.finished_utc: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "ok": True,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "state": self.state,
            "decision": self.decision,
            "capability_ids": list(self.capability_ids),
            "original_capability_ids": list(self.original_capability_ids),
            "rationale": self.rationale,
            "job_ids": list(self.job_ids),
            "completed_capability_ids": list(self.completed_capability_ids),
            "pending_capability_ids": list(self.pending_capability_ids),
            "plan_success": self.plan_success,
            "regressions": [list(goal) for goal in self.regressions],
            "error": self.error,
            "repair_history": [dict(entry) for entry in self.repair_history],
            "scene_version": self.scene_version,
        }


class JobRecord:
    """One capability execution inside a session."""

    def __init__(self, job_id: str, request_id: str, session_id: str, capability_id: str,
                 run_dir: Path) -> None:
        self.job_id = job_id
        self.request_id = request_id
        self.session_id = session_id
        self.capability_id = capability_id
        self.run_dir = Path(run_dir)
        self.state = "queued"  # queued -> running -> completed | error | cancelled
        self.steps = 0
        self.total_steps = 0
        self.success: bool | None = None
        self.ended_reason: str | None = None
        self.error: str | None = None
        self.instruction: str | None = None
        self.scene_version_before: int | None = None
        self.scene_version_after: int | None = None
        self.state_before_sha: str | None = None
        self.state_after_sha: str | None = None
        self.env_instance_id: int | None = None
        self.episode_resets: int | None = None
        self.latest_png: str | None = None
        self.rollout_path: str | None = None
        self.wall_s: float | None = None
        self._t0: float | None = None

    def public(self) -> dict[str, Any]:
        wall = self.wall_s
        if wall is None and self._t0 is not None and self.state == "running":
            wall = round(time.monotonic() - self._t0, 3)
        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "capability_id": self.capability_id,
            "state": self.state,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "success": self.success,
            "ended_reason": self.ended_reason,
            "error": self.error,
            "instruction": self.instruction,
            "scene_version_before": self.scene_version_before,
            "scene_version_after": self.scene_version_after,
            "state_before_sha": self.state_before_sha,
            "state_after_sha": self.state_after_sha,
            "env_instance_id": self.env_instance_id,
            "episode_resets": self.episode_resets,
            "run_dir": str(self.run_dir),
            "latest_png": self.latest_png,
            "rollout_path": self.rollout_path,
            "wall_s": wall,
        }


# --- the service -------------------------------------------------------------


class SceneService:
    """Owns one persistent scene, one worker thread and all request state.

    All LIBERO/MuJoCo/CUDA access happens on the worker thread.  Every public
    method is safe to call from an HTTP thread: it only reads lock-protected
    state or pushes work onto the queue.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        run_root: str = DEFAULT_RUN_ROOT,
        *,
        env_factory: Any = None,
        policy_loader: Any = None,
        action_function: Any = None,
        batch_builder: Any = None,
    ) -> None:
        self.model_path = str(model_path)
        self.run_root = Path(run_root)

        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue()
        self._sessions: dict[str, SessionRecord] = {}
        self._session_order: list[str] = []
        self._plans: dict[str, PlanRecord] = {}
        self._plan_order: list[str] = []
        self._jobs: dict[str, JobRecord] = {}
        self._job_order: list[str] = []

        self._ready = False
        self._worker_error: str | None = None
        self._model_revision = MODEL_REVISION_DEFAULT
        self._active_request_id: str | None = None

        # Worker-thread-owned simulation state.
        self._env: Any = None
        self._env_session_id: str | None = None
        self._env_instance_counter = 0
        self._current_env_instance_id: int | None = None
        self._last_obs: Any = None
        self._total_steps = 0
        self._processor: Any = None

        # Test seams (None -> the real lerobot/LIBERO path).
        self._env_factory = env_factory
        self._policy_loader = policy_loader
        self._action_function = action_function
        self._batch_builder = batch_builder

        self._stop = threading.Event()
        # Instantiated, but never started: only ``_load_policy`` is reused, and
        # only on the worker thread.
        self._v1 = v1_libero_service.LiberoService(self.model_path, str(self.run_root))
        self._worker = threading.Thread(target=self._worker_loop, name="scene-worker", daemon=True)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._model_revision = v1_libero_service._detect_model_revision(self.model_path)
        self._worker.start()
        log("worker started (model_path=%s, run_root=%s)" % (self.model_path, self.run_root))

    def stop(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=60.0)

    # -- HTTP-facing reads ----------------------------------------------------

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = self._active_plan_locked()
            return {
                "ok": True,
                "backend": BACKEND_NAME,
                "robot": ROBOT_NAME,
                "workflow": WORKFLOW_NAME,
                "model_revision": self._model_revision,
                "ready": bool(self._ready),
                "worker_error": self._worker_error,
                "model_path": self.model_path,
                "device": "cuda",
                "n_action_steps": getattr(self._v1, "_n_action_steps", None),
                "dtype": getattr(self._v1, "_dtype", None),
                "control_mode": CONTROL_MODE,
                "session_step_limit": SESSION_STEP_LIMIT,
                "active_request_id": active.request_id if active is not None else None,
            }

    def scenes(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for scene_id, scene in catalog.SCENES.items():
            entries.append(
                {
                    "scene_id": scene_id,
                    "label": scene.get("label"),
                    "suite": scene.get("suite"),
                    "task_id": scene.get("task_id"),
                    "variant": scene.get("variant"),
                    "description": scene.get("description"),
                    "storage_policy": dict(scene.get("storage_policy") or {}),
                    "capabilities": [c["capability_id"] for c in catalog.scene_capabilities(scene_id)],
                }
            )
        return {"ok": True, "workflow": WORKFLOW_NAME, "n_scenes": len(entries), "scenes": entries}

    def session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._sessions.get(session_id)
            return record.public() if record is not None else None

    def plan(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            plan = self._plans.get(request_id)
            return plan.public() if plan is not None else None

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.public() if record is not None else None

    # -- worker plumbing ------------------------------------------------------

    def _require_ready(self) -> None:
        if not self._ready:
            raise NotReadyError(self._worker_error or "policy not loaded")

    def _active_plan_locked(self) -> PlanRecord | None:
        if self._active_request_id is None:
            return None
        plan = self._plans.get(self._active_request_id)
        if plan is None or plan.state not in ACTIVE_PLAN_STATES:
            return None
        return plan

    def _sync_work(self, kind: str, fn: Any, timeout: float = WORKER_WAIT_S) -> dict[str, Any]:
        """Queue a worker task and wait for it -- never holding ``self._lock``."""

        if not self._worker.is_alive():
            return _err("not_ready", "worker thread is not running")
        work = _Work(kind, fn)
        self._queue.put(work)
        if not work.done.wait(timeout):
            work.cancelled = True
            return _err("busy", "worker did not finish within %.0fs" % timeout)
        if work.error is not None:
            if isinstance(work.error, NotReadyError):
                return _err("not_ready", str(work.error))
            return _err("internal_error", _format_exc(work.error))
        if isinstance(work.result, dict):
            return work.result
        return _err("internal_error", "worker returned a non-dict result")

    def _worker_loop(self) -> None:
        try:
            self._load_policy()
        except BaseException as exc:  # noqa: BLE001 - report, never crash silently
            message = _format_exc(exc)
            with self._lock:
                self._worker_error = message
                self._ready = False
            log("policy load FAILED:\n%s" % message)
        else:
            with self._lock:
                self._ready = True
            log("policy ready (n_action_steps=%s)" % getattr(self._v1, "_n_action_steps", None))

        while not self._stop.is_set():
            try:
                work = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if work.cancelled:
                work.done.set()
                continue
            try:
                work.result = work.fn()
            except NotReadyError as exc:
                work.error = exc
            except SceneError as exc:
                # The worker owns the simulator: a failed session creation is
                # torn down here, on the worker thread, before the error result
                # is handed back to the (MuJoCo-free) HTTP caller.
                if work.kind == "create_session":
                    self._close_env()
                work.result = _err(exc.reason, exc.detail)
            except BaseException as exc:  # noqa: BLE001 - last-resort guard
                if work.kind == "create_session":
                    self._close_env()
                work.error = exc
                log("worker task %s crashed:\n%s" % (work.kind, _format_exc(exc)))
            finally:
                work.done.set()

    def _load_policy(self) -> None:
        if self._policy_loader is not None:
            self._policy_loader(self)
            return
        self._v1._load_policy()

    def _reset_policy_queues(self) -> None:
        """Clear only capability-scoped state -- never the environment."""

        policy = getattr(self._v1, "_policy", None)
        if policy is not None and hasattr(policy, "reset"):
            policy.reset()
        for name in ("_pre", "_post"):
            processor = getattr(self._v1, name, None)
            if processor is not None and hasattr(processor, "reset"):
                processor.reset()

    # -- environment construction --------------------------------------------

    def _build_env(self, suite_name: str, task_id: int, seed: int, init_state_index: int) -> Any:
        if self._env_factory is not None:
            return self._env_factory(
                suite_name=suite_name,
                task_id=task_id,
                seed=seed,
                init_state_index=init_state_index,
            )
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[suite_name]()
        return PersistentLiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            observation_width=OBS_WIDTH,
            observation_height=OBS_HEIGHT,
            control_mode=CONTROL_MODE,
            init_states=True,
            episode_index=init_state_index,
            hard_reset=True,
            num_steps_wait=NUM_STEPS_WAIT,
        )

    def _close_env(self) -> None:
        """Tear down the live simulator and mark its session closed."""

        env, self._env = self._env, None
        session_id, self._env_session_id = self._env_session_id, None
        self._current_env_instance_id = None
        self._last_obs = None
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                log("env.close failed: %s" % exc)
        if session_id is not None:
            with self._lock:
                record = self._sessions.get(session_id)
                if record is not None and record.state != "error":
                    record.state = "closed"

    def _refresh_observation(self, env: Any) -> Any:
        """Re-read the current observation *without* ever stepping the simulator.

        The real ``LiberoEnv`` wraps a native robosuite env whose
        ``_get_observations(force_update=True)`` returns the raw observation and
        whose ``env._format_raw_obs`` turns it into the env's observation dict;
        the native getter supports ``force_update``.  A fake env (used only by
        the GPU-free tests) may expose a plain, argument-less getter instead, so
        a ``TypeError`` from ``force_update`` is retried without it purely for
        mock compatibility.  If no observation can be produced the environment
        is unusable, so this fails closed with ``invalid_fixture`` rather than
        silently taking a hidden zero-action step.
        """

        inner = getattr(getattr(env, "_env", None), "env", None)
        getter = getattr(inner, "_get_observations", None) if inner is not None else None
        formatter = getattr(env, "_format_raw_obs", None)
        if callable(getter):
            try:
                raw = getter(force_update=True)
            except TypeError:
                # Mock compatibility only: a fake getter need not accept the
                # native ``force_update`` keyword.
                try:
                    raw = getter()
                except Exception as exc:  # noqa: BLE001
                    raise SceneError(
                        "invalid_fixture", "observation refresh failed: %s" % exc
                    ) from exc
            except Exception as exc:  # noqa: BLE001
                raise SceneError(
                    "invalid_fixture", "observation refresh failed: %s" % exc
                ) from exc
            if raw is not None:
                if callable(formatter):
                    try:
                        return formatter(raw)
                    except Exception as exc:  # noqa: BLE001
                        raise SceneError(
                            "invalid_fixture", "observation formatting failed: %s" % exc
                        ) from exc
                return raw

        # Mock-compatible plain getters on the env or its wrapped inner env.
        for source in (env, inner):
            if source is None:
                continue
            for name in ("_get_observation", "_get_obs"):
                fn = getattr(source, name, None)
                if not callable(fn):
                    continue
                try:
                    value = fn()
                except Exception as exc:  # noqa: BLE001
                    raise SceneError(
                        "invalid_fixture", "observation refresh failed: %s" % exc
                    ) from exc
                if value is not None:
                    return value

        raise SceneError(
            "invalid_fixture",
            "cannot refresh the observation without stepping the simulator",
        )

    def _snapshot_session_state(self, record: SessionRecord, env: Any) -> None:
        """Persist the initial simulator state, XML hash and object positions."""

        inner = _inner_env(env)
        sim = inner.sim
        try:
            # Native MjSimState: flatten() exposes the real numeric state.
            state = np.asarray(sim.get_state().flatten(), dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            state = np.concatenate(
                [np.asarray(sim.data.qpos, dtype=np.float64), np.asarray(sim.data.qvel, dtype=np.float64)]
            )
        state_path = record.run_dir / "initial_state.npy"
        np.save(str(state_path), state)
        record.initial_state_path = str(state_path)
        # Private equal-state digest kept with the seed for independent checks.
        record.initial_state_hash = state_sha(env)
        try:
            xml = sim.model.get_xml()
            record.xml_sha = _sha256_bytes(xml.encode("utf-8") if isinstance(xml, str) else bytes(xml))
        except Exception:  # noqa: BLE001
            record.xml_sha = None
        record.initial_positions = object_positions(env)
        record.initial_object_states = dict(_object_states(inner))
        record.camera_names = list(getattr(sim.model, "camera_names", []) or [])

    def _update_images(self, record: SessionRecord, env: Any) -> None:
        """Refresh the cached agentview/wrist PNGs and the main image."""

        images = capture_vla_images(env)
        entries: list[dict[str, Any]] = []
        for view, frame in images.items():
            path = record.run_dir / ("%s.png" % view)
            try:
                _save_png(path, frame)
            except Exception as exc:  # noqa: BLE001
                log("image save failed (%s): %s" % (view, exc))
                continue
            entries.append(
                {
                    "view": view,
                    "image_path": str(path),
                    "sha256": _sha256_bytes(np.ascontiguousarray(frame).tobytes()),
                    "kind": "vla_observation",
                }
            )
        main = images.get(VIEW_AGENTVIEW)
        if main is None and images:
            main = next(iter(images.values()))
        if main is not None:
            main_path = record.run_dir / "latest.png"
            try:
                _save_png(main_path, main)
                record.latest_png = str(main_path)
            except Exception as exc:  # noqa: BLE001
                log("main image save failed: %s" % exc)
        with self._lock:
            record.images = entries

    # -- session creation -----------------------------------------------------

    def create_session(self, scene_id: str, seed: int = 0, init_state_index: int = 0) -> dict[str, Any]:
        if not isinstance(scene_id, str) or not scene_id:
            return _err("invalid_body", "scene_id must be a non-empty string")
        if scene_id not in catalog.SCENES:
            return _err("unknown_scene", "unknown scene_id %r" % (scene_id,))
        if isinstance(seed, bool) or not isinstance(seed, int):
            return _err("invalid_body", "seed must be an integer")
        if isinstance(init_state_index, bool) or not isinstance(init_state_index, int) or init_state_index < 0:
            return _err("invalid_body", "init_state_index must be a non-negative integer")

        with self._lock:
            if self._active_plan_locked() is not None:
                return _err("busy", "a plan is active; finish or cancel it first")
            session_id = uuid.uuid4().hex
            run_dir = self.run_root / ("%s_%s_%s" % (_utc_stamp(), scene_id, session_id[:8]))
            record = SessionRecord(
                session_id, scene_id, catalog.SCENES[scene_id], seed, init_state_index, run_dir
            )
            self._sessions[session_id] = record
            self._session_order.append(session_id)

        result = self._sync_work(
            "create_session",
            lambda: self._do_create_session(record, seed, init_state_index),
        )
        if not result.get("ok"):
            # A failed creation must leave no live environment or busy flags
            # behind.  The *worker* already tore the simulator down (see
            # ``_worker_loop``); this HTTP-facing thread must never touch MuJoCo,
            # so it only records the terminal state.
            with self._lock:
                record.active_request_id = None
                record.state = "error"
                record.error = result.get("detail") or result.get("reason")
            return result
        return result

    def _do_create_session(self, record: SessionRecord, seed: int, init_state_index: int) -> dict[str, Any]:
        self._require_ready()
        scene = record.scene
        record.run_dir.mkdir(parents=True, exist_ok=True)

        # A new session is only allowed with no active plan; tear down the old.
        self._close_env()

        # Reproducibility: seed every RNG before any environment construction,
        # exactly as v1 did.  A failed construction must leave nothing behind.
        _seed_everything(seed)

        env = None
        try:
            env = self._build_env(scene["suite"], int(scene["task_id"]), seed, init_state_index)
            obs, _reset_info = env.reset(seed=seed)  # the single logical reset
        except SceneError:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
            raise
        except BaseException as exc:  # noqa: BLE001
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
            raise SceneError("internal_error", _format_exc(exc)) from exc

        self._env = env
        self._env_session_id = record.session_id
        self._env_instance_counter += 1
        env_instance_id = self._env_instance_counter
        self._current_env_instance_id = env_instance_id
        self._last_obs = obs
        self._total_steps = 0

        try:
            patches = list(scene.get("patches") or [])
            for patch in patches:
                apply_scene_patch(env, patch)
            if patches:
                refreshed = self._refresh_observation(env)
                if refreshed is not None:
                    self._last_obs = refreshed
                for patch in patches:
                    if patch.get("kind") != "place_on":
                        continue
                    goal = ["on", patch["object_id"], patch["target_id"]]
                    if not eval_goal_predicate(env, goal):
                        raise SceneError(
                            "invalid_fixture",
                            "%s is not on %s after the patch"
                            % (patch["object_id"], patch["target_id"]),
                        )
        except SceneError:
            # Close the partially created environment; leave no busy/active
            # state behind (the caller marks the session 'error').
            self._close_env()
            raise

        self._snapshot_session_state(record, env)
        self._reset_policy_queues()

        with self._lock:
            record.state = "ready"
            record.env_instance_id = env_instance_id
            record.episode_resets = 1
            record.policy_resets = 0
            record.total_steps = 0
            record.scene_version = 0
            record.error = None
        self._update_images(record, env)
        with self._lock:
            return record.public()

    # -- observation ----------------------------------------------------------

    def observe(self, session_id: str, extra_views: bool = False) -> dict[str, Any]:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return _err("unknown_session", "no such session")
            if not extra_views:
                return record.public()
            if self._active_plan_locked() is not None:
                return _err("busy", "extra views are only available while idle")
            if record.state != "ready":
                return _err("busy", "session is %s; extra views are idle-only" % record.state)
        return self._sync_work("observe", lambda: self._do_extra_views(record))

    def _do_extra_views(self, record: SessionRecord) -> dict[str, Any]:
        self._require_ready()
        if self._env is None or self._env_session_id != record.session_id:
            return _err("unknown_session", "no live environment for this session")
        entries: list[dict[str, Any]] = []
        for name, frame in render_extra_views(self._env):
            path = record.run_dir / ("extra_%s.png" % name)
            try:
                _save_png(path, frame)
            except Exception as exc:  # noqa: BLE001
                log("extra view save failed (%s): %s" % (name, exc))
                continue
            entries.append(
                {
                    "view": name,
                    "image_path": str(path),
                    "sha256": _sha256_bytes(np.ascontiguousarray(frame).tobytes()),
                    "kind": "simulation_extra_view",
                }
            )
        payload = record.public()
        payload["extra_views"] = entries
        payload["extra_views_available"] = bool(entries)
        return payload

    # -- policy interfacing (worker thread only) ------------------------------

    def _observation_batch(self, obs: Any, instruction: str) -> Any:
        if self._batch_builder is not None:
            return self._batch_builder(obs, instruction)
        from lerobot.envs.utils import preprocess_observation
        from lerobot.processor.env_processor import LiberoProcessorStep

        if self._processor is None:
            self._processor = LiberoProcessorStep()
        batch = preprocess_observation(_batch_observation(obs))
        batch = self._processor._process_observation(batch)
        batch["task"] = [instruction]
        return batch

    def _select_action(self, batch: Any) -> np.ndarray:
        if self._action_function is not None:
            return np.asarray(self._action_function(batch), dtype=np.float64).reshape(-1)
        import torch

        batch = self._v1._pre(batch)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = self._v1._policy.select_action(batch)
        action = self._v1._post(action)
        return np.asarray(action.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

    # -- plan submission / execution -----------------------------------------

    def submit_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return _err("invalid_body", "body must be a JSON object")
        session_id = payload.get("session_id")
        with self._lock:
            record = self._sessions.get(session_id) if isinstance(session_id, str) else None
            session_public = record.public() if record is not None else None
            active = self._active_plan_locked()
            active_public = active.public() if active is not None else None
            error = validate_plan_request(session_public, active_public, payload)
            if error is not None:
                return error
            # A request_id is a permanent identifier: once a plan (completed,
            # blocked, cancelled or errored) has used it, the same id may never
            # be reused.  History is never overwritten and replacement is not
            # idempotent.  (An active plan is already reported as ``busy`` above,
            # so busy correctly takes precedence.)
            request_id = payload["request_id"]
            if request_id in self._plans:
                return _err(
                    "request_conflict",
                    "request_id %r already exists and cannot be reused" % request_id,
                )
            if not self._worker.is_alive():
                return _err("not_ready", "worker thread is not running")
            # Never hardcode routing from the whole user request: the caller's
            # decision/capability list is executed verbatim, only validated.
            plan = PlanRecord(payload["request_id"], session_id, payload)
            self._plans[plan.request_id] = plan
            self._plan_order.append(plan.request_id)
            self._active_request_id = plan.request_id
            record.active_request_id = plan.request_id
        self._queue.put(_Work("plan", lambda: self._do_run_plan(plan)))
        return plan.public()

    def cancel(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            plan = self._plans.get(request_id)
            if plan is None:
                return _err("unknown_plan", "no such request_id")
            if plan.state not in TERMINAL_PLAN_STATES:
                plan.cancel_event.set()
                if plan.state == "queued":
                    self._terminalise_plan_locked(plan, "cancelled", None)
            return plan.public()

    def resume_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return _err("invalid_body", "body must be a JSON object")
        request_id = payload.get("request_id")
        with self._lock:
            plan = self._plans.get(request_id) if isinstance(request_id, str) else None
            session = self._sessions.get(payload.get("session_id")) if isinstance(payload.get("session_id"), str) else None
            # Resume is only for blocked plans.  audit is *never* inherited
            # implicitly: it must be passed explicitly, so a repair can never
            # silently keep audit-only rights.  The effective budget defaults to
            # the plan's already-validated budget.
            resume_payload = dict(payload)
            if "budget_per_subgoal" not in resume_payload and plan is not None:
                resume_payload["budget_per_subgoal"] = plan.budget_per_subgoal
            error = validate_resume_request(
                session.public() if session is not None else None,
                plan.public() if plan is not None else None,
                resume_payload,
            )
            if error is not None:
                return error
            new_capabilities = list(payload["capability_ids"])
            # ``capability_ids`` is ordered *unique* goal identity: a completed
            # goal that the repair also re-declares (a regressed goal being
            # re-executed) must be counted once, so the capability cap is not
            # consumed twice by the same goal.  ``pending_capability_ids`` stays
            # the repair list verbatim -- that is the next execution schedule,
            # so re-declaring an already-completed goal runs it again.
            merged_capabilities = list(dict.fromkeys(list(plan.completed_capability_ids) + new_capabilities))
            if len(merged_capabilities) > MAX_CAPABILITIES:
                return _err("invalid_capabilities", "repaired plan would exceed %d capabilities" % MAX_CAPABILITIES)
            plan.repair_history.append(
                {
                    "capability_ids": list(new_capabilities),
                    "rationale": str(payload.get("rationale") or ""),
                    "at": _now_utc(),
                }
            )
            plan.capability_ids = merged_capabilities
            plan.pending_capability_ids = list(new_capabilities)
            # The resumed plan targets the *current* scene version (validated
            # above to equal the session's), so the worker re-check passes.
            plan.scene_version = int(resume_payload["scene_version"])
            plan.budget_per_subgoal = int(resume_payload["budget_per_subgoal"])
            plan.rationale = str(payload.get("rationale") or plan.rationale)
            plan.error = None
            plan.cancel_event = threading.Event()
            plan.state = "queued"
            self._active_request_id = plan.request_id
            if session is not None:
                session.active_request_id = plan.request_id
        self._queue.put(_Work("plan", lambda: self._do_run_plan(plan)))
        return plan.public()

    def _finish_plan_locked(self, plan: PlanRecord) -> None:
        if self._active_request_id == plan.request_id:
            self._active_request_id = None
        record = self._sessions.get(plan.session_id)
        if record is not None and record.active_request_id == plan.request_id:
            record.active_request_id = None
            if record.state == "running":
                record.state = "ready"

    def _terminalise_plan_locked(self, plan: PlanRecord, state: str, error: str | None) -> None:
        """Move a plan to a terminal state and sync its scene version.

        Every terminal path (completed/blocked/error/cancelled and the
        clarify/unsupported completion) records the owning session's *current*
        scene version on the plan, so a normal evaluation matches and a later
        scene change correctly invalidates the finished plan.
        """

        record = self._sessions.get(plan.session_id)
        if record is not None:
            plan.scene_version = record.scene_version
        plan.state = state
        if error is not None:
            plan.error = error
        plan.finished_utc = _now_utc()
        self._finish_plan_locked(plan)

    def _do_run_plan(self, plan: PlanRecord) -> dict[str, Any]:
        try:
            self._require_ready()
            with self._lock:
                if plan.state == "cancelled":
                    self._finish_plan_locked(plan)
                    return plan.public()
                record = self._sessions.get(plan.session_id)
                if record is None:
                    self._terminalise_plan_locked(
                        plan, "error", "session vanished before the plan started"
                    )
                    return plan.public()
                # Worker-side re-validation of scene support + version + budget.
                worker_error = None
                if plan.decision == "execute":
                    worker_error = validate_capability_list(record.scene_id, plan.capability_ids, plan.audit)
                if (
                    worker_error is None
                    and isinstance(plan.scene_version, int)
                    and not isinstance(plan.scene_version, bool)
                    and plan.scene_version != record.scene_version
                ):
                    worker_error = _err(
                        "stale_scene_version",
                        "session scene_version is %s, plan submitted for %s"
                        % (record.scene_version, plan.scene_version),
                    )
                if worker_error is None and plan.budget_per_subgoal * len(plan.pending_capability_ids) > SESSION_STEP_LIMIT:
                    worker_error = _err("invalid_budget", "plan budget exceeds the session limit")
                if worker_error is not None:
                    self._terminalise_plan_locked(
                        plan, "error", worker_error.get("detail") or worker_error.get("reason")
                    )
                    return worker_error
                plan.state = "running"
                plan.started_utc = _now_utc()
                record.state = "running"

            if plan.decision != "execute":
                with self._lock:
                    plan.plan_success = None
                    self._terminalise_plan_locked(plan, "completed", None)
                return plan.public()

            for capability_id in list(plan.pending_capability_ids):
                if self._stop.is_set() or plan.cancel_event.is_set():
                    with self._lock:
                        self._terminalise_plan_locked(plan, "cancelled", None)
                    return plan.public()
                job = self._start_job(record, plan, capability_id)
                outcome = self._run_capability(record, plan, job, capability_id)
                with self._lock:
                    if capability_id in plan.pending_capability_ids:
                        plan.pending_capability_ids.remove(capability_id)
                if outcome.get("cancelled"):
                    with self._lock:
                        self._terminalise_plan_locked(plan, "cancelled", None)
                    return plan.public()
                if not outcome.get("ok"):
                    with self._lock:
                        self._terminalise_plan_locked(
                            plan, "blocked", outcome.get("detail") or outcome.get("reason")
                        )
                    return plan.public()
                with self._lock:
                    if capability_id not in plan.completed_capability_ids:
                        plan.completed_capability_ids.append(capability_id)

            self._finalise_plan_success(plan)
            return plan.public()
        except NotReadyError as exc:
            with self._lock:
                self._terminalise_plan_locked(plan, "error", str(exc))
            return _err("not_ready", str(exc))
        except BaseException as exc:  # noqa: BLE001
            with self._lock:
                self._terminalise_plan_locked(plan, "error", _format_exc(exc))
            log("plan %s crashed:\n%s" % (plan.request_id, _format_exc(exc)))
            return _err("internal_error", _format_exc(exc))

    def _finalise_plan_success(self, plan: PlanRecord) -> None:
        # The originally declared capability ids come first, then the current
        # (possibly repaired) ones; ordered deduplication keeps the first
        # occurrence of each.  A repair may reorder or add recovery
        # capabilities, but the initial goals can never be dropped from the
        # final AND: an omitted failed goal still has to be satisfied.
        declared_ids: list[str] = []
        seen_ids: set[str] = set()
        for capability_id in list(plan.original_capability_ids) + list(plan.capability_ids):
            if capability_id in seen_ids:
                continue
            seen_ids.add(capability_id)
            declared_ids.append(capability_id)
        declared = catalog.deduplicate_goals(declared_ids)
        values: dict[str, bool] = {}
        missing_declared: list[list[str]] = []
        for goal in declared:
            key = catalog.goal_key(goal)
            if key not in values:
                try:
                    values[key] = bool(eval_goal_predicate(self._env, goal))
                except SceneError:  # unknown truth can never be success
                    values[key] = False
            if not values.get(key, False):
                missing_declared.append(list(goal))
        plan_success = all(values.get(catalog.goal_key(g), False) for g in declared) if declared else None

        # Actual completed-goal regressions are preserved separately.
        regressions: list[list[str]] = []
        seen: set[str] = set()
        for capability_id in plan.completed_capability_ids:
            capability = catalog.CAPABILITIES.get(capability_id) or {}
            for goal in capability.get("goals") or []:
                key = catalog.goal_key(goal)
                if key in seen:
                    continue
                seen.add(key)
                if not values.get(key, False):
                    regressions.append(list(goal))

        with self._lock:
            plan.plan_success = plan_success
            plan.regressions = regressions
            if missing_declared:
                error = "final declared goals not satisfied: %s" % (
                    [catalog.goal_key(g) for g in missing_declared],
                )
                self._terminalise_plan_locked(plan, "blocked", error)
            elif regressions:
                error = "completed goal regressed: %s" % ([catalog.goal_key(g) for g in regressions],)
                self._terminalise_plan_locked(plan, "blocked", error)
            else:
                self._terminalise_plan_locked(plan, "completed", None)

    # -- capability execution -------------------------------------------------

    def _start_job(self, record: SessionRecord, plan: PlanRecord, capability_id: str) -> JobRecord:
        capability = catalog.CAPABILITIES[capability_id]
        job_id = uuid.uuid4().hex
        index = len(plan.completed_capability_ids) + 1
        job_dir = record.run_dir / ("cap_%02d_%s_%s" % (index, capability_id, job_id[:8]))
        job = JobRecord(job_id, plan.request_id, record.session_id, capability_id, job_dir)
        job.instruction = str(capability["instruction"])
        job.env_instance_id = self._current_env_instance_id
        job.episode_resets = 1
        with self._lock:
            self._jobs[job_id] = job
            self._job_order.append(job_id)
            plan.job_ids.append(job_id)
        return job

    def _target_occupant(self, env: Any, capability: dict[str, Any]) -> str | None:
        target = capability.get("target_id")
        if not target:
            return None
        objects = getattr(_inner_env(env), "objects_dict", None) or {}
        for object_id in objects:
            if object_id in (capability.get("object_id"), target):
                continue
            try:
                if eval_goal_predicate(env, ["on", object_id, target]):
                    return object_id
            except Exception:  # noqa: BLE001
                continue
        return None

    def _safe_render(self, env: Any) -> np.ndarray | None:
        try:
            frame = env.render()
        except Exception:  # noqa: BLE001
            return None
        if frame is None:
            return None
        try:
            return _to_uint8(frame)
        except Exception:  # noqa: BLE001
            return None

    def _save_frame(self, path: Path, frame: Any) -> None:
        try:
            _save_png(path, frame)
        except Exception as exc:  # noqa: BLE001
            log("frame save failed (%s): %s" % (path, exc))

    def _run_capability(self, session: SessionRecord, plan: PlanRecord, job: JobRecord,
                        capability_id: str) -> dict[str, Any]:
        capability = catalog.CAPABILITIES[capability_id]
        instruction = str(capability["instruction"])
        goals = [list(goal) for goal in capability["goals"]]
        budget = int(plan.budget_per_subgoal)
        started = time.monotonic()
        job.run_dir.mkdir(parents=True, exist_ok=True)
        events_path = job.run_dir / "events.jsonl"
        first_png = job.run_dir / "first.png"
        last_png = job.run_dir / "last.png"
        latest_png = job.run_dir / "latest.png"
        rollout_path = job.run_dir / "rollout.mp4"

        frames: list[np.ndarray] = []
        steps = 0
        success = False
        ended_reason: str | None = None
        error_text: str | None = None

        with self._lock:
            job.state = "running"
            job.instruction = instruction
            job.scene_version_before = session.scene_version
            job.latest_png = str(latest_png)
            job.rollout_path = str(rollout_path)
            job._t0 = started

        try:
            self._require_ready()
            env = self._env
            if env is None or self._env_session_id != session.session_id:
                raise SceneError("unknown_session", "no live environment for this session")

            before_sha = state_sha(env)
            self._reset_policy_queues()
            with self._lock:
                job.state_before_sha = before_sha
                session.policy_resets += 1

            initial_frame = self._safe_render(env)
            if initial_frame is not None:
                frames.append(initial_frame)
                self._save_frame(first_png, initial_frame)

            already_satisfied = bool(goals) and all(eval_goal_predicate(env, goal) for goal in goals)
            occupant = None
            if (
                not already_satisfied
                and capability.get("exclusive_target")
                and capability.get("target_id")
            ):
                occupant = self._target_occupant(env, capability)

            if already_satisfied:
                success = True
                ended_reason = "already_satisfied"
            elif occupant is not None:
                success = False
                ended_reason = "target_occupied"
                error_text = "target %s already holds %s" % (capability.get("target_id"), occupant)
            else:
                remaining = SESSION_STEP_LIMIT - self._total_steps
                limit = min(budget, remaining)
                if limit <= 0:
                    ended_reason = "session_limit"
                else:
                    obs = self._last_obs
                    consecutive = 0
                    with open(events_path, "w", encoding="utf-8") as events_file:
                        for step in range(1, limit + 1):
                            if self._stop.is_set():
                                ended_reason = "shutdown"
                                break
                            if plan.cancel_event.is_set():
                                ended_reason = "cancelled"
                                break
                            if self._total_steps >= SESSION_STEP_LIMIT:
                                ended_reason = "session_limit"
                                break
                            batch = self._observation_batch(obs, instruction)
                            action = self._select_action(batch)
                            if action.shape != (ACTION_DIM,):
                                raise RuntimeError(
                                    "policy action shape %s != (%d,); refusing to truncate"
                                    % (action.shape, ACTION_DIM)
                                )
                            if not np.all(np.isfinite(action)):
                                raise RuntimeError("policy produced a non-finite action: %s" % (action.tolist(),))
                            send = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
                            obs, reward, terminated, truncated, info = env.step(send)
                            self._last_obs = obs
                            steps += 1
                            self._total_steps += 1
                            native_success = bool(info.get("is_success", False))
                            predicates = {
                                catalog.goal_key(goal): bool(eval_goal_predicate(env, goal))
                                for goal in goals
                            }
                            consecutive = consecutive + 1 if predicates and all(predicates.values()) else 0
                            events_file.write(
                                json.dumps(
                                    {
                                        "step": step,
                                        "action": [float(v) for v in np.asarray(send).reshape(-1).tolist()],
                                        "total_steps": self._total_steps,
                                        "native_success": native_success,
                                        "declared_predicates": predicates,
                                    }
                                )
                                + "\n"
                            )
                            events_file.flush()
                            with self._lock:
                                job.steps = steps
                                job.total_steps = self._total_steps
                                session.total_steps = self._total_steps
                            frame = self._safe_render(env)
                            if frame is not None:
                                frames.append(frame)
                                if step % PNG_EVERY == 0:
                                    self._save_frame(latest_png, frame)
                                    self._update_images(session, env)
                            if consecutive >= GOAL_CONSECUTIVE_STEPS:
                                success = True
                                ended_reason = "success"
                                break
                        else:
                            ended_reason = ended_reason or "budget_exhausted"
        except SceneError as exc:
            ended_reason = exc.reason
            error_text = exc.detail or exc.reason
        except BaseException as exc:  # noqa: BLE001 - persist the real exception
            ended_reason = "error"
            error_text = _format_exc(exc)
        finally:
            wall_s = round(time.monotonic() - started, 3)
            cancelled = ended_reason == "cancelled"
            try:
                after_sha = state_sha(self._env) if self._env is not None else None
            except Exception:  # noqa: BLE001
                after_sha = None
            if frames:
                self._save_frame(last_png, frames[-1])
                self._save_frame(latest_png, frames[-1])
                if len(frames) >= 2:
                    try:
                        _save_video(rollout_path, frames, VIDEO_FPS)
                    except Exception as exc:  # noqa: BLE001
                        log("video write failed: %s" % exc)
            with self._lock:
                session.scene_version += 1  # a capability ended -> new scene version
                job.scene_version_after = session.scene_version
                job.steps = steps
                job.total_steps = self._total_steps
                job.success = bool(success)
                job.wall_s = wall_s
                job.ended_reason = ended_reason or "unknown"
                job.error = error_text
                job.state_after_sha = after_sha
                if error_text and ended_reason == "error":
                    job.state = "error"
                elif cancelled:
                    job.state = "cancelled"
                else:
                    job.state = "completed"
                job.rollout_path = str(rollout_path) if rollout_path.exists() else None
                job.latest_png = str(latest_png) if latest_png.exists() else job.latest_png
            result = {
                "ok": bool(success) and not cancelled,
                "job": job.public(),
                "frames": len(frames),
                "state_before_sha": job.state_before_sha,
                "state_after_sha": job.state_after_sha,
            }
            try:
                (job.run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                log("job result write failed: %s" % exc)
            log(
                "job %s %s | capability=%s steps=%d success=%s ended=%s"
                % (job.job_id, job.state, capability_id, steps, success, job.ended_reason)
            )
        return {
            "ok": bool(success),
            "cancelled": cancelled,
            "reason": ended_reason or "unknown",
            "detail": error_text or "",
        }

    # -- independent evaluation ----------------------------------------------

    def evaluate(self, session_id: str, case_id: str, request_id: str) -> dict[str, Any]:
        try:
            cases = oracle.load_cases()
        except Exception as exc:  # noqa: BLE001
            return _err("internal_error", _format_exc(exc))
        case = cases.get(case_id) if isinstance(case_id, str) else None
        with self._lock:
            record = self._sessions.get(session_id)
            plan = self._plans.get(request_id)
            # Never evaluate while a plan still owns the live scene.
            if self._active_plan_locked() is not None:
                return _err("busy", "a plan is active; finish or cancel it before evaluating")
            error = validate_evaluate_request(
                record.public() if record is not None else None,
                plan.public() if plan is not None else None,
                case,
                request_id,
            )
            if error is not None:
                return error
        return self._sync_work("evaluate", lambda: self._do_evaluate(record, plan, case, case_id))

    def _do_evaluate(self, record: SessionRecord, plan: PlanRecord, case: dict[str, Any],
                     case_id: str) -> dict[str, Any]:
        self._require_ready()
        # Re-validate on the worker immediately before reading any truth, so a
        # plan that started (or a scene version that moved) between the public
        # check and this task cannot cause a later scene to be scored under an
        # earlier request.
        with self._lock:
            if self._active_plan_locked() is not None:
                return _err("busy", "a plan is active; finish or cancel it before evaluating")
            error = validate_evaluate_request(
                record.public(), plan.public(), case, plan.request_id
            )
            if error is not None:
                return error
        if self._env is None or self._env_session_id != record.session_id:
            return _err("unknown_session", "no live environment for this session")

        # executed_objects: only capabilities whose jobs actually moved the sim.
        executed: set[str] = set()
        with self._lock:
            jobs = [self._jobs[job_id] for job_id in self._job_order if job_id in self._jobs]
        for job in jobs:
            if job.request_id != plan.request_id or not job.steps:
                continue
            capability = catalog.CAPABILITIES.get(job.capability_id) or {}
            if capability.get("object_id"):
                executed.add(capability["object_id"])
            else:  # composite: union of the goal object ids
                for goal in capability.get("goals") or []:
                    if len(goal) >= 2:
                        executed.add(goal[1])

        predicates: list[Any] = []
        for option in case.get("goal_options") or []:
            predicates.extend(option)
        predicates.extend(case.get("protected_goals") or [])
        values: dict[str, bool] = {}
        for predicate in predicates:
            base = list(predicate[1:]) if predicate and predicate[0] == "not" else list(predicate)
            key = catalog.goal_key(base)
            if key in values:
                continue
            try:
                values[key] = bool(eval_goal_predicate(self._env, base))
            except Exception:  # noqa: BLE001 - leave it missing so it fails
                continue

        initial_positions = record.initial_positions
        final_positions = object_positions(self._env)
        result = oracle.evaluate_case(
            case,
            values,
            sorted(executed),
            plan.decision,
            initial_positions=initial_positions,
            final_positions=final_positions,
        )
        result["ok"] = True
        result["case_id"] = case_id
        result["request_id"] = plan.request_id
        result["session_id"] = record.session_id
        return result


# --- HTTP layer --------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    service: SceneService


def _segments(path: str) -> list[str]:
    return [unquote(part) for part in path.strip("/").split("/") if part != ""]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SceneService/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102 - stderr trace
        log("http %s - %s" % (self.address_string(), fmt % args))

    # -- helpers --------------------------------------------------------------

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_payload(self, result: dict[str, Any]) -> None:
        reason = result.get("reason") or "invalid_body"
        self._send_json(
            status_for_reason(reason),
            {"ok": False, "reason": reason, "detail": result.get("detail", "")},
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("body must be a JSON object")
        return parsed

    # -- artifact serving -----------------------------------------------------

    def _serve_range(self, path: Path, content_type: str, size: int, range_header: str) -> None:
        try:
            units, _, spec = range_header.partition("=")
            if units.strip().lower() != "bytes":
                raise ValueError("unsupported range unit")
            start_text, _, end_text = spec.partition("-")
            if start_text == "":
                length = int(end_text)
                if length <= 0:
                    raise ValueError("bad suffix range")
                start = max(0, size - length)
                end = size - 1
            else:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            if start < 0 or end >= size or start > end:
                raise ValueError("range out of bounds")
        except Exception:  # noqa: BLE001
            self.send_response(416)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        with open(path, "rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)

    def _serve_artifact(self, relative: str) -> None:
        service = self.server.service
        path = resolve_artifact_path(service.run_root, relative)
        if path is None:
            return self._send_json(404, _err("unknown_artifact", "artifact not available"))
        content_types = {".png": "image/png", ".mp4": "video/mp4", ".json": "application/json"}
        content_type = content_types[path.suffix.lower()]
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        if path.suffix.lower() == ".mp4" and range_header:
            return self._serve_range(path, content_type, size, range_header)
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)

    # -- verbs ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        service = self.server.service
        segments = _segments(urlparse(self.path).path)
        if not segments:
            return self._send_json(404, _err("unknown_artifact", "not found"))
        head = segments[0]
        if head == "health" and len(segments) == 1:
            return self._send_json(200, service.health())
        if head == "scenes" and len(segments) == 1:
            return self._send_json(200, service.scenes())
        if head == "sessions" and len(segments) == 2:
            payload = service.session(segments[1])
            if payload is None:
                return self._send_json(404, _err("unknown_session", "no such session"))
            return self._send_json(200, payload)
        if head == "plans" and len(segments) == 2:
            payload = service.plan(segments[1])
            if payload is None:
                return self._send_json(404, _err("unknown_plan", "no such plan"))
            return self._send_json(200, payload)
        if head == "jobs" and len(segments) == 2:
            payload = service.job(segments[1])
            if payload is None:
                return self._send_json(404, _err("unknown_job", "no such job"))
            return self._send_json(200, payload)
        if head == "artifacts" and len(segments) >= 2:
            return self._serve_artifact("/".join(segments[1:]))
        return self._send_json(404, _err("unknown_artifact", "not found"))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        service = self.server.service
        segments = _segments(urlparse(self.path).path)
        try:
            payload = self._read_json()
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, _err("invalid_body", str(exc)))

        if segments == ["sessions"]:
            result = service.create_session(
                payload.get("scene_id"),
                payload.get("seed", 0),
                payload.get("init_state_index", 0),
            )
            if not result.get("ok"):
                return self._send_error_payload(result)
            return self._send_json(201, result)

        if segments == ["observe"]:
            result = service.observe(payload.get("session_id"), bool(payload.get("extra_views", False)))
            if not result.get("ok"):
                return self._send_error_payload(result)
            return self._send_json(200, result)

        if segments == ["plans"]:
            result = service.submit_plan(payload)
            if not result.get("ok"):
                return self._send_error_payload(result)
            return self._send_json(202, result)

        if segments == ["evaluate"]:
            result = service.evaluate(
                payload.get("session_id"), payload.get("case_id"), payload.get("request_id")
            )
            if not result.get("ok"):
                return self._send_error_payload(result)
            return self._send_json(200, result)

        if len(segments) == 3 and segments[0] == "plans" and segments[2] == "cancel":
            result = service.cancel(segments[1])
            if not result.get("ok"):
                return self._send_error_payload(result)
            return self._send_json(202, result)

        if len(segments) == 3 and segments[0] == "plans" and segments[2] == "resume":
            payload["request_id"] = segments[1]
            result = service.resume_plan(payload)
            if not result.get("ok"):
                return self._send_error_payload(result)
            return self._send_json(202, result)

        return self._send_json(404, _err("unknown_artifact", "not found"))


# --- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent-scene SmolVLA-on-LIBERO HTTP service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run-root", type=str, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args(argv)

    service = SceneService(args.model, args.run_root)
    service.start()

    httpd = _Server((args.host, args.port), _Handler)
    httpd.service = service

    def _handle_signal(signum: int, _frame: Any) -> None:
        log("received signal %s; shutting down" % signum)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log("listening on http://%s:%d (workflow=%s)" % (args.host, args.port, WORKFLOW_NAME))
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        service.stop()
        log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
