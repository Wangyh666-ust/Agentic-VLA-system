#!/usr/bin/env python3
"""Fixed paired wine start-pose diagnostic: replay-only (A) versus prepared (B).

This module hosts :class:`PoseDiagnosticService` -- a subclass of the accepted
``preparation_diagnostics.PreparedService`` -- plus ONE small, fixed campaign that
runs the SAME actual starting state (the exact recorded 102-action bowl prefix on
the shared ``goal_table`` scene) at the FIXED model seeds 0 and 1 under two fixed
branches: ``A`` replays only the 102 recorded actions and then runs the single
wine subgoal exactly as the accepted context diagnostic does (no preparation);
``B`` replays the same 102 actions and then runs the accepted, UNCHANGED
``preparation_diagnostics`` physical preparation hook -- an assistant, non-VLA
open-loop servo -- before the single wine subgoal.

The fixed order is A0, B0, B1, A1 and the report is atomically persisted after
every trial.  The native start pose comes from an external, strictly validated
calibration report (``native_target`` + ``target_source``) and overrides the
shared stored pre-bowl target, so the preparation aligns the hand to the MEASURED
native ``libero_goal/9`` start pose.  An end-effector target does not guarantee
the full robot joint configuration.

Nothing here retrains, downloads, resets the environment, teleports an object,
forces a release, calls Hermes, runs an assessment/repair/retry or re-decides a
plan.  Only stdlib + numpy are imported at import time; ``torch`` is imported
lazily by the pinned base, so ``--help`` and the GPU-free unit tests never
initialise CUDA, load a model or create a live environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
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

import guard_validation as gv  # noqa: E402,F401 - required sibling module surface
import paired_config_experiments as paired  # noqa: E402
import placement_experiments as pe  # noqa: E402
import preparation_diagnostics as pd  # noqa: E402
import service  # noqa: E402
import skill_context_diagnostics as context  # noqa: E402
import wine_diagnostics as wd  # noqa: E402

EXPERIMENT_NAME = "wine_pose_diagnostics"

# The FIXED paired design: two branches, two model seeds, one fixed order.
MODEL_SEEDS = (0, 1)
PLAN = (("A", 0), ("B", 0), ("B", 1), ("A", 1))

PREFIX_CONDITION = pd.PREFIX_CONDITION  # "shared_after_bowl"
PROFILE = context.PROFILE
COMPLETION_MODE = context.COMPLETION_MODE
GRASP_GUARD_MODE = context.GRASP_GUARD_MODE
BUDGET = context.BUDGET
INSTRUCTION = context.WINE_INSTRUCTION
WINE_CAPABILITY_ID = context.WINE_CAPABILITY_ID
WINE_ORACLE_GOALS = [list(goal) for goal in context.WINE_ORACLE_GOALS]

# The three fixed preparation action bounds raised process-locally ONLY for this
# entry; every gain, clamp and tolerance stays unchanged and no file is edited.
LIFT_STAGE_MAX_ACTIONS = 120
ALIGN_STAGE_MAX_ACTIONS = 180
AUX_ACTION_CAP = 300

# The exact calibration contract of the measured native start pose.
TARGET_SOURCE = "libero_goal/9 seed0 init0 reset"
NATIVE_STATE_SHA = "6511e57b416fc70e9fc6ac3e533033dcf9c19f9b1b46046fddfdf7981a5b0b40"

JOINT_CONFIGURATION_NOTE = (
    "the end-effector start-pose target is a measured XYZ/orientation and does NOT "
    "guarantee the full robot joint configuration"
)

LIMITATIONS = {
    "assistant_preparation": (
        "branch B runs the accepted assistant open-loop preparation (0 VLA actions) "
        "after the recorded bowl prefix; it is not a learned VLA skill"
    ),
    "recorded_policy_replay": (
        "the 102-action bowl prefix is the preserved recorded-policy replay, not a "
        "freshly generated VLA bowl execution, a recovery or a forced release"
    ),
    "one_starting_state": (
        "one actual starting state at two model seeds is not a statistical "
        "reliability claim and no success rate is inferred"
    ),
    "joint_configuration": JOINT_CONFIGURATION_NOTE,
    "no_repair": "no assessment, repair, retry, forced release or Hermes call is performed",
    "preparation_changes": (
        "Preparation changes end-effector position/orientation, gripper opening, robot "
        "velocities and elapsed simulation time. This A/B estimates the preparation "
        "procedure effect, not an isolated position-only or orientation-only causal "
        "effect. Objects are protected by the fixed displacement gate."
    ),
}

SOURCE_FILES = (
    "wine_pose_diagnostics.py",
    "preparation_diagnostics.py",
    "skill_context_diagnostics.py",
    "service.py",
    "wine_diagnostics.py",
    "placement_experiments.py",
    "guard_validation.py",
    "paired_config_experiments.py",
    "placement_completion.py",
    "catalog.py",
)


# --- small helpers -----------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _progress(message: str) -> None:
    sys.stderr.write("[wine-pose %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the exact bytes of every module this runner reads."""

    digests: dict[str, str | None] = {}
    for name in SOURCE_FILES:
        try:
            digests[name] = _sha256_hex((_HERE / name).read_bytes())
        except Exception:  # noqa: BLE001 - absent/unreadable is recorded as unknown
            digests[name] = None
    return digests


def _finite_vector(value: Any, shape: tuple) -> Any:
    """The finite float64 array of EXACTLY ``shape``, or ``None`` (never guessed)."""

    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception:  # noqa: BLE001 - a malformed value is unknown, never coerced
        return None
    if array.shape != tuple(shape) or not bool(np.all(np.isfinite(array))):
        return None
    return array


# --- the strict calibration contract -----------------------------------------


def load_calibration(path: Path) -> dict[str, Any]:
    """Load and STRICTLY validate the native start-pose calibration report.

    The report must carry ``ok is True``, ``preparation.ok is True``, the exact
    ``target_source`` and a ``native_target`` whose ``position`` is a finite
    3-vector, whose ``orientation`` is a finite 3x3 matrix and whose ``state_sha``
    is the exact recorded digest.  Anything else -- a missing file, malformed
    JSON, a false ``ok``, a wrong target source/state digest or a
    non-finite/mis-shaped pose -- is rejected with ``ValueError``.  No value is
    inferred from any other (older, richer) schema and no success is fabricated.

    The EXTERNAL report key is ``orientation``; the returned internal normalized
    mapping deliberately exposes it as ``orientation_matrix`` so every downstream
    consumer of the copied native target keeps its single internal key.
    """

    target = Path(path)
    if not target.is_file():
        raise ValueError("calibration report is not a readable file: %s" % (path,))
    try:
        report = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("calibration report is not valid JSON: %s" % (exc,)) from exc
    if not isinstance(report, dict):
        raise ValueError("calibration report is not a JSON object")
    if report.get("ok") is not True:
        raise ValueError("calibration ok is not True: %r" % (report.get("ok"),))
    preparation = report.get("preparation")
    if not isinstance(preparation, dict) or preparation.get("ok") is not True:
        raise ValueError("calibration preparation.ok is not True: %r" % (preparation,))
    native = report.get("native_target")
    if not isinstance(native, dict):
        raise ValueError("calibration native_target is missing or not an object")
    target_source = native.get("target_source", report.get("target_source"))
    if target_source != TARGET_SOURCE:
        raise ValueError("calibration target_source %r != %r" % (target_source, TARGET_SOURCE))
    position = _finite_vector(native.get("position"), (3,))
    if position is None:
        raise ValueError("calibration native_target.position is not a finite 3-vector")
    orientation = _finite_vector(native.get("orientation"), (3, 3))
    if orientation is None:
        raise ValueError(
            "calibration native_target.orientation is not a finite 3x3 matrix"
        )
    if native.get("state_sha") != NATIVE_STATE_SHA:
        raise ValueError(
            "calibration native_target.state_sha %r != %r"
            % (native.get("state_sha"), NATIVE_STATE_SHA)
        )
    return {
        "position": [float(v) for v in position.tolist()],
        "orientation_matrix": orientation.tolist(),
        "state_sha": native["state_sha"],
        "target_source": target_source,
    }


def _copy_native_target(native_target: Any) -> dict[str, Any] | None:
    """An np.float64 copy of a validated native target, or ``None``."""

    if not isinstance(native_target, dict):
        return None
    position = _finite_vector(native_target.get("position"), (3,))
    matrix = _finite_vector(native_target.get("orientation_matrix"), (3, 3))
    if position is None or matrix is None:
        return None
    return {
        "position": position.astype(np.float64).copy(),
        "orientation_matrix": matrix.astype(np.float64).copy(),
        "state_sha": native_target.get("state_sha"),
        "target_source": native_target.get("target_source"),
    }


@contextlib.contextmanager
def preparation_limits():
    """Temporarily raise the three fixed preparation action bounds (process-local).

    ``pd.LIFT_STAGE_MAX_ACTIONS`` / ``pd.ALIGN_STAGE_MAX_ACTIONS`` /
    ``pd.AUX_ACTION_CAP`` are assigned for the duration of the campaign and
    restored to their exact originals in ``finally`` -- even on error.  No source
    file is edited; only this process's module attributes change.  The yielded
    record lists the effective values and the previous ones.
    """

    originals = {
        "lift_stage_max_actions": pd.LIFT_STAGE_MAX_ACTIONS,
        "align_stage_max_actions": pd.ALIGN_STAGE_MAX_ACTIONS,
        "aux_action_cap": pd.AUX_ACTION_CAP,
    }
    pd.LIFT_STAGE_MAX_ACTIONS = LIFT_STAGE_MAX_ACTIONS
    pd.ALIGN_STAGE_MAX_ACTIONS = ALIGN_STAGE_MAX_ACTIONS
    pd.AUX_ACTION_CAP = AUX_ACTION_CAP
    try:
        yield {
            "lift_stage_max_actions": pd.LIFT_STAGE_MAX_ACTIONS,
            "align_stage_max_actions": pd.ALIGN_STAGE_MAX_ACTIONS,
            "aux_action_cap": pd.AUX_ACTION_CAP,
            "previous": originals,
        }
    finally:
        pd.LIFT_STAGE_MAX_ACTIONS = originals["lift_stage_max_actions"]
        pd.ALIGN_STAGE_MAX_ACTIONS = originals["align_stage_max_actions"]
        pd.AUX_ACTION_CAP = originals["aux_action_cap"]


# --- the paired service -------------------------------------------------------


class PoseDiagnosticService(pd.PreparedService):
    """The accepted prepared service plus a branchable preparation switch.

    Everything else -- the worker thread, the ``strict=True`` model load, the
    ``release_verified`` completion gate, the ``shadow`` wine-only grasp guard,
    the recorded-prefix replay, the inherited first-input fingerprint capture and
    the accepted physical preparation hook -- is inherited verbatim.  Only two
    methods are overridden:

    * ``_do_create_session`` calls the ACTUAL parent implementation first and, only
      on a successful creation, overwrites ``self._pre_bowl_pose`` with the
      MEASURED native target position/orientation as np.float64 copies (the
      environment itself is never touched).  The parent's own success fields and
      return value are reused unchanged.  If the copied internal native target is
      missing/invalid/nonfinite -- which the constructor never permits -- the
      session is refused with ``service.SceneError('native_target_unavailable')``
      instead of returning a created success;
    * ``_sync_work`` dispatches to the accepted preparation hook ONLY while
      ``preparation_enabled`` is True (branch B); otherwise it calls the plain
      context worker plumbing directly (branch A), so A never runs any preparation.
    """

    def __init__(self, *args: Any, native_target: Any = None, **kwargs: Any) -> None:
        copied = _copy_native_target(native_target)
        if copied is None:
            raise ValueError(
                "PoseDiagnosticService requires a valid native_target holding a finite "
                "position and orientation_matrix; none was supplied or it was invalid"
            )
        kwargs.setdefault("completion_mode", COMPLETION_MODE)
        kwargs.setdefault("grasp_guard_mode", GRASP_GUARD_MODE)
        super().__init__(*args, **kwargs)
        self.native_target = copied
        self.preparation_enabled = False

    def _do_create_session(self, record: Any, seed: int, init_state_index: int) -> dict[str, Any]:
        created = super()._do_create_session(record, seed, init_state_index)
        if not isinstance(created, dict) or created.get("ok") is not True:
            return created
        native = self.native_target
        position = native.get("position") if isinstance(native, dict) else None
        matrix = native.get("orientation_matrix") if isinstance(native, dict) else None
        if (
            not isinstance(position, np.ndarray)
            or not isinstance(matrix, np.ndarray)
            or position.shape != (3,)
            or matrix.shape != (3, 3)
            or not bool(np.all(np.isfinite(position)))
            or not bool(np.all(np.isfinite(matrix)))
        ):
            raise service.SceneError(
                "native_target_unavailable", "native calibration target is unavailable"
            )
        self._pre_bowl_pose = {
            "position": position.astype(np.float64).copy(),
            "orientation_matrix": matrix.astype(np.float64).copy(),
        }
        return created

    def _sync_work(self, kind: str, fn: Any, timeout: float = service.WORKER_WAIT_S) -> dict[str, Any]:
        """The accepted hook for B; the plain context plumbing for A."""

        if self.preparation_enabled:
            return pd.PreparedService._sync_work(self, kind, fn, timeout)
        return context.ContextDiagnosticService._sync_work(self, kind, fn, timeout)


# --- the outer per-trial view -------------------------------------------------


def _target_pose_error(preparation: Any) -> dict[str, Any] | None:
    """The native target pose error from the LAST align trace entry, or ``None``."""

    if not isinstance(preparation, dict):
        return None
    align = preparation.get("align")
    trace = align.get("trace") if isinstance(align, dict) else None
    if not isinstance(trace, list) or not trace or not isinstance(trace[-1], dict):
        return None
    last = trace[-1]
    return {
        "position_error_m": last.get("position_error_m"),
        "rotation_error_rad": last.get("rotation_error_rad"),
    }


def _wine_attempted(raw_trial: Any) -> bool:
    """Whether the raw trial carries ACTUAL wine subgoal job evidence.

    Derived ONLY from the recorded wine job evidence (a wine job in ``jobs``),
    never from the planned branch: an entry whose wine subgoal was never submitted
    -- for example a physical preparation block -- is NOT an attempt.
    """

    if not isinstance(raw_trial, dict):
        return False
    for evidence in raw_trial.get("jobs") or []:
        if not isinstance(evidence, dict):
            continue
        job = evidence.get("job")
        if isinstance(job, dict) and job.get("capability_id") == WINE_CAPABILITY_ID:
            return True
    return False


def _combined_success(raw: dict[str, Any]) -> bool:
    """Exactly raw strict wine success AND raw final-bowl strict, never inferred."""

    return raw.get("strict_wine_success") is True and raw.get("final_bowl_strict") is True


def _branch_view(branch: str, model_seed: int, raw_trial: Any, preparation: Any) -> dict[str, Any]:
    """The outer report entry: the raw trial UNCHANGED plus derived fields.

    Only a PHYSICAL preparation failure blocks the wine subgoal (the accepted
    ``_trial_view`` filters exactly the ``PreparationBlocked`` marker); an
    operational failure keeps its raw operational errors.  Branch A passes
    ``preparation=None`` so the raw operational errors are used directly and it
    can never report any auxiliary action (``aux_actions`` is exactly 0).

    ``wine_attempted`` is derived from the ACTUAL wine job evidence, never from
    the planned branch.  A physically blocked B entry never submitted the wine
    subgoal, so it claims no attempt (``wine_attempted`` False) and no wine
    actions (``wine_action_count`` 0), can never be a combined success, and stays
    a PHYSICAL preparation failure rather than an operational or VLA wine failure.
    """

    view = pd._trial_view(
        {"entry": branch, "model_seed": int(model_seed)}, raw_trial, preparation
    )
    raw = view["raw_trial"]
    view["branch"] = branch
    view["model_seed"] = int(model_seed)
    auxiliary = view.get("auxiliary_action_count")
    view["aux_actions"] = 0 if branch == "A" or auxiliary is None else int(auxiliary)
    view["before_wine_state_sha"] = raw.get("before_wine_state_sha")
    view["strict_wine_success"] = raw.get("strict_wine_success")
    view["raw_stable_grasp_first_step"] = raw.get("raw_stable_grasp_first_step")
    view["final_bowl_predicate"] = raw.get("final_bowl_predicate")
    view["final_bowl_strict"] = raw.get("final_bowl_strict")
    view["native_target_pose_error"] = _target_pose_error(preparation)
    if view.get("status") == "physical_preparation_failed":
        view["wine_attempted"] = False
        view["wine_action_count"] = 0
        view["combined_success"] = False
    else:
        view["wine_attempted"] = _wine_attempted(raw)
        view["combined_success"] = _combined_success(raw)
    return view


def _check_provenance(
    view: dict[str, Any], branch: str, raw_trial: dict[str, Any], prefix_result: Any
) -> None:
    """Record every provenance/consistency violation as an operational error."""

    ops = view["operational_errors"]
    prefix = prefix_result if isinstance(prefix_result, dict) else {}
    if prefix.get("origin_state_sha") != context.REPLAY_ORIGIN_SHA:
        ops.append(
            "prefix_origin_state_sha: %r != %r"
            % (prefix.get("origin_state_sha"), context.REPLAY_ORIGIN_SHA)
        )
    if prefix.get("final_state_sha") != context.REPLAY_FINAL_SHA:
        ops.append(
            "prefix_final_state_sha: %r != %r"
            % (prefix.get("final_state_sha"), context.REPLAY_FINAL_SHA)
        )
    if prefix.get("replay_action_count") != context.REPLAY_ACTION_COUNT:
        ops.append(
            "prefix_replay_action_count: %r != %d"
            % (prefix.get("replay_action_count"), context.REPLAY_ACTION_COUNT)
        )
    if branch == "A" and raw_trial.get("before_wine_state_sha") != context.REPLAY_FINAL_SHA:
        ops.append(
            "a_before_wine_state_sha: %r != %r"
            % (raw_trial.get("before_wine_state_sha"), context.REPLAY_FINAL_SHA)
        )
    if branch == "B":
        preparation = view.get("preparation")
        prepared = isinstance(preparation, dict) and preparation.get("ok") is True
        if not prepared and int(view.get("wine_action_count") or 0) > 0:
            ops.append(
                "b_wine_action_without_preparation: %d" % view["wine_action_count"]
            )


def _run_branch_trial(
    svc: Any,
    branch: str,
    model_seed: int,
    actions: Any,
    run_root: Path,
    context_seen: dict[str, Any],
    b_cross_seen: dict[str, Any],
) -> dict[str, Any]:
    """Run ONE fixed trial through the accepted context runner, then view it.

    The raw trial returned by :func:`skill_context_diagnostics._run_trial` is
    preserved UNCHANGED.  Branch A is checked with the existing context cross-seed
    comparison; branch B additionally uses the context key map plus its own
    separate ``_cross_seed_check`` map.  A is never given the B-only helper.
    """

    svc.prefix_result = None
    svc.preparation_result = None
    svc.preparation_enabled = branch == "B"
    try:
        raw_trial = context._run_trial(
            svc,
            {"condition": PREFIX_CONDITION, "model_seed": int(model_seed)},
            actions,
            Path(run_root) / branch,
            context_seen,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve the raw record
        message = "trial_exception: %s" % _format_exc(exc)
        raw_trial = {
            "trial_id": "%s_m%s" % (PREFIX_CONDITION, model_seed),
            "condition": PREFIX_CONDITION,
            "model_seed": int(model_seed),
            "errors": [message],
            "operational_errors": [message],
            "jobs": [],
            "physical_failures": [],
            "after_bowl_state_sha": None,
        }
    preparation = svc.preparation_result if branch == "B" else None
    view = _branch_view(branch, model_seed, raw_trial, preparation)
    prefix_result = svc.prefix_result if isinstance(svc.prefix_result, dict) else raw_trial.get("prefix")
    view["prefix_result"] = prefix_result
    if isinstance(prefix_result, dict) and prefix_result.get("final_state_sha") is not None:
        view["after_bowl_state_sha"] = prefix_result["final_state_sha"]
    _check_provenance(view, branch, raw_trial, prefix_result)
    if branch == "A":
        context._compare_across_seeds(raw_trial, context_seen, view["operational_errors"].append)
    else:
        cross = pd._cross_seed_check(view, b_cross_seen, int(model_seed))
        view["cross_seed"] = cross
        if cross.get("differing"):
            view["operational_errors"].append("cross_seed_B_mismatch: %r" % (cross.get("differing"),))
    return view


def _exception_view(entry: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    """A faithful outer view for a trial that raised before any record existed."""

    message = "trial_exception: %s" % _format_exc(exc)
    raw = {
        "trial_id": "%s_m%s" % (PREFIX_CONDITION, entry["model_seed"]),
        "condition": PREFIX_CONDITION,
        "model_seed": int(entry["model_seed"]),
        "errors": [message],
        "operational_errors": [message],
        "jobs": [],
        "physical_failures": [],
        "after_bowl_state_sha": None,
    }
    view = _branch_view(entry["entry"], entry["model_seed"], raw, None)
    view["prefix_result"] = None
    return view


# --- the fixed plan -----------------------------------------------------------


def build_campaign_plan() -> list[dict[str, Any]]:
    """The FIXED four-trial order: A0, B0, B1, A1."""

    return [{"entry": entry, "model_seed": int(model_seed)} for entry, model_seed in PLAN]


def _build_service(args: argparse.Namespace, native_target: dict[str, Any]) -> PoseDiagnosticService:
    """Construct the one paired service with the exact required arguments."""

    return PoseDiagnosticService(
        model_path=service.DEFAULT_MODEL_PATH,
        run_root=str(args.run_root),
        completion_mode=COMPLETION_MODE,
        grasp_guard_mode=GRASP_GUARD_MODE,
        native_target=native_target,
    )


# --- preregistration and report ----------------------------------------------


def _build_preregistration(
    args: argparse.Namespace,
    plan: list[dict[str, Any]],
    actions: Any,
    native_target: dict[str, Any],
    limits_previous: dict[str, Any],
) -> dict[str, Any]:
    """The frozen plan, written BEFORE the model / service starts."""

    return {
        "experiment": EXPERIMENT_NAME,
        "created_utc": _now_utc(),
        "commitment": (
            "written before the model starts; the raw SHA-256 of this file's bytes is "
            "recorded in the campaign report and the fixed order is never re-decided"
        ),
        "scope": (
            "ONE actual starting state (%s, scene seed %s / init state %s: the exact "
            "recorded 102-action bowl prefix) measured at TWO model seeds; this is not "
            "a statistical reliability claim and no success rate is inferred"
            % (PREFIX_CONDITION, context.SCENE_SEED, context.INIT_STATE_INDEX)
        ),
        "fixed_spec": {
            "condition": PREFIX_CONDITION,
            "scene_id": wd.SHARED_SCENE_ID,
            "scene_seed": context.SCENE_SEED,
            "init_state_index": context.INIT_STATE_INDEX,
            "model_seeds": [int(seed) for seed in MODEL_SEEDS],
            "capability_id": WINE_CAPABILITY_ID,
            "instruction": INSTRUCTION,
            "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
            "profile": PROFILE,
            "completion_mode": COMPLETION_MODE,
            "grasp_guard_mode": GRASP_GUARD_MODE,
            "budget": BUDGET,
            "wine_vla_action_cap": BUDGET,
            "replay_action_count": context.REPLAY_ACTION_COUNT,
            "replay_origin_state_sha": context.REPLAY_ORIGIN_SHA,
            "replay_final_state_sha": context.REPLAY_FINAL_SHA,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "target_revision": paired.FIXED_SOURCE_REVISION,
        },
        "branches": {
            "A": {"preparation": False, "replay_only": True, "auxiliary_actions": 0},
            "B": {
                "preparation": True,
                "assistant": True,
                "is_vla": False,
                "vla_actions_from_preparation": 0,
            },
        },
        "preparation_limits": {
            "lift_stage_max_actions": LIFT_STAGE_MAX_ACTIONS,
            "align_stage_max_actions": ALIGN_STAGE_MAX_ACTIONS,
            "aux_action_cap": AUX_ACTION_CAP,
            "previous": limits_previous,
            "note": (
                "only the three fixed action bounds are raised, process-locally; every "
                "gain, clamp and tolerance is unchanged and no source file is edited"
            ),
        },
        "unchanged": {
            "servo_gain": pd.SERVO_GAIN,
            "servo_translation_clamp": pd.SERVO_TRANSLATION_CLAMP,
            "servo_rotation_clamp": pd.SERVO_ROTATION_CLAMP,
            "servo_position_tolerance_m": pd.SERVO_POSITION_TOLERANCE_M,
            "servo_rotation_tolerance_rad": pd.SERVO_ROTATION_TOLERANCE_RAD,
            "servo_success_streak": pd.SERVO_SUCCESS_STREAK,
            "protection_tolerance_m": pd.PROTECTION_TOLERANCE_M,
            "protected_objects": list(pd.PROTECTED_OBJECTS),
        },
        "native_start_pose_target": {
            "source": TARGET_SOURCE,
            "state_sha": native_target.get("state_sha"),
            "position": native_target.get("position"),
            "orientation_matrix": native_target.get("orientation_matrix"),
            "note": JOINT_CONFIGURATION_NOTE,
        },
        "plan": plan,
        "inputs": {
            "input_actions_path": str(args.input_actions),
            "input_actions_count": len(actions),
            "input_actions_sha256": _sha256_hex(Path(args.input_actions).read_bytes()),
            "calibration_path": str(args.calibration),
            "calibration_sha256": _sha256_hex(Path(args.calibration).read_bytes()),
        },
        "source_sha256": _source_sha256(),
        "source_git_sha": getattr(args, "source_git_sha", None),
        "limitations": dict(LIMITATIONS),
    }


def _base_report(args: argparse.Namespace, plan: list[dict[str, Any]]) -> dict[str, Any]:
    """The self-consistent report shell, current at every atomic persistence."""

    return {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "ok": False,
        "fatal_error": None,
        "paired": True,
        "metadata": {
            "output": str(args.output),
            "run_root": str(args.run_root),
            "input_actions_path": str(args.input_actions),
            "calibration_path": str(args.calibration),
            "input_actions_count": None,
            "input_actions_sha256": None,
            "calibration_sha256": None,
            "native_target": None,
            "plan": plan,
            "model_seeds": [int(seed) for seed in MODEL_SEEDS],
            "condition": PREFIX_CONDITION,
            "scene_id": wd.SHARED_SCENE_ID,
            "budget": BUDGET,
            "profile": PROFILE,
            "completion_mode": COMPLETION_MODE,
            "grasp_guard_mode": GRASP_GUARD_MODE,
            "capability_id": WINE_CAPABILITY_ID,
            "instruction": INSTRUCTION,
            "model_path": service.DEFAULT_MODEL_PATH,
            "model_revision": None,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "target_revision": paired.FIXED_SOURCE_REVISION,
            "source_git_sha": None,
            "source_sha256": _source_sha256(),
            "replay_action_count": context.REPLAY_ACTION_COUNT,
            "replay_origin_state_sha": context.REPLAY_ORIGIN_SHA,
            "replay_final_state_sha": context.REPLAY_FINAL_SHA,
            "preparation_limits": {
                "lift_stage_max_actions": LIFT_STAGE_MAX_ACTIONS,
                "align_stage_max_actions": ALIGN_STAGE_MAX_ACTIONS,
                "aux_action_cap": AUX_ACTION_CAP,
            },
            "assistant_preparation_is_vla": False,
            "production_health_url": context.HEALTH_URL,
            "production_health": [],
            "campaign_wall_s": None,
        },
        "preregistration": None,
        "service": {},
        "trials": [],
        "health_checks": [],
        "operational_errors": [],
        "limitations": dict(LIMITATIONS),
        "aggregate": {},
    }


def _aggregate(trials: list[dict[str, Any]], expected_trials: int) -> dict[str, Any]:
    """Explicit A/B counts; the 102-action replay stays separate from VLA/aux."""

    def _branch(name: str) -> dict[str, Any]:
        rows = [trial for trial in trials if trial.get("branch") == name]
        return {
            "n_trials": len(rows),
            "n_strict_wine_success": sum(1 for t in rows if t.get("strict_wine_success") is True),
            "n_wine_success": sum(
                1
                for t in rows
                if t.get("wine_attempted") is True and t.get("strict_wine_success") is True
            ),
            "n_combined_success": sum(1 for t in rows if t.get("combined_success") is True),
            # A wine FAILURE requires an ACTUAL attempt: an unattempted physically
            # blocked entry (wine_attempted False) is never counted as a wine failure.
            "n_strict_wine_failure": sum(
                1
                for t in rows
                if t.get("wine_attempted") is True and t.get("strict_wine_success") is False
            ),
            "n_physical_preparation_failed": sum(
                1 for t in rows if t.get("status") == "physical_preparation_failed"
            ),
            "n_operational_error": sum(1 for t in rows if t.get("status") == "operational_error"),
            "n_raw_stable_grasp": sum(
                1 for t in rows if t.get("raw_stable_grasp_first_step") is not None
            ),
            "raw_stable_grasp_first_steps": [t.get("raw_stable_grasp_first_step") for t in rows],
            "n_auxiliary_actions": sum(int(t.get("aux_actions") or 0) for t in rows),
            "n_wine_actions": sum(int(t.get("wine_action_count") or 0) for t in rows),
            "n_final_bowl_native_true": sum(1 for t in rows if t.get("final_bowl_predicate") is True),
            "n_final_bowl_strict_true": sum(1 for t in rows if t.get("final_bowl_strict") is True),
            "native_target_pose_errors": [t.get("native_target_pose_error") for t in rows],
        }

    return {
        "n_trials": len(trials),
        "expected_trials": int(expected_trials),
        "n_replay_actions": sum(
            int((t.get("prefix_result") or {}).get("replay_action_count") or 0) for t in trials
        ),
        "n_auxiliary_actions": sum(int(t.get("aux_actions") or 0) for t in trials),
        "n_wine_actions": sum(int(t.get("wine_action_count") or 0) for t in trials),
        "branches": {"A": _branch("A"), "B": _branch("B")},
        "reliability_note": (
            "explicit counts only; one starting state at two model seeds is not a "
            "statistical reliability claim and no success rate is inferred"
        ),
    }


def _aggregate_operational(trials: list[dict[str, Any]], fatal_error: str | None) -> list[str]:
    """Every real operational error, plus a non-``None`` fatal error."""

    messages: list[str] = []
    if fatal_error:
        messages.append("fatal: %s" % (fatal_error,))
    for trial in trials:
        for message in trial.get("operational_errors") or []:
            messages.append(
                "%s_m%s: %s" % (trial.get("branch"), trial.get("model_seed"), message)
            )
    return messages


# --- the campaign -------------------------------------------------------------


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed four-trial paired campaign and persist the report."""

    output_path = Path(args.output)
    run_root = Path(args.run_root)
    started = time.monotonic()
    plan = build_campaign_plan()
    expected_trials = len(plan)
    report = _base_report(args, plan)
    trials: list[dict[str, Any]] = report["trials"]
    health_checks: list[dict[str, Any]] = report["health_checks"]
    persisted_errors: list[str] = []
    context_seen: dict[str, dict] = {"A": {}, "B": {}}
    b_cross_seen: dict[str, Any] = {}
    fatal_error: str | None = None
    svc: Any = None

    def _persist() -> None:
        report["fatal_error"] = fatal_error
        report["operational_errors"] = _aggregate_operational(trials, fatal_error)
        report["aggregate"] = _aggregate(trials, expected_trials)
        report["metadata"]["production_health"] = health_checks
        report["metadata"]["campaign_wall_s"] = round(time.monotonic() - started, 3)
        report["ok"] = bool(
            fatal_error is None
            and not persisted_errors
            and len(trials) == expected_trials
            and all(not trial.get("operational_errors") for trial in trials)
        )
        try:
            pe._write_json_atomic(output_path, report)
        except Exception as exc:  # noqa: BLE001 - a report write failure is fatal
            persisted_errors.append(_format_exc(exc))
            _progress("report write failed: %s" % exc)

    try:
        run_root.mkdir(parents=True, exist_ok=True)
        actions = context.load_actions(args.input_actions)
        native_target = load_calibration(Path(args.calibration))
        report["metadata"]["input_actions_count"] = len(actions)
        report["metadata"]["input_actions_sha256"] = _sha256_hex(
            Path(args.input_actions).read_bytes()
        )
        report["metadata"]["calibration_sha256"] = _sha256_hex(Path(args.calibration).read_bytes())
        report["metadata"]["native_target"] = dict(native_target)
        report["metadata"]["source_git_sha"] = args.source_git_sha or pe._git_rev_parse()
        limits_previous = {
            "lift_stage_max_actions": pd.LIFT_STAGE_MAX_ACTIONS,
            "align_stage_max_actions": pd.ALIGN_STAGE_MAX_ACTIONS,
            "aux_action_cap": pd.AUX_ACTION_CAP,
        }

        # 1. Freeze the preregistration BEFORE the model / service starts.
        prereg_path = run_root / "preregistration.json"
        pe._write_json_atomic(
            prereg_path, _build_preregistration(args, plan, actions, native_target, limits_previous)
        )
        report["preregistration"] = {
            "path": str(prereg_path),
            "sha256": _sha256_hex(prereg_path.read_bytes()),
            "frozen_before_model_start": True,
        }
        _persist()

        # 2. ONE service / ONE model load, under the raised preparation limits.
        with wd.register_native_wine_scene(), preparation_limits():
            svc = _build_service(args, native_target)
            svc.start()
            readiness = context._wait_ready(svc, context.READY_TIMEOUT_S)
            report["service"] = readiness if isinstance(readiness, dict) else {"ready": False}
            if not isinstance(readiness, dict) or readiness.get("ready") is not True:
                raise RuntimeError(
                    "service not ready: %s"
                    % ((readiness or {}).get("worker_error") or "no ready health reported")
                )
            report["metadata"]["model_revision"] = readiness.get("model_revision")

            total = len(plan)
            for index, entry in enumerate(plan, start=1):
                record, gate_fatal = pd.production_health_gate()
                health_checks.append(
                    {
                        "trial_index": index,
                        "branch": entry["entry"],
                        "model_seed": entry["model_seed"],
                        "result": record,
                    }
                )
                _persist()
                if gate_fatal is not None:
                    fatal_error = "%s (trial %d/%d)" % (gate_fatal, index, total)
                    _progress(fatal_error)
                    break

                _progress(
                    "START trial %d/%d branch=%s model_seed=%s actions=%d"
                    % (index, total, entry["entry"], entry["model_seed"], len(actions))
                )
                try:
                    view = _run_branch_trial(
                        svc,
                        entry["entry"],
                        entry["model_seed"],
                        actions,
                        run_root,
                        context_seen[entry["entry"]],
                        b_cross_seen,
                    )
                except BaseException as exc:  # noqa: BLE001 - preserve, then stop
                    view = _exception_view(entry, exc)
                trials.append(view)
                _persist()
                _progress(
                    "END trial %d/%d branch=%s model_seed=%s replay=%s aux=%s wine=%s "
                    "success=%s ops=%d"
                    % (
                        index,
                        total,
                        view.get("branch"),
                        view.get("model_seed"),
                        (view.get("prefix_result") or {}).get("replay_action_count"),
                        view.get("aux_actions"),
                        view.get("wine_action_count"),
                        view.get("strict_wine_success"),
                        len(view.get("operational_errors") or []),
                    )
                )
                if view.get("operational_errors"):
                    fatal_error = (
                        "operational error(s); stopping campaign (no continuation): %s"
                        % ("; ".join(view["operational_errors"]),)
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
        try:
            record, gate_fatal = pd.production_health_gate()
            health_checks.append({"phase": "after_campaign", "result": record})
            report["final_health_gate_reason"] = gate_fatal
        except Exception as exc:  # noqa: BLE001 - the final read is best effort
            report["final_health_error"] = _format_exc(exc)

    _persist()
    if report.get("ok"):
        print("WINE_POSE_DIAGNOSTICS_PASS")
    _progress("report written to %s (%d trials)" % (output_path, len(trials)))
    return report


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed paired wine start-pose diagnostic: replay-only (A) versus the "
            "accepted assistant preparation (B) on ONE recorded bowl-prefix starting "
            "state at the fixed model seeds 0 and 1.  No training, downloads, "
            "assessment, repair, forced release or HTTP server."
        )
    )
    parser.add_argument(
        "--input-actions",
        required=True,
        type=str,
        help="absolute path of the preserved 102-action bowl prefix (events.jsonl or JSON list)",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        type=str,
        help="absolute path of the strictly validated native start-pose calibration report",
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
    for name in ("input_actions", "calibration", "output", "run_root"):
        if not os.path.isabs(str(getattr(args, name))):
            parser.error("--%s must be an absolute path" % name.replace("_", "-"))
    for name in ("input_actions", "calibration"):
        if not Path(str(getattr(args, name))).is_file():
            parser.error(
                "--%s must be an existing readable file: %s"
                % (name.replace("_", "-"), getattr(args, name))
            )
    for name in ("output", "run_root"):
        if Path(str(getattr(args, name))).exists():
            parser.error(
                "--%s already exists; refusing to reuse %s" % (name.replace("_", "-"), getattr(args, name))
            )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    report = run_campaign(args)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
