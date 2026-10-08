#!/usr/bin/env python3
"""Isolated, separately preregistered *physical preparation* diagnostic.

This module hosts :class:`PreparedService` -- a subclass of the accepted
:class:`skill_context_diagnostics.ContextDiagnosticService` -- plus ONE fixed,
explicitly-assisted physical preparation control that runs **after the exact
recorded bowl prefix** and **before** the single wine subgoal.

Explicit assisted preparation, not a VLA trial
==============================================

Two fixed entries (model seeds 0 and 1) run the SAME preparation pipeline: an
open-loop servo that lifts the end effector clear of the bowl and then aligns it
back to the recorded pre-bowl pose.  The motion is performed by this module,
never by the policy, and is reported as ``assisted`` evidence -- never as a
learned VLA skill, a production feature, a recovery or a forced release.

Nothing here retrains, downloads, resets the environment, calls ``set_state`` /
``teleport`` / ``forward``, moves a task object, forces a release, calls Hermes,
runs an assessment or runs a repair.  The base service lifecycle is reused
verbatim; the accepted ``skill_context_diagnostics`` runner/helpers are reused
rather than copied.

The fixed pipeline
==================

Both entries start from the *same* recorded-policy replay of the preserved 102
user bowl actions (fixed origin ``REPLAY_ORIGIN_SHA`` before the first step and
final ``REPLAY_FINAL_SHA`` before any preparation action).  The servo then runs
exactly two stages on the single worker thread:

* lift: target = the after-bowl XY, the current orientation, and
  ``Z = max(original initial target Z, current Z + 0.10)`` (max 40 actions);
* align: target = the original initial end-effector XYZ/orientation (max 60
  actions).

Each stage succeeds only after five consecutive REAL post-action samples with a
position norm ``<= 0.005 m`` and an orientation norm ``<= 0.05 rad``.  At most
100 auxiliary controls are sent per entry and the gains/scales are fixed (no
parameter adaptation).  Every servo action opens the already-empty hand.

Fail-closed evidence
====================

BEFORE and AFTER every real ``env.step`` a read-only
``placement_experiments.capture_snapshot(env, [BOWL_GOAL])`` must show the native
bowl predicate True, the gripper empty and the grasp observation complete, and
all four protected objects (``wine_bottle_1``, ``akita_black_bowl_1``,
``plate_1``, ``cream_cheese_1``) present, finite and within EXACTLY
``PROTECTION_TOLERANCE_M`` (0.005 m) of their pre-preparation positions.  An
unknown or violated value prevents the next step and is recorded, including the
failure-causing action.  A preparation failure is physical: the wine subgoal is
never submitted and the context model RNG is never reseeded for that entry.

Only stdlib + numpy are imported at module import time; ``torch`` is imported
lazily by the pinned base ``_select_action`` and ``scipy`` (if present at all)
only inside :func:`_matrix_to_rotvec`, so ``--help`` and the GPU-free unit tests
never initialise CUDA, load a model or create a live environment.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import catalog  # noqa: E402
import grasp_guard  # noqa: E402
import guard_validation as gv  # noqa: E402
import paired_config_experiments as paired  # noqa: E402
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402
import skill_context_diagnostics as context  # noqa: E402
import wine_diagnostics as wd  # noqa: E402

EXPERIMENT_NAME = "preparation_diagnostics"

ACTION_DIM = 7
CONTROL_FREQUENCY_HZ = 20

# The two fixed physical-state digests of the preserved bowl prefix (reused from
# the accepted context runner; never re-typed or re-derived here).
REPLAY_ORIGIN_SHA = context.REPLAY_ORIGIN_SHA
REPLAY_FINAL_SHA = context.REPLAY_FINAL_SHA
REPLAY_ACTION_COUNT = context.REPLAY_ACTION_COUNT

# The literal goals / capability / instruction of the recorded bowl-prefix scene.
BOWL_GOAL: list[str] = list(context.BOWL_GOAL)
WINE_ORACLE_GOALS: list[list[str]] = [list(goal) for goal in context.WINE_ORACLE_GOALS]
WINE_CAPABILITY_ID = context.WINE_CAPABILITY_ID
WINE_INSTRUCTION = context.WINE_INSTRUCTION
BOWL_GOAL_KEY = catalog.goal_key(BOWL_GOAL)

# The single recorded bowl-prefix condition this module extends.  It is the
# accepted ``skill_context_diagnostics`` condition whose scene is the shared
# ``goal_table`` and whose scene state is the exact recorded replay final SHA.
PREFIX_CONDITION = "shared_after_bowl"
SCENE_ID = wd.SHARED_SCENE_ID  # "goal_table"
TASK_ID = int(context.CONDITION_SPECS[PREFIX_CONDITION]["task_id"])

PROFILE = context.PROFILE  # "baseline_bf16"
COMPLETION_MODE = context.COMPLETION_MODE  # "release_verified"
GRASP_GUARD_MODE = context.GRASP_GUARD_MODE  # "shadow"

# The fixed wine budget (unchanged) and the fixed preparation allowance.  The
# TOTAL auxiliary (non-VLA) simulator steps of ONE preparation are capped at
# ``AUX_ACTION_CAP`` (lift 40 + align 60); no wine action is ever produced by it.
BUDGET = context.BUDGET  # 300
AUX_ACTION_CAP = 100
TIMEOUT_S = context.TIMEOUT_S
READY_TIMEOUT_S = context.READY_TIMEOUT_S
PREPARE_TIMEOUT_S = context.REPLAY_TIMEOUT_S
POLL_INTERVAL_S = context.POLL_INTERVAL_S

HEALTH_URL = context.HEALTH_URL
HEALTH_TIMEOUT_S = 5.0

RATIONALE = "wine handoff preparation diagnostic"

# The two FIXED model seeds of this diagnostic (0 then 1); the environment seed
# and the init state index are BOTH fixed at 0 for every entry, and there is no
# variable seed/branch CLI and no arbitrary extra case.
MODEL_SEEDS = (0, 1)
SCENE_SEED = 0
INIT_STATE_INDEX = 0
MAX_WINE_TRIALS = len(MODEL_SEEDS)

# The fixed read-only baseline report whose bytes must remain untouched.  It is
# the accepted context campaign's ``shared_after_bowl`` evidence: both seeds are
# paired to it (before-wine state ``f0a5...405e``, 300 wine steps, success
# False).  It is only ever read, and its digest is required exactly.
BASELINE_REPORT_SHA = "65317bab9e9ee7d628579fa9cddf7f5b7054cf21084c3d943a3164a6504a7aa7"
BASELINE_CONDITION = PREFIX_CONDITION
BASELINE_BEFORE_WINE_STATE_SHA = REPLAY_FINAL_SHA
BASELINE_WINE_STEPS = 300
BASELINE_WINE_SUCCESS = False

PREPARATION_EVIDENCE_DIRNAME = "preparation"
PREPARATION_EVENTS_FILENAME = "preparation_events.jsonl"
PREPARATION_SUMMARY_FILENAME = "preparation.json"

# --- the protected objects and the EXACT physical protection threshold -------
# ALL FOUR objects are protected; the threshold is exactly 0.005 m (never the
# invented 0.08 m README-section-6 fixture claim).  A missing baseline/current
# position is a fail-closed violation and finite XYZ is required.
PROTECTED_OBJECTS = ("wine_bottle_1", "akita_black_bowl_1", "plate_1", "cream_cheese_1")
PROTECTION_TOLERANCE_M = 0.005

# --- the fixed robosuite OSC delta-controller contract (one Panda) -----------
CONTROLLER_CONTROL_DIM = 6
CONTROLLER_INPUT_MIN = -1.0
CONTROLLER_INPUT_MAX = 1.0
CONTROLLER_OUTPUT_MIN = (-0.05, -0.05, -0.05, -0.5, -0.5, -0.5)
CONTROLLER_OUTPUT_MAX = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
OSC_LIMIT_TOLERANCE = 1e-9

# --- the fixed open-loop servo ----------------------------------------------
# The EXACT sent action (np.float32, shape 7):
#   translation[i] = clip(0.5 * (target - current)[i] / 0.05, -0.35, +0.35)
#   rotation[i]    = clip(0.5 * rotvec(target_ori @ current_ori.T)[i] / 0.5,
#                         -0.25, +0.25)
#   gripper        = -1  (every servo action opens the already-empty hand)
# The gain is exactly 0.5 and the clamp is exactly ±0.35 / ±0.25; a gain of 1
# and a ±1 unit clamp are forbidden.
SERVO_GAIN = 0.5
SERVO_TRANSLATION_SCALE = (0.05, 0.05, 0.05)
SERVO_ROTATION_SCALE = (0.5, 0.5, 0.5)
SERVO_TRANSLATION_CLAMP = 0.35
SERVO_ROTATION_CLAMP = 0.25
SERVO_GRIPPER_COMMAND = -1.0
SERVO_POSITION_TOLERANCE_M = 0.005
SERVO_ROTATION_TOLERANCE_RAD = 0.05
SERVO_SUCCESS_STREAK = 5
LIFT_STAGE_MAX_ACTIONS = 40
ALIGN_STAGE_MAX_ACTIONS = 60
LIFT_CLEAR_M = 0.10
MIN_OPEN_GAP_M = 0.07

# The fixed failure reasons that are PHYSICAL preparation outcomes (the entry's
# own preregistered gate) rather than infrastructure faults.
PREPARATION_PHYSICAL_REASONS = frozenset(
    {
        "grasp_observation_incomplete",
        "held_objects_not_empty",
        "bowl_predicate_false",
        "protection_baseline_incomplete",
        "protection_violation",
        "action_cap_exceeded",
        "lift_target_not_reached",
        "align_target_not_reached",
        "final_held_objects_not_empty",
        "final_gap_too_small",
        "final_strict_not_true",
    }
)

PREPARATION_GOALS: list[list[str]] = [list(BOWL_GOAL)]

ASSISTED_NOTE = (
    "the preparation is an explicit, preregistered open-loop servo performed by "
    "this module after the recorded bowl prefix and before the wine subgoal; it "
    "is NOT a learned VLA skill, NOT a pure VLA trial and NOT a production feature"
)

LIMITATIONS: dict[str, str] = {
    "assisted_not_vla": ASSISTED_NOTE,
    "recorded_policy_replay": (
        "the bowl prefix is the preserved 102-action recorded-policy replay through "
        "env.step; it is not a freshly generated VLA bowl execution, a recovery, a "
        "repair or a forced release"
    ),
    "open_hand_not_release": (
        "the hand is already provably empty (native bowl predicate True, held "
        "objects empty, grasp observation complete) BEFORE and AFTER every servo "
        "action, so opening the empty hand can never assist the release of a held "
        "wine bottle"
    ),
    "protection_proxy": (
        "protection is an exact Euclidean displacement proxy (threshold %.3f m) "
        "over ALL FOUR protected objects measured against their pre-preparation "
        "positions; a missing baseline/current position is a fail-closed violation "
        "and it is not a full contact audit" % PROTECTION_TOLERANCE_M
    ),
    "controller_proxy": (
        "the controller contract is read from the live robosuite OSC delta "
        "controller (one Panda: use_delta / control_dim / input and output limits); "
        "a mismatch is a configuration error, not a physical outcome"
    ),
    "no_success_rate": (
        "two prescribed entries are not a statistical reliability claim and no "
        "success rate is inferred"
    ),
    "no_assessment": (
        "no assessment action, repair, forced release, recovery, object move or "
        "Hermes call is performed"
    ),
    "no_source_edits": (
        "this module only READS the accepted sibling modules and the read-only "
        "baseline report; it never writes, patches or re-generates them"
    ),
}


# --- small helpers -----------------------------------------------------------


def _err(reason: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "reason": reason, "detail": detail}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _progress(message: str) -> None:
    sys.stderr.write("[preparation %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _finite_scalar(value: Any) -> float | None:
    """A finite float scalar, or ``None`` for a missing/malformed/nonfinite one."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_vec3(value: Any) -> list[float] | None:
    """A finite 3-vector, or ``None`` when missing/malformed/nonfinite."""

    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        sequence = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if len(sequence) < 3:
        return None
    vector = sequence[:3]
    if any(not math.isfinite(component) for component in vector):
        return None
    return vector


def _finite_matrix3(value: Any) -> np.ndarray | None:
    """A finite 3x3 float64 matrix, or ``None``."""

    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    except Exception:  # noqa: BLE001 - malformed shape/dtype is unknown
        return None
    if not bool(np.all(np.isfinite(matrix))):
        return None
    return matrix


def _as_float_list(value: Any) -> list[float] | None:
    """A flat finite float list of whatever length ``value`` actually has."""

    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        sequence = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if not sequence or any(not math.isfinite(component) for component in sequence):
        return None
    return sequence


def _norm3(first: Any, second: Any) -> float | None:
    """The finite Euclidean distance between two 3-vectors, or ``None``."""

    left = _finite_vec3(first)
    right = _finite_vec3(second)
    if left is None or right is None:
        return None
    distance = math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))
    return distance if math.isfinite(distance) else None


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _close(first: Any, second: Any, tolerance: float = OSC_LIMIT_TOLERANCE) -> bool:
    """Whether two finite numbers are equal within a fixed absolute tolerance."""

    left = _finite_scalar(first)
    right = _finite_scalar(second)
    if left is None or right is None:
        return False
    return abs(left - right) <= float(tolerance)


# --- world-frame orientation error (pure except for an optional scipy seam) ---


def _matrix_to_rotvec_numpy(matrix: Any) -> np.ndarray:
    """Rotation matrix -> axis-angle vector, pure numpy (Shepperd's method)."""

    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    w = math.sqrt(max(0.0, 1.0 + trace)) / 2.0
    if w < 1e-8:
        # Angle ~ pi: recover the axis from the symmetric part of R.
        x = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) / 2.0
        y = math.sqrt(max(0.0, 1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2])) / 2.0
        z = math.sqrt(max(0.0, 1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2])) / 2.0
        if rotation[2, 1] - rotation[1, 2] < 0.0:
            x = -x
        if rotation[0, 2] - rotation[2, 0] < 0.0:
            y = -y
        if rotation[1, 0] - rotation[0, 1] < 0.0:
            z = -z
        vector = np.asarray([x, y, z], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return vector / norm * math.pi
    denominator = 4.0 * w
    vector = np.asarray(
        [
            (rotation[2, 1] - rotation[1, 2]) / denominator,
            (rotation[0, 2] - rotation[2, 0]) / denominator,
            (rotation[1, 0] - rotation[0, 1]) / denominator,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, w)
    return vector / norm * angle


def _matrix_to_rotvec(matrix: Any) -> np.ndarray:
    """The named seam turning a rotation matrix into an axis-angle vector.

    ``scipy.spatial.transform.Rotation.from_matrix(...).as_rotvec()`` is used when
    scipy is importable; the pure-numpy implementation is a documented, exact
    fallback so the GPU-free tests never need scipy.
    """

    try:
        from scipy.spatial.transform import Rotation  # noqa: PLC0415 - optional
    except Exception:  # noqa: BLE001 - scipy is optional; fall back to numpy
        return _matrix_to_rotvec_numpy(matrix)
    try:
        return np.asarray(
            Rotation.from_matrix(np.asarray(matrix, dtype=np.float64).reshape(3, 3)).as_rotvec(),
            dtype=np.float64,
        ).reshape(3)
    except Exception:  # noqa: BLE001 - a failing conversion falls back
        return _matrix_to_rotvec_numpy(matrix)


def orientation_error_rad(current_orientation: Any, target_orientation: Any) -> float | None:
    """The world-frame misalignment (radians) between two end-effector frames."""

    current = _finite_matrix3(current_orientation)
    target = _finite_matrix3(target_orientation)
    if current is None or target is None:
        return None
    error = _matrix_to_rotvec(target @ current.T)
    if not bool(np.all(np.isfinite(error))):
        return None
    magnitude = float(np.linalg.norm(error))
    return magnitude if math.isfinite(magnitude) else None


# --- the pure open-loop servo -------------------------------------------------


def servo_action(
    target_position: Any,
    current_position: Any,
    target_orientation: Any,
    current_orientation: Any,
) -> np.ndarray:
    """One EXACT 7-D ``np.float32`` OSC delta action driving the hand to a target.

    ``translation[i] = clip(0.5 * (target - current)[i] / 0.05, -0.35, +0.35)``
    and ``rotation[i] = clip(0.5 * rotvec(target_ori @ current_ori.T)[i] / 0.5,
    -0.25, +0.25)``; the gripper channel is the fixed open command ``-1``.  The
    gain is exactly 0.5 and the clamp is exactly ``±0.35`` / ``±0.25`` -- never a
    ``±1`` unit clamp.  Every returned action therefore opens the already-empty
    hand.  This function never reads or mutates the simulator.
    """

    current = _finite_vec3(current_position)
    target = _finite_vec3(target_position)
    if current is None or target is None:
        raise ValueError("servo_action needs two finite 3-D positions")
    current_matrix = _finite_matrix3(current_orientation)
    target_matrix = _finite_matrix3(target_orientation)
    if current_matrix is None or target_matrix is None:
        raise ValueError("servo_action needs two finite 3x3 orientations")

    rotation = _matrix_to_rotvec(target_matrix @ current_matrix.T)
    channels: list[float] = []
    for index in range(3):
        error = float(target[index]) - float(current[index])
        channels.append(
            _clip(
                SERVO_GAIN * error / SERVO_TRANSLATION_SCALE[index],
                -SERVO_TRANSLATION_CLAMP,
                SERVO_TRANSLATION_CLAMP,
            )
        )
    for index in range(3):
        channels.append(
            _clip(
                SERVO_GAIN * float(rotation[index]) / SERVO_ROTATION_SCALE[index],
                -SERVO_ROTATION_CLAMP,
                SERVO_ROTATION_CLAMP,
            )
        )
    channels.append(_clip(SERVO_GRIPPER_COMMAND, -1.0, 1.0))
    return np.asarray(channels, dtype=np.float32)


# --- the live OSC delta-controller contract (worker thread only) -------------


def _controller_of(env: Any) -> Any:
    """The single live robosuite robot controller, or a ``SceneError``."""

    inner = service._inner_env(env)
    robots = getattr(inner, "robots", None)
    if not isinstance(robots, (list, tuple)) or len(robots) != 1:
        raise service.SceneError(
            "controller_unavailable",
            "expected exactly one robot, found %s" % (len(robots) if isinstance(robots, (list, tuple)) else None,),
        )
    controller = getattr(robots[0], "controller", None)
    if controller is None:
        raise service.SceneError("controller_unavailable", "robot exposes no controller")
    return controller


def controller_facts(env: Any) -> dict[str, Any]:
    """Read the live controller contract + the current end-effector pose (read-only).

    Every field is either the measured value or an explicit ``None``; an
    unreadable contract is recorded as an ``error`` string and never invented.
    ``controller.update(force=True)`` only refreshes robosuite's internal
    kinematics -- it neither steps nor resets the simulator.
    """

    facts: dict[str, Any] = {
        "n_robots": None,
        "controller_type": None,
        "use_delta": None,
        "control_dim": None,
        "input_min": None,
        "input_max": None,
        "output_min": None,
        "output_max": None,
        "ee_pos": None,
        "ee_ori_mat": None,
        "error": None,
    }
    try:
        inner = service._inner_env(env)
    except Exception as exc:  # noqa: BLE001 - no inner env -> everything unknown
        facts["error"] = "inner env unavailable: %s" % exc
        return facts
    robots = getattr(inner, "robots", None)
    count = len(robots) if isinstance(robots, (list, tuple)) else None
    facts["n_robots"] = count
    if count != 1:
        facts["error"] = "expected exactly one robot, found %r" % (count,)
        return facts
    controller = getattr(robots[0], "controller", None)
    if controller is None:
        facts["error"] = "robot exposes no controller"
        return facts
    facts["controller_type"] = type(controller).__name__
    try:
        controller.update(force=True)
    except Exception as exc:  # noqa: BLE001 - a controller that cannot update is unusable
        facts["error"] = "controller.update(force=True) failed: %s" % exc
        return facts
    facts["use_delta"] = getattr(controller, "use_delta", None)
    facts["control_dim"] = getattr(controller, "control_dim", None)
    facts["input_min"] = _as_float_list(getattr(controller, "input_min", None))
    facts["input_max"] = _as_float_list(getattr(controller, "input_max", None))
    facts["output_min"] = _as_float_list(getattr(controller, "output_min", None))
    facts["output_max"] = _as_float_list(getattr(controller, "output_max", None))
    facts["ee_pos"] = _finite_vec3(getattr(controller, "ee_pos", None))
    matrix = _finite_matrix3(getattr(controller, "ee_ori_mat", None))
    facts["ee_ori_mat"] = None if matrix is None else matrix.tolist()
    return facts


def _limit_channels_match(values: Any, expected: Any) -> bool:
    if not isinstance(values, list) or len(values) != len(expected):
        return False
    return all(_close(value, target) for value, target in zip(values, expected))


def controller_mismatch(facts: Any) -> str | None:
    """Why the live controller does not match the fixed delta contract, or ``None``.

    The contract is the accepted 6-D robosuite OSC delta controller: a genuine
    ``use_delta is True``, an integer ``control_dim`` of 6, ``[-1, +1]`` input
    limits, the fixed ``±[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`` output limits, and a
    readable finite end-effector position and orientation.  Anything else is a
    mismatch -- nothing is coerced, and the caller must reject it *before* moving.
    """

    if not isinstance(facts, dict):
        return "controller facts unavailable"
    if facts.get("error"):
        return str(facts["error"])
    if facts.get("use_delta") is not True:
        return "use_delta is not exactly True: %r" % (facts.get("use_delta"),)
    control_dim = facts.get("control_dim")
    if type(control_dim) is not int or control_dim != CONTROLLER_CONTROL_DIM:
        return "control_dim is not the integer %d: %r" % (CONTROLLER_CONTROL_DIM, control_dim)
    for key in ("input_min", "input_max", "output_min", "output_max"):
        values = facts.get(key)
        if not isinstance(values, list) or len(values) != CONTROLLER_CONTROL_DIM:
            return "%s is not a %d-channel vector: %r" % (key, CONTROLLER_CONTROL_DIM, values)
    if not all(_close(value, CONTROLLER_INPUT_MIN) for value in facts["input_min"]):
        return "input_min is not all %+g: %r" % (CONTROLLER_INPUT_MIN, facts["input_min"])
    if not all(_close(value, CONTROLLER_INPUT_MAX) for value in facts["input_max"]):
        return "input_max is not all %+g: %r" % (CONTROLLER_INPUT_MAX, facts["input_max"])
    if not _limit_channels_match(facts["output_min"], list(CONTROLLER_OUTPUT_MIN)):
        return "output_min is not the fixed delta contract: %r" % (facts["output_min"],)
    if not _limit_channels_match(facts["output_max"], list(CONTROLLER_OUTPUT_MAX)):
        return "output_max is not the fixed delta contract: %r" % (facts["output_max"],)
    if _finite_vec3(facts.get("ee_pos")) is None:
        return "ee_pos is not a finite 3-vector: %r" % (facts.get("ee_pos"),)
    if _finite_matrix3(facts.get("ee_ori_mat")) is None:
        return "ee_ori_mat is not a finite 3x3 matrix"
    return None


def read_eef_pose(env: Any) -> dict[str, Any]:
    """The current end-effector pose (worker-owned, read-only)."""

    controller = _controller_of(env)
    controller.update(force=True)
    position = _finite_vec3(getattr(controller, "ee_pos", None))
    matrix = _finite_matrix3(getattr(controller, "ee_ori_mat", None))
    if position is None or matrix is None:
        raise service.SceneError("controller_unavailable", "end-effector pose is unreadable")
    return {
        "position": list(position),
        "orientation_matrix": matrix,
        "orientation": matrix.tolist(),
    }


# --- protection (read-only displacement proxy) -------------------------------


def snapshot_gate_violations(snapshot: Any) -> list[str]:
    """The before/after gate: native bowl True, hand empty, grasp complete.

    Every required field must be the exact expected value; a missing/malformed
    value is unknown and is reported as a violation (never silently accepted).
    """

    if not isinstance(snapshot, dict):
        return ["snapshot_unavailable"]
    violations: list[str] = []
    predicates = snapshot.get("predicates")
    bowl = predicates.get(BOWL_GOAL_KEY) if isinstance(predicates, dict) else None
    if bowl is not True:
        violations.append("bowl_predicate_false: %r" % (bowl,))
    held = snapshot.get("held_objects")
    if not isinstance(held, list) or held:
        violations.append("held_objects_not_empty: %r" % (held,))
    if snapshot.get("grasp_observation_complete") is not True:
        violations.append("grasp_observation_incomplete")
    return violations


def snapshot_positions(snapshot: Any) -> dict[str, list[float] | None]:
    """The finite world position of every protected object from a snapshot.

    Only the native snapshot ``objects[object_id].position`` is read; an object
    that is missing, or whose position is unreadable/nonfinite, maps to ``None``
    (absence is never silently reported as ``[0, 0, 0]``).
    """

    positions: dict[str, list[float] | None] = {object_id: None for object_id in PROTECTED_OBJECTS}
    objects = snapshot.get("objects") if isinstance(snapshot, dict) else None
    if not isinstance(objects, dict):
        return positions
    for object_id in PROTECTED_OBJECTS:
        entry = objects.get(object_id)
        if isinstance(entry, dict):
            positions[object_id] = _finite_vec3(entry.get("position"))
    return positions


def protection_baseline_violations(baseline: Any) -> list[str]:
    """The four protected objects missing/nonfinite at the pre-preparation baseline."""

    baseline_map = baseline if isinstance(baseline, dict) else {}
    return [
        "protected_baseline_missing: %s" % object_id
        for object_id in PROTECTED_OBJECTS
        if _finite_vec3(baseline_map.get(object_id)) is None
    ]


def protection_violations(
    baseline: Any,
    current: Any,
    tolerance: float = PROTECTION_TOLERANCE_M,
) -> list[str]:
    """Every protected object missing or moved beyond ``tolerance`` (fail-closed).

    ALL FOUR protected objects are targets: a baseline position that is missing
    or nonfinite is itself a violation, and an object that becomes unreadable
    afterwards is a violation -- unknown is never silently accepted.
    """

    baseline_map = baseline if isinstance(baseline, dict) else {}
    current_map = current if isinstance(current, dict) else {}
    limit = float(tolerance)
    violations: list[str] = []
    for object_id in PROTECTED_OBJECTS:
        base = _finite_vec3(baseline_map.get(object_id))
        if base is None:
            violations.append("protected_object_unknown: %s" % object_id)
            continue
        now = _finite_vec3(current_map.get(object_id))
        if now is None:
            violations.append("protected_object_unknown: %s" % object_id)
            continue
        distance = math.sqrt(sum((base[index] - now[index]) ** 2 for index in range(3)))
        if not math.isfinite(distance) or distance > limit:
            violations.append(
                "protected_object_moved: %s %.4f m > %.4f m" % (object_id, distance, limit)
            )
    return violations


# --- preparation evidence ----------------------------------------------------


def preparation_evidence_dir(svc: Any, session_id: Any) -> Path | None:
    """The directory the preparation evidence is written to (best effort)."""

    record = None
    sessions = getattr(svc, "_sessions", None)
    if isinstance(sessions, dict):
        record = sessions.get(session_id)
    run_dir = getattr(record, "run_dir", None) if record is not None else None
    if run_dir:
        return Path(run_dir) / PREPARATION_EVIDENCE_DIRNAME
    root = getattr(svc, "run_root", None)
    if root:
        return Path(root) / ("%s_%s" % (PREPARATION_EVIDENCE_DIRNAME, session_id))
    return None


def save_preparation_evidence(evidence_dir: Any, result: dict[str, Any]) -> None:
    """Persist the real float64 states, the per-step event log and the summary."""

    if not evidence_dir:
        return
    try:
        directory = Path(evidence_dir)
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 - evidence saving never breaks the run
        return
    for key, filename in (
        ("before_prepare_state", "preparation_before_state.npy"),
        ("after_prepare_state", "preparation_after_state.npy"),
        ("state_flatten", "preparation_state.npy"),
    ):
        try:
            flat = result.get(key)
            if flat is not None:
                np.save(
                    str(directory / filename),
                    np.ascontiguousarray(np.asarray(flat, dtype=np.float64)),
                )
        except Exception:  # noqa: BLE001
            pass
    try:
        events = result.get("events")
        if events is not None:
            with (directory / PREPARATION_EVENTS_FILENAME).open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass
    try:
        summary = {
            key: value
            for key, value in result.items()
            if key not in ("events", "before_prepare_state", "after_prepare_state", "state_flatten")
        }
        (directory / PREPARATION_SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass


# --- the preparation service -------------------------------------------------


class PreparationBlocked(RuntimeError):
    """Raised from the ``_sync_work`` hook when the physical preparation failed."""


# A unique, targeted marker: the derived operational-error filter matches ONLY
# messages carrying the PreparationBlocked exception type -- never a broad
# string filter and never a rewrite of the raw error.
BLOCKED_MARKER = "preparation_diagnostics.PreparationBlocked"


class PreparedService(context.ContextDiagnosticService):
    """The accepted context service plus ONE worker-owned preparation hook.

    Everything else -- the worker thread, the ``strict=True`` model load, the
    ``release_verified`` completion gate, the ``shadow`` wine-only grasp guard,
    the recorded-prefix replay, the inherited first-input fingerprint capture,
    ``events.jsonl`` / PNG / MP4 and the wine telemetry -- is inherited verbatim.

    Exactly two methods are overridden:

    * ``_do_create_session`` validates the live OSC delta-controller contract
      immediately after the session's single reset -- i.e. **before any movement**
      -- tears the session down on a mismatch, and captures the ORIGINAL initial
      hand target (``controller.update(force=True)`` then ``ee_pos`` /
      ``ee_ori_mat``);
    * ``_sync_work`` delegates to the base implementation unchanged and, ONLY when
      the ``replay_prefix`` result is exactly ``ok``, dispatches the preparation
      with a DIFFERENT kind (``prepare_after_prefix``) through the base
      ``_sync_work``.  A failed preparation stores ``preparation_result`` and
      RAISES :class:`PreparationBlocked` from the hook -- strictly before the
      context model RNG reseed, the first-input capture, any inference and any
      wine submission.  On success the ORIGINAL prefix result is returned
      unchanged.

    The hook can never recurse (the inner kind differs), and every simulator
    operation is worker-owned.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("completion_mode", COMPLETION_MODE)
        kwargs.setdefault("grasp_guard_mode", GRASP_GUARD_MODE)
        super().__init__(*args, **kwargs)
        self.preparation_result: dict | None = None
        self.prefix_result: dict | None = None
        self._prepared_session_id: Any = None
        self._pre_bowl_pose: dict | None = None

    # -- controller contract + original initial target, before any movement -----

    def _do_create_session(self, record: Any, seed: int, init_state_index: int) -> dict[str, Any]:
        """Create the session, then require the fixed one-Panda controller contract.

        Runs on the worker thread (it *is* the ``create_session`` worker task), so
        the controller read is worker-owned.  A mismatch is detected immediately
        after the single logical reset and before any prefix replay or auxiliary
        step: the environment is closed on the worker and the trial is refused.
        """

        created = super()._do_create_session(record, seed, init_state_index)
        if not isinstance(created, dict) or not created.get("ok"):
            return created
        env = self._env
        facts = controller_facts(env)
        mismatch = controller_mismatch(facts)
        if mismatch is not None:
            self._close_env()
            return service._err("controller_mismatch", mismatch)
        self._prepared_session_id = record.session_id
        try:
            self._pre_bowl_pose = read_eef_pose(env)
        except Exception:  # noqa: BLE001 - an unreadable initial target is recorded as unknown
            self._pre_bowl_pose = None
        return created

    # -- the preparation hook --------------------------------------------------

    def _sync_work(self, kind: str, fn: Any, timeout: float = service.WORKER_WAIT_S) -> dict[str, Any]:
        """Delegate to the base worker plumbing, then prepare after a good prefix.

        ``super()._sync_work(kind, fn, timeout)`` is called UNCHANGED.  Only a
        successful ``replay_prefix`` triggers a second base ``_sync_work`` call
        (``prepare_after_prefix``).  A failed preparation raises
        :class:`PreparationBlocked` -- before the caller can reseed the model RNG,
        arm the capture, run inference or submit anything -- and the ORIGINAL
        prefix result is returned unchanged on success.
        """

        result = super()._sync_work(kind, fn, timeout)
        if kind == "replay_prefix" and isinstance(result, dict) and result.get("ok") is True:
            # Preserve the ORIGINAL successful prefix evidence on the service.
            self.prefix_result = result
            outcome = super()._sync_work("prepare_after_prefix", self._prepare, PREPARE_TIMEOUT_S)
            self.preparation_result = (
                outcome
                if isinstance(outcome, dict)
                else {
                    "ok": False,
                    "reason": "prepare_worker_error",
                    "kind": "operational",
                    "detail": repr(outcome),
                }
            )
            if self.preparation_result.get("ok") is not True:
                reason = self.preparation_result.get("reason")
                detail = self.preparation_result.get("detail")
                if self.preparation_result.get("kind") == "physical":
                    raise PreparationBlocked(
                        "physical preparation failed before the wine subgoal: reason=%s detail=%s"
                        % (reason, detail)
                    )
                # A configuration/infrastructure failure is NOT a physical block:
                # it stays operational and surfaces as an ordinary RuntimeError, so
                # the raw operational errors and the ``operational_error`` status
                # are retained by the caller.
                raise RuntimeError(
                    "operational preparation failure before the wine subgoal: reason=%s detail=%s"
                    % (reason, detail)
                )
        return result

    def _prepare(self) -> dict[str, Any]:
        """The worker thunk: prepare the session recorded by the last prefix."""

        session_id = self._prepared_session_id
        return self._prepare_work(session_id, preparation_evidence_dir(self, session_id))

    # -- the worker-owned preparation -----------------------------------------

    def _prepare_work(self, session_id: Any, evidence_dir: Any) -> dict[str, Any]:
        """Validate, servo the two fixed stages, then verify (worker-owned).

        Runs entirely on the worker thread and never raises -- every outcome is a
        record.  BEFORE and AFTER every real ``env.step`` it reads a read-only
        ``capture_snapshot`` and requires the native bowl predicate True, the hand
        empty, the grasp observation complete and all four protected objects
        present/finite within exactly ``PROTECTION_TOLERANCE_M`` of their
        pre-preparation positions; an unknown or violation prevents the next step.
        It never produces a VLA action, a reset, a ``set_state``, a teleport or an
        object move.
        """

        result: dict[str, Any] = {
            "ok": False,
            "reason": None,
            "kind": None,
            "detail": "",
            "pipeline": "lift_then_align",
            "expected_before_state_sha": REPLAY_FINAL_SHA,
            "before_prepare_state_sha": None,
            "after_prepare_state_sha": None,
            "controller": None,
            "controller_mismatch": None,
            "pre_snapshot": None,
            "pre_gate_violations": [],
            "protection_baseline": None,
            "protection_baseline_violations": [],
            "protection_violations": [],
            "protection_tolerance_m": PROTECTION_TOLERANCE_M,
            "lift_target_position": None,
            "lift": None,
            "align": None,
            "final_snapshot": None,
            "final_gate_violations": [],
            "final_bowl_predicate": None,
            "final_strict": None,
            "final_held_objects": None,
            "final_grasp_observation_complete": None,
            "gripper_gap_after": None,
            "failure_step": None,
            "failed_action": None,
            "events": [],
            "aux_actions": 0,
            "aux_action_cap": AUX_ACTION_CAP,
            "evidence_dir": str(evidence_dir) if evidence_dir else None,
        }

        sessions = getattr(self, "_sessions", None)
        record = sessions.get(session_id) if isinstance(sessions, dict) else None
        env = getattr(self, "_env", None)
        counter = [int(getattr(self, "_total_steps", 0) or 0)]
        base_steps = counter[0]

        def _finish(reason: str | None, kind: str | None, detail: str = "") -> dict[str, Any]:
            result["ok"] = reason is None
            result["reason"] = reason
            result["kind"] = kind
            if detail:
                result["detail"] = detail
            # ALWAYS record the ACTUAL auxiliary count, even when it failed.
            result["aux_actions"] = counter[0] - base_steps
            if env is not None:
                try:
                    result["after_prepare_state_sha"] = service.state_sha(env)
                except Exception as exc:  # noqa: BLE001
                    result["after_prepare_state_sha"] = None
                    result["after_state_error"] = _format_exc(exc)
                try:
                    flat, source = context._flatten_sim_state(env)
                    result["after_prepare_state"] = flat
                    result["after_prepare_state_source"] = source
                    result["after_prepare_state_sha256"] = _sha256_hex(
                        np.ascontiguousarray(np.asarray(flat, dtype=np.float64)).tobytes()
                    )
                except Exception:  # noqa: BLE001 - a missing flatten is recorded, never faked
                    result["after_prepare_state_source"] = "unavailable"
            save_preparation_evidence(evidence_dir, result)
            return result

        if env is None or record is None:
            return _finish("no_env", "operational", "no live environment / session record")

        try:
            # -- save the BEFORE preparation float64 state ----------------------
            try:
                before_flat, before_source = context._flatten_sim_state(env)
                result["before_prepare_state"] = before_flat
                result["before_prepare_state_source"] = before_source
                result["before_prepare_state_sha"] = service.state_sha(env)
            except Exception as exc:  # noqa: BLE001
                return _finish("state_unreadable", "operational", _format_exc(exc))
            if result["before_prepare_state_sha"] != REPLAY_FINAL_SHA:
                return _finish(
                    "state_sha_mismatch",
                    "operational",
                    "actual %r != expected %r"
                    % (result["before_prepare_state_sha"], REPLAY_FINAL_SHA),
                )

            facts = controller_facts(env)
            result["controller"] = facts
            mismatch = controller_mismatch(facts)
            result["controller_mismatch"] = mismatch
            if mismatch is not None:
                return _finish("controller_mismatch", "operational", mismatch)

            def _step(action: Any) -> None:
                sent = np.asarray(action, dtype=np.float32).reshape(-1)
                step_result = env.step(sent)
                if isinstance(step_result, tuple) and len(step_result) >= 1:
                    self._last_obs = step_result[0]
                counter[0] += 1
                self._total_steps = counter[0]
                record.total_steps = counter[0]

            def _snapshot() -> Any:
                return pe.capture_snapshot(env, PREPARATION_GOALS)

            def _gap() -> float | None:
                try:
                    probe = grasp_guard.read_probe(env, grasp_guard.WINE_OBJECT_ID, grasp_guard.WINE_GOAL_KEY)
                except Exception:  # noqa: BLE001 - an unreadable probe is unknown, never a False
                    return None
                if not isinstance(probe, dict):
                    return None
                value = probe.get("gap")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                number = float(value)
                return number if math.isfinite(number) else None

            def _log(phase: str, index: int, action: Any, snapshot: Any, violations: Any) -> None:
                result["events"].append(
                    {
                        "phase": phase,
                        "index": int(index),
                        "action": (
                            None
                            if action is None
                            else [float(v) for v in np.asarray(action, dtype=np.float32).reshape(-1)]
                        ),
                        "action_dtype": None if action is None else "float32",
                        "snapshot": snapshot,
                        "violations": list(violations),
                    }
                )

            def _guard(phase: str, index: int, action: Any) -> list[str]:
                snapshot = _snapshot()
                violations = snapshot_gate_violations(snapshot) + protection_violations(
                    baseline, snapshot_positions(snapshot), PROTECTION_TOLERANCE_M
                )
                _log(phase, index, action, snapshot, violations)
                return violations

            # -- preconditions + the pre-preparation protection baseline --------
            pre_snapshot = _snapshot()
            result["pre_snapshot"] = pre_snapshot
            pre_gate = snapshot_gate_violations(pre_snapshot)
            result["pre_gate_violations"] = list(pre_gate)
            if pre_gate:
                return _finish(pre_gate[0].split(":", 1)[0], "physical", "; ".join(pre_gate))
            baseline = snapshot_positions(pre_snapshot)
            result["protection_baseline"] = {key: value for key, value in baseline.items()}
            baseline_bad = protection_baseline_violations(baseline)
            result["protection_baseline_violations"] = list(baseline_bad)
            if baseline_bad:
                return _finish("protection_baseline_incomplete", "physical", "; ".join(baseline_bad))

            def _run_stage(
                stage_name: str, target_position: Any, target_orientation: Any, max_actions: int
            ) -> tuple[bool, str | None, str, list[dict[str, Any]], int]:
                trace: list[dict[str, Any]] = []
                streak = 0
                last: tuple[float, float] | None = None
                for index in range(1, int(max_actions) + 1):
                    if counter[0] - base_steps >= AUX_ACTION_CAP:
                        return (
                            False,
                            "action_cap_exceeded",
                            "auxiliary action cap %d reached during the %s stage"
                            % (AUX_ACTION_CAP, stage_name),
                            trace,
                            index,
                        )
                    before_bad = _guard("%s_before" % stage_name, index, None)
                    if before_bad:
                        result["failure_step"] = index
                        return False, "protection_violation", "; ".join(before_bad), trace, index
                    pose = read_eef_pose(env)
                    action = servo_action(
                        target_position, pose["position"], target_orientation, pose["orientation_matrix"]
                    )
                    _step(action)
                    after_bad = _guard("%s_after" % stage_name, index, action)
                    if after_bad:
                        result["failure_step"] = index
                        result["failed_action"] = [
                            float(v) for v in np.asarray(action, dtype=np.float32).reshape(-1)
                        ]
                        return False, "protection_violation", "; ".join(after_bad), trace, index
                    pose = read_eef_pose(env)
                    position_error = math.sqrt(
                        sum(
                            (float(pose["position"][axis]) - float(target_position[axis])) ** 2
                            for axis in range(3)
                        )
                    )
                    rotation_error = orientation_error_rad(pose["orientation_matrix"], target_orientation)
                    if rotation_error is None:
                        return (
                            False,
                            "orientation_unreadable",
                            "the end-effector orientation is unreadable during the %s stage" % stage_name,
                            trace,
                            index,
                        )
                    trace.append(
                        {
                            "index": index,
                            "action": [float(v) for v in np.asarray(action, dtype=np.float32).reshape(-1)],
                            "ee_position": list(pose["position"]),
                            "position_error_m": position_error,
                            "rotation_error_rad": rotation_error,
                        }
                    )
                    last = (position_error, rotation_error)
                    if (
                        position_error <= SERVO_POSITION_TOLERANCE_M
                        and rotation_error <= SERVO_ROTATION_TOLERANCE_RAD
                    ):
                        streak += 1
                        if streak >= SERVO_SUCCESS_STREAK:
                            return True, None, "", trace, index
                    else:
                        streak = 0
                detail = "target not reached within %d actions" % int(max_actions)
                if last is not None:
                    detail += " (last position error %.4f m, rotation error %.4f rad)" % last
                return False, "%s_target_not_reached" % stage_name, detail, trace, int(max_actions)

            # -- stage 1: vertical lift clear of the bowl -----------------------
            start_pose = read_eef_pose(env)
            current_z = float(start_pose["position"][2])
            original_z = None
            if isinstance(self._pre_bowl_pose, dict):
                original_position = _finite_vec3(self._pre_bowl_pose.get("position"))
                if original_position is not None:
                    original_z = original_position[2]
            lift_target = [
                float(start_pose["position"][0]),
                float(start_pose["position"][1]),
                max(original_z if original_z is not None else current_z, current_z + LIFT_CLEAR_M),
            ]
            result["lift_target_position"] = list(lift_target)
            lifted, lift_reason, lift_detail, lift_trace, lift_steps = _run_stage(
                "lift", lift_target, start_pose["orientation_matrix"], LIFT_STAGE_MAX_ACTIONS
            )
            result["lift"] = {
                "ok": lifted,
                "reason": lift_reason,
                "detail": lift_detail,
                "target_position": list(lift_target),
                "steps": lift_steps,
                "trace": lift_trace,
            }
            if not lifted:
                return _finish(lift_reason or "lift_target_not_reached", "physical", lift_detail)

            # -- stage 2: align to the ORIGINAL initial XYZ/orientation ---------
            target_pose = self._pre_bowl_pose
            if not isinstance(target_pose, dict) or _finite_vec3(target_pose.get("position")) is None:
                return _finish(
                    "align_target_unknown",
                    "operational",
                    "the original initial end-effector pose is unavailable",
                )
            aligned, align_reason, align_detail, align_trace, align_steps = _run_stage(
                "align",
                list(target_pose["position"]),
                target_pose["orientation_matrix"],
                ALIGN_STAGE_MAX_ACTIONS,
            )
            result["align"] = {
                "ok": aligned,
                "reason": align_reason,
                "detail": align_detail,
                "target_position": list(target_pose["position"]),
                "steps": align_steps,
                "trace": align_trace,
            }
            if not aligned:
                return _finish(align_reason or "align_target_not_reached", "physical", align_detail)

            # -- final read-only verification -----------------------------------
            final_snapshot = _snapshot()
            result["final_snapshot"] = final_snapshot
            final_gate = snapshot_gate_violations(final_snapshot)
            result["final_gate_violations"] = list(final_gate)
            if final_gate:
                return _finish(final_gate[0].split(":", 1)[0], "physical", "; ".join(final_gate))
            final_held = final_snapshot.get("held_objects") if isinstance(final_snapshot, dict) else None
            result["final_held_objects"] = list(final_held) if isinstance(final_held, list) else None
            result["final_grasp_observation_complete"] = (
                final_snapshot.get("grasp_observation_complete") if isinstance(final_snapshot, dict) else None
            )
            final_predicates = final_snapshot.get("predicates") if isinstance(final_snapshot, dict) else None
            result["final_bowl_predicate"] = (
                final_predicates.get(BOWL_GOAL_KEY) if isinstance(final_predicates, dict) else None
            )
            final_bad = protection_violations(
                baseline, snapshot_positions(final_snapshot), PROTECTION_TOLERANCE_M
            )
            result["protection_violations"] = list(final_bad)
            if final_bad:
                return _finish("protection_violation", "physical", "; ".join(final_bad))
            strict = context._strict_of(final_snapshot)
            result["final_strict"] = strict
            if strict is not True:
                return _finish(
                    "final_strict_not_true",
                    "physical",
                    "context._strict_of(final_bowl_snapshot) is %r" % (strict,),
                )
            gap = _gap()
            result["gripper_gap_after"] = gap
            if gap is None or gap < MIN_OPEN_GAP_M:
                return _finish(
                    "final_gap_too_small",
                    "physical",
                    "the gripper gap is %r < %.3f m" % (gap, MIN_OPEN_GAP_M),
                )
            return _finish(None, None, "")
        except Exception as exc:  # noqa: BLE001 - the preparation never raises
            return _finish("exception", "operational", _format_exc(exc))


# --- source provenance --------------------------------------------------------

SOURCE_FILES = (
    "preparation_diagnostics.py",
    "skill_context_diagnostics.py",
    "service.py",
    "grasp_guard.py",
    "wine_diagnostics.py",
    "placement_experiments.py",
    "guard_validation.py",
    "paired_config_experiments.py",
    "placement_completion.py",
    "catalog.py",
)


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the exact bytes of every module this runner reads."""

    digests: dict[str, str | None] = {}
    for name in SOURCE_FILES:
        try:
            digests[name] = _sha256_hex((_HERE / name).read_bytes())
        except Exception:  # noqa: BLE001 - absent/unreadable is recorded as unknown
            digests[name] = None
    return digests


# --- service construction / readiness / production health --------------------


def _build_service(args: argparse.Namespace) -> PreparedService:
    """Construct the prepared service with the exact required keyword arguments."""

    return PreparedService(
        model_path=service.DEFAULT_MODEL_PATH,
        run_root=str(args.run_root),
        completion_mode=COMPLETION_MODE,
        grasp_guard_mode=GRASP_GUARD_MODE,
    )


def _wait_ready(svc: Any, timeout_s: float = READY_TIMEOUT_S) -> dict[str, Any]:
    """Poll the inherited health method until ready / errored / timed out."""

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


def production_health_gate() -> tuple[dict[str, Any], str | None]:
    """Read production ``/health`` READ-ONLY and require ready / no active request.

    Uses the accepted ``ProxyHandler({})`` reader, so no proxy environment variable
    is ever honoured and no write, stop, reset or cancel is ever sent to
    production.  Returns ``(record, fatal_message)``; a non-``None`` message means
    the trial must not start.
    """

    health = context._production_health(HEALTH_URL, HEALTH_TIMEOUT_S)
    record = context._health_record(health)
    if not health.get("ok"):
        record["gate"] = "unreachable"
        return record, "production health unreachable before the trial: %s" % (
            health.get("reason") or health.get("detail"),
        )
    raw = health.get("raw")
    ready = raw.get("ready") if isinstance(raw, dict) else None
    if ready is not True:
        record["gate"] = "not_ready"
        return record, "production health reports ready=%r; refusing to start a trial" % (ready,)
    if health.get("active_request_id") is not None:
        record["gate"] = "busy"
        return record, (
            "production busy before the trial: active_request_id=%r; refusing to start a "
            "trial (production is never stopped)" % (health["active_request_id"],)
        )
    record["gate"] = "ready_noactive"
    return record, None


# --- the fixed plan -----------------------------------------------------------


def build_campaign_plan() -> list[dict[str, Any]]:
    """The FIXED two-entry order: the SAME pipeline at model seeds 0 then 1."""

    return [
        {
            "entry": "B",
            "model_seed": int(model_seed),
            "pipeline": "lift_then_align",
            "prefix_condition": PREFIX_CONDITION,
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "scene_seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "wine_capability_id": WINE_CAPABILITY_ID,
        }
        for model_seed in MODEL_SEEDS
    ]


# --- per-trial view over the raw context trial -------------------------------


def _is_blocked_message(message: Any) -> bool:
    """Whether an operational message carries the PreparationBlocked marker.

    This is a targeted exception-type marker, never a broad string filter and
    never a rewrite of the raw error.
    """

    return isinstance(message, str) and any(
        line.strip().startswith(("PreparationBlocked:", "preparation_diagnostics.PreparationBlocked:", "__main__.PreparationBlocked:"))
        for line in message.splitlines()
    )


def wine_action_count(raw_trial: Any) -> int:
    """The ACTUAL wine subgoal action count of a raw context trial (0 when none)."""

    if not isinstance(raw_trial, dict):
        return 0
    total = 0
    for evidence in raw_trial.get("jobs") or []:
        if not isinstance(evidence, dict):
            continue
        job = evidence.get("job") or {}
        if job.get("capability_id") != WINE_CAPABILITY_ID:
            continue
        steps = job.get("steps")
        if isinstance(steps, int) and not isinstance(steps, bool) and steps > 0:
            total += steps
    return total


def _cross_seed_check(view: dict[str, Any], seen: dict[str, Any], model_seed: int) -> dict[str, Any]:
    """Compare the two B entries (model seeds 0/1) on preparation/state/input.

    Both entries run the SAME pipeline on the SAME shared scene seed, so their
    prefix inputs, pre/post-preparation state, auxiliary count, raw XML and
    preparation outcome must be equal.  The required physical fields are NON-NULL:
    ``None == None`` is never evidence of a match.  A *successful* preparation has
    a ``None`` reason (compared by equality only, never as "missing") and must
    carry a non-null before-wine state SHA plus a COMPLETE first-input fingerprint
    (``paired.fingerprint_is_complete``) that is identical across both seeds.  A
    *failed* preparation must carry a nonempty reason, must NOT carry a
    fingerprint and is compared by its actual failure preparation state.  The
    shared ``seen`` map keeps the context condition key(s) alongside this ``B``
    key; the raw trial is never mutated.
    """

    prefix = view.get("prefix_result") or {}
    preparation = view.get("preparation") or {}
    raw_trial = view.get("raw_trial") or {}
    ok = preparation.get("ok") is True
    fingerprint = raw_trial.get("first_fingerprint")
    complete = paired.fingerprint_is_complete(fingerprint)
    current = {
        "model_seed": int(model_seed),
        "origin_state_sha": prefix.get("origin_state_sha"),
        "final_state_sha": prefix.get("final_state_sha"),
        "before_prepare_state_sha": preparation.get("before_prepare_state_sha"),
        "after_prepare_state_sha": preparation.get("after_prepare_state_sha"),
        "before_wine_state_sha": raw_trial.get("before_wine_state_sha"),
        "auxiliary_action_count": view.get("auxiliary_action_count"),
        "xml_sha": raw_trial.get("xml_sha"),
        "preparation_ok": ok,
        "preparation_reason": preparation.get("reason"),
        "first_fingerprint_sha256": (
            fingerprint.get("combined_sha256")
            if isinstance(fingerprint, dict) and complete
            else None
        ),
    }
    reference = seen.get("B")
    if reference is None:
        seen["B"] = current
        return {"reference_model_seed": None, "differing": {}}
    differing: dict[str, Any] = {}
    # Required NON-NULL known physical fields: any unknown or mismatch differs.
    for field in (
        "origin_state_sha",
        "final_state_sha",
        "before_prepare_state_sha",
        "after_prepare_state_sha",
        "auxiliary_action_count",
        "xml_sha",
    ):
        left = reference.get(field)
        right = current.get(field)
        if left is None or right is None or left != right:
            differing[field] = [left, right]
    if reference.get("preparation_ok") != ok:
        differing["preparation_ok"] = [reference.get("preparation_ok"), ok]
    # A ``None`` reason is a VALID successful outcome: compare by equality only.
    if reference.get("preparation_reason") != current.get("preparation_reason"):
        differing["preparation_reason"] = [
            reference.get("preparation_reason"),
            current.get("preparation_reason"),
        ]
    if ok:
        # Success: a NON-NULL before-wine SHA and a COMPLETE, identical fingerprint.
        left = reference.get("before_wine_state_sha")
        right = current.get("before_wine_state_sha")
        if left is None or right is None or left != right:
            differing["before_wine_state_sha"] = [left, right]
        if not complete:
            differing["first_fingerprint_incomplete"] = [
                reference.get("first_fingerprint_sha256"),
                None,
            ]
        elif reference.get("first_fingerprint_sha256") != current.get("first_fingerprint_sha256"):
            differing["first_fingerprint_sha256"] = [
                reference.get("first_fingerprint_sha256"),
                current.get("first_fingerprint_sha256"),
            ]
    else:
        # Failure: a nonempty reason and NO fingerprint; the actual failure
        # preparation state is already compared by the required fields above.
        if not (isinstance(current.get("preparation_reason"), str) and current.get("preparation_reason")):
            differing["preparation_reason_missing"] = [None, current.get("preparation_reason")]
        if complete:
            differing["first_fingerprint_unexpected"] = [
                reference.get("first_fingerprint_sha256"),
                current.get("first_fingerprint_sha256"),
            ]
    return {"reference_model_seed": reference.get("model_seed"), "differing": differing}


def _trial_view(entry: dict[str, Any], raw_trial: Any, preparation: Any) -> dict[str, Any]:
    """The OUTER report entry: the raw trial UNCHANGED plus derived fields."""

    raw = raw_trial if isinstance(raw_trial, dict) else {}
    record = preparation if isinstance(preparation, dict) else {}
    # ONLY a PHYSICAL preparation failure blocks the wine subgoal; an operational
    # (configuration/infrastructure) failure keeps its raw operational errors and
    # the ordinary ``operational_error`` status.
    blocked = bool(record) and record.get("ok") is not True and record.get("kind") == "physical"
    raw_operational = list(raw.get("operational_errors") or [])
    if blocked:
        # ONLY PreparationBlocked-tagged messages leave the operational set;
        # every unrelated error stays operational.
        operational = [message for message in raw_operational if not _is_blocked_message(message)]
        status = "physical_preparation_failed"
    else:
        operational = raw_operational
        status = "ok" if not operational else "operational_error"
    physical = list(raw.get("physical_failures") or [])
    if blocked:
        physical.append("preparation_failed: reason=%s" % (record.get("reason"),))
    return {
        "trial_id": raw.get("trial_id"),
        "condition": raw.get("condition"),
        "model_seed": raw.get("model_seed"),
        "status": status,
        "preparation": preparation,
        "raw_trial": raw,
        "operational_errors": operational,
        "physical_failures": physical,
        "wine_action_count": wine_action_count(raw),
        "auxiliary_action_count": record.get("aux_actions"),
        "after_bowl_state_sha": raw.get("after_bowl_state_sha"),
    }


# --- per-trial runner ---------------------------------------------------------


def _run_trial(svc: Any, entry: dict[str, Any], actions: Any, run_root: Path, seen: dict[str, Any]) -> dict[str, Any]:
    """Run the accepted prefix-condition trial, then build the outer view.

    ``skill_context_diagnostics._run_trial`` is reused verbatim (its helpers and
    report framework are NOT copied).  The preparation is applied by the service
    hook during the exact prefix replay; when it fails, the hook raises
    :class:`PreparationBlocked`, which the context runner catches and retains as a
    raw ``trial_exception``.  The raw trial is kept UNCHANGED and the original
    prefix evidence is preserved separately on the service.
    """

    model_seed = int(entry["model_seed"])
    try:
        raw_trial = context._run_trial(
            svc,
            {"condition": PREFIX_CONDITION, "model_seed": model_seed},
            actions,
            run_root,
            seen,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve the raw record
        raw_trial = {
            "trial_id": "%s_m%s" % (PREFIX_CONDITION, model_seed),
            "condition": PREFIX_CONDITION,
            "model_seed": model_seed,
            "errors": ["trial_exception: %s" % _format_exc(exc)],
            "operational_errors": ["trial_exception: %s" % _format_exc(exc)],
            "jobs": [],
            "physical_failures": [],
            "after_bowl_state_sha": None,
        }
    view = _trial_view(entry, raw_trial, svc.preparation_result)
    prefix_result = svc.prefix_result
    view["prefix_result"] = prefix_result
    # The after-bowl final state SHA is derived from the ORIGINAL successful prefix
    # evidence preserved on the service, EVEN when the hook blocked before the wine
    # subgoal; the raw trial itself is NEVER mutated.
    if isinstance(prefix_result, dict) and prefix_result.get("final_state_sha") is not None:
        view["after_bowl_state_sha"] = prefix_result.get("final_state_sha")
    cross = _cross_seed_check(view, seen, model_seed)
    view["cross_seed"] = cross
    if cross["differing"]:
        view["operational_errors"].append("cross_seed_B_mismatch: %r" % (cross["differing"],))
    return view


# --- preregistration and report ----------------------------------------------


def _build_preregistration(
    args: argparse.Namespace, entries: list[dict[str, Any]], actions: Any
) -> dict[str, Any]:
    """The frozen plan (BOTH fixed entries), written BEFORE the model is loaded."""

    return {
        "experiment": EXPERIMENT_NAME,
        "created_utc": _now_utc(),
        "commitment": (
            "written before the model / service starts; the raw SHA-256 of this "
            "file's bytes is recorded in the campaign report and the fixed order is "
            "never re-decided after data collection begins"
        ),
        "selection_rationale": (
            "in the read-only baseline report the seed-1 shared_initial condition "
            "succeeds while its shared_after_bowl condition fails (before-wine state "
            "%s, %d wine steps, success %r); this diagnostic therefore tests the "
            "previously planned handoff preparation after that exact recorded bowl "
            "prefix. Both model seeds (0 and 1) and the whole pipeline are fixed "
            "BEFORE any B outcome is observed, and no reliability is inferred."
            % (BASELINE_BEFORE_WINE_STATE_SHA, BASELINE_WINE_STEPS, BASELINE_WINE_SUCCESS)
        ),
        "baseline": {
            "path": str(args.baseline_report),
            "sha256_expected": BASELINE_REPORT_SHA,
            "condition": BASELINE_CONDITION,
            "before_wine_state_sha": BASELINE_BEFORE_WINE_STATE_SHA,
            "wine_steps": BASELINE_WINE_STEPS,
            "wine_success": BASELINE_WINE_SUCCESS,
        },
        "fixed_spec": {
            "entry": "B",
            "pipeline": "lift_then_align",
            "prefix_condition": PREFIX_CONDITION,
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "scene_seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "model_seeds": [int(seed) for seed in MODEL_SEEDS],
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
            "protected_objects": list(PROTECTED_OBJECTS),
            "protection_tolerance_m": PROTECTION_TOLERANCE_M,
            "aux_action_cap": AUX_ACTION_CAP,
            "servo_gain": SERVO_GAIN,
            "servo_translation_scale": list(SERVO_TRANSLATION_SCALE),
            "servo_rotation_scale": list(SERVO_ROTATION_SCALE),
            "servo_translation_clamp": SERVO_TRANSLATION_CLAMP,
            "servo_rotation_clamp": SERVO_ROTATION_CLAMP,
            "servo_gripper_command": SERVO_GRIPPER_COMMAND,
            "servo_position_tolerance_m": SERVO_POSITION_TOLERANCE_M,
            "servo_rotation_tolerance_rad": SERVO_ROTATION_TOLERANCE_RAD,
            "servo_success_streak": SERVO_SUCCESS_STREAK,
            "lift_stage_max_actions": LIFT_STAGE_MAX_ACTIONS,
            "align_stage_max_actions": ALIGN_STAGE_MAX_ACTIONS,
            "lift_clear_m": LIFT_CLEAR_M,
            "min_open_gap_m": MIN_OPEN_GAP_M,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "target_revision": paired.FIXED_SOURCE_REVISION,
            "baseline_report_sha256": BASELINE_REPORT_SHA,
            "max_wine_trials": MAX_WINE_TRIALS,
            "vla_actions_from_preparation": 0,
            "assessment_actions": 0,
            "forced_release": False,
            "recovery": False,
            "object_moves": 0,
            "resets": 0,
            "hermes_calls": 0,
        },
        "entries": entries,
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
        "assisted_preparation": True,
        "metadata": {
            "output": str(args.output),
            "run_root": str(args.run_root),
            "baseline_report_path": str(args.baseline_report),
            "baseline_report_sha256": None,
            "baseline_report_sha256_expected": BASELINE_REPORT_SHA,
            "baseline_report_sha256_after": None,
            "entry": "B",
            "pipeline": "lift_then_align",
            "model_seeds": [int(seed) for seed in MODEL_SEEDS],
            "max_wine_trials": MAX_WINE_TRIALS,
            "prefix_condition": PREFIX_CONDITION,
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "scene_seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "budget": BUDGET,
            "timeout_s": TIMEOUT_S,
            "profile": PROFILE,
            "completion_mode": COMPLETION_MODE,
            "grasp_guard_mode": GRASP_GUARD_MODE,
            "capability_id": WINE_CAPABILITY_ID,
            "instruction": WINE_INSTRUCTION,
            "model_path": service.DEFAULT_MODEL_PATH,
            "model_revision": None,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "target_revision": paired.FIXED_SOURCE_REVISION,
            "source_git_sha": None,
            "source_sha256": _source_sha256(),
            "replay_action_count": REPLAY_ACTION_COUNT,
            "replay_origin_state_sha": REPLAY_ORIGIN_SHA,
            "replay_final_state_sha": REPLAY_FINAL_SHA,
            "protected_objects": list(PROTECTED_OBJECTS),
            "protection_tolerance_m": PROTECTION_TOLERANCE_M,
            "aux_action_cap": AUX_ACTION_CAP,
            "wine_budget": BUDGET,
            "production_health_url": HEALTH_URL,
            "vla_actions_from_preparation": 0,
            "assessment_actions": 0,
            "forced_release": False,
            "recovery": False,
            "object_moves": 0,
            "resets": 0,
            "hermes_calls": 0,
            "campaign_wall_s": None,
        },
        "preregistration": None,
        "service": {},
        "health_checks": [],
        "trials": [],
        "operational_errors": [],
        "preparation": {
            "pipeline": "lift_then_align",
            "assisted_note": ASSISTED_NOTE,
            "physical_reasons": sorted(PREPARATION_PHYSICAL_REASONS),
        },
        "limitations": dict(LIMITATIONS),
        "aggregate": {},
    }


def _aggregate(trials: list[dict[str, Any]], expected_trials: int) -> dict[str, Any]:
    return {
        "n_trials": len(trials),
        "expected_trials": int(expected_trials),
        "n_status_ok": sum(1 for trial in trials if trial.get("status") == "ok"),
        "n_physical_preparation_failed": sum(
            1 for trial in trials if trial.get("status") == "physical_preparation_failed"
        ),
        "n_operational_error": sum(
            1 for trial in trials if trial.get("status") == "operational_error"
        ),
        "n_operational_errors": sum(1 for trial in trials if trial.get("operational_errors")),
        "n_physical_failures": sum(1 for trial in trials if trial.get("physical_failures")),
        "n_wine_actions": sum(int(trial.get("wine_action_count") or 0) for trial in trials),
        "n_auxiliary_actions": sum(
            int(trial.get("auxiliary_action_count") or 0) for trial in trials
        ),
        "n_protection_violations": sum(
            1
            for trial in trials
            if isinstance(trial.get("preparation"), dict)
            and trial["preparation"].get("protection_violations")
        ),
        "reliability_note": (
            "explicit counts only; two prescribed assisted-preparation entries are "
            "not a statistical reliability claim and no success rate is inferred"
        ),
    }


def _aggregate_operational(trials: list[dict[str, Any]], fatal_error: str | None) -> list[str]:
    messages: list[str] = []
    if fatal_error:
        messages.append("fatal: %s" % fatal_error)
    for trial in trials:
        for message in trial.get("operational_errors") or []:
            messages.append(
                "%s_m%s: %s" % (trial.get("condition"), trial.get("model_seed"), message)
            )
    return messages


# --- the campaign -------------------------------------------------------------


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed two-entry preparation campaign and persist the report."""

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
    expected_trials = 0

    def _persist() -> None:
        report["fatal_error"] = fatal_error
        report["operational_errors"] = _aggregate_operational(trials, fatal_error)
        report["aggregate"] = _aggregate(trials, expected_trials)
        report["metadata"]["campaign_wall_s"] = round(time.monotonic() - started, 3)
        report["ok"] = bool(
            fatal_error is None
            and not persisted_errors
            and len(trials) == expected_trials
            and expected_trials > 0
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
            actions = context.load_actions(args.input_actions)
        except Exception as exc:  # noqa: BLE001 - no actions -> no run
            raise RuntimeError("load_actions: %s" % exc) from exc
        report["metadata"]["input_actions_count"] = len(actions)
        report["metadata"]["input_actions_sha256"] = _sha256_hex(
            Path(args.input_actions).read_bytes()
        )

        # The read-only baseline report must be byte-identical to the fixed digest
        # BEFORE the model is loaded; it is only ever read.
        try:
            baseline_sha = _sha256_hex(Path(str(args.baseline_report)).read_bytes())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("baseline report read failed: %s" % _format_exc(exc)) from exc
        report["metadata"]["baseline_report_sha256"] = baseline_sha
        if baseline_sha != BASELINE_REPORT_SHA:
            raise RuntimeError(
                "baseline report SHA %r does not match the frozen %r"
                % (baseline_sha, BASELINE_REPORT_SHA)
            )

        plan = build_campaign_plan()
        expected_trials = len(plan)
        report["metadata"]["plan"] = plan
        report["metadata"]["source_git_sha"] = args.source_git_sha or pe._git_rev_parse()

        # 1. Freeze ONE fresh preregistration (both fixed entries) BEFORE the model.
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

        # 2. One resident service / one model load for the whole campaign.
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
                record, gate_fatal = production_health_gate()
                health_checks.append(
                    {
                        "trial_index": index,
                        "model_seed": entry["model_seed"],
                        "result": record,
                    }
                )
                _persist()
                if gate_fatal is not None:
                    fatal_error = "%s (entry %d/%d)" % (gate_fatal, index, total)
                    _progress(fatal_error)
                    break

                _progress("entry %d/%d model_seed=%s" % (index, total, entry["model_seed"]))
                try:
                    trial = _run_trial(svc, entry, actions, run_root, seen)
                except BaseException as exc:  # noqa: BLE001 - preserve, then stop
                    trial = {
                        "trial_id": "%s_m%s" % (PREFIX_CONDITION, entry["model_seed"]),
                        "condition": PREFIX_CONDITION,
                        "model_seed": entry["model_seed"],
                        "status": "operational_error",
                        "preparation": None,
                        "raw_trial": {
                            "errors": ["trial_exception: %s" % _format_exc(exc)],
                            "operational_errors": ["trial_exception: %s" % _format_exc(exc)],
                        },
                        "operational_errors": ["trial_exception: %s" % _format_exc(exc)],
                        "physical_failures": [],
                        "wine_action_count": 0,
                        "auxiliary_action_count": None,
                    }
                    trials.append(trial)
                    fatal_error = "trial_exception: %s" % _format_exc(exc)
                    _persist()
                    break
                trials.append(trial)
                _persist()
                _progress(
                    "entry %d/%d done status=%s aux=%s wine_actions=%s operational=%d"
                    % (
                        index,
                        total,
                        trial.get("status"),
                        trial.get("auxiliary_action_count"),
                        trial.get("wine_action_count"),
                        len(trial.get("operational_errors") or []),
                    )
                )
                if trial.get("operational_errors"):
                    fatal_error = (
                        "operational error(s); stopping campaign (no continuation): %s"
                        % ("; ".join(trial["operational_errors"]),)
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
            report["metadata"]["baseline_report_sha256_after"] = _sha256_hex(
                Path(str(args.baseline_report)).read_bytes()
            )
        except Exception:  # noqa: BLE001
            report["metadata"]["baseline_report_sha256_after"] = None

    _persist()
    if report.get("ok"):
        print("PREPARATION_DIAGNOSTICS_PASS")
    _progress("report written to %s (%d trials)" % (output_path, len(trials)))
    return report


# --- CLI ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed two-entry assisted-preparation wine diagnostic: after the exact "
            "recorded bowl prefix, an explicit preregistered physical preparation "
            "(lift then align) runs before the single wine subgoal.  The two entries "
            "are the SAME pipeline at FIXED model seeds 0 and 1; there is no variable "
            "seed/branch option.  No training, downloads, assessment, forced release, "
            "recovery or HTTP server."
        )
    )
    parser.add_argument(
        "--input-actions",
        required=True,
        type=str,
        help="absolute path of the preserved 102-action bowl prefix (events.jsonl or JSON list)",
    )
    parser.add_argument(
        "--baseline-report",
        required=True,
        type=str,
        help="absolute path of the read-only baseline report (its bytes must stay unchanged)",
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
    if not os.path.isabs(str(args.baseline_report)):
        parser.error("--baseline-report must be an absolute path")
    if not os.path.isabs(str(args.output)):
        parser.error("--output must be an absolute path")
    if not os.path.isabs(str(args.run_root)):
        parser.error("--run-root must be an absolute path")
    if not Path(str(args.input_actions)).is_file():
        parser.error("--input-actions must be an existing readable file: %s" % args.input_actions)
    if not Path(str(args.baseline_report)).is_file():
        parser.error(
            "--baseline-report must be an existing readable file: %s" % args.baseline_report
        )
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
