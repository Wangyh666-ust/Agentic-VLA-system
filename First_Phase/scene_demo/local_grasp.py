#!/usr/bin/env python3
"""Isolated, wine-only *local* grasp assistance (experimental, not production).

This module is a self-contained, additive, **read-only** helper for exactly one
experimental situation: a single ``wine_bottle_1`` grasp attempt inside the
persistent ``goal_table`` scene, assisted by an explicit local state machine.
It is simulator-truth evidence (``assisted=True``) and is deliberately *not* a
learned VLA skill, *not* a production feature and *not* wired into the service.

Hard boundaries
===============

* it **never** calls ``env.step`` / ``env.reset`` / ``set_state`` / ``forward``
  and never mutates any simulator state; every simulator access is a read-only
  probe (``service._inner_env`` / ``preparation_diagnostics.read_eef_pose`` /
  ``placement_experiments.capture_snapshot``);
* the only motion primitive is the already-accepted
  :func:`preparation_diagnostics.servo_action` (unchanged gains / OSC relative
  scales) plus :func:`preparation_diagnostics.orientation_error_rad`;
* the caller owns the actual stepping: it asks :meth:`LocalGraspController
  .next_action` for the next 7-D action, performs exactly one step, then calls
  :meth:`observe_after` with a fresh reading.  Action/sample counters and
  success streaks therefore only ever advance in ``observe_after``.

The controller is a small phase machine -- ``idle / above / descend / close /
lift / confirmed / bypass / failed`` -- with at most one assistance per job.
It triggers only when the VLA proposes a positive gripper command while the
wine is known *not* grasped, the grasp observation is complete, the hand holds
nothing and the end-effector is within ``0.14 m`` of the bottle.  A missing or
malformed reading never counts as a boolean: unknown stays ``None`` and can
never advance a streak.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import placement_experiments as pe
import preparation_diagnostics as pd
import service

# --- fixed wine-only identity -------------------------------------------------

WINE_OBJECT_ID = "wine_bottle_1"
WINE_GOAL_KEY = "on|wine_bottle_1|wine_rack_1_top_region"

# The exact read-only snapshot goals this module screens (the protected bowl
# placement plus the wine target).  Nothing is derived from a capability
# schedule; the two literal goals are written here verbatim.
SNAPSHOT_GOALS: list[list[str]] = [
    ["on", "akita_black_bowl_1", "plate_1"],
    ["on", "wine_bottle_1", "wine_rack_1_top_region"],
]

# The three objects a grasp attempt must never disturb (the wine bottle is the
# intended motion and is therefore NOT protected here).
PROTECTED_OBJECTS = ("akita_black_bowl_1", "plate_1", "cream_cheese_1")

# --- fixed geometry / thresholds ---------------------------------------------

# The wine XML neck collision centre recorded as ``0.10286`` m; the pad target
# sits this far above the wine body origin along the wine +Z axis.  This is a
# calibration constant and is NEVER replaced by a guessed world XYZ.
WINE_PAD_HEIGHT_M = 0.105

# ``above`` lifts the frozen end-effector straight up in world Z; ``lift``
# lifts the closed hand a smaller amount once the grasp is confirmed.
ABOVE_CLEARANCE_M = 0.06
LIFT_CLEARANCE_M = 0.035

# Servo alignment tolerances + consecutive real post-action sample counts.
ALIGN_TOLERANCE_M = 0.003
ALIGN_TOLERANCE_RAD = 0.05
ALIGN_STREAK = 3
GRASP_STREAK = 3
LIFT_STREAK = 5

# A confirmed grasp must have really raised the bottle by this much.
LIFT_SUCCESS_M = 0.02

# Trigger radius: the end-effector must already be this close to the bottle.
NEAR_DISTANCE_M = 0.14

# Protected-object displacement tolerance (fail-closed).
PROTECTION_TOLERANCE_M = 0.005

# During ``above`` / ``descend`` the bottle must stay put.
WINE_TRANSLATION_DRIFT_M = 0.015
WINE_ROTATION_DRIFT_RAD = math.radians(20.0)

# --- fixed phase machine ------------------------------------------------------

IDLE = "idle"
ABOVE = "above"
DESCEND = "descend"
CLOSE = "close"
LIFT = "lift"
CONFIRMED = "confirmed"
BYPASS = "bypass"
FAILED = "failed"

PHASES = (IDLE, ABOVE, DESCEND, CLOSE, LIFT, CONFIRMED, BYPASS, FAILED)
STAGE_ORDER = (ABOVE, DESCEND, CLOSE, LIFT)
STAGE_BUDGETS = {ABOVE: 60, DESCEND: 80, CLOSE: 30, LIFT: 60}
TOTAL_MAX_ACTIONS = 200

TERMINAL_PHASES = (CONFIRMED, BYPASS, FAILED)

ACTION_DIM = 7

GRASP_OPEN = -1.0
GRASP_CLOSED = 1.0


# --- small pure helpers -------------------------------------------------------


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


def _distance(first: Any, second: Any) -> float | None:
    """The finite Euclidean distance between two 3-vectors, or ``None``."""

    left = _finite_vec3(first)
    right = _finite_vec3(second)
    if left is None or right is None:
        return None
    distance = math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))
    return distance if math.isfinite(distance) else None


def _action_gripper(proposed_action: Any) -> float | None:
    """The finite gripper channel (index 6) of a proposed VLA action, or ``None``."""

    if proposed_action is None:
        return None
    try:
        array = np.asarray(proposed_action, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001 - a malformed action is unknown
        return None
    if array.size < ACTION_DIM:
        return None
    value = float(array[ACTION_DIM - 1])
    if not math.isfinite(value):
        return None
    return value


def _snapshot_entry(snapshot: Any, object_id: str) -> dict | None:
    if not isinstance(snapshot, dict):
        return None
    objects = snapshot.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(object_id)
    return entry if isinstance(entry, dict) else None


def _snapshot_object_position(snapshot: Any, object_id: str) -> list[float] | None:
    entry = _snapshot_entry(snapshot, object_id)
    if entry is None:
        return None
    return _finite_vec3(entry.get("position"))


def _snapshot_object_grasped(snapshot: Any, object_id: str) -> bool | None:
    entry = _snapshot_entry(snapshot, object_id)
    if entry is None:
        return None
    value = entry.get("grasped")
    return value if isinstance(value, bool) else None


def _protected_positions(snapshot: Any) -> dict[str, list[float] | None]:
    return {
        object_id: _snapshot_object_position(snapshot, object_id)
        for object_id in PROTECTED_OBJECTS
    }


# --- quaternion handling (never assume xyzw) ---------------------------------


def _quat_wxyz_to_matrix_numpy(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Pure-numpy fallback for a MuJoCo ``wxyz`` quaternion -> rotation matrix."""

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_matrix(quat_wxyz: Any) -> np.ndarray | None:
    """Convert a MuJoCo ``wxyz`` free-joint quaternion into a 3x3 matrix.

    The input order is MuJoCo's ``[w, x, y, z]``; scipy's
    ``Rotation.from_quat`` expects ``[x, y, z, w]`` so the components are
    explicitly reordered.  The xyzw layout is never assumed.
    """

    full = _finite_scalar_vector(quat_wxyz, 4)
    if full is None:
        return None
    w, x, y, z = full
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    try:
        from scipy.spatial.transform import Rotation  # noqa: PLC0415 - optional

        matrix = Rotation.from_quat([x, y, z, w]).as_matrix()
    except Exception:  # noqa: BLE001 - scipy is optional; documented numpy fallback
        matrix = _quat_wxyz_to_matrix_numpy(w, x, y, z)
    result = _finite_matrix3(matrix)
    return result


def _finite_scalar_vector(value: Any, length: int) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        sequence = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if len(sequence) < length:
        return None
    vector = sequence[:length]
    if any(not math.isfinite(component) for component in vector):
        return None
    return vector


def _wine_freejoint_qpos(inner: Any) -> np.ndarray | None:
    """The raw ``qpos`` of the wine free joint (7 floats), or ``None``."""

    objects = getattr(inner, "objects_dict", None)
    if not isinstance(objects, dict):
        return None
    obj = objects.get(WINE_OBJECT_ID)
    if obj is None:
        return None
    joints = getattr(obj, "joints", None) or []
    if not joints or not isinstance(joints[0], str):
        return None
    sim = getattr(inner, "sim", None)
    data = getattr(sim, "data", None)
    if data is None or not callable(getattr(data, "get_joint_qpos", None)):
        return None
    try:
        qpos = np.asarray(data.get_joint_qpos(joints[0]), dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001 - unreadable qpos stays unknown
        return None
    if qpos.size < 7 or not bool(np.all(np.isfinite(qpos))):
        return None
    return qpos


def _geom_xpos(data: Any, model: Any, geom_name: str) -> list[float] | None:
    if not isinstance(geom_name, str):
        return None
    getter = getattr(data, "get_geom_xpos", None)
    if callable(getter):
        try:
            return _finite_vec3(getter(geom_name))
        except Exception:  # noqa: BLE001 - fall through to the array form
            pass
    geom_xpos = getattr(data, "geom_xpos", None)
    name2id = getattr(model, "geom_name2id", None)
    if geom_xpos is None or not callable(name2id):
        return None
    try:
        index = int(name2id(geom_name))
        return _finite_vec3(np.asarray(geom_xpos)[index])
    except Exception:  # noqa: BLE001
        return None


def _pad_center(data: Any, model: Any, important_geoms: Any, key: str) -> list[float] | None:
    """The mean ``geom_xpos`` of one gripper pad (``key``), or ``None``.

    ``important_geoms`` values may be a single geom name or a sequence of geom
    names; every resolvable geom is averaged.
    """

    if not isinstance(important_geoms, dict) or key not in important_geoms:
        return None
    names = important_geoms[key]
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)):
        return None
    positions: list[list[float]] = []
    for name in names:
        position = _geom_xpos(data, model, name)
        if position is not None:
            positions.append(position)
    if not positions:
        return None
    array = np.asarray(positions, dtype=np.float64).mean(axis=0)
    return [float(value) for value in array]


def _site_xpos(data: Any, model: Any, site_name: str) -> list[float] | None:
    getter = getattr(data, "get_site_xpos", None)
    if callable(getter):
        try:
            return _finite_vec3(getter(site_name))
        except Exception:  # noqa: BLE001 - fall through
            pass
    site_xpos = getattr(data, "site_xpos", None)
    name2id = getattr(model, "site_name2id", None)
    if site_xpos is None or not callable(name2id):
        return None
    try:
        index = int(name2id(site_name))
        return _finite_vec3(np.asarray(site_xpos)[index])
    except Exception:  # noqa: BLE001
        return None


def _site_xmat(data: Any, model: Any, site_name: str) -> np.ndarray | None:
    getter = getattr(data, "get_site_xmat", None)
    if callable(getter):
        try:
            return _finite_matrix3(getter(site_name))
        except Exception:  # noqa: BLE001 - fall through
            pass
    site_xmat = getattr(data, "site_xmat", None)
    name2id = getattr(model, "site_name2id", None)
    if site_xmat is None or not callable(name2id):
        return None
    try:
        index = int(name2id(site_name))
        return _finite_matrix3(np.asarray(site_xmat)[index])
    except Exception:  # noqa: BLE001
        return None


# --- read-only geometry probe -------------------------------------------------


def read_geometry(env: Any) -> dict[str, Any]:
    """Read the wine-only grasp geometry from the live simulator (read-only).

    Returns a mapping with the fields the controller needs; every field is
    either a measured value or an explicit ``None`` (unknown).  Missing or
    invalid readings are never coerced to a boolean ``False``.  No probe here
    can move, reset or forward the simulator -- a probing failure is simply an
    unknown reading.
    """

    reading: dict[str, Any] = {
        "pose": None,
        "snapshot": None,
        "wine_rotation": None,
        "pad_midpoint": None,
        "pad_offset_local": None,
        "pad_separation_m": None,
        "grip_site_position": None,
        "grip_site_orientation_matrix": None,
        "controller_grip_site_error_m": None,
    }

    # --- native snapshot (protected bowl/wine goals) -------------------------
    try:
        snapshot = pe.capture_snapshot(env, [list(goal) for goal in SNAPSHOT_GOALS])
    except Exception:  # noqa: BLE001 - a failed snapshot is an unknown reading
        snapshot = None
    if isinstance(snapshot, dict):
        reading["snapshot"] = snapshot

    # --- end-effector pose (worker-owned, read-only) -------------------------
    try:
        pose = pd.read_eef_pose(env)
    except Exception:  # noqa: BLE001 - an unreadable pose is unknown
        pose = None
    if isinstance(pose, dict):
        position = _finite_vec3(pose.get("position"))
        matrix = _finite_matrix3(pose.get("orientation_matrix"))
        if position is not None and matrix is not None:
            reading["pose"] = {"position": position, "orientation_matrix": matrix}

    # --- inner env for the gripper geometry + wine free joint ----------------
    try:
        inner = service._inner_env(env)
    except Exception:  # noqa: BLE001 - no inner env -> everything below unknown
        inner = None

    gripper = None
    data = None
    model = None
    if inner is not None:
        robots = getattr(inner, "robots", None) or []
        robot = robots[0] if robots else None
        gripper = getattr(robot, "gripper", None) if robot is not None else None
        sim = getattr(inner, "sim", None)
        data = getattr(sim, "data", None) if sim is not None else None
        model = getattr(sim, "model", None) if sim is not None else None

    # --- finger-pad midpoint / separation ------------------------------------
    if gripper is not None and data is not None:
        important_geoms = getattr(gripper, "important_geoms", None)
        left = _pad_center(data, model, important_geoms, "left_fingerpad")
        right = _pad_center(data, model, important_geoms, "right_fingerpad")
        if left is not None and right is not None:
            reading["pad_midpoint"] = [
                (left[index] + right[index]) / 2.0 for index in range(3)
            ]
            reading["pad_separation_m"] = _distance(left, right)

    # --- grip site position / matrix -----------------------------------------
    if gripper is not None and data is not None:
        important_sites = getattr(gripper, "important_sites", None)
        site_name = (
            important_sites.get("grip_site") if isinstance(important_sites, dict) else None
        )
        if isinstance(site_name, str):
            reading["grip_site_position"] = _site_xpos(data, model, site_name)
            reading["grip_site_orientation_matrix"] = _site_xmat(data, model, site_name)

    # --- wine rotation (MuJoCo wxyz free-joint quaternion) -------------------
    qpos = _wine_freejoint_qpos(inner) if inner is not None else None
    if qpos is not None:
        reading["wine_rotation"] = _quat_wxyz_to_matrix(qpos[3:7])

    # --- derived comparisons (controller frame) ------------------------------
    pose = reading["pose"]
    if pose is not None:
        if reading["grip_site_position"] is not None:
            reading["controller_grip_site_error_m"] = _distance(
                pose["position"], reading["grip_site_position"]
            )
        if reading["pad_midpoint"] is not None:
            offset = np.asarray(reading["pad_midpoint"], dtype=np.float64).reshape(3) - np.asarray(
                pose["position"], dtype=np.float64
            ).reshape(3)
            local = np.asarray(pose["orientation_matrix"], dtype=np.float64).reshape(3, 3).T @ offset
            reading["pad_offset_local"] = [float(value) for value in local]

    return reading


# --- the frozen contact target ------------------------------------------------


def _default_approach_offset(reading: dict, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """The frozen default above-clearance offset (world +Z)."""

    return np.array([0.0, 0.0, ABOVE_CLEARANCE_M], dtype=np.float64)


def make_target(reading: dict) -> tuple[np.ndarray, np.ndarray]:
    """The frozen end-effector target for the wine grasp.

    ``pad_target = wine_xyz + R_wine @ [0, 0, WINE_PAD_HEIGHT_M]`` and the
    desired hand orientation is the symmetric cylindrical neck frame closer to
    the current end-effector frame.  The two symmetric candidates are
    ``R1 = R_wine @ diag(1, -1, -1)`` and ``R2 = R1 @ diag(-1, -1, 1)``; the one
    with the lower :func:`preparation_diagnostics.orientation_error_rad` against
    ``reading.pose.orientation_matrix`` is selected (a tie selects ``R1``).  The
    returned end-effector target compensates the measured local pad offset with
    the selected orientation: ``eef_target = pad_target - orientation @
    pad_offset_local``.

    Raises ``ValueError`` when a required reading is unknown; it never falls
    back to a guessed fixed world XYZ.
    """

    snapshot = reading.get("snapshot") if isinstance(reading, dict) else None
    wine_position = _snapshot_object_position(snapshot, WINE_OBJECT_ID)
    wine_rotation = _finite_matrix3(reading.get("wine_rotation")) if isinstance(reading, dict) else None
    pad_offset_local = (
        _finite_vec3(reading.get("pad_offset_local")) if isinstance(reading, dict) else None
    )
    if wine_position is None:
        raise ValueError("make_target needs a finite wine position")
    if wine_rotation is None:
        raise ValueError("make_target needs a finite wine rotation")
    if pad_offset_local is None:
        raise ValueError("make_target needs a finite local pad offset")

    candidate_r1 = wine_rotation @ np.diag([1.0, -1.0, -1.0])
    candidate_r2 = candidate_r1 @ np.diag([-1.0, -1.0, 1.0])
    pose = reading.get("pose") if isinstance(reading, dict) else None
    current_orientation = (
        _finite_matrix3(pose.get("orientation_matrix")) if isinstance(pose, dict) else None
    )
    orientation = candidate_r1
    if current_orientation is not None:
        error_r1 = pd.orientation_error_rad(current_orientation, candidate_r1)
        error_r2 = pd.orientation_error_rad(current_orientation, candidate_r2)
        if error_r1 is not None and error_r2 is not None and error_r2 < error_r1:
            orientation = candidate_r2

    pad_offset_world = wine_rotation @ np.array([0.0, 0.0, WINE_PAD_HEIGHT_M], dtype=np.float64)
    pad_target = np.asarray(wine_position, dtype=np.float64).reshape(3) + pad_offset_world
    eef_target = pad_target - orientation @ np.asarray(pad_offset_local, dtype=np.float64).reshape(3)
    return eef_target, orientation


# --- trigger evaluation -------------------------------------------------------


def evaluate_trigger(reading: dict, proposed_action: Any) -> dict[str, Any]:
    """The explicit trigger evidence for one ``next_action`` read.

    Satisfied only when the proposed gripper command is positive, the wine is
    known *not* grasped, the grasp observation is complete, the hand holds
    nothing and the end-effector is within ``NEAR_DISTANCE_M`` of the bottle.
    Every unknown field stays ``None`` and cannot satisfy the trigger.
    """

    result: dict[str, Any] = {
        "command": None,
        "grasped": None,
        "grasp_observation_complete": None,
        "held_objects": None,
        "distance_m": None,
        "satisfied": False,
    }
    if not isinstance(reading, dict):
        return result

    result["command"] = _action_gripper(proposed_action)

    snapshot = reading.get("snapshot")
    result["grasped"] = _snapshot_object_grasped(snapshot, WINE_OBJECT_ID)
    complete = snapshot.get("grasp_observation_complete") if isinstance(snapshot, dict) else None
    result["grasp_observation_complete"] = complete if isinstance(complete, bool) else None
    held = snapshot.get("held_objects") if isinstance(snapshot, dict) else None
    result["held_objects"] = list(held) if isinstance(held, list) else None

    pose = reading.get("pose")
    position = _finite_vec3(pose.get("position")) if isinstance(pose, dict) else None
    wine_position = _snapshot_object_position(snapshot, WINE_OBJECT_ID)
    result["distance_m"] = _distance(position, wine_position)

    command = result["command"]
    result["satisfied"] = bool(
        command is not None
        and command > 0.0
        and result["grasped"] is False
        and result["grasp_observation_complete"] is True
        and result["held_objects"] == []
        and result["distance_m"] is not None
        and result["distance_m"] <= NEAR_DISTANCE_M
    )
    return result


# --- the phase machine --------------------------------------------------------


class LocalGraspController:
    """A focused, wine-only local grasp assistance state machine.

    The controller is *pure* with respect to the simulator: :meth:`next_action`
    returns the next 7-D float32 action (or ``None``) and :meth:`observe_after`
    is called exactly once after each real step with a fresh reading.  Counters
    and streaks advance only in :meth:`observe_after`, so repeated
    ``next_action`` reads never fabricate progress.
    """

    def __init__(self, *, target_builder: Any = None, approach_offset_builder: Any = None) -> None:
        self._target_builder = target_builder if target_builder is not None else make_target
        self._approach_offset_builder = (
            approach_offset_builder if approach_offset_builder is not None else _default_approach_offset
        )
        self.phase: str = IDLE
        self.reason: str | None = None
        self.total_actions: int = 0

        self._armed = False
        self._stage_counts: dict[str, int] = {stage: 0 for stage in STAGE_ORDER}
        self._stage_streak: dict[str, int] = {stage: 0 for stage in STAGE_ORDER}
        self._outstanding_phase: str | None = None

        self._arm_wine_position: list[float] | None = None
        self._arm_wine_rotation: np.ndarray | None = None
        self._wine_initial_z: float | None = None
        self._protected_baseline: dict[str, list[float] | None] = {}

        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        self._above_target_position: np.ndarray | None = None
        self._above_target_orientation: np.ndarray | None = None
        self._lift_target_position: np.ndarray | None = None
        self._lift_target_orientation: np.ndarray | None = None

        self._trigger: dict[str, Any] = {
            "command": None,
            "grasped": None,
            "grasp_observation_complete": None,
            "held_objects": None,
            "distance_m": None,
            "satisfied": False,
        }
        self._last_errors: dict[str, Any] = {}

    # -- public API -----------------------------------------------------------

    def next_action(
        self, reading: dict, proposed_action: Any = None
    ) -> np.ndarray | None:
        """Return the next action to step, or ``None`` to leave control to the VLA.

        Calling this repeatedly without an intervening :meth:`observe_after`
        never advances any counter or streak.
        """

        if self.phase in TERMINAL_PHASES:
            return None

        if not self._armed:
            return self._try_arm(reading, proposed_action)

        failure = self._active_failure(reading)
        if failure is not None:
            return self._fail(failure)

        if self.total_actions >= TOTAL_MAX_ACTIONS:
            return self._fail("total_budget_exceeded")
        if self._stage_counts.get(self.phase, 0) >= STAGE_BUDGETS.get(self.phase, 0):
            return self._fail("%s_timeout" % self.phase)

        return self._stage_action(reading)

    def observe_after(self, reading: dict) -> None:
        """Consume ONE real post-action reading for the action actually issued.

        The sample is attributed to the phase of the last action returned by
        :meth:`next_action`; a reading with no action outstanding is ignored.
        Only here do ``total_actions``, the per-stage counts and the success
        streaks advance.
        """

        phase = self._outstanding_phase
        if phase is None:
            return
        self._outstanding_phase = None
        if phase not in self._stage_counts:
            return

        self.total_actions += 1
        self._stage_counts[phase] += 1

        failure = self._active_failure(reading)
        if failure is not None:
            self._fail(failure)
            return

        if phase == ABOVE:
            self._observe_alignment(reading, ABOVE, DESCEND)
        elif phase == DESCEND:
            self._observe_alignment(reading, DESCEND, CLOSE)
        elif phase == CLOSE:
            self._observe_close(reading)
        elif phase == LIFT:
            self._observe_lift(reading)

    def summary(self) -> dict[str, Any]:
        """A JSON-friendly snapshot of the controller's frozen evidence."""

        target = {
            "eef_position": _list_or_none(self._target_position),
            "orientation": _matrix_list_or_none(self._target_orientation),
            "above_position": _list_or_none(self._above_target_position),
            "lift_position": _list_or_none(self._lift_target_position),
        }
        return {
            "assisted": True,
            "phase": self.phase,
            "reason": self.reason,
            "total_actions": self.total_actions,
            "stage_counts": dict(self._stage_counts),
            "trigger": dict(self._trigger),
            "target": dict(target),
            "targets": dict(target),
            "last_errors": dict(self._last_errors),
        }

    # -- arming ---------------------------------------------------------------

    def _try_arm(self, reading: dict, proposed_action: Any) -> np.ndarray | None:
        trigger = evaluate_trigger(reading, proposed_action)
        self._trigger = trigger

        command = trigger["command"]
        if command is not None and command > 0.0 and trigger["grasped"] is True:
            self.phase = BYPASS
            self.reason = "wine_already_held"
            return None

        if not trigger["satisfied"]:
            return None

        try:
            target_position, target_orientation = self._target_builder(reading)
            target_position = _finite_vec3(target_position)
            target_orientation = _finite_matrix3(target_orientation)
            if target_position is None or target_orientation is None:
                raise ValueError("invalid target")
            approach_offset = self._approach_offset_builder(
                reading, np.asarray(target_position, dtype=np.float64).reshape(3), np.asarray(target_orientation, dtype=np.float64).reshape(3, 3)
            )
            approach_offset = _finite_vec3(approach_offset)
            if approach_offset is None:
                raise ValueError("invalid approach offset")
        except Exception:  # noqa: BLE001 - an uncomputable target is not a trigger
            trigger["satisfied"] = False
            trigger["target_ready"] = False
            return None

        snapshot = reading.get("snapshot")
        pose = reading.get("pose")
        wine_position = _snapshot_object_position(snapshot, WINE_OBJECT_ID)
        pose_position = _finite_vec3(pose.get("position")) if isinstance(pose, dict) else None
        if wine_position is None or pose_position is None:
            trigger["satisfied"] = False
            return None

        self._armed = True
        self._arm_wine_position = list(wine_position)
        self._arm_wine_rotation = _finite_matrix3(reading.get("wine_rotation"))
        self._wine_initial_z = float(wine_position[2])
        self._protected_baseline = _protected_positions(snapshot)
        self._target_position = target_position
        self._target_orientation = target_orientation
        self._above_target_position = (
            np.asarray(target_position, dtype=np.float64).reshape(3)
            + np.asarray(approach_offset, dtype=np.float64).reshape(3)
        )
        self._above_target_orientation = target_orientation
        self.phase = ABOVE
        self._stage_streak[ABOVE] = 0
        trigger["satisfied"] = True
        trigger["target_ready"] = True
        return self._stage_action(reading)

    # -- failure checks (active assistance only) ------------------------------

    def _active_failure(self, reading: dict) -> str | None:
        if not isinstance(reading, dict):
            return "unknown_geometry"
        pose = reading.get("pose")
        if not isinstance(pose, dict):
            return "unknown_geometry"
        if _finite_vec3(pose.get("position")) is None:
            return "unknown_geometry"
        if _finite_matrix3(pose.get("orientation_matrix")) is None:
            return "unknown_geometry"

        snapshot = reading.get("snapshot")
        if not isinstance(snapshot, dict):
            return "unknown_geometry"
        held = snapshot.get("held_objects")
        if not isinstance(held, list):
            return "unknown_geometry"
        if any(object_id != WINE_OBJECT_ID for object_id in held):
            return "foreign_held_object"

        if _snapshot_object_position(snapshot, WINE_OBJECT_ID) is None:
            return "unknown_geometry"

        # In EVERY active phase the grasp observation must be explicitly
        # complete and the wine grasp must be an explicit bool/np.bool_; a
        # missing or non-boolean value is unknown and fails, never `False`.
        if snapshot.get("grasp_observation_complete") is not True:
            return "unknown_grasp_observation"
        entry = _snapshot_entry(snapshot, WINE_OBJECT_ID)
        grasped = entry.get("grasped") if isinstance(entry, dict) else None
        if not isinstance(grasped, (bool, np.bool_)):
            return "unknown_grasp"

        protected = self._protected_violation(snapshot)
        if protected is not None:
            return protected

        if self.phase in (ABOVE, DESCEND):
            drift = self._wine_drift(reading)
            if drift is not None:
                return drift
        return None

    def _protected_violation(self, snapshot: dict) -> str | None:
        for object_id in PROTECTED_OBJECTS:
            baseline = self._protected_baseline.get(object_id)
            current = _snapshot_object_position(snapshot, object_id)
            if baseline is None or current is None:
                return "protection_unknown:%s" % object_id
            distance = _distance(baseline, current)
            if distance is None or distance > PROTECTION_TOLERANCE_M:
                return "protected_displacement:%s" % object_id
        return None

    def _wine_drift(self, reading: dict) -> str | None:
        if self._arm_wine_position is None:
            return None
        snapshot = reading.get("snapshot")
        wine_position = _snapshot_object_position(snapshot, WINE_OBJECT_ID)
        if wine_position is None:
            return "unknown_geometry"
        translation = _distance(wine_position, self._arm_wine_position)
        if translation is None:
            return "unknown_geometry"
        if translation > WINE_TRANSLATION_DRIFT_M:
            return "wine_translation_drift"
        current_rotation = _finite_matrix3(reading.get("wine_rotation"))
        if current_rotation is None or self._arm_wine_rotation is None:
            return "unknown_geometry"
        angle = pd.orientation_error_rad(current_rotation, self._arm_wine_rotation)
        if angle is None:
            return "unknown_geometry"
        if angle > WINE_ROTATION_DRIFT_RAD:
            return "wine_rotation_drift"
        return None

    # -- per-stage actions ----------------------------------------------------

    def _stage_action(self, reading: dict) -> np.ndarray | None:
        phase = self.phase
        pose = reading.get("pose") if isinstance(reading, dict) else None
        if not isinstance(pose, dict):
            return self._fail("unknown_geometry")
        position = _finite_vec3(pose.get("position"))
        orientation = _finite_matrix3(pose.get("orientation_matrix"))
        if position is None or orientation is None:
            return self._fail("unknown_geometry")

        if phase == ABOVE:
            if self._above_target_position is None:
                return self._fail("target_unavailable")
            action = pd.servo_action(
                self._above_target_position,
                position,
                self._above_target_orientation,
                orientation,
            )
            action[-1] = GRASP_OPEN
        elif phase == DESCEND:
            if self._target_position is None:
                return self._fail("target_unavailable")
            action = pd.servo_action(
                self._target_position, position, self._target_orientation, orientation
            )
            action[-1] = GRASP_OPEN
        elif phase == CLOSE:
            if self._target_position is None:
                return self._fail("target_unavailable")
            action = pd.servo_action(
                self._target_position, position, self._target_orientation, orientation
            )
            action[-1] = GRASP_CLOSED
        elif phase == LIFT:
            if self._lift_target_position is None:
                return self._fail("target_unavailable")
            action = pd.servo_action(
                self._lift_target_position,
                position,
                self._lift_target_orientation,
                orientation,
            )
            action[-1] = GRASP_CLOSED
        else:
            return self._fail("invalid_phase")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != ACTION_DIM or not bool(np.all(np.isfinite(action))):
            return self._fail("nonfinite_action")
        self._outstanding_phase = phase
        return action

    # -- per-stage observations -----------------------------------------------

    def _observe_alignment(self, reading: dict, stage: str, next_phase: str) -> None:
        pose = reading.get("pose")
        position = _finite_vec3(pose.get("position")) if isinstance(pose, dict) else None
        orientation = _finite_matrix3(pose.get("orientation_matrix")) if isinstance(pose, dict) else None
        target_position = (
            self._above_target_position if stage == ABOVE else self._target_position
        )
        target_orientation = (
            self._above_target_orientation if stage == ABOVE else self._target_orientation
        )
        if position is None or orientation is None or target_position is None:
            self._stage_streak[stage] = 0
            return
        position_error = _distance(position, target_position)
        rotation_error = pd.orientation_error_rad(orientation, target_orientation)
        self._last_errors["%s_position_error_m" % stage] = position_error
        self._last_errors["%s_rotation_error_rad" % stage] = rotation_error
        if (
            position_error is not None
            and rotation_error is not None
            and position_error <= ALIGN_TOLERANCE_M
            and rotation_error <= ALIGN_TOLERANCE_RAD
        ):
            self._stage_streak[stage] += 1
            if self._stage_streak[stage] >= ALIGN_STREAK:
                self.phase = next_phase
                self._stage_streak[next_phase] = 0
        else:
            self._stage_streak[stage] = 0

    def _observe_close(self, reading: dict) -> None:
        snapshot = reading.get("snapshot") if isinstance(reading, dict) else None
        grasped = _snapshot_object_grasped(snapshot, WINE_OBJECT_ID)
        self._last_errors["close_grasped"] = grasped
        if grasped is True:
            self._stage_streak[CLOSE] += 1
            if self._stage_streak[CLOSE] >= GRASP_STREAK:
                self.phase = LIFT
                self._stage_streak[LIFT] = 0
                pose = reading.get("pose") if isinstance(reading, dict) else None
                position = _finite_vec3(pose.get("position")) if isinstance(pose, dict) else None
                orientation = (
                    _finite_matrix3(pose.get("orientation_matrix"))
                    if isinstance(pose, dict)
                    else None
                )
                if position is not None and orientation is not None:
                    self._lift_target_position = (
                        np.asarray(position, dtype=np.float64).reshape(3)
                        + np.array([0.0, 0.0, LIFT_CLEARANCE_M], dtype=np.float64)
                    )
                    self._lift_target_orientation = orientation
        else:
            self._stage_streak[CLOSE] = 0

    def _observe_lift(self, reading: dict) -> None:
        snapshot = reading.get("snapshot") if isinstance(reading, dict) else None
        grasped = _snapshot_object_grasped(snapshot, WINE_OBJECT_ID)
        wine_position = _snapshot_object_position(snapshot, WINE_OBJECT_ID)
        lift: float | None = None
        if wine_position is not None and self._wine_initial_z is not None:
            lift = float(wine_position[2]) - self._wine_initial_z
            if not math.isfinite(lift):
                lift = None
        self._last_errors["lift_grasped"] = grasped
        self._last_errors["lift_metres"] = lift
        if grasped is True and lift is not None and lift >= LIFT_SUCCESS_M:
            self._stage_streak[LIFT] += 1
            if self._stage_streak[LIFT] >= LIFT_STREAK:
                self.phase = CONFIRMED
                self.reason = None
        else:
            self._stage_streak[LIFT] = 0

    # -- failure --------------------------------------------------------------

    def _fail(self, reason: str) -> None:
        self.phase = FAILED
        self.reason = reason
        self._outstanding_phase = None
        return None


# --- small serialization helpers ---------------------------------------------


def _list_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(component) for component in array]


def _matrix_list_or_none(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    return [[float(component) for component in row] for row in matrix]
