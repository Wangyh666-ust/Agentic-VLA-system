"""Diagnostic-only joint-space homing helpers and a torque adapter.

Pure import: no top-level I/O, no robosuite/mujoco imports, no simulator
mutation.  Every function that touches the simulator is read-only except
``screen_joint_path`` (which uses a private ``mujoco.MjData`` scratch copy)
and ``JointHomeAdapter`` (which returns arm torques for the *caller* to apply
via the standard robosuite ``env.step`` path).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class HomeError(ValueError):
    """Raised on any contract violation in the joint-home diagnostics."""


JOINT_STEP_RAD = 0.01
JOINT_TOL_RAD = 0.01
SPEED_TOL_RAD_S = 0.02
FINGER_TOL_M = 0.002
CONFIRM_SAMPLES = 5
RETURN_ACTION_CAP = 360

_ARM_DOF = 7
_FINGER_DOF = 2
_COLLISION_CAP = 20


# --------------------------------------------------------------------------- #
# small strict numeric helpers
# --------------------------------------------------------------------------- #
def _finite_vec(values: Any, length: int, name: str) -> List[float]:
    """Return a finite length-``length`` list of floats or raise HomeError."""

    if values is None:
        raise HomeError("%s is missing" % name)
    if isinstance(values, (bool, np.bool_)):
        raise HomeError("%s must not be bool" % name)
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1 or arr.shape[0] != length:
        raise HomeError("%s must be a length-%d 1-D sequence, got %r" % (name, length, getattr(values, "shape", type(values).__name__)))
    out: List[float] = []
    for index, item in enumerate(arr.tolist()):
        if type(item) is bool or isinstance(item, np.bool_):
            raise HomeError("%s[%d] must not be bool" % (name, index))
        if not isinstance(item, (int, np.integer, float, np.floating)):
            raise HomeError("%s[%d] is not numeric: %r" % (name, index, item))
        number = float(item)
        if not math.isfinite(number):
            raise HomeError("%s[%d] is not finite: %r" % (name, index, item))
        out.append(number)
    return out


def _finite_scalar(value: Any, name: str) -> float:
    if type(value) is bool or isinstance(value, np.bool_):
        raise HomeError("%s must not be bool" % name)
    if not isinstance(value, (int, np.integer, float, np.floating)):
        raise HomeError("%s is not numeric: %r" % (name, value))
    number = float(value)
    if not math.isfinite(number):
        raise HomeError("%s is not finite: %r" % (name, value))
    return number


def _list_of_floats(values: Any, name: str) -> List[float]:
    if values is None:
        raise HomeError("%s is missing" % name)
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise HomeError("%s must be a non-empty 1-D sequence" % name)
    return _finite_vec(arr, arr.shape[0], name)


def _limits_matrix(values: Any, name: str) -> List[List[float]]:
    if values is None:
        raise HomeError("%s is missing" % name)
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 2:
        raise HomeError("%s must be a 2-D matrix" % name)
    rows: List[List[float]] = []
    for index in range(arr.shape[0]):
        rows.append(_finite_vec(arr[index], arr.shape[1], "%s[%d]" % (name, index)))
    return rows


def _require_arm7(values: Any, name: str) -> np.ndarray:
    return np.asarray(_finite_vec(values, _ARM_DOF, name), dtype=float)


def _require_finger2(values: Any, name: str) -> np.ndarray:
    return np.asarray(_finite_vec(values, _FINGER_DOF, name), dtype=float)


# --------------------------------------------------------------------------- #
# environment / robot introspection
# --------------------------------------------------------------------------- #
def _inner_env(env: Any) -> Any:
    if env is None:
        raise HomeError("env is None")
    try:
        from scene_demo import service  # local import: no top-level side effects
    except Exception:  # noqa: BLE001 - fall back to installed abs module name
        import service  # type: ignore
    try:
        return service._inner_env(env)
    except Exception as exc:  # noqa: BLE001
        raise HomeError("inner env unavailable: %s" % exc) from exc


def _single_robot(env: Any) -> Any:
    inner = _inner_env(env)
    robots = getattr(inner, "robots", None)
    if not isinstance(robots, (list, tuple)) or len(robots) != 1:
        raise HomeError("expected exactly one robot, found %r" % (len(robots) if isinstance(robots, (list, tuple)) else robots,))
    return robots[0]


def _check_indexes(values: List[int], name: str) -> List[int]:
    if len(set(values)) != len(values):
        raise HomeError("%s must not contain duplicate indexes" % name)
    for item in values:
        if item < 0:
            raise HomeError("%s must not contain negative indexes: %r" % (name, item))
    return values


def _require_index7(value: Any, name: str) -> List[int]:
    if value is None:
        raise HomeError("%s is missing" % name)
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1 or arr.shape[0] != _ARM_DOF:
        raise HomeError("%s must have exactly %d entries" % (name, _ARM_DOF))
    out: List[int] = []
    for item in arr.tolist():
        if type(item) is bool:
            raise HomeError("%s must not contain bool" % name)
        if not isinstance(item, (int, np.integer)):
            raise HomeError("%s entries must be integers: %r" % (name, item))
        out.append(int(item))
    return _check_indexes(out, name)


def _require_index2(value: Any, name: str) -> List[int]:
    if value is None:
        raise HomeError("%s is missing" % name)
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1 or arr.shape[0] != _FINGER_DOF:
        raise HomeError("%s must have exactly %d entries" % (name, _FINGER_DOF))
    out: List[int] = []
    for item in arr.tolist():
        if type(item) is bool:
            raise HomeError("%s must not contain bool" % name)
        if not isinstance(item, (int, np.integer)):
            raise HomeError("%s entries must be integers: %r" % (name, item))
        out.append(int(item))
    return _check_indexes(out, name)


def _joint_limits_from_model(sim: Any, joint_names: Sequence[str]) -> Tuple[List[float], List[float]]:
    model = getattr(sim, "model", None)
    if model is None:
        raise HomeError("sim.model is missing")
    limited_attr = getattr(model, "jnt_limited", None)
    range_attr = getattr(model, "jnt_range", None)
    if limited_attr is None or range_attr is None:
        raise HomeError("sim.model does not expose joint limits")
    limited = np.asarray(limited_attr, dtype=object)
    ranges = np.asarray(range_attr, dtype=float)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise HomeError("sim.model.jnt_range has unexpected shape %r" % (ranges.shape,))
    lows: List[float] = []
    highs: List[float] = []
    for name in joint_names:
        try:
            joint_id = int(model.joint_name2id(name))
        except Exception:  # noqa: BLE001
            joint_id = -1
        if joint_id < 0 or joint_id >= ranges.shape[0] or joint_id >= limited.shape[0]:
            raise HomeError("joint %s has no model id" % name)
        if not bool(limited[joint_id]):
            raise HomeError("joint %s is not limited" % name)
        lo = float(ranges[joint_id, 0])
        hi = float(ranges[joint_id, 1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or not (lo < hi):
            raise HomeError("joint %s has invalid limits [%r, %r]" % (name, lo, hi))
        lows.append(lo)
        highs.append(hi)
    return lows, highs


def _torque_limits_from_robot(robot: Any) -> Tuple[List[float], List[float]]:
    inner = getattr(robot, "robot_model", None)
    if inner is None:
        raise HomeError("robot.robot_model is missing")
    torque_attr = getattr(robot, "torque_limits", None)
    if torque_attr is None:
        raise HomeError("robot.torque_limits is missing")
    arr = np.asarray(torque_attr, dtype=object)
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == _ARM_DOF:
        lows = _finite_vec(arr[0], _ARM_DOF, "robot.torque_limits[0]")
        highs = _finite_vec(arr[1], _ARM_DOF, "robot.torque_limits[1]")
    elif arr.ndim == 1 and arr.shape[0] == _ARM_DOF:
        vals = _finite_vec(arr, _ARM_DOF, "robot.torque_limits")
        lows = [-abs(v) for v in vals]
        highs = [abs(v) for v in vals]
    else:
        raise HomeError("robot.torque_limits must be 7- or 2x7-channel")
    for lo, hi in zip(lows, highs):
        if not (lo < hi):
            raise HomeError("robot.torque_limits must satisfy lo<hi per axis")
    return lows, highs


# --------------------------------------------------------------------------- #
# public API: state / reference / metrics
# --------------------------------------------------------------------------- #
def read_robot_state(env: Any) -> Dict[str, Any]:
    """Read-only snapshot of the 7-joint arm + 2-finger gripper contract."""

    robot = _single_robot(env)
    controller = getattr(robot, "controller", None)
    if controller is None:
        raise HomeError("robot.controller is missing")

    joint_indexes = _require_index7(getattr(controller, "joint_index", None), "controller.joint_index")
    qpos_indexes = _require_index7(getattr(controller, "qpos_index", None), "controller.qpos_index")
    qvel_indexes = _require_index7(getattr(controller, "qvel_index", None), "controller.qvel_index")

    sim = getattr(robot, "sim", None)
    if sim is None:
        raise HomeError("robot.sim is missing")
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        raise HomeError("sim.model / sim.data is missing")

    id2name = getattr(model, "joint_id2name", None)
    if not callable(id2name):
        raise HomeError("sim.model.joint_id2name is not callable")

    joint_names: List[str] = []
    for index in joint_indexes:
        try:
            name = id2name(int(index))
        except Exception as exc:  # noqa: BLE001
            raise HomeError("joint_id2name failed for %r: %s" % (index, exc)) from exc
        if not isinstance(name, str) or not name:
            raise HomeError("joint id %r has no name" % (index,))
        joint_names.append(name)

    qpos = np.asarray(getattr(data, "qpos", None), dtype=object)
    qvel = np.asarray(getattr(data, "qvel", None), dtype=object)
    if qpos.ndim != 1 or qvel.ndim != 1:
        raise HomeError("sim.data qpos/qvel must be 1-D")
    if max(qpos_indexes) >= qpos.shape[0] or max(qvel_indexes) >= qvel.shape[0]:
        raise HomeError("arm qpos/qvel indexes out of range")

    arm_qpos = _finite_vec([qpos[i] for i in qpos_indexes], _ARM_DOF, "arm_qpos")
    arm_qvel = _finite_vec([qvel[i] for i in qvel_indexes], _ARM_DOF, "arm_qvel")

    finger_qpos_idx = _require_index2(
        getattr(robot, "_ref_gripper_joint_pos_indexes", None),
        "robot._ref_gripper_joint_pos_indexes",
    )
    finger_qvel_idx = _require_index2(
        getattr(robot, "_ref_gripper_joint_vel_indexes", None),
        "robot._ref_gripper_joint_vel_indexes",
    )
    if max(finger_qpos_idx) >= qpos.shape[0] or max(finger_qvel_idx) >= qvel.shape[0]:
        raise HomeError("finger qpos/qvel indexes out of range")
    if set(qpos_indexes) & set(finger_qpos_idx):
        raise HomeError("arm and finger qpos indexes must not overlap")
    if set(qvel_indexes) & set(finger_qvel_idx):
        raise HomeError("arm and finger qvel indexes must not overlap")
    finger_qpos = _finite_vec([qpos[i] for i in finger_qpos_idx], _FINGER_DOF, "finger_qpos")
    finger_qvel = _finite_vec([qvel[i] for i in finger_qvel_idx], _FINGER_DOF, "finger_qvel")

    low, high = _joint_limits_from_model(sim, joint_names)
    torque_low, torque_high = _torque_limits_from_robot(robot)

    return {
        "robot_name": getattr(robot, "name", None),
        "joint_names": list(joint_names),
        "joint_indexes": list(joint_indexes),
        "qpos_indexes": list(qpos_indexes),
        "qvel_indexes": list(qvel_indexes),
        "joint_limits_low": list(low),
        "joint_limits_high": list(high),
        "arm_qpos": list(arm_qpos),
        "arm_qvel": list(arm_qvel),
        "finger_qpos_indexes": list(finger_qpos_idx),
        "finger_qvel_indexes": list(finger_qvel_idx),
        "finger_qpos": list(finger_qpos),
        "finger_qvel": list(finger_qvel),
        "torque_limits_low": list(torque_low),
        "torque_limits_high": list(torque_high),
    }


def _read_eef_pose(env: Any) -> Dict[str, Any]:
    try:
        try:
            from scene_demo import preparation_diagnostics as pd
        except Exception:  # noqa: BLE001
            import preparation_diagnostics as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise HomeError("preparation_diagnostics unavailable: %s" % exc) from exc
    try:
        pose = pd.read_eef_pose(env)
    except Exception as exc:  # noqa: BLE001
        raise HomeError("read_eef_pose failed: %s" % exc) from exc
    if not isinstance(pose, Mapping):
        raise HomeError("read_eef_pose did not return a mapping")
    position = _finite_vec(pose.get("position"), 3, "eef.position")
    matrix = _limits_matrix(pose.get("orientation_matrix"), "eef.orientation_matrix")
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise HomeError("eef.orientation_matrix must be 3x3")
    return {
        "position": list(position),
        "orientation_matrix": matrix,
    }


def capture_home_reference(env: Any, *, origin_sha256: str) -> Dict[str, Any]:
    """Capture an immutable reference snapshot for later joint-home checks."""

    if not isinstance(origin_sha256, str) or not origin_sha256:
        raise HomeError("origin_sha256 must be a non-empty string")
    state = read_robot_state(env)
    eef_pose = _read_eef_pose(env)
    reference: Dict[str, Any] = {
        "schema_version": 1,
        "origin_sha256": origin_sha256,
        "joint_names": list(state["joint_names"]),
        "joint_indexes": list(state["joint_indexes"]),
        "qpos_indexes": list(state["qpos_indexes"]),
        "qvel_indexes": list(state["qvel_indexes"]),
        "joint_limits_low": list(state["joint_limits_low"]),
        "joint_limits_high": list(state["joint_limits_high"]),
        "homeq": list(state["arm_qpos"]),
        "home_joint_speed": list(state["arm_qvel"]),
        "finger_qpos_indexes": list(state["finger_qpos_indexes"]),
        "finger_qvel_indexes": list(state["finger_qvel_indexes"]),
        "finger_home": list(state["finger_qpos"]),
        "finger_speed": list(state["finger_qvel"]),
        "eef_pose": eef_pose,
    }
    return reference


def _reference_identity(reference: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    for key in ("joint_names", "joint_indexes", "qpos_indexes", "qvel_indexes"):
        if list(reference.get(key, [])) != list(state.get(key, [])):
            raise HomeError("reference/state %s mismatch" % key)
    ref_low = _finite_vec(reference.get("joint_limits_low"), _ARM_DOF, "reference.joint_limits_low")
    ref_high = _finite_vec(reference.get("joint_limits_high"), _ARM_DOF, "reference.joint_limits_high")
    st_low = _finite_vec(state.get("joint_limits_low"), _ARM_DOF, "state.joint_limits_low")
    st_high = _finite_vec(state.get("joint_limits_high"), _ARM_DOF, "state.joint_limits_high")
    if ref_low != st_low or ref_high != st_high:
        raise HomeError("reference/state joint_limits mismatch")
    if list(reference.get("finger_qpos_indexes", [])) != list(state.get("finger_qpos_indexes", [])):
        raise HomeError("reference/state finger_qpos_indexes mismatch")
    if list(reference.get("finger_qvel_indexes", [])) != list(state.get("finger_qvel_indexes", [])):
        raise HomeError("reference/state finger_qvel_indexes mismatch")
    homeq = _require_arm7(reference.get("homeq"), "reference.homeq")
    for index, value in enumerate(homeq):
        if value < ref_low[index] or value > ref_high[index]:
            raise HomeError("reference.homeq[%d] outside joint limits" % index)
    _require_arm7(reference.get("home_joint_speed"), "reference.home_joint_speed")
    _require_finger2(reference.get("finger_home"), "reference.finger_home")
    _require_finger2(reference.get("finger_speed"), "reference.finger_speed")


def home_metrics(reference: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-axis joint/finger errors and the deterministic ``ready`` verdict."""

    if not isinstance(reference, Mapping) or not isinstance(state, Mapping):
        raise HomeError("reference and state must be mappings")
    _reference_identity(reference, state)

    homeq = _require_arm7(reference["homeq"], "reference.homeq")
    home_speed = _require_arm7(reference["home_joint_speed"], "reference.home_joint_speed")
    finger_home = _require_finger2(reference["finger_home"], "reference.finger_home")
    finger_home_speed = _require_finger2(reference["finger_speed"], "reference.finger_speed")
    arm_qpos = _require_arm7(state["arm_qpos"], "state.arm_qpos")
    arm_qvel = _require_arm7(state["arm_qvel"], "state.arm_qvel")
    finger_qpos = _require_finger2(state["finger_qpos"], "state.finger_qpos")
    finger_qvel = _require_finger2(state["finger_qvel"], "state.finger_qvel")

    arm_error = [float(abs(a - b)) for a, b in zip(homeq, arm_qpos)]
    arm_speed = [float(abs(v)) for v in arm_qvel]
    finger_error = [float(abs(a - b)) for a, b in zip(finger_home, finger_qpos)]
    finger_speed = [float(abs(v)) for v in finger_qvel]

    max_joint_error = max(arm_error) if arm_error else 0.0
    max_joint_speed = max(arm_speed) if arm_speed else 0.0
    max_finger_error = max(finger_error) if finger_error else 0.0
    max_finger_speed = max(finger_speed) if finger_speed else 0.0

    ready = (
        max_joint_error <= JOINT_TOL_RAD
        and max_joint_speed <= SPEED_TOL_RAD_S
        and max_finger_error <= FINGER_TOL_M
        and max_finger_speed <= SPEED_TOL_RAD_S
    )

    return {
        "joint_names": list(state["joint_names"]),
        "joint_indexes": list(state["joint_indexes"]),
        "arm_error_rad": arm_error,
        "arm_speed_rad_s": arm_speed,
        "max_joint_error_rad": float(max_joint_error),
        "max_joint_speed_rad_s": float(max_joint_speed),
        "finger_error_m": finger_error,
        "finger_speed_m_s": finger_speed,
        "max_finger_error_m": float(max_finger_error),
        "max_finger_speed_m_s": float(max_finger_speed),
        "ready": bool(ready),
    }


# --------------------------------------------------------------------------- #
# waypoint generation
# --------------------------------------------------------------------------- #
def joint_waypoints(
    start: Any,
    target: Any,
    *,
    step_rad: float = JOINT_STEP_RAD,
) -> List[List[float]]:
    """Shared-alpha linear interpolation between two strict 7-joint poses."""

    start_vec = _require_arm7(start, "start")
    target_vec = _require_arm7(target, "target")
    step = _finite_scalar(step_rad, "step_rad")
    if step <= 0.0:
        raise HomeError("step_rad must be positive")

    delta = target_vec - start_vec
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    n = max(1, int(math.ceil(max_abs / step)))
    waypoints: List[List[float]] = []
    for index in range(1, n):
        alpha = float(index) / float(n)
        point = start_vec + alpha * delta
        waypoints.append([float(v) for v in point])
    waypoints.append([float(v) for v in target_vec])
    return waypoints


# --------------------------------------------------------------------------- #
# scratch-path collision screen
# --------------------------------------------------------------------------- #
def _robot_collision_geoms(robot: Any) -> List[str]:
    robot_model = getattr(robot, "robot_model", None)
    gripper = getattr(robot, "gripper", None)
    if robot_model is None or gripper is None:
        raise HomeError("robot.robot_model / robot.gripper missing")
    arm_geoms = getattr(robot_model, "contact_geoms", None)
    grip_geoms = getattr(gripper, "contact_geoms", None)
    if not isinstance(arm_geoms, (list, tuple)) or not isinstance(grip_geoms, (list, tuple)):
        raise HomeError("contact_geoms must be sequences")
    names = list(arm_geoms) + list(grip_geoms)
    if not names:
        raise HomeError("robot exposes no collision geoms")
    for name in names:
        if not isinstance(name, str) or not name:
            raise HomeError("unknown collision geom name: %r" % (name,))
    return names


def _snapshot_live(sim: Any) -> Tuple[np.ndarray, np.ndarray, float]:
    data = sim.data
    qpos = np.array(data.qpos, copy=True)
    qvel = np.array(data.qvel, copy=True)
    time = float(data.time)
    return qpos, qvel, time


def _live_matches(sim: Any, qpos: np.ndarray, qvel: np.ndarray, time: float) -> bool:
    import struct

    data = sim.data
    live_qpos = np.asarray(data.qpos)
    snap_qpos = np.asarray(qpos)
    if live_qpos.shape != snap_qpos.shape or live_qpos.dtype != snap_qpos.dtype:
        return False
    if live_qpos.tobytes() != snap_qpos.tobytes():
        return False
    live_qvel = np.asarray(data.qvel)
    snap_qvel = np.asarray(qvel)
    if live_qvel.shape != snap_qvel.shape or live_qvel.dtype != snap_qvel.dtype:
        return False
    if live_qvel.tobytes() != snap_qvel.tobytes():
        return False
    return struct.pack("d", float(data.time)) == struct.pack("d", float(time))


def screen_joint_path(
    env: Any,
    reference: Mapping[str, Any],
    waypoints: Sequence[Any],
) -> Dict[str, Any]:
    """Sample the nominal joint path on a private ``MjData`` scratch copy.

    Live ``qpos`` / ``qvel`` / ``time`` are preserved byte-for-byte; any
    scratch contact between robot geoms (including self and target contacts)
    with ``dist <= 0`` is reported.  The result only characterises the sampled
    nominal path -- it is not a full-physics avoidance guarantee.
    """

    if not isinstance(reference, Mapping):
        raise HomeError("reference must be a mapping")
    state = read_robot_state(env)
    _reference_identity(reference, state)

    robot = _single_robot(env)
    sim = getattr(robot, "sim", None)
    if sim is None or getattr(sim, "model", None) is None or getattr(sim, "data", None) is None:
        raise HomeError("sim.model / sim.data is missing")

    points = list(waypoints) if waypoints is not None else []
    if not points:
        raise HomeError("waypoints must be a non-empty sequence")
    current_arm_qpos = _require_arm7(state["arm_qpos"], "state.arm_qpos")
    parsed: List[np.ndarray] = [current_arm_qpos]
    parsed.extend(
        _require_arm7(point, "waypoints[%d]" % index) for index, point in enumerate(points)
    )

    ref_low = _finite_vec(reference.get("joint_limits_low"), _ARM_DOF, "reference.joint_limits_low")
    ref_high = _finite_vec(reference.get("joint_limits_high"), _ARM_DOF, "reference.joint_limits_high")
    for point_index, arm_target in enumerate(parsed):
        for axis, value in enumerate(arm_target):
            if value < ref_low[axis] or value > ref_high[axis]:
                raise HomeError(
                    "point %d axis %d=%r outside joint limits [%r, %r]"
                    % (point_index, axis, value, ref_low[axis], ref_high[axis])
                )

    robot_geoms = _robot_collision_geoms(robot)
    model = sim.model
    geom_name2id = getattr(model, "geom_name2id", None)
    if not callable(geom_name2id):
        raise HomeError("sim.model.geom_name2id is not callable")
    robot_geom_ids = set()
    for name in robot_geoms:
        try:
            geom_id = int(geom_name2id(name))
        except Exception as exc:  # noqa: BLE001
            raise HomeError("unknown robot collision geom %r: %s" % (name, exc)) from exc
        if geom_id < 0:
            raise HomeError("unknown robot collision geom %r" % (name,))
        robot_geom_ids.add(geom_id)

    try:
        import mujoco  # lazy import: no top-level mujoco dependency
    except Exception as exc:  # noqa: BLE001
        raise HomeError("mujoco is unavailable: %s" % exc) from exc

    raw_model = getattr(model, "_model", None)
    if raw_model is None:
        raise HomeError("sim.model._model is missing")

    qpos_indexes = list(state["qpos_indexes"])
    finger_qpos_indexes = list(state["finger_qpos_indexes"])
    finger_home = _require_finger2(reference.get("finger_home"), "reference.finger_home")

    live_qpos, live_qvel, live_time = _snapshot_live(sim)
    scratch = mujoco.MjData(raw_model)
    if scratch.qpos.shape[0] != live_qpos.shape[0]:
        raise HomeError("scratch MjData qpos size does not match live sim")
    scratch.qpos[:] = live_qpos
    scratch.qvel[:] = live_qvel
    scratch.time = live_time

    collisions: List[Dict[str, Any]] = []
    executed_samples = 0
    for point_index, arm_target in enumerate(parsed):
        scratch.qpos[:] = live_qpos
        scratch.qvel[:] = live_qvel
        scratch.time = live_time
        for axis, qpos_index in enumerate(qpos_indexes):
            scratch.qpos[qpos_index] = float(arm_target[axis])
        for axis, qpos_index in enumerate(finger_qpos_indexes):
            scratch.qpos[qpos_index] = float(finger_home[axis])
        try:
            mujoco.mj_forward(raw_model, scratch)
        except Exception as exc:  # noqa: BLE001
            raise HomeError("mj_forward failed at point %d: %s" % (point_index, exc)) from exc
        executed_samples += 1
        contacts = getattr(scratch, "contact", None)
        if contacts is None:
            raise HomeError("scratch MjData has no contact array")
        ncon = getattr(scratch, "ncon", None)
        if type(ncon) is not int and not isinstance(ncon, np.integer):
            raise HomeError("scratch MjData.ncon is missing or not an integer")
        ncon = int(ncon)
        if ncon < 0 or ncon > len(contacts):
            raise HomeError("scratch MjData.ncon is invalid: %r" % (ncon,))
        for contact_index in range(ncon):
            contact = contacts[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if not (0 <= geom1 < model.ngeom) or not (0 <= geom2 < model.ngeom):
                raise HomeError("contact %d has invalid geom id(s): %r, %r" % (contact_index, geom1, geom2))
            if geom1 not in robot_geom_ids and geom2 not in robot_geom_ids:
                continue
            dist = float(contact.dist)
            if not math.isfinite(dist):
                raise HomeError("contact %d has non-finite dist: %r" % (contact_index, dist))
            if dist > 0.0:
                continue
            if len(collisions) >= _COLLISION_CAP:
                break
            name1 = getattr(model, "geom_id2name", None)
            name2 = name1
            n1 = name1(geom1) if callable(name1) else str(geom1)
            n2 = name2(geom2) if callable(name2) else str(geom2)
            collisions.append(
                {
                    "point_index": int(point_index),
                    "point": [float(v) for v in arm_target],
                    "geom1": int(geom1),
                    "geom2": int(geom2),
                    "name1": str(n1),
                    "name2": str(n2),
                    "dist": dist,
                }
            )
        if len(collisions) >= _COLLISION_CAP:
            break

    live_unchanged = _live_matches(sim, live_qpos, live_qvel, live_time)
    if not live_unchanged:
        raise HomeError("scratch path screen mutated live simulator state")

    return {
        "sample_count": int(executed_samples),
        "planned_samples": int(len(parsed)),
        "collisions": collisions,
        "collision_count": int(len(collisions)),
        "ok": bool(not collisions),
        "live_unchanged": True,
    }


# --------------------------------------------------------------------------- #
# torque adapter
# --------------------------------------------------------------------------- #
def _build_joint_controller(original: Any, env: Any, reference: Mapping[str, Any], state: Mapping[str, Any]):
    try:
        from robosuite.controllers import JointPositionController
    except Exception as exc:  # noqa: BLE001
        raise HomeError("robosuite JointPositionController unavailable: %s" % exc) from exc

    robot = _single_robot(env)
    controller = getattr(robot, "controller", None)
    if controller is None:
        raise HomeError("robot.controller is missing")
    if controller is not original:
        raise HomeError("original controller is not robot.controller")
    sim = getattr(robot, "sim", None)
    if sim is None:
        raise HomeError("robot.sim is missing")
    eef_name = getattr(original, "eef_name", None)
    if not isinstance(eef_name, str) or not eef_name:
        raise HomeError("original controller exposes no eef_name")

    inner = _inner_env(env)
    control_freq = getattr(inner, "control_freq", None)
    if control_freq is None:
        raise HomeError("inner env control_freq is missing")

    joint_indexes = list(state["joint_indexes"])
    qpos_indexes = list(state["qpos_indexes"])
    qvel_indexes = list(state["qvel_indexes"])
    low = _finite_vec(state["torque_limits_low"], _ARM_DOF, "state.torque_limits_low")
    high = _finite_vec(state["torque_limits_high"], _ARM_DOF, "state.torque_limits_high")
    actuator_range = np.array([low, high], dtype=float)

    ref_low = _finite_vec(reference.get("joint_limits_low"), _ARM_DOF, "reference.joint_limits_low")
    ref_high = _finite_vec(reference.get("joint_limits_high"), _ARM_DOF, "reference.joint_limits_high")
    qpos_limits = np.array([ref_low, ref_high], dtype=float)

    return JointPositionController(
        sim=sim,
        eef_name=eef_name,
        joint_indexes={"joints": joint_indexes, "qpos": qpos_indexes, "qvel": qvel_indexes},
        actuator_range=actuator_range,
        kp=50,
        damping_ratio=1,
        impedance_mode="fixed",
        policy_freq=control_freq,
        qpos_limits=qpos_limits,
        interpolator=None,
    )


class JointHomeAdapter:
    """6-channel arm-torque adapter that preserves the 7-D env action layout.

    ``control_dim`` is fixed to 6 so the host OSC wrapper's gripper slot
    (last action channel) is still consumed by robosuite's ``env.step``
    convention.  Torques are produced from a freshly built
    ``JointPositionController``; no live ``sim.data.qpos`` is ever written.
    """

    def __init__(self, original: Any, env: Any, reference: Mapping[str, Any]):
        if original is None:
            raise HomeError("original controller is None")
        if not isinstance(reference, Mapping):
            raise HomeError("reference must be a mapping")
        state = read_robot_state(env)
        _reference_identity(reference, state)

        robot = _single_robot(env)
        live_original = getattr(robot, "controller", None)
        if live_original is None:
            raise HomeError("robot.controller is missing")
        if live_original is not original:
            raise HomeError("original is not robot.controller")

        try:
            try:
                from scene_demo import preparation_diagnostics as pd
            except Exception:  # noqa: BLE001
                import preparation_diagnostics as pd  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise HomeError("preparation_diagnostics unavailable: %s" % exc) from exc
        mismatch = pd.controller_mismatch(pd.controller_facts(env))
        if mismatch is not None:
            raise HomeError("original OSC controller mismatch: %s" % mismatch)

        sim = getattr(robot, "sim", None)
        if sim is None or getattr(sim, "data", None) is None:
            raise HomeError("robot.sim / sim.data is missing")
        snapshot = _snapshot_live(sim)

        self.original = original
        self.env = env
        self.reference = reference
        self.control_dim = 6
        self.robot = robot
        self._state = state
        self._target: Optional[List[float]] = [float(v) for v in _require_arm7(state["arm_qpos"], "state.arm_qpos")]
        self._internal = _build_joint_controller(original, env, reference, state)

        if not _live_matches(sim, snapshot[0], snapshot[1], snapshot[2]):
            raise HomeError("adapter construction mutated live simulator state")

    # -- transparent passthrough of original OSC metadata -------------------- #
    def __getattr__(self, name: str) -> Any:
        original = self.__dict__.get("original")
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)

    # -- controller access --------------------------------------------------- #
    def _ensure_internal(self):
        return self._internal

    # -- public controller surface ------------------------------------------- #
    def select_target(self, q7: Any) -> List[float]:
        target = [float(v) for v in _require_arm7(q7, "q7")]
        low = _finite_vec(self._state["joint_limits_low"], _ARM_DOF, "joint_limits_low")
        high = _finite_vec(self._state["joint_limits_high"], _ARM_DOF, "joint_limits_high")
        for axis, value in enumerate(target):
            if value < low[axis] or value > high[axis]:
                raise HomeError("target[%d]=%r outside joint limits [%r, %r]" % (axis, value, low[axis], high[axis]))
        self._target = list(target)
        return list(self._target)

    def set_goal(self, action6: Any) -> None:
        _finite_vec(action6, self.control_dim, "action6")
        target = self._target
        if target is None:
            raise HomeError("select_target must be called before set_goal")
        controller = self._ensure_internal()
        controller.set_goal(np.zeros(_ARM_DOF), set_qpos=np.asarray(target, dtype=float))

    def run_controller(self) -> List[float]:
        controller = self._ensure_internal()
        torques = controller.run_controller()
        vec = _finite_vec(torques, _ARM_DOF, "torques")
        updater = getattr(self.original, "update", None)
        if callable(updater):
            updater(force=True)
        return vec

    def update(self, force: bool = False) -> None:
        for controller in (self.original, self._internal):
            updater = getattr(controller, "update", None)
            if callable(updater):
                updater(force=force)


__all__ = [
    "HomeError",
    "JOINT_STEP_RAD",
    "JOINT_TOL_RAD",
    "SPEED_TOL_RAD_S",
    "FINGER_TOL_M",
    "CONFIRM_SAMPLES",
    "RETURN_ACTION_CAP",
    "read_robot_state",
    "capture_home_reference",
    "home_metrics",
    "joint_waypoints",
    "screen_joint_path",
    "JointHomeAdapter",
]
