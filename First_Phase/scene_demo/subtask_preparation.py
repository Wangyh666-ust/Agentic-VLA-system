"""Empty-hand approach preparation route checker and servo state machine.

This module plans and validates a collision-checked, empty-hand approach route
that stops *above* the target object.  It never descends to grasp and never
closes the gripper.  All collision reasoning is performed on sampled gripper OBBs
only and is explicitly *not* a full-arm or all-physics-substep guarantee.

The module has no filesystem, network, simulator, model or bot-service side
effects at import time.  ``preparation_diagnostics`` and ``scipy`` are imported
lazily inside runtime methods.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


class PreparationError(ValueError):
    """Raised when a preparation reading, plan or route cannot be trusted."""


_EPS = 1e-10

# Workspace limits (candidate poses and every checked sample must satisfy these).
_WORKSPACE_X = (-0.6, 0.6)
_WORKSPACE_Y = (-0.6, 0.6)
_WORKSPACE_Z = (0.90, 1.55)

# Sampled-route spacing limits.
_POSITION_STEP = 0.005
_ANGULAR_STEP = 0.05

# Objective displacement tolerance for baseline snapshot objects.
_BASELINE_DISPLACEMENT = 0.005

# Servo arrival tolerances.
_WAYPOINT_POSITION_TOL = 0.005
_WAYPOINT_ORIENTATION_TOL = 0.05
_CONFIRMATION_SAMPLES = 5

# Budgets.
_DEFAULT_MAX_ACTIONS = 200
_WAYPOINT_ACTION_LIMIT = 60

_ORIENTATION_FLOOR = -0.85

_SAFE_GAP_MIN = 0.07


# --------------------------------------------------------------------------- #
# Low-level validation helpers
# --------------------------------------------------------------------------- #
def _finite_scalar(value: Any) -> Optional[float]:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _finite_vec3(value: Any) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape[0] != 3:
        return None
    if not bool(np.all(np.isfinite(array))):
        return None
    return array.astype(np.float64)


def _finite_matrix3(value: Any) -> Optional[np.ndarray]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (3, 3):
        return None
    if not bool(np.all(np.isfinite(matrix))):
        return None
    return matrix.astype(np.float64)


def _orthonormal(matrix: np.ndarray) -> bool:
    if matrix is None or matrix.shape != (3, 3):
        return False
    if not bool(np.all(np.isfinite(matrix))):
        return False
    gram = matrix.T @ matrix
    if not bool(np.all(np.isfinite(gram))):
        return False
    return bool(np.allclose(gram, np.eye(3), atol=1e-6, rtol=0.0))


def _obb_corners(center: np.ndarray, rotation: np.ndarray, half: np.ndarray) -> np.ndarray:
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    local = signs * half.reshape(1, 3)
    return (rotation @ local.T).T + center.reshape(1, 3)


def _validate_obb(obb: dict, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(obb, dict):
        raise PreparationError(f"{label} is not an OBB dict")
    center = _finite_vec3(obb.get("center"))
    rotation = _finite_matrix3(obb.get("rotation"))
    half = _finite_vec3(obb.get("half"))
    if center is None:
        raise PreparationError(f"{label} center is not finite 3-D")
    if rotation is None:
        raise PreparationError(f"{label} rotation is not finite 3x3")
    if not _orthonormal(rotation):
        raise PreparationError(f"{label} rotation axes are not orthonormal")
    if half is None:
        raise PreparationError(f"{label} half extents are not finite 3-D")
    if bool(np.any(half < -_EPS)):
        raise PreparationError(f"{label} half extents must be nonnegative")
    return center, rotation, np.abs(half)


def _obb_axes(rotation: np.ndarray, half: np.ndarray) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    for index in range(3):
        axis = rotation[:, index].astype(np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= _EPS:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / norm
        if float(half[index]) <= _EPS:
            axis = axis * 0.0
        axes.append(axis)
    return axes


def obb_overlap(a: dict, b: dict, margin: float = 0.003) -> bool:
    """Standard 15-axis separating-axis test for two oriented bounding boxes.

    Boundary contact counts as collision.  A small positive ``margin`` inflates
    the test so that kissing geometry is rejected.  Returns ``True`` when the two
    OBBs overlap (or touch within the margin).
    """

    a_center, a_rotation, a_half = _validate_obb(a, "obb.a")
    b_center, b_rotation, b_half = _validate_obb(b, "obb.b")

    margin_value = _finite_scalar(margin)
    if margin_value is None or margin_value < 0.0:
        raise PreparationError("obb_overlap margin must be finite and nonnegative")

    a_half = np.asarray(a_half, dtype=np.float64) + margin_value
    b_half = np.asarray(b_half, dtype=np.float64) + margin_value

    a_axes = _obb_axes(a_rotation, a_half)
    b_axes = _obb_axes(b_rotation, b_half)

    center_delta = b_center - a_center

    test_axes: list[np.ndarray] = []
    test_axes.extend(a_axes)
    test_axes.extend(b_axes)
    for axis_a in a_axes:
        for axis_b in b_axes:
            cross = np.cross(axis_a, axis_b)
            norm = float(np.linalg.norm(cross))
            if norm > _EPS:
                test_axes.append(cross / norm)

    for axis in test_axes:
        norm = float(np.linalg.norm(axis))
        if norm <= _EPS:
            continue
        unit = axis / norm
        projection_a = sum(
            float(a_half[i]) * abs(float(np.dot(unit, a_axes[i]))) for i in range(3)
        )
        projection_b = sum(
            float(b_half[i]) * abs(float(np.dot(unit, b_axes[i]))) for i in range(3)
        )
        distance = abs(float(np.dot(unit, center_delta)))
        if distance > projection_a + projection_b + _EPS:
            return False
    return True


# --------------------------------------------------------------------------- #
# Reading helpers
# --------------------------------------------------------------------------- #
def _reading_pose(reading: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(reading, dict):
        raise PreparationError("reading is not a dict")
    pose = reading.get("pose")
    if not isinstance(pose, dict):
        raise PreparationError("reading.pose missing")
    position = _finite_vec3(pose.get("position"))
    orientation = _finite_matrix3(pose.get("orientation_matrix"))
    if position is None:
        raise PreparationError("reading.pose.position not finite 3-D")
    if orientation is None:
        raise PreparationError("reading.pose.orientation_matrix not finite 3x3")
    return position, orientation


def _reading_home_orientation(reading: Any) -> np.ndarray:
    home = _finite_matrix3(reading.get("home_orientation"))
    if home is None:
        raise PreparationError("reading.home_orientation not finite 3x3")
    if not _orthonormal(home):
        raise PreparationError("reading.home_orientation not orthonormal")
    return home


def _reading_hand(reading: Any) -> list[dict]:
    hand = reading.get("hand")
    if not isinstance(hand, list) or len(hand) == 0:
        raise PreparationError("reading.hand must be a non-empty list of OBBs")
    for index, obb in enumerate(hand):
        _validate_obb(obb, f"reading.hand[{index}]")
    return hand


def _reading_obstacles(reading: Any) -> list[dict]:
    obstacles = reading.get("obstacles")
    if not isinstance(obstacles, list):
        raise PreparationError("reading.obstacles must be a list")
    for index, obb in enumerate(obstacles):
        _validate_obb(obb, f"reading.obstacles[{index}]")
    return obstacles


def _obb_name(obb: dict, fallback: str) -> str:
    name = obb.get("name")
    if isinstance(name, str) and name:
        return name
    if isinstance(name, str):
        return fallback
    return fallback


def transformed_hand(reading: Any, position: Any, orientation: Any) -> list[dict]:
    """Return hand OBBs transformed to a candidate world pose.

    The supplied ``position``/``orientation`` is treated as the current end
    effector reference frame.  Each hand OBB is converted to a local relative
    pose (position and rotation) with respect to that reference, then mapped into
    the candidate world pose.  Half extents are preserved unchanged.
    """

    current_position, current_orientation = _reading_pose(reading)
    if not isinstance(position, (tuple, list, np.ndarray)):
        raise PreparationError("transformed_hand position is not a vector")
    if not isinstance(orientation, (tuple, list, np.ndarray)):
        raise PreparationError("transformed_hand orientation is not a matrix")
    candidate_position = _finite_vec3(position)
    candidate_orientation = _finite_matrix3(orientation)
    if candidate_position is None:
        raise PreparationError("transformed_hand candidate position not finite 3-D")
    if candidate_orientation is None:
        raise PreparationError("transformed_hand candidate orientation not finite 3x3")
    if not _orthonormal(candidate_orientation):
        raise PreparationError("transformed_hand candidate orientation not orthonormal")

    hand = _reading_hand(reading)

    reference_inverse = current_orientation.T
    transformed: list[dict] = []
    for index, obb in enumerate(hand):
        center, rotation, half = _validate_obb(obb, f"reading.hand[{index}]")
        local_center = reference_inverse @ (center - current_position)
        local_rotation = reference_inverse @ rotation
        world_center = candidate_orientation @ local_center + candidate_position
        world_rotation = candidate_orientation @ local_rotation
        transformed.append(
            {
                "name": _obb_name(obb, f"hand[{index}]"),
                "center": world_center,
                "rotation": world_rotation,
                "half": half.copy(),
                "convex": obb.get("convex"),
            }
        )
    return transformed


def pose_collisions(reading: Any, position: Any, orientation: Any) -> list[str]:
    """Return ``a|b`` pair labels for every hand/obstacle OBB overlap.

    Every transformed hand OBB is checked against *every* obstacle OBB,
    including the target object.  No object is exempted and no classification is
    inferred from object identifiers.
    """

    hand = transformed_hand(reading, position, orientation)
    obstacles = _reading_obstacles(reading)
    collisions: list[str] = []
    broadphase_padding = 0.006
    for hand_index, hand_obb in enumerate(hand):
        hand_name = _obb_name(hand_obb, f"hand[{hand_index}]")
        h_center, h_rotation, h_half = _validate_obb(
            hand_obb, f"transformed_hand[{hand_index}]"
        )
        h_abs_r = np.abs(h_rotation)
        h_extent = h_abs_r @ h_half + broadphase_padding
        for obstacle_index, obstacle_obb in enumerate(obstacles):
            obstacle_name = _obb_name(obstacle_obb, f"obstacle[{obstacle_index}]")
            o_center, o_rotation, o_half = _validate_obb(
                obstacle_obb, f"reading.obstacles[{obstacle_index}]"
            )
            o_abs_r = np.abs(o_rotation)
            o_extent = o_abs_r @ o_half + broadphase_padding
            if bool(
                np.any(
                    np.abs(o_center - h_center)
                    > h_extent + o_extent + _EPS
                )
            ):
                continue
            if obb_overlap(hand_obb, obstacle_obb):
                if support_certificate(hand_obb, obstacle_obb)["certified_clear"]:
                    continue
                collisions.append(f"{hand_name}|{obstacle_name}")
    return collisions


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _validate_target(reading: Any) -> tuple[np.ndarray, np.ndarray]:
    target = reading.get("target")
    if not isinstance(target, dict):
        raise PreparationError("reading.target missing")
    minimum = _finite_vec3(target.get("world_aabb_min"))
    maximum = _finite_vec3(target.get("world_aabb_max"))
    if minimum is None:
        raise PreparationError("target.world_aabb_min not finite 3-D")
    if maximum is None:
        raise PreparationError("target.world_aabb_max not finite 3-D")
    if bool(np.any(maximum < minimum - _EPS)):
        raise PreparationError("target AABB min exceeds max")
    return minimum, maximum


def _validate_empty_hand_context(ctx: Any, reading: Any) -> float:
    snapshot = reading.get("snapshot")
    if not isinstance(snapshot, dict):
        raise PreparationError("reading.snapshot missing")

    complete = snapshot.get("grasp_observation_complete")
    if complete is not True:
        raise PreparationError("grasp_observation_complete must be exactly True")

    held = snapshot.get("held_objects")
    if not isinstance(held, list) or len(held) != 0:
        raise PreparationError("snapshot.held_objects must be an empty list")

    gap = _finite_scalar(reading.get("gripper_gap_m"))
    if gap is None:
        raise PreparationError("gripper_gap_m not finite")
    if gap < _SAFE_GAP_MIN - _EPS:
        raise PreparationError(
            f"gripper_gap_m {gap:.4f} below required empty-hand clearance"
        )

    contacts = reading.get("robot_contacts")
    if not isinstance(contacts, list) or len(contacts) != 0:
        raise PreparationError("robot_contacts must be a known empty list")

    return gap


def _workspace_ok(position: np.ndarray) -> bool:
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    if not (_WORKSPACE_X[0] - _EPS <= x <= _WORKSPACE_X[1] + _EPS):
        return False
    if not (_WORKSPACE_Y[0] - _EPS <= y <= _WORKSPACE_Y[1] + _EPS):
        return False
    if not (_WORKSPACE_Z[0] - _EPS <= z <= _WORKSPACE_Z[1] + _EPS):
        return False
    return True


def _max_hand_reach(reading: Any, current_position: np.ndarray) -> float:
    hand = _reading_hand(reading)
    reach = 0.0
    for index, obb in enumerate(hand):
        center, _rotation, half = _validate_obb(obb, f"reading.hand[{index}]")
        reach = max(
            reach,
            float(np.linalg.norm(center - current_position) + np.linalg.norm(half)),
        )
    return reach


def _obstacle_highest_z(obb: dict) -> float:
    center, rotation, half = _validate_obb(obb, "obstacle")
    corners = _obb_corners(center, rotation, half)
    return float(np.max(corners[:, 2]))


def _relevant_obstacles(
    reading: Any,
    current_xy: np.ndarray,
    target_xy: np.ndarray,
    reach: float,
) -> list[dict]:
    obstacles = _reading_obstacles(reading)
    expand = reach + 0.10
    low_x = min(float(current_xy[0]), float(target_xy[0])) - expand
    high_x = max(float(current_xy[0]), float(target_xy[0])) + expand
    low_y = min(float(current_xy[1]), float(target_xy[1])) - expand
    high_y = max(float(current_xy[1]), float(target_xy[1])) + expand

    relevant: list[dict] = []
    for index, obb in enumerate(obstacles):
        center, rotation, half = _validate_obb(obb, f"reading.obstacles[{index}]")
        corners = _obb_corners(center, rotation, half)
        min_cx = float(np.min(corners[:, 0]))
        max_cx = float(np.max(corners[:, 0]))
        min_cy = float(np.min(corners[:, 1]))
        max_cy = float(np.max(corners[:, 1]))
        if max_cx < low_x or min_cx > high_x:
            continue
        if max_cy < low_y or min_cy > high_y:
            continue
        relevant.append(obb)
    return relevant


def _down_extent(reading: Any, target_home_orientation: np.ndarray) -> float:
    """Minimum relative hand corner Z at the target home orientation."""

    zero_position = np.zeros(3, dtype=np.float64)
    hand = transformed_hand(reading, zero_position, target_home_orientation)
    minimum_corner_z = math.inf
    for index, obb in enumerate(hand):
        center, rotation, half = _validate_obb(obb, f"transformed_hand[{index}]")
        corners = _obb_corners(center, rotation, half)
        minimum_corner_z = min(minimum_corner_z, float(np.min(corners[:, 2])))
    if not math.isfinite(minimum_corner_z):
        raise PreparationError("could not compute hand corner Z extent")
    return max(0.0, -minimum_corner_z)


def _interpolate_positions(
    start: np.ndarray, end: np.ndarray
) -> list[np.ndarray]:
    delta = end - start
    length = float(np.linalg.norm(delta))
    steps = max(1, int(math.ceil(length / _POSITION_STEP)))
    samples: list[np.ndarray] = []
    for index in range(steps + 1):
        fraction = index / steps
        samples.append(start + delta * fraction)
    return samples


def _interpolate_orientations(
    start: np.ndarray, end: np.ndarray
) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation, Slerp  # lazy import

    start_rotation = Rotation.from_matrix(start)
    end_rotation = Rotation.from_matrix(end)
    start_vector = start_rotation.as_rotvec()
    end_vector = end_rotation.as_rotvec()
    angle = float(np.linalg.norm(end_vector - start_vector))
    steps = max(1, int(math.ceil(angle / _ANGULAR_STEP)))
    key_rotations = Rotation.from_matrix(np.stack([start, end], axis=0))
    slerp = Slerp([0.0, 1.0], key_rotations)
    samples: list[np.ndarray] = []
    for index in range(steps + 1):
        fraction = index / steps
        samples.append(slerp([fraction])[0].as_matrix())
    return samples


def _segment_length(start: np.ndarray, end: np.ndarray) -> float:
    return float(np.linalg.norm(end - start))


def _check_route(
    reading: Any,
    origin_position: np.ndarray,
    origin_orientation: np.ndarray,
    start_position: np.ndarray,
    start_orientation: np.ndarray,
    waypoints: list[dict],
) -> tuple[bool, str, int]:
    """Validate a single route; return (ok, reason, sample_count)."""

    from scipy.spatial.transform import Rotation, Slerp  # lazy import

    samples = 0

    def check_pose(position: np.ndarray, orientation: np.ndarray) -> Optional[str]:
        nonlocal samples
        if not _workspace_ok(position):
            return "workspace_violation"
        collisions = pose_collisions(reading, position, orientation)
        if collisions:
            return "pose_collision"
        samples += 1
        return None

    reason = check_pose(origin_position, origin_orientation)
    if reason is not None:
        return False, reason, samples

    def check_segment(
        from_position: np.ndarray,
        from_orientation: np.ndarray,
        to_position: np.ndarray,
        to_orientation: np.ndarray,
    ) -> Optional[str]:
        nonlocal samples
        delta_position = to_position - from_position
        delta_length = float(np.linalg.norm(delta_position))
        relative_rotation = Rotation.from_matrix(
            to_orientation @ from_orientation.T
        )
        relative_angle = float(relative_rotation.magnitude())
        grid = max(
            1,
            int(math.ceil(delta_length / _POSITION_STEP)),
            int(math.ceil(relative_angle / _ANGULAR_STEP)),
        )
        key_rotations = Rotation.from_matrix(
            np.stack([from_orientation, to_orientation], axis=0)
        )
        slerp = Slerp([0.0, 1.0], key_rotations)
        for index in range(grid + 1):
            fraction = index / grid
            position = from_position + delta_position * fraction
            orientation = slerp([fraction])[0].as_matrix()
            if not _workspace_ok(position):
                return "workspace_violation"
            collisions = pose_collisions(reading, position, orientation)
            if collisions:
                return "pose_collision"
            samples += 1
        return None

    current_position = origin_position
    current_orientation = origin_orientation

    reason = check_segment(
        current_position,
        current_orientation,
        start_position,
        start_orientation,
    )
    if reason is not None:
        return False, reason, samples
    current_position = start_position
    current_orientation = start_orientation

    for waypoint in waypoints:
        target_position = np.asarray(waypoint["position"], dtype=np.float64)
        target_orientation = np.asarray(waypoint["orientation"], dtype=np.float64)
        reason = check_segment(
            current_position,
            current_orientation,
            target_position,
            target_orientation,
        )
        if reason is not None:
            return False, reason, samples
        current_position = target_position
        current_orientation = target_orientation

    return True, "", samples


def plan_route(ctx: Any, reading: Any) -> dict:
    """Plan a checked empty-hand approach route that stops above the object."""

    if ctx is None:
        raise PreparationError("ctx is required")

    gap = _validate_empty_hand_context(ctx, reading)

    current_position, current_orientation = _reading_pose(reading)
    if not _orthonormal(current_orientation):
        raise PreparationError("reading.pose.orientation_matrix not orthonormal")

    home_orientation = _reading_home_orientation(reading)
    if float(home_orientation[2, 2]) >= _ORIENTATION_FLOOR:
        raise PreparationError(
            "home_orientation does not point sufficiently downward (R[2,2] >= -0.85)"
        )

    ctx_position = _finite_vec3(getattr(ctx, "position", None))
    if ctx_position is None:
        raise PreparationError("ctx.position not finite 3-D")

    target_min, target_max = _validate_target(reading)
    target_xy = (target_min[:2] + target_max[:2]) / 2.0

    ready_xy = ctx_position[:2]
    down_extent = _down_extent(reading, home_orientation)
    ready_z = max(
        float(ctx_position[2]) + 0.20,
        float(target_max[2]) + down_extent + 0.08,
    )
    ready_position = np.array(
        [float(ready_xy[0]), float(ready_xy[1]), float(ready_z)],
        dtype=np.float64,
    )

    if not _workspace_ok(ready_position):
        raise PreparationError(
            f"ready position {ready_position.tolist()} outside workspace"
        )

    reach = _max_hand_reach(reading, current_position)

    current_xy = current_position[:2]
    relevant = _relevant_obstacles(reading, current_xy, target_xy, reach)
    highest_relevant_z = 0.0
    for obb in relevant:
        highest_relevant_z = max(highest_relevant_z, _obstacle_highest_z(obb))

    high_z = max(
        float(current_position[2]) + 0.10,
        float(ready_z),
        highest_relevant_z + reach + 0.025,
    )

    high_position = np.array(
        [float(current_xy[0]), float(current_xy[1]), float(high_z)],
        dtype=np.float64,
    )
    if not _workspace_ok(high_position):
        raise PreparationError(
            f"high lift Z {high_z:.4f} exceeds workspace limits"
        )

    fixed_waypoints = [
        {
            "name": "lift_clear",
            "position": high_position.copy(),
            "orientation": current_orientation.copy(),
        },
        {
            "name": "align_clear",
            "position": high_position.copy(),
            "orientation": home_orientation.copy(),
        },
        {
            "name": "transit",
            "position": np.array(
                [float(ctx_position[0]), float(ctx_position[1]), float(high_z)],
                dtype=np.float64,
            ),
            "orientation": home_orientation.copy(),
        },
        {
            "name": "ready",
            "position": ready_position.copy(),
            "orientation": home_orientation.copy(),
        },
    ]

    for waypoint in fixed_waypoints:
        if not _workspace_ok(waypoint["position"]):
            raise PreparationError(
                f"candidate waypoint {waypoint['name']} outside workspace"
            )

    escape_directions: list[np.ndarray] = []
    to_target = (
        np.array([float(target_xy[0]), float(target_xy[1])], dtype=np.float64)
        - current_xy
    )
    escape_directions.append(to_target)
    escape_directions.append(np.array([1.0, 0.0], dtype=np.float64))
    escape_directions.append(np.array([-1.0, 0.0], dtype=np.float64))
    escape_directions.append(np.array([0.0, 1.0], dtype=np.float64))
    escape_directions.append(np.array([0.0, -1.0], dtype=np.float64))

    obstacles = _reading_obstacles(reading)
    nearest_center: Optional[np.ndarray] = None
    nearest_distance = math.inf
    for index, obb in enumerate(obstacles):
        center, _rotation, _half = _validate_obb(obb, f"reading.obstacles[{index}]")
        distance = float(np.linalg.norm(center[:2] - current_xy))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_center = center[:2]
    if nearest_center is not None:
        escape_directions.append(current_xy - nearest_center)
    else:
        escape_directions.append(np.array([0.0, 0.0], dtype=np.float64))

    candidates: list[dict] = []
    rejection_reasons: list[str] = []

    def build_candidate(escape_xy: Optional[np.ndarray], index: int) -> Optional[dict]:
        if escape_xy is None:
            return None
        norm = float(np.linalg.norm(escape_xy))
        if norm <= _EPS:
            return None
        unit = escape_xy / norm
        escape_xy_world = current_xy + unit * 0.08
        escape_z = float(current_position[2]) + 0.04
        lift_position = np.array(
            [float(escape_xy_world[0]), float(escape_xy_world[1]), float(escape_z)],
            dtype=np.float64,
        )
        lift_clear_position = np.array(
            [float(escape_xy_world[0]), float(escape_xy_world[1]), float(high_z)],
            dtype=np.float64,
        )
        candidate_waypoints = [
            {
                "name": "lift_escape",
                "position": lift_position,
                "orientation": current_orientation.copy(),
            },
            {
                "name": "lift_clear",
                "position": lift_clear_position,
                "orientation": current_orientation.copy(),
            },
            {
                "name": "align_clear",
                "position": lift_clear_position.copy(),
                "orientation": home_orientation.copy(),
            },
            {
                "name": "transit",
                "position": np.array(
                    [float(ctx_position[0]), float(ctx_position[1]), float(high_z)],
                    dtype=np.float64,
                ),
                "orientation": home_orientation.copy(),
            },
            {
                "name": "ready",
                "position": ready_position.copy(),
                "orientation": home_orientation.copy(),
            },
        ]
        for waypoint in candidate_waypoints:
            if not _workspace_ok(waypoint["position"]):
                return None
        return {
            "index": index,
            "name": f"escape_{index}",
            "waypoints": candidate_waypoints,
            "origin_position": current_position.copy(),
            "origin_orientation": current_orientation.copy(),
            "start_position": current_position.copy(),
            "start_orientation": current_orientation.copy(),
        }

    fixed_candidate = {
        "index": 0,
        "name": "fixed",
        "waypoints": fixed_waypoints,
        "origin_position": current_position.copy(),
        "origin_orientation": current_orientation.copy(),
        "start_position": current_position.copy(),
        "start_orientation": current_orientation.copy(),
    }
    candidates.append(fixed_candidate)

    for offset, direction in enumerate(escape_directions, start=1):
        candidate = build_candidate(direction, offset)
        if candidate is None:
            rejection_reasons.append(f"escape_{offset}:zero_or_out_of_workspace")
            continue
        candidates.append(candidate)

    def total_length(candidate: dict) -> float:
        length = 0.0
        previous = candidate["start_position"]
        for waypoint in candidate["waypoints"]:
            length += _segment_length(previous, waypoint["position"])
            previous = waypoint["position"]
        return length

    ordered = sorted(
        candidates,
        key=lambda candidate: (total_length(candidate), candidate["index"]),
    )

    for candidate in ordered:
        ok, reason, sample_count = _check_route(
            reading,
            candidate["origin_position"],
            candidate["origin_orientation"],
            candidate["start_position"],
            candidate["start_orientation"],
            candidate["waypoints"],
        )
        if ok:
            return {
                "chosen_index": int(candidate["index"]),
                "waypoints": [
                    {
                        "name": waypoint["name"],
                        "position": waypoint["position"].copy(),
                        "orientation": waypoint["orientation"].copy(),
                    }
                    for waypoint in candidate["waypoints"]
                ],
                "check_samples": int(sample_count),
                "rejection_reasons": list(rejection_reasons),
                "geometry_source": reading.get("geometry_source")
                or getattr(ctx, "geometry_source", None),
                "collision_scope": "gripper_obb_samples_only",
                "high_z": float(high_z),
                "ready_z": float(ready_z),
                "max_hand_reach": float(reach),
                "note": (
                    "Sampled gripper OBBs only; not a full-arm or all-physics-substep "
                    "collision guarantee. Preparation only: no descent to grasp, no closing."
                ),
            }
        else:
            rejection_reasons.append(
                f"{candidate['name']}:{reason} (samples={sample_count})"
            )

    raise PreparationError("no_checked_path")


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class PreparationController:
    """Fixed-plan empty-hand approach servo state machine.

    The controller never calls ``env.step`` and never reads or mutates a model or
    environment.  ``next_action`` returns one servo delta from the existing
    ``preparation_diagnostics.servo_action`` helper while the phase is
    ``preparing``; it returns ``None`` once the plan is confirmed or has failed.
    """

    def __init__(self, ctx: Any, reading: Any, max_actions: int = 200) -> None:
        if ctx is None:
            raise PreparationError("ctx is required")
        actions_limit = int(max_actions)
        if actions_limit <= 0:
            raise PreparationError("max_actions must be positive")
        if actions_limit > _DEFAULT_MAX_ACTIONS:
            raise PreparationError("max_actions must not exceed 200")

        self._ctx = ctx
        self._plan = plan_route(ctx, reading)
        self._max_actions = actions_limit

        self._baseline_objects = self._freeze_baseline_objects(reading)
        self._baseline_true_goals = self._freeze_baseline_goals(reading)

        self._phase = "preparing"
        self._reason: Optional[str] = None
        self._waypoint_index = 0
        self._stage_counts: dict[str, int] = {}
        self._actual_aux_actions = 0
        self._last_errors: Optional[dict] = None
        self._consecutive_confirmations = 0
        self._ready_confirmed = False
        self._initial_position, self._initial_orientation = _reading_pose(reading)

        plan_position = self._plan["waypoints"][self._waypoint_index]["position"]
        plan_orientation = self._plan["waypoints"][self._waypoint_index]["orientation"]
        self._validate_runtime_reading(
            reading,
            plan_position,
            plan_orientation,
            source_position=self._recover_position(reading),
            source_orientation=self._recover_orientation(reading),
        )

    # -- freezing ---------------------------------------------------------- #
    @staticmethod
    def _prepare_snapshot_objects(ctx: Any, reading: Any) -> dict:
        snapshot = reading.get("snapshot")
        if not isinstance(snapshot, dict):
            raise PreparationError("reading.snapshot missing")
        objects = snapshot.get("objects")
        if not isinstance(objects, dict):
            raise PreparationError("reading.snapshot.objects must be a dict of objects")
        return objects

    def _freeze_baseline_objects(self, reading: Any) -> dict:
        objects = self._prepare_snapshot_objects(self._ctx, reading)
        baseline: dict[str, np.ndarray] = {}
        for name, obj in objects.items():
            if not isinstance(obj, dict):
                raise PreparationError(f"snapshot object {name} is not a dict")
            position = _finite_vec3(obj.get("position"))
            if position is None:
                raise PreparationError(
                    f"snapshot object {name} position unknown; cannot baseline"
                )
            baseline[str(name)] = position
        return baseline

    def _freeze_baseline_goals(self, reading: Any) -> dict[str, bool]:
        snapshot = reading.get("snapshot")
        if not isinstance(snapshot, dict):
            raise PreparationError("reading.snapshot missing")
        predicates = snapshot.get("predicates")
        if not isinstance(predicates, dict):
            raise PreparationError("snapshot.predicates must be a dict")
        baseline: dict[str, bool] = {}
        for key, value in predicates.items():
            if value is None:
                continue
            if not isinstance(value, bool):
                raise PreparationError(
                    f"snapshot predicate {key} is neither exact bool nor None"
                )
            if value is True:
                baseline[str(key)] = True
        return baseline

    # -- runtime validation ------------------------------------------------ #
    def _recover_position(self, reading: Any) -> np.ndarray:
        position, _orientation = _reading_pose(reading)
        return position

    def _recover_orientation(self, reading: Any) -> np.ndarray:
        _position, orientation = _reading_pose(reading)
        return orientation

    def _validate_runtime_reading(
        self,
        reading: Any,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        source_position: np.ndarray,
        source_orientation: np.ndarray,
    ) -> None:
        if not isinstance(reading, dict):
            raise PreparationError("runtime reading is not a dict")

        snapshot = reading.get("snapshot")
        if not isinstance(snapshot, dict):
            raise PreparationError("reading.snapshot missing")

        complete = snapshot.get("grasp_observation_complete")
        if complete is not True:
            raise PreparationError("grasp_observation_complete not True during runtime")

        held = snapshot.get("held_objects")
        if not isinstance(held, list) or len(held) != 0:
            raise PreparationError("snapshot.held_objects must be an empty list")

        gap = _finite_scalar(reading.get("gripper_gap_m"))
        if gap is None:
            raise PreparationError("gripper_gap_m not finite during runtime")
        if gap < _SAFE_GAP_MIN - _EPS:
            raise PreparationError("gripper_gap_m below empty-hand clearance")

        contacts = reading.get("robot_contacts")
        if not isinstance(contacts, list) or len(contacts) != 0:
            raise PreparationError("robot_contacts must be a known empty list")

        objects = snapshot.get("objects")
        if not isinstance(objects, dict):
            raise PreparationError("reading.snapshot.objects must be a dict of objects")
        for name, baseline_position in self._baseline_objects.items():
            if name not in objects:
                raise PreparationError(f"baseline object {name} missing at runtime")
            obj = objects[name]
            if not isinstance(obj, dict):
                raise PreparationError(f"baseline object {name} not a dict")
            position = _finite_vec3(obj.get("position"))
            if position is None:
                raise PreparationError(
                    f"baseline object {name} position unknown at runtime"
                )
            displacement = float(np.linalg.norm(position - baseline_position))
            if displacement > _BASELINE_DISPLACEMENT + _EPS:
                raise PreparationError(
                    f"baseline object {name} moved by {displacement:.4f}m"
                )

        predicates = snapshot.get("predicates")
        if not isinstance(predicates, dict):
            raise PreparationError("snapshot.predicates must be a dict at runtime")
        for key in self._baseline_true_goals:
            if key not in predicates:
                raise PreparationError(
                    f"baseline true predicate {key} missing at runtime"
                )
            if predicates[key] is not True:
                raise PreparationError(
                    f"baseline true predicate {key} no longer exactly True"
                )

        current_position, current_orientation = _reading_pose(reading)
        if not _orthonormal(current_orientation):
            raise PreparationError(
                "reading.pose.orientation_matrix not orthonormal during runtime"
            )
        if not _workspace_ok(current_position):
            raise PreparationError("current pose position outside workspace")
        current_collisions = pose_collisions(
            reading, current_position, current_orientation
        )
        if current_collisions:
            raise PreparationError(
                "current pose in collision: " + ",".join(current_collisions)
            )

        if not _workspace_ok(target_position):
            raise PreparationError("target waypoint position outside workspace")
        collisions = pose_collisions(reading, target_position, target_orientation)
        if collisions:
            raise PreparationError(
                "target waypoint in collision: " + ",".join(collisions)
            )

    # -- public API -------------------------------------------------------- #
    def next_action(self, reading: Any) -> Optional[np.ndarray]:
        """Return one servo delta while ``preparing``; else ``None``."""

        if self._phase != "preparing":
            return None

        if self._actual_aux_actions >= self._max_actions:
            self._phase = "failed"
            self._reason = "prepare_budget_exhausted"
            return None

        if self._waypoint_index >= len(self._plan["waypoints"]):
            self._phase = "failed"
            self._reason = "prepare_budget_exhausted"
            return None

        waypoint = self._plan["waypoints"][self._waypoint_index]
        target_position = np.asarray(waypoint["position"], dtype=np.float64)
        target_orientation = np.asarray(waypoint["orientation"], dtype=np.float64)

        try:
            source_position = self._recover_position(reading)
            source_orientation = self._recover_orientation(reading)
            self._validate_runtime_reading(
                reading,
                target_position,
                target_orientation,
                source_position,
                source_orientation,
            )
        except PreparationError as exc:
            self._phase = "failed"
            self._reason = str(exc)
            return None

        try:
            # Lazy import avoids model/service import at module import time.
            from preparation_diagnostics import servo_action
        except Exception as exc:  # pragma: no cover - defensive
            raise PreparationError(f"servo_action unavailable: {exc}") from exc

        return servo_action(
            target_position,
            source_position,
            target_orientation,
            source_orientation,
        )

    def observe_after(self, reading: Any) -> None:
        """Count one real auxiliary action and possibly advance the waypoint."""

        if self._phase != "preparing":
            return

        self._actual_aux_actions += 1

        if self._waypoint_index >= len(self._plan["waypoints"]):
            self._phase = "failed"
            self._reason = "prepare_budget_exhausted"
            return

        waypoint = self._plan["waypoints"][self._waypoint_index]
        target_position = np.asarray(waypoint["position"], dtype=np.float64)
        target_orientation = np.asarray(waypoint["orientation"], dtype=np.float64)

        try:
            source_position = self._recover_position(reading)
            source_orientation = self._recover_orientation(reading)
            self._validate_runtime_reading(
                reading,
                target_position,
                target_orientation,
                source_position,
                source_orientation,
            )
        except PreparationError as exc:
            self._phase = "failed"
            self._reason = str(exc)
            return

        stage_name = str(waypoint.get("name", f"waypoint_{self._waypoint_index}"))

        try:
            from preparation_diagnostics import orientation_error_rad
        except Exception as exc:  # pragma: no cover - defensive
            raise PreparationError(
                f"orientation_error_rad unavailable: {exc}"
            ) from exc

        position_error = float(np.linalg.norm(source_position - target_position))
        orientation_error = orientation_error_rad(
            source_orientation, target_orientation
        )
        if orientation_error is None or not math.isfinite(float(orientation_error)):
            self._phase = "failed"
            self._reason = "orientation_error_unknown"
            return

        orientation_error = float(orientation_error)
        self._last_errors = {
            "position_error": position_error,
            "orientation_error": orientation_error,
        }

        self._stage_counts[stage_name] = self._stage_counts.get(stage_name, 0) + 1

        if (
            position_error <= _WAYPOINT_POSITION_TOL + _EPS
            and orientation_error <= _WAYPOINT_ORIENTATION_TOL + _EPS
        ):
            self._consecutive_confirmations += 1
        else:
            self._consecutive_confirmations = 0

        if self._consecutive_confirmations >= _CONFIRMATION_SAMPLES:
            self._consecutive_confirmations = 0
            self._waypoint_index += 1
            if self._waypoint_index >= len(self._plan["waypoints"]):
                if self._actual_aux_actions <= self._max_actions:
                    self._phase = "ready"
                    self._reason = None
                    self._ready_confirmed = True
                else:
                    self._phase = "failed"
                    self._reason = "prepare_budget_exhausted"
                return
            self._last_errors = None

        if self._actual_aux_actions >= self._max_actions:
            self._phase = "failed"
            self._reason = "prepare_budget_exhausted"
            return

        stage_actions = self._stage_counts.get(stage_name, 0)
        if stage_actions >= _WAYPOINT_ACTION_LIMIT:
            self._phase = "failed"
            self._reason = "stage_timeout"
            return

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def summary(self) -> dict:
        return {
            "phase": self._phase,
            "reason": self._reason,
            "current_stage": (
                self._plan["waypoints"][self._waypoint_index]["name"]
                if self._waypoint_index < len(self._plan["waypoints"])
                else "done"
            ),
            "stage_counts": dict(self._stage_counts),
            "actual_aux_actions": int(self._actual_aux_actions),
            "last_errors": dict(self._last_errors) if self._last_errors else None,
            "plan": {
                "chosen_index": int(self._plan["chosen_index"]),
                "check_samples": int(self._plan["check_samples"]),
                "rejection_reasons": list(self._plan["rejection_reasons"]),
                "geometry_source": self._plan.get("geometry_source"),
                "collision_scope": self._plan.get("collision_scope"),
                "high_z": self._plan.get("high_z"),
                "ready_z": self._plan.get("ready_z"),
                "max_hand_reach": self._plan.get("max_hand_reach"),
                "waypoints": [
                    {
                        "name": waypoint["name"],
                        "position": [float(value) for value in waypoint["position"]],
                        "orientation": [
                            [float(value) for value in row]
                            for row in waypoint["orientation"]
                        ],
                    }
                    for waypoint in self._plan["waypoints"]
                ],
            },
            "ready_confirmed": bool(self._ready_confirmed),
            "confirmation": "preparation_only_not_grasp_or_placement",
        }


def _convex_axes_from_local(convex, rotation, label):
    if not isinstance(convex, dict):
        raise PreparationError(f"{label} convex is not a dict")
    vertices = convex.get("vertices")
    face_axes = convex.get("face_axes")
    edge_axes = convex.get("edge_axes")
    if vertices is None or face_axes is None or edge_axes is None:
        raise PreparationError(f"{label} convex members missing")
    try:
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(face_axes, dtype=np.float64)
        e = np.asarray(edge_axes, dtype=np.float64)
    except Exception:
        raise PreparationError(f"{label} convex arrays malformed")
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 4:
        raise PreparationError(f"{label} convex vertices malformed")
    if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 1:
        raise PreparationError(f"{label} convex face axes malformed")
    if e.ndim != 2 or e.shape[1] != 3 or e.shape[0] < 1:
        raise PreparationError(f"{label} convex edge axes malformed")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(e)):
        raise PreparationError(f"{label} convex arrays nonfinite")
    if not _orthonormal(rotation):
        raise PreparationError(f"{label} rotation not orthonormal")
    world_v = v @ rotation.T
    world_f = f @ rotation.T
    world_e = e @ rotation.T
    if not np.all(np.isfinite(world_v)) or not np.all(np.isfinite(world_f)) or not np.all(np.isfinite(world_e)):
        raise PreparationError(f"{label} convex world arrays nonfinite")
    return world_v, world_f, world_e


def _support_unit_axes(rows):
    out = []
    seen = set()
    for row in np.asarray(rows, dtype=np.float64).reshape(-1, 3):
        norm = float(np.linalg.norm(row))
        if norm <= 1e-9:
            continue
        unit = row / norm
        if not np.all(np.isfinite(unit)):
            continue
        nonzero = np.flatnonzero(np.abs(unit) > 1e-9)
        if len(nonzero) and unit[nonzero[0]] < 0.0:
            unit = -unit
        key = tuple(np.round(unit, 9))
        if key not in seen:
            seen.add(key)
            out.append(unit)
    if not out:
        raise PreparationError("no usable separating axes")
    return np.asarray(out, dtype=np.float64)


def support_certificate(a: dict, b: dict) -> dict:
    """Return a pure convex separation certificate for one OBB pair."""

    if not isinstance(a, dict) or not isinstance(b, dict):
        raise PreparationError("support_certificate requires OBB dicts")
    a_center, a_rotation, a_half = _validate_obb(a, "support_certificate.a")
    b_center, b_rotation, b_half = _validate_obb(b, "support_certificate.b")

    a_convex = a.get("convex")
    b_convex = b.get("convex")

    if a_convex is not None and not isinstance(a_convex, dict):
        raise PreparationError("support_certificate.a convex is not a dict")
    if b_convex is not None and not isinstance(b_convex, dict):
        raise PreparationError("support_certificate.b convex is not a dict")

    if isinstance(a_convex, dict):
        a_v, a_f, a_e = _convex_axes_from_local(a_convex, a_rotation, "support_certificate.a")
        a_points = a_v + a_center.reshape(1, 3)
        a_meta = {
            "basis": "actual_mesh_convex_hull",
            "vertex_count": int(a_v.shape[0]),
            "face_axis_count": int(a_f.shape[0]),
            "edge_axis_count": int(a_e.shape[0]),
        }
    else:
        a_points = _obb_corners(a_center, a_rotation, a_half)
        a_f = a_rotation.T.copy()
        a_e = a_rotation.T.copy()
        a_meta = {
            "basis": "unchanged_conservative_obb",
            "vertex_count": 8,
            "face_axis_count": 3,
            "edge_axis_count": 3,
        }

    if isinstance(b_convex, dict):
        b_v, b_f, b_e = _convex_axes_from_local(b_convex, b_rotation, "support_certificate.b")
        b_points = b_v + b_center.reshape(1, 3)
        b_meta = {
            "basis": "actual_mesh_convex_hull",
            "vertex_count": int(b_v.shape[0]),
            "face_axis_count": int(b_f.shape[0]),
            "edge_axis_count": int(b_e.shape[0]),
        }
    else:
        b_points = _obb_corners(b_center, b_rotation, b_half)
        b_f = b_rotation.T.copy()
        b_e = b_rotation.T.copy()
        b_meta = {
            "basis": "unchanged_conservative_obb",
            "vertex_count": 8,
            "face_axis_count": 3,
            "edge_axis_count": 3,
        }

    if not np.all(np.isfinite(a_points)) or not np.all(np.isfinite(b_points)):
        raise PreparationError("support_certificate world vertices nonfinite")

    crosses = np.cross(a_e[:, None, :], b_e[None, :, :]).reshape(-1, 3)
    candidates = np.concatenate((a_f, b_f, a_rotation.T, b_rotation.T, crosses), axis=0)
    axes = _support_unit_axes(candidates)

    pa = a_points @ axes.T
    pb = b_points @ axes.T
    a_min = pa.min(axis=0)
    a_max = pa.max(axis=0)
    b_min = pb.min(axis=0)
    b_max = pb.max(axis=0)
    gaps = np.maximum(b_min - a_max, a_min - b_max)
    best = int(np.argmax(gaps))
    gap = float(gaps[best])
    required = 0.006 + _EPS

    return {
        "certified_clear": bool(gap > required),
        "maximum_separation_gap_m": gap,
        "required_gap_m": required,
        "separating_axis": [float(value) for value in axes[best]],
        "projection_a": [float(a_min[best]), float(a_max[best])],
        "projection_b": [float(b_min[best]), float(b_max[best])],
        "tested_axis_count": int(axes.shape[0]),
        "hand": a_meta,
        "obstacle": b_meta,
        "hand_center": [float(value) for value in a_center],
        "hand_rotation": [[float(a_rotation[r, c]) for c in range(3)] for r in range(3)],
        "obstacle_center": [float(value) for value in b_center],
        "obstacle_rotation": [[float(b_rotation[r, c]) for c in range(3)] for r in range(3)],
    }
