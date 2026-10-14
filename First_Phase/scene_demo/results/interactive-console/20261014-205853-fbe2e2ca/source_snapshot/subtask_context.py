from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


class ContextError(ValueError):
    """Raised when subtask context input is missing, malformed or unknown."""


@dataclass(frozen=True)
class SubtaskContext:
    operation: str
    object_id: str
    target_id: Optional[str]
    goals: tuple
    position: tuple
    orientation: tuple
    dimensions: tuple
    shape_family: str
    geometry_source: str
    calibration_key: Optional[str]
    held_objects: tuple
    observation_complete: bool

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "object_id": self.object_id,
            "target_id": self.target_id,
            "goals": [list(goal) for goal in self.goals],
            "position": list(self.position),
            "orientation": [list(row) for row in self.orientation],
            "dimensions": list(self.dimensions),
            "shape_family": self.shape_family,
            "geometry_source": self.geometry_source,
            "calibration_key": self.calibration_key,
            "held_objects": list(self.held_objects),
            "observation_complete": self.observation_complete,
        }


@dataclass(frozen=True)
class AssistSelection:
    prepare_strategy: Optional[str]
    grasp_strategy: Optional[str]
    status: str
    reason: str
    shape_family: str

    def to_dict(self) -> dict:
        return {
            "prepare_strategy": self.prepare_strategy,
            "grasp_strategy": self.grasp_strategy,
            "status": self.status,
            "reason": self.reason,
            "shape_family": self.shape_family,
        }


REGISTRY: dict = {
    "clearance_preposition_v1": {
        "operation": "pick_place",
        "shape_families": (
            "upright_elongated",
            "low_wide",
            "compact",
            "tilted",
        ),
        "role": "preparation",
        "grasp_strategy": None,
        "description": (
            "Preparation-only clearance strategy applicable to all finite "
            "classified pick-place geometries; it does not select or perform "
            "a grasp."
        ),
    },
    "calibrated_side_v1": {
        "operation": "pick_place",
        "shape_family": "upright_elongated",
        "calibration_key": "libero_wine_side_v1",
        "reference": "libero_wine_side_v1",
        "compatibility": "bounded_old_controller",
        "generalized": False,
        "description": (
            "Side-grasp strategy for the specific validated upright elongated "
            "geometry profile matched by calibration key libero_wine_side_v1; "
            "compatible with the bounded old controller, not generalized to "
            "all elongated objects."
        ),
    },
}


def operation_from_goals(goals) -> str:
    if not isinstance(goals, (list, tuple)) or len(goals) == 0:
        return "unsupported"

    normalized = []
    for goal in goals:
        if not isinstance(goal, (list, tuple)) or len(goal) < 2:
            return "unsupported"
        normalized.append(tuple(goal))

    first = normalized[0]
    first_object = first[1]

    if first[0] in ("on", "in") and len(first) == 3:
        for goal in normalized:
            if (
                len(goal) != 3
                or goal[0] not in ("on", "in")
                or goal[1] != first_object
            ):
                return "unsupported"
        return "pick_place"

    if first[0] in ("turnon", "turnoff") and len(first) == 2:
        for goal in normalized:
            if len(goal) != 2 or goal[0] not in ("turnon", "turnoff"):
                return "unsupported"
        return "toggle"

    return "unsupported"


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _validate_dimensions(dimensions) -> tuple:
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3:
        raise ContextError("dimensions must be exactly three numbers")
    values = []
    for value in dimensions:
        if not _is_finite_number(value):
            raise ContextError("dimensions must be finite numbers excluding bool")
        number = float(value)
        if number <= 0.0:
            raise ContextError("dimensions must be strictly positive")
        values.append(number)
    return tuple(values)


def _validate_orientation(orientation) -> np.ndarray:
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 3:
        raise ContextError("orientation must be a 3x3 matrix")
    matrix = np.empty((3, 3), dtype=float)
    for row_index, row in enumerate(orientation):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ContextError("orientation must be a 3x3 matrix")
        for column_index, value in enumerate(row):
            if not _is_finite_number(value):
                raise ContextError(
                    "orientation entries must be finite numbers excluding bool"
                )
            matrix[row_index, column_index] = float(value)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-5):
        raise ContextError("orientation must be orthogonal")
    determinant = float(np.linalg.det(matrix))
    if abs(determinant - 1.0) > 1e-5:
        raise ContextError("orientation determinant must be close to 1")
    return matrix


def classify_geometry(dimensions, orientation) -> str:
    values = _validate_dimensions(dimensions)
    matrix = _validate_orientation(orientation)
    width, depth, height = values
    up = abs(float(matrix[2, 2]))
    if up < 0.9:
        return "tilted"
    smaller_plan = min(width, depth)
    larger_plan = max(width, depth)
    if height / larger_plan >= 1.6:
        return "upright_elongated"
    if height / smaller_plan <= 0.9:
        return "low_wide"
    return "compact"


def _require_nonempty_str(mapping: dict, key: str, label: str) -> str:
    if key not in mapping:
        raise ContextError("missing %s" % label)
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ContextError("%s must be a nonempty str" % label)
    return value


def _require_position(position) -> tuple:
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ContextError("position must be exactly three numbers")
    values = []
    for value in position:
        if not _is_finite_number(value):
            raise ContextError("position must be finite numbers excluding bool")
        values.append(float(value))
    return tuple(values)


def _require_str_list(value, label: str) -> tuple:
    if not isinstance(value, list):
        raise ContextError("%s must be a list[str]" % label)
    for item in value:
        if not isinstance(item, str):
            raise ContextError("%s must be a list[str]" % label)
    return tuple(value)


def _require_calibration_key(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ContextError("calibration_key must be None or nonempty str")
    return value


def build_context(capability: dict, reading: dict) -> SubtaskContext:
    if not isinstance(capability, dict):
        raise ContextError("capability must be a dict")
    if not isinstance(reading, dict):
        raise ContextError("reading must be a dict")

    for key in ("object_id", "target_id", "goals"):
        if key not in capability:
            raise ContextError("missing capability field %r" % key)

    object_id = capability["object_id"]
    if not isinstance(object_id, str) or object_id == "":
        raise ContextError("object_id must be a nonempty str")

    target_id = capability["target_id"]
    if target_id is not None and (
        not isinstance(target_id, str) or target_id == ""
    ):
        raise ContextError("target_id must be None or nonempty str")

    goals = capability["goals"]
    if not isinstance(goals, (list, tuple)):
        raise ContextError("goals must be a list of goal sequences")
    goal_tuple = tuple(
        tuple(goal)
        if isinstance(goal, (list, tuple))
        else _raise_goal_error()
        for goal in goals
    )

    for key in ("target", "geometry_source", "snapshot"):
        if key not in reading:
            raise ContextError("missing reading field %r" % key)

    target = reading["target"]
    if not isinstance(target, dict):
        raise ContextError("target must be a dict")

    geometry_source = reading["geometry_source"]
    if not isinstance(geometry_source, str) or geometry_source == "":
        raise ContextError("geometry_source must be a nonempty str")

    snapshot = reading["snapshot"]
    if not isinstance(snapshot, dict):
        raise ContextError("snapshot must be a dict")

    if "held_objects" not in snapshot:
        raise ContextError("missing snapshot.held_objects")
    held_objects = _require_str_list(
        snapshot["held_objects"], "snapshot.held_objects"
    )

    if "grasp_observation_complete" not in snapshot:
        raise ContextError("missing snapshot.grasp_observation_complete")
    observation_complete = snapshot["grasp_observation_complete"]
    if not isinstance(observation_complete, bool):
        raise ContextError(
            "snapshot.grasp_observation_complete must be an actual bool"
        )

    if "position" not in target:
        raise ContextError("missing target.position")
    position = _require_position(target["position"])

    if "orientation" not in target:
        raise ContextError("missing target.orientation")
    orientation_matrix = _validate_orientation(target["orientation"])
    orientation = tuple(
        tuple(float(value) for value in row) for row in orientation_matrix
    )

    if "dimensions" not in target:
        raise ContextError("missing target.dimensions")
    dimensions = _validate_dimensions(target["dimensions"])

    calibration_key = _require_calibration_key(target.get("calibration_key"))

    shape_family = classify_geometry(dimensions, orientation)

    operation = operation_from_goals(goal_tuple)
    if operation == "pick_place":
        if target_id is None or any(g[1] != object_id or g[2] != target_id for g in goal_tuple):
            raise ContextError("pick_place goals must match object_id and target_id")
    elif operation == "toggle":
        if any(g[1] != object_id for g in goal_tuple):
            raise ContextError("toggle goals must match object_id")

    return SubtaskContext(
        operation=operation,
        object_id=object_id,
        target_id=target_id,
        goals=goal_tuple,
        position=position,
        orientation=orientation,
        dimensions=dimensions,
        shape_family=shape_family,
        geometry_source=geometry_source,
        calibration_key=calibration_key,
        held_objects=held_objects,
        observation_complete=observation_complete,
    )


def _raise_goal_error():
    raise ContextError("each goal must be a list or tuple")


_FINITE_SHAPES = ("upright_elongated", "low_wide", "compact", "tilted")


def select_strategies(ctx: SubtaskContext) -> AssistSelection:
    if not isinstance(ctx, SubtaskContext):
        raise ContextError("ctx must be a SubtaskContext")

    if ctx.operation != "pick_place":
        return AssistSelection(
            prepare_strategy=None,
            grasp_strategy=None,
            status="vla_only",
            reason="operation_has_no_registered_assist",
            shape_family=ctx.shape_family,
        )

    if not isinstance(ctx.observation_complete, bool):
        raise ContextError("observation_complete must be an actual bool")

    if ctx.observation_complete is False:
        return AssistSelection(
            prepare_strategy=None,
            grasp_strategy=None,
            status="blocked",
            reason="grasp_observation_unknown",
            shape_family=ctx.shape_family,
        )

    if len(ctx.held_objects) > 0:
        if all(str(item) == ctx.object_id for item in ctx.held_objects):
            return AssistSelection(
                prepare_strategy=None,
                grasp_strategy=None,
                status="vla_only",
                reason="already_holding_target",
                shape_family=ctx.shape_family,
            )
        return AssistSelection(
            prepare_strategy=None,
            grasp_strategy=None,
            status="blocked",
            reason="foreign_object_held",
            shape_family=ctx.shape_family,
        )

    if ctx.shape_family not in _FINITE_SHAPES:
        raise ContextError("unknown shape_family: %r" % (ctx.shape_family,))

    prepare_strategy = "clearance_preposition_v1"

    if (
        ctx.shape_family == "upright_elongated"
        and ctx.calibration_key == "libero_wine_side_v1"
    ):
        return AssistSelection(
            prepare_strategy=prepare_strategy,
            grasp_strategy="calibrated_side_v1",
            status="selected",
            reason="matched_validated_geometry_profile",
            shape_family=ctx.shape_family,
        )

    return AssistSelection(
        prepare_strategy=prepare_strategy,
        grasp_strategy=None,
        status="selected",
        reason="grasp_strategy_unverified",
        shape_family=ctx.shape_family,
    )