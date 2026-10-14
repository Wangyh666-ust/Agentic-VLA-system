import copy
import math
from typing import Any, Optional

import numpy as np

import subtask_preparation as sp


class EntryError(sp.PreparationError):
    """Raised when an entry reference or target is invalid."""


# Diagnostic-only capabilities. These do not alter the baseline preparation
# controller, which keeps its original default of 200 real actions.
GENERIC_ACTION_CAP = 200
BRIDGE_ACTION_CAP = 100
TOTAL_PREPARE_CAP = 300
DIMENSION_RATIO_TOL = 0.10
GAP_TOL_M = 0.005

_ATOL = 1e-6
_GAP_MIN = 0.07
_DOWNWARD_ROW2 = -0.85
_DIM_RATIO_LO = 1.0 - DIMENSION_RATIO_TOL
_DIM_RATIO_HI = 1.0 + DIMENSION_RATIO_TOL
_SCHEMA_VERSION = 1


def _is_real_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, np.integer):
        return True
    if isinstance(value, np.floating):
        return True
    return False


def _finite_scalar(value: Any) -> Optional[float]:
    if not _is_real_number(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_vec3(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None

    try:
        raw = np.asarray(value, dtype=object)
    except Exception:
        return None

    if raw.shape != (3,):
        return None

    for item in raw.flat:
        if not _is_real_number(item):
            return None

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None

    if arr.shape != (3,):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _positive_vec3(value: Any) -> Optional[np.ndarray]:
    arr = _finite_vec3(value)
    if arr is None:
        return None
    if not np.all(arr > 0.0):
        return None
    return arr


def _finite_matrix3(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None

    try:
        raw = np.asarray(value, dtype=object)
    except Exception:
        return None

    if raw.shape != (3, 3):
        return None

    for item in raw.flat:
        if not _is_real_number(item):
            return None

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None

    if arr.shape != (3, 3):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _is_so3(matrix: Any) -> bool:
    arr = _finite_matrix3(matrix)
    if arr is None:
        return False
    if not np.allclose(arr.T @ arr, np.eye(3), rtol=0.0, atol=_ATOL):
        return False
    if abs(float(np.linalg.det(arr)) - 1.0) > _ATOL:
        return False
    return True


def _so3_or_error(value: Any, label: str) -> np.ndarray:
    arr = _finite_matrix3(value)
    if arr is None:
        raise EntryError(f"{label} must be a finite 3x3 matrix")
    if not _is_so3(arr):
        raise EntryError(f"{label} must be orthonormal with det +1")
    return arr


def _vec3_or_error(value: Any, label: str) -> np.ndarray:
    arr = _finite_vec3(value)
    if arr is None:
        raise EntryError(f"{label} must be a finite 3-vector")
    return arr


def _positive_vec3_or_error(value: Any, label: str) -> np.ndarray:
    arr = _positive_vec3(value)
    if arr is None:
        raise EntryError(f"{label} must be a positive finite 3-vector")
    return arr


def _scalar_or_error(value: Any, label: str) -> float:
    out = _finite_scalar(value)
    if out is None:
        raise EntryError(f"{label} must be finite")
    return out


def _validate_context(ctx: Any) -> dict:
    if ctx is None:
        raise EntryError("ctx is required")

    operation = getattr(ctx, "operation", None)
    if operation != "pick_place":
        raise EntryError("ctx.operation must be 'pick_place'")

    shape_family = getattr(ctx, "shape_family", None)
    if not isinstance(shape_family, str) or not shape_family:
        raise EntryError("ctx.shape_family must be a nonempty string")

    object_id = getattr(ctx, "object_id", None)
    if not isinstance(object_id, str) or not object_id:
        raise EntryError("ctx.object_id must be a nonempty string")

    position = _vec3_or_error(getattr(ctx, "position", None), "ctx.position")
    orientation = _so3_or_error(
        getattr(ctx, "orientation", None), "ctx.orientation"
    )
    dimensions = _positive_vec3_or_error(
        getattr(ctx, "dimensions", None), "ctx.dimensions"
    )

    return {
        "operation": operation,
        "shape_family": shape_family,
        "object_id": object_id,
        "position": position,
        "orientation": orientation,
        "dimensions": dimensions,
    }


def _validate_reading_common(reading: Any, ctx_info: dict) -> dict:
    if not isinstance(reading, dict):
        raise EntryError("reading must be a dict")

    target = reading.get("target")
    if not isinstance(target, dict):
        raise EntryError("reading.target must be a dict")

    target_position = _vec3_or_error(
        target.get("position"), "reading.target.position"
    )
    target_orientation = _so3_or_error(
        target.get("orientation"), "reading.target.orientation"
    )
    target_dimensions = _positive_vec3_or_error(
        target.get("dimensions"), "reading.target.dimensions"
    )

    if not np.allclose(
        target_position, ctx_info["position"], atol=_ATOL, rtol=0.0
    ):
        raise EntryError("reading.target.position disagrees with ctx")
    if not np.allclose(
        target_orientation, ctx_info["orientation"], atol=_ATOL, rtol=0.0
    ):
        raise EntryError("reading.target.orientation disagrees with ctx")
    if not np.allclose(
        target_dimensions, ctx_info["dimensions"], atol=_ATOL, rtol=0.0
    ):
        raise EntryError("reading.target.dimensions disagrees with ctx")

    pose = reading.get("pose")
    if not isinstance(pose, dict):
        raise EntryError("reading.pose must be a dict")
    pose_position = _vec3_or_error(pose.get("position"), "reading.pose.position")
    pose_orientation = _so3_or_error(
        pose.get("orientation_matrix"), "reading.pose.orientation_matrix"
    )

    snapshot = reading.get("snapshot")
    if not isinstance(snapshot, dict):
        raise EntryError("reading.snapshot must be a dict")
    if snapshot.get("grasp_observation_complete") is not True:
        raise EntryError("reading.snapshot.grasp_observation_complete not True")

    held = snapshot.get("held_objects")
    if not isinstance(held, list) or len(held) != 0:
        raise EntryError("reading.snapshot.held_objects must be exactly []")

    objects = snapshot.get("objects")
    if not isinstance(objects, dict) or len(objects) == 0:
        raise EntryError("reading.snapshot.objects must be a nonempty dict")

    object_id = ctx_info["object_id"]
    if object_id not in objects:
        raise EntryError(
            "reading.snapshot.objects must contain ctx.object_id"
        )

    for name, obj in objects.items():
        if not isinstance(obj, dict):
            raise EntryError(f"snapshot object {name} must be a dict")
        if obj.get("grasped") is not False:
            raise EntryError(f"snapshot object {name}.grasped must be False")
        _vec3_or_error(obj.get("position"), f"snapshot object {name}.position")

    target_obj = objects[object_id]
    if not isinstance(target_obj, dict):
        raise EntryError(
            f"snapshot object {object_id} must be a dict"
        )
    target_obj_position = _vec3_or_error(
        target_obj.get("position"),
        f"snapshot object {object_id}.position",
    )
    if not np.allclose(
        target_obj_position, ctx_info["position"], atol=_ATOL, rtol=0.0
    ):
        raise EntryError(
            "snapshot target object position disagrees with ctx.position"
        )

    contacts = reading.get("robot_contacts")
    if not isinstance(contacts, list) or len(contacts) != 0:
        raise EntryError("reading.robot_contacts must be exactly []")

    gap = _scalar_or_error(reading.get("gripper_gap_m"), "reading.gripper_gap_m")
    if gap < _GAP_MIN - _ATOL:
        raise EntryError("reading.gripper_gap_m below 0.07")

    return {
        "target_position": target_position,
        "target_orientation": target_orientation,
        "target_dimensions": target_dimensions,
        "pose_position": pose_position,
        "pose_orientation": pose_orientation,
        "snapshot": snapshot,
        "gripper_gap_m": gap,
    }


def _validate_reference_value(reference: Any) -> dict:
    if not isinstance(reference, dict):
        raise EntryError("reference must be a dict")

    schema_version = reference.get("schema_version")
    if type(schema_version) is not int or schema_version != _SCHEMA_VERSION:
        raise EntryError("reference.schema_version mismatch")

    operation = reference.get("operation")
    if operation != "pick_place":
        raise EntryError("reference.operation must be 'pick_place'")

    shape_family = reference.get("shape_family")
    if not isinstance(shape_family, str) or not shape_family:
        raise EntryError("reference.shape_family must be a nonempty string")

    provenance = reference.get("provenance")
    if not isinstance(provenance, dict) or len(provenance) == 0:
        raise EntryError("reference.provenance must be a nonempty dict")

    ref_dims = _positive_vec3_or_error(
        reference.get("reference_dimensions"), "reference.reference_dimensions"
    )
    pos_norm = _vec3_or_error(
        reference.get("position_normalized"), "reference.position_normalized"
    )
    orient_rel = _so3_or_error(
        reference.get("orientation_relative"), "reference.orientation_relative"
    )
    gap = _scalar_or_error(
        reference.get("gripper_gap_m"), "reference.gripper_gap_m"
    )
    if gap < _GAP_MIN - _ATOL:
        raise EntryError("reference.gripper_gap_m below 0.07")

    return {
        "schema_version": _SCHEMA_VERSION,
        "operation": operation,
        "shape_family": shape_family,
        "reference_dimensions": ref_dims,
        "position_normalized": pos_norm,
        "orientation_relative": orient_rel,
        "gripper_gap_m": gap,
    }


def make_entry_reference(ctx: Any, reading: dict, *, provenance: dict) -> dict:
    """Build an isolated object-relative entry reference (diagnostic only)."""

    if not isinstance(provenance, dict) or len(provenance) == 0:
        raise EntryError("provenance must be a nonempty dict")

    ctx_info = _validate_context(ctx)
    common = _validate_reading_common(reading, ctx_info)

    object_position = ctx_info["position"]
    object_orientation = ctx_info["orientation"]
    dimensions = ctx_info["dimensions"]

    eef_position = common["pose_position"]
    eef_orientation = common["pose_orientation"]

    if float(eef_orientation[2, 2]) >= _DOWNWARD_ROW2:
        raise EntryError(
            "reading.pose.orientation_matrix R[2,2] must be < -0.85"
        )

    delta_world = eef_position - object_position
    position_normalized = object_orientation.T @ delta_world
    position_normalized = position_normalized / dimensions

    orientation_relative = object_orientation.T @ eef_orientation

    provenance_copy = copy.deepcopy(provenance)

    reference = {
        "schema_version": _SCHEMA_VERSION,
        "operation": ctx_info["operation"],
        "shape_family": ctx_info["shape_family"],
        "reference_dimensions": [float(v) for v in dimensions],
        "position_normalized": [float(v) for v in position_normalized],
        "orientation_relative": [
            [float(v) for v in row] for row in orientation_relative
        ],
        "gripper_gap_m": float(common["gripper_gap_m"]),
        "provenance": provenance_copy,
    }
    return reference


def entry_target(reference: dict, ctx: Any, reading: dict) -> dict:
    """Resolve an object-relative entry target from a validated reference."""

    ref = _validate_reference_value(reference)
    ctx_info = _validate_context(ctx)
    common = _validate_reading_common(reading, ctx_info)

    if ctx_info["operation"] != ref["operation"]:
        raise EntryError("ctx.operation does not match reference.operation")
    if ctx_info["shape_family"] != ref["shape_family"]:
        raise EntryError("ctx.shape_family does not match reference.shape_family")

    current_dims = ctx_info["dimensions"]
    ref_dims = ref["reference_dimensions"]
    ratio = current_dims / ref_dims
    if not np.all(ratio >= _DIM_RATIO_LO - _ATOL) or not np.all(
        ratio <= _DIM_RATIO_HI + _ATOL
    ):
        raise EntryError("current dimensions out of tolerance vs reference")

    current_gap = common["gripper_gap_m"]
    if abs(current_gap - ref["gripper_gap_m"]) > GAP_TOL_M + _ATOL:
        raise EntryError("gripper gap out of tolerance vs reference")

    object_position = ctx_info["position"]
    object_orientation = ctx_info["orientation"]

    position = object_position + object_orientation @ (
        ref["position_normalized"] * current_dims
    )
    orientation = object_orientation @ ref["orientation_relative"]

    if float(orientation[2, 2]) >= _DOWNWARD_ROW2:
        raise EntryError("entry orientation no longer points sufficiently downward")

    if not sp._workspace_ok(position):
        raise EntryError("entry target outside workspace")

    return {
        "position": [float(v) for v in position],
        "orientation": [[float(v) for v in row] for row in orientation],
        "gripper_gap_m": float(current_gap),
        "geometry_compatibility": "diagnostic_reference_only",
    }


class EntryPreparationController(sp.PreparationController):
    """Diagnostic-only controller adding an isolated entry bridge."""

    def __init__(self, ctx: Any, reading: dict, reference: dict) -> None:
        super().__init__(ctx, reading, max_actions=GENERIC_ACTION_CAP)

        self._entry_reference = copy.deepcopy(reference)

        original_plan = copy.deepcopy(self._plan)
        original_waypoints = original_plan.get("waypoints")
        if not isinstance(original_waypoints, list) or not original_waypoints:
            raise EntryError("parent plan has no ready waypoints")
        original_n = len(original_waypoints)

        target = entry_target(self._entry_reference, ctx, reading)
        entry_position = np.asarray(target["position"], dtype=np.float64)
        entry_orientation = np.asarray(target["orientation"], dtype=np.float64)

        ready_waypoint = original_waypoints[original_n - 1]
        ready_position = np.asarray(
            ready_waypoint["position"], dtype=np.float64
        )

        align_waypoint = {
            "name": "entry_align",
            "position": [float(v) for v in ready_position],
            "orientation": [
                [float(v) for v in row] for row in entry_orientation
            ],
        }
        ready_entry_waypoint = {
            "name": "entry_ready",
            "position": [float(v) for v in entry_position],
            "orientation": [
                [float(v) for v in row] for row in entry_orientation
            ],
        }

        new_waypoints = list(original_waypoints) + [
            align_waypoint,
            ready_entry_waypoint,
        ]

        current_position, current_orientation = sp._reading_pose(reading)
        ok, reason, samples = sp._check_route(
            reading,
            current_position,
            current_orientation,
            current_position,
            current_orientation,
            new_waypoints,
        )
        if not ok:
            raise EntryError(f"entry bridge route rejected: {reason}")

        new_plan = original_plan
        new_plan["waypoints"] = new_waypoints
        new_plan["check_samples"] = int(samples)
        new_plan["entry_reference"] = copy.deepcopy(reference)
        new_plan["entry_target"] = target
        new_plan["entry_original_waypoint_count"] = int(original_n)

        self._plan = new_plan
        self._max_actions = TOTAL_PREPARE_CAP
        self._original_waypoint_count = int(original_n)
        self._bridge_start_actions: Optional[int] = None

    @property
    def _bridge_actions(self) -> int:
        if self._bridge_start_actions is None:
            return 0
        return int(self._actual_aux_actions) - int(self._bridge_start_actions)

    def _generic_actions(self) -> int:
        if self._bridge_start_actions is None:
            return int(self._actual_aux_actions)
        return int(self._bridge_start_actions)

    def _ensure_bridge_started(self) -> None:
        if self._bridge_start_actions is None:
            self._bridge_start_actions = int(self._actual_aux_actions)

    def _is_bridge_phase(self) -> bool:
        return self._waypoint_index >= self._original_waypoint_count

    def _generic_ready(self) -> bool:
        if self._waypoint_index >= self._original_waypoint_count:
            return True
        return False

    def _generic_failed(self) -> bool:
        return self._phase == "failed" and self._reason in (
            "prepare_budget_exhausted",
            "generic_prepare_budget_exhausted",
        )

    def _check_budgets(self) -> bool:
        if self._phase != "preparing":
            return False

        if not self._is_bridge_phase():
            if self._actual_aux_actions >= GENERIC_ACTION_CAP:
                self._phase = "failed"
                self._reason = "generic_prepare_budget_exhausted"
                return False
            return True

        self._ensure_bridge_started()
        if self._bridge_actions >= BRIDGE_ACTION_CAP:
            self._phase = "failed"
            self._reason = "entry_bridge_budget_exhausted"
            return False
        if self._actual_aux_actions >= TOTAL_PREPARE_CAP:
            self._phase = "failed"
            self._reason = "entry_bridge_budget_exhausted"
            return False
        return True

    def next_action(self, reading: Any) -> Optional[np.ndarray]:
        if self._phase != "preparing":
            return None

        if self._is_bridge_phase():
            self._ensure_bridge_started()

        if not self._check_budgets():
            return None

        result = super().next_action(reading)

        if self._phase == "failed" and self._reason == "prepare_budget_exhausted":
            if not self._is_bridge_phase():
                self._reason = "generic_prepare_budget_exhausted"
            else:
                self._reason = "entry_bridge_budget_exhausted"

        return result

    def observe_after(self, reading: Any) -> None:
        if self._phase != "preparing":
            return

        if self._is_bridge_phase():
            self._ensure_bridge_started()

        prebudget = self._check_budgets()
        if not prebudget:
            return

        was_preparing = self._phase == "preparing"
        super().observe_after(reading)

        if was_preparing and self._phase == "preparing" and self._is_bridge_phase():
            self._ensure_bridge_started()

        if self._phase == "failed" and self._reason == "prepare_budget_exhausted":
            if not self._is_bridge_phase():
                self._reason = "generic_prepare_budget_exhausted"
            else:
                self._reason = "entry_bridge_budget_exhausted"

        if not self._check_budgets():
            return

    def summary(self) -> dict:
        base = super().summary()
        base["entry_reference"] = copy.deepcopy(self._entry_reference)
        base["entry_target"] = copy.deepcopy(self._plan.get("entry_target"))
        base["generic_action_cap"] = int(GENERIC_ACTION_CAP)
        base["bridge_action_cap"] = int(BRIDGE_ACTION_CAP)
        base["total_prepare_cap"] = int(TOTAL_PREPARE_CAP)
        base["generic_actions"] = int(self._generic_actions())
        base["bridge_actions"] = int(self._bridge_actions)
        base["generic_ready"] = bool(self._generic_ready())
        return base
