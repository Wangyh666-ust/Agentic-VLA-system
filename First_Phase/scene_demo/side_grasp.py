#!/usr/bin/env python3
"""Demonstrated-side wine grasp: reuse the base machine with injected builders."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import local_grasp as base
import service

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "results/2026-10-08-skill-context/physical/trials/native_goal9_m0_a9270b6b/job/wine_telemetry.jsonl.gz"
)
_REFERENCE_SHA256 = "3b546bcf49a3b01f31799212ff509a30c8a73c6836373680d07584f2c4800c10"
_REFERENCE_STEP = 93

P_RELATIVE = np.array([-0.00299415138700734, 0.010524242443417493, 0.09384945135656642])
R_RELATIVE = np.array([
    [0.3217877768160636, -0.9463167125463995, -0.030615457650522],
    [-0.3520318440699384, -0.14959777039620167, 0.923955674182049],
    [-0.8789347003347261, -0.28654002625787023, -0.38127235134434445],
])


def _is_finite_vec3(value: Any) -> bool:
    return base._finite_vec3(value) is not None


def _is_finite_matrix3(value: Any) -> bool:
    return base._finite_matrix3(value) is not None


@lru_cache(maxsize=1)
def load_reference() -> dict:
    """Load and verify the frozen side-grasp reference (CPU-only)."""

    raw = _REFERENCE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _REFERENCE_SHA256:
        raise ValueError("reference sha mismatch")
    record = None
    with gzip.open(_REFERENCE_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if entry.get("step") == _REFERENCE_STEP:
                record = entry
                break
    if record is None:
        raise ValueError("reference step missing")
    snapshot = record.get("before_snapshot") or {}
    objects = snapshot.get("objects") or {}
    wine = objects.get(base.WINE_OBJECT_ID) or {}
    if wine.get("grasped") is not True:
        raise ValueError("reference step not grasped")
    state = record.get("policy_input_state")
    wine_xyz = wine.get("position")
    wine_wxyz = wine.get("quaternion")
    if state is None or wine_xyz is None or wine_wxyz is None:
        raise ValueError("reference state missing")
    state = np.asarray(state, dtype=np.float64)
    wine_xyz = np.asarray(wine_xyz, dtype=np.float64)
    if state.shape != (8,) or wine_xyz.shape != (3,):
        raise ValueError("reference state invalid")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(wine_xyz)):
        raise ValueError("reference state nonfinite")
    from scipy.spatial.transform import Rotation

    rbody = Rotation.from_rotvec(state[3:6]).as_matrix()
    rwine = base._quat_wxyz_to_matrix(wine_wxyz)
    if rbody is None or rwine is None:
        raise ValueError("reference rotation invalid")
    p_rel = rwine.T @ (state[:3] - wine_xyz)
    r_rel = rwine.T @ rbody
    if not np.allclose(p_rel, P_RELATIVE, atol=1e-8, rtol=0):
        raise ValueError("reference relative position mismatch")
    if not np.allclose(r_rel, R_RELATIVE, atol=1e-8, rtol=0):
        raise ValueError("reference relative orientation mismatch")
    return {
        "source_path": str(_REFERENCE_PATH),
        "source_sha256": _REFERENCE_SHA256,
        "sha256": _REFERENCE_SHA256,
        "step": _REFERENCE_STEP,
        "quaternion_order": "wxyz",
        "policy_orientation": "axis_angle_body",
        "relative_position": [float(v) for v in p_rel],
        "relative_orientation": [[float(v) for v in row] for row in r_rel],
    }


def reference_metadata() -> dict:
    """JSON-safe metadata for the frozen demonstrated guide."""

    return load_reference()


def read_geometry(env: Any) -> dict:
    """Base reading plus the measured body-to-controller rotation (fail closed)."""

    reading = base.read_geometry(env)
    inner = service._inner_env(env)
    if inner is None:
        raise ValueError("missing inner env")
    robots = getattr(inner, "robots", None)
    if not robots:
        raise ValueError("missing robot")
    robot = robots[0]
    robot_model = getattr(robot, "robot_model", None)
    body_name = getattr(robot_model, "eef_name", None)
    if body_name is None:
        raise ValueError("missing body")
    sim = getattr(inner, "sim", None)
    data = getattr(sim, "data", None)
    get_body_xmat = getattr(data, "get_body_xmat", None)
    if get_body_xmat is None:
        raise ValueError("missing body read")
    rbody = get_body_xmat(body_name)
    pose = reading.get("pose") if isinstance(reading, dict) else None
    rcontroller = pose.get("orientation_matrix") if isinstance(pose, dict) else None
    rbody = base._finite_matrix3(rbody)
    rcontroller = base._finite_matrix3(rcontroller)
    if rbody is None:
        raise ValueError("missing body rotation")
    if rcontroller is None:
        raise ValueError("missing controller orientation")
    reading["eef_body_rotation"] = rbody
    reading["body_to_controller_rotation"] = rbody.T @ rcontroller
    return reading


def make_target(reading: dict) -> tuple[np.ndarray, np.ndarray]:
    """Demonstrated relative side pose applied to the current wine frame."""

    snapshot = reading.get("snapshot") if isinstance(reading, dict) else None
    wine_position = base._snapshot_object_position(snapshot, base.WINE_OBJECT_ID)
    wine_rotation = base._finite_matrix3(reading.get("wine_rotation"))
    body_to_controller = base._finite_matrix3(reading.get("body_to_controller_rotation"))
    if wine_position is None or wine_rotation is None or body_to_controller is None:
        raise ValueError("missing relative side geometry")
    p = np.asarray(wine_position, dtype=np.float64).reshape(3) + wine_rotation @ P_RELATIVE
    r = wine_rotation @ R_RELATIVE @ body_to_controller
    return p, r


def make_approach_offset(reading: dict, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Retreat along the hand approach axis: -clearance * R[:, 2]."""

    return -base.ABOVE_CLEARANCE_M * np.asarray(orientation, dtype=np.float64).reshape(3, 3)[:, 2]


def _default_approach_offset(reading: dict, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    return base._default_approach_offset(reading, position, orientation) if hasattr(base, "_default_approach_offset") else -base.ABOVE_CLEARANCE_M * np.asarray(orientation, dtype=np.float64).reshape(3, 3)[:, 2]


class SideGraspController(base.LocalGraspController):
    """The base phase machine driven by the demonstrated-side builders."""

    def __init__(self) -> None:
        load_reference()
        super().__init__(target_builder=make_target, approach_offset_builder=make_approach_offset)

    def summary(self) -> dict:
        result = super().summary()
        reference = reference_metadata()
        result["grasp_mode"] = "demonstrated_side"
        result["reference"] = reference
        result["approach_offset_explanation"] = (
            "above_target = target - ABOVE_CLEARANCE_M * R[:, 2] (hand approach axis), "
            "not the legacy world +Z offset"
        )
        return result


LocalGraspController = SideGraspController


def __getattr__(name: str) -> Any:
    return getattr(base, name)
