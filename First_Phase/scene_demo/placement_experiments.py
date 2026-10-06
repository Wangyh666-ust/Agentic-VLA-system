#!/usr/bin/env python3
"""Isolated placement diagnostic experiment runner (read-only service host).

This module hosts ``DiagnosticService`` -- a subclass of the *actual*
``scene_demo/service.py`` ``SceneService`` -- and a small campaign runner that
drives the persistent-scene SmolVLA service through a fixed grid of precision /
denoising / action-chunk profiles and placement conditions.

Nothing here retrains, downloads, resets the environment between subgoals, or
calls Hermes.  The runner reuses the base service lifecycle verbatim (worker
started once, model loaded once with ``strict=True``, native termination /
``events.jsonl`` / PNG / MP4 logic untouched) and only adds two isolated
diagnostic seams:

* it wraps *only* the worker-owned ``env.step`` of one capability so that every
  action is followed by a contact/kinematics ``capture_snapshot`` written to a
  per-job ``telemetry.jsonl`` (the original action and the original step call
  are passed through unchanged, and ``env.step`` is restored in ``finally``);
* it overrides ``_select_action`` so the profile's precision / denoising
  iterations / action-chunk length are applied to the resident policy config
  (both ``policy.config`` and ``policy.model.config``) and the policy latency is
  measured.

Grasp-proxy limitation
======================

The grasp signal used by ``capture_snapshot`` is the robosuite **contact-geom
proxy** ``inner._check_grasp(gripper, obj.contact_geoms)``: it reports whether
the gripper contact geoms touch the object's contact geoms.  It is *not* a
tactile or force measurement.  It can be ``True`` while the object is merely
brushing past the fingers and ``False`` while a stable pinch is held through a
non-contact geom, and it is undefined when the simulator does not expose the
probe at all.  A ``None`` grasp is therefore never treated as success: an unknown
grasp or an unknown velocity makes ``strict_candidate`` ``None`` ("unknown"),
and unknown is never equal to success.

The *final* success of a trial is scored from a literal, independent oracle goal
set (``FINAL_ORACLE_GOALS``) that is fixed per condition and is never derived
from the capability schedule that was actually submitted or executed.

Only stdlib + numpy are imported at module import time; ``torch`` is imported
lazily inside ``_select_action`` so ``--help`` and the GPU-free unit tests never
initialise CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types
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
import service  # noqa: E402

EXPERIMENT_NAME = "placement_experiments"
CONTROL_FREQUENCY_HZ = 20
ACTION_DIM = 7
IMAGE_SIZE = 256
CHUNK_SIZE = 50
SOURCE_REVISION_EXPECTED = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"

# Velocities below these magnitudes count as "released and at rest".
LINEAR_SPEED_TOLERANCE = 0.02  # m/s
ANGULAR_SPEED_TOLERANCE = 0.2  # rad/s
STRICT_STREAK = 5

PROXY_LIMITATION = (
    "grasp is inferred from robosuite contact geoms via inner._check_grasp, a "
    "contact-based proxy, not a tactile/force measurement; it can be wrong at "
    "contact transitions and is unknown when the probe is unavailable. Unknown "
    "grasp or velocity is never treated as success."
)

# --- immutable configuration -------------------------------------------------


def _freeze(value: Any) -> Any:
    """Recursively freeze a config literal into an immutable structure.

    Dicts become ``MappingProxyType`` and sequences become tuples, so neither a
    test nor a caller can mutate the exported ``PROFILES`` / ``CONDITIONS``
    tables in place.
    """

    if isinstance(value, dict):
        return types.MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


# amp=True -> bfloat16 autocast; amp=False -> plain fp32 (autocast disabled).
# num_steps  -> flow-matching denoising iterations per action prediction.
# n_action_steps -> actions executed before a new prediction is made.
PROFILES = _freeze(
    {
        "baseline_bf16": {"amp": True, "num_steps": 10, "n_action_steps": 1},
        "fp32": {"amp": False, "num_steps": 10, "n_action_steps": 1},
        "fp32_d1": {"amp": False, "num_steps": 1, "n_action_steps": 1},
        "fp32_h5": {"amp": False, "num_steps": 10, "n_action_steps": 5},
        "fp32_d1_h5": {"amp": False, "num_steps": 1, "n_action_steps": 5},
    }
)

CONDITIONS = _freeze(
    {
        "soup_fresh": {
            "scene_id": "basket_two",
            "capability_ids": ["soup_to_basket"],
            "budget_per_subgoal": 300,
            "audit": False,
            "mode": "single",
            "native_instruction": False,
        },
        "sauce_fresh": {
            "scene_id": "basket_two",
            "capability_ids": ["sauce_to_basket"],
            "budget_per_subgoal": 300,
            "audit": False,
            "mode": "single",
            "native_instruction": False,
        },
        "basket_native": {
            "scene_id": "basket_two",
            "capability_ids": ["basket_both"],
            "budget_per_subgoal": 600,
            "audit": True,
            "mode": "single",
            # Temporarily replace ONLY the basket_both instruction with the native
            # env.task_description for the isolated process, restored in finally.
            "native_instruction": True,
        },
        "basket_split": {
            "scene_id": "basket_two",
            "capability_ids": ["soup_to_basket", "sauce_to_basket"],
            "budget_per_subgoal": 300,
            "audit": False,
            # ONE plan; the base plan loop aborts on the first failed subgoal.
            "mode": "single",
            "native_instruction": False,
        },
        "basket_forced_handoff": {
            "scene_id": "basket_two",
            "capability_ids": ["soup_to_basket"],
            "budget_per_subgoal": 300,
            "audit": False,
            # Deliberately experimental: a SECOND plan in the SAME session even
            # after the first failed or timed out. Never a production default.
            "mode": "forced_handoff",
            "forced_handoff_capability_ids": ["sauce_to_basket"],
            "forced_handoff_budget_per_subgoal": 300,
            "deliberately_forced_after_failure": True,
            "native_instruction": False,
        },
        "bowl_control": {
            "scene_id": "goal_table",
            "capability_ids": ["bowl_to_plate"],
            "budget_per_subgoal": 300,
            "audit": False,
            "mode": "single",
            "native_instruction": False,
        },
        "soup_extended": {
            "scene_id": "basket_two",
            "capability_ids": ["soup_to_basket"],
            "budget_per_subgoal": 600,
            "audit": False,
            "mode": "single",
            "native_instruction": False,
        },
        "soup_retry": {
            "scene_id": "basket_two",
            "capability_ids": ["soup_to_basket"],
            "budget_per_subgoal": 300,
            "audit": False,
            # Fixed same-goal continuation: a SECOND plan in the SAME session
            # only when the first plan blocks without an operational timeout.
            "mode": "same_goal_retry",
            "native_instruction": False,
        },
    }
)

# Literal, independent FINAL oracle goals.  These are fixed here and are never
# derived from the capability schedule a condition submits or executes.
_SOUP_GOAL = ["in", "alphabet_soup_1", "basket_1_contain_region"]
_SAUCE_GOAL = ["in", "tomato_sauce_1", "basket_1_contain_region"]
_BOWL_GOAL = ["on", "akita_black_bowl_1", "plate_1"]

FINAL_ORACLE_GOALS: dict[str, list[list[str]]] = {
    "soup_fresh": [list(_SOUP_GOAL)],
    "sauce_fresh": [list(_SAUCE_GOAL)],
    "basket_native": [list(_SOUP_GOAL), list(_SAUCE_GOAL)],
    "basket_split": [list(_SOUP_GOAL), list(_SAUCE_GOAL)],
    "basket_forced_handoff": [list(_SOUP_GOAL), list(_SAUCE_GOAL)],
    "bowl_control": [list(_BOWL_GOAL)],
    # Literally written, independent oracle goals for the fixed continuation
    # conditions; they are NEVER derived from the submitted capability schedule.
    "soup_extended": [["in", "alphabet_soup_1", "basket_1_contain_region"]],
    "soup_retry": [["in", "alphabet_soup_1", "basket_1_contain_region"]],
}

# --- small helpers -----------------------------------------------------------


def _err(reason: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "reason": reason, "detail": detail}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON via a ``.tmp`` file then ``os.replace``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _git_rev_parse(cwd: Path | None = None) -> str | None:
    """Read the source git SHA via ``git rev-parse`` (read-only, best-effort)."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else str(_HERE),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:  # noqa: BLE001 - a missing git binary is not fatal
        return None
    if completed.returncode != 0:
        return None
    text = (completed.stdout or "").strip()
    return text or None


def _native_task_instruction(env: Any) -> str | None:
    """Return the native task language, preferring ``env.task_description``.

    Defensive across the native attribute spellings: the official LIBERO env
    exposes the task language as ``task_description``; older robosuite-style
    wrappers keep it on a ``task.language`` attribute.  Nothing is invented: an
    unavailable instruction returns ``None`` and the caller records the miss.
    """

    def _text(candidate: Any) -> str | None:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return None

    for attr in ("task_description", "_task_description"):
        found = _text(getattr(env, attr, None))
        if found:
            return found
    task = getattr(env, "task", None)
    if task is not None:
        found = _text(getattr(task, "language", None))
        if found:
            return found
    try:
        inner = service._inner_env(env)
    except Exception:  # noqa: BLE001
        inner = None
    if inner is not None:
        for attr in ("task_description", "language"):
            found = _text(getattr(inner, attr, None))
            if found:
                return found
        inner_task = getattr(inner, "task", None)
        if inner_task is not None:
            found = _text(getattr(inner_task, "language", None))
            if found:
                return found
    return None


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the *exact bytes* of the campaign source files."""

    digests: dict[str, str | None] = {}
    for name in (
        "placement_experiments.py",
        "service.py",
        "catalog.py",
        "placement_completion.py",
    ):
        try:
            digests[name] = hashlib.sha256((_HERE / name).read_bytes()).hexdigest()
        except Exception:  # noqa: BLE001 - an absent/unreadable file is recorded as unknown
            digests[name] = None
    return digests


def _package_versions() -> dict[str, str | None]:
    """Best-effort installed versions; a missing distribution is ``None``.

    Measured, never asserted: a missing package is reported as unknown and is
    never turned into a version-based explanation of a run failure.
    """

    versions: dict[str, str | None] = {}
    try:
        from importlib import metadata as importlib_metadata
    except Exception:  # noqa: BLE001 - importlib.metadata always exists on 3.8+
        importlib_metadata = None  # type: ignore[assignment]
    for name in ("torch", "lerobot", "libero", "robosuite", "mujoco", "transformers"):
        if importlib_metadata is None:
            versions[name] = None
            continue
        try:
            versions[name] = importlib_metadata.version(name)
        except Exception:  # noqa: BLE001 - distribution not installed
            versions[name] = None
    return versions


def _finite_vec(values: Any, expected: int = 3) -> list[float] | None:
    """Return a finite float vector of exactly ``expected`` components or None."""

    if values is None:
        return None
    try:
        vector = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(vector) != expected:
        return None
    if any(not math.isfinite(v) for v in vector):
        return None
    return vector


def _speed(vector: Any) -> float | None:
    """Euclidean magnitude of a 3-vector, or None when malformed/nonfinite."""

    finite = _finite_vec(vector, 3)
    if finite is None:
        return None
    return math.sqrt(sum(v * v for v in finite))


# --- offline diagnostics -----------------------------------------------------


def capture_snapshot(env: Any, goals: Any) -> dict[str, Any]:
    """Read contact/kinematic diagnostics from the live simulator (read-only).

    Never supplies privileged data to the policy or Hermes: this is a diagnostic
    reader only.  Every field is either a measured value or an explicit
    ``None`` (unknown); nothing is fabricated.
    """

    goal_list = [list(goal) for goal in (goals or [])]

    predicates: dict[str, bool | None] = {}
    predicate_unknown = False
    for goal in goal_list:
        key = catalog.goal_key(goal)
        try:
            predicates[key] = bool(service.eval_goal_predicate(env, goal))
        except Exception:  # noqa: BLE001 - unknown truth is never success
            predicates[key] = None
            predicate_unknown = True

    try:
        inner = service._inner_env(env)
    except Exception as exc:  # noqa: BLE001
        return {
            "predicates": predicates,
            "objects": {},
            "held_objects": [],
            "gripper_qpos": None,
            "eef_position": None,
            "grasp_observation_complete": False,
            "strict_candidate": None,
            "phases": {catalog.goal_key(g): "unknown" for g in goal_list},
            "goal_objects": [g[1] for g in goal_list if len(g) >= 2],
            "error": "inner env unavailable: %s" % exc,
        }

    sim = getattr(inner, "sim", None)
    data = getattr(sim, "data", None)
    robots = getattr(inner, "robots", None) or []
    robot = robots[0] if robots else None
    gripper = getattr(robot, "gripper", None) if robot is not None else None

    objects = getattr(inner, "objects_dict", None)
    if not isinstance(objects, dict):
        objects = {}

    entries: dict[str, dict[str, Any]] = {}
    held_objects: list[str] = []
    grasp_unknown = False

    for object_id, obj in objects.items():
        joints = getattr(obj, "joints", None) or []
        if not joints or not isinstance(joints[0], str):
            continue
        joint_name = joints[0]
        entry: dict[str, Any] = {
            "position": None,
            "quaternion": None,
            "linear_velocity": None,
            "angular_velocity": None,
            "grasped": None,
            "error": None,
        }

        # --- geometry (position/quaternion): nonfinite qpos is unknown ---
        qpos = None
        try:
            raw_qpos = np.asarray(data.get_joint_qpos(joint_name), dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - qpos unavailable
            entry["error"] = "qpos unavailable: %s" % exc
        else:
            if not np.all(np.isfinite(raw_qpos)):
                entry["error"] = "nonfinite qpos"
            else:
                qpos = raw_qpos

        if qpos is not None and qpos.size >= 7:
            # A free joint: 3 position + 4 quaternion (and 6 velocity) DOF.
            entry["position"] = [float(v) for v in qpos[:3]]
            entry["quaternion"] = [float(v) for v in qpos[3:7]]
        elif qpos is not None and qpos.size >= 3:
            entry["position"] = [float(v) for v in qpos[:3]]

        # --- velocity: nonfinite/malformed components are unknown ---
        qvel = None
        try:
            qvel = np.asarray(data.get_joint_qvel(joint_name), dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001 - velocity unsupported -> nullable
            qvel = None
        if qvel is not None and not np.all(np.isfinite(qvel)):
            entry["error"] = entry["error"] or "nonfinite velocity"
            qvel = None
        if qvel is not None and qvel.size >= 6:
            entry["linear_velocity"] = _finite_vec(qvel[:3])
            entry["angular_velocity"] = _finite_vec(qvel[3:6])
        elif qvel is not None and qvel.size >= 3:
            entry["linear_velocity"] = _finite_vec(qvel[:3])

        # --- grasp: probed for EVERY jointed object, even when qpos failed ---
        # An unknown grasp of *any* jointed object (goal or not) makes the
        # strict observation incomplete; it is never silently skipped.
        grasp: bool | None = None
        if gripper is not None:
            try:
                grasp = bool(inner._check_grasp(gripper, getattr(obj, "contact_geoms", None)))
            except Exception:  # noqa: BLE001 - probe unavailable -> unknown
                grasp = None
        entry["grasped"] = grasp
        if grasp is None:
            grasp_unknown = True
        elif grasp:
            held_objects.append(object_id)

        entries[object_id] = entry

    gripper_qpos: list[float] | None = None
    if robot is not None and data is not None:
        indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
        if indexes is not None:
            try:
                full_qpos = np.asarray(sim.data.qpos, dtype=np.float64).reshape(-1)
                gripper_qpos = [float(full_qpos[int(i)]) for i in indexes]
            except Exception:  # noqa: BLE001
                gripper_qpos = None

    eef_position: list[float] | None = None
    if gripper is not None and data is not None:
        important_sites = getattr(gripper, "important_sites", None)
        site_name = important_sites.get("grip_site") if isinstance(important_sites, dict) else None
        if site_name is not None:
            try:
                eef_position = [
                    float(v)
                    for v in np.asarray(data.get_site_xpos(site_name), dtype=np.float64)[:3]
                ]
            except Exception:  # noqa: BLE001
                eef_position = None

    required_objects = [goal[1] for goal in goal_list if len(goal) >= 2]
    required_missing = any(obj not in entries for obj in required_objects)
    required_grasp_unknown = any(
        entries.get(obj, {}).get("grasped") is None for obj in required_objects
    )
    # A required goal object whose qpos could not be read (or was nonfinite) has
    # null geometry: its position is unknown, so success cannot be claimed.
    required_geometry_unknown = any(
        bool(entries.get(obj, {}).get("error")) for obj in required_objects
    )
    # Both linear AND angular velocity must be known (finite, correct length)
    # for every required goal object; a missing either way is unknown.
    required_velocity_unknown = any(
        _speed(entries.get(obj, {}).get("linear_velocity")) is None
        or _speed(entries.get(obj, {}).get("angular_velocity")) is None
        for obj in required_objects
    )

    if (
        predicate_unknown
        or grasp_unknown
        or required_missing
        or required_grasp_unknown
        or required_geometry_unknown
        or required_velocity_unknown
    ):
        strict_candidate: bool | None = None
    else:
        all_goals_true = bool(predicates) and all(value is True for value in predicates.values())
        no_held = not held_objects
        speeds_ok = True
        for obj in required_objects:
            entry = entries.get(obj) or {}
            linear = _speed(entry.get("linear_velocity"))
            angular = _speed(entry.get("angular_velocity"))
            if linear is None or angular is None:
                speeds_ok = False
                break
            if linear > LINEAR_SPEED_TOLERANCE or angular > ANGULAR_SPEED_TOLERANCE:
                speeds_ok = False
                break
        strict_candidate = bool(all_goals_true and no_held and speeds_ok)

    # Phases are computed PER goal from that goal's own predicate, grasp and
    # physics: another goal being incomplete can never relabel a stable,
    # released goal as unsettled.
    phases: dict[str, str] = {}
    for goal in goal_list:
        key = catalog.goal_key(goal)
        predicate = predicates.get(key)
        obj = goal[1] if len(goal) >= 2 else None
        entry = entries.get(obj) if obj is not None else None
        grasp = entry.get("grasped") if isinstance(entry, dict) else None
        if predicate is None:
            phase = "unknown"
        elif entry is None or grasp is None:
            phase = "unknown"
        elif predicate is True and grasp is True:
            phase = "goal_held"
        elif grasp is True and predicate is False:
            phase = "transporting"
        elif predicate is True and grasp is False:
            linear = _speed(entry.get("linear_velocity"))
            angular = _speed(entry.get("angular_velocity"))
            settled = (
                linear is not None
                and angular is not None
                and linear <= LINEAR_SPEED_TOLERANCE
                and angular <= ANGULAR_SPEED_TOLERANCE
            )
            phase = "released_stable" if settled else "released_unsettled"
        else:
            phase = "ungrasped"
        phases[key] = phase

    return {
        "predicates": predicates,
        "objects": entries,
        "held_objects": held_objects,
        "gripper_qpos": gripper_qpos,
        "eef_position": eef_position,
        "grasp_observation_complete": not grasp_unknown,
        "strict_candidate": strict_candidate,
        "phases": phases,
        "goal_objects": required_objects,
    }


def summarize_samples(samples: Any) -> dict[str, Any]:
    """Aggregate a telemetry sample list into streak / count diagnostics.

    A sample is a mapping with ``snapshot`` (carrying ``strict_candidate``,
    ``objects``, ``predicates``), an optional ``native_success`` and an optional
    ``error``.  Unknown values are counted as nulls, never as success, and a
    five-sample strict streak is required for either success flag.
    """

    sample_list = list(samples or [])
    strict_values: list[Any] = []
    native_values: list[Any] = []
    telemetry_errors = 0
    telemetry_nulls = 0

    for sample in sample_list:
        snapshot = sample.get("snapshot") or {}
        strict_values.append(snapshot.get("strict_candidate"))
        native_values.append(sample.get("native_success"))
        if sample.get("error") or snapshot.get("error"):
            telemetry_errors += 1
        if snapshot.get("strict_candidate") is None:
            telemetry_nulls += 1
        for predicate_value in (snapshot.get("predicates") or {}).values():
            if predicate_value is None:
                telemetry_nulls += 1
        for entry in (snapshot.get("objects") or {}).values():
            if entry.get("grasped") is None:
                telemetry_nulls += 1
            if entry.get("linear_velocity") is None:
                telemetry_nulls += 1
            if entry.get("angular_velocity") is None:
                telemetry_nulls += 1
            if entry.get("error"):
                telemetry_errors += 1

    max_streak = 0
    run = 0
    for value in strict_values:
        if value is True:
            run += 1
            max_streak = max(max_streak, run)
        else:
            run = 0

    strict_success_ever = max_streak >= STRICT_STREAK
    strict_success_final = len(strict_values) >= STRICT_STREAK and all(
        value is True for value in strict_values[-STRICT_STREAK:]
    )

    per_object_grasp_counts: dict[str, dict[str, int]] = {}
    phase_counts_last20: dict[str, int] = {}
    for sample in sample_list[-20:]:
        snapshot = sample.get("snapshot") or {}
        for object_id, entry in (snapshot.get("objects") or {}).items():
            counts = per_object_grasp_counts.setdefault(
                object_id, {"true": 0, "false": 0, "null": 0}
            )
            grasped = entry.get("grasped")
            if grasped is True:
                counts["true"] += 1
            elif grasped is False:
                counts["false"] += 1
            else:
                counts["null"] += 1
        for phase in (snapshot.get("phases") or {}).values():
            phase_counts_last20[phase] = phase_counts_last20.get(phase, 0) + 1

    return {
        "n_samples": len(sample_list),
        "max_strict_streak": max_streak,
        "strict_success_ever": bool(strict_success_ever),
        "strict_success_final": bool(strict_success_final),
        "native_success_ever": any(value is True for value in native_values),
        "native_success_final": native_values[-1] if native_values else None,
        "n_strict_true": sum(1 for value in strict_values if value is True),
        "n_strict_false": sum(1 for value in strict_values if value is False),
        "n_strict_unknown": sum(1 for value in strict_values if value is None),
        "telemetry_errors": telemetry_errors,
        "telemetry_nulls": telemetry_nulls,
        "per_object_grasp_counts": per_object_grasp_counts,
        "phase_counts_last20": phase_counts_last20,
        "final_snapshot": sample_list[-1].get("snapshot") if sample_list else None,
    }


def _oracle_sample_strict(sample: Any) -> Any:
    """The ``strict_candidate`` of a fixed-oracle snapshot or sample mapping."""

    if not isinstance(sample, dict):
        return None
    if "strict_candidate" in sample:
        return sample.get("strict_candidate")
    snapshot = sample.get("snapshot")
    if isinstance(snapshot, dict):
        return snapshot.get("strict_candidate")
    return None


def strict_final_score(
    final_snapshot: dict | None, oracle_samples: list[dict]
) -> bool | None:
    """Score the FIXED literal-oracle final state of one condition.

    The score is deliberately independent of the capability schedule that was
    submitted: ``final_snapshot`` is the fixed literal-goal snapshot taken after
    the final job and ``oracle_samples`` are the *fixed-goal* oracle snapshots
    recorded during the final job's actions (never the declared-goal summaries).

    * a missing final snapshot, or an unknown ``strict_candidate`` in the final
      snapshot, scores Unknown;
    * an unknown value anywhere in the required final window of ``STRICT_STREAK``
      consecutive samples scores Unknown;
    * a known False (in the final snapshot or the window) or fewer than
      ``STRICT_STREAK`` samples scores False -- never success;
    * only a True final snapshot AND every one of the last ``STRICT_STREAK``
      samples being True scores True.
    """

    if not isinstance(final_snapshot, dict) or "strict_candidate" not in final_snapshot:
        return None
    final_strict = final_snapshot.get("strict_candidate")
    if final_strict is None:
        return None

    samples = list(oracle_samples or [])
    if len(samples) < STRICT_STREAK:
        return False
    window = [_oracle_sample_strict(sample) for sample in samples[-STRICT_STREAK:]]
    if any(value is None for value in window):
        return None
    if final_strict is not True:
        return False
    return all(value is True for value in window)


# --- the diagnostic service --------------------------------------------------


class DiagnosticService(service.SceneService):
    """``SceneService`` with a profile-driven action seam and step telemetry.

    The base worker, seeding, environment construction (one reset per session),
    ``strict=True`` model load, termination, ``events.jsonl`` and PNG/MP4 logic
    are inherited unchanged.  Only two methods are overridden:

    * ``_select_action`` -- applies the active profile's ``use_amp`` /
      ``num_steps`` / ``n_action_steps`` to the resident policy and measures the
      action latency;
    * ``_run_capability`` -- wraps *only* the worker-owned ``env.step`` of this
      one capability to record a snapshot per action, then delegates to the base
      implementation unchanged.
    """

    def __init__(self, *args: Any, completion_mode: str = "native", **kwargs: Any) -> None:
        super().__init__(*args, completion_mode=completion_mode, **kwargs)
        self._diag_profile: Any = PROFILES["baseline_bf16"]
        self._diag_profile_name: str = "baseline_bf16"
        self._diag_condition: str | None = None
        self._last_action_latency_s: float | None = None
        self._diag_instruction_override: dict[str, Any] | None = None

    # -- profile configuration (worker thread only) ---------------------------

    def _do_configure_profile(self, profile_name: str) -> dict[str, Any]:
        """Apply one profile to the resident policy config and reset its queues.

        Fails closed: both ``policy.config`` and ``policy.model.config`` must be
        present, and every applied ``use_amp`` / ``num_steps`` /
        ``n_action_steps`` value must read back exactly, or an ``ok=False`` error
        is returned and the profile is *not* recorded as active.  The caller
        must then submit ZERO plans for the trial.
        """

        profile = PROFILES[profile_name]
        policy = getattr(self._v1, "_policy", None)
        config = getattr(policy, "config", None) if policy is not None else None
        model = getattr(policy, "model", None) if policy is not None else None
        model_config = getattr(model, "config", None) if model is not None else None

        if config is None or model_config is None:
            return _err(
                "invalid_fixture",
                "profile %r needs both policy.config and policy.model.config "
                "(policy present=%s, config present=%s, model.config present=%s)"
                % (profile_name, policy is not None, config is not None, model_config is not None),
            )

        expected = {
            "use_amp": bool(profile["amp"]),
            "num_steps": int(profile["num_steps"]),
            "n_action_steps": int(profile["n_action_steps"]),
        }

        def _matches(observed: Any, want: Any) -> bool:
            try:
                if isinstance(want, bool):
                    return observed is not None and bool(observed) == want
                return observed is not None and int(observed) == want
            except (TypeError, ValueError):
                return False

        applied: dict[str, Any] = {}
        mismatches: list[str] = []
        for label, target in (("policy.config", config), ("policy.model.config", model_config)):
            try:
                setattr(target, "use_amp", expected["use_amp"])
                setattr(target, "num_steps", expected["num_steps"])
                setattr(target, "n_action_steps", expected["n_action_steps"])
            except Exception as exc:  # noqa: BLE001 - a write that cannot land is a failure
                mismatches.append("%s: write failed: %s" % (label, exc))
                applied[label] = None
                continue
            observed = {
                "use_amp": getattr(target, "use_amp", None),
                "num_steps": getattr(target, "num_steps", None),
                "n_action_steps": getattr(target, "n_action_steps", None),
            }
            applied[label] = observed
            for key, want in expected.items():
                if not _matches(observed.get(key), want):
                    mismatches.append("%s.%s=%r != %r" % (label, key, observed.get(key), want))

        if mismatches:
            return _err(
                "invalid_fixture",
                "profile %r readback mismatch: %s" % (profile_name, "; ".join(mismatches)),
            )

        self._diag_profile = profile
        self._diag_profile_name = profile_name
        self._reset_policy_queues()
        return {
            "ok": True,
            "profile": profile_name,
            "policy_present": policy is not None,
            "applied": applied,
        }

    def configure_profile(self, profile_name: str) -> dict[str, Any]:
        """Queue the profile configuration while the worker is idle."""

        if profile_name not in PROFILES:
            return _err("invalid_body", "unknown profile %r" % (profile_name,))
        return self._sync_work("profile", lambda: self._do_configure_profile(profile_name))

    # -- action selection (worker thread only) --------------------------------

    def _select_action(self, batch: Any) -> np.ndarray:
        """Select one 7-D action, applying the active profile's precision.

        The preprocessing/postprocessing helpers and the policy are exactly the
        base service's; only autocast is toggled by ``profile['amp']`` and the
        denoising/action-chunk config is the one queued by ``configure_profile``.
        """

        if self._action_function is not None:
            # Test seam: never touch torch/CUDA.
            started = time.monotonic()
            action = np.asarray(self._action_function(batch), dtype=np.float64).reshape(-1)
            self._last_action_latency_s = time.monotonic() - started
            return action

        import torch

        profile = self._diag_profile
        started = time.monotonic()
        batch = self._v1._pre(batch)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=bool(profile["amp"])
        ):
            action = self._v1._policy.select_action(batch)
        action = self._v1._post(action)
        # The timer ends only AFTER the CUDA->CPU device transfer
        # (``.detach().cpu().numpy()``) and the float64 reshape, so it includes
        # the CUDA synchronization and the host conversion.
        action = np.asarray(action.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        self._last_action_latency_s = time.monotonic() - started
        return action

    # -- instruction override (worker thread only) ----------------------------

    def _do_override_basket_instruction(self) -> dict[str, Any]:
        env = self._env
        if env is None:
            return _err("invalid_fixture", "no live environment for instruction override")
        native = _native_task_instruction(env)
        if not native:
            return _err("invalid_fixture", "no native task instruction available")
        capability = catalog.CAPABILITIES["basket_both"]
        self._diag_instruction_override = {
            "capability_id": "basket_both",
            "old": capability.get("instruction"),
            "new": native,
        }
        capability["instruction"] = native
        return {"ok": True, "old": self._diag_instruction_override["old"], "new": native}

    def override_basket_instruction(self) -> dict[str, Any]:
        return self._sync_work("native_instruction", self._do_override_basket_instruction)

    def _do_restore_basket_instruction(self) -> dict[str, Any]:
        info = self._diag_instruction_override
        if info is not None:
            catalog.CAPABILITIES[info["capability_id"]]["instruction"] = info["old"]
            self._diag_instruction_override = None
        return {"ok": True}

    def restore_basket_instruction(self) -> dict[str, Any]:
        return self._sync_work("restore_instruction", self._do_restore_basket_instruction)

    # -- capability execution (worker thread only) ----------------------------

    def _run_capability(
        self,
        session: service.SessionRecord,
        plan: service.PlanRecord,
        job: service.JobRecord,
        capability_id: str,
    ) -> dict[str, Any]:
        """Wrap only this capability's ``env.step``, then run the base verbatim."""

        env = self._env
        if env is None or self._env_session_id != session.session_id:
            # No live environment: let the base report ``unknown_session``.
            return super()._run_capability(session, plan, job, capability_id)

        goals = [list(goal) for goal in catalog.CAPABILITIES[capability_id]["goals"]]
        # Literal, fixed FINAL-oracle goals for this condition (never derived
        # from the capability schedule actually submitted).
        oracle_goals = [
            list(goal) for goal in FINAL_ORACLE_GOALS.get(self._diag_condition or "", [])
        ]
        job.run_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = job.run_dir / "telemetry.jsonl"

        initial_snapshot = capture_snapshot(env, goals)
        samples: list[dict[str, Any]] = []
        counter = {"step": 0}

        original_step = env.step
        telemetry_file = open(telemetry_path, "w", encoding="utf-8")

        def wrapped_step(action: Any) -> Any:
            # Call the original step exactly once, unmodified.
            result = original_step(action)
            counter["step"] += 1
            try:
                snapshot = capture_snapshot(env, goals)
            except Exception as exc:  # noqa: BLE001 - keep the real failure as evidence
                snapshot = {"error": _format_exc(exc)}
            try:
                oracle_snapshot = capture_snapshot(env, oracle_goals)
            except Exception as exc:  # noqa: BLE001 - keep the real failure as evidence
                oracle_snapshot = {"error": _format_exc(exc)}
            native_success = None
            if isinstance(result, tuple) and len(result) >= 5 and isinstance(result[4], dict):
                native_success = bool(result[4].get("is_success", False))
            sample = {
                "step": counter["step"],
                "action": [float(v) for v in np.asarray(action).reshape(-1).tolist()],
                "policy_action_latency_s": self._last_action_latency_s,
                "profile": self._diag_profile_name,
                "condition": self._diag_condition,
                "native_success": native_success,
                "snapshot": snapshot,
                "oracle_snapshot": oracle_snapshot,
            }
            self._last_action_latency_s = None
            samples.append(sample)
            try:
                telemetry_file.write(json.dumps(sample, default=str) + "\n")
                telemetry_file.flush()
            except Exception:  # noqa: BLE001 - telemetry must never break the run
                pass
            return result

        env.step = wrapped_step
        try:
            result = super()._run_capability(session, plan, job, capability_id)
        finally:
            try:
                env.step = original_step
            except Exception:  # noqa: BLE001
                pass
            try:
                telemetry_file.close()
            except Exception:  # noqa: BLE001
                pass

        summary = summarize_samples(samples)
        # The fixed-oracle samples are summarized SEPARATELY from the declared
        # capability metrics: the protocol's success score is computed from
        # these, not from the declared schedule.
        oracle_summary = summarize_samples(
            [{"snapshot": sample.get("oracle_snapshot"), "native_success": None} for sample in samples]
        )
        oracle_samples = [
            sample.get("oracle_snapshot") for sample in samples if isinstance(sample, dict)
        ]
        native_events = _native_from_events(job.run_dir / "events.jsonl")
        diagnostic = {
            "job_id": job.job_id,
            "capability_id": capability_id,
            "condition": self._diag_condition,
            "profile": self._diag_profile_name,
            "profile_config": dict(PROFILES[self._diag_profile_name]),
            "goals": goals,
            "oracle_goals": oracle_goals,
            "n_samples": summary["n_samples"],
            "native_success_ever": native_events["ever"],
            "native_success_final": native_events["final"],
            "native_success_from_events_available": native_events["available"],
            "declared_predicates_final": (summary["final_snapshot"] or {}).get("predicates"),
            "strict_success_ever": summary["strict_success_ever"],
            "strict_success_final": summary["strict_success_final"],
            "max_strict_streak": summary["max_strict_streak"],
            "oracle_strict_success_final": oracle_summary["strict_success_final"],
            "oracle_max_strict_streak": oracle_summary["max_strict_streak"],
            "oracle_final_snapshot": oracle_summary["final_snapshot"],
            "oracle_samples": oracle_samples,
            "telemetry_errors": summary["telemetry_errors"],
            "telemetry_nulls": summary["telemetry_nulls"],
            "per_object_grasp_counts": summary["per_object_grasp_counts"],
            "phase_counts_last20": summary["phase_counts_last20"],
            "final_snapshot": summary["final_snapshot"],
            "initial_snapshot": initial_snapshot,
            "telemetry_path": str(telemetry_path),
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "proxy_limitation": PROXY_LIMITATION,
            "assisted": False,
            "Hermes_calls": 0,
        }
        try:
            _write_json_atomic(job.run_dir / "diagnostic.json", diagnostic)
        except Exception as exc:  # noqa: BLE001
            service.log("diagnostic write failed: %s" % exc)
        return result

    # -- final worker-side snapshot -------------------------------------------

    def _do_final_snapshot(self, goals: Any) -> dict[str, Any]:
        env = self._env
        if env is None:
            return _err("unknown_session", "no live environment for the final snapshot")
        try:
            snapshot = capture_snapshot(env, goals)
        except Exception as exc:  # noqa: BLE001
            return _err("internal_error", _format_exc(exc))
        return {"ok": True, "snapshot": snapshot}

    def final_snapshot(self, goals: Any) -> dict[str, Any]:
        return self._sync_work("final_snapshot", lambda: self._do_final_snapshot(goals))


def _native_from_events(events_path: Path) -> dict[str, Any]:
    """Read native ``is_success`` from the service ``events.jsonl`` format."""

    ever = False
    final: Any = None
    count = 0
    try:
        with open(events_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                count += 1
                value = record.get("native_success")
                if value is True:
                    ever = True
                final = value
    except Exception:  # noqa: BLE001
        return {"available": False, "ever": None, "final": None, "n_events": 0}
    return {"available": True, "ever": ever, "final": final, "n_events": count}


# --- campaign runner ---------------------------------------------------------


def _progress(message: str) -> None:
    sys.stderr.write("[placement-exp %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _poll_interval(deadline: float) -> float:
    """A polling step that is always <= 1 s and never overshoots the deadline."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 0.05
    return max(0.05, min(0.5, remaining))


def _submit_and_wait(
    service_: DiagnosticService,
    session_id: str,
    capability_ids: Any,
    budget: int,
    audit: bool,
    request_id: str,
    timeout: float,
    rationale: str,
) -> dict[str, Any]:
    """Submit one plan and poll it to a terminal state, cancelling on timeout."""

    session_public = service_.session(session_id) or {}
    payload = {
        "session_id": session_id,
        "scene_version": session_public.get("scene_version"),
        "request_id": request_id,
        "capability_ids": list(capability_ids),
        "rationale": rationale,
        "decision": "execute",
        "audit": bool(audit),
        "budget_per_subgoal": int(budget),
    }
    submitted = service_.submit_plan(payload)
    record: dict[str, Any] = {
        "request_id": request_id,
        "capability_ids": list(capability_ids),
        "budget_per_subgoal": int(budget),
        "audit": bool(audit),
        "payload": payload,
        "submitted": bool(submitted.get("ok")),
        "submit_error": None if submitted.get("ok") else submitted,
        "timed_out": False,
        "cancel_nonterminal": False,
        "cancelled": False,
        "plan": None,
        "wall_s": None,
    }
    if not submitted.get("ok"):
        return record

    started = time.monotonic()
    deadline = started + float(timeout)
    plan = None
    while True:
        plan = service_.plan(request_id)
        if plan is not None and plan["state"] in service.TERMINAL_PLAN_STATES:
            break
        if time.monotonic() >= deadline:
            record["timed_out"] = True
            break
        time.sleep(_poll_interval(deadline))

    if record["timed_out"]:
        # Cancel the exact request, then await terminal for at most 30 s.
        service_.cancel(request_id)
        cancel_deadline = time.monotonic() + 30.0
        while True:
            plan = service_.plan(request_id)
            if plan is not None and plan["state"] in service.TERMINAL_PLAN_STATES:
                record["cancelled"] = True
                break
            if time.monotonic() >= cancel_deadline:
                record["cancel_nonterminal"] = True
                break
            time.sleep(0.25)

    record["plan"] = plan
    record["wall_s"] = round(time.monotonic() - started, 3)
    return record


def _job_evidence(service_: DiagnosticService, job_id: str) -> dict[str, Any]:
    job = service_.job(job_id)
    if job is None:
        return {"job_id": job_id, "available": False}
    run_dir = Path(job.get("run_dir") or ".")
    diagnostic_path = run_dir / "diagnostic.json"
    telemetry_path = run_dir / "telemetry.jsonl"
    diagnostic: Any = None
    if diagnostic_path.is_file():
        try:
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            diagnostic = {"error": _format_exc(exc)}
    return {
        "job_id": job_id,
        "available": True,
        "job": job,
        "diagnostic": diagnostic,
        "diagnostic_path": str(diagnostic_path) if diagnostic_path.is_file() else None,
        "telemetry_path": str(telemetry_path) if telemetry_path.is_file() else None,
        "video_path": job.get("rollout_path"),
        "latest_png": job.get("latest_png"),
    }


def _run_condition(
    service_: DiagnosticService,
    condition: str,
    session_id: str,
    timeout: float,
    trial_key: str,
) -> dict[str, Any]:
    """Execute one condition against a fresh session and return its evidence."""

    spec = CONDITIONS[condition]
    result: dict[str, Any] = {
        "condition": condition,
        "mode": spec["mode"],
        "plans": [],
        "native_instruction": None,
        "instruction_override_error": None,
        "campaign_stopped": False,
        "stop_reason": None,
        "forced_handoff": None,
        "retry_same_goal": spec["mode"] == "same_goal_retry",
        "retry": None,
    }

    override_active = False
    try:
        if spec["native_instruction"]:
            override = service_.override_basket_instruction()
            if override.get("ok"):
                override_active = True
                result["native_instruction"] = override.get("new")
            else:
                result["instruction_override_error"] = override

        first = _submit_and_wait(
            service_,
            session_id,
            spec["capability_ids"],
            int(spec["budget_per_subgoal"]),
            bool(spec["audit"]),
            "%s-plan1" % trial_key,
            timeout,
            "placement diagnostic %s (%s)" % (condition, spec["mode"]),
        )
        result["plans"].append(first)

        if first["cancel_nonterminal"]:
            result["campaign_stopped"] = True
            result["stop_reason"] = "first plan cancel left the request nonterminal"
            return result

        if spec["mode"] == "forced_handoff":
            second = _submit_and_wait(
                service_,
                session_id,
                spec["forced_handoff_capability_ids"],
                int(spec["forced_handoff_budget_per_subgoal"]),
                bool(spec["audit"]),
                "%s-plan2" % trial_key,
                timeout,
                "forced handoff after first-plan outcome for %s" % condition,
            )
            result["plans"].append(second)
            result["forced_handoff"] = {
                "deliberately_forced_after_failure": True,
                "first_plan_state": (first.get("plan") or {}).get("state"),
                "first_plan_timed_out": first["timed_out"],
                "second_plan_state": (second.get("plan") or {}).get("state"),
                "note": "strictly experimental; never a production default recovery",
            }
            if second["cancel_nonterminal"]:
                result["campaign_stopped"] = True
                result["stop_reason"] = "second plan cancel left the request nonterminal"

        if spec["mode"] == "same_goal_retry":
            # Fixed same-goal continuation: a NEW second plan in the SAME session
            # with the SAME soup_to_basket goal and budget, submitted ONLY when
            # the first plan blocked without an operational timeout.  The
            # environment is never reset and the RNG is never reseeded; the
            # policy queues reset naturally per job.  No scripted release,
            # action injection or foreign-object schedule is used.
            first_state = (first.get("plan") or {}).get("state")
            if first.get("submit_error"):
                result["campaign_stopped"] = True
                result["stop_reason"] = "first plan submit error; no same-goal retry"
                result["retry"] = {"retried": False, "reason": "submit_error"}
                return result
            if first["timed_out"]:
                result["campaign_stopped"] = True
                result["stop_reason"] = "first plan operational timeout; no same-goal retry"
                result["retry"] = {"retried": False, "reason": "operational_timeout"}
                return result
            if first_state in ("cancelled", "error"):
                result["campaign_stopped"] = True
                result["stop_reason"] = "first plan %s; no same-goal retry" % first_state
                result["retry"] = {"retried": False, "reason": first_state}
                return result
            if first_state != "blocked":
                # completed (or any other terminal state) -> no retry.
                result["retry"] = {
                    "retried": False,
                    "reason": "first plan %s" % first_state,
                }
                return result
            # A blocked plan whose owned job ended in an operational error
            # (state=="error", ended_reason=="error" or a truthy job.error) must
            # NOT be retried: the same-goal continuation recovers from a
            # non-operational block only.  The job list is inspected through the
            # existing _job_evidence seam; the errored job's real detail is kept.
            for job_id in (first.get("plan") or {}).get("job_ids") or []:
                evidence = _job_evidence(service_, job_id)
                if not evidence.get("available"):
                    continue
                job_public = evidence.get("job") or {}
                if (
                    job_public.get("state") == "error"
                    or job_public.get("ended_reason") == "error"
                    or job_public.get("error")
                ):
                    result["campaign_stopped"] = True
                    result["stop_reason"] = (
                        "first plan blocked with an operational job error; "
                        "no same-goal retry"
                    )
                    result["retry"] = {"retried": False, "reason": "job_error"}
                    return result
            second = _submit_and_wait(
                service_,
                session_id,
                spec["capability_ids"],
                int(spec["budget_per_subgoal"]),
                bool(spec["audit"]),
                "%s-plan2" % trial_key,
                timeout,
                "same-goal retry after a blocked first plan for %s" % condition,
            )
            result["plans"].append(second)
            result["retry"] = {
                "retried": True,
                "same_goal": True,
                "first_plan_state": first_state,
                "second_plan_state": (second.get("plan") or {}).get("state"),
                "second_submit_error": second.get("submit_error"),
                "second_timed_out": second["timed_out"],
            }
            if second["cancel_nonterminal"]:
                result["campaign_stopped"] = True
                result["stop_reason"] = "second plan cancel left the request nonterminal"
        return result
    finally:
        if override_active:
            restored = service_.restore_basket_instruction()
            if not restored.get("ok"):
                result["instruction_override_error"] = restored


def _build_trial(
    service_: DiagnosticService,
    profile: str,
    condition: str,
    pair_text: str,
    seed: int,
    init_state_index: int,
    session: dict[str, Any],
    condition_result: dict[str, Any],
    final_snapshot: dict[str, Any] | None,
    model_revision: str | None,
    git_sha: str | None,
    wall_s: float,
    trial_key: str,
) -> dict[str, Any]:
    session_id = session.get("session_id")
    session_record = service_._sessions.get(session_id) if session_id else None
    initial_state_hash = getattr(session_record, "initial_state_hash", None)

    plan_records: list[dict[str, Any]] = []
    job_evidence: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    instructions: list[str] = []
    for plan_record in condition_result["plans"]:
        plan_public = plan_record.get("plan") or {}
        plan_records.append(
            {
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
        )
        for job_id in plan_public.get("job_ids") or []:
            evidence = _job_evidence(service_, job_id)
            job_evidence.append(evidence)
            if evidence.get("available"):
                instruction = (evidence.get("job") or {}).get("instruction")
                if instruction:
                    instructions.append(instruction)
                if evidence.get("diagnostic"):
                    diagnostics.append(evidence["diagnostic"])

    final_predicates = None
    if isinstance(final_snapshot, dict):
        final_predicates = final_snapshot.get("predicates")

    if isinstance(final_predicates, dict) and final_predicates and all(
        value is not None for value in final_predicates.values()
    ):
        task_success: bool | None = all(final_predicates.values())
    else:
        task_success = None

    # Fixed-oracle final score: the fixed literal final snapshot AND the LAST
    # FIVE fixed-oracle strict observations of the FINAL job.  Earlier
    # ``strict_success_ever`` peaks are deliberately ignored.
    final_oracle_samples: list[dict] = []
    for evidence in reversed(job_evidence):
        diagnostic = evidence.get("diagnostic")
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("oracle_samples"), list):
            final_oracle_samples = diagnostic["oracle_samples"]
            break
    strict_task_success = strict_final_score(
        final_snapshot if isinstance(final_snapshot, dict) else None,
        final_oracle_samples,
    )

    native_available = [
        diagnostic.get("native_success_ever")
        for diagnostic in diagnostics
        if diagnostic.get("native_success_ever") is not None
    ]
    native_benchmark_success: bool | None = any(native_available) if native_available else None

    errors: list[str] = []
    if condition_result.get("instruction_override_error"):
        errors.append("instruction_override: %s" % condition_result["instruction_override_error"])
    if condition_result.get("campaign_stopped"):
        errors.append("campaign_stopped: %s" % condition_result.get("stop_reason"))
    for plan_record in condition_result["plans"]:
        if plan_record.get("submit_error"):
            errors.append("plan_submit: %s" % plan_record["submit_error"])
        if plan_record.get("timed_out"):
            errors.append("plan_timeout: %s" % plan_record.get("request_id"))
    for diagnostic in diagnostics:
        if diagnostic.get("telemetry_errors"):
            errors.append(
                "telemetry_errors job=%s n=%s" % (diagnostic.get("job_id"), diagnostic["telemetry_errors"])
            )
    if final_snapshot is None:
        errors.append("final_snapshot_unavailable")

    # A capability that ended in a *job error* (state=="error", a truthy
    # ``job.error`` or ``ended_reason=="error"``) is an operational failure: it
    # is recorded with its real detail and is NEVER mistaken for an ordinary
    # ``budget_exhausted`` physical failure.
    job_operational_errors: list[str] = []
    for evidence in job_evidence:
        if not evidence.get("available"):
            continue
        job_public = evidence.get("job") or {}
        state = job_public.get("state")
        ended = job_public.get("ended_reason")
        detail = job_public.get("error")
        if state == "error" or ended == "error" or detail:
            job_operational_errors.append(
                "job_error: job=%s state=%s ended_reason=%s error=%s"
                % (evidence.get("job_id"), state, ended, detail or "")
            )
    errors.extend(job_operational_errors)

    # Operational failures (submit/timeout/config/telemetry) are kept distinct
    # from ``budget_exhausted`` physical failures, which are NOT errors.
    physical_failures: list[str] = []
    for evidence in job_evidence:
        job_public = evidence.get("job") or {}
        state = job_public.get("state")
        ended = job_public.get("ended_reason")
        detail = job_public.get("error")
        if state == "error" or ended == "error" or detail:
            # An operationally errored job (state=="error", ended_reason=="error"
            # or a truthy job.error) is never a physical budget exhaustion -- even
            # when it also reports ended_reason=="budget_exhausted".
            continue
        if ended == "budget_exhausted":
            physical_failures.append(
                "budget_exhausted: job=%s capability=%s"
                % (evidence.get("job_id"), job_public.get("capability_id"))
            )

    null_metrics: list[str] = []
    if task_success is None:
        null_metrics.append("task_success")
    if strict_task_success is None:
        null_metrics.append("strict_task_success")
    if native_benchmark_success is None:
        null_metrics.append("native_benchmark_success")
    if initial_state_hash is None:
        null_metrics.append("initial_state_hash")

    trial_id = trial_key
    return {
        "trial_id": trial_id,
        "profile": profile,
        "condition": condition,
        "pair": pair_text,
        "seed": seed,
        "init_state_index": init_state_index,
        "scene_id": session.get("scene_id"),
        "session_id": session_id,
        "initial_state_hash": initial_state_hash,
        "env_instance_id": session.get("env_instance_id"),
        "episode_resets": session.get("episode_resets"),
        "policy_resets": session.get("policy_resets"),
        "model_revision": model_revision,
        "source_git_sha": git_sha,
        "completion_mode": getattr(service_, "completion_mode", None),
        "instructions": instructions,
        "native_instruction": condition_result.get("native_instruction"),
        "plans": plan_records,
        "jobs": job_evidence,
        "diagnostics": [
            {
                "job_id": diagnostic.get("job_id"),
                "strict_success_ever": diagnostic.get("strict_success_ever"),
                "strict_success_final": diagnostic.get("strict_success_final"),
                "oracle_strict_success_final": diagnostic.get("oracle_strict_success_final"),
                "oracle_max_strict_streak": diagnostic.get("oracle_max_strict_streak"),
                "native_success_ever": diagnostic.get("native_success_ever"),
                "telemetry_errors": diagnostic.get("telemetry_errors"),
                "telemetry_nulls": diagnostic.get("telemetry_nulls"),
                "telemetry_path": diagnostic.get("telemetry_path"),
            }
            for diagnostic in diagnostics
        ],
        "final_oracle_goals": [list(goal) for goal in FINAL_ORACLE_GOALS.get(condition, [])],
        "final_snapshot": final_snapshot,
        "task_success": task_success,
        "strict_task_success": strict_task_success,
        "native_benchmark_success": native_benchmark_success,
        "forced_handoff": condition_result.get("forced_handoff"),
        "retry_same_goal": bool(condition_result.get("retry_same_goal")),
        "retry": condition_result.get("retry"),
        "job_operational_errors": job_operational_errors,
        "final_oracle_samples_n": len(final_oracle_samples),
        "final_oracle_strict_window": [
            _oracle_sample_strict(sample) for sample in final_oracle_samples[-STRICT_STREAK:]
        ],
        "timings": {"trial_wall_s": round(wall_s, 3)},
        "errors": errors,
        "operational_errors": list(errors),
        "budget_exhausted": bool(physical_failures),
        "physical_failures": physical_failures,
        "null_metrics": null_metrics,
    }


def _run_trial(
    service_: DiagnosticService,
    profile: str,
    condition: str,
    pair_text: str,
    timeout: float,
    model_revision: str | None,
    git_sha: str | None,
    run_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    seed_text, _, state_text = pair_text.partition(":")
    seed = int(seed_text or 0)
    init_state_index = int(state_text or 0)

    # A FRESH trial key per trial: it is unique even when the same profile is
    # re-run for a different condition/pair, and it is carried into the
    # per-plan request ids so repeated (profile, condition, pair) runs never
    # collide on ``request_conflict``.
    trial_key = "%s_%s_%s_%s" % (
        condition,
        profile,
        str(pair_text).replace(":", "-"),
        uuid.uuid4().hex[:8],
    )

    spec = CONDITIONS[condition]
    service_._diag_condition = condition

    session = service_.create_session(spec["scene_id"], seed=seed, init_state_index=init_state_index)
    if not session.get("ok"):
        message = "create_session: %s" % session
        return {
            "trial_id": trial_key,
            "profile": profile,
            "condition": condition,
            "pair": pair_text,
            "seed": seed,
            "init_state_index": init_state_index,
            "scene_id": spec["scene_id"],
            "session_id": None,
            "errors": [message],
            "operational_errors": [message],
            "null_metrics": ["session"],
            "timings": {"trial_wall_s": round(time.monotonic() - started, 3)},
        }

    profile_result = service_.configure_profile(profile)
    profile_ok = bool(profile_result.get("ok"))

    if profile_ok:
        condition_result = _run_condition(
            service_, condition, session["session_id"], timeout, trial_key
        )
    else:
        # A configuration failure submits ZERO plans: no partial profile may run.
        condition_result = {
            "condition": condition,
            "mode": spec["mode"],
            "plans": [],
            "native_instruction": None,
            "instruction_override_error": None,
            "campaign_stopped": True,
            "stop_reason": "profile configuration failed",
            "forced_handoff": None,
        }

    final_snapshot: dict[str, Any] | None = None
    goals = FINAL_ORACLE_GOALS.get(condition, [])
    if profile_ok and service_._active_request_id is None and service_._env is not None:
        final = service_.final_snapshot(goals)
        if final.get("ok"):
            final_snapshot = final.get("snapshot")

    session_after = service_.session(session["session_id"]) or session
    trial = _build_trial(
        service_,
        profile,
        condition,
        pair_text,
        seed,
        init_state_index,
        session_after,
        condition_result,
        final_snapshot,
        model_revision,
        git_sha,
        time.monotonic() - started,
        trial_key,
    )
    trial["profile_result"] = profile_result
    if not profile_ok:
        message = "profile_config: %s" % profile_result
        trial.setdefault("errors", []).append(message)
        trial.setdefault("operational_errors", []).append(message)
    if not trial.get("errors"):
        trial.pop("errors", None)
    if not trial.get("operational_errors"):
        trial.pop("operational_errors", None)
    _write_json_atomic(run_root / ("trial_%s.json" % trial["trial_id"]), trial)
    return trial


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated placement diagnostic experiment runner (no training, no downloads)."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(PROFILES),
        default=list(PROFILES),
        help="precision/denoising/chunk profiles to screen",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=["soup_fresh"],
        help="placement conditions to run",
    )
    parser.add_argument("--pairs", nargs="+", default=["0:0"], help="seed:state pairs, e.g. 0:0")
    parser.add_argument("--output", required=True, type=str, help="absolute path of the JSON report (must not exist)")
    parser.add_argument("--run-root", required=True, type=str, help="absolute run root for per-trial artifacts")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-plan deadline in seconds")
    parser.add_argument(
        "--completion-mode",
        choices=service.COMPLETION_MODES,
        default=service.DEFAULT_COMPLETION_MODE,
        help="completion gate passed to the diagnostic service (native or release_verified)",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isabs(args.output):
        parser.error("--output must be an absolute path")
    if not os.path.isabs(args.run_root):
        parser.error("--run-root must be an absolute path")
    if Path(args.output).exists():
        parser.error("--output already exists; refusing to overwrite %s" % args.output)
    for pair in args.pairs:
        seed_text, sep, state_text = str(pair).partition(":")
        if not sep:
            parser.error("pair %r is not in seed:state form" % (pair,))
        try:
            int(seed_text)
            int(state_text)
        except ValueError:
            parser.error("pair %r has non-integer components" % (pair,))


def _campaign_report(
    args: argparse.Namespace,
    trials: list[dict[str, Any]],
    model_revision: str | None,
    git_sha: str | None,
    started: float,
    fatal_error: str | None,
) -> dict[str, Any]:
    """Build the full campaign report (metadata + trials + aggregate).

    Shared by the per-append atomic persistence and the final write so the
    on-disk report is always a complete, self-consistent snapshot.
    """

    output_path = Path(args.output)
    run_root = Path(args.run_root)
    trial_errors = sum(1 for trial in trials if trial.get("errors"))
    return {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "metadata": {
            "output": str(output_path),
            "run_root": str(run_root),
            "profiles": list(args.profiles),
            "conditions": list(args.conditions),
            "pairs": list(args.pairs),
            "timeout_s": float(args.timeout),
            "completion_mode": getattr(args, "completion_mode", "native"),
            "model_path": service.DEFAULT_MODEL_PATH,
            "model_revision": model_revision,
            "source_git_sha": git_sha,
            "source_revision_expected": SOURCE_REVISION_EXPECTED,
            "source_sha256": _source_sha256(),
            "package_versions": _package_versions(),
            "config": {
                "chunk_size": CHUNK_SIZE,
                "action_dim": ACTION_DIM,
                "image_size": IMAGE_SIZE,
                "control_mode": "relative",
                "render_fps": CONTROL_FREQUENCY_HZ,
                "num_steps": "denoising iterations per action prediction",
                "n_action_steps": "actions executed before a new prediction",
            },
            "assisted": False,
            "hermes_calls": 0,
            "campaign_wall_s": round(time.monotonic() - started, 3),
            "fatal_error": fatal_error,
        },
        "trials": trials,
        "aggregate": {
            "n_trials": len(trials),
            "n_errors": trial_errors,
            "n_task_success": sum(1 for t in trials if t.get("task_success") is True),
            "n_strict_task_success": sum(1 for t in trials if t.get("strict_task_success") is True),
            "n_native_success": sum(1 for t in trials if t.get("native_benchmark_success") is True),
            "n_timed_out": sum(
                1 for t in trials if any(p.get("timed_out") for p in (t.get("plans") or []))
            ),
            "n_budget_exhausted": sum(1 for t in trials if t.get("budget_exhausted") is True),
            "reliability_note": (
                "explicit counts only; a single seed/state pair is not a "
                "statistical reliability claim"
            ),
        },
    }


def _run_campaign(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    git_sha = _git_rev_parse()
    trials: list[dict[str, Any]] = []
    fatal_error: str | None = None
    model_revision: str | None = None
    started = time.monotonic()
    persist_errors: list[str] = []

    def _persist() -> None:
        """Atomically write the current report snapshot; never raises."""

        try:
            _write_json_atomic(
                output_path,
                _campaign_report(args, trials, model_revision, git_sha, started, fatal_error),
            )
        except Exception as exc:  # noqa: BLE001 - a report write failure is fatal
            persist_errors.append(_format_exc(exc))
            _progress("report write failed: %s" % exc)

    diagnostic_service: DiagnosticService | None = None
    ready = False
    try:
        diagnostic_service = DiagnosticService(
            service.DEFAULT_MODEL_PATH,
            str(run_root),
            completion_mode=args.completion_mode,
        )
        diagnostic_service.start()
        deadline = time.monotonic() + 120.0
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
            fatal_error = "worker did not become ready within 120s"
        model_revision = diagnostic_service.health().get("model_revision")

        if ready and model_revision != SOURCE_REVISION_EXPECTED:
            # ZERO actions: the resident model revision must match the audited
            # source revision before any plan is submitted.
            fatal_error = (
                "model_revision %r does not match SOURCE_REVISION_EXPECTED %r"
                % (model_revision, SOURCE_REVISION_EXPECTED)
            )
            ready = False

        if ready:
            planned = [
                (profile, condition, pair)
                for profile in args.profiles
                for condition in args.conditions
                for pair in args.pairs
            ]
            total = len(planned)
            for index, (profile, condition, pair) in enumerate(planned, start=1):
                _progress(
                    "trial %d/%d profile=%s condition=%s pair=%s"
                    % (index, total, profile, condition, pair)
                )
                try:
                    trial = _run_trial(
                        diagnostic_service,
                        profile,
                        condition,
                        pair,
                        float(args.timeout),
                        model_revision,
                        git_sha,
                        run_root,
                    )
                except BaseException as exc:  # noqa: BLE001 - preserve the record, then stop
                    fatal_error = _format_exc(exc)
                    trial = {
                        "trial_id": "%s_%s_%s" % (condition, profile, str(pair).replace(":", "-")),
                        "profile": profile,
                        "condition": condition,
                        "pair": pair,
                        "errors": ["trial_exception: %s" % _format_exc(exc)],
                        "operational_errors": ["trial_exception: %s" % _format_exc(exc)],
                        "null_metrics": ["trial"],
                    }
                    trials.append(trial)
                    _persist()  # preserve the record BEFORE stopping
                    _progress("trial exception; stopping campaign (no continuation)")
                    break
                trials.append(trial)
                # The report is atomically persisted immediately after every
                # appended trial, so a crash can never lose a finished trial.
                _persist()
                _progress(
                    "trial %d/%d done state=%s task_success=%s strict=%s errors=%d"
                    % (
                        index,
                        total,
                        (trial.get("plans") or [{}])[-1].get("state") if trial.get("plans") else None,
                        trial.get("task_success"),
                        trial.get("strict_task_success"),
                        len(trial.get("errors") or []),
                    )
                )
                if (trial.get("errors") or []) and "campaign_stopped" in " ".join(trial["errors"]):
                    _progress("campaign stopped after nonterminal cancellation")
                    break
                # A job that ended in a real operational error (state=="error",
                # truthy job.error or ended_reason=="error") stops the campaign
                # with an explanatory fatal_error -- it is never treated as an
                # ordinary budget_exhausted physical failure.  The record has
                # already been atomically persisted above.
                if trial.get("job_operational_errors"):
                    fatal_error = (
                        "job operational error(s); stopping campaign (no continuation): %s"
                        % ("; ".join(trial["job_operational_errors"]),)
                    )
                    _progress(fatal_error)
                    break
    except BaseException as exc:  # noqa: BLE001 - never lose the partial report
        fatal_error = _format_exc(exc)
    finally:
        if diagnostic_service is not None:
            try:
                if diagnostic_service._ready and diagnostic_service._active_request_id is None:
                    diagnostic_service._sync_work(
                        "close_env", lambda: (diagnostic_service._close_env() or {"ok": True})
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

    report_write_failed = bool(persist_errors)
    # Report-write failures and any operational error must never exit 0.
    operational_failure = any(bool(trial.get("operational_errors")) for trial in trials)
    return 1 if (fatal_error or report_write_failed or operational_failure) else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return _run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
