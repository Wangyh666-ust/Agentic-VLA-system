#!/usr/bin/env python3
"""Fixed isolated wine-stage diagnostic module (read-only service host).

This module hosts :class:`WineDiagnosticService` -- a subclass of the *actual*
``placement_experiments.DiagnosticService`` (itself a subclass of the real
``scene_demo/service.py`` ``SceneService``) -- and a small campaign runner that
drives one real worker / one real model load through the native LIBERO
``libero_goal/9`` wine scene and the shared ``libero_goal/8`` table scene.

Nothing here retrains, downloads, resets between subgoals, calls Hermes, opens a
socket, or forces a release.  The runner reuses the base service lifecycle
verbatim (worker started once, model loaded once, native termination /
``events.jsonl`` / PNG / MP4 logic untouched).  It adds three isolated seams:

* :func:`register_native_wine_scene` -- a context manager that adds the
  in-memory ``wine_native_goal9`` scene (``libero_goal``, task 9, no patches)
  and appends only that scene id to the existing ``wine_to_rack`` capability,
  then restores BOTH ``catalog.SCENES`` and ``catalog.CAPABILITIES`` to their
  exact deep-copied contents in a ``finally`` (no catalog file is written and no
  HTTP server is started);
* :class:`WineStepWrapper` -- a read-only observation wrapper around the
  worker-owned ``env.step`` of one capability.  It calls the saved original step
  exactly once with the unchanged action object and returns the *same* tuple
  object, recording a fixed-goal contact/kinematics snapshot plus the gripper
  state immediately before and after every real action;
* :class:`PostActionCapture` -- a context manager that temporarily replaces the
  policy postprocessor ``self._v1._post`` with a wrapper that copies the raw
  policy action *before* the original post mutates it (so the true raw dimension
  is preserved), invokes the original post exactly once, and returns its result
  object unchanged.

Grasp-proxy limitation
======================

The grasp signal recorded here is the robosuite **contact-geom proxy**
``inner._check_grasp(gripper, obj.contact_geoms)``: it reports whether the
gripper contact geoms touch the object's contact geoms.  It is a *contact /
kinematic screening proxy*, not a tactile or force measurement.  It can be
``True`` while the object merely brushes past the fingers and ``False`` while a
stable pinch is held through a non-contact geom, and it is undefined when the
simulator does not expose the probe at all.  Unknown values are therefore
recorded as ``None`` and are never turned into success.

Only stdlib + numpy are imported at module import time; ``torch`` is imported
lazily by the pinned base :meth:`SceneService._select_action` (never by the pure
probes or the summarizer), so ``--help`` and the GPU-free unit tests never
initialise CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import sys
import time
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
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402

EXPERIMENT_NAME = "wine_stage_diagnostics"
CONTROL_FREQUENCY_HZ = 20
ACTION_DIM = 7
STRICT_STREAK = service.GOAL_CONSECUTIVE_STEPS  # five consecutive satisfied steps

# The exact, fixed wine oracle goal set.  It is literally written here and is
# never derived from the capability schedule that was submitted or executed.
WINE_ORACLE_GOALS: list[list[str]] = [["on", "wine_bottle_1", "wine_rack_1_top_region"]]
WINE_GOAL_KEY = catalog.goal_key(WINE_ORACLE_GOALS[0])
WINE_OBJECT_ID = "wine_bottle_1"
WINE_TARGET_ID = "wine_rack_1_top_region"

# The exact fixed ``placement_experiments.FINAL_ORACLE_GOALS`` key under which the
# isolated wine goals are temporarily registered (never ``table_wine_only``).
FINAL_ORACLE_KEY = "wine_stage_diagnostic"

WINE_CAPABILITY_ID = "wine_to_rack"
WINE_INSTRUCTION = "put the wine bottle on the rack"
WINE_SCENE_ID = "wine_native_goal9"
SHARED_SCENE_ID = "goal_table"

BASELINE_PROFILE = "baseline_bf16"
COMPLETION_MODE = "release_verified"

# The two fixed conditions, in their fixed order.  ``native_goal9`` runs the
# new libero_goal/9 wine scene; ``shared_goal8`` runs the original goal_table
# (libero_goal/8) scene with no patches.  Both use a fresh original scene.
CONDITION_ORDER = ("native_goal9", "shared_goal8")
CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "native_goal9": {"scene_id": WINE_SCENE_ID, "task_id": 9, "variant": "original"},
    "shared_goal8": {"scene_id": SHARED_SCENE_ID, "task_id": 8, "variant": "original"},
}

READY_TIMEOUT_S = 120.0

# Summarizer geometry / thresholds (measured, never assumed).
LIFT_THRESHOLD_M = 0.02
WIDTH_CHANGE_EPSILON_M = 0.0001

PROXY_LIMITATION = (
    "grasp is screened from robosuite contact geoms via inner._check_grasp, a "
    "contact/kinematic proxy -- not a tactile or force measurement; it can be "
    "wrong at contact transitions and is unknown when the probe is unavailable. "
    "gripper width is a jaw-joint gap, not an object identity or grasp-force "
    "measurement. An unknown value is never treated as success."
)


# --- small helpers -----------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _finite_scalar(value: Any) -> float | None:
    """A finite float scalar, or ``None`` for a missing/malformed/nonfinite one."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _flat_float_list(value: Any) -> list[float] | None:
    """``detach().float().cpu().numpy().flatten().tolist()`` when possible.

    The actual length of ``value`` is preserved (never forced to seven).  A
    value that cannot be flattened to a finite-agnostic float list returns
    ``None``.  The conversion is pure (it never mutates ``value``).  The
    ``float()`` widening runs on the detached observation copy *before* ``cpu()``
    so a BF16 tensor -- whose dtype numpy cannot convert directly -- is widened
    without touching the caller-owned tensor.
    """

    if value is None:
        return None
    obj = value
    detach = getattr(obj, "detach", None)
    if callable(detach):
        try:
            obj = detach()
        except Exception:  # noqa: BLE001 - not a tensor; keep the original
            obj = value
    to_float = getattr(obj, "float", None)
    if callable(to_float):
        try:
            obj = to_float()
        except Exception:  # noqa: BLE001 - keep the detached observation copy
            pass
    cpu = getattr(obj, "cpu", None)
    if callable(cpu):
        try:
            obj = cpu()
        except Exception:  # noqa: BLE001
            pass
    numpy = getattr(obj, "numpy", None)
    if callable(numpy):
        try:
            obj = numpy()
        except Exception:  # noqa: BLE001
            pass
    try:
        array = np.asarray(obj).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    try:
        return [float(component) for component in array.tolist()]
    except (TypeError, ValueError):
        return None


def _batch_task_text(batch: Any) -> list[str] | None:
    """The actual policy input task text (as a list), or ``None``."""

    task: Any = None
    if isinstance(batch, dict):
        task = batch.get("task")
    if task is None:
        task = getattr(batch, "task", None)
    if isinstance(task, (list, tuple)):
        texts = [str(item) for item in task if isinstance(item, str) and item.strip()]
        return texts or None
    if isinstance(task, str) and task.strip():
        return [task.strip()]
    return None


def _batch_state_list(batch: Any) -> list[float] | None:
    """The actual policy input ``observation.state`` (as a list), or ``None``."""

    state: Any = None
    if isinstance(batch, dict):
        state = batch.get("observation.state")
        if state is None:
            observation = batch.get("observation")
            if isinstance(observation, dict):
                state = observation.get("state")
    if state is None:
        return None
    flattened = _flat_float_list(state)
    return flattened


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


# --- fixed scene registration ------------------------------------------------


@contextlib.contextmanager
def register_native_wine_scene(scene_id: str = WINE_SCENE_ID):
    """Temporarily register the native ``libero_goal/9`` wine scene.

    The original *contents* of ``catalog.SCENES``, ``catalog.CAPABILITIES`` and
    ``placement_experiments.FINAL_ORACLE_GOALS`` are deep-copied up front, the
    in-memory scene is added, and only the new scene id is appended to the
    existing ``wine_to_rack`` capability.  The fixed ``FINAL_ORACLE_KEY`` entry is
    set to a literal, independent copy of ``WINE_ORACLE_GOALS`` -- never derived
    from the submitted or executed capability schedule.  On exit -- including on
    an exception -- all three *original dictionary objects* are cleared and
    repopulated from the saved contents, so each dictionary's identity and its
    contents are restored exactly.  No catalog file is written, no HTTP server is
    started and no ``placement_experiments`` source is modified.
    """

    scenes_saved = copy.deepcopy(catalog.SCENES)
    capabilities_saved = copy.deepcopy(catalog.CAPABILITIES)
    final_oracle_saved = copy.deepcopy(pe.FINAL_ORACLE_GOALS)
    try:
        catalog.SCENES[scene_id] = {
            "label": "Native LIBERO goal/9 wine table",
            "description": (
                "Original libero_goal task 9 wine scene built for the isolated "
                "wine-stage diagnostic; no layout patch is applied."
            ),
            "suite": "libero_goal",
            "task_id": 9,
            "variant": "original",
            "patches": [],
            "storage_policy": {WINE_OBJECT_ID: WINE_TARGET_ID},
        }
        wine = catalog.CAPABILITIES[WINE_CAPABILITY_ID]
        if scene_id not in wine["scene_ids"]:
            wine["scene_ids"].append(scene_id)
        pe.FINAL_ORACLE_GOALS[FINAL_ORACLE_KEY] = copy.deepcopy(WINE_ORACLE_GOALS)
        yield scene_id
    finally:
        catalog.SCENES.clear()
        catalog.SCENES.update(scenes_saved)
        catalog.CAPABILITIES.clear()
        catalog.CAPABILITIES.update(capabilities_saved)
        pe.FINAL_ORACLE_GOALS.clear()
        pe.FINAL_ORACLE_GOALS.update(final_oracle_saved)


# --- read-only gripper probe -------------------------------------------------


def _gripper_of(inner: Any) -> Any:
    robots = getattr(inner, "robots", None) or []
    robot = robots[0] if robots else None
    return getattr(robot, "gripper", None) if robot is not None else None


def capture_gripper_state(env: Any) -> dict[str, Any]:
    """Read the gripper joint/actuator state from the live simulator (read-only).

    Never steps, resets, forwards or otherwise mutates the simulator: it only
    reads named joints through ``sim.data.get_joint_qpos`` /
    ``get_joint_qvel`` and named actuators through
    ``model.actuator_name2id(name)`` + ``data.ctrl``.  Any unreadable, malformed or
    nonfinite value is recorded as ``None`` with a short ``error`` note, and a
    partially readable record keeps its known fields.  ``width_m`` is
    ``abs(qpos[0] - qpos[1])`` ONLY when exactly two finite scalar jaw values
    were read; otherwise it is ``None`` (no guessed finger threshold).
    """

    result: dict[str, Any] = {
        "joint_names": None,
        "qpos": None,
        "qvel": None,
        "current_action": None,
        "actuator_names": None,
        "actuator_ctrl": None,
        "width_m": None,
        "error": None,
    }
    errors: list[str] = []

    try:
        inner = service._inner_env(env)
    except Exception as exc:  # noqa: BLE001 - no inner env -> everything unknown
        result["error"] = "inner env unavailable: %s" % exc
        return result

    gripper = _gripper_of(inner)

    data = getattr(getattr(inner, "sim", None), "data", None)
    model = getattr(getattr(inner, "sim", None), "model", None)

    # --- gripper joint names ---
    names: list[str] | None = None
    if gripper is None:
        errors.append("no gripper available")
    else:
        raw_joints = getattr(gripper, "joints", None)
        if raw_joints:
            try:
                names = [str(joint) for joint in raw_joints]
            except Exception as exc:  # noqa: BLE001
                errors.append("gripper joint names unavailable: %s" % exc)
        if not names:
            errors.append("gripper joint names unavailable")
            names = None
    result["joint_names"] = names

    # --- per-joint qpos / qvel ---
    if names is not None and data is not None:
        qpos_values: list[float | None] = []
        qvel_values: list[float | None] = []
        for name in names:
            try:
                raw_qpos = np.asarray(data.get_joint_qpos(name), dtype=np.float64).reshape(-1)
                qpos_values.append(_finite_scalar(raw_qpos[0]) if raw_qpos.size else None)
                if raw_qpos.size == 0:
                    errors.append("empty qpos for joint %r" % name)
            except Exception as exc:  # noqa: BLE001
                qpos_values.append(None)
                errors.append("qpos unavailable for joint %r: %s" % (name, exc))
            try:
                raw_qvel = np.asarray(data.get_joint_qvel(name), dtype=np.float64).reshape(-1)
                qvel_values.append(_finite_scalar(raw_qvel[0]) if raw_qvel.size else None)
                if raw_qvel.size == 0:
                    errors.append("empty qvel for joint %r" % name)
            except Exception as exc:  # noqa: BLE001
                qvel_values.append(None)
                errors.append("qvel unavailable for joint %r: %s" % (name, exc))
        result["qpos"] = qpos_values
        result["qvel"] = qvel_values
        # Panda width: ONLY exactly two finite scalar jaw values.
        finite_jaws = [value for value in qpos_values if value is not None]
        if (
            len(qpos_values) == 2
            and all(value is not None for value in qpos_values)
            and len(finite_jaws) == 2
        ):
            result["width_m"] = abs(qpos_values[0] - qpos_values[1])

    # --- current action (a plain read-only attribute, if present) ---
    if gripper is not None:
        current = getattr(gripper, "current_action", None)
        if current is not None:
            result["current_action"] = _flat_float_list(current)

    # --- actuator names -> model.actuator_name2id -> data.ctrl ---
    if gripper is not None:
        raw_actuators = getattr(gripper, "actuators", None)
        if raw_actuators:
            act_names: list[str] = []
            for actuator in raw_actuators:
                name = getattr(actuator, "name", None)
                if name is None and isinstance(actuator, str):
                    name = actuator
                act_names.append(str(name) if name is not None else "")
            result["actuator_names"] = act_names
            name_to_id = getattr(model, "actuator_name2id", None)
            ctrl = getattr(data, "ctrl", None) if data is not None else None
            # The native ``sim.model.actuator_name2id`` is callable(name) -> int;
            # the ``dict.get`` form is kept only for compatible mocks.
            if callable(name_to_id):
                lookup: Any = name_to_id
            elif isinstance(name_to_id, dict):
                lookup = name_to_id.get
            else:
                lookup = None
            if lookup is not None and ctrl is not None:
                ctrl_values: list[float | None] = []
                for name in act_names:
                    value: float | None = None
                    try:
                        index = lookup(name)
                    except Exception as exc:  # noqa: BLE001 - per-name failure
                        errors.append("actuator id unavailable for %r: %s" % (name, exc))
                        ctrl_values.append(None)
                        continue
                    if index is not None:
                        try:
                            value = _finite_scalar(np.asarray(ctrl).reshape(-1)[int(index)])
                        except Exception as exc:  # noqa: BLE001
                            errors.append("ctrl unavailable for actuator %r: %s" % (name, exc))
                            value = None
                    else:
                        errors.append("unknown actuator id for %r" % name)
                    ctrl_values.append(value)
                result["actuator_ctrl"] = ctrl_values
            else:
                errors.append("actuator id map or ctrl unavailable")

    if errors:
        result["error"] = "; ".join(errors)
    return result


# --- wine telemetry summarizer -----------------------------------------------


def _snapshot_of(sample: Any, key: str) -> dict | None:
    if not isinstance(sample, dict):
        return None
    snapshot = sample.get(key)
    return snapshot if isinstance(snapshot, dict) else None


def _action_field(sample: Any, key: str) -> list[float] | None:
    if not isinstance(sample, dict):
        return None
    return _flat_float_list(sample.get(key))


def _snapshot_held(snapshot: dict | None, object_id: str) -> bool | None:
    """Whether ``object_id`` is reported held, or ``None`` when unknown/missing."""

    if not isinstance(snapshot, dict):
        return None
    held = snapshot.get("held_objects")
    if not isinstance(held, list):
        return None
    return object_id in [str(entry) for entry in held]


def _snapshot_predicate(snapshot: dict | None, key: str) -> bool | None:
    if not isinstance(snapshot, dict):
        return None
    predicates = snapshot.get("predicates")
    if not isinstance(predicates, dict) or key not in predicates:
        return None
    value = predicates.get(key)
    return value if isinstance(value, bool) else None


def _snapshot_wine_z(snapshot: dict | None) -> float | None:
    if not isinstance(snapshot, dict):
        return None
    objects = snapshot.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(WINE_OBJECT_ID)
    if not isinstance(entry, dict):
        return None
    position = entry.get("position")
    if not _is_sequence(position) or len(position) < 3:
        return None
    return _finite_scalar(position[2])


def _graphic_width(gripper: Any, key: str) -> float | None:
    if not isinstance(gripper, dict):
        return None
    return _finite_scalar(gripper.get(key))


def _sample_step(sample: Any, index: int) -> int:
    if isinstance(sample, dict):
        step = sample.get("step")
        if isinstance(step, int) and not isinstance(step, bool):
            return step
    return index + 1


def _dimension_record(values: list[list[float] | None]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    nulls = 0
    for value in values:
        if value is None:
            nulls += 1
            continue
        key = str(len(value))
        counts[key] = counts.get(key, 0) + 1
    return {
        "dimension_counts": counts,
        "dims": sorted(int(key) for key in counts),
        "n_known": sum(counts.values()),
        "n_null": nulls,
    }


def _empty_wine_summary() -> dict[str, Any]:
    return {
        "n_samples": 0,
        "first_grasp_proxy_step": None,
        "max_grasp_proxy_streak": None,
        "max_lift_m": None,
        "lift_reference_m": None,
        "lift_reference_step": None,
        "first_lift_2cm_step": None,
        "lift_threshold_m": LIFT_THRESHOLD_M,
        "first_goal_true_step": None,
        "first_strict_five_end_step": None,
        "strict_streak_required": STRICT_STREAK,
        "sent_open_count": None,
        "sent_close_count": None,
        "sent_open_while_goal_true_count": None,
        "sent_open_while_grasp_proxy_before_count": None,
        "width_min_m": None,
        "width_max_m": None,
        "width_change_epsilon_m": WIDTH_CHANGE_EPSILON_M,
        "open_command_width_increased_count": None,
        "raw_policy_action_dim": _dimension_record([]),
        "postprocessed_action_dim": _dimension_record([]),
        "sent_action_dim": _dimension_record([]),
        "n_native_success_true": None,
        "n_native_success_known": None,
        "native_success_final": None,
        "null_counts": {},
        "error_counts": {},
        "grasp_proxy_limitation": PROXY_LIMITATION,
    }


def summarize_wine_samples(samples: Any) -> dict[str, Any]:
    """Aggregate one job's wine telemetry samples into measured diagnostics.

    Every metric is computed only from *actually observed* values: a missing or
    unreadable field is ``None`` ("unknown") and is never silently turned into a
    ``False`` or a ``0``.  Metrics that cannot be computed from the available
    samples (e.g. a lift without a readable wine position, an open command
    without a readable sent action) are ``None``, and an empty sample list yields
    an all-null metric record.  The grasp signal is a contact screening proxy,
    not a physical grasp identity, so no causal or training conclusion is drawn.
    """

    sample_list = [sample for sample in (samples or []) if isinstance(sample, dict)]
    if not sample_list:
        return _empty_wine_summary()

    raw_values: list[list[float] | None] = []
    post_values: list[list[float] | None] = []
    sent_values: list[list[float] | None] = []

    proxies: list[bool | None] = []
    proxy_before: list[bool | None] = []
    goal_true: list[bool | None] = []
    strict_values: list[Any] = []
    native_values: list[Any] = []
    width_before: list[float | None] = []
    width_after: list[float | None] = []

    null_counts: dict[str, int] = {
        "raw_policy_action": 0,
        "postprocessed_action": 0,
        "sent_action": 0,
        "instruction": 0,
        "policy_input_state": 0,
        "before_snapshot": 0,
        "after_snapshot": 0,
        "gripper_before": 0,
        "gripper_after": 0,
    }
    error_counts: dict[str, int] = {
        "before_snapshot_errors": 0,
        "after_snapshot_errors": 0,
        "gripper_before_errors": 0,
        "gripper_after_errors": 0,
    }

    for sample in sample_list:
        before_snapshot = _snapshot_of(sample, "before_snapshot")
        after_snapshot = _snapshot_of(sample, "after_snapshot")
        gripper_before = _snapshot_of(sample, "gripper_before")
        gripper_after = _snapshot_of(sample, "gripper_after")

        raw = _action_field(sample, "raw_policy_action")
        post = _action_field(sample, "postprocessed_action")
        sent = _action_field(sample, "sent_action")
        raw_values.append(raw)
        post_values.append(post)
        sent_values.append(sent)

        if raw is None:
            null_counts["raw_policy_action"] += 1
        if post is None:
            null_counts["postprocessed_action"] += 1
        if sent is None:
            null_counts["sent_action"] += 1
        if sample.get("instruction") is None:
            null_counts["instruction"] += 1
        if sample.get("policy_input_state") is None:
            null_counts["policy_input_state"] += 1
        if before_snapshot is None:
            null_counts["before_snapshot"] += 1
        if after_snapshot is None:
            null_counts["after_snapshot"] += 1
        if gripper_before is None:
            null_counts["gripper_before"] += 1
        if gripper_after is None:
            null_counts["gripper_after"] += 1
        if isinstance(before_snapshot, dict) and before_snapshot.get("error"):
            error_counts["before_snapshot_errors"] += 1
        if isinstance(after_snapshot, dict) and after_snapshot.get("error"):
            error_counts["after_snapshot_errors"] += 1
        if isinstance(gripper_before, dict) and gripper_before.get("error"):
            error_counts["gripper_before_errors"] += 1
        if isinstance(gripper_after, dict) and gripper_after.get("error"):
            error_counts["gripper_after_errors"] += 1

        held_before = _snapshot_held(before_snapshot, WINE_OBJECT_ID)
        held_after = _snapshot_held(after_snapshot, WINE_OBJECT_ID)
        proxy_before.append(held_before)
        known = [value for value in (held_before, held_after) if value is not None]
        if any(value is True for value in known):
            proxies.append(True)
        elif len(known) == 2 and not any(known):
            proxies.append(False)
        else:
            proxies.append(None)

        after_goal = _snapshot_predicate(after_snapshot, WINE_GOAL_KEY)
        before_goal = _snapshot_predicate(before_snapshot, WINE_GOAL_KEY)
        goal_true.append(after_goal if after_goal is not None else before_goal)

        strict_values.append(after_snapshot.get("strict_candidate") if isinstance(after_snapshot, dict) else None)
        native_values.append(sample.get("native_success"))
        width_before.append(_graphic_width(gripper_before, "width_m"))
        width_after.append(_graphic_width(gripper_after, "width_m"))

    n = len(sample_list)

    # --- grasp proxy streak / first step ---
    max_streak: int | None = None
    run = 0
    any_known_proxy = False
    for value in proxies:
        if value is True:
            any_known_proxy = True
            run += 1
            max_streak = run if max_streak is None else max(max_streak, run)
        elif value is False:
            any_known_proxy = True
            run = 0
        else:
            run = 0  # an unknown sample breaks the streak
    if not any_known_proxy:
        max_streak = None

    first_grasp_proxy_step = None
    for index, value in enumerate(proxies):
        if value is True:
            first_grasp_proxy_step = _sample_step(sample_list[index], index)
            break

    # --- lift relative to the IMMUTABLE first sample's before-snapshot wine z ---
    # The baseline is taken ONLY from the first sample's before-snapshot wine z.
    # A later readable before-snapshot (or any after-snapshot) is never used as a
    # substitute baseline: when the first before-snapshot wine z is missing or
    # unreadable the entire lift family is unknown (``None``).  Every per-action
    # height -- including the final action -- is measured from that sample's
    # after-snapshot wine z relative to the immutable first-before reference.
    reference_z = _snapshot_wine_z(_snapshot_of(sample_list[0], "before_snapshot"))
    reference_step = _sample_step(sample_list[0], 0) if reference_z is not None else None
    max_lift: float | None = None
    first_lift_step = None
    if reference_z is not None:
        for index, sample in enumerate(sample_list):
            value = _snapshot_wine_z(_snapshot_of(sample, "after_snapshot"))
            if value is None:
                continue
            lift = value - reference_z
            if max_lift is None or lift > max_lift:
                max_lift = lift
            if first_lift_step is None and lift >= LIFT_THRESHOLD_M:
                first_lift_step = _sample_step(sample_list[index], index)

    # --- goal-true / five-consecutive-strict first steps ---
    first_goal_true_step = None
    for index, value in enumerate(goal_true):
        if value is True:
            first_goal_true_step = _sample_step(sample_list[index], index)
            break

    first_strict_five_end_step = None
    for index in range(len(strict_values) - STRICT_STREAK + 1):
        window = strict_values[index : index + STRICT_STREAK]
        if all(value is True for value in window):
            first_strict_five_end_step = _sample_step(sample_list[index + STRICT_STREAK - 1], index + STRICT_STREAK - 1)
            break

    # --- gripper command counts (sent[6] sign) ---
    sent_open = 0
    sent_close = 0
    open_goal_true = 0
    open_proxy_before = 0
    open_width_increased = 0
    n_sent_known = 0
    for index, sent in enumerate(sent_values):
        if sent is None or len(sent) <= 6:
            continue
        n_sent_known += 1
        gripper_command = sent[6]
        is_open = gripper_command < 0
        if is_open:
            sent_open += 1
            if goal_true[index] is True:
                open_goal_true += 1
            if proxy_before[index] is True:
                open_proxy_before += 1
            before_w = width_before[index]
            after_w = width_after[index]
            if (
                before_w is not None
                and after_w is not None
                and (after_w - before_w) > WIDTH_CHANGE_EPSILON_M
            ):
                open_width_increased += 1
        else:
            sent_close += 1

    # --- width range ---
    known_widths = [
        value
        for value in (width_before + width_after)
        if value is not None
    ]
    width_min = min(known_widths) if known_widths else None
    width_max = max(known_widths) if known_widths else None

    native_known = [value for value in native_values if isinstance(value, bool)]

    # A command/success count whose denominator of known values is empty is
    # unknown (``None``) -- never a fabricated zero.
    if n_sent_known == 0:
        sent_open = sent_close = None
        open_goal_true = open_proxy_before = open_width_increased = None

    return {
        "n_samples": n,
        "first_grasp_proxy_step": first_grasp_proxy_step,
        "max_grasp_proxy_streak": max_streak,
        "max_lift_m": max_lift,
        "lift_reference_m": reference_z,
        "lift_reference_step": reference_step,
        "first_lift_2cm_step": first_lift_step,
        "lift_threshold_m": LIFT_THRESHOLD_M,
        "first_goal_true_step": first_goal_true_step,
        "first_strict_five_end_step": first_strict_five_end_step,
        "strict_streak_required": STRICT_STREAK,
        "sent_open_count": sent_open,
        "sent_close_count": sent_close,
        "sent_open_while_goal_true_count": open_goal_true,
        "sent_open_while_grasp_proxy_before_count": open_proxy_before,
        "sent_known_count": n_sent_known,
        "width_min_m": width_min,
        "width_max_m": width_max,
        "width_change_epsilon_m": WIDTH_CHANGE_EPSILON_M,
        "open_command_width_increased_count": open_width_increased,
        "raw_policy_action_dim": _dimension_record(raw_values),
        "postprocessed_action_dim": _dimension_record(post_values),
        "sent_action_dim": _dimension_record(sent_values),
        "n_native_success_true": (sum(1 for value in native_known if value) if native_known else None),
        "n_native_success_known": len(native_known),
        "native_success_final": native_known[-1] if native_known else None,
        "null_counts": null_counts,
        "error_counts": error_counts,
        "grasp_proxy_limitation": PROXY_LIMITATION,
    }


# --- raw/post action capture -------------------------------------------------


class PostActionCapture:
    """Temporarily wrap ``v1._post`` so the raw action is copied *before* post.

    On enter, ``v1._post`` is replaced with a wrapper that (1) copies the raw
    policy action into ``target._wine_raw_action`` *before* the original post
    runs, so an in-place post mutation can never hide the true raw dimension;
    (2) invokes the original post exactly once; (3) copies the returned
    postprocessed action into ``target._wine_post_action``; and (4) returns the
    original post's result object unchanged.  On exit -- including on an
    exception -- the original ``_post`` is restored.  No RNG is consumed.
    """

    def __init__(self, v1: Any, target: Any) -> None:
        self._v1 = v1
        self._target = target
        self.original_post: Any = None
        self.installed = False
        self.raw_capture_count = 0
        self.post_call_count = 0

    def _wrapped_post(self, raw_action: Any) -> Any:
        # Copy the raw action BEFORE the original post may mutate it in place.
        self.raw_capture_count += 1
        try:
            self._target._wine_raw_action = _flat_float_list(raw_action)
        except Exception:  # noqa: BLE001 - never break the run on a copy failure
            self._target._wine_raw_action = None
        self.post_call_count += 1
        returned = self.original_post(raw_action)
        try:
            self._target._wine_post_action = _flat_float_list(returned)
        except Exception:  # noqa: BLE001
            self._target._wine_post_action = None
        return returned

    def __enter__(self) -> "PostActionCapture":
        self.original_post = getattr(self._v1, "_post", None)
        if self.original_post is not None:
            try:
                self._v1._post = self._wrapped_post
                self.installed = True
            except Exception:  # noqa: BLE001 - a non-settable _post stays unwrapped
                self.installed = False
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.installed:
            try:
                self._v1._post = self.original_post
            except Exception:  # noqa: BLE001
                pass
        self.installed = False
        return False


# --- step observation wrapper ------------------------------------------------


class WineStepWrapper:
    """Read-only observation wrapper around a worker-owned ``env.step``.

    ``install`` saves the current ``env.step`` and replaces it with
    :meth:`wrapped_step`; ``restore`` puts the saved bound method back.  The
    wrapped step reads the fixed-goal snapshot and gripper state immediately
    before the real action, calls the saved original step *exactly once* with the
    unchanged ``action`` object, reads both states again, records one sample via
    ``sample_sink`` and returns the *same* tuple object the original step
    returned.  It never steps/forwards/resets more than the one real action, and
    it never feeds any measurement back to the model or agent.  It is usable both
    explicitly (``install``/``restore``) and as a context manager.
    """

    def __init__(
        self,
        env: Any,
        goals: Any,
        sample_sink: Any,
        state_provider: Any = None,
    ) -> None:
        self.env = env
        self.goals = [list(goal) for goal in (goals or [])]
        self.sample_sink = sample_sink
        self.state_provider = state_provider or (lambda: {})
        self.original_step: Any = None
        self.installed = False
        self.counter = 0

    def _snapshot(self) -> Any:
        try:
            return pe.capture_snapshot(self.env, self.goals)
        except Exception as exc:  # noqa: BLE001 - keep the real failure as evidence
            return {"error": _format_exc(exc)}

    def _gripper(self) -> Any:
        try:
            return capture_gripper_state(self.env)
        except Exception as exc:  # noqa: BLE001 - keep the real failure as evidence
            return {"error": _format_exc(exc)}

    def wrapped_step(self, action: Any) -> Any:
        state = {}
        try:
            provider = self.state_provider()
            if isinstance(provider, dict):
                state = provider
        except Exception:  # noqa: BLE001
            state = {}
        before_snapshot = self._snapshot()
        gripper_before = self._gripper()
        # The real step is called exactly once, with the unchanged action object.
        result = self.original_step(action)
        self.counter += 1
        after_snapshot = self._snapshot()
        gripper_after = self._gripper()
        native_success = None
        if (
            isinstance(result, tuple)
            and len(result) >= 5
            and isinstance(result[4], dict)
        ):
            native_success = bool(result[4].get("is_success", False))
        sample = {
            "step": self.counter,
            "raw_policy_action": state.get("raw_policy_action"),
            "postprocessed_action": state.get("postprocessed_action"),
            "sent_action": _flat_float_list(action),
            "instruction": state.get("instruction"),
            "policy_input_state": state.get("policy_input_state"),
            "before_snapshot": before_snapshot,
            "after_snapshot": after_snapshot,
            "gripper_before": gripper_before,
            "gripper_after": gripper_after,
            "native_success": native_success,
        }
        try:
            self.sample_sink(sample)
        except Exception:  # noqa: BLE001 - telemetry must never break the run
            pass
        return result

    def install(self) -> "WineStepWrapper":
        self.original_step = self.env.step
        self.env.step = self.wrapped_step
        self.installed = True
        return self

    def restore(self) -> None:
        if self.installed and self.original_step is not None:
            try:
                self.env.step = self.original_step
            except Exception:  # noqa: BLE001
                pass
        self.installed = False

    def __enter__(self) -> "WineStepWrapper":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.restore()
        return False


# --- the diagnostic service --------------------------------------------------


class WineDiagnosticService(pe.DiagnosticService):
    """``pe.DiagnosticService`` with raw/post capture and wine step telemetry.

    The base worker, seeding, environment construction (one reset per session),
    ``strict=True`` model load, termination, ``events.jsonl`` and PNG/MP4 logic
    are inherited unchanged.  Only two methods are overridden:

    * ``_select_action`` -- records the true raw policy action (before post), the
      postprocessed action and the actual policy input task/state through
      :class:`PostActionCapture`;
    * ``_run_capability`` -- installs a :class:`WineStepWrapper` around this job's
      worker-owned ``env.step``, then delegates to the base implementation
      unchanged.
    """

    def __init__(self, *args: Any, completion_mode: str = COMPLETION_MODE, **kwargs: Any) -> None:
        super().__init__(*args, completion_mode=completion_mode, **kwargs)
        self._wine_raw_action: list[float] | None = None
        self._wine_post_action: list[float] | None = None
        self._wine_input_task: list[str] | None = None
        self._wine_input_state: list[float] | None = None

    # -- action selection (worker thread only) --------------------------------

    def _select_action(self, batch: Any) -> np.ndarray:
        """Select one action, capturing the raw action before post mutates it.

        The base behaviour is preserved exactly (including the lazy ``torch``
        import and the base's postprocessing); this override only *observes* the
        raw/post boundary.  The returned action is the base's object, unchanged.
        """

        self._wine_raw_action = None
        self._wine_post_action = None
        self._wine_input_task = _batch_task_text(batch)
        self._wine_input_state = _batch_state_list(batch)

        v1 = getattr(self, "_v1", None)
        if self._action_function is not None or v1 is None:
            # The existing action_function seam has no ``_post``: the returned
            # action IS the postprocessed action and the raw action is unknown.
            action = super()._select_action(batch)
            self._wine_post_action = _flat_float_list(action)
            return action

        with PostActionCapture(v1, self):
            return super()._select_action(batch)

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

        job.run_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = job.run_dir / "wine_telemetry.jsonl"
        goals = [list(goal) for goal in WINE_ORACLE_GOALS]
        samples: list[dict[str, Any]] = []
        telemetry_file = open(telemetry_path, "w", encoding="utf-8")

        def _sink(sample: dict[str, Any]) -> None:
            samples.append(sample)
            try:
                telemetry_file.write(json.dumps(sample, default=str) + "\n")
                telemetry_file.flush()
            except Exception:  # noqa: BLE001 - telemetry must never break the run
                pass

        def _provider() -> dict[str, Any]:
            return {
                "raw_policy_action": self._wine_raw_action,
                "postprocessed_action": self._wine_post_action,
                "instruction": self._wine_input_task,
                "policy_input_state": self._wine_input_state,
            }

        # The wrapper is installed BEFORE the base ``_run_capability``; the base
        # then installs its own wrapper on top.  A nested wrapper still results in
        # exactly one physical ``env.step`` per action.
        wrapper = WineStepWrapper(env, goals, _sink, _provider)
        wrapper.install()
        try:
            result = super()._run_capability(session, plan, job, capability_id)
        finally:
            wrapper.restore()
            try:
                telemetry_file.close()
            except Exception:  # noqa: BLE001
                pass

        summary = summarize_wine_samples(samples)
        diagnostic = {
            "job_id": job.job_id,
            "capability_id": capability_id,
            "condition": self._diag_condition,
            "profile": self._diag_profile_name,
            "oracle_goals": goals,
            "instruction": WINE_INSTRUCTION,
            "n_samples": summary.get("n_samples"),
            "wine_summary": summary,
            "telemetry_path": str(telemetry_path),
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "proxy_limitation": PROXY_LIMITATION,
            "assisted": False,
            "hermes_calls": 0,
        }
        try:
            pe._write_json_atomic(job.run_dir / "wine_diagnostic.json", diagnostic)
        except Exception as exc:  # noqa: BLE001
            service.log("wine diagnostic write failed: %s" % exc)
        return result


# --- campaign runner ---------------------------------------------------------


def _progress(message: str) -> None:
    sys.stderr.write("[wine-diag %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the exact bytes of the pinned modules this runner depends on."""

    digests: dict[str, str | None] = {}
    for name in ("wine_diagnostics.py", "service.py", "placement_experiments.py"):
        try:
            digests[name] = hashlib.sha256((_HERE / name).read_bytes()).hexdigest()
        except Exception:  # noqa: BLE001 - an absent/unreadable file is recorded as unknown
            digests[name] = None
    return digests


def _read_task_language(service_: WineDiagnosticService) -> dict[str, Any]:
    """Read the live native task language on the worker thread."""

    def _work() -> dict[str, Any]:
        env = service_._env
        if env is None:
            return {"ok": True, "task_language": None}
        return {"ok": True, "task_language": pe._native_task_instruction(env)}

    return service_._sync_work("task_language", _work)


def _read_wine_samples(job_public: Any) -> list[dict[str, Any]]:
    """Read one job's ``wine_telemetry.jsonl`` samples (best effort)."""

    run_dir = job_public.get("run_dir") if isinstance(job_public, dict) else None
    if not run_dir:
        return []
    path = Path(run_dir) / "wine_telemetry.jsonl"
    samples: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(record, dict):
                    samples.append(record)
    except Exception:  # noqa: BLE001 - a missing telemetry file is simply empty
        return []
    return samples


def _job_evidence(service_: WineDiagnosticService, job_id: str) -> dict[str, Any]:
    job = service_.job(job_id)
    if job is None:
        return {"job_id": job_id, "available": False}
    run_dir = Path(job.get("run_dir") or ".")
    wine_diag_path = run_dir / "wine_diagnostic.json"
    diagnostic: Any = None
    if wine_diag_path.is_file():
        try:
            diagnostic = json.loads(wine_diag_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            diagnostic = {"error": _format_exc(exc)}
    samples = _read_wine_samples(job)

    def _maybe(path: Path) -> str | None:
        return str(path) if path.is_file() else None

    return {
        "job_id": job_id,
        "available": True,
        "job": job,
        "wine_samples": samples,
        "wine_diagnostic": diagnostic,
        "run_dir": str(run_dir),
        "wine_telemetry_path": _maybe(run_dir / "wine_telemetry.jsonl"),
        "wine_diagnostic_path": _maybe(wine_diag_path),
        "events_path": _maybe(run_dir / "events.jsonl"),
        "result_path": _maybe(run_dir / "result.json"),
        "rollout_path": job.get("rollout_path"),
        "latest_png": job.get("latest_png"),
    }


def _plan_summary(plan_record: dict[str, Any]) -> dict[str, Any]:
    plan_public = plan_record.get("plan") or {}
    return {
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


def _job_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    if not evidence.get("available"):
        return {"job_id": evidence.get("job_id"), "available": False}
    job = evidence.get("job") or {}
    samples = evidence.get("wine_samples") or []
    return {
        "job_id": evidence.get("job_id"),
        "available": True,
        "state": job.get("state"),
        "ended_reason": job.get("ended_reason"),
        "error": job.get("error"),
        "success": job.get("success"),
        "instruction": job.get("instruction"),
        "steps": job.get("steps"),
        "total_steps": job.get("total_steps"),
        "completion_mode": job.get("completion_mode"),
        "phase": job.get("phase"),
        "held_objects": job.get("held_objects"),
        "completion_ready": job.get("completion_ready"),
        "state_before_sha": job.get("state_before_sha"),
        "state_after_sha": job.get("state_after_sha"),
        "run_dir": job.get("run_dir"),
        "rollout_path": evidence.get("rollout_path"),
        "latest_png": evidence.get("latest_png"),
        "wine_samples_n": len(samples),
        "wine_statistics": summarize_wine_samples(samples),
        "wine_telemetry_path": evidence.get("wine_telemetry_path"),
        "wine_diagnostic_path": evidence.get("wine_diagnostic_path"),
        "events_path": evidence.get("events_path"),
        "result_path": evidence.get("result_path"),
    }


def _wine_task_success(final_snapshot: Any) -> bool | None:
    """The fixed wine predicate over the final snapshot (``None`` when unknown)."""

    if not isinstance(final_snapshot, dict):
        return None
    predicates = final_snapshot.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        return None
    if any(value is None for value in predicates.values()):
        return None
    return all(bool(value) for value in predicates.values())


def _run_trial(
    service_: WineDiagnosticService,
    condition: str,
    pair_text: str,
    timeout: float,
    budget: int,
    model_revision: str | None,
    git_sha: str | None,
    run_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    seed_text, _, state_text = pair_text.partition(":")
    seed = int(seed_text or 0)
    init_state_index = int(state_text or 0)
    spec = CONDITION_SPECS[condition]
    trial_key = "%s_%s_%s" % (
        condition,
        str(pair_text).replace(":", "-"),
        uuid.uuid4().hex[:8],
    )
    service_._diag_condition = FINAL_ORACLE_KEY

    trial: dict[str, Any] = {
        "trial_id": trial_key,
        "condition": condition,
        "scene_id": spec["scene_id"],
        "suite": "libero_goal",
        "task_id": spec["task_id"],
        "variant": spec["variant"],
        "pair": pair_text,
        "seed": seed,
        "init_state_index": init_state_index,
        "model_revision": model_revision,
        "completion_mode": getattr(service_, "completion_mode", None),
        "profile": BASELINE_PROFILE,
        "profile_readback": None,
        "source_git_sha": git_sha,
        "source_sha256": _source_sha256(),
        "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
        "instruction": WINE_INSTRUCTION,
        "session_id": None,
        "initial_state_sha": None,
        "xml_sha": None,
        "task_language": None,
        "policy_instructions": [],
        "plan": None,
        "plan_terminal_state": None,
        "jobs": [],
        "wine_statistics": None,
        "wine_samples_n": None,
        "final_snapshot": None,
        "task_success": None,
        "strict_task_success": None,
        "artifacts": {},
        "assisted": False,
        "hermes_calls": 0,
        "contact_kinematic_proxy_limitation": PROXY_LIMITATION,
        "errors": [],
        "operational_errors": [],
        "physical_failures": [],
        "null_metrics": [],
        "timings": {},
    }
    operational = trial["operational_errors"]

    def _op(message: str) -> None:
        operational.append(message)
        trial["errors"].append(message)

    try:
        session = service_.create_session(spec["scene_id"], seed=seed, init_state_index=init_state_index)
        if not session.get("ok"):
            _op("create_session: %s" % session)
            trial["null_metrics"].append("session")
            return trial
        session_id = session["session_id"]
        trial["session_id"] = session_id
        trial["session"] = {
            "env_instance_id": session.get("env_instance_id"),
            "episode_resets": session.get("episode_resets"),
            "policy_resets": session.get("policy_resets"),
            "scene_version": session.get("scene_version"),
        }
        record = service_._sessions.get(session_id)
        trial["initial_state_sha"] = getattr(record, "initial_state_hash", None)
        trial["xml_sha"] = getattr(record, "xml_sha", None)

        language = _read_task_language(service_)
        if language.get("ok"):
            trial["task_language"] = language.get("task_language")

        profile_result = service_.configure_profile(BASELINE_PROFILE)
        trial["profile_readback"] = profile_result
        if not profile_result.get("ok"):
            _op("profile_config: %s" % profile_result)
            trial["null_metrics"].append("profile")
            return trial

        plan_record = pe._submit_and_wait(
            service_,
            session_id,
            [WINE_CAPABILITY_ID],
            int(budget),
            False,
            "%s-plan1" % trial_key,
            float(timeout),
            "wine stage diagnostic",
        )
        plan_public = plan_record.get("plan") or {}
        trial["plan"] = _plan_summary(plan_record)
        trial["plan_terminal_state"] = plan_public.get("state")

        job_ids = plan_public.get("job_ids") or []
        evidence_list = [_job_evidence(service_, job_id) for job_id in job_ids]
        trial["jobs"] = [_job_summary(evidence) for evidence in evidence_list]

        instructions: list[Any] = []
        for evidence in evidence_list:
            if evidence.get("available"):
                instruction = (evidence.get("job") or {}).get("instruction")
                if instruction is not None:
                    instructions.append(instruction)
        trial["policy_instructions"] = instructions
        for instruction in instructions:
            if instruction != WINE_INSTRUCTION:
                _op("instruction_mismatch: submitted policy instruction %r != %r" % (instruction, WINE_INSTRUCTION))

        if evidence_list:
            last = evidence_list[-1]
            trial["artifacts"] = {
                "run_dir": last.get("run_dir"),
                "wine_telemetry": last.get("wine_telemetry_path"),
                "wine_diagnostic": last.get("wine_diagnostic_path"),
                "events": last.get("events_path"),
                "result": last.get("result_path"),
                "rollout": last.get("rollout_path"),
                "latest_png": last.get("latest_png"),
            }

        # Wine statistics and the strict score use ONLY the final job's
        # fixed-goal telemetry: never stitched across jobs or trials.
        final_job_samples: list[dict[str, Any]] = []
        if evidence_list:
            final_job_samples = evidence_list[-1].get("wine_samples") or []
        trial["wine_statistics"] = summarize_wine_samples(final_job_samples)
        trial["wine_samples_n"] = len(final_job_samples)

        final_snapshot: dict[str, Any] | None = None
        if service_._active_request_id is None and service_._env is not None:
            final = service_.final_snapshot(WINE_ORACLE_GOALS)
            if final.get("ok"):
                final_snapshot = final.get("snapshot")
            else:
                _op("final_snapshot: %s" % final)
        else:
            _op(
                "final_snapshot_unavailable: active_request=%r env_present=%s"
                % (service_._active_request_id, service_._env is not None)
            )
        trial["final_snapshot"] = final_snapshot
        trial["task_success"] = _wine_task_success(final_snapshot)
        trial["strict_task_success"] = pe.strict_final_score(
            final_snapshot if isinstance(final_snapshot, dict) else None,
            [
                sample.get("after_snapshot")
                for sample in final_job_samples
                if isinstance(sample, dict)
            ],
        )

        # Operational vs physical classification: an errored job or a
        # nonterminal cancellation is operational; a budget exhaustion is an
        # ordinary physical failure that must not stop the campaign.
        for evidence in evidence_list:
            if not evidence.get("available"):
                continue
            job_public = evidence.get("job") or {}
            state = job_public.get("state")
            ended = job_public.get("ended_reason")
            detail = job_public.get("error")
            if state == "error" or ended == "error" or detail:
                _op(
                    "job_error: job=%s state=%s ended_reason=%s error=%s"
                    % (evidence.get("job_id"), state, ended, detail or "")
                )
            elif ended == "budget_exhausted":
                trial["physical_failures"].append(
                    "budget_exhausted: job=%s" % evidence.get("job_id")
                )

        if plan_record.get("submit_error"):
            _op("plan_submit: %s" % plan_record["submit_error"])
        if plan_record.get("timed_out"):
            # A timed-out plan is an incomplete trajectory: it is an operational
            # error (stops the campaign), never an ordinary physical 300-step
            # budget failure nor valid data.
            _op("plan_timeout: %s" % plan_record.get("request_id"))
        if plan_record.get("cancel_nonterminal"):
            _op("cancel_nonterminal: %s" % plan_record.get("request_id"))

        if trial["task_success"] is None:
            trial["null_metrics"].append("task_success")
        if trial["strict_task_success"] is None:
            trial["null_metrics"].append("strict_task_success")
        if trial["initial_state_sha"] is None:
            trial["null_metrics"].append("initial_state_sha")
        if trial["xml_sha"] is None:
            trial["null_metrics"].append("xml_sha")
        if trial["task_language"] is None:
            trial["null_metrics"].append("task_language")
        return trial
    except BaseException as exc:  # noqa: BLE001 - preserve the record
        _op("trial_exception: %s" % _format_exc(exc))
        return trial
    finally:
        trial["timings"]["trial_wall_s"] = round(time.monotonic() - started, 3)


def _campaign_report(
    args: argparse.Namespace,
    trials: list[dict[str, Any]],
    model_revision: str | None,
    git_sha: str | None,
    started: float,
    fatal_error: str | None,
) -> dict[str, Any]:
    output_path = Path(args.output)
    run_root = Path(args.run_root)
    return {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "metadata": {
            "output": str(output_path),
            "run_root": str(run_root),
            "conditions": list(args.conditions),
            "condition_order": [c for c in CONDITION_ORDER if c in set(args.conditions)],
            "pairs": list(args.pairs),
            "timeout_s": float(args.timeout),
            "budget_per_subgoal": int(args.budget),
            "profile": BASELINE_PROFILE,
            "completion_mode": COMPLETION_MODE,
            "model_path": service.DEFAULT_MODEL_PATH,
            "model_revision": model_revision,
            "source_revision_expected": pe.SOURCE_REVISION_EXPECTED,
            "source_git_sha": git_sha,
            "source_sha256": _source_sha256(),
            "oracle_goals": [list(goal) for goal in WINE_ORACLE_GOALS],
            "instruction": WINE_INSTRUCTION,
            "assisted": False,
            "hermes_calls": 0,
            "contact_kinematic_proxy_limitation": PROXY_LIMITATION,
            "campaign_wall_s": round(time.monotonic() - started, 3),
            "fatal_error": fatal_error,
        },
        "trials": trials,
        "aggregate": {
            "n_trials": len(trials),
            "n_operational_errors": sum(1 for t in trials if t.get("operational_errors")),
            "n_task_success": sum(1 for t in trials if t.get("task_success") is True),
            "n_task_failed": sum(1 for t in trials if t.get("task_success") is False),
            "n_strict_task_success": sum(1 for t in trials if t.get("strict_task_success") is True),
            "n_budget_exhausted": sum(1 for t in trials if t.get("physical_failures")),
            "reliability_note": (
                "explicit counts only; a single seed/state pair is not a "
                "statistical reliability claim"
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated fixed wine-stage diagnostic runner (no training, no "
            "downloads, no forced release, no HTTP server)."
        )
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
        help="absolute run root for per-trial artifacts (must not exist; created)",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["0:0", "1:1", "2:2"],
        help="seed:state pairs, e.g. 0:0",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITION_ORDER),
        default=list(CONDITION_ORDER),
        help="fixed wine conditions (native_goal9 then shared_goal8)",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="per-plan deadline in seconds")
    parser.add_argument("--budget", type=int, default=300, help="per-subgoal action budget")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isabs(args.output):
        parser.error("--output must be an absolute path")
    if not os.path.isabs(args.run_root):
        parser.error("--run-root must be an absolute path")
    if Path(args.output).exists():
        parser.error("--output already exists; refusing to overwrite %s" % args.output)
    if Path(args.run_root).exists():
        parser.error("--run-root already exists; refusing to reuse %s" % args.run_root)
    for pair in args.pairs:
        seed_text, sep, state_text = str(pair).partition(":")
        if not sep:
            parser.error("pair %r is not in seed:state form" % (pair,))
        try:
            int(seed_text)
            int(state_text)
        except ValueError:
            parser.error("pair %r has non-integer components" % (pair,))


def _run_campaign(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    git_sha = pe._git_rev_parse()
    trials: list[dict[str, Any]] = []
    fatal_error: str | None = None
    model_revision: str | None = None
    started = time.monotonic()
    persist_errors: list[str] = []
    conditions_selected = [c for c in CONDITION_ORDER if c in set(args.conditions)]

    def _persist() -> None:
        """Atomically write the current report snapshot; never raises."""

        try:
            pe._write_json_atomic(
                output_path,
                _campaign_report(args, trials, model_revision, git_sha, started, fatal_error),
            )
        except Exception as exc:  # noqa: BLE001 - a report write failure is fatal
            persist_errors.append(_format_exc(exc))
            _progress("report write failed: %s" % exc)

    diagnostic_service: WineDiagnosticService | None = None
    ready = False

    with register_native_wine_scene():
        try:
            diagnostic_service = WineDiagnosticService(
                service.DEFAULT_MODEL_PATH,
                str(run_root),
                completion_mode=COMPLETION_MODE,
            )
            diagnostic_service.start()
            deadline = time.monotonic() + READY_TIMEOUT_S
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
                fatal_error = "worker did not become ready within %.0fs" % READY_TIMEOUT_S
            model_revision = diagnostic_service.health().get("model_revision")

            if ready and model_revision != pe.SOURCE_REVISION_EXPECTED:
                # ZERO actions: the resident model revision must match the audited
                # source revision before any plan is submitted.
                fatal_error = (
                    "model_revision %r does not match SOURCE_REVISION_EXPECTED %r"
                    % (model_revision, pe.SOURCE_REVISION_EXPECTED)
                )
                ready = False

            if ready:
                # One fresh session per trial; for every seed:state pair, the fixed
                # condition order native_goal9 then shared_goal8 (filtered by CLI).
                planned = [
                    (condition, pair)
                    for pair in args.pairs
                    for condition in conditions_selected
                ]
                total = len(planned)
                for index, (condition, pair) in enumerate(planned, start=1):
                    _progress(
                        "trial %d/%d condition=%s pair=%s" % (index, total, condition, pair)
                    )
                    try:
                        trial = _run_trial(
                            diagnostic_service,
                            condition,
                            pair,
                            float(args.timeout),
                            int(args.budget),
                            model_revision,
                            git_sha,
                            run_root,
                        )
                    except BaseException as exc:  # noqa: BLE001 - preserve, then stop
                        fatal_error = _format_exc(exc)
                        trial = {
                            "trial_id": "%s_%s" % (condition, str(pair).replace(":", "-")),
                            "condition": condition,
                            "pair": pair,
                            "errors": ["trial_exception: %s" % _format_exc(exc)],
                            "operational_errors": ["trial_exception: %s" % _format_exc(exc)],
                            "null_metrics": ["trial"],
                        }
                        trials.append(trial)
                        _persist()
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
                            trial.get("plan_terminal_state"),
                            trial.get("task_success"),
                            trial.get("strict_task_success"),
                            len(trial.get("errors") or []),
                        )
                    )
                    if trial.get("operational_errors"):
                        # An operational error (job error, nonterminal cancel,
                        # instruction mismatch, submit failure, ...) stops the
                        # campaign with an explanatory fatal_error.  Ordinary
                        # budget/task failures are NOT operational and continue.
                        fatal_error = (
                            "operational error(s); stopping campaign (no continuation): %s"
                            % ("; ".join(trial["operational_errors"]),)
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
                            "close_env",
                            lambda: (diagnostic_service._close_env() or {"ok": True}),
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
    operational_failure = any(bool(trial.get("operational_errors")) for trial in trials)
    return 1 if (fatal_error or report_write_failed or operational_failure) else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return _run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
