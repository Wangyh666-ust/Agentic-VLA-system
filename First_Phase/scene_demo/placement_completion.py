#!/usr/bin/env python3
"""Opt-in release-verified placement completion screening probe.

This module is deliberately standalone: it imports **neither** ``service`` nor
``placement_experiments`` (so it can be reused by both without creating an
import cycle) and it only ever *reads* the live simulator -- it never steps,
resets, applies a patch or mutates any simulator state.  It also never
evaluates a predicate itself: the caller passes the raw, already-evaluated
predicate mapping keyed by the canonical ``"|"``-joined goal string.

Why a release screen?
=====================

The native LIBERO predicate evaluator only reports *whether the goal is
satisfied*, not *whether the manipulation is finished*.  A can can read as
``in`` the basket region while the gripper is still closed around it, so a
purely predicate-based termination can succeed on a grasp that was never
released.  This probe adds a screening release check on top of the exact same
raw predicate truth:

* for every declared ``on``/``in`` placement goal the placement target object's
  named free joint velocity must be a finite six-vector whose linear norm is
  ``<= 0.02 m/s`` and angular norm ``<= 0.2 rad/s`` (i.e. released and at rest);
* **every** named-joint object in ``inner.objects_dict`` -- goal or not -- is
  probed for a grasp so a foreign object that is still held can never be hidden;
* any unreadable/invalid velocity or any unknown grasp makes ``ready`` ``None``
  ("unknown"), and unknown is never success.

Grasp-proxy limitation
======================

The grasp signal is the robosuite **contact-geom proxy**
``inner._check_grasp(gripper, obj.contact_geoms)``: it reports whether the
gripper contact geoms touch the object's contact geoms.  It is a *screening
proxy*, not a tactile or force measurement.  It can be ``True`` while the object
only brushes past the fingers and ``False`` while a stable pinch is held through
a non-contact geom, and it is undefined when the simulator does not expose the
probe.  An unknown grasp is therefore never treated as success.  This proxy is
used only as telemetry to gate completion; it is never fed back to the policy.
"""

from __future__ import annotations

import math
from typing import Any

# Released-and-at-rest tolerances (the same thresholds used by the diagnostic
# screening runner).
LINEAR_SPEED_TOLERANCE = 0.02  # m/s
ANGULAR_SPEED_TOLERANCE = 0.2  # rad/s
VELOCITY_COMPONENTS = 6  # 3 linear + 3 angular for a free joint

PLACEMENT_PREDICATES = ("on", "in")

# The allowed completion phases.  Classification is driven purely by the
# declared placement targets and their raw predicate truth; goals are never
# derived from the oracle and tactile truth is never inferred.
PHASES = (
    "holding_target_outside_goal",
    "goal_still_held",
    "holding_foreign_object",
    "not_holding",
    "unknown",
)

PROXY_LIMITATION = (
    "grasp is screened from robosuite contact geoms via inner._check_grasp, a "
    "contact-based proxy, not a tactile/force measurement; unknown grasp or "
    "velocity is never treated as completion."
)


def goal_key(goal: Any) -> str:
    """Canonical ``"|"``-joined key for one goal predicate."""

    return "|".join(str(part) for part in goal)


def _named_joint(obj: Any) -> str | None:
    """The object's first named joint, or ``None`` when it has none."""

    joints = getattr(obj, "joints", None) or []
    if joints and isinstance(joints[0], str):
        return joints[0]
    return None


def _finite_vector(values: Any, expected: int) -> list[float] | None:
    """A finite float vector of exactly ``expected`` components, or ``None``."""

    if values is None:
        return None
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(vector) != expected:
        return None
    if any(not math.isfinite(value) for value in vector):
        return None
    return vector


def _norm(vector: list[float] | None) -> float | None:
    if vector is None:
        return None
    return math.sqrt(sum(value * value for value in vector))


def probe_placement_completion(inner: Any, goals: list, predicates: dict) -> dict:
    """Screen one released-placement completion sample (read-only).

    ``goals`` are the declared goal predicates (e.g. ``["in", "soup",
    "basket_region"]``) and ``predicates`` is the raw, already-evaluated
    ``{"|".join(goal): bool}`` mapping produced by the caller.  Returns::

        {
          "ready": bool | None,
          "held_objects": list[str],
          "grasp_observation_complete": bool,
          "goal_speeds": {goal_key: {"linear_speed": float|None,
                                     "angular_speed": float|None}},
          "phase": str,   # one of PHASES
        }

    ``ready`` is ``None`` ("unknown") whenever a required observation is
    missing; unknown is never success.
    """

    goal_list = [list(goal) for goal in (goals or [])]
    predicate_map = predicates if isinstance(predicates, dict) else {}

    # Declared placement targets: only on/in goals name a physical placement
    # object whose release + rest is meaningful.
    placement: list[tuple[list, str, str]] = []
    placement_targets: list[str] = []
    for goal in goal_list:
        if len(goal) >= 2 and goal[0] in PLACEMENT_PREDICATES:
            object_id = str(goal[1])
            placement.append((goal, goal_key(goal), object_id))
            if object_id not in placement_targets:
                placement_targets.append(object_id)

    objects = getattr(inner, "objects_dict", None)
    if not isinstance(objects, dict):
        objects = {}

    robots = getattr(inner, "robots", None) or []
    robot = robots[0] if robots else None
    gripper = getattr(robot, "gripper", None) if robot is not None else None

    data = getattr(getattr(inner, "sim", None), "data", None)

    # --- grasp screen: probe EVERY named-joint object, not only the targets ---
    held_objects: list[str] = []
    grasp_unknown = False
    for object_id, obj in objects.items():
        if _named_joint(obj) is None:
            continue
        # A missing or ``None`` ``contact_geoms`` cannot be probed at all: the
        # grasp stays unknown (it is never coerced to False) and ``_check_grasp``
        # is not called for this object.
        contact_geoms = getattr(obj, "contact_geoms", None)
        grasp: bool | None = None
        if gripper is not None and contact_geoms is not None:
            try:
                raw = inner._check_grasp(gripper, contact_geoms)
            except Exception:  # noqa: BLE001 - an unavailable probe is unknown
                raw = None
            # Only a non-None probe result may be converted to bool; a None probe
            # return means the grasp is unknown, never a released (False) grasp.
            if raw is not None:
                grasp = bool(raw)
        if grasp is None:
            grasp_unknown = True
        elif grasp:
            held_objects.append(object_id)
    grasp_complete = not grasp_unknown

    # --- release screen: target free-joint velocity must be known and at rest --
    goal_speeds: dict[str, dict[str, float | None]] = {}
    velocity_unknown = False
    for _goal, key, object_id in placement:
        linear: float | None = None
        angular: float | None = None
        joint_name = _named_joint(objects.get(object_id))
        if data is not None and joint_name is not None:
            try:
                raw = data.get_joint_qvel(joint_name)
            except Exception:  # noqa: BLE001 - velocity unavailable -> unknown
                raw = None
            six = _finite_vector(raw, VELOCITY_COMPONENTS)
            if six is not None:
                linear = _norm(six[:3])
                angular = _norm(six[3:6])
        if linear is None or angular is None:
            velocity_unknown = True
        goal_speeds[key] = {"linear_speed": linear, "angular_speed": angular}

    conjunction = bool(predicate_map) and all(
        bool(value) for value in predicate_map.values()
    )

    if not placement:
        # A non-placement goal (e.g. "turn on the stove") carries no physical
        # release object: only the non-empty raw conjunction is required.
        ready: bool | None = conjunction
    elif not grasp_complete or velocity_unknown:
        ready = None
    else:
        speeds_ok = all(
            entry["linear_speed"] is not None
            and entry["angular_speed"] is not None
            and entry["linear_speed"] <= LINEAR_SPEED_TOLERANCE
            and entry["angular_speed"] <= ANGULAR_SPEED_TOLERANCE
            for entry in goal_speeds.values()
        )
        ready = bool(conjunction and not held_objects and speeds_ok)

    # --- phase classification (targets + raw predicates only) ----------------
    if not grasp_complete:
        phase = "unknown"
    else:
        target_true: dict[str, bool] = {}
        for _goal, key, object_id in placement:
            target_true[object_id] = target_true.get(object_id, False) or bool(
                predicate_map.get(key, False)
            )
        held_targets = [o for o in placement_targets if o in held_objects]
        foreign_held = [o for o in held_objects if o not in placement_targets]
        if any(target_true.get(o, False) for o in held_targets):
            phase = "goal_still_held"
        elif held_targets:
            phase = "holding_target_outside_goal"
        elif foreign_held:
            phase = "holding_foreign_object"
        else:
            phase = "not_holding"

    return {
        "ready": ready,
        "held_objects": held_objects,
        "grasp_observation_complete": bool(grasp_complete),
        "goal_speeds": goal_speeds,
        "phase": phase,
    }
