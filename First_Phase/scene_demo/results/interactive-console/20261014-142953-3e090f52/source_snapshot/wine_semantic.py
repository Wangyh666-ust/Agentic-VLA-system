#!/usr/bin/env python3
"""Fixed independent semantic wine-on-rack evaluator (``semantic_wine_rack_v1``).

This module is a *self-contained*, read-only semantic evaluator for one physical
question: **is the released wine bottle persistently resting on the wine rack
with a genuine upward-support-like physical contact?**

Operational rubric (limited -- NOT a certified safety guarantee)
===============================================================

A sample is semantically successful **only** when ALL of the following hold at
the same simulator instant:

* every jointed object in the scene is observed (the all-object grasp screen is
  complete) and **no** object is held (the wine bottle or a foreign object);
* the wine bottle free-joint linear speed is ``<= 0.02 m/s`` and its angular
  speed is ``<= 0.2 rad/s`` (released and at rest);
* there is at least one **native** rack/wine contact whose world normal has
  ``|n_z| >= 0.5`` (an upward-support-like contact), whose contact point is not
  higher than the bottle centre of mass by more than ``0.01 m`` and whose signed
  contact distance is ``<= 0.001 m``; and
* that contact carries a **finite native contact normal force ``> 0``**
  (``mujoco.mj_contactForce``).

This is deliberately a *limited operational rubric*, not a certified safety
result and not a replacement for the LIBERO benchmark.  It intentionally does
**not** require the narrow LIBERO ``on``/``in`` placement strip and does **not**
require an upright bottle orientation, so a **tilted** rack placement is
allowed.  The native standard literal predicate
(``['on', 'wine_bottle_1', 'wine_rack_1_top_region']``) is recorded separately
for display only and can **never** change the semantic result.

Independence and provenance
============================

* The evaluator never takes a Hermes plan, capability list, task goal or any
  other caller-supplied decision as input: the only input is the environment.
* The rubric and its constants are frozen in :data:`SPEC` *before* any future
  trial; a caller cannot mutate them.
* ``read_wine_semantic`` never steps, resets, forwards or otherwise mutates the
  simulator and never loads weights.  A semantic score is distinct from a job's
  ``success`` flag and can never override the VLA action termination.

Evidence reuse (and why)
========================

``read_wine_semantic`` lazily imports the existing diagnostic
``placement_experiments.capture_snapshot`` **only** for the wine object's
position/velocity and the separately-displayed standard predicate, and reuses
``placement_completion.probe_placement_completion`` **only** for its
``held_objects`` / ``grasp_observation_complete`` fields.  The completion
helper's ``ready`` result is deliberately ignored: it depends on the narrow
native predicates.  The extra reuse is required because ``capture_snapshot``
boolean-converts an unknown raw grasp (``bool(None) -> False``), whereas the
completion helper correctly treats a missing/``None`` grasp as unknown.  Missing
or unknown evidence always fails closed (``None``), never ``True``.

Only ``numpy`` and the standard library are imported at module import time; the
existing modules and the native ``mujoco`` binding are imported lazily inside
:func:`read_wine_semantic`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "SPEC",
    "SPEC_ID",
    "OBJECT_ID",
    "TARGET_ID",
    "STANDARD_GOAL",
    "STANDARD_PREDICATE_KEY",
    "OPERATIONAL_RUBRIC",
    "SemanticTracker",
    "score_wine_semantic",
    "read_wine_semantic",
]

SPEC_ID = "semantic_wine_rack_v1"
OBJECT_ID = "wine_bottle_1"
TARGET_ID = "wine_rack_1"
# The standard LIBERO predicate literal: recorded for separate display only.
STANDARD_GOAL = (("on", "wine_bottle_1", "wine_rack_1_top_region"),)
STANDARD_PREDICATE_KEY = "on|wine_bottle_1|wine_rack_1_top_region"


@dataclass(frozen=True)
class WineRackSemanticSpec:
    """Immutable fixed constants for ``semantic_wine_rack_v1``.

    Frozen *before* any future trial; a caller cannot mutate an instance.
    """

    spec_id: str
    object_id: str
    target_id: str
    standard_goal: tuple
    stable_samples: int
    linear_speed_max: float
    angular_speed_max: float
    support_abs_normal_z_min: float
    contact_height_slack_m: float
    signed_distance_max: float

    def as_dict(self) -> dict:
        """Return a plain, printable copy of the fixed constants."""

        result = asdict(self)
        result["standard_goal"] = [list(goal) for goal in self.standard_goal]
        return result


SPEC = WineRackSemanticSpec(
    spec_id=SPEC_ID,
    object_id=OBJECT_ID,
    target_id=TARGET_ID,
    standard_goal=STANDARD_GOAL,
    stable_samples=20,
    linear_speed_max=0.02,  # m/s
    angular_speed_max=0.2,  # rad/s
    support_abs_normal_z_min=0.5,
    contact_height_slack_m=0.01,  # m
    signed_distance_max=0.001,  # m
)

OPERATIONAL_RUBRIC = (
    "Limited operational rubric for semantic_wine_rack_v1: the released wine "
    "bottle is treated as semantically supported only when it is unheld and at "
    "rest (linear <= 0.02 m/s, angular <= 0.2 rad/s) and at least one native "
    "rack/wine contact has |normal_z| >= 0.5, a contact point no higher than the "
    "bottle centre of mass plus 0.01 m, a signed contact distance <= 0.001 m and "
    "a finite native contact normal force > 0. This is not a certified safety "
    "guarantee and not a benchmark replacement; it deliberately does not require "
    "the narrow LIBERO XY placement strip or an upright bottle orientation, so a "
    "tilted rack placement is allowed."
)


# --- small pure helpers ------------------------------------------------------


def _finite_vec(values: Any, expected: int) -> list[float] | None:
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


def _finite_scalar(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm(vector: list[float] | None) -> float | None:
    if vector is None:
        return None
    return math.sqrt(sum(value * value for value in vector))


def _known_bool(value: Any) -> bool | None:
    """Return ``True``/``False`` for a genuine bool, else ``None`` (unknown)."""

    if value is True or value is False:
        return bool(value)
    return None


def _frame_normal(frame: Any) -> list[float] | None:
    """First three components of a contact ``frame`` (the world normal)."""

    if frame is None:
        return None
    try:
        values = [float(value) for value in frame]
    except (TypeError, ValueError):
        return None
    if len(values) < 3:
        return None
    normal = values[:3]
    if any(not math.isfinite(value) for value in normal):
        return None
    return normal


def _import_mujoco() -> Any:
    """Lazily import the installed ``mujoco`` binding, or ``None``."""

    try:
        import mujoco
    except Exception:  # noqa: BLE001 - an absent binding is unknown evidence
        return None
    return mujoco


def _contact_normal_force(raw_model: Any, raw_data: Any, index: int) -> float | None:
    """Native contact normal force via ``mujoco.mj_contactForce``, or ``None``.

    Uses the *raw* (unwrapped) model/data when the caller passes the robosuite
    ``_model`` / ``_data`` wrappers.  Any unsupported binding or failed call is
    unknown evidence (``None``) and never assumed to be a positive force.
    """

    if raw_model is None or raw_data is None:
        return None
    mujoco = _import_mujoco()
    if mujoco is None:
        return None
    force = np.zeros(6, dtype=np.float64)
    try:
        mujoco.mj_contactForce(raw_model, raw_data, int(index), force)
    except Exception:  # noqa: BLE001 - unsupported force evidence
        return None
    if getattr(force, "size", len(force)) < 1:
        return None
    return _finite_scalar(force[0])


def _contact_geom_ids(obj: Any, model: Any) -> list[int] | None:
    """Resolve an object's ``contact_geoms`` to a list of native geom ids."""

    geoms = getattr(obj, "contact_geoms", None)
    if geoms is None:
        return None
    if isinstance(geoms, str):
        names = [geoms]
    else:
        try:
            names = list(geoms)
        except TypeError:
            return None
    ids: list[int] = []
    for geom in names:
        if isinstance(geom, bool):
            return None
        if isinstance(geom, int):
            if geom >= 0:
                ids.append(int(geom))
            continue
        if isinstance(geom, str):
            resolver = getattr(model, "geom_name2id", None)
            if not callable(resolver):
                return None
            try:
                geom_id = int(resolver(geom))
            except Exception:  # noqa: BLE001 - an unknown geom name is unknown
                return None
            if geom_id >= 0:
                ids.append(geom_id)
            continue
        return None
    if not ids:
        return None
    return ids


def _state_truth(state: Any) -> bool | None:
    """Best-effort truth of an object state, or ``None`` when it is ambiguous."""

    if isinstance(state, bool):
        return state
    if state is None:
        return None
    is_true = getattr(state, "is_true", None)
    if callable(is_true):
        try:
            return bool(is_true())
        except Exception:  # noqa: BLE001
            return None
    return None


def _object_state_parent_contact(inner: Any) -> bool | None:
    """Best-effort rack contact read from the wine object's recorded states."""

    states = getattr(inner, "object_states_dict", None)
    if not isinstance(states, dict):
        return None
    entry = states.get(SPEC.object_id)
    if not isinstance(entry, dict):
        return None
    for name, state in entry.items():
        lowered = str(name).lower()
        if "contact" not in lowered and "parent" not in lowered:
            continue
        value = _state_truth(state)
        if value is not None:
            return value
    return None


def _native_rack_contact(inner: Any, rack_obj: Any, wine_obj: Any) -> bool | None:
    """Native parent rack contact: ``inner.check_contact`` or object states."""

    check = getattr(inner, "check_contact", None)
    if callable(check) and rack_obj is not None and wine_obj is not None:
        for args in ((rack_obj, wine_obj), (wine_obj, rack_obj)):
            try:
                return bool(check(*args))
            except Exception:  # noqa: BLE001 - fall through to the state probe
                continue
    return _object_state_parent_contact(inner)


def _get_native_object(inner: Any, object_id: str) -> Any:
    """Obtain one native object/fixture via ``inner.get_object`` or ``None``.

    The confirmed LIBERO environment exposes ``wine_rack_1`` as a *fixture* that
    is absent from ``inner.objects_dict``; the only supported way to read it (or
    the wine) is the native getter ``inner.get_object(object_id)``.  A missing or
    uncallable getter, a raised exception or a ``None`` result is unknown
    evidence (``None``) and is never silently replaced by an ``objects_dict``
    lookup or an arbitrary name/catalog goal.
    """

    getter = getattr(inner, "get_object", None)
    if not callable(getter):
        return None
    try:
        return getter(object_id)
    except Exception:  # noqa: BLE001 - an unavailable getter is unknown evidence
        return None


def _read_support_contacts(
    inner: Any, wine_position: list[float] | None
) -> tuple[bool | None, bool | None, list[dict]]:
    """Read native fixture rack contact + upward-support evidence (read-only).

    The rack (``wine_rack_1``) is a LIBERO *fixture*: it is absent from
    ``inner.objects_dict`` and is obtained ONLY through the confirmed native
    getter ``inner.get_object('wine_rack_1')``; the wine is read the same way via
    ``inner.get_object('wine_bottle_1')``.  Their ``contact_geoms`` are resolved
    to native geom ids and the native parent contact is read with
    ``inner.check_contact(rack_object, wine_object)`` (the existing replay-audit
    approach).  ``objects_dict`` is deliberately NOT consulted here -- it stays
    in use only by the all-object held-screen helper.

    Returns ``(rack_contact, support_contact, contacts)`` where each value is a
    tri-state: ``True``/``False``/``None`` (unknown).  A missing/unavailable
    getter or object makes BOTH contact values ``None`` (unknown evidence), never
    a released/support success; support is never assumed to be ``True``.
    """

    rack_obj = _get_native_object(inner, SPEC.target_id)
    wine_obj = _get_native_object(inner, SPEC.object_id)
    if rack_obj is None or wine_obj is None:
        return None, None, []
    native_contact = _native_rack_contact(inner, rack_obj, wine_obj)

    sim = getattr(inner, "sim", None)
    model = getattr(sim, "model", None) if sim is not None else None
    data = getattr(sim, "data", None) if sim is not None else None
    if model is None or data is None:
        return native_contact, None, []

    rack_ids = _contact_geom_ids(rack_obj, model)
    wine_ids = _contact_geom_ids(wine_obj, model)
    if rack_ids is None or wine_ids is None:
        return native_contact, None, []

    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    sequence = getattr(raw_data, "contact", None) if raw_data is not None else None
    if sequence is None:
        sequence = getattr(data, "contact", None)
    if sequence is None:
        return native_contact, None, []

    wine_z = wine_position[2] if wine_position is not None else None
    contacts: list[dict] = []
    support = False
    evidence_missing = False
    for index, contact in enumerate(sequence):
        geom1 = _finite_int(getattr(contact, "geom1", None))
        geom2 = _finite_int(getattr(contact, "geom2", None))
        if geom1 is None or geom2 is None:
            continue
        rack_wine = (geom1 in rack_ids and geom2 in wine_ids) or (
            geom2 in rack_ids and geom1 in wine_ids
        )
        if not rack_wine:
            continue
        position = _finite_vec(getattr(contact, "pos", None), 3)
        normal = _frame_normal(getattr(contact, "frame", None))
        distance = _finite_scalar(getattr(contact, "dist", None))
        force = _contact_normal_force(raw_model, raw_data, index)
        contacts.append(
            {
                "index": index,
                "geom1": geom1,
                "geom2": geom2,
                "position": position,
                "normal": normal,
                "distance": distance,
                "normal_force": force,
            }
        )
        if position is None or normal is None or distance is None or force is None:
            evidence_missing = True
            continue
        if abs(normal[2]) < SPEC.support_abs_normal_z_min:
            continue
        if wine_z is None:
            evidence_missing = True
            continue
        if not (position[2] <= wine_z + SPEC.contact_height_slack_m):
            continue
        if not (distance <= SPEC.signed_distance_max):
            continue
        if not (force > 0.0):
            continue
        support = True

    if support:
        support_contact: bool | None = True
    elif not contacts:
        support_contact = False
    elif evidence_missing or wine_z is None:
        support_contact = None
    else:
        support_contact = False

    rack_contact: bool | None = True if contacts else native_contact
    return rack_contact, support_contact, contacts


# --- the fixed semantic score ------------------------------------------------


def score_wine_semantic(sample: Any) -> bool | None:
    """Score one raw sample against the *fixed* semantic criteria.

    Only the semantic evidence fields are consulted; any caller-supplied
    task/plan/capability/goal field is ignored, and the separately recorded
    native standard predicate never gates the result.

    * ``None`` -- required evidence is missing/nonfinite (including an
      incomplete all-object held screen, unknown rack/support contact or unknown
      velocity): fail closed, never ``True``;
    * ``False`` -- a known held object, an excessive speed, or a known-false
      rack/support contact;
    * ``True`` -- unheld, at rest and an upward-support-like native contact with
      a finite positive normal force.
    """

    if not isinstance(sample, dict):
        return None

    observation_complete = sample.get("observation_complete")
    held_objects = sample.get("held_objects")
    if observation_complete is not True or not isinstance(held_objects, list):
        return None
    if held_objects:
        return False

    linear_speed = _finite_scalar(sample.get("linear_speed"))
    angular_speed = _finite_scalar(sample.get("angular_speed"))
    if linear_speed is None or angular_speed is None:
        return None
    if linear_speed > SPEC.linear_speed_max or angular_speed > SPEC.angular_speed_max:
        return False

    rack_contact = _known_bool(sample.get("rack_contact"))
    support_contact = _known_bool(sample.get("support_contact"))
    if rack_contact is None or support_contact is None:
        return None
    if not rack_contact or not support_contact:
        return False
    return True


# --- pure streak tracker -----------------------------------------------------


class SemanticTracker:
    """Accumulate consecutive ``True`` semantic candidates (no latch).

    ``stable_samples`` consecutive ``True`` simulator samples are required to
    report ``semantic_complete``.  These are simulator samples (~1 s at 20 Hz),
    not Hermes calls.  Any unknown sample resets the streak and reports
    ``semantic_success = None``; any known-false sample resets the streak and
    reports ``False`` (a later failure un-completes an earlier success).
    """

    def __init__(self, stable_samples: int = 20) -> None:
        self.stable_samples = int(stable_samples)
        self.candidate_streak = 0
        self.semantic_success: bool | None = None

    def update(self, sample: Any) -> dict:
        value = score_wine_semantic(sample)
        if value is None:
            self.candidate_streak = 0
            self.semantic_success = None
            state = "unknown"
        elif value is False:
            self.candidate_streak = 0
            self.semantic_success = False
            state = "incomplete"
        else:
            self.candidate_streak += 1
            if self.candidate_streak >= self.stable_samples:
                self.semantic_success = True
                state = "semantic_complete"
            else:
                self.semantic_success = False
                state = "incomplete"
        return {
            "state": state,
            "candidate_streak": self.candidate_streak,
            "semantic_success": self.semantic_success,
        }


# --- the read-only probe -----------------------------------------------------


def _null_sample(error: str | None = None) -> dict:
    sample = {
        "spec_id": SPEC.spec_id,
        "object_id": SPEC.object_id,
        "target_id": SPEC.target_id,
        "rack_contact": None,
        "support_contact": None,
        "linear_speed": None,
        "angular_speed": None,
        "held_objects": None,
        "observation_complete": False,
        "standard_predicate": None,
        "semantic_candidate": None,
        "contacts": [],
    }
    if error is not None:
        sample["error"] = error
    return sample


def read_wine_semantic(env: Any) -> dict:
    """Read one raw semantic sample from the live environment (read-only).

    Never steps/resets/forwards the simulator and never loads weights.  Missing
    or unknown evidence yields ``semantic_candidate = None`` (fail closed); the
    native standard predicate is recorded separately and never gates the score.
    """

    try:
        import placement_completion as pc
        import placement_experiments as pe
        import service as svc
    except Exception as exc:  # noqa: BLE001 - a missing module is unknown evidence
        return _null_sample("lazy import failed: %s" % exc)

    goals = [list(goal) for goal in SPEC.standard_goal]
    key = STANDARD_PREDICATE_KEY

    try:
        snapshot = pe.capture_snapshot(env, goals)
    except Exception as exc:  # noqa: BLE001 - a failed read is unknown evidence
        return _null_sample("capture_snapshot failed: %s" % exc)

    objects = snapshot.get("objects")
    entry = objects.get(SPEC.object_id) if isinstance(objects, dict) else None
    if not isinstance(entry, dict):
        entry = {}
    position = _finite_vec(entry.get("position"), 3)
    linear_speed = _norm(_finite_vec(entry.get("linear_velocity"), 3))
    angular_speed = _norm(_finite_vec(entry.get("angular_velocity"), 3))

    predicates = snapshot.get("predicates")
    standard_predicate = (
        _known_bool(predicates.get(key)) if isinstance(predicates, dict) else None
    )

    try:
        inner = svc._inner_env(env)
    except Exception as exc:  # noqa: BLE001
        sample = _null_sample("inner env unavailable: %s" % exc)
        sample["standard_predicate"] = standard_predicate
        sample["linear_speed"] = linear_speed
        sample["angular_speed"] = angular_speed
        return sample

    # --- all-object held screen: reuse the completion helper, held fields only
    held_objects: list | None = None
    observation_complete = False
    try:
        completion = pc.probe_placement_completion(inner, goals, {key: standard_predicate})
    except Exception:  # noqa: BLE001 - an unavailable screen fails closed
        completion = None
    if isinstance(completion, dict):
        candidate_held = completion.get("held_objects")
        if isinstance(candidate_held, list):
            held_objects = candidate_held
        observation_complete = bool(completion.get("grasp_observation_complete"))

    rack_contact, support_contact, contacts = _read_support_contacts(inner, position)

    sample = {
        "spec_id": SPEC.spec_id,
        "object_id": SPEC.object_id,
        "target_id": SPEC.target_id,
        "rack_contact": rack_contact,
        "support_contact": support_contact,
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
        "held_objects": held_objects,
        "observation_complete": observation_complete,
        "standard_predicate": standard_predicate,
        "semantic_candidate": None,
        "contacts": contacts,
    }
    sample["semantic_candidate"] = score_wine_semantic(sample)
    return sample
