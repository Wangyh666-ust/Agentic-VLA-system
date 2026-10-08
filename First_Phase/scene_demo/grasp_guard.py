#!/usr/bin/env python3
"""Wine-only grasp monitor for the persistent-scene execution service.

This module is deliberately standalone and **stdlib + numpy only**: it imports
neither ``service`` nor ``placement_experiments`` (so it can be reused by the
service without creating an import cycle) and it only ever *reads* the live
simulator -- it never steps, resets, teleports or mutates any simulator state.

The guard is a *wine-only candidate screen*: the thresholds below are
candidate values pending validation for the single wine bottle / wine rack
capability -- they were never measured or calibrated from a single wine trial
and are not a universal skill truth.  They screen one very specific failure
mode -- a "ghost" grasp where the end-effector retreats upward away from a
bottle that never actually rose, so the goal predicate can never become true --
and must not be generalised to other objects without re-validation.

Grasp-proxy limitation
======================

The grasp signal is the robosuite **contact-geom proxy**
``inner._check_grasp(gripper, obj.contact_geoms)``: it reports whether the
gripper contact geoms touch the object's contact geoms.  It is a *screening
proxy*, not a tactile or force measurement, and it is undefined when the
simulator does not expose the probe.  An unknown grasp is therefore never
treated as a confirmation or a failure.  The gripper jaw gap is *supplementary
evidence only* and never identifies the object on its own.

The monitor is a privileged diagnostic reader.  Its status is never fed back to
the policy or to Hermes, and the service only ever *stops* a failing wine job
(it never fabricates a success).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- wine-only fixed identity ------------------------------------------------
#
# The guard is enabled for exactly one literal goal: the wine bottle onto the
# wine rack top region.  The keys are written literally here (never derived
# from the capability schedule) so an unrelated object can never be screened.
WINE_OBJECT_ID = "wine_bottle_1"
WINE_TARGET_ID = "wine_rack_1_top_region"
WINE_GOAL_KEY = "on|wine_bottle_1|wine_rack_1_top_region"

# The monitor stages.  ``failed_grasp`` is the only stage whose ``should_stop``
# status is true; ``already_placed`` is the disabled state used when the target
# goal is already satisfied before any action.
STAGES = (
    "approaching",
    "attempting",
    "grasp_confirmed",
    "failed_grasp",
    "unknown",
    "already_placed",
)


@dataclass(frozen=True)
class GuardConfig:
    """Frozen wine-only candidate thresholds (metres / samples), pending validation.

    These are candidate values pending validation -- never measured or
    calibrated from a single wine trial -- and are not a universal skill truth.
    """

    near_distance_m: float = 0.14
    lift_success_m: float = 0.02
    bottle_still_m: float = 0.01
    retreat_rise_m: float = 0.03
    separation_growth_m: float = 0.05
    confirm_samples: int = 5
    failure_samples: int = 5


# --- small pure helpers ------------------------------------------------------


def _finite(value: Any) -> float | None:
    """A finite float, or ``None`` for a missing/malformed/nonfinite value."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _vec3(value: Any) -> list[float] | None:
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


def _euclidean(first: list[float] | None, second: list[float] | None) -> float | None:
    """The finite Euclidean distance between two 3-vectors, or ``None``."""

    if first is None or second is None:
        return None
    squared = sum((first[index] - second[index]) ** 2 for index in range(3))
    distance = math.sqrt(squared)
    if not math.isfinite(distance):
        return None
    return distance


def _object_entry(snapshot: Any, object_id: str) -> dict | None:
    if not isinstance(snapshot, dict):
        return None
    objects = snapshot.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(object_id)
    return entry if isinstance(entry, dict) else None


def _object_position(snapshot: Any, object_id: str) -> list[float] | None:
    entry = _object_entry(snapshot, object_id)
    if entry is None:
        return None
    return _vec3(entry.get("position"))


def _object_grasped(snapshot: Any, object_id: str) -> bool | None:
    entry = _object_entry(snapshot, object_id)
    if entry is None:
        return None
    value = entry.get("grasped")
    return value if isinstance(value, bool) else None


def _snapshot_eef(snapshot: Any) -> list[float] | None:
    if not isinstance(snapshot, dict):
        return None
    return _vec3(snapshot.get("eef_position"))


def _snapshot_predicate(snapshot: Any, key: str) -> bool | None:
    if not isinstance(snapshot, dict):
        return None
    predicates = snapshot.get("predicates")
    if not isinstance(predicates, dict) or key not in predicates:
        return None
    value = predicates.get(key)
    return value if isinstance(value, bool) else None


# --- the stateful monitor ----------------------------------------------------


class GraspMonitor:
    """Stateful wine-only grasp monitor (never modifies the simulator).

    ``start`` fixes the initial bottle z baseline forever and records whether the
    target goal is already satisfied (which disables the guard).  ``update`` is
    called once per real action with a read-only snapshot and the sent gripper
    command, and returns the status mapping.  Confirmation latches permanently;
    a later intentional release can never become ``failed_grasp``.
    """

    def __init__(self, config: GuardConfig = GuardConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._started = False
        self._initial_z: float | None = None
        self._disabled = False
        self._confirmed_step: int | None = None
        self._failure_step: int | None = None
        self._armed_step: int | None = None
        self._armed_eef_z: float | None = None
        self._armed_distance: float | None = None
        self._confirm_streak = 0
        self._candidate_streak = 0
        self._stage = "unknown"

    def start(self, snapshot: Any) -> dict:
        """Fix the baseline from ``snapshot`` and return the initial status."""

        self.reset()
        self._initial_z = None
        position = _object_position(snapshot, WINE_OBJECT_ID)
        if position is not None:
            self._initial_z = position[2]
        self._disabled = _snapshot_predicate(snapshot, WINE_GOAL_KEY) is True
        self._started = True
        self._stage = "already_placed" if self._disabled else "unknown"
        return self.status()

    def _arm(self, step: int, command: float | None, distance: float | None,
             eef: list[float] | None) -> None:
        """Arm once -- with a finite positive command and a near object."""

        if self._armed_step is not None:
            return
        if (
            command is not None
            and command > 0.0
            and distance is not None
            and distance <= self.config.near_distance_m
            and eef is not None
        ):
            self._armed_step = int(step)
            # The arm EEF z and separation are stored once and never overwritten.
            self._armed_eef_z = eef[2]
            self._armed_distance = distance

    def update(self, step: int, snapshot: Any, gripper_command: Any) -> dict:
        """Screen one real action sample and return the current status."""

        if not self._started:
            self._confirm_streak = 0
            self._candidate_streak = 0
            self._stage = "unknown"
            return self.status()

        if self._disabled:
            self._stage = "already_placed"
            return self.status()

        # Latched terminal stages: neither is ever recomputed.
        if self._confirmed_step is not None:
            self._stage = "grasp_confirmed"
            return self.status()
        if self._failure_step is not None:
            self._stage = "failed_grasp"
            return self.status()

        command = _finite(gripper_command)
        position = _object_position(snapshot, WINE_OBJECT_ID)
        eef = _snapshot_eef(snapshot)
        grasped = _object_grasped(snapshot, WINE_OBJECT_ID)
        predicate = _snapshot_predicate(snapshot, WINE_GOAL_KEY)
        distance = _euclidean(eef, position)

        self._arm(step, command, distance, eef)
        armed = self._armed_step is not None

        # Bottle rise from the immutable initial-z baseline (unknown when either
        # the baseline or the current position is unreadable/nonfinite).
        lift: float | None = None
        if position is not None and self._initial_z is not None:
            lift = position[2] - self._initial_z
            if not math.isfinite(lift):
                lift = None

        # Confirmation: a real lift above the threshold while the bottle is
        # grasped, held for ``confirm_samples`` consecutive known samples.
        confirm_ok = (
            lift is not None
            and lift >= self.config.lift_success_m
            and grasped is True
        )

        # Failure candidate (before confirmation only): the gripper has retreated
        # away while the bottle never rose and the native wine goal stays false.
        failure_ok = False
        if armed and self._confirmed_step is None:
            eef_rise: float | None = None
            if eef is not None and self._armed_eef_z is not None:
                eef_rise = eef[2] - self._armed_eef_z
            separation_growth: float | None = None
            if distance is not None and self._armed_distance is not None:
                separation_growth = distance - self._armed_distance
            failure_ok = (
                grasped is False
                and lift is not None
                and lift < self.config.bottle_still_m
                and eef_rise is not None
                and eef_rise >= self.config.retreat_rise_m
                and separation_growth is not None
                and separation_growth >= self.config.separation_growth_m
                and predicate is False
            )

        # Unknown/missing/nonfinite samples satisfy neither condition and reset
        # the relevant streak, so they can never yield a confirmation or failure.
        if confirm_ok:
            self._confirm_streak += 1
            self._candidate_streak = 0
        elif failure_ok:
            self._candidate_streak += 1
            self._confirm_streak = 0
        else:
            self._confirm_streak = 0
            self._candidate_streak = 0

        if self._confirm_streak >= self.config.confirm_samples:
            self._confirmed_step = int(step)
            self._candidate_streak = 0
            self._stage = "grasp_confirmed"
        elif self._candidate_streak >= self.config.failure_samples:
            self._failure_step = int(step)
            self._stage = "failed_grasp"
        elif armed:
            self._stage = "attempting"
        elif distance is not None:
            self._stage = "approaching"
        else:
            self._stage = "unknown"

        return self.status()

    def status(self) -> dict:
        return {
            "stage": self._stage,
            "armed_step": self._armed_step,
            "grasp_confirmed_step": self._confirmed_step,
            "failure_step": self._failure_step,
            "candidate_streak": self._candidate_streak,
            "should_stop": self._stage == "failed_grasp",
        }


# --- read-only probe ---------------------------------------------------------


def _inner_env(env: Any) -> Any:
    """Return the robosuite inner env wrapped by ``LiberoEnv`` (or ``None``)."""

    try:
        inner = getattr(getattr(env, "_env", None), "env", None)
    except Exception:  # noqa: BLE001 - a defensive attribute read
        return None
    return inner


def _read_site_xpos(data: Any, model: Any, site_name: str) -> Any:
    """Read a named site position via the native method or the ``site_xpos`` map.

    The native robosuite ``sim.data`` exposes ``get_site_xpos(name)``; the array
    form ``sim.data.site_xpos[sim.model.site_name2id(name)]`` is accepted as a
    compatible fallback.  Only *reads* happen here -- the simulator is never
    stepped, reset or forwarded.
    """

    getter = getattr(data, "get_site_xpos", None)
    if callable(getter):
        try:
            return getter(site_name)
        except Exception:  # noqa: BLE001 - fall through to the array form
            pass
    site_xpos = getattr(data, "site_xpos", None)
    if site_xpos is None:
        return None
    name2id = getattr(model, "site_name2id", None)
    if not callable(name2id):
        return None
    try:
        index = int(name2id(site_name))
        return np.asarray(site_xpos)[index]
    except Exception:  # noqa: BLE001
        return None


def _empty_probe(object_id: str, goal_key: str) -> dict:
    return {
        "objects": {object_id: {"position": None, "grasped": None}},
        "eef_position": None,
        "predicates": {goal_key: None},
        "gripper_qpos": None,
        "gap": None,
    }


def read_probe(env: Any, object_id: str, goal_key: str) -> dict:
    """Read one wine-only, snapshot-compatible probe from the live simulator.

    The returned mapping mirrors the relevant fields of
    ``placement_experiments.capture_snapshot`` -- ``objects[object_id]`` with a
    finite ``position`` and a contact-proxy ``grasped``, the gripper
    ``eef_position``, ``predicates[goal_key]`` and the ``gripper_qpos`` plus a
    finite jaw ``gap``.  Every unreadable/malformed/nonfinite field is left
    ``None`` ("unknown") and is never invented as a ``False``.  The jaw gap is
    supplementary evidence only and never identifies the object on its own.  It
    never steps, resets or teleports the simulator.
    """

    probe = _empty_probe(object_id, goal_key)
    inner = _inner_env(env)
    if inner is None:
        return probe

    sim = getattr(inner, "sim", None)
    data = getattr(sim, "data", None)
    model = getattr(sim, "model", None)
    objects = getattr(inner, "objects_dict", None)
    if not isinstance(objects, dict):
        objects = {}
    robots = getattr(inner, "robots", None) or []
    robot = robots[0] if robots else None
    gripper = getattr(robot, "gripper", None) if robot is not None else None

    # --- object position (native named free joint) + contact-proxy grasp ---
    obj = objects.get(object_id)
    position: list[float] | None = None
    grasped: bool | None = None
    if obj is not None:
        joints = getattr(obj, "joints", None) or []
        if data is not None and joints and isinstance(joints[0], str):
            try:
                qpos = np.asarray(data.get_joint_qpos(joints[0]), dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001 - unreadable qpos stays unknown
                qpos = None
            if qpos is not None and qpos.size >= 3 and bool(np.all(np.isfinite(qpos[:3]))):
                position = [float(value) for value in qpos[:3]]
        contact_geoms = getattr(obj, "contact_geoms", None)
        # A missing/None/empty contact-geom set means the proxy is undefined for
        # this object: the grasp stays unknown and ``_check_grasp`` is never
        # invoked, because "no contact geoms" must never be read as "not held".
        contact_geoms_available = False
        if contact_geoms is not None:
            try:
                contact_geoms_available = len(contact_geoms) > 0
            except TypeError:
                contact_geoms_available = True
        check = getattr(inner, "_check_grasp", None)
        if gripper is not None and contact_geoms_available and callable(check):
            try:
                raw_grasp = check(gripper, contact_geoms)
            except Exception:  # noqa: BLE001 - unavailable probe stays unknown
                raw_grasp = None
            # Only a genuine bool (or numpy bool) is accepted; any other return
            # (None, int, array, ...) stays unknown and is never coerced to a
            # Python truth value.
            if isinstance(raw_grasp, (bool, np.bool_)):
                grasped = bool(raw_grasp)
    probe["objects"][object_id] = {"position": position, "grasped": grasped}

    # --- end-effector position (gripper grip site) ---
    eef: list[float] | None = None
    if gripper is not None and data is not None:
        important_sites = getattr(gripper, "important_sites", None)
        site_name = important_sites.get("grip_site") if isinstance(important_sites, dict) else None
        if site_name is not None:
            raw_site = _read_site_xpos(data, model, site_name)
            if raw_site is not None:
                site = np.asarray(raw_site, dtype=np.float64).reshape(-1)
                if site.size >= 3 and bool(np.all(np.isfinite(site[:3]))):
                    eef = [float(value) for value in site[:3]]
    probe["eef_position"] = eef

    # --- gripper jaw qpos + supplementary finite gap ---
    qpos_values: list[float] | None = None
    if robot is not None and data is not None:
        indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
        if indexes is not None:
            try:
                full_qpos = np.asarray(data.qpos, dtype=np.float64).reshape(-1)
                values = [float(full_qpos[int(index)]) for index in indexes]
            except Exception:  # noqa: BLE001
                values = None
            if values and all(math.isfinite(value) for value in values):
                qpos_values = values
    probe["gripper_qpos"] = qpos_values
    if qpos_values is not None and len(qpos_values) == 2:
        probe["gap"] = abs(qpos_values[0] - qpos_values[1])

    # --- native wine-target predicate ---
    predicate: bool | None = None
    evaluator = getattr(inner, "_eval_predicate", None)
    if callable(evaluator) and isinstance(goal_key, str) and goal_key:
        try:
            raw_predicate = evaluator(goal_key.split("|"))
        except Exception:  # noqa: BLE001 - unknown truth is never success
            raw_predicate = None
        if raw_predicate is not None:
            predicate = bool(raw_predicate)
    probe["predicates"][goal_key] = predicate

    return probe
