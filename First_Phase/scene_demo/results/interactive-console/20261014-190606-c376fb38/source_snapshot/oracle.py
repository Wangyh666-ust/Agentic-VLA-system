"""Independent fixture-based oracle (standard library only).

``evaluate_case`` is a pure function: it judges a single preauthored case from
explicit predicate truth values, executed objects, a decision and optional
position maps. It never reads a plan, never consults a catalog scene hidden
expectation and never trusts agent narration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from catalog import goal_key
except ImportError:  # pragma: no cover - package style import
    from .catalog import goal_key

ORACLE_SOURCE = "preauthored_fixture"


def load_cases(path=None) -> dict[str, dict]:
    """Load the preauthored fixtures and map them by ``case_id``."""

    if path is None:
        path = Path(__file__).resolve().parent / "fixtures.json"
    fixture_path = Path(path)
    with open(fixture_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    cases: dict[str, dict] = {}
    for case in data.get("cases", []):
        cases[case["case_id"]] = case
    return cases


def _remember(key: str, missing: list[str]) -> None:
    """Append ``key`` to ``missing`` at most once, preserving first occurrence."""

    if key not in missing:
        missing.append(key)


def _evaluate_predicate(predicate, predicate_values, missing) -> bool:
    """Judge one predicate. Missing truth entries are false, never inferred."""

    if not predicate:
        return False
    if predicate[0] == "not":
        base_key = goal_key(list(predicate[1:]))
        if base_key not in predicate_values:
            _remember(base_key, missing)
            return False
        return predicate_values[base_key] is False
    key = goal_key(list(predicate))
    if key not in predicate_values:
        _remember(key, missing)
        return False
    return predicate_values[key] is True


def _coerce_coords(position):
    """Normalise a position into a list of floats, or ``None`` if unusable.

    A legal position is exactly three finite coordinates: a mapping carrying
    all of ``x``, ``y`` and ``z``, or a list/tuple of exactly three values.
    Everything else is rejected with ``None``: empty, one- and two-dimensional
    sequences, mappings missing any axis, non-numeric values, and non-finite
    values such as ``nan`` or ``inf``.
    """

    if position is None:
        return None
    if isinstance(position, dict):
        if not all(axis in position for axis in ("x", "y", "z")):
            return None
        try:
            coords = [float(position[axis]) for axis in ("x", "y", "z")]
        except (TypeError, ValueError):
            return None
    elif isinstance(position, (list, tuple)):
        if len(position) != 3:
            return None
        try:
            coords = [float(value) for value in position]
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not all(math.isfinite(value) for value in coords):
        return None
    return coords


def _evaluate_positions(protected_positions, initial_positions, final_positions):
    """Return per-object position evidence and an overall boolean."""

    initial = initial_positions or {}
    final = final_positions or {}
    checks: dict[str, dict] = {}
    all_ok = True
    for object_id, raw_threshold in (protected_positions or {}).items():
        threshold = float(raw_threshold)
        entry = {
            "ok": False,
            "threshold": threshold,
            "displacement": None,
            "reason": "",
        }
        start = _coerce_coords(initial.get(object_id))
        end = _coerce_coords(final.get(object_id))
        if start is None or end is None:
            entry["reason"] = "missing initial or final position"
            all_ok = False
        elif len(start) != len(end):
            entry["reason"] = "coordinate dimension mismatch"
            all_ok = False
        else:
            displacement = math.sqrt(sum((a - b) ** 2 for a, b in zip(start, end)))
            entry["displacement"] = displacement
            if displacement <= threshold:
                entry["ok"] = True
                entry["reason"] = "within tolerance"
            else:
                entry["reason"] = "displacement exceeds tolerance"
                all_ok = False
        checks[object_id] = entry
    return checks, all_ok


def evaluate_case(
    case,
    predicate_values,
    executed_objects,
    decision,
    initial_positions=None,
    final_positions=None,
) -> dict:
    """Purely evaluate one preauthored case.

    Goal options are an OR of ANDs; every predicate needs an explicit truth
    entry (including the base predicate of a negation). The executed objects
    must be a subset of ``allowed_objects`` and a non-execute decision must
    carry zero executed objects. Protected predicates and optional Euclidean
    protected-position limits are always checked.
    """

    values = dict(predicate_values or {})
    executed = list(executed_objects or [])
    missing: list[str] = []

    options = case.get("goal_options") or []
    goal_option_satisfied: list[bool] = []
    for option in options:
        satisfied = True
        for predicate in option:
            if not _evaluate_predicate(predicate, values, missing):
                satisfied = False
        goal_option_satisfied.append(satisfied)
    goal_ok = any(goal_option_satisfied) if options else True

    protected_satisfied = True
    for predicate in case.get("protected_goals") or []:
        if not _evaluate_predicate(predicate, values, missing):
            protected_satisfied = False

    decision_ok = decision in (case.get("allowed_decisions") or [])

    allowed_objects = set(case.get("allowed_objects") or [])
    objects_ok = set(executed).issubset(allowed_objects)
    if decision != "execute" and executed:
        objects_ok = False

    position_checks, positions_ok = _evaluate_positions(
        case.get("protected_positions") or {},
        initial_positions,
        final_positions,
    )

    task_success = bool(
        goal_ok
        and protected_satisfied
        and decision_ok
        and objects_ok
        and positions_ok
    )

    return {
        "task_success": task_success,
        "goal_option_satisfied": goal_option_satisfied,
        "protected_satisfied": protected_satisfied,
        "decision_ok": decision_ok,
        "objects_ok": objects_ok,
        "position_checks": position_checks,
        "missing_truth": missing,
        "oracle_source": ORACLE_SOURCE,
    }
