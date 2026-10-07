#!/usr/bin/env python3
"""Private recorded-action replay audit for the native wine-rack predicate.

This file is a *private*, non-executed auditor.  It replays a fixed, already
chosen set of recorded wine trials action-for-action and separately re-derives
the native wine-rack region / contact components, so a reviewer can see whether
the recorded predicate is reproducible from the recorded actions and the
installed native geometry -- without instantiating any policy, VLA or model.

The four replayed trials are the *exact ordered* selections made by the
primary -- one success control plus all three lifted-but-goal-false trials:

    (native_goal9, 0:0)   -- the one recorded success control
    (shared_goal8, 1:1)   -- a lifted-but-goal-false example
    (native_goal9, 2:2)   -- a lifted-but-goal-false example
    (shared_goal8, 2:2)   -- a lifted-but-goal-false example

They are hard-coded here on purpose.  There is deliberately NO automatic
success/phase-based selection and NO stage or causal inference anywhere in this
script: the source trial outcomes are preserved verbatim and never
re-interpreted.

What future execution does:

* it never starts the service, never loads a policy/model/VLA/weights, never
  runs HTTP, inference, training, GUI or any download;
* it builds each selected environment through the exact parent-specified
  ``service.SceneService._build_env(...)`` interface and a single
  ``env.reset(seed)``;
* it refuses to replay a trial unless the freshly built environment reproduces
  the recorded ``initial_state_sha`` EXACTLY (no substituted/reseeded state);
* it replays exactly one ``env.step`` per recorded telemetry record using the
  recorded ``sent_action`` converted to ``np.float32`` (7 values, unchanged),
  with no resampling, extra, scripted or forced action and no mid-trial reset;
* it compares the replayed observation against the recorded ``after_snapshot``
  (wine-bottle position, gripper qpos, fixed goal predicate) and re-derives the
  native region/contact components for every step;
* it writes ``replay_summary.json`` plus one ``replay.jsonl`` per selected
  trial, and exits non-zero on any mismatch, missing field or exception.

Nothing here is executed by the authoring task: the file is written, then
re-read, and compiled externally with ``py_compile`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Future execution runs inside the WSL tree where the real service lives.  The
# path is inserted before importing the sibling modules; importing them does NOT
# start the service, load a policy or touch a model (both modules stay import
# -clean and only import stdlib + numpy at import time).
_SCENE_DEMO_DIR = "/mnt/d/FYP/First_Phase/scene_demo"
if _SCENE_DEMO_DIR not in sys.path:
    sys.path.insert(0, _SCENE_DEMO_DIR)

import numpy as np  # noqa: E402

import service  # noqa: E402
import placement_experiments as pe  # noqa: E402

# --- fixed literals ----------------------------------------------------------

# The single, fixed wine oracle goal and the exact predicate key used by the
# recorded telemetry (``catalog.goal_key`` of the goal below).
WINE_GOAL = ["on", "wine_bottle_1", "wine_rack_1_top_region"]
WINE_PREDICATE_KEY = "on|wine_bottle_1|wine_rack_1_top_region"
WINE_OBJECT_ID = "wine_bottle_1"
WINE_RACK_ID = "wine_rack_1"
WINE_TARGET_SITE = "wine_rack_1_top_region"

SUITE_NAME = "libero_goal"
ACTION_DIM = 7
NUMERIC_TOLERANCE = 1e-6
REQUIRED_ROOT_TRIALS = 6

# The exact ordered selections, chosen by the primary.  NEVER derived from the
# recorded success flag, phase or any automatic criterion.
SELECTED_TRIAL_IDENTITIES = (
    ("native_goal9", "0:0"),
    ("shared_goal8", "1:1"),
    ("native_goal9", "2:2"),
    ("shared_goal8", "2:2"),
)

# Exact installed native sources whose bytes are hashed into the summary.
_INSTALLED_SOURCE_PATHS = {
    "base_predicates.py": (
        "/home/yhwang/fyp/libero_demo/venv/lib/python3.12/site-packages/libero/"
        "libero/envs/predicates/base_predicates.py"
    ),
    "base_object_states.py": (
        "/home/yhwang/fyp/libero_demo/venv/lib/python3.12/site-packages/libero/"
        "libero/envs/object_states/base_object_states.py"
    ),
    "site_object.py": (
        "/home/yhwang/fyp/libero_demo/venv/lib/python3.12/site-packages/libero/"
        "libero/envs/objects/site_object.py"
    ),
}

_THIS_SCRIPT_NAME = "replay_wine_predicate_20261007.py"


class ReplayAuditError(Exception):
    """A hard, non-substitutable replay-audit failure."""


# --- small helpers -----------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_file(path: Path) -> str | None:
    """SHA-256 of the exact bytes of ``path`` (``None`` when unreadable)."""

    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:  # noqa: BLE001 - an unreadable source is recorded as unknown
        return None


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a JSONL file into ``(records, errors)`` (dict records only)."""

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except Exception as exc:  # noqa: BLE001
                    errors.append("line %d is not valid JSON: %s" % (number, exc))
                    continue
                if not isinstance(parsed, dict):
                    errors.append("line %d is not a JSON object" % number)
                    continue
                records.append(parsed)
    except Exception as exc:  # noqa: BLE001
        errors.append("cannot read %s: %s" % (path, exc))
    return records, errors


def _write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_line(handle: Any, line: dict[str, Any]) -> None:
    try:
        handle.write(json.dumps(line, default=str) + "\n")
        handle.flush()
    except Exception:  # noqa: BLE001 - telemetry must never abort the audit
        pass


def _fvec(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values).reshape(-1)]


def _fmat(values: Any) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(values)]


def _sanitize(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in text)
    return safe or "unknown"


def _source_sha256() -> dict[str, str | None]:
    """SHA-256 of the exact installed native sources and of this private script."""

    digests: dict[str, str | None] = {}
    for name, path in _INSTALLED_SOURCE_PATHS.items():
        digests[name] = _sha256_file(Path(path))
    digests[_THIS_SCRIPT_NAME] = _sha256_file(Path(__file__).resolve())
    return digests


# --- campaign validation and exact selection ---------------------------------


def _validate_campaign(campaign: Any) -> None:
    """Require a dict root, an explicit null ``metadata.fatal_error`` and six trials."""

    if not isinstance(campaign, dict):
        raise ReplayAuditError("campaign root must be a JSON object")
    metadata = campaign.get("metadata")
    if not isinstance(metadata, dict):
        raise ReplayAuditError("campaign metadata must be a dict")
    if "fatal_error" not in metadata:
        raise ReplayAuditError("campaign metadata.fatal_error must be explicitly present")
    if metadata.get("fatal_error") is not None:
        raise ReplayAuditError(
            "campaign metadata.fatal_error must be null, got %r" % (metadata.get("fatal_error"),)
        )
    trials = campaign.get("trials")
    if not isinstance(trials, list):
        raise ReplayAuditError("campaign trials must be a list")
    if len(trials) != REQUIRED_ROOT_TRIALS:
        raise ReplayAuditError(
            "campaign must contain exactly %d root trials, got %d"
            % (REQUIRED_ROOT_TRIALS, len(trials))
        )


def _select_trials(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the exact ordered selections, refusing missing/duplicate identities."""

    trials = campaign["trials"]
    selected: list[dict[str, Any]] = []
    for condition, pair in SELECTED_TRIAL_IDENTITIES:
        matches = [
            trial
            for trial in trials
            if isinstance(trial, dict)
            and trial.get("condition") == condition
            and trial.get("pair") == pair
        ]
        if not matches:
            raise ReplayAuditError(
                "selected trial identity (condition=%r, pair=%r) is missing" % (condition, pair)
            )
        if len(matches) > 1:
            raise ReplayAuditError(
                "selected trial identity (condition=%r, pair=%r) is duplicated (%d matches)"
                % (condition, pair, len(matches))
            )
        selected.append(matches[0])
    return selected


# --- recorded-snapshot readers -----------------------------------------------


def _snapshot_position(snapshot: Any) -> list[float] | None:
    """``objects.wine_bottle_1.position`` or ``None`` when missing/unreadable."""

    if not isinstance(snapshot, dict):
        return None
    objects = snapshot.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(WINE_OBJECT_ID)
    if not isinstance(entry, dict):
        return None
    position = entry.get("position")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return None
    try:
        values = [float(v) for v in position]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _snapshot_gripper_qpos(snapshot: Any) -> list[float] | None:
    """``gripper_qpos`` or ``None`` when missing/unreadable/nonfinite."""

    if not isinstance(snapshot, dict):
        return None
    qpos = snapshot.get("gripper_qpos")
    if not isinstance(qpos, (list, tuple)) or not qpos:
        return None
    try:
        values = [float(v) for v in qpos]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _snapshot_predicate(snapshot: Any, key: str) -> bool | None:
    """The fixed goal predicate bool, or ``None`` when unknown/missing."""

    if not isinstance(snapshot, dict):
        return None
    predicates = snapshot.get("predicates")
    if not isinstance(predicates, dict):
        return None
    value = predicates.get(key)
    return value if isinstance(value, bool) else None


# --- native component audit --------------------------------------------------


def _audit_components(env: Any) -> dict[str, Any]:
    """Re-derive the native wine-rack region/contact components, exactly.

    The formulas below mirror the installed native ``SiteObject.under`` and
    ``SiteObjectState.check_ontop`` behaviour byte-for-byte: strict local-XY
    bounds, the local-Z window ``(sizeZ - 0.005, sizeZ + 0.10)`` and, on top of
    that, the parent-rack contact required by the native ``on`` predicate.  A
    missing API / property / access is raised as an error -- never replaced by a
    substituted boolean or transposed geometry.
    """

    inner = service._inner_env(env)

    sites = getattr(inner, "object_sites_dict", None)
    if not isinstance(sites, dict):
        raise ReplayAuditError("inner env exposes no object_sites_dict")
    if WINE_TARGET_SITE not in sites:
        raise ReplayAuditError("object_sites_dict has no %r" % WINE_TARGET_SITE)
    target = sites[WINE_TARGET_SITE]
    name = getattr(target, "name", None)
    if not isinstance(name, str) or not name:
        raise ReplayAuditError("target site object exposes no name")
    size = getattr(target, "size", None)
    if size is None:
        raise ReplayAuditError("target site object exposes no size")
    under = getattr(target, "under", None)
    if not callable(under):
        raise ReplayAuditError("target site object exposes no under()")

    sim = getattr(inner, "sim", None)
    data = getattr(sim, "data", None)
    if data is None:
        raise ReplayAuditError("inner env exposes no sim.data")
    get_site_xpos = getattr(data, "get_site_xpos", None)
    get_site_xmat = getattr(data, "get_site_xmat", None)
    body_xpos = getattr(data, "body_xpos", None)
    if not callable(get_site_xpos) or not callable(get_site_xmat) or body_xpos is None:
        raise ReplayAuditError("sim.data lacks the site/body readers")

    body_map = getattr(inner, "obj_body_id", None)
    if not isinstance(body_map, dict) or WINE_OBJECT_ID not in body_map:
        raise ReplayAuditError("obj_body_id has no %r" % WINE_OBJECT_ID)

    get_object = getattr(inner, "get_object", None)
    check_contact = getattr(inner, "check_contact", None)
    if not callable(get_object) or not callable(check_contact):
        raise ReplayAuditError("inner env lacks get_object/check_contact")

    sitepos = np.asarray(get_site_xpos(name), dtype=np.float64).reshape(-1)
    sitemat = np.asarray(get_site_xmat(name), dtype=np.float64)
    bottlepos = np.asarray(body_xpos[body_map[WINE_OBJECT_ID]], dtype=np.float64).reshape(-1)
    size_arr = np.asarray(size, dtype=np.float64).reshape(-1)

    if sitepos.shape != (3,) or bottlepos.shape != (3,) or size_arr.shape != (3,):
        raise ReplayAuditError("native site/body/size geometry is not three-dimensional")
    if sitemat.shape != (3, 3):
        raise ReplayAuditError("native site matrix is not 3x3")
    if not (
        np.all(np.isfinite(sitepos))
        and np.all(np.isfinite(sitemat))
        and np.all(np.isfinite(bottlepos))
        and np.all(np.isfinite(size_arr))
    ):
        raise ReplayAuditError("native geometry contains non-finite values")

    delta = sitemat @ (bottlepos - sitepos)
    region_xy = bool(np.all(np.abs(delta[:2]) < size_arr[:2]))
    region_z = bool(size_arr[2] - 0.005 < delta[2] < size_arr[2] + 0.10)
    region_under = bool(under(sitepos, sitemat, bottlepos))
    rack_contact = bool(
        check_contact(get_object(WINE_RACK_ID), get_object(WINE_OBJECT_ID))
    )
    native_wine_predicate = bool(service.eval_goal_predicate(env, list(WINE_GOAL)))

    return {
        "site_key": WINE_TARGET_SITE,
        "site_name": name,
        "sitepos": _fvec(sitepos),
        "sitemat": _fmat(sitemat),
        "bottlepos": _fvec(bottlepos),
        "delta": _fvec(delta),
        "size": _fvec(size_arr),
        "region_xy": region_xy,
        "region_z": region_z,
        "region_under": region_under,
        "rack_contact": rack_contact,
        "native_wine_predicate": native_wine_predicate,
    }


# --- one-trial replay --------------------------------------------------------


def _replay_one_trial(trial: dict[str, Any], trial_dir: Path, job_steps: Any) -> dict[str, Any]:
    """Replay one selected trial and return its per-trial audit record."""

    record: dict[str, Any] = {
        "condition": trial.get("condition"),
        "pair": trial.get("pair"),
        "trial_id": trial.get("trial_id"),
        "task_id": trial.get("task_id"),
        "seed": trial.get("seed"),
        "init_state_index": trial.get("init_state_index"),
        "telemetry_path": None,
        "initial_sha_original": trial.get("initial_state_sha"),
        "initial_sha_replayed": None,
        "initial_sha_equal": False,
        "recorded_steps": None,
        "replayed_steps": 0,
        "job_steps": job_steps,
        "counts_equal": False,
        "replay_valid": False,
        "max_position_abs_diff": None,
        "max_gripper_abs_diff": None,
        "goal_mismatch_count": None,
        "final_audited_components": None,
        # Source outcomes are preserved verbatim; they are never re-scored.
        "source_outcome": {
            "task_success": trial.get("task_success"),
            "strict_task_success": trial.get("strict_task_success"),
            "plan_terminal_state": trial.get("plan_terminal_state"),
        },
        "errors": [],
    }
    errors: list[str] = record["errors"]
    env: Any = None
    replay_file: Any = None

    try:
        # -- required construction fields -----------------------------------
        seed = trial.get("seed")
        task_id = trial.get("task_id")
        init_state_index = trial.get("init_state_index")
        for label, value in (
            ("seed", seed),
            ("task_id", task_id),
            ("init_state_index", init_state_index),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append("trial field %r must be an integer, got %r" % (label, value))
        if errors:
            return record

        # -- build exactly as the recorded session did ----------------------
        service._seed_everything(int(seed))
        env = service.SceneService._build_env(
            types.SimpleNamespace(_env_factory=None),
            suite_name=SUITE_NAME,
            task_id=int(task_id),
            seed=int(seed),
            init_state_index=int(init_state_index),
        )
        # The single logical reset, exactly as recorded (never reset again).
        env.reset(seed=int(seed))

        # -- initial-state SHA guard (never replay on mismatch/missing) -----
        replayed_sha = service.state_sha(env)
        record["initial_sha_replayed"] = replayed_sha
        original_sha = trial.get("initial_state_sha")
        if not isinstance(original_sha, str) or not original_sha:
            errors.append("trial initial_state_sha is missing; refusing to replay this trial")
            return record
        record["initial_sha_equal"] = replayed_sha == original_sha
        if replayed_sha != original_sha:
            errors.append(
                "initial-state SHA mismatch: replayed=%s original=%s; refusing to replay this trial"
                % (replayed_sha, original_sha)
            )
            return record

        # -- read the trial's exact recorded telemetry ----------------------
        artifacts = trial.get("artifacts")
        telemetry_path = artifacts.get("wine_telemetry") if isinstance(artifacts, dict) else None
        record["telemetry_path"] = telemetry_path
        if not isinstance(telemetry_path, str) or not telemetry_path:
            errors.append("trial artifacts.wine_telemetry is missing; nothing to replay")
            return record
        telemetry_file = Path(telemetry_path)
        if not telemetry_file.is_file():
            errors.append("wine_telemetry file does not exist: %s" % telemetry_path)
            return record
        records, parse_errors = _read_jsonl(telemetry_file)
        for message in parse_errors:
            errors.append("wine_telemetry: %s" % message)
        record["recorded_steps"] = len(records)

        if not isinstance(job_steps, int) or isinstance(job_steps, bool):
            errors.append("trial jobs[0].steps is missing; cannot validate the step count")

        # Every expected record must be present, in order.
        for index, sample in enumerate(records):
            if sample.get("step") != index + 1:
                errors.append(
                    "record %d carries step=%r (expected %d)" % (index, sample.get("step"), index + 1)
                )

        trial_dir.mkdir(parents=True, exist_ok=True)
        replay_file = open(trial_dir / "replay.jsonl", "w", encoding="utf-8")

        max_position_diff: float | None = None
        max_gripper_diff: float | None = None
        goal_mismatch_count = 0
        replayed_steps = 0
        last_components: dict[str, Any] | None = None

        for index, sample in enumerate(records):
            step_number = index + 1
            line: dict[str, Any] = {
                "step": sample.get("step"),  # the ORIGINAL recorded step
                "replay_step": step_number,
                "sent_action": None,
                "observed_position": None,
                "original_position": None,
                "position_abs_diff": None,
                "observed_gripper_qpos": None,
                "original_gripper_qpos": None,
                "gripper_abs_diff": None,
                "observed_predicate": None,
                "original_predicate": None,
                "predicate_match": None,
                "audited_components": None,
                "errors": [],
            }
            line_errors: list[str] = line["errors"]

            # -- the recorded action, unchanged, float32, exactly 7 values --
            action_values = sample.get("sent_action")
            try:
                action = np.asarray(action_values, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                errors.append("record %d: sent_action is unreadable: %s" % (step_number, exc))
                line_errors.append("sent_action unreadable")
                _write_line(replay_file, line)
                break
            if action.shape != (ACTION_DIM,):
                errors.append(
                    "record %d: sent_action has shape %s, expected (%d,)"
                    % (step_number, action.shape, ACTION_DIM)
                )
                line_errors.append("sent_action dimension")
                _write_line(replay_file, line)
                break
            if not np.all(np.isfinite(action)):
                errors.append("record %d: sent_action contains non-finite values" % step_number)
                line_errors.append("sent_action non-finite")
                _write_line(replay_file, line)
                break
            line["sent_action"] = _fvec(action)

            # Exactly one physical step, with the unchanged recorded action.
            env.step(action)
            replayed_steps += 1

            original_snapshot = sample.get("after_snapshot")
            try:
                observed_snapshot = pe.capture_snapshot(env, [list(WINE_GOAL)])
            except Exception as exc:  # noqa: BLE001
                observed_snapshot = None
                errors.append("record %d: capture_snapshot failed: %s" % (step_number, exc))
                line_errors.append("capture_snapshot")

            # -- wine-bottle position ---------------------------------------
            observed_position = _snapshot_position(observed_snapshot)
            original_position = _snapshot_position(original_snapshot)
            if observed_position is None or original_position is None:
                errors.append(
                    "record %d: wine_bottle_1 position unreadable (observed=%s original=%s)"
                    % (step_number, observed_position is not None, original_position is not None)
                )
                line_errors.append("position unreadable")
            else:
                diff = max(abs(o - p) for o, p in zip(observed_position, original_position))
                line["observed_position"] = observed_position
                line["original_position"] = original_position
                line["position_abs_diff"] = diff
                max_position_diff = diff if max_position_diff is None else max(max_position_diff, diff)

            # -- gripper qpos -----------------------------------------------
            observed_gripper = _snapshot_gripper_qpos(observed_snapshot)
            original_gripper = _snapshot_gripper_qpos(original_snapshot)
            if observed_gripper is None or original_gripper is None:
                errors.append(
                    "record %d: gripper_qpos unreadable (observed=%s original=%s)"
                    % (step_number, observed_gripper is not None, original_gripper is not None)
                )
                line_errors.append("gripper_qpos unreadable")
            elif len(observed_gripper) != len(original_gripper):
                errors.append(
                    "record %d: gripper_qpos length %d != recorded %d"
                    % (step_number, len(observed_gripper), len(original_gripper))
                )
                line_errors.append("gripper_qpos length")
            else:
                diff = max(abs(o - p) for o, p in zip(observed_gripper, original_gripper))
                line["observed_gripper_qpos"] = observed_gripper
                line["original_gripper_qpos"] = original_gripper
                line["gripper_abs_diff"] = diff
                max_gripper_diff = diff if max_gripper_diff is None else max(max_gripper_diff, diff)

            # -- fixed goal predicate ---------------------------------------
            observed_predicate = _snapshot_predicate(observed_snapshot, WINE_PREDICATE_KEY)
            original_predicate = _snapshot_predicate(original_snapshot, WINE_PREDICATE_KEY)
            line["observed_predicate"] = observed_predicate
            line["original_predicate"] = original_predicate
            if observed_predicate is None or original_predicate is None:
                errors.append("record %d: goal predicate unreadable" % step_number)
                line_errors.append("predicate unreadable")
            else:
                match = observed_predicate == original_predicate
                line["predicate_match"] = match
                if not match:
                    goal_mismatch_count += 1

            # -- native region/contact components ---------------------------
            try:
                components = _audit_components(env)
            except Exception as exc:  # noqa: BLE001
                components = None
                errors.append("record %d: native component audit failed: %s" % (step_number, exc))
                line_errors.append("components")
            line["audited_components"] = components
            if components is not None:
                last_components = components

            _write_line(replay_file, line)

        record["replayed_steps"] = replayed_steps
        record["max_position_abs_diff"] = max_position_diff
        record["max_gripper_abs_diff"] = max_gripper_diff
        record["goal_mismatch_count"] = goal_mismatch_count
        record["final_audited_components"] = last_components

        counts_equal = (
            record["recorded_steps"] is not None
            and isinstance(job_steps, int)
            and not isinstance(job_steps, bool)
            and record["recorded_steps"] == replayed_steps == job_steps
        )
        record["counts_equal"] = bool(counts_equal)
        if not counts_equal:
            errors.append(
                "step count mismatch: recorded=%r replayed=%r job_steps=%r"
                % (record["recorded_steps"], replayed_steps, job_steps)
            )

        record["replay_valid"] = bool(
            record["initial_sha_equal"]
            and record["counts_equal"]
            and max_position_diff is not None
            and max_position_diff <= NUMERIC_TOLERANCE
            and max_gripper_diff is not None
            and max_gripper_diff <= NUMERIC_TOLERANCE
            and goal_mismatch_count == 0
            and not errors
        )
        return record
    except BaseException as exc:  # noqa: BLE001 - keep the partial record
        errors.append("replay exception: %s" % _format_exc(exc))
        return record
    finally:
        if replay_file is not None:
            try:
                replay_file.close()
            except Exception:  # noqa: BLE001
                pass
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                errors.append("env.close failed: %s" % exc)


def _trial_dir(output_dir: Path, index: int, trial: dict[str, Any]) -> Path:
    root = output_dir.resolve()
    name = "%02d_%s_%s" % (
        index,
        _sanitize(trial.get("condition")),
        _sanitize(str(trial.get("pair", "unknown")).replace(":", "-")),
    )
    candidate = root / name
    try:
        candidate.resolve().relative_to(root)
    except ValueError:
        raise ReplayAuditError("selected trial directory escapes the output root")
    return candidate


# --- entry point -------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    campaign_path = Path(args.campaign)
    output_dir = Path(args.output_dir)

    if not campaign_path.is_file():
        sys.stderr.write("campaign path is not a readable file: %s\n" % campaign_path)
        return 1
    if output_dir.exists():
        sys.stderr.write("output directory already exists; refusing to overwrite: %s\n" % output_dir)
        return 1

    try:
        campaign = _read_json(campaign_path)
        _validate_campaign(campaign)
        selected = _select_trials(campaign)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("campaign validation failed:\n%s\n" % _format_exc(exc))
        return 1

    source_sha = _source_sha256()
    output_dir.mkdir(parents=True, exist_ok=False)

    trial_records: list[dict[str, Any]] = []
    fatal_errors: list[str] = []
    for index, trial in enumerate(selected, start=1):
        try:
            trial_dir = _trial_dir(output_dir, index, trial)
            trial_dir.mkdir(parents=True, exist_ok=True)
            jobs = trial.get("jobs")
            job_steps = (
                jobs[0].get("steps")
                if isinstance(jobs, list) and jobs and isinstance(jobs[0], dict)
                else None
            )
            record = _replay_one_trial(trial, trial_dir, job_steps)
        except Exception as exc:  # noqa: BLE001 - preserve a partial record
            message = "selected trial %d (condition=%r, pair=%r) failed:\n%s" % (
                index,
                trial.get("condition"),
                trial.get("pair"),
                _format_exc(exc),
            )
            fatal_errors.append(message)
            record = {
                "condition": trial.get("condition"),
                "pair": trial.get("pair"),
                "trial_id": trial.get("trial_id"),
                "task_id": trial.get("task_id"),
                "seed": trial.get("seed"),
                "init_state_index": trial.get("init_state_index"),
                "telemetry_path": None,
                "initial_sha_original": trial.get("initial_state_sha"),
                "initial_sha_replayed": None,
                "initial_sha_equal": False,
                "recorded_steps": None,
                "replayed_steps": 0,
                "job_steps": None,
                "counts_equal": False,
                "replay_valid": False,
                "max_position_abs_diff": None,
                "max_gripper_abs_diff": None,
                "goal_mismatch_count": None,
                "final_audited_components": None,
                "source_outcome": {
                    "task_success": trial.get("task_success"),
                    "strict_task_success": trial.get("strict_task_success"),
                    "plan_terminal_state": trial.get("plan_terminal_state"),
                },
                "errors": [message],
            }
        trial_records.append(record)
        status = "valid" if record.get("replay_valid") else "invalid"
        sys.stderr.write(
            "replay %d/%d condition=%s pair=%s -> %s\n"
            % (index, len(selected), trial.get("condition"), trial.get("pair"), status)
        )

    all_valid = bool(trial_records) and all(
        item.get("replay_valid") is True for item in trial_records
    )

    summary = {
        "script": str(Path(__file__).resolve()),
        "campaign_path": str(campaign_path),
        "output_dir": str(output_dir),
        "generated_utc": _now_utc(),
        "selected_trial_identities": [list(identity) for identity in SELECTED_TRIAL_IDENTITIES],
        "root_trial_count": len(campaign["trials"]),
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "source_paths": {
            "installed": dict(_INSTALLED_SOURCE_PATHS),
            "script": str(Path(__file__).resolve()),
        },
        "source_sha256": source_sha,
        "trials": trial_records,
        "fatal_errors": fatal_errors,
        "all_replays_valid": all_valid,
    }
    try:
        _write_json_atomic(output_dir / "replay_summary.json", summary)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("failed to write replay_summary.json:\n%s\n" % _format_exc(exc))
        return 1

    return 0 if (all_valid and not fatal_errors) else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Private recorded-action wine-rack replay audit. No policy, model, "
            "VLA, HTTP server, inference, training, GUI or download."
        )
    )
    parser.add_argument(
        "--campaign",
        required=True,
        type=str,
        help="path to the recorded campaign JSON (read-only)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="nonexistent output directory for the replay audit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
