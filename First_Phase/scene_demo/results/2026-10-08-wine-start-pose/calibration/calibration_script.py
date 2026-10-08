#!/usr/bin/env python3
"""Isolated, simulation-only native-wine start-pose measurement + fixed preparation.

This private calibration script measures the ACTUAL starting end-effector pose of
the native ``libero_goal`` wine scene (task 9, seed 0, init state 0) and then
runs the ONE fixed, already-preregistered physical preparation
(:meth:`preparation_diagnostics.PreparedService._prepare_work`) after the exact
preserved 102-action recorded bowl prefix on the shared ``libero_goal`` table
scene (task 8).

It is deliberately NOT a VLA run:

* no model/policy is constructed, no ``service.start`` / worker thread is used
  and no inference is ever performed (0 VLA actions);
* exactly two simulator environments are built directly through
  ``SceneService._build_env`` on the calling thread, each reset exactly once
  (``env.reset(seed=0)``) -- no ``set_state``, no teleport, no ``forward``, no
  object move and no extra reset;
* the recorded bowl prefix is the preserved 102-action replay performed by the
  existing :func:`skill_context_diagnostics._replay_prefix_work` helper, and the
  physical preparation is the existing
  :meth:`preparation_diagnostics.PreparedService._prepare_work` helper -- both
  reused verbatim, never re-implemented here;
* only the three fixed action bounds of the preparation are raised for this one
  isolated calibration (lift 120 / align 180 / aux cap 300); every gain, clamp,
  tolerance and the protected-object displacement threshold are left unchanged.

A physical preparation failure (a real measured servo/protection outcome) is an
allowed result and must never be reported as success, an infrastructure fault or
a VLA failure.  An unexpected exception is recorded with its traceback and exits
non-zero after the evidence has been saved.

Only stdlib + numpy are imported at import time; the accepted sibling modules are
reused rather than copied.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# The accepted, read-only scene_demo package (never modified).
sys.path.insert(0, "/mnt/c/Users/Admin1/.codex/worktrees/wine-start-pose/FYP/First_Phase/scene_demo")

import numpy as np  # noqa: E402

import preparation_diagnostics as pd  # noqa: E402
import skill_context_diagnostics as context  # noqa: E402
import service  # noqa: E402

EXPERIMENT_NAME = "wine_pose_20261008_calibration"

OUTPUT_ROOT = Path("/home/yhwang/fyp/scene_demo/skill_context/2026-10-08-pose-calibration")
ACTIONS_PATH = "/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-08-skill-context/inputs/actions.json"

NATIVE_SUITE = "libero_goal"
NATIVE_TASK_ID = 9
SHARED_SUITE = "libero_goal"
SHARED_TASK_ID = 8
SCENE_SEED = 0
INIT_STATE_INDEX = 0

# The three fixed action bounds raised ONLY for this isolated calibration.
LIFT_STAGE_MAX_ACTIONS = 120
ALIGN_STAGE_MAX_ACTIONS = 180
AUX_ACTION_CAP = 300

# The exact source modules this script reads; hashed (never written) for evidence.
SOURCE_FILES = (
    "preparation_diagnostics.py",
    "skill_context_diagnostics.py",
    "service.py",
)

CALIBRATION_FILENAME = "calibration.json"
ACCEPTANCE_FILENAME = "acceptance.json"
PREPARATION_DIRNAME = "preparation"
PREFIX_REPLAY_DIRNAME = "prefix_replay"


# --- tiny stdlib helpers ------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:  # noqa: BLE001 - an unreadable file is recorded as unknown
        return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _progress(message: str) -> None:
    sys.stderr.write("[pose-calibration %s] %s\n" % (_now_utc(), message))
    sys.stderr.flush()


def _to_jsonable(value):
    """Recursively convert NumPy/arbitrary values to strict JSON-native values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None


def _write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_jsonable(payload), indent=2, default=str, allow_nan=False)
    path.write_text(text, encoding="utf-8")


def _file_hashes(root: Path) -> dict:
    root = Path(root)
    out: dict = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = _sha256_file(path)
    return out


def _source_sha256() -> dict:
    here = Path(pd.__file__).resolve().parent
    digests: dict = {}
    for name in SOURCE_FILES:
        digests[name] = _sha256_file(here / name)
    return digests


def _command_metadata() -> dict:
    return {
        "argv": list(sys.argv),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "implemented_by": "mechanical bridge (external); DeepSeek authored this file only",
    }


# --- pose + robotics read helpers (read-only) --------------------------------


def _pose_record(pose) -> dict:
    return {
        "position": [float(v) for v in pose["position"]],
        "orientation": [[float(v) for v in row] for row in pose["orientation"]],
        "orientation_matrix": [[float(v) for v in row] for row in pose["orientation_matrix"]],
    }


def _dist3(first, second):
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.size < 3 or right.size < 3:
        return None
    distance = math.sqrt(sum((float(left[i]) - float(right[i])) ** 2 for i in range(3)))
    return distance if math.isfinite(distance) else None


def _xml_sha(env):
    """(sha256, error) of the actual ``inner.sim.model.get_xml()`` bytes."""

    try:
        inner = service._inner_env(env)
        xml = inner.sim.model.get_xml()
    except Exception as exc:  # noqa: BLE001 - an unreadable XML is unknown
        return None, "get_xml failed: %s" % exc
    try:
        data = xml.encode("utf-8") if isinstance(xml, str) else bytes(xml)
    except Exception as exc:  # noqa: BLE001
        return None, "get_xml bytes failed: %s" % exc
    return _sha256_bytes(data), None


_JOINT_POS_ATTRS = ("joint_pos", "_joint_pos", "joint_positions", "_joint_positions")
_JOINT_VEL_ATTRS = ("joint_vel", "_joint_vel", "joint_velocities", "_joint_velocities")


def _finite_floats(value):
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001 - malformed/deferred values are unknown
        return None
    if array.size == 0 or not bool(np.all(np.isfinite(array))):
        return None
    return [float(v) for v in array.tolist()]


def _read_joint_vector(robot, data, attr_names, index_attr, data_attr):
    """The actual robot joint vector (direct attribute first, index fallback)."""

    for name in attr_names:
        try:
            value = getattr(robot, name, None)
        except Exception:  # noqa: BLE001 - a raising property is unknown
            value = None
        if value is None:
            continue
        finite = _finite_floats(value)
        if finite is not None:
            return finite, "robot.%s" % name
    try:
        indexes = getattr(robot, index_attr, None)
    except Exception:  # noqa: BLE001
        indexes = None
    if indexes is not None and data is not None:
        try:
            full = np.asarray(getattr(data, data_attr), dtype=np.float64).reshape(-1)
            values = [float(full[int(i)]) for i in indexes]
        except Exception:  # noqa: BLE001
            values = None
        if values is not None:
            finite = _finite_floats(values)
            if finite is not None:
                return finite, "robot.%s->sim.data.%s" % (index_attr, data_attr)
    return None, None


def _read_robot_joints(env) -> dict:
    """The actual installed robot joint positions/velocities, or null + error."""

    record = {
        "joint_pos": None,
        "joint_vel": None,
        "n_joint_pos": None,
        "n_joint_vel": None,
        "joint_pos_source": None,
        "joint_vel_source": None,
        "error": None,
    }
    errors: list = []
    try:
        inner = service._inner_env(env)
    except Exception as exc:  # noqa: BLE001 - no inner env -> everything unknown
        record["error"] = "inner env unavailable: %s" % exc
        return record
    robots = getattr(inner, "robots", None)
    if not isinstance(robots, (list, tuple)) or len(robots) != 1:
        found = len(robots) if isinstance(robots, (list, tuple)) else None
        record["error"] = "expected exactly one robot, found %r" % (found,)
        return record
    robot = robots[0]
    data = getattr(getattr(inner, "sim", None), "data", None)

    joint_pos, pos_source = _read_joint_vector(
        robot, data, _JOINT_POS_ATTRS, "_ref_joint_pos_indexes", "qpos"
    )
    joint_vel, vel_source = _read_joint_vector(
        robot, data, _JOINT_VEL_ATTRS, "_ref_joint_vel_indexes", "qvel"
    )

    if joint_pos is None:
        errors.append("robot joint positions unavailable")
    else:
        record["joint_pos"] = joint_pos
        record["n_joint_pos"] = len(joint_pos)
        record["joint_pos_source"] = pos_source
    if joint_vel is None:
        errors.append("robot joint velocities unavailable")
    else:
        record["joint_vel"] = joint_vel
        record["n_joint_vel"] = len(joint_vel)
        record["joint_vel_source"] = vel_source
    record["error"] = "; ".join(errors) if errors else None
    return record


def _joints_record(joints: dict) -> dict:
    return {
        "joint_pos": joints.get("joint_pos"),
        "joint_vel": joints.get("joint_vel"),
        "n_joint_pos": joints.get("n_joint_pos"),
        "n_joint_vel": joints.get("n_joint_vel"),
        "joint_pos_source": joints.get("joint_pos_source"),
        "joint_vel_source": joints.get("joint_vel_source"),
        "error": joints.get("error"),
    }


def _close_env(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:  # noqa: BLE001 - the close is best effort
        pass


def _health_request() -> dict:
    """Read-only production health probe (stdlib urllib, no mutation)."""

    url = "http://127.0.0.1:8767/health"
    record = {
        "url": url,
        "method": "GET",
        "timeout_s": 10,
        "ok": False,
        "ready": None,
        "active_request_id": None,
        "error": None,
    }
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=10) as response:
            raw = response.read()
        body = json.loads(raw.decode("utf-8"))
        record["ready"] = body.get("ready")
        record["active_request_id"] = body.get("active_request_id")
        record["ok"] = (record["ready"] is True) and (
            record["active_request_id"] is None
        )
    except Exception as exc:  # noqa: BLE001 - a failed probe is a failed gate
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
    return record


def _canonical_record(record) -> dict:
    """Canonical flat pose/joint view for one nested report record."""

    record = record if isinstance(record, dict) else {}
    pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
    joints = record.get("joints") if isinstance(record.get("joints"), dict) else {}
    position = pose.get("position")
    orientation = pose.get("orientation_matrix")
    return {
        "position": [float(v) for v in position] if position is not None else None,
        "orientation": (
            [[float(v) for v in row] for row in orientation]
            if orientation is not None
            else None
        ),
        "state_sha": record.get("state_sha"),
        "joint_position": joints.get("joint_pos"),
        "joint_velocity": joints.get("joint_vel"),
    }


def _calibration_canonical_fields(report: dict) -> dict:
    """Canonical top-level calibration.json fields (shallow copy only)."""

    prep = report.get("preparation")
    compact_prep = None
    if isinstance(prep, dict):
        compact_prep = {
            key: value for key, value in prep.items()
            if key not in {"events", "before_prepare_state", "after_prepare_state"}
        }
    return {
        "native_target": _canonical_record(report.get("native")),
        "shared_initial": _canonical_record(report.get("shared_initial")),
        "after_bowl": _canonical_record(report.get("after_bowl")),
        "final_pose": _canonical_record(report.get("final")),
        "target_source": "libero_goal/9 seed0 init0 reset",
        "vla_actions": 0,
        "limits": {
            "lift": LIFT_STAGE_MAX_ACTIONS,
            "align": ALIGN_STAGE_MAX_ACTIONS,
            "total": AUX_ACTION_CAP,
        },
        "preparation": compact_prep,
    }


# --- main ---------------------------------------------------------------------


def main() -> int:
    started = time.monotonic()

    if OUTPUT_ROOT.exists():
        sys.stderr.write(
            "refusing to run: output root already exists: %s\n" % OUTPUT_ROOT
        )
        return 2

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "ok": False,
        "mode": "simulation_only_native_wine_start_pose_calibration",
        "vla_actions": 0,
        "output_root": str(OUTPUT_ROOT),
        "actions_path": ACTIONS_PATH,
        "errors": [],
        "fatal_error": None,
        "traceback": None,
        "native": {},
        "shared_initial": {},
        "after_bowl": {},
        "replay": {},
        "preparation": {},
        "final": {},
        "stage_errors": {},
        "protected_displacements": {},
        "distances": {},
        "constants": {},
        "measured": {
            "vla_actions": 0,
            "replay_action_count": None,
            "aux_actions": None,
            "replay_expected": context.REPLAY_ACTION_COUNT,
        },
        "outputs": {},
        "wall_s": None,
    }

    native_env = None
    env = None
    measured_prep = None

    try:
        # --- read-only production health gate (before env construction) --------
        health_before = _health_request()
        report["health_checks"] = {"before": health_before, "after": None}
        _require(
            health_before["ok"] is True,
            "production health gate (before) not ready: %r" % (health_before,),
        )

        # --- step 2: native task9 seed0 init0 starting pose --------------------
        service._seed_everything(0)
        native_svc = object.__new__(service.SceneService)
        native_svc._env_factory = None
        native_env = native_svc._build_env(NATIVE_SUITE, NATIVE_TASK_ID, SCENE_SEED, INIT_STATE_INDEX)
        native_obs, _native_reset_info = native_env.reset(seed=SCENE_SEED)

        native_pose = pd.read_eef_pose(native_env)
        native_state_sha = service.state_sha(native_env)
        native_xml_sha, native_xml_error = _xml_sha(native_env)
        native_joints = _read_robot_joints(native_env)

        report["native"] = {
            "suite": NATIVE_SUITE,
            "task_id": NATIVE_TASK_ID,
            "scene_seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "state_sha": native_state_sha,
            "xml_sha": native_xml_sha,
            "xml_error": native_xml_error,
            "pose": _pose_record(native_pose),
            "joints": _joints_record(native_joints),
            "target_note": (
                "this measured native starting end-effector XYZ/orientation is the "
                "preparation align target (not a robot joint restoration target)"
            ),
        }
        _progress("native task9 seed0 init0 pose measured; closing native env")
        _close_env(native_env)
        native_env = None

        # --- step 3: shared task8 seed0 init0 (replay origin) ------------------
        service._seed_everything(0)
        svc = object.__new__(service.SceneService)
        svc._env_factory = None
        env = svc._build_env(SHARED_SUITE, SHARED_TASK_ID, SCENE_SEED, INIT_STATE_INDEX)
        obs, _reset_info = env.reset(seed=SCENE_SEED)

        shared_pose = pd.read_eef_pose(env)
        shared_state_sha = service.state_sha(env)
        shared_xml_sha, shared_xml_error = _xml_sha(env)
        shared_joints = _read_robot_joints(env)

        report["shared_initial"] = {
            "suite": SHARED_SUITE,
            "task_id": SHARED_TASK_ID,
            "scene_seed": SCENE_SEED,
            "init_state_index": INIT_STATE_INDEX,
            "state_sha": shared_state_sha,
            "xml_sha": shared_xml_sha,
            "xml_error": shared_xml_error,
            "pose": _pose_record(shared_pose),
            "joints": _joints_record(shared_joints),
            "expected_replay_origin_sha": context.REPLAY_ORIGIN_SHA,
        }
        _require(
            shared_state_sha == context.REPLAY_ORIGIN_SHA,
            "shared initial state sha %r != REPLAY_ORIGIN_SHA %r"
            % (shared_state_sha, context.REPLAY_ORIGIN_SHA),
        )

        # --- the one prepared service (worker plumbing bypassed on purpose) ----
        prep_svc = object.__new__(pd.PreparedService)
        prep_svc._env = env
        prep_svc._sessions = {"cal": SimpleNamespace(total_steps=0, run_dir=OUTPUT_ROOT)}
        prep_svc._total_steps = 0
        prep_svc._last_obs = obs
        prep_svc._pre_bowl_pose = {
            "position": list(native_pose["position"]),
            "orientation_matrix": np.array(native_pose["orientation_matrix"], dtype=np.float64),
            "orientation": [list(row) for row in native_pose["orientation"]],
        }
        prep_svc._prepared_session_id = "cal"

        # --- step 4: exact recorded bowl prefix replay (102 steps, 0 VLA) ------
        actions = context.load_actions(ACTIONS_PATH)
        _require(
            len(actions) == context.REPLAY_ACTION_COUNT,
            "loaded %d actions != expected %d" % (len(actions), context.REPLAY_ACTION_COUNT),
        )
        replay = context._replay_prefix_work(
            prep_svc, "cal", actions, OUTPUT_ROOT / PREFIX_REPLAY_DIRNAME
        )
        report["replay"] = _to_jsonable(replay)
        _require(
            isinstance(replay, dict) and replay.get("ok") is True,
            "prefix replay failed: %r"
            % (replay.get("reason") if isinstance(replay, dict) else replay,),
        )
        _require(
            replay.get("final_state_sha") == context.REPLAY_FINAL_SHA,
            "replay final_state_sha %r != REPLAY_FINAL_SHA %r"
            % (replay.get("final_state_sha"), context.REPLAY_FINAL_SHA),
        )
        _require(
            replay.get("replay_action_count") == context.REPLAY_ACTION_COUNT,
            "replay_action_count %r != %d"
            % (replay.get("replay_action_count"), context.REPLAY_ACTION_COUNT),
        )
        _require(
            replay.get("vla_action_count") == 0,
            "replay reported non-zero vla_action_count %r" % (replay.get("vla_action_count"),),
        )
        report["measured"]["replay_action_count"] = replay.get("replay_action_count")
        report["measured"]["vla_actions"] = 0

        before_pose = pd.read_eef_pose(env)
        before_joints = _read_robot_joints(env)
        after_bowl_state_sha = service.state_sha(env)
        report["after_bowl"] = {
            "state_sha": after_bowl_state_sha,
            "pose": _pose_record(before_pose),
            "joints": _joints_record(before_joints),
        }

        # --- step 5: the ONE fixed physical preparation ------------------------
        pd.LIFT_STAGE_MAX_ACTIONS = LIFT_STAGE_MAX_ACTIONS
        pd.ALIGN_STAGE_MAX_ACTIONS = ALIGN_STAGE_MAX_ACTIONS
        pd.AUX_ACTION_CAP = AUX_ACTION_CAP

        report["constants"] = {
            "lift_stage_max_actions": pd.LIFT_STAGE_MAX_ACTIONS,
            "align_stage_max_actions": pd.ALIGN_STAGE_MAX_ACTIONS,
            "aux_action_cap": pd.AUX_ACTION_CAP,
            "servo_gain": pd.SERVO_GAIN,
            "servo_translation_scale": list(pd.SERVO_TRANSLATION_SCALE),
            "servo_rotation_scale": list(pd.SERVO_ROTATION_SCALE),
            "servo_translation_clamp": pd.SERVO_TRANSLATION_CLAMP,
            "servo_rotation_clamp": pd.SERVO_ROTATION_CLAMP,
            "servo_gripper_command": pd.SERVO_GRIPPER_COMMAND,
            "servo_position_tolerance_m": pd.SERVO_POSITION_TOLERANCE_M,
            "servo_rotation_tolerance_rad": pd.SERVO_ROTATION_TOLERANCE_RAD,
            "servo_success_streak": pd.SERVO_SUCCESS_STREAK,
            "lift_clear_m": pd.LIFT_CLEAR_M,
            "min_open_gap_m": pd.MIN_OPEN_GAP_M,
            "protection_tolerance_m": pd.PROTECTION_TOLERANCE_M,
            "protected_objects": list(pd.PROTECTED_OBJECTS),
            "unchanged_note": (
                "only the three fixed action bounds were raised for this isolated "
                "calibration; every gain, clamp, tolerance and the protected-object "
                "displacement threshold are unchanged"
            ),
        }

        preparation = prep_svc._prepare_work("cal", OUTPUT_ROOT / PREPARATION_DIRNAME)
        measured_prep = preparation if isinstance(preparation, dict) else None
        report["preparation"] = _to_jsonable(preparation)
        report["measured"]["aux_actions"] = (
            measured_prep.get("aux_actions") if measured_prep is not None else None
        )

        after_pose = pd.read_eef_pose(env)
        after_joints = _read_robot_joints(env)
        after_state_sha = service.state_sha(env)
        report["final"] = {
            "state_sha": after_state_sha,
            "pose": _pose_record(after_pose),
            "joints": _joints_record(after_joints),
        }

        # --- measured distances + explicit stage/protection separation ---------
        report["distances"] = {
            "native_vs_shared_initial_position_m": _dist3(
                native_pose["position"], shared_pose["position"]
            ),
            "native_vs_shared_initial_orientation_rad": pd.orientation_error_rad(
                shared_pose["orientation_matrix"], native_pose["orientation_matrix"]
            ),
            "native_vs_after_bowl_position_m": _dist3(
                native_pose["position"], before_pose["position"]
            ),
            "native_vs_after_bowl_orientation_rad": pd.orientation_error_rad(
                before_pose["orientation_matrix"], native_pose["orientation_matrix"]
            ),
            "native_vs_final_position_m": _dist3(
                native_pose["position"], after_pose["position"]
            ),
            "native_vs_final_orientation_rad": pd.orientation_error_rad(
                after_pose["orientation_matrix"], native_pose["orientation_matrix"]
            ),
        }

        if measured_prep is not None:
            lift = measured_prep.get("lift") or {}
            align = measured_prep.get("align") or {}
            report["stage_errors"] = {
                "lift": {
                    "ok": lift.get("ok"),
                    "reason": lift.get("reason"),
                    "detail": lift.get("detail"),
                    "steps": lift.get("steps"),
                },
                "align": {
                    "ok": align.get("ok"),
                    "reason": align.get("reason"),
                    "detail": align.get("detail"),
                    "steps": align.get("steps"),
                },
                "failure_step": measured_prep.get("failure_step"),
                "failed_action": measured_prep.get("failed_action"),
            }
            report["protected_displacements"] = {
                "tolerance_m": measured_prep.get("protection_tolerance_m"),
                "baseline": measured_prep.get("protection_baseline"),
                "baseline_violations": measured_prep.get("protection_baseline_violations"),
                "violations": measured_prep.get("protection_violations"),
                "final_gate_violations": measured_prep.get("final_gate_violations"),
            }

        report["ok"] = True
        _progress("measurement complete (preparation ok=%s)" % (measured_prep or {}).get("ok"))

    except BaseException as exc:  # noqa: BLE001 - preserve the real failure
        report["fatal_error"] = "%s: %s" % (type(exc).__name__, exc)
        report["traceback"] = _format_exc(exc)
        report["errors"].append(report["fatal_error"])
        _progress("measurement failed: %s" % report["fatal_error"])
    finally:
        _close_env(env)
        _close_env(native_env)

    # --- read-only production health gate (after closing owned envs) -----------
    try:
        health_after = _health_request()
        report.setdefault("health_checks", {})["after"] = health_after
        _require(
            health_after["ok"] is True,
            "production health gate (after) not ready: %r" % (health_after,),
        )
    except BaseException as exc:  # noqa: BLE001 - a failed gate is recorded
        report["errors"].append("health gate (after) failed: %s" % _format_exc(exc))
        if report.get("fatal_error") is None:
            report["fatal_error"] = "%s: %s" % (type(exc).__name__, exc)
        report["ok"] = False

    # --- outcome classification (physical failure is an allowed result) --------
    prep = measured_prep
    physical_outcome = bool(prep) and (prep.get("ok") is True or prep.get("kind") == "physical")
    report["preparation_measured"] = prep is not None
    report["preparation_physical_outcome"] = physical_outcome
    report["wall_s"] = round(time.monotonic() - started, 3)
    report["outputs"] = {
        "calibration": str(OUTPUT_ROOT / CALIBRATION_FILENAME),
        "acceptance": str(OUTPUT_ROOT / ACCEPTANCE_FILENAME),
        "preparation_dir": str(OUTPUT_ROOT / PREPARATION_DIRNAME),
        "prefix_replay_dir": str(OUTPUT_ROOT / PREFIX_REPLAY_DIRNAME),
    }

    calibration_path = OUTPUT_ROOT / CALIBRATION_FILENAME
    try:
        # Shallow copy: only calibration.json gains the canonical top-level
        # fields; the internal nested report stays intact for acceptance.
        calibration_payload = dict(report)
        calibration_payload.update(_calibration_canonical_fields(report))
        _write_json(calibration_path, calibration_payload)
    except BaseException as exc:  # noqa: BLE001 - never lose the failure
        report["errors"].append("calibration write failed: %s" % _format_exc(exc))

    preparation_summary = None
    if prep is not None:
        preparation_summary = {
            "ok": prep.get("ok"),
            "reason": prep.get("reason"),
            "kind": prep.get("kind"),
            "detail": prep.get("detail"),
            "pipeline": prep.get("pipeline"),
            "aux_actions": prep.get("aux_actions"),
            "aux_action_cap": prep.get("aux_action_cap"),
            "expected_before_state_sha": prep.get("expected_before_state_sha"),
            "before_prepare_state_sha": prep.get("before_prepare_state_sha"),
            "after_prepare_state_sha": prep.get("after_prepare_state_sha"),
            "controller_mismatch": prep.get("controller_mismatch"),
            "protection_tolerance_m": prep.get("protection_tolerance_m"),
            "protection_violations": prep.get("protection_violations"),
            "failure_step": prep.get("failure_step"),
        }

    acceptance = {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _now_utc(),
        "ok": bool(report.get("ok") and physical_outcome),
        "measurement_complete": bool(report.get("ok")),
        "preparation_physical_outcome": physical_outcome,
        "preparation_reason": prep.get("reason") if prep else None,
        "preparation_kind": prep.get("kind") if prep else None,
        "measured": dict(report["measured"]),
        "constants": dict(report["constants"]),
        "native_pose": report["native"].get("pose"),
        "shared_initial_pose": report["shared_initial"].get("pose"),
        "after_bowl_pose": report["after_bowl"].get("pose"),
        "final_pose": report["final"].get("pose"),
        "distances": report["distances"],
        "stage_errors": report["stage_errors"],
        "protected_displacements": report["protected_displacements"],
        "preparation_summary": preparation_summary,
        "target_is_measured_native_eef_pose": True,
        "no_production_http_mutation": True,
        "no_model_invocation": True,
        "source_sha256": _source_sha256(),
        "file_sha256": {},
        "command": _command_metadata(),
        "external_bridge_note": (
            "the actual subprocess command and the DeepSeek evidence are added and "
            "verified separately by the external mechanical bridge in its private "
            "acceptance; this file records only in-process measurements, command "
            "metadata and file hashes"
        ),
    }
    try:
        acceptance["file_sha256"] = _file_hashes(OUTPUT_ROOT)
    except BaseException as exc:  # noqa: BLE001
        acceptance["file_sha256_error"] = _format_exc(exc)

    try:
        _write_json(OUTPUT_ROOT / ACCEPTANCE_FILENAME, acceptance)
    except BaseException as exc:  # noqa: BLE001
        sys.stderr.write("acceptance write failed: %s\n" % _format_exc(exc))

    _progress(
        "done ok=%s physical=%s aux=%s wall=%.1fs"
        % (
            report.get("ok"),
            physical_outcome,
            report["measured"].get("aux_actions"),
            report["wall_s"] or 0.0,
        )
    )
    return 0 if (report.get("ok") and physical_outcome) else 1


if __name__ == "__main__":
    raise SystemExit(main())
