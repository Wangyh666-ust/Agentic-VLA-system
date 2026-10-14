"""Audited/resumable wrapper around the original X-VLA LIBERO evaluation.

This runner measures 40 native LIBERO tasks x 5 episodes without altering
inference, control or scoring. It inherits the pinned author ClientModel and
LIBEROEval behaviors, adding only observation, persistence and audit records.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import json_numpy
from PIL import Image

import evaluation.libero.libero_client as author

SUITES = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
LIMITS = {"libero_10": 900, "libero_spatial": 800, "libero_goal": 800, "libero_object": 800}
HOST = "127.0.0.1"
PORT = 8794

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "First_Phase/scene_demo/plans/20261012-xvla-native-benchmark.json"
OUT_DEFAULT = ROOT / "First_Phase/scene_demo/results/2026-10-12-xvla-native-benchmark"


class StopRequested(Exception):
    pass


class ContractError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config() -> dict:
    return json.loads(PROPOSAL.read_text(encoding="utf-8"))["configuration"]


def episode_slots() -> list[tuple[str, int, int]]:
    slots = []
    for suite in SUITES:
        for task_id in range(10):
            for episode in range(5):
                slots.append((suite, task_id, episode))
    return slots


def validate_episode(row: dict, horizon: int) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("status") != "completed" or row.get("valid") is not True:
        return False
    success = row.get("success")
    author_score = row.get("author_score")
    independent = row.get("independent_success")
    if not isinstance(success, bool) or author_score not in (0, 1) or not isinstance(independent, bool):
        return False
    if bool(author_score) != bool(success) or bool(independent) != bool(success):
        return False
    if type(author_score) is not int:
        return False
    if not all(type(row.get(k)) is int for k in ("policy_actions", "setup_actions", "model_calls", "horizon")):
        return False
    try:
        policy_actions = int(row.get("policy_actions"))
        setup_actions = int(row.get("setup_actions"))
        model_calls = int(row.get("model_calls"))
        row_horizon = int(row.get("horizon"))
    except (TypeError, ValueError):
        return False
    if model_calls <= 0 or policy_actions <= 0 or policy_actions > horizon:
        return False
    if setup_actions != 10:
        return False
    if row_horizon != horizon:
        return False
    if success:
        return row.get("end_reason") == "native_success"
    return row.get("end_reason") == "budget_exhausted" and policy_actions == horizon


def _slot_dir(out: Path, suite: str, task_id: int, episode: int) -> Path:
    return out / suite / f"task_{task_id:02d}" / f"episode_{episode}"


def _slot_result_config(out: Path) -> dict:
    meta = out / "metadata.json"
    if not meta.exists():
        raise ContractError(f"missing metadata: {meta}")
    return json.loads(meta.read_text(encoding="utf-8")).get("fingerprint", "")


def _next_attempt(slot: Path) -> Path:
    index = 1
    while True:
        candidate = slot / f"attempt_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def _read_slot_result(slot: Path) -> dict | None:
    result = slot / "result.json"
    if not result.exists():
        return None
    return json.loads(result.read_text(encoding="utf-8"))


def summarize(out: Path) -> dict:
    meta_path = out / "metadata.json"
    meta = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_fp = meta.get("fingerprint", "")
    else:
        expected_fp = None
    suites: dict[str, dict[str, int]] = {s: {"valid": 0, "success": 0} for s in SUITES}
    tasks: dict[tuple[str, int], dict[str, int]] = {(s, t): {"valid": 0, "success": 0} for s in SUITES for t in range(10)}
    valid_episodes = 0
    successes = 0
    policy_total = 0
    setup_total = 0
    completed_slots: set[tuple[str, int, int]] = set()
    unknown_paths: list[str] = []

    for suite, task_id, episode in episode_slots():
        slot = _slot_dir(out, suite, task_id, episode)
        row = _read_slot_result(slot)
        attempts = sorted(slot.glob("attempt_*")) if slot.exists() else []
        if row is None:
            for attempt in attempts:
                attempt_result = attempt / "result.json"
                if attempt_result.exists():
                    try:
                        at_row = json.loads(attempt_result.read_text(encoding="utf-8"))
                    except Exception:
                        unknown_paths.append(str(attempt_result))
                        continue
                    if at_row.get("status") != "completed":
                        unknown_paths.append(str(attempt_result))
                else:
                    unknown_paths.append(str(attempt))
            continue
        if expected_fp is None:
            raise ContractError(f"missing metadata with existing result: {meta_path}")
        if row.get("fingerprint") != expected_fp:
            raise ContractError(f"fingerprint mismatch at {slot / 'result.json'}")
        if row.get("suite") != suite or row.get("task_id") != task_id or row.get("episode") != episode:
            raise ContractError(f"row identity mismatch at {slot / 'result.json'}")
        horizon = LIMITS[suite]
        if validate_episode(row, horizon):
            valid_episodes += 1
            suites[suite]["valid"] += 1
            tasks[(suite, task_id)]["valid"] += 1
            policy_total += int(row["policy_actions"])
            setup_total += int(row["setup_actions"])
            if row["success"]:
                successes += 1
                suites[suite]["success"] += 1
                tasks[(suite, task_id)]["success"] += 1
            completed_slots.add((suite, task_id, episode))
        else:
            raise ContractError(f"invalid completed slot result at {slot / 'result.json'}")

    completed_tasks = []
    for suite, task_id, episode in episode_slots():
        if (suite, task_id, episode) in completed_slots and episode == 4:
            if all((suite, task_id, e) in completed_slots for e in range(5)):
                completed_tasks.append(f"{suite}/task_{task_id:02d}")

    row_list = []
    for suite in SUITES:
        for task_id in range(10):
            entry = tasks[(suite, task_id)]
            rate = (entry["success"] / entry["valid"]) if entry["valid"] else None
            row_list.append({
                "suite": suite,
                "task_id": task_id,
                "valid_episodes": entry["valid"],
                "successes": entry["success"],
                "rate": rate,
            })

    summary = {
        "planned": 200,
        "valid_episodes": valid_episodes,
        "successes": successes,
        "rate": (successes / valid_episodes) if valid_episodes else None,
        "per_suite": {s: {"valid_episodes": suites[s]["valid"], "successes": suites[s]["success"],
                          "rate": (suites[s]["success"] / suites[s]["valid"]) if suites[s]["valid"] else None}
                      for s in SUITES},
        "completed_tasks": completed_tasks,
        "policy_actions_total": policy_total,
        "setup_actions_total": setup_total,
        "complete": valid_episodes == 200,
        "unknown_paths": unknown_paths,
        "invocation_utc": utc_now(),
    }
    atomic_write_json(out / "summary.json", summary)
    csv_path = out / "tasks.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "task_id", "valid_episodes", "successes", "rate"])
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)
    return summary


class AuditedClient(author.ClientModel):
    def __init__(self, host: str, port: int, directory: Path, counters: dict):
        super().__init__(host, port)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.counters = counters
        for key, default in (
            ("model_requests", 0),
            ("model_calls", 0),
            ("policy_actions", 0),
            ("setup_actions", 0),
            ("last_native_success", False),
        ):
            counters.setdefault(key, default)
        self._request_index = 0

    def _post(self, payload: dict) -> np.ndarray:
        self.counters["model_requests"] += 1
        self._request_index += 1
        index = self._request_index
        (self.directory / f"request_{index:03d}.json.gz").write_bytes(
            gzip.compress(json.dumps(payload).encode("utf-8"))
        )
        start = time.time()
        resp = requests.post(self.url, json=payload, timeout=(10, 180))
        (self.directory / f"response_{index:03d}.json").write_bytes(resp.content)
        resp.raise_for_status()
        data = json_numpy.loads(resp.content)
        action = np.array(data["action"], dtype=np.float64)
        if action.shape != (30, 20) or not np.all(np.isfinite(action)):
            raise ContractError(f"malformed policy action shape={action.shape}")
        self.counters["model_calls"] += 1
        with open(self.directory / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"call": index, "shape": list(action.shape),
                                "seconds": time.time() - start}) + "\n")
        return action


class _AuditedEnv:
    def __init__(self, original, ctx):
        self._original = original
        self._ctx = ctx
        self.env = original.env
        self.check_success = original.check_success
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._original, name)

    def step(self, action):
        ctx = self._ctx
        if (ctx["out"] / "STOP").exists():
            raise StopRequested("STOP file present")
        arr = np.asarray(action, dtype=np.float64)
        if arr.shape != (7,) or not np.all(np.isfinite(arr)):
            raise ContractError(f"invalid action shape/value: {arr.shape}")
        if ctx["phase"] == "policy":
            controller = ctx["env"].env.robots[0].controller
            if getattr(controller, "name", None) != "OSC_POSE":
                raise ContractError("controller.name must be OSC_POSE during policy phase")
            if controller.use_delta is not False:
                raise ContractError("controller.use_delta must be False during policy phase")
        ee_before = np.asarray(ctx["env"].env.robots[0].controller.ee_pos, dtype=np.float64).copy()
        sim_before = float(ctx["env"].env.sim.data.time)
        obs, reward, done, info = self._original.step(action)
        ee_after = np.asarray(ctx["env"].env.robots[0].controller.ee_pos, dtype=np.float64).copy()
        sim_after = float(ctx["env"].env.sim.data.time)
        if ctx["phase"] == "settling":
            ctx["counters"]["setup_actions"] += 1
            record = {"phase": "settling", "action": arr.tolist(), "done": bool(done),
                      "independent_success": None, "ee_pos_before": ee_before.tolist(),
                      "ee_pos_after": ee_after.tolist(), "sim_time": sim_after}
            ctx["last_obs"] = obs
        else:
            independent = bool(ctx["env"].check_success())
            if independent != bool(done):
                ctx["disagreement"] = True
            ctx["counters"]["policy_actions"] += 1
            ctx["counters"]["last_native_success"] = bool(done)
            ctx["last_obs"] = obs
            ctx["last_independent_success"] = independent
            record = {"phase": "policy", "action": arr.tolist(), "done": bool(done),
                      "independent_success": independent, "ee_pos_before": ee_before.tolist(),
                      "ee_pos_after": ee_after.tolist(), "sim_time": sim_after,
                      "sim_time_before": sim_before}
            with open(ctx["episode_dir"] / "steps.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
            if ctx["counters"]["policy_actions"] % 10 == 0:
                _write_progress(ctx)
            if ctx["disagreement"]:
                raise ContractError("independent success disagrees with native done")
            return obs, reward, done, info
        with open(ctx["episode_dir"] / "steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        if ctx["disagreement"]:
            raise ContractError("independent success disagrees with native done")
        return obs, reward, done, info

    def close(self):
        if self._closed:
            return None
        self._closed = True
        return self._original.close()


def _write_progress(ctx: dict) -> None:
    _write_progress_file(ctx["out"], ctx.get("slot_index", 0), ctx["counters"], ctx.get("status", "running"))


def _write_progress_file(out: Path, slot_index: int, counters: dict, status: str) -> None:
    completed = sum(1 for s in SUITES for t in range(10) for e in range(5)
                    if (_read_slot_result(_slot_dir(out, s, t, e)) or {}).get("status") == "completed")
    atomic_write_json(out / "progress.json", {
        "slot_index": slot_index,
        "counters": {k: counters.get(k) for k in ("model_requests", "model_calls", "policy_actions",
                                                   "setup_actions", "last_native_success")},
        "status": status,
        "completed": completed,
        "invocation_utc": utc_now(),
    })


class AuditedEval(author.LIBEROEval):
    def _init_env(self, task_suite, task_id: int = 0, ep: int = 0):
        env, task_description, obs = super()._init_env(task_suite, task_id, ep)
        if not isinstance(env, _AuditedEnv):
            raise ContractError("environment was not wrapped by audit factory")
        ctx = env._ctx
        controller = env.env.robots[0].controller
        if getattr(controller, "name", None) != "OSC_POSE":
            raise ContractError("unexpected controller name")
        if controller.use_delta is not False:
            raise ContractError("unexpected controller delta mode")
        if getattr(controller, "control_dim", None) != 6:
            raise ContractError("unexpected controller control_dim")
        if int(ctx["counters"].get("setup_actions", 0)) != 10:
            raise ContractError("setup_actions must be 10 before policy phase")
        if float(env.env.control_freq) != 20:
            raise ContractError(f"unexpected control_freq {env.env.control_freq}")
        robot_model_name = type(env.env.robots[0].robot_model).__name__
        if "Panda" not in robot_model_name:
            raise ContractError(f"unexpected robot model {robot_model_name}")
        task = task_suite.get_task(task_id)
        ctx["init_index"] = int(ep)
        ctx["seed"] = 142 + ep
        ctx["robot_model"] = robot_model_name
        ctx["gripper"] = type(env.env.robots[0].gripper).__name__
        ctx["control_freq"] = float(env.env.control_freq)
        bddl_root = Path(author.get_libero_path("bddl_files"))
        bddl = bddl_root / task.problem_folder / task.bddl_file
        ctx["bddl_path"] = str(bddl)
        ctx["bddl_sha"] = sha256_file(bddl) if bddl.exists() else None
        init_root = Path(author.get_libero_path("init_states"))
        init_path = init_root / task.problem_folder / task.init_states_file
        ctx["init_path"] = str(init_path)
        ctx["init_sha"] = sha256_file(init_path) if init_path.exists() else None
        _save_png(ctx["episode_dir"] / "initial_agentview.png", obs["agentview_image"])
        _save_png(ctx["episode_dir"] / "initial_wrist.png", obs["robot0_eye_in_hand_image"])
        np.save(ctx["episode_dir"] / "initial_state.npy", env.env.sim.get_state().flatten())
        audit = {
            "robot_model": ctx["robot_model"],
            "controller_name": getattr(controller, "name", None),
            "controller_use_delta": controller.use_delta,
            "controller_control_dim": getattr(controller, "control_dim", None),
            "gripper": ctx["gripper"],
            "control_freq": ctx["control_freq"],
            "init_index": ctx["init_index"],
            "seed": ctx["seed"],
            "bddl_path": ctx["bddl_path"],
            "bddl_sha": ctx["bddl_sha"],
            "init_path": ctx["init_path"],
            "init_sha": ctx["init_sha"],
            "invocation_utc": utc_now(),
        }
        atomic_write_json(ctx["episode_dir"] / "initial_audit.json", audit)
        ctx["initial_audit"] = audit
        ctx["phase"] = "policy"
        return env, task_description, obs

    def _log_results(self, metrics: dict) -> None:
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            numeric = [v for v in metrics.values()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(numeric) != 1 or float(numeric[0]) not in (0.0, 1.0):
                raise ContractError(f"expected exactly one numeric score in {{0,1}}, got {metrics}")
            ctx["author_score"] = int(numeric[0])
        super()._log_results(metrics)


def _save_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def freeze_metadata(out: Path, config: dict, runner_sha: str, author_sha: str) -> dict:
    versions = {"python": sys.version, "numpy": np.__version__}
    for name in ("torch", "mujoco", "robosuite", "gym"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "unavailable"
    libero_root = Path("/home/yhwang/fyp/xvla_native/LIBERO")
    upstream_root = Path("/home/yhwang/fyp/xvla_native/upstream")
    libero_head = _git_head(libero_root)
    upstream_head = _git_head(upstream_root)
    expected_libero = config.get("source_libero")
    expected_xvla = config.get("source_xvla")
    if expected_libero is None or expected_xvla is None:
        raise ContractError("config missing source_libero/source_xvla")
    if libero_head != expected_libero:
        raise ContractError(f"LIBERO head {libero_head} != config {expected_libero}")
    if upstream_head != expected_xvla:
        raise ContractError(f"upstream head {upstream_head} != config {expected_xvla}")
    install_path = ROOT / "First_Phase/tmp/xvla_20261012_setup/install_verification.json"
    if not install_path.exists():
        raise ContractError(f"missing install_verification.json: {install_path}")
    install = json.loads(install_path.read_text(encoding="utf-8"))
    weight = install.get("weight", {})
    if weight.get("sha256") != config.get("model_sha256") or weight.get("revision") != config.get("model_revision"):
        raise ContractError("install_verification weight mismatch with config")
    model_file = Path(weight["path"])
    if not model_file.is_file():
        raise ContractError(f"model.safetensors missing: {model_file}")
    size = model_file.stat().st_size
    if size != weight.get("bytes"):
        raise ContractError(f"model bytes {size} != recorded {weight.get('bytes')}")
    fingerprint = sha256_bytes(
        json.dumps({
            "config": config,
            "runner": runner_sha,
            "author": author_sha,
        }, sort_keys=True).encode()
    )
    meta = {
        "configuration": config,
        "fingerprint": fingerprint,
        "versions": versions,
        "git_heads": {
            "libero": libero_head,
            "upstream": upstream_head,
        },
        "weight_verification": {
            "source": str(install_path),
            "bytes": size,
            "sha256": weight.get("sha256"),
            "revision": weight.get("revision"),
            "reused_sha": True,
        },
        "runner_sha": runner_sha,
        "author_sha": author_sha,
        "server_url": f"http://{HOST}:{PORT}/act",
        "started_utc": utc_now(),
    }
    path = out / "metadata.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint or existing.get("versions") != versions:
            raise ContractError("existing metadata does not match fingerprint/versions")
    else:
        atomic_write_json(path, meta)
    return meta


def run_episode(out: Path, suite: str, task_id: int, episode: int, meta: dict,
                max_new_episodes: int, slots_done: int, fp: str) -> tuple[str, dict]:
    slot = _slot_dir(out, suite, task_id, episode)
    existing = _read_slot_result(slot)
    if existing is not None and validate_episode(existing, LIMITS[suite]) and existing.get("fingerprint") == fp:
        return "reused", existing
    if existing is not None:
        raise ContractError(f"existing invalid/mismatched result at {slot / 'result.json'}")
    attempt = _next_attempt(slot)
    attempt.mkdir(parents=True)
    counters: dict = {}
    ctx = {
        "out": out,
        "episode_dir": attempt,
        "counters": counters,
        "phase": "settling",
        "horizon": LIMITS[suite],
        "env": None,
        "slot_index": slots_done,
        "status": "running",
        "disagreement": False,
        "last_obs": None,
        "last_independent_success": None,
    }
    start = time.time()
    row: dict
    try:
        original_class = author.OffScreenRenderEnv

        def factory(**kwargs):
            env = original_class(**kwargs)
            wrapped = _AuditedEnv(env, ctx)
            ctx["env"] = wrapped
            return wrapped

        author.OffScreenRenderEnv = factory
        client = AuditedClient(HOST, PORT, attempt, counters)
        evaluator = AuditedEval(task_suite_name=suite, eval_horizon=LIMITS[suite],
                                act_type="abs", num_episodes=5, init_seed=42)
        evaluator.base_dir = attempt
        evaluator._ctx = ctx
        score = evaluator._rollout(evaluator.task_suite_list[0], client, task_id, episode)
        captured_author_score = ctx.get("author_score")
        last_independent = ctx.get("last_independent_success")
        if captured_author_score is None:
            raise ContractError("author_score not captured from _log_results")
        returned_score = int(score)
        if returned_score != captured_author_score:
            raise ContractError(f"author score mismatch: returned {returned_score} captured {captured_author_score}")
        if last_independent is None:
            raise ContractError("last_independent_success not captured")
        final_obs = None
        if ctx.get("env") is not None:
            final_obs = ctx.get("last_obs")
        if final_obs is not None:
            _save_png(attempt / "final_agentview.png", final_obs["agentview_image"])
            _save_png(attempt / "final_wrist.png", final_obs["robot0_eye_in_hand_image"])
        ctx["phase"] = "done"
        last_native = bool(ctx["counters"].get("last_native_success", False))
        author_score = captured_author_score
        independent = bool(last_independent)
        if independent != author_score or last_native != bool(author_score):
            raise ContractError("success/score/native consistency failed")
        success = bool(author_score)
        policy_actions = int(counters.get("policy_actions", 0))
        setup_actions = int(counters.get("setup_actions", 0))
        model_calls = int(counters.get("model_calls", 0))
        end_reason = "native_success" if success else "budget_exhausted"
        row = {
            "status": "completed",
            "valid": True,
            "suite": suite,
            "task_id": task_id,
            "episode": episode,
            "success": success,
            "author_score": author_score,
            "independent_success": independent,
            "policy_actions": policy_actions,
            "setup_actions": setup_actions,
            "model_requests": int(counters.get("model_requests", 0)),
            "model_calls": model_calls,
            "horizon": LIMITS[suite],
            "end_reason": end_reason,
            "fingerprint": fp,
            "attempt_dir": str(attempt),
            "wall_seconds": time.time() - start,
            "invocation_utc": utc_now(),
            "reset_internal_physics_steps": "not separately observed",
            "init_index": ctx.get("init_index"),
            "seed": ctx.get("seed"),
            "robot_model": ctx.get("robot_model"),
            "controller_name": ctx["initial_audit"]["controller_name"],
            "controller_use_delta": ctx["initial_audit"]["controller_use_delta"],
            "controller_control_dim": ctx["initial_audit"]["controller_control_dim"],
            "gripper": ctx.get("gripper"),
            "control_freq": ctx.get("control_freq"),
            "bddl_path": ctx.get("bddl_path"),
            "bddl_sha": ctx.get("bddl_sha"),
            "init_path": ctx.get("init_path"),
            "init_sha": ctx.get("init_sha"),
        }
        if not validate_episode(row, LIMITS[suite]):
            row["status"] = "unknown"
            row["valid"] = False
            row["success"] = None
            row["error"] = "contract validation failed after rollout"
            atomic_write_json(attempt / "result.json", row)
            return "unknown", row
        atomic_write_json(attempt / "result.json", row)
        atomic_write_json(slot / "result.json", row)
        return "completed", row
    except StopRequested as exc:
        row = _error_row(suite, task_id, episode, counters, attempt, fp, "interrupted", exc)
        atomic_write_json(attempt / "result.json", row)
        return "interrupted", row
    except Exception as exc:
        row = _error_row(suite, task_id, episode, counters, attempt, fp, "unknown", exc)
        atomic_write_json(attempt / "result.json", row)
        return "unknown", row
    finally:
        ctx["phase"] = "done"
        try:
            if ctx.get("env") is not None:
                ctx["env"].close()
        except Exception:
            pass
        try:
            author.OffScreenRenderEnv = original_class
        except Exception:
            pass


def _error_row(suite, task_id, episode, counters, attempt, fp, status, exc) -> dict:
    return {
        "status": status,
        "valid": False,
        "suite": suite,
        "task_id": task_id,
        "episode": episode,
        "success": None,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "policy_actions": int(counters.get("policy_actions", 0)),
        "setup_actions": int(counters.get("setup_actions", 0)),
        "model_requests": int(counters.get("model_requests", 0)),
        "model_calls": int(counters.get("model_calls", 0)),
        "fingerprint": fp,
        "attempt_dir": str(attempt),
        "invocation_utc": utc_now(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("xvla-native-benchmark")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-new-episodes", type=int, default=200)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_episodes <= 0 or args.max_new_episodes > 200:
        parser.error("--max-new-episodes must be positive and <= 200")
    out = args.out
    if args.report_only:
        summarize(out)
        return 0
    config = load_config()
    runner_sha = sha256_file(Path(__file__))
    author_sha = sha256_file(Path(author.__file__))
    out.mkdir(parents=True, exist_ok=True)
    meta = freeze_metadata(out, config, runner_sha, author_sha)
    fp = meta["fingerprint"]
    done = 0
    slots = episode_slots()
    for index, (suite, task_id, episode) in enumerate(slots):
        if (out / "STOP").exists():
            summarize(out)
            return 130
        if done >= args.max_new_episodes:
            _write_progress_file(out, index, {}, "paused_after_limit")
            summarize(out)
            return 0
        slot = _slot_dir(out, suite, task_id, episode)
        existing = _read_slot_result(slot)
        if existing is not None and validate_episode(existing, LIMITS[suite]) and existing.get("fingerprint") == fp:
            continue
        if existing is not None:
            _write_progress_file(out, index, {}, "blocked")
            summarize(out)
            return 2
        attempts = sorted(slot.glob("attempt_*")) if slot.exists() else []
        if attempts and existing is None:
            _write_progress_file(out, index, {}, "blocked")
            summarize(out)
            return 2
        status, row = run_episode(out, suite, task_id, episode, meta, args.max_new_episodes, index, fp)
        _write_progress_file(out, index, {"model_requests": row.get("model_requests", 0),
                                          "model_calls": row.get("model_calls", 0),
                                          "policy_actions": row.get("policy_actions", 0),
                                          "setup_actions": row.get("setup_actions", 0),
                                          "last_native_success": bool(row.get("independent_success"))},
                             status)
        if status == "interrupted":
            summarize(out)
            return 130
        if status == "unknown":
            summarize(out)
            return 2
        if status == "completed":
            done += 1
    _write_progress_file(out, len(slots), {}, "completed")
    summarize(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
