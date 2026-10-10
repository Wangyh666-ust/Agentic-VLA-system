"""First_Phase/scene_demo/finish_v1/pilot.py

Fixed five historical-state trial runner.  No mocking, no stubs, real
runtime imports after argparse.  Implements the exact contract described
in the task specification.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# argparse (help / import path must not touch heavy imports)
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="finish_v1 fixed five-case historical-state pilot runner"
    )
    p.add_argument("--proposal", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=5)
    a = p.parse_args(argv)
    if a.limit < 1 or a.limit > 5:
        p.error("--limit must be between 1 and 5")
    return a


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as _np
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return repr(o)


def _append_jsonl(fp, obj) -> None:
    fp.write(json.dumps(obj, default=_json_default) + "\n")
    fp.flush()


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# gold predicate fixed literal lookup
# ---------------------------------------------------------------------------
GOLD_BY_CASE_ID = {
    "bowl_held83": [["on", "akita_black_bowl_1", "plate_1"]],
    "wine_exit": [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
    "stove_exit": [["turnon", "flat_stove_1"]],
    "soup_negative": [["in", "alphabet_soup_1", "basket_1_contain_region"]],
    "sauce_negative": [["in", "tomato_sauce_1", "basket_1_contain_region"]],
}


# ---------------------------------------------------------------------------
# Pure helper functions (unit-testable)
# ---------------------------------------------------------------------------
def validate_context_origin(case: dict):
    if case.get("context_origin") != "fixed_historical_subtask_fixture":
        raise RuntimeError("context_origin mismatch")


def sensor_frames_list(captures, existing_frames=None):
    """Merge captures into timestamp-ordered sensor frame list.

    captures: iterable of dicts with keys timestamp, frames (list).
    existing_frames: optional list of already accepted frame dicts.
    Returns list of {frame:int, view:str, timestamp:float}.
    """
    ordered = list(existing_frames or [])
    for cap in captures:
        ts = cap.get("timestamp")
        if ts is None:
            continue
        for fr in cap.get("frames") or []:
            ordered.append({"timestamp": float(ts), "view": fr.get("view"), "frame": None})
    ordered.sort(key=lambda x: x["timestamp"])
    for i, row in enumerate(ordered):
        row["frame"] = i
    return ordered


def dedup_last_n_ordered(rows, n=3):
    """Return last n rows sorted by timestamp, without duplicate timestamps."""
    by_ts = {}
    for r in rows:
        by_ts[r["timestamp"]] = r
    uniq = list(by_ts.values())
    uniq.sort(key=lambda x: x["timestamp"])
    return uniq[-n:]


def predicate_key(goal):
    return "|".join(goal)


def negative_routing(decision, allow_helper):
    """Returns one of 'correct_negative' or 'false_positive' for non-complete states."""
    state = decision.get("state")
    if allow_helper:
        return None
    if state in ("incomplete", "unknown"):
        return "correct_negative"
    return "false_positive"


def compute_unique_actual_streak(post_rows):
    """post_rows: sorted list of dicts with actual_step, ready, home_ready, contacts_empty.
    Reject duplicate step IDs. Return physical streak."""
    seen = set()
    streak = 0
    for row in post_rows:
        step = row.get("actual_step")
        if step in seen:
            raise RuntimeError(f"duplicate actual_step in post rows: {step}")
        seen.add(step)
        if row.get("ready") and row.get("home_ready") and row.get("contacts_empty"):
            streak += 1
        else:
            streak = 0
    return streak


def turn_ready_without_static_entry(snapshot):
    """For stove turnon, no strict_candidate needed."""
    preds = snapshot.get("predicates", {})
    if preds.get("turnon|flat_stove_1") is not True:
        return False
    if not snapshot.get("grasp_observation_complete"):
        return False
    objects = snapshot.get("objects", {})
    for obj in objects.values():
        if obj.get("grasped"):
            return False
    if snapshot.get("held_objects"):
        return False
    return True


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
class PilotRunner:
    def __init__(self, args):
        self.args = args
        self.proposal_path = Path(args.proposal).resolve()
        self.inputs_path = Path(args.inputs).resolve()
        self.output_dir = Path(args.output).resolve()
        self.limit = args.limit
        self.campaign_summary = {
            "cases": [],
            "statuses": [],
            "phase_counts": {
                "replay": 0,
                "release": 0,
                "movement": 0,
                "confirm": 0,
                "home": 0,
            },
            "new_vla": 0,
            "new_hermes": 0,
            "qwen_calls": 0,
        }

    # --- preflight ---------------------------------------------------------
    def verify_preflight(self):
        evidence = (
            ROOT
            / "First_Phase"
            / "scene_demo"
            / "results"
            / "2026-10-10-visual-finish"
            / "evidence"
        )
        preflight_path = evidence / "preflight.json"
        if not preflight_path.is_file():
            raise RuntimeError(f"preflight not found: {preflight_path}")
        with open(preflight_path, "r", encoding="utf-8") as f:
            pre = json.load(f)

        prop_sha = _sha256_file(self.proposal_path)
        if pre.get("proposal_sha256") != prop_sha:
            raise RuntimeError(
                "preflight proposal_sha256 mismatch: "
                f"{pre.get('proposal_sha256')} != {prop_sha}"
            )
        inp_sha = _sha256_file(self.inputs_path)
        if pre.get("inputs_sha256") != inp_sha:
            raise RuntimeError(
                "preflight inputs_sha256 mismatch: "
                f"{pre.get('inputs_sha256')} != {inp_sha}"
            )
        src_map = pre.get("source_sha256") or {}
        for rel, expected in src_map.items():
            p = ROOT / rel
            if not p.is_file():
                raise RuntimeError(f"source file missing: {rel}")
            actual = _sha256_file(p)
            if actual != expected:
                raise RuntimeError(
                    f"source_sha256 mismatch for {rel}: {expected} != {actual}"
                )
        return pre

    def load_inputs(self):
        with open(self.inputs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frozen = data.get("frozen_files")
        if not isinstance(frozen, dict):
            raise RuntimeError("inputs.frozen_files missing or not dict")
        for v in frozen.values():
            if not isinstance(v, dict):
                raise RuntimeError("frozen_files entry not dict")
            rel = v.get("path")
            nbytes = v.get("bytes")
            sha = v.get("sha256")
            if not isinstance(rel, str):
                raise RuntimeError("frozen_files entry missing path")
            p = ROOT / rel
            if not p.is_file():
                raise RuntimeError(f"frozen file missing: {rel}")
            actual_bytes = p.stat().st_size
            actual_sha = _sha256_file(p)
            if int(nbytes) != int(actual_bytes):
                raise RuntimeError(
                    f"frozen bytes mismatch for {rel}: {nbytes} != {actual_bytes}"
                )
            if str(sha) != actual_sha:
                raise RuntimeError(
                    f"frozen sha mismatch for {rel}: {sha} != {actual_sha}"
                )

        cases = data.get("cases")
        if not isinstance(cases, list) or len(cases) != 5:
            raise RuntimeError("inputs.cases must be exactly 5 cases")
        return data, cases

    # --- runtime imports ---------------------------------------------------
    def import_runtime(self):
        sys.path.insert(0, str(ROOT / "First_Phase"))
        from scene_demo import service  # noqa
        from scene_demo import placement_experiments as pe  # noqa
        from scene_demo import joint_home as jh  # noqa
        from scene_demo import safe_exit_contacts as sc  # noqa
        from scene_demo.finish_v1 import control  # noqa
        from scene_demo.finish_v1 import vision  # noqa
        from scene_demo.finish_v1.sensors import SensorPort  # noqa
        import numpy as np  # noqa
        import imageio.v2 as imageio  # noqa

        self.service = service
        self.pe = pe
        self.jh = jh
        self.sc = sc
        self.control = control
        self.vision = vision
        self.SensorPort = SensorPort
        self.np = np
        self.imageio = imageio

    # --- review subprocess -------------------------------------------------
    def run_review(self):
        script = ROOT / ".agents" / "skills" / "fyp-experiment-review" / "scripts" / "review.py"
        if not script.is_file():
            raise RuntimeError(f"review script missing: {script}")
        cmd = [
            sys.executable,
            str(script),
            "check",
            "--proposal",
            str(self.proposal_path),
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"preflight review failed rc={proc.returncode}"
            )

    # --- per-case execution ------------------------------------------------
    def run_case(self, case, case_idx, qwen_client, qwen_budget):
        case_id = case["case_id"]
        case_out = self.output_dir / case_id
        case_out.mkdir(parents=True, exist_ok=False)

        result = {
            "case_id": case_id,
            "context_origin": "fixed_historical_subtask_fixture",
            "origin_sha": case["origin_sha"],
            "terminal_sha": case.get("terminal_sha"),
            "origin_check": None,
            "terminal_check": None,
            "phase_counts": {"replay": 0, "release": 0, "movement": 0, "confirm": 0, "home": 0},
            "initial_vision_decision": None,
            "finish": None,
            "final_snapshot": None,
            "home": None,
            "contacts": None,
            "physical_streak": 0,
            "video": None,
            "errors": [],
            "status": "not_run",
        }

        env = None
        port = None
        video_writer = None
        oracle_fp = None
        checks_fp = None
        events_fp = None
        helper_fp = None
        orig_step = None

        try:
            validate_context_origin(case)
            suite = case["suite"]
            task = case["task"]
            init = case["init"]
            env_seed = case["env_seed"]
            replay_count = int(case["replay_count"])
            action_path = case["action_path"]
            action_key = case["action_key"]
            reference = case["reference"]
            context = case["context"]
            allow_helper = bool(case["allow_helper"])
            gold = GOLD_BY_CASE_ID[case_id]

            # reverify sources before each case
            self.verify_preflight()

            np = self.np
            service = self.service
            pe = self.pe
            jh = self.jh
            sc = self.sc
            control = self.control
            vision = self.vision
            SensorPort = self.SensorPort
            imageio = self.imageio

            builder = object.__new__(service.SceneService)
            builder._env_factory = None
            env = builder._build_env(suite, task, env_seed, init)
            env.reset(seed=env_seed)
            result["initialization_steps"] = 10

            init_sha = service.state_sha(env)
            result["initial_state_sha"] = init_sha
            if init_sha != case["origin_sha"]:
                raise RuntimeError(
                    f"initial state sha mismatch: {init_sha} != {case['origin_sha']}"
                )
            if init_sha != reference.get("origin_sha256"):
                raise RuntimeError(
                    "reference.origin_sha256 mismatch: "
                    f"{init_sha} != {reference.get('origin_sha256')}"
                )

            actions = self._load_actions(action_path, action_key, replay_count, np)

            oracle_path = case_out / "oracle.jsonl"
            checks_path = case_out / "checks.jsonl"
            events_path = case_out / "events.jsonl"
            helper_path = case_out / "helper.jsonl"
            oracle_fp = open(oracle_path, "w", encoding="utf-8")
            checks_fp = open(checks_path, "w", encoding="utf-8")
            events_fp = open(events_path, "w", encoding="utf-8")
            helper_fp = open(helper_path, "w", encoding="utf-8")

            frames_dir = "sensors"
            port = SensorPort(
                env,
                reference,
                str(case_out / frames_dir),
                step_callback=lambda row: _append_jsonl(helper_fp, row),
            )
            port.phase = "release"

            video_path = case_out / "all_actions.mp4"
            video_writer = imageio.get_writer(str(video_path), fps=20)
            result["video"] = str(video_path)

            frame_deque = deque(maxlen=3)
            actual_counts = {"release": 0, "movement": 0, "confirm": 0, "home": 0}
            replay_counter = {"n": 0}
            caps = {"release": 20, "movement": 80, "confirm": 20, "home": 360}
            actual_step_counter = {"n": 0}
            last_capture = {"cap": None}

            def _capture_and_record(record_oracle=False):
                pre_sha = service.state_sha(env)
                cap = port.capture()
                post_sha = service.state_sha(env)
                if pre_sha != post_sha:
                    raise RuntimeError(
                        f"sensor capture mutated state: {pre_sha} != {post_sha}"
                    )
                frames = cap.get("frames") or []
                rgb_paths = [f["rgb_path"] for f in frames]
                if len(rgb_paths) >= 2:
                    img_a = imageio.imread(rgb_paths[0])
                    img_b = imageio.imread(rgb_paths[1])
                    if img_a.shape[0] != img_b.shape[0]:
                        h = min(img_a.shape[0], img_b.shape[0])
                        img_a = img_a[:h]
                        img_b = img_b[:h]
                    combined = np.concatenate([img_a, img_b], axis=1)
                    video_writer.append_data(combined)
                if record_oracle:
                    snap = self._oracle_snapshot(env, gold, reference, service, jh, sc)
                    _append_jsonl(oracle_fp, {"tag": "actual", "snapshot": snap, "phase": port.phase, "actual_step": actual_step_counter["n"]})
                frame_deque.append(cap)
                last_capture["cap"] = cap
                return cap

            orig_step = env.step

            # Replay wrapper: dedicated, no capture, no oracle post
            def replay_step(action, *a, **kw):
                out = orig_step(action, *a, **kw)
                replay_counter["n"] += 1
                result["phase_counts"]["replay"] += 1
                _capture_and_record(record_oracle=False)
                replay_state_sha = service.state_sha(env)
                _append_jsonl(oracle_fp, {"tag": "replay", "state_sha": replay_state_sha, "action": action})
                return out

            env.step = replay_step

            for act in actions:
                replay_step(act)

            # now after replay, do initial capture/oracle (non-post)
            _capture_and_record(record_oracle=False)
            initial_oracle = self._oracle_snapshot(env, gold, reference, service, jh, sc)
            _append_jsonl(oracle_fp, {"tag": "initial", "snapshot": initial_oracle})

            terminal_sha = service.state_sha(env)
            result["terminal_state_sha"] = terminal_sha
            if case["terminal_sha"] is not None:
                if terminal_sha != case["terminal_sha"]:
                    raise RuntimeError(
                        f"terminal_sha mismatch: {terminal_sha} != {case['terminal_sha']}"
                    )
                result["terminal_check"] = "sha_match"
            else:
                self._verify_embedded_terminal(env, case.get("terminal_snapshot_embedded"), gold, reference, service)
                result["terminal_check"] = "embedded_match"

            safety_baseline = self._init_safety_baseline(env, gold, service, port, sc)

            def check_fn(context_arg, observations):
                nonlocal last_verdict
                if not isinstance(observations, list):
                    raise TypeError("observations must be a list")
                caps = list(frame_deque) + list(observations) + [port.capture()]
                by_ts = {float(cap['timestamp']): cap for cap in caps}
                selected = [by_ts[t] for t in sorted(by_ts)[-3:]]
                frame_deque.clear()
                frame_deque.extend(selected)
                sensor_frames = []
                image_paths = []
                for cap in selected:
                    for fr in cap.get("frames", []):
                        sensor_frames.append({"frame": len(sensor_frames), "view": fr.get("view"), "timestamp": cap.get("timestamp")})
                        image_paths.append(fr.get("rgb_path"))
                safe_ctx = {
                    "user_request": copy.deepcopy(context_arg.get("user_request")),
                    "current_subtask": copy.deepcopy(context_arg.get("current_subtask")),
                    "criteria": copy.deepcopy(context_arg.get("criteria")),
                    "sensor_frames": copy.deepcopy(sensor_frames),
                }
                call_index = qwen_budget["n"]
                qwen_budget["n"] += 1
                if qwen_budget["n"] > qwen_budget["max"]:
                    raise RuntimeError("qwen call budget exceeded")
                meta = qwen_client.analyze(safe_ctx, image_paths)
                verdict = meta.get("verdict", {}) if isinstance(meta, dict) else {}
                row = {
                    "call_index": call_index,
                    "status": meta.get("status") if isinstance(meta, dict) else None,
                    "model": meta.get("model") if isinstance(meta, dict) else None,
                    "usage": meta.get("usage") if isinstance(meta, dict) else None,
                    "http_status": meta.get("http_status") if isinstance(meta, dict) else None,
                    "request_sha256": meta.get("request_sha256") if isinstance(meta, dict) else None,
                    "response_sha256": meta.get("response_sha256") if isinstance(meta, dict) else None,
                    "sensor_paths": image_paths,
                    "sensor_frames": sensor_frames,
                    "verdict": verdict,
                }
                _append_jsonl(checks_fp, row)
                last_verdict = verdict
                return verdict

            last_verdict = None
            initial_ctx = copy.deepcopy(context)
            initial_verdict = check_fn(initial_ctx, [])
            result["initial_vision_decision"] = initial_verdict

            decision = vision.decide(context, initial_verdict)
            decision_state = decision.get("state")

            if not allow_helper:
                status = negative_routing(decision, allow_helper)
                result["status"] = status
                final_snap = self._oracle_snapshot(env, gold, reference, service, jh, sc)
                _append_jsonl(oracle_fp, {"tag": "final", "snapshot": final_snap})
                result["final_snapshot"] = final_snap
                result["finish"] = {"status": "skipped_negative"}
                return result

            # Positive path: run_finish
            env.step = orig_step

            def wrapped_step(action, *a, **kw):
                phase = port.phase
                if phase in caps:
                    if actual_counts[phase] >= caps[phase]:
                        raise RuntimeError(f"phase cap exceeded: {phase} >= {caps[phase]}")
                out = orig_step(action, *a, **kw)
                actual_counts[phase] = actual_counts.get(phase, 0) + 1
                result["phase_counts"][phase] += 1
                actual_step_counter["n"] += 1
                _capture_and_record(record_oracle=False)
                snap = self._oracle_snapshot(env, gold, reference, service, jh, sc)
                _append_jsonl(oracle_fp, {"tag": "actual", "snapshot": snap, "phase": phase, "actual_step": actual_step_counter["n"]})
                self._safety_check(snap, result, safety_baseline, env, gold, service, sc, np)
                if result.get("oracle_safety_abort"):
                    raise RuntimeError("oracle_safety_abort:" + result["oracle_safety_abort_reason"])
                return out

            env.step = wrapped_step

            def emit(row):
                _append_jsonl(events_fp, row)

            try:
                finish_result = control.run_finish(
                    port,
                    context,
                    initial_verdict,
                    vision.decide,
                    check_fn,
                    emit,
                )
                result["finish"] = finish_result
                reason = finish_result.get("reason", "")
                if reason.startswith("exception:") and "oracle_safety_abort" not in reason:
                    raise RuntimeError("core reason " + reason)
                if "oracle_safety_abort" in reason:
                    result["oracle_safety_abort"] = True
                    result["oracle_safety_abort_reason"] = reason
                if not result.get("oracle_safety_abort") and finish_result["counts"] != actual_counts:
                    raise RuntimeError("core reason action counts mismatch")
                result["core_counts"] = finish_result.get("counts")
                result["actual_helper_counts"] = dict(actual_counts)
            except RuntimeError as exc:
                msg = str(exc)
                if msg.startswith("oracle_safety_abort"):
                    result["errors"].append(msg)
                    result["oracle_safety_abort"] = True
                    result["oracle_safety_abort_reason"] = msg
                    result["finish"] = {"status": "aborted", "reason": msg}
                else:
                    raise

            # restore
            env.step = orig_step

            decision_final = vision.decide(context, last_verdict)
            latest_state_complete = decision_final.get("state") == "complete"

            final_snap = self._oracle_snapshot(env, gold, reference, service, jh, sc)
            _append_jsonl(oracle_fp, {"tag": "final", "snapshot": final_snap})
            result["final_snapshot"] = final_snap
            result["home"] = final_snap["home"]
            result["contacts"] = final_snap["contacts"]

            # Compute streak from actual post rows only
            post_rows = []
            oracle_fp.flush()
            with open(oracle_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("tag") == "actual" and "actual_step" in row:
                        snap = row["snapshot"]
                        gold_snap = snap.get("gold", {})
                        home = snap.get("home", {})
                        ready = self._is_ready_for_goal(gold_snap, gold, case_id)
                        post_rows.append({
                            "actual_step": row["actual_step"],
                            "ready": ready,
                            "home_ready": home.get("ready"),
                            "contacts_empty": len(snap.get("contacts", [])) == 0
                        })
            streak = compute_unique_actual_streak(post_rows)
            result["physical_streak"] = streak

            if result.get("oracle_safety_abort"):
                result["status"] = "blocked"
            elif streak >= 5 and latest_state_complete and result["finish"].get("ready") is True:
                result["status"] = "physical_success"
            else:
                result["status"] = "blocked"

            video_writer.close()
            video_writer = None

            return result

        except Exception as exc:
            tb = traceback.format_exc()
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            result["traceback"] = tb
            if result.get("oracle_safety_abort") or (isinstance(exc, RuntimeError) and str(exc).startswith("oracle_safety_abort")):
                result["status"] = "blocked"
            elif "Unexpected exception" in str(exc) or "core reason" in str(exc):
                result["status"] = "unknown"
            else:
                if result["status"] == "not_run":
                    result["status"] = "unknown"
            return result
        finally:
            cleanup_errors = []
            if env is not None and orig_step is not None:
                try:
                    env.step = orig_step
                except Exception as e:
                    cleanup_errors.append(f"restore env.step: {e}")
            if port is not None:
                try:
                    port.close()
                except Exception as e:
                    cleanup_errors.append(f"port.close: {e}")
            if env is not None:
                try:
                    env.close()
                except Exception as e:
                    cleanup_errors.append(f"env.close: {e}")
            for fp in (oracle_fp, checks_fp, events_fp, helper_fp):
                if fp is not None:
                    try:
                        fp.close()
                    except Exception as e:
                        cleanup_errors.append(f"close file: {e}")
            if video_writer is not None:
                try:
                    video_writer.close()
                except Exception as e:
                    cleanup_errors.append(f"video_writer.close: {e}")
            if cleanup_errors:
                result.setdefault("errors", []).extend([f"cleanup: {e}" for e in cleanup_errors])
                result["status"] = "unknown"

    # ------------------------------------------------------------------
    # helpers used inside run_case
    # ------------------------------------------------------------------
    def _load_actions(self, action_path, action_key, replay_count, np):
        p = Path(action_path)
        if not p.is_absolute():
            p = ROOT / p
        rows = []
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                arr = obj[action_key]
                rows.append(np.asarray(arr, dtype=np.float32))
                if len(rows) >= replay_count:
                    break
        if len(rows) != replay_count:
            raise RuntimeError(
                f"expected {replay_count} actions, got {len(rows)}"
            )
        for i, a in enumerate(rows):
            if a.shape != (7,):
                raise RuntimeError(f"action {i} shape {a.shape} != (7,)")
            if not np.all(np.isfinite(a)):
                raise RuntimeError(f"action {i} non-finite")
        return rows

    def _verify_embedded_terminal(self, env, embed, gold, reference, service):
        if not embed:
            raise RuntimeError("terminal_snapshot_embedded missing")
        np = self.np
        pe = self.pe
        snap = pe.capture_snapshot(env, gold)
        for goal in gold:
            key = predicate_key(goal)
            expected_predicate = embed.get("predicates", {}).get(key)
            live_predicate = snap.get("predicates", {}).get(key)
            if type(expected_predicate) is not bool or type(live_predicate) is not bool or expected_predicate != live_predicate:
                raise RuntimeError("embedded predicate mismatch: " + key)
        held_expected = embed.get("held_objects")
        held_live = set(snap.get("held_objects") or [])
        if set(held_expected) != held_live:
            raise RuntimeError(
                f"held_objects mismatch: {held_live} != {set(held_expected)}"
            )
        objects = snap.get("objects", {})
        for goal in gold:
            obj_id = goal[1] if len(goal) >= 2 else None
            if obj_id and obj_id in embed.get("objects", {}):
                if obj_id not in objects:
                    raise RuntimeError(f"{obj_id} missing")
                expected = np.asarray(embed["objects"][obj_id]["position"], dtype=np.float64)
                live = np.asarray(objects[obj_id].get("position"), dtype=np.float64)
                if expected.shape != live.shape or np.max(np.abs(expected - live)) > 1e-7:
                    raise RuntimeError(f"{obj_id} position mismatch >1e-7")
        for k in ("eef_position", "gripper_qpos"):
            if k in embed:
                if k not in snap or snap[k] is None:
                    raise RuntimeError(f"{k} missing")
                expected = np.asarray(embed[k], dtype=np.float64)
                live = np.asarray(snap[k], dtype=np.float64)
                if expected.shape != live.shape or np.max(np.abs(expected - live)) > 1e-7:
                    raise RuntimeError(f"{k} mismatch >1e-7")

    def _oracle_snapshot(self, env, gold, reference, service, jh, sc):
        snap = self.pe.capture_snapshot(env, gold)
        robot_state = jh.read_robot_state(env)
        home = jh.home_metrics(reference, robot_state)
        contacts = sc.contacts(env)
        rows = []
        for c in contacts or []:
            rows.append(c)  # retain all raw fields
        return {
            "state_sha": service.state_sha(env),
            "gold": snap,
            "home": home,
            "contacts": rows,
        }

    def _init_safety_baseline(self, env, gold, service, port, sc):
        np = self.np
        base = self.pe.capture_snapshot(env, gold)
        non_target_positions = {}
        objs = base.get("objects") if isinstance(base, dict) else {}
        target_ids = {p[1] for p in gold if len(p) >= 2}
        for k, v in objs.items():
            if k in target_ids:
                continue
            pos = v.get("position") if isinstance(v, dict) else None
            if pos is not None:
                non_target_positions[k] = np.asarray(pos, dtype=np.float64)
        contacts0 = sc.contacts(env) or []
        contact_depth = {}
        for c in contacts0:
            pair = c.get("pair")
            dist = c.get("dist")
            if pair is None or dist is None:
                continue
            key = (int(pair[0]), int(pair[1]))
            contact_depth[key] = float(max(-float(dist), 0.0))
        gold_true = self._eval_gold_predicates(base, gold)
        return {
            "gold_true": gold_true,
            "non_target_positions": non_target_positions,
            "contact_depth": contact_depth,
        }

    def _eval_gold_predicates(self, snapshot, gold):
        preds = snapshot.get("predicates", {})
        if not isinstance(preds, dict):
            # Oracle missing key => unknown
            raise RuntimeError("snapshot missing predicates dict")
        result = []
        for goal in gold:
            key = predicate_key(goal)
            if key not in preds or type(preds[key]) is not bool:
                raise RuntimeError(f"missing predicate key: {key}")
            result.append(preds[key])
        return result

    def _safety_check(self, snapshot, result, baseline, env, gold, service, sc, np):
        # snapshot is oracle snapshot dict
        gold_snap = snapshot.get("gold", {})
        gold_true_now = self._eval_gold_predicates(gold_snap, gold)
        base_true = baseline["gold_true"]
        for i, was in enumerate(base_true):
            if was and not gold_true_now[i]:
                result["oracle_safety_abort"] = True
                result["oracle_safety_abort_reason"] = f"gold_predicate_dropped_{i}"
                return
        # non-target displacement
        objs = gold_snap.get("objects", {})
        for k, pos0 in baseline["non_target_positions"].items():
            v = objs.get(k)
            if v is None or not isinstance(v, dict):
                result["oracle_safety_abort"] = True
                result["oracle_safety_abort_reason"] = f"non_target_missing_{k}"
                return
            pos = v.get("position")
            if pos is None:
                result["oracle_safety_abort"] = True
                result["oracle_safety_abort_reason"] = f"non_target_position_missing_{k}"
                return
            pos1 = np.asarray(pos, dtype=np.float64)
            disp = float(np.linalg.norm(pos1 - pos0))
            if disp > 0.005:
                result["oracle_safety_abort"] = True
                result["oracle_safety_abort_reason"] = f"non_target_disp_{k}"
                return
        # contacts
        contacts_now = sc.contacts(env) or []
        for c in contacts_now:
            pair = c.get("pair")
            dist = c.get("dist")
            if pair is None or dist is None:
                continue
            key = (int(pair[0]), int(pair[1]))
            pen = float(max(-float(dist), 0.0))
            base_pen = baseline["contact_depth"].get(key, 0.0)
            if key not in baseline["contact_depth"] and pen > 0.00005:
                result["oracle_safety_abort"] = True
                result["oracle_safety_abort_reason"] = f"new_pair_penetration_{key}"
                return
            if key in baseline["contact_depth"] and pen > base_pen + 0.00005:
                result["oracle_safety_abort"] = True
                result["oracle_safety_abort_reason"] = f"deepen_penetration_{key}"
                return

    def _is_ready_for_goal(self, gold_snap, gold, case_id):
        # Strict readiness per goal
        if not gold_snap.get("grasp_observation_complete"):
            return False
        objects = gold_snap.get("objects", {})
        for obj in objects.values():
            if obj.get("grasped"):
                return False
        if gold_snap.get("held_objects"):
            return False
        preds_ok = self._eval_gold_predicates(gold_snap, gold)
        if not all(preds_ok):
            return False
        if case_id == "stove_exit":
            return turn_ready_without_static_entry(gold_snap)
        return gold_snap.get("strict_candidate") is True

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def run(self):
        if self.output_dir.exists():
            raise RuntimeError(f"output already exists: {self.output_dir}")

        self.verify_preflight()

        inputs_data, cases = self.load_inputs()

        self.run_review()

        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.import_runtime()

        qwen_budget = {"n": 0, "max": 18}
        qwen_client = self.vision.QwenClient(max_calls=18)

        stop_remaining = False
        for idx, case in enumerate(cases[: self.limit]):
            if stop_remaining:
                self.campaign_summary["cases"].append({
                    "case_id": case["case_id"],
                    "status": "not_run",
                })
                self.campaign_summary["statuses"].append("not_run")
                continue
            case_id = case["case_id"]
            _log(f"[case-begin] {case_id}")
            try:
                res = self.run_case(case, idx, qwen_client, qwen_budget)
            except Exception as exc:
                res = {
                    "case_id": case_id,
                    "status": "unknown",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "context_origin": "fixed_historical_subtask_fixture",
                }
            status = res.get("status", "unknown")
            self.campaign_summary["statuses"].append(status)
            self.campaign_summary["cases"].append({
                "case_id": case_id,
                "status": status,
                "phase_counts": res.get("phase_counts"),
                "errors": res.get("errors"),
            })
            for k, v in (res.get("phase_counts") or {}).items():
                if k in self.campaign_summary["phase_counts"]:
                    self.campaign_summary["phase_counts"][k] += int(v)
            _log(f"[case-end] {case_id} status={status}")
            _write_json(self.output_dir / case_id / "result.json", res)
            if status == "unknown":
                stop_remaining = True

        self.campaign_summary["qwen_calls"] = qwen_budget["n"]
        self.campaign_summary["new_vla"] = 0
        self.campaign_summary["new_hermes"] = 0
        _write_json(self.output_dir / "summary.json", self.campaign_summary)
        return self.campaign_summary


def main(argv=None):
    args = _parse_args(argv)
    runner = PilotRunner(args)
    try:
        runner.run()
    except Exception as exc:
        _log(f"[fatal] {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
