#!/usr/bin/env python3
"""Read-only extraction of raw facts for four existing wine success/failure entries."""
import argparse, gzip, hashlib, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT_REL = "First_Phase/scene_demo/results/2026-10-09-safe-exit/analysis/wine_existing.json"

OLD_REL = "First_Phase/scene_demo/results/2026-10-09-joint-home-20"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def diff(a, b):
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)):
        return None
    vals = []
    for x, y in zip(a, b):
        if not (num(x) and num(y)):
            return None
        vals.append(abs(x - y))
    return max(vals) if vals else None


def edist(a, b):
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)):
        return None
    s = 0.0
    for x, y in zip(a, b):
        if not (num(x) and num(y)):
            return None
        s += (x - y) ** 2
    return math.sqrt(s)


class Reader:
    def __init__(self):
        self.files = {}

    def read(self, p, binary=True):
        data = p.read_bytes() if binary else p.read_text(encoding="utf-8").encode("utf-8")
        rel = p.resolve().relative_to(ROOT).as_posix()
        self.files[rel] = {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def json(self, p):
        self.read(p)
        return json.loads(p.read_text(encoding="utf-8"))


def stream_lines(p, r):
    r.read(p)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def wine_obj(snap):
    o = (snap or {}).get("objects", {}).get("wine_bottle_1", {})
    return o


def bowl_obj(snap):
    return (snap or {}).get("objects", {}).get("akita_black_bowl_1", {})


def predicate(snap):
    pr = (snap or {}).get("predicates", {})
    if isinstance(pr, dict):
        return pr.get("on|wine_bottle_1|wine_rack_1_top_region")
    return None


def relpath(p):
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return p.as_posix()


def extract_stage_metrics(stg):
    """Return the subset of wine stage fields we care about, preserving actual values."""
    if not isinstance(stg, dict):
        return None
    out = {}
    for k in ("row_count", "current_predicate_ever_true", "object_grasped_true_samples"):
        if k in stg:
            out[k] = stg[k]
    sc = stg.get("score")
    if isinstance(sc, dict):
        out["score"] = {"physical_gold_success": sc.get("physical_gold_success")}
    elif "score" in stg:
        out["score"] = stg["score"]
    return out


def extract_job(job):
    if not isinstance(job, dict):
        return None
    out = {}
    for k in ("steps", "instruction", "success", "ended_reason", "error"):
        if k in job:
            out[k] = job[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    expected = (ROOT / OUT_REL).resolve()
    scene = ROOT / "First_Phase" / "scene_demo"
    errs = []
    if not scene.exists():
        print("missing scene_demo", file=sys.stderr)
        return 2
    if out != expected:
        print("bad output path", file=sys.stderr)
        return 2
    if out.exists():
        print("output exists, refusing overwrite", file=sys.stderr)
        return 2

    r = Reader()
    OLD = ROOT / "First_Phase" / "scene_demo" / "results" / "2026-10-09-joint-home-20"
    metrics = None
    metrics_path = OLD / "evidence" / "metrics12.json"
    if metrics_path.exists():
        try:
            metrics = r.json(metrics_path)
        except Exception as e:
            print("failed to read metrics12.json:", e, file=sys.stderr)
            return 2
    if metrics is None:
        print("missing metrics12.json", file=sys.stderr)
        return 2

    pilot = OLD / "pilot"

    cases_meta = {c.get("case_id"): c for c in metrics.get("cases", []) if isinstance(c, dict)}

    cases = []
    unknowns = []
    checks = []
    errors = []

    for cid in ("case_01", "case_02", "case_03", "case_04"):
        cm = cases_meta.get(cid, {})
        wine_stage = None
        stages = cm.get("stages") if isinstance(cm, dict) else None
        if isinstance(stages, list):
            for s in stages:
                if isinstance(s, dict) and s.get("capability_id") == "wine_to_rack":
                    wine_stage = s
                    break
        rec = {"case_id": cid, "metrics": None, "spec": None, "source_files": []}
        if wine_stage is None:
            errors.append({"case": cid, "check": "wine_stage_missing", "error": "no stage with capability_id=='wine_to_rack'"})
        else:
            rec["metrics"] = extract_stage_metrics(wine_stage)
            job = wine_stage.get("job")
            if isinstance(job, dict):
                rec["job"] = extract_job(job)
            else:
                rec["job"] = None
                errors.append({"case": cid, "check": "job_missing", "error": "wine stage has no job dict"})
            for req in ("row_count", "current_predicate_ever_true", "object_grasped_true_samples"):
                if not isinstance(wine_stage, dict) or req not in wine_stage:
                    errors.append({"case": cid, "check": "required_metric_field_missing", "field": req})
        spec = cm.get("spec") if isinstance(cm, dict) else None
        if spec is None:
            errors.append({"case": cid, "check": "spec_missing", "error": "cm['spec'] absent"})
            rec["spec"] = None
        else:
            rec["spec"] = spec
        cases.append(rec)

    # Native entries 01/02
    for cid in ("case_01", "case_02"):
        rec = next(c for c in cases if c["case_id"] == cid)
        cdir = pilot / cid
        origin = cdir / "native_origin" / "robot_state.json"
        snap = cdir / "native_origin" / "snapshot.json"
        eef = cdir / "native_origin" / "eef_pose.json"
        home = cdir / "native_home.json"
        ent = {"robot_state_source": relpath(origin),
               "snapshot_source": relpath(snap),
               "eef_pose_source": relpath(eef),
               "native_home_source": relpath(home)}
        for key, path in (("robot_state", origin), ("native_snapshot", snap),
                          ("eef_pose", eef), ("native_home", home)):
            if path.exists():
                try:
                    ent[key] = r.json(path)
                except Exception as e:
                    errors.append({"case": cid, "file": relpath(path), "error": str(e)})
                    ent[key] = None
            else:
                ent[key] = None
                errors.append({"case": cid, "file": relpath(path), "error": "missing required file"})
        rs = ent.get("robot_state") or {}
        hs = ent.get("native_home") or {}
        if not isinstance(rs, dict):
            rs = {}
        if not isinstance(hs, dict):
            hs = {}
        ent["arm_qpos"] = rs.get("arm_qpos")
        ent["arm_qvel"] = rs.get("arm_qvel")
        ent["finger_qpos"] = rs.get("finger_qpos")
        ent["finger_qvel"] = rs.get("finger_qvel")
        ns = ent.get("native_snapshot") or {}
        if not isinstance(ns, dict):
            ns = {}
        wo = wine_obj(ns)
        bo = bowl_obj(ns)
        ent["wine_bottle_position"] = wo.get("position")
        ent["wine_bottle_quaternion"] = wo.get("quaternion")
        ent["bowl_position"] = bo.get("position")
        ent["bowl_quaternion"] = bo.get("quaternion")
        ep = ent.get("eef_pose") or {}
        if isinstance(ep, dict):
            ent["eef_pose"] = ep
        else:
            ent["eef_pose"] = None
        homeq = hs.get("homeq")
        finger_home = hs.get("finger_home")
        ent["max_abs_arm_qpos_diff_vs_homeq"] = diff(rs.get("arm_qpos"), homeq)
        ent["max_abs_finger_qpos_diff_vs_finger_home"] = diff(rs.get("finger_qpos"), finger_home)
        rec["entry"] = ent

    # Entries 03/04 from last row subtask_01_bowl_to_plate/home/home.jsonl.gz
    for cid in ("case_03", "case_04"):
        rec = next(c for c in cases if c["case_id"] == cid)
        cdir = pilot / cid
        gz = cdir / "subtask_01_bowl_to_plate" / "home" / "home.jsonl.gz"
        raw = cdir / "subtask_01_bowl_to_plate" / "home" / "home.jsonl"
        src = gz if gz.exists() else (raw if raw.exists() else None)
        if src is None:
            errors.append({"case": cid, "check": "home_stream_missing",
                           "source": relpath(cdir / "subtask_01_bowl_to_plate" / "home")})
            rec["entry"] = None
            continue
        last = None
        try:
            for row in stream_lines(src, r):
                last = row
        except Exception as e:
            errors.append({"case": cid, "file": relpath(src), "error": str(e)})
            rec["entry"] = None
            continue
        if last is None:
            rec["entry"] = None
            errors.append({"case": cid, "file": relpath(src), "error": "empty stream"})
            continue
        ent = {"source": relpath(src)}
        ent["step"] = last.get("step")
        ent["robot_state"] = last.get("robot_state")
        ent["eef_pose"] = last.get("eef_pose")
        ent["oracle_snapshot"] = last.get("oracle_snapshot")
        ent["state_sha256"] = last.get("state_sha256")
        ent["metrics"] = last.get("metrics")
        rec["entry"] = ent

        # native_home.json reference
        home = cdir / "native_home.json"
        if home.exists():
            try:
                nh = r.json(home)
            except Exception as e:
                errors.append({"case": cid, "file": relpath(home), "error": str(e)})
                nh = None
        else:
            nh = None
            errors.append({"case": cid, "file": relpath(home), "error": "missing required file"})
        rs = ent.get("robot_state") or {}
        if not isinstance(rs, dict):
            rs = {}
        if not isinstance(nh, dict):
            nh = {}
        ent["native_home_source"] = relpath(home)
        ent["native_home"] = nh if isinstance(nh, dict) else None
        ent["max_abs_arm_qpos_diff_vs_homeq"] = diff(rs.get("arm_qpos"), nh.get("homeq"))
        ent["max_abs_finger_qpos_diff_vs_finger_home"] = diff(rs.get("finger_qpos"), nh.get("finger_home"))

    # Observer streams and cross-checks
    for cid in ("case_01", "case_02", "case_03", "case_04"):
        rec = next(c for c in cases if c["case_id"] == cid)
        cdir = pilot / cid
        if cid in ("case_01", "case_02"):
            subdir = cdir / "subtask_01_wine_to_rack"
        else:
            subdir = cdir / "subtask_02_wine_to_rack"
        gz = subdir / "observer.jsonl.gz"
        raw = subdir / "observer.jsonl"
        src = gz if gz.exists() else (raw if raw.exists() else None)
        obs = {
            "grasp_true_step": None,
            "predicate_true_step": None,
            "current_ready_true_step": None,
            "first_wine_position": None,
            "first_wine_quaternion": None,
            "last_wine_position": None,
            "last_wine_quaternion": None,
            "first_post_action_wine_z": None,
            "max_wine_z_minus_first_post_action_z": None,
            "reference_step": None,
            "z_definition": None,
            "count_rows": 0,
            "grasp_true_count": 0,
            "predicate_true_count": 0,
            "current_ready_count": 0,
            "source": relpath(src) if src is not None else None,
        }
        if src is None:
            errors.append({"case": cid, "check": "observer_missing",
                           "source": relpath(subdir)})
        else:
            last = None
            first_seen = False
            first_post_action_z = None
            max_z = None
            try:
                for row in stream_lines(src, r):
                    obs["count_rows"] += 1
                    step = row.get("step")
                    snap = row.get("snapshot", {}) or {}
                    wo = wine_obj(snap)
                    pos = wo.get("position")
                    quat = wo.get("quaternion")
                    grasped = wo.get("grasped")
                    pred = predicate(snap)
                    ready = (row.get("current_ready") or {}).get("ready")
                    if not first_seen and pos is not None:
                        obs["first_wine_position"] = pos
                        obs["first_wine_quaternion"] = quat
                        first_seen = True
                    if grasped is True and obs["grasp_true_step"] is None:
                        obs["grasp_true_step"] = step
                    if pred is True and obs["predicate_true_step"] is None:
                        obs["predicate_true_step"] = step
                    if ready is True and obs["current_ready_true_step"] is None:
                        obs["current_ready_true_step"] = step
                    if grasped is True:
                        obs["grasp_true_count"] += 1
                    if pred is True:
                        obs["predicate_true_count"] += 1
                    if ready is True:
                        obs["current_ready_count"] += 1
                    if first_post_action_z is None and isinstance(pos, list) and len(pos) > 2:
                        first_post_action_z = pos[2]
                        obs["first_post_action_wine_z"] = first_post_action_z
                        obs["reference_step"] = step
                        obs["z_definition"] = "wine Z minus FIRST POST-ACTION observer wine Z; post-action = first observer row"
                    if isinstance(pos, list) and len(pos) > 2 and num(pos[2]):
                        if max_z is None or pos[2] > max_z:
                            max_z = pos[2]
                    last = row
            except Exception as e:
                errors.append({"case": cid, "file": relpath(src), "error": str(e)})
            if last is not None:
                wo = wine_obj(last.get("snapshot", {}))
                obs["last_wine_position"] = wo.get("position")
                obs["last_wine_quaternion"] = wo.get("quaternion")
            if max_z is not None and first_post_action_z is not None:
                obs["max_wine_z_minus_first_post_action_z"] = max_z - first_post_action_z
        rec["observer"] = obs

    # Mandatory cross-checks
    wine_success_values = []
    for cid, exp_rows, exp_success in (("case_01", 262, True), ("case_02", 180, True),
                                       ("case_03", 336, False), ("case_04", 336, False)):
        rec = next(c for c in cases if c["case_id"] == cid)
        m = rec.get("metrics")
        if not isinstance(m, dict):
            m = {}
        job = rec.get("job") if isinstance(rec.get("job"), dict) else {}
        obs = rec.get("observer") or {}
        # stage row_count
        got_rows = m.get("row_count")
        row_ok = (got_rows == exp_rows)
        checks.append({"case": cid, "check": "stage_row_count",
                       "expected": exp_rows, "actual": got_rows, "ok": row_ok})
        if got_rows is None:
            errors.append({"case": cid, "check": "row_count_missing"})
        elif not row_ok:
            errors.append({"case": cid, "check": "row_count_mismatch",
                           "expected": exp_rows, "actual": got_rows})
        # job steps
        got_steps = job.get("steps")
        steps_ok = (got_steps == exp_rows)
        checks.append({"case": cid, "check": "job_steps",
                       "expected": exp_rows, "actual": got_steps, "ok": steps_ok})
        if got_steps is None:
            errors.append({"case": cid, "check": "job_steps_missing"})
        elif not steps_ok:
            errors.append({"case": cid, "check": "job_steps_mismatch",
                           "expected": exp_rows, "actual": got_steps})
        # score.physical_gold_success
        sc = m.get("score") if isinstance(m.get("score"), dict) else {}
        got_succ = sc.get("physical_gold_success")
        succ_ok = (got_succ is exp_success)
        checks.append({"case": cid, "check": "physical_gold_success",
                       "expected": exp_success, "actual": got_succ, "ok": succ_ok})
        wine_success_values.append(got_succ)
        if got_succ is None:
            errors.append({"case": cid, "check": "physical_gold_success_missing"})
        elif not succ_ok:
            errors.append({"case": cid, "check": "physical_gold_success_mismatch",
                           "expected": exp_success, "actual": got_succ})
        # observer count == stage.row_count
        obs_rows = obs.get("count_rows")
        obs_ok = (obs_rows == got_rows) and (got_rows is not None)
        checks.append({"case": cid, "check": "observer_count_vs_stage_row_count",
                       "stage": got_rows, "observer": obs_rows, "ok": obs_ok})
        if obs_rows is None:
            errors.append({"case": cid, "check": "observer_count_missing"})
        elif got_rows is not None and obs_rows != got_rows:
            errors.append({"case": cid, "check": "observer_count_mismatch",
                           "stage": got_rows, "observer": obs_rows})
        # grasp true count == object_grasped_true_samples
        mg = m.get("object_grasped_true_samples")
        og = obs.get("grasp_true_count")
        if mg is None:
            errors.append({"case": cid, "check": "object_grasped_true_samples_missing"})
            checks.append({"case": cid, "check": "grasp_true_count_vs_samples",
                           "stage": None, "observer": og, "ok": False})
        else:
            gok = (og == mg)
            checks.append({"case": cid, "check": "grasp_true_count_vs_samples",
                           "stage": mg, "observer": og, "ok": gok})
            if og is None:
                errors.append({"case": cid, "check": "observer_grasp_count_missing"})
            elif not gok:
                errors.append({"case": cid, "check": "grasp_count_mismatch",
                               "stage": mg, "observer": og})
        # actual ever predicate true == current_predicate_ever_true
        mp = m.get("current_predicate_ever_true")
        actual_ever = obs.get("predicate_true_step") is not None if src is not None else None
        if mp is None:
            errors.append({"case": cid, "check": "current_predicate_ever_true_missing"})
            checks.append({"case": cid, "check": "ever_predicate_vs_metric",
                           "stage": None, "observer": actual_ever, "ok": False})
        else:
            pok = (actual_ever is not None) and (bool(mp) == bool(actual_ever))
            checks.append({"case": cid, "check": "ever_predicate_vs_metric",
                           "stage": mp, "observer": actual_ever, "ok": pok})
            if actual_ever is None:
                errors.append({"case": cid, "check": "observer_predicate_unverified"})
            elif not pok:
                errors.append({"case": cid, "check": "ever_predicate_mismatch",
                               "stage": mp, "observer": actual_ever})

    # First false / no True => null not absolute no grasp
    # (already encoded as None defaults; preserve)

    # Pairs 01vs03 and 02vs04
    pairs = []
    for a, b in (("case_01", "case_03"), ("case_02", "case_04")):
        ca = next(c for c in cases if c["case_id"] == a)
        cb = next(c for c in cases if c["case_id"] == b)
        ea = ca.get("entry") or {}
        eb = cb.get("entry") or {}
        ra = ea.get("robot_state") if isinstance(ea.get("robot_state"), dict) else {}
        rb = eb.get("robot_state") if isinstance(eb.get("robot_state"), dict) else {}
        sa = ea.get("native_snapshot") if isinstance(ea.get("native_snapshot"), dict) else {}
        sb = eb.get("native_snapshot") if isinstance(eb.get("native_snapshot"), dict) else {}
        if not sa:
            sa = ea.get("oracle_snapshot") if isinstance(ea.get("oracle_snapshot"), dict) else {}
        if not sb:
            sb = eb.get("oracle_snapshot") if isinstance(eb.get("oracle_snapshot"), dict) else {}
        p = {
            "pair": f"{a}_vs_{b}",
            "max_abs_arm_qpos": diff(ra.get("arm_qpos"), rb.get("arm_qpos")),
            "max_abs_arm_qvel": diff(ra.get("arm_qvel"), rb.get("arm_qvel")),
            "max_abs_finger_qpos": diff(ra.get("finger_qpos"), rb.get("finger_qpos")),
            "max_abs_finger_qvel": diff(ra.get("finger_qvel"), rb.get("finger_qvel")),
            "wine_translation_euclid": edist(wine_obj(sa).get("position"), wine_obj(sb).get("position")),
            "bowl_translation_euclid": edist(bowl_obj(sa).get("position"), bowl_obj(sb).get("position")),
        }
        for k, v in p.items():
            if k == "pair":
                continue
            if v is None:
                unknowns.append({"pair": p["pair"], "field": k, "source": "missing numeric arrays"})
        pairs.append(p)

    known_runner_facts = [
        "init_state and model_seed are fixed pairs in the historical runner (from root-reviewed metrics spec); "
        "seed is called once per case, and policy queues are cleared at subtask boundaries; "
        "no per-subtask RNG snapshots were recorded, and the later wine subtask follows preceding bowl policy "
        "consumption. These are acknowledged historical runner facts, not strict-pair conclusions."
    ]
    unknowns.extend([
        {"field": "native_two_view_actual_policy_input", "source": "not present in supplied inputs"},
        {"field": "native_two_view_processed_input", "source": "not derived from PNG"},
        {"field": "per_subtask_rng_snapshot", "source": "not snapshotted"},
    ])

    checks.append({"check": "metrics_wine_success_values", "values": wine_success_values})
    checks.append({"check": "extraction_only", "value": True})
    all_checks_ok = all((c.get("ok") is not False) for c in checks if "ok" in c)
    checks.append({"check": "mandatory_crosschecks_all_ok", "value": all_checks_ok})

    source_files = sorted(r.files.values(), key=lambda x: x["path"])
    for f in source_files:
        try:
            now = sha(ROOT / f["path"])
            if now != f["sha256"]:
                errors.append({"file": f["path"], "check": "changed_during_run"})
        except Exception as e:
            errors.append({"file": f["path"], "error": str(e)})

    ok = len(errors) == 0
    out_doc = {
        "cases": cases,
        "pairs": pairs,
        "known_runner_facts": known_runner_facts,
        "unknowns": unknowns,
        "source_files": source_files,
        "registry_evidence": {"checked": 0, "note": "not traversed per contract"},
        "checks": checks,
        "errors": errors,
        "ok": ok,
        "physical_actions": 0,
        "new_vla": 0,
        "new_hermes": 0,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out_doc, allow_nan=False, indent=1, sort_keys=True)
    try:
        with open(out, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except FileExistsError:
        print("output appeared, refusing overwrite", file=sys.stderr)
        return 2

    summary = []
    for c in cases:
        m = c.get("metrics") or {}
        sc = (m.get("score") or {}).get("physical_gold_success") if isinstance(m.get("score"), dict) else None
        summary.append({
            "case_id": c["case_id"],
            "row_count": m.get("row_count"),
            "success": sc,
            "pred": m.get("current_predicate_ever_true"),
            "grasp": m.get("object_grasped_true_samples"),
        })
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    print("output:", str(out))
    print("ok:", ok, "errors:", len(errors))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
