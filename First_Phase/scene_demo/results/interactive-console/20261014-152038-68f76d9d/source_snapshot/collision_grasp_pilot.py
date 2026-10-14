#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collision_grasp_pilot.py - thin collision-assisted pilot entry.

Software only: import and --help load no model, build no environment, open no
socket and never call Hermes.  All five literal requests, the shared 500-action
budget (including the 200-action helper cap), zero retries, the seeds, the true
Hermes plans and the independent literal gold stay inherited unchanged from
``side_grasp_pilot``.  This entry only injects ``collision_grasp`` as the grasp
module, rewrites the single preregistration.json payload to record the collision
condition, and refuses to run without a validated collision confirmation whose
recorded source hashes match the current raw source files.
"""

from __future__ import annotations

import json, math, os
from pathlib import Path
from types import SimpleNamespace

import collision_grasp, grasp_assist_service as gas, placement_experiments as pe, service, side_grasp_pilot as pilot  # noqa: E402

_HERE = Path(__file__).resolve().parent
EXPERIMENT_NAME = CONDITION = "collision_grasp_pilot"
HOST, PORT = "127.0.0.1", 8780
ENDPOINT = "http://127.0.0.1:8780"
PREREG_NAME = "preregistration.json"
COLLISION_SOURCE_FILES = ("collision_grasp.py", "collision_geometry.py", "side_grasp.py", "local_grasp.py")
PREREG_COLLISION_FILES = ("collision_grasp.py", "collision_geometry.py", "collision_grasp_pilot.py")


def _sha256_file(path):
    try: return pilot._sha256_file(path)
    except Exception: return None


def _raw_sha256(dir_path, name):
    try: return _sha256_file(Path(dir_path) / name)
    except Exception: return None


def validate_collision_confirmation(path, source_dir=None):
    """Fail-closed check of a collision-confirmation report BEFORE any run.

    The recorded ``source_sha256_after`` mapping MUST contain entries for
    collision_grasp.py, collision_geometry.py, side_grasp.py and local_grasp.py,
    each equal to the CURRENT raw source bytes under ``source_dir`` (defaulting
    to this module's directory).  Every required field must be exactly present
    and true; a missing or malformed value is rejected.
    """
    report_path = Path(path)
    if not report_path.is_file(): return None, "collision confirmation missing: %s" % path
    try: raw = report_path.read_bytes()
    except Exception as exc: return None, "collision confirmation unreadable: %s" % exc
    try: data = json.loads(raw.decode("utf-8"))
    except Exception as exc: return None, "collision confirmation JSON unreadable: %s" % exc
    if not isinstance(data, dict): return None, "collision confirmation is not a mapping"
    if data.get("candidate_confirmed") is not True: return None, "candidate_confirmed is not True"
    if data.get("phase") != "confirmed": return None, "phase is not 'confirmed'"
    if data.get("lift_five_sample_gate") is not True: return None, "lift_five_sample_gate is not True"
    lift_m = data.get("lift_m")
    if type(lift_m) not in (int, float) or not math.isfinite(lift_m) or lift_m < 0.02:
        return None, "lift_m is not a finite value >= 0.02"
    if data.get("final_bowl_predicate") is not True: return None, "final_bowl_predicate is not True"
    final = data.get("final_snapshot")
    objects = final.get("objects") if isinstance(final, dict) else None
    entry = objects.get("wine_bottle_1") if isinstance(objects, dict) else None
    grasped = entry.get("grasped") if isinstance(entry, dict) else None
    if grasped is not True: return None, "final_snapshot.objects.wine_bottle_1.grasped is not True"
    mapped = data.get("source_sha256_after")
    if not isinstance(mapped, dict): return None, "source_sha256_after missing"
    source_dir = Path(source_dir) if source_dir is not None else _HERE
    for name in COLLISION_SOURCE_FILES:
        recorded = mapped.get(name)
        if not isinstance(recorded, str) or not recorded:
            return None, "source_sha256_after[%s] missing/unknown" % name
        current = _raw_sha256(source_dir, name)
        if not isinstance(current, str) or recorded != current:
            return None, "%s SHA mismatch (recorded=%r current=%r)" % (name, recorded, current)
    return {"path": str(report_path), "sha256": _sha256_file(report_path),
            "source_sha256_after": dict(mapped)}, None


class CollisionPilotService(gas.GraspAssistService):
    """GraspAssistService that always uses the collision grasp module."""

    def __init__(self, *args, **kwargs):
        kwargs["grasp_module"] = collision_grasp
        super().__init__(*args, **kwargs)


def _collision_writer(actual_writer, prereg_dir, confirmation):
    """Build a writer intercepting ONLY prereg_dir/preregistration.json."""
    target = Path(prereg_dir)

    def _write(path, payload):
        if Path(path) != target:
            return actual_writer(path, payload)
        updated = dict(payload) if isinstance(payload, dict) else payload
        if isinstance(updated, dict):
            source_map = dict(updated.get("source_sha256") or {})
            for name in PREREG_COLLISION_FILES:
                source_map[name] = _raw_sha256(_HERE, name)
            updated["source_sha256"] = source_map
            updated["grasp_module"] = "collision_grasp"
            updated["controller"] = "collision_grasp.LocalGraspController"
            updated["collision_confirmation_path"] = confirmation.get("path")
            updated["collision_confirmation_sha256"] = confirmation.get("sha256")
        return actual_writer(path, updated)

    return _write


def run_collision_pilot(args, confirmation):
    """Run the inherited pilot with collision-only module/writer injection."""
    saved_gas, saved_pe = pilot.gas, pilot.pe
    saved_name, saved_condition = pilot.EXPERIMENT_NAME, pilot.CONDITION
    saved_host, saved_port, saved_endpoint = pilot.HOST, pilot.PORT, pilot.ENDPOINT
    actual_writer = pe._write_json_atomic
    output_dir = Path(args.output_dir)
    try:
        proxy_gas = SimpleNamespace(**vars(gas))
        proxy_gas.GraspAssistService = CollisionPilotService
        proxy_pe = SimpleNamespace(**vars(pe))
        proxy_pe._write_json_atomic = _collision_writer(actual_writer, output_dir / PREREG_NAME, confirmation)
        pilot.gas, pilot.pe = proxy_gas, proxy_pe
        pilot.EXPERIMENT_NAME = pilot.CONDITION = CONDITION
        pilot.HOST, pilot.PORT, pilot.ENDPOINT = HOST, PORT, ENDPOINT
        return pilot.run_pilot(args)
    finally:
        pilot.gas, pilot.pe = saved_gas, saved_pe
        pilot.EXPERIMENT_NAME, pilot.CONDITION = saved_name, saved_condition
        pilot.HOST, pilot.PORT, pilot.ENDPOINT = saved_host, saved_port, saved_endpoint


def _build_parser():
    parser = pilot._build_parser()
    parser.add_argument("--collision-confirmation", required=True,
                        help="absolute collision confirmation JSON path")
    return parser


def main(argv=None):
    parser = _build_parser(); args = parser.parse_args(argv)
    for name in ("output_dir", "calibration", "hermes_home", "collision_confirmation"):
        if not os.path.isabs(getattr(args, name)):
            parser.error("--%s must be an absolute path" % name.replace("_", "-"))
    if Path(args.output_dir).exists():
        parser.error("--output-dir already exists; refusing to reuse %s" % args.output_dir)
    if not Path(args.calibration).is_file(): parser.error("--calibration must be an existing file")
    if Path(args.hermes_home).exists():
        parser.error("--hermes-home already exists; refusing to reuse %s" % args.hermes_home)
    if not Path(args.collision_confirmation).is_file():
        parser.error("--collision-confirmation must be an existing file")
    confirmation, error = validate_collision_confirmation(args.collision_confirmation, _HERE)
    if error:
        print("collision confirmation rejected: %s" % error, file=os.sys.stderr)
        return 1
    return run_collision_pilot(args, confirmation)


if __name__ == "__main__":
    raise SystemExit(main())