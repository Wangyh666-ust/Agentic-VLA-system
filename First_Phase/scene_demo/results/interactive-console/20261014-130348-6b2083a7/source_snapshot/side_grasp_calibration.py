#!/usr/bin/env python3
"""Isolated wrapper: run grasp_calibration with the demonstrated-side grasp."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import grasp_calibration as cal  # noqa: E402
import side_grasp  # noqa: E402

cal.local_grasp = side_grasp
cal.EXPERIMENT_NAME = "side_grasp_calibration"
cal.SOURCE_FILES = tuple(cal.SOURCE_FILES) + ("side_grasp.py", "side_grasp_calibration.py")

_original_build_preregistration = cal._build_preregistration


def _build_preregistration(action_sha, historical_sha, action_count, historical_count):
    report = _original_build_preregistration(action_sha, historical_sha, action_count, historical_count)
    report["target_formula"] = (
        "p_target=wine_xyz+Rwine@p_relative; "
        "R_target=Rwine@R_relative@reading['body_to_controller_rotation']; "
        "approach_offset=-ABOVE_CLEARANCE_M*R_target[:,2]; above=p_target+approach_offset"
    )
    report["reference"] = side_grasp.reference_metadata()
    report["grasp_mode"] = "demonstrated_side"
    return report


cal._build_preregistration = _build_preregistration

main = cal.main

if __name__ == "__main__":
    sys.exit(main())
