import hashlib
import json
import math
import mimetypes
import os
import time
import urllib.request
from pathlib import Path

_BOOL_FIELDS = (
    "target_visible",
    "destination_visible",
    "at_destination",
    "supported",
    "released",
    "stable",
    "operation_achieved",
    "clear_of_target",
)
_VERDICT_KEYS = set(_BOOL_FIELDS) | {"boxes", "evidence", "unknown_reasons"}
_BOX_KEYS = ("target", "destination", "gripper")
_ALLOWED_SUBTASK_KEYS = {"operation", "target", "destination", "phase"}


def _check_no_bool(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool")


def _validate_context(context):
    if not isinstance(context, dict):
        raise TypeError("context must be a dict")
    user_request = context.get("user_request")
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("user_request must be a nonempty string")
    subtask = context.get("current_subtask")
    if not isinstance(subtask, dict):
        raise TypeError("current_subtask must be a dict")
    operation = subtask.get("operation")
    if operation not in ("place", "turn", "articulate"):
        raise ValueError("current_subtask.operation must be place, turn or articulate")
    target = subtask.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("current_subtask.target must be a nonempty string")
    phase = subtask.get("phase")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("current_subtask.phase must be a nonempty string")
    destination = subtask.get("destination", None)
    if operation == "place":
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError("current_subtask.destination must be a nonempty string for place")
    else:
        if destination is not None:
            raise ValueError("current_subtask.destination must be None for non-placement operations")
    criteria = context.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("criteria must be a nonempty dict")
    return user_request, subtask, criteria


def _validate_frames(frames):
    if frames is None:
        return []
    if not isinstance(frames, list):
        raise TypeError("sensor_frames must be a list")
    cleaned = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(f"sensor_frames[{index}] must be a dict")
        out = {}
        fv = frame.get("frame")
        if isinstance(fv, bool) or not isinstance(fv, int):
            raise TypeError(f"sensor_frames[{index}].frame must be an int")
        out["frame"] = fv
        view = frame.get("view")
        if not isinstance(view, str) or not view.strip():
            raise ValueError(f"sensor_frames[{index}].view must be a nonempty string")
        out["view"] = view
        ts = frame.get("timestamp")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            raise TypeError(f"sensor_frames[{index}].timestamp must be numeric")
        if not math.isfinite(float(ts)):
            raise ValueError(f"sensor_frames[{index}].timestamp must be finite")
        out["timestamp"] = ts
        cleaned.append(out)
    return cleaned


def _validate_actuator_measurements(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("actuator_measurements must be a dict or None")
    stage = value.get("stage")
    finger_qpos = value.get("finger_qpos")
    max_abs_arm_qvel = value.get("max_abs_arm_qvel")
    if not isinstance(stage, str) or not stage:
        raise ValueError("actuator_measurements.stage must be a non-empty str")
    if not isinstance(finger_qpos, list) or len(finger_qpos) != 2:
        raise ValueError("actuator_measurements.finger_qpos must be a list of length 2")
    norm_fingers = []
    for item in finger_qpos:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("actuator_measurements.finger_qpos entries must be numbers")
        f = float(item)
        if not math.isfinite(f):
            raise ValueError("actuator_measurements.finger_qpos entries must be finite")
        norm_fingers.append(f)
    if isinstance(max_abs_arm_qvel, bool) or not isinstance(max_abs_arm_qvel, (int, float)):
        raise TypeError("actuator_measurements.max_abs_arm_qvel must be a number")
    max_speed = float(max_abs_arm_qvel)
    if not math.isfinite(max_speed) or max_speed < 0.0:
        raise ValueError("actuator_measurements.max_abs_arm_qvel must be finite and non-negative")
    normalized = {
        "stage": stage,
        "finger_qpos": norm_fingers,
        "max_abs_arm_qvel": max_speed,
    }
    if "standard_open_finger_qpos" in value:
        standard_open = _finite_pair(
            value["standard_open_finger_qpos"],
            "actuator_measurements.standard_open_finger_qpos",
        )
        normalized["standard_open_finger_qpos"] = [float(standard_open[0]), float(standard_open[1])]
    if "timestamp" in value:
        stamp = value["timestamp"]
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            raise TypeError("actuator_measurements.timestamp must be numeric")
        if not math.isfinite(float(stamp)):
            raise ValueError("actuator_measurements.timestamp must be finite")
        normalized["timestamp"] = float(stamp)
    return normalized

def build_prompt(context):
    user_request, subtask, criteria = _validate_context(context)
    frames = _validate_frames(context.get("sensor_frames"))

    safe_subtask = {
        "operation": subtask["operation"],
        "target": subtask["target"],
        "destination": subtask.get("destination"),
        "phase": subtask["phase"],
    }
    safe = {
        "user_request": user_request,
        "current_subtask": safe_subtask,
        "criteria": criteria,
    }
    if frames:
        safe["sensor_frames"] = frames
    measurements = _validate_actuator_measurements(context.get("actuator_measurements"))
    if measurements is not None and "timestamp" in measurements and frames:
        if abs(measurements["timestamp"] - max(f["timestamp"] for f in frames)) > 1e-9:
            raise ValueError("actuator measurement timestamp must match latest frame")
    if measurements is not None:
        safe["actuator_measurements"] = measurements

    lines = [
        "You are an independent vision inspector for this task; the whitelisted user_request, current_subtask, criteria and optional real encoders/frames are the only context.",
        "Hermes subtask guidance tells you what to attend to, but the user's original request and the fixed criteria must not be rewritten by any plan.",
        "First ground the target, destination, and gripper in the images; then judge their relationships; only then fill the exact original keys.",
        "Return exactly one JSON object with EXACTLY these keys and no others:",
        "target_visible, destination_visible, at_destination, supported, released,",
        "stable, operation_achieved, clear_of_target, boxes, evidence, unknown_reasons.",
        "Each of target_visible, destination_visible, at_destination, supported,",
        "released, stable, operation_achieved, clear_of_target must be true, false,",
        "or null. Use null whenever the property cannot be judged from the images.",
        "boxes must be an object keyed by camera/view name. Every camera entry must",
        "contain exactly target, destination, gripper. Each of those is either null",
        "or a normalized [x1, y1, x2, y2] array of four numbers with",
        "0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1. boxes may be an empty object when",
        "geometry is insufficient. Locate boxes independently per view; use each view's",
        "last image and never copy agentview coordinates into side or wrist views.",
        "evidence and unknown_reasons must be lists of strings.",
        "evidence should state the recognized category, the support relationship and the",
        "actual frame/view used; do not merely restate the task.",
        "Do not add fields. Do not add confidence. Do not add commentary outside JSON.",
        "An object ID is an alias: a name like black or any color word is NOT a guaranteed rendered color. Locate by category shape.",
        "LIBERO asset aliases alphabet_soup and tomato_sauce refer to cylindrical food cans, not soup bowls or sauce bottles; milk refers to a carton. Use the actual asset category and visible category/label evidence to identify the requested object. A target visible in one usable view is not globally invisible merely because another view is occluded; retain null if identity genuinely remains ambiguous.",
        "Generic category notes: bowl is a deep open container with an inner wall; plate is a shallow flat dish with a rim, and when a bowl sits on it usually only the outer ring is visible; stove is a cooktop with burner rings and knobs, and a round burner ring is not a plate; rack/shelf is a support frame; basket is a container; bottle is a bottle-shaped object; knob is a knob.",
        "Use the current ID's category; do NOT supply fixed scene positions, colors, or answers.",
        "If a bowl hides the center of a plate but the plate's outer rim is visible, do not say the plate is empty or invisible just because its center is hidden.",
        "If the bowl is beside the plate on the table rather than on the plate, at_destination is False.",
        "Position is not release. For place operation_achieved describes the placement relation at the destination, separately from released/clear/stable; do not require historical release footage or home posture.",
        "Judge the current instant, not a required video history of the release or carry event.",
        "In the current two views, if the fingers are visibly away from the target, released/clear may be True even without seeing the moment of opening; an open encoder alone does not prove release.",
        "This is short-window observed stability, not guaranteed future stability. With two or more distinct timestamps, judge visible support and motion over the supplied window, even if 0.05 or 0.1 seconds; do not invent a minimum duration, require a perturbation, or treat the absence of motion as absence of evidence. Set true only when support is visible and no target motion relative to its support is discernible; set false for visible displacement; null for single timestamp, unresolved support or inadequate image detail. Same-time different-camera frames and duplicate frames are not distinct-time evidence.",
        "Partial occlusion is not automatic success; if contact or support is unclear, fill Null; with no evidence do not claim grasp or contact.",
        "place.supported is support at the declared destination, not any table or the gripper.",
        "clear_of_target is local separation of the fingers/hand from target and support surface; arm may remain in the workspace. Do not demand leaving the entire workspace. For turn also check the relevant knob; do not expand it to unrelated knobs.",
        "For turn, judge by criteria.operation_semantics and the desired_state on/off with the burner ring/knob state: turn/on means rotate the stove knob to ignite; a visible red burner ring can be on evidence and is never a lid opening.",
        "Null is better than fabricating.",
        "Keep two stages: an operation that is achieved while still holding or touching hands off to a finishing stage; complete only after release and retreat, based on current evidence rather than a required view of a past event.",
        "Ignore and do not use any plan claim of success; judge only images and",
        "the fixed criteria. Do not accept self-reported model confidence.",
        "Report unknown whenever an object is occluded, evidence is insufficient,",
        "or images contradict. Null is better than guessing.",
        "",
        "Structured task context (safe fields only):",
        json.dumps(safe, ensure_ascii=False, sort_keys=True),
    ]
    if frames:
        lines.append("")
        lines.append(
            "Ordered image frames are provided below in this exact order; each image"
        )
        lines.append(
            "corresponds to the frame/view/timestamp entries listed above:"
        )
        for entry in frames:
            lines.append(
                f"- frame={entry['frame']} view={entry['view']} timestamp={entry['timestamp']}"
            )
    if measurements is not None:
        lines.append("Measurements bind only to the latest frame timestamp, never earlier frames. Closed fingers are not evidence of grasp; arm velocity is not object stability. Deterministic fusion applies encoder thresholds.")
        lines.append("Measurement context: actuator_measurements are actual joint encoder measurements and the current controller stage, not a claim that the task succeeded. An open gripper does not alone prove release. Inspect actual object support and separation in both camera views. Do not mistake projected overlap or an overhead gripper for grasp contact. Stable means observed support and no visible target motion across the supplied distinct timestamps; no perturbation experiment is required. If the fingers/object gap is occluded, return null, not an invented claim of holding.")
        lines.append("Fill released, clear_of_target, support and stable from images. Deterministic code compares the calibrated standard-open pair with measured fingers and applies the 0.002 encoder veto after your visual verdict. Closed fingers do not prove grasp, and open fingers do not prove success. Report visual support independently.")
        lines.append("Controller stage confirm is the post-retreat stopped-arm confirmation stage; stage home is the return-to-home stage. A stage label does not prove the preceding actions succeeded.")
    return "\n".join(lines)


def _validate_bool(value, name):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool or None")


def _validate_box(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be null or a four-number list")
    coords = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        coords.append(number)
    x1, y1, x2, y2 = coords
    if not (0.0 <= x1 < x2 <= 1.0):
        raise ValueError(f"{name} x bounds invalid")
    if not (0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{name} y bounds invalid")
    return coords


def _validate_boxes(boxes):
    if not isinstance(boxes, dict):
        raise TypeError("boxes must be a dict")
    out = {}
    for camera, entry in boxes.items():
        if not isinstance(camera, str) or not camera:
            raise ValueError("boxes camera keys must be nonempty strings")
        if not isinstance(entry, dict):
            raise TypeError("boxes camera entry must be a dict")
        extra = set(entry.keys()) - set(_BOX_KEYS)
        if extra:
            raise ValueError("boxes camera entry has unexpected keys")
        missing = set(_BOX_KEYS) - set(entry.keys())
        if missing:
            raise ValueError("boxes camera entry missing required keys")
        out[camera] = {
            key: _validate_box(entry[key], f"boxes.{camera}.{key}")
            for key in _BOX_KEYS
        }
    return out


def _validate_string_list(value, name):
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    out = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{index}] must be a string")
        out.append(item)
    return out


def validate_verdict(raw):
    if not isinstance(raw, dict):
        raise TypeError("verdict must be a dict")
    extra = set(raw.keys()) - _VERDICT_KEYS
    if extra:
        raise ValueError("verdict contains unexpected keys")
    missing = _VERDICT_KEYS - set(raw.keys())
    if missing:
        raise ValueError("verdict missing required keys")
    result = {}
    for field in _BOOL_FIELDS:
        result[field] = _validate_bool(raw[field], field)
    result["boxes"] = _validate_boxes(raw["boxes"])
    result["evidence"] = _validate_string_list(raw["evidence"], "evidence")
    result["unknown_reasons"] = _validate_string_list(
        raw["unknown_reasons"], "unknown_reasons"
    )
    return result


def _decision(state, reason, allow_release=False, allow_retreat=False):
    return {
        "state": state,
        "reason": reason,
        "allow_release": bool(allow_release),
        "allow_retreat": bool(allow_retreat),
    }


def _decide_place(verdict):
    tv = verdict["target_visible"]
    dv = verdict["destination_visible"]
    ad = verdict["at_destination"]
    supported = verdict["supported"]
    released = verdict["released"]
    stable = verdict["stable"]
    achieved = verdict["operation_achieved"]
    clear = verdict["clear_of_target"]

    if tv is None or dv is None:
        return _decision("unknown", "object or destination visibility unknown")
    if tv is False or dv is False:
        return _decision("unknown", "target or destination not visible")
    if ad is None:
        return _decision("unknown", "placement position unknown")
    if ad is False:
        return _decision("incomplete", "target not at destination")

    if released is None:
        return _decision("unknown", "release state unknown")
    if released is False:
        if supported is False:
            return _decision(
                "incomplete", "unsupported release unsafe"
            )
        if supported is None:
            return _decision("unknown", "support state unknown before release")
        allow = stable is not False
        return _decision(
            "needs_release",
            "target in place and supported; release required",
            allow_release=allow,
        )

    if supported is False or stable is False:
        return _decision(
            "incomplete", "released but support or stability failed"
        )
    if supported is None or stable is None:
        return _decision(
            "unknown", "release confirmed but support or stability unknown"
        )

    if clear is False:
        return _decision(
            "needs_retreat",
            "placed and stable but gripper still obstructs target",
            allow_retreat=True,
        )
    if clear is None:
        return _decision("unknown", "gripper clearance unknown after placement")
    return _decision("complete", "placement stable and clear", allow_retreat=False)


def _decide_turn(verdict):
    tv = verdict["target_visible"]
    achieved = verdict["operation_achieved"]
    clear = verdict["clear_of_target"]
    released = verdict["released"]

    if tv is None:
        return _decision("unknown", "target visibility unknown for turn")
    if tv is False:
        return _decision("unknown", "target not visible for turn")
    if achieved is None:
        return _decision("unknown", "turn completion unknown")
    if achieved is False:
        return _decision("incomplete", "turn not yet achieved")
    if released is False:
        return _decision(
            "needs_release",
            "knob-operation done release fingers",
            allow_release=True,
        )
    if clear is False:
        return _decision(
            "needs_retreat",
            "turn achieved but clearance not satisfied",
            allow_retreat=True,
        )
    if clear is None:
        return _decision("unknown", "clearance unknown after turn")
    if released is None:
        return _decision("unknown", "release state unknown before retreat")
    return _decision("complete", "turn achieved and clear")


def _finite_pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair of two finite numbers")
    out = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        out.append(number)
    return out


def _box_separation_norm(box_a, box_b):
    dx = max(
        0.0,
        max(box_a[0], box_b[0]) - min(box_a[2], box_b[2]),
    )
    dy = max(
        0.0,
        max(box_a[1], box_b[1]) - min(box_a[3], box_b[3]),
    )
    return (dx * dx + dy * dy) ** 0.5


def reconcile_verdict(context, verdict, robot, reference):
    _validate_context(context)
    clean = validate_verdict(verdict)
    out = {
        field: clean[field]
        for field in _BOOL_FIELDS
    }
    out["boxes"] = {
        camera: dict(entry) for camera, entry in clean["boxes"].items()
    }
    out["evidence"] = list(clean["evidence"])
    out["unknown_reasons"] = list(clean["unknown_reasons"])

    if not isinstance(robot, dict):
        raise TypeError("robot must be a dict")
    if not isinstance(reference, dict):
        raise TypeError("reference must be a dict")

    if context.get("sensor_frames"):
        try:
            frames = _validate_frames(context["sensor_frames"])
            measurements = _validate_actuator_measurements(context.get("actuator_measurements"))
            valid = measurements is not None and "timestamp" in measurements and "standard_open_finger_qpos" in measurements
            valid = valid and abs(measurements["timestamp"] - max(f["timestamp"] for f in frames)) <= 1e-9
            robot_pair = _finite_pair(robot.get("finger_qpos"), "robot.finger_qpos")
            home_pair = _finite_pair(reference.get("finger_home"), "reference.finger_home")
            valid = valid and all(abs(a-b) <= 1e-9 for a,b in zip(measurements["finger_qpos"], robot_pair))
            valid = valid and all(abs(a-b) <= 1e-9 for a,b in zip(measurements["standard_open_finger_qpos"], home_pair))
        except (TypeError, ValueError, KeyError):
            valid = False
        if not valid:
            out["released"] = None
            out["clear_of_target"] = None
            out["unknown_reasons"].append("missing_or_stale_encoder")
            out["evidence"].append("missing_or_stale_encoder: latest visual frame lacks matching measured feedback and reference")
            return out

    finger_qpos = _finite_pair(robot.get("finger_qpos"), "robot.finger_qpos")
    finger_home = _finite_pair(reference.get("finger_home"), "reference.finger_home")

    deviation = max(
        abs(finger_qpos[0] - finger_home[0]),
        abs(finger_qpos[1] - finger_home[1]),
    )
    if deviation > 0.002:
        out["released"] = False
        out["clear_of_target"] = False
        out["evidence"].append(
            "encoder fingers not at standard open position; this does not prove object grasp"
        )

    clear_near = False
    for entry in out["boxes"].values():
        gripper = entry["gripper"]
        if gripper is None:
            continue
        for key in ("target", "destination"):
            box = entry[key]
            if box is None:
                continue
            if _box_separation_norm(gripper, box) <= 0.03:
                clear_near = True
                break
        if clear_near:
            break

    if clear_near:
        out["evidence"].append(
            "2D projection overlap near target/support; not 3D contact evidence"
        )

    return out


def needs_relation_review(context, raw, robot, reference):
    try:
        _, subtask, _ = _validate_context(context)
        clean = validate_verdict(raw)
        frames = _validate_frames(context.get("sensor_frames"))
        if not frames or len({f["timestamp"] for f in frames}) < 2 or len({f["view"] for f in frames}) < 2:
            return False
        probe = {**clean, **{k: True for k in _BOOL_FIELDS}}
        fused = reconcile_verdict(context, probe, robot, reference)
        if fused["released"] is None or fused["clear_of_target"] is None or "missing_or_stale_encoder" in fused["unknown_reasons"]:
            return False
        latest = max(f["timestamp"] for f in frames)
        views = {f["view"] for f in frames if f["timestamp"] == latest}
        if subtask["operation"] == "turn":
            return clean["target_visible"] is True and clean["operation_achieved"] is not True and sum(clean["boxes"].get(v, {}).get("target") is not None for v in views) >= 2
        if len(views) >= 2 and clean["target_visible"] is True and clean["destination_visible"] is True and (clean["at_destination"] is not True or clean["supported"] is not True):
            for view in views:
                boxes = clean["boxes"].get(view, {})
                target, destination = boxes.get("target"), boxes.get("destination")
                if target is not None and destination is not None:
                    if min(target[2], destination[2]) - max(target[0], destination[0]) > 0 and min(target[3], destination[3]) - max(target[1], destination[1]) > 0:
                        return True
        if not all(clean[k] is True for k in ("target_visible", "destination_visible", "at_destination", "supported")):
            return False
        if clean["released"] is not None or clean["clear_of_target"] is not False or clean["stable"] is not None:
            return False
        if fused["released"] is not True or fused["clear_of_target"] is not True:
            return False
        qualified = 0
        for view in views:
            boxes = clean["boxes"].get(view)
            if boxes and all(boxes[k] is not None for k in _BOX_KEYS):
                if all(_box_separation_norm(boxes["gripper"], boxes[k]) >= .10 for k in ("target", "destination")):
                    qualified += 1
        return qualified >= 2
    except (ValueError, TypeError, KeyError):
        return False


def build_relation_prompt(context):
    _, subtask, criteria = _validate_context(context)
    frames = _validate_frames(context.get("sensor_frames"))
    measurements = _validate_actuator_measurements(context.get("actuator_measurements"))
    if not frames or measurements is None or "timestamp" not in measurements:
        raise ValueError("relation review requires synchronized measurements and frames")
    if abs(measurements["timestamp"] - max(f["timestamp"] for f in frames)) > 1e-9:
        raise ValueError("actuator measurement timestamp must match latest frame")
    safe = {"operation": subtask["operation"], "target": subtask["target"], "destination": subtask.get("destination"),
            "sensor_frames": frames,
            "actuator_measurements": {k: measurements[k] for k in ("timestamp", "finger_qpos", "standard_open_finger_qpos") if k in measurements}}
    if subtask["operation"] == "turn":
        desired = criteria.get("desired_state")
        if desired not in ("on", "off"):
            raise ValueError("turn desired_state must be on or off")
        safe["desired_state"] = desired
    derivatives = context.get("relation_derivatives") or []
    if not isinstance(derivatives, list):
        raise TypeError("relation_derivatives must be a list")
    mappings = []
    allowed = {"frame", "source_frame", "view", "timestamp", "source_size_px", "pixel_bounds", "original_roi"}
    for d in derivatives:
        if not isinstance(d, dict) or set(d) != allowed:
            raise ValueError("relation derivative mapping has unexpected or missing fields")
        frame, source = d["frame"], d["source_frame"]
        if isinstance(frame,bool) or isinstance(source,bool) or not isinstance(frame,int) or not isinstance(source,int) or not (0 <= source < frame < len(frames)):
            raise ValueError("relation derivative frame mapping invalid")
        if frames[frame]["view"] != d["view"] or frames[source]["view"] != d["view"] or frames[frame]["timestamp"] != d["timestamp"] or frames[source]["timestamp"] != d["timestamp"] or d["timestamp"] != measurements["timestamp"]:
            raise ValueError("relation derivative view/time mismatch")
        size, bounds, roi = d["source_size_px"], d["pixel_bounds"], d["original_roi"]
        if not isinstance(size,list) or len(size)!=2 or any(isinstance(x,bool) or not isinstance(x,int) or x<=0 for x in size):
            raise ValueError("relation derivative source dimensions invalid")
        if not isinstance(bounds,list) or len(bounds)!=4 or any(isinstance(x,bool) or not isinstance(x,int) for x in bounds):
            raise ValueError("relation derivative pixel bounds invalid")
        w,h=size
        if not (0<=bounds[0]<bounds[2]<=w and 0<=bounds[1]<bounds[3]<=h):
            raise ValueError("relation derivative pixel bounds outside source")
        clean_roi = _validate_box(roi,"relation derivative original_roi")
        if clean_roi is None or any(abs(a-b)>1e-9 for a,b in zip(clean_roi,[bounds[0]/w,bounds[1]/h,bounds[2]/w,bounds[3]/h])):
            raise ValueError("relation derivative ROI mapping mismatch")
        mappings.append(dict(d))
    if mappings:
        safe["relation_derivatives"] = mappings
    return "\n".join([
        "Independently inspect the current image relationships. Locate the target, destination and gripper in each view.",
        ("For turn, inspect the CURRENT visible burner-ring/knob state. turn/on means rotate the stove knob to ignite; a red burner ring is visible on evidence, not opening a lid. Historical rotation footage is unnecessary. Wrist occlusion means unknown in that view and does not override visible evidence in another view; both views need not show the burner. With insufficient current state evidence use null. Encoders cannot establish ignition. at_destination and supported may be null; operation_achieved independently states whether desired_state is currently achieved." if subtask["operation"] == "turn" else "For place, inspect the current placement relation."),
        "When derivative mappings are supplied, four complete images are followed by two local details cropped from the latest original views. These crops are not new views or timestamps. Candidate regions may be wrong: reidentify categories and support, using null for ambiguity. Output boxes in original full-image normalized coordinates, mapping detail coordinates through original_roi. Neither crop selection nor boxes establish contact or completion.",
        "An object ID is an alias: a name like black or any color word is NOT a guaranteed rendered color. Locate by category shape.",
        "LIBERO asset aliases alphabet_soup and tomato_sauce refer to cylindrical food cans, not soup bowls or sauce bottles; milk refers to a carton. Use the actual asset category and visible category/label evidence to identify the requested object. A target visible in one usable view is not globally invisible merely because another view is occluded; retain null if identity genuinely remains ambiguous.",
        "Generic category notes: bowl is a deep open container with an inner wall; plate is a shallow flat dish with a rim, and when a bowl sits on it usually only the outer ring is visible; stove is a cooktop with burner rings and knobs, and a round burner ring is not a plate; rack/shelf is a support frame; basket is a container; bottle is a bottle-shaped object; knob is a knob.",
        "If a bowl hides the center of a plate but the plate's outer rim is visible, do not say the plate is empty or invisible just because its center is hidden.",
        "target_visible and destination_visible mean the identified target and destination are visible. at_destination describes the current target relation to the identified destination. supported means the destination supports the target, distinct from support by the table.",
        "released means the current fingers no longer hold or contact the target, independent of placement success. Visibly distant hands can establish released=true without historical opening footage, target movement, or resolving distant finger gaps. clear_of_target means local separation of the entire hand/fingers from the target and destination support, without requiring departure from the whole workspace; projected overhead position or 2D overlap does not establish contact.",
        "stable means no visible target motion relative to its visible support across distinct timestamps, independent of open fingers or arm velocity. For place, operation_achieved describes only whether the current placement relation holds; do not require historical carrying, release, or return home. Use null for occlusion or unclear evidence; a negative value requires visible contrary evidence. Open fingers alone do not prove placement, support, or separation.",
        "Observe separately: is the target at the destination; is the destination supporting it; are the current fingers/hand separated from target and support; is there visible target motion relative to support across distinct timestamps?",
        "Visible support and no discernible relative motion can establish short-window stability. Historical release footage is unnecessary. Placement position and operation relation are independent from release and clearance. Open fingers alone do not establish success. Occlusion, inadequate detail or conflict means null. Encoders bind only to the latest timestamp; deterministic code applies the encoder veto.",
        "Return exactly one JSON object with these keys: " + ", ".join(_BOOL_FIELDS) + ", boxes, evidence, unknown_reasons.",
        "Each boolean field is true, false or null. boxes is keyed by view; each view contains exactly target, destination, gripper, each null or normalized [x1,y1,x2,y2] with 0<=x1<x2<=1 and 0<=y1<y2<=1. Use the latest image per view. evidence and unknown_reasons are lists of strings. No other keys or commentary.",
        json.dumps(safe, ensure_ascii=False, sort_keys=True),
    ])


def decide(context, verdict):
    _, subtask, _ = _validate_context(context)
    clean_verdict = validate_verdict(verdict)
    if subtask["operation"] == "place":
        return _decide_place(clean_verdict)
    return _decide_turn(clean_verdict)


def _safe_status(status):
    if status is None:
        return "unknown"
    try:
        return int(status)
    except Exception:
        return "unknown"


def _read_env_key(env_path):
    try:
        path = Path(env_path)
    except Exception:
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != "DASHSCOPE_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value or None
    return None


def _guess_mime(path):
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed in ("image/jpeg", "image/jpg"):
        return "image/jpeg"
    if guessed == "image/png":
        return "image/png"
    if guessed == "image/webp":
        return "image/webp"
    suffix = Path(str(path)).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return None


def _extract_content(payload):
    if not isinstance(payload, dict):
        raise ValueError("server response not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("server response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("server choice not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("server choice missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("server message content not a string")
    return content


def _parse_verdict_content(content):
    text = content.strip()
    if text.startswith('```'):
        if not text.endswith('```') or text.count('```') != 2:
            raise ValueError("invalid fenced block")
        lines = text.splitlines()
        if len(lines) < 2:
            raise ValueError("invalid fenced block")
        first = lines[0]
        if first not in ('```json', '```JSON', '```'):
            raise ValueError("invalid fence opener")
        last = lines[-1]
        if last != '```':
            raise ValueError("invalid fence closer")
        body = '\n'.join(lines[1:-1])
    else:
        body = text
    try:
        obj = json.loads(body)
    except Exception as e:
        raise ValueError("invalid JSON") from e
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


class QwenClient:
    def __init__(
        self,
        env_path="/home/yhwang/fyp/scene_demo/hermes_home/.env",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max",
        max_calls=18,
        request_timeout=90,
    ):
        if not isinstance(env_path, (str, os.PathLike)):
            raise TypeError("env_path must be path-like")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a nonempty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if model != "qwen3.8-max":
            raise ValueError("All scene VLM clients require qwen3.8-max")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int):
            raise TypeError("max_calls must be an int")
        if max_calls < 0:
            raise ValueError("max_calls must be nonnegative")
        if isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float)) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError('request_timeout must be finite and positive')
        self.env_path = env_path
        self.base_url = base_url
        self.model = model
        self.max_calls = max_calls
        self._calls = 0
        self.request_timeout = request_timeout

    def _resolve_key(self):
        key = os.environ.get("DASHSCOPE_API_KEY")
        if isinstance(key, str) and key.strip():
            return key.strip()
        return _read_env_key(self.env_path)

    def _load_images(self, image_paths):
        if not isinstance(image_paths, list):
            raise TypeError("image_paths must be a list")
        if not (1 <= len(image_paths) <= 6):
            raise ValueError("image_paths must contain 1..6 entries")
        items = []
        for index, entry in enumerate(image_paths):
            if not isinstance(entry, (str, os.PathLike)):
                raise TypeError(f"image_paths[{index}] must be path-like")
            path = Path(entry)
            data = path.read_bytes()
            mime = _guess_mime(path)
            if mime is None:
                raise ValueError(f"image_paths[{index}] unsupported type")
            items.append(f"data:{mime};base64,{__import__('base64').b64encode(data).decode('ascii')}")
        return items

    def analyze(self, context, image_paths):
        return self._request(context, image_paths, build_prompt(context))

    def inspect_relations(self, context, image_paths):
        return self._request(context, image_paths, build_relation_prompt(context))

    def _request(self, context, image_paths, prompt):
        if self._calls >= self.max_calls:
            raise RuntimeError("Qwen call budget exhausted")
        key = self._resolve_key()
        if not key:
            raise RuntimeError("Qwen API key unavailable")
        data_urls = self._load_images(image_paths)

        frames = _validate_frames(context.get("sensor_frames")) if isinstance(context, dict) else []
        if context.get("sensor_frames") is not None and len(frames) != len(data_urls):
            raise ValueError("sensor_frames length must match image_paths length")
        for index, frame in enumerate(frames):
            if frame["frame"] != index:
                raise ValueError("sensor_frames frame indices must be 0..n-1 in order")

        content = [{"type": "text", "text": prompt}]
        if frames:
            for index, url in enumerate(data_urls):
                frame = frames[index]
                content.append({"type": "text", "text": f"Image frame={frame['frame']}; view={frame['view']}; timestamp={frame['timestamp']}"})
                content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            for index, url in enumerate(data_urls):
                content.append({"type": "text", "text": f"Image frame={index}; view=unspecified; timestamp=unspecified"})
                content.append({"type": "image_url", "image_url": {"url": url}})
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1600,
            "messages": [{"role": "user", "content": content}],
        }
        body.update(getattr(self, 'request_options', {}))
        if body['model'] != 'qwen3.8-max':
            raise ValueError('VLM request model must be qwen3.8-max')
        request_bytes = json.dumps(body).encode("utf-8")
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=request_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
        )

        self._calls += 1
        call_index = self._calls
        if hasattr(self, 'record_request'):
            self.record_request(context, image_paths, prompt, request_bytes, call_index)
        started = time.monotonic()
        def transport_error(exc, stage):
            reason = getattr(exc, 'reason', exc)
            error = {'stage': stage, 'error_type': type(exc).__name__,
                     'reason_type': type(reason).__name__, 'errno': getattr(reason, 'errno', None),
                     'http_status': _safe_status(getattr(exc, 'code', None)),
                     'timeout_seconds': self.request_timeout,
                     'elapsed_seconds': round(time.monotonic() - started, 3)}
            if hasattr(self, 'record_transport_error'):
                self.record_transport_error(call_index, error)
            if isinstance(reason, TimeoutError):
                return f'Qwen request timed out after {self.request_timeout:g}s ({stage})'
            return f'Qwen network failure ({type(reason).__name__}; {stage})'
        try:
            response = urllib.request.urlopen(request, timeout=self.request_timeout)
        except urllib.error.HTTPError as exc:
            transport_error(exc, 'http_response')
            raise RuntimeError(f"Qwen HTTP status {_safe_status(getattr(exc, 'code', None))}") from None
        except Exception as exc:
            raise RuntimeError(transport_error(exc, 'awaiting_response')) from None

        try:
            try:
                status = response.getcode()
            except Exception:
                status = getattr(response, "status", None)
            raw = response.read()
        except Exception as exc:
            raise RuntimeError(transport_error(exc, 'reading_response')) from None
        finally:
            if hasattr(response, 'close'):
                response.close()

        if hasattr(self, 'record_response'):
            self.record_response(context, image_paths, prompt, request_bytes, raw, call_index, status)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            raise RuntimeError("Qwen response parse failure") from None

        if not isinstance(payload, dict):
            raise RuntimeError('Qwen response parse failure')
        if payload.get('model') and payload['model'] != 'qwen3.8-max':
            raise RuntimeError('Qwen returned an unexpected model: ' + str(payload['model']))

        try:
            response_content = _extract_content(payload)
            parser = getattr(self, 'parse_verdict', _parse_verdict_content)
            raw_verdict = parser(response_content)
            verdict = validate_verdict(raw_verdict)
        except ValueError:
            raise RuntimeError("Qwen response parse failure") from None
        except TypeError:
            raise RuntimeError("Qwen response parse failure") from None

        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            usage = {}
        server_model = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(server_model, str) or not server_model:
            server_model = self.model
        request_sha = hashlib.sha256(request_bytes).hexdigest()
        response_sha = hashlib.sha256(raw).hexdigest()
        return {
            "verdict": verdict,
            "usage": usage,
            "model": server_model,
            "http_status": _safe_status(status),
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "call_index": call_index,
        }
