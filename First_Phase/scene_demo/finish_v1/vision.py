import hashlib
import json
import math
import mimetypes
import os
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
    if operation not in ("place", "turn"):
        raise ValueError("current_subtask.operation must be 'place' or 'turn'")
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
            raise ValueError("current_subtask.destination must be None for turn")
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

    lines = [
        "You are the Hermes vision verifier for a robot task.",
        "Read the user's original request and the fixed completion criteria below.",
        "You must NEVER modify, reinterpret, or replace the user's original request",
        "or the fixed completion criteria. Judge only what the provided images show.",
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
        "geometry is insufficient.",
        "evidence and unknown_reasons must be lists of strings.",
        "Do not add fields. Do not add confidence. Do not add commentary outside JSON.",
        "Never infer task completion from a commanded gripper opening alone.",
        "Actual image-observed release, support, and stability decide completion.",
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
        allow = stable is True
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

    if tv is None:
        return _decision("unknown", "target visibility unknown for turn")
    if tv is False:
        return _decision("unknown", "target not visible for turn")
    if achieved is None:
        return _decision("unknown", "turn completion unknown")
    if achieved is False:
        return _decision("incomplete", "turn not yet achieved")
    if clear is False:
        return _decision(
            "needs_retreat",
            "turn achieved but clearance not satisfied",
            allow_retreat=True,
        )
    if clear is None:
        return _decision("unknown", "clearance unknown after turn")
    return _decision("complete", "turn achieved and clear")


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
        model="qwen3-vl-plus",
        max_calls=18,
    ):
        if not isinstance(env_path, (str, os.PathLike)):
            raise TypeError("env_path must be path-like")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a nonempty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int):
            raise TypeError("max_calls must be an int")
        if max_calls < 0:
            raise ValueError("max_calls must be nonnegative")
        self.env_path = env_path
        self.base_url = base_url
        self.model = model
        self.max_calls = max_calls
        self._calls = 0

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
        if self._calls >= self.max_calls:
            raise RuntimeError("Qwen call budget exhausted")
        prompt = build_prompt(context)
        key = self._resolve_key()
        if not key:
            raise RuntimeError("Qwen API key unavailable")
        data_urls = self._load_images(image_paths)

        content = [{"type": "text", "text": prompt}]
        for url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1600,
            "messages": [{"role": "user", "content": content}],
        }
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

        try:
            response = urllib.request.urlopen(request, timeout=90)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Qwen HTTP status {_safe_status(getattr(exc, 'code', None))}") from None
        except Exception:
            raise RuntimeError("Qwen network failure") from None

        try:
            try:
                status = response.getcode()
            except Exception:
                status = getattr(response, "status", None)
            raw = response.read()
        except Exception:
            raise RuntimeError("Qwen response read failure") from None

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            raise RuntimeError("Qwen response parse failure") from None

        try:
            response_content = _extract_content(payload)
            raw_verdict = _parse_verdict_content(response_content)
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
