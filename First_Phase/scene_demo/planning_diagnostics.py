#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""planning_diagnostics.py — capture-only Hermes planning benchmark (no execution).

Purpose
-------
Measure whether the REAL installed Hermes (``run_agent.HermesRunner``) can
*plan* against a small, fixed set of scene requests -- and nothing more.  This
module NEVER executes a robot, NEVER touches the simulator/policy/VLA service
(8767), NEVER calls the production service profile and NEVER repairs or
resubmits.  It stands up a tiny 127.0.0.1:8778 fake scene service that only
*records* the plan the agent submits; scoring is a pure local function.

Design boundaries
-----------------
* Module import and ``--help`` must not invoke Hermes, the network or the GPU:
  ``run_agent``/``catalog``/``yaml`` are imported lazily inside functions.
* The public scene returned to the agent is derived from the input package's
  ``session.json`` PUBLIC fields only, with ``scene_version=0``, ``state=ready``
  and ``total_steps=0`` over a fresh fake session uuid.  The ONLY image is the
  real native agentview ``.../cap_01_bowl_to_plate_5eb0b463/first.png`` -- the
  overwritten original agentview/wrist are never used and no second camera or
  extra view is fabricated.
* Case ids and independent-oracle fixture fields NEVER enter the public data or
  the prompt; they are asserted absent.
* ``score_plan`` is a pure scorer over the LITERAL fixture fields
  (``goal_options`` / ``allowed_decisions`` / ``allowed_objects``); it is not a
  plan-derived oracle and it never claims physical/task success.

CLI
---
  python3 planning_diagnostics.py --input-package PATH --output-dir PATH \
                                  --hermes-home PATH

Only three files are ever authored: this module and its focused test module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- #
# Fixed constants
# --------------------------------------------------------------------------- #
# The managed scene_demo directory is this module's own directory; nothing is
# hard-coded to a host path that could drift.
MANAGED_SCENE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8778
ENDPOINT = "http://127.0.0.1:8778"

# Real source runtime home (copied FROM; never modified).
SOURCE_HOME = "/home/yhwang/fyp/scene_demo/hermes_home"

HERMES_TOOLS = "scene_tools,vision"
MODEL_NAME = "qwen3-vl-plus"
HERMES_TIMEOUT = 180

# The ONLY image the agent may see: the real native agentview capture.
CAP_IMAGE_RELPATH = ("cap_01_bowl_to_plate_5eb0b463", "first.png")

# Five fixed cases, in this exact order, from the sibling fixtures.json.
CASE_IDS = (
    "table_tidy",
    "table_bowl_only",
    "table_wine_only",
    "missing_bin",
    "ambiguous_cleanup",
)

# Metadata that must NEVER appear in public data or in the prompt.
FORBIDDEN_PUBLIC_KEYS = (
    "case_id",
    "goal_options",
    "allowed_decisions",
    "allowed_objects",
    "protected_goals",
    "protected_positions",
    "oracle_source",
)

# Sentinel for an unparseable request body.
_BAD_BODY = object()

__all__ = [
    "MANAGED_SCENE_DIR",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ENDPOINT",
    "SOURCE_HOME",
    "HERMES_TOOLS",
    "MODEL_NAME",
    "HERMES_TIMEOUT",
    "CASE_IDS",
    "FORBIDDEN_PUBLIC_KEYS",
    "build_public_session",
    "score_plan",
    "FakeSceneContext",
    "make_capture_server",
    "prepare_home",
    "patch_config",
    "config_sha",
    "run_benchmark",
    "main",
]


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path, payload):
    """Write JSON atomically: temp file in the same directory + os.replace."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _write_bytes(path, data):
    with open(path, "wb") as handle:
        handle.write(data)


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# --------------------------------------------------------------------------- #
# Public scene construction (public data only)
# --------------------------------------------------------------------------- #
def build_public_session(input_session, session_id, image_path):
    """Build the immutable public session the agent is allowed to see.

    Only PUBLIC fields from ``input_session`` are copied; any case/evaluation/
    fixture-expectation key (``FORBIDDEN_PUBLIC_KEYS``) is stripped if present.
    ``scene_version`` is pinned to 0, ``state`` to ``ready``, ``total_steps``
    to 0 and the session id to the supplied fake uuid.  The image list holds
    exactly ONE native agentview image (``image_path``) -- no second camera and
    no extra view is fabricated.
    """
    src = input_session if isinstance(input_session, dict) else {}

    capabilities = []
    for capability in src.get("capabilities") or []:
        if not isinstance(capability, dict):
            continue
        entry = {
            key: copy.deepcopy(value)
            for key, value in capability.items()
            if key not in FORBIDDEN_PUBLIC_KEYS
        }
        capabilities.append(entry)

    session = {
        "ok": True,
        "session_id": session_id,
        "scene_id": src.get("scene_id"),
        "state": "ready",
        "scene_version": 0,
        "total_steps": 0,
        "description": src.get("description"),
        "storage_policy": copy.deepcopy(src.get("storage_policy") or {}),
        "capabilities": capabilities,
        "images": [
            {
                "view": "agentview",
                "kind": "vla_observation",
                "image_path": str(image_path),
            }
        ],
        "latest_png": str(image_path),
    }
    for key in FORBIDDEN_PUBLIC_KEYS:
        session.pop(key, None)
    return session


def _oracle_excluded(case, public_session, prompt):
    """Concrete assertion that no oracle/case metadata leaked.

    Returns False (never raises) if any forbidden token appears in the prompt
    or in the serialized public session, so the benchmark can record the leak
    as an operational error while preserving the five raw case results.
    """
    blob = json.dumps(public_session, ensure_ascii=False)
    tokens = list(FORBIDDEN_PUBLIC_KEYS) + [str(case.get("case_id"))]
    try:
        for token in tokens:
            assert token, "empty forbidden token"
            assert token not in prompt, "oracle token in prompt: %s" % token
            assert token not in blob, "oracle token in public session: %s" % token
    except AssertionError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Pure scorer (literal fixture fields only -- no plan-derived oracle)
# --------------------------------------------------------------------------- #
def _normalize_capabilities(capabilities):
    """Return ``{capability_id: capability_dict}`` from a list or a mapping."""
    out = {}
    if isinstance(capabilities, dict):
        for capability_id, capability in capabilities.items():
            if isinstance(capability, dict):
                entry = copy.deepcopy(capability)
                entry.setdefault("capability_id", capability_id)
                out[capability_id] = entry
        return out
    if isinstance(capabilities, list):
        for capability in capabilities:
            if not isinstance(capability, dict):
                continue
            capability_id = capability.get("capability_id")
            if capability_id in (None, ""):
                continue
            out[capability_id] = capability
    return out


def _goal_key(goal):
    return "|".join(str(part) for part in goal)


def _plan_goal_keys(selected):
    """Deduplicated set of goal keys contributed by the selected capabilities."""
    keys = set()
    for capability in selected:
        for goal in capability.get("goals") or []:
            keys.add(_goal_key(goal))
    return keys


def _validate_execute_ids(capability_ids, capabilities, errors):
    """Validate a nonempty, known, duplicate-free ``capability_ids`` list."""
    if not isinstance(capability_ids, list) or not capability_ids:
        errors.append("execute requires a nonempty capability_ids list")
        return False, []
    seen = set()
    selected = []
    for capability_id in capability_ids:
        if capability_id in seen:
            errors.append("duplicate capability id: %r" % (capability_id,))
            return False, []
        seen.add(capability_id)
        if capability_id not in capabilities:
            errors.append("unknown capability id: %r" % (capability_id,))
            return False, []
        selected.append(capabilities[capability_id])
    return True, selected


def score_plan(plan, case, capabilities, submission_count=1):
    """Score one captured plan against the literal fixture case (pure).

    The oracle is the fixture's own ``goal_options`` / ``allowed_decisions`` /
    ``allowed_objects`` -- nothing is derived from the plan itself.  Returns
    ``decision_correct``, ``goal_set_correct``, ``objects_correct``,
    ``single_submission`` and their conjunction ``planning_correct``.

    ``submission_count`` is supplied by the benchmark's capture count; a value
    of exactly 1 (with a supplied plan) is the only correct single submission.
    This scorer never reports task/physical success and never claims any
    unrestricted vision ability.
    """
    capabilities = _normalize_capabilities(capabilities)
    case = case if isinstance(case, dict) else {}

    allowed_decisions = list(case.get("allowed_decisions") or [])
    goal_options = case.get("goal_options") or []
    allowed_objects = set(case.get("allowed_objects") or [])

    errors = []
    supplied = isinstance(plan, dict) and bool(plan)
    try:
        count = int(submission_count)
    except (TypeError, ValueError):
        count = 0
    single_submission = bool(supplied and count == 1)

    decision_correct = False
    goal_set_correct = False
    objects_correct = False

    if supplied:
        decision = plan.get("decision")
        capability_ids = plan.get("capability_ids")

        if decision == "execute":
            valid, selected = _validate_execute_ids(capability_ids, capabilities, errors)
            decision_correct = bool(valid and "execute" in allowed_decisions)
            if valid:
                plan_keys = _plan_goal_keys(selected)
                option_keys = [
                    set(_goal_key(goal) for goal in (option or []))
                    for option in goal_options
                ]
                goal_set_correct = bool(option_keys) and any(
                    keys == plan_keys for keys in option_keys
                )
                object_ids = {
                    capability.get("object_id")
                    for capability in selected
                    if capability.get("object_id")
                }
                objects_correct = object_ids.issubset(allowed_objects)
        elif decision in ("clarify", "unsupported"):
            if not isinstance(capability_ids, list):
                errors.append("capability_ids must be a list")
            else:
                empty = len(capability_ids) == 0
                decision_correct = bool(empty and decision in allowed_decisions)
                goal_set_correct = bool(empty)
                objects_correct = bool(empty)
        else:
            errors.append("unknown decision: %r" % (decision,))
    else:
        errors.append("missing or invalid plan")

    planning_correct = bool(
        single_submission and decision_correct and goal_set_correct and objects_correct
    )
    return {
        "decision_correct": decision_correct,
        "goal_set_correct": goal_set_correct,
        "objects_correct": objects_correct,
        "single_submission": single_submission,
        "planning_correct": planning_correct,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Capture-only fake scene service
# --------------------------------------------------------------------------- #
def _body_for_log(body):
    """Return a JSON-safe copy of a request body for the transport log.

    The fake service handles only nonsecret, locally-generated payloads; the
    body is preserved verbatim so rejected attempts (wrong request id, duplicate
    submissions, unknown routes) stay auditable.  The ``_BAD_BODY`` sentinel is
    represented as an explicit marker and nothing is ever reinterpreted or
    forwarded anywhere.
    """
    if body is _BAD_BODY:
        return {"_unparseable": True}
    if body is None:
        return None
    if isinstance(body, dict):
        return copy.deepcopy(body)
    return {"_repr": repr(body)}


class FakeSceneContext(object):
    """Thread-safe, capture-only fake scene state.

    It answers the exact routes the real ``mcp_server.py`` calls, validates the
    *current* case context identifiers, records valid plan submissions and
    logs every transport attempt / protocol error.  It NEVER executes a plan,
    NEVER calls any service client or upstream HTTP and NEVER reads fixture
    gold: the independent-oracle fields never reach this object.

    ``transport_path`` is optional; when omitted, entries are kept in memory
    only (used by the HTTP unit tests so they never touch disk).
    """

    def __init__(self, public_session, transport_path=None):
        self._lock = threading.RLock()
        self._session = copy.deepcopy(public_session or {})
        self._session_id = self._session.get("session_id")
        try:
            self._scene_version = int(self._session.get("scene_version") or 0)
        except (TypeError, ValueError):
            self._scene_version = 0
        self._case_id = None
        self._request_id = None
        self._plans = {}        # request_id -> latest queued plan (public)
        self._captures = {}     # request_id -> [valid captured payloads]
        self._transport_entries = []
        self._protocol_errors = []
        # Must stay empty forever: the handler never forwards anything.  The
        # tests assert this explicitly.
        self.forwarded = []
        self._transport_path = transport_path
        self._closed = False

    # ---- properties ------------------------------------------------------- #
    @property
    def session_id(self):
        return self._session_id

    @property
    def scene_version(self):
        return self._scene_version

    @property
    def case_id(self):
        return self._case_id

    @property
    def request_id(self):
        return self._request_id

    @property
    def transport_entries(self):
        with self._lock:
            return list(self._transport_entries)

    @property
    def protocol_errors(self):
        with self._lock:
            return list(self._protocol_errors)

    def protocol_errors_for(self, case_id):
        """Protocol errors belonging to ``case_id``'s current context.

        Errors are attributed to the *current* case context (from
        ``begin_case``) even when the attempted body carried a wrong request id,
        so a stray/wrong-id attempt is still charged to the case that made it.
        """
        with self._lock:
            return [
                dict(error)
                for error in self._protocol_errors
                if error.get("case_id") == case_id
            ]

    # ---- case context ----------------------------------------------------- #
    def begin_case(self, case_id, request_id=None):
        """Set the per-case current context; returns the fresh request_id.

        The request_id is a random uuid unless one is supplied.  ``case_id`` is
        kept only for local bookkeeping -- it is never returned to the agent and
        never enters public data or the prompt.
        """
        with self._lock:
            request_id = request_id or uuid.uuid4().hex
            self._case_id = case_id
            self._request_id = request_id
            return request_id

    def set_expected(self, session_id, scene_version):
        with self._lock:
            self._session_id = session_id
            self._scene_version = int(scene_version)

    # ---- public views (immutable copies) ---------------------------------- #
    def public_session(self):
        with self._lock:
            return copy.deepcopy(self._session)

    def public_images(self):
        with self._lock:
            return copy.deepcopy(self._session.get("images") or [])

    def captures_for(self, request_id):
        with self._lock:
            return copy.deepcopy(list(self._captures.get(request_id) or []))

    def capture_count(self, request_id):
        with self._lock:
            return len(self._captures.get(request_id) or [])

    def plan_for(self, request_id):
        with self._lock:
            plan = self._plans.get(request_id)
            return copy.deepcopy(plan) if plan is not None else None

    # ---- transport log ---------------------------------------------------- #
    def _log(self, entry):
        self._transport_entries.append(entry)
        if self._transport_path:
            with open(self._transport_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def close(self):
        with self._lock:
            self._closed = True

    # ---- routing core ----------------------------------------------------- #
    def request(self, method, path, body):
        """Route one request; returns ``(status, payload)`` and logs it."""
        with self._lock:
            try:
                status, payload, note = self._route(method, path, body)
            except Exception as exc:  # noqa: BLE001 - never leak a 500 trace
                status = 500
                payload = {"ok": False, "reason": "internal_error", "detail": str(exc)}
                note = "internal_error: %s" % exc
            entry = {
                "ts": _utcnow_iso(),
                "event": "http",
                "method": method,
                "path": path,
                "status": status,
                "case_id": self._case_id,
                "request_id": self._request_id,
                "body": _body_for_log(body),
                "protocol_error": note,
            }
            self._log(entry)
            if note:
                self._protocol_errors.append(
                    {
                        "ts": entry["ts"],
                        "case_id": self._case_id,
                        "request_id": self._request_id,
                        "method": method,
                        "path": path,
                        "note": note,
                    }
                )
            return status, payload

    def _route(self, method, path, body):
        parts = [part for part in path.split("/") if part]
        if method == "GET" and len(parts) == 2 and parts[0] == "sessions":
            return self._get_session(parts[1])
        if method == "GET" and len(parts) == 2 and parts[0] == "plans":
            return self._get_plan(parts[1])
        if method == "POST" and parts == ["observe"]:
            return self._observe(body)
        if method == "POST" and parts == ["plans"]:
            return self._submit_plan(body)
        if method == "POST" and len(parts) == 3 and parts[0] == "plans" and parts[2] == "resume":
            return (
                405,
                {"ok": False, "reason": "unsupported_route",
                 "detail": "resume_scene_plan is not supported"},
                "unsupported_route: POST resume",
            )
        return (
            404,
            {"ok": False, "reason": "unsupported_route",
             "detail": "%s %s" % (method, path)},
            "unsupported_route: %s %s" % (method, path),
        )

    def _get_session(self, session_id):
        if session_id != self._session_id:
            return 404, {"ok": False, "reason": "unknown_session"}, (
                "wrong_session_id: %r" % (session_id,))
        return 200, self.public_session(), None

    def _get_plan(self, request_id):
        plan = self._plans.get(request_id)
        if plan is None:
            # A missing plan is an honest 404, not a protocol error.
            return 404, {"ok": False, "reason": "plan_not_found"}, None
        return 200, copy.deepcopy(plan), None

    def _observe(self, body):
        if body is _BAD_BODY or not isinstance(body, dict):
            return 400, {"ok": False, "reason": "bad_body"}, "bad_body: observe"
        if body.get("session_id") != self._session_id:
            return 409, {"ok": False, "reason": "wrong_session"}, (
                "wrong_session_id: %r" % (body.get("session_id"),))
        payload = {
            "ok": True,
            "session_id": self._session_id,
            "scene_version": self._scene_version,
            "images": self.public_images(),
            "latest_png": self._session.get("latest_png"),
            "extra_views": False,
            "note": "no extra views available",
        }
        return 200, payload, None

    def _submit_plan(self, body):
        if body is _BAD_BODY or not isinstance(body, dict):
            return 400, {"ok": False, "reason": "bad_body"}, "bad_body: plans"

        session_id = body.get("session_id")
        request_id = body.get("request_id")
        scene_version = body.get("scene_version")
        decision = body.get("decision")
        capability_ids = body.get("capability_ids")

        if session_id != self._session_id:
            return 409, {"ok": False, "reason": "wrong_session"}, (
                "wrong_session_id: %r" % (session_id,))
        if not isinstance(scene_version, int) or scene_version != self._scene_version:
            return 409, {"ok": False, "reason": "wrong_scene_version"}, (
                "wrong_scene_version: %r" % (scene_version,))
        if not self._request_id or request_id != self._request_id:
            return 409, {"ok": False, "reason": "wrong_request_id"}, (
                "wrong_request_id: %r" % (request_id,))
        if not isinstance(decision, str) or not decision:
            return 400, {"ok": False, "reason": "bad_decision"}, (
                "bad_decision: %r" % (decision,))
        if not isinstance(capability_ids, list) or not all(
            isinstance(item, str) for item in capability_ids
        ):
            return 400, {"ok": False, "reason": "bad_capability_ids"}, "bad_capability_ids"

        rationale = body.get("rationale")
        stored = {
            "request_id": request_id,
            "session_id": session_id,
            "scene_version": self._scene_version,
            "decision": decision,
            "capability_ids": list(capability_ids),
            "rationale": rationale if isinstance(rationale, str) else "",
        }
        # Capture EVERY valid submission for this attempt (duplicates retained).
        self._captures.setdefault(request_id, []).append(copy.deepcopy(stored))
        queued = dict(stored)
        queued.update({
            "ok": True,
            "state": "queued",
            "job_ids": [],
            "plan_success": None,
            "repair_history": [],
            "executed": False,
        })
        self._plans[request_id] = copy.deepcopy(queued)
        return 200, queued, None


class _CaptureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "planning_diagnostics_fake/1.0"

    def log_message(self, fmt, *args):  # silence default stderr chatter
        return

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return _BAD_BODY

    def _dispatch(self, method):
        context = getattr(self.server, "context", None)
        path = self.path.split("?", 1)[0]
        body = self._read_body() if method == "POST" else None
        if context is None:
            status, payload = 503, {"ok": False, "reason": "no_context"}
        else:
            status, payload = context.request(method, path, body)
        self._send(status, payload)

    def do_GET(self):  # noqa: N802 (http.server API)
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802 (http.server API)
        self._dispatch("POST")

    def _send(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _CaptureServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_capture_server(context, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Build the capture server bound to ``host:port`` (default 127.0.0.1:8778).

    Tests may pass ``port=0`` to bind an ephemeral loopback port.  The handler
    never forwards and never executes a plan.
    """
    server = _CaptureServer((host, port), _CaptureHandler)
    server.context = context
    return server


def serve_capture_server(context, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Start the server in a background thread; returns ``(server, thread)``."""
    server = make_capture_server(context, host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


# --------------------------------------------------------------------------- #
# Isolated Hermes home preparation (never touches the source home)
# --------------------------------------------------------------------------- #
def patch_config(config_path, mcp_server_path=None, scene_dir=None):
    """Patch ONLY the copied config.yaml's scene_tools MCP wiring.

    The ``yaml`` dependency is imported lazily so module import stays safe.
    Command stays ``/usr/bin/python3``; args become ``['-u', <managed
    mcp_server.py>]``; cwd becomes the managed scene_demo dir; the prior env is
    preserved with ``SCENE_SERVICE_URL`` set explicitly to the 8778 endpoint.
    Model/provider/supports_vision/role/other settings are preserved.  Asserts
    the endpoint is exactly 8778 and that no ``8767`` survives.
    """
    import yaml  # lazy: never needed for import/--help

    mcp_server_path = mcp_server_path or os.path.join(MANAGED_SCENE_DIR, "mcp_server.py")
    scene_dir = scene_dir or MANAGED_SCENE_DIR

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = {}
        config["mcp_servers"] = servers
    scene_tools = servers.get("scene_tools")
    if not isinstance(scene_tools, dict):
        scene_tools = {}
        servers["scene_tools"] = scene_tools

    scene_tools["command"] = "/usr/bin/python3"
    scene_tools["args"] = ["-u", mcp_server_path]
    scene_tools["cwd"] = scene_dir
    scene_tools["enabled"] = True

    env = scene_tools.get("env")
    if not isinstance(env, dict):
        env = {}
    env = dict(env)  # preserve prior env
    env["SCENE_SERVICE_URL"] = ENDPOINT
    scene_tools["env"] = env

    text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    # Concrete pre-call assertions.
    with open(config_path, "rb") as handle:
        raw = handle.read()
    assert b"8767" not in raw, "8767 must not survive in the isolated config"
    assert ENDPOINT in text, "endpoint must be the explicit 8778 endpoint"
    assert scene_tools["env"]["SCENE_SERVICE_URL"] == ENDPOINT
    assert scene_tools["args"] == ["-u", mcp_server_path]
    assert scene_tools["cwd"] == scene_dir
    assert scene_tools["command"] == "/usr/bin/python3"
    model = config.get("model") or {}
    assert model.get("default") == MODEL_NAME, "model must be qwen3-vl-plus"
    assert model.get("supports_vision") is True, "native vision must be enabled"
    return config


def config_summary(config):
    """Non-secret summary: model, provider, endpoint (never any credential)."""
    model = config.get("model") or {}
    return {
        "model": model.get("default"),
        "provider": model.get("provider"),
        "supports_vision": model.get("supports_vision"),
        "endpoint": ENDPOINT,
    }


def config_sha(config_path):
    with open(config_path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def prepare_home(hermes_home, source_home=SOURCE_HOME):
    """Create the NEW isolated home by copying config.yaml/SOUL.md/.env.

    The ``.env`` contents are NEVER inspected or printed; it is copied and then
    chmod'ed 0o600.  Only the copied config.yaml is modified.  The new home must
    not already exist, so nothing is ever overwritten in place.
    """
    if os.path.exists(hermes_home):
        raise FileExistsError("isolated home already exists: %s" % hermes_home)
    os.makedirs(hermes_home, mode=0o700)

    for name in ("config.yaml", "SOUL.md", ".env"):
        src = os.path.join(source_home, name)
        dst = os.path.join(hermes_home, name)
        if not os.path.isfile(src):
            raise FileNotFoundError("missing source profile file: %s" % src)
        shutil.copy2(src, dst)

    env_path = os.path.join(hermes_home, ".env")
    os.chmod(env_path, 0o600)

    config_path = os.path.join(hermes_home, "config.yaml")
    config = patch_config(config_path)
    return config_path, config


# --------------------------------------------------------------------------- #
# Benchmark orchestration
# --------------------------------------------------------------------------- #
def _load_fixtures(path=None):
    path = path or os.path.join(MANAGED_SCENE_DIR, "fixtures.json")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    by_id = {}
    for case in data.get("cases") or []:
        if isinstance(case, dict) and case.get("case_id"):
            by_id[case["case_id"]] = case
    return by_id


def _scoring_capabilities(public_session):
    """Public catalog capabilities for scoring (never independent gold)."""
    scene_id = public_session.get("scene_id")
    try:
        import catalog  # lazy; stdlib-only, no side effects

        return catalog.scene_capabilities(scene_id)
    except Exception:  # noqa: BLE001 - fall back to the public session list
        return public_session.get("capabilities") or []


def _report_ok(report):
    cases = report.get("cases") or []
    if len(cases) != len(CASE_IDS):
        return False
    for case in cases:
        if case.get("exit_code") != 0:
            return False
        if case.get("submission_count") != 1:
            return False
        if case.get("errors"):
            return False
    return report.get("oracle_not_in_prompt") is True


def _print_row(case_entry):
    score = case_entry.get("score") or {}
    print(
        "case=%s exit=%s submissions=%d planning=%s wall=%.1fs"
        % (
            case_entry.get("case_id"),
            case_entry.get("exit_code"),
            case_entry.get("submission_count"),
            score.get("planning_correct"),
            float(case_entry.get("wall_s") or 0.0),
        ),
        flush=True,
    )


def run_benchmark(input_package, output_dir, hermes_home):
    """Run the five fixed capture-only cases; returns a process exit code."""
    import run_agent  # lazy: import/--help must not pull this in

    if os.path.exists(output_dir):
        sys.stderr.write("output-dir already exists: %s\n" % output_dir)
        return 2
    if os.path.exists(hermes_home):
        sys.stderr.write("hermes-home already exists: %s\n" % hermes_home)
        return 2

    input_session_path = os.path.join(input_package, "session.json")
    with open(input_session_path, "r", encoding="utf-8") as handle:
        input_session = json.load(handle)

    first_png = os.path.join(input_package, *CAP_IMAGE_RELPATH)
    if not os.path.isfile(first_png):
        sys.stderr.write("missing input image: %s\n" % first_png)
        return 2

    fixtures = _load_fixtures()
    cases = [fixtures[case_id] for case_id in CASE_IDS]

    os.makedirs(output_dir, mode=0o700)

    fake_session_id = uuid.uuid4().hex
    public_session = build_public_session(input_session, fake_session_id, first_png)

    report = {
        "cases": [],
        "ok": False,
        "robot_actions": 0,
        "model": MODEL_NAME,
        "endpoint": ENDPOINT,
        "oracle_not_in_prompt": True,
        "config_sha": None,
        "config": None,
        "output_dir": output_dir,
        "hermes_home": hermes_home,
    }
    report_path = os.path.join(output_dir, "report.json")

    transport_path = os.path.join(output_dir, "transport.jsonl")
    context = None
    server = None
    thread = None
    try:
        config_path, config = prepare_home(hermes_home)
        report["config_sha"] = config_sha(config_path)
        report["config"] = config_summary(config)

        context = FakeSceneContext(public_session, transport_path=transport_path)
        server, thread = serve_capture_server(context, host=DEFAULT_HOST, port=DEFAULT_PORT)

        runner = run_agent.HermesRunner(
            home=hermes_home, cwd=MANAGED_SCENE_DIR, tools=HERMES_TOOLS
        )

        scoring_capabilities = _scoring_capabilities(public_session)

        for case in cases:
            case_id = case["case_id"]
            case_dir = os.path.join(output_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)

            request_id = context.begin_case(case_id)
            image_paths = [first_png]
            prompt = run_agent.build_initial_prompt(
                public_session, case["request"], request_id, image_paths
            )
            _write_text(os.path.join(case_dir, "prompt.txt"), prompt)

            errors = []
            oracle_ok = _oracle_excluded(case, public_session, prompt)
            if not oracle_ok:
                # Fail-closed: a leaking prompt is an operational error and the
                # model is NEVER invoked for this case.  The prompt file written
                # above is still preserved as evidence of what was withheld.
                errors.append("oracle_leak")
                report["oracle_not_in_prompt"] = False

            usage_path = os.path.join(case_dir, "usage.json")
            started = time.monotonic()
            if not oracle_ok:
                result = {"exit_code": None, "output": b"", "timed_out": False}
            else:
                try:
                    result = runner(prompt, first_png, usage_path, HERMES_TIMEOUT) or {}
                except Exception as exc:  # noqa: BLE001 - record, never crash the run
                    result = {
                        "exit_code": None,
                        "output": str(exc).encode("utf-8"),
                        "timed_out": False,
                        "error": "hermes_call_failed: %s" % exc,
                    }
            wall_s = round(time.monotonic() - started, 3)

            output = result.get("output") or b""
            if isinstance(output, str):
                output = output.encode("utf-8")
            _write_bytes(os.path.join(case_dir, "hermes.log"), output)

            captures = context.captures_for(request_id)
            submission_count = len(captures)
            plan = captures[0] if captures else None

            score = score_plan(
                plan, case, scoring_capabilities, submission_count=submission_count
            )

            exit_code = result.get("exit_code")
            if result.get("timed_out"):
                errors.append("hermes_timeout")
            if result.get("error"):
                errors.append(str(result["error"]))
            if exit_code != 0:
                errors.append("exit_code=%r" % (exit_code,))
            if submission_count == 0:
                errors.append("no_plan_submitted")
            elif submission_count > 1:
                errors.append("duplicate_submission=%d" % submission_count)
            # Per-case protocol errors (wrong-id / duplicate attempts) are
            # operational errors for THIS case, even when the attempted body
            # carried a wrong request id.
            case_protocol_errors = context.protocol_errors_for(case_id)
            if case_protocol_errors:
                errors.append("protocol_error=%d" % len(case_protocol_errors))
            if errors:
                # Operational failure can never be planning-correct, but the
                # captured plan and the component scores are retained as-is.
                score = dict(score)
                score["planning_correct"] = False

            _write_json_atomic(os.path.join(case_dir, "submitted_plan.json"), plan)
            _write_json_atomic(os.path.join(case_dir, "score.json"), score)

            entry = {
                "case_id": case_id,
                "request": case["request"],
                "request_id": request_id,
                "exit_code": exit_code,
                "submission_count": submission_count,
                "plan": plan,
                "score": score,
                "protocol_errors": case_protocol_errors,
                "wall_s": wall_s,
                "errors": errors,
            }
            report["cases"].append(entry)
            report["ok"] = _report_ok(report)
            _write_json_atomic(report_path, report)
            _print_row(entry)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=10)
        if context is not None:
            context.close()
        report["ok"] = _report_ok(report)
        _write_json_atomic(report_path, report)

    return 0 if report.get("ok") else 1


# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="planning_diagnostics.py",
        description=(
            "Capture-only Hermes planning benchmark. Records the plans a real "
            "Hermes submits against a fixed 127.0.0.1:8778 fake scene service; "
            "it executes no robot and never contacts the production service."
        ),
    )
    parser.add_argument("--input-package", required=True,
                        help="input package dir containing session.json and the cap image")
    parser.add_argument("--output-dir", required=True,
                        help="private output dir (must not already exist)")
    parser.add_argument("--hermes-home", required=True,
                        help="new isolated Hermes home to create (must not already exist)")
    args = parser.parse_args(argv)

    return run_benchmark(args.input_package, args.output_dir, args.hermes_home)


if __name__ == "__main__":
    raise SystemExit(main())
