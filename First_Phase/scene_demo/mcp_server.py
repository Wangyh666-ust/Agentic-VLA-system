#!/usr/bin/env python3
"""Hermes MCP bridge (stdio) for the persistent-scene SmolVLA service.

Thin client in the same ``anyio`` / HTTP style as the older ``libero_demo``
bridge: it owns no model, no simulation and no evaluation.  Every tool forwards
to ``scene_demo/service.py``'s HTTP JSON interface (default
``http://127.0.0.1:8767``) and returns the service payload largely unchanged, so
Hermes and a plain HTTP runner observe identical execution semantics.

Exactly SIX tools are registered:

  get_scene_session       read one persistent session (public state only)
  list_scene_capabilities read the scene's public storage policy + capabilities
  observe_scene           current images + public data (extra views optional)
  submit_scene_plan       submit a plan and return (host owns waiting)
  get_scene_plan          read one plan by its exact request_id
  resume_scene_plan       resume a blocked plan (repair turn only)

Scene creation, independent evaluation, capability audit and generic arbitrary
HTTP are deliberately NOT exposed: those belong to the host / the fixed service,
never to the planner.  The planner picks capabilities, submits a plan, says it
submitted the plan and finishes -- it must not poll the robot or sleep in
45-second rounds.

Run:  /usr/bin/python3 -u mcp_server.py     (stdio)
"""

from __future__ import annotations

import functools
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import anyio  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

SERVER_NAME = "scene_tools"
SERVER_VERSION = "0.1.0"
SERVICE_URL = os.environ.get("SCENE_SERVICE_URL", "http://127.0.0.1:8767").rstrip("/")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

INSTRUCTIONS = (
    "Tools for planning inside ONE persistent robot scene (SmolVLA-on-LIBERO). "
    "The scene already exists: it is created and reset by the host, and its "
    "objects are never touched by these tools. Typical flow: get_scene_session("
    "session_id) to read the current scene_version / storage_policy / "
    "capabilities / images, optionally observe_scene(session_id, extra_views) "
    "and look at the returned image paths with the native vision tool, then "
    "submit_scene_plan(session_id, scene_version, request_id, capability_ids, "
    "rationale, decision) ONCE and finish. Choose capability ids yourself from "
    "the public capabilities -- nothing here routes on keywords. If the request "
    "is ambiguous use decision='clarify' with an empty capability_ids list; if "
    "the scene cannot support it use decision='unsupported' with an empty list. "
    "After submit_scene_plan returns, state plainly that the plan was submitted "
    "and stop: the host owns waiting, error checking and cancellation. Do NOT "
    "poll get_scene_plan in a loop, do NOT wait in 45-second rounds, and do NOT "
    "call the robot yourself."
)


mcp = MCPServer(SERVER_NAME, version=SERVER_VERSION, instructions=INSTRUCTIONS)


def _http(method: str, path: str, body: dict | None = None, timeout: float = 15.0) -> dict:
    """One JSON round trip to the execution service; never raises."""
    url = SERVICE_URL + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("ok", False)
                payload.setdefault("http_status", exc.code)
                return payload
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "reason": "service_http_error", "http_status": exc.code,
                "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "service_unreachable", "detail": str(exc)}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "bad_response", "detail": str(exc)}
    if isinstance(payload, dict):
        return payload
    return {"ok": False, "reason": "bad_response", "detail": "non-object response"}


async def _call(method: str, path: str, body: dict | None = None, timeout: float = 15.0) -> dict:
    """Run one blocking HTTP call off the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(_http, method, path, body, timeout))


def _quote(value: Any) -> str:
    return urllib.parse.quote(str(value), safe="")


def _atomic_capabilities(capabilities: Any) -> list[dict]:
    """Keep only atomic capabilities.

    The service's public capability list already excludes audit-only composites;
    this filter is a defensive second guard so a composite entry (audit flag set,
    or no single ``object_id``) can never reach the planner as an executable
    choice.  Caller data is never mutated.
    """
    out: list[dict] = []
    if not isinstance(capabilities, list):
        return out
    for capability in capabilities:
        if not isinstance(capability, dict):
            continue
        if capability.get("audit_only") is True:
            continue
        if capability.get("object_id") in (None, ""):
            continue
        out.append(capability)
    return out


@mcp.tool(description="Read one persistent scene session by id: state "
                      "(ready|running|closed|error), scene_version, "
                      "env_instance_id, episode_resets, total_steps, "
                      "storage_policy, capabilities, images and latest_png. Call "
                      "this FIRST to confirm the scene_version you are planning "
                      "against. Only public data is returned; there are no hidden "
                      "coordinates or answer fixtures.")
async def get_scene_session(session_id: str) -> dict[str, Any]:
    return await _call("GET", "/sessions/" + _quote(session_id), None, 10.0)


@mcp.tool(description="List the atomic capabilities usable in this scene together "
                      "with the public storage_policy and the current scene_version. "
                      "Use the returned capability_id values when submitting a plan. "
                      "Every entry also carries an 'evidence' tag: a 'candidate' tag "
                      "is NOT a reliability guarantee, so do not claim a capability "
                      "is verified.")
async def list_scene_capabilities(session_id: str) -> dict[str, Any]:
    session = await _call("GET", "/sessions/" + _quote(session_id), None, 10.0)
    if not isinstance(session, dict) or session.get("ok") is False:
        return session
    return {
        "ok": True,
        "session_id": session.get("session_id", str(session_id)),
        "scene_version": session.get("scene_version"),
        "storage_policy": session.get("storage_policy"),
        "capabilities": _atomic_capabilities(session.get("capabilities")),
    }


@mcp.tool(description="Observe the CURRENT scene: returns the present images "
                      "(agentview and, with extra_views=true, additional views) plus "
                      "public data. The returned image_path values are real on-disk "
                      "PNG paths that you can pass to the native vision tool as "
                      "vision_analyze(image_url=..., question=...) to actually look "
                      "at the scene. Re-observe only if you need fresher pixels or "
                      "extra views.")
async def observe_scene(session_id: str, extra_views: bool = False) -> dict[str, Any]:
    body = {"session_id": str(session_id), "extra_views": bool(extra_views)}
    return await _call("POST", "/observe", body, 20.0)


@mcp.tool(description="Submit your plan for the current scene and return immediately "
                      "with the queued plan (request_id, state, decision). Choose "
                      "capability_ids yourself from list_scene_capabilities; their "
                      "order is the execution order. Use decision='execute' with the "
                      "chosen ids for a normal plan, or decision='clarify' / "
                      "'unsupported' with an EMPTY capability_ids list. After this "
                      "call returns, say that the plan was submitted and FINISH -- do "
                      "not poll for progress, do not wait for the robot, do not sleep "
                      "in rounds; the host owns waiting and cancellation.")
async def submit_scene_plan(
    session_id: str,
    scene_version: int,
    request_id: str,
    capability_ids: list[str],
    rationale: str,
    decision: str = "execute",
) -> dict[str, Any]:
    body = {
        "session_id": str(session_id),
        "scene_version": int(scene_version),
        "request_id": str(request_id),
        "capability_ids": [str(capability_id) for capability_id in (capability_ids or [])],
        "rationale": str(rationale),
        "decision": str(decision or "execute"),
        "audit": False,
        "budget_per_subgoal": 300,
    }
    return await _call("POST", "/plans", body, 20.0)


@mcp.tool(description="Read one plan by its EXACT request_id: state "
                      "(queued|running|completed|blocked|error|cancelled), decision, "
                      "capability_ids, job_ids, completed/pending capability ids, "
                      "plan_success, regressions, error and repair_history. This is a "
                      "read-only lookup: the host owns waiting and polling, so do NOT "
                      "repeatedly query progress in a loop.")
async def get_scene_plan(request_id: str) -> dict[str, Any]:
    return await _call("GET", "/plans/" + _quote(request_id), None, 10.0)


@mcp.tool(description="Resume a BLOCKED plan for the same request_id with a corrected "
                      "capability list and order (repair turn only). Pass "
                      "capability_ids chosen from the public capabilities; empty lists "
                      "are not a repair. Use this only when the host asks you to repair "
                      "a blocked plan; if the plan truly cannot be repaired, do not call "
                      "this tool -- explain why and leave the plan blocked.")
async def resume_scene_plan(
    session_id: str,
    scene_version: int,
    request_id: str,
    capability_ids: list[str],
    rationale: str,
) -> dict[str, Any]:
    body = {
        "session_id": str(session_id),
        "scene_version": int(scene_version),
        "capability_ids": [str(capability_id) for capability_id in (capability_ids or [])],
        "rationale": str(rationale),
        "audit": False,
        "budget_per_subgoal": 300,
    }
    path = "/plans/" + _quote(request_id) + "/resume"
    return await _call("POST", path, body, 20.0)


def main() -> None:
    sys.stderr.write(
        "[%s mcp] stdio server %s v%s pid=%d service=%s\n"
        % (SERVER_NAME, SERVER_NAME, SERVER_VERSION, os.getpid(), SERVICE_URL)
    )
    sys.stderr.flush()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
