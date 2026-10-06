#!/usr/bin/env python3
"""Hermes MCP bridge for the resident SmolVLA-on-LIBERO HTTP service.

This is a thin client in the same style as ``vla_mcp_server.py``: it owns no
model, no simulation and no environment.  Every tool forwards to
``service.py``'s HTTP JSON interface (default ``http://127.0.0.1:8766``) and
returns the service payload largely unchanged, so Hermes and a plain HTTP
runner see identical execution semantics.

The tasks are the *standard* LIBERO benchmark tasks.  ``service.py`` builds the
task catalogue from ``libero.libero.benchmark.get_benchmark_dict()`` and, for a
requested ``(suite, task_id)``, automatically instantiates the matching scene
(hard reset, ``num_steps_wait`` settle steps) before running the policy.  A job's
``success`` is the environment's own ``info['is_success']`` judgement -- it is a
benchmark result, never an LLM verdict.  Clients must therefore read the task
list from ``list_libero_tasks`` and never invent instructions; a suite/task the
service does not expose cannot be executed, and this bridge must not fake a
result with another tool.

FastMCP note: on mcp 2.x the high-level server class is
``mcp.server.mcpserver.MCPServer`` (``FastMCP`` was renamed).

Run:  /usr/bin/python3 mcp_server.py     (stdio)
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import anyio  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

SERVER_NAME = "libero_tools"
SERVER_VERSION = "0.1.0"
SERVICE_URL = os.environ.get("LIBERO_SERVICE_URL", "http://127.0.0.1:8766").rstrip("/")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

INSTRUCTIONS = (
    "Tools for evaluating a SmolVLA checkpoint on the standard LIBERO benchmark "
    "(Franka Panda, 4 suites x 10 tasks). Typical loop: list_libero_tasks -> "
    "execute_libero_task(suite, task_id) -> wait_for_libero_execution(job_id) -> "
    "observe_libero_scene. The service creates the requested task's scene itself and "
    "resets it before every episode; you cannot pass a free-form instruction -- the "
    "instruction and the step budget are taken from the service's own task catalogue. "
    "A job's success flag is the LIBERO environment's own \"is_success\" judgement, i.e. "
    "a benchmark result, and it is only available once the job has completed. If a "
    "suite/task is not listed by list_libero_tasks it cannot be executed: report that "
    "plainly and do NOT fabricate a result with another tool. Only one execution may be "
    "active at a time; a second execute returns an error until the first finishes."
)


mcp = MCPServer(SERVER_NAME, version=SERVER_VERSION, instructions=INSTRUCTIONS)


def _http(method: str, path: str, body: dict | None = None, timeout: float = 10.0) -> dict:
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
            return json.loads(exc.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "reason": "service_unreachable", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "service_unreachable", "detail": str(exc)}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "bad_response", "detail": str(exc)}
    return payload if isinstance(payload, dict) else {"ok": False, "reason": "bad_response",
                                                     "detail": "non-object response"}


async def _call(method: str, path: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    """Run one blocking HTTP call off the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(_http, method, path, body, timeout))


@mcp.tool(description="List the standard LIBERO benchmark tasks the service can run: "
                      "for each suite (libero_spatial, libero_object, libero_goal, libero_10) "
                      "and task_id it returns the task name, the exact instruction the policy "
                      "will be given, and the step budget. Read instructions from here -- do "
                      "not invent them. Pass a (suite, task_id) pair to execute_libero_task.")
async def list_libero_tasks() -> dict[str, Any]:
    return await _call("GET", "/tasks", None, 10.0)


@mcp.tool(description="Queue one LIBERO episode and return immediately with a job_id "
                      "(state 'queued'). The service creates that task's scene, hard-resets it "
                      "with the given seed and init_state_index, and runs the SmolVLA policy "
                      "until success or the suite's step budget. Only one job may be active; a "
                      "second call returns an error while the first runs. Poll "
                      "get_libero_status / wait_for_libero_execution for the outcome.")
async def execute_libero_task(
    suite: str,
    task_id: int,
    seed: int = 0,
    init_state_index: int = 0,
) -> dict[str, Any]:
    body = {
        "suite": str(suite),
        "task_id": int(task_id),
        "seed": int(seed),
        "init_state_index": int(init_state_index),
    }
    return await _call("POST", "/execute", body, 10.0)


@mcp.tool(description="Read one job's state machine: state is queued|running|completed|error, "
                      "plus steps, success (null until the job finishes; when set it is the "
                      "LIBERO environment's is_success judgement), suite, task_id, the exact "
                      "instruction used, wall_s, seed, init_state_index and any real exception. "
                      "Also returns the run directory and the paths of result.json/rollout.mp4.")
async def get_libero_status(job_id: str) -> dict[str, Any]:
    path = "/status?job_id=" + urllib.parse.quote(str(job_id))
    return await _call("GET", path, None, 10.0)


@mcp.tool(description="Poll a job every 1 s until it reaches 'completed' or 'error' or "
                      "timeout_s elapses (clamped to 1..60 s). Returns the last status payload "
                      "plus waited_s and timed_out; timed_out=true means the execution is still "
                      "running, so call again. Includes the benchmark success flag once done.")
async def wait_for_libero_execution(job_id: str, timeout_s: float = 45.0) -> dict[str, Any]:
    timeout = max(1.0, min(60.0, float(timeout_s)))
    started = time.monotonic()
    status: dict[str, Any] = {}
    timed_out = False
    while True:
        path = "/status?job_id=" + urllib.parse.quote(str(job_id))
        status = await _call("GET", path, None, 10.0)
        state = status.get("state")
        if state in ("completed", "error"):
            break
        if not status.get("ok", False):
            # service unreachable / unknown job_id: nothing more to wait for
            break
        if time.monotonic() - started >= timeout:
            timed_out = True
            break
        await anyio.sleep(1.0)
    out = dict(status)
    out["waited_s"] = round(time.monotonic() - started, 2)
    out["timed_out"] = timed_out
    return out


@mcp.tool(description="Return the most recently rendered RGB frame of the active/last job "
                      "(absolute path to the saved PNG) together with that job's state, steps "
                      "and success flag. Use a vision tool on the returned image_path if you "
                      "need to look at the scene; success itself is still the environment's "
                      "judgement, not your visual verdict.")
async def observe_libero_scene() -> dict[str, Any]:
    return await _call("GET", "/observe", None, 10.0)


def main() -> None:
    sys.stderr.write(
        "[%s mcp] stdio server %s v%s pid=%d service=%s\n"
        % (SERVER_NAME, SERVER_NAME, SERVER_VERSION, os.getpid(), SERVICE_URL)
    )
    sys.stderr.flush()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
