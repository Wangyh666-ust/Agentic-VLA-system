#!/usr/bin/env python3
"""Reproducible direct-VLA vs manual-subgoal persistent-scene audit runner.

This module drives the *existing* persistent-scene HTTP service (workflow
``persistent_scene_v2``) through a strictly serial sequence of capability
audits.  It is a capability/instrument audit, **not** an agent benchmark:

* every condition is a *preauthored* request -- either a single composite
  direct-VLA capability (``direct_vla``) or a fixed list of human-authored
  subgoals (``manual_subgoals``);
* the independent success signal comes only from the preauthored fixture
  oracle (``POST /evaluate``), never from the plan, the narration or any
  visual inference;
* there is no Hermes planner, no model-state reset and no per-task
  environment reset performed by this runner -- the service owns the single
  persistent scene and the runner never reaches into it.

Importing this module performs no networking and starts no experiment: the
urllib opener and all constants are inert.  Only :func:`main` talks to the
service, and only after a strict health gate.

The runner is deliberately standard-library only, and every HTTP call goes
through :func:`http_json`, which the mock contract tests patch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

# --- fixed service contract --------------------------------------------------

SERVICE = "http://127.0.0.1:8767"
WORKFLOW = "persistent_scene_v2"
MODEL_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
ORACLE_SOURCE = "preauthored_fixture"
RATIONALE = "preauthored manual/direct capability audit"

TERMINAL_PLAN_STATES = ("completed", "blocked", "error", "cancelled")
EVALUATED_PLAN_STATES = ("completed", "blocked")
POLL_INTERVAL_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 60.0
CANCEL_TIMEOUT_S = 5.0

NOTE = (
    "These trials are direct-VLA / manual-subgoal capability audits. Each "
    "condition is a preauthored request; no Hermes planner, no agent "
    "narration and no visual inference participates. Independent success is "
    "judged only by the preauthored fixture oracle, and every trial runs in "
    "its own fresh session with a fixed seed and init_state_index. Because "
    "both arms share the same preauthored final target case, these results "
    "cannot establish any agent gain."
)

# One condition: exact scene, capability ids, per-subgoal budget, preauthored
# final evaluation case, and execution kind (direct VLA vs manual subgoals).
CONDITIONS = {
    "table_direct": {
        "scene_id": "goal_table",
        "capability_ids": ["table_both"],
        "budget_per_subgoal": 600,
        "case_id": "table_tidy",
        "execution_kind": "direct_vla",
    },
    "table_forward": {
        "scene_id": "goal_table",
        "capability_ids": ["bowl_to_plate", "wine_to_rack"],
        "budget_per_subgoal": 300,
        "case_id": "table_tidy",
        "execution_kind": "manual_subgoals",
    },
    "table_reverse": {
        "scene_id": "goal_table",
        "capability_ids": ["wine_to_rack", "bowl_to_plate"],
        "budget_per_subgoal": 300,
        "case_id": "table_tidy",
        "execution_kind": "manual_subgoals",
    },
    "table_shifted": {
        "scene_id": "goal_table_shifted",
        "capability_ids": ["bowl_to_plate", "wine_to_rack"],
        "budget_per_subgoal": 300,
        "case_id": "table_shifted_tidy",
        "execution_kind": "manual_subgoals",
    },
    "basket_direct": {
        "scene_id": "basket_two",
        "capability_ids": ["basket_both"],
        "budget_per_subgoal": 600,
        "case_id": "basket_two_cans",
        "execution_kind": "direct_vla",
    },
    "basket_split": {
        "scene_id": "basket_two",
        "capability_ids": ["soup_to_basket", "sauce_to_basket"],
        "budget_per_subgoal": 300,
        "case_id": "basket_two_cans",
        "execution_kind": "manual_subgoals",
    },
    "mugs_direct": {
        "scene_id": "mugs_two",
        "capability_ids": ["mugs_both"],
        "budget_per_subgoal": 600,
        "case_id": "mugs_standard",
        "execution_kind": "direct_vla",
    },
    "mugs_split": {
        "scene_id": "mugs_two",
        "capability_ids": ["white_mug_left", "yellow_mug_right"],
        "budget_per_subgoal": 300,
        "case_id": "mugs_standard",
        "execution_kind": "manual_subgoals",
    },
    "free_right": {
        "scene_id": "mugs_left_occupied",
        "capability_ids": ["white_mug_right"],
        "budget_per_subgoal": 300,
        "case_id": "mugs_free_right",
        "execution_kind": "manual_subgoals",
    },
    "free_left": {
        "scene_id": "mugs_right_occupied",
        "capability_ids": ["white_mug_left"],
        "budget_per_subgoal": 300,
        "case_id": "mugs_free_left",
        "execution_kind": "manual_subgoals",
    },
}

# A proxy-free opener.  Constructing it touches no network.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HttpError(RuntimeError):
    """Raised when the service returns a non-2xx response."""


# --- HTTP --------------------------------------------------------------------


def http_json(method, path, body=None, timeout=DEFAULT_REQUEST_TIMEOUT_S):
    """Issue one JSON HTTP request against :data:`SERVICE`.

    ``body=None`` sends no payload.  The opener ignores every proxy
    (``ProxyHandler({})``).  This is the only function that touches the
    network, which keeps the contract tests trivial to mock.
    """

    url = SERVICE + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:  # non-2xx
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - detail is best-effort only
            detail = ""
        raise HttpError("HTTP %s on %s %s: %s" % (exc.code, method, path, detail)) from exc
    except urllib.error.URLError as exc:
        raise HttpError("connection failed on %s %s: %s" % (method, path, exc.reason)) from exc
    text = raw.decode("utf-8") if raw else ""
    if not text.strip():
        return {}
    return json.loads(text)


def _call(method, path, body=None, timeout=DEFAULT_REQUEST_TIMEOUT_S):
    """Wrap :func:`http_json`; return ``(response_or_None, error_or_None)``."""

    try:
        return http_json(method, path, body, timeout=timeout), None
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, never raised
        return None, "%s: %s" % (type(exc).__name__, exc)


def _cancel_plan(plan_id):
    """Best-effort, bounded cancel of a single plan.  Never raises."""

    try:
        return http_json("POST", "/plans/%s/cancel" % plan_id, None, timeout=CANCEL_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - the timeout failure is already recorded
        return None


# --- trial execution ---------------------------------------------------------


def run_trial(condition_name, state, seed, timeout):
    """Run exactly one capability audit trial and return its record.

    A trial always gets a **new** session with the requested scene, seed and
    ``init_state_index``; a **fresh** ``request_id``; and one monotonic
    deadline.  Nothing is reused between trials and no failure is silently
    turned into a pass: any HTTP error, missing/invalid evaluation or oracle
    mismatch leaves ``task_success`` as ``None``.  The function never raises
    for service-side failures and never resets model/environment state.
    """

    if condition_name not in CONDITIONS:
        raise ValueError("unknown condition: %r" % (condition_name,))
    condition = CONDITIONS[condition_name]

    trial = {
        "condition": condition_name,
        "state": state,
        "seed": seed,
        "scene_id": condition["scene_id"],
        "case_id": condition["case_id"],
        "execution_kind": condition["execution_kind"],
        "capability_ids": list(condition["capability_ids"]),
        "budget_per_subgoal": condition["budget_per_subgoal"],
        "request_id": uuid.uuid4().hex,
        "plan_id": None,
        "session_id": None,
        "job_ids": [],
        "env_instance_id": None,
        "initial_images": None,
        "session": None,
        "plan": None,
        "jobs": [],
        "evaluation": None,
        "plan_success": None,
        "task_success": None,
        "error": None,
        "elapsed_s": None,
    }

    started = time.monotonic()
    deadline = started + float(timeout)

    def remaining():
        return deadline - time.monotonic()

    def request_timeout():
        left = remaining()
        if left <= 0:
            return 0.0
        return min(left, DEFAULT_REQUEST_TIMEOUT_S)

    def fail(message):
        trial["error"] = message if not trial["error"] else "%s | %s" % (trial["error"], message)
        trial["elapsed_s"] = round(time.monotonic() - started, 3)
        return trial

    # 1. Fresh session (never reused, never created while a plan is active).
    if remaining() <= 0:
        return fail("timeout before the session could be created")
    session, error = _call(
        "POST",
        "/sessions",
        {
            "scene_id": condition["scene_id"],
            "seed": seed,
            "init_state_index": state,
        },
        timeout=request_timeout(),
    )
    if error is not None:
        return fail("session request failed: %s" % error)
    if not isinstance(session, dict) or not session.get("session_id"):
        return fail("session response missing session_id")
    trial["session"] = session
    trial["session_id"] = session["session_id"]
    trial["env_instance_id"] = session.get("env_instance_id")
    # Retain the raw public images (with their explicit sha256 metadata).
    trial["initial_images"] = session.get("images")

    # 2. One fresh plan request.
    if remaining() <= 0:
        return fail("timeout before the plan could be created")
    plan_body = {
        "session_id": trial["session_id"],
        "scene_version": session.get("scene_version"),
        "request_id": trial["request_id"],
        "capability_ids": list(condition["capability_ids"]),
        "rationale": RATIONALE,
        "decision": "execute",
        "audit": True,
        "budget_per_subgoal": condition["budget_per_subgoal"],
    }
    plan, error = _call("POST", "/plans", plan_body, timeout=request_timeout())
    if error is not None:
        return fail("plan request failed: %s" % error)
    if not isinstance(plan, dict):
        return fail("plan response is not an object")
    trial["plan"] = plan
    plan_id = plan.get("plan_id") or trial["request_id"]
    trial["plan_id"] = plan_id

    # 3. Poll ONLY this plan until it reaches a terminal state (never resume).
    while plan.get("state") not in TERMINAL_PLAN_STATES:
        if remaining() <= 0:
            _cancel_plan(plan_id)
            return fail("trial deadline exceeded; plan %s cancelled" % plan_id)
        time.sleep(max(0.0, min(POLL_INTERVAL_S, remaining())))
        if remaining() <= 0:
            _cancel_plan(plan_id)
            return fail("trial deadline exceeded; plan %s cancelled" % plan_id)
        polled, error = _call("GET", "/plans/%s" % plan_id, None, timeout=request_timeout())
        if error is not None:
            return fail("plan poll failed: %s" % error)
        if not isinstance(polled, dict):
            return fail("plan poll returned a non-object")
        plan = polled
        trial["plan"] = plan

    trial["job_ids"] = list(plan.get("job_ids") or [])
    trial["plan_success"] = plan.get("plan_success")

    plan_state = plan.get("state")
    if plan_state not in EVALUATED_PLAN_STATES:
        # error / cancelled (or the timeout path above) -> never a success.
        return fail("plan ended in state %r" % plan_state)

    # 4. Preserve every raw job for this plan.
    for job_id in trial["job_ids"]:
        job, error = _call("GET", "/jobs/%s" % job_id, None, timeout=request_timeout())
        if error is not None:
            return fail("job %s fetch failed: %s" % (job_id, error))
        if not isinstance(job, dict):
            return fail("job %s response is not an object" % job_id)
        trial["jobs"].append(job)

    # 5. Independent evaluation with the SAME session_id and SAME request_id.
    if remaining() <= 0:
        return fail("timeout before independent evaluation")
    evaluation, error = _call(
        "POST",
        "/evaluate",
        {
            "session_id": trial["session_id"],
            "case_id": condition["case_id"],
            "request_id": trial["request_id"],
        },
        timeout=request_timeout(),
    )
    if error is not None:
        return fail("evaluation request failed: %s" % error)
    trial["evaluation"] = evaluation
    if not isinstance(evaluation, dict):
        return fail("evaluation response is not an object")
    reason = evaluation.get("reason")
    if evaluation.get("ok") is False or evaluation.get("error") or reason == "invalid_fixture":
        return fail("evaluation error: %s" % (reason or evaluation.get("error") or "invalid_fixture"))
    if evaluation.get("oracle_source") != ORACLE_SOURCE:
        return fail("evaluation oracle_source invalid: %r" % evaluation.get("oracle_source"))
    value = evaluation.get("task_success")
    if not isinstance(value, bool):
        return fail("evaluation task_success is not a boolean")
    trial["task_success"] = value
    trial["elapsed_s"] = round(time.monotonic() - started, 3)
    return trial


# --- statistics --------------------------------------------------------------


def wilson(successes, n):
    """95% Wilson score interval as ``[lower, upper]`` clamped to ``[0, 1]``.

    Returns ``None`` when there are no evaluable samples (``n <= 0``).
    """

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        return None
    try:
        s = int(successes)
    except (TypeError, ValueError):
        s = 0
    if s < 0:
        s = 0
    if s > n:
        s = n
    z = 1.959963984540054
    phat = s / float(n)
    denominator = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denominator
    margin = (z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return [lower, upper]


def summarize(trials):
    """Per-condition and overall counts plus 95% Wilson intervals.

    ``n`` counts every recorded trial; ``evaluable`` counts literal boolean
    ``task_success`` results; ``successes``/``failures`` split those booleans;
    ``errors`` counts null / missing / invalid-fixture / service failures.
    Error trials are never dropped from ``n``.
    """

    conditions = {}
    order = []
    totals = {"n": 0, "evaluable": 0, "successes": 0, "failures": 0, "errors": 0}
    for trial in trials:
        name = trial.get("condition")
        if name not in conditions:
            conditions[name] = {
                "condition": name,
                "n": 0,
                "evaluable": 0,
                "successes": 0,
                "failures": 0,
                "errors": 0,
                "wilson_95": None,
            }
            order.append(name)
        entry = conditions[name]
        value = trial.get("task_success")
        is_bool = isinstance(value, bool)
        entry["n"] += 1
        totals["n"] += 1
        if is_bool:
            entry["evaluable"] += 1
            totals["evaluable"] += 1
            if value:
                entry["successes"] += 1
                totals["successes"] += 1
            else:
                entry["failures"] += 1
                totals["failures"] += 1
        else:
            entry["errors"] += 1
            totals["errors"] += 1

    for name in order:
        entry = conditions[name]
        entry["wilson_95"] = wilson(entry["successes"], entry["evaluable"])

    totals["wilson_95"] = wilson(totals["successes"], totals["evaluable"])
    return {"conditions": conditions, "totals": totals}


# --- reporting ---------------------------------------------------------------


def _atomic_write(path, report):
    """Write ``report`` as JSON via a sibling temp file then ``os.replace``."""

    payload = json.dumps(report, indent=2, allow_nan=False, default=str)
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:  # noqa: BLE001 - fsync is a nicety, not a requirement
            pass
    os.replace(tmp_path, str(path))


def _check_health(health):
    """Return an error string when the health gate is not satisfied."""

    if not isinstance(health, dict):
        return "health response is not an object"
    if health.get("ready") is not True:
        return "service is not ready (ready=%r)" % health.get("ready")
    if health.get("workflow") != WORKFLOW:
        return "workflow mismatch (got %r, expected %r)" % (health.get("workflow"), WORKFLOW)
    if health.get("model_revision") != MODEL_REVISION:
        return "model_revision mismatch (got %r, expected %r)" % (
            health.get("model_revision"),
            MODEL_REVISION,
        )
    return None


def _progress(index, total, trial):
    print(
        "[%d/%d] %s state=%s seed=%s -> task_success=%s plan_success=%s "
        "elapsed=%ss error=%s"
        % (
            index,
            total,
            trial.get("condition"),
            trial.get("state"),
            trial.get("seed"),
            trial.get("task_success"),
            trial.get("plan_success"),
            trial.get("elapsed_s"),
            trial.get("error"),
        )
    )


def main(argv=None):
    """Parse CLI arguments, gate on health, run the serial trial grid."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS.keys()),
        default=None,
        help="conditions to run (default: all ten)",
    )
    parser.add_argument("--states", nargs="+", type=int, default=[0], help="init_state_index values")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-trial timeout in seconds")
    args = parser.parse_args(argv)

    conditions = list(args.conditions) if args.conditions else list(CONDITIONS.keys())
    states = list(args.states)
    output = args.output

    report = {
        "experiment": "persistent_scene_direct_vs_manual_subgoal_audit",
        "checkpoint_revision": MODEL_REVISION,
        "service": SERVICE,
        "seed": args.seed,
        "requested_conditions": conditions,
        "requested_states": states,
        "n_requested": len(states) * len(conditions),
        "trial_timeout_s": args.timeout,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "health": None,
        "is_error": False,
        "error": None,
        "trials": [],
        "summary": summarize([]),
        "note": NOTE,
    }

    health, health_error = _call("GET", "/health", timeout=min(DEFAULT_REQUEST_TIMEOUT_S, args.timeout))
    report["health"] = health
    if health_error is None:
        health_error = _check_health(health)
    if health_error is not None:
        report["error"] = "health check failed: %s" % health_error
        report["is_error"] = True
        _atomic_write(output, report)
        print("HEALTH CHECK FAILED: %s" % health_error)
        return 1

    total = report["n_requested"]
    index = 0
    for state in states:
        for condition in conditions:
            index += 1
            try:
                trial = run_trial(condition, state, args.seed, args.timeout)
            except Exception as exc:  # noqa: BLE001 - a trial must never abort the run
                trial = {
                    "condition": condition,
                    "state": state,
                    "seed": args.seed,
                    "task_success": None,
                    "plan_success": None,
                    "session": None,
                    "plan": None,
                    "jobs": [],
                    "evaluation": None,
                    "error": "runner exception: %s" % exc,
                    "elapsed_s": None,
                }
            report["trials"].append(trial)
            report["summary"] = summarize(report["trials"])
            _atomic_write(output, report)
            _progress(index, total, trial)

    return 0


if __name__ == "__main__":
    sys.exit(main())
