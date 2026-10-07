#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-free HTTP contract tests for ``scene_demo/app.py``.

These run the *real* app ``Handler`` on an ephemeral in-process HTTP server with a
fake service proxy (no GPU/Hermes/third-party network) and a fake background
runner (``subprocess.run`` replaced), so the actual ownership, marker and status
logic of ``app.py`` is exercised end to end:

* exact request/session-owned cancellation (unknown 404 / wrong owner 409);
* a pre-plan cancel marker written only after the service acknowledgement;
* idempotent repeat Stop and terminal noop;
* honest failure when the service does not acknowledge (no marker, no success);
* RUNNING stays busy until the real runner exits;
* a real ``cancelled_by_user`` / ``plan.state == cancelled`` result maps to the
  ``cancelled`` public status;
* internal process objects / Events never leak into the public job view.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import app  # noqa: E402


class _FakeService:
    """Records proxy calls and returns canned JSON; can be made to fail cancel."""

    def __init__(self):
        self.calls = []
        self.fail_cancel = False
        self.plans = {}          # request_id -> canned read-only plan
        self.jobs = {}           # job_id -> canned read-only public job record
        self.plan_failures = 0   # remaining /plans calls that raise (unreachable)
        self.job_failures = 0    # remaining /jobs calls that raise (unreachable)
        self.lock = threading.Lock()

    def __call__(self, method, path, body=None, timeout=None):
        with self.lock:
            self.calls.append((method, path, body, timeout))
        if method == "GET" and path.startswith("/plans/"):
            rid = urllib.parse.unquote(path[len("/plans/"):])
            with self.lock:
                if self.plan_failures > 0:
                    self.plan_failures -= 1
                    raise RuntimeError("service unreachable")
                plan = self.plans.get(rid)
            if plan is None:
                raise urllib.error.HTTPError(
                    path, 404, "not found", {}, io.BytesIO(b"{}"))
            return plan
        if method == "GET" and path.startswith("/jobs/"):
            jid = urllib.parse.unquote(path[len("/jobs/"):])
            with self.lock:
                if self.job_failures > 0:
                    self.job_failures -= 1
                    raise RuntimeError("service unreachable")
                job = self.jobs.get(jid)
            if job is None:
                raise urllib.error.HTTPError(
                    path, 404, "not found", {}, io.BytesIO(b"{}"))
            return job
        if method == "GET" and path.startswith("/sessions/"):
            sid = urllib.parse.unquote(path[len("/sessions/"):])
            return {"ok": True, "session_id": sid, "state": "ready",
                    "scene_version": 0, "images": [], "capabilities": [],
                    "storage_policy": {}}
        if method == "GET" and path == "/health":
            return {"ok": True, "ready": True}
        if method == "POST" and path.endswith("/cancel"):
            if self.fail_cancel:
                raise urllib.error.HTTPError(
                    path, 502, "bad gateway", {}, io.BytesIO(b"{}"))
            rid = path.split("/")[2] if path.startswith("/requests/") else None
            return {"ok": True, "request_id": rid,
                    "session_id": (body or {}).get("session_id"),
                    "cancel_requested": True, "state": "cancelling", "plan": None}
        if method == "GET" and path == "/scenes":
            return {"ok": True, "scenes": []}
        return {"ok": True}

    def cancel_calls(self):
        with self.lock:
            return [c for c in self.calls if c[0] == "POST" and c[1].endswith("/cancel")]


class _FakeRunner:
    """Stands in for subprocess.run: blocks until released, then writes a result."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cmds = []
        self.results = {}  # request_id -> result payload written to agent_result.json
        self.lock = threading.Lock()

    def __call__(self, cmd, cwd=None, stdout=None, stderr=None, **kwargs):
        with self.lock:
            self.cmds.append(list(cmd))
        self.entered.set()
        self.release.wait(timeout=15)
        run_dir = None
        request_id = None
        for index, arg in enumerate(cmd):
            if arg == "--run-dir":
                run_dir = cmd[index + 1]
            if arg == "--request-id":
                request_id = cmd[index + 1]
        payload = self.results.get(request_id, {}) if request_id else {}
        if run_dir:
            with open(os.path.join(run_dir, "agent_result.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"")


class AppContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service = _FakeService()
        self.runner = _FakeRunner()

        self._orig_runs = app.RUNS_DIR
        self._orig_proxy = app.proxy_json
        self._orig_subprocess = app.subprocess
        self._orig_jobs = app.JOBS
        self._orig_running = app.RUNNING
        self._orig_plan_interval = app.CANCEL_PLAN_POLL_INTERVAL

        app.RUNS_DIR = self.tmp.name
        app.proxy_json = self.service
        app.subprocess = types.SimpleNamespace(
            PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, run=self.runner)
        app.JOBS = {}
        app.RUNNING = {"request_id": None}
        app.CANCEL_PLAN_POLL_INTERVAL = 0.02   # speed up pending-cancel polling

        self.server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._teardown)

    def _teardown(self):
        self.runner.release.set()  # unblock any still-parked background thread
        self.server.shutdown()
        self.server.server_close()
        app.RUNS_DIR = self._orig_runs
        app.proxy_json = self._orig_proxy
        app.subprocess = self._orig_subprocess
        app.JOBS = self._orig_jobs
        app.RUNNING = self._orig_running
        app.CANCEL_PLAN_POLL_INTERVAL = self._orig_plan_interval

    # -- helpers ------------------------------------------------------------ #
    def _request(self, method, path, body=None):
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw.decode("utf-8")) if raw else None)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                payload = None
            return exc.code, payload

    def _start_agent(self, session_id="sess-1", request="整理桌面"):
        status, payload = self._request(
            "POST", "/api/agent", {"session_id": session_id, "request": request})
        self.assertEqual(status, 202, payload)
        self.assertTrue(self.runner.entered.wait(5), "runner never started")
        return payload["request_id"]

    def _cancel(self, rid, session_id="sess-1"):
        return self._request("POST", "/api/agent/%s/cancel" % rid,
                             {"session_id": session_id})

    def _wait_terminal(self, rid, timeout=10):
        deadline = time.time() + timeout
        job = None
        while time.time() < deadline:
            _, job = self._request("GET", "/api/agent/%s" % rid)
            if job and job.get("status") in ("completed", "error", "cancelled"):
                return job
            time.sleep(0.05)
        self.fail("job %s never reached terminal; last=%r" % (rid, job))

    # -- tests -------------------------------------------------------------- #
    def test_cancel_unknown_request_is_404(self):
        status, payload = self._cancel("deadbeef")
        self.assertEqual(status, 404)
        self.assertEqual(payload.get("request_id"), "deadbeef")

    def test_cancel_wrong_owner_is_409(self):
        rid = self._start_agent("sess-1")
        status, payload = self._cancel(rid, "sess-2")
        self.assertEqual(status, 409)
        self.assertFalse(app.JOBS[rid].get("cancel_requested"))

    def test_cancel_requires_session_id(self):
        rid = self._start_agent("sess-1")
        status, _ = self._request("POST", "/api/agent/%s/cancel" % rid, {})
        self.assertEqual(status, 400)

    def test_pre_plan_marker_written_only_after_ack(self):
        rid = self._start_agent("sess-1")
        # No plan was ever submitted (the fake runner is parked before any plan).
        status, payload = self._cancel(rid)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["cancel_requested"])
        self.assertEqual(payload["status"], "cancelling")
        self.assertFalse(payload.get("noop"))

        job = app.JOBS[rid]
        marker = os.path.join(job["run_dir"], "cancel_requested.json")
        self.assertTrue(os.path.isfile(marker))
        with open(marker, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(sorted(data.keys()),
                         ["request_id", "requested_at", "session_id"])
        self.assertEqual(data["request_id"], rid)
        self.assertEqual(data["session_id"], "sess-1")
        self.assertTrue(data["requested_at"])
        self.assertTrue(job["cancel_requested"])
        self.assertEqual(job["status"], "cancelling")

        # The exact-request cancel was proxied once; the runner got --cancel-file.
        self.assertEqual([c[1] for c in self.service.cancel_calls()],
                         ["/requests/%s/cancel" % rid])
        cmd = self.runner.cmds[-1]
        self.assertIn("--cancel-file", cmd)
        self.assertEqual(cmd[cmd.index("--cancel-file") + 1], marker)

    def test_service_failure_is_honest_and_writes_no_marker(self):
        rid = self._start_agent("sess-1")
        self.service.fail_cancel = True
        status, payload = self._cancel(rid)
        self.assertGreaterEqual(status, 400)
        self.assertLess(status, 600)
        job = app.JOBS[rid]
        self.assertFalse(job.get("cancel_requested"))
        self.assertNotEqual(job.get("status"), "cancelling")
        marker = os.path.join(job["run_dir"], "cancel_requested.json")
        self.assertFalse(os.path.exists(marker))

    def test_repeat_stop_is_idempotent_before_terminal(self):
        rid = self._start_agent("sess-1")
        first = self._cancel(rid)
        second = self._cancel(rid)
        self.assertEqual(first[0], 200)
        self.assertEqual(second[0], 200)
        self.assertTrue(second[1]["cancel_requested"])
        self.assertEqual(second[1]["status"], "cancelling")
        marker = os.path.join(app.JOBS[rid]["run_dir"], "cancel_requested.json")
        self.assertTrue(os.path.isfile(marker))

    def test_running_stays_busy_until_actual_exit(self):
        rid = self._start_agent("sess-1")
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertIn(job["status"], ("queued", "running"))

        # A second submission while RUNNING is rejected (never silently replaced).
        status, payload = self._request(
            "POST", "/api/agent", {"session_id": "sess-1", "request": "另一个请求"})
        self.assertEqual(status, 409)
        self.assertTrue(payload.get("busy"))

        self.assertEqual(self._cancel(rid)[0], 200)
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertNotIn(job["status"], ("completed", "error", "cancelled"))

        # Only the real runner exit produces a terminal status.
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "plan": None,
            "run_ok": False, "error": "cancelled_by_user"}
        self.runner.release.set()
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(app.RUNNING["request_id"])

        # Terminal Stop is a noop returning the existing status.
        status, payload = self._cancel(rid)
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("noop"))
        self.assertEqual(payload["status"], "cancelled")

    def test_cancelled_status_from_real_plan_state(self):
        rid = self._start_agent("sess-1")
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "plan": {"state": "cancelled"}, "run_ok": False}
        self.runner.release.set()
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")

    def test_public_job_excludes_internal_objects_and_events(self):
        rid = self._start_agent("sess-1")
        with app.JOBS_LOCK:
            app.JOBS[rid]["_proc"] = threading.Event()
            app.JOBS[rid]["_thread"] = threading.current_thread()
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertNotIn("_proc", job)
        self.assertNotIn("_thread", job)
        _, listing = self._request("GET", "/api/agent")
        entry = next(j for j in listing["jobs"] if j["request_id"] == rid)
        self.assertNotIn("_proc", entry)
        self.assertNotIn("_thread", entry)

    # -- public status snapshots ------------------------------------------- #
    def test_listing_is_a_snapshot_not_a_live_reference(self):
        rid = self._start_agent("sess-1")
        _, listing = self._request("GET", "/api/agent")
        entry = next(j for j in listing["jobs"] if j["request_id"] == rid)
        # The exposed dict is a fresh copy, never the live JOBS entry.
        self.assertIsNot(entry, app.JOBS[rid])
        self.assertIsNot(app.public_job(app.JOBS[rid]), app.JOBS[rid])
        before = dict(entry)
        # Mutating the live job afterwards cannot retroactively change the snapshot.
        with app.JOBS_LOCK:
            app.JOBS[rid]["status"] = "cancelling"
            app.JOBS[rid]["cancel_requested"] = True
        self.assertEqual(entry, before)

    def test_agent_status_is_a_snapshot_copy(self):
        rid = self._start_agent("sess-1")
        _, snapshot = self._request("GET", "/api/agent/%s" % rid)
        self.assertIsNot(snapshot, app.JOBS[rid])
        self.assertEqual(snapshot.get("request_id"), rid)

    # -- cancel acknowledgement racing real completion --------------------- #
    def test_cancel_ack_race_returns_actual_terminal_status(self):
        rid = self._start_agent("sess-1")
        original = self.service.__call__

        def racing(method, path, body=None, timeout=None):
            reply = original(method, path, body, timeout)
            if method == "POST" and path.endswith("/cancel"):
                # The real runner reaches terminal *during* the service ack.
                with app.JOBS_LOCK:
                    app.JOBS[rid]["status"] = "completed"
                    app.JOBS[rid]["finished"] = "now"
            return reply

        app.proxy_json = racing
        status, payload = self._cancel(rid)
        self.assertEqual(status, 200, payload)
        # Honest: report the job's actual current status, never hardcoded cancelling.
        self.assertEqual(payload["status"], "completed")
        self.assertNotEqual(payload["status"], "cancelling")

    # -- cancellation_pending: hold RUNNING until the real plan is terminal - #
    def test_cancellation_pending_keeps_busy_until_real_plan_terminal(self):
        rid = self._start_agent("sess-1")
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "running", "decision": "execute", "job_ids": []}
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True,
            "plan": {"request_id": rid, "session_id": "sess-1", "state": "running"},
            "run_ok": False}
        self.runner.release.set()
        time.sleep(0.4)

        # Plan still active: NOT cancelled, RUNNING still occupied, still cancelling.
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")
        self.assertIsNotNone(app.RUNNING["request_id"])
        status, payload = self._request(
            "POST", "/api/agent", {"session_id": "sess-1", "request": "另一个"})
        self.assertEqual(status, 409)
        self.assertTrue(payload.get("busy"))

        # The real exact plan finally reaches its true terminal state.
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "cancelled", "decision": "execute",
                                   "job_ids": []}
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(app.RUNNING["request_id"])
        # Only the real exact plan was merged; nothing invented.
        self.assertEqual(job["result"]["plan"]["request_id"], rid)
        self.assertEqual(job["result"]["plan"]["state"], "cancelled")

    def test_cancellation_pending_unreachable_notes_then_recovers(self):
        rid = self._start_agent("sess-1")
        self.service.plan_failures = 10 ** 6   # backend unreachable
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True,
            "plan": {"request_id": rid, "session_id": "sess-1", "state": "running"},
            "run_ok": False}
        self.runner.release.set()
        time.sleep(0.4)

        # Unreachable is NOT confirmation: stays cancelling with an explicit note.
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")
        self.assertTrue(job.get("cancel_note"))
        self.assertNotIn(job["status"], ("completed", "error", "cancelled"))
        self.assertIsNotNone(app.RUNNING["request_id"])

        # Recovery: reachable backend reports the real terminal plan.
        self.service.plan_failures = 0
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "cancelled", "decision": "execute",
                                   "job_ids": []}
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(app.RUNNING["request_id"])
        self.assertNotIn("cancel_note", job)

    def test_cancellation_without_plan_is_real_terminal_cancelled(self):
        rid = self._start_agent("sess-1")
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": False, "plan": None}
        self.runner.release.set()
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")

    def test_cancellation_pending_without_plan_is_terminal_cancelled(self):
        # No plan was ever submitted: a real terminal cancellation, never a poll loop.
        rid = self._start_agent("sess-1")
        self.service.plan_failures = 10 ** 6   # polling here would hang forever
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True, "plan": None}
        self.runner.release.set()
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(app.RUNNING["request_id"])

    def test_cancelled_plan_beats_cancelled_by_user_flag(self):
        # A real terminal (non-cancelled) plan must win over cancelled_by_user.
        rid = self._start_agent("sess-1")
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": False,
            "plan": {"request_id": rid, "session_id": "sess-1",
                     "state": "completed", "job_ids": []}}
        self.runner.release.set()
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "completed")

    # -- exact plan ownership: request_id AND session_id both required ------ #
    def _job_get_paths(self):
        return [c[1] for c in self.service.calls
                if c[0] == "GET" and c[1].startswith("/jobs/")]

    def _plan_get_paths(self):
        return [c[1] for c in self.service.calls
                if c[0] == "GET" and c[1].startswith("/plans/")]

    def test_pending_terminal_plan_missing_request_id_is_ignored(self):
        rid = self._start_agent("sess-1")
        # Backend reports a terminal plan WITHOUT request_id: it is NOT accepted.
        self.service.plans[rid] = {"session_id": "sess-1", "state": "cancelled",
                                   "decision": "execute", "job_ids": []}
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True,
            "plan": {"request_id": rid, "session_id": "sess-1", "state": "running"},
            "run_ok": False}
        self.runner.release.set()
        time.sleep(0.3)

        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")
        self.assertNotIn(job["status"], ("completed", "error", "cancelled"))
        self.assertIsNotNone(app.RUNNING["request_id"])

        # The exact owned plan (request_id AND session_id) is finally accepted.
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "cancelled", "decision": "execute",
                                   "job_ids": []}
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(app.RUNNING["request_id"])

    def test_pending_terminal_plan_wrong_or_missing_session_is_ignored(self):
        rid = self._start_agent("sess-1")
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True,
            "plan": {"request_id": rid, "session_id": "sess-1", "state": "running"},
            "run_ok": False}
        # (a) missing session_id: NOT accepted.
        self.service.plans[rid] = {"request_id": rid, "state": "cancelled",
                                   "decision": "execute", "job_ids": []}
        self.runner.release.set()
        time.sleep(0.3)
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")

        # (b) foreign session_id: NOT accepted.
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-2",
                                   "state": "cancelled", "decision": "execute",
                                   "job_ids": []}
        time.sleep(0.3)
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")
        self.assertIsNotNone(app.RUNNING["request_id"])

        # (c) exact ownership finally allows terminal recovery.
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "cancelled", "decision": "execute",
                                   "job_ids": []}
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(app.RUNNING["request_id"])

    # -- terminal job refresh: owned public records only -------------------- #
    def test_terminal_refresh_replaces_stale_jobs_with_owned_records(self):
        rid = self._start_agent("sess-1")
        # The runner's 10-second window left a STALE snapshot (partial steps).
        stale = {"job_id": "job-1", "capability_id": "cap-a",
                 "request_id": rid, "session_id": "sess-1",
                 "state": "running", "success": None, "steps": 3, "total_steps": 3}
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True,
            "plan": {"request_id": rid, "session_id": "sess-1", "state": "running"},
            "jobs": [stale], "usage": {"tokens": 5}, "wall_s": 12.5,
            "evaluation": {"task_success": True}, "run_ok": False}
        # Fresh real public records from GET /jobs/<exact id>.
        self.service.jobs["job-1"] = {"job_id": "job-1", "request_id": rid,
                                      "session_id": "sess-1", "capability_id": "cap-a",
                                      "state": "success", "success": True,
                                      "steps": 9, "total_steps": 9}
        self.service.jobs["job-2"] = {"job_id": "job-2", "request_id": rid,
                                      "session_id": "sess-1", "capability_id": "cap-b",
                                      "state": "success", "success": True,
                                      "steps": 4, "total_steps": 4}
        # A foreign job that must never be fetched/adopted (no latest fallback).
        self.service.jobs["job-FOREIGN"] = {"job_id": "job-FOREIGN",
                                            "request_id": rid, "session_id": "sess-1",
                                            "steps": 1, "total_steps": 1}
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "completed", "decision": "execute",
                                   "job_ids": ["job-1", "job-2"]}
        self.runner.release.set()
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "completed")

        result = job["result"]
        # Only the freshly refreshed owned records are retained, in exact order.
        self.assertEqual([j["job_id"] for j in result["jobs"]], ["job-1", "job-2"])
        self.assertEqual([j["steps"] for j in result["jobs"]], [9, 4])
        self.assertEqual([j["total_steps"] for j in result["jobs"]], [9, 4])
        self.assertEqual(result["plan"]["request_id"], rid)
        self.assertEqual(result["plan"]["state"], "completed")
        # Real usage/logs/scores preserved; cancellation_pending cleared.
        self.assertEqual(result["usage"], {"tokens": 5})
        self.assertEqual(result["wall_s"], 12.5)
        self.assertEqual(result["evaluation"], {"task_success": True})
        self.assertNotIn("cancellation_pending", result)
        # Exact read-only GET paths only; no latest fallback fetch.
        self.assertEqual(self._job_get_paths(), ["/jobs/job-1", "/jobs/job-2"])
        self.assertNotIn("/jobs/job-FOREIGN", self._job_get_paths())
        self.assertIn("/plans/%s" % rid, self._plan_get_paths())

    def test_terminal_refresh_rejects_foreign_jobs(self):
        rid = self._start_agent("sess-1")
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True,
            "plan": {"request_id": rid, "session_id": "sess-1", "state": "running"},
            "jobs": [{"job_id": "job-1", "request_id": rid, "session_id": "sess-1",
                      "steps": 99, "total_steps": 99}],
            "run_ok": False}
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "cancelled", "decision": "execute",
                                   "job_ids": ["job-1"]}
        self.runner.release.set()

        # (a) wrong request_id for the same job_id: NOT adopted.
        self.service.jobs["job-1"] = {"job_id": "job-1", "request_id": "other-req",
                                      "session_id": "sess-1", "steps": 1}
        time.sleep(0.3)
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")
        self.assertIsNotNone(app.RUNNING["request_id"])

        # (b) wrong session_id: NOT adopted.
        self.service.jobs["job-1"] = {"job_id": "job-1", "request_id": rid,
                                      "session_id": "sess-9", "steps": 1}
        time.sleep(0.3)
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")

        # (c) job_id echoed back mismatched: NOT adopted.
        self.service.jobs["job-1"] = {"job_id": "job-other", "request_id": rid,
                                      "session_id": "sess-1", "steps": 1}
        time.sleep(0.3)
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")

        # (d) exact ownership finally arrives: adopted and stale snapshot replaced.
        self.service.jobs["job-1"] = {"job_id": "job-1", "request_id": rid,
                                      "session_id": "sess-1", "steps": 7,
                                      "total_steps": 7}
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual([j["steps"] for j in job["result"]["jobs"]], [7])

    def test_terminal_job_refresh_unavailable_stays_busy_until_valid(self):
        rid = self._start_agent("sess-1")
        self.runner.results[rid] = {
            "request_id": rid, "session_id": "sess-1",
            "cancelled_by_user": True, "cancellation_pending": True,
            "plan": {"request_id": rid, "session_id": "sess-1", "state": "running"},
            "run_ok": False}
        self.service.plans[rid] = {"request_id": rid, "session_id": "sess-1",
                                   "state": "cancelled", "decision": "execute",
                                   "job_ids": ["job-1"]}
        self.service.job_failures = 10 ** 6   # /plans terminal but /jobs unreachable
        self.runner.release.set()
        time.sleep(0.3)

        # Terminal plan confirmed, but no owned job data: STILL cancelling/busy.
        _, job = self._request("GET", "/api/agent/%s" % rid)
        self.assertEqual(job["status"], "cancelling")
        self.assertTrue(job.get("cancel_note"))
        self.assertNotIn(job["status"], ("completed", "error", "cancelled"))
        self.assertIsNotNone(app.RUNNING["request_id"])

        # Backend recovers with the exact owned job: only then terminal recovery.
        self.service.job_failures = 0
        self.service.jobs["job-1"] = {"job_id": "job-1", "request_id": rid,
                                      "session_id": "sess-1", "steps": 2,
                                      "total_steps": 2}
        job = self._wait_terminal(rid)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["result"]["jobs"][0]["steps"], 2)
        self.assertIsNone(app.RUNNING["request_id"])


if __name__ == "__main__":
    unittest.main()
