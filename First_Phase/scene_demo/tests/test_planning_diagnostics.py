#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused, GPU-free tests for planning_diagnostics.py.

These tests never import a model, the simulator, the production service or any
credential.  They exercise the pure scorer, the public/prompt oracle exclusion
and the local capture HTTP handler (correct/wrong context, queued plans,
duplicate retention) while proving the handler never forwards anything: any
outbound ``urllib`` call is monkeypatched to raise, and the test client talks
over ``http.client`` instead.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import planning_diagnostics as pd  # noqa: E402

with open(os.path.join(ROOT, "fixtures.json"), "r", encoding="utf-8") as _fh:
    _FIXTURE_DATA = json.load(_fh)
FIX = {case["case_id"]: case for case in _FIXTURE_DATA["cases"]}


def _caps(scene_id):
    import catalog  # stdlib-only sibling; safe

    return catalog.scene_capabilities(scene_id)


def _plan(decision, capability_ids, rationale="r"):
    return {
        "decision": decision,
        "capability_ids": list(capability_ids),
        "rationale": rationale,
    }


def _minimal_input_session(scene_id="goal_table"):
    return {
        "ok": True,
        "session_id": "original-should-be-replaced",
        "scene_id": scene_id,
        "state": "ready",
        "scene_version": 3,
        "total_steps": 590,
        "description": "scene",
        "storage_policy": {"wine_bottle_1": "wine_rack_1_top_region"},
        "capabilities": _caps(scene_id),
        "images": [
            {"view": "agentview", "image_path": "/old/agentview.png"},
            {"view": "wrist", "image_path": "/old/wrist.png"},
        ],
        "latest_png": "/old/latest.png",
    }


# --------------------------------------------------------------------------- #
class ScorePlanTests(unittest.TestCase):
    def test_bowl_and_wine_in_either_order_is_correct(self):
        case = FIX["table_tidy"]
        caps = _caps("goal_table")
        for order in (["bowl_to_plate", "wine_to_rack"],
                      ["wine_to_rack", "bowl_to_plate"]):
            score = pd.score_plan(_plan("execute", order), case, caps)
            self.assertTrue(score["planning_correct"], score)
            self.assertTrue(score["decision_correct"])
            self.assertTrue(score["goal_set_correct"])
            self.assertTrue(score["objects_correct"])
            self.assertTrue(score["single_submission"])

    def test_goal_deduplication_is_order_flexible(self):
        case = FIX["table_tidy"]
        caps = _caps("goal_table")
        score = pd.score_plan(
            _plan("execute", ["wine_to_rack", "bowl_to_plate"]), case, caps
        )
        self.assertTrue(score["goal_set_correct"])

    def test_missing_wine_fails(self):
        case = FIX["table_wine_only"]
        caps = _caps("goal_table")
        score = pd.score_plan(_plan("execute", ["bowl_to_plate"]), case, caps)
        self.assertFalse(score["planning_correct"])
        self.assertFalse(score["goal_set_correct"])
        empty = pd.score_plan(_plan("execute", []), case, caps)
        self.assertFalse(empty["planning_correct"])
        self.assertFalse(empty["decision_correct"])

    def test_added_stove_fails(self):
        case = FIX["table_tidy"]
        caps = _caps("goal_table")
        score = pd.score_plan(
            _plan("execute", ["bowl_to_plate", "wine_to_rack", "stove_on"]),
            case,
            caps,
        )
        self.assertFalse(score["planning_correct"])
        self.assertFalse(score["objects_correct"])
        self.assertFalse(score["goal_set_correct"])

    def test_clarify_with_nonempty_capabilities_fails(self):
        case = FIX["missing_bin"]
        caps = _caps("goal_table")
        score = pd.score_plan(_plan("clarify", ["bowl_to_plate"]), case, caps)
        self.assertFalse(score["decision_correct"])
        self.assertFalse(score["planning_correct"])

    def test_missing_bin_unsupported_empty_passes(self):
        case = FIX["missing_bin"]
        caps = _caps("goal_table")
        score = pd.score_plan(_plan("unsupported", []), case, caps)
        self.assertTrue(score["planning_correct"], score)
        self.assertTrue(score["decision_correct"])
        self.assertTrue(score["single_submission"])

    def test_ambiguous_cleanup_clarify_empty_passes(self):
        case = FIX["ambiguous_cleanup"]
        caps = _caps("goal_table")
        score = pd.score_plan(_plan("clarify", []), case, caps)
        self.assertTrue(score["planning_correct"], score)

    def test_missing_bin_execute_fails(self):
        case = FIX["missing_bin"]
        caps = _caps("goal_table")
        score = pd.score_plan(_plan("execute", ["wine_to_rack"]), case, caps)
        self.assertFalse(score["planning_correct"])
        self.assertFalse(score["decision_correct"])

    def test_invalid_duplicate_and_missing_ids_fail(self):
        case = FIX["table_tidy"]
        caps = _caps("goal_table")
        unknown = pd.score_plan(_plan("execute", ["nope"]), case, caps)
        self.assertFalse(unknown["decision_correct"])
        self.assertFalse(unknown["planning_correct"])

        duplicate = pd.score_plan(
            _plan("execute", ["bowl_to_plate", "bowl_to_plate"]), case, caps
        )
        self.assertFalse(duplicate["decision_correct"])
        self.assertFalse(duplicate["planning_correct"])

        missing_list = pd.score_plan(_plan("execute", []), case, caps)
        self.assertFalse(missing_list["decision_correct"])

        missing_key = pd.score_plan(
            {"decision": "execute", "rationale": "r"}, case, caps
        )
        self.assertFalse(missing_key["planning_correct"])

    def test_non_execute_decision_not_allowed_fails(self):
        case = FIX["table_tidy"]  # allowed_decisions == ["execute"]
        caps = _caps("goal_table")
        score = pd.score_plan(_plan("clarify", []), case, caps)
        self.assertFalse(score["decision_correct"])
        self.assertFalse(score["planning_correct"])

    def test_single_submission_required(self):
        case = FIX["table_tidy"]
        caps = _caps("goal_table")
        good = _plan("execute", ["bowl_to_plate", "wine_to_rack"])
        one = pd.score_plan(good, case, caps, submission_count=1)
        self.assertTrue(one["planning_correct"])
        two = pd.score_plan(good, case, caps, submission_count=2)
        self.assertFalse(two["single_submission"])
        self.assertFalse(two["planning_correct"])
        zero = pd.score_plan(None, case, caps, submission_count=0)
        self.assertFalse(zero["single_submission"])
        self.assertFalse(zero["planning_correct"])

    def test_missing_or_error_plan_cannot_be_correct(self):
        case = FIX["table_tidy"]
        caps = _caps("goal_table")
        for bad in (None, {}, "not-a-plan", []):
            score = pd.score_plan(bad, case, caps)
            self.assertFalse(score["planning_correct"])
            self.assertFalse(score["single_submission"])

    def test_scorer_never_reports_task_or_physical_success(self):
        case = FIX["table_tidy"]
        caps = _caps("goal_table")
        score = pd.score_plan(
            _plan("execute", ["bowl_to_plate", "wine_to_rack"]), case, caps
        )
        blob = json.dumps(score)
        self.assertNotIn("task_success", blob)
        self.assertNotIn("physical", blob)


# --------------------------------------------------------------------------- #
class PublicDataTests(unittest.TestCase):
    def test_public_session_pins_scene_and_image(self):
        session = pd.build_public_session(
            _minimal_input_session(), "fake-0001", "/pkg/cap/first.png"
        )
        self.assertEqual(session["session_id"], "fake-0001")
        self.assertEqual(session["state"], "ready")
        self.assertEqual(session["scene_version"], 0)
        self.assertEqual(session["total_steps"], 0)
        self.assertEqual(len(session["images"]), 1)
        self.assertEqual(session["images"][0]["view"], "agentview")
        self.assertEqual(session["images"][0]["image_path"], "/pkg/cap/first.png")
        self.assertEqual(session["latest_png"], "/pkg/cap/first.png")
        # No stale original agentview/wrist path survives.
        blob = json.dumps(session)
        self.assertNotIn("/old/agentview.png", blob)
        self.assertNotIn("/old/wrist.png", blob)

    def test_build_public_session_strips_oracle_metadata(self):
        dirty = _minimal_input_session()
        dirty["case_id"] = "LEAK"
        dirty["goal_options"] = [["LEAK"]]
        dirty["allowed_decisions"] = ["LEAK"]
        dirty["capabilities"] = [{
            "capability_id": "c1",
            "goals": [["on", "a", "b"]],
            "object_id": "a",
            "case_id": "LEAK",
            "allowed_objects": ["LEAK"],
        }]
        session = pd.build_public_session(dirty, "fake-0002", "/pkg/first.png")
        blob = json.dumps(session, ensure_ascii=False)
        for token in pd.FORBIDDEN_PUBLIC_KEYS:
            self.assertNotIn(token, blob)

    def test_public_and_prompt_exclude_oracle_tokens(self):
        import run_agent  # safe: no side effects on import

        session = pd.build_public_session(
            _minimal_input_session(), "fake-0003", "/pkg/cap/first.png"
        )
        blob = json.dumps(session, ensure_ascii=False)
        for case_id in pd.CASE_IDS:
            case = FIX[case_id]
            prompt = run_agent.build_initial_prompt(
                session, case["request"], "rid-1", ["/pkg/cap/first.png"]
            )
            for token in pd.FORBIDDEN_PUBLIC_KEYS:
                self.assertNotIn(token, prompt)
                self.assertNotIn(token, blob)
            self.assertNotIn(case_id, prompt)
            self.assertNotIn(case_id, blob)
            self.assertIn(case["request"], prompt)

    def test_oracle_excluded_helper_detects_leak(self):
        case = FIX["table_tidy"]
        clean = pd.build_public_session(
            _minimal_input_session(), "fake-0004", "/pkg/cap/first.png"
        )
        self.assertTrue(pd._oracle_excluded(case, clean, "a clean prompt"))
        leaked = dict(clean)
        leaked["allowed_objects"] = ["akita_black_bowl_1"]
        self.assertFalse(pd._oracle_excluded(case, leaked, "a clean prompt"))
        self.assertFalse(pd._oracle_excluded(case, clean, "case_id: table_tidy"))


# --------------------------------------------------------------------------- #
class CaptureHandlerTests(unittest.TestCase):
    def setUp(self):
        self.session = pd.build_public_session(
            _minimal_input_session(), "fake-session-0", "/pkg/cap/first.png"
        )
        self.ctx = pd.FakeSceneContext(self.session, transport_path=None)
        self.server, self.thread = pd.serve_capture_server(
            self.ctx, host="127.0.0.1", port=0
        )
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.ctx.close()

    def _request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            payload = None if body is None else json.dumps(body)
            headers = {} if body is None else {"Content-Type": "application/json"}
            conn.request(method, path, payload, headers)
            response = conn.getresponse()
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
        finally:
            conn.close()

    def _plan_body(self, request_id, capability_ids=("bowl_to_plate",),
                   decision="execute", session_id="fake-session-0",
                   scene_version=0):
        return {
            "session_id": session_id,
            "scene_version": scene_version,
            "request_id": request_id,
            "capability_ids": list(capability_ids),
            "rationale": "r",
            "decision": decision,
        }

    def test_session_and_observe_share_one_native_view(self):
        status, payload = self._request("GET", "/sessions/fake-session-0")
        self.assertEqual(status, 200)
        self.assertEqual(payload["session_id"], "fake-session-0")
        self.assertEqual(payload["scene_version"], 0)
        self.assertEqual(len(payload["images"]), 1)
        self.assertEqual(payload["images"][0]["view"], "agentview")

        status, payload = self._request(
            "POST", "/observe",
            {"session_id": "fake-session-0", "extra_views": True},
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["extra_views"])
        self.assertEqual(len(payload["images"]), 1)
        self.assertEqual(payload["images"][0]["image_path"], "/pkg/cap/first.png")

        status, _ = self._request("GET", "/sessions/wrong-session")
        self.assertEqual(status, 404)

    def test_submit_then_get_queued_plan(self):
        request_id = self.ctx.begin_case("table_tidy")
        status, payload = self._request(
            "POST", "/plans", self._plan_body(request_id)
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"], "queued")
        self.assertEqual(payload["job_ids"], [])
        self.assertEqual(payload["request_id"], request_id)
        self.assertFalse(payload["executed"])
        self.assertEqual(self.ctx.capture_count(request_id), 1)

        status, payload = self._request("GET", "/plans/" + request_id)
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"], "queued")
        self.assertEqual(payload["capability_ids"], ["bowl_to_plate"])

        status, _ = self._request("GET", "/plans/does-not-exist")
        self.assertEqual(status, 404)

    def test_wrong_context_is_rejected_and_not_captured(self):
        request_id = self.ctx.begin_case("table_tidy")

        status, _ = self._request(
            "POST", "/plans", self._plan_body(request_id, session_id="other")
        )
        self.assertEqual(status, 409)

        status, _ = self._request(
            "POST", "/plans", self._plan_body(request_id, scene_version=1)
        )
        self.assertEqual(status, 409)

        status, _ = self._request(
            "POST", "/plans", self._plan_body("not-the-current-request")
        )
        self.assertEqual(status, 409)

        self.assertEqual(self.ctx.capture_count(request_id), 0)
        self.assertTrue(self.ctx.protocol_errors)

    def test_wrong_request_body_is_logged_and_case_error_retained(self):
        # A rejected wrong-request-id attempt must stay auditable: the nonsecret
        # request body is preserved in transport.jsonl, and the protocol error
        # is attributed to the current case context (not the bogus request id).
        case_id = "table_tidy"
        with tempfile.TemporaryDirectory() as tmp:
            transport_path = os.path.join(tmp, "transport.jsonl")
            ctx = pd.FakeSceneContext(self.session, transport_path=transport_path)
            request_id = ctx.begin_case(case_id)
            body = self._plan_body("stale-request-id", ("bowl_to_plate",))
            body["rationale"] = "nonsecret rejected payload"

            status, payload = ctx.request("POST", "/plans", body)
            self.assertEqual(status, 409)
            self.assertEqual(payload["reason"], "wrong_request_id")
            ctx.close()

            with open(transport_path, "r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            rejected = [
                record for record in records
                if record["path"] == "/plans" and record["status"] == 409
            ]
            self.assertTrue(rejected)
            logged = rejected[-1]["body"]
            self.assertEqual(logged["request_id"], "stale-request-id")
            self.assertEqual(logged["capability_ids"], ["bowl_to_plate"])
            self.assertEqual(logged["rationale"], "nonsecret rejected payload")

            case_errors = ctx.protocol_errors_for(case_id)
            self.assertTrue(case_errors)
            self.assertTrue(all(e.get("case_id") == case_id for e in case_errors))
            self.assertEqual(ctx.capture_count(request_id), 0)

    def test_resume_and_unknown_routes_rejected(self):
        request_id = self.ctx.begin_case("table_tidy")
        status, _ = self._request(
            "POST", "/plans/" + request_id + "/resume", self._plan_body(request_id)
        )
        self.assertEqual(status, 405)
        status, _ = self._request("GET", "/nope")
        self.assertEqual(status, 404)
        status, _ = self._request("POST", "/jobs", {})
        self.assertEqual(status, 404)

    def test_duplicate_submissions_retained_and_scored_false(self):
        request_id = self.ctx.begin_case("table_tidy")
        body = self._plan_body(request_id, ("bowl_to_plate", "wine_to_rack"))
        self._request("POST", "/plans", body)
        self._request("POST", "/plans", body)
        captures = self.ctx.captures_for(request_id)
        self.assertEqual(len(captures), 2)
        score = pd.score_plan(
            captures[0], FIX["table_tidy"], _caps("goal_table"),
            submission_count=len(captures),
        )
        self.assertFalse(score["single_submission"])
        self.assertFalse(score["planning_correct"])

    def test_handler_never_forwards_outbound(self):
        import urllib.request

        original_urlopen = urllib.request.urlopen
        original_build_opener = urllib.request.build_opener

        def _boom(*args, **kwargs):
            raise RuntimeError("outbound urllib is forbidden in tests")

        urllib.request.urlopen = _boom
        urllib.request.build_opener = _boom
        try:
            request_id = self.ctx.begin_case("table_tidy")
            status, payload = self._request(
                "POST", "/plans", self._plan_body(request_id)
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload["state"], "queued")
            status, _ = self._request("GET", "/sessions/fake-session-0")
            self.assertEqual(status, 200)
            status, _ = self._request(
                "POST", "/observe", {"session_id": "fake-session-0"}
            )
            self.assertEqual(status, 200)
        finally:
            urllib.request.urlopen = original_urlopen
            urllib.request.build_opener = original_build_opener

        self.assertEqual(self.ctx.forwarded, [])
        self.assertTrue(self.ctx.transport_entries)


# --------------------------------------------------------------------------- #
class CliSafetyTests(unittest.TestCase):
    def test_help_exits_zero_without_running(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as caught:
                pd.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--input-package", buffer.getvalue())


# --------------------------------------------------------------------------- #
class OracleLeakBlockTests(unittest.TestCase):
    """Fail-closed: an oracle-leaking prompt must never reach Hermes."""

    def test_oracle_leak_blocks_hermes_and_records_case_error(self):
        import run_agent
        from unittest import mock

        tmp_root = tempfile.mkdtemp()
        try:
            input_package = os.path.join(tmp_root, "input")
            cap_dir = os.path.join(input_package, *pd.CAP_IMAGE_RELPATH[:-1])
            os.makedirs(cap_dir)
            with open(os.path.join(cap_dir, pd.CAP_IMAGE_RELPATH[-1]), "wb") as fh:
                fh.write(b"\x89PNG\r\n\x1a\n")
            with open(os.path.join(input_package, "session.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(_minimal_input_session("goal_table"), fh)

            output_dir = os.path.join(tmp_root, "out")
            hermes_home = os.path.join(tmp_root, "home")

            calls = []

            class _ForbiddenRunner(object):
                def __init__(self, *args, **kwargs):
                    pass

                def __call__(self, *args, **kwargs):
                    calls.append(args)
                    raise AssertionError("Hermes must never be invoked on a leak")

            real_serve = pd.serve_capture_server

            def _serve_ephemeral(context, host=pd.DEFAULT_HOST,
                                 port=pd.DEFAULT_PORT):
                return real_serve(context, host=host, port=0)

            def _fake_prepare_home(home, source_home=None):
                os.makedirs(home, mode=0o700)
                config_path = os.path.join(home, "config.yaml")
                with open(config_path, "w", encoding="utf-8") as fh:
                    fh.write("model:\n  default: %s\n" % pd.MODEL_NAME)
                return config_path, {}

            with mock.patch.object(pd, "prepare_home", _fake_prepare_home), \
                    mock.patch.object(pd, "serve_capture_server", _serve_ephemeral), \
                    mock.patch.object(pd, "_oracle_excluded", return_value=False), \
                    mock.patch.object(run_agent, "HermesRunner", _ForbiddenRunner):
                exit_code = pd.run_benchmark(input_package, output_dir, hermes_home)

            # Hermes was never called, and the run failed closed.
            self.assertEqual(calls, [])
            self.assertNotEqual(exit_code, 0)

            with open(os.path.join(output_dir, "report.json"),
                      "r", encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertFalse(report["oracle_not_in_prompt"])
            self.assertFalse(report["ok"])
            self.assertEqual(len(report["cases"]), len(pd.CASE_IDS))
            for entry in report["cases"]:
                self.assertIn("oracle_leak", entry["errors"])
                self.assertFalse(entry["score"]["planning_correct"])
                # Evidence preserved: the withheld prompt is still on disk.
                self.assertTrue(os.path.isfile(
                    os.path.join(output_dir, entry["case_id"], "prompt.txt")
                ))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
