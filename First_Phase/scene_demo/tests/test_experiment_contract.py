#!/usr/bin/env python3
"""Mock-only HTTP contract tests for ``run_experiments``.

Every test patches ``run_experiments.http_json`` (and, where needed,
``run_experiments.time``) so no real service, GPU, LIBERO, network or Hermes
call ever happens and no test ever sleeps.  The tests exercise the
*human-authored / direct-VLA capability audit* contract: fresh sessions,
fixed independent evaluation cases, honest error handling and the Wilson
summary semantics.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

# Make ``run_experiments`` importable from this file's own location.
_HERE = Path(__file__).resolve()
_SCENE_DEMO = _HERE.parent.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import run_experiments as R  # noqa: E402


EXPECTED_CONDITIONS = {
    "table_direct": ("goal_table", ["table_both"], 600, "table_tidy", "direct_vla"),
    "table_forward": (
        "goal_table",
        ["bowl_to_plate", "wine_to_rack"],
        300,
        "table_tidy",
        "manual_subgoals",
    ),
    "table_reverse": (
        "goal_table",
        ["wine_to_rack", "bowl_to_plate"],
        300,
        "table_tidy",
        "manual_subgoals",
    ),
    "table_shifted": (
        "goal_table_shifted",
        ["bowl_to_plate", "wine_to_rack"],
        300,
        "table_shifted_tidy",
        "manual_subgoals",
    ),
    "basket_direct": ("basket_two", ["basket_both"], 600, "basket_two_cans", "direct_vla"),
    "basket_split": (
        "basket_two",
        ["soup_to_basket", "sauce_to_basket"],
        300,
        "basket_two_cans",
        "manual_subgoals",
    ),
    "mugs_direct": ("mugs_two", ["mugs_both"], 600, "mugs_standard", "direct_vla"),
    "mugs_split": (
        "mugs_two",
        ["white_mug_left", "yellow_mug_right"],
        300,
        "mugs_standard",
        "manual_subgoals",
    ),
    "free_right": ("mugs_left_occupied", ["white_mug_right"], 300, "mugs_free_right", "manual_subgoals"),
    "free_left": ("mugs_right_occupied", ["white_mug_left"], 300, "mugs_free_left", "manual_subgoals"),
}


class FakeTime:
    """Deterministic clock: only ``sleep`` advances time."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(0.0, float(seconds))


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        return FakeResponse(b'{"ok": true}')


def make_completed_backend(calls, sessions=("s1",)):
    """Return a fake ``http_json`` that completes a clean trial."""

    state = {"session_index": 0}

    def fake(method, path, body=None, timeout=60):
        calls.append((method, path, body))
        if path == "/sessions":
            session_id = sessions[min(state["session_index"], len(sessions) - 1)]
            state["session_index"] += 1
            return {
                "ok": True,
                "session_id": session_id,
                "scene_id": body["scene_id"],
                "scene_version": 0,
                "env_instance_id": 7,
                "episode_resets": 1,
                "seed": body["seed"],
                "init_state_index": body["init_state_index"],
                "images": [{"view": "agentview", "sha256": "deadbeef", "kind": "vla_observation"}],
            }
        if path == "/plans":
            return {
                "ok": True,
                "plan_id": "p1",
                "request_id": body["request_id"],
                "state": "queued",
                "job_ids": [],
                "plan_success": None,
                "scene_version": 0,
            }
        if path == "/plans/p1":
            return {
                "ok": True,
                "plan_id": "p1",
                "state": "completed",
                "job_ids": ["j1"],
                "plan_success": True,
                "scene_version": 0,
            }
        if path == "/jobs/j1":
            return {
                "job_id": "j1",
                "state": "completed",
                "steps": 12,
                "success": True,
                "state_before_sha": "before",
                "state_after_sha": "after",
                "episode_resets": 1,
                "env_instance_id": 7,
                "wall_s": 3.5,
                "rollout_path": "runs/j1.mp4",
            }
        if path == "/evaluate":
            return {"task_success": True, "oracle_source": "preauthored_fixture"}
        raise AssertionError("unexpected path %r" % path)

    return fake


class TestConditionTable(unittest.TestCase):
    def test_exact_conditions_and_budgets(self):
        self.assertEqual(set(R.CONDITIONS.keys()), set(EXPECTED_CONDITIONS.keys()))
        for name, expected in EXPECTED_CONDITIONS.items():
            scene_id, capability_ids, budget, case_id, execution_kind = expected
            condition = R.CONDITIONS[name]
            self.assertEqual(condition["scene_id"], scene_id, name)
            self.assertEqual(condition["capability_ids"], capability_ids, name)
            self.assertEqual(condition["budget_per_subgoal"], budget, name)
            self.assertEqual(condition["case_id"], case_id, name)
            self.assertEqual(condition["execution_kind"], execution_kind, name)

    def test_paired_conditions_share_budget_and_case(self):
        # direct vs forward vs reverse share the same preauthored final case.
        self.assertEqual(R.CONDITIONS["table_direct"]["case_id"], "table_tidy")
        self.assertEqual(R.CONDITIONS["table_forward"]["case_id"], "table_tidy")
        self.assertEqual(R.CONDITIONS["table_reverse"]["case_id"], "table_tidy")
        # balanced per-comparison budgets.
        self.assertEqual(R.CONDITIONS["table_direct"]["budget_per_subgoal"], 600)
        self.assertEqual(R.CONDITIONS["table_forward"]["budget_per_subgoal"], 300)
        self.assertEqual(R.CONDITIONS["mugs_direct"]["budget_per_subgoal"], 600)
        self.assertEqual(R.CONDITIONS["mugs_split"]["budget_per_subgoal"], 300)

    def test_unknown_condition_raises_value_error(self):
        with self.assertRaises(ValueError):
            R.run_trial("not_a_condition", 0, 0, 10.0)

    def test_expected_service_constants(self):
        self.assertEqual(R.SERVICE, "http://127.0.0.1:8767")
        self.assertEqual(R.WORKFLOW, "persistent_scene_v2")
        self.assertEqual(R.MODEL_REVISION, "6721902bc4d61e50a3bfdb11dfb4cb626f05d102")
        self.assertEqual(R.RATIONALE, "preauthored manual/direct capability audit")


class TestHttpLayer(unittest.TestCase):
    def test_opener_ignores_proxies(self):
        original = R._OPENER
        with mock.patch(
            "urllib.request.build_opener", return_value=original
        ) as constructor_mock, mock.patch(
            "urllib.request.getproxies",
            return_value={"http": "http://invalid.invalid:9"},
        ) as getproxies_mock:
            importlib.reload(R)
        constructor_mock.assert_called_once()
        argument = constructor_mock.call_args.args[0]
        self.assertIsInstance(argument, urllib.request.ProxyHandler)
        self.assertEqual(argument.proxies, {})
        getproxies_mock.assert_not_called()
        self.assertIs(R._OPENER, original)

    def test_http_json_sends_json_without_network(self):
        fake = FakeOpener()
        with mock.patch.object(R, "_OPENER", fake):
            result = R.http_json("POST", "/plans", {"a": 1})
        self.assertEqual(result, {"ok": True})
        request = fake.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.data, b'{"a": 1}')
        self.assertTrue(request.full_url.startswith(R.SERVICE + "/plans"))


class TestRunTrialContract(unittest.TestCase):
    def test_completed_trial_association_and_raw_preservation(self):
        calls = []
        with mock.patch.object(R, "http_json", make_completed_backend(calls)), mock.patch.object(
            R, "time", FakeTime()
        ):
            trial = R.run_trial("table_direct", 3, 11, timeout=100.0)

        self.assertTrue(trial["task_success"])
        self.assertTrue(trial["plan_success"])
        self.assertIsNone(trial["error"])
        self.assertEqual(trial["session_id"], "s1")
        self.assertEqual(trial["plan_id"], "p1")
        self.assertEqual(trial["job_ids"], ["j1"])
        self.assertEqual(trial["env_instance_id"], 7)
        self.assertEqual(trial["initial_images"][0]["sha256"], "deadbeef")
        self.assertEqual(len(trial["jobs"]), 1)
        self.assertEqual(trial["jobs"][0]["state_after_sha"], "after")
        self.assertEqual(trial["evaluation"]["oracle_source"], "preauthored_fixture")

        session_body = [b for (m, p, b) in calls if p == "/sessions"][0]
        plan_body = [b for (m, p, b) in calls if p == "/plans"][0]
        eval_body = [b for (m, p, b) in calls if p == "/evaluate"][0]
        self.assertEqual(session_body["scene_id"], "goal_table")
        self.assertEqual(session_body["seed"], 11)
        self.assertEqual(session_body["init_state_index"], 3)
        # exact preauthored request.
        self.assertEqual(plan_body["rationale"], R.RATIONALE)
        self.assertEqual(plan_body["decision"], "execute")
        self.assertIs(plan_body["audit"], True)
        self.assertEqual(plan_body["capability_ids"], ["table_both"])
        self.assertEqual(plan_body["budget_per_subgoal"], 600)
        self.assertEqual(plan_body["request_id"], trial["request_id"])
        # evaluation is bound to the SAME session and request id.
        self.assertEqual(eval_body["session_id"], plan_body["session_id"])
        self.assertEqual(eval_body["request_id"], plan_body["request_id"])
        self.assertEqual(eval_body["case_id"], "table_tidy")

    def test_new_session_per_trial_and_controls(self):
        calls = []
        backend = make_completed_backend(calls, sessions=("s1", "s2"))
        with mock.patch.object(R, "http_json", backend), mock.patch.object(R, "time", FakeTime()):
            first = R.run_trial("table_direct", 2, 5, timeout=100.0)
            second = R.run_trial("mugs_direct", 2, 5, timeout=100.0)

        session_calls = [b for (m, p, b) in calls if p == "/sessions"]
        self.assertEqual(len(session_calls), 2)
        self.assertNotEqual(first["session_id"], second["session_id"])
        # fair controls: same seed and init_state_index across matching conditions.
        for body in session_calls:
            self.assertEqual(body["seed"], 5)
            self.assertEqual(body["init_state_index"], 2)

    def test_blocked_plan_is_preserved_and_evaluated(self):
        calls = []

        def fake(method, path, body=None, timeout=60):
            calls.append((method, path, body))
            if path == "/sessions":
                return {"session_id": "s1", "scene_version": 0, "env_instance_id": 1, "images": []}
            if path == "/plans":
                return {"plan_id": "p1", "state": "queued", "job_ids": []}
            if path == "/plans/p1":
                return {"plan_id": "p1", "state": "blocked", "job_ids": ["j1"], "plan_success": False}
            if path == "/jobs/j1":
                return {"job_id": "j1", "state": "blocked", "success": False}
            if path == "/evaluate":
                return {"task_success": True, "oracle_source": "preauthored_fixture"}
            raise AssertionError(path)

        with mock.patch.object(R, "http_json", fake), mock.patch.object(R, "time", FakeTime()):
            trial = R.run_trial("table_forward", 0, 0, timeout=100.0)

        self.assertEqual(trial["plan"]["state"], "blocked")
        self.assertIs(trial["plan_success"], False)  # plan_success recorded separately
        self.assertIs(trial["task_success"], True)  # independent final state still evaluated
        self.assertTrue(any(p == "/evaluate" for (_, p, _b) in calls))

    def test_session_error_is_persisted_with_null_success(self):
        def fake(method, path, body=None, timeout=60):
            raise RuntimeError("boom")

        with mock.patch.object(R, "http_json", fake), mock.patch.object(R, "time", FakeTime()):
            trial = R.run_trial("table_direct", 0, 0, timeout=100.0)

        self.assertIsNone(trial["task_success"])
        self.assertIsNone(trial["session"])
        self.assertIn("session request failed", trial["error"])

        summary = R.summarize([trial])
        self.assertEqual(summary["conditions"]["table_direct"]["n"], 1)
        self.assertEqual(summary["conditions"]["table_direct"]["errors"], 1)
        self.assertEqual(summary["conditions"]["table_direct"]["evaluable"], 0)
        self.assertIsNone(summary["conditions"]["table_direct"]["wilson_95"])

    def _backend_with_evaluation(self, evaluation):
        def fake(method, path, body=None, timeout=60):
            if path == "/sessions":
                return {"session_id": "s1", "scene_version": 0, "env_instance_id": 1, "images": []}
            if path == "/plans":
                return {"plan_id": "p1", "state": "queued", "job_ids": []}
            if path == "/plans/p1":
                return {"plan_id": "p1", "state": "completed", "job_ids": [], "plan_success": True}
            if path == "/evaluate":
                return evaluation
            raise AssertionError(path)

        return fake

    def test_missing_oracle_source_is_not_success(self):
        backend = self._backend_with_evaluation({"task_success": True})
        with mock.patch.object(R, "http_json", backend), mock.patch.object(R, "time", FakeTime()):
            trial = R.run_trial("table_direct", 0, 0, timeout=100.0)
        self.assertIsNone(trial["task_success"])
        self.assertIn("oracle_source", trial["error"])

    def test_invalid_fixture_is_not_success(self):
        evaluation = {"ok": False, "reason": "invalid_fixture", "task_success": True}
        backend = self._backend_with_evaluation(evaluation)
        with mock.patch.object(R, "http_json", backend), mock.patch.object(R, "time", FakeTime()):
            trial = R.run_trial("table_direct", 0, 0, timeout=100.0)
        self.assertIsNone(trial["task_success"])
        self.assertIn("invalid_fixture", trial["error"])

    def test_non_boolean_task_success_is_not_success(self):
        evaluation = {"task_success": 1, "oracle_source": "preauthored_fixture"}
        backend = self._backend_with_evaluation(evaluation)
        with mock.patch.object(R, "http_json", backend), mock.patch.object(R, "time", FakeTime()):
            trial = R.run_trial("table_direct", 0, 0, timeout=100.0)
        self.assertIsNone(trial["task_success"])
        self.assertIn("not a boolean", trial["error"])

    def test_job_fetch_error_is_persisted(self):
        calls = []

        def fake(method, path, body=None, timeout=60):
            calls.append((method, path))
            if path == "/sessions":
                return {"session_id": "s1", "scene_version": 0, "env_instance_id": 1, "images": []}
            if path == "/plans":
                return {"plan_id": "p1", "state": "queued", "job_ids": []}
            if path == "/plans/p1":
                return {"plan_id": "p1", "state": "completed", "job_ids": ["j1"], "plan_success": True}
            if path == "/jobs/j1":
                raise RuntimeError("job gone")
            raise AssertionError(path)

        with mock.patch.object(R, "http_json", fake), mock.patch.object(R, "time", FakeTime()):
            trial = R.run_trial("table_direct", 0, 0, timeout=100.0)

        self.assertIsNone(trial["task_success"])
        self.assertIn("job j1 fetch failed", trial["error"])
        self.assertFalse(any(p == "/evaluate" for (_m, p) in calls))

    def test_deadline_cancels_without_success(self):
        calls = []
        fake_time = FakeTime()

        def fake(method, path, body=None, timeout=60):
            calls.append((method, path))
            if path == "/sessions":
                return {"session_id": "s1", "scene_version": 0, "env_instance_id": 1, "images": []}
            if path == "/plans":
                return {"plan_id": "p1", "state": "running", "job_ids": []}
            if path == "/plans/p1":
                return {"plan_id": "p1", "state": "running", "job_ids": []}
            if path == "/plans/p1/cancel":
                return {"ok": True, "state": "cancelled"}
            if path == "/evaluate":
                return {"task_success": True, "oracle_source": "preauthored_fixture"}
            raise AssertionError(path)

        with mock.patch.object(R, "http_json", fake), mock.patch.object(R, "time", fake_time):
            trial = R.run_trial("table_direct", 0, 0, timeout=5.0)

        self.assertIsNone(trial["task_success"])  # never a pass
        self.assertIn("deadline", trial["error"])
        self.assertIn(("POST", "/plans/p1/cancel"), calls)
        # even though a true evaluation is available, it is never consulted.
        self.assertFalse(any(p == "/evaluate" for (_m, p) in calls))


class TestHealthGate(unittest.TestCase):
    def _run_main(self, health, calls):
        def fake(method, path, body=None, timeout=60):
            calls.append((method, path))
            if path == "/health":
                return health
            raise AssertionError("runner proceeded past a failed health gate: %r" % path)

        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "report.json")
            with mock.patch.object(R, "http_json", fake):
                with mock.patch.object(sys, "argv", ["run_experiments.py", "--output", output]):
                    code = R.main()
            with open(output, "r", encoding="utf-8") as handle:
                return code, json.load(handle)

    def _good_health(self):
        return {
            "ready": True,
            "workflow": R.WORKFLOW,
            "model_revision": R.MODEL_REVISION,
        }

    def test_rejects_not_ready(self):
        health = self._good_health()
        health["ready"] = False
        calls = []
        code, report = self._run_main(health, calls)
        self.assertEqual(code, 1)
        self.assertTrue(report["is_error"])
        self.assertIn("health check failed", report["error"])
        self.assertEqual(report["trials"], [])
        self.assertFalse(any(p == "/sessions" for (_m, p) in calls))

    def test_rejects_wrong_workflow(self):
        health = self._good_health()
        health["workflow"] = "something_else"
        code, report = self._run_main(health, [])
        self.assertEqual(code, 1)
        self.assertIn("workflow mismatch", report["error"])

    def test_rejects_wrong_model_revision(self):
        health = self._good_health()
        health["model_revision"] = "0" * 40
        code, report = self._run_main(health, [])
        self.assertEqual(code, 1)
        self.assertIn("model_revision mismatch", report["error"])


class TestWilsonAndSummary(unittest.TestCase):
    def test_wilson_no_sample_is_none(self):
        self.assertIsNone(R.wilson(0, 0))
        self.assertIsNone(R.wilson(3, 0))

    def test_wilson_small_sample_bounds(self):
        interval = R.wilson(1, 1)
        self.assertIsNotNone(interval)
        lower, upper = interval
        self.assertLessEqual(0.0, lower)
        self.assertLessEqual(lower, upper)
        self.assertLessEqual(upper, 1.0)
        # the interval must not collapse to a point at n = 1.
        self.assertLess(lower, upper)
        self.assertAlmostEqual(R.wilson(0, 5)[0], 0.0, places=9)
        self.assertAlmostEqual(R.wilson(5, 5)[1], 1.0, places=9)

    def test_summarize_counts_and_denominator(self):
        trials = [
            {"condition": "table_direct", "task_success": True},
            {"condition": "table_direct", "task_success": False},
            {"condition": "table_direct", "task_success": None},
            {"condition": "table_forward", "task_success": None},
        ]
        summary = R.summarize(trials)
        direct = summary["conditions"]["table_direct"]
        self.assertEqual(direct["n"], 3)  # errored trial is not dropped
        self.assertEqual(direct["evaluable"], 2)
        self.assertEqual(direct["successes"], 1)
        self.assertEqual(direct["failures"], 1)
        self.assertEqual(direct["errors"], 1)
        self.assertIsNotNone(direct["wilson_95"])  # denominator is evaluable = 2

        forward = summary["conditions"]["table_forward"]
        self.assertEqual(forward["n"], 1)
        self.assertEqual(forward["evaluable"], 0)
        self.assertIsNone(forward["wilson_95"])  # no evaluable sample -> None

        self.assertEqual(summary["totals"]["n"], 4)
        self.assertEqual(summary["totals"]["evaluable"], 2)


if __name__ == "__main__":
    unittest.main()
