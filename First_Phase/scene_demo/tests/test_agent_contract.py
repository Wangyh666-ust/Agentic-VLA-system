# -*- coding: utf-8 -*-
"""Mock contracts for run_agent.py (持久场景 v2).

These tests are PURE MOCKS: they never call the real Hermes CLI, the real
service, the GPU or Git.  They pin the host-side contract of ``Runner`` and the
prompt builders:

  * the initial prompt carries public data only -- no case id and no evaluation
    standard, and a provided case_id value is never leaked into any prompt;
  * a request that never submits a plan attaches NO jobs/results and returns an
    explicit no-plan error;
  * polling always uses the EXACT request_id (never a "latest plan" fallback);
  * a normal run (queued -> completed) invokes the real model exactly once;
  * task_success is null unless an independent evaluation ran;
  * the total deadline cancels the same plan and reports execution_timeout;
  * a blocked plan is repaired exactly once (a second real model invocation);
  * a FINAL blocked plan (no repair budget, or repair cannot resume) is still
    eligible for one independent POST /evaluate and its task_success comes only
    from that evaluator -- while queued/running stay ineligible;
  * the runner verifies the fixed service identity (ready + workflow
    persistent_scene_v2 + the pinned model_revision) before reading any session
    or spawning Hermes, and fails honestly on any mismatch.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import run_agent  # noqa: E402

REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def make_session(**overrides):
    session = {
        "ok": True,
        "session_id": "sess-1",
        "scene_id": "goal_table",
        "state": "ready",
        "scene_version": 3,
        "env_instance_id": "env-1",
        "episode_resets": 0,
        "total_steps": 0,
        "storage_policy": {"akita_black_bowl_1": "plate_1"},
        "capabilities": [
            {
                "capability_id": "bowl_to_plate",
                "instruction": "put the bowl on the plate",
                "object_id": "akita_black_bowl_1",
                "target_id": "plate_1",
                "evidence": "candidate",
            }
        ],
        "images": [
            {"view": "agentview",
             "image_path": "/home/yhwang/fyp/scene_demo/runs/s/agentview.png",
             "kind": "png"},
            {"view": "wrist",
             "image_path": "/home/yhwang/fyp/scene_demo/runs/s/wrist.png",
             "kind": "png"},
        ],
        "latest_png": "/home/yhwang/fyp/scene_demo/runs/s/agentview.png",
        "run_dir": "/home/yhwang/fyp/scene_demo/runs/s",
        "active_request_id": None,
        "error": None,
    }
    session.update(overrides)
    return session


def make_plan(state, decision="execute", **overrides):
    plan = {
        "request_id": "req-123",
        "session_id": "sess-1",
        "state": state,
        "decision": decision,
        "capability_ids": ["bowl_to_plate"],
        "rationale": "整理桌面",
        "job_ids": [],
        "completed_capability_ids": [],
        "pending_capability_ids": [],
        "plan_success": None,
        "regressions": [],
        "error": None,
        "repair_history": [],
        "scene_version": 3,
    }
    plan.update(overrides)
    return plan


class FakeClock:
    """Deterministic monotonic clock: only time.sleep moves it forward."""

    def __init__(self, start=1000.0):
        self.t = start

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(0.0, float(seconds))


class FakeService:
    def __init__(self, session=None, plans=None, jobs=None, health=None,
                 cancelled_plan=None, evaluation=None):
        self.session = session if session is not None else make_session()
        self.health = health or {"ready": True, "workflow": "persistent_scene_v2",
                                 "model_revision": REVISION}
        self.plans = list(plans) if plans else []
        self._index = 0
        self.jobs = jobs or {}
        self.cancelled_plan = cancelled_plan
        self.evaluation = evaluation or {"task_success": True,
                                         "oracle_source": "preauthored_fixture"}
        self.plan_lookups = []
        self.cancel_calls = []
        self.evaluate_calls = []
        self.session_calls = []

    def get_health(self):
        return self.health

    def get_session(self, session_id):
        self.session_calls.append(session_id)
        return self.session

    def get_plan(self, request_id):
        self.plan_lookups.append(request_id)
        if not self.plans:
            return None
        if self._index < len(self.plans) - 1:
            plan = self.plans[self._index]
            self._index += 1
            return plan
        return self.plans[-1]

    def cancel_plan(self, request_id):
        self.cancel_calls.append(request_id)
        if self.cancelled_plan is not None:
            self.plans = [self.cancelled_plan]
        return {"ok": True}

    def get_job(self, job_id):
        return self.jobs.get(job_id, {
            "job_id": job_id, "state": "completed", "capability_id": "bowl_to_plate",
            "steps": 10, "total_steps": 300, "success": True, "rollout_path": None,
        })

    def evaluate(self, session_id, case_id, request_id):
        self.evaluate_calls.append((session_id, case_id, request_id))
        return dict(self.evaluation)


class FakeHermes:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, prompt, image_path, usage_path, timeout):
        self.calls.append({"prompt": prompt, "image": image_path, "timeout": timeout})
        try:
            with open(usage_path, "w", encoding="utf-8") as handle:
                json.dump({"model": "deepseek-flash", "is_error": False}, handle)
        except OSError:
            pass
        if self.result is not None:
            return dict(self.result)
        return {"exit_code": 0, "output": b"plan submitted", "timed_out": False,
                "error": None}


class Config:
    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id", "sess-1")
        self.request = kwargs.get("request", "请帮我整理一下桌面，把碗和酒瓶收好。")
        self.case_id = kwargs.get("case_id")
        self.request_id = kwargs.get("request_id", "req-123")
        self.timeout = kwargs.get("timeout", 30)
        self.max_repairs = kwargs.get("max_repairs", 1)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
class PromptContractTests(unittest.TestCase):
    def test_initial_prompt_excludes_case_and_evaluation_standards(self):
        session = make_session()
        prompt = run_agent.build_initial_prompt(session, "整理桌面", "req-1", ["/a.png"])
        for forbidden in ("case_id", "case-id", "task_success", "evaluate",
                          "evaluation", "oracle", "fixture", "评测", "标准"):
            self.assertNotIn(forbidden, prompt, "initial prompt leaked %r" % forbidden)
        self.assertIn("phase", prompt)
        self.assertIn("initial", prompt)
        self.assertIn("req-1", prompt)
        self.assertIn("scene_version", prompt)
        self.assertIn("整理桌面", prompt)
        self.assertIn("/a.png", prompt)

    def test_initial_prompt_never_contains_case_id_value(self):
        service = FakeService(session=make_session(),
                              plans=[make_plan("completed")])
        hermes = FakeHermes()
        runner = self._runner(service, hermes, case_id="secret_case_42")
        runner.run()
        self.assertTrue(hermes.calls)
        self.assertNotIn("secret_case_42", hermes.calls[0]["prompt"])

    def test_repair_prompt_has_no_oracle_or_fixture(self):
        session = make_session()
        plan = make_plan("blocked", error="capability failed",
                         pending_capability_ids=["bowl_to_plate"])
        prompt = run_agent.build_repair_prompt(session, "整理桌面", "req-1", plan, ["/b.png"])
        for forbidden in ("oracle", "fixture", "case_id"):
            self.assertNotIn(forbidden, prompt)
        self.assertIn("blocked", prompt)
        self.assertIn("pending_capability_ids", prompt)

    def _runner(self, service, hermes, case_id=None, timeout=30, max_repairs=1,
                request_id="req-123"):
        run_dir = tempfile.mkdtemp()
        config = Config(case_id=case_id, timeout=timeout, max_repairs=max_repairs,
                        request_id=request_id)
        return run_agent.Runner(config, service, hermes, FakeClock(), run_dir)


class RunnerContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _runner(self, service, hermes, case_id=None, timeout=30, max_repairs=1,
                request_id="req-123", session_id="sess-1"):
        run_dir = tempfile.mkdtemp(dir=self.tmp)
        config = Config(session_id=session_id, case_id=case_id, timeout=timeout,
                        max_repairs=max_repairs, request_id=request_id)
        return run_agent.Runner(config, service, hermes, FakeClock(), run_dir)

    def test_no_submitted_plan_never_attaches_old_results(self):
        service = FakeService(session=make_session(), plans=[])
        service.jobs = {"old_job": {"job_id": "old_job", "state": "completed",
                                    "success": True}}
        hermes = FakeHermes()
        result = self._runner(service, hermes, timeout=5).run()

        self.assertIsNone(result["plan"])
        self.assertEqual(result["jobs"], [])
        self.assertIsNone(result["task_success"])
        self.assertIsNone(result["evaluation"])
        self.assertIn("no_plan_submitted", result["error"] or "")
        self.assertEqual(service.evaluate_calls, [])
        self.assertEqual(result["hermes_invocations"], 1)
        self.assertFalse(result["run_ok"] and result["chain_ok"])

    def test_polling_uses_exact_request_id(self):
        service = FakeService(session=make_session(),
                              plans=[make_plan("running"), make_plan("completed")])
        hermes = FakeHermes()
        result = self._runner(service, hermes, request_id="req-exact").run()

        self.assertTrue(service.plan_lookups)
        self.assertEqual(set(service.plan_lookups), {"req-exact"})
        self.assertEqual(result["request_id"], "req-exact")
        self.assertEqual(result["plan"]["state"], "completed")

    def test_normal_polling_invokes_model_once(self):
        service = FakeService(session=make_session(),
                              plans=[make_plan("running"), make_plan("completed")])
        hermes = FakeHermes()
        result = self._runner(service, hermes).run()

        self.assertEqual(len(hermes.calls), 1)
        self.assertEqual(result["hermes_invocations"], 1)
        self.assertEqual(service.cancel_calls, [])
        self.assertTrue(result["run_ok"])
        self.assertTrue(result["chain_ok"])

    def test_task_success_null_without_evaluation(self):
        service = FakeService(session=make_session(), plans=[make_plan("completed")])
        hermes = FakeHermes()
        result = self._runner(service, hermes, case_id=None).run()

        self.assertIsNone(result["task_success"])
        self.assertIsNone(result["evaluation"])
        self.assertEqual(service.evaluate_calls, [])

    def test_task_success_from_independent_evaluation_only(self):
        service = FakeService(session=make_session(),
                              plans=[make_plan("completed", decision="execute")])
        hermes = FakeHermes()
        result = self._runner(service, hermes, case_id="mugs_standard").run()

        self.assertEqual(service.evaluate_calls, [("sess-1", "mugs_standard", "req-123")])
        self.assertTrue(result["task_success"])
        self.assertEqual(result["evaluation"]["oracle_source"], "preauthored_fixture")

    def test_total_deadline_cancel_semantics(self):
        cancelled = make_plan("cancelled")
        service = FakeService(session=make_session(), plans=[make_plan("running")],
                              cancelled_plan=cancelled)
        hermes = FakeHermes()
        result = self._runner(service, hermes, timeout=6).run()

        self.assertTrue(result["execution_timeout"])
        self.assertEqual(service.cancel_calls, ["req-123"])
        self.assertEqual(result["plan"]["state"], "cancelled")
        self.assertFalse(result["chain_ok"])
        self.assertEqual(set(service.plan_lookups), {"req-123"})
        self.assertEqual(result["hermes_invocations"], 1)

    def test_blocked_plan_repaired_exactly_once(self):
        service = FakeService(
            session=make_session(),
            plans=[make_plan("blocked", error="failed"),
                   make_plan("running"), make_plan("completed")],
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes, max_repairs=1).run()

        self.assertEqual(len(hermes.calls), 2)      # initial + one repair
        self.assertEqual(result["hermes_invocations"], 2)
        self.assertEqual(result["plan"]["state"], "completed")

    def test_blocked_plan_without_repair_budget_returns_blocked(self):
        service = FakeService(session=make_session(), plans=[make_plan("blocked")])
        hermes = FakeHermes()
        result = self._runner(service, hermes, max_repairs=0).run()

        self.assertEqual(len(hermes.calls), 1)
        self.assertEqual(result["plan"]["state"], "blocked")
        self.assertFalse(result["chain_ok"])

    def test_session_not_ready_stops_before_hermes(self):
        service = FakeService(session=make_session(state="closed"), plans=[])
        hermes = FakeHermes()
        result = self._runner(service, hermes).run()

        self.assertEqual(len(hermes.calls), 0)
        self.assertEqual(result["hermes_invocations"], 0)
        self.assertIsNone(result["plan"])

    # ---- final blocked independent evaluation ----------------------------- #
    def test_final_blocked_with_case_id_evaluates_once_with_oracle_false(self):
        service = FakeService(
            session=make_session(),
            plans=[make_plan("blocked", error="unrecoverable")],
            evaluation={"task_success": False, "oracle_source": "preauthored_fixture"},
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes, case_id="mugs_standard",
                              max_repairs=0).run()

        self.assertEqual(result["plan"]["state"], "blocked")
        self.assertEqual(service.evaluate_calls,
                         [("sess-1", "mugs_standard", "req-123")])
        self.assertFalse(result["task_success"])
        self.assertFalse(result["chain_ok"])
        self.assertEqual(result["evaluation"]["oracle_source"], "preauthored_fixture")
        self.assertEqual(len(hermes.calls), 1)              # initial only, no repair
        self.assertEqual(result["hermes_invocations"], 1)

    def test_final_blocked_after_repair_cannot_resume_evaluates_once(self):
        service = FakeService(
            session=make_session(),
            plans=[make_plan("blocked", error="failed"),
                   make_plan("blocked", error="still failed")],
            evaluation={"task_success": False, "oracle_source": "preauthored_fixture"},
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes, case_id="mugs_standard",
                              max_repairs=1).run()

        self.assertEqual(result["plan"]["state"], "blocked")
        self.assertEqual(len(hermes.calls), 2)              # initial + one repair
        self.assertEqual(service.evaluate_calls,
                         [("sess-1", "mugs_standard", "req-123")])
        self.assertFalse(result["task_success"])
        self.assertFalse(result["chain_ok"])

    def test_blocked_is_not_terminal_but_is_evaluable(self):
        self.assertFalse(run_agent.is_terminal("blocked"))
        self.assertTrue(run_agent.is_evaluable("blocked"))
        for state in ("completed", "error", "cancelled"):
            self.assertTrue(run_agent.is_terminal(state))
            self.assertTrue(run_agent.is_evaluable(state))

    def test_queued_and_running_never_call_evaluator(self):
        for state in ("queued", "running"):
            self.assertFalse(run_agent.is_evaluable(state))
            service = FakeService(session=make_session(), plans=[make_plan(state)])
            hermes = FakeHermes()
            runner = self._runner(service, hermes, case_id="mugs_standard")
            # Direct eligibility check: the evaluator must never be called.
            self.assertIsNone(runner._evaluate(make_plan(state)))
            self.assertEqual(service.evaluate_calls, [])


class HealthIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _runner(self, service, hermes, case_id=None, request_id="req-123"):
        run_dir = tempfile.mkdtemp(dir=self.tmp)
        config = Config(case_id=case_id, timeout=30, max_repairs=1,
                        request_id=request_id)
        return run_agent.Runner(config, service, hermes, FakeClock(), run_dir)

    def _assert_honest_stop(self, result, service, hermes, needle):
        self.assertEqual(len(hermes.calls), 0)
        self.assertEqual(result["hermes_invocations"], 0)
        self.assertEqual(service.session_calls, [])          # no session read
        self.assertEqual(service.plan_lookups, [])           # no plan polling
        self.assertEqual(service.evaluate_calls, [])         # no evaluation
        self.assertIsNone(result["plan"])
        self.assertIn(needle, result["error"] or "")

    def test_valid_health_identity_is_accepted(self):
        service = FakeService(
            session=make_session(),
            plans=[make_plan("completed")],
            health={"ready": True, "workflow": "persistent_scene_v2",
                    "model_revision": REVISION},
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes).run()

        self.assertTrue(hermes.calls)
        self.assertIsNone(result["error"])
        self.assertTrue(result["chain_ok"])

    def test_wrong_workflow_stops_before_model(self):
        service = FakeService(
            session=make_session(),
            plans=[make_plan("completed")],
            health={"ready": True, "workflow": "legacy_scene_v1",
                    "model_revision": REVISION},
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes, case_id="mugs_standard").run()

        self._assert_honest_stop(result, service, hermes, "workflow")
        self.assertIn("persistent_scene_v2", result["error"] or "")

    def test_wrong_revision_stops_before_model(self):
        service = FakeService(
            session=make_session(),
            plans=[make_plan("completed")],
            health={"ready": True, "workflow": "persistent_scene_v2",
                    "model_revision": "0" * 40},
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes, case_id="mugs_standard").run()

        self._assert_honest_stop(result, service, hermes, "model_revision")
        self.assertIn(REVISION, result["error"] or "")

    def test_not_ready_stops_before_model(self):
        service = FakeService(
            session=make_session(),
            plans=[make_plan("completed")],
            health={"ready": False, "workflow": "persistent_scene_v2",
                    "model_revision": REVISION},
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes, case_id="mugs_standard").run()

        self._assert_honest_stop(result, service, hermes, "ready")


if __name__ == "__main__":
    unittest.main()
