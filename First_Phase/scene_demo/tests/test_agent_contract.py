# -*- coding: utf-8 -*-
"""Mock contracts for run_agent.py (持久场景 v2).

These tests are PURE MOCKS: they never call the real Hermes CLI, the real
service, the GPU or Git.  They pin the host-side contract of ``Runner`` and the
prompt builders:

  * the initial prompt carries public data only -- no case id and no evaluation
    standard, and a provided case_id value is never leaked into any prompt;
  * both phase prompts (and the isolated profile's SOUL) carry IDENTICAL lazy
    MCP routing guidance (tool_search -> tool_describe -> tool_call) plus the
    native-agentview image-efficiency guidance;
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
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import run_agent  # noqa: E402
import setup_profile  # noqa: E402  (pure module: constants only, no side effects)

REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def _prompt_payload(prompt, head):
    """Parse the JSON data section appended after a prompt-head constant."""
    if not prompt.startswith(head):
        raise AssertionError("prompt does not start with the expected head")
    return json.loads(prompt[len(head):])


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

    # ---- real per-job execution evidence --------------------------------- #
    def test_repair_prompt_carries_actual_steps_and_ended_reason(self):
        """The repair payload must expose the REAL execution fields (steps /
        total_steps / ended_reason / ...) of this plan's jobs, so the model can
        reason about a budget_exhausted subgoal instead of guessing."""
        session = make_session()
        plan = make_plan("blocked", job_ids=["job-1"])
        jobs = [{
            "job_id": "job-1",
            "request_id": "req-1",
            "session_id": "sess-1",
            "capability_id": "bowl_to_plate",
            "state": "error",
            "steps": 137,
            "total_steps": 300,
            "success": False,
            "ended_reason": "budget_exhausted",
            "error": "budget_exhausted after 300 steps",
            "wall_s": 12.5,
            "scene_version_before": 3,
            "scene_version_after": 3,
        }]
        prompt = run_agent.build_repair_prompt(session, "整理桌面", "req-1", plan,
                                               ["/b.png"], jobs)
        evidence = _prompt_payload(prompt, run_agent.REPAIR_PROMPT_HEAD)["execution_evidence"]

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["job_id"], "job-1")
        self.assertEqual(evidence[0]["capability_id"], "bowl_to_plate")
        self.assertEqual(evidence[0]["steps"], 137)
        self.assertEqual(evidence[0]["total_steps"], 300)
        self.assertEqual(evidence[0]["ended_reason"], "budget_exhausted")
        self.assertIs(evidence[0]["success"], False)
        self.assertEqual(evidence[0]["wall_s"], 12.5)

    def test_repair_prompt_filters_foreign_jobs_and_never_copies_oracle_truth(self):
        """Wrong job / request / session are dropped; a retrieval-error entry
        with only an allowed job_id+error is kept; embedded evaluation / oracle /
        fixture / case fields are NEVER copied into the public evidence."""
        session = make_session()  # session_id == "sess-1"
        plan = make_plan("blocked", job_ids=["job-1"])
        jobs = [
            {"job_id": "job-1", "request_id": "req-1", "session_id": "sess-1",
             "state": "error", "steps": 10, "ended_reason": "budget_exhausted",
             "evaluation": {"task_success": True},
             "oracle_source": "preauthored_fixture",
             "fixtures": {"secret": 1}, "case_id": "secret_case"},
            {"job_id": "foreign-job", "request_id": "req-1", "session_id": "sess-1",
             "steps": 99},
            {"job_id": "job-1", "request_id": "other-request", "session_id": "sess-1",
             "steps": 88},
            {"job_id": "job-1", "request_id": "req-1", "session_id": "other-session",
             "steps": 77},
            {"job_id": "job-1", "error": "GET /jobs/job-1 failed"},
            "not-a-dict",
        ]
        prompt = run_agent.build_repair_prompt(session, "整理桌面", "req-1", plan,
                                               ["/b.png"], jobs)
        evidence = _prompt_payload(prompt, run_agent.REPAIR_PROMPT_HEAD)["execution_evidence"]

        # Only the in-plan, identity-matching job plus the retrieval-error entry.
        self.assertEqual([entry["job_id"] for entry in evidence], ["job-1", "job-1"])
        self.assertEqual(evidence[0]["steps"], 10)
        self.assertEqual(evidence[0]["ended_reason"], "budget_exhausted")
        self.assertEqual(evidence[1]["error"], "GET /jobs/job-1 failed")
        # Absent values are not invented for the error entry.
        self.assertNotIn("steps", evidence[1])
        self.assertNotIn("state", evidence[1])
        # Whitelist filtering: evaluation truth never appears anywhere.
        for entry in evidence:
            for leaked in ("evaluation", "oracle_source", "fixtures", "case_id"):
                self.assertNotIn(leaked, entry)
        for forbidden in ("evaluation", "oracle", "fixture", "case_id",
                          "task_success", "secret_case"):
            self.assertNotIn(forbidden, prompt, "repair prompt leaked %r" % forbidden)

    def test_repair_prompt_has_empty_evidence_when_no_jobs_supplied(self):
        """Old 5-argument calls stay compatible: execution_evidence is present
        but empty, and nothing extra is invented."""
        session = make_session()
        plan = make_plan("blocked")
        prompt = run_agent.build_repair_prompt(session, "整理桌面", "req-1", plan,
                                               ["/b.png"])
        payload = _prompt_payload(prompt, run_agent.REPAIR_PROMPT_HEAD)
        self.assertEqual(payload["execution_evidence"], [])

    def test_both_phase_heads_state_separate_per_subgoal_budget(self):
        """Both prompts must state the MCP gives EACH subgoal its OWN 300-step
        budget (not one shared budget), and the repair head must explain
        budget_exhausted without inventing a larger budget."""
        session = make_session()
        plan = make_plan("blocked")
        initial = run_agent.build_initial_prompt(session, "整理桌面", "req-1", ["/a.png"])
        repair = run_agent.build_repair_prompt(session, "整理桌面", "req-1", plan, ["/b.png"])

        for prompt in (initial, repair):
            self.assertIn("300", prompt)
            self.assertIn("subgoal", prompt)
            self.assertIn("per subgoal", prompt)
        self.assertIn("budget_exhausted", repair)
        self.assertIn("共享", repair)
        self.assertIn("completed", repair)

    # ---- lazy MCP tool routing ------------------------------------------- #
    def test_both_prompts_and_soul_carry_identical_lazy_tool_routing(self):
        """Initial + repair prompts (and the isolated SOUL) must carry the SAME
        lazy tool_call routing guidance: the installed Hermes may expose only
        tool_search / tool_describe / tool_call, so a non-directly-callable MCP
        method is reached via tool_search -> tool_describe -> tool_call, never by
        re-attempting a discovered MCP name as a direct function.

        Pure-function check: only the prompt builders and the setup_profile
        constants are read -- no runner, model, service or Hermes call.
        """
        session = make_session()
        plan = make_plan("blocked", error="failed")
        initial = run_agent.build_initial_prompt(session, "整理桌面", "req-1", ["/a.png"])
        repair = run_agent.build_repair_prompt(session, "整理桌面", "req-1", plan, ["/b.png"])

        guidance = run_agent.MCP_ROUTING_GUIDANCE
        # Identical text in the isolated profile's SOUL as in the host prompts.
        self.assertEqual(setup_profile.MCP_ROUTING_GUIDANCE, guidance)
        for prompt in (initial, repair):
            self.assertIn(guidance, prompt)
        self.assertIn(guidance, setup_profile.SOUL_MD)

        # Spell the routing contract out, not merely name it.
        call_template = ('tool_call(calls=[{"name": "mcp__scene_tools__<method>", '
                         '"arguments": {...}}])')
        for prompt in (initial, repair):
            self.assertIn("tool_search", prompt)
            self.assertIn("tool_describe", prompt)
            self.assertIn("tool_call", prompt)
            self.assertIn(call_template, prompt)
            self.assertIn("never repeatedly attempt a discovered MCP name", prompt)
            self.assertIn("Only call functions currently exposed as callable", prompt)

    def test_both_prompts_and_soul_carry_image_efficiency_guidance(self):
        """The native agentview image is already attached: reuse it, do not
        repeat vision_analyze when it suffices; only observe_scene extra views
        when the imagery is insufficient, and actually inspect them.  Checking
        the latest session_version and submitting a plan stay mandatory."""
        session = make_session()
        plan = make_plan("blocked", error="failed")
        initial = run_agent.build_initial_prompt(session, "整理桌面", "req-1", ["/a.png"])
        repair = run_agent.build_repair_prompt(session, "整理桌面", "req-1", plan, ["/b.png"])

        guidance = run_agent.IMAGE_EFFICIENCY_GUIDANCE
        self.assertEqual(setup_profile.IMAGE_EFFICIENCY_GUIDANCE, guidance)
        for prompt in (initial, repair):
            self.assertIn(guidance, prompt)
            self.assertIn("do NOT repeat vision_analyze", prompt)
            self.assertIn("observe_scene(extra_views=true)", prompt)
        self.assertIn(guidance, setup_profile.SOUL_MD)

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

    def test_repair_runner_prompt_includes_exact_executed_job_summaries(self):
        """The SECOND real Hermes call (repair) must carry the exact executed
        job summaries of THIS request, collected via ServiceClient.get_job, and
        never any evaluation/oracle truth."""
        blocked = make_plan("blocked", error="failed", job_ids=["job-1"])
        service = FakeService(
            session=make_session(),
            plans=[blocked, make_plan("running"), make_plan("completed")],
            jobs={"job-1": {
                "job_id": "job-1", "request_id": "req-123", "session_id": "sess-1",
                "capability_id": "bowl_to_plate", "state": "error", "steps": 137,
                "total_steps": 300, "success": False,
                "ended_reason": "budget_exhausted", "wall_s": 12.5,
                "evaluation": {"task_success": True},
                "oracle_source": "preauthored_fixture",
            }},
        )
        hermes = FakeHermes()
        result = self._runner(service, hermes, max_repairs=1).run()

        self.assertEqual(len(hermes.calls), 2)          # initial + one repair
        self.assertEqual(result["hermes_invocations"], 2)
        payload = _prompt_payload(hermes.calls[1]["prompt"], run_agent.REPAIR_PROMPT_HEAD)
        evidence = payload["execution_evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["job_id"], "job-1")
        self.assertEqual(evidence[0]["steps"], 137)
        self.assertEqual(evidence[0]["total_steps"], 300)
        self.assertEqual(evidence[0]["ended_reason"], "budget_exhausted")
        self.assertEqual(evidence[0]["capability_id"], "bowl_to_plate")
        self.assertNotIn("evaluation", evidence[0])
        self.assertNotIn("oracle_source", hermes.calls[1]["prompt"])

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


class HermesCommandContractTests(unittest.TestCase):
    """Pin the EXACT installed-Hermes CLI invocation and honest usage reporting.

    ``subprocess.run`` is the ONLY mocked boundary: no real Hermes binary, model
    or service is ever executed.  These tests prove the native-image invocation
    targets the installed ``chat`` subcommand, that ``--usage-file`` is a
    top-level option ordered BEFORE ``chat``, that the image is attached via the
    chat ``--image`` option (the old top-level ``-z`` is gone), that the literal
    prompt stays exactly one unchanged argument, and that a missing native-image
    usage file is reported honestly (nulls) instead of faked counts.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # A real (but never executed) file so HermesRunner's isfile() guard passes.
        self.bin_path = os.path.join(self.tmp, "hermes")
        with open(self.bin_path, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n")
        self.usage_path = os.path.join(self.tmp, "usage_initial.json")
        self.image_path = "/home/yhwang/fyp/scene_demo/runs/s/agentview.png"
        self.prompt = "literal prompt: 整理桌面，把碗放好 (do not mutate)"

    def _runner(self):
        return run_agent.HermesRunner(
            bin_path=self.bin_path, home=os.path.join(self.tmp, "hermes_home"),
            cwd=self.tmp, tools="scene_tools,vision")

    def _call(self, runner, image_path, returncode=0, stdout=b"plan submitted"):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=returncode, stdout=stdout)

        with mock.patch("run_agent.subprocess.run", side_effect=fake_run) as patched:
            result = runner(self.prompt, image_path, self.usage_path, 42)
        captured["mock"] = patched
        return captured, result

    # ---- exact argument contract ----------------------------------------- #
    def test_chat_command_exact_argument_order_with_image(self):
        captured, result = self._call(self._runner(), self.image_path)
        cmd = captured["cmd"]
        self.assertEqual(cmd, [
            self.bin_path,
            "--usage-file", self.usage_path,      # top-level, BEFORE chat
            "chat", "--cli", "--oneshot", "-Q",
            "-t", "scene_tools,vision",
            "-q", self.prompt,                    # literal prompt, one argument
            "--image", self.image_path,
        ])
        self.assertNotIn("-z", cmd)
        self.assertEqual(cmd.count(self.prompt), 1)
        self.assertEqual(cmd[cmd.index("-q") + 1], self.prompt)
        self.assertLess(cmd.index("--usage-file"), cmd.index("chat"))
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])

    def test_chat_command_exact_argument_order_without_image(self):
        captured, _ = self._call(self._runner(), None)
        cmd = captured["cmd"]
        self.assertEqual(cmd, [
            self.bin_path,
            "--usage-file", self.usage_path,
            "chat", "--cli", "--oneshot", "-Q",
            "-t", "scene_tools,vision",
            "-q", self.prompt,
        ])
        self.assertNotIn("--image", cmd)
        self.assertNotIn("-z", cmd)
        self.assertLess(cmd.index("--usage-file"), cmd.index("chat"))

    def test_chat_flags_present_and_quiet_ordering(self):
        captured, _ = self._call(self._runner(), self.image_path)
        cmd = captured["cmd"]
        for flag in ("chat", "--cli", "--oneshot", "-Q"):
            self.assertIn(flag, cmd)
        # tools precede the query; the query precedes any image.
        self.assertLess(cmd.index("-t"), cmd.index("-q"))
        self.assertLess(cmd.index("-q"), cmd.index("--image"))
        # subprocess boundary untouched: env/cwd/stdout/error handling unchanged.
        self.assertEqual(captured["kwargs"]["cwd"], self.tmp)
        self.assertEqual(captured["kwargs"]["timeout"], 42)
        self.assertNotIn("http_proxy", captured["kwargs"]["env"])
        self.assertEqual(captured["kwargs"]["env"]["HERMES_HOME"],
                         os.path.join(self.tmp, "hermes_home"))

    # ---- honest usage reporting ------------------------------------------ #
    def test_missing_usage_reports_nulls_not_fake_counts(self):
        self.assertFalse(os.path.exists(self.usage_path))
        _, result = self._call(self._runner(), self.image_path, returncode=0)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(os.path.isfile(self.usage_path))
        with open(self.usage_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertIs(data["available"], False)
        self.assertEqual(
            data["reason"],
            "installed Hermes chat native-image path did not export usage")
        self.assertIsNone(data["api_calls"])
        self.assertIsNone(data["input_tokens"])
        self.assertIsNone(data["output_tokens"])
        # Never invent zero calls / zero cost.
        self.assertNotEqual(data["api_calls"], 0)
        self.assertNotEqual(data["input_tokens"], 0)
        self.assertNotEqual(data["output_tokens"], 0)

    def test_genuine_existing_usage_is_preserved(self):
        genuine = {"model": "deepseek-flash", "is_error": False,
                   "api_calls": 3, "input_tokens": 111, "output_tokens": 222}
        with open(self.usage_path, "w", encoding="utf-8") as handle:
            json.dump(genuine, handle)
        _, result = self._call(self._runner(), self.image_path, returncode=0)
        self.assertEqual(result["exit_code"], 0)
        with open(self.usage_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data, genuine)

    def test_failed_process_does_not_create_success_like_metadata(self):
        _, result = self._call(self._runner(), self.image_path, returncode=1,
                               stdout=b"agentview.png is not a hermes command")
        self.assertEqual(result["exit_code"], 1)
        self.assertFalse(os.path.exists(self.usage_path))

    def test_runner_collects_honest_metadata_into_usages(self):
        """The existing runner keeps collecting whatever usage file exists."""
        service = FakeService(session=make_session(), plans=[make_plan("completed")])
        run_dir = tempfile.mkdtemp(dir=self.tmp)
        config = Config(case_id=None, timeout=30, max_repairs=1, request_id="req-123")
        hermes = run_agent.HermesRunner(
            bin_path=self.bin_path, home=os.path.join(self.tmp, "home"),
            cwd=self.tmp, tools="scene_tools,vision")

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stdout=b"plan submitted")

        with mock.patch("run_agent.subprocess.run", side_effect=fake_run):
            result = run_agent.Runner(
                config, service, hermes, FakeClock(), run_dir).run()

        self.assertEqual(result["hermes_invocations"], 1)
        self.assertEqual(len(result["usage"]), 1)
        usage = result["usage"][0]
        self.assertIs(usage["available"], False)
        self.assertIsNone(usage["api_calls"])


if __name__ == "__main__":
    unittest.main()
