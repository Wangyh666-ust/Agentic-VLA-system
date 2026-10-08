#!/usr/bin/env python3
"""GPU-free contract tests for ``scene_demo/grasp_assist_service.py``.

These tests never load a checkpoint, never build a LIBERO scene, never touch
CUDA and never open a socket.  They exercise, with an injected fake simulator and
the REAL ``local_grasp`` module (its public interface is never stubbed away):

* the fixed pilot budget override for ``execute`` plans (and only for them),
  with the caller's payload left byte-for-byte unchanged;
* the isolation of a non-wine capability: the original VLA action is returned
  unchanged and every recorded source is ``vla``;
* a far/idle wine assistant that also leaves the VLA action unchanged;
* the wine trigger handoff (auxiliary action replaces the VLA proposal, the
  stale VLA raw/input evidence is cleared, the queues reset exactly once);
* one real ``env.step`` and one ``helper.observe_after`` per actual local action,
  with the honest local/VLA action counts;
* the confirmed return to VLA (queues reset exactly once) ;
* a physical assist failure raised BEFORE the next action;
* an unreadable geometry raised honestly as ``local_grasp_unknown``;
* cooperative cancellation still winning, using the existing fake-env pattern.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import grasp_assist_service as gas  # noqa: E402
import local_grasp  # noqa: E402
import service  # noqa: E402

WINE_GOAL_KEY = "on|wine_bottle_1|wine_rack_1_top_region"
BOWL_GOAL_KEY = "on|akita_black_bowl_1|plate_1"


# --- fake simulator (mirrors the accepted contract-test fixtures) ------------


class _AssistObject:
    def __init__(self, object_id):
        self.joints = ["%s_joint0" % object_id]
        self.contact_geoms = object_id


class _AssistSimData:
    def __init__(self):
        self.qpos = np.zeros(7, dtype=np.float64)
        self.qvel = np.zeros(7, dtype=np.float64)
        self.body_xpos = np.zeros((8, 3), dtype=np.float64)
        self.joint_qpos: dict[str, np.ndarray] = {}

    def get_joint_qpos(self, name):
        return self.joint_qpos.setdefault(name, np.zeros(7, dtype=np.float64))

    def set_joint_qpos(self, name, value):
        self.joint_qpos[name] = np.asarray(value, dtype=np.float64).copy()

    def get_joint_qvel(self, name):  # noqa: ARG002
        return np.zeros(6, dtype=np.float64)


class _FakeController:
    """A live OSC delta-controller stand-in satisfying the fixed contract."""

    def __init__(self):
        self.use_delta = True
        self.control_dim = 6
        self.input_min = [-1.0] * 6
        self.input_max = [1.0] * 6
        self.output_min = [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
        self.output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        self.ee_pos = [0.0, 0.0, 0.0]
        self.ee_ori_mat = np.eye(3, dtype=np.float64)

    def update(self, force=False):  # noqa: ARG002
        pass


class _AssistInnerEnv:
    OBJECT_IDS = (
        "akita_black_bowl_1",
        "plate_1",
        "wine_bottle_1",
        "wine_rack_1_top_region",
        "cream_cheese_1",
    )

    def __init__(self, truth_after=None):
        self.sim = types.SimpleNamespace(
            data=_AssistSimData(),
            model=types.SimpleNamespace(camera_names=[], get_xml=lambda: "<xml/>"),
        )
        self.steps = 0
        self.truth_after = dict(truth_after or {})
        self.objects_dict = {oid: _AssistObject(oid) for oid in self.OBJECT_IDS}
        self.object_states_dict = {}
        self.obj_body_id = {oid: i for i, oid in enumerate(self.OBJECT_IDS)}
        self.robots = [
            types.SimpleNamespace(
                gripper=types.SimpleNamespace(), controller=_FakeController()
            )
        ]

    def _get_observations(self, force_update=False):  # noqa: ARG002
        return {}

    def _check_grasp(self, gripper, geoms):  # noqa: ARG002
        return False

    def _eval_predicate(self, predicate):
        key = "|".join(str(part) for part in predicate)
        threshold = self.truth_after.get(key)
        if threshold is None:
            return False
        return self.steps >= threshold

    def step(self, action):  # noqa: ARG002
        self.steps += 1
        self.sim.data.qpos[0] = float(self.steps)
        return {}


class _AssistEnv:
    def __init__(self, inner, render_counter=None):
        self._inner = inner
        self._env = types.SimpleNamespace(env=inner)
        self.action_space = types.SimpleNamespace(low=-np.ones(7), high=np.ones(7))
        self.closed = False
        self._render_counter = render_counter

    def reset(self, seed=0):  # noqa: ARG002
        return types.SimpleNamespace(pixels={}, robot_state={}), {}

    def step(self, action):
        self._inner.step(action)
        return (
            types.SimpleNamespace(pixels={}, robot_state={}),
            0.0,
            False,
            False,
            {"is_success": False},
        )

    def render(self):
        if self._render_counter is not None:
            self._render_counter["n"] += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class _ScriptedController:
    """Deterministic test double for the REAL ``local_grasp`` controller.

    It mirrors the exact public interface (``phase`` / ``reason`` /
    ``total_actions`` / ``next_action`` / ``observe_after`` / ``summary``); the
    production module itself is never replaced.
    """

    def __init__(self, arm=True, observe_phases=(), fail_reason="wine_translation_drift"):
        self.phase = local_grasp.IDLE
        self.reason = None
        self.total_actions = 0
        self._arm = bool(arm)
        self._observe_phases = list(observe_phases)
        self._fail_reason = fail_reason
        self.next_calls = 0
        self.observe_calls = 0

    def next_action(self, reading, proposed_action=None):  # noqa: ARG002
        self.next_calls += 1
        if self.phase == local_grasp.FAILED:
            return None
        if self.phase == local_grasp.IDLE:
            if not self._arm:
                return None
            self.phase = local_grasp.ABOVE
            return np.full(7, 0.10, dtype=np.float32)
        if self.phase in local_grasp.STAGE_ORDER:
            return np.full(7, 0.20, dtype=np.float32)
        return None

    def observe_after(self, reading):  # noqa: ARG002
        self.total_actions += 1
        self.observe_calls += 1
        if self._observe_phases:
            nxt = self._observe_phases.pop(0)
            if nxt == local_grasp.FAILED:
                self.reason = self._fail_reason
                self.phase = local_grasp.FAILED
            else:
                self.phase = nxt
                if nxt == local_grasp.CONFIRMED:
                    self.reason = None

    def summary(self):
        return {
            "phase": self.phase,
            "reason": self.reason,
            "total_actions": self.total_actions,
            "assisted": True,
        }


class _RaisingObserveController(_ScriptedController):
    """A scripted controller whose ``observe_after`` raises after a real step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observe_raised = 0

    def observe_after(self, reading):  # noqa: ARG002
        self.observe_calls += 1
        self.observe_raised += 1
        raise RuntimeError("observer exploded")


class _RaisingAuditFile:
    """A minimal ``action_sources.jsonl`` handle whose write/flush raises."""

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.closed = False

    def write(self, data):
        if self.fail_on == "write":
            raise OSError("audit write exploded")
        return len(data)

    def flush(self):
        if self.fail_on == "flush":
            raise OSError("audit flush exploded")

    def close(self):
        self.closed = True


def _read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- bare-service unit tests (no worker, no env) -----------------------------


class SelectActionUnitTests(unittest.TestCase):
    """``_select_action`` assistant integration, without the worker loop."""

    def _bare(self, helper):
        svc = object.__new__(gas.GraspAssistService)
        svc._env = object()
        svc._grasp_assist_helper = helper
        svc._grasp_assist_capability = gas.WINE_CAPABILITY_ID
        svc._grasp_assist_job_id = "job"
        svc._grasp_assist_source = "vla"
        svc._grasp_assist_trigger_proposal = None
        svc._grasp_assist_observer_error = None
        svc._grasp_assist_handoff_reset_done = False
        svc._grasp_assist_return_reset_done = False
        svc._action_function = lambda batch: np.full(7, 0.25)
        svc._wine_raw_action = [9.0] * 7
        svc._wine_post_action = [9.0] * 7
        svc._wine_input_task = ["stale"]
        svc._wine_input_state = [9.0]
        counters = {"resets": 0}
        svc._reset_policy_queues = lambda: counters.__setitem__(
            "resets", counters["resets"] + 1
        )
        return svc, counters

    def test_no_helper_returns_the_original_action(self):
        svc, counters = self._bare(None)
        action = svc._select_action({})
        np.testing.assert_allclose(action, [0.25] * 7)
        self.assertEqual(svc._grasp_assist_source, "vla")
        self.assertEqual(counters["resets"], 0)

    def test_idle_not_triggered_keeps_the_vla_action_and_clears_source(self):
        helper = _ScriptedController(arm=False)
        svc, counters = self._bare(helper)
        with mock.patch.object(local_grasp, "read_geometry", lambda env: {}):
            action = svc._select_action({})
        np.testing.assert_allclose(action, [0.25] * 7)
        self.assertEqual(svc._grasp_assist_source, "vla")
        self.assertEqual(counters["resets"], 0)
        self.assertEqual(helper.phase, local_grasp.IDLE)

    def test_trigger_handoff_replaces_proposal_resets_once_and_clears_evidence(self):
        helper = _ScriptedController(arm=True)
        svc, counters = self._bare(helper)
        with mock.patch.object(local_grasp, "read_geometry", lambda env: {}):
            action = svc._select_action({})
        np.testing.assert_allclose(action, [0.10] * 7)
        self.assertEqual(svc._grasp_assist_source, "local_grasp")
        self.assertEqual(counters["resets"], 1)
        self.assertEqual(svc._grasp_assist_trigger_proposal, [0.25] * 7)
        # The stale VLA raw/post/input evidence can no longer masquerade.
        self.assertIsNone(svc._wine_raw_action)
        self.assertIsNone(svc._wine_post_action)
        self.assertIsNone(svc._wine_input_task)
        self.assertIsNone(svc._wine_input_state)

    def test_active_stage_returns_the_auxiliary_action_directly(self):
        helper = _ScriptedController(arm=True)
        helper.phase = local_grasp.DESCEND
        svc, counters = self._bare(helper)
        with mock.patch.object(local_grasp, "read_geometry", lambda env: {}):
            action = svc._select_action({})
        np.testing.assert_allclose(action, [0.20] * 7)
        self.assertEqual(svc._grasp_assist_source, "local_grasp")
        self.assertIsNone(svc._wine_raw_action)
        self.assertEqual(counters["resets"], 0)  # handoff reset already happened

    def test_confirmed_return_resets_exactly_once_then_stays_vla(self):
        helper = _ScriptedController(arm=True)
        helper.phase = local_grasp.CONFIRMED
        svc, counters = self._bare(helper)
        first = svc._select_action({})
        second = svc._select_action({})
        np.testing.assert_allclose(first, [0.25] * 7)
        np.testing.assert_allclose(second, [0.25] * 7)
        self.assertEqual(counters["resets"], 1)  # exactly once, not per call
        self.assertEqual(svc._grasp_assist_source, "vla")

    def test_failed_phase_raises_before_any_action(self):
        helper = _ScriptedController()
        helper.phase = local_grasp.FAILED
        helper.reason = "total_budget_exceeded"
        svc, _ = self._bare(helper)
        with self.assertRaises(service.SceneError) as ctx:
            svc._select_action({})
        self.assertEqual(ctx.exception.reason, "local_grasp_failed")
        self.assertEqual(ctx.exception.detail, "total_budget_exceeded")

    def test_latched_observer_error_stops_before_any_inference(self):
        helper = _ScriptedController(arm=True)
        svc, _ = self._bare(helper)
        svc._grasp_assist_observer_error = "Traceback: observer exploded"
        inferences = {"n": 0}

        def _counting_action(batch):  # noqa: ARG001
            inferences["n"] += 1
            return np.full(7, 0.25)

        svc._action_function = _counting_action
        with self.assertRaises(service.SceneError) as ctx:
            svc._select_action({})
        self.assertEqual(ctx.exception.reason, "local_grasp_unknown")
        self.assertIn("observer exploded", ctx.exception.detail)
        # The latch short-circuits BEFORE the VLA/helper inference.
        self.assertEqual(inferences["n"], 0)
        self.assertEqual(helper.next_calls, 0)

    def test_unknown_geometry_raises_honestly(self):
        helper = _ScriptedController(arm=False)
        svc, _ = self._bare(helper)

        def _boom(env):  # noqa: ARG001
            raise RuntimeError("probe exploded")

        with mock.patch.object(local_grasp, "read_geometry", _boom):
            with self.assertRaises(service.SceneError) as ctx:
                svc._select_action({})
        self.assertEqual(ctx.exception.reason, "local_grasp_unknown")
        self.assertIn("probe exploded", ctx.exception.detail)
        self.assertEqual(service.status_for_reason("local_grasp_unknown"), 400)


# --- worker integration tests (fake env, no GPU) -----------------------------


class _AssistWorkerHarness:
    """Builds a real ``GraspAssistService`` around a fake simulator."""

    def __init__(self, test_case, *, completion_mode="native", grasp_guard_mode="shadow"):
        self._tmp = tempfile.TemporaryDirectory()
        test_case.addCleanup(self._tmp.cleanup)
        self.action_calls = {"n": 0}
        self.renders = {"n": 0}
        self.action_gate = threading.Event()
        self.action_gate.set()
        self.inner = _AssistInnerEnv()
        self.svc = gas.GraspAssistService(
            run_root=self._tmp.name,
            env_factory=lambda **kwargs: _AssistEnv(self.inner, self.renders),
            policy_loader=lambda s: setattr(s._v1, "_n_action_steps", 10),
            action_function=self._action,
            batch_builder=lambda obs, instruction: {},
            completion_mode=completion_mode,
            grasp_guard_mode=grasp_guard_mode,
        )
        self.svc.start()
        test_case.addCleanup(self._stop)
        deadline = time.time() + 20.0
        while time.time() < deadline and not self.svc.health()["ready"]:
            time.sleep(0.05)

    def _stop(self):
        self.action_gate.set()
        self.svc.stop()

    def _action(self, batch):  # noqa: ARG002
        self.action_calls["n"] += 1
        self.action_gate.wait(timeout=15.0)
        return np.full(7, 0.25)

    def submit(self, session_id, request_id, capability_ids, budget=12, decision="execute"):
        return self.svc.submit_plan(
            {
                "session_id": session_id,
                "scene_version": self.svc.session(session_id)["scene_version"],
                "request_id": request_id,
                "capability_ids": capability_ids,
                "decision": decision,
                "budget_per_subgoal": budget,
            }
        )

    def wait(self, request_id, deadline_s=20.0):
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            payload = self.svc.plan(request_id)
            if payload is not None and payload["state"] in service.TERMINAL_PLAN_STATES:
                return payload
            time.sleep(0.05)
        raise AssertionError("plan %s did not reach a terminal state" % request_id)


class WorkerIntegrationTests(unittest.TestCase):
    def test_local_assist_records_and_confirms(self):
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="shadow"
        )
        harness.inner.truth_after[WINE_GOAL_KEY] = 3
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        self.assertTrue(session["ok"], session)
        sid = session["session_id"]

        controller = _ScriptedController(
            arm=True,
            observe_phases=[local_grasp.ABOVE, local_grasp.CONFIRMED],
        )
        reset_steps: list[int] = []
        original_reset = harness.svc._reset_policy_queues

        def _spy_reset():
            reset_steps.append(harness.inner.steps)
            original_reset()

        harness.svc._reset_policy_queues = _spy_reset

        caller_payload = {
            "session_id": sid,
            "scene_version": harness.svc.session(sid)["scene_version"],
            "request_id": "req-assist",
            "capability_ids": [gas.WINE_CAPABILITY_ID],
            "decision": "execute",
            "budget_per_subgoal": 12,
        }
        frozen = copy.deepcopy(caller_payload)

        with mock.patch.object(
            local_grasp, "LocalGraspController", lambda: controller
        ), mock.patch.object(local_grasp, "read_geometry", lambda env: {}):
            submitted = harness.svc.submit_plan(caller_payload)
            self.assertTrue(submitted["ok"], submitted)
            # The fixed pilot budget override: 12 -> 500.
            self.assertEqual(harness.svc._plans["req-assist"].budget_per_subgoal, 500)
            plan = harness.wait("req-assist")

        # The caller's payload object was not mutated.
        self.assertEqual(caller_payload, frozen)

        job = harness.svc.job(plan["job_ids"][0])
        self.assertEqual(job["ended_reason"], "success", job)
        self.assertTrue(job["success"])
        steps = job["steps"]
        self.assertEqual(steps, harness.inner.steps)

        rows = _read_rows(Path(job["run_dir"]) / gas.ACTION_SOURCES_FILENAME)
        self.assertEqual(len(rows), steps)
        local_rows = [row for row in rows if row["source"] == "local_grasp"]
        vla_rows = [row for row in rows if row["source"] == "vla"]
        self.assertEqual(len(local_rows), 2)
        self.assertEqual(len(vla_rows), steps - 2)
        # Exactly one real step per row, and one observe per local action.
        self.assertEqual([row["step"] for row in rows], list(range(1, steps + 1)))
        self.assertEqual(controller.observe_calls, 2)
        self.assertEqual(controller.total_actions, 2)
        # The VLA proposal is sampled exactly once per returned VLA action, plus
        # the single replaced trigger proposal; the two local actions never call
        # the VLA at all.
        self.assertEqual(harness.action_calls["n"], len(vla_rows) + 1)
        self.assertEqual(len(vla_rows), 5)
        # A local row carries no VLA raw action; the replaced proposal is saved
        # for the FIRST local action only (never repeated for later ones).
        for index, row in enumerate(local_rows):
            self.assertIsNone(row["raw_action"])
            self.assertEqual(len(row["sent_action"]), gas.ACTION_DIM)
            if index == 0:
                self.assertEqual(row["proposed_vla_action"], [0.25] * 7)
            else:
                self.assertIsNone(row["proposed_vla_action"])
        for row in vla_rows:
            self.assertEqual(row["proposed_vla_action"], None)
            self.assertEqual(row["sent_action"], [0.25] * 7)
        # The confirmed return reset the queues exactly once, at the local end.
        self.assertEqual(reset_steps.count(2), 1)

        report = json.loads(
            (Path(job["run_dir"]) / gas.ASSIST_REPORT_FILENAME).read_text(encoding="utf-8")
        )
        self.assertTrue(report["assisted"])
        self.assertEqual(report["actual_local_actions"], 2)
        self.assertEqual(report["actual_vla_actions"], steps - 2)
        self.assertEqual(report["requested_budget"], 12)
        self.assertEqual(report["effective_budget"], 500)
        self.assertTrue(report["confirmation_is_not_placement_success"])

        for name in gas.ASSIST_RESULT_FILENAMES:
            payload = json.loads((Path(job["run_dir"]) / name).read_text(encoding="utf-8"))
            self.assertIs(payload["assisted"], True, name)
            self.assertEqual(payload["actual_local_actions"], 2, name)
            self.assertEqual(payload["actual_vla_actions"], steps - 2, name)
            self.assertTrue(payload["action_sources_path"].endswith(gas.ACTION_SOURCES_FILENAME))

    def test_non_wine_capability_is_untouched(self):
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="shadow"
        )
        harness.inner.truth_after[BOWL_GOAL_KEY] = 3
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        sid = session["session_id"]
        with mock.patch.object(local_grasp, "LocalGraspController") as factory:
            submitted = harness.submit(sid, "req-bowl", ["bowl_to_plate"], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait("req-bowl")
        factory.assert_not_called()  # no controller for a non-wine capability
        job = harness.svc.job(plan["job_ids"][0])
        self.assertTrue(job["success"])
        rows = _read_rows(Path(job["run_dir"]) / gas.ACTION_SOURCES_FILENAME)
        self.assertTrue(rows)
        self.assertTrue(all(row["source"] == "vla" for row in rows))
        # The original VLA action is returned unchanged for a non-wine capability.
        self.assertTrue(all(row["sent_action"] == [0.25] * 7 for row in rows))
        # One VLA inference per real step: nothing injected, nothing doubled.
        self.assertEqual(harness.action_calls["n"], len(rows))
        self.assertEqual(len(rows), harness.inner.steps)
        report = json.loads(
            (Path(job["run_dir"]) / gas.ASSIST_REPORT_FILENAME).read_text(encoding="utf-8")
        )
        self.assertFalse(report["assisted"])
        self.assertEqual(report["actual_local_actions"], 0)
        self.assertEqual(report["helper"], False)

    def test_far_wine_idle_leaves_the_vla_action_unchanged(self):
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="shadow"
        )
        harness.inner.truth_after[WINE_GOAL_KEY] = 3
        # Push the wine bottle far away so the REAL local controller never arms.
        harness.inner.sim.data.joint_qpos["wine_bottle_1_joint0"] = np.array(
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]
        )
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        sid = session["session_id"]
        submitted = harness.submit(sid, "req-far", [gas.WINE_CAPABILITY_ID], budget=12)
        self.assertTrue(submitted["ok"], submitted)
        plan = harness.wait("req-far")
        job = harness.svc.job(plan["job_ids"][0])
        rows = _read_rows(Path(job["run_dir"]) / gas.ACTION_SOURCES_FILENAME)
        self.assertTrue(rows)
        self.assertTrue(all(row["source"] == "vla" for row in rows))
        self.assertTrue(all(row["sent_action"] == [0.25] * 7 for row in rows))
        # The far wine never arms the REAL controller: pure VLA, one step each.
        self.assertEqual(harness.action_calls["n"], len(rows))
        self.assertEqual(len(rows), harness.inner.steps)
        report = json.loads(
            (Path(job["run_dir"]) / gas.ASSIST_REPORT_FILENAME).read_text(encoding="utf-8")
        )
        self.assertFalse(report["assisted"])
        self.assertEqual(report["actual_local_actions"], 0)

    def test_failed_assist_raises_before_the_next_action(self):
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="shadow"
        )
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        sid = session["session_id"]
        controller = _ScriptedController(arm=True, observe_phases=[local_grasp.FAILED])
        with mock.patch.object(
            local_grasp, "LocalGraspController", lambda: controller
        ), mock.patch.object(local_grasp, "read_geometry", lambda env: {}):
            submitted = harness.submit(sid, "req-fail", [gas.WINE_CAPABILITY_ID], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait("req-fail")
        job = harness.svc.job(plan["job_ids"][0])
        self.assertFalse(job["success"])
        self.assertEqual(job["ended_reason"], "local_grasp_failed")
        self.assertIn("wine_translation_drift", job["error"])
        # Exactly one action was stepped; the failure stopped before the next.
        self.assertEqual(job["steps"], 1)
        self.assertEqual(harness.inner.steps, 1)
        self.assertEqual(harness.action_calls["n"], 1)

    def test_unknown_geometry_fails_honestly_without_stepping(self):
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="shadow"
        )
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        sid = session["session_id"]

        def _boom(env):  # noqa: ARG001
            raise RuntimeError("no geometry here")

        with mock.patch.object(local_grasp, "read_geometry", _boom):
            submitted = harness.submit(sid, "req-unknown", [gas.WINE_CAPABILITY_ID], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait("req-unknown")
        job = harness.svc.job(plan["job_ids"][0])
        self.assertFalse(job["success"])
        self.assertEqual(job["ended_reason"], "local_grasp_unknown")
        self.assertIn("no geometry here", job["error"])
        self.assertEqual(job["steps"], 0)
        self.assertEqual(harness.inner.steps, 0)

    def test_cancellation_still_wins_and_is_never_a_failed_grasp(self):
        gate = threading.Event()
        gate.set()
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="enforce"
        )
        self.addCleanup(gate.set)
        harness.action_gate = gate
        # The real controller never arms for a far wine bottle -> pure VLA path.
        harness.inner.sim.data.joint_qpos["wine_bottle_1_joint0"] = np.array(
            [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]
        )
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        sid = session["session_id"]

        gate.clear()  # park the first inference
        submitted = harness.submit(sid, "req-cancel", [gas.WINE_CAPABILITY_ID], budget=12)
        self.assertTrue(submitted["ok"], submitted)
        deadline = time.time() + 10.0
        while time.time() < deadline and harness.action_calls["n"] == 0:
            time.sleep(0.02)
        self.assertGreaterEqual(harness.action_calls["n"], 1)
        ack = harness.svc.cancel_request("req-cancel", sid)
        self.assertTrue(ack["ok"], ack)
        self.assertEqual(ack["state"], "cancelling")
        gate.set()
        plan = harness.wait("req-cancel")

        self.assertEqual(plan["state"], "cancelled", plan)
        job = harness.svc.job(plan["job_ids"][0])
        self.assertEqual(job["state"], "cancelled")
        self.assertEqual(job["ended_reason"], "cancelled")
        self.assertIsNone(job["error"])
        self.assertNotEqual(job["ended_reason"], "failed_grasp")
        self.assertEqual(harness.inner.steps, 0)
        report = json.loads(
            (Path(job["run_dir"]) / gas.ASSIST_REPORT_FILENAME).read_text(encoding="utf-8")
        )
        self.assertFalse(report["assisted"])


class FailClosedIntegrationTests(unittest.TestCase):
    """Observer/audit failures latch and stop before the next action/inference."""

    def _armed(self, request_id, **controller_kwargs):
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="shadow"
        )
        harness.inner.truth_after[WINE_GOAL_KEY] = 3
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        self.assertTrue(session["ok"], session)
        sid = session["session_id"]
        controller = _ScriptedController(arm=True, **controller_kwargs)
        return harness, sid, controller

    def test_observer_exception_stops_before_the_next_action(self):
        harness, sid, _ = self._armed("req-observe")
        controller = _RaisingObserveController(arm=True)
        with mock.patch.object(
            local_grasp, "LocalGraspController", lambda: controller
        ), mock.patch.object(local_grasp, "read_geometry", lambda env: {}):
            submitted = harness.submit(sid, "req-observe", [gas.WINE_CAPABILITY_ID], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait("req-observe")
        job = harness.svc.job(plan["job_ids"][0])
        self.assertFalse(job["success"])
        self.assertEqual(job["ended_reason"], "local_grasp_unknown")
        self.assertIn("observer exploded", job["error"])
        # The just-sent action was still counted and rendered by the base ...
        self.assertEqual(job["steps"], 1)
        self.assertEqual(harness.inner.steps, 1)
        self.assertGreaterEqual(harness.renders["n"], 1)
        # ... but the very next selection stopped before a new inference/action.
        self.assertEqual(harness.action_calls["n"], 1)
        self.assertEqual(controller.observe_raised, 1)
        rows = _read_rows(Path(job["run_dir"]) / gas.ACTION_SOURCES_FILENAME)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "local_grasp")

    def test_post_step_geometry_exception_stops_before_the_next_action(self):
        harness, sid, _ = self._armed("req-geom")
        controller = _ScriptedController(arm=True)
        reads = {"n": 0}

        def _flaky(env):  # noqa: ARG001
            reads["n"] += 1
            if reads["n"] >= 2:  # only the POST-step read (after the trigger read) fails
                raise RuntimeError("post-step geometry exploded")
            return {}

        with mock.patch.object(
            local_grasp, "LocalGraspController", lambda: controller
        ), mock.patch.object(local_grasp, "read_geometry", _flaky):
            submitted = harness.submit(sid, "req-geom", [gas.WINE_CAPABILITY_ID], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait("req-geom")
        job = harness.svc.job(plan["job_ids"][0])
        self.assertFalse(job["success"])
        self.assertEqual(job["ended_reason"], "local_grasp_unknown")
        self.assertIn("post-step geometry exploded", job["error"])
        self.assertEqual(job["steps"], 1)
        self.assertEqual(harness.inner.steps, 1)
        self.assertGreaterEqual(harness.renders["n"], 1)
        self.assertEqual(harness.action_calls["n"], 1)
        rows = _read_rows(Path(job["run_dir"]) / gas.ACTION_SOURCES_FILENAME)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "local_grasp")
        self.assertIn("error", rows[0]["geometry"])

    def _run_with_failing_audit(self, request_id, fail_on):
        harness, sid, _ = self._armed(request_id)
        controller = _ScriptedController(arm=True)
        real_open = open

        def _fake_open(path, *args, **kwargs):
            if str(path).endswith(gas.ACTION_SOURCES_FILENAME):
                return _RaisingAuditFile(fail_on)
            return real_open(path, *args, **kwargs)

        with mock.patch.object(
            local_grasp, "LocalGraspController", lambda: controller
        ), mock.patch.object(
            local_grasp, "read_geometry", lambda env: {}
        ), mock.patch("builtins.open", _fake_open):
            submitted = harness.submit(sid, request_id, [gas.WINE_CAPABILITY_ID], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait(request_id)
        return harness, harness.svc.job(plan["job_ids"][0])

    def test_action_sources_write_failure_stops_before_the_next_action(self):
        harness, job = self._run_with_failing_audit("req-write", "write")
        self.assertFalse(job["success"])
        self.assertEqual(job["ended_reason"], "local_grasp_unknown")
        self.assertIn("audit write exploded", job["error"])
        # The real step finished (counted) before the latched write failure stopped.
        self.assertEqual(job["steps"], 1)
        self.assertEqual(harness.inner.steps, 1)
        self.assertEqual(harness.action_calls["n"], 1)

    def test_action_sources_flush_failure_stops_before_the_next_action(self):
        harness, job = self._run_with_failing_audit("req-flush", "flush")
        self.assertFalse(job["success"])
        self.assertEqual(job["ended_reason"], "local_grasp_unknown")
        self.assertIn("audit flush exploded", job["error"])
        self.assertEqual(job["steps"], 1)
        self.assertEqual(harness.inner.steps, 1)
        self.assertEqual(harness.action_calls["n"], 1)

    def test_audit_open_failure_stops_before_any_action(self):
        harness, sid, _ = self._armed("req-open")
        controller = _ScriptedController(arm=True)
        real_open = open

        def _fake_open(path, *args, **kwargs):
            if str(path).endswith(gas.ACTION_SOURCES_FILENAME):
                raise OSError("audit open exploded")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(
            local_grasp, "LocalGraspController", lambda: controller
        ), mock.patch("builtins.open", _fake_open):
            submitted = harness.submit(sid, "req-open", [gas.WINE_CAPABILITY_ID], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait("req-open")
        job = harness.svc.job(plan["job_ids"][0])
        self.assertFalse(job["success"])
        self.assertEqual(job["state"], "error")
        self.assertEqual(job["ended_reason"], "audit_unavailable")
        self.assertIn("audit_unavailable", job["error"])
        self.assertIn("audit open exploded", job["error"])
        # The wrapper finalized its own pre-action job: a public terminal error
        # with zero actions and the current service step budget.
        self.assertEqual(job["total_steps"], harness.svc._total_steps)
        self.assertEqual(job["wall_s"], 0.0)
        # No physics and no inference happened without the audit.
        self.assertEqual(job["steps"], 0)
        self.assertEqual(harness.inner.steps, 0)
        self.assertEqual(harness.action_calls["n"], 0)
        self.assertEqual(controller.next_calls, 0)
        self.assertFalse((Path(job["run_dir"]) / gas.ACTION_SOURCES_FILENAME).exists())

    def test_trigger_proposal_is_recorded_only_for_the_first_local_action(self):
        harness, sid, _ = self._armed("req-first")
        controller = _ScriptedController(
            arm=True,
            observe_phases=[local_grasp.ABOVE, local_grasp.DESCEND, local_grasp.CONFIRMED],
        )
        with mock.patch.object(
            local_grasp, "LocalGraspController", lambda: controller
        ), mock.patch.object(local_grasp, "read_geometry", lambda env: {}):
            submitted = harness.submit(sid, "req-first", [gas.WINE_CAPABILITY_ID], budget=12)
            self.assertTrue(submitted["ok"], submitted)
            plan = harness.wait("req-first")
        job = harness.svc.job(plan["job_ids"][0])
        rows = _read_rows(Path(job["run_dir"]) / gas.ACTION_SOURCES_FILENAME)
        local_rows = [row for row in rows if row["source"] == "local_grasp"]
        self.assertEqual(len(local_rows), 3)
        self.assertEqual(local_rows[0]["proposed_vla_action"], [0.25] * 7)
        for row in local_rows[1:]:
            self.assertIsNone(row["proposed_vla_action"])
        for row in rows:
            if row["source"] == "vla":
                self.assertIsNone(row["proposed_vla_action"])


class JsonDefaultTests(unittest.TestCase):
    """``_json_default`` keeps numpy numbers numeric (never stringified)."""

    def test_ndarray_is_a_nested_numeric_list(self):
        matrix = np.array([[1.5, 2.0], [3.25, 4.0]], dtype=np.float32)
        converted = gas._json_default(matrix)
        self.assertNotIsInstance(converted, str)
        self.assertIsInstance(converted, list)
        self.assertIsInstance(converted[0], list)
        self.assertIsInstance(converted[0][0], float)
        self.assertEqual(converted, [[1.5, 2.0], [3.25, 4.0]])

    def test_numpy_scalar_item_and_other_fallback(self):
        self.assertEqual(gas._json_default(np.float32(2.5)), 2.5)
        self.assertIsInstance(gas._json_default(np.float64(3.0)), float)
        self.assertEqual(gas._json_default(np.int64(7)), 7)
        self.assertIsInstance(gas._json_default(np.int64(7)), int)
        self.assertIs(gas._json_default(np.bool_(True)), True)
        # Non-numpy objects keep the previous ``default=str`` behaviour.
        self.assertEqual(gas._json_default("already a string"), "already a string")
        self.assertEqual(gas._json_default(None), "None")

    def test_dumps_round_trips_numeric_without_stringifying(self):
        record = {
            "matrix": np.array([[1, 2], [3, 4]], dtype=np.float64),
            "scalar": np.float32(0.5),
            "count": np.int64(3),
        }
        payload = json.loads(json.dumps(record, default=gas._json_default))
        self.assertEqual(payload["matrix"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertNotIsInstance(payload["matrix"], str)
        self.assertEqual(payload["scalar"], 0.5)
        self.assertEqual(payload["count"], 3)


class BudgetOverrideTests(unittest.TestCase):
    """``submit_plan`` overrides only ``execute`` and never mutates the caller."""

    def test_execute_overridden_and_clarify_untouched(self):
        harness = _AssistWorkerHarness(
            self, completion_mode="native", grasp_guard_mode="shadow"
        )
        session = harness.svc.create_session("goal_table", seed=0, init_state_index=0)
        sid = session["session_id"]

        clarify = {
            "session_id": sid,
            "scene_version": harness.svc.session(sid)["scene_version"],
            "request_id": "req-clarify",
            "capability_ids": [],
            "decision": "clarify",
            "budget_per_subgoal": 50,
        }
        clarify_frozen = copy.deepcopy(clarify)
        self.assertTrue(harness.svc.submit_plan(clarify)["ok"])
        self.assertEqual(clarify, clarify_frozen)  # caller payload unchanged
        self.assertEqual(harness.svc._plans["req-clarify"].budget_per_subgoal, 50)
        # Drain the clarify plan (no jobs) so the execute submission is not busy.
        self.assertEqual(harness.wait("req-clarify")["state"], "completed")

        execute = {
            "session_id": sid,
            "scene_version": harness.svc.session(sid)["scene_version"],
            "request_id": "req-exec",
            "capability_ids": ["bowl_to_plate"],
            "decision": "execute",
            "budget_per_subgoal": 50,
        }
        execute_frozen = copy.deepcopy(execute)
        harness.inner.truth_after[BOWL_GOAL_KEY] = 3
        self.assertTrue(harness.svc.submit_plan(execute)["ok"])
        self.assertEqual(execute, execute_frozen)  # caller payload unchanged
        self.assertEqual(harness.svc._plans["req-exec"].budget_per_subgoal, gas.ASSIST_BUDGET_PER_SUBGOAL)
        self.assertEqual(gas.ASSIST_BUDGET_PER_SUBGOAL, 500)
        self.assertEqual(harness.wait("req-exec")["state"], "completed")


class CliContractTests(unittest.TestCase):
    """``--help`` and the process environment convention, without a GPU."""

    def test_parser_defaults(self):
        parser = gas._build_parser()
        args = parser.parse_args(["--run-root", "/tmp/assist"])
        self.assertEqual(args.port, gas.DEFAULT_ASSIST_PORT)
        self.assertEqual(args.port, 8779)
        self.assertEqual(args.run_root, "/tmp/assist")

    def test_run_root_is_required(self):
        parser = gas._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_process_environment_convention(self):
        # ``mock.patch.dict`` snapshots the WHOLE os.environ and restores it on
        # exit, so no global environment state leaks into the other test modules.
        with mock.patch.dict(os.environ, {"http_proxy": "http://proxy"}, clear=False):
            gas.configure_process_environment()
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")
            self.assertEqual(os.environ["LIBERO_CONFIG_PATH"], gas.LIBERO_CONFIG_PATH)
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], gas.WSL_LIBRARY_PATH)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            for name in gas.PROXY_ENV_VARS:
                self.assertNotIn(name, os.environ)


if __name__ == "__main__":
    unittest.main()
