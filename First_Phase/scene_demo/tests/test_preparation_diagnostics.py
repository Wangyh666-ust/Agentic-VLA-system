#!/usr/bin/env python3
"""GPU-free focused unit tests for the assisted-preparation diagnostic.

Everything (service, policy, model, simulator) is faked: no CUDA, no model load
and no live environment.  Five focused contracts are exercised against the
CURRENT seams:

1. the live OSC delta-controller contract is checked *before any movement* and a
   mismatch refuses and closes the session without stepping;
2. the pure servo action has the exact float32 shape, the +/-0.35 position clamp,
   zero-orientation channels, the +/-0.25 yaw clamp and the fixed -1 gripper;
3. the REAL ``_prepare_work`` lift+align pipeline records the actual auxiliary
   count and the failed action and stops BEFORE the next step on a move;
4. ``_sync_work`` raises PreparationBlocked ONLY for a physical failure (an
   operational failure is an ordinary RuntimeError) and preserves the original
   prefix result identity;
5. ``_trial_view`` filters ONLY the physical marker (raw trial identity intact)
   and ``_cross_seed_check`` accepts a ``None`` success reason with identical
   state/fingerprint while flagging a changed after-preparation SHA.
"""

from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import numpy as np  # noqa: E402

import preparation_diagnostics as pd  # noqa: E402
import service  # noqa: E402


def _rot_z(angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _bare_service(env: object, sessions: dict, total_steps: int) -> "pd.PreparedService":
    """A ``PreparedService`` whose real ``__init__`` chain never runs torch."""

    svc = object.__new__(pd.PreparedService)
    svc._env = env
    svc._sessions = sessions
    svc._total_steps = total_steps
    svc._last_obs = None
    svc._pre_bowl_pose = None
    svc.preparation_result = None
    svc.prefix_result = None
    svc._prepared_session_id = None
    return svc


class _CountingEnv:
    """An env whose ``step`` only records that it was (wrongly) called."""

    def __init__(self) -> None:
        self.steps = 0

    def step(self, action):  # noqa: ARG002
        self.steps += 1
        raise AssertionError("env.step must not be called here")


class _MovingWineEnv:
    """A fake env whose single step moves ``wine_bottle_1`` by 0.006 m."""

    def __init__(self) -> None:
        self.step_calls: list = []
        self.positions = {
            "wine_bottle_1": [0.0, 0.0, 0.0],
            "akita_black_bowl_1": [1.0, 0.0, 0.0],
            "plate_1": [2.0, 0.0, 0.0],
            "cream_cheese_1": [3.0, 0.0, 0.0],
        }

    def step(self, action):
        self.step_calls.append(np.asarray(action, dtype=np.float64).copy())
        self.positions["wine_bottle_1"] = [0.006, 0.0, 0.0]
        return ({"obs": len(self.step_calls)}, 0.0, False, {})


def _snapshot_of(env: "_MovingWineEnv", goals: object) -> dict:
    return {
        "predicates": {pd.BOWL_GOAL_KEY: True},
        "held_objects": [],
        "grasp_observation_complete": True,
        "strict_candidate": True,
        "objects": {key: {"position": list(value)} for key, value in env.positions.items()},
    }


def _complete_fingerprint(sha: str = "c" * 64) -> dict:
    return {
        "errors": [],
        "cameras": [{"raw_sha256": sha}, {"raw_sha256": sha}],
        "state": {"raw_sha256": sha},
        "task": "put the wine bottle on the rack",
        "task_sha256": sha,
        "combined_sha256": sha,
    }


def _view(after_sha: str = "d" * 64, fingerprint_sha: str = "c" * 64) -> dict:
    return {
        "prefix_result": {
            "origin_state_sha": pd.REPLAY_ORIGIN_SHA,
            "final_state_sha": pd.REPLAY_FINAL_SHA,
        },
        "preparation": {
            "ok": True,
            "kind": None,
            "reason": None,
            "before_prepare_state_sha": pd.REPLAY_FINAL_SHA,
            "after_prepare_state_sha": after_sha,
        },
        "auxiliary_action_count": 7,
        "raw_trial": {
            "xml_sha": "a" * 64,
            "before_wine_state_sha": pd.REPLAY_FINAL_SHA,
            "first_fingerprint": _complete_fingerprint(fingerprint_sha),
        },
    }


class TestControllerContract(unittest.TestCase):
    def test_controller_mismatch_refuses_before_movement(self) -> None:
        env = _CountingEnv()
        svc = _bare_service(env, {}, 0)
        svc._close_env = mock.Mock()
        with mock.patch.object(
            pd.context.ContextDiagnosticService, "_do_create_session", create=True,
            return_value={"ok": True},
        ), mock.patch.object(
            pd, "controller_facts", return_value={"sentinel": 1}
        ), mock.patch.object(
            pd, "controller_mismatch", return_value="use_delta is not exactly True: 1"
        ):
            result = svc._do_create_session(object(), 0, 0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "controller_mismatch")
        self.assertTrue(result["detail"])
        svc._close_env.assert_called_once()
        self.assertEqual(env.steps, 0)
        self.assertIsNone(svc._pre_bowl_pose)


class TestServoAction(unittest.TestCase):
    def test_servo_action_clamps_and_opens_the_empty_gripper(self) -> None:
        identity = np.eye(3)
        action = pd.servo_action([5.0, -5.0, 1.0], [0.0, 0.0, 0.0], identity, identity)
        self.assertEqual(action.dtype, np.float32)
        self.assertEqual(action.shape, (pd.ACTION_DIM,))
        self.assertAlmostEqual(float(action[0]), pd.SERVO_TRANSLATION_CLAMP, places=6)
        self.assertAlmostEqual(float(action[1]), -pd.SERVO_TRANSLATION_CLAMP, places=6)
        self.assertAlmostEqual(float(action[2]), pd.SERVO_TRANSLATION_CLAMP, places=6)
        self.assertTrue(np.allclose(action[3:6], 0.0))  # identity -> zero rotation
        self.assertAlmostEqual(float(action[6]), -1.0)
        yaw = pd.servo_action([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], _rot_z(math.pi / 2), identity)
        self.assertTrue(np.allclose(yaw[:5], 0.0))
        self.assertAlmostEqual(float(yaw[5]), pd.SERVO_ROTATION_CLAMP, places=6)


class TestPrepareWork(unittest.TestCase):
    def test_prepare_work_stops_on_a_protected_object_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _MovingWineEnv()
            record = types.SimpleNamespace(total_steps=102, run_dir=Path(tmp))
            svc = _bare_service(env, {"s1": record}, 102)
            svc._pre_bowl_pose = {"position": [0.0, 0.0, 0.1], "orientation_matrix": np.eye(3)}
            pose = {
                "position": [0.0, 0.0, 0.1],
                "orientation_matrix": np.eye(3),
                "orientation": np.eye(3).tolist(),
            }
            with mock.patch.object(
                pd.service, "state_sha", return_value=pd.REPLAY_FINAL_SHA
            ), mock.patch.object(
                pd.context, "_flatten_sim_state", return_value=([0.0], "fake")
            ), mock.patch.object(
                pd, "controller_facts", return_value={}
            ), mock.patch.object(
                pd, "controller_mismatch", return_value=None
            ), mock.patch.object(
                pd, "read_eef_pose", return_value=pose
            ), mock.patch.object(
                pd.pe, "capture_snapshot", side_effect=_snapshot_of
            ):
                result = svc._prepare_work("s1", Path(tmp))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "protection_violation")
        self.assertEqual(result["kind"], "physical")
        self.assertEqual(result["aux_actions"], 1)  # the ACTUAL auxiliary count
        self.assertEqual(len(env.step_calls), 1)  # no second step
        self.assertIsInstance(result["failed_action"], list)
        self.assertEqual(len(result["failed_action"]), pd.ACTION_DIM)  # action retained
        self.assertEqual(
            [event["phase"] for event in result["events"]], ["lift_before", "lift_after"]
        )
        self.assertEqual(result["protection_baseline_violations"], [])
        for object_id in pd.PROTECTED_OBJECTS:  # before: all four present
            self.assertIsNotNone(result["protection_baseline"][object_id])
        self.assertGreater(env.positions["wine_bottle_1"][0], pd.PROTECTION_TOLERANCE_M)
        self.assertIn("wine_bottle_1", result["detail"])


class TestPreparationHook(unittest.TestCase):
    def test_sync_work_blocks_only_physical_and_preserves_prefix(self) -> None:
        original = {"ok": True, "final_state_sha": pd.REPLAY_FINAL_SHA, "sentinel": "orig"}
        kinds: list = []

        def fake_sync(self_, kind, fn, timeout=service.WORKER_WAIT_S):  # noqa: ARG001
            kinds.append(kind)
            if kind == "replay_prefix":
                return original
            return {"ok": False, "reason": "protection_violation", "kind": "physical", "detail": "moved"}

        svc = _bare_service(None, {}, 0)
        with mock.patch.object(pd.context.ContextDiagnosticService, "_sync_work", new=fake_sync):
            with self.assertRaises(pd.PreparationBlocked):
                svc._sync_work("replay_prefix", lambda: None, 1.0)
        self.assertEqual(kinds, ["replay_prefix", "prepare_after_prefix"])  # no recursion
        self.assertIs(svc.prefix_result, original)  # original result preserved untouched
        self.assertEqual(svc.preparation_result["kind"], "physical")

        # The physical block happens BEFORE the caller's seed marker: simulate the
        # real caller and require it never reaches the model-RNG reseed.
        order: list = []

        def fake_context_run_trial(svc_, entry, actions, run_root, seen):  # noqa: ARG001
            svc_._sync_work("replay_prefix", lambda: None, 1.0)
            order.append("seed_model_rng")
            return {
                "trial_id": "t", "condition": pd.PREFIX_CONDITION, "model_seed": 0,
                "errors": [], "operational_errors": [], "jobs": [],
                "physical_failures": [], "after_bowl_state_sha": None,
            }

        blocked_svc = _bare_service(None, {}, 0)
        with mock.patch.object(
            pd.context.ContextDiagnosticService, "_sync_work", new=fake_sync
        ), mock.patch.object(pd.context, "_run_trial", new=fake_context_run_trial):
            view = pd._run_trial(blocked_svc, {"model_seed": 0}, [], Path("/tmp/unused"), {})
        self.assertNotIn("seed_model_rng", order)  # blocked before the seed marker
        self.assertEqual(view["status"], "physical_preparation_failed")
        self.assertEqual(view["after_bowl_state_sha"], pd.REPLAY_FINAL_SHA)  # even blocked
        self.assertIs(blocked_svc.prefix_result, original)  # original prefix preserved

        # A successful preparation returns the SAME original result object.
        def fake_sync_ok(self_, kind, fn, timeout=service.WORKER_WAIT_S):  # noqa: ARG001
            if kind == "replay_prefix":
                return original
            return {"ok": True, "reason": None, "kind": None}

        svc_ok = _bare_service(None, {}, 0)
        with mock.patch.object(pd.context.ContextDiagnosticService, "_sync_work", new=fake_sync_ok):
            result = svc_ok._sync_work("replay_prefix", lambda: None, 1.0)
        self.assertIs(result, original)

        # An OPERATIONAL failure is an ordinary RuntimeError, never a physical block.
        def fake_sync_op(self_, kind, fn, timeout=service.WORKER_WAIT_S):  # noqa: ARG001
            if kind == "replay_prefix":
                return original
            return {"ok": False, "reason": "no_env", "kind": "operational", "detail": "no env"}

        svc_op = _bare_service(None, {}, 0)
        with mock.patch.object(pd.context.ContextDiagnosticService, "_sync_work", new=fake_sync_op):
            with self.assertRaises(RuntimeError) as raised:
                svc_op._sync_work("replay_prefix", lambda: None, 1.0)
        self.assertNotIsInstance(raised.exception, pd.PreparationBlocked)


class TestViewAndCrossSeed(unittest.TestCase):
    def test_view_filters_only_physical_and_cross_seed_compares_state(self) -> None:
        marker = "trial_exception:\n%s: physical failure" % pd.BLOCKED_MARKER
        self.assertTrue(pd._is_blocked_message("trial_exception:\nPreparationBlocked: physical failure"))
        self.assertFalse(pd._is_blocked_message("unrelated mention of PreparationBlocked"))
        raw = {
            "trial_id": "t", "condition": pd.PREFIX_CONDITION, "model_seed": 0,
            "operational_errors": [marker, "unrelated: boom"],
            "physical_failures": [], "jobs": [], "after_bowl_state_sha": None,
        }
        physical = pd._trial_view(
            {"model_seed": 0}, raw, {"ok": False, "reason": "protection_violation", "kind": "physical"}
        )
        self.assertIs(physical["raw_trial"], raw)  # raw trial identity retained
        self.assertNotIn(marker, physical["operational_errors"])  # physical marker filtered
        self.assertIn("unrelated: boom", physical["operational_errors"])  # unrelated kept
        self.assertEqual(physical["status"], "physical_preparation_failed")
        operational = pd._trial_view(
            {"model_seed": 0}, raw, {"ok": False, "reason": "no_env", "kind": "operational"}
        )
        self.assertEqual(operational["status"], "operational_error")
        self.assertIn(marker, operational["operational_errors"])  # only physical is filtered

        seen = {"shared_after_bowl": {"kept": True}}
        self.assertEqual(pd._cross_seed_check(_view(), seen, 0)["differing"], {})
        self.assertEqual(pd._cross_seed_check(_view(), seen, 1)["differing"], {})  # None reason valid
        self.assertEqual(seen["shared_after_bowl"], {"kept": True})  # context key untouched
        changed = pd._cross_seed_check(_view(after_sha="e" * 64), seen, 1)
        self.assertIn("after_prepare_state_sha", changed["differing"])

if __name__ == "__main__":
    unittest.main()
