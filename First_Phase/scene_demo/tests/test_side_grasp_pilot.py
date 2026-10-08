#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-free CPU contract tests for the side_grasp pilot.

Tests mock the real module functions (never production stubs) and never load a
model, build an environment, open an HTTP socket or call Hermes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_SCENE = _HERE.parent
if str(_SCENE) not in sys.path:
    sys.path.insert(0, str(_SCENE))

import side_grasp_pilot as pilot  # noqa: E402
import grasp_assist_service as gas  # noqa: E402
import local_grasp  # noqa: E402

EXACT_CASES = (
    "\u8bf7\u5e2e\u6211\u6574\u7406\u684c\u9762\u4e0a\u7684\u7897\u548c\u9152\u74f6\u3002",
    "\u628a\u7897\u653e\u5230\u76d8\u5b50\u4e0a\uff0c\u7136\u540e\u628a\u9152\u74f6\u653e\u5230\u67b6\u5b50\u4e0a\u3002",
    "\u8bf7\u628a\u9152\u74f6\u548c\u7897\u90fd\u6536\u56de\u5404\u81ea\u7684\u4f4d\u7f6e\u3002",
    "\u5148\u628a\u9152\u74f6\u653e\u5230\u67b6\u5b50\u4e0a\uff0c\u518d\u628a\u7897\u653e\u5230\u76d8\u5b50\u4e0a\u3002",
    "\u6574\u7406\u8fd9\u4e24\u4ef6\u4e1c\u897f\uff1a\u7897\u653e\u76d8\u5b50\uff0c\u9152\u74f6\u653e\u67b6\u5b50\u3002",
)


class CasesContractTest(unittest.TestCase):
    """The five literal requests and identical two-goal literal gold."""

    def test_exact_five_cases_in_order(self):
        self.assertEqual(tuple(pilot.CASES), EXACT_CASES)
        self.assertEqual(len(pilot.CASES), 5)

    def test_literal_two_goal_gold(self):
        self.assertEqual(pilot.GOLD, [["on", "akita_black_bowl_1", "plate_1"],
                                      ["on", "wine_bottle_1", "wine_rack_1_top_region"]])
        self.assertEqual(len(pilot.GOLD_KEYS), 2)

    def test_fixed_budget_and_seeds(self):
        self.assertEqual(pilot.BUDGET_PER_SUBGOAL, 500)
        self.assertEqual(pilot.MODEL_SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual((pilot.SEED, pilot.INIT_STATE_INDEX), (0, 0))

    def test_order_expectation_indices(self):
        self.assertEqual(pilot.ORDER_EXPECTATIONS[1], ("bowl_to_plate", "wine_to_rack"))
        self.assertEqual(pilot.ORDER_EXPECTATIONS[3], ("wine_to_rack", "bowl_to_plate"))
        for index in (0, 2, 4):
            self.assertNotIn(index, pilot.ORDER_EXPECTATIONS)


class CalibrationGateTest(unittest.TestCase):
    """Nested calibration/prereg/hash/reference gates fail closed."""

    def setUp(self):
        self.tmp = Path(self._testMethodName + "_dir")
        self.tmp.mkdir(exist_ok=True)
        self.addCleanup(self._cleanup)
        self.source_dir = self.tmp / "src"
        self.source_dir.mkdir()
        self.source_map = {}
        for name in pilot.PREREG_SOURCE_FILES:
            payload = ("payload for %s" % name).encode("utf-8")
            path = self.source_dir / name
            path.write_bytes(payload)
            self.source_map[name] = hashlib.sha256(payload).hexdigest()

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, data, prereg):
        cal_path = self.tmp / "calibration.json"
        prereg_bytes = json.dumps(prereg, sort_keys=True).encode("utf-8")
        (self.tmp / "preregistration.json").write_bytes(prereg_bytes)
        data = dict(data)
        data["preregistration_sha256"] = hashlib.sha256(prereg_bytes).hexdigest()
        cal_path.write_text(json.dumps(data), encoding="utf-8")
        return cal_path

    def _valid_prereg(self):
        return {"experiment": "side_grasp_calibration",
                "source_sha256": dict(self.source_map),
                "reference": {"path": "side_grasp.py",
                              "source_sha256": pilot.REFERENCE_SHA256,
                              "step": pilot.REFERENCE_STEP,
                              "relative_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}

    def _reference_metadata(self):
        return {"path": "side_grasp.py", "source_sha256": pilot.REFERENCE_SHA256,
                "step": pilot.REFERENCE_STEP,
                "relative_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    def test_accepts_nested_schema(self):
        prereg = self._valid_prereg()
        cal = self._write({"ok": True, "calibration": {"confirmed": True}}, prereg)
        with mock.patch.object(pilot.side_grasp, "reference_metadata",
                               return_value=self._reference_metadata()):
            data, error = pilot.validate_calibration(cal, self.source_dir)
        self.assertIsNone(error)
        self.assertEqual(data["ok"], True)

    def test_rejects_top_level_confirmed(self):
        prereg = self._valid_prereg()
        cal = self._write({"ok": True, "confirmed": True}, prereg)
        with mock.patch.object(pilot.side_grasp, "reference_metadata",
                               return_value=self._reference_metadata()):
            data, error = pilot.validate_calibration(cal, self.source_dir)
        self.assertIsNone(data)
        self.assertIn("confirmed", error)

    def test_rejects_tampered_source(self):
        prereg = self._valid_prereg()
        cal = self._write({"ok": True, "calibration": {"confirmed": True}}, prereg)
        (self.source_dir / "side_grasp.py").write_bytes(b"tampered")
        with mock.patch.object(pilot.side_grasp, "reference_metadata",
                               return_value=self._reference_metadata()):
            data, error = pilot.validate_calibration(cal, self.source_dir)
        self.assertIsNone(data)
        self.assertIn("SHA mismatch", error)

    def test_rejects_unknown_source_entry(self):
        prereg = self._valid_prereg()
        prereg["source_sha256"].pop("side_grasp.py")
        cal = self._write({"ok": True, "calibration": {"confirmed": True}}, prereg)
        with mock.patch.object(pilot.side_grasp, "reference_metadata",
                               return_value=self._reference_metadata()):
            data, error = pilot.validate_calibration(cal, self.source_dir)
        self.assertIsNone(data)
        self.assertIn("missing", error)

    def test_rejects_wrong_reference_step(self):
        prereg = self._valid_prereg()
        prereg["reference"]["step"] = "93"
        cal = self._write({"ok": True, "calibration": {"confirmed": True}}, prereg)
        with mock.patch.object(pilot.side_grasp, "reference_metadata",
                               return_value=self._reference_metadata()):
            data, error = pilot.validate_calibration(cal, self.source_dir)
        self.assertIsNone(data)
        self.assertIn("step", error)

    def test_rejects_preregistration_sha_mismatch(self):
        prereg = self._valid_prereg()
        cal = self._write({"ok": True, "calibration": {"confirmed": True}}, prereg)
        payload = json.loads(cal.read_text(encoding="utf-8"))
        payload["preregistration_sha256"] = "0" * 64
        cal.write_text(json.dumps(payload), encoding="utf-8")
        with mock.patch.object(pilot.side_grasp, "reference_metadata",
                               return_value=self._reference_metadata()):
            data, error = pilot.validate_calibration(cal, self.source_dir)
        self.assertIsNone(data)
        self.assertIn("preregistration sha", error)


class FailedCalibrationNoModelTest(unittest.TestCase):
    """A failed calibration aborts before any model/Hermes invocation."""

    def test_failed_calibration_blocks_hermes_and_model(self):
        tmp = Path(self._testMethodName + "_dir")
        tmp.mkdir(exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        cal = tmp / "calibration.json"
        cal.write_text(json.dumps({"ok": True, "calibration": {"confirmed": False}}), encoding="utf-8")
        output = tmp / "out"
        hermes = tmp / "home"
        args = type("A", (), {"output_dir": str(output), "calibration": str(cal),
                              "hermes_home": str(hermes)})()
        with mock.patch.object(pilot.gas, "GraspAssistService") as svc_cls, \
             mock.patch.object(pilot.run_agent, "HermesRunner") as hermes_cls, \
             mock.patch.object(pilot.run_agent, "Runner") as runner_cls:
            assert pilot.run_pilot(args) == 1
            svc_cls.assert_not_called()
            hermes_cls.assert_not_called()
            runner_cls.assert_not_called()
        self.assertTrue((output / pilot.REPORT_NAME).is_file())


class GoldScorerTest(unittest.TestCase):
    """The independent literal-gold scorer never derives from a plan."""

    def _snapshot(self, predicates, held, position):
        return {"predicates": predicates, "held_objects": held,
                "objects": {pilot.CHEESE_OBJECT: {"position": position}}}

    def test_missing_wine_cannot_pass_even_with_planner_success(self):
        predicates = {pilot.GOLD_KEYS[0]: True, pilot.GOLD_KEYS[1]: None}
        rows = [{"strict_candidate": True} for _ in range(pilot.STRICT_STREAK)]
        final = self._snapshot(predicates, [], [0.0, 0.0, 0.0])
        before = self._snapshot(None, None, [0.0, 0.0, 0.0])
        score = pilot.score_gold(rows, final, before, True)
        self.assertFalse(score["combined_success"])

    def test_unknown_stove_never_success(self):
        predicates = {key: True for key in pilot.GOLD_KEYS}
        rows = [{"strict_candidate": True} for _ in range(pilot.STRICT_STREAK)]
        final = self._snapshot(predicates, [], [0.0, 0.0, 0.0])
        before = self._snapshot(None, None, [0.0, 0.0, 0.0])
        score = pilot.score_gold(rows, final, before, None)
        self.assertFalse(score["combined_success"])

    def test_all_known_good_is_success(self):
        predicates = {key: True for key in pilot.GOLD_KEYS}
        rows = [{"strict_candidate": True} for _ in range(pilot.STRICT_STREAK)]
        final = self._snapshot(predicates, [], [0.0, 0.0, 0.0])
        before = self._snapshot(None, None, [0.001, 0.0, 0.0])
        score = pilot.score_gold(rows, final, before, True)
        self.assertTrue(score["combined_success"])


class OrderAndClassifyTest(unittest.TestCase):
    """Actual requested-order checks and physical-vs-operational classification."""

    def test_order_ok_uses_actual_jobs(self):
        jobs = [{"capability_id": "bowl_to_plate"}, {"capability_id": "wine_to_rack"}]
        self.assertTrue(pilot._request_order(jobs, ("bowl_to_plate", "wine_to_rack")))
        self.assertFalse(pilot._request_order(jobs, ("wine_to_rack", "bowl_to_plate")))

    def test_request_order_true_when_no_expectation(self):
        self.assertTrue(pilot._request_order([{"capability_id": "anything"}], None))

    def _ok_summary(self):
        return {"hermes_invocations": 1, "run_ok": True}

    def test_local_grasp_errors_are_physical(self):
        for ended in ("local_grasp_failed", "local_grasp_unknown", "failed_grasp", "budget_exhausted"):
            jobs = [{"job_id": "j", "ended_reason": ended, "state": "error", "error": "x"}]
            ops, phys = pilot.classify_case(self._ok_summary(), {"state": "completed"}, jobs)
            self.assertEqual(ops, [], ended)
            self.assertTrue(any(ended in item for item in phys), ended)

    def test_unrelated_errors_are_operational(self):
        jobs = [{"job_id": "j", "state": "error", "ended_reason": "error", "error": "boom"}]
        ops, phys = pilot.classify_case(self._ok_summary(), {"state": "completed"}, jobs)
        self.assertTrue(any("job_error" in item for item in ops))
        self.assertFalse(any("job_error" in item for item in phys))

    def test_plan_error_is_operational(self):
        ops, phys = pilot.classify_case(self._ok_summary(), {"state": "error", "error": "bad"}, [])
        self.assertTrue(any("plan_error" in item for item in ops))
        self.assertEqual(phys, [])

    def test_plan_error_after_known_physical_is_physical(self):
        jobs = [{"job_id": "wine", "ended_reason": "local_grasp_failed", "state": "error", "error": "x"}]
        ops, phys = pilot.classify_case(self._ok_summary(), {"state": "error", "error": "wine"}, jobs)
        self.assertEqual(ops, [])
        self.assertTrue(any("plan_error_after_physical" in item for item in phys))


class ScoreWithOrderTest(unittest.TestCase):
    """score_with_order separates placement success from request order."""

    def test_placement_true_order_false_is_final_false(self):
        original = {"combined_success": True, "gold_completed": True}
        result = pilot.score_with_order(original, False)
        self.assertTrue(result["placement_success"])
        self.assertFalse(result["combined_success"])
        self.assertEqual(original, {"combined_success": True, "gold_completed": True})


class RunCaseOrderRegressionTest(unittest.TestCase):
    """run_case must apply order AFTER the actual job order is known."""

    def test_run_case_index1_reversed_order_scores_placement_only(self):
        tmp = Path(self._testMethodName + "_dir")
        tmp.mkdir(exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))

        gold_combined = {"gold_completed": True, "combined_success": True,
                         "n_actual_samples": pilot.STRICT_STREAK,
                         "final_predicates": {key: True for key in pilot.GOLD_KEYS},
                         "held_objects": [], "stove_off": True,
                         "cheese_displacement_m": 0.0, "cheese_within_tolerance": True,
                         "unknown": []}

        jobs_by_id = {
            "job_wine": {"job_id": "job_wine", "capability_id": "wine_to_rack", "state": "completed",
                         "ended_reason": "completed", "error": None, "success": True, "steps": 1,
                         "phase": "done", "held_objects": [], "run_dir": None},
            "job_bowl": {"job_id": "job_bowl", "capability_id": "bowl_to_plate", "state": "completed",
                         "ended_reason": "completed", "error": None, "success": True, "steps": 1,
                         "phase": "done", "held_objects": [], "run_dir": None},
        }

        class FakeSession:
            def create_session(self, scene_id, seed=0, init_state_index=0):
                return {"ok": True, "session_id": "sess-1"}

            def final_snapshot(self, goals):
                return {"ok": True, "snapshot": {"predicates": {}, "held_objects": [], "objects": {}}}

            def plan(self, request_id):
                return {"request_id": request_id, "state": "completed", "plan_success": True,
                        "completed_capability_ids": ["wine_to_rack", "bowl_to_plate"],
                        "pending_capability_ids": [],
                        "job_ids": ["job_wine", "job_bowl"]}

            def job(self, job_id):
                return jobs_by_id[job_id]

        svc = FakeSession()

        agent_result = {"request_id": "req-1", "run_ok": True, "chain_ok": True, "plan_success": True,
                        "decision": "completed", "hermes_invocations": 1, "execution_timeout": False,
                        "cancelled_by_user": False, "cancellation_pending": False, "wall_s": 1.0,
                        "error": None, "plan": {"state": "completed"}}

        class FakeRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run(self):
                return dict(agent_result)

        with mock.patch.object(pilot.paired, "_seed_model_rng",
                               return_value={"ok": True, "seeded": True}), \
             mock.patch.object(pilot, "score_gold", return_value=dict(gold_combined)), \
             mock.patch.object(pilot, "read_oracle_rows", return_value=([], [])), \
             mock.patch.object(pilot.run_agent, "Runner", FakeRunner), \
             mock.patch.object(pilot.run_agent, "ServiceClient", lambda *a, **k: None), \
             mock.patch.object(pilot.run_agent, "HermesRunner", lambda *a, **k: None), \
             mock.patch.object(pilot.run_agent, "SystemClock", lambda *a, **k: None), \
             mock.patch.object(pilot.pe, "_write_json_atomic", lambda *a, **k: None):
            entry = pilot.run_case(svc, tmp / "home", tmp, 1)

        self.assertFalse(entry["order_ok"])
        self.assertEqual(entry["actual_order"], ["wine_to_rack", "bowl_to_plate"])
        self.assertEqual(entry["expected_order"], ["bowl_to_plate", "wine_to_rack"])
        self.assertIsNotNone(entry["score"])
        self.assertTrue(entry["score"]["placement_success"])
        self.assertFalse(entry["score"]["combined_success"])


class InjectionTest(unittest.TestCase):
    """Backward-compatible grasp-module injection preserves the existing fallback."""

    def test_default_module_is_local_grasp(self):
        svc = object.__new__(gas.GraspAssistService)
        gas.GraspAssistService.__init__(svc)
        self.assertIs(svc._grasp_module, local_grasp)

    def test_injected_module_is_consumed(self):
        class FakeModule:
            marker = object()
        svc = object.__new__(gas.GraspAssistService)
        fake = FakeModule()
        gas.GraspAssistService.__init__(svc, grasp_module=fake)
        self.assertIs(svc._grasp_module, fake)

    def test_injected_module_drives_helper_construction(self):
        called = {}

        class FakeController:
            def __init__(self):
                called["built"] = True

        class FakeModule:
            LocalGraspController = FakeController

            @staticmethod
            def read_geometry(env):
                return None

        svc = object.__new__(gas.GraspAssistService)
        gas.GraspAssistService.__init__(svc, grasp_module=FakeModule)
        module = getattr(svc, "_grasp_module", local_grasp)
        module.LocalGraspController()
        self.assertTrue(called.get("built"))


if __name__ == "__main__":
    unittest.main()
