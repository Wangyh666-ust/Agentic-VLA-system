#!/usr/bin/env python3
"""GPU-free unit tests for ``skill_context_diagnostics``.

These tests never import ``torch``, never load a model, never create a real
simulator and never touch the network or production.  Every service is a pure
Python fake; the only ``scene_demo`` code exercised is the module's own pure
helpers (``load_actions``, ``build_campaign_plan``, ``_trial_scores``,
``replay_prefix`` dispatch and the CLI contract).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCENE_DEMO = _HERE.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import skill_context_diagnostics as diag  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402


def _row(value: float = 0.0) -> list[float]:
    return [float(value)] * diag.ACTION_DIM


def _actions(count: int = diag.REPLAY_ACTION_COUNT) -> list[list[float]]:
    return [_row(index / 100.0) for index in range(count)]


def _probe(z, grasped):  # noqa: ANN001, ANN202
    """One actual ``grasp_guard.read_probe``-shaped raw probe mapping."""

    position = [0.0, 0.0, float(z)] if z is not None else None
    return {
        "objects": {wd.WINE_OBJECT_ID: {"position": position, "grasped": grasped}},
        "eef_position": None,
        "predicates": {},
        "gripper_qpos": None,
        "gap": None,
    }


def _complete_fingerprint() -> dict:
    """A fingerprint that satisfies ``paired.fingerprint_is_complete``."""

    return {
        "cameras": [
            {"key": "observation.images.a", "raw_sha256": "aa"},
            {"key": "observation.images.b", "raw_sha256": "bb"},
        ],
        "state": {"raw_sha256": "cc"},
        "task": ["put the wine bottle on the rack"],
        "task_sha256": "dd",
        "combined_sha256": "ee",
    }


class _FakeSession:
    """A session record stand-in with only the attributes the replay reads."""

    def __init__(self, run_dir=None, total_steps: int = 0) -> None:
        self.run_dir = run_dir
        self.total_steps = total_steps


class _FakeEnv:
    """A fake environment whose ``step`` records calls and returns a tuple."""

    def __init__(self) -> None:
        self.steps: list[list] = []
        self.step_calls = 0

    def step(self, action):  # noqa: ANN001, ANN201
        self.step_calls += 1
        self.steps.append(np.asarray(action).tolist())
        return ("obs", 0.0, False, {})


class _FakeService:
    """A worker-less service stand-in whose ``_sync_work`` runs the thunk inline."""

    def __init__(self) -> None:
        self._env = _FakeEnv()
        self._sessions = {"s1": _FakeSession()}
        self._total_steps = 0
        self._last_obs = None
        self.run_root = None
        self.kinds: list[str] = []

    def _sync_work(self, kind, fn, timeout=60.0):  # noqa: ANN001, ANN201
        self.kinds.append(kind)
        return fn()


class LoadActionsTests(unittest.TestCase):
    """``load_actions`` accepts the real formats and rejects everything else."""

    def test_events_jsonl_records_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            lines = [
                json.dumps({"step": i, "action": _row(i / 50.0), "total_steps": i})
                for i in range(diag.REPLAY_ACTION_COUNT)
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rows = diag.load_actions(path)
        self.assertEqual(len(rows), diag.REPLAY_ACTION_COUNT)
        self.assertTrue(all(len(row) == diag.ACTION_DIM for row in rows))

    def test_whole_document_json_list_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.json"
            path.write_text(json.dumps(_actions()), encoding="utf-8")
            rows = diag.load_actions(path)
        self.assertEqual(rows, _actions())

    def test_wrapper_object_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.json"
            path.write_text(json.dumps({"actions": _actions()}), encoding="utf-8")
            rows = diag.load_actions(path)
        self.assertEqual(len(rows), diag.REPLAY_ACTION_COUNT)

    def test_blank_jsonl_lines_do_not_drop_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            lines = [json.dumps({"action": _row()}) for _ in range(diag.REPLAY_ACTION_COUNT)]
            path.write_text("\n\n".join(lines) + "\n\n", encoding="utf-8")
            rows = diag.load_actions(path)
        self.assertEqual(len(rows), diag.REPLAY_ACTION_COUNT)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            diag.load_actions("does/not/exist.jsonl")

    def test_malformed_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            path.write_text("{not json}\n" * diag.REPLAY_ACTION_COUNT, encoding="utf-8")
            with self.assertRaises(ValueError):
                diag.load_actions(path)

    def test_wrong_row_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.json"
            path.write_text(json.dumps(_actions(diag.REPLAY_ACTION_COUNT - 1)), encoding="utf-8")
            with self.assertRaises(ValueError):
                diag.load_actions(path)

    def test_wrong_row_width_is_rejected(self) -> None:
        rows = _actions()
        rows[0] = [0.0] * (diag.ACTION_DIM - 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                diag.load_actions(path)

    def test_non_numeric_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.jsonl"
            bad = json.dumps({"action": [0.0] * (diag.ACTION_DIM - 1) + ["oops"]})
            path.write_text("\n".join([bad] * diag.REPLAY_ACTION_COUNT), encoding="utf-8")
            with self.assertRaises(ValueError):
                diag.load_actions(path)

    def test_bool_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.json"
            rows = _actions()
            rows[0] = [True] + [0.0] * (diag.ACTION_DIM - 1)
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                diag.load_actions(path)

    def test_empty_document_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                diag.load_actions(path)


class CampaignPlanTests(unittest.TestCase):
    """The fixed six-trial order is exactly two model seeds x three conditions."""

    def test_plan_length_and_model_seed_order(self) -> None:
        plan = diag.build_campaign_plan()
        self.assertEqual(len(plan), diag.EXPECTED_TRIAL_COUNT)
        self.assertEqual(len(plan), 6)
        self.assertEqual([entry["model_seed"] for entry in plan], [0, 0, 0, 1, 1, 1])

    def test_plan_condition_order_is_fixed(self) -> None:
        plan = diag.build_campaign_plan()
        self.assertEqual(
            [entry["condition"] for entry in plan],
            [
                "native_goal9",
                "shared_initial",
                "shared_after_bowl",
                "native_goal9",
                "shared_initial",
                "shared_after_bowl",
            ],
        )

    def test_plan_scene_and_prefix_flags(self) -> None:
        plan = diag.build_campaign_plan()
        native = [entry for entry in plan if entry["condition"] == "native_goal9"]
        after = [entry for entry in plan if entry["condition"] == "shared_after_bowl"]
        self.assertTrue(all(entry["scene_id"] == wd.WINE_SCENE_ID for entry in native))
        self.assertTrue(all(entry["scene_id"] == wd.SHARED_SCENE_ID for entry in plan if entry["condition"] != "native_goal9"))
        self.assertTrue(all(entry["use_prefix"] is True for entry in after))
        self.assertTrue(all(not entry["use_prefix"] for entry in plan if entry["condition"] == "shared_initial"))

    def test_condition_order_constant(self) -> None:
        self.assertEqual(
            tuple(diag.CONDITION_ORDER),
            ("native_goal9", "shared_initial", "shared_after_bowl"),
        )


class TrialScoreTests(unittest.TestCase):
    """Strict (actual wine job) and semantic (shadow observer) stay separate."""

    def test_strict_true_and_final_semantic_false_are_separate(self) -> None:
        job_evidence = [{"job": {"capability_id": diag.WINE_CAPABILITY_ID, "success": True}}]
        summary = {"last_semantic_status": False}
        rows = [{"semantic_success": False}, {"semantic_success": False}]
        strict, final_semantic, ever_semantic = diag._trial_scores(job_evidence, summary, rows)
        self.assertIs(strict, True)
        self.assertIs(final_semantic, False)
        self.assertIs(ever_semantic, False)

    def test_strict_unknown_is_never_promoted_from_semantic(self) -> None:
        job_evidence = [{"job": {"capability_id": diag.WINE_CAPABILITY_ID, "success": None}}]
        summary = {"last_semantic_status": True}
        strict, final_semantic, ever_semantic = diag._trial_scores(job_evidence, summary, [])
        self.assertIsNone(strict)
        self.assertIs(final_semantic, True)
        self.assertIsNone(ever_semantic)

    def test_ever_semantic_is_independent_of_final(self) -> None:
        job_evidence = [{"job": {"capability_id": diag.WINE_CAPABILITY_ID, "success": False}}]
        summary = {"last_semantic_status": False}
        rows = [{"semantic_success": True}, {"semantic_success": False}]
        strict, final_semantic, ever_semantic = diag._trial_scores(job_evidence, summary, rows)
        self.assertIs(strict, False)
        self.assertIs(final_semantic, False)
        self.assertIs(ever_semantic, True)

    def test_non_wine_job_does_not_set_strict(self) -> None:
        job_evidence = [{"job": {"capability_id": "some_other_cap", "success": True}}]
        strict, _, _ = diag._trial_scores(job_evidence, {}, [])
        self.assertIsNone(strict)

    def test_unknown_semantic_status_is_none(self) -> None:
        strict, final_semantic, ever_semantic = diag._trial_scores([], {"last_semantic_status": None}, [])
        self.assertIsNone(strict)
        self.assertIsNone(final_semantic)
        self.assertIsNone(ever_semantic)


class ReplayPrefixTests(unittest.TestCase):
    """``replay_prefix`` checks the fixed origin SHA before ANY step."""

    def test_origin_mismatch_aborts_before_any_step(self) -> None:
        svc = _FakeService()
        with mock.patch.object(service, "state_sha", return_value="deadbeef" * 8):
            result = diag.replay_prefix(svc, "s1", _actions())
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "origin_state_sha_mismatch")
        self.assertEqual(svc._env.step_calls, 0)
        self.assertEqual(result["replay_action_count"], 0)
        self.assertEqual(result["origin_state_sha"], "deadbeef" * 8)
        self.assertEqual(result["expected_origin_state_sha"], diag.REPLAY_ORIGIN_SHA)

    def test_origin_match_steps_once_per_action_then_checks_final(self) -> None:
        svc = _FakeService()
        digests = [diag.REPLAY_ORIGIN_SHA, "00" * 32]

        def _fake_state_sha(_env):  # noqa: ANN001, ANN202
            return digests.pop(0)

        with mock.patch.object(service, "state_sha", side_effect=_fake_state_sha):
            result = diag.replay_prefix(svc, "s1", _actions())
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "final_state_sha_mismatch")
        self.assertEqual(result["origin_state_sha"], diag.REPLAY_ORIGIN_SHA)
        self.assertEqual(result["replay_action_count"], diag.REPLAY_ACTION_COUNT)
        self.assertEqual(svc._env.step_calls, diag.REPLAY_ACTION_COUNT)
        self.assertEqual(len(digests), 0)

    def test_invalid_action_count_aborts_without_stepping(self) -> None:
        svc = _FakeService()
        result = diag.replay_prefix(svc, "s1", _actions(diag.REPLAY_ACTION_COUNT - 1))
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "invalid_prefix_actions")
        self.assertEqual(svc._env.step_calls, 0)

    def test_replay_runs_through_the_worker(self) -> None:
        svc = _FakeService()
        with mock.patch.object(service, "state_sha", return_value="deadbeef" * 8):
            diag.replay_prefix(svc, "s1", _actions())
        self.assertEqual(svc.kinds, ["replay_prefix"])

    def test_unknown_session_is_reported(self) -> None:
        svc = _FakeService()
        result = diag.replay_prefix(svc, "missing", _actions())
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "unknown_session")

    def test_worker_error_dict_is_normalised(self) -> None:
        class _ErrorService(_FakeService):
            def _sync_work(self, kind, fn, timeout=60.0):  # noqa: ANN001, ANN201
                return {"ok": False, "reason": "worker_error", "detail": "boom"}

        result = diag.replay_prefix(_ErrorService(), "s1", _actions())
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "worker_error")
        self.assertEqual(result["replay_action_count"], 0)


class CliContractTests(unittest.TestCase):
    """``--help`` must never load a model, open CUDA or build the service."""

    def test_help_exits_zero_without_building_service(self) -> None:
        with mock.patch.object(diag, "ContextDiagnosticService") as service_cls:
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                with self.assertRaises(SystemExit) as caught:
                    diag.main(["--help"])
            self.assertEqual(caught.exception.code, 0)
            service_cls.assert_not_called()
        self.assertIn("--input-actions", stdout.getvalue())

    def test_missing_required_args_exits_two_without_building_service(self) -> None:
        with mock.patch.object(diag, "ContextDiagnosticService") as service_cls:
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as caught:
                    diag.main([])
            self.assertEqual(caught.exception.code, 2)
            service_cls.assert_not_called()

    def test_relative_paths_are_rejected_without_building_service(self) -> None:
        with mock.patch.object(diag, "ContextDiagnosticService") as service_cls:
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as caught:
                    diag.main(["--input-actions", "rel.jsonl", "--output", "rel.json", "--run-root", "rel"])
            self.assertEqual(caught.exception.code, 2)
            service_cls.assert_not_called()

    def test_existing_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actions = Path(tmp) / "a.json"
            actions.write_text(json.dumps(_actions()), encoding="utf-8")
            output = Path(tmp) / "out.json"
            output.write_text("{}", encoding="utf-8")
            with mock.patch.object(diag, "ContextDiagnosticService") as service_cls:
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    with self.assertRaises(SystemExit) as caught:
                        diag.main(
                            [
                                "--input-actions",
                                str(actions),
                                "--output",
                                str(output),
                                "--run-root",
                                str(Path(tmp) / "run"),
                            ]
                        )
                self.assertEqual(caught.exception.code, 2)
                service_cls.assert_not_called()

    def test_missing_input_actions_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(diag, "ContextDiagnosticService") as service_cls:
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    with self.assertRaises(SystemExit) as caught:
                        diag.main(
                            [
                                "--input-actions",
                                str(Path(tmp) / "nope.json"),
                                "--output",
                                str(Path(tmp) / "out.json"),
                                "--run-root",
                                str(Path(tmp) / "run"),
                            ]
                        )
                self.assertEqual(caught.exception.code, 2)
                service_cls.assert_not_called()


class RawStableGraspTests(unittest.TestCase):
    """The independent statistic reads the raw probe only, never the monitor."""

    def _rows(self, start_z, samples, stage="failed_grasp"):  # noqa: ANN001, ANN202
        rows = [
            {
                "phase": "start",
                "probe": _probe(start_z, None),
                "monitor": {"stage": "unknown", "grasp_confirmed_step": None, "failure_step": None},
            }
        ]
        for step, (z, grasped) in enumerate(samples, start=1):
            rows.append(
                {
                    "step": step,
                    "command": None,
                    "probe": _probe(z, grasped),
                    "monitor": {
                        "stage": stage,
                        "grasp_confirmed_step": None,
                        "failure_step": 5 if stage == "failed_grasp" else None,
                    },
                    "semantic": None,
                    "semantic_status": None,
                    "semantic_success": None,
                    "native_success": None,
                }
            )
        return rows

    def test_failed_grasp_monitor_still_reports_later_raw_stable_grasp(self) -> None:
        # The monitor reports failed_grasp with a null confirmation while five
        # later raw samples rise >= 0.02 m and are grasped: the independent
        # statistic must report the fifth confirming row's step.
        start_z = 0.50
        samples = [(0.50, False)] * 3 + [(0.53, True)] * 5
        rows = self._rows(start_z, samples, stage="failed_grasp")
        summary = diag.summarize_raw_stable_grasp(rows)
        self.assertEqual(summary["raw_stable_grasp_first_step"], 8)
        self.assertEqual(summary["raw_stable_grasp_start_z"], start_z)
        # The monitor field stays null: the statistic is independent of it.
        self.assertIsNone(rows[8]["monitor"]["grasp_confirmed_step"])
        self.assertEqual(rows[8]["monitor"]["stage"], "failed_grasp")

    def test_unknown_position_interrupts_the_streak(self) -> None:
        start_z = 0.50
        samples = [(0.53, True)] * 4 + [(None, True)] + [(0.53, True)] * 5
        rows = self._rows(start_z, samples)
        summary = diag.summarize_raw_stable_grasp(rows)
        self.assertEqual(summary["raw_stable_grasp_first_step"], 10)

    def test_nonboolean_grasp_resets_the_streak(self) -> None:
        start_z = 0.50
        samples = [(0.53, True), (0.53, True), (0.53, None), (0.53, True), (0.53, True)]
        rows = self._rows(start_z, samples)
        summary = diag.summarize_raw_stable_grasp(rows)
        self.assertIsNone(summary["raw_stable_grasp_first_step"])

    def test_unknown_initial_probe_leaves_statistic_null(self) -> None:
        rows = self._rows(None, [(0.53, True)] * 6)
        summary = diag.summarize_raw_stable_grasp(rows)
        self.assertIsNone(summary["raw_stable_grasp_first_step"])
        self.assertIsNone(summary["raw_stable_grasp_start_z"])

    def test_below_threshold_lift_is_not_a_stable_grasp(self) -> None:
        start_z = 0.50
        samples = [(0.51, True)] * 8  # only a 0.01 m rise
        rows = self._rows(start_z, samples)
        summary = diag.summarize_raw_stable_grasp(rows)
        self.assertIsNone(summary["raw_stable_grasp_first_step"])

    def test_non_list_input_stays_null(self) -> None:
        summary = diag.summarize_raw_stable_grasp(None)
        self.assertIsNone(summary["raw_stable_grasp_first_step"])


class _FakeTrackerService:
    """A stand-in carrying exactly the attributes ``_reset_trial_context`` clears."""

    def __init__(self) -> None:
        self._context_tracker = "stale-tracker"
        self._context_observer_summary = {"stale": True}
        self._diag_condition = None
        self.first_fingerprint = {"stale": True}
        self.action_selection_count = 7
        self.first_input_evidence = {"stale": True}
        self._first_input_captured = True


class TrialContextTests(unittest.TestCase):
    """The per-trial reset clears the persistent tracker and selects the oracle."""

    def test_reset_clears_persistent_tracker_and_selects_wine_oracle(self) -> None:
        svc = _FakeTrackerService()
        diag._reset_trial_context(svc)
        self.assertIsNone(svc._context_tracker)
        self.assertIsNone(svc._context_observer_summary)
        self.assertEqual(svc._diag_condition, wd.FINAL_ORACLE_KEY)
        self.assertIsNone(svc.first_fingerprint)
        self.assertEqual(svc.action_selection_count, 0)
        self.assertIsNone(svc.first_input_evidence)
        self.assertIs(svc._first_input_captured, False)


class _CaptureRecorder:
    """A service stand-in that records every ``arm_input_capture`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def arm_input_capture(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.calls.append((args, kwargs))
        return {"ok": True}


class CaptureContextTests(unittest.TestCase):
    """The capture is armed with the ACTUAL pre-wine state SHA, not the session SHA."""

    def test_arm_uses_before_wine_state_sha(self) -> None:
        svc = _CaptureRecorder()
        trial = {
            "before_wine_state_sha": "post-bowl-pre-wine-sha",
            "initial_state_sha": "original-session-sha",
            "xml_sha": "a" * 64,
            "camera_names": ["agentview", "wrist"],
            "control_frequency_hz": 20,
        }
        diag._arm_first_input_capture(svc, session_id="s1", model_seed=1, trial=trial)
        self.assertEqual(len(svc.calls), 1)
        args, kwargs = svc.calls[0]
        self.assertEqual(args[0], "s1")
        self.assertEqual(args[1], diag.INIT_STATE_INDEX)
        self.assertEqual(args[2], 1)
        self.assertEqual(kwargs["initial_state_sha"], "post-bowl-pre-wine-sha")
        self.assertNotEqual(kwargs["initial_state_sha"], trial["initial_state_sha"])
        self.assertEqual(kwargs["xml_sha"], "a" * 64)
        self.assertEqual(kwargs["camera_names"], ["agentview", "wrist"])


class FirstInputProvenanceTests(unittest.TestCase):
    """A missing XML SHA or an incomplete fingerprint is an operational error."""

    def test_none_none_is_not_evidence(self) -> None:
        errors = diag._first_input_operational_errors(None, None)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("xml_sha" in message for message in errors))
        self.assertTrue(any("fingerprint" in message for message in errors))

    def test_empty_xml_sha_is_rejected(self) -> None:
        errors = diag._first_input_operational_errors("   ", _complete_fingerprint())
        self.assertTrue(any("xml_sha" in message for message in errors))

    def test_incomplete_fingerprint_is_rejected(self) -> None:
        errors = diag._first_input_operational_errors("a" * 64, {"cameras": []})
        self.assertEqual(errors, ["first_fingerprint_incomplete"])

    def test_valid_xml_and_complete_fingerprint_accepted(self) -> None:
        self.assertEqual(
            diag._first_input_operational_errors("a" * 64, _complete_fingerprint()), []
        )


class CrossSeedComparisonTests(unittest.TestCase):
    """``before_wine_state_sha`` must match across model seeds (null is no match)."""

    def test_before_wine_state_sha_mismatch_is_flagged(self) -> None:
        seen: dict = {}
        ops: list[str] = []
        base = {
            "condition": "shared_initial",
            "xml_sha": "a" * 64,
            "first_fingerprint": _complete_fingerprint(),
            "before_wine_state_sha": "sha-one",
            "after_bowl_state_sha": None,
        }
        diag._compare_across_seeds(dict(base), seen, ops.append)
        diag._compare_across_seeds(dict(base, before_wine_state_sha="sha-two"), seen, ops.append)
        self.assertTrue(any("cross_seed_before_wine_state_sha" in message for message in ops))

    def test_null_before_wine_state_sha_is_not_a_match(self) -> None:
        seen: dict = {}
        ops: list[str] = []
        row = {
            "condition": "native_goal9",
            "xml_sha": "a" * 64,
            "first_fingerprint": _complete_fingerprint(),
            "before_wine_state_sha": None,
            "after_bowl_state_sha": None,
        }
        diag._compare_across_seeds(dict(row), seen, ops.append)
        diag._compare_across_seeds(dict(row), seen, ops.append)
        self.assertTrue(any("cross_seed_before_wine_state_sha" in message for message in ops))


class ModuleContractTests(unittest.TestCase):
    """Fixed constants and the required public interface are present."""

    def test_required_interface_exists(self) -> None:
        self.assertTrue(callable(diag.main))
        self.assertTrue(callable(diag.load_actions))
        self.assertTrue(callable(diag.replay_prefix))
        self.assertTrue(callable(diag.build_campaign_plan))
        self.assertTrue(issubclass(diag.ContextDiagnosticService, diag.gv.GuardValidationService))

    def test_fixed_constants(self) -> None:
        self.assertEqual(diag.GRASP_GUARD_MODE, "shadow")
        self.assertEqual(diag.COMPLETION_MODE, wd.COMPLETION_MODE)
        self.assertEqual(diag.PROFILE, wd.BASELINE_PROFILE)
        self.assertEqual(diag.WINE_CAPABILITY_ID, "wine_to_rack")
        self.assertEqual(diag.REPLAY_ACTION_COUNT, 102)
        self.assertEqual(diag.BUDGET, 300)
        self.assertEqual(len(diag.REPLAY_ORIGIN_SHA), 64)
        self.assertEqual(len(diag.REPLAY_FINAL_SHA), 64)
        self.assertEqual(diag.WINE_ORACLE_GOALS, [["on", "wine_bottle_1", "wine_rack_1_top_region"]])

    def test_raw_stable_grasp_contract(self) -> None:
        self.assertTrue(callable(diag.summarize_raw_stable_grasp))
        self.assertEqual(diag.RAW_STABLE_GRASP_WINDOW, 5)
        self.assertEqual(diag.RAW_STABLE_GRASP_LIFT_M, wd.LIFT_THRESHOLD_M)
        self.assertEqual(diag.RAW_STABLE_GRASP_LIFT_M, 0.02)
        record = diag.summarize_raw_stable_grasp([])
        self.assertIsNone(record["raw_stable_grasp_first_step"])
        self.assertEqual(record["raw_stable_grasp_window"], diag.RAW_STABLE_GRASP_WINDOW)
        self.assertEqual(record["raw_stable_grasp_source"], "shadow_observer_probe")
        self.assertIn("raw_stable_grasp", diag.LIMITATIONS)
        self.assertIn("monitor_confirmation_null", diag.LIMITATIONS)
        self.assertIn("semantic_window_vs_strict_stop", diag.LIMITATIONS)

    def test_health_url_uses_stdlib_defaults(self) -> None:
        self.assertEqual(
            diag.HEALTH_URL,
            "http://%s:%d/health" % (service.DEFAULT_HOST, service.DEFAULT_PORT),
        )
        self.assertEqual(diag.HEALTH_URL, "http://127.0.0.1:8767/health")

    def test_module_does_not_import_torch(self) -> None:
        self.assertFalse(hasattr(diag, "torch"))


if __name__ == "__main__":
    unittest.main()
