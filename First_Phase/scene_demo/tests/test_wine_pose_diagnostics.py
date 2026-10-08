#!/usr/bin/env python3
"""Focused, GPU-free unit tests for :mod:`wine_pose_diagnostics`.

Nothing here starts a model, a worker, CUDA or a live environment: only the pure
calibration/view helpers are exercised for real, while the heavy service parent
methods are replaced with ``unittest.mock`` objects.  The ``scene_demo`` sibling
directory is inserted on ``sys.path`` before the runner is imported, mirroring
the existing test import pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCENE_DEMO = Path(__file__).resolve().parents[1]
if str(SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(SCENE_DEMO))

import preparation_diagnostics as pd  # noqa: E402
import service  # noqa: E402
import skill_context_diagnostics as context  # noqa: E402
import wine_pose_diagnostics as wpd  # noqa: E402


BLOCKED_MESSAGE = (
    "trial_exception: Traceback (most recent call last):\n"
    '  File "x.py", line 1, in f\n'
    "    raise PreparationBlocked(...)\n"
    "preparation_diagnostics.PreparationBlocked: physical preparation failed "
    "before the wine subgoal: reason=protection_violation detail=x"
)


def _valid_calibration() -> dict:
    return {
        "ok": True,
        "preparation": {"ok": True},
        "native_target": {
            "target_source": wpd.TARGET_SOURCE,
            "position": [0.1, 0.2, 0.3],
            "orientation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "state_sha": wpd.NATIVE_STATE_SHA,
        },
    }


class _FakeService:
    """A GPU-free service stub: it only counts its lifecycle calls.

    It deliberately carries NO ``_close_env`` attribute (so the campaign's
    best-effort close is skipped) and never loads a model, starts a worker or
    touches CUDA.
    """

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class WinePoseDiagnosticsTests(unittest.TestCase):
    # -- the fixed plan -------------------------------------------------------

    def test_campaign_plan_is_fixed_order(self):
        self.assertEqual(
            wpd.build_campaign_plan(),
            [
                {"entry": "A", "model_seed": 0},
                {"entry": "B", "model_seed": 0},
                {"entry": "B", "model_seed": 1},
                {"entry": "A", "model_seed": 1},
            ],
        )

    # -- process-local preparation limits -------------------------------------

    def test_preparation_limits_values_and_restore(self):
        before = (
            pd.LIFT_STAGE_MAX_ACTIONS,
            pd.ALIGN_STAGE_MAX_ACTIONS,
            pd.AUX_ACTION_CAP,
        )
        with wpd.preparation_limits() as record:
            self.assertEqual(pd.LIFT_STAGE_MAX_ACTIONS, 120)
            self.assertEqual(pd.ALIGN_STAGE_MAX_ACTIONS, 180)
            self.assertEqual(pd.AUX_ACTION_CAP, 300)
            self.assertEqual(record["lift_stage_max_actions"], 120)
            self.assertEqual(record["align_stage_max_actions"], 180)
            self.assertEqual(record["aux_action_cap"], 300)
        self.assertEqual(
            (pd.LIFT_STAGE_MAX_ACTIONS, pd.ALIGN_STAGE_MAX_ACTIONS, pd.AUX_ACTION_CAP),
            before,
        )

    def test_preparation_limits_restored_on_exception(self):
        before = (
            pd.LIFT_STAGE_MAX_ACTIONS,
            pd.ALIGN_STAGE_MAX_ACTIONS,
            pd.AUX_ACTION_CAP,
        )
        with self.assertRaises(RuntimeError):
            with wpd.preparation_limits():
                self.assertEqual(pd.AUX_ACTION_CAP, 300)
                raise RuntimeError("boom")
        self.assertEqual(
            (pd.LIFT_STAGE_MAX_ACTIONS, pd.ALIGN_STAGE_MAX_ACTIONS, pd.AUX_ACTION_CAP),
            before,
        )

    # -- the calibration contract ---------------------------------------------

    def _write_and_load(self, payload: dict):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return wpd.load_calibration(path)

    def test_load_calibration_accepts_orientation(self):
        loaded = self._write_and_load(_valid_calibration())
        self.assertEqual(loaded["position"], [0.1, 0.2, 0.3])
        self.assertEqual(
            loaded["orientation_matrix"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(loaded["state_sha"], wpd.NATIVE_STATE_SHA)
        self.assertEqual(loaded["target_source"], wpd.TARGET_SOURCE)

    def test_load_calibration_rejects_orientation_matrix_only(self):
        payload = _valid_calibration()
        native = payload["native_target"]
        native["orientation_matrix"] = native.pop("orientation")
        with self.assertRaises(ValueError):
            self._write_and_load(payload)

    def test_load_calibration_rejects_invalid_fields(self):
        def ok_false(payload):
            payload["ok"] = False

        def bad_state(payload):
            payload["native_target"]["state_sha"] = "deadbeef"

        def bad_source(payload):
            payload["native_target"]["target_source"] = "other"

        def bad_shape(payload):
            payload["native_target"]["orientation"] = [[1.0, 0.0], [0.0, 1.0]]

        def nonfinite(payload):
            payload["native_target"]["orientation"][0][0] = float("nan")

        for name, mutate in (
            ("ok_false", ok_false),
            ("bad_state", bad_state),
            ("bad_source", bad_source),
            ("bad_shape", bad_shape),
            ("nonfinite", nonfinite),
        ):
            payload = _valid_calibration()
            mutate(payload)
            with self.subTest(case=name), self.assertRaises(ValueError):
                self._write_and_load(payload)

    # -- the service constructor ----------------------------------------------

    def test_constructor_rejects_invalid_native_target(self):
        with mock.patch.object(pd.PreparedService, "__init__", return_value=None):
            with self.assertRaises(ValueError):
                wpd.PoseDiagnosticService(native_target=None)
            with self.assertRaises(ValueError):
                wpd.PoseDiagnosticService(
                    native_target={
                        "position": [1.0, 2.0, 3.0],
                        "orientation_matrix": [
                            [float("nan"), 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                )

    def test_constructor_copies_valid_native_target(self):
        native = {
            "position": np.array([1.0, 2.0, 3.0]),
            "orientation_matrix": np.eye(3),
            "state_sha": "sha",
            "target_source": "src",
        }
        with mock.patch.object(pd.PreparedService, "__init__", return_value=None):
            svc = wpd.PoseDiagnosticService(native_target=native)
        self.assertIsNot(svc.native_target, native)
        self.assertIsNot(svc.native_target["position"], native["position"])
        self.assertIsNot(svc.native_target["orientation_matrix"], native["orientation_matrix"])
        self.assertEqual(svc.native_target["position"].dtype, np.float64)
        self.assertFalse(svc.preparation_enabled)

    # -- _do_create_session ----------------------------------------------------

    def _bare_service(self) -> wpd.PoseDiagnosticService:
        return wpd.PoseDiagnosticService.__new__(wpd.PoseDiagnosticService)

    def test_do_create_session_missing_internal_target_raises_scene_error(self):
        for bad in (
            None,
            {"position": [1.0, 2.0, 3.0]},
            {"position": [1.0, 2.0], "orientation_matrix": np.eye(3)},
            {
                "position": np.array([1.0, 2.0, float("inf")]),
                "orientation_matrix": np.eye(3),
            },
        ):
            svc = self._bare_service()
            svc.native_target = bad
            with self.subTest(bad=bad), mock.patch.object(
                pd.PreparedService, "_do_create_session", return_value={"ok": True}
            ):
                with self.assertRaises(service.SceneError) as ctx:
                    svc._do_create_session(object(), 0, 0)
                self.assertEqual(ctx.exception.reason, "native_target_unavailable")

    def test_do_create_session_parent_failure_returned_unchanged(self):
        svc = self._bare_service()
        svc.native_target = None
        failure = {"ok": False, "reason": "controller_mismatch"}
        with mock.patch.object(
            pd.PreparedService, "_do_create_session", return_value=failure
        ):
            self.assertEqual(svc._do_create_session(object(), 0, 0), failure)

    def test_do_create_session_valid_overlay_copies_without_env(self):
        svc = self._bare_service()
        position = np.array([1.0, 2.0, 3.0])
        matrix = np.eye(3)
        svc.native_target = {"position": position, "orientation_matrix": matrix}
        env_sentinel = object()
        svc._env = env_sentinel
        with mock.patch.object(
            pd.PreparedService, "_do_create_session", return_value={"ok": True}
        ) as parent:
            created = svc._do_create_session(object(), 0, 0)
        self.assertEqual(created, {"ok": True})
        parent.assert_called_once()
        pose = svc._pre_bowl_pose
        self.assertEqual(pose["position"].dtype, np.float64)
        self.assertEqual(pose["orientation_matrix"].dtype, np.float64)
        self.assertIsNot(pose["position"], position)
        self.assertIsNot(pose["orientation_matrix"], matrix)
        np.testing.assert_array_equal(pose["position"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(pose["orientation_matrix"], np.eye(3))
        self.assertIs(svc._env, env_sentinel)

    # -- _sync_work dispatch ---------------------------------------------------

    def test_sync_work_a_bypasses_pd_hook(self):
        svc = self._bare_service()
        svc.preparation_enabled = False
        with mock.patch.object(
            context.ContextDiagnosticService, "_sync_work", return_value={"ok": True}
        ) as ctx_hook, mock.patch.object(
            pd.PreparedService, "_sync_work", return_value={"ok": True}
        ) as pd_hook:
            result = svc._sync_work("replay_prefix", lambda: None, 5.0)
        self.assertEqual(result, {"ok": True})
        ctx_hook.assert_called_once()
        pd_hook.assert_not_called()

    def test_sync_work_b_delegates_pd_hook_and_propagates_blocked(self):
        svc = self._bare_service()
        svc.preparation_enabled = True
        with mock.patch.object(
            context.ContextDiagnosticService, "_sync_work", return_value={"ok": True}
        ) as ctx_hook, mock.patch.object(
            pd.PreparedService, "_sync_work", side_effect=pd.PreparationBlocked("blocked")
        ) as pd_hook:
            with self.assertRaises(pd.PreparationBlocked):
                svc._sync_work("replay_prefix", lambda: None, 5.0)
        pd_hook.assert_called_once()
        ctx_hook.assert_not_called()

    # -- _branch_view ----------------------------------------------------------

    def _raw(self, **overrides) -> dict:
        raw = {
            "trial_id": "shared_after_bowl_m0",
            "condition": wpd.PREFIX_CONDITION,
            "model_seed": 0,
            "errors": [],
            "operational_errors": [],
            "jobs": [],
            "physical_failures": [],
            "after_bowl_state_sha": None,
        }
        raw.update(overrides)
        return raw

    def _wine_job(self, steps: int) -> dict:
        return {"job": {"capability_id": wpd.WINE_CAPABILITY_ID, "steps": steps}}

    def test_branch_view_a_aux_zero_and_combined_scoring(self):
        raw = self._raw(
            jobs=[self._wine_job(300)],
            strict_wine_success=True,
            final_bowl_predicate=True,
            final_bowl_strict=True,
            before_wine_state_sha="sha",
        )
        view = wpd._branch_view("A", 0, raw, None)
        self.assertEqual(view["status"], "ok")
        self.assertEqual(view["aux_actions"], 0)
        self.assertTrue(view["wine_attempted"])
        self.assertTrue(view["combined_success"])
        self.assertEqual(view["wine_action_count"], 300)

    def test_branch_view_b_physically_blocked_zeroes_wine(self):
        raw = self._raw(
            errors=[BLOCKED_MESSAGE],
            operational_errors=[BLOCKED_MESSAGE],
            jobs=[],
        )
        preparation = {
            "ok": False,
            "kind": "physical",
            "reason": "protection_violation",
            "aux_actions": 17,
        }
        view = wpd._branch_view("B", 0, raw, preparation)
        self.assertEqual(view["status"], "physical_preparation_failed")
        self.assertEqual(view["operational_errors"], [])
        self.assertFalse(view["wine_attempted"])
        self.assertEqual(view["wine_action_count"], 0)
        self.assertFalse(view["combined_success"])

    def test_branch_view_wine_attempted_follows_evidence_not_branch(self):
        # A planned B entry with a good preparation but NO wine job evidence is
        # not an attempt, and an attempted-but-failed wine row is not a success.
        raw = self._raw(jobs=[], strict_wine_success=None, final_bowl_strict=None)
        view = wpd._branch_view("B", 0, raw, {"ok": True, "aux_actions": 40})
        self.assertEqual(view["status"], "ok")
        self.assertFalse(view["wine_attempted"])
        self.assertFalse(view["combined_success"])

    def test_aggregate_counts_exclude_unattempted_blocked(self):
        good_a = wpd._branch_view(
            "A",
            0,
            self._raw(
                jobs=[self._wine_job(300)],
                strict_wine_success=True,
                final_bowl_strict=True,
            ),
            None,
        )
        blocked_b = wpd._branch_view(
            "B",
            0,
            self._raw(
                errors=[BLOCKED_MESSAGE],
                operational_errors=[BLOCKED_MESSAGE],
                jobs=[],
            ),
            {"ok": False, "kind": "physical", "reason": "protection_violation", "aux_actions": 17},
        )
        failed_b = wpd._branch_view(
            "B",
            1,
            self._raw(
                jobs=[self._wine_job(300)],
                strict_wine_success=False,
                final_bowl_strict=False,
            ),
            {"ok": True, "aux_actions": 40},
        )
        aggregate = wpd._aggregate([good_a, blocked_b, failed_b], 4)
        branch_a = aggregate["branches"]["A"]
        branch_b = aggregate["branches"]["B"]
        self.assertEqual(branch_a["n_wine_success"], 1)
        self.assertEqual(branch_a["n_combined_success"], 1)
        self.assertEqual(branch_b["n_wine_success"], 0)
        self.assertEqual(branch_b["n_combined_success"], 0)
        # The unattempted physically blocked row is not counted as a wine failure.
        self.assertEqual(branch_b["n_strict_wine_failure"], 1)
        self.assertEqual(branch_b["n_physical_preparation_failed"], 1)
        self.assertEqual(aggregate["n_wine_actions"], 600)

    # -- full GPU-free run_campaign entry smoke tests --------------------------

    def _write_real_inputs(self, tmp: str):
        """Write the REAL, loadable 102-row actions file and calibration report."""

        root = Path(tmp)
        actions_path = root / "input_actions.json"
        actions_path.write_text(
            json.dumps([[0.0] * 7 for _ in range(102)]), encoding="utf-8"
        )
        calibration_path = root / "calibration.json"
        calibration_path.write_text(json.dumps(_valid_calibration()), encoding="utf-8")
        return actions_path, calibration_path

    def _smoke_args(self, tmp: str, actions_path: Path, calibration_path: Path) -> argparse.Namespace:
        return argparse.Namespace(
            input_actions=str(actions_path),
            calibration=str(calibration_path),
            output=str(Path(tmp) / "report.json"),
            run_root=str(Path(tmp) / "run"),
            source_git_sha="0" * 40,
        )

    @staticmethod
    def _fake_trial_view(branch: str, model_seed: int) -> dict:
        return {
            "branch": branch,
            "model_seed": int(model_seed),
            "status": "ok",
            "operational_errors": [],
            "prefix_result": {"replay_action_count": 102},
            "aux_actions": 0 if branch == "A" else 125,
            "wine_action_count": 300,
            "wine_attempted": True,
            "strict_wine_success": False,
            "combined_success": False,
            "raw_stable_grasp_first_step": None,
            "final_bowl_predicate": True,
            "final_bowl_strict": True,
            "native_target_pose_error": None,
        }

    def test_run_campaign_complete_success_smoke(self):
        # REAL input loading, preregistration and report JSON serialization run
        # for real; only the model/service boundary and the per-trial runner are
        # replaced with GPU-free doubles.
        limits_before = (
            pd.LIFT_STAGE_MAX_ACTIONS,
            pd.ALIGN_STAGE_MAX_ACTIONS,
            pd.AUX_ACTION_CAP,
        )
        order: list = []

        def fake_run_branch_trial(
            svc, branch, model_seed, actions, run_root, context_seen, b_cross_seen
        ):
            order.append((branch, int(model_seed)))
            return self._fake_trial_view(branch, model_seed)

        health_record = {"ready": True, "active_request_id": None}
        with tempfile.TemporaryDirectory() as tmp:
            actions_path, calibration_path = self._write_real_inputs(tmp)
            args = self._smoke_args(tmp, actions_path, calibration_path)
            output = Path(args.output)
            run_root = Path(args.run_root)
            fake_service = _FakeService()
            with mock.patch.object(
                wpd, "_build_service", return_value=fake_service
            ), mock.patch.object(
                context, "_wait_ready", return_value={"ready": True}
            ), mock.patch.object(
                pd, "production_health_gate", return_value=(health_record, None)
            ), mock.patch.object(
                wpd, "_run_branch_trial", side_effect=fake_run_branch_trial
            ):
                result = wpd.run_campaign(args)

            self.assertTrue(result["ok"])
            self.assertEqual(order, [("A", 0), ("B", 0), ("B", 1), ("A", 1)])
            self.assertEqual(
                [(t["branch"], t["model_seed"]) for t in result["trials"]],
                [("A", 0), ("B", 0), ("B", 1), ("A", 1)],
            )
            self.assertEqual(fake_service.start_calls, 1)
            self.assertEqual(fake_service.stop_calls, 1)
            checks = result["health_checks"]
            self.assertEqual(len(checks), 5)
            self.assertEqual([c.get("trial_index") for c in checks[:4]], [1, 2, 3, 4])
            self.assertEqual(checks[4].get("phase"), "after_campaign")
            self.assertEqual(result["aggregate"]["n_replay_actions"], 408)
            self.assertEqual(result["aggregate"]["n_auxiliary_actions"], 250)
            self.assertEqual(result["aggregate"]["n_wine_actions"], 1200)
            # A physical failure (no wine success) still leaves the campaign
            # operationally ok.
            self.assertTrue(all(t["strict_wine_success"] is False for t in result["trials"]))
            self.assertEqual(
                (pd.LIFT_STAGE_MAX_ACTIONS, pd.ALIGN_STAGE_MAX_ACTIONS, pd.AUX_ACTION_CAP),
                limits_before,
            )
            self.assertTrue(output.is_file())
            prereg_path = run_root / "preregistration.json"
            self.assertTrue(prereg_path.is_file())
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(persisted["ok"])
            self.assertEqual(len(persisted["health_checks"]), 5)
            json.loads(prereg_path.read_text(encoding="utf-8"))

    def test_run_campaign_readiness_failure_smoke(self):
        health_record = {"ready": True, "active_request_id": None}
        with tempfile.TemporaryDirectory() as tmp:
            actions_path, calibration_path = self._write_real_inputs(tmp)
            args = self._smoke_args(tmp, actions_path, calibration_path)
            output = Path(args.output)
            fake_service = _FakeService()
            with mock.patch.object(
                wpd, "_build_service", return_value=fake_service
            ), mock.patch.object(
                context,
                "_wait_ready",
                return_value={"ready": False, "worker_error": "fixture readiness failure"},
            ), mock.patch.object(
                pd, "production_health_gate", return_value=(health_record, None)
            ), mock.patch.object(wpd, "_run_branch_trial") as run_trial:
                result = wpd.run_campaign(args)

            self.assertFalse(result["ok"])
            self.assertEqual(result["trials"], [])
            self.assertEqual(fake_service.stop_calls, 1)
            self.assertIn("fixture readiness failure", result["fatal_error"])
            run_trial.assert_not_called()
            self.assertTrue(output.is_file())
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(persisted["ok"])


if __name__ == "__main__":
    unittest.main()
