#!/usr/bin/env python3
"""GPU-free unit tests for the isolated grasp-guard validation runner.

Every service / policy / model interaction is faked: no CUDA, no Hermes, no model
load and no live environment is ever created.  The tests exercise the runner's
*contracts* only -- the CLI refusals, the single first-input fingerprint capture,
the read-only freeze probes, the operational-vs-physical classification, the
source/checkpoint metadata serialization and the constructor/model-seed/profile
ordering.
"""

from __future__ import annotations

import hashlib
import json
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

import guard_validation as gv  # noqa: E402
import paired_config_experiments as paired  # noqa: E402
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402


# A deterministic, well-formed physical-state digest (64 lowercase hex chars).
_VALID_HASH = "0123456789abcdef" * 4
_OTHER_HASH = "fedcba9876543210" * 4


def _probe(
    physical_hash=_VALID_HASH,
    total_steps=10,
    action_selection_count=3,
    action_queue_length=0,
    ok=True,
):
    """A well-formed freeze-probe payload (individual fields may be overridden)."""

    return {
        "ok": ok,
        "physical_hash": physical_hash,
        "total_steps": total_steps,
        "action_selection_count": action_selection_count,
        "action_queue_length": action_queue_length,
    }


def _profile_result(use_amp=True, num_steps=10, n_action_steps=1, ok=True):
    """A ``configure_profile`` readback with both applied mappings."""

    applied = {
        "policy.config": {"use_amp": use_amp, "num_steps": num_steps, "n_action_steps": n_action_steps},
        "policy.model.config": {
            "use_amp": use_amp,
            "num_steps": num_steps,
            "n_action_steps": n_action_steps,
        },
    }
    return {"ok": ok, "profile": "baseline_bf16", "applied": applied}


# --- shared fakes ------------------------------------------------------------


def _argv(tmp: str, **overrides) -> list:
    values = {
        "output": str(Path(tmp) / "report.json"),
        "run_root": str(Path(tmp) / "run-root"),
        "scene": wd.WINE_SCENE_ID,
        "state_index": "0",
        "model_seed": "5",
        "guard_mode": "enforce",
        "capabilities": "wine",
        "source_git_sha": "deadbeef",
    }
    values.update(overrides)
    argv: list = []
    for key, value in values.items():
        if value is None:
            continue
        argv += ["--" + key.replace("_", "-"), str(value)]
    return argv


def _parse(argv: list):
    parser = gv._build_parser()
    args = parser.parse_args(argv)
    gv._validate_args(args, parser)
    return args


def _make_service():
    """A ``GuardValidationService`` whose base ``__init__`` never runs torch."""

    with mock.patch.object(wd.WineDiagnosticService, "__init__", return_value=None):
        svc = gv.GuardValidationService()
    return svc


class _FakeSimState:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float64)

    def flatten(self):
        return self._values


class _FakeSim:
    def __init__(self):
        self._state = _FakeSimState([1.0, 2.0, 3.0, 4.0])
        self.get_state_calls = 0

    def get_state(self):
        self.get_state_calls += 1
        return self._state


class _FakeInner:
    def __init__(self):
        self.sim = _FakeSim()


class _FakeEnv:
    """A read-only environment stand-in that raises on any mutating call."""

    def __init__(self):
        self.inner = _FakeInner()
        self._env = types.SimpleNamespace(env=self.inner)
        self.step_calls = 0
        self.reset_calls = 0
        self.forward_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        raise AssertionError("env.step must never be called during the freeze")

    def reset(self, *args, **kwargs):
        self.reset_calls += 1
        raise AssertionError("env.reset must never be called during the freeze")

    def forward(self):
        self.forward_calls += 1
        raise AssertionError("env.forward must never be called during the freeze")


class _FakeFreezeService:
    def __init__(self, env, policy, total_steps=42, count=7):
        self._env = env
        self._v1 = types.SimpleNamespace(_policy=policy)
        self._total_steps = total_steps
        self.action_selection_count = count
        self._sessions = {"sess-1": types.SimpleNamespace(total_steps=total_steps)}
        self.first_fingerprint = None

    def _sync_work(self, kind, fn, timeout=60.0):
        return fn()


class _OrderFakeService:
    """Records the call order of the run orchestration."""

    def __init__(self, events, record):
        self.events = events
        self._record = record
        self._sessions = {}
        self._env = None
        self._v1 = types.SimpleNamespace(_policy=None)
        self.action_selection_count = 3
        self.first_fingerprint = {"combined_sha256": "abc"}
        self.first_input_evidence = None

    def start(self):
        self.events.append("start")

    def health(self):
        return {"ready": True, "worker_error": None, "model_revision": "rev"}

    def create_session(self, scene_id, seed=0, init_state_index=0):
        self.events.append("create_session")
        self._sessions["sess-1"] = self._record
        return {"ok": True, "session_id": "sess-1", "scene_version": 0}

    def arm_input_capture(self, *args, **kwargs):
        self.events.append("arm_input_capture")

    def configure_profile(self, name):
        self.events.append("configure_profile")
        applied = {"use_amp": True, "num_steps": 10, "n_action_steps": 1}
        return {
            "ok": True,
            "profile": name,
            "policy_present": True,
            "applied": {"policy.config": dict(applied), "policy.model.config": dict(applied)},
        }

    def _sync_work(self, kind, fn, timeout=60.0):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "internal_error", "detail": str(exc)}

    def _close_env(self):
        self.events.append("close_env")
        return {"ok": True}

    def stop(self):
        self.events.append("stop")


# --- 1. CLI contract ---------------------------------------------------------


class CliContractTests(unittest.TestCase):
    def test_scene_choices_are_the_exact_catalog_ids(self):
        self.assertEqual(gv.SCENE_CHOICES, (wd.WINE_SCENE_ID, wd.SHARED_SCENE_ID))
        self.assertEqual(gv.SCENE_CHOICES, ("wine_native_goal9", "goal_table"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                gv.main(_argv(tmp, scene="wine_native_goals9"))

    def test_relative_paths_are_refused(self):
        with self.assertRaises(SystemExit):
            _parse(_argv("/nonexistent-dir", output="relative/report.json"))
        with self.assertRaises(SystemExit):
            _parse(_argv("/nonexistent-dir", run_root="relative/run-root"))

    def test_existing_output_and_run_root_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing_output = Path(tmp) / "report.json"
            existing_output.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                _parse(_argv(tmp, output=str(existing_output)))
            existing_root = Path(tmp) / "run-root"
            existing_root.mkdir()
            with self.assertRaises(SystemExit):
                _parse(
                    _argv(
                        tmp,
                        output=str(Path(tmp) / "fresh.json"),
                        run_root=str(existing_root),
                    )
                )

    def test_native_scene_requires_wine_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                _parse(_argv(tmp, scene=wd.WINE_SCENE_ID, capabilities="wine_bowl"))

    def test_goal_table_permits_wine_and_wine_bowl(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                _parse(_argv(tmp, scene=wd.SHARED_SCENE_ID, capabilities="wine")).capabilities,
                "wine",
            )
            self.assertEqual(
                _parse(
                    _argv(tmp, scene=wd.SHARED_SCENE_ID, capabilities="wine_bowl")
                ).capabilities,
                "wine_bowl",
            )

    def test_unknown_guard_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                gv.main(_argv(tmp, guard_mode="shadow"))

    def test_budget_and_timeout_are_fixed(self):
        self.assertEqual(gv.BUDGET, 300)
        self.assertEqual(gv.TIMEOUT_S, 900.0)
        parser = gv._build_parser()
        option_strings = {opt for action in parser._actions for opt in action.option_strings}
        self.assertNotIn("--budget", option_strings)
        self.assertNotIn("--timeout", option_strings)

    def test_help_does_not_build_a_service(self):
        with mock.patch.object(gv, "GuardValidationService") as mock_cls:
            with self.assertRaises(SystemExit):
                gv.main(["--help"])
            mock_cls.assert_not_called()


# --- 2. first-input fingerprint capture --------------------------------------


class SelectActionCaptureTests(unittest.TestCase):
    def test_fingerprint_captured_exactly_once_and_action_unchanged(self):
        svc = _make_service()
        batch = {"observation.state": [1, 2, 3], "task": "put the wine bottle on the rack"}
        snapshot = json.dumps(batch, sort_keys=True)
        sentinel = object()
        with mock.patch.object(
            paired, "fingerprint_batch", return_value={"combined_sha256": "fp"}
        ) as fingerprint, mock.patch.object(
            wd.WineDiagnosticService, "_select_action", return_value=sentinel
        ) as base:
            first = svc._select_action(batch)
            second = svc._select_action(batch)

        self.assertEqual(fingerprint.call_count, 1)
        self.assertEqual(svc.action_selection_count, 2)
        self.assertEqual(svc.first_fingerprint, {"combined_sha256": "fp"})
        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(base.call_count, 2)
        # The batch was never mutated by the capture.
        self.assertEqual(json.dumps(batch, sort_keys=True), snapshot)

    def test_input_evidence_matches_input_record_schema(self):
        svc = _make_service()
        record = types.SimpleNamespace(
            initial_state_hash="sha-init",
            xml_sha="sha-xml",
            camera_names=["camera_top", "camera_wrist"],
        )
        svc._sessions = {"sess-1": record}
        svc._env = types.SimpleNamespace(control_freq=20)
        svc.arm_input_capture("sess-1", 3, 7, "baseline_bf16")

        fingerprint = {
            "cameras": [],
            "state": {},
            "task": ["put the wine bottle on the rack"],
            "task_sha256": "task-hash",
            "combined_sha256": "combined",
            "errors": [],
        }
        with mock.patch.object(paired, "fingerprint_batch", return_value=fingerprint), mock.patch.object(
            wd.WineDiagnosticService, "_select_action", return_value=object()
        ):
            svc._select_action({})

        captured = svc.first_input_evidence
        self.assertEqual(captured["state_index"], 3)
        self.assertEqual(captured["model_seed"], 7)
        self.assertEqual(captured["profile"], "baseline_bf16")
        # The physical facts come from the REAL session record / live worker.
        self.assertEqual(captured["initial_state_sha"], "sha-init")
        self.assertEqual(captured["xml_sha"], "sha-xml")
        self.assertEqual(captured["camera_names"], ["camera_top", "camera_wrist"])
        self.assertEqual(captured["control_frequency_hz"], 20)

        comparable = paired._input_record(captured)
        self.assertEqual(comparable["state_index"], 3)
        self.assertEqual(comparable["initial_state_sha"], "sha-init")
        self.assertEqual(comparable["xml_sha"], "sha-xml")
        self.assertEqual(comparable["camera_names"], ["camera_top", "camera_wrist"])
        self.assertEqual(comparable["control_frequency_hz"], 20)
        self.assertEqual(comparable["combined_sha256"], "combined")

    def test_missing_context_stays_unknown(self):
        svc = _make_service()
        with mock.patch.object(
            paired, "fingerprint_batch", return_value={"combined_sha256": "fp"}
        ), mock.patch.object(wd.WineDiagnosticService, "_select_action", return_value=object()):
            svc._select_action({})
        captured = svc.first_input_evidence
        self.assertIsNone(captured["state_index"])
        self.assertIsNone(captured["initial_state_sha"])
        self.assertIsNone(captured["xml_sha"])
        self.assertEqual(captured["camera_names"], [])
        self.assertIsNone(captured["control_frequency_hz"])
        self.assertEqual(captured["fingerprint"], {"combined_sha256": "fp"})


# --- 3. read-only freeze probes ----------------------------------------------


class FreezeProbeTests(unittest.TestCase):
    def test_freeze_probes_are_read_only_with_exact_sleep(self):
        env = _FakeEnv()
        policy = types.SimpleNamespace(_queues={"action": [0.0, 1.0, 2.0, 3.0]})
        svc = _FakeFreezeService(env, policy)

        with mock.patch.object(gv.time, "sleep") as sleep:
            result = gv._run_freeze_probes(svc, "sess-1")

        sleep.assert_called_once_with(gv.FREEZE_SLEEP_S)
        self.assertEqual(gv.FREEZE_SLEEP_S, 0.25)
        self.assertTrue(result["equal"])
        self.assertEqual(
            result["probe_1"]["physical_hash"], result["probe_2"]["physical_hash"]
        )
        self.assertIsInstance(result["probe_1"]["physical_hash"], str)
        self.assertEqual(result["probe_1"]["total_steps"], 42)
        self.assertEqual(result["probe_1"]["action_selection_count"], 7)
        self.assertEqual(result["probe_1"]["action_queue_length"], 4)
        # No mutating simulator call happened during the freeze.
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(env.reset_calls, 0)
        self.assertEqual(env.forward_calls, 0)

    def test_fallback_hash_matches_service_state_sha(self):
        env = _FakeEnv()
        self.assertEqual(gv._direct_state_sha(env), service.state_sha(env))

    def test_action_queue_length_uses_installed_key_without_resetting(self):
        policy = types.SimpleNamespace(_queues={"action": [1, 2, 3]}, reset=mock.Mock())
        self.assertEqual(gv._action_queue_length(policy), 3)
        self.assertEqual(list(policy._queues["action"]), [1, 2, 3])
        policy.reset.assert_not_called()

        # A mapping whose only key is a non-default action key still resolves.
        single = types.SimpleNamespace(_queues={"act": [1, 2]})
        self.assertEqual(gv._action_queue_length(single), 2)

        # A module-level ACTION constant is honoured.
        module_name = "_gv_fake_policy_module"
        fake_module = types.ModuleType(module_name)
        fake_module.ACTION = "act"
        sys.modules[module_name] = fake_module
        try:
            policy_cls = type("_FakePolicy", (), {})
            policy_cls.__module__ = module_name
            constant_policy = policy_cls()
            constant_policy._queues = {"act": [1, 2, 3, 4, 5]}
            self.assertEqual(gv._action_queue_length(constant_policy), 5)
        finally:
            del sys.modules[module_name]

        self.assertIsNone(gv._action_queue_length(None))
        self.assertIsNone(gv._action_queue_length(types.SimpleNamespace(_queues=None)))


# --- 3b. strict probe equality (fail closed) ---------------------------------


class ProbeEqualityTests(unittest.TestCase):
    def test_error_and_null_pairs_are_not_equal(self):
        error_probe = {"ok": False, "error": "boom"}
        self.assertFalse(gv._probe_equal(error_probe, dict(error_probe)))
        self.assertFalse(gv._probe_equal(None, None))
        self.assertFalse(gv._probe_equal(_probe(), None))
        self.assertFalse(gv._probe_equal("error", "error"))

    def test_valid_same_hashes_and_integer_counts_are_equal(self):
        first = _probe(
            physical_hash=_VALID_HASH,
            total_steps=0,
            action_selection_count=5,
            action_queue_length=2,
        )
        second = _probe(
            physical_hash=_VALID_HASH,
            total_steps=0,
            action_selection_count=5,
            action_queue_length=2,
        )
        self.assertTrue(gv._probe_equal(first, second))

    def test_mismatched_counts_are_not_equal(self):
        base = _probe(total_steps=7, action_selection_count=4, action_queue_length=1)
        self.assertFalse(gv._probe_equal(base, _probe(total_steps=8, action_selection_count=4, action_queue_length=1)))
        self.assertFalse(gv._probe_equal(base, _probe(total_steps=7, action_selection_count=5, action_queue_length=1)))
        self.assertFalse(gv._probe_equal(base, _probe(total_steps=7, action_selection_count=4, action_queue_length=2)))
        self.assertFalse(gv._probe_equal(base, _probe(physical_hash=_OTHER_HASH)))

    def test_ok_must_be_exactly_true(self):
        for bad_ok in (None, False, 1, "true", 0, "yes"):
            self.assertFalse(
                gv._probe_equal(_probe(ok=bad_ok), _probe(ok=bad_ok))
            )
        missing_ok = _probe()
        missing_ok.pop("ok")
        self.assertFalse(gv._probe_equal(missing_ok, dict(missing_ok)))

    def test_malformed_hashes_are_rejected(self):
        for bad_hash in (
            None,
            "",
            "abc",
            _VALID_HASH.upper(),
            _VALID_HASH[:-1],
            _VALID_HASH + "0",
            _VALID_HASH + "\n",
            _VALID_HASH + "\r\n",
            "z" * 64,
            "0x" + _VALID_HASH[:60],
            b"0" * 64,
            123,
        ):
            self.assertFalse(
                gv._probe_equal(_probe(physical_hash=bad_hash), _probe(physical_hash=bad_hash))
            )
        missing = _probe()
        missing.pop("physical_hash")
        self.assertFalse(gv._probe_equal(missing, dict(missing)))

    def test_coerced_and_negative_counts_are_rejected(self):
        for bad in ("10", 10.0, True, False, None, -1, -5):
            self.assertFalse(gv._probe_equal(_probe(total_steps=bad), _probe(total_steps=bad)))
        for field in ("total_steps", "action_selection_count", "action_queue_length"):
            payload = _probe()
            payload.pop(field)
            self.assertFalse(gv._probe_equal(payload, dict(payload)))
        self.assertFalse(
            gv._probe_equal(_probe(action_queue_length=None), _probe(action_queue_length=None))
        )


# --- 3c. strict profile readback (fail closed) -------------------------------


class ProfileReadbackTests(unittest.TestCase):
    def test_valid_readback_passes(self):
        self.assertTrue(gv._profile_readback_ok(_profile_result()))
        self.assertTrue(gv._profile_readback_ok(_profile_result(use_amp=True)))

    def test_ok_must_be_exactly_true(self):
        for bad_ok in (None, False, 1, "true", "yes"):
            self.assertFalse(gv._profile_readback_ok(_profile_result(ok=bad_ok)))
        self.assertFalse(gv._profile_readback_ok({"applied": _profile_result()["applied"]}))

    def test_missing_either_readback_fails(self):
        result = _profile_result()
        del result["applied"]["policy.config"]
        self.assertFalse(gv._profile_readback_ok(result))

        result = _profile_result()
        del result["applied"]["policy.model.config"]
        self.assertFalse(gv._profile_readback_ok(result))

        self.assertFalse(gv._profile_readback_ok({"ok": True}))
        self.assertFalse(gv._profile_readback_ok({"ok": True, "applied": None}))
        self.assertFalse(gv._profile_readback_ok({"ok": True, "applied": {}}))

    def test_string_and_fractional_values_are_rejected(self):
        for bad in ("10", 10.0, 10.5):
            self.assertFalse(gv._profile_readback_ok(_profile_result(num_steps=bad)))
        for bad in ("1", 1.0):
            self.assertFalse(gv._profile_readback_ok(_profile_result(n_action_steps=bad)))

    def test_bool_as_int_is_rejected(self):
        self.assertFalse(gv._profile_readback_ok(_profile_result(num_steps=True)))
        self.assertFalse(gv._profile_readback_ok(_profile_result(n_action_steps=False)))

    def test_use_amp_must_be_a_genuine_bool(self):
        for bad in (1, 0, "true", "True", None):
            self.assertFalse(gv._profile_readback_ok(_profile_result(use_amp=bad)))

    def test_malformed_applied_mappings_are_rejected(self):
        result = _profile_result()
        result["applied"]["policy.config"] = "not-a-dict"
        self.assertFalse(gv._profile_readback_ok(result))

        result = _profile_result()
        result["applied"]["policy.model.config"] = None
        self.assertFalse(gv._profile_readback_ok(result))

        result = _profile_result()
        result["applied"]["policy.config"].pop("num_steps")
        self.assertFalse(gv._profile_readback_ok(result))


# --- 4. plan result collection ----------------------------------------------


class ResultCollectionTests(unittest.TestCase):
    def _plan(self, record=None, **plan_overrides):
        plan = {
            "state": "blocked",
            "plan_success": False,
            "completed_capability_ids": [],
            "pending_capability_ids": ["bowl_to_plate"],
            "job_ids": ["j1"],
        }
        plan.update(plan_overrides)
        out = {
            "request_id": "r1",
            "submitted": True,
            "submit_error": None,
            "timed_out": False,
            "cancel_nonterminal": False,
            "cancelled": False,
            "wall_s": 2.0,
            "plan": plan,
        }
        if record:
            out.update(record)
        return out

    def test_failed_grasp_and_skipped_bowl_are_physical(self):
        evidence = {
            "job_id": "j1",
            "available": True,
            "job": {
                "capability_id": "wine_to_rack",
                "state": "completed",
                "ended_reason": "failed_grasp",
                "error": None,
                "success": False,
                "steps": 40,
                "total_steps": 40,
                "grasp_guard_mode": "enforce",
                "grasp_stage": "failed_grasp",
                "grasp_guard_status": {"blocked": True},
            },
            "wine_telemetry_path": "/t/wine_telemetry.jsonl",
            "wine_diagnostic_path": None,
            "events_path": "/t/events.jsonl",
            "result_path": None,
            "rollout_path": None,
            "latest_png": None,
        }
        with mock.patch.object(wd, "_job_evidence", return_value=evidence):
            result = gv._collect_plan_result(
                object(), self._plan(), ("wine_to_rack", "bowl_to_plate")
            )

        self.assertEqual(result["operational_errors"], [])
        self.assertEqual(len(result["physical_failures"]), 1)
        self.assertIn("failed_grasp", result["physical_failures"][0])
        self.assertEqual(result["skipped_capabilities"], ["bowl_to_plate"])
        self.assertEqual(result["executed_capabilities"], ["wine_to_rack"])
        job = result["jobs"][0]
        self.assertEqual(job["ended_reason"], "failed_grasp")
        self.assertEqual(job["steps"], 40)
        self.assertEqual(job["guard"]["grasp_guard_mode"], "enforce")
        self.assertEqual(job["guard"]["grasp_stage"], "failed_grasp")
        self.assertEqual(job["wine_telemetry_path"], "/t/wine_telemetry.jsonl")
        self.assertEqual(job["events_path"], "/t/events.jsonl")

    def test_budget_exhausted_is_physical_only(self):
        evidence = {
            "job_id": "j1",
            "available": True,
            "job": {
                "capability_id": "wine_to_rack",
                "state": "completed",
                "ended_reason": "budget_exhausted",
                "error": None,
                "success": False,
            },
        }
        with mock.patch.object(wd, "_job_evidence", return_value=evidence):
            result = gv._collect_plan_result(object(), self._plan(), ("wine_to_rack",))
        self.assertEqual(result["operational_errors"], [])
        self.assertIn("budget_exhausted", result["physical_failures"][0])

    def test_job_error_is_operational(self):
        evidence = {
            "job_id": "j1",
            "available": True,
            "job": {
                "capability_id": "wine_to_rack",
                "state": "error",
                "ended_reason": "error",
                "error": "worker blew up",
                "success": None,
            },
        }
        with mock.patch.object(wd, "_job_evidence", return_value=evidence):
            result = gv._collect_plan_result(object(), self._plan(), ("wine_to_rack",))
        self.assertEqual(result["physical_failures"], [])
        self.assertTrue(any("job_error" in err for err in result["operational_errors"]))

    def test_timeout_and_cancel_are_operational(self):
        with mock.patch.object(
            wd, "_job_evidence", return_value={"job_id": "j1", "available": False}
        ):
            timed_out = gv._collect_plan_result(
                object(),
                self._plan(record={"timed_out": True}, job_ids=[]),
                ("wine_to_rack",),
            )
            self.assertTrue(any("plan_timeout" in err for err in timed_out["operational_errors"]))

            cancelled = gv._collect_plan_result(
                object(),
                self._plan(record={"cancel_nonterminal": True}, job_ids=[], state="cancelled"),
                ("wine_to_rack",),
            )
            self.assertTrue(
                any("cancel_nonterminal" in err for err in cancelled["operational_errors"])
            )

            plan_cancelled = gv._collect_plan_result(
                object(), self._plan(job_ids=[], state="cancelled"), ("wine_to_rack",)
            )
            self.assertTrue(
                any("plan_cancelled" in err for err in plan_cancelled["operational_errors"])
            )


# --- 5. metadata serialization ----------------------------------------------


class MetadataTests(unittest.TestCase):
    def test_source_sha256_covers_every_pinned_file(self):
        digests = gv._source_sha256()
        self.assertEqual(set(gv.SOURCE_FILES), set(digests))
        runner_digest = digests["guard_validation.py"]
        self.assertIsNotNone(runner_digest)
        self.assertRegex(runner_digest, r"^[0-9a-f]{64}$")
        # The exact runner bytes are hashed.
        self.assertEqual(
            runner_digest,
            hashlib.sha256((_SCENE_DEMO / "guard_validation.py").read_bytes()).hexdigest(),
        )
        json.dumps(digests)  # must be JSON-serializable

    def test_checkpoint_sha256_matches_pinned_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = b"pinned model.safetensors bytes"
            (Path(tmp) / gv.CHECKPOINT_FILENAME).write_bytes(payload)
            self.assertEqual(
                gv._checkpoint_sha256(tmp), hashlib.sha256(payload).hexdigest()
            )
            json.dumps({"checkpoint_sha256": gv._checkpoint_sha256(tmp)})
        # A missing checkpoint is unknown, never a fabricated digest.
        self.assertIsNone(gv._checkpoint_sha256(str(Path(tempfile.gettempdir()) / "no-such-dir")))


# --- 6. orchestration ordering and exit codes --------------------------------


class RunValidationOrderingTests(unittest.TestCase):
    def _run(self, tmp, **overrides):
        events: list = []
        record = types.SimpleNamespace(
            initial_state_hash="init-sha",
            xml_sha="xml-sha",
            camera_names=["camera_top", "camera_wrist"],
        )
        fake_service = _OrderFakeService(events, record)
        args = types.SimpleNamespace(
            output=str(Path(tmp) / "report.json"),
            run_root=str(Path(tmp) / "run-root"),
            scene=wd.WINE_SCENE_ID,
            capabilities=overrides.get("capabilities", "wine"),
            guard_mode=overrides.get("guard_mode", "enforce"),
            state_index=2,
            model_seed=9,
            source_git_sha="deadbeef",
        )
        override_errors = overrides.get("operational_errors")
        collected = {
            "plan": {"state": "completed", "plan_success": True},
            "jobs": [],
            "declared_capabilities": list(gv._capability_ids(args.capabilities)),
            "executed_capabilities": ["wine_to_rack"],
            "skipped_capabilities": [],
            "operational_errors": list(override_errors) if override_errors else [],
            "physical_failures": overrides.get("physical_failures", []),
        }

        seed_result = overrides.get("seed_result")
        freeze_result = overrides.get(
            "freeze_result", {"sleep_s": 0.25, "equal": True}
        )

        def _seed(service_, seed):
            events.append("seed_model_rng")
            if seed_result is not None:
                return seed_result
            return {"ok": True, "seeded": True, "seed": seed}

        def _submit(service_, session_id, capability_ids, budget, audit, request_id, timeout, rationale):
            events.append("submit_plan")
            self.assertEqual(budget, gv.BUDGET)
            self.assertEqual(timeout, gv.TIMEOUT_S)
            self.assertEqual(session_id, "sess-1")
            self.assertEqual(list(capability_ids), list(gv._capability_ids(args.capabilities)))
            return {
                "request_id": request_id,
                "submitted": True,
                "submit_error": None,
                "timed_out": False,
                "cancel_nonterminal": False,
                "cancelled": False,
                "wall_s": 1.0,
                "plan": {
                    "state": "completed",
                    "plan_success": True,
                    "completed_capability_ids": ["wine_to_rack"],
                    "pending_capability_ids": [],
                    "job_ids": [],
                },
            }

        with mock.patch.object(gv, "GuardValidationService") as mock_cls, mock.patch.object(
            paired, "_seed_model_rng", side_effect=_seed
        ), mock.patch.object(pe, "_submit_and_wait", side_effect=_submit), mock.patch.object(
            gv, "_collect_plan_result", return_value=collected
        ), mock.patch.object(
            gv, "_run_freeze_probes", return_value=freeze_result
        ):
            mock_cls.return_value = fake_service
            report = gv.run_validation(args)

        return report, events, mock_cls, args

    def test_constructor_keywords_and_seed_before_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, events, mock_cls, args = self._run(tmp)
            output_written = Path(args.output).is_file()

        kwargs = mock_cls.call_args.kwargs
        self.assertEqual(kwargs.get("model_path"), service.DEFAULT_MODEL_PATH)
        self.assertNotIn("model", kwargs)
        self.assertEqual(kwargs.get("run_root"), args.run_root)
        self.assertEqual(kwargs.get("completion_mode"), "release_verified")
        self.assertEqual(kwargs.get("grasp_guard_mode"), "enforce")

        self.assertIn("create_session", events)
        self.assertIn("seed_model_rng", events)
        self.assertIn("configure_profile", events)
        self.assertIn("submit_plan", events)
        self.assertLess(events.index("create_session"), events.index("seed_model_rng"))
        self.assertLess(events.index("seed_model_rng"), events.index("configure_profile"))
        self.assertLess(events.index("configure_profile"), events.index("submit_plan"))
        self.assertIn("stop", events)

        self.assertTrue(report["ok"])
        self.assertIsNone(report["fatal_error"])
        self.assertEqual(report["session"]["initial_state_sha"], "init-sha")
        self.assertEqual(report["session"]["xml_sha"], "xml-sha")
        self.assertEqual(report["session"]["camera_names"], ["camera_top", "camera_wrist"])
        self.assertTrue(report["freeze"]["equal"])
        # The report was written atomically to the fresh CLI output path.
        self.assertTrue(output_written)

    def test_wine_bowl_plan_uses_the_bowl_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _events, _mock_cls, _args = self._run(
                tmp, capabilities="wine_bowl", guard_mode="off"
            )
        self.assertEqual(
            report["declared_capabilities"], ["wine_to_rack", "bowl_to_plate"]
        )

    def test_operational_error_sets_ok_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _events, _mock_cls, _args = self._run(
                tmp, operational_errors=["job_error: job=j1 state=error"]
            )
            output_written = Path(_args.output).is_file()
        self.assertFalse(report["ok"])
        self.assertIn("job_error: job=j1 state=error", report["operational_errors"])
        self.assertTrue(output_written)

    def test_unseeded_model_rng_blocks_profile_and_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, events, _mock_cls, _args = self._run(
                tmp, seed_result={"ok": True, "seeded": False, "reason": "no_policy"}
            )
        self.assertFalse(report["ok"])
        self.assertIn("seed_model_rng", events)
        self.assertNotIn("configure_profile", events)
        self.assertNotIn("submit_plan", events)
        self.assertIsNotNone(report["fatal_error"])
        self.assertIn("model_rng_seed", report["fatal_error"])
        # The complete raw seeding result is preserved verbatim.
        self.assertEqual(report["model_rng_seed"]["result"]["reason"], "no_policy")
        self.assertFalse(report["model_rng_seed"]["seeded"])

    def test_seed_ok_and_seeded_must_be_exactly_true(self):
        for bad_seed in (
            {"ok": True, "seeded": 1},
            {"ok": True, "seeded": "true"},
            {"ok": False, "seeded": True},
            {"seeded": True},
            "not-a-dict",
        ):
            with tempfile.TemporaryDirectory() as tmp:
                report, events, _mock_cls, _args = self._run(tmp, seed_result=bad_seed)
            self.assertFalse(report["ok"], bad_seed)
            self.assertNotIn("configure_profile", events, bad_seed)
            self.assertNotIn("submit_plan", events, bad_seed)

    def test_valid_physical_failure_with_valid_freeze_stays_physical(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _events, _mock_cls, _args = self._run(
                tmp, physical_failures=["failed_grasp: job=j1 capability=wine_to_rack"]
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["operational_errors"], [])
        self.assertIn("failed_grasp", report["physical_failures"][0])
        self.assertTrue(report["freeze"]["equal"])

    def test_changed_freeze_is_operational_and_exit_1(self):
        changed = {
            "sleep_s": gv.FREEZE_SLEEP_S,
            "probe_1": _probe(total_steps=7),
            "probe_2": _probe(total_steps=8),
            "equal": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            report, _events, _mock_cls, _args = self._run(tmp, freeze_result=changed)
            argv = _argv(str(Path(tmp) / "cli"))
            with mock.patch.object(gv, "run_validation", return_value=report):
                exit_code = gv.main(argv)
        self.assertFalse(report["ok"])
        self.assertIn("freeze_probe_invalid_or_changed", report["operational_errors"])
        # The full freeze evidence is preserved (never fabricated or dropped).
        self.assertFalse(report["freeze"]["equal"])
        self.assertEqual(report["freeze"]["probe_1"]["total_steps"], 7)
        self.assertEqual(exit_code, 1)

    def test_non_dict_freeze_is_operational(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _events, _mock_cls, _args = self._run(tmp, freeze_result=None)
        self.assertFalse(report["ok"])
        self.assertIn("freeze_probe_invalid_or_changed", report["operational_errors"])
        self.assertIsNone(report["freeze"])

    def test_main_exit_code_follows_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _argv(tmp)
            with mock.patch.object(gv, "run_validation", return_value={"ok": True}):
                self.assertEqual(gv.main(argv), 0)
            with mock.patch.object(
                gv, "run_validation", return_value={"ok": False, "operational_errors": ["x"]}
            ):
                self.assertEqual(gv.main(argv), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
