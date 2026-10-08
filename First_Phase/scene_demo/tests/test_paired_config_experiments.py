#!/usr/bin/env python3
"""GPU-free unit tests for ``scene_demo/paired_config_experiments.py``.

These tests never load a checkpoint, never build a LIBERO scene and never touch
CUDA or the network: they exercise the *real production helpers* against fake
simulator / tensor interfaces.  They prove, in particular, that

* ``fingerprint_batch`` records the exact shape, dtype and raw-byte SHA-256 of the
  two camera tensors, the ``observation.state`` tensor and the task text, hashes a
  BF16 payload without widening it (two bytes per element), and never mutates the
  batch;
* a missing/incomplete or mismatched first input raises *before* any prediction or
  ``env.step`` and registers an abortable error;
* the independent shadow observer never changes the returned step result and
  records an unknown probe/semantic as unknown (never as success);
* ``PassiveStepObserver`` calls the saved original step exactly once with the
  unchanged action object and returns the *same* result object, restoring the
  original ``env.step`` on the normal AND the exception path;
* the discovery grid is exactly 18 non-control trials plus one A/A duplicate of
  state 0 / model seed 0 / ``baseline_bf16`` immediately after the first trial, and
  the holdout grid never reselects;
* the pre-declared selection rule (strict successes, then post-assessment semantic
  successes, then the lowest median policy wall time, then the exact tie order
  ``baseline_bf16 > fp32 > fp32_h5``) is applied exactly, with the A/A control
  excluded;
* ``_run_trial`` treats a profile/plan operational failure as an abortable
  operational error with ZERO submitted plans and NO assessment, while a plain
  ``budget_exhausted`` task failure continues and does receive the 20-action
  assessment;
* the checkpoint audit fails closed unless the root holds exactly seven readable
  files, and the preregistration is a pure, model-free frozen record;
* the post-policy assessment drives the *real* ``wine_semantic.SemanticTracker``
  (exact ``update(sample)`` once per step): 20 known candidates complete ``True``,
  unknown evidence stays ``None`` in every row and the final status, a completed
  success followed by a failing end becomes ``False`` (never ``any(samples)``) and
  19 candidates are not enough;
* a semantic success is read only from the exact tracked ``semantic_success``; the
  retired generic aliases and a raw ``semantic_candidate`` are never success.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import grasp_guard  # noqa: E402
import paired_config_experiments as paired  # noqa: E402
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402
import wine_semantic  # noqa: E402


# --- small fake tensor / batch helpers ---------------------------------------


def _batch(n_cameras: int = 2, state=(0.1, 0.2, 0.3), task=(wd.WINE_INSTRUCTION,)):
    batch = {}
    for index in range(n_cameras):
        batch["observation.images.cam%d" % index] = np.full(
            (2, 2, 3), index + 1, dtype=np.uint8
        )
    batch["observation.state"] = np.asarray(state, dtype=np.float32)
    batch["task"] = list(task)
    return batch


def _batch_bytes(batch):
    """A comparable, immutable snapshot of every array payload in the batch."""

    return {
        key: (value.tobytes() if isinstance(value, np.ndarray) else repr(value))
        for key, value in batch.items()
    }


class _FakeByteView:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def numpy(self) -> np.ndarray:
        return np.frombuffer(self._data, dtype=np.uint8)


class _FakeTorchTensor:
    """A GPU-free BF16 stand-in: ``numpy()`` refuses, ``view(uint8)`` is exact."""

    def __init__(self, data: bytes, dtype_name: str = "bfloat16") -> None:
        self._data = bytes(data)
        self.dtype = dtype_name
        self.shape = (max(1, len(self._data) // 2),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def view(self, _dtype):
        return _FakeByteView(self._data)

    def numpy(self):
        raise TypeError("numpy cannot convert bfloat16 without a float() cast")

    def tobytes(self) -> bytes:
        return self._data


def _fake_torch_module() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.Tensor = _FakeTorchTensor
    module.uint8 = "uint8"
    return module


# --- input fingerprint --------------------------------------------------------


class FingerprintBatchTests(unittest.TestCase):
    def test_full_fingerprint_records_shape_dtype_and_raw_sha(self):
        batch = _batch()
        fingerprint = paired.fingerprint_batch(batch)
        self.assertTrue(paired.fingerprint_is_complete(fingerprint))
        self.assertEqual(fingerprint["errors"], [])
        self.assertEqual(len(fingerprint["cameras"]), 2)
        self.assertEqual(
            sorted(fingerprint["camera_keys"]),
            ["observation.images.cam0", "observation.images.cam1"],
        )
        for entry in fingerprint["cameras"]:
            self.assertEqual(entry["shape"], [2, 2, 3])
            self.assertEqual(entry["dtype"], "uint8")
            self.assertEqual(entry["raw_nbytes"], 12)
            self.assertIsNotNone(entry["raw_sha256"])
        self.assertEqual(fingerprint["state"]["shape"], [3])
        self.assertEqual(fingerprint["state"]["dtype"], "float32")
        self.assertIsNotNone(fingerprint["state"]["raw_sha256"])
        self.assertEqual(fingerprint["task"], [wd.WINE_INSTRUCTION])
        self.assertIsNotNone(fingerprint["task_sha256"])
        self.assertIsNotNone(fingerprint["combined_sha256"])

    def test_fingerprint_is_stable_for_equal_input_and_does_not_mutate(self):
        batch = _batch()
        before = _batch_bytes(batch)
        first = paired.fingerprint_batch(batch)
        second = paired.fingerprint_batch(batch)
        self.assertEqual(before, _batch_bytes(batch))  # never mutates the batch
        self.assertEqual(first["combined_sha256"], second["combined_sha256"])
        self.assertEqual(
            [entry["raw_sha256"] for entry in first["cameras"]],
            [entry["raw_sha256"] for entry in second["cameras"]],
        )

    def test_camera_byte_change_changes_the_combined_digest(self):
        batch = _batch()
        reference = paired.fingerprint_batch(batch)
        batch["observation.images.cam1"][0, 0, 0] = 99
        changed = paired.fingerprint_batch(batch)
        self.assertNotEqual(
            reference["cameras"][1]["raw_sha256"], changed["cameras"][1]["raw_sha256"]
        )
        self.assertNotEqual(reference["combined_sha256"], changed["combined_sha256"])

    def test_incomplete_input_is_never_complete(self):
        one_camera = _batch(n_cameras=1)
        fingerprint = paired.fingerprint_batch(one_camera)
        self.assertFalse(paired.fingerprint_is_complete(fingerprint))
        self.assertTrue(any("two camera" in e for e in fingerprint["errors"]))

        missing_state = _batch()
        del missing_state["observation.state"]
        state_fingerprint = paired.fingerprint_batch(missing_state)
        self.assertFalse(paired.fingerprint_is_complete(state_fingerprint))
        self.assertTrue(any("state" in e for e in state_fingerprint["errors"]))

    def test_non_mapping_batch_is_incomplete(self):
        fingerprint = paired.fingerprint_batch(None)
        self.assertFalse(paired.fingerprint_is_complete(fingerprint))
        self.assertFalse(fingerprint["is_mapping"])

    def test_bf16_raw_bytes_are_preserved_without_widening(self):
        data = bytes([1, 2, 3, 4, 5, 6])  # three 2-byte BF16 elements
        tensor = _FakeTorchTensor(data)
        with mock.patch.dict(sys.modules, {"torch": _fake_torch_module()}):
            raw = paired._raw_bytes(tensor)
        self.assertEqual(raw, data)
        self.assertEqual(len(raw), 6)  # 2 bytes/element: never widened to fp32
        described = paired._describe_tensor("camera", tensor)
        self.assertEqual(described["dtype"], "bfloat16")
        self.assertEqual(described["raw_nbytes"], 6)

    def test_raw_bytes_edge_cases(self):
        self.assertIsNone(paired._raw_bytes(None))
        np_bytes = paired._raw_bytes(np.zeros(4, dtype=np.float16))
        self.assertEqual(len(np_bytes), 8)  # exact 2-byte elements preserved
        duck = types.SimpleNamespace(tobytes=lambda: b"abc")
        self.assertEqual(paired._raw_bytes(duck), b"abc")


class PairingRegistryTests(unittest.TestCase):
    def _evidence(self, state_index, batch):
        return {
            "state_index": state_index,
            "initial_state_sha": "state-sha-%s" % state_index,
            "xml_sha": "xml-sha",
            "camera_names": ["agentview", "wrist"],
            "control_frequency_hz": 20,
            "fingerprint": paired.fingerprint_batch(batch),
        }

    def test_first_record_registers_and_an_identical_repeat_matches(self):
        registry: dict = {}
        paired.validate_input(registry, self._evidence(0, _batch()))
        self.assertIn(0, registry)
        # An identical repeat for the same state is accepted (no raise).
        paired.validate_input(registry, self._evidence(0, _batch()))
        self.assertEqual(registry.get("errors", []), [])

    def test_mismatch_raises_and_is_registered(self):
        registry: dict = {}
        paired.validate_input(registry, self._evidence(0, _batch()))
        other_state = _batch(state=(9.0, 9.0, 9.0))
        with self.assertRaises(RuntimeError) as caught:
            paired.validate_input(registry, self._evidence(0, other_state))
        self.assertIn("paired_input_mismatch", str(caught.exception))
        self.assertTrue(any("paired_input_mismatch" in e for e in registry["errors"]))

    def test_different_states_are_independent(self):
        registry: dict = {}
        paired.validate_input(registry, self._evidence(0, _batch()))
        # A different state may legitimately differ: its first record is accepted.
        paired.validate_input(registry, self._evidence(1, _batch(state=(1.0, 1.0, 1.0))))
        self.assertEqual(set(registry) - {"errors"}, {0, 1})

    def test_incomplete_input_raises_before_prediction(self):
        registry: dict = {}
        evidence = self._evidence(0, _batch(n_cameras=1))
        with self.assertRaises(RuntimeError) as caught:
            paired.validate_input(registry, evidence)
        self.assertIn("paired_input_incomplete", str(caught.exception))
        self.assertNotIn(0, registry)  # nothing was registered from a bad input


class SemanticHelperTests(unittest.TestCase):
    def test_semantic_flag_reads_only_an_explicit_bool_semantic_success(self):
        self.assertIs(paired._semantic_flag(True), True)
        self.assertIs(paired._semantic_flag(False), False)
        self.assertIs(paired._semantic_flag(None), None)
        self.assertIs(paired._semantic_flag({"semantic_success": True}), True)
        self.assertIs(paired._semantic_flag({"semantic_success": False}), False)
        self.assertIsNone(paired._semantic_flag({"semantic_success": None}))
        self.assertIsNone(paired._semantic_flag({"semantic_success": 1}))  # not a bool
        self.assertIsNone(paired._semantic_flag({}))
        self.assertIsNone(paired._semantic_flag({"semantic_candidate": True}))
        self.assertIsNone(paired._semantic_flag("semantic_success"))
        # The retired generic aliases are no longer success signals.
        for alias in ("success", "satisfied", "task_success", "is_success", "goal_true"):
            self.assertIsNone(paired._semantic_flag({alias: True}))
            self.assertFalse(paired.semantic_is_success({alias: True}))

    def test_semantic_is_success_truth_table(self):
        self.assertTrue(paired.semantic_is_success(True))
        self.assertTrue(paired.semantic_is_success({"semantic_success": True}))
        self.assertFalse(paired.semantic_is_success(None))
        self.assertFalse(paired.semantic_is_success(False))
        self.assertFalse(paired.semantic_is_success({}))
        self.assertFalse(paired.semantic_is_success({"semantic_success": False}))
        self.assertFalse(paired.semantic_is_success({"semantic_success": None}))
        self.assertFalse(paired.semantic_is_success({"semantic_candidate": True}))
        self.assertFalse(paired.semantic_is_success({"success": True}))
        self.assertFalse(paired.semantic_is_success("semantic_success"))

    def test_tracker_update_calls_update_once_with_the_sample_only(self):
        self.assertIsNone(paired.tracker_update(None, 1, {"a": 1}))

        class _Real:
            def __init__(self):
                self.calls = []

            def update(self, sample):
                self.calls.append(sample)
                return {"semantic_success": True}

        real = _Real()
        self.assertEqual(
            paired.tracker_update(real, 7, {"a": 1}), {"semantic_success": True}
        )
        # Exactly once, with the sample only: the ``step`` argument is never
        # forwarded and there is no signature guessing, introspection or retry.
        self.assertEqual(real.calls, [{"a": 1}])

        class _Raises:
            def __init__(self):
                self.calls = 0

            def update(self, sample):
                self.calls += 1
                raise ValueError("boom")

        raises = _Raises()
        self.assertIsNone(paired.tracker_update(raises, 1, {}))
        self.assertEqual(raises.calls, 1)  # a failed update is never retried

        class _WrongArity:
            def __init__(self):
                self.calls = 0

            def update(self, step, sample):  # the old two-argument signature
                self.calls += 1
                return {"semantic_success": True}

        wrong = _WrongArity()
        # A tracker whose update needs two arguments is never guessed: the call
        # fails closed and the body is never entered (no alternate call is made).
        self.assertIsNone(paired.tracker_update(wrong, 1, {"a": 1}))
        self.assertEqual(wrong.calls, 0)

        class _NoUpdate:
            pass

        self.assertIsNone(paired.tracker_update(_NoUpdate(), 1, {}))

    def test_median_odd_even_and_empty(self):
        self.assertEqual(paired._median([3, 1, 2]), 2)
        self.assertEqual(paired._median([4, 1, 3, 2]), 2.5)
        self.assertIsNone(paired._median([]))
        self.assertIsNone(paired._median(None))
        self.assertEqual(paired._median([1.5, None, "x", True]), 1.5)


# --- real ``wine_semantic`` module integration --------------------------------


def _raw_semantic_sample(
    *,
    linear_speed=0.0,
    angular_speed=0.0,
    rack_contact=True,
    support_contact=True,
    held_objects=None,
    observation_complete=True,
    standard_predicate=False,
    semantic_candidate=None,
    contacts=None,
):
    """A complete raw sample shaped exactly like ``read_wine_semantic``.

    Every field the real module emits is present.  ``semantic_candidate`` defaults
    to the module's own ``score_wine_semantic`` result and may be overridden so a
    test can prove a raw candidate is never a tracked completion.
    """

    sample = {
        "spec_id": wine_semantic.SPEC.spec_id,
        "object_id": wine_semantic.OBJECT_ID,
        "target_id": wine_semantic.TARGET_ID,
        "rack_contact": rack_contact,
        "support_contact": support_contact,
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
        "held_objects": [] if held_objects is None else list(held_objects),
        "observation_complete": observation_complete,
        "standard_predicate": standard_predicate,
        "semantic_candidate": semantic_candidate,
        "contacts": list(contacts or []),
    }
    if semantic_candidate is None:
        sample["semantic_candidate"] = wine_semantic.score_wine_semantic(sample)
    return sample


class _ReadFeeder:
    """A ``paired.read_semantic`` stand-in returning a fixed, ordered sequence.

    The assessment reads the environment once before the loop, once per action and
    once after the loop, so the sequence is consumed in that exact order and the
    last value is repeated if the sequence runs short (a fixed sample never
    exhausts it).
    """

    def __init__(self, samples):
        self._samples = list(samples)
        self.calls = 0

    def __call__(self, env):  # noqa: ARG002
        index = self.calls
        self.calls += 1
        if index < len(self._samples):
            return self._samples[index]
        return self._samples[-1] if self._samples else None


class _FakeAssessmentEnv:
    """A GPU-free fake env whose ``step`` returns a native five-tuple result."""

    def __init__(self, native_success=False):
        self.actions = []
        self.native_success = native_success

    def step(self, action):
        self.actions.append(action)
        return (
            {"obs": True},
            0.0,
            False,
            False,
            {"is_success": bool(self.native_success)},
        )


class _FakeAssessmentService:
    """The minimal service surface ``_do_assessment_work`` reads/writes."""

    def __init__(self, env):
        self._env = env
        self._last_obs = None


def _read_assessment_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_assessment(tmp, samples, *, native_success=False, tracker=None):
    """Run ``_do_assessment_work`` against a fake env and an ordered sample feed."""

    env = _FakeAssessmentEnv(native_success=native_success)
    svc = _FakeAssessmentService(env)
    tracker = tracker if tracker is not None else wine_semantic.SemanticTracker()
    feeder = _ReadFeeder(samples)
    with mock.patch.object(paired, "read_semantic", feeder), mock.patch.object(
        pe, "capture_snapshot", return_value={"predicates": {}, "objects": {}}
    ), mock.patch.object(pe, "strict_final_score", return_value=False):
        outcome = paired._do_assessment_work(svc, Path(tmp), tracker)
    return env, svc, tracker, feeder, outcome


class RealSemanticTrackerIntegrationTests(unittest.TestCase):
    """The 20-action assessment against the real ``wine_semantic`` tracker."""

    def test_twenty_known_candidates_complete_true(self):
        good = _raw_semantic_sample(standard_predicate=False)
        with tempfile.TemporaryDirectory() as tmp:
            env, svc, tracker, feeder, outcome = _run_assessment(
                tmp, [good] * 22, native_success=False
            )
            rows = _read_assessment_rows(outcome["path"])
        self.assertEqual(outcome["action_count"], paired.ASSESSMENT_ACTION_COUNT)
        self.assertEqual(len(env.actions), paired.ASSESSMENT_ACTION_COUNT)
        # Exactly 20 real neutral float32 seven-dimensional zero actions.
        for action in env.actions:
            self.assertEqual(action.dtype, np.float32)
            self.assertEqual(action.shape, (7,))
            self.assertTrue(np.all(action == 0.0))
        # The tracked status is False until the 20th consecutive known candidate.
        self.assertIs(rows[0]["semantic_success"], False)
        self.assertIs(rows[-2]["semantic_success"], False)
        self.assertIs(rows[-1]["semantic_success"], True)
        self.assertIs(outcome["semantic_before"], None)  # unknown before any sample
        self.assertIs(outcome["semantic_after"], True)
        self.assertIs(outcome["semantic_success"], True)
        self.assertEqual(tracker.candidate_streak, 20)
        # The raw pre/step/post samples are retained verbatim and separate.
        self.assertEqual(outcome["semantic_before_sample"], good)
        self.assertEqual(outcome["semantic_after_sample"], good)
        self.assertEqual(rows[-1]["semantic"], good)
        # A raw candidate is True yet is NOT itself a completion; the display-only
        # standard_predicate and a native False never change the tracked result.
        self.assertIs(rows[-1]["semantic"]["semantic_candidate"], True)
        self.assertIs(rows[-1]["semantic"]["standard_predicate"], False)
        self.assertIs(outcome["native_success_final"], False)
        self.assertIs(outcome["native_success_ever"], False)

    def test_unknown_evidence_is_none_in_every_row_and_the_final_status(self):
        for unknown in (
            _raw_semantic_sample(linear_speed=None),
            _raw_semantic_sample(angular_speed=None),
            _raw_semantic_sample(rack_contact=None),
            _raw_semantic_sample(support_contact=None),
            _raw_semantic_sample(observation_complete=False),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                env, svc, tracker, feeder, outcome = _run_assessment(
                    tmp, [unknown] * 22
                )
                rows = _read_assessment_rows(outcome["path"])
            self.assertEqual(len(rows), paired.ASSESSMENT_ACTION_COUNT)
            for row in rows:
                self.assertIsNone(row["semantic_success"])
            self.assertIsNone(outcome["semantic_before"])
            self.assertIsNone(outcome["semantic_after"])
            self.assertIsNone(outcome["semantic_success"])
            # The raw sample is still retained verbatim (with its null fields),
            # separate from the tracked status which stays unknown.
            self.assertEqual(outcome["semantic_after_sample"], unknown)
            self.assertIsNone(outcome["semantic_after_sample"]["semantic_candidate"])
            self.assertEqual(tracker.candidate_streak, 0)
            self.assertIsNone(tracker.semantic_success)

    def test_a_completed_success_followed_by_a_failing_end_becomes_false(self):
        good = _raw_semantic_sample()
        failing = _raw_semantic_sample(held_objects=[wine_semantic.OBJECT_ID])
        tracker = wine_semantic.SemanticTracker()
        for _ in range(19):
            tracker.update(good)
        self.assertEqual(tracker.candidate_streak, 19)
        with tempfile.TemporaryDirectory() as tmp:
            step_samples = [good] * 19 + [failing]
            env, svc, tracker, feeder, outcome = _run_assessment(
                tmp, [good] + step_samples + [failing], tracker=tracker
            )
            rows = _read_assessment_rows(outcome["path"])
        self.assertIs(outcome["semantic_before"], False)  # 19 < 20 samples
        self.assertIs(rows[0]["semantic_success"], True)  # the 20th candidate completes
        self.assertTrue(any(row["semantic_success"] is True for row in rows))
        self.assertIs(rows[-1]["semantic_success"], False)  # the failing end
        self.assertIs(outcome["semantic_after"], False)
        self.assertIs(outcome["semantic_success"], False)  # LAST status, never any()
        self.assertIs(outcome["semantic_after_sample"], failing)

    def test_nineteen_candidates_are_not_a_success(self):
        good = _raw_semantic_sample()
        failing = _raw_semantic_sample(held_objects=[wine_semantic.OBJECT_ID])
        with tempfile.TemporaryDirectory() as tmp:
            step_samples = [good] * 19 + [failing]
            env, svc, tracker, feeder, outcome = _run_assessment(
                tmp, [good] + step_samples + [failing]
            )
            rows = _read_assessment_rows(outcome["path"])
        self.assertFalse(any(row["semantic_success"] is True for row in rows))
        self.assertIs(outcome["semantic_after"], False)
        self.assertIs(outcome["semantic_success"], False)
        self.assertEqual(tracker.candidate_streak, 0)

    def test_the_tracker_requires_twenty_consecutive_true_candidates(self):
        tracker = wine_semantic.SemanticTracker()
        good = _raw_semantic_sample()
        for _ in range(19):
            status = tracker.update(good)
            self.assertIs(status["semantic_success"], False)
        self.assertIs(tracker.semantic_success, False)  # 19 candidates: not enough
        status = tracker.update(good)  # the 20th consecutive candidate
        self.assertIs(status["semantic_success"], True)
        self.assertEqual(tracker.candidate_streak, 20)
        # A failing 21st sample un-completes the earlier success.
        status = tracker.update(_raw_semantic_sample(held_objects=[wine_semantic.OBJECT_ID]))
        self.assertIs(status["semantic_success"], False)
        self.assertIs(tracker.semantic_success, False)

    def test_a_raw_candidate_without_a_tracked_window_is_not_a_completion(self):
        raw = _raw_semantic_sample(semantic_candidate=True)
        self.assertIs(raw["semantic_candidate"], True)
        # A raw candidate is not a tracked status: no ``semantic_success`` key.
        self.assertIsNone(paired._semantic_flag(raw))
        self.assertFalse(paired.semantic_is_success(raw))
        tracker = wine_semantic.SemanticTracker()
        status = tracker.update(raw)  # a single candidate is a length-1 streak
        self.assertIs(status["semantic_success"], False)
        self.assertIs(paired._semantic_flag(status), False)
        self.assertFalse(paired.semantic_is_success(status))

    def test_one_action_and_one_tracker_update_per_step(self):
        good = _raw_semantic_sample()
        tracker = wine_semantic.SemanticTracker()
        seen = []
        original_update = tracker.update

        def counting_update(sample):
            seen.append(sample)
            return original_update(sample)

        tracker.update = counting_update
        with tempfile.TemporaryDirectory() as tmp:
            env, svc, tracker, feeder, outcome = _run_assessment(
                tmp, [good] * 22, tracker=tracker
            )
            rows = _read_assessment_rows(outcome["path"])
        self.assertEqual(len(env.actions), paired.ASSESSMENT_ACTION_COUNT)
        self.assertEqual(len(seen), paired.ASSESSMENT_ACTION_COUNT)  # one per step
        # before + 20 steps + after reads, and exactly one tracked update per step.
        self.assertEqual(feeder.calls, paired.ASSESSMENT_ACTION_COUNT + 2)
        self.assertIs(outcome["semantic_after"], True)
        self.assertIs(rows[-1]["semantic_success"], True)
        # A full-True assessment yields a streak of exactly 20; if the separate
        # pre-assessment read had advanced the candidate it would be 21, so the
        # before read never feeds the tracker.
        self.assertEqual(tracker.candidate_streak, 20)


# --- independent shadow observer ---------------------------------------------


def _probe(z=0.5, grasped=False, eef_z=0.6, predicate=False):
    return {
        "objects": {paired.WINE_OBJECT_ID: {"position": [0.0, 0.0, z], "grasped": grasped}},
        "eef_position": [0.0, 0.0, eef_z],
        "predicates": {paired.WINE_GOAL_KEY: predicate},
        "gripper_qpos": None,
        "gap": None,
    }


class GuardSemanticObserverTests(unittest.TestCase):
    def test_start_creates_a_monitor_and_writes_the_start_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = types.SimpleNamespace()
            with mock.patch.object(grasp_guard, "read_probe", return_value=_probe()), \
                 mock.patch.object(paired, "read_semantic", return_value=None):
                observer = paired.GuardSemanticObserver(env, Path(tmp))
                observer.start()
                self.assertIsInstance(observer.monitor, grasp_guard.GraspMonitor)
                self.assertIsNotNone(observer.initial_status)
                sample = observer.on_action(
                    np.array([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32),
                    ("obs", 0.0, False, False, {"is_success": False}),
                )
                observer.close()
            lines = observer.path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)  # the start line + one action line
        self.assertEqual(sample["command"], 0.5)
        self.assertEqual(sample["native_success"], False)
        self.assertFalse(sample["semantic_success"])  # unknown is never success
        summary = observer.summary()
        self.assertTrue(summary["observer_only"])
        self.assertEqual(summary["n_samples"], 1)
        self.assertEqual(summary["semantic_unknown_count"], 1)

    def test_unknown_probe_is_unknown_and_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = types.SimpleNamespace()
            with mock.patch.object(grasp_guard, "read_probe", side_effect=RuntimeError("no sim")), \
                 mock.patch.object(paired, "read_semantic", return_value=None):
                observer = paired.GuardSemanticObserver(env, Path(tmp))
                observer.start()
                sample = observer.on_action(np.zeros(7, dtype=np.float32), None)
                observer.close()
        self.assertIsNone(sample["probe"]["objects"][paired.WINE_OBJECT_ID]["grasped"])
        self.assertIsNone(sample["native_success"])
        self.assertFalse(sample["semantic_success"])
        self.assertIsNone(sample["monitor"]["grasp_confirmed_step"])
        self.assertEqual(observer.semantic_unknown_count, 1)

    def test_grasp_confirmed_step_is_reported_from_the_shadow_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = types.SimpleNamespace()
            start_probe = _probe(z=0.50, grasped=False, eef_z=0.55)
            lifted_probe = _probe(z=0.60, grasped=True, eef_z=0.55)  # +0.10 m rise
            with mock.patch.object(
                grasp_guard, "read_probe", side_effect=[start_probe] + [lifted_probe] * 5
            ), mock.patch.object(paired, "read_semantic", return_value=None):
                observer = paired.GuardSemanticObserver(env, Path(tmp))
                observer.start()
                for _ in range(5):
                    observer.on_action(np.zeros(7, dtype=np.float32), None)
                observer.close()
        # The shadow monitor latches a confirmation; it is only *reported*.
        summary = observer.summary()
        self.assertEqual(summary["first_grasp_confirmed_step"], 5)
        self.assertTrue(summary["observer_only"])

    def test_semantic_known_count_requires_a_bool_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = types.SimpleNamespace()
            null_dict = {"semantic_candidate": None, "note": None, "confidence": None}
            with mock.patch.object(grasp_guard, "read_probe", return_value=_probe()), \
                 mock.patch.object(
                     paired,
                     "read_semantic",
                     side_effect=[True, null_dict, False, None],
                 ):
                observer = paired.GuardSemanticObserver(env, Path(tmp))
                observer.start()
                samples = [
                    observer.on_action(np.zeros(7, dtype=np.float32), None)
                    for _ in range(4)
                ]
                observer.close()
        # Only the two real booleans are "known"; the dict whose candidate is None
        # and the None read are both unknown (never known/False/success).
        self.assertEqual(observer.semantic_known_count, 2)
        self.assertEqual(observer.semantic_unknown_count, 2)
        self.assertEqual(observer.summary()["semantic_known_count"], 2)
        self.assertEqual(observer.summary()["semantic_unknown_count"], 2)
        # The raw dict is preserved verbatim, with its null fields intact, and it
        # is never promoted into a success.
        self.assertEqual(samples[1]["semantic"], null_dict)
        self.assertIsNone(samples[1]["semantic"]["semantic_candidate"])
        self.assertIsNone(samples[1]["semantic"]["note"])
        self.assertFalse(samples[1]["semantic_success"])
        # A genuine bool candidate is counted as known, but with no tracker there is
        # no tracked status: the row's ``semantic_success`` stays ``None`` (a raw
        # bool / raw candidate is never a fallback completion).
        self.assertIs(samples[0]["semantic"], True)
        self.assertIsNone(samples[0]["semantic_success"])
        self.assertIsNone(samples[2]["semantic_success"])

    def _drive_observer(self, tmp, sample, tracker, actions):
        """Drive the real observer over ``actions`` identical raw samples.

        The probe and steps are mocked and the env is a bare fake object, but
        ``tracker`` is a REAL ``wine_semantic.SemanticTracker`` and its ``update``
        is wrapped only to count the exact number of calls.  Returns the observer,
        the returned samples, the tracker updates seen and the parsed JSONL sample
        rows (the ``phase=start`` line is filtered out).
        """

        env = types.SimpleNamespace()
        seen = []
        original_update = tracker.update

        def counting_update(sample_arg):
            seen.append(sample_arg)
            return original_update(sample_arg)

        tracker.update = counting_update
        with mock.patch.object(grasp_guard, "read_probe", return_value=_probe()), \
             mock.patch.object(paired, "read_semantic", return_value=sample):
            observer = paired.GuardSemanticObserver(env, Path(tmp), tracker=tracker)
            observer.start()
            samples = [
                observer.on_action(np.zeros(7, dtype=np.float32), None)
                for _ in range(actions)
            ]
            observer.close()
            file_rows = [
                row
                for row in _read_assessment_rows(observer.path)
                if isinstance(row, dict) and "step" in row
            ]
        return observer, samples, seen, file_rows

    def test_real_tracker_counts_known_candidates_and_tracks_true(self):
        good = _raw_semantic_sample()  # accepted full raw sample, candidate True
        self.assertIs(good["semantic_candidate"], True)
        tracker = wine_semantic.SemanticTracker()
        with tempfile.TemporaryDirectory() as tmp:
            observer, samples, seen, file_rows = self._drive_observer(
                tmp, good, tracker, paired.ASSESSMENT_ACTION_COUNT
            )
        # Every raw candidate is an explicit bool -> every action is "known".
        self.assertEqual(observer.semantic_known_count, 20)
        self.assertEqual(observer.semantic_unknown_count, 0)
        # Exactly one tracker.update per action, with the raw sample only.
        self.assertEqual(len(seen), 20)
        self.assertEqual(len(file_rows), 20)
        self.assertEqual(len(samples), 20)
        self.assertTrue(all(update is good for update in seen))
        # 20 consecutive known candidates complete the tracked success.
        self.assertIs(tracker.semantic_success, True)
        # Every JSONL row carries a real bool (never null); the last is True.
        for row in file_rows:
            self.assertIsInstance(row["semantic_success"], bool)
        self.assertIs(file_rows[0]["semantic_success"], False)  # not yet complete
        self.assertIs(file_rows[-1]["semantic_success"], True)
        # The raw semantic and the tracked status are retained separately, and the
        # tracked status is the exact last ``update`` result.
        self.assertEqual(file_rows[-1]["semantic"], good)
        self.assertEqual(file_rows[-1]["semantic_status"], samples[-1]["semantic_status"])

    def test_real_tracker_reads_unknown_evidence_as_unknown_in_every_row(self):
        unknown = _raw_semantic_sample(linear_speed=None)  # required evidence missing
        self.assertIsNone(unknown["semantic_candidate"])
        tracker = wine_semantic.SemanticTracker()
        with tempfile.TemporaryDirectory() as tmp:
            observer, samples, seen, file_rows = self._drive_observer(
                tmp, unknown, tracker, paired.ASSESSMENT_ACTION_COUNT
            )
        # A dict whose candidate is None is UNKNOWN, never known/False/success.
        self.assertEqual(observer.semantic_known_count, 0)
        self.assertEqual(observer.semantic_unknown_count, 20)
        self.assertEqual(len(seen), 20)
        self.assertIsNone(tracker.semantic_success)
        # Every written JSONL row and every returned sample keeps a null status.
        for row in file_rows:
            self.assertIsNone(row["semantic_success"])
        for sample in samples:
            self.assertIsNone(sample["semantic_success"])
        # The raw sample is still retained verbatim, with its null field intact.
        self.assertEqual(file_rows[-1]["semantic"], unknown)
        self.assertIsNone(file_rows[-1]["semantic"]["semantic_candidate"])

    def test_real_tracker_keeps_a_valid_false_candidate_known_and_false(self):
        failing = _raw_semantic_sample(rack_contact=False)  # known-false rack contact
        self.assertIs(failing["semantic_candidate"], False)
        tracker = wine_semantic.SemanticTracker()
        with tempfile.TemporaryDirectory() as tmp:
            observer, samples, seen, file_rows = self._drive_observer(
                tmp, failing, tracker, 1
            )
        # An explicit bool False is a genuine *known* candidate (not unknown).
        self.assertEqual(observer.semantic_known_count, 1)
        self.assertEqual(observer.semantic_unknown_count, 0)
        self.assertEqual(len(seen), 1)  # exactly one tracker update per action
        self.assertIs(tracker.semantic_success, False)
        self.assertIsInstance(file_rows[0]["semantic_success"], bool)
        self.assertIs(file_rows[0]["semantic_success"], False)
        self.assertIs(samples[0]["semantic_success"], False)
        self.assertEqual(file_rows[0]["semantic"], failing)
        self.assertEqual(file_rows[0]["semantic_status"], samples[0]["semantic_status"])


class PassiveStepObserverTests(unittest.TestCase):
    class _Env:
        def __init__(self, raise_exc=None):
            self.calls = []
            self.result = ("obs", 1.0, False, False, {"is_success": True})
            self.raise_exc = raise_exc

        def step(self, action):
            self.calls.append(action)
            if self.raise_exc is not None:
                raise self.raise_exc
            return self.result

    class _Observer:
        def __init__(self, explode=False):
            self.started = 0
            self.seen = []
            self.explode = explode

        def start(self):
            self.started += 1
            return self

        def on_action(self, action, result):
            self.seen.append((action, result))
            if self.explode:
                raise RuntimeError("observer boom")

    def test_one_step_same_objects_and_restore(self):
        env = self._Env()
        original = env.step
        observer = self._Observer()
        step_observer = paired.PassiveStepObserver(env, observer)
        step_observer.install()
        self.assertIsNot(env.step, original)

        action = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        result = env.step(action)

        self.assertIs(result, env.result)  # the SAME result object
        self.assertEqual(len(env.calls), 1)  # exactly one physical step
        self.assertIs(env.calls[0], action)  # the unchanged action object
        self.assertEqual(observer.started, 1)  # started before the first action
        self.assertEqual(len(observer.seen), 1)
        self.assertIs(observer.seen[0][1], env.result)

        step_observer.restore()
        self.assertIs(env.step.__self__, original.__self__)
        self.assertIs(env.step.__func__, original.__func__)

    def test_exception_still_steps_once_and_restores(self):
        env = self._Env(raise_exc=RuntimeError("step boom"))
        original = env.step
        observer = self._Observer()
        step_observer = paired.PassiveStepObserver(env, observer)
        step_observer.install()
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7, dtype=np.float32))
        step_observer.restore()
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(observer.seen, [])  # no result was observed
        self.assertIs(env.step.__self__, original.__self__)
        self.assertIs(env.step.__func__, original.__func__)
        self.assertFalse(step_observer.installed)

    def test_observer_failure_never_changes_the_returned_result(self):
        env = self._Env()
        observer = self._Observer(explode=True)
        step_observer = paired.PassiveStepObserver(env, observer)
        with step_observer:
            result = env.step(np.zeros(7, dtype=np.float32))
        self.assertIs(result, env.result)
        self.assertEqual(len(env.calls), 1)


# --- selection rule -----------------------------------------------------------


def _trial(profile, state_index, model_seed, *, strict, semantic=None, wall=None,
           control=False, trial_id=None):
    return {
        "trial_id": trial_id or "%s-s%s-m%s" % (profile, state_index, model_seed),
        "profile": profile,
        "state_index": state_index,
        "model_seed": model_seed,
        "is_control": control,
        "strict_policy_success": strict,
        "strict_task_success": strict,
        "assessment_semantic_success": semantic,
        "policy_wall_s": wall,
        "sent_actions": [],
        "first_input": {},
    }


def _discovery_trials(strict_by_profile, semantic_by_profile=None, wall_by_profile=None):
    """A synthetic 6-trials-per-profile discovery grid (counts are per profile)."""

    semantic_by_profile = semantic_by_profile or {}
    wall_by_profile = wall_by_profile or {}
    trials = []
    for profile in paired.PROFILE_ORDER:
        count = 0
        for state_index in range(3):
            for offset in (0, 1000):
                trials.append(
                    _trial(
                        profile,
                        state_index,
                        state_index + offset,
                        strict=count < strict_by_profile.get(profile, 0),
                        semantic=count < semantic_by_profile.get(profile, 0),
                        wall=wall_by_profile.get(profile),
                    )
                )
                count += 1
    return trials


class SelectionRuleTests(unittest.TestCase):
    def test_primary_strict_success_count_wins(self):
        trials = _discovery_trials(
            {paired.PROFILE_A: 3, paired.PROFILE_B: 6, paired.PROFILE_C: 5}
        )
        result = paired.select_profile(trials)
        self.assertEqual(result["winner"], paired.PROFILE_B)
        self.assertEqual(result["stats"][paired.PROFILE_B]["strict_successes"], 6)
        self.assertFalse(result["reselects"])

    def test_secondary_semantic_then_tertiary_median_wall(self):
        trials = _discovery_trials(
            {paired.PROFILE_A: 4, paired.PROFILE_B: 4, paired.PROFILE_C: 4},
            semantic_by_profile={paired.PROFILE_B: 2, paired.PROFILE_C: 1},
            wall_by_profile={paired.PROFILE_B: 5.0, paired.PROFILE_C: 1.0},
        )
        result = paired.select_profile(trials)
        # Equal strict counts -> the semantic count decides (fp32 over fp32_h5).
        self.assertEqual(result["winner"], paired.PROFILE_B)

        wall_only = _discovery_trials(
            {paired.PROFILE_A: 4, paired.PROFILE_B: 4, paired.PROFILE_C: 4},
            wall_by_profile={paired.PROFILE_A: 9.0, paired.PROFILE_B: 9.0, paired.PROFILE_C: 2.0},
        )
        # Equal strict and semantic counts -> the lowest median wall time wins.
        self.assertEqual(paired.select_profile(wall_only)["winner"], paired.PROFILE_C)

    def test_exact_tie_uses_the_predeclared_order(self):
        trials = _discovery_trials(
            {paired.PROFILE_A: 5, paired.PROFILE_B: 5, paired.PROFILE_C: 5}
        )
        result = paired.select_profile(trials)
        self.assertEqual(result["winner"], paired.PROFILE_A)  # baseline_bf16 first
        self.assertEqual(
            result["ranking"], [paired.PROFILE_A, paired.PROFILE_B, paired.PROFILE_C]
        )

    def test_the_aa_control_trial_is_excluded(self):
        trials = _discovery_trials({paired.PROFILE_A: 1, paired.PROFILE_B: 1, paired.PROFILE_C: 1})
        trials.append(
            _trial(
                paired.PROFILE_C,
                0,
                0,
                strict=True,
                semantic=True,
                control=True,
                trial_id="control",
            )
        )
        result = paired.select_profile(trials)
        self.assertEqual(result["n_non_control_trials"], 18)
        self.assertEqual(result["stats"][paired.PROFILE_C]["strict_successes"], 1)
        self.assertEqual(result["stats"][paired.PROFILE_C]["n_trials"], 6)


class AssessmentReportingTests(unittest.TestCase):
    """The post-assessment tri-state is counted exactly and never promoted."""

    def _grid(self):
        trials = []
        for profile in paired.PROFILE_ORDER:
            for state_index in range(3):
                for offset in (0, 1000):
                    trials.append(
                        _trial(
                            profile,
                            state_index,
                            state_index + offset,
                            strict=False,
                            semantic=None,
                        )
                    )
        return trials

    def test_semantic_success_counts_only_explicit_true(self):
        trials = self._grid()
        # Profile C (indices 12-17): one explicit True, one explicit False, the
        # rest unknown (None).  Only the True is counted.
        trials[12]["assessment_semantic_success"] = True
        trials[13]["assessment_semantic_success"] = False
        result = paired.select_profile(trials)
        self.assertEqual(result["stats"][paired.PROFILE_C]["semantic_successes"], 1)
        self.assertEqual(result["stats"][paired.PROFILE_A]["semantic_successes"], 0)
        self.assertEqual(result["stats"][paired.PROFILE_B]["semantic_successes"], 0)
        # No assessment outcome is ever promoted into a strict policy success.
        for profile in paired.PROFILE_ORDER:
            self.assertEqual(result["stats"][profile]["strict_successes"], 0)

    def test_campaign_report_never_promotes_the_assessment(self):
        args = types.SimpleNamespace(
            phase="discovery",
            selected_profile=None,
            budget=300,
            timeout=900.0,
            source_git_sha="deadbeef",
            output="/tmp/report.json",
            run_root="/tmp/paired_runs",
        )
        # Explicit ``strict=False`` is a *known* physical failure: it is a strict
        # policy failure, never an unknown.
        trials = [
            _trial(paired.PROFILE_A, 0, 0, strict=False, semantic=True),
            _trial(paired.PROFILE_A, 0, 1000, strict=False, semantic=None),
        ]
        report = paired._build_campaign_report(
            args,
            trials,
            model_revision=None,
            git_sha="deadbeef",
            started=paired.time.monotonic(),
            fatal_error=None,
            plan=paired.build_discovery_plan(),
            preregistration_path=Path("/tmp/prereg.json"),
            preregistration_sha256="frozen-sha",
            selection=paired.select_profile(trials),
            aa_comparison=None,
        )
        # The assessment outcome stays out of the policy benchmark aggregate.
        self.assertFalse(report["metadata"]["assessment_promoted_to_benchmark"])
        self.assertEqual(report["aggregate"]["n_strict_policy_success"], 0)
        self.assertEqual(report["aggregate"]["n_strict_policy_unknown"], 0)

        # A separately built report whose two inputs are *unknown* (``strict=None``):
        # the assessment outcome is again counted separately from the strict policy
        # outcome, so the unknown inputs raise the unknown count but neither the
        # strict success count nor the assessment promotion.
        unknown_trials = [
            _trial(paired.PROFILE_A, 0, 0, strict=None, semantic=True),
            _trial(paired.PROFILE_A, 0, 1000, strict=None, semantic=None),
        ]
        unknown_report = paired._build_campaign_report(
            args,
            unknown_trials,
            model_revision=None,
            git_sha="deadbeef",
            started=paired.time.monotonic(),
            fatal_error=None,
            plan=paired.build_discovery_plan(),
            preregistration_path=Path("/tmp/prereg.json"),
            preregistration_sha256="frozen-sha",
            selection=paired.select_profile(unknown_trials),
            aa_comparison=None,
        )
        self.assertFalse(unknown_report["metadata"]["assessment_promoted_to_benchmark"])
        self.assertEqual(unknown_report["aggregate"]["n_strict_policy_success"], 0)
        self.assertEqual(unknown_report["aggregate"]["n_strict_policy_unknown"], 2)


# --- fixed grids --------------------------------------------------------------


class PlanTests(unittest.TestCase):
    def test_discovery_plan_is_18_plus_one_control(self):
        plan = paired.build_discovery_plan()
        self.assertEqual(len(plan), 19)
        self.assertEqual(sum(1 for entry in plan if entry["is_control"]), 1)
        # The A/A duplicate sits immediately after the very first trial.
        self.assertFalse(plan[0]["is_control"])
        self.assertTrue(plan[1]["is_control"])
        self.assertEqual(plan[0]["state_index"], 0)
        self.assertEqual(plan[0]["model_seed"], 0)
        self.assertEqual(plan[0]["profile"], paired.PROFILE_A)
        self.assertEqual(plan[1]["state_index"], 0)
        self.assertEqual(plan[1]["model_seed"], 0)
        self.assertEqual(plan[1]["profile"], paired.PROFILE_A)
        self.assertEqual(plan[1]["control_of"], "state0_modelseed0_baseline")
        # States in order, env seed == state index, model seeds and profile order.
        non_control = [entry for entry in plan if not entry["is_control"]]
        self.assertEqual([entry["state_index"] for entry in non_control[:6]], [0] * 6)
        self.assertEqual(
            [entry["profile"] for entry in non_control[:3]], list(paired.PROFILE_ORDER)
        )
        self.assertEqual([entry["model_seed"] for entry in non_control[:3]], [0, 0, 0])
        self.assertEqual([entry["model_seed"] for entry in non_control[3:6]], [1000] * 3)
        for entry in non_control:
            self.assertEqual(entry["envseed"], entry["state_index"])
        self.assertEqual(
            sorted({entry["state_index"] for entry in non_control}),
            list(paired.DISCOVERY_STATES),
        )

    def test_holdout_plan_baseline_winner_has_three_trials(self):
        plan = paired.build_holdout_plan(paired.PROFILE_A)
        self.assertEqual(len(plan), 3)
        self.assertTrue(all(entry["profile"] == paired.PROFILE_A for entry in plan))
        self.assertEqual(
            [entry["state_index"] for entry in plan], list(paired.HOLDOUT_STATES)
        )
        for entry in plan:
            self.assertEqual(entry["envseed"], entry["state_index"])
            self.assertEqual(entry["model_seed"], entry["state_index"])
            self.assertFalse(entry["is_control"])

    def test_holdout_plan_other_winner_has_six_trials_and_never_reselects(self):
        plan = paired.build_holdout_plan(paired.PROFILE_C)
        self.assertEqual(len(plan), 6)
        self.assertEqual(
            sorted({entry["profile"] for entry in plan}),
            [paired.PROFILE_A, paired.PROFILE_C],
        )
        # The unselected profile is never present, and the grid is fixed.
        self.assertNotIn(paired.PROFILE_B, {entry["profile"] for entry in plan})
        self.assertEqual(
            [entry["state_index"] for entry in plan],
            [3, 3, 4, 4, 5, 5],
        )


# --- A/A repeatability --------------------------------------------------------


class AaComparisonTests(unittest.TestCase):
    def _pair(self):
        control = _trial(paired.PROFILE_A, 0, 0, strict=True, trial_id="control")
        repeat = _trial(
            paired.PROFILE_A, 0, 0, strict=True, control=True, trial_id="repeat"
        )
        control["sent_actions"] = [[0.0] * 6 + [0.5], [0.0] * 7]
        repeat["sent_actions"] = [[0.0] * 6 + [0.5], [0.0] * 6 + [0.1]]
        control["first_input"] = {"combined_sha256": "abc"}
        repeat["first_input"] = {"combined_sha256": "abc"}
        control["plan_terminal_state"] = "completed"
        repeat["plan_terminal_state"] = "completed"
        control["policy_action_count"] = 2
        repeat["policy_action_count"] = 2
        control["assessment_action_count"] = 20
        repeat["assessment_action_count"] = 20
        return control, repeat

    def test_compare_aa_reports_lengths_diff_stop_and_inputs(self):
        control, repeat = self._pair()
        comparison = paired.compare_aa(control, repeat)
        self.assertEqual(comparison["control_length"], 2)
        self.assertEqual(comparison["repeat_length"], 2)
        self.assertTrue(comparison["length_match"])
        self.assertEqual(comparison["compared_count"], 2)
        self.assertEqual(comparison["exact_match_count"], 1)  # only the first 7-D array
        self.assertAlmostEqual(comparison["max_abs_diff"], 0.1, places=6)  # the gripper dim
        self.assertTrue(comparison["stop_result_match"])
        self.assertTrue(comparison["inputs_match"])
        self.assertEqual(comparison["length_mismatches"], [])
        # Repeatability evidence only -- never a bitwise CUDA claim.
        self.assertIn("bitwise CUDA determinism", comparison["note"])

    def test_compare_aa_flags_a_length_mismatch_and_missing_input(self):
        control, repeat = self._pair()
        repeat["sent_actions"] = repeat["sent_actions"][:1]
        repeat["first_input"] = {"combined_sha256": None}
        comparison = paired.compare_aa(control, repeat)
        self.assertFalse(comparison["length_match"])
        self.assertFalse(comparison["inputs_match"])
        self.assertTrue(comparison["first_input_errors"])

    def test_aa_from_trials_pairs_the_first_trial_with_its_duplicate(self):
        control, repeat = self._pair()
        trials = [control, repeat]
        comparison = paired._aa_from_trials(trials)
        self.assertEqual(comparison["control_trial_id"], "control")
        self.assertEqual(comparison["repeat_trial_id"], "repeat")
        self.assertIsNone(paired._aa_from_trials([control]))


# --- checkpoint audit and preregistration ------------------------------------


class CheckpointAuditTests(unittest.TestCase):
    def test_exactly_seven_readable_files_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(paired.AUDITED_CHECKPOINT_FILE_COUNT):
                (root / ("file_%d.bin" % index)).write_bytes(b"x" * (index + 1))
            audit = paired._audit_checkpoint_files(str(root))
            self.assertTrue(audit["ok"])
            self.assertIsNone(audit["reason"])
            self.assertEqual(len(audit["files"]), 7)
            for entry in audit["files"]:
                self.assertIsNotNone(entry["size"])
                self.assertIsNotNone(entry["sha256"])
                self.assertIsNone(entry["error"])

    def test_a_wrong_file_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(paired.AUDITED_CHECKPOINT_FILE_COUNT - 1):
                (root / ("file_%d.bin" % index)).write_bytes(b"x")
            audit = paired._audit_checkpoint_files(str(root))
            self.assertFalse(audit["ok"])
            self.assertIn("expected exactly", audit["reason"])

    def test_a_missing_directory_fails_closed(self):
        audit = paired._audit_checkpoint_files(str(Path(tempfile.gettempdir()) / "no_such_dir_xyz"))
        self.assertFalse(audit["ok"])
        self.assertIn("missing", audit["reason"])


class PreregistrationTests(unittest.TestCase):
    def _args(self, phase="discovery", selected=None):
        return types.SimpleNamespace(
            phase=phase,
            selected_profile=selected,
            budget=300,
            timeout=900.0,
            source_git_sha="deadbeef",
            output="/tmp/report.json",
            run_root="/tmp/paired_runs",
        )

    def test_preregistration_is_frozen_and_model_free(self):
        plan = paired.build_discovery_plan()
        audit = {"ok": True, "files": [{"name": "f", "size": 1, "sha256": "s"}]}
        record = paired._build_preregistration(
            self._args(), {"paired_config_experiments.py": "sha"}, audit, plan
        )
        self.assertEqual(record["experiment"], paired.EXPERIMENT_NAME)
        self.assertEqual(record["phase"], "discovery")
        self.assertEqual(record["selection_rule"], dict(paired._SELECTION_RULE))
        self.assertEqual(record["assessment_protocol"], dict(paired._ASSESSMENT_PROTOCOL))
        self.assertEqual(record["assessment_protocol"]["action_count"], 20)
        self.assertEqual(record["source_sha256"], {"paired_config_experiments.py": "sha"})
        self.assertEqual(record["checkpoint_audit"], audit)
        self.assertEqual(
            record["fixed_spec"]["oracle_goals"],
            [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
        )
        self.assertEqual(record["fixed_spec"]["guard_mode"], "off")
        self.assertEqual(record["parameters"]["budget_per_subgoal"], 300)
        self.assertIsNotNone(record["created_utc"])
        self.assertEqual(len(record["plan"]), len(plan))

    def test_campaign_report_freezes_the_preregistration_sha(self):
        args = self._args(phase="holdout", selected=paired.PROFILE_C)
        report = paired._build_campaign_report(
            args,
            [],
            model_revision="6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
            git_sha="deadbeef",
            started=paired.time.monotonic(),
            fatal_error=None,
            plan=paired.build_holdout_plan(paired.PROFILE_C),
            preregistration_path=Path("/tmp/prereg.json"),
            preregistration_sha256="frozen-sha",
            selection={"reselects": False, "selected_profile": paired.PROFILE_C},
            aa_comparison=None,
        )
        self.assertTrue(report["preregistration"]["frozen_before_model_start"])
        self.assertEqual(report["preregistration"]["sha256"], "frozen-sha")
        self.assertIsNone(report["discovery_summary"])  # holdout: no re-selection
        self.assertFalse(report["selection"]["reselects"])
        self.assertEqual(report["aggregate"]["n_trials"], 0)


# --- one trial ----------------------------------------------------------------


class _FakeSessionRecord:
    initial_state_hash = "init-sha"
    xml_sha = "xml-sha"
    camera_names = ["agentview", "wrist"]


class _FakePairedService:
    """A pure stand-in exposing exactly what ``paired.run_trial`` reads/writes."""

    def __init__(self, jobs=None, profile_ok=True):
        self._jobs = dict(jobs or {})
        self._sessions = {}
        self._active_request_id = None
        self._env = object()
        self._diag_condition = None
        self._paired_tracker = None
        self._paired_input_evidence = None
        self._paired_observer_summary = None
        self.completion_mode = wd.COMPLETION_MODE
        self.run_root = Path(tempfile.gettempdir())
        self._profile_ok = profile_ok
        self.profile_calls = 0

    def create_session(self, scene_id, seed=0, init_state_index=0):  # noqa: ARG002
        self._sessions["sess-1"] = _FakeSessionRecord()
        return {
            "ok": True,
            "session_id": "sess-1",
            "env_instance_id": 1,
            "episode_resets": 1,
            "policy_resets": 1,
            "scene_version": 1,
        }

    def configure_profile(self, name):  # noqa: ARG002
        self.profile_calls += 1
        if not self._profile_ok:
            return {"ok": False, "reason": "invalid_fixture", "detail": "no config"}
        return {"ok": True, "profile": name}

    def job(self, job_id):
        return self._jobs.get(job_id)

    def final_snapshot(self, goals):  # noqa: ARG002
        return {
            "ok": True,
            "snapshot": {
                "predicates": {paired.WINE_GOAL_KEY: False},
                "phases": {paired.WINE_GOAL_KEY: "ungrasped"},
                "held_objects": [],
                "strict_candidate": False,
                "goal_objects": [paired.WINE_OBJECT_ID],
                "objects": {},
            },
        }


def _plan_record(state, job_ids, *, timed_out=False, submit_error=None, cancelled=False):
    return {
        "request_id": "req-1",
        "submitted": submit_error is None,
        "submit_error": submit_error,
        "timed_out": timed_out,
        "cancel_nonterminal": False,
        "cancelled": cancelled,
        "wall_s": 1.0,
        "plan": {
            "state": state,
            "plan_success": state == "completed",
            "completed_capability_ids": [],
            "pending_capability_ids": [paired.WINE_CAPABILITY_ID],
            "job_ids": list(job_ids),
        },
    }


def _assessment_outcome(tmp):
    return {
        "ok": True,
        "path": str(Path(tmp) / "assessment.jsonl"),
        "action_count": paired.ASSESSMENT_ACTION_COUNT,
        "expected_action_count": paired.ASSESSMENT_ACTION_COUNT,
        "action_dtype": "float32",
        "action_shape": [7],
        "strict_final_score": False,
        "final_snapshot": {"strict_candidate": False},
        "semantic_before": None,
        "semantic_after": None,
        "semantic_success": None,
        "native_success_final": False,
        "native_success_ever": False,
        "error": None,
    }


class RunTrialTests(unittest.TestCase):
    """``run_trial`` against a fake service: the operational/physical split."""

    def _patch_common(self, plan_record):
        """Everything except the assessment, which each test patches itself."""

        patches = [
            mock.patch.object(
                paired,
                "_read_control_frequency",
                return_value={"ok": True, "control_frequency_hz": 20},
            ),
            mock.patch.object(paired, "_seed_model_rng", return_value={"ok": True, "seeded": True}),
            mock.patch.object(
                wd,
                "_read_task_language",
                return_value={"ok": True, "task_language": wd.WINE_INSTRUCTION},
            ),
            mock.patch.object(pe, "_submit_and_wait", return_value=plan_record),
        ]
        started = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)
        self.submit_mock = started[-1]

    def _run(self, svc, plan_record, tmp):
        self._patch_common(plan_record)
        with mock.patch.object(paired, "run_assessment", return_value=_assessment_outcome(tmp)) as m:
            self.assessment_mock = m
            return paired.run_trial(
                svc,
                phase="discovery",
                state_index=0,
                model_seed=0,
                profile=paired.PROFILE_A,
                registry={},
            )

    def test_profile_failure_is_operational_with_zero_plans_and_no_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _FakePairedService(profile_ok=False)
            trial = self._run(svc, _plan_record("blocked", []), tmp)
            self.submit_mock.assert_not_called()
            self.assessment_mock.assert_not_called()
        self.assertTrue(trial["operational_errors"])
        self.assertTrue(any("profile_config" in e for e in trial["operational_errors"]))
        self.assertIsNone(trial["policy_action_count"])
        self.assertEqual(trial["assessment_action_count"], 0)
        self.assertIsNone(trial["assessment"])  # returned before the assessment stage
        self.assertEqual(svc.profile_calls, 1)

    def test_budget_exhaustion_continues_and_receives_the_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = {
                "job-1": {
                    "job_id": "job-1",
                    "run_dir": tmp,
                    "state": "completed",
                    "ended_reason": "budget_exhausted",
                    "error": None,
                    "success": False,
                    "instruction": wd.WINE_INSTRUCTION,
                    "steps": 300,
                    "total_steps": 300,
                    "wall_s": 12.5,
                }
            }
            svc = _FakePairedService(jobs=jobs)
            trial = self._run(svc, _plan_record("blocked", ["job-1"]), tmp)
            self.assessment_mock.assert_called_once()
        self.assertEqual(trial["operational_errors"], [])
        self.assertTrue(any("budget_exhausted" in f for f in trial["physical_failures"]))
        self.assertEqual(trial["policy_action_count"], 300)
        self.assertEqual(trial["assessment_action_count"], 20)
        self.assertTrue(trial["assessment_physics"])
        self.assertTrue(trial["assessment"]["performed"])
        self.assertIsNone(trial["semantic_before_assessment"])
        self.assertIsNone(trial["semantic_after_assessment"])
        self.assertFalse(trial["assisted_policy"])
        self.assertEqual(trial["hermes_calls"], 0)
        self.assertEqual(trial["guard_mode"], "off")
        self.assertTrue(trial["assessment_included_in_walltime"])

    def test_an_operational_job_error_stops_without_an_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = {
                "job-1": {
                    "job_id": "job-1",
                    "run_dir": tmp,
                    "state": "error",
                    "ended_reason": "error",
                    "error": "boom",
                    "success": False,
                    "instruction": wd.WINE_INSTRUCTION,
                    "steps": 3,
                    "total_steps": 3,
                }
            }
            svc = _FakePairedService(jobs=jobs)
            trial = self._run(svc, _plan_record("blocked", ["job-1"]), tmp)
            self.assessment_mock.assert_not_called()
        self.assertTrue(any("job_error" in e for e in trial["operational_errors"]))
        self.assertEqual(trial["assessment_action_count"], 0)
        self.assertFalse(trial["assessment"]["performed"])

    def test_a_cancelled_plan_and_job_are_operational_not_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = {
                "job-1": {
                    "job_id": "job-1",
                    "run_dir": tmp,
                    "state": "cancelled",
                    "ended_reason": "cancelled",
                    "error": None,
                    "success": False,
                    "instruction": wd.WINE_INSTRUCTION,
                    "steps": 4,
                    "total_steps": 300,
                    "wall_s": 3.0,
                }
            }
            svc = _FakePairedService(jobs=jobs)
            trial = self._run(
                svc, _plan_record("cancelled", ["job-1"], cancelled=True), tmp
            )
            self.assessment_mock.assert_not_called()
        # Cancellation is recorded as an operational error (plan AND job).
        self.assertTrue(any("job_cancelled" in e for e in trial["operational_errors"]))
        self.assertTrue(any("plan_cancelled" in e for e in trial["operational_errors"]))
        # No 20-step assessment is run (assessment0).
        self.assertEqual(trial["assessment_action_count"], 0)
        self.assertFalse(trial["assessment"]["performed"])
        self.assertFalse(trial["assessment_physics"])
        # Cancellation is never reinterpreted as a physical budget failure.
        self.assertEqual(trial["physical_failures"], [])
        # The original cancellation state/reason/artifacts are preserved verbatim.
        self.assertEqual(trial["plan"]["state"], "cancelled")
        self.assertTrue(trial["plan"]["cancelled"])
        self.assertEqual(trial["plan_terminal_state"], "cancelled")
        self.assertEqual(trial["jobs"][0]["state"], "cancelled")
        self.assertEqual(trial["jobs"][0]["ended_reason"], "cancelled")
        self.assertEqual(trial["jobs"][0]["run_dir"], tmp)

    def test_a_cancelled_job_after_a_completed_plan_is_operational(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = {
                "job-1": {
                    "job_id": "job-1",
                    "run_dir": tmp,
                    "state": "cancelled",
                    "ended_reason": "cancelled",
                    "error": None,
                    "success": False,
                    "instruction": wd.WINE_INSTRUCTION,
                    "steps": 2,
                    "total_steps": 300,
                }
            }
            svc = _FakePairedService(jobs=jobs)
            trial = self._run(svc, _plan_record("completed", ["job-1"]), tmp)
            self.assessment_mock.assert_not_called()
        self.assertTrue(any("job_cancelled" in e for e in trial["operational_errors"]))
        self.assertEqual(trial["physical_failures"], [])
        self.assertEqual(trial["assessment_action_count"], 0)
        self.assertFalse(trial["assessment"]["performed"])

    def test_a_cancelled_trial_records_an_op_error_and_no_later_trials_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = {
                "job-1": {
                    "job_id": "job-1",
                    "run_dir": tmp,
                    "state": "cancelled",
                    "ended_reason": "cancelled",
                    "error": None,
                    "success": False,
                    "instruction": wd.WINE_INSTRUCTION,
                    "steps": 1,
                    "total_steps": 300,
                }
            }
            svc = _FakePairedService(jobs=jobs)
            plan_record = _plan_record("cancelled", ["job-1"], cancelled=True)
            plan = [
                {"state_index": 0, "model_seed": 0, "profile": paired.PROFILE_A},
                {"state_index": 0, "model_seed": 0, "profile": paired.PROFILE_B},
            ]
            executed = []
            with mock.patch.object(
                paired,
                "_read_control_frequency",
                return_value={"ok": True, "control_frequency_hz": 20},
            ), mock.patch.object(
                paired, "_seed_model_rng", return_value={"ok": True, "seeded": True}
            ), mock.patch.object(
                wd,
                "_read_task_language",
                return_value={"ok": True, "task_language": wd.WINE_INSTRUCTION},
            ), mock.patch.object(
                pe, "_submit_and_wait", return_value=plan_record
            ), mock.patch.object(
                paired, "run_assessment", return_value=_assessment_outcome(tmp)
            ) as assessment:
                for entry in plan:
                    trial = paired.run_trial(
                        svc,
                        phase="discovery",
                        state_index=entry["state_index"],
                        model_seed=entry["model_seed"],
                        profile=entry["profile"],
                        registry={},
                    )
                    executed.append(trial)
                    # This is the exact campaign abort condition used by
                    # ``_run_campaign``: an operational error stops the campaign.
                    if trial.get("operational_errors"):
                        break
            assessment.assert_not_called()
        self.assertEqual(len(executed), 1)  # no later trials ran
        self.assertTrue(executed[0]["operational_errors"])
        self.assertEqual(executed[0]["assessment_action_count"], 0)
        self.assertFalse(executed[0]["assessment"]["performed"])


# --- the service wrapper ------------------------------------------------------


class PairedServiceTests(unittest.TestCase):
    def _service(self):
        """A subclass instance whose heavy base ``__init__`` is recorded only.

        ``wd.WineDiagnosticService.__init__`` (and therefore the whole
        worker/simulator/torch chain) is replaced by a recorder, so these tests
        stay GPU-free: no worker, no model, no simulator, no network.  The
        subclass' own ``__init__`` body still runs, so its ``_paired_*`` state and
        the exact kwargs it forwards to its superclass are both inspected.
        """

        with mock.patch.object(
            wd.WineDiagnosticService, "__init__", return_value=None
        ) as init:
            svc = paired.PairedWineDiagnosticService("model-path", "run-root")
        self.init_mock = init
        return svc

    def test_service_forces_the_guard_off_and_release_verified(self):
        self._service()
        kwargs = self.init_mock.call_args.kwargs
        self.assertEqual(kwargs.get("grasp_guard_mode"), "off")
        self.assertEqual(kwargs.get("completion_mode"), wd.COMPLETION_MODE)
        self.assertEqual(wd.COMPLETION_MODE, "release_verified")

    def test_service_rejects_a_non_off_guard_mode_request(self):
        with mock.patch.object(
            wd.WineDiagnosticService, "__init__", return_value=None
        ) as init:
            with self.assertRaises(ValueError) as caught:
                paired.PairedWineDiagnosticService(
                    "model-path", "run-root", grasp_guard_mode="enforce"
                )
        init.assert_not_called()  # rejected before the base is ever constructed
        self.assertIn("grasp_guard_mode", str(caught.exception))

    def test_service_accepts_omitted_or_off_and_forwards_literal_off(self):
        with mock.patch.object(
            wd.WineDiagnosticService, "__init__", return_value=None
        ) as init:
            paired.PairedWineDiagnosticService("model-path", "run-root")
            self.assertEqual(init.call_args.kwargs.get("grasp_guard_mode"), "off")
        with mock.patch.object(
            wd.WineDiagnosticService, "__init__", return_value=None
        ) as init:
            paired.PairedWineDiagnosticService(
                "model-path", "run-root", grasp_guard_mode="off"
            )
            self.assertEqual(init.call_args.kwargs.get("grasp_guard_mode"), "off")
            self.assertEqual(
                init.call_args.kwargs.get("completion_mode"), wd.COMPLETION_MODE
            )

    def test_off_guard_keeps_the_independent_observer_passive(self):
        svc = self._service()
        self.assertIsNone(svc._paired_observer_summary)
        with tempfile.TemporaryDirectory() as tmp:
            env = types.SimpleNamespace()
            with mock.patch.object(grasp_guard, "read_probe", return_value=_probe()), \
                 mock.patch.object(paired, "read_semantic", return_value=None):
                observer = paired.GuardSemanticObserver(env, Path(tmp))
                observer.start()
                observer.on_action(np.zeros(7, dtype=np.float32), None)
                observer.close()
                summary = observer.summary()
        # The independent observer stays read-only/passive under the forced-off guard.
        self.assertTrue(summary["observer_only"])
        self.assertEqual(summary["n_samples"], 1)

    def test_capture_paired_input_validates_only_the_first_batch(self):
        svc = self._service()
        svc._paired_state_index = 0
        svc._paired_model_seed = 0
        svc._paired_profile = paired.PROFILE_A
        svc._paired_registry = {}
        first = svc._capture_paired_input(_batch())
        # A later, different batch for the SAME trial is not re-validated.
        svc._capture_paired_input(_batch(state=(9.0, 9.0, 9.0)))
        self.assertEqual(first["state_index"], 0)
        self.assertTrue(paired.fingerprint_is_complete(first["fingerprint"]))
        self.assertIn(0, svc._paired_registry)

    def test_capture_paired_input_mismatch_raises_before_prediction(self):
        svc = self._service()
        svc._paired_state_index = 0
        svc._paired_registry = {
            0: {
                "initial_state_sha": "other",
                "xml_sha": "other",
                "camera_names": ["a"],
                "control_frequency_hz": 7,
                "camera_raw_sha256": {},
                "state_raw_sha256": "other",
                "state_shape": [1],
                "state_dtype": "float32",
                "task": ["other"],
                "task_sha256": "other",
                "combined_sha256": "other",
            }
        }
        with mock.patch.object(
            wd.WineDiagnosticService,
            "_select_action",
            side_effect=AssertionError("prediction must not run"),
        ) as predict:
            with self.assertRaises(RuntimeError) as caught:
                svc._select_action(_batch())
        predict.assert_not_called()  # raised BEFORE any prediction ran
        self.assertIn("paired_input_mismatch", str(caught.exception))

    def test_capture_paired_input_fingerprints_only_the_first_batch(self):
        svc = self._service()
        svc._paired_state_index = 0
        svc._paired_model_seed = 0
        svc._paired_profile = paired.PROFILE_A
        svc._paired_registry = {}
        changed = _batch(state=(9.0, 9.0, 9.0))  # a DIFFERENT later action batch
        with mock.patch.object(
            paired, "fingerprint_batch", wraps=paired.fingerprint_batch
        ) as fp:
            first = svc._capture_paired_input(_batch())
            self.assertEqual(fp.call_count, 1)  # the first call fingerprints once
            second = svc._capture_paired_input(changed)
            self.assertEqual(fp.call_count, 1)  # a later batch is never fingerprinted
        # The existing first-input evidence is returned unchanged and never
        # overwritten by the changed later batch.
        self.assertIs(second, first)
        self.assertIs(svc._paired_input_evidence, first)
        self.assertEqual(
            second["fingerprint"]["state"]["raw_sha256"],
            first["fingerprint"]["state"]["raw_sha256"],
        )
        self.assertEqual(
            second["fingerprint"]["combined_sha256"],
            first["fingerprint"]["combined_sha256"],
        )
        self.assertIn(0, svc._paired_registry)

    def test_capture_paired_input_missing_first_input_prevents_step(self):
        svc = self._service()
        svc._paired_state_index = 0
        svc._paired_registry = {}
        with mock.patch.object(
            paired, "fingerprint_batch", wraps=paired.fingerprint_batch
        ) as fp:
            with self.assertRaises(RuntimeError) as caught:
                svc._capture_paired_input(_batch(n_cameras=1))
        self.assertIn("paired_input_incomplete", str(caught.exception))
        # The rejection stands: nothing was captured or registered, so a later
        # re-attempt still has to pass, and the first batch was fingerprinted once.
        self.assertIsNone(svc._paired_input_evidence)
        self.assertNotIn(0, svc._paired_registry)
        self.assertEqual(fp.call_count, 1)


# --- CLI contract -------------------------------------------------------------


class CliContractTests(unittest.TestCase):
    def test_defaults(self):
        parser = paired._build_parser()
        args = parser.parse_args(["--output", "/tmp/p.json", "--run-root", "/tmp/p_runs"])
        self.assertEqual(args.phase, "discovery")
        self.assertIsNone(args.selected_profile)
        self.assertEqual(args.budget, 300)
        self.assertEqual(args.timeout, 900.0)

    def test_holdout_requires_a_selected_profile(self):
        parser = paired._build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            args = parser.parse_args(
                [
                    "--phase",
                    "holdout",
                    "--output",
                    str(Path(tmp) / "out.json"),
                    "--run-root",
                    str(Path(tmp) / "runs"),
                ]
            )
            with self.assertRaises(SystemExit):
                paired._validate_args(args, parser)

    def test_unknown_profile_is_rejected(self):
        parser = paired._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--selected-profile",
                    "bogus",
                    "--output",
                    "/tmp/p.json",
                    "--run-root",
                    "/tmp/p_runs",
                ]
            )

    def test_relative_and_existing_paths_are_rejected(self):
        parser = paired._build_parser()
        args = parser.parse_args(["--output", "relative.json", "--run-root", "/tmp/p_runs"])
        with self.assertRaises(SystemExit):
            paired._validate_args(args, parser)


def _main() -> int:
    unittest.main(verbosity=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
