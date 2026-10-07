#!/usr/bin/env python3
"""GPU-free unit tests for ``scene_demo/wine_diagnostics.py``.

These tests never load a checkpoint, never build a LIBERO scene and never touch
CUDA: they exercise the *real production helpers* against fake simulator /
robot interfaces and mock simple tensor-like objects.  In particular they prove

* ``capture_gripper_state`` is read-only (zero step/reset/forward) and turns
  missing/nonfinite fields into ``None`` without guessing a width;
* ``WineStepWrapper`` calls the saved original ``env.step`` exactly once with the
  unchanged action object, returns the *same* tuple object, and restores the
  original ``env.step`` on the normal AND the exception path;
* ``PostActionCapture`` copies the raw action *before* an in-place post mutation,
  calls the original post exactly once, returns its result object unchanged, and
  restores ``_post`` on the normal AND the exception path;
* ``register_native_wine_scene`` restores all THREE dictionaries -- the two
  catalog dictionaries and ``placement_experiments.FINAL_ORACLE_GOALS`` --
  identity and contents -- on the normal AND the exception path;
* ``_run_trial`` classifies a plan timeout as an operational error (never an
  ordinary physical 300-step failure) while a plain budget exhaustion remains an
  ordinary, continuable task failure;
* ``summarize_wine_samples`` reports measured metrics and never turns missing
  data into a ``False``/``0`` success, and distinguishes a jaw-width change from
  an object lift.
"""

from __future__ import annotations

import copy
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

import catalog  # noqa: E402
import placement_experiments as pe  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402


# --- fake gripper / simulator interfaces -------------------------------------


class _GripperData:
    """Minimal native ``sim.data`` for the gripper probe."""

    def __init__(self) -> None:
        self.qpos_map: dict[str, np.ndarray] = {}
        self.qvel_map: dict[str, np.ndarray] = {}
        self.ctrl = np.zeros(4, dtype=np.float64)

    def get_joint_qpos(self, name):
        if name not in self.qpos_map:
            raise KeyError(name)
        return np.asarray(self.qpos_map[name], dtype=np.float64)

    def get_joint_qvel(self, name):
        if name not in self.qvel_map:
            raise KeyError(name)
        return np.asarray(self.qvel_map[name], dtype=np.float64)


class _MutatingSim:
    """A sim stand-in that records any step/forward/reset call (all forbidden)."""

    def __init__(self, data, model) -> None:
        self.data = data
        self.model = model
        self.calls = {"step": 0, "reset": 0, "forward": 0}

    def step(self, *args):  # noqa: ARG002
        self.calls["step"] += 1

    def reset(self, *args):  # noqa: ARG002
        self.calls["reset"] += 1

    def forward(self, *args):  # noqa: ARG002
        self.calls["forward"] += 1


def _gripper_env(
    joints,
    qpos_map,
    qvel_map=None,
    actuator_names=(),
    ctrl=None,
    model_map=None,
    current_action=None,
    native_lookup=None,
):
    """Build a mock env wrapping a native-shaped inner env.

    ``native_lookup`` -- when given -- is installed as the *callable* native
    ``sim.model.actuator_name2id(name) -> int``; otherwise a plain ``dict`` is
    installed (the compatible-mock fallback).  Returns ``(env, sim)`` so a test
    can assert the sim was never stepped.
    """

    data = _GripperData()
    data.qpos_map = dict(qpos_map)
    data.qvel_map = dict(qvel_map or {})
    if ctrl is not None:
        data.ctrl = np.asarray(ctrl, dtype=np.float64)
    if native_lookup is not None:
        sim_model = types.SimpleNamespace(actuator_name2id=native_lookup)
    else:
        sim_model = types.SimpleNamespace(actuator_name2id=dict(model_map or {}))
    sim = _MutatingSim(data, sim_model)

    gripper = types.SimpleNamespace(
        joints=list(joints),
        actuators=[types.SimpleNamespace(name=name) for name in actuator_names],
    )
    if current_action is not None:
        gripper.current_action = current_action
    robot = types.SimpleNamespace(gripper=gripper)

    class _Inner:
        def __init__(self) -> None:
            self.robots = [robot]
            self.sim = sim
            self.objects_dict = {}

    env = types.SimpleNamespace(_env=types.SimpleNamespace(env=_Inner()))
    return env, sim


class GripperStateTests(unittest.TestCase):
    def test_read_only_and_two_finite_jaws(self):
        env, sim = _gripper_env(
            ["g0", "g1"],
            {"g0": [0.04], "g1": [-0.02]},
            {"g0": [0.0], "g1": [0.0]},
            actuator_names=["a0", "a1"],
            ctrl=[0.1, 0.2, 0.3, 0.4],
            model_map={"a0": 0, "a1": 1},
            current_action=[0.5, -0.5],
        )
        state = wd.capture_gripper_state(env)
        # Read-only: never step/reset/forward.
        self.assertEqual(sim.calls, {"step": 0, "reset": 0, "forward": 0})
        self.assertEqual(state["joint_names"], ["g0", "g1"])
        self.assertEqual(state["qpos"], [0.04, -0.02])
        self.assertEqual(state["qvel"], [0.0, 0.0])
        self.assertAlmostEqual(state["width_m"], 0.06)
        self.assertEqual(state["current_action"], [0.5, -0.5])
        self.assertEqual(state["actuator_names"], ["a0", "a1"])
        self.assertEqual(state["actuator_ctrl"], [0.1, 0.2])
        self.assertIsNone(state["error"])

    def test_native_callable_actuator_lookup_reads_nonzero_ctrl(self):
        queried: list[str] = []

        def name2id(name):
            queried.append(name)
            return {"a0": 0, "a1": 1}[name]

        env, sim = _gripper_env(
            ["g0", "g1"],
            {"g0": [0.01], "g1": [0.02]},
            {"g0": [0.0], "g1": [0.0]},
            actuator_names=["a0", "a1"],
            ctrl=[0.7, -0.3],
            native_lookup=name2id,
        )
        state = wd.capture_gripper_state(env)
        # Read-only: never step/reset/forward even with the callable lookup.
        self.assertEqual(sim.calls, {"step": 0, "reset": 0, "forward": 0})
        self.assertEqual(state["actuator_names"], ["a0", "a1"])
        self.assertEqual(queried, ["a0", "a1"])  # exact queried names, in order
        self.assertEqual(state["actuator_ctrl"], [0.7, -0.3])  # real ctrl values
        self.assertIsNone(state["error"])

    def test_native_callable_lookup_failure_is_null_not_zero(self):
        def name2id(name):
            if name == "a0":
                return 0
            raise KeyError(name)

        env, _sim = _gripper_env(
            ["g0", "g1"],
            {"g0": [0.01], "g1": [0.02]},
            actuator_names=["a0", "bogus"],
            ctrl=[0.9, 0.1],
            native_lookup=name2id,
        )
        state = wd.capture_gripper_state(env)
        # The failing name is null with an error -- never a fabricated 0.
        self.assertEqual(state["actuator_ctrl"], [0.9, None])
        self.assertIsNotNone(state["error"])

    def test_missing_joint_is_null_with_error(self):
        env, sim = _gripper_env(["g0", "g1"], {"g0": [0.04]})
        state = wd.capture_gripper_state(env)
        self.assertEqual(sim.calls["step"], 0)
        self.assertEqual(state["joint_names"], ["g0", "g1"])
        self.assertEqual(state["qpos"], [0.04, None])
        self.assertIsNone(state["width_m"])  # not exactly two finite jaws
        self.assertIsNotNone(state["error"])

    def test_nonfinite_qpos_is_null(self):
        env, _sim = _gripper_env(["g0", "g1"], {"g0": [float("nan")], "g1": [0.0]})
        state = wd.capture_gripper_state(env)
        self.assertEqual(state["qpos"], [None, 0.0])
        self.assertIsNone(state["width_m"])
        self.assertIsNotNone(state["error"])

    def test_single_jaw_has_no_width(self):
        env, _sim = _gripper_env(["g0"], {"g0": [0.04]})
        state = wd.capture_gripper_state(env)
        self.assertEqual(state["qpos"], [0.04])
        self.assertIsNone(state["width_m"])

    def test_absent_inner_env_is_fully_null(self):
        state = wd.capture_gripper_state(types.SimpleNamespace())
        self.assertIsNone(state["joint_names"])
        self.assertIsNone(state["qpos"])
        self.assertIsNone(state["width_m"])
        self.assertIsNotNone(state["error"])


# --- observation wrapper -----------------------------------------------------


class _StepEnv:
    """A minimal env whose ``step`` records the exact action object."""

    def __init__(self, result=None, raise_exc=None) -> None:
        self.calls: list = []
        self.result = (
            result if result is not None else ("obs", 1.0, False, False, {"is_success": True})
        )
        self.raise_exc = raise_exc

    def step(self, action):
        self.calls.append(action)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result


class WineStepWrapperTests(unittest.TestCase):
    def test_calls_original_once_same_object_and_same_tuple(self):
        env = _StepEnv()
        original = env.step
        sink: list = []
        wrapper = wd.WineStepWrapper(env, wd.WINE_ORACLE_GOALS, sink.append)
        wrapper.install()
        self.assertIsNot(env.step, original)

        action = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        result = env.step(action)

        self.assertIs(result, env.result)  # the SAME tuple object
        self.assertEqual(len(env.calls), 1)  # exactly one physical step
        self.assertIs(env.calls[0], action)  # unchanged action object
        self.assertEqual(list(env.calls[0]), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0]["step"], 1)
        self.assertEqual(sink[0]["sent_action"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        wrapper.restore()
        # A Python descriptor builds a *fresh* bound-method object on every
        # attribute read, so the restored bound method is not the same object;
        # its underlying instance and function must be identical.
        self.assertIs(env.step.__self__, original.__self__)
        self.assertIs(env.step.__func__, original.__func__)

    def test_records_provider_state_and_preserves_raw_dimension(self):
        env = _StepEnv()
        state = {
            "raw_policy_action": [1.0, 2.0],  # deliberately NOT seven
            "postprocessed_action": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "instruction": [wd.WINE_INSTRUCTION],
            "policy_input_state": [0.1, 0.2],
        }
        sink: list = []
        wrapper = wd.WineStepWrapper(env, wd.WINE_ORACLE_GOALS, sink.append, lambda: state)
        with wrapper:
            env.step([0.0] * 7)
        sample = sink[0]
        self.assertEqual(sample["raw_policy_action"], [1.0, 2.0])
        self.assertEqual(sample["postprocessed_action"], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        self.assertEqual(sample["instruction"], [wd.WINE_INSTRUCTION])
        self.assertEqual(sample["policy_input_state"], [0.1, 0.2])

    def test_exception_restores_original_step(self):
        env = _StepEnv(raise_exc=RuntimeError("boom"))
        original = env.step
        wrapper = wd.WineStepWrapper(env, [], lambda _sample: None)
        with wrapper:
            with self.assertRaises(RuntimeError):
                env.step([0.0] * 7)
        self.assertEqual(len(env.calls), 1)  # still exactly one attempt
        # Descriptor rebinds a fresh object; instance + function must match.
        self.assertIs(env.step.__self__, original.__self__)
        self.assertIs(env.step.__func__, original.__func__)
        self.assertFalse(wrapper.installed)

    def test_no_extra_step_across_many_actions(self):
        env = _StepEnv()
        sink: list = []
        wrapper = wd.WineStepWrapper(env, [], sink.append)
        with wrapper:
            for _ in range(4):
                env.step([0.0] * 7)
        self.assertEqual(len(env.calls), 4)
        self.assertEqual(len(sink), 4)
        self.assertEqual([sample["step"] for sample in sink], [1, 2, 3, 4])


# --- raw/post capture --------------------------------------------------------


class _CaptureTarget:
    pass


class _MutatingPostV1:
    """A v1 stand-in whose post mutates the raw action in place (like torch)."""

    def __init__(self, raw) -> None:
        self.raw = raw

        def _post(action):
            action[:] = action + 100.0
            return action

        self._post = _post


class _RaisingPostV1:
    def __init__(self) -> None:
        def _post(action):  # noqa: ARG001
            raise RuntimeError("post failed")

        self._post = _post


class _NoPostV1:
    pass


class _FakeBF16Tensor:
    """A GPU-free BF16 stand-in.

    ``numpy()`` refuses to convert until ``float()`` widening has actually run on
    the detached observation copy -- exactly the bfloat16 limitation the real
    tensor path must handle.  Every method returns a new object (never mutating
    the receiver), so a copy taken by ``_flat_float_list`` is independent of the
    original tensor the caller keeps.
    """

    def __init__(self, values, *, converted=False, record=None) -> None:
        self._values = list(values)
        self._converted = converted
        self._record = record

    def detach(self):
        return _FakeBF16Tensor(self._values, converted=self._converted, record=self._record)

    def float(self):
        if self._record is not None:
            self._record.append("float")
        return _FakeBF16Tensor(self._values, converted=True, record=self._record)

    def cpu(self):
        return self

    def numpy(self):
        if not self._converted:
            raise TypeError("numpy cannot convert bfloat16 without a float() cast")
        return np.asarray(self._values, dtype=np.float32)

    def mutate_in_place(self, delta):
        for index in range(len(self._values)):
            self._values[index] += delta


class _MutatingBF16PostV1:
    """A v1 whose post mutates the BF16 raw action in place (like torch)."""

    def __init__(self, raw) -> None:
        self.raw = raw
        self.received: list = []

        def _post(action):
            self.received.append(action)
            action.mutate_in_place(100.0)
            return action

        self._post = _post


class PostActionCaptureTests(unittest.TestCase):
    def test_raw_copied_before_in_place_post_mutation(self):
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v1 = _MutatingPostV1(raw)
        original = v1._post
        target = _CaptureTarget()
        with wd.PostActionCapture(v1, target) as capture:
            returned = v1._post(v1.raw)
        # Raw captured BEFORE mutation; post result captured after.
        self.assertEqual(target._wine_raw_action, [1.0, 2.0, 3.0])
        self.assertEqual(target._wine_post_action, [101.0, 102.0, 103.0])
        self.assertIs(returned, v1.raw)  # result object unchanged
        self.assertEqual(capture.post_call_count, 1)  # original post called once
        self.assertIs(v1._post, original)  # restored

    def test_exception_restores_post(self):
        v1 = _RaisingPostV1()
        original = v1._post
        target = _CaptureTarget()
        with self.assertRaises(RuntimeError):
            with wd.PostActionCapture(v1, target):
                v1._post([1.0, 2.0, 3.0])
        self.assertIs(v1._post, original)

    def test_absent_post_is_not_installed(self):
        v1 = _NoPostV1()
        target = _CaptureTarget()
        with wd.PostActionCapture(v1, target) as capture:
            self.assertFalse(capture.installed)
        self.assertFalse(capture.installed)
        self.assertFalse(hasattr(v1, "_post"))

    def test_flat_float_list_widens_bf16_without_mutating_original(self):
        record: list = []
        raw = _FakeBF16Tensor([0.5, -1.5], record=record)
        values = wd._flat_float_list(raw)
        self.assertEqual(values, [0.5, -1.5])
        self.assertIn("float", record)  # float() widening was really invoked
        self.assertEqual(raw._values, [0.5, -1.5])  # original untouched
        self.assertFalse(raw._converted)  # original stayed BF16

    def test_bf16_raw_copied_before_in_place_post_keeps_original_object(self):
        raw = _FakeBF16Tensor([1, 2, 3], record=[])
        v1 = _MutatingBF16PostV1(raw)
        original_post = v1._post
        target = _CaptureTarget()
        with wd.PostActionCapture(v1, target) as capture:
            returned = v1._post(v1.raw)
        # The raw copy succeeded (BF16 was widened) and kept the pre-mutation
        # values even though the post mutated the raw tensor in place.
        self.assertEqual(target._wine_raw_action, [1.0, 2.0, 3.0])
        self.assertIn("float", raw._record)
        # The original post received the EXACT original tensor object.
        self.assertEqual(len(v1.received), 1)
        self.assertIs(v1.received[0], raw)
        self.assertEqual(raw._values, [101.0, 102.0, 103.0])
        self.assertIs(returned, raw)  # original post result returned unchanged
        self.assertEqual(capture.post_call_count, 1)  # called exactly once
        self.assertIs(v1._post, original_post)  # restored

    def test_action_function_seam_reports_raw_null(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            svc = wd.WineDiagnosticService(run_root=tmp)
            svc._action_function = lambda batch: np.arange(7)  # noqa: ARG005
            action = svc._select_action({"task": [wd.WINE_INSTRUCTION]})
        self.assertEqual(action.shape, (7,))
        self.assertIsNone(svc._wine_raw_action)  # no raw without a _post
        self.assertEqual(svc._wine_post_action, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(svc._wine_input_task, [wd.WINE_INSTRUCTION])


# --- summarizer --------------------------------------------------------------


def _wine_sample(
    step,
    *,
    raw=(0.0,) * 7,
    post=(0.0,) * 7,
    sent=None,
    held_before=False,
    held_after=False,
    z=0.0,
    before_z=None,
    after_z=None,
    goal=False,
    strict=None,
    width_before=None,
    width_after=None,
    native=None,
):
    if sent is None:
        sent = (0.0,) * 7

    # ``prior``/``post`` default to the single ``z`` (backward compatible); the
    # lift contract tests pass them explicitly to separate the pre-action and
    # post-action wine heights.
    prior_z = z if before_z is None else before_z
    post_z = z if after_z is None else after_z

    def snap(held, predicate, strict_value, z_value):
        return {
            "held_objects": [wd.WINE_OBJECT_ID] if held else [],
            "predicates": {wd.WINE_GOAL_KEY: predicate},
            "objects": {wd.WINE_OBJECT_ID: {"position": [0.0, 0.0, z_value]}},
            "strict_candidate": strict_value,
        }

    return {
        "step": step,
        "raw_policy_action": list(raw),
        "postprocessed_action": list(post),
        "sent_action": list(sent),
        "instruction": [wd.WINE_INSTRUCTION],
        "policy_input_state": [0.0, 0.0],
        "before_snapshot": snap(held_before, goal, strict, prior_z),
        "after_snapshot": snap(held_after, goal, strict, post_z),
        "gripper_before": {"width_m": width_before, "error": None},
        "gripper_after": {"width_m": width_after, "error": None},
        "native_success": native,
    }


class SummarizeWineSamplesTests(unittest.TestCase):
    def _observed(self):
        return [
            _wine_sample(
                1, held_before=True, held_after=True, z=0.10, goal=False, strict=None,
                sent=[0, 0, 0, 0, 0, 0, -1],
            ),
            _wine_sample(2, held_after=True, z=0.23, goal=False, strict=None,
                         sent=[0, 0, 0, 0, 0, 0, 1]),
            _wine_sample(3, z=0.24, goal=True, strict=True, sent=[0, 0, 0, 0, 0, 0, -1],
                         width_before=0.01, width_after=0.05),
            _wine_sample(4, z=0.24, goal=True, strict=True),
            _wine_sample(5, z=0.24, goal=True, strict=True),
            _wine_sample(6, z=0.24, goal=True, strict=True),
            _wine_sample(7, z=0.24, goal=True, strict=True),
        ]

    def test_observed_metrics(self):
        summary = wd.summarize_wine_samples(self._observed())
        self.assertEqual(summary["n_samples"], 7)
        self.assertEqual(summary["first_grasp_proxy_step"], 1)
        self.assertEqual(summary["max_grasp_proxy_streak"], 2)
        self.assertAlmostEqual(summary["lift_reference_m"], 0.10)
        self.assertAlmostEqual(summary["max_lift_m"], 0.14)
        self.assertEqual(summary["first_lift_2cm_step"], 2)
        self.assertEqual(summary["first_goal_true_step"], 3)
        self.assertEqual(summary["first_strict_five_end_step"], 7)
        self.assertEqual(summary["sent_open_count"], 2)
        self.assertEqual(summary["sent_close_count"], 5)
        self.assertEqual(summary["sent_open_while_goal_true_count"], 1)
        self.assertEqual(summary["sent_open_while_grasp_proxy_before_count"], 1)
        self.assertAlmostEqual(summary["width_min_m"], 0.01)
        self.assertAlmostEqual(summary["width_max_m"], 0.05)
        self.assertEqual(summary["open_command_width_increased_count"], 1)
        self.assertEqual(summary["raw_policy_action_dim"]["dims"], [7])
        self.assertEqual(summary["sent_action_dim"]["dims"], [7])
        self.assertEqual(summary["raw_policy_action_dim"]["n_null"], 0)

    def test_five_consecutive_strict_requires_a_full_window(self):
        samples = [_wine_sample(i, z=0.0, goal=True, strict=True) for i in range(1, 5)]
        summary = wd.summarize_wine_samples(samples)
        self.assertIsNone(summary["first_strict_five_end_step"])

    def test_missing_fields_are_null_not_false_or_zero(self):
        samples = [{"step": i} for i in range(1, 4)]
        summary = wd.summarize_wine_samples(samples)
        self.assertEqual(summary["n_samples"], 3)
        self.assertIsNone(summary["first_grasp_proxy_step"])
        self.assertIsNone(summary["max_grasp_proxy_streak"])
        self.assertIsNone(summary["max_lift_m"])
        self.assertIsNone(summary["first_lift_2cm_step"])
        self.assertIsNone(summary["first_goal_true_step"])
        self.assertIsNone(summary["first_strict_five_end_step"])
        self.assertIsNone(summary["sent_open_count"])
        self.assertIsNone(summary["sent_close_count"])
        self.assertIsNone(summary["width_min_m"])
        self.assertIsNone(summary["width_max_m"])
        self.assertIsNone(summary["n_native_success_true"])
        self.assertEqual(summary["n_native_success_known"], 0)
        self.assertEqual(summary["raw_policy_action_dim"]["n_known"], 0)
        self.assertEqual(summary["raw_policy_action_dim"]["n_null"], 3)
        self.assertGreater(summary["null_counts"]["before_snapshot"], 0)

    def test_empty_summary_is_all_null(self):
        summary = wd.summarize_wine_samples([])
        self.assertEqual(summary["n_samples"], 0)
        for key in (
            "first_grasp_proxy_step",
            "max_grasp_proxy_streak",
            "max_lift_m",
            "first_lift_2cm_step",
            "first_goal_true_step",
            "first_strict_five_end_step",
            "sent_open_count",
            "width_min_m",
        ):
            self.assertIsNone(summary[key], key)

    def test_lift_baseline_is_first_before_and_post_action_heights(self):
        # The immutable baseline is the FIRST sample's before-snapshot wine z
        # (0.10).  Per-action heights come from the after-snapshot z of EVERY
        # sample, INCLUDING the final action (0.20).
        samples = [
            _wine_sample(1, before_z=0.10, after_z=0.13),
            _wine_sample(2, before_z=0.13, after_z=0.20),
        ]
        summary = wd.summarize_wine_samples(samples)
        self.assertAlmostEqual(summary["lift_reference_m"], 0.10)
        self.assertEqual(summary["lift_reference_step"], 1)
        self.assertEqual(summary["first_lift_2cm_step"], 1)
        self.assertAlmostEqual(summary["max_lift_m"], 0.10)

    def test_missing_first_before_z_nulls_all_lift_metrics(self):
        # The first sample's before-snapshot wine z is unreadable while a LATER
        # sample has a readable before AND after z.  A later height must NEVER be
        # substituted as the baseline: every lift metric stays null.
        first = _wine_sample(1, z=0.05)
        first["before_snapshot"]["objects"] = {}
        later = _wine_sample(2, before_z=0.05, after_z=0.25)
        summary = wd.summarize_wine_samples([first, later])
        self.assertIsNone(summary["lift_reference_m"])
        self.assertIsNone(summary["lift_reference_step"])
        self.assertIsNone(summary["max_lift_m"])
        self.assertIsNone(summary["first_lift_2cm_step"])

    def test_width_change_is_distinguished_from_object_lift(self):
        # The jaw opens (negative sent command, width grows) while the object z
        # is unchanged: a width change must NOT be reported as an object lift.
        samples = [
            _wine_sample(1, z=0.50, width_before=0.01, width_after=0.02,
                         sent=[0, 0, 0, 0, 0, 0, -1]),
            _wine_sample(2, z=0.50, width_before=0.02, width_after=0.05,
                         sent=[0, 0, 0, 0, 0, 0, -1]),
        ]
        summary = wd.summarize_wine_samples(samples)
        self.assertAlmostEqual(summary["max_lift_m"], 0.0)
        self.assertIsNone(summary["first_lift_2cm_step"])
        self.assertGreaterEqual(summary["open_command_width_increased_count"], 1)
        self.assertAlmostEqual(summary["width_max_m"], 0.05)

    def test_unknown_grasp_never_counts_as_hold(self):
        samples = [{"step": 1, "before_snapshot": {}, "after_snapshot": {}}]
        summary = wd.summarize_wine_samples(samples)
        self.assertIsNone(summary["first_grasp_proxy_step"])
        self.assertIsNone(summary["max_grasp_proxy_streak"])


# --- scene registration ------------------------------------------------------


class RegisterNativeWineSceneTests(unittest.TestCase):
    def _snapshot(self):
        return (
            copy.deepcopy(catalog.SCENES),
            copy.deepcopy(catalog.CAPABILITIES),
            copy.deepcopy(pe.FINAL_ORACLE_GOALS),
        )

    def test_success_restores_identity_and_contents(self):
        scenes_obj = catalog.SCENES
        caps_obj = catalog.CAPABILITIES
        goals_obj = pe.FINAL_ORACLE_GOALS
        scenes_before, caps_before, goals_before = self._snapshot()
        wine_ids_before = list(catalog.CAPABILITIES["wine_to_rack"]["scene_ids"])

        with wd.register_native_wine_scene() as scene_id:
            self.assertEqual(scene_id, wd.WINE_SCENE_ID)
            self.assertIn(scene_id, catalog.SCENES)
            scene = catalog.SCENES[scene_id]
            self.assertEqual(scene["suite"], "libero_goal")
            self.assertEqual(scene["task_id"], 9)
            self.assertEqual(scene["variant"], "original")
            self.assertEqual(scene["patches"], [])
            self.assertEqual(scene["storage_policy"], {"wine_bottle_1": "wine_rack_1_top_region"})
            self.assertIn(scene_id, catalog.CAPABILITIES["wine_to_rack"]["scene_ids"])
            # Only the wine capability's scene id list changed.
            self.assertEqual(
                catalog.CAPABILITIES["wine_to_rack"]["instruction"],
                "put the wine bottle on the rack",
            )
            # The exact fixed key holds the literal preauthored wine goals.
            self.assertEqual(wd.FINAL_ORACLE_KEY, "wine_stage_diagnostic")
            self.assertIn(wd.FINAL_ORACLE_KEY, pe.FINAL_ORACLE_GOALS)
            self.assertEqual(
                pe.FINAL_ORACLE_GOALS[wd.FINAL_ORACLE_KEY],
                [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
            )
            # ...registered as an *independent* copy, never the shared literal.
            self.assertIsNot(
                pe.FINAL_ORACLE_GOALS[wd.FINAL_ORACLE_KEY], wd.WINE_ORACLE_GOALS
            )
            pe.FINAL_ORACLE_GOALS[wd.FINAL_ORACLE_KEY].append(["sentinel"])
            self.assertNotIn(["sentinel"], wd.WINE_ORACLE_GOALS)

        self.assertIs(catalog.SCENES, scenes_obj)
        self.assertIs(catalog.CAPABILITIES, caps_obj)
        self.assertIs(pe.FINAL_ORACLE_GOALS, goals_obj)
        self.assertEqual(catalog.SCENES, scenes_before)
        self.assertEqual(catalog.CAPABILITIES, caps_before)
        self.assertEqual(pe.FINAL_ORACLE_GOALS, goals_before)
        self.assertEqual(catalog.CAPABILITIES["wine_to_rack"]["scene_ids"], wine_ids_before)

    def test_exception_restores_identity_and_contents(self):
        scenes_obj = catalog.SCENES
        caps_obj = catalog.CAPABILITIES
        goals_obj = pe.FINAL_ORACLE_GOALS
        scenes_before, caps_before, goals_before = self._snapshot()

        with self.assertRaises(RuntimeError):
            with wd.register_native_wine_scene():
                self.assertIn(wd.WINE_SCENE_ID, catalog.SCENES)
                self.assertEqual(
                    pe.FINAL_ORACLE_GOALS[wd.FINAL_ORACLE_KEY],
                    [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
                )
                raise RuntimeError("boom")

        self.assertIs(catalog.SCENES, scenes_obj)
        self.assertIs(catalog.CAPABILITIES, caps_obj)
        self.assertIs(pe.FINAL_ORACLE_GOALS, goals_obj)
        self.assertEqual(catalog.SCENES, scenes_before)
        self.assertEqual(catalog.CAPABILITIES, caps_before)
        self.assertEqual(pe.FINAL_ORACLE_GOALS, goals_before)

    def test_literal_contract(self):
        self.assertEqual(
            wd.WINE_ORACLE_GOALS,
            [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
        )
        self.assertEqual(wd.FINAL_ORACLE_KEY, "wine_stage_diagnostic")
        self.assertEqual(wd.WINE_INSTRUCTION, "put the wine bottle on the rack")


# --- campaign trial classification -------------------------------------------


class _FakeSessionRecord:
    initial_state_hash = "init-sha"
    xml_sha = "xml-sha"


class _FakeTrialService:
    """A pure stand-in exposing exactly what ``_run_trial`` reads/writes.

    No worker, no CUDA, no network: ``create_session`` returns a synthetic
    session, ``configure_profile`` succeeds, ``job`` reads a fixed mapping and
    ``final_snapshot`` returns an unknown snapshot.  ``pe._submit_and_wait`` is
    patched by the tests.
    """

    def __init__(self, jobs=None, task_language=None) -> None:
        self._jobs = dict(jobs or {})
        self._sessions = {}
        self._active_request_id = None
        self._env = object()  # present, so the final-snapshot branch runs
        self._diag_condition = None
        self.completion_mode = wd.COMPLETION_MODE
        self._task_language = task_language
        self.create_calls = 0
        self.profile_calls = 0

    def create_session(self, scene_id, seed=0, init_state_index=0):  # noqa: ARG002
        self.create_calls += 1
        self._sessions["sess-1"] = _FakeSessionRecord()
        return {
            "ok": True,
            "session_id": "sess-1",
            "env_instance_id": "env-1",
            "episode_resets": 1,
            "policy_resets": 1,
            "scene_version": 1,
        }

    def _sync_work(self, name, work):  # noqa: ARG002
        return {"ok": True, "task_language": self._task_language}

    def configure_profile(self, name):  # noqa: ARG002
        self.profile_calls += 1
        return {"ok": True}

    def job(self, job_id):
        return self._jobs.get(job_id)

    def final_snapshot(self, goals):  # noqa: ARG002
        return {"ok": True, "snapshot": None}


class RunTrialClassificationTests(unittest.TestCase):
    def _run(self, plan_record, jobs=None):
        service = _FakeTrialService(jobs=jobs, task_language=wd.WINE_INSTRUCTION)
        with mock.patch.object(wd.pe, "_submit_and_wait", return_value=plan_record):
            trial = wd._run_trial(
                service,
                "native_goal9",
                "0:0",
                5.0,
                300,
                None,
                None,
                Path("/tmp"),
            )
        # The exact new diagnostic key is used for the condition.
        self.assertEqual(service._diag_condition, "wine_stage_diagnostic")
        return trial

    def test_plan_timeout_is_operational_never_physical(self):
        plan_record = {
            "request_id": "req-timeout",
            "submitted": True,
            "submit_error": None,
            "timed_out": True,
            "cancel_nonterminal": False,
            "cancelled": False,
            "wall_s": 5.0,
            "plan": {
                "state": "timeout",
                "plan_success": None,
                "completed_capability_ids": [],
                "pending_capability_ids": [wd.WINE_CAPABILITY_ID],
                "job_ids": [],
            },
        }
        trial = self._run(plan_record)
        self.assertEqual(trial["plan_terminal_state"], "timeout")
        self.assertTrue(any("plan_timeout" in e for e in trial["operational_errors"]))
        # An incomplete timed-out trajectory is NOT an ordinary physical failure.
        self.assertEqual(trial["physical_failures"], [])

    def test_budget_exhaustion_is_ordinary_failure_not_operational(self):
        with tempfile.TemporaryDirectory() as run_dir:
            plan_record = {
                "request_id": "req-budget",
                "submitted": True,
                "submit_error": None,
                "timed_out": False,
                "cancel_nonterminal": False,
                "cancelled": False,
                "wall_s": 3.0,
                "plan": {
                    "state": "blocked",
                    "plan_success": False,
                    "completed_capability_ids": [],
                    "pending_capability_ids": [wd.WINE_CAPABILITY_ID],
                    "job_ids": ["job-1"],
                },
            }
            jobs = {
                "job-1": {
                    "job_id": "job-1",
                    "run_dir": run_dir,
                    "state": "blocked",
                    "ended_reason": "budget_exhausted",
                    "error": None,
                    "success": False,
                    "instruction": wd.WINE_INSTRUCTION,
                    "steps": 300,
                    "total_steps": 300,
                }
            }
            trial = self._run(plan_record, jobs=jobs)
        self.assertEqual(trial["operational_errors"], [])
        self.assertTrue(any("budget_exhausted" in f for f in trial["physical_failures"]))


# --- CLI contract ------------------------------------------------------------


class CliContractTests(unittest.TestCase):
    def test_defaults_without_gpu(self):
        parser = wd._build_parser()
        args = parser.parse_args(["--output", "/tmp/wine.json", "--run-root", "/tmp/wine_runs"])
        self.assertEqual(args.pairs, ["0:0", "1:1", "2:2"])
        self.assertEqual(args.conditions, ["native_goal9", "shared_goal8"])
        self.assertEqual(args.timeout, 900.0)
        self.assertEqual(args.budget, 300)

    def test_unknown_condition_is_rejected(self):
        parser = wd._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--conditions", "bogus", "--output", "/tmp/w.json", "--run-root", "/tmp/wr"]
            )

    def test_absolute_and_non_existing_validation(self):
        parser = wd._build_parser()
        args = parser.parse_args(
            ["--output", "relative.json", "--run-root", "/tmp/wine_runs"]
        )
        with self.assertRaises(SystemExit):
            wd._validate_args(args, parser)


if __name__ == "__main__":
    unittest.main(verbosity=2)
