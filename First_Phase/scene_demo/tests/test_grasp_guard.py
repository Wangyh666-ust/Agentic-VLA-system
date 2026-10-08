#!/usr/bin/env python3
"""Synthetic protocol tests for the wine-only ``grasp_guard`` monitor.

These tests are pure Python/numpy: they never build a simulator, never load a
checkpoint and never touch a GPU.  They drive :class:`grasp_guard.GraspMonitor`
with hand-built, snapshot-compatible mappings and assert the frozen candidate-
calibration protocol:

* a monitor that never arms (approaching) can never fail;
* an empty-lift retreat is only a failure after ``failure_samples`` consecutive
  candidates;
* a transient lift/contact can never confirm;
* a real lift held for ``confirm_samples`` latches ``grasp_confirmed`` forever,
  so a later intentional release can never become ``failed_grasp``;
* unknown/missing/nonfinite samples satisfy neither condition;
* an already-satisfied target disables the guard;
* the initial-z baseline never drifts;
* both arming and failure are gated by finite distance, finite command and the
  independent retreat/separation thresholds.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity with the native contract)

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import grasp_guard  # noqa: E402
from grasp_guard import GraspMonitor, GuardConfig  # noqa: E402

WINE = grasp_guard.WINE_OBJECT_ID
GOAL = grasp_guard.WINE_GOAL_KEY

_STATUS_KEYS = {
    "stage",
    "armed_step",
    "grasp_confirmed_step",
    "failure_step",
    "candidate_streak",
    "should_stop",
}


def _snapshot(position=None, grasped=None, eef=None, predicate=None):
    """A snapshot mapping with exactly the fields ``GraspMonitor`` reads."""

    return {
        "objects": {WINE: {"position": position, "grasped": grasped}},
        "eef_position": eef,
        "predicates": {GOAL: predicate},
    }


def _armed_monitor(initial_z=0.0):
    """A started monitor armed once at step 1 with a finite near command."""

    monitor = GraspMonitor()
    monitor.start(_snapshot([0.0, 0.0, initial_z], False, [0.0, 0.0, initial_z], False))
    status = monitor.update(
        1, _snapshot([0.0, 0.0, initial_z], False, [0.0, 0.0, initial_z], False), 0.5
    )
    assert status["armed_step"] == 1
    return monitor


class GraspMonitorProtocolTests(unittest.TestCase):
    """The wine-only synthetic protocol, one behaviour per test."""

    # -- arming ---------------------------------------------------------------

    def test_arming_requires_a_finite_positive_command_and_a_near_object(self):
        # A strong, positive command with the object far away must not arm.
        far = GraspMonitor()
        far.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False))
        far.update(1, _snapshot([0.0, 0.0, 0.0], False, [1.0, 0.0, 0.0], False), 0.9)
        self.assertIsNone(far.status()["armed_step"])
        self.assertEqual(far.status()["stage"], "approaching")

        # A zero / negative / nonfinite command never arms, even when near.
        for command in (0.0, -0.5, float("nan"), float("inf"), None):
            monitor = GraspMonitor()
            monitor.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False))
            monitor.update(1, _snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False), command)
            self.assertIsNone(monitor.status()["armed_step"], repr(command))

        # A nonfinite object/eef position makes the distance unknown -> no arm.
        unknown = GraspMonitor()
        unknown.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False))
        unknown.update(1, _snapshot([0.0, 0.0, 0.0], False, [float("nan"), 0.0, 0.0], False), 0.9)
        self.assertIsNone(unknown.status()["armed_step"])

        # On the near boundary it arms; a hair beyond it does not.
        boundary = GraspMonitor()
        boundary.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False))
        boundary.update(1, _snapshot([0.0, 0.0, 0.0], False, [0.14, 0.0, 0.0], False), 0.5)
        self.assertEqual(boundary.status()["armed_step"], 1)
        beyond = GraspMonitor()
        beyond.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False))
        beyond.update(1, _snapshot([0.0, 0.0, 0.0], False, [0.1401, 0.0, 0.0], False), 0.5)
        self.assertIsNone(beyond.status()["armed_step"])

    def test_arming_stores_eef_z_and_distance_once_without_overwriting(self):
        monitor = _armed_monitor(initial_z=0.0)
        # A later sample moves the eef but the stored arm reference stays fixed:
        # the failure test below would be impossible if it drifted.
        monitor.update(2, _snapshot([0.0, 0.0, 0.0], False, [0.2, 0.0, 0.1], False), 0.5)
        status = monitor.status()
        self.assertEqual(status["armed_step"], 1)

    # -- approaching cannot fail ---------------------------------------------

    def test_approaching_monitor_retreat_can_never_fail(self):
        # The eef rises and separates, but the gripper never had a positive
        # command near the bottle, so the monitor must stay unarmed and never
        # report a failure.
        monitor = GraspMonitor()
        monitor.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False))
        for step in range(1, 20):
            status = monitor.update(
                step,
                _snapshot([0.0, 0.0, 0.0], False, [0.5, 0.0, 0.4], False),
                0.0,
            )
            self.assertIsNone(status["failure_step"])
            self.assertFalse(status["should_stop"])
        self.assertNotEqual(monitor.status()["stage"], "failed_grasp")

    # -- failure needs the full streak ----------------------------------------

    def test_empty_lift_retreat_fails_only_after_five_samples(self):
        monitor = _armed_monitor(initial_z=0.0)
        retreat = _snapshot([0.0, 0.0, 0.0], False, [0.06, 0.0, 0.03], False)
        # Four candidates are not enough.
        for step in range(2, 6):
            status = monitor.update(step, retreat, 0.5)
            self.assertIsNone(status["failure_step"], step)
            self.assertFalse(status["should_stop"], step)
        self.assertEqual(monitor.status()["candidate_streak"], 4)
        # The fifth consecutive candidate latches the failure.
        status = monitor.update(6, retreat, 0.5)
        self.assertEqual(status["failure_step"], 6)
        self.assertEqual(status["stage"], "failed_grasp")
        self.assertTrue(status["should_stop"])

    def test_failure_needs_both_retreat_rise_and_separation_growth(self):
        # A purely vertical retreat of 0.03 m satisfies the eef-rise threshold
        # but not the separation-growth one.
        vertical = _armed_monitor(initial_z=0.0)
        for step in range(2, 12):
            status = vertical.update(step, _snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.03], False), 0.5)
            self.assertIsNone(status["failure_step"], step)
        self.assertNotEqual(vertical.status()["stage"], "failed_grasp")

        # A purely lateral retreat satisfies separation but never rises enough.
        lateral = _armed_monitor(initial_z=0.0)
        for step in range(2, 12):
            status = lateral.update(step, _snapshot([0.0, 0.0, 0.0], False, [0.2, 0.0, 0.0], False), 0.5)
            self.assertIsNone(status["failure_step"], step)
        self.assertNotEqual(lateral.status()["stage"], "failed_grasp")

    def test_failure_requires_the_native_predicate_to_be_false(self):
        monitor = _armed_monitor(initial_z=0.0)
        retreat_true = _snapshot([0.0, 0.0, 0.0], False, [0.06, 0.0, 0.03], True)
        for step in range(2, 12):
            status = monitor.update(step, retreat_true, 0.5)
            self.assertIsNone(status["failure_step"], step)
        self.assertNotEqual(monitor.status()["stage"], "failed_grasp")

    # -- confirmation ---------------------------------------------------------

    def test_transient_lift_and_contact_cannot_confirm(self):
        monitor = _armed_monitor(initial_z=0.0)
        step = 2
        for _ in range(4):
            lifted = monitor.update(
                step, _snapshot([0.0, 0.0, 0.03], True, [0.0, 0.0, 0.0], False), 0.5
            )
            self.assertIsNone(lifted["grasp_confirmed_step"], step)
            step += 1
            # An unknown grasp sample resets the running confirm streak.
            monitor.update(
                step, _snapshot([0.0, 0.0, 0.03], None, [0.0, 0.0, 0.0], False), 0.5
            )
            step += 1
        self.assertIsNone(monitor.status()["grasp_confirmed_step"])
        self.assertNotEqual(monitor.status()["stage"], "grasp_confirmed")

    def test_grasped_true_without_lift_cannot_confirm(self):
        monitor = _armed_monitor(initial_z=0.0)
        for step in range(2, 12):
            status = monitor.update(
                step, _snapshot([0.0, 0.0, 0.0], True, [0.0, 0.0, 0.0], False), 0.5
            )
            self.assertIsNone(status["grasp_confirmed_step"])
            self.assertIsNone(status["failure_step"])
        self.assertEqual(monitor.status()["stage"], "attempting")

    def test_real_lift_confirms_then_later_release_can_never_fail(self):
        monitor = _armed_monitor(initial_z=0.0)
        lifted = _snapshot([0.0, 0.0, 0.03], True, [0.0, 0.0, 0.0], False)
        for step in range(2, 6):
            status = monitor.update(step, lifted, 0.5)
            self.assertIsNone(status["grasp_confirmed_step"], step)
        status = monitor.update(6, lifted, 0.5)
        self.assertEqual(status["grasp_confirmed_step"], 6)
        self.assertEqual(status["stage"], "grasp_confirmed")
        self.assertFalse(status["should_stop"])

        # An intentional late release (drop + retreat + no grasp) must never be
        # reported as ``failed_grasp`` once the grasp was confirmed.
        released = _snapshot([0.0, 0.0, 0.0], False, [0.2, 0.0, 0.2], False)
        for step in range(7, 20):
            status = monitor.update(step, released, 0.5)
            self.assertEqual(status["stage"], "grasp_confirmed")
            self.assertIsNone(status["failure_step"])
            self.assertFalse(status["should_stop"])

    # -- unknown / missing / nonfinite ---------------------------------------

    def test_null_or_nonfinite_samples_cannot_confirm_or_fail(self):
        monitor = _armed_monitor(initial_z=0.0)
        for step in range(2, 14):
            status = monitor.update(
                step,
                _snapshot([float("nan"), 0.0, 0.0], None, [float("inf"), 0.0, 0.0], None),
                0.5,
            )
            self.assertIsNone(status["grasp_confirmed_step"], step)
            self.assertIsNone(status["failure_step"], step)
            self.assertFalse(status["should_stop"], step)
        # A missing snapshot entirely is likewise unknown.
        status = monitor.update(20, None, 0.5)
        self.assertIsNone(status["grasp_confirmed_step"])
        self.assertIsNone(status["failure_step"])

    def test_update_before_start_is_unknown_and_status_schema_is_exact(self):
        monitor = GraspMonitor()
        status = monitor.update(1, _snapshot([0.0, 0.0, 0.0], True, [0.0, 0.0, 0.0], False), 0.5)
        self.assertEqual(status["stage"], "unknown")
        self.assertIsNone(status["armed_step"])
        self.assertFalse(status["should_stop"])
        self.assertEqual(set(status.keys()), _STATUS_KEYS)

    # -- already placed -------------------------------------------------------

    def test_already_satisfied_target_disables_the_guard(self):
        monitor = GraspMonitor()
        start = monitor.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], True))
        self.assertEqual(start["stage"], "already_placed")
        # Even a perfect ghost-grasp signature never stops an already-placed run.
        for step in range(1, 12):
            status = monitor.update(
                step, _snapshot([0.0, 0.0, 0.0], False, [0.06, 0.0, 0.03], False), 0.9
            )
            self.assertEqual(status["stage"], "already_placed")
            self.assertFalse(status["should_stop"])
            self.assertIsNone(status["failure_step"])

    # -- baseline stability ---------------------------------------------------

    def test_baseline_z_is_fixed_at_start_and_does_not_drift(self):
        # Baseline z = 0.0.  A first sample at z = 0.015 is below the lift
        # threshold; five later samples at z = 0.03 confirm ONLY because the
        # baseline stayed at 0.0 (0.03 - 0.015 would be below 0.02).
        monitor = GraspMonitor()
        monitor.start(_snapshot([0.0, 0.0, 0.0], False, [0.0, 0.0, 0.0], False))
        monitor.update(1, _snapshot([0.0, 0.0, 0.015], False, [0.0, 0.0, 0.015], False), 0.5)
        self.assertIsNone(monitor.status()["grasp_confirmed_step"])
        lifted = _snapshot([0.0, 0.0, 0.03], True, [0.0, 0.0, 0.03], False)
        for step in range(2, 6):
            status = monitor.update(step, lifted, 0.5)
            self.assertIsNone(status["grasp_confirmed_step"], step)
        status = monitor.update(6, lifted, 0.5)
        self.assertEqual(status["grasp_confirmed_step"], 6)

    # -- frozen config --------------------------------------------------------

    def test_default_config_matches_the_frozen_candidate_calibration(self):
        config = GuardConfig()
        self.assertEqual(config.near_distance_m, 0.14)
        self.assertEqual(config.lift_success_m, 0.02)
        self.assertEqual(config.bottle_still_m, 0.01)
        self.assertEqual(config.retreat_rise_m, 0.03)
        self.assertEqual(config.separation_growth_m, 0.05)
        self.assertEqual(config.confirm_samples, 5)
        self.assertEqual(config.failure_samples, 5)
        with self.assertRaises(Exception):
            config.near_distance_m = 0.9  # frozen dataclass


class _FakeGripper:
    important_sites = None
    _ref_gripper_joint_pos_indexes = None


class _FakeRobot:
    def __init__(self):
        self.gripper = _FakeGripper()


class _FakeObject:
    def __init__(self, contact_geoms=(), *, has_contact_geoms=True):
        self.joints = []
        if has_contact_geoms:
            self.contact_geoms = contact_geoms


class _RecordingCheck:
    """A stand-in ``_check_grasp`` that records every invocation."""

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def __call__(self, gripper, contact_geoms):
        self.calls.append((gripper, contact_geoms))
        return self.result


class _FakeInner:
    def __init__(self, obj, check):
        self.sim = None  # no data/model -> position/eef/qpos stay unknown
        self.objects_dict = {WINE: obj}
        self.robots = [_FakeRobot()]
        self._check_grasp = check


def _probe_env(obj, check):
    return types.SimpleNamespace(_env=types.SimpleNamespace(env=_FakeInner(obj, check)))


class ReadProbeGraspUncertaintyTests(unittest.TestCase):
    """Fail-closed contact-proxy behaviour of ``read_probe`` (no simulator).

    The raw ``inner._check_grasp`` return is validated rather than coerced, and
    a missing/empty contact-geom set never triggers the probe.  Unknown is never
    reported as ``False``.
    """

    def _grasp(self, obj, check):
        probe = grasp_guard.read_probe(_probe_env(obj, check), WINE, GOAL)
        return probe["objects"][WINE]["grasped"]

    def test_none_raw_return_is_unknown(self):
        check = _RecordingCheck(None)
        self.assertIsNone(self._grasp(_FakeObject(["g0"]), check))
        self.assertEqual(len(check.calls), 1)

    def test_nonbool_raw_return_is_unknown_and_never_coerced(self):
        # ``bool()`` on these would produce True/False; the probe must not.
        for raw in (1, 0, -1, np.int64(1), "True", "", np.array([True]), object()):
            with self.subTest(raw=repr(raw)):
                check = _RecordingCheck(raw)
                self.assertIsNone(self._grasp(_FakeObject(["g0"]), check))
                self.assertEqual(len(check.calls), 1)

    def test_numpy_bool_raw_return_is_accepted(self):
        self.assertIs(self._grasp(_FakeObject(["g0"]), _RecordingCheck(np.bool_(True))), True)
        self.assertIs(self._grasp(_FakeObject(["g0"]), _RecordingCheck(np.bool_(False))), False)

    def test_python_bool_raw_return_is_accepted(self):
        self.assertIs(self._grasp(_FakeObject(["g0"]), _RecordingCheck(True)), True)
        self.assertIs(self._grasp(_FakeObject(["g0"]), _RecordingCheck(False)), False)

    def test_missing_none_or_empty_contact_geoms_stay_unknown_without_probe(self):
        cases = {
            "missing": _FakeObject(has_contact_geoms=False),
            "none": _FakeObject(None),
            "empty": _FakeObject([]),
        }
        for name, obj in cases.items():
            with self.subTest(case=name):
                check = _RecordingCheck(True)
                self.assertIsNone(self._grasp(obj, check))
                self.assertEqual(check.calls, [], "_check_grasp must not be invoked")


if __name__ == "__main__":
    unittest.main(verbosity=2)
