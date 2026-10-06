#!/usr/bin/env python3
"""GPU-free unit tests for ``scene_demo/placement_experiments.py``.

These tests never load a checkpoint, never build a LIBERO scene and never touch
CUDA: they exercise the diagnostic ``capture_snapshot`` / ``summarize_samples``
seams against mock object interfaces, validate the immutable profile/condition
contracts and the literal final oracle goals, and confirm that the
``DiagnosticService`` overrides apply the profile to a mock policy config
without starting the worker.
"""

from __future__ import annotations

import json
import sys
import tempfile
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

import placement_experiments as pe  # noqa: E402
import service  # noqa: E402

FREE = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
REST = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _SnapData:
    """Minimal native ``sim.data``."""

    def __init__(self) -> None:
        self.joint_qpos: dict[str, np.ndarray] = {}
        self.joint_qvel: dict[str, np.ndarray] = {}
        self.site_xpos: dict[str, np.ndarray] = {}
        self.qpos = np.zeros(32, dtype=np.float64)

    def get_joint_qpos(self, name):
        if name not in self.joint_qpos:
            raise KeyError(name)
        return np.asarray(self.joint_qpos[name], dtype=np.float64)

    def get_joint_qvel(self, name):
        if name not in self.joint_qvel:
            raise KeyError(name)
        return np.asarray(self.joint_qvel[name], dtype=np.float64)

    def get_site_xpos(self, name):
        return np.asarray(self.site_xpos[name], dtype=np.float64)


def _make_env(objects, predicate_map, grasp_map, gripper_indexes=(0, 1)):
    """Build a mock env wrapping a native-shaped inner env.

    ``objects`` is a list of ``(object_id, joint_name, qpos, qvel_or_None)``.
    A ``qpos``/``qvel`` of ``"raise"`` means the joint is not registered, so the
    native ``get_joint_qpos``/``get_joint_qvel`` raises (unreadable joint).
    ``predicate_map`` / ``grasp_map`` may map a key to ``"raise"`` to simulate an
    unavailable probe.
    """

    data = _SnapData()
    inner_objects: dict[str, types.SimpleNamespace] = {}
    for object_id, joint, qpos, qvel in objects:
        inner_objects[object_id] = types.SimpleNamespace(joints=[joint], contact_geoms=object_id)
        if not (isinstance(qpos, str) and qpos == "raise"):
            data.joint_qpos[joint] = np.asarray(qpos, dtype=np.float64)
        if qvel is not None and not (isinstance(qvel, str) and qvel == "raise"):
            data.joint_qvel[joint] = np.asarray(qvel, dtype=np.float64)
    for index in gripper_indexes:
        data.qpos[index] = 0.04
    data.site_xpos["gripper0_grip_site"] = np.array([0.5, 0.1, 0.2])

    gripper = types.SimpleNamespace(important_sites={"grip_site": "gripper0_grip_site"})
    robot = types.SimpleNamespace(
        gripper=gripper, _ref_gripper_joint_pos_indexes=list(gripper_indexes)
    )

    class _Inner:
        def __init__(self) -> None:
            self.robots = [robot]
            self.sim = types.SimpleNamespace(data=data)
            self.objects_dict = inner_objects

        def _check_grasp(self, gripper_, geoms):  # noqa: ARG002
            outcome = grasp_map.get(geoms, False)
            if outcome == "raise":
                raise RuntimeError("grasp probe failed")
            return outcome

        def _eval_predicate(self, predicate):
            key = "|".join(str(part) for part in predicate)
            outcome = predicate_map.get(key, False)
            if outcome == "raise":
                raise RuntimeError("predicate failed")
            return outcome

    inner = _Inner()
    return types.SimpleNamespace(_env=types.SimpleNamespace(env=inner))


_SOUP_GOAL = ["in", "alphabet_soup_1", "basket_1_contain_region"]
_SOUP_KEY = "in|alphabet_soup_1|basket_1_contain_region"
_SAUCE_GOAL = ["in", "tomato_sauce_1", "basket_1_contain_region"]
_SAUCE_KEY = "in|tomato_sauce_1|basket_1_contain_region"

_BASE_OBJECTS = [
    ("alphabet_soup_1", "soup_j", list(FREE), list(REST)),
    ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
]


class CaptureSnapshotTests(unittest.TestCase):
    def test_strict_candidate_true_when_goals_true_and_at_rest(self):
        env = _make_env(
            _BASE_OBJECTS,
            {_SOUP_KEY: True},
            {"alphabet_soup_1": False, "basket_1": False},
        )
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIs(snap["strict_candidate"], True)
        self.assertEqual(snap["held_objects"], [])
        self.assertIs(snap["grasp_observation_complete"], True)
        self.assertEqual(snap["phases"][_SOUP_KEY], "released_stable")
        self.assertEqual(snap["predicates"][_SOUP_KEY], True)
        self.assertEqual(snap["gripper_qpos"], [0.04, 0.04])
        self.assertEqual(snap["eef_position"], [0.5, 0.1, 0.2])
        entry = snap["objects"]["alphabet_soup_1"]
        self.assertEqual(entry["position"], [0.0, 0.0, 0.1])
        self.assertEqual(entry["quaternion"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(entry["linear_velocity"], [0.0, 0.0, 0.0])

    def test_unknown_grasp_fails_closed(self):
        env = _make_env(
            _BASE_OBJECTS, {_SOUP_KEY: True}, {"alphabet_soup_1": "raise", "basket_1": False}
        )
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIsNone(snap["strict_candidate"])
        self.assertIs(snap["grasp_observation_complete"], False)
        self.assertEqual(snap["phases"][_SOUP_KEY], "unknown")
        self.assertIsNone(snap["objects"]["alphabet_soup_1"]["grasped"])

    def test_unknown_velocity_fails_closed(self):
        objects = [
            ("alphabet_soup_1", "soup_j", list(FREE), None),  # velocity unsupported
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIsNone(snap["objects"]["alphabet_soup_1"]["linear_velocity"])
        self.assertIsNone(snap["strict_candidate"])

    def test_held_object_is_strict_false_despite_native_predicate_true(self):
        # Native predicate (from the mock sim truth) is True, but the contact
        # proxy says the gripper still holds the can -> strict release is False.
        env = _make_env(_BASE_OBJECTS, {_SOUP_KEY: True}, {"alphabet_soup_1": True, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertEqual(snap["predicates"][_SOUP_KEY], True)
        self.assertEqual(snap["held_objects"], ["alphabet_soup_1"])
        self.assertIs(snap["strict_candidate"], False)
        self.assertEqual(snap["phases"][_SOUP_KEY], "goal_held")

    def test_linear_stability_threshold(self):
        moving = [
            ("alphabet_soup_1", "soup_j", list(FREE), [0.03, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(moving, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIs(snap["strict_candidate"], False)
        self.assertEqual(snap["phases"][_SOUP_KEY], "released_unsettled")

        at_limit = [
            ("alphabet_soup_1", "soup_j", list(FREE), [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(at_limit, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIs(snap["strict_candidate"], True)

    def test_angular_stability_threshold(self):
        spinning = [
            ("alphabet_soup_1", "soup_j", list(FREE), [0.0, 0.0, 0.0, 0.3, 0.0, 0.0]),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(spinning, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        self.assertIs(pe.capture_snapshot(env, [_SOUP_GOAL])["strict_candidate"], False)

        at_limit = [
            ("alphabet_soup_1", "soup_j", list(FREE), [0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(at_limit, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        self.assertIs(pe.capture_snapshot(env, [_SOUP_GOAL])["strict_candidate"], True)

    def test_transporting_and_ungrasped_phases(self):
        transporting = _make_env(
            _BASE_OBJECTS, {_SOUP_KEY: False}, {"alphabet_soup_1": True, "basket_1": False}
        )
        snap = pe.capture_snapshot(transporting, [_SOUP_GOAL])
        self.assertEqual(snap["phases"][_SOUP_KEY], "transporting")
        self.assertIs(snap["strict_candidate"], False)

        ungrasped = _make_env(
            _BASE_OBJECTS, {_SOUP_KEY: False}, {"alphabet_soup_1": False, "basket_1": False}
        )
        snap = pe.capture_snapshot(ungrasped, [_SOUP_GOAL])
        self.assertEqual(snap["phases"][_SOUP_KEY], "ungrasped")

    def test_unknown_predicate_is_never_success(self):
        env = _make_env(
            _BASE_OBJECTS, {_SOUP_KEY: "raise"}, {"alphabet_soup_1": False, "basket_1": False}
        )
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIsNone(snap["predicates"][_SOUP_KEY])
        self.assertIsNone(snap["strict_candidate"])
        self.assertEqual(snap["phases"][_SOUP_KEY], "unknown")

    def test_non_movable_fixture_is_probed_but_not_held(self):
        objects = list(_BASE_OBJECTS) + [
            ("plate_1", "plate_j", [0.1, 0.2, 0.3], None),  # 3-DOF, not a free joint
        ]
        env = _make_env(objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        # The jointed, non-goal fixture IS probed for grasp (never silently
        # skipped), but a known-unknown-free False grasp does not hold it.
        self.assertIs(snap["objects"]["plate_1"]["grasped"], False)
        self.assertEqual(snap["objects"]["plate_1"]["position"], [0.1, 0.2, 0.3])
        self.assertNotIn("plate_1", snap["held_objects"])
        self.assertIs(snap["grasp_observation_complete"], True)
        self.assertIs(snap["strict_candidate"], True)

    def test_speed_rejects_malformed_and_nonfinite(self):
        self.assertIsNone(pe._speed(None))
        self.assertIsNone(pe._speed([1.0, 2.0]))
        self.assertIsNone(pe._speed([1.0, 2.0, 3.0, 4.0]))
        self.assertIsNone(pe._speed([float("nan"), 0.0, 0.0]))
        self.assertIsNone(pe._speed([float("inf"), 0.0, 0.0]))
        self.assertIsNone(pe._speed("not a vector"))
        self.assertAlmostEqual(pe._speed([3.0, 4.0, 0.0]), 5.0)

    def test_angular_velocity_missing_is_unknown(self):
        # A 3-component qvel gives linear velocity but no angular velocity.
        objects = [
            ("alphabet_soup_1", "soup_j", list(FREE), [0.0, 0.0, 0.0]),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertEqual(snap["objects"]["alphabet_soup_1"]["linear_velocity"], [0.0, 0.0, 0.0])
        self.assertIsNone(snap["objects"]["alphabet_soup_1"]["angular_velocity"])
        self.assertIsNone(snap["strict_candidate"])
        self.assertEqual(snap["phases"][_SOUP_KEY], "released_unsettled")

    def test_nonfinite_velocity_is_unknown(self):
        objects = [
            ("alphabet_soup_1", "soup_j", list(FREE), [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        entry = snap["objects"]["alphabet_soup_1"]
        self.assertIsNone(entry["linear_velocity"])
        self.assertIsNone(entry["angular_velocity"])
        self.assertIsNone(snap["strict_candidate"])

    def test_nonfinite_angular_velocity_is_unknown(self):
        objects = [
            ("alphabet_soup_1", "soup_j", list(FREE), [0.0, 0.0, 0.0, float("inf"), 0.0, 0.0]),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIsNone(snap["strict_candidate"])

    def test_nonfinite_qpos_gives_null_geometry_and_blocks(self):
        objects = [
            ("alphabet_soup_1", "soup_j", [float("nan"), 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], list(REST)),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "basket_1": False})
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        entry = snap["objects"]["alphabet_soup_1"]
        self.assertIsNone(entry["position"])
        self.assertIsNone(entry["quaternion"])
        self.assertEqual(entry["error"], "nonfinite qpos")
        self.assertIsNone(snap["strict_candidate"])

    def test_qpos_failure_of_other_jointed_object_blocks_completion(self):
        # The goal object is fine, but another jointed (non-goal) object's qpos
        # is unreadable AND its grasp probe is unavailable: the declared goal may
        # NOT silently complete.
        objects = [
            ("alphabet_soup_1", "soup_j", list(FREE), list(REST)),
            ("tomato_sauce_1", "sauce_j", "raise", list(REST)),
        ]
        env = _make_env(
            objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "tomato_sauce_1": "raise"}
        )
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        entry = snap["objects"]["tomato_sauce_1"]
        self.assertIsNone(entry["position"])
        self.assertIsNotNone(entry["error"])
        self.assertIsNone(entry["grasped"])
        self.assertIs(snap["grasp_observation_complete"], False)
        self.assertIsNone(snap["strict_candidate"])

    def test_qpos_failure_of_other_jointed_object_held_blocks(self):
        objects = [
            ("alphabet_soup_1", "soup_j", list(FREE), list(REST)),
            ("tomato_sauce_1", "sauce_j", "raise", list(REST)),
        ]
        env = _make_env(
            objects, {_SOUP_KEY: True}, {"alphabet_soup_1": False, "tomato_sauce_1": True}
        )
        snap = pe.capture_snapshot(env, [_SOUP_GOAL])
        self.assertIn("tomato_sauce_1", snap["held_objects"])
        self.assertIs(snap["strict_candidate"], False)

    def test_stable_goal_not_unsettled_by_another_goal(self):
        # Soup is released and stable; sauce is still held.  The held goal can
        # never relabel the stable, released goal as unsettled.
        objects = [
            ("alphabet_soup_1", "soup_j", list(FREE), list(REST)),
            ("tomato_sauce_1", "sauce_j", list(FREE), list(REST)),
            ("basket_1", "basket_j", [0.5, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], list(REST)),
        ]
        env = _make_env(
            objects,
            {_SOUP_KEY: True, _SAUCE_KEY: True},
            {"alphabet_soup_1": False, "tomato_sauce_1": True, "basket_1": False},
        )
        snap = pe.capture_snapshot(env, [_SOUP_GOAL, _SAUCE_GOAL])
        self.assertEqual(snap["phases"][_SOUP_KEY], "released_stable")
        self.assertEqual(snap["phases"][_SAUCE_KEY], "goal_held")
        # The global strict verdict still fails because an object is held.
        self.assertIs(snap["strict_candidate"], False)
        self.assertIn("tomato_sauce_1", snap["held_objects"])


def _sample(strict, native=None, objects=None, predicates=None, phases=None):
    snapshot = {"strict_candidate": strict}
    if objects is not None:
        snapshot["objects"] = objects
    if predicates is not None:
        snapshot["predicates"] = predicates
    if phases is not None:
        snapshot["phases"] = phases
    return {"step": 1, "action": [0.0] * 7, "snapshot": snapshot, "native_success": native}


class SummarizeSamplesTests(unittest.TestCase):
    def test_five_consecutive_strict_true(self):
        summary = pe.summarize_samples([_sample(True) for _ in range(5)])
        self.assertTrue(summary["strict_success_ever"])
        self.assertTrue(summary["strict_success_final"])
        self.assertEqual(summary["max_strict_streak"], 5)

    def test_fewer_than_five_is_never_success(self):
        summary = pe.summarize_samples([_sample(True) for _ in range(4)])
        self.assertFalse(summary["strict_success_ever"])
        self.assertFalse(summary["strict_success_final"])
        self.assertEqual(summary["max_strict_streak"], 4)

    def test_zero_samples_never_fake_success(self):
        summary = pe.summarize_samples([])
        self.assertEqual(summary["n_samples"], 0)
        self.assertFalse(summary["strict_success_ever"])
        self.assertFalse(summary["strict_success_final"])
        self.assertIsNone(summary["native_success_final"])
        self.assertIsNone(summary["final_snapshot"])

    def test_interrupted_streak(self):
        values = [True, True, True, True, True, False, True, True, True, True]
        summary = pe.summarize_samples([_sample(value) for value in values])
        self.assertTrue(summary["strict_success_ever"])
        self.assertEqual(summary["max_strict_streak"], 5)
        # The final window contains the interrupting False -> not final success.
        self.assertFalse(summary["strict_success_final"])

    def test_final_five_all_true_after_longer_run(self):
        summary = pe.summarize_samples([_sample(True) for _ in range(7)])
        self.assertTrue(summary["strict_success_ever"])
        self.assertTrue(summary["strict_success_final"])

    def test_native_and_strict_are_separate(self):
        samples = [_sample(False, native=True) for _ in range(6)]
        summary = pe.summarize_samples(samples)
        self.assertTrue(summary["native_success_ever"])
        self.assertTrue(summary["native_success_final"])
        self.assertFalse(summary["strict_success_ever"])
        self.assertFalse(summary["strict_success_final"])

    def test_unknown_strict_is_never_success(self):
        summary = pe.summarize_samples([_sample(None) for _ in range(5)])
        self.assertFalse(summary["strict_success_ever"])
        self.assertFalse(summary["strict_success_final"])
        self.assertEqual(summary["n_strict_unknown"], 5)
        self.assertGreater(summary["telemetry_nulls"], 0)

    def test_per_object_grasp_counts_and_phases_last20(self):
        objects = {"alphabet_soup_1": {"grasped": True, "linear_velocity": [0, 0, 0],
                                       "angular_velocity": [0, 0, 0]}}
        samples = [
            _sample(False, objects=objects, phases={_SOUP_KEY: "goal_held"}) for _ in range(25)
        ]
        summary = pe.summarize_samples(samples)
        # Only the last 20 samples count.
        self.assertEqual(summary["per_object_grasp_counts"]["alphabet_soup_1"]["true"], 20)
        self.assertEqual(summary["phase_counts_last20"]["goal_held"], 20)


def _oracle_sample(strict):
    return {"strict_candidate": strict}


class StrictFinalScoreTests(unittest.TestCase):
    """The pure fixed-oracle final-window scorer."""

    def test_final_and_window_all_true(self):
        self.assertIs(
            pe.strict_final_score(
                {"strict_candidate": True}, [_oracle_sample(True) for _ in range(5)]
            ),
            True,
        )

    def test_missing_final_snapshot_is_unknown(self):
        self.assertIsNone(
            pe.strict_final_score(None, [_oracle_sample(True) for _ in range(5)])
        )

    def test_unknown_final_snapshot_is_unknown(self):
        self.assertIsNone(
            pe.strict_final_score({"strict_candidate": None}, [_oracle_sample(True) for _ in range(5)])
        )

    def test_fewer_than_five_samples_is_false(self):
        self.assertIs(
            pe.strict_final_score(
                {"strict_candidate": True}, [_oracle_sample(True) for _ in range(4)]
            ),
            False,
        )

    def test_known_false_final_is_false(self):
        self.assertIs(
            pe.strict_final_score(
                {"strict_candidate": False}, [_oracle_sample(True) for _ in range(5)]
            ),
            False,
        )

    def test_unknown_in_required_window_is_unknown(self):
        samples = [_oracle_sample(True) for _ in range(4)] + [_oracle_sample(None)]
        self.assertIsNone(pe.strict_final_score({"strict_candidate": True}, samples))

    def test_earlier_five_true_then_final_unstable_fails(self):
        # Earlier samples were stable, but the final window is not -> no success.
        samples = [_oracle_sample(True) for _ in range(5)] + [_oracle_sample(False)]
        self.assertIs(
            pe.strict_final_score({"strict_candidate": False}, samples), False
        )

    def test_final_success_but_earlier_oracle_unsettled_fails(self):
        # The final snapshot claims success, but an earlier settled-then-unsettled
        # object is inside the required final window -> the score fails.
        samples = [
            _oracle_sample(True),
            _oracle_sample(True),
            _oracle_sample(False),
            _oracle_sample(True),
            _oracle_sample(True),
            _oracle_sample(True),
        ]
        self.assertIs(pe.strict_final_score({"strict_candidate": True}, samples), False)

    def test_accepts_full_oracle_snapshots(self):
        samples = [{"snapshot": _oracle_sample(True)} for _ in range(5)]
        self.assertIs(pe.strict_final_score({"strict_candidate": True}, samples), True)


class ProfileAndConditionContractTests(unittest.TestCase):
    def test_profile_values_are_exact(self):
        expected = {
            "baseline_bf16": {"amp": True, "num_steps": 10, "n_action_steps": 1},
            "fp32": {"amp": False, "num_steps": 10, "n_action_steps": 1},
            "fp32_d1": {"amp": False, "num_steps": 1, "n_action_steps": 1},
            "fp32_h5": {"amp": False, "num_steps": 10, "n_action_steps": 5},
            "fp32_d1_h5": {"amp": False, "num_steps": 1, "n_action_steps": 5},
        }
        self.assertEqual(set(pe.PROFILES), set(expected))
        for name, profile in expected.items():
            self.assertEqual(dict(pe.PROFILES[name]), profile, name)

    def test_profiles_are_immutable(self):
        with self.assertRaises(TypeError):
            pe.PROFILES["fp32"]["amp"] = True  # type: ignore[index]
        with self.assertRaises(TypeError):
            pe.PROFILES["new"] = {}  # type: ignore[index]

    def test_condition_contracts(self):
        self.assertEqual(tuple(pe.CONDITIONS["soup_fresh"]["capability_ids"]), ("soup_to_basket",))
        self.assertEqual(pe.CONDITIONS["soup_fresh"]["budget_per_subgoal"], 300)
        self.assertEqual(tuple(pe.CONDITIONS["basket_native"]["capability_ids"]), ("basket_both",))
        self.assertEqual(pe.CONDITIONS["basket_native"]["budget_per_subgoal"], 600)
        self.assertTrue(pe.CONDITIONS["basket_native"]["native_instruction"])
        self.assertEqual(
            tuple(pe.CONDITIONS["basket_split"]["capability_ids"]),
            ("soup_to_basket", "sauce_to_basket"),
        )
        self.assertEqual(pe.CONDITIONS["basket_split"]["mode"], "single")
        self.assertEqual(pe.CONDITIONS["basket_forced_handoff"]["mode"], "forced_handoff")
        self.assertTrue(pe.CONDITIONS["basket_forced_handoff"]["deliberately_forced_after_failure"])
        self.assertEqual(pe.CONDITIONS["bowl_control"]["scene_id"], "goal_table")

    def test_conditions_are_immutable(self):
        with self.assertRaises(TypeError):
            pe.CONDITIONS["soup_fresh"]["budget_per_subgoal"] = 1  # type: ignore[index]

    def test_final_oracle_goals_are_literal_and_independent(self):
        soup = ["in", "alphabet_soup_1", "basket_1_contain_region"]
        sauce = ["in", "tomato_sauce_1", "basket_1_contain_region"]
        bowl = ["on", "akita_black_bowl_1", "plate_1"]
        self.assertEqual(pe.FINAL_ORACLE_GOALS["soup_fresh"], [soup])
        self.assertEqual(pe.FINAL_ORACLE_GOALS["sauce_fresh"], [sauce])
        self.assertEqual(pe.FINAL_ORACLE_GOALS["bowl_control"], [bowl])
        for condition in ("basket_native", "basket_split", "basket_forced_handoff"):
            self.assertEqual(pe.FINAL_ORACLE_GOALS[condition], [soup, sauce], condition)
        # The oracle goals are independent objects, not aliases of each other.
        self.assertIsNot(pe.FINAL_ORACLE_GOALS["basket_native"], pe.FINAL_ORACLE_GOALS["basket_split"])

    def test_literal_goals_survive_a_capability_substitution(self):
        # Whatever a condition's capability schedule is, the literal oracle goals
        # stay the fixed physical goals for that condition.
        self.assertEqual(
            pe.FINAL_ORACLE_GOALS["soup_fresh"],
            [["in", "alphabet_soup_1", "basket_1_contain_region"]],
        )
        self.assertNotIn("sauce_to_basket", " ".join(pe.FINAL_ORACLE_GOALS["soup_fresh"][0]))


class _MockConfig:
    """A mutable stand-in for a lerobot policy config object."""


class DiagnosticServiceTests(unittest.TestCase):
    def test_subclasses_actual_scene_service(self):
        self.assertTrue(issubclass(pe.DiagnosticService, service.SceneService))
        self.assertIsNot(pe.DiagnosticService._run_capability, service.SceneService._run_capability)
        self.assertIsNot(pe.DiagnosticService._select_action, service.SceneService._select_action)

    def test_profile_applied_to_both_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = pe.DiagnosticService(run_root=tmp)
            policy = types.SimpleNamespace(config=_MockConfig(), model=types.SimpleNamespace(config=_MockConfig()))
            svc._v1._policy = policy
            result = svc._do_configure_profile("fp32_d1_h5")
            self.assertTrue(result["ok"])
            self.assertEqual(policy.config.use_amp, False)
            self.assertEqual(policy.config.num_steps, 1)
            self.assertEqual(policy.config.n_action_steps, 5)
            self.assertEqual(policy.model.config.use_amp, False)
            self.assertEqual(policy.model.config.num_steps, 1)
            self.assertEqual(policy.model.config.n_action_steps, 5)
            self.assertEqual(svc._diag_profile_name, "fp32_d1_h5")

    def test_unknown_profile_is_rejected_without_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = pe.DiagnosticService(run_root=tmp)
            self.assertEqual(svc.configure_profile("nope")["reason"], "invalid_body")

    def test_select_action_test_seam_returns_7d_float64(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = pe.DiagnosticService(run_root=tmp)
            svc._action_function = lambda batch: np.arange(7)
            action = svc._select_action({})
            self.assertEqual(action.shape, (7,))
            self.assertEqual(action.dtype, np.float64)
            self.assertIsInstance(svc._last_action_latency_s, float)

    def test_missing_model_config_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = pe.DiagnosticService(run_root=tmp)
            policy = types.SimpleNamespace(config=_MockConfig())  # no .model
            svc._v1._policy = policy
            result = svc._do_configure_profile("fp32")
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "invalid_fixture")
            # The profile is not recorded as active after a failure.
            self.assertEqual(svc._diag_profile_name, "baseline_bf16")

    def test_missing_policy_config_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = pe.DiagnosticService(run_root=tmp)
            svc._v1._policy = None
            result = svc._do_configure_profile("fp32")
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "invalid_fixture")

    def test_readback_mismatch_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = pe.DiagnosticService(run_root=tmp)
            policy = types.SimpleNamespace(
                config=_ReadOnlyConfig(), model=types.SimpleNamespace(config=_ReadOnlyConfig())
            )
            svc._v1._policy = policy
            result = svc._do_configure_profile("fp32_d1")
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "invalid_fixture")


class _ReadOnlyConfig:
    """A stand-in config whose attribute writes raise (cannot read back)."""

    def __setattr__(self, name, value):  # noqa: ARG002 - deliberately reject writes
        raise AttributeError("read-only config attribute %s" % name)


class _FakeDiagnosticService:
    """A GPU/model-free stand-in for ``DiagnosticService`` for campaign tests."""

    TERMINAL_PLAN_STATES = service.TERMINAL_PLAN_STATES
    instances: list = []

    def __init__(self, model_path=None, run_root=None, **kwargs):  # noqa: ARG002
        self.model_path = model_path
        self.run_root = Path(run_root) if run_root is not None else None
        self._ready = True
        self._worker_error = None
        self._active_request_id = None
        self._env = object()
        self._sessions: dict = {}
        self._diag_condition = None
        self._model_revision = pe.SOURCE_REVISION_EXPECTED
        self.completion_mode = kwargs.get("completion_mode", "native")
        self.submitted: list = []
        self._plans: dict = {}
        self._counter = 0
        self.configure_result = {
            "ok": True,
            "profile": None,
            "policy_present": True,
            "applied": {},
        }
        _FakeDiagnosticService.instances.append(self)

    def start(self):
        return None

    def stop(self):
        return None

    def health(self):
        return {
            "ok": True,
            "ready": self._ready,
            "model_revision": self._model_revision,
            "worker_error": self._worker_error,
        }

    def _close_env(self):
        return {"ok": True}

    def _sync_work(self, kind, fn, timeout=None):  # noqa: ARG002
        return fn()

    def create_session(self, scene_id, seed=0, init_state_index=0):  # noqa: ARG002
        self._counter += 1
        session_id = "sess%d" % self._counter
        self._sessions[session_id] = types.SimpleNamespace(
            initial_state_hash="hash%d" % self._counter
        )
        return {
            "ok": True,
            "session_id": session_id,
            "scene_id": scene_id,
            "scene_version": 0,
            "env_instance_id": 1,
            "episode_resets": 1,
            "policy_resets": 0,
        }

    def configure_profile(self, profile):
        result = dict(self.configure_result)
        result["profile"] = profile
        return result

    def session(self, session_id):
        if session_id not in self._sessions:
            return None
        return {
            "ok": True,
            "session_id": session_id,
            "scene_id": "basket_two",
            "scene_version": 0,
            "env_instance_id": 1,
            "episode_resets": 1,
            "policy_resets": 0,
        }

    def submit_plan(self, payload):
        request_id = payload["request_id"]
        self.submitted.append(request_id)
        plan = {
            "ok": True,
            "request_id": request_id,
            "state": "completed",
            "job_ids": [],
            "plan_success": False,
            "completed_capability_ids": list(payload.get("capability_ids") or []),
            "pending_capability_ids": [],
        }
        self._plans[request_id] = plan
        return plan

    def plan(self, request_id):
        return self._plans.get(request_id)

    def cancel(self, request_id):  # noqa: ARG002
        return {"ok": True}

    def job(self, job_id):  # noqa: ARG002
        return None

    def override_basket_instruction(self):
        return {"ok": True, "new": "native"}

    def restore_basket_instruction(self):
        return {"ok": True}

    def final_snapshot(self, goals):  # noqa: ARG002
        return {
            "ok": True,
            "snapshot": {"strict_candidate": True, "predicates": {}, "objects": {}, "phases": {}},
        }


class _PollingStubWithoutTerminalConstant:
    """A plan-polling stub with NO ``TERMINAL_PLAN_STATES`` attribute at all.

    ``_submit_and_wait`` needs only ``session`` / ``submit_plan`` / ``plan`` /
    ``cancel``.  Neither this class nor its instances define
    ``TERMINAL_PLAN_STATES``, so any ``service_.TERMINAL_PLAN_STATES`` lookup
    raises ``AttributeError``; a poll can therefore only reach a terminal state
    by reading the module-level ``service.TERMINAL_PLAN_STATES``.  The ``plan``
    mapping carries the exact fields the production poll reads.

    By default (``auto_complete_after=None``) the plan never reaches a terminal
    state on its own, so only ``cancel`` drives it to ``terminal_state`` (the
    timeout path).  ``auto_complete_after=n`` models a plan that transitions to
    ``terminal_state`` deterministically on its ``n``-th ``plan`` poll, so the
    completion path is exercised without waiting on a wall-clock deadline.
    """

    def __init__(self, non_terminal_state, terminal_state, auto_complete_after=None):
        self.state = non_terminal_state
        self.terminal_state = terminal_state
        self.auto_complete_after = auto_complete_after
        self.poll_count = 0
        self.submitted: list = []
        self.cancelled: list = []

    def session(self, session_id):
        return {"ok": True, "session_id": session_id, "scene_version": 3}

    def submit_plan(self, payload):
        self.submitted.append(payload["request_id"])
        return {"ok": True}

    def plan(self, request_id):
        self.poll_count += 1
        if (
            self.auto_complete_after is not None
            and self.poll_count >= self.auto_complete_after
        ):
            self.state = self.terminal_state
        return {
            "ok": True,
            "request_id": request_id,
            "state": self.state,
            "plan_success": self.state == self.terminal_state,
            "completed_capability_ids": [],
            "pending_capability_ids": [],
            "job_ids": [],
        }

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        self.state = self.terminal_state
        return {"ok": True}


class SubmitAndWaitModuleConstantTests(unittest.TestCase):
    """Regression: plan polling must read the *module* ``service`` constant.

    Both ``_submit_and_wait`` poll loops once read
    ``service_.TERMINAL_PLAN_STATES`` off the service *instance*.  The constant
    lives on the imported ``service`` module -- ``DiagnosticService`` /
    ``SceneService`` do not carry it -- so the instance lookup raised
    ``AttributeError`` at runtime.  The stub below deliberately omits even a
    class attribute, so a regression fails loudly instead of being hidden by a
    mock inventing an instance constant.
    """

    @staticmethod
    def _pick_terminal_state():
        states = service.TERMINAL_PLAN_STATES
        for candidate in ("completed", "cancelled", "failed"):
            if candidate in states:
                return candidate
        return sorted(states)[0]

    def _assert_stub_lacks_terminal_constant(self, stub):
        self.assertFalse(hasattr(stub, "TERMINAL_PLAN_STATES"))
        self.assertNotIn("TERMINAL_PLAN_STATES", vars(stub))
        self.assertNotIn("TERMINAL_PLAN_STATES", vars(type(stub)))

    def test_module_constant_is_absent_from_production_class(self):
        self.assertTrue(hasattr(service, "TERMINAL_PLAN_STATES"))
        self.assertNotIn("TERMINAL_PLAN_STATES", vars(pe.DiagnosticService))
        self.assertFalse(hasattr(pe.DiagnosticService, "TERMINAL_PLAN_STATES"))

    def test_completion_poll_uses_module_constant(self):
        terminal = self._pick_terminal_state()
        # The plan reaches ``terminal`` deterministically on the second poll: the
        # first poll observes the non-terminal state, the second completes.
        stub = _PollingStubWithoutTerminalConstant("running", terminal, auto_complete_after=2)
        self._assert_stub_lacks_terminal_constant(stub)

        # The no-op sleep keeps the single running-then-completed transition
        # deterministic without a real wall-clock wait.  Patched for THIS test
        # only; the timeout/cancel test keeps the real sleep.
        with mock.patch.object(pe.time, "sleep"):
            record = pe._submit_and_wait(
                stub, "sess1", ["soup_to_basket"], 300, False, "req-complete", 5.0, "regression"
            )

        self.assertEqual(stub.submitted, ["req-complete"])
        self.assertTrue(record["submitted"])
        self.assertFalse(record["timed_out"])
        self.assertFalse(record["cancel_nonterminal"])
        self.assertFalse(record["cancelled"])
        self.assertEqual(stub.cancelled, [])
        self.assertEqual(record["plan"]["state"], terminal)
        # Two polls: one observed running, the next completed -> no cancel path.
        self.assertEqual(stub.poll_count, 2)

    def test_timeout_then_cancel_poll_uses_module_constant(self):
        terminal = self._pick_terminal_state()
        stub = _PollingStubWithoutTerminalConstant("running", terminal)
        self._assert_stub_lacks_terminal_constant(stub)
        self.assertNotIn("running", set(service.TERMINAL_PLAN_STATES))

        # timeout=0.0 forces the deadline on the first poll, exercising the
        # cancel-then-await-terminal loop.
        record = pe._submit_and_wait(
            stub, "sess2", ["soup_to_basket"], 300, False, "req-timeout", 0.0, "regression"
        )

        self.assertTrue(record["timed_out"])
        self.assertEqual(stub.cancelled, ["req-timeout"])
        self.assertTrue(record["cancelled"])
        self.assertFalse(record["cancel_nonterminal"])
        self.assertEqual(record["plan"]["state"], terminal)


class RunTrialTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_repeated_profile_gets_unique_request_ids(self):
        fake = _FakeDiagnosticService(run_root=self.run_root)
        first = pe._run_trial(
            fake, "fp32", "soup_fresh", "0:0", 10.0, pe.SOURCE_REVISION_EXPECTED, "sha", self.run_root
        )
        second = pe._run_trial(
            fake, "fp32", "soup_fresh", "0:1", 10.0, pe.SOURCE_REVISION_EXPECTED, "sha", self.run_root
        )
        self.assertEqual(len(fake.submitted), 2)
        self.assertNotEqual(fake.submitted[0], fake.submitted[1])
        self.assertIn("-plan1", fake.submitted[0])
        self.assertNotEqual(first["trial_id"], second["trial_id"])
        self.assertIn("0-0", first["trial_id"])
        self.assertIn("0-1", second["trial_id"])
        self.assertIn("soup_fresh", first["trial_id"])
        self.assertIn("fp32", first["trial_id"])

    def test_configure_failure_submits_zero_plans_and_preserves_identity(self):
        fake = _FakeDiagnosticService(run_root=self.run_root)
        fake.configure_result = {
            "ok": False,
            "reason": "invalid_fixture",
            "detail": "policy.model.config missing",
            "applied": {},
        }
        trial = pe._run_trial(
            fake, "fp32", "soup_fresh", "0:0", 10.0, pe.SOURCE_REVISION_EXPECTED, "sha", self.run_root
        )
        self.assertEqual(fake.submitted, [])  # ZERO plans after a config error
        self.assertFalse(trial["profile_result"]["ok"])
        self.assertEqual(trial["profile_result"]["reason"], "invalid_fixture")
        # Session identity is preserved even though nothing ran.
        self.assertIsNotNone(trial["session_id"])
        self.assertEqual(trial["initial_state_hash"], "hash1")
        self.assertIsNone(trial["task_success"])
        self.assertIsNone(trial["strict_task_success"])
        self.assertIn("task_success", trial["null_metrics"])
        self.assertIn("strict_task_success", trial["null_metrics"])
        self.assertTrue(
            any("profile_config" in entry for entry in (trial.get("errors") or []))
        )
        # The per-trial artifact was persisted.
        self.assertTrue((self.run_root / ("trial_%s.json" % trial["trial_id"])).is_file())

    def test_campaign_report_records_versions_and_hashes(self):
        args = pe._build_parser().parse_args(
            [
                "--profiles", "fp32",
                "--conditions", "soup_fresh",
                "--pairs", "0:0",
                "--output", "/tmp/placement_report.json",
                "--run-root", "/tmp/placement_runs",
            ]
        )
        report = pe._campaign_report(args, [], pe.SOURCE_REVISION_EXPECTED, "sha", time.monotonic(), None)
        metadata = report["metadata"]
        self.assertEqual(
            set(metadata["package_versions"]),
            {"torch", "lerobot", "libero", "robosuite", "mujoco", "transformers"},
        )
        for value in metadata["package_versions"].values():
            self.assertTrue(value is None or isinstance(value, str))
        self.assertEqual(
            set(metadata["source_sha256"]),
            {
                "placement_experiments.py",
                "service.py",
                "catalog.py",
                "placement_completion.py",
            },
        )
        # The three original digests are retained unchanged alongside the
        # newly tracked placement_completion.py module.
        for name in ("placement_experiments.py", "service.py", "catalog.py"):
            self.assertIn(name, metadata["source_sha256"])
        digest = metadata["source_sha256"]["placement_experiments.py"]
        self.assertIsNotNone(digest)
        self.assertEqual(len(digest), 64)
        self.assertEqual(metadata["source_git_sha"], "sha")
        self.assertEqual(report["aggregate"]["n_trials"], 0)


class CampaignPersistenceTests(unittest.TestCase):
    def test_report_is_written_after_every_trial_and_finally(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "screening.json"
            run_root = Path(tmp) / "screening_runs"
            args = pe._build_parser().parse_args(
                [
                    "--profiles", "baseline_bf16", "fp32",
                    "--conditions", "soup_fresh",
                    "--pairs", "0:0",
                    "--output", str(output),
                    "--run-root", str(run_root),
                ]
            )
            writes: list = []
            original = pe._write_json_atomic

            def wrapper(path, payload):
                if isinstance(payload, dict):
                    writes.append((str(path), len(payload.get("trials", []))))
                return original(path, payload)

            with mock.patch.object(pe, "DiagnosticService", _FakeDiagnosticService), mock.patch.object(
                pe, "_git_rev_parse", lambda cwd=None: "deadbeef"
            ), mock.patch.object(pe, "_write_json_atomic", side_effect=wrapper):
                rc = pe._run_campaign(args)

            self.assertEqual(rc, 0)
            report_writes = [count for (path, count) in writes if path == str(output)]
            # One write immediately after each of the two trials, plus the
            # final write in the ``finally`` block.
            self.assertIn(1, report_writes)
            self.assertIn(2, report_writes)
            self.assertEqual(report_writes[-1], 2)
            self.assertGreaterEqual(len(report_writes), 3)

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["aggregate"]["n_trials"], 2)
            self.assertEqual(report["metadata"]["source_git_sha"], "deadbeef")
            self.assertIn("source_sha256", report["metadata"])
            self.assertIn("package_versions", report["metadata"])

    def test_model_revision_mismatch_runs_zero_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "screening.json"
            args = pe._build_parser().parse_args(
                [
                    "--profiles", "fp32",
                    "--conditions", "soup_fresh",
                    "--pairs", "0:0",
                    "--output", str(output),
                    "--run-root", str(Path(tmp) / "runs"),
                ]
            )

            def _mismatched(*positional, **keywords):
                svc = _FakeDiagnosticService(*positional, **keywords)
                svc._model_revision = "0" * 40
                return svc

            with mock.patch.object(pe, "DiagnosticService", _mismatched), mock.patch.object(
                pe, "_git_rev_parse", lambda cwd=None: "sha"
            ):
                rc = pe._run_campaign(args)

            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["aggregate"]["n_trials"], 0)
            self.assertIsNotNone(report["metadata"]["fatal_error"])
            self.assertEqual(_FakeDiagnosticService.instances[-1].submitted, [])

    def test_trial_exception_stops_campaign_after_preserving_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "screening.json"
            args = pe._build_parser().parse_args(
                [
                    "--profiles", "baseline_bf16", "fp32", "fp32_d1",
                    "--conditions", "soup_fresh",
                    "--pairs", "0:0",
                    "--output", str(output),
                    "--run-root", str(Path(tmp) / "runs"),
                ]
            )
            real_run_trial = pe._run_trial
            calls = {"n": 0}

            def _flaky(*positional, **keywords):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("boom")
                return real_run_trial(*positional, **keywords)

            with mock.patch.object(pe, "DiagnosticService", _FakeDiagnosticService), mock.patch.object(
                pe, "_git_rev_parse", lambda cwd=None: "sha"
            ), mock.patch.object(pe, "_run_trial", side_effect=_flaky):
                rc = pe._run_campaign(args)

            self.assertEqual(rc, 1)  # an unexpected trial exception is fatal
            self.assertEqual(calls["n"], 2)  # the third trial never ran
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["aggregate"]["n_trials"], 2)  # trial 1 + error record
            self.assertTrue(
                any("trial_exception" in entry for entry in (report["trials"][1].get("errors") or []))
            )


class CliContractTests(unittest.TestCase):
    def test_parser_accepts_the_screening_arguments_without_cuda(self):
        parser = pe._build_parser()
        args = parser.parse_args(
            [
                "--profiles",
                "baseline_bf16",
                "fp32",
                "fp32_d1",
                "fp32_h5",
                "fp32_d1_h5",
                "--conditions",
                "soup_fresh",
                "--pairs",
                "0:0",
                "--output",
                "/home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/screening.json",
                "--run-root",
                "/home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/screening_runs",
            ]
        )
        self.assertEqual(len(args.profiles), 5)
        self.assertEqual(args.conditions, ["soup_fresh"])
        self.assertEqual(args.pairs, ["0:0"])
        self.assertEqual(args.timeout, 900.0)

    def test_parser_rejects_unknown_profile(self):
        parser = pe._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--profiles", "not_a_profile", "--output", "/x.json", "--run-root", "/r"])


class ContinuationConditionContractTests(unittest.TestCase):
    """The two fixed continuation conditions and their literal oracle goals."""

    def test_soup_extended_contract(self):
        spec = pe.CONDITIONS["soup_extended"]
        self.assertEqual(spec["scene_id"], "basket_two")
        self.assertEqual(tuple(spec["capability_ids"]), ("soup_to_basket",))
        self.assertEqual(spec["budget_per_subgoal"], 600)
        self.assertEqual(spec["mode"], "single")
        self.assertFalse(spec["native_instruction"])
        self.assertFalse(spec["audit"])

    def test_soup_retry_contract(self):
        spec = pe.CONDITIONS["soup_retry"]
        self.assertEqual(spec["scene_id"], "basket_two")
        self.assertEqual(tuple(spec["capability_ids"]), ("soup_to_basket",))
        self.assertEqual(spec["budget_per_subgoal"], 300)
        self.assertEqual(spec["mode"], "same_goal_retry")
        self.assertFalse(spec["native_instruction"])
        self.assertFalse(spec["audit"])
        # No foreign-object schedule is declared.
        self.assertNotIn("forced_handoff_capability_ids", spec)

    def test_continuation_oracle_goals_are_literal(self):
        literal = [["in", "alphabet_soup_1", "basket_1_contain_region"]]
        self.assertEqual(pe.FINAL_ORACLE_GOALS["soup_extended"], literal)
        self.assertEqual(pe.FINAL_ORACLE_GOALS["soup_retry"], literal)
        # Independent list objects, never aliases of each other.
        self.assertIsNot(pe.FINAL_ORACLE_GOALS["soup_extended"], pe.FINAL_ORACLE_GOALS["soup_retry"])


class _ConditionStub:
    """A model-free service stub driving ``_run_condition``'s retry logic."""

    def __init__(self, first_state="blocked", submit_error=False, timeout=False, job_ids=()):
        self.first_state = first_state
        self.submit_error = submit_error
        self.timeout = timeout
        self.job_ids = list(job_ids)
        self.submitted: list = []
        self.payloads: list = []
        self.cancelled: list = []
        self._cancelled: set = set()

    def session(self, session_id):
        return {"ok": True, "session_id": session_id, "scene_version": 0}

    def submit_plan(self, payload):
        request_id = payload["request_id"]
        self.submitted.append(request_id)
        self.payloads.append(dict(payload))
        if self.submit_error and request_id.endswith("plan1"):
            return {"ok": False, "reason": "busy", "detail": "mock submit failure"}
        return {"ok": True}

    def plan(self, request_id):
        if request_id in self._cancelled:
            state = "cancelled"
        elif request_id.endswith("plan1"):
            state = "running" if self.timeout else self.first_state
        else:
            state = "completed"
        return {
            "ok": True,
            "request_id": request_id,
            "state": state,
            "plan_success": state == "completed",
            "completed_capability_ids": [],
            "pending_capability_ids": [],
            "job_ids": list(self.job_ids),
        }

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        self._cancelled.add(request_id)
        return {"ok": True}

    def override_basket_instruction(self):
        return {"ok": True, "new": "native"}

    def restore_basket_instruction(self):
        return {"ok": True}


class SameGoalRetryTests(unittest.TestCase):
    def _run(self, stub, timeout=10.0):
        return pe._run_condition(stub, "soup_retry", "sess-1", timeout, "trialkey")

    def test_blocked_first_plan_submits_same_goal_second_plan(self):
        stub = _ConditionStub(first_state="blocked")
        result = self._run(stub)
        self.assertTrue(result["retry_same_goal"])
        self.assertEqual(
            [plan["request_id"] for plan in result["plans"]],
            ["trialkey-plan1", "trialkey-plan2"],
        )
        self.assertEqual(stub.submitted, ["trialkey-plan1", "trialkey-plan2"])
        self.assertTrue(result["retry"]["retried"])
        self.assertTrue(result["retry"]["same_goal"])
        # Same session, same soup goal, same 300 budget for both plans.
        self.assertEqual(len(stub.payloads), 2)
        for payload in stub.payloads:
            self.assertEqual(payload["session_id"], "sess-1")
            self.assertEqual(list(payload["capability_ids"]), ["soup_to_basket"])
            self.assertEqual(payload["budget_per_subgoal"], 300)
        self.assertIsNone(result["forced_handoff"])
        self.assertFalse(result["campaign_stopped"])

    def test_blocked_first_plan_with_state_error_job_stops_without_retry(self):
        # A blocked plan whose owned job reports state=="error" is operational:
        # no same-goal second plan is submitted.
        stub = _ConditionStub(first_state="blocked", job_ids=["job-err"])
        evidence = {
            "job_id": "job-err",
            "available": True,
            "job": {"state": "error", "ended_reason": None, "error": None},
        }
        with mock.patch.object(pe, "_job_evidence", return_value=evidence):
            result = self._run(stub)
        self.assertEqual(stub.submitted, ["trialkey-plan1"])
        self.assertEqual(len(result["plans"]), 1)
        self.assertFalse(result["retry"]["retried"])
        self.assertEqual(result["retry"]["reason"], "job_error")
        self.assertTrue(result["campaign_stopped"])

    def test_blocked_first_plan_with_ended_reason_error_job_stops_without_retry(self):
        stub = _ConditionStub(first_state="blocked", job_ids=["job-err"])
        evidence = {
            "job_id": "job-err",
            "available": True,
            "job": {"state": "completed", "ended_reason": "error", "error": None},
        }
        with mock.patch.object(pe, "_job_evidence", return_value=evidence):
            result = self._run(stub)
        self.assertEqual(stub.submitted, ["trialkey-plan1"])
        self.assertEqual(len(result["plans"]), 1)
        self.assertFalse(result["retry"]["retried"])
        self.assertEqual(result["retry"]["reason"], "job_error")
        self.assertTrue(result["campaign_stopped"])

    def test_blocked_first_plan_with_truthy_job_error_stops_without_retry(self):
        stub = _ConditionStub(first_state="blocked", job_ids=["job-err"])
        evidence = {
            "job_id": "job-err",
            "available": True,
            "job": {
                "state": "completed",
                "ended_reason": "budget_exhausted",
                "error": "RuntimeError: boom",
            },
        }
        with mock.patch.object(pe, "_job_evidence", return_value=evidence):
            result = self._run(stub)
        self.assertEqual(stub.submitted, ["trialkey-plan1"])
        self.assertEqual(len(result["plans"]), 1)
        self.assertFalse(result["retry"]["retried"])
        self.assertEqual(result["retry"]["reason"], "job_error")
        self.assertTrue(result["campaign_stopped"])

    def test_completed_first_plan_skips_retry(self):
        stub = _ConditionStub(first_state="completed")
        result = self._run(stub)
        self.assertTrue(result["retry_same_goal"])
        self.assertFalse(result["retry"]["retried"])
        self.assertEqual(stub.submitted, ["trialkey-plan1"])
        self.assertEqual(len(result["plans"]), 1)

    def test_operational_timeout_stops_without_retry(self):
        stub = _ConditionStub(first_state="blocked", timeout=True)
        result = self._run(stub, timeout=0.0)
        self.assertFalse(result["retry"]["retried"])
        self.assertEqual(result["retry"]["reason"], "operational_timeout")
        self.assertTrue(result["campaign_stopped"])
        self.assertEqual(stub.submitted, ["trialkey-plan1"])
        self.assertEqual(stub.cancelled, ["trialkey-plan1"])

    def test_error_and_cancelled_first_plan_stop_without_retry(self):
        for state in ("error", "cancelled"):
            stub = _ConditionStub(first_state=state)
            result = self._run(stub)
            self.assertFalse(result["retry"]["retried"], state)
            self.assertEqual(result["retry"]["reason"], state)
            self.assertTrue(result["campaign_stopped"])
            self.assertEqual(stub.submitted, ["trialkey-plan1"])

    def test_submit_error_stops_without_retry(self):
        stub = _ConditionStub(submit_error=True)
        result = self._run(stub)
        self.assertFalse(result["retry"]["retried"])
        self.assertEqual(result["retry"]["reason"], "submit_error")
        self.assertTrue(result["campaign_stopped"])
        self.assertEqual(stub.submitted, ["trialkey-plan1"])

    def test_retry_uses_same_session_and_never_reseeds_rng(self):
        stub = _ConditionStub(first_state="blocked")
        with mock.patch.object(service, "_seed_everything") as seeded:
            result = pe._run_condition(stub, "soup_retry", "sess-1", 10.0, "trialkey")
        seeded.assert_not_called()
        self.assertTrue(result["retry"]["retried"])
        self.assertEqual({p["session_id"] for p in stub.payloads}, {"sess-1"})

    def test_single_mode_never_retries(self):
        stub = _ConditionStub(first_state="blocked")
        result = pe._run_condition(stub, "soup_extended", "sess-1", 10.0, "trialkey")
        self.assertFalse(result["retry_same_goal"])
        self.assertEqual(stub.submitted, ["trialkey-plan1"])


class OperationalJobErrorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _session(self):
        return {
            "session_id": "sess-1",
            "scene_id": "basket_two",
            "env_instance_id": 1,
            "episode_resets": 1,
            "policy_resets": 1,
        }

    def _condition_result(self, job_ids):
        return {
            "condition": "soup_retry",
            "mode": "same_goal_retry",
            "plans": [
                {
                    "request_id": "trialkey-plan1",
                    "submitted": True,
                    "submit_error": None,
                    "timed_out": False,
                    "cancel_nonterminal": False,
                    "cancelled": False,
                    "wall_s": 1.0,
                    "plan": {
                        "state": "completed",
                        "plan_success": False,
                        "completed_capability_ids": ["soup_to_basket"],
                        "pending_capability_ids": [],
                        "job_ids": list(job_ids),
                    },
                }
            ],
            "native_instruction": None,
            "instruction_override_error": None,
            "campaign_stopped": False,
            "stop_reason": None,
            "forced_handoff": None,
            "retry_same_goal": True,
            "retry": {"retried": False, "reason": "completed"},
        }

    def _stub(self, job_public):
        class _Stub:
            completion_mode = "native"
            _sessions: dict = {}

            def job(self, job_id):
                return dict(job_public, job_id=job_id)

        return _Stub()

    def _build(self, job_public):
        return pe._build_trial(
            self._stub(job_public),
            "fp32",
            "soup_retry",
            "0:0",
            0,
            0,
            self._session(),
            self._condition_result(["job-1"]),
            {"strict_candidate": False, "predicates": {}},
            None,
            None,
            1.0,
            "trialkey",
        )

    def test_errored_job_is_operational_not_physical(self):
        trial = self._build(
            {
                "state": "error",
                "ended_reason": "error",
                "error": "RuntimeError: boom",
                "capability_id": "soup_to_basket",
                "run_dir": str(self.run_root),
            }
        )
        self.assertTrue(trial["job_operational_errors"])
        self.assertTrue(any("job_error" in e for e in trial["operational_errors"]))
        self.assertTrue(any("RuntimeError: boom" in e for e in trial["operational_errors"]))
        self.assertFalse(trial["budget_exhausted"])
        self.assertEqual(trial["physical_failures"], [])

    def test_ended_reason_error_is_operational(self):
        trial = self._build(
            {
                "state": "completed",
                "ended_reason": "error",
                "error": "scene died",
                "capability_id": "soup_to_basket",
                "run_dir": str(self.run_root),
            }
        )
        self.assertTrue(trial["job_operational_errors"])
        self.assertFalse(trial["budget_exhausted"])

    def test_budget_exhausted_job_stays_a_physical_failure_with_no_errors(self):
        trial = self._build(
            {
                "state": "completed",
                "ended_reason": "budget_exhausted",
                "error": None,
                "capability_id": "soup_to_basket",
                "run_dir": str(self.run_root),
            }
        )
        self.assertTrue(trial["budget_exhausted"])
        self.assertEqual(trial["job_operational_errors"], [])
        self.assertEqual(trial["operational_errors"], [])

    def test_budget_exhausted_with_truthy_error_is_operational_only(self):
        # A completed job carrying a truthy ``job.error`` AND an
        # ``ended_reason=="budget_exhausted"`` is an operational failure: it must
        # never be counted simultaneously as a physical budget exhaustion.
        trial = self._build(
            {
                "state": "completed",
                "ended_reason": "budget_exhausted",
                "error": "RuntimeError: boom",
                "capability_id": "soup_to_basket",
                "run_dir": str(self.run_root),
            }
        )
        self.assertTrue(trial["job_operational_errors"])
        self.assertIn("RuntimeError: boom", " ".join(trial["job_operational_errors"]))
        self.assertFalse(trial["budget_exhausted"])
        self.assertEqual(trial["physical_failures"], [])

    def test_campaign_stops_on_job_operational_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "screening.json"
            args = pe._build_parser().parse_args(
                [
                    "--profiles", "baseline_bf16", "fp32",
                    "--conditions", "soup_retry",
                    "--pairs", "0:0",
                    "--output", str(output),
                    "--run-root", str(Path(tmp) / "runs"),
                ]
            )
            calls = {"n": 0}

            def _trial(*positional, **keywords):  # noqa: ARG001
                calls["n"] += 1
                return {
                    "trial_id": "t%d" % calls["n"],
                    "profile": "x",
                    "condition": "soup_retry",
                    "pair": "0:0",
                    "errors": ["job_error: boom"],
                    "operational_errors": ["job_error: boom"],
                    "job_operational_errors": ["job_error: boom"],
                    "null_metrics": [],
                }

            with mock.patch.object(pe, "DiagnosticService", _FakeDiagnosticService), mock.patch.object(
                pe, "_git_rev_parse", lambda cwd=None: "sha"
            ), mock.patch.object(pe, "_run_trial", side_effect=_trial):
                rc = pe._run_campaign(args)

            self.assertEqual(rc, 1)  # an operational job error is fatal
            self.assertEqual(calls["n"], 1)  # stopped after the first trial
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNotNone(report["metadata"]["fatal_error"])
            self.assertEqual(report["aggregate"]["n_trials"], 1)


class CompletionModeMetadataTests(unittest.TestCase):
    def test_trial_records_completion_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            fake = _FakeDiagnosticService(run_root=run_root, completion_mode="release_verified")
            trial = pe._run_trial(
                fake,
                "fp32",
                "soup_retry",
                "0:0",
                10.0,
                pe.SOURCE_REVISION_EXPECTED,
                "sha",
                run_root,
            )
            self.assertEqual(trial["completion_mode"], "release_verified")
            self.assertTrue(trial["retry_same_goal"])

    def test_campaign_metadata_records_completion_mode_and_source_hash(self):
        args = pe._build_parser().parse_args(
            [
                "--profiles", "fp32",
                "--conditions", "soup_extended",
                "--pairs", "0:0",
                "--output", "/tmp/placement_report.json",
                "--run-root", "/tmp/placement_runs",
                "--completion-mode", "release_verified",
            ]
        )
        report = pe._campaign_report(
            args, [], pe.SOURCE_REVISION_EXPECTED, "sha", time.monotonic(), None
        )
        self.assertEqual(report["metadata"]["completion_mode"], "release_verified")
        self.assertEqual(
            set(report["metadata"]["source_sha256"]),
            {
                "placement_experiments.py",
                "service.py",
                "catalog.py",
                "placement_completion.py",
            },
        )

    def test_completion_mode_cli_defaults_to_native(self):
        parser = pe._build_parser()
        args = parser.parse_args(["--output", "/x.json", "--run-root", "/r"])
        self.assertEqual(args.completion_mode, "native")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--completion-mode", "bogus", "--output", "/x.json", "--run-root", "/r"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
