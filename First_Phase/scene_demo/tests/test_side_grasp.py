#!/usr/bin/env python3
"""CPU-only tests for the demonstrated-side grasp seams."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import local_grasp  # noqa: E402
import side_grasp  # noqa: E402

WINE = local_grasp.WINE_OBJECT_ID
COMMAND = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _reading(position=(0.0, 0.0, 0.10), wine=(0.0, 0.0, 0.10), wine_rotation=None, b2c=None):
    if wine_rotation is None:
        wine_rotation = np.eye(3)
    if b2c is None:
        b2c = np.eye(3)
    snapshot = {
        "objects": {WINE: {"position": list(wine), "grasped": False}},
        "held_objects": [],
        "grasp_observation_complete": True,
        "predicates": {},
    }
    return {
        "pose": {"position": list(position), "orientation_matrix": np.eye(3)},
        "snapshot": snapshot,
        "wine_rotation": np.asarray(wine_rotation, dtype=np.float64),
        "body_to_controller_rotation": np.asarray(b2c, dtype=np.float64),
        "pad_offset_local": [0.0, 0.0, 0.0],
    }


class ReferenceTests(unittest.TestCase):
    def test_actual_reference_derives_relative_pose(self):
        ref = side_grasp.load_reference()
        self.assertEqual(ref["step"], 93)
        self.assertEqual(ref["source_sha256"], "3b546bcf49a3b01f31799212ff509a30c8a73c6836373680d07584f2c4800c10")
        self.assertTrue(np.allclose(ref["relative_position"], side_grasp.P_RELATIVE, atol=1e-8))
        self.assertTrue(np.allclose(ref["relative_orientation"], side_grasp.R_RELATIVE, atol=1e-8))

    def test_reference_independent_recompute(self):
        import gzip, json, hashlib
        from scipy.spatial.transform import Rotation
        raw = side_grasp._REFERENCE_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), side_grasp._REFERENCE_SHA256)
        rec = None
        with gzip.open(side_grasp._REFERENCE_PATH, "rt") as fh:
            for line in fh:
                e = json.loads(line)
                if e.get("step") == 93:
                    rec = e
                    break
        self.assertIsNotNone(rec)
        before = rec["before_snapshot"]["objects"][WINE]
        self.assertIs(before["grasped"], True)
        s = np.array(rec["policy_input_state"], dtype=float)
        Rw = local_grasp._quat_wxyz_to_matrix(before["quaternion"])
        Rbody = Rotation.from_rotvec(s[3:6]).as_matrix()
        p_rel = Rw.T @ (s[:3] - np.array(before["position"], dtype=float))
        self.assertTrue(np.allclose(p_rel, side_grasp.P_RELATIVE, atol=1e-8, rtol=0))
        self.assertTrue(np.allclose(Rw.T @ Rbody, side_grasp.R_RELATIVE, atol=1e-8, rtol=0))


class EquivarianceTests(unittest.TestCase):
    def test_translation_equivariance(self):
        r0 = _reading(wine=(0.0, 0.0, 0.10))
        p0, _ = side_grasp.make_target(r0)
        shift = np.array([1.0, -2.0, 3.0])
        r1 = _reading(wine=tuple(np.array([0.0, 0.0, 0.10]) + shift))
        p1, _ = side_grasp.make_target(r1)
        self.assertTrue(np.allclose(p1 - p0, shift, atol=1e-9))

    def test_rotation_equivariance(self):
        yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        r0 = _reading(wine_rotation=np.eye(3))
        p0, R0 = side_grasp.make_target(r0)
        r1 = _reading(wine_rotation=yaw)
        p1, R1 = side_grasp.make_target(r1)
        self.assertTrue(np.allclose(p1, yaw @ p0, atol=1e-9))
        self.assertTrue(np.allclose(R1, yaw @ R0, atol=1e-9))

    def test_body_to_controller_compensation(self):
        z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        r0 = _reading(b2c=np.eye(3))
        _, R0 = side_grasp.make_target(r0)
        r1 = _reading(b2c=z90)
        _, R1 = side_grasp.make_target(r1)
        self.assertTrue(np.allclose(R1, R0 @ z90, atol=1e-9))


class OffsetTests(unittest.TestCase):
    def test_approach_offset_is_along_hand_z(self):
        R = np.eye(3)
        off = side_grasp.make_approach_offset({}, np.zeros(3), R)
        self.assertTrue(np.allclose(off, [0.0, 0.0, -local_grasp.ABOVE_CLEARANCE_M], atol=1e-9))
        self.assertFalse(np.allclose(off, [0.0, 0.0, local_grasp.ABOVE_CLEARANCE_M], atol=1e-9))

    def test_approach_offset_non_world_z(self):
        x90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        off = side_grasp.make_approach_offset({}, np.zeros(3), x90)
        self.assertTrue(np.allclose(off, [0.0, 0.06, 0.0], atol=1e-9))
        self.assertFalse(np.allclose(off, [0.0, 0.0, -local_grasp.ABOVE_CLEARANCE_M], atol=1e-9))


class FailureTests(unittest.TestCase):
    def test_missing_or_nonfinite_geometry_raises(self):
        for key in ("wine_rotation", "body_to_controller_rotation"):
            for bad in (None, np.full((3, 3), np.nan)):
                reading = _reading()
                reading[key] = bad
                with self.assertRaises(ValueError):
                    side_grasp.make_target(reading)

    def test_read_geometry_missing_body_raises(self):
        with mock.patch.object(side_grasp.base, "read_geometry", return_value={}):
            with mock.patch.object(side_grasp.service, "_inner_env", return_value=SimpleNamespace(robots=[SimpleNamespace(robot_model=SimpleNamespace())])):
                with self.assertRaisesRegex(ValueError, "missing body"):
                    side_grasp.read_geometry(object())

    def test_injected_builders_counter_preservation(self):
        calls = {"target": 0, "offset": 0}

        def target(reading):
            calls["target"] += 1
            return side_grasp.make_target(reading)

        def offset(reading, p, R):
            calls["offset"] += 1
            return side_grasp.make_approach_offset(reading, p, R)

        c = local_grasp.LocalGraspController(target_builder=target, approach_offset_builder=offset)
        reading = _reading()
        self.assertIsNotNone(c.next_action(reading, COMMAND))
        c.observe_after(reading)
        self.assertEqual(calls["target"], 1)
        self.assertEqual(calls["offset"], 1)
        self.assertEqual(c.total_actions, 1)

    def test_invalid_offset_never_arms(self):
        def bad_offset(reading, p, R):
            return [float("nan"), 0.0, 0.0]
        c = local_grasp.LocalGraspController(approach_offset_builder=bad_offset)
        self.assertIsNone(c.next_action(_reading(), COMMAND))
        self.assertEqual(c.phase, "idle")
        self.assertEqual(c.total_actions, 0)


class BaseSeamTests(unittest.TestCase):
    def test_base_default_target_and_counters(self):
        reading = _reading()
        c = local_grasp.LocalGraspController()
        action = c.next_action(reading, COMMAND)
        self.assertIsNotNone(action)
        expected = local_grasp.make_target(reading)[0] + [0.0, 0.0, 0.06]
        self.assertTrue(np.allclose(c._above_target_position, expected, atol=1e-9))
        self.assertEqual(c.total_actions, 0)
        c.observe_after(reading)
        self.assertEqual(c.total_actions, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
