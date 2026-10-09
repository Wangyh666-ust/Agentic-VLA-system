"""Contract tests for the three-segment collision grasp controller.

CPU only: no physical env, VLA, or Hermes calls. All controller methods
are exercised for real; only side.load_reference/reference_metadata and
collision_geometry's service._inner_env are mocked.
"""

import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

_SCENE_DEMO = Path(__file__).resolve().parents[1]
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import collision_grasp as cg  # noqa: E402
import collision_geometry  # noqa: E402
import local_grasp as base  # noqa: E402
import side_grasp as side  # noqa: E402

_REF_META = {"name": "synthetic_side_reference", "version": 1}


def _eye3():
    return np.eye(3, dtype=np.float64).tolist()


def _make_reading():
    return {
        "pose": {
            "position": [0.06, 0.0, 1.0],
            "orientation_matrix": _eye3(),
        },
        "wine_rotation": _eye3(),
        "body_to_controller_rotation": _eye3(),
        "snapshot": {
            "objects": {
                base.WINE_OBJECT_ID: {
                    "position": [0.0, 0.0, 0.9],
                    "grasped": False,
                },
                "akita_black_bowl_1": {"position": [0.4, 0.4, 0.9], "grasped": False},
                "plate_1": {"position": [0.4, 0.4, 0.89], "grasped": False},
                "cream_cheese_1": {"position": [-0.4, 0.4, 0.9], "grasped": False},
            },
            "held_objects": [],
            "grasp_observation_complete": True,
            "predicates": {"on|akita_black_bowl_1|plate_1": True},
        },
        "collision_geometry": {
            "bottle_top_z_m": 1.06,
            "hand_sweep_radius_m": 0.20,
            "wine_robot_contacts": [],
        },
    }


def _proposed():
    action = np.zeros(base.ACTION_DIM, dtype=np.float64)
    action[-1] = base.GRASP_CLOSED
    return action


def _controller():
    with mock.patch.object(side, "load_reference", lambda: None), mock.patch.object(
        side, "reference_metadata", lambda: dict(_REF_META)
    ):
        return cg.CollisionGraspController()


def _armed_controller(reading):
    controller = _controller()
    action = controller.next_action(reading, _proposed())
    assert action is not None
    return controller


def _advance(controller, reading):
    """Real next_action -> real post-sample aligned to CURRENT route target."""
    summary = controller.summary()
    targets = summary["route_targets"]
    index = summary["route_index"]
    target = targets[index]
    pre = controller.next_action(reading, _proposed())
    assert pre is not None
    post = copy.deepcopy(reading)
    post["pose"]["position"] = list(target["position"])
    post["pose"]["orientation_matrix"] = [list(r) for r in target["orientation"]]
    controller.observe_after(post)
    reading.clear()
    reading.update(post)
    return pre


def _stage_three(controller, reading):
    for _ in range(3):
        _advance(controller, reading)


class TestFirstAction(unittest.TestCase):
    def test_first_action_shape_and_direction(self):
        reading = _make_reading()
        controller = _armed_controller(reading)
        action = controller.next_action(reading, _proposed())
        self.assertEqual(action.shape, (base.ACTION_DIM,))
        np.testing.assert_allclose(action[3:6], [0, 0, 0])
        self.assertGreater(action[0], 0.0)  # escape radial +X, away from wine
        self.assertEqual(action[1], 0.0)
        self.assertGreater(action[2], 0.0)  # Z rises
        self.assertEqual(action[-1], base.GRASP_OPEN)


class TestMissingDescriptor(unittest.TestCase):
    def test_missing_descriptor_fails(self):
        reading = _make_reading()
        reading.pop("collision_geometry")
        controller = _controller()
        action = controller.next_action(reading, _proposed())
        self.assertIsNone(action)
        self.assertEqual(controller.phase, base.FAILED)
        self.assertEqual(controller.total_actions, 0)
        self.assertEqual(controller.summary()["route_counts"], [0, 0, 0])


class TestRepeatedNextActionFrozen(unittest.TestCase):
    def test_repeated_calls_do_not_advance(self):
        reading = _make_reading()
        controller = _armed_controller(reading)
        targets_before = controller.summary()["route_targets"]
        actions = [controller.next_action(reading, _proposed()) for _ in range(20)]
        self.assertTrue(all(a is not None for a in actions))
        self.assertEqual(controller.total_actions, 0)
        self.assertEqual(sum(controller._stage_streak.values()), 0)
        self.assertEqual(controller.summary()["route_targets"], targets_before)
        self.assertEqual(controller.summary()["route_counts"], [0, 0, 0])


class TestThreeStageRoute(unittest.TestCase):
    def test_three_aligned_samples_per_stage(self):
        reading = _make_reading()
        controller = _armed_controller(reading)
        self.assertEqual(controller.phase, cg.ABOVE)
        _stage_three(controller, reading)
        self.assertEqual(controller.phase, cg.ABOVE)
        self.assertEqual(controller.summary()["route_index"], 1)
        _stage_three(controller, reading)
        self.assertEqual(controller.phase, cg.ABOVE)
        self.assertEqual(controller.summary()["route_index"], 2)
        _stage_three(controller, reading)
        self.assertEqual(controller.phase, cg.DESCEND)
        self.assertEqual(controller.total_actions, 9)
        self.assertEqual(controller.summary()["route_counts"], [3, 3, 3])


class TestContactDuringAlign(unittest.TestCase):
    def test_injected_contact_fails_and_counts(self):
        reading = _make_reading()
        controller = _armed_controller(reading)
        _stage_three(controller, reading)
        self.assertEqual(controller.summary()["route_index"], 1)
        controller.next_action(reading, _proposed())
        post = copy.deepcopy(reading)
        post["collision_geometry"]["wine_robot_contacts"] = [
            {"pair": ["wine_bottle_1", "gripper0"], "dist": 0.0, "pos": [0.0, 0.0, 0.9]}
        ]
        controller.observe_after(post)
        self.assertEqual(controller.phase, base.FAILED)
        self.assertEqual(controller.total_actions, 4)


class TestRotationClearance(unittest.TestCase):
    def test_insufficient_clearance_fails(self):
        reading = _make_reading()
        controller = _armed_controller(reading)
        _stage_three(controller, reading)
        self.assertEqual(controller.summary()["route_index"], 1)
        low = copy.deepcopy(reading)
        low["pose"]["position"] = [0.06, 0.0, 1.1]
        action = controller.next_action(low, _proposed())
        self.assertIsNone(action)
        self.assertEqual(controller.phase, base.FAILED)
        self.assertEqual(controller.reason, "rotation_clearance_lost")


class TestWineRotationDrift(unittest.TestCase):
    def test_20_1_degrees_is_drift(self):
        angle = math.radians(20.1)
        c, s = math.cos(angle), math.sin(angle)
        reading = _make_reading()
        controller = _armed_controller(reading)
        reading["wine_rotation"] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        controller.next_action(reading, _proposed())
        post = copy.deepcopy(reading)
        post["wine_rotation"] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        controller.observe_after(post)
        self.assertEqual(controller.phase, base.FAILED)
        self.assertEqual(controller.reason, "wine_rotation_drift")


class TestTotalBudget(unittest.TestCase):
    def test_total_200_refuses(self):
        reading = _make_reading()
        controller = _armed_controller(reading)
        controller.total_actions = 200
        action = controller.next_action(reading, _proposed())
        self.assertIsNone(action)
        self.assertEqual(controller.phase, base.FAILED)
        self.assertEqual(controller.reason, "total_budget_exceeded")


class TestFinalTargets(unittest.TestCase):
    def test_targets_match_side_make_target(self):
        reading = _make_reading()
        with mock.patch.object(side, "load_reference", lambda: None), mock.patch.object(
            side, "reference_metadata", lambda: dict(_REF_META)
        ):
            controller = cg.CollisionGraspController()
        with mock.patch.object(side, "load_reference", lambda: None), mock.patch.object(
            side, "reference_metadata", lambda: dict(_REF_META)
        ):
            expected_position, expected_orientation = side.make_target(reading)
        controller.next_action(reading, _proposed())
        np.testing.assert_allclose(controller._target_position, expected_position)
        np.testing.assert_allclose(controller._target_orientation, expected_orientation)


class TestGeometryDescribe(unittest.TestCase):
    def _fake_env(self):
        def geom(name, contype=1, conaffinity=1, rbound=0.05, pos=(0.0, 0.0, 0.0)):
            return SimpleNamespace(name=name, contype=contype, conaffinity=conaffinity, rbound=rbound, pos=pos)

        geoms = [
            geom("gripper0_left", pos=(0.0, 0.0, 0.0)),
            geom("gripper0_right", pos=(0.0, 0.02, 0.0)),
            geom("wine_bottle_1_body", pos=(0.0, 0.0, 0.85)),
            geom("table", contype=0, conaffinity=0),
        ]
        names = {i: g.name for i, g in enumerate(geoms)}
        model = SimpleNamespace(
            ngeom=len(geoms),
            geom_contype=[g.contype for g in geoms],
            geom_conaffinity=[g.conaffinity for g in geoms],
            geom_rbound=[g.rbound for g in geoms],
            geom_id2name=lambda i: names.get(i),
        )
        positions = [np.asarray(g.pos, dtype=float) for g in geoms]
        contact = [
            SimpleNamespace(geom1=2, geom2=3, dist=-0.01, pos=np.array([0.0, 0.0, 0.9])),
            SimpleNamespace(geom1=2, geom2=0, dist=0.001, pos=np.array([0.0, 0.0, 0.9])),
        ]
        data = SimpleNamespace(
            geom_xpos=positions,
            ncon=len(contact),
            contact=contact,
        )
        inner = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
        return SimpleNamespace(inner=inner)

    def test_describe_records_only_wine_gripper(self):
        env = self._fake_env()
        with mock.patch.object(collision_geometry.service, "_inner_env", lambda e: e.inner):
            result = collision_geometry.describe(env, [0.0, 0.0, 1.0])
        self.assertIsNotNone(result)
        pairs = [c["pair"] for c in result["wine_robot_contacts"]]
        self.assertEqual(len(pairs), 1)
        self.assertTrue(any("gripper0" in p[0] or "gripper0" in p[1] for p in pairs))

    def test_nan_rbound_returns_none(self):
        env = self._fake_env()
        model = env.inner.sim.model
        model.geom_rbound[0] = float("nan")
        with mock.patch.object(collision_geometry.service, "_inner_env", lambda e: e.inner):
            result = collision_geometry.describe(env, [0.0, 0.0, 1.0])
        self.assertIsNone(result)


class TestRotationLowerExtent(unittest.TestCase):
    def test_legacy_descriptor_missing_spheres_returns_sweep(self):
        descriptor = {
            "bottle_top_z_m": 1.06,
            "hand_sweep_radius_m": 0.20,
            "wine_robot_contacts": [],
        }
        extent = collision_geometry.rotation_lower_extent(descriptor, _eye3(), _eye3())
        self.assertEqual(extent, 0.20)

    def test_single_sphere_identity_rotation_extent_zero(self):
        descriptor = {
            "bottle_top_z_m": 1.06,
            "hand_sweep_radius_m": 0.20,
            "hand_spheres": [
                {"offset_world": [0.0, 0.0, 0.1], "radius": 0.02},
            ],
            "wine_robot_contacts": [],
        }
        extent = collision_geometry.rotation_lower_extent(descriptor, _eye3(), _eye3())
        self.assertIsNotNone(extent)
        self.assertGreaterEqual(extent, 0.0)
        self.assertLessEqual(extent, 0.02)

    def test_single_sphere_rotated_180_returns_larger_extent(self):
        descriptor = {
            "bottle_top_z_m": 1.06,
            "hand_sweep_radius_m": 0.20,
            "hand_spheres": [
                {"offset_world": [0.0, 0.0, 0.1], "radius": 0.02},
            ],
            "wine_robot_contacts": [],
        }
        rx180 = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        extent = collision_geometry.rotation_lower_extent(descriptor, _eye3(), rx180)
        self.assertIsNotNone(extent)
        self.assertAlmostEqual(extent, 0.12, places=6)

    def test_invalid_inputs_return_none(self):
        base = {
            "bottle_top_z_m": 1.06,
            "hand_sweep_radius_m": 0.20,
            "wine_robot_contacts": [],
        }
        bad_radius = dict(base)
        bad_radius["hand_spheres"] = [
            {"offset_world": [0.0, 0.0, 0.1], "radius": -0.01},
        ]
        self.assertIsNone(
            collision_geometry.rotation_lower_extent(bad_radius, _eye3(), _eye3())
        )
        bad_offset = dict(base)
        bad_offset["hand_spheres"] = [
            {"offset_world": [0.0, 0.0, float("nan")], "radius": 0.02},
        ]
        self.assertIsNone(
            collision_geometry.rotation_lower_extent(bad_offset, _eye3(), _eye3())
        )
        good = dict(base)
        good["hand_spheres"] = [
            {"offset_world": [0.0, 0.0, 0.1], "radius": 0.02},
        ]
        self.assertIsNone(
            collision_geometry.rotation_lower_extent(good, _eye3(), [[1.0, 0.0], [0.0, 1.0]])
        )
        self.assertIsNone(
            collision_geometry.rotation_lower_extent(good, _eye3(), [[1.0, 0.0, float("inf")], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )


if __name__ == "__main__":
    unittest.main()
