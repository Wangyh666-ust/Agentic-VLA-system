#!/usr/bin/env python3
"""GPU-free unit tests for the isolated wine-only local grasp controller.

These tests are pure Python/NumPy: they never build a simulator, never load a
checkpoint and never touch a GPU.  They drive :class:`local_grasp
.LocalGraspController` with hand-built, reading-compatible mappings and assert
the frozen phase machine:

* the frozen pad target is derived from the wine pose, never a guessed XYZ;
* the local pad offset is compensated in the controller frame;
* the trigger rejects far / open-hand / unknown observations;
* an already-held wine bypasses, unknown geometry during an active attempt
  fails, a foreign held object fails and a protected displacement fails;
* ``above`` -> ``descend`` -> ``close`` -> ``lift`` requires the real
  consecutive post-action sample counts;
* a confirmation needs a real ``>= 0.02 m`` bottle raise;
* repeated ``next_action`` reads never advance a counter;
* every stage budget and the total budget bound the action count;
* every issued action is a finite ``float32`` shape-``(7,)`` vector whose
  gripper channel is open in ``above`` / ``descend`` and closed in ``close`` /
  ``lift``.

``read_geometry`` is exercised with a synthetic, read-only fake environment to
pin the MuJoCo ``wxyz`` quaternion conversion and the controller-frame pad
offset.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import local_grasp  # noqa: E402

WINE = local_grasp.WINE_OBJECT_ID
BOWL = "akita_black_bowl_1"
PLATE = "plate_1"
CREAM = "cream_cheese_1"

TARGET_ORI = np.diag([1.0, -1.0, -1.0])
COMMAND = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
OPEN_COMMAND = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

ARM_POS = [0.0, 0.0, 0.10]
# above target = frozen grasp target (PAD_TARGET_POS) + ABOVE_CLEARANCE_M.
ABOVE_POS = [0.0, 0.0, 0.265]
PAD_TARGET_POS = [0.0, 0.0, 0.205]
LIFT_POS = [0.0, 0.0, 0.24]

BOWL_POS = [0.10, 0.20, 0.30]
PLATE_POS = [0.10, 0.20, 0.00]
CREAM_POS = [-0.10, 0.00, 0.05]


def make_reading(
    position,
    orientation=None,
    wine_position=(0.0, 0.0, 0.10),
    grasped=False,
    held=(),
    complete=True,
    wine_rotation=None,
    pad_offset_local=(0.0, 0.0, 0.0),
    protected=None,
):
    """A reading mapping matching the ``read_geometry`` contract."""

    if orientation is None:
        orientation = np.eye(3)
    if wine_rotation is None:
        wine_rotation = np.eye(3)
    if protected is None:
        protected = {BOWL: BOWL_POS, PLATE: PLATE_POS, CREAM: CREAM_POS}
    objects = {WINE: {"position": list(wine_position), "grasped": grasped}}
    for object_id, pos in protected.items():
        objects[object_id] = {"position": list(pos)}
    snapshot = {
        "objects": objects,
        "held_objects": list(held),
        "grasp_observation_complete": complete,
        "predicates": {},
    }
    return {
        "pose": {
            "position": list(position),
            "orientation_matrix": np.asarray(orientation, dtype=np.float64),
        },
        "snapshot": snapshot,
        "wine_rotation": np.asarray(wine_rotation, dtype=np.float64),
        "pad_midpoint": [0.0, 0.0, 0.0],
        "pad_offset_local": list(pad_offset_local),
        "pad_separation_m": 0.08,
        "grip_site_position": [0.0, 0.0, 0.0],
        "controller_grip_site_error_m": 0.0,
    }


def _step(controller, reading, proposed=COMMAND):
    """One real cycle: ask for an action, then feed the post-action reading."""

    action = controller.next_action(reading, proposed)
    if action is not None:
        controller.observe_after(reading)
    return action


def _advance_to(controller, stage):
    """Drive a fresh controller to the start of ``stage`` with real samples."""

    _step(controller, make_reading(ARM_POS))  # arms -> above
    if stage == "above":
        return
    for _ in range(local_grasp.ALIGN_STREAK):
        _step(controller, make_reading(ABOVE_POS, orientation=TARGET_ORI))
    if stage == "descend":
        return
    for _ in range(local_grasp.ALIGN_STREAK):
        _step(controller, make_reading(PAD_TARGET_POS, orientation=TARGET_ORI))
    if stage == "close":
        return
    for _ in range(local_grasp.GRASP_STREAK):
        _step(
            controller,
            make_reading(PAD_TARGET_POS, orientation=TARGET_ORI, grasped=True, held=(WINE,)),
        )
    assert controller.phase == "lift", controller.phase


def _drive_until_failure(controller, reading, max_steps=500):
    for _ in range(max_steps):
        action = controller.next_action(reading, COMMAND)
        if action is None:
            return
        controller.observe_after(reading)
    raise AssertionError("controller never failed")


class TargetTests(unittest.TestCase):
    def test_make_target_uses_wine_rotation_and_the_fixed_neck_height(self):
        # A current hand frame already matching R1 selects R1.
        reading = make_reading(ARM_POS, orientation=TARGET_ORI, wine_rotation=np.eye(3))
        eef_target, orientation = local_grasp.make_target(reading)
        # pad target = wine + Rwine @ [0, 0, 0.105]; no local offset -> eef = pad.
        self.assertTrue(np.allclose(eef_target, PAD_TARGET_POS, atol=1e-9))
        self.assertTrue(np.allclose(orientation, TARGET_ORI, atol=1e-9))

        # A 90-degree wine yaw rotates the pad target and the hand frame.
        yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        reading = make_reading(ARM_POS, orientation=expected, wine_rotation=yaw)
        eef_target, orientation = local_grasp.make_target(reading)
        self.assertTrue(np.allclose(eef_target, [0.0, 0.0, 0.205], atol=1e-9))
        self.assertTrue(np.allclose(orientation, expected, atol=1e-9))

    def test_make_target_compensates_the_local_pad_offset(self):
        # A local +X pad offset with an identity wine rotation shifts the eef
        # target by -X by the same amount.
        reading = make_reading(
            ARM_POS, orientation=TARGET_ORI, pad_offset_local=(0.02, 0.0, 0.0)
        )
        eef_target, _ = local_grasp.make_target(reading)
        self.assertTrue(np.allclose(eef_target, [-0.02, 0.0, 0.205], atol=1e-9))

        # With a 90-degree yaw the local +X offset maps to +Y in the world, so
        # the compensation is applied in the controller frame (not the world).
        yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        r1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        reading = make_reading(
            ARM_POS, orientation=r1, wine_rotation=yaw, pad_offset_local=(0.02, 0.0, 0.0)
        )
        eef_target, _ = local_grasp.make_target(reading)
        self.assertTrue(np.allclose(eef_target, [0.0, -0.02, 0.205], atol=1e-9))

    def test_make_target_picks_the_closer_symmetric_neck_frame(self):
        # The two symmetric cylindrical neck frames differ by a 180-degree yaw:
        #   R1 = Rwine @ diag(1, -1, -1); R2 = R1 @ diag(-1, -1, 1).
        r1 = np.diag([1.0, -1.0, -1.0])
        r2 = r1 @ np.diag([-1.0, -1.0, 1.0])

        # A current hand frame already matching R2 selects R2, and the local
        # pad-offset compensation uses that actually-selected orientation.
        reading = make_reading(ARM_POS, orientation=r2, pad_offset_local=(0.02, 0.0, 0.0))
        eef_target, orientation = local_grasp.make_target(reading)
        self.assertTrue(np.allclose(orientation, r2, atol=1e-9))
        self.assertTrue(np.allclose(eef_target, [0.02, 0.0, 0.205], atol=1e-9))

        # A current hand frame equidistant from R1 and R2 is a tie: R1 wins.
        tie = np.diag([-1.0, -1.0, 1.0])
        reading = make_reading(ARM_POS, orientation=tie, pad_offset_local=(0.02, 0.0, 0.0))
        eef_target, orientation = local_grasp.make_target(reading)
        self.assertTrue(np.allclose(orientation, r1, atol=1e-9))
        self.assertTrue(np.allclose(eef_target, [-0.02, 0.0, 0.205], atol=1e-9))

    def test_make_target_requires_known_readings(self):
        unknown_offset = make_reading(ARM_POS)
        unknown_offset["pad_offset_local"] = None
        with self.assertRaises(ValueError):
            local_grasp.make_target(unknown_offset)
        unknown_rotation = make_reading(ARM_POS)
        unknown_rotation["wine_rotation"] = None
        with self.assertRaises(ValueError):
            local_grasp.make_target(unknown_rotation)


class TriggerTests(unittest.TestCase):
    def test_far_or_open_hand_never_triggers(self):
        controller = local_grasp.LocalGraspController()
        far = make_reading([0.3, 0.0, 0.10])
        self.assertIsNone(controller.next_action(far, COMMAND))
        self.assertEqual(controller.phase, "idle")

        near = make_reading(ARM_POS)
        self.assertIsNone(controller.next_action(near, OPEN_COMMAND))
        self.assertIsNone(controller.next_action(near, None))
        self.assertEqual(controller.phase, "idle")

    def test_unknown_observations_never_trigger(self):
        for reading in (
            make_reading(ARM_POS, grasped=None),
            make_reading(ARM_POS, complete=False),
            make_reading(ARM_POS, held=(PLATE,)),
        ):
            controller = local_grasp.LocalGraspController()
            self.assertIsNone(controller.next_action(reading, COMMAND))
            self.assertEqual(controller.phase, "idle")

        # A trigger that cannot produce a target (unknown wine rotation) stays
        # idle rather than failing.
        controller = local_grasp.LocalGraspController()
        no_rotation = make_reading(ARM_POS)
        no_rotation["wine_rotation"] = None
        self.assertIsNone(controller.next_action(no_rotation, COMMAND))
        self.assertEqual(controller.phase, "idle")

    def test_already_wine_held_bypasses(self):
        controller = local_grasp.LocalGraspController()
        held = make_reading(ARM_POS, grasped=True, held=(WINE,))
        self.assertIsNone(controller.next_action(held, COMMAND))
        self.assertEqual(controller.phase, "bypass")
        self.assertEqual(controller.reason, "wine_already_held")


class StateMachineTests(unittest.TestCase):
    def test_alignment_stages_then_five_sample_confirmation(self):
        controller = local_grasp.LocalGraspController()
        _step(controller, make_reading(ARM_POS))
        self.assertEqual(controller.phase, "above")

        for _ in range(local_grasp.ALIGN_STREAK):
            _step(controller, make_reading(ABOVE_POS, orientation=TARGET_ORI))
        self.assertEqual(controller.phase, "descend")

        for _ in range(local_grasp.ALIGN_STREAK):
            _step(controller, make_reading(PAD_TARGET_POS, orientation=TARGET_ORI))
        self.assertEqual(controller.phase, "close")

        for _ in range(local_grasp.GRASP_STREAK):
            _step(
                controller,
                make_reading(PAD_TARGET_POS, orientation=TARGET_ORI, grasped=True, held=(WINE,)),
            )
        self.assertEqual(controller.phase, "lift")

        lifted = make_reading(LIFT_POS, orientation=TARGET_ORI, wine_position=(0.0, 0.0, 0.125), grasped=True, held=(WINE,))
        for _ in range(local_grasp.LIFT_STREAK - 1):
            _step(controller, lifted)
        self.assertNotEqual(controller.phase, "confirmed")
        _step(controller, lifted)
        self.assertEqual(controller.phase, "confirmed")
        self.assertIsNone(controller.reason)
        self.assertIsNone(controller.next_action(lifted, COMMAND))

    def test_lift_confirmation_requires_a_real_raise_and_known_grasp(self):
        controller = local_grasp.LocalGraspController()
        _advance_to(controller, "lift")
        up = make_reading(LIFT_POS, orientation=TARGET_ORI, wine_position=(0.0, 0.0, 0.125), grasped=True, held=(WINE,))
        low = make_reading(LIFT_POS, orientation=TARGET_ORI, wine_position=(0.0, 0.0, 0.115), grasped=True, held=(WINE,))
        not_grasped = make_reading(LIFT_POS, orientation=TARGET_ORI, wine_position=(0.0, 0.0, 0.125), grasped=False)

        for _ in range(4):
            _step(controller, up)
        _step(controller, low)  # a below-threshold sample resets the streak
        self.assertNotEqual(controller.phase, "confirmed")
        _step(controller, not_grasped)  # a not-held sample resets the streak
        for _ in range(local_grasp.LIFT_STREAK - 1):
            _step(controller, up)
        self.assertNotEqual(controller.phase, "confirmed")
        _step(controller, up)
        self.assertEqual(controller.phase, "confirmed")

    def test_foreign_held_object_fails(self):
        controller = local_grasp.LocalGraspController()
        _step(controller, make_reading(ARM_POS))
        foreign = make_reading(ARM_POS, held=(BOWL,))
        self.assertIsNone(controller.next_action(foreign, COMMAND))
        self.assertEqual(controller.phase, "failed")
        self.assertEqual(controller.reason, "foreign_held_object")

    def test_active_unknown_geometry_fails(self):
        controller = local_grasp.LocalGraspController()
        _step(controller, make_reading(ARM_POS))
        broken = make_reading(ARM_POS)
        broken["pose"] = None
        self.assertIsNone(controller.next_action(broken, COMMAND))
        self.assertEqual(controller.phase, "failed")
        self.assertEqual(controller.reason, "unknown_geometry")

    def test_above_target_is_the_frozen_grasp_target_plus_clearance(self):
        controller = local_grasp.LocalGraspController()
        _step(controller, make_reading(ARM_POS))
        self.assertEqual(controller.phase, "above")
        self.assertTrue(
            np.allclose(controller._above_target_position, [0.0, 0.0, 0.265], atol=1e-9)
        )
        expected = controller._target_position + np.array(
            [0.0, 0.0, local_grasp.ABOVE_CLEARANCE_M], dtype=np.float64
        )
        self.assertTrue(np.allclose(controller._above_target_position, expected, atol=1e-9))
        # It tracks the frozen grasp target, never the live pose + clearance.
        live_pose_plus_clearance = np.asarray(ARM_POS, dtype=np.float64) + np.array(
            [0.0, 0.0, local_grasp.ABOVE_CLEARANCE_M], dtype=np.float64
        )
        self.assertFalse(
            np.allclose(controller._above_target_position, live_pose_plus_clearance, atol=1e-9)
        )

    def test_active_incomplete_grasp_observation_fails(self):
        controller = local_grasp.LocalGraspController()
        _step(controller, make_reading(ARM_POS))
        incomplete = make_reading(ABOVE_POS, orientation=TARGET_ORI, complete=False)
        self.assertIsNone(controller.next_action(incomplete, COMMAND))
        self.assertEqual(controller.phase, "failed")
        self.assertEqual(controller.reason, "unknown_grasp_observation")

    def test_active_unknown_or_nonbool_grasp_fails(self):
        for bad in (None, 1, 0.0, "yes"):
            controller = local_grasp.LocalGraspController()
            _step(controller, make_reading(ARM_POS))
            reading = make_reading(ABOVE_POS, orientation=TARGET_ORI, grasped=bad)
            self.assertIsNone(controller.next_action(reading, COMMAND))
            self.assertEqual(controller.phase, "failed")
            self.assertEqual(controller.reason, "unknown_grasp")

    def test_active_unknown_rotation_fails(self):
        for bad in (None, np.full((3, 3), np.nan)):
            controller = local_grasp.LocalGraspController()
            _step(controller, make_reading(ARM_POS))
            reading = make_reading(ABOVE_POS, orientation=TARGET_ORI)
            reading["wine_rotation"] = bad
            self.assertIsNone(controller.next_action(reading, COMMAND))
            self.assertEqual(controller.phase, "failed")
            self.assertEqual(controller.reason, "unknown_geometry")

    def test_idle_unknown_trigger_still_returns_none(self):
        for reading in (
            make_reading(ARM_POS, complete=False),
            make_reading(ARM_POS, grasped=None),
            make_reading(ARM_POS, grasped=1),
        ):
            controller = local_grasp.LocalGraspController()
            self.assertIsNone(controller.next_action(reading, COMMAND))
            self.assertEqual(controller.phase, "idle")

        controller = local_grasp.LocalGraspController()
        unknown_rotation = make_reading(ARM_POS)
        unknown_rotation["wine_rotation"] = None
        self.assertIsNone(controller.next_action(unknown_rotation, COMMAND))
        self.assertEqual(controller.phase, "idle")

    def test_protected_displacement_fails(self):
        controller = local_grasp.LocalGraspController()
        _step(controller, make_reading(ARM_POS))
        moved = make_reading(
            ARM_POS,
            protected={BOWL: [0.10, 0.20, 0.31], PLATE: PLATE_POS, CREAM: CREAM_POS},
        )
        self.assertIsNone(controller.next_action(moved, COMMAND))
        self.assertEqual(controller.phase, "failed")
        self.assertTrue(controller.reason.startswith("protected_displacement"))

    def test_repeated_next_action_never_counts(self):
        controller = local_grasp.LocalGraspController()
        near = make_reading(ARM_POS)
        for _ in range(5):
            self.assertIsNotNone(controller.next_action(near, COMMAND))
        self.assertEqual(controller.total_actions, 0)
        self.assertEqual(sum(controller._stage_counts.values()), 0)

        controller.observe_after(near)  # one real post-action sample
        self.assertEqual(controller.total_actions, 1)
        self.assertEqual(controller._stage_counts["above"], 1)

        for _ in range(3):
            controller.next_action(make_reading(ABOVE_POS, orientation=TARGET_ORI), COMMAND)
        self.assertEqual(controller.total_actions, 1)

    def test_total_budget_boundary_fails_before_another_action(self):
        controller = local_grasp.LocalGraspController()
        _advance_to(controller, "lift")
        neutral = make_reading(LIFT_POS, orientation=TARGET_ORI, grasped=True, held=(WINE,))

        controller.total_actions = local_grasp.TOTAL_MAX_ACTIONS - 1
        self.assertIsNotNone(controller.next_action(neutral, COMMAND))
        self.assertEqual(controller.phase, "lift")

        controller.total_actions = local_grasp.TOTAL_MAX_ACTIONS
        self.assertIsNone(controller.next_action(neutral, COMMAND))
        self.assertEqual(controller.phase, "failed")
        self.assertEqual(controller.reason, "total_budget_exceeded")

    def test_each_stage_budget_times_out(self):
        cases = (
            ("above", make_reading(ARM_POS), 0, make_reading([0.0, 0.0, 0.10])),
            ("descend", None, 0, make_reading(ABOVE_POS, orientation=TARGET_ORI)),
            ("close", None, 0, make_reading(PAD_TARGET_POS, orientation=TARGET_ORI)),
            ("lift", None, 0, make_reading(LIFT_POS, orientation=TARGET_ORI)),
        )
        for stage, _, _, stall in cases:
            controller = local_grasp.LocalGraspController()
            _advance_to(controller, stage)
            _drive_until_failure(controller, stall)
            self.assertEqual(controller.phase, "failed", stage)
            self.assertEqual(controller.reason, "%s_timeout" % stage, stage)
            self.assertEqual(
                controller._stage_counts[stage], local_grasp.STAGE_BUDGETS[stage], stage
            )

    def test_actions_are_finite_float32_with_the_expected_gripper(self):
        controller = local_grasp.LocalGraspController()
        above_action = controller.next_action(make_reading(ARM_POS), COMMAND)
        self._assert_action(above_action, local_grasp.GRASP_OPEN)

        _advance_to(controller, "descend")
        descend_action = controller.next_action(
            make_reading(ABOVE_POS, orientation=TARGET_ORI), COMMAND
        )
        self._assert_action(descend_action, local_grasp.GRASP_OPEN)

        _advance_to(controller, "close")
        close_action = controller.next_action(
            make_reading(PAD_TARGET_POS, orientation=TARGET_ORI), COMMAND
        )
        self._assert_action(close_action, local_grasp.GRASP_CLOSED)

        _advance_to(controller, "lift")
        lift_action = controller.next_action(
            make_reading(LIFT_POS, orientation=TARGET_ORI), COMMAND
        )
        self._assert_action(lift_action, local_grasp.GRASP_CLOSED)

    def _assert_action(self, action, gripper):
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (7,))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(bool(np.all(np.isfinite(action))))
        self.assertAlmostEqual(float(action[-1]), float(gripper), places=6)

    def test_summary_reports_the_frozen_evidence(self):
        controller = local_grasp.LocalGraspController()
        _step(controller, make_reading(ARM_POS))
        summary = controller.summary()
        self.assertTrue(summary["assisted"])
        self.assertEqual(summary["phase"], "above")
        self.assertEqual(summary["total_actions"], 1)
        self.assertIn("stage_counts", summary)
        self.assertIn("trigger", summary)
        self.assertIn("last_errors", summary)
        self.assertIsNotNone(summary["target"])
        self.assertIsNotNone(summary["targets"])


class _FakeController:
    def __init__(self, ee_pos, ee_ori_mat):
        self.ee_pos = list(ee_pos)
        self.ee_ori_mat = np.asarray(ee_ori_mat, dtype=np.float64)

    def update(self, force=False):  # noqa: ARG002 - matches the robosuite contract
        return None


class _FakeGripper:
    important_sites = {"grip_site": "grip_site"}
    important_geoms = {
        "left_fingerpad": ["left_pad"],
        "right_fingerpad": ["right_pad"],
    }


class _FakeRobot:
    def __init__(self, controller):
        self.controller = controller
        self.gripper = _FakeGripper()
        self._ref_gripper_joint_pos_indexes = None


class _FakeObject:
    def __init__(self, joints):
        self.joints = list(joints)
        self.contact_geoms = ["cg"]


class _FakeData:
    def __init__(self, qpos):
        self._qpos = qpos

    def get_joint_qpos(self, name):
        return np.asarray(self._qpos[name], dtype=np.float64)

    def get_geom_xpos(self, name):
        return {"left_pad": [0.0, 0.0, 0.50], "right_pad": [0.0, 0.0, 0.60]}[name]

    def get_site_xpos(self, name):  # noqa: ARG002
        return [0.0, 0.0, 0.55]

    def get_site_xmat(self, name):  # noqa: ARG002
        return np.eye(3)


class _FakeModel:
    pass


class _FakeSim:
    def __init__(self, data):
        self.data = data
        self.model = _FakeModel()


class _FakeInner:
    def __init__(self, data, objects, robot):
        self.sim = _FakeSim(data)
        self.objects_dict = objects
        self.robots = [robot]
        self._eval_predicate = lambda predicate: True  # noqa: ARG005
        self._check_grasp = lambda gripper, contact_geoms: False  # noqa: ARG005


class ReadGeometryTests(unittest.TestCase):
    """Read-only geometry probe against a synthetic fake environment."""

    def _env(self):
        quaternion_z90 = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]  # wxyz
        qpos = {
            "wine_joint": [0.0, 0.0, 0.10] + quaternion_z90,
            "bowl_joint": [0.10, 0.20, 0.30, 1.0, 0.0, 0.0, 0.0],
            "plate_joint": [0.10, 0.20, 0.00, 1.0, 0.0, 0.0, 0.0],
            "cream_joint": [-0.10, 0.00, 0.05, 1.0, 0.0, 0.0, 0.0],
        }
        objects = {
            WINE: _FakeObject(["wine_joint"]),
            BOWL: _FakeObject(["bowl_joint"]),
            PLATE: _FakeObject(["plate_joint"]),
            CREAM: _FakeObject(["cream_joint"]),
        }
        robot = _FakeRobot(_FakeController([0.0, 0.0, 0.50], np.eye(3)))
        inner = _FakeInner(_FakeData(qpos), objects, robot)
        return types.SimpleNamespace(_env=types.SimpleNamespace(env=inner))

    def test_geometry_reads_pads_site_and_wxyz_rotation(self):
        reading = local_grasp.read_geometry(self._env())

        self.assertTrue(np.allclose(reading["pad_midpoint"], [0.0, 0.0, 0.55], atol=1e-9))
        self.assertAlmostEqual(reading["pad_separation_m"], 0.10, places=9)
        self.assertTrue(np.allclose(reading["grip_site_position"], [0.0, 0.0, 0.55], atol=1e-9))
        self.assertAlmostEqual(reading["controller_grip_site_error_m"], 0.05, places=9)
        # local offset = R^T @ (pad_midpoint - controller ee_pos), identity frame.
        self.assertTrue(np.allclose(reading["pad_offset_local"], [0.0, 0.0, 0.05], atol=1e-9))
        self.assertTrue(np.allclose(reading["pose"]["position"], [0.0, 0.0, 0.50], atol=1e-9))

        # MuJoCo wxyz [w, x, y, z] must convert to a 90-degree z rotation, which
        # only holds if the components are explicitly reordered to [x, y, z, w].
        expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.allclose(reading["wine_rotation"], expected, atol=1e-9))

        # The snapshot carries the wine grasp and the evaluated predicates.
        self.assertIs(reading["snapshot"]["objects"][WINE]["grasped"], False)
        self.assertEqual(reading["snapshot"]["held_objects"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
