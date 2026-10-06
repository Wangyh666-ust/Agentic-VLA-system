#!/usr/bin/env python3
"""GPU-free tests for ``scene_demo/placement_completion.py``.

These tests build a *native-shaped* fake inner environment (objects_dict with
named free joints, a gripper, a contact-geom grasp probe and a
``sim.data.get_joint_qvel`` reader).  No simulator, model, checkpoint or CUDA is
ever created: only the pure ``probe_placement_completion`` screening logic is
exercised.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

_SCENE_DEMO = Path(__file__).resolve().parent.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import placement_completion as pc  # noqa: E402

BOWL = ["on", "akita_black_bowl_1", "plate_1"]
BOWL_KEY = "on|akita_black_bowl_1|plate_1"
STOVE = ["turnon", "flat_stove_1"]
STOVE_KEY = "turnon|flat_stove_1"

REST = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _Data:
    """Minimal native ``sim.data``: free-joint velocity by joint name."""

    def __init__(self, qvel_by_joint):
        self._qvel = dict(qvel_by_joint)

    def get_joint_qvel(self, name):
        if name not in self._qvel:
            raise KeyError(name)
        return self._qvel[name]


_MISSING = object()


def _inner(objects, grasp_map, qvel_map, gripper=True):
    """Build a native-shaped inner env.

    ``objects`` is a list of ``(object_id, joint_name)`` or
    ``(object_id, joint_name, contact_geoms)``; the optional third element is the
    object's ``contact_geoms`` -- passing ``_MISSING`` omits the attribute
    entirely (an unprobeable object whose grasp must stay unknown).  ``grasp_map``
    maps an object id to ``True``/``False``/``None``/``"raise"``; ``qvel_map``
    maps a joint name to a velocity vector (or omits it so the read raises).
    Every ``_check_grasp`` call records its ``geoms`` argument in
    ``inner.probe_calls`` so tests can prove the probe was *not* called.
    """

    objs: dict[str, types.SimpleNamespace] = {}
    for entry in objects:
        object_id, joint = entry[0], entry[1]
        contact = entry[2] if len(entry) > 2 else object_id
        obj = types.SimpleNamespace(joints=[joint])
        if contact is not _MISSING:
            obj.contact_geoms = contact
        objs[object_id] = obj

    data = _Data(qvel_map)

    class _Gripper:
        pass

    gripper_obj = _Gripper() if gripper else None
    robots = [types.SimpleNamespace(gripper=gripper_obj)] if gripper else []

    class _Inner:
        def __init__(self):
            self.objects_dict = objs
            self.sim = types.SimpleNamespace(data=data)
            self.robots = robots
            self.probe_calls = []

        def _check_grasp(self, g, geoms):  # noqa: ARG002
            self.probe_calls.append(geoms)
            outcome = grasp_map.get(geoms, False)
            if outcome == "raise":
                raise RuntimeError("grasp probe failed")
            return outcome

    return _Inner()


class ProbePlacementCompletionTests(unittest.TestCase):
    def test_held_bowl_raw_on_true_is_not_ready(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": True, "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIs(result["ready"], False)
        self.assertEqual(result["held_objects"], ["akita_black_bowl_1"])
        self.assertIs(result["grasp_observation_complete"], True)
        self.assertEqual(result["phase"], "goal_still_held")

    def test_released_but_high_linear_velocity_is_not_ready(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": [0.03, 0.0, 0.0, 0.0, 0.0, 0.0], "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIs(result["ready"], False)
        self.assertEqual(result["phase"], "not_holding")
        self.assertAlmostEqual(result["goal_speeds"][BOWL_KEY]["linear_speed"], 0.03)

    def test_released_but_high_angular_velocity_is_not_ready(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": [0.0, 0.0, 0.0, 0.3, 0.0, 0.0], "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIs(result["ready"], False)
        self.assertAlmostEqual(result["goal_speeds"][BOWL_KEY]["angular_speed"], 0.3)

    def test_stable_released_is_ready(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIs(result["ready"], True)
        self.assertEqual(result["held_objects"], [])
        self.assertEqual(result["phase"], "not_holding")

    def test_raw_false_is_not_ready(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: False})
        self.assertIs(result["ready"], False)

    def test_unknown_grasp_is_unknown(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": "raise", "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIsNone(result["ready"])
        self.assertIs(result["grasp_observation_complete"], False)
        self.assertEqual(result["phase"], "unknown")

    def test_invalid_angular_velocity_is_unknown(self):
        # A 5-component qvel yields linear but no valid angular velocity.
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": [0.0, 0.0, 0.0, 0.0, 0.0], "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIsNone(result["ready"])
        self.assertIsNone(result["goal_speeds"][BOWL_KEY]["angular_speed"])

    def test_nonfinite_velocity_is_unknown(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0], "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIsNone(result["ready"])

    def test_missing_target_velocity_is_unknown(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"plate_j": REST},  # bowl joint absent -> read raises
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIsNone(result["ready"])

    def test_nonplacement_stove_raw_goal_is_unaffected(self):
        # No on/in placement target: only the raw conjunction matters, and no
        # gripper is required at all.
        inner = _inner(
            [("flat_stove_1", "stove_j")],
            {},
            {"stove_j": REST},
            gripper=False,
        )
        result = pc.probe_placement_completion(inner, [STOVE], {STOVE_KEY: True})
        self.assertIs(result["ready"], True)
        self.assertEqual(result["goal_speeds"], {})

    def test_nonplacement_stove_raw_false_is_not_ready(self):
        inner = _inner([("flat_stove_1", "stove_j")], {}, {"stove_j": REST}, gripper=False)
        result = pc.probe_placement_completion(inner, [STOVE], {STOVE_KEY: False})
        self.assertIs(result["ready"], False)

    def test_held_foreign_object_is_not_ready(self):
        inner = _inner(
            [
                ("akita_black_bowl_1", "bowl_j"),
                ("plate_1", "plate_j"),
                ("tomato_sauce_1", "sauce_j"),
            ],
            {"akita_black_bowl_1": False, "plate_1": False, "tomato_sauce_1": True},
            {"bowl_j": REST, "plate_j": REST, "sauce_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIs(result["ready"], False)
        self.assertEqual(result["held_objects"], ["tomato_sauce_1"])
        self.assertEqual(result["phase"], "holding_foreign_object")

    def test_unknown_foreign_grasp_fails_closed(self):
        # The goal object is fine, but a *foreign* object's grasp probe is
        # unavailable: the all-object screen makes the observation incomplete.
        inner = _inner(
            [
                ("akita_black_bowl_1", "bowl_j"),
                ("plate_1", "plate_j"),
                ("tomato_sauce_1", "sauce_j"),
            ],
            {"akita_black_bowl_1": False, "plate_1": False, "tomato_sauce_1": "raise"},
            {"bowl_j": REST, "plate_j": REST, "sauce_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIsNone(result["ready"])
        self.assertIs(result["grasp_observation_complete"], False)

    def test_check_grasp_none_on_target_is_unknown(self):
        # The probe itself returns None (an unavailable measurement): it must
        # never be coerced to False and counted as released.
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": None, "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIsNone(result["ready"])
        self.assertIs(result["grasp_observation_complete"], False)
        self.assertEqual(result["phase"], "unknown")
        self.assertEqual(result["held_objects"], [])

    def test_check_grasp_none_on_foreign_is_unknown(self):
        inner = _inner(
            [
                ("akita_black_bowl_1", "bowl_j"),
                ("plate_1", "plate_j"),
                ("tomato_sauce_1", "sauce_j"),
            ],
            {"akita_black_bowl_1": False, "plate_1": False, "tomato_sauce_1": None},
            {"bowl_j": REST, "plate_j": REST, "sauce_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertIsNone(result["ready"])
        self.assertIs(result["grasp_observation_complete"], False)
        self.assertEqual(result["phase"], "unknown")

    def test_missing_contact_geoms_on_target_is_unknown_without_probe(self):
        # The target object exposes NO contact_geoms attribute: the probe must
        # not be called for it at all (a fallback probe would return False), and
        # its grasp stays unknown.
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j", _MISSING), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertNotIn(None, inner.probe_calls)
        self.assertNotIn("akita_black_bowl_1", inner.probe_calls)
        self.assertIsNone(result["ready"])
        self.assertIs(result["grasp_observation_complete"], False)
        self.assertEqual(result["phase"], "unknown")

    def test_missing_contact_geoms_on_foreign_is_unknown_without_probe(self):
        inner = _inner(
            [
                ("akita_black_bowl_1", "bowl_j"),
                ("plate_1", "plate_j"),
                ("tomato_sauce_1", "sauce_j", _MISSING),
            ],
            {"akita_black_bowl_1": False, "plate_1": False, "tomato_sauce_1": False},
            {"bowl_j": REST, "plate_j": REST, "sauce_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: True})
        self.assertNotIn(None, inner.probe_calls)
        self.assertNotIn("tomato_sauce_1", inner.probe_calls)
        self.assertIsNone(result["ready"])
        self.assertIs(result["grasp_observation_complete"], False)
        self.assertEqual(result["phase"], "unknown")

    def test_target_held_outside_goal_phase(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": True, "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        result = pc.probe_placement_completion(inner, [BOWL], {BOWL_KEY: False})
        self.assertEqual(result["phase"], "holding_target_outside_goal")
        self.assertIs(result["ready"], False)

    def test_empty_predicates_are_never_ready(self):
        inner = _inner(
            [("akita_black_bowl_1", "bowl_j"), ("plate_1", "plate_j")],
            {"akita_black_bowl_1": False, "plate_1": False},
            {"bowl_j": REST, "plate_j": REST},
        )
        self.assertIs(pc.probe_placement_completion(inner, [BOWL], {})["ready"], False)

    def test_phases_constant_is_exact(self):
        self.assertEqual(
            tuple(pc.PHASES),
            (
                "holding_target_outside_goal",
                "goal_still_held",
                "holding_foreign_object",
                "not_holding",
                "unknown",
            ),
        )

    def test_module_has_no_service_or_runner_import(self):
        # The probe must stay import-cycle free.
        self.assertNotIn("service", sys.modules.get("placement_completion").__dict__)
        source = Path(pc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import service", source)
        self.assertNotIn("import placement_experiments", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
