import copy
import unittest

from scene_demo.finish_v1 import vision


def _box(x1, y1, x2, y2):
    return [x1, y1, x2, y2]


def _verdict(**overrides):
    verdict = {
        "target_visible": True,
        "destination_visible": True,
        "at_destination": True,
        "supported": True,
        "released": True,
        "stable": True,
        "operation_achieved": True,
        "clear_of_target": True,
        "boxes": {
            "front": {
                "target": _box(0.10, 0.10, 0.20, 0.20),
                "destination": _box(0.30, 0.30, 0.40, 0.40),
                "gripper": _box(0.80, 0.80, 0.90, 0.90),
            }
        },
        "evidence": [],
        "unknown_reasons": [],
    }
    verdict.update(overrides)
    return verdict


def _place_context():
    return {
        "user_request": "Put the bottle on the shelf.",
        "current_subtask": {
            "operation": "place",
            "target": "bottle",
            "destination": "shelf",
            "phase": "execute",
        },
        "criteria": {
            "operation_semantics": "place the target on the destination",
            "desired_state": "on",
        },
    }


def _turn_context():
    return {
        "user_request": "Turn on the stove.",
        "current_subtask": {
            "operation": "turn",
            "target": "stove knob",
            "destination": None,
            "phase": "execute",
        },
        "criteria": {
            "operation_semantics": "rotate the stove knob to ignite the burner",
            "desired_state": "on",
        },
    }


class ReconcileVerdictTest(unittest.TestCase):
    def test_closed_fingers_downgrade_release_and_clear(self):
        context = _place_context()
        verdict = _verdict(
            released=True,
            clear_of_target=True,
            boxes={
                "front": {
                    "target": _box(0.10, 0.10, 0.20, 0.20),
                    "destination": _box(0.30, 0.30, 0.40, 0.40),
                    "gripper": _box(0.80, 0.80, 0.90, 0.90),
                }
            },
        )
        robot = {"finger_qpos": [0.0, 0.0]}
        reference = {"finger_home": [0.04, 0.04]}
        out = vision.reconcile_verdict(context, verdict, robot, reference)
        self.assertIs(out["released"], False)
        self.assertIs(out["clear_of_target"], False)
        self.assertIn(
            "encoder fingers not at standard open position; this does not prove object grasp",
            out["evidence"],
        )
        decision = vision.decide(context, out)
        self.assertEqual(decision["state"], "needs_release")
        self.assertTrue(decision["allow_release"])

    def test_open_fingers_do_not_change_null_release(self):
        context = _place_context()
        verdict = _verdict(released=None, clear_of_target=None)
        robot = {"finger_qpos": [0.04, 0.04]}
        reference = {"finger_home": [0.04, 0.04]}
        out = vision.reconcile_verdict(context, verdict, robot, reference)
        self.assertIsNone(out["released"])
        self.assertIsNone(out["clear_of_target"])

    def test_near_support_projection_retains_raw_clear(self):
        context = _place_context()
        verdict = _verdict(
            released=True,
            stable=True,
            clear_of_target=True,
            boxes={
                "front": {
                    "target": _box(0.90, 0.90, 0.95, 0.95),
                    "destination": _box(0.10, 0.10, 0.15, 0.15),
                    "gripper": _box(0.10, 0.10, 0.15, 0.15),
                }
            },
        )
        robot = {"finger_qpos": [0.04, 0.04]}
        reference = {"finger_home": [0.04, 0.04]}
        out = vision.reconcile_verdict(context, verdict, robot, reference)
        self.assertIs(out["clear_of_target"], True)
        self.assertIn(
            "2D projection overlap near target/support; not 3D contact evidence",
            out["evidence"],
        )
        decision = vision.decide(context, out)
        self.assertEqual(decision["state"], "complete")
        self.assertFalse(decision["allow_retreat"])

    def test_invalid_finger_data_is_rejected_without_mutation(self):
        context = _place_context()
        verdict = _verdict()
        before = copy.deepcopy(verdict)
        robot = {"finger_qpos": [0.04, "bad"]}
        reference = {"finger_home": [0.04, 0.04]}
        with self.assertRaises(TypeError):
            vision.reconcile_verdict(context, verdict, robot, reference)
        self.assertEqual(verdict, before)



    def test_closed_fingers_override_null_release_and_clear(self):
        context = _place_context()
        verdict = _verdict(released=None, clear_of_target=None)
        robot = {"finger_qpos": [0.0, 0.0]}
        reference = {"finger_home": [0.04, 0.04]}
        before = copy.deepcopy((context, verdict, robot, reference))
        out = vision.reconcile_verdict(context, verdict, robot, reference)
        self.assertIs(out["released"], False)
        self.assertIs(out["clear_of_target"], False)
        self.assertEqual((context, verdict, robot, reference), before)

class TurnDecisionTest(unittest.TestCase):
    def test_turn_on_with_closed_fingers_needs_release(self):
        context = _turn_context()
        verdict = _verdict(
            operation_achieved=True,
            released=False,
            clear_of_target=False,
        )
        decision = vision.decide(context, verdict)
        self.assertEqual(decision["state"], "needs_release")
        self.assertTrue(decision["allow_release"])
        self.assertIn("knob-operation done release fingers", decision["reason"])

    def test_turn_not_achieved_is_incomplete(self):
        context = _turn_context()
        verdict = _verdict(operation_achieved=False, released=False)
        decision = vision.decide(context, verdict)
        self.assertEqual(decision["state"], "incomplete")

    def test_turn_achieved_and_clear_and_released_is_complete(self):
        context = _turn_context()
        verdict = _verdict(operation_achieved=True, released=True, clear_of_target=True)
        decision = vision.decide(context, verdict)
        self.assertEqual(decision["state"], "complete")

    def test_turn_clear_unknown_release_unknown_is_unknown(self):
        context = _turn_context()
        verdict = _verdict(operation_achieved=True, released=None, clear_of_target=None)
        decision = vision.decide(context, verdict)
        self.assertEqual(decision["state"], "unknown")


class BuildPromptTest(unittest.TestCase):
    def test_measurement_whitelist_prompt_and_extra_dropped(self):
        context = _place_context()
        context["actuator_measurements"] = {
            "stage": "confirm",
            "finger_qpos": [0.04, -0.04],
            "max_abs_arm_qvel": 0.0,
            "sim_truth": "truth",
            "gold": "gold",
        }
        prompt = vision.build_prompt(context)
        self.assertIn("Measurement context:", prompt)
        self.assertIn("Max Abs Arm Qvel", prompt.replace("max_abs_arm_qvel", "Max Abs Arm Qvel"))
        self.assertNotIn("sim_truth", prompt)
        self.assertNotIn("gold", prompt)

    def test_measurement_invalid_values_raise(self):
        for bad in [
            {"stage": "confirm", "finger_qpos": [float("nan"), 0.0], "max_abs_arm_qvel": 0.0},
            {"stage": "confirm", "finger_qpos": [0.0, 0.0], "max_abs_arm_qvel": -0.1},
            {"stage": "confirm", "finger_qpos": [True, 0.0], "max_abs_arm_qvel": 0.0},
            {"stage": "confirm", "finger_qpos": [0.0], "max_abs_arm_qvel": 0.0},
        ]:
            context = _place_context()
            context["actuator_measurements"] = bad
            with self.assertRaises((ValueError, TypeError)):
                vision.build_prompt(context)

    def test_measurement_absent_compatible(self):
        context = _place_context()
        prompt = vision.build_prompt(context)
        self.assertNotIn("Measurement context:", prompt)

    def test_open_measurements_released_none_not_complete(self):
        context = _place_context()
        context["actuator_measurements"] = {
            "stage": "confirm",
            "finger_qpos": [0.04, -0.04],
            "max_abs_arm_qvel": 0.0,
        }
        verdict = _verdict(released=None)
        result = vision.decide(context, verdict)
        self.assertNotEqual(result.get("state"), "complete")

    def test_prompt_contains_turn_support_and_no_gold(self):
        context = _place_context()
        context["sim_truth"] = "secret"
        context["plan"] = {"success": True}
        prompt = vision.build_prompt(context)
        self.assertIn("turn/on", prompt)
        self.assertIn("support surface", prompt)
        self.assertNotIn("sim_truth", prompt)
        self.assertNotIn("plan.success", prompt)
        self.assertNotIn("secret", prompt)

    def test_prompt_mentions_stove_semantics(self):
        prompt = vision.build_prompt(_turn_context())
        self.assertIn("red burner", prompt)
        self.assertIn("rotate the stove knob", prompt)


if __name__ == "__main__":
    unittest.main()
