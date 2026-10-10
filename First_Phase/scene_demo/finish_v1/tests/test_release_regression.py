"""GPT fixed acceptance definitions, mechanically applied; no physical proof."""
import unittest
from unittest.mock import patch
import numpy as np
from scene_demo.finish_v1 import control, pilot, vision
from scene_demo.finish_v1.tests.test_core import FakePort
from scene_demo.finish_v1.tests.test_takeover_vision import _place_context, _verdict


class ReleaseRegression(unittest.TestCase):
    def test_pre_release_unknown_stability_and_protections(self):
        context = _place_context()
        self.assertTrue(vision.decide(context, _verdict(released=False, stable=None))["allow_release"])
        for overrides in ({"stable": False}, {"supported": False}, {"at_destination": False}):
            value = _verdict(released=False, stable=None)
            value.update(overrides)
            self.assertFalse(vision.decide(context, value)["allow_release"])

    def run_sequence(self, postretreat):
        port = FakePort()
        initial = _verdict(released=False, stable=None, clear_of_target=False)
        postopen = _verdict(released=None, stable=None, clear_of_target=False)
        checks = iter([postopen, postretreat, _verdict()])
        events = []
        with patch.object(control, "plan_retreat", return_value={"ok": True, "waypoints": [np.zeros(7), np.full(7, .02)]}), patch.object(control, "clouds", return_value=np.array([[10, 10, 10]])), patch.object(control, "screen", return_value={"ok": True}), patch.object(control.jh, "home_metrics", side_effect=lambda ref, st: {"ready": np.max(np.abs(st["arm_qpos"])) < 1e-12}):
            result = control.run_finish(port, _place_context(), initial, vision.decide, lambda context, observations: next(checks), events.append)
        return port, result, events

    def test_real_decision_settle_open_retreat_complete(self):
        port, result, events = self.run_sequence(_verdict())
        self.assertTrue(result["ready"], result)
        self.assertEqual(result["counts"]["release"], 10)
        self.assertEqual(result["release_settle_actions"], 5)
        self.assertEqual(result["release_open_actions"], 5)
        self.assertEqual(set(result["counts"]), {"release", "movement", "confirm", "home"})
        np.testing.assert_allclose(np.stack(port.targets_fingers[:5]), np.zeros((5, 2)))
        np.testing.assert_allclose(np.stack(port.targets_fingers[5:10]), np.tile(port.reference["finger_home"], (5, 1)))
        self.assertEqual([row["command_kind"] for row in events[:10]], ["release_settle"] * 5 + ["release_open"] * 5)
        self.assertGreater(result["counts"]["movement"], 0)
        self.assertIsNone(result["checks"][0]["released"])
        self.assertTrue(result["opening_confirmed"])

    def test_postretreat_unknown_release_blocks_home(self):
        port, result, events = self.run_sequence(_verdict(released=None))
        self.assertGreater(result["counts"]["movement"], 0)
        self.assertEqual(result["counts"]["home"], 0)
        self.assertFalse(result["ready"])

    def test_phase_only_waiting_is_not_actual_release(self):
        case = {"allow_helper": True, "expected_release": True, "expected_retreat": True}
        phases = {"release": 6, "movement": 1}
        self.assertFalse(pilot.takeover_metrics(case, phases, 0)["target_executed"])
        self.assertFalse(pilot.takeover_metrics(case, phases)["target_executed"])
        self.assertTrue(pilot.takeover_metrics(case, phases, 1)["target_executed"])
        result = {"status": "blocked", "phase_counts": phases, "release_open_actions": 0}
        self.assertTrue(pilot.should_stop_after_case(case, result))
        result["release_open_actions"] = 1
        self.assertFalse(pilot.should_stop_after_case(case, result))

    def run_online_screen(self, reason):
        port = FakePort()
        initial_path = [np.full(7, i * .01) for i in range(21)]
        replacement_path = [np.full(7, i * .01) for i in range(10, 21)]
        initial_plan = {"ok": True, "waypoints": initial_path}
        replacement_plan = {"ok": True, "waypoints": replacement_path}
        failed_screen = {"ok": False, "reason": reason, "min_distance": .005, "endpoint_hand_clear": False}
        clear_screen = {"ok": True, "reason": "clear", "min_distance": .02, "endpoint_hand_clear": True}
        checks = iter([_verdict(released=None, stable=None, clear_of_target=False), _verdict(), _verdict()])
        with patch.object(control, "plan_retreat", side_effect=[initial_plan, replacement_plan]) as planning, patch.object(control, "clouds", return_value=np.array([[10, 10, 10]])), patch.object(control, "screen", side_effect=[failed_screen, clear_screen]) as screening, patch.object(control.jh, "home_metrics", side_effect=lambda ref, st: {"ready": np.max(np.abs(st["arm_qpos"])) < 1e-12}):
            result = control.run_finish(port, _place_context(), _verdict(released=False, stable=None, clear_of_target=False), vision.decide, lambda context, observations: next(checks), lambda row: None)
        return port, result, planning, screening, initial_plan, replacement_plan

    def test_endpoint_clear_failure_replans_once_without_resetting_counts(self):
        port, result, planning, screening, initial_plan, replacement_plan = self.run_online_screen("endpoint_hand_clear_fail")
        self.assertTrue(result["ready"], result)
        self.assertEqual(len(result["retreat_replans"]), 1)
        self.assertEqual(planning.call_count, 2)
        self.assertEqual(screening.call_count, 2)
        self.assertEqual(result["counts"]["movement"], 20)
        self.assertEqual(port.phases.count("movement"), 20)
        self.assertIs(result["plan"], initial_plan)
        self.assertIs(result["active_retreat_plan"], replacement_plan)
        self.assertEqual(result["movement_screens"][0]["reason"], "endpoint_hand_clear_fail")
        self.assertTrue(result["movement_screens"][1]["ok"])
        np.testing.assert_allclose(port.targets[20], replacement_plan["waypoints"][1])

    def test_approach_violation_blocks_without_replan(self):
        port, result, planning, screening, initial_plan, replacement_plan = self.run_online_screen("approach_violation")
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "movement screen blocked")
        self.assertEqual(len(result.get("retreat_replans", [])), 0)
        self.assertEqual(planning.call_count, 1)
        self.assertEqual(result["counts"]["movement"], 10)
        self.assertEqual(result["counts"]["confirm"], 0)
        self.assertEqual(result["counts"]["home"], 0)


if __name__ == "__main__":
    unittest.main()
