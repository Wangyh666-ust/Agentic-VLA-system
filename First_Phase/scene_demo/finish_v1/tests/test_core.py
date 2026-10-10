"""GPT-designed, worker-applied focused software acceptance; no physical proof."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scene_demo.finish_v1 import control


class FakePort:
    def __init__(self, q=None, fail_close=False, screen_ok=True, follow=1.0):
        self.reference = {"homeq": np.zeros(7), "finger_home": [.04, -.04]}
        self.q = np.zeros(7) if q is None else np.asarray(q, float)
        self.f = np.zeros(2)
        self.started = False
        self.closed = False
        self.fail_close = fail_close
        self.phases = []
        self.screen_ok = screen_ok
        self.follow = follow
        self.targets = []
        self.targets_fingers = []
        self.screens = []

    def home_screen(self, waypoints):
        self.screens.append([np.asarray(w, float).tolist() for w in waypoints])
        return {"ok": self.screen_ok, "reason": "fixture"}

    def start(self):
        self.started = True

    def read_robot(self):
        return {"arm_qpos": self.q.copy(), "arm_qvel": np.zeros(7), "finger_qpos": self.f.copy()}

    def step(self, q, f=None):
        if not self.started:
            raise RuntimeError("step before start")
        target = np.asarray(q, float)
        self.targets.append(target.copy())
        self.targets_fingers.append(None if f is None else np.asarray(f, float).copy())
        self.q = self.q + self.follow * (target - self.q)
        if f is not None:
            self.f = np.asarray(f, float).copy()
        self.phases.append(self.phase)
        return self.read_robot()

    def capture(self):
        return {"frames": [], "eef_position": np.zeros(3), "eef_rotation": np.eye(3), "robot": self.read_robot()}

    def robot_spheres(self, q):
        return [{"id": "hand", "center": [0, 0, 0], "radius": .001, "hand": True}]

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("restore failed")


def verdict(state):
    result = {name: True for name in ("target_visible", "destination_visible", "at_destination", "supported", "released", "stable", "operation_achieved", "clear_of_target")}
    result["decision"] = {"state": state, "allow_release": state == "needs_release", "allow_retreat": state == "needs_retreat", "reason": "fixture"}
    if state == "needs_release":
        result["released"] = False
    return result


class CoreTests(unittest.TestCase):
    def run_case(self, port, state, post="unknown"):
        return control.run_finish(port, {"current_subtask": {"operation": "place"}}, verdict(state), lambda context, value: value["decision"], lambda context, observations: verdict(post), lambda row: None)

    def test_incomplete_zero_actions_restored(self):
        port = FakePort()
        result = self.run_case(port, "incomplete")
        self.assertEqual(result["counts"], {"release": 0, "movement": 0, "confirm": 0, "home": 0})
        self.assertEqual(result["reason"], "initial stop")
        self.assertFalse(port.started)
        self.assertTrue(port.closed)

    def test_release_requires_five_actual_posts(self):
        port = FakePort()
        result = self.run_case(port, "needs_release")
        self.assertEqual(result["counts"], {"release": 10, "movement": 0, "confirm": 0, "home": 0})
        self.assertEqual(port.phases, ["release"] * 10)
        self.assertFalse(result["ready"])
        self.assertTrue(result["restored"])

    def test_movement_global_budget(self):
        port = FakePort()
        path = [np.full(7, i * .01) for i in range(91)]
        with patch.object(control, "plan_retreat", return_value={"ok": True, "waypoints": path}), patch.object(control, "clouds", return_value=np.array([[10, 10, 10]])), patch.object(control, "screen", return_value={"ok": True}):
            result = self.run_case(port, "needs_retreat")
        self.assertEqual(result["counts"], {"release": 0, "movement": 80, "confirm": 0, "home": 0})
        self.assertEqual(result["reason"], "movement budget")
        self.assertEqual(len(port.phases), 80)
        self.assertTrue(port.closed)

    def test_confirmation_separate_from_movement(self):
        port = FakePort()
        with patch.object(control, "plan_retreat", return_value={"ok": True, "waypoints": [np.zeros(7), np.full(7, .02)]}), patch.object(control, "clouds", return_value=np.array([[10, 10, 10]])), patch.object(control, "screen", return_value={"ok": True}):
            result = self.run_case(port, "needs_retreat")
        self.assertEqual(result["counts"], {"release": 0, "movement": 1, "confirm": 5, "home": 0})
        self.assertEqual(port.phases, ["movement"] + ["confirm"] * 5)
        self.assertFalse(result["ready"])

    def test_home_approach_plus_five_posts(self):
        port = FakePort(np.full(7, .02))
        with patch.object(control.jh, "home_metrics", side_effect=lambda ref, st: {"ready": np.max(np.abs(st["arm_qpos"])) < 1e-12}):
            result = self.run_case(port, "complete", "complete")
        self.assertEqual(result["counts"], {"release": 0, "movement": 0, "confirm": 0, "home": 6})
        self.assertEqual(port.phases, ["home"] * 6)
        points = control.jh.joint_waypoints(np.full(7, .02), np.zeros(7))
        np.testing.assert_allclose(np.stack(port.targets), np.stack([points[min(i, len(points)-1)] for i in range(6)]))
        self.assertTrue(result["ready"])
        self.assertTrue(result["ok"])
        self.assertTrue(result["restored"])

    def test_home_streaming_delayed_follow(self):
        port = FakePort(np.full(7, .02), follow=.25)
        with patch.object(control.jh, "home_metrics", side_effect=lambda ref, st: {"ready": np.max(np.abs(st["arm_qpos"])) <= .002}):
            result = self.run_case(port, "complete", "complete")
        points = control.jh.joint_waypoints(np.full(7, .02), np.zeros(7))
        np.testing.assert_allclose(np.stack(port.targets[:2]), np.stack(points[:2]))
        self.assertTrue(result["ready"])
        self.assertEqual(result["home_streak"], 5)
        self.assertLess(result["counts"]["home"], 360)

    def run_release_sequence(self, operation):
        port = FakePort()
        initial = verdict("needs_release")
        checks = [verdict("needs_retreat"), verdict("complete"), verdict("complete")]
        if operation == "turn":
            for value in [initial] + checks:
                for key in ("at_destination", "supported", "stable"):
                    value[key] = None
        responses = iter(checks)
        with patch.object(control, "plan_retreat", return_value={"ok": True, "waypoints": [np.zeros(7), np.full(7, .02)]}), patch.object(control, "clouds", return_value=np.array([[10, 10, 10]])), patch.object(control, "screen", return_value={"ok": True}), patch.object(control.jh, "home_metrics", side_effect=lambda ref, st: {"ready": np.max(np.abs(st["arm_qpos"])) < 1e-12}):
            result = control.run_finish(port, {"current_subtask": {"operation": operation}}, initial, lambda context, value: value["decision"], lambda context, observations: next(responses), lambda row: None)
        return port, result

    def test_release_retreat_home_order(self):
        port, result = self.run_release_sequence("place")
        for phase in ("release", "movement", "confirm", "home"):
            self.assertGreater(result["counts"][phase], 0)
        self.assertEqual(list(dict.fromkeys(port.phases)), ["release", "movement", "confirm", "home"])
        self.assertTrue(result["ready"])

    def test_turn_release_does_not_require_place_fields(self):
        port, result = self.run_release_sequence("turn")
        for phase in ("release", "movement", "confirm", "home"):
            self.assertGreater(result["counts"][phase], 0, result)
        self.assertTrue(result["ready"])

    def test_home_screen_blocked_zero_home_actions(self):
        port = FakePort(np.full(7, .02), screen_ok=False)
        result = self.run_case(port, "complete", "complete")
        self.assertEqual(result["counts"]["home"], 0)
        self.assertEqual(port.phases, [])
        self.assertEqual(result["reason"], "home screen blocked")
        self.assertTrue(port.closed)

    def test_close_error_propagates(self):
        port = FakePort(fail_close=True)
        with self.assertRaisesRegex(RuntimeError, "restore failed"):
            self.run_case(port, "incomplete")
        self.assertEqual(port.phases, [])

    def test_cloud_transform_and_invalid_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            depth = Path(directory) / "depth.npy"
            np.save(depth, np.full((8, 8), .5))
            transform = np.eye(4)
            transform[:3, 3] = [1, 2, 3]
            frame = {"depth_path": str(depth), "K": np.eye(3), "T_world_camera": transform}
            points = control.clouds([frame])
            np.testing.assert_allclose(points[0], [1, 2, 3.5])
            frame["T_world_camera"] = np.eye(3)
            self.assertEqual(control.clouds([frame]).shape, (0, 3))

    def test_first_segment_approach_blocked(self):
        start = [{"id": "hand", "center": [0, 0, 0], "radius": .1, "hand": True}]
        path = [[{"id": "hand", "center": [.002, 0, 0], "radius": .1, "hand": True}]]
        result = control.screen(np.array([[.105, 0, 0]]), start, path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "approach_violation")


if __name__ == "__main__":
    unittest.main()
