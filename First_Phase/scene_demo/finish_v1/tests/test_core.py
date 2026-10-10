"""GPT-designed, worker-applied focused software acceptance; no physical proof."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scene_demo.finish_v1 import control


class FakePort:
    def __init__(self, q=None, fail_close=False):
        self.reference = {"homeq": np.zeros(7), "finger_home": [.04, -.04]}
        self.q = np.zeros(7) if q is None else np.asarray(q, float)
        self.f = np.zeros(2)
        self.started = False
        self.closed = False
        self.fail_close = fail_close
        self.phases = []

    def start(self):
        self.started = True

    def read_robot(self):
        return {"arm_qpos": self.q.copy(), "arm_qvel": np.zeros(7), "finger_qpos": self.f.copy()}

    def step(self, q, f=None):
        if not self.started:
            raise RuntimeError("step before start")
        self.q = np.asarray(q, float).copy()
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
        self.assertEqual(result["counts"], {"release": 5, "movement": 0, "confirm": 0, "home": 0})
        self.assertEqual(port.phases, ["release"] * 5)
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
        with patch.object(control, "clouds", return_value=np.array([[10, 10, 10]])), patch.object(control, "screen", return_value={"ok": True}), patch.object(control.jh, "home_metrics", return_value={"ready": True}):
            result = self.run_case(port, "complete", "complete")
        self.assertEqual(result["counts"], {"release": 0, "movement": 0, "confirm": 0, "home": 7})
        self.assertEqual(port.phases, ["home"] * 7)
        self.assertTrue(result["ready"])
        self.assertTrue(result["ok"])
        self.assertTrue(result["restored"])

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
