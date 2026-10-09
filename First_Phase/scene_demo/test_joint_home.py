import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import joint_home as jh


def vec7(*vals):
    return [float(v) for v in vals]


def vec2(*vals):
    return [float(v) for v in vals]


def make_sim(qpos=None, qvel=None, time=0.0, joint_limited=True, joint_range=None):
    if qpos is None:
        qpos = [0.0] * 12
    if qvel is None:
        qvel = [0.0] * 12
    if joint_range is None:
        joint_range = [[-2.0, 2.0]] * 10
    names = {}

    def joint_name2id(name):
        idx = names.get(name, -1)
        return idx

    def joint_id2name(idx):
        for name, value in names.items():
            if value == idx:
                return name
        return ""

    geom_map = {"arm_geom": 0, "gripper_geom": 1, "object_a": 2, "object_b": 3}

    def geom_name2id(name):
        return geom_map.get(name, -1)

    def geom_id2name(idx):
        for name, value in geom_map.items():
            if value == idx:
                return name
        return ""

    model = SimpleNamespace(
        joint_name2id=joint_name2id,
        joint_id2name=joint_id2name,
        jnt_limited=[joint_limited] * 10,
        jnt_range=np.asarray(joint_range, dtype=float),
        ngeom=4,
        geom_name2id=geom_name2id,
        geom_id2name=geom_id2name,
        _model=object(),
    )
    data = SimpleNamespace(qpos=np.asarray(qpos, dtype=float), qvel=np.asarray(qvel, dtype=float), time=time)
    return SimpleNamespace(model=model, data=data), names


def make_state(joint_names=None, limits_low=None, limits_high=None, arm_qpos=None, arm_qvel=None, finger_qpos=None, finger_qvel=None):
    if joint_names is None:
        joint_names = ["j%d" % i for i in range(7)]
    if limits_low is None:
        limits_low = [-2.0] * 7
    if limits_high is None:
        limits_high = [2.0] * 7
    if arm_qpos is None:
        arm_qpos = [0.0] * 7
    if arm_qvel is None:
        arm_qvel = [0.0] * 7
    if finger_qpos is None:
        finger_qpos = [0.0] * 2
    if finger_qvel is None:
        finger_qvel = [0.0] * 2
    return {
        "joint_names": list(joint_names),
        "joint_indexes": list(range(7)),
        "qpos_indexes": list(range(7)),
        "qvel_indexes": list(range(7)),
        "joint_limits_low": list(limits_low),
        "joint_limits_high": list(limits_high),
        "arm_qpos": list(arm_qpos),
        "arm_qvel": list(arm_qvel),
        "finger_qpos_indexes": [7, 8],
        "finger_qvel_indexes": [7, 8],
        "finger_qpos": list(finger_qpos),
        "finger_qvel": list(finger_qvel),
        "torque_limits_low": [-10.0] * 7,
        "torque_limits_high": [10.0] * 7,
    }


def make_env(qpos=None, qvel=None, joint_names=None, limits_low=None, limits_high=None, ctrl=None):
    if qpos is None:
        qpos = [0.0] * 7 + [0.04, -0.04] + [0.0] * 3
    if qvel is None:
        qvel = [0.0] * 12
    if joint_names is None:
        joint_names = ["j%d" % i for i in range(7)]
    if limits_low is None:
        limits_low = [-2.0] * 7
    if limits_high is None:
        limits_high = [2.0] * 7
    sim, names = make_sim(qpos=qpos, qvel=qvel, joint_range=[[-2.0, 2.0]] * 7)
    for i, name in enumerate(joint_names):
        names[name] = i
    if ctrl is None:
        ctrl = SimpleNamespace(
            joint_index=list(range(7)),
            qpos_index=list(range(7)),
            qvel_index=list(range(7)),
            eef_name="eef",
            update=mock.Mock(),
        )
    if not callable(getattr(ctrl, "update", None)):
        ctrl.update = mock.Mock()
    ctrl.joint_index = list(range(7))
    ctrl.qpos_index = list(range(7))
    ctrl.qvel_index = list(range(7))
    robot = SimpleNamespace(
        name="robot0",
        controller=ctrl,
        sim=sim,
        _ref_gripper_joint_pos_indexes=[7, 8],
        _ref_gripper_joint_vel_indexes=[7, 8],
        robot_model=SimpleNamespace(contact_geoms=["arm_geom"]),
        gripper=SimpleNamespace(contact_geoms=["gripper_geom"]),
        torque_limits=np.asarray([[-10.0] * 7, [10.0] * 7], dtype=float),
    )
    env = SimpleNamespace(robots=[robot], control_freq=20)
    return env


def make_reference(state=None, **extra):
    if state is None:
        state = make_state()
    ref = {
        "schema_version": 1,
        "origin_sha256": "a" * 64,
        "joint_names": list(state["joint_names"]),
        "joint_indexes": list(state["joint_indexes"]),
        "qpos_indexes": list(state["qpos_indexes"]),
        "qvel_indexes": list(state["qvel_indexes"]),
        "joint_limits_low": list(state["joint_limits_low"]),
        "joint_limits_high": list(state["joint_limits_high"]),
        "homeq": list(state["arm_qpos"]),
        "home_joint_speed": list(state["arm_qvel"]),
        "finger_qpos_indexes": list(state["finger_qpos_indexes"]),
        "finger_qvel_indexes": list(state["finger_qvel_indexes"]),
        "finger_home": list(state["finger_qpos"]),
        "finger_speed": list(state["finger_qvel"]),
        "eef_pose": {"position": [0.0, 0.0, 0.0], "orientation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
    }
    ref.update(extra)
    return ref


def _fake_preparation_diagnostics():
    module = types.ModuleType("preparation_diagnostics")
    module.read_eef_pose = lambda env: {
        "position": [0.0, 0.0, 0.0],
        "orientation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    module.controller_facts = lambda env: {
        "type": "OSC",
        "control_dim": 6,
        "use_delta": True,
        "input_min": [-1.0] * 6,
        "input_max": [1.0] * 6,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "ee_pos": [0.0, 0.0, 0.0],
        "ee_ori_mat": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "error": None,
    }
    module.controller_mismatch = lambda facts: facts.get("error")
    return module


class TestJointHome(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(jh, "_inner_env", side_effect=lambda env: env)
        patcher.start()
        self.addCleanup(patcher.stop)

        fake_pd = _fake_preparation_diagnostics()
        fake_scene_demo = types.ModuleType("scene_demo")
        fake_scene_demo.preparation_diagnostics = fake_pd
        modules = {
            "preparation_diagnostics": fake_pd,
            "scene_demo": fake_scene_demo,
            "scene_demo.preparation_diagnostics": fake_pd,
        }
        sys_patcher = mock.patch.dict(sys.modules, modules)
        sys_patcher.start()
        self.addCleanup(sys_patcher.stop)

    def test_joint_waypoints_reaches_target_and_increments(self):
        start = vec7(0, 0, 0, 0, 0, 0, 0)
        target = vec7(0.05, 0, 0, 0, 0, 0, 0)
        wps = jh.joint_waypoints(start, target)
        self.assertEqual(wps[-1], target)
        prev = start
        for wp in wps:
            for a, b in zip(prev, wp):
                self.assertLessEqual(abs(a - b), jh.JOINT_STEP_RAD + 1e-9)
            prev = wp

    def test_joint_waypoints_each_axis_reaches_endpoint(self):
        start = vec7(0, 0, 0, 0, 0, 0, 0)
        target = vec7(0.03, -0.02, 0.01, -0.03, 0.02, -0.01, 0.03)
        wps = jh.joint_waypoints(start, target)
        self.assertEqual(wps[-1], target)
        self.assertLessEqual(len(wps), 10)

    def test_joint_waypoints_shared_alpha_across_axes(self):
        start = vec7(0, 0, 0, 0, 0, 0, 0)
        target = vec7(0.04, 0.02, 0, 0, 0, 0, 0)
        wps = jh.joint_waypoints(start, target)
        mid = wps[0]
        self.assertAlmostEqual(mid[0] / 0.04, mid[1] / 0.02, places=7)

    def test_joint_waypoints_cross_zero_no_angular_wrap(self):
        start = vec7(-3, 0, 0, 0, 0, 0, 0)
        target = vec7(3, 0, 0, 0, 0, 0, 0)
        wps = jh.joint_waypoints(start, target, step_rad=1.0)
        self.assertEqual(wps[-1], target)
        self.assertEqual(len(wps), 6)
        self.assertAlmostEqual(wps[0][0], -2.0)

    def test_joint_waypoints_inputs_unchanged(self):
        start = vec7(0, 0, 0, 0, 0, 0, 0)
        target = vec7(0.01, 0, 0, 0, 0, 0, 0)
        s0 = list(start)
        t0 = list(target)
        jh.joint_waypoints(start, target)
        self.assertEqual(start, s0)
        self.assertEqual(target, t0)

    def test_joint_waypoints_reject_bad_inputs(self):
        ok = vec7(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(jh.HomeError):
            jh.joint_waypoints([True] + [0.0] * 6, ok)
        with self.assertRaises(jh.HomeError):
            jh.joint_waypoints([float("nan")] + [0.0] * 6, ok)
        with self.assertRaises(jh.HomeError):
            jh.joint_waypoints([float("inf")] + [0.0] * 6, ok)
        with self.assertRaises(jh.HomeError):
            jh.joint_waypoints([0.0] * 6, ok)
        with self.assertRaises(jh.HomeError):
            jh.joint_waypoints(ok, ok, step_rad=0)
        with self.assertRaises(jh.HomeError):
            jh.joint_waypoints(ok, ok, step_rad=-1)
        with self.assertRaises(jh.HomeError):
            jh.joint_waypoints(ok, ok, step_rad=True)

    def test_home_metrics_ready_on_exact(self):
        st = make_state()
        ref = make_reference(st)
        m = jh.home_metrics(ref, st)
        self.assertTrue(m["ready"])
        self.assertEqual(m["arm_error_rad"], [0.0] * 7)
        self.assertEqual(m["finger_error_m"], [0.0] * 2)

    def test_home_metrics_each_arm_position_axis_affects_ready(self):
        for axis in range(7):
            st = make_state(arm_qpos=[0.0] * 7)
            st["arm_qpos"][axis] = jh.JOINT_TOL_RAD + 0.001
            ref = make_reference()
            m = jh.home_metrics(ref, st)
            self.assertFalse(m["ready"], "axis %d" % axis)

    def test_home_metrics_each_finger_error_affects_ready(self):
        for axis in range(2):
            st = make_state(finger_qpos=[0.0, 0.0])
            st["finger_qpos"][axis] = jh.FINGER_TOL_M + 0.001
            ref = make_reference()
            m = jh.home_metrics(ref, st)
            self.assertFalse(m["ready"], "finger %d" % axis)

    def test_home_metrics_arm_speed_gate_each_axis(self):
        for axis in range(7):
            st = make_state(arm_qvel=[0.0] * 7)
            st["arm_qvel"][axis] = jh.SPEED_TOL_RAD_S + 0.001
            ref = make_reference()
            self.assertFalse(jh.home_metrics(ref, st)["ready"])

    def test_home_metrics_finger_speed_gate_each_axis(self):
        for axis in range(2):
            st = make_state(finger_qvel=[0.0, 0.0])
            st["finger_qvel"][axis] = jh.SPEED_TOL_RAD_S + 0.001
            ref = make_reference()
            self.assertFalse(jh.home_metrics(ref, st)["ready"])

    def test_home_metrics_boundaries_accepted_and_beyond_fail(self):
        st = make_state(arm_qpos=[jh.JOINT_TOL_RAD] * 7, arm_qvel=[jh.SPEED_TOL_RAD_S] * 7, finger_qpos=[jh.FINGER_TOL_M] * 2, finger_qvel=[jh.SPEED_TOL_RAD_S] * 2)
        ref = make_reference()
        self.assertTrue(jh.home_metrics(ref, st)["ready"])
        st2 = make_state(arm_qpos=[jh.JOINT_TOL_RAD + 1e-6] * 7)
        self.assertFalse(jh.home_metrics(ref, st2)["ready"])

    def test_home_metrics_identity_mismatch_raises(self):
        st = make_state()
        ref = make_reference(st)
        ref["joint_names"] = ["other"] + list(ref["joint_names"][1:])
        with self.assertRaises(jh.HomeError):
            jh.home_metrics(ref, st)
        ref = make_reference(st)
        ref["joint_limits_high"] = [2.0] * 6 + [3.0]
        with self.assertRaises(jh.HomeError):
            jh.home_metrics(ref, st)

    def test_read_robot_state_extracts_arm_and_fingers(self):
        qpos = [float(i) for i in range(12)]
        qvel = [float(100 + i) for i in range(12)]
        env = make_env(qpos=qpos, qvel=qvel)
        state = jh.read_robot_state(env)
        self.assertEqual(state["arm_qpos"], [float(i) for i in range(7)])
        self.assertEqual(state["finger_qpos"], [7.0, 8.0])
        self.assertEqual(state["arm_qvel"], [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        self.assertEqual(state["finger_qvel"], [107.0, 108.0])
        self.assertEqual(state["joint_names"], ["j%d" % i for i in range(7)])
        self.assertEqual(state["joint_indexes"], list(range(7)))
        self.assertEqual(state["joint_limits_low"], [-2.0] * 7)
        self.assertEqual(state["joint_limits_high"], [2.0] * 7)

    def test_read_robot_state_indexes_only_on_controller(self):
        env = make_env()
        robot = env.robots[0]
        self.assertFalse(hasattr(robot, "joint_index"))
        self.assertTrue(hasattr(robot.controller, "joint_index"))
        state = jh.read_robot_state(env)
        self.assertEqual(state["arm_qpos"], env.robots[0].sim.data.qpos[:7].tolist())
        self.assertEqual(state["qpos_indexes"], list(range(7)))

    def test_read_robot_state_rejects_bad_counts_and_nonfinite(self):
        env = make_env()
        env.robots[0].controller.joint_index = [0, 1, 2]
        with self.assertRaises(jh.HomeError):
            jh.read_robot_state(env)
        env = make_env(qpos=[float("nan")] + [float(i) for i in range(11)])
        with self.assertRaises(jh.HomeError):
            jh.read_robot_state(env)

    def test_screen_joint_path_unchanged_and_collisions(self):
        qpos = [0.1 * i for i in range(12)]
        qvel = [0.0] * 12
        env = make_env(qpos=qpos, qvel=qvel)
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        ref["joint_limits_low"] = [-2.0] * 7
        ref["joint_limits_high"] = [2.0] * 7
        contacts = [SimpleNamespace(geom1=0, geom2=0, dist=-0.01), SimpleNamespace(geom1=999, geom2=999, dist=-1.0)]

        class Scratch:
            def __init__(self, model):
                self.qpos = np.zeros(12)
                self.qvel = np.zeros(12)
                self.time = 0.0
                self.contact = contacts
                self.ncon = 0

        fake_mj = types.ModuleType("mujoco")
        fake_mj.MjData = Scratch

        def forward(model, data):
            data.ncon = 1

        fake_mj.mj_forward = forward
        with mock.patch.dict(sys.modules, {"mujoco": fake_mj}):
            before = (np.array(env.robots[0].sim.data.qpos, copy=True), np.array(env.robots[0].sim.data.qvel, copy=True), env.robots[0].sim.data.time)
            res = jh.screen_joint_path(env, ref, [list(state["arm_qpos"])])
            after = (np.array(env.robots[0].sim.data.qpos, copy=True), np.array(env.robots[0].sim.data.qvel, copy=True), env.robots[0].sim.data.time)
        self.assertTrue(res["live_unchanged"])
        self.assertEqual(res["collision_count"], 2)
        self.assertEqual(res["sample_count"], 2)
        self.assertEqual(res["planned_samples"], 2)
        self.assertFalse(res["ok"])
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2], after[2])

    def test_screen_joint_path_ignores_object_object_contact(self):
        qpos = [0.1 * i for i in range(12)]
        qvel = [0.0] * 12
        env = make_env(qpos=qpos, qvel=qvel)
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        contacts = [SimpleNamespace(geom1=2, geom2=3, dist=-0.5)]

        class Scratch:
            def __init__(self, model):
                self.qpos = np.zeros(12)
                self.qvel = np.zeros(12)
                self.time = 0.0
                self.contact = contacts
                self.ncon = 1

        fake_mj = types.ModuleType("mujoco")
        fake_mj.MjData = Scratch
        fake_mj.mj_forward = lambda m, d: None
        with mock.patch.dict(sys.modules, {"mujoco": fake_mj}):
            res = jh.screen_joint_path(env, ref, [list(state["arm_qpos"])])
        self.assertEqual(res["collision_count"], 0)
        self.assertTrue(res["ok"])

    def test_adapter_set_goal_layout(self):
        qpos = [0.0] * 12
        env = make_env(qpos=qpos)
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        fake_jpc = mock.MagicMock()
        fake_jpc.set_goal = mock.MagicMock()
        fake_jpc.run_controller = mock.MagicMock(return_value=[0.0] * 7)
        fake_jpc.update = mock.MagicMock()
        fake_cls = mock.MagicMock(return_value=fake_jpc)
        fake_mod = types.ModuleType("robosuite.controllers")
        fake_mod.JointPositionController = fake_cls
        fake_rob = types.ModuleType("robosuite")
        fake_rob.controllers = fake_mod
        with mock.patch.dict(sys.modules, {"robosuite": fake_rob, "robosuite.controllers": fake_mod}):
            adapter = jh.JointHomeAdapter(env.robots[0].controller, env, ref)
        self.assertEqual(adapter.control_dim, 6)
        adapter.select_target([0.1] * 7)
        adapter.set_goal([0.0] * 6)
        fake_jpc.set_goal.assert_called_once()
        args, kwargs = fake_jpc.set_goal.call_args
        self.assertEqual(list(args[0]), [0.0] * 7)
        np.testing.assert_array_equal(kwargs["set_qpos"], np.asarray([0.1] * 7, dtype=float))

    def test_adapter_control_dim_and_run_controller(self):
        env = make_env()
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        fake_jpc = mock.MagicMock()
        fake_jpc.run_controller.return_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        fake_cls = mock.MagicMock(return_value=fake_jpc)
        fake_mod = types.ModuleType("robosuite.controllers")
        fake_mod.JointPositionController = fake_cls
        fake_rob = types.ModuleType("robosuite")
        fake_rob.controllers = fake_mod
        with mock.patch.dict(sys.modules, {"robosuite": fake_rob, "robosuite.controllers": fake_mod}):
            adapter = jh.JointHomeAdapter(env.robots[0].controller, env, ref)
        self.assertEqual(adapter.control_dim, 6)
        result = adapter.run_controller()
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        adapter.run_controller()
        env.robots[0].controller.update.assert_called()

    def test_adapter_control_dim_and_getattr_passthrough(self):
        env = make_env()
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        original = mock.MagicMock()
        original.eef_name = "eef"
        original.joint_index = list(range(7))
        original.qpos_index = list(range(7))
        original.qvel_index = list(range(7))
        env.robots[0].controller = original
        fake_jpc = mock.MagicMock()
        fake_cls = mock.MagicMock(return_value=fake_jpc)
        fake_mod = types.ModuleType("robosuite.controllers")
        fake_mod.JointPositionController = fake_cls
        fake_rob = types.ModuleType("robosuite")
        fake_rob.controllers = fake_mod
        with mock.patch.dict(sys.modules, {"robosuite": fake_rob, "robosuite.controllers": fake_mod}):
            adapter = jh.JointHomeAdapter(original, env, ref)
        self.assertEqual(adapter.control_dim, 6)
        self.assertEqual(adapter.eef_name, "eef")

    def test_adapter_set_goal_rejects_invalid_shape(self):
        env = make_env()
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        fake_jpc = mock.MagicMock()
        fake_cls = mock.MagicMock(return_value=fake_jpc)
        fake_mod = types.ModuleType("robosuite.controllers")
        fake_mod.JointPositionController = fake_cls
        fake_rob = types.ModuleType("robosuite")
        fake_rob.controllers = fake_mod
        with mock.patch.dict(sys.modules, {"robosuite": fake_rob, "robosuite.controllers": fake_mod}):
            adapter = jh.JointHomeAdapter(env.robots[0].controller, env, ref)
        with self.assertRaises(jh.HomeError):
            adapter.set_goal([0.0] * 5)
        with self.assertRaises(jh.HomeError):
            adapter.run_controller = lambda: None
            adapter.set_goal([float("nan")] * 6)

    def test_adapter_verify_qpos_limits_shape(self):
        env = make_env()
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        captured = {}

        def ctor(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        fake_mod = types.ModuleType("robosuite.controllers")
        fake_mod.JointPositionController = ctor
        fake_rob = types.ModuleType("robosuite")
        fake_rob.controllers = fake_mod
        with mock.patch.dict(sys.modules, {"robosuite": fake_rob, "robosuite.controllers": fake_mod}):
            jh.JointHomeAdapter(env.robots[0].controller, env, ref)
        self.assertEqual(captured["qpos_limits"].shape, (2, 7))

    def test_adapter_original_mismatch_rejected(self):
        env = make_env()
        state = jh.read_robot_state(env)
        ref = make_reference(state)
        other = SimpleNamespace(eef_name="eef")
        fake_jpc = mock.MagicMock()
        fake_cls = mock.MagicMock(return_value=fake_jpc)
        fake_mod = types.ModuleType("robosuite.controllers")
        fake_mod.JointPositionController = fake_cls
        fake_rob = types.ModuleType("robosuite")
        fake_rob.controllers = fake_mod
        with mock.patch.dict(sys.modules, {"robosuite": fake_rob, "robosuite.controllers": fake_mod}):
            with self.assertRaises(jh.HomeError):
                jh.JointHomeAdapter(other, env, ref)


if __name__ == "__main__":
    unittest.main()
