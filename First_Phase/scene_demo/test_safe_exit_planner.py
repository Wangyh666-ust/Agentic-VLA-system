import sys
import struct
import types

import numpy as np
import pytest

from scene_demo import safe_exit_planner as sp
from scene_demo import safe_exit_contacts as sc
from scene_demo import joint_home as jh


class _FakeModel:
    def __init__(self):
        self.nv = 11
        self._model = types.SimpleNamespace(nv=11)

    def site_name2id(self, name):
        return 0


class _FakeData:
    def __init__(self, qpos, qvel, time):
        self.qpos = np.array(qpos, dtype=float)
        self.qvel = np.array(qvel, dtype=float)
        self.time = float(time)
        self.site_xpos = np.zeros((1, 3))
        self.site_xmat = np.zeros((1, 9))


class _FakeSim:
    def __init__(self, model, data):
        self.model = model
        self.data = data


class _FakeController:
    eef_name = "eef_site"


class _FakeRobot:
    def __init__(self, sim):
        self.sim = sim
        self.controller = _FakeController()


class _FakeEnv:
    def __init__(self, robot):
        self.robot = robot


def test_plan_safe_exit_simulated(monkeypatch):
    mujoco = types.ModuleType("mujoco")

    scratch_records = []

    class MjData:
        def __init__(self, model):
            self.qpos = np.zeros(11)
            self.qvel = np.zeros(11)
            self.time = 0.0
            self.site_xpos = np.zeros((1, 3))
            self.site_xmat = np.zeros((1, 9))
            scratch_records.append(self)

    def mj_forward(model, data):
        data.site_xpos[0] = np.array(data.qpos[:3], dtype=float)
        data.site_xmat[0] = np.eye(3).reshape(9)

    def mj_jacSite(model, data, jacp, jacr, site_id):
        jacp[:] = 0.0
        jacr[:] = 0.0
        jacp[:3, :3] = np.eye(3)
        jacr[:3, 3:6] = np.eye(3)

    mujoco.MjData = MjData
    mujoco.mj_forward = mj_forward
    mujoco.mj_jacSite = mj_jacSite
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)

    pd_mod = types.ModuleType("scene_demo.preparation_diagnostics")

    def _matrix_to_rotvec(R):
        return np.zeros(3)

    pd_mod._matrix_to_rotvec = _matrix_to_rotvec
    monkeypatch.setitem(sys.modules, "scene_demo.preparation_diagnostics", pd_mod)
    monkeypatch.setattr(sys.modules["scene_demo"], "preparation_diagnostics", pd_mod, raising=False)

    live_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.03, 0.7, 0.8])
    live_qvel = np.full(11, 0.01)
    live_time = 1.23

    model = _FakeModel()
    live_data = _FakeData(live_qpos, live_qvel, live_time)
    sim = _FakeSim(model, live_data)
    robot = _FakeRobot(sim)
    env = _FakeEnv(robot)

    qpos0_bytes = live_data.qpos.tobytes()
    qvel0_bytes = live_data.qvel.tobytes()
    time0_bytes = struct.pack("<d", float(live_data.time))

    monkeypatch.setattr(jh, "_single_robot", lambda e: e.robot)

    state_dict = {
        "arm_qpos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "finger_qpos": [0.02, -0.03],
        "qpos_indexes": [0, 1, 2, 3, 4, 5, 6],
        "qvel_indexes": [0, 1, 2, 3, 4, 5, 6],
        "finger_qpos_indexes": [7, 8],
        "joint_limits_low": [-3.0] * 7,
        "joint_limits_high": [3.0] * 7,
    }
    monkeypatch.setattr(jh, "read_robot_state", lambda e: state_dict)

    ref = {
        "homeq": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "finger_home": [0.04, -0.04],
    }

    def _reference_identity(reference, state):
        assert reference is ref
        assert state is state_dict

    monkeypatch.setattr(jh, "_reference_identity", _reference_identity)

    contacts_calls = []

    def contacts(e, data=None):
        d = data if data is not None else live_data
        qpos_copy = np.array(d.qpos, copy=True)
        contacts_calls.append(qpos_copy)
        if d.qpos[0] < 0.0005:
            return [
                {
                    "geom1": 1,
                    "geom2": 2,
                    "pair": [1, 2],
                    "geom1_name": "g1",
                    "geom2_name": "g2",
                    "dist": -0.0001,
                    "pos": [0.0, 0.0, 0.0],
                    "normal": [1.0, 0.0, 0.0],
                    "outward_normal": [1.0, 0.0, 0.0],
                    "self": False,
                }
            ]
        return []

    monkeypatch.setattr(sc, "contacts", contacts)

    result = sp.plan_safe_exit(env, ref)

    assert result["ok"] is True
    assert result["needed"] is True
    assert result["home_nominal_clear"] is True
    assert result["live_unchanged"] is True
    assert result["reason"] == "planned"
    assert result["direction"] == [1.0, 0.0, 0.0]
    assert result["distance"] == pytest.approx(0.02)
    assert len(result["waypoints"]) == 20
    assert result["waypoints"][-1][0] == pytest.approx(0.02, abs=1e-4)

    assert live_data.qpos.tobytes() == qpos0_bytes
    assert live_data.qvel.tobytes() == qvel0_bytes
    assert struct.pack("<d", float(live_data.time)) == time0_bytes

    assert len(contacts_calls) > 0
    for q in contacts_calls:
        assert list(q[9:11]) == [0.7, 0.8]

    seen_fingers = set()
    seen_home_fingers = set()
    for q in contacts_calls:
        seen_fingers.add((round(float(q[7]), 6), round(float(q[8]), 6)))
    for q in contacts_calls:
        seen_home_fingers.add((round(float(q[7]), 6), round(float(q[8]), 6)))

    assert (0.02, -0.03) in seen_fingers
    assert (0.04, -0.04) in seen_home_fingers
    assert "scene_demo.service" not in sys.modules
