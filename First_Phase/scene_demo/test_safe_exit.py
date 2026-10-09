import sys
import types

import numpy as np
import pytest

from scene_demo import safe_exit as s
from scene_demo import safe_exit_contacts as c
from scene_demo import joint_home as jh


class FakeController:
    def __init__(self):
        self.update_calls = 0
        self.reset_goal_calls = 0

    def update(self, force=True):
        self.update_calls += 1

    def reset_goal(self):
        self.reset_goal_calls += 1


class FakeGripper:
    def __init__(self):
        self.current_action = np.zeros(2)


class FakeAdapter:
    def __init__(self):
        self.target = None
        self.target_records = []

    def select_target(self, q7):
        self.target = np.array(q7, dtype=float).copy()
        self.target_records.append(self.target.copy())


class FakeEnv:
    def __init__(self, num_steps=10):
        self.num_steps = num_steps
        self.steps = 0
        self.actions = []
        self.robot = types.SimpleNamespace(
            controller=FakeController(),
            gripper=FakeGripper(),
        )
        self.adapter = FakeAdapter()
        self.q = np.zeros(7)
        self.ready = False
        self._last_obs = {"ok": True}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        assert action.shape == (7,), f"action shape {action.shape}"
        assert np.allclose(action, 0.0), f"action not zero: {action}"
        assert self.robot.controller is self.adapter, "controller/adapter mismatch"
        if self.adapter.target is not None:
            self.q = self.adapter.target.copy()
        self.actions.append(np.array(action, dtype=float))
        self.steps += 1
        return self._last_obs, 0.0, False, {}


def _make_contacts_dict(rows):
    return {
        "geom1": rows["geom1"],
        "geom2": rows["geom2"],
        "pair": rows["pair"],
        "geom1_name": rows["geom1_name"],
        "geom2_name": rows["geom2_name"],
        "dist": rows["dist"],
        "pos": np.asarray(rows["pos"], dtype=float),
        "normal": np.asarray(rows["normal"], dtype=float),
        "outward_normal": np.asarray(rows["outward_normal"], dtype=float),
        "self": rows["self"],
    }


def _initial_contact(dist=-0.0001):
    return _make_contacts_dict(
        {
            "geom1": 1,
            "geom2": 2,
            "pair": (1, 2),
            "geom1_name": "g1",
            "geom2_name": "g2",
            "dist": dist,
            "pos": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "outward_normal": [0.0, 0.0, 1.0],
            "self": False,
        }
    )


def _reference():
    return {
        "home_q7": np.array([0.0] * 7, dtype=float),
        "finger_home": np.array([0.02, -0.02], dtype=float),
        "normal_indices": np.array([0, 1, 2], dtype=int),
        "limits": np.array([1.0] * 7, dtype=float),
        "contact": _initial_contact(),
    }


def _fake_state():
    return {
        "arm_qpos": np.zeros(7),
        "arm_qvel": np.zeros(7),
        "finger_qpos": np.array([0.02, -0.02]),
        "finger_qvel": np.zeros(2),
        "finger_indexes": np.array([7, 8]),
        "arm_indexes": np.array([0, 1, 2, 3, 4, 5, 6]),
        "joint_limits": np.array([1.0] * 7),
    }


@pytest.fixture(autouse=True)
def _patch_single_robot(monkeypatch):
    monkeypatch.setattr(jh, "_single_robot", lambda env: env.robot)
    yield


@pytest.fixture
def fake_pd(monkeypatch):
    fake = types.ModuleType("scene_demo.preparation_diagnostics")

    def controller_facts(env):
        return {
            "ok": True,
            "base_protection": True,
        }

    def controller_mismatch(facts):
        return None

    fake.controller_facts = controller_facts
    fake.controller_mismatch = controller_mismatch
    monkeypatch.setitem(sys.modules, "scene_demo.preparation_diagnostics", fake)
    pkg = sys.modules.get("scene_demo")
    if pkg is not None:
        monkeypatch.setattr(pkg, "preparation_diagnostics", fake, raising=False)
    return fake


def _install_joint_home(monkeypatch, state=None):
    state = state if state is not None else _fake_state()
    adapter = FakeAdapter()

    def read_robot_state(env):
        snap = {
            k: (np.array(v, dtype=float).copy() if isinstance(v, np.ndarray) else v)
            for k, v in state.items()
        }
        snap["arm_qpos"] = env.q.copy()
        return snap

    def adapter_factory(original, env, reference):
        env.adapter = adapter
        return adapter

    monkeypatch.setattr(jh, "read_robot_state", read_robot_state)
    monkeypatch.setattr(jh, "JointHomeAdapter", adapter_factory)
    monkeypatch.setattr(s, "_finger_hold", lambda env, state: np.array([0.0, -0.5]))
    return adapter


def _install_clean_contacts(monkeypatch):
    rows = [_initial_contact()]

    def read_contacts(env):
        if env.steps >= 1:
            return []
        return [dict(r) for r in rows]

    def contacts(env, data=None):
        if env.steps >= 1:
            return []
        return [dict(r) for r in rows]

    monkeypatch.setattr(s, "read_contacts", read_contacts)
    monkeypatch.setattr(c, "contacts", contacts)
    return rows


def _plan(contact):
    return {
        "ok": True,
        "needed": True,
        "initial_contacts": [dict(contact)],
        "waypoints": [[0.1] * 7],
        "candidate_reports": [],
        "home_nominal_clear": True,
        "live_unchanged": True,
    }


def _guard_ok():
    return {"ok": True, "base_protection": True}


def _emit_collector(rows):
    def emit(row):
        rows.append(row)

    return emit


def test_contact_check_new_pair_rejected():
    initial = _initial_contact(dist=-0.0001)
    previous = _initial_contact(dist=-0.0001)
    current = _make_contacts_dict(
        {
            "geom1": 3,
            "geom2": 4,
            "pair": (3, 4),
            "geom1_name": "g3",
            "geom2_name": "g4",
            "dist": -0.0001,
            "pos": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "outward_normal": [0.0, 0.0, 1.0],
            "self": False,
        }
    )
    init_map = c.pair_map([initial])
    prev_map = c.pair_map([previous])
    curr_map = c.pair_map([current])
    result = c.contact_check(init_map, prev_map, curr_map, [])
    assert result["ok"] is False


def test_contact_check_deeper_initial_rejected():
    initial = {(1, 2): -0.001}
    previous = {(1, 2): -0.0001}
    current = {(1, 2): -0.002}
    result = c.contact_check(initial, previous, current, [])
    assert result["ok"] is False


def test_contact_check_deeper_previous_rejected():
    initial = {(1, 2): -0.001}
    previous = {(1, 2): -0.0001}
    current = {(1, 2): -0.0002}
    result = c.contact_check(initial, previous, current, [])
    assert result["ok"] is False


def test_contact_check_disappearance_new_cleared_reusable():
    initial = {(1, 2): -0.001}
    previous = {(1, 2): -0.0001}
    current = {}
    result = c.contact_check(initial, previous, current, [])
    assert result["ok"] is True
    assert isinstance(result["newcleared"], (list, tuple))
    cleared = result["newcleared"]
    reusable = result["newcleared"]
    assert reusable == cleared
    result2 = c.contact_check(initial, current, current, cleared)
    assert result2["ok"] is True


def test_contact_check_recontact_rejected():
    initial = {(1, 2): -0.001}
    previous = {}
    current = {(1, 2): -0.0001}
    cleared = {(1, 2)}
    result = c.contact_check(initial, previous, current, cleared)
    assert result["ok"] is False


def test_contact_check_inputs_unchanged():
    initial = {(1, 2): -0.001}
    previous = {(1, 2): -0.0001}
    current = {(1, 2): -0.0002}
    cleared = set()
    initial_copy = dict(initial)
    previous_copy = dict(previous)
    current_copy = dict(current)
    cleared_copy = set(cleared)
    c.contact_check(initial, previous, current, cleared)
    assert initial == initial_copy
    assert previous == previous_copy
    assert current == current_copy
    assert cleared == cleared_copy


def _direct_model(frame, geom1=1, geom2=2, dist=-0.001):
    class FakeModel:
        ngeom = 5

        def geom_name2id(self, name):
            return {"g1": 1, "g2": 2, "g3": 3}[name]

        def geom_id2name(self, gid):
            return {1: "g1", 2: "g2", 3: "g3"}[gid]

    class FakeSim:
        def __init__(self):
            self.model = FakeModel()
            self.data = types.SimpleNamespace(
                ncon=1,
                contact=[
                    types.SimpleNamespace(
                        geom1=geom1,
                        geom2=geom2,
                        dist=dist,
                        pos=np.zeros(3),
                        frame=frame,
                    )
                ],
            )

    class FakeRobot:
        def __init__(self):
            self.sim = FakeSim()
            self.robot_model = types.SimpleNamespace(contact_geoms=["g1"])
            self.gripper = types.SimpleNamespace(contact_geoms=[])

    return types.SimpleNamespace(robot=FakeRobot())


def test_contacts_direct_fake_model():
    frame = np.eye(3).astype(float).reshape(9)
    env = _direct_model(frame, geom1=1, geom2=2, dist=-0.001)
    rows = c.contacts(env)
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert isinstance(rows[0], dict)
    assert rows[0]["outward_normal"] == [-1.0, 0.0, 0.0]


def test_contacts_reverse_pair_normal_sign():
    frame = np.eye(3).astype(float).reshape(9)
    env = _direct_model(frame, geom1=2, geom2=1)
    rows = c.contacts(env)
    row = rows[0]
    assert row["geom1"] == 2
    assert row["geom2"] == 1
    assert row["outward_normal"] == [1.0, 0.0, 0.0]


def test_contacts_self_true_for_two_robot_ids():
    class FakeModel:
        ngeom = 5

        def geom_name2id(self, name):
            return {"g1": 1, "g2": 2}[name]

        def geom_id2name(self, gid):
            return {1: "g1", 2: "g2"}[gid]

    class FakeSim:
        def __init__(self):
            self.model = FakeModel()
            self.data = types.SimpleNamespace(
                ncon=1,
                contact=[
                    types.SimpleNamespace(
                        geom1=1,
                        geom2=2,
                        dist=-0.001,
                        pos=np.zeros(3),
                        frame=np.eye(3).astype(float).reshape(9),
                    )
                ],
            )

    class FakeRobot:
        def __init__(self):
            self.sim = FakeSim()
            self.robot_model = types.SimpleNamespace(contact_geoms=["g1", "g2"])
            self.gripper = types.SimpleNamespace(contact_geoms=[])

    env = types.SimpleNamespace(robot=FakeRobot())
    rows = c.contacts(env)
    assert rows
    assert rows[0]["self"] is True
    assert rows[0]["geom1_name"] == "g1"
    assert rows[0]["geom2_name"] == "g2"


def test_contacts_zero_normal_raises():
    frame = np.zeros(9)
    env = _direct_model(frame)
    with pytest.raises(jh.HomeError, match="zero"):
        c.contacts(env)


def test_contacts_positive_distance_filtered():
    frame = np.eye(3).astype(float).reshape(9)
    env = _direct_model(frame, dist=0.001)
    rows = c.contacts(env)
    assert rows == []


def test_contacts_nan_normal_raises():
    frame = np.eye(3).astype(float).reshape(9)
    frame[0] = np.nan
    env = _direct_model(frame)
    with pytest.raises(jh.HomeError, match="finite"):
        c.contacts(env)


def test_run_zero_action_plan_not_ok(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: {
        "ok": False,
        "needed": True,
        "reason": "no_safe_exit_candidate",
        "initial_contacts": [],
        "waypoints": [[0.1] * 7],
        "candidate_reports": [],
        "home_nominal_clear": True,
        "live_unchanged": True,
    })
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is True
    assert result["ready"] is False
    assert len(env.actions) == 0
    assert env.steps == 0


def test_run_zero_action_preguard_false(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))

    def guard_false():
        return {"ok": False, "base_protection": True}

    rows = []
    result = s.run_safe_exit(env, ref, guard_false, _emit_collector(rows))
    assert result["ok"] is True
    assert result["ready"] is False
    assert len(env.actions) == 0
    assert env.steps == 0


def test_run_zero_action_stop_requested(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))
    rows = []
    result = s.run_safe_exit(
        env, ref, _guard_ok, _emit_collector(rows), stop_requested=lambda: True
    )
    assert result["ok"] is True
    assert result["ready"] is False
    assert len(env.actions) == 0
    assert env.steps == 0


def test_run_exit_cap_four(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))
    monkeypatch.setattr(s, "EXIT_CAP", 4)
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is True
    assert result["ready"] is False
    assert result["confirmed_samples"] <= 4
    assert len(env.actions) == 4
    assert env.steps == 4
    assert env.robot.controller.update_calls >= 1
    assert env.robot.controller.reset_goal_calls >= 1


def test_run_guard_false_at_step_five(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    rows = [_initial_contact()]
    call_count = {"n": 0}

    def read_contacts(env):
        return [dict(r) for r in rows]

    monkeypatch.setattr(s, "read_contacts", read_contacts)
    monkeypatch.setattr(c, "contacts", lambda env, data=None: [dict(r) for r in rows])
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))

    def guard():
        call_count["n"] += 1
        if env.steps == 5:
            return {"ok": False, "base_protection": True}
        return {"ok": True, "base_protection": True}

    emit_rows = []
    result = s.run_safe_exit(env, ref, guard, _emit_collector(emit_rows))
    assert result["ok"] is True
    assert result["ready"] is False
    assert len(env.actions) == 5
    assert env.steps == 5
    assert result["confirmed_samples"] == 0
    assert len(emit_rows) == 5


def test_run_single_final_target_success(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    adapter = _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is True
    assert result["ready"] is True
    assert len(env.actions) == 5
    assert env.steps == 5
    assert result["confirmed_samples"] == 5
    assert env.robot.controller.update_calls >= 1
    assert env.robot.controller.reset_goal_calls >= 1
    assert len(adapter.target_records) >= 5
    assert all(rec.tolist() == [0.1] * 7 for rec in adapter.target_records)


def test_run_two_waypoints_streak(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    adapter = _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    plan = _plan(ref["contact"])
    plan["waypoints"] = [[0.1] * 7, [0.2] * 7]
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: plan)
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is True
    assert result["ready"] is True
    assert len(env.actions) == 6
    assert len(rows) == 6
    assert adapter.target_records[0].tolist() == [0.1] * 7
    assert adapter.target_records[-1].tolist() == [0.2] * 7
    streaks = [row["streak"] for row in rows]
    assert streaks[0] == 0
    assert streaks[1:] == [1, 2, 3, 4, 5]
    first_sent = np.asarray(rows[0]["sent_action"], dtype=float)
    assert np.allclose(first_sent, 0.0)
    second_sent = np.asarray(rows[1]["sent_action"], dtype=float)
    assert np.allclose(second_sent, 0.0)
    first_before = np.asarray(rows[0]["before_robot_state"]["arm_qpos"], dtype=float)
    assert np.allclose(first_before, 0.0)
    for i, row in enumerate(rows):
        after_arm = np.asarray(row["after_robot_state"]["arm_qpos"], dtype=float)
        assert np.allclose(after_arm, 0.1 if i == 0 else 0.2)
        before_arm = np.asarray(row["before_robot_state"]["arm_qpos"], dtype=float)
        assert np.allclose(before_arm, 0.0 if i == 0 else (0.1 if i == 1 else 0.2))
    assert np.allclose(np.asarray(rows[0]["before_robot_state"]["arm_qpos"], dtype=float), 0.0)
    assert np.allclose(np.asarray(rows[0]["after_robot_state"]["arm_qpos"], dtype=float), 0.1)


def test_run_expected_row_fields(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    plan = _plan(ref["contact"])
    plan["waypoints"] = [[0.1] * 7, [0.2] * 7]
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: plan)
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is True
    assert len(rows) >= 6
    first = rows[0]
    for key in (
        "before_robot_state",
        "after_robot_state",
        "streak",
        "step",
        "qtarget",
        "sent_action",
        "guard",
        "contact_check",
        "before_contacts",
        "after_contacts",
    ):
        assert key in first, key
    sent = np.asarray(first["sent_action"], dtype=float)
    assert sent.shape == (7,)


def test_run_step_first_throws(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))

    def step_throw(action):
        raise RuntimeError("boom")

    env.step = step_throw
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is False
    assert result["ready"] is False
    assert len(env.actions) == 0


def test_run_emit_throws(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))

    def emit_throw(row):
        raise RuntimeError("emit boom")

    result = s.run_safe_exit(env, ref, _guard_ok, emit_throw)
    assert result["ok"] is False
    assert result["ready"] is False
    assert len(env.actions) == 1


def test_run_restore_throws(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    _install_clean_contacts(monkeypatch)
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(ref["contact"]))

    def reset_goal_throw():
        raise RuntimeError("restore boom")

    env.robot.controller.reset_goal = reset_goal_throw
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is False
    assert result["ready"] is False
    assert len(env.actions) == 5


def test_run_new_contact_first_step_stops(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    initial = _initial_contact()
    new_contact = _make_contacts_dict(
        {
            "geom1": 3,
            "geom2": 4,
            "pair": (3, 4),
            "geom1_name": "g3",
            "geom2_name": "g4",
            "dist": -0.0001,
            "pos": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "outward_normal": [0.0, 0.0, 1.0],
            "self": False,
        }
    )
    state = {"phase": "initial"}

    def read_contacts(env):
        if state["phase"] == "initial":
            return [dict(initial)]
        return [dict(new_contact)]

    monkeypatch.setattr(s, "read_contacts", read_contacts)
    monkeypatch.setattr(c, "contacts", lambda env, data=None: [dict(new_contact)])
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(initial))

    original_step = env.step

    def step(action):
        state["phase"] = "after_first"
        return original_step(action)

    env.step = step

    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is True
    assert result["ready"] is False
    assert len(env.actions) == 1


def test_run_recontact_second_step_stops(monkeypatch, fake_pd):
    env = FakeEnv()
    ref = _reference()
    _install_joint_home(monkeypatch)
    initial = _initial_contact()
    same_pair_later = _initial_contact(dist=-0.0001)
    state = {"step": 0}

    def read_contacts(env):
        if state["step"] == 0:
            return [dict(initial)]
        if state["step"] == 1:
            return []
        return [dict(same_pair_later)]

    monkeypatch.setattr(s, "read_contacts", read_contacts)
    monkeypatch.setattr(c, "contacts", lambda env, data=None: [])
    monkeypatch.setattr(s, "plan_safe_exit", lambda env, ref: _plan(initial))

    original_step = env.step

    def step(action):
        state["step"] += 1
        return original_step(action)

    env.step = step
    rows = []
    result = s.run_safe_exit(env, ref, _guard_ok, _emit_collector(rows))
    assert result["ok"] is True
    assert result["ready"] is False
    assert len(env.actions) == 2


@pytest.fixture
def finger_hold_env(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.nu = 10
            self.njnt = 9
            self.actuator_trntype = np.zeros(10, dtype=int)
            self.actuator_trnid = np.zeros((10, 2), dtype=int)
            self.actuator_trnid[8] = [7, 0]
            self.actuator_trnid[9] = [8, 0]
            self.jnt_qposadr = np.zeros(9, dtype=int)
            self.jnt_qposadr[7] = 7
            self.jnt_qposadr[8] = 8
            self.actuator_ctrlrange = np.tile(
                np.array([0.0, 0.04]), (10, 1)
            )
            self.actuator_ctrlrange[8] = np.array([0.0, 0.04])
            self.actuator_ctrlrange[9] = np.array([-0.04, 0.0])
            self.actuator_gear = np.zeros((10, 6))
            self.actuator_gear[:, 0] = 1.0
            self.actuator_gaintype = np.zeros(10, dtype=int)
            self.actuator_biastype = np.ones(10, dtype=int)
            self.actuator_dyntype = np.zeros(10, dtype=int)
            self.actuator_gainprm = np.zeros((10, 3))
            self.actuator_gainprm[:, 0] = 100.0
            self.actuator_biasprm = np.zeros((10, 3))
            self.actuator_biasprm[:, 1] = -100.0
            self.actuator_forcerange = np.tile(
                np.array([-1.0, 1.0]), (10, 1)
            )
            self.actuator_forcelimited = np.zeros(10, dtype=int)

        def geom_name2id(self, name):
            return {"g1": 1, "g2": 2}[name]

        def geom_id2name(self, gid):
            return {1: "g1", 2: "g2"}[gid]

    class FakeSim:
        def __init__(self):
            self.model = FakeModel()

    class FakeRobot:
        def __init__(self):
            self.sim = FakeSim()
            self._ref_joint_gripper_actuator_indexes = [8, 9]
            self.gripper = types.SimpleNamespace(
                current_action=np.array([0.02, -0.03])
            )

    env = types.SimpleNamespace(robot=FakeRobot())
    state = {
        "finger_qpos": np.array([0.02, -0.03]),
        "finger_qpos_indexes": [7, 8],
        "finger_qvel": np.zeros(2),
    }
    return env, state


def test_finger_hold_real(finger_hold_env):
    env, state = finger_hold_env
    result = s._finger_hold(env, state)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.allclose(result[0], 0.0)
    assert np.allclose(result[1], -0.5)


def test_finger_hold_invalid_gear_first(finger_hold_env):
    env, state = finger_hold_env
    env.robot.sim.model.actuator_gear[8, 0] = 2.0
    with pytest.raises(jh.HomeError, match="first"):
        s._finger_hold(env, state)


def test_finger_hold_invalid_rest_gear(finger_hold_env):
    env, state = finger_hold_env
    env.robot.sim.model.actuator_gear[9, 5] = 1.0
    with pytest.raises(jh.HomeError, match="rest"):
        s._finger_hold(env, state)


def test_finger_hold_invalid_dyntype(finger_hold_env):
    env, state = finger_hold_env
    env.robot.sim.model.actuator_dyntype = np.ones(10, dtype=int)
    with pytest.raises(jh.HomeError, match="dyntype"):
        s._finger_hold(env, state)


def test_finger_hold_invalid_gaintype(finger_hold_env):
    env, state = finger_hold_env
    env.robot.sim.model.actuator_gaintype = np.ones(10, dtype=int)
    with pytest.raises(jh.HomeError, match="gaintype"):
        s._finger_hold(env, state)


def test_finger_hold_invalid_biastype(finger_hold_env):
    env, state = finger_hold_env
    env.robot.sim.model.actuator_biastype = np.zeros(10, dtype=int)
    with pytest.raises(jh.HomeError, match="biastype"):
        s._finger_hold(env, state)


def test_finger_hold_invalid_gainprm(finger_hold_env):
    env, state = finger_hold_env
    env.robot.sim.model.actuator_gainprm = np.ones((10, 3))
    with pytest.raises(jh.HomeError, match="biasprm1 must equal -gainprm0"):
        s._finger_hold(env, state)


def test_finger_hold_invalid_biasprm(finger_hold_env):
    env, state = finger_hold_env
    env.robot.sim.model.actuator_biasprm = np.ones((10, 3))
    with pytest.raises(jh.HomeError, match="biasprm1 must equal -gainprm0"):
        s._finger_hold(env, state)


def test_no_service_imports():
    assert "scene_demo.service" not in sys.modules


def test_finger_hold_uses_installed_single_arm_references(finger_hold_env):
    from robosuite.robots.single_arm import SingleArm

    env, state = finger_hold_env
    model = env.robot.sim.model

    joint_qpos = {
        "arm_joint_%d" % i: i for i in range(7)
    }
    joint_qpos.update({
        "finger_joint_0": 7,
        "finger_joint_1": 8,
    })
    joint_qvel = dict(joint_qpos)
    joint_ids = dict(joint_qpos)
    actuator_ids = {
        "arm_act_0": 0,
        "arm_act_1": 1,
        "arm_act_2": 2,
        "arm_act_3": 3,
        "arm_act_4": 4,
        "arm_act_5": 5,
        "arm_act_6": 6,
        "finger_act_0": 8,
        "finger_act_1": 9,
    }
    site_ids = {"grip_site": 0, "grip_cylinder": 1}

    model.get_joint_qpos_addr = lambda name: joint_qpos[name]
    model.get_joint_qvel_addr = lambda name: joint_qvel[name]
    model.joint_name2id = lambda name: joint_ids[name]
    model.actuator_name2id = lambda name: actuator_ids[name]
    model.site_name2id = lambda name: site_ids[name]

    r = object.__new__(SingleArm)
    r.sim = env.robot.sim
    r.robot_model = types.SimpleNamespace(
        joints=["arm_joint_%d" % i for i in range(7)],
        actuators=["arm_act_%d" % i for i in range(7)],
    )
    r.gripper = env.robot.gripper
    r.gripper.joints = ["finger_joint_0", "finger_joint_1"]
    r.gripper.actuators = ["finger_act_0", "finger_act_1"]
    r.gripper.important_sites = {
        "grip_site": "grip_site",
        "grip_cylinder": "grip_cylinder",
    }
    r.has_gripper = True

    SingleArm.setup_references(r)
    assert r._ref_joint_gripper_actuator_indexes == [8, 9]
    assert not hasattr(r, "_ref_gripper_actuator_indexes")

    env.robot = r
    result = s._finger_hold(env, state)
    np.testing.assert_allclose(result, [0.0, -0.5])
