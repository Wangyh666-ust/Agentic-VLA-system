import os, sys, json, types
import numpy as np
import pytest


def test_missing_eef_rejects():
    from sensors import SensorPort

    class RC:
        eef_name = None

    class Robot:
        eef_site_id = None
        controller = RC()
        sim = types.SimpleNamespace(model=types.SimpleNamespace(site_name2id=lambda n: -1))

    p = SensorPort.__new__(SensorPort)
    p._robot = Robot()
    with pytest.raises(RuntimeError):
        SensorPort._site_id(p)


def test_scratch_uses_qpos_indexes(monkeypatch, tmp_path):
    from sensors import SensorPort

    seen = {}

    class Data:
        def __init__(self, raw):
            self.qpos = np.zeros(raw.nq)
            self.qvel = np.zeros(raw.nv)
            self.time = 0.0

    fake_mj = types.ModuleType('mujoco')
    fake_mj.MjData = Data
    fake_mj.mj_forward = lambda raw, d: seen.setdefault('forward', True)
    monkeypatch.setitem(sys.modules, 'mujoco', fake_mj)

    raw = types.SimpleNamespace(nq=20, nv=20)
    sim = types.SimpleNamespace(model=types.SimpleNamespace(_model=raw),
                                data=types.SimpleNamespace(qpos=np.arange(20.0),
                                                           qvel=np.ones(20), time=1.5))
    robot = types.SimpleNamespace(sim=sim)
    p = SensorPort.__new__(SensorPort)
    p._robot = robot
    p.env = types.SimpleNamespace()
    import scene_demo.joint_home as jh
    monkeypatch.setattr(jh, 'read_robot_state', lambda env: {'qpos_indexes': [2, 4, 6, 8, 10, 12, 14]})
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    raw2, data = SensorPort._scratch(p, q)
    assert data.qpos[2] == pytest.approx(0.1)
    assert data.qpos[14] == pytest.approx(0.7)
    assert data.qpos[0] == 0.0
    assert data.qpos[1] == 1.0


def test_start_close_restores(monkeypatch, tmp_path):
    from sensors import SensorPort
    import scene_demo.joint_home as jh
    import scene_demo.safe_exit as safe_exit

    class StubCtrl:
        def update(self, force=False):
            pass
        def reset_goal(self):
            pass

    class StubAdapter:
        def __init__(self, original, env, reference):
            self.original = original

    class StubGrip:
        def __init__(self):
            self.current_action = np.array([0.25, -0.25], dtype=np.float32)

    robot = types.SimpleNamespace(controller=StubCtrl(), gripper=StubGrip())
    monkeypatch.setattr(jh, '_single_robot', lambda env: robot)
    monkeypatch.setattr(jh, 'read_robot_state', lambda env: {'qpos_indexes': [2, 4, 6, 8, 10, 12, 14]})
    monkeypatch.setattr(safe_exit, '_finger_hold', lambda env, state: np.array([1.0, 1.0], dtype=np.float32))
    monkeypatch.setattr(jh, 'JointHomeAdapter', StubAdapter)

    p = SensorPort(types.SimpleNamespace(), {}, str(tmp_path))
    p.start()
    assert np.allclose(robot.gripper.current_action, [1.0, 1.0])
    assert isinstance(robot.controller, StubAdapter)
    p.close()
    assert np.allclose(robot.gripper.current_action, [0.25, -0.25])
    assert isinstance(robot.controller, StubCtrl)


def test_capture_rejects_bad_T(monkeypatch, tmp_path):
    from sensors import SensorPort
    import PIL.Image

    def fake_render(width, height, camera_name, depth):
        return (np.zeros((height, width, 3), np.uint8), np.ones((height, width), np.float32))

    class CU:
        @staticmethod
        def get_real_depth_map(sim, d):
            return np.asarray(d)
        @staticmethod
        def get_camera_intrinsic_matrix(sim, name, w, h):
            return np.eye(3)
        @staticmethod
        def get_camera_extrinsic_matrix(sim, name):
            return np.eye(3)

    monkeypatch.setattr('sensors._cu', lambda: CU)

    sim = types.SimpleNamespace(render=fake_render,
                                data=types.SimpleNamespace(time=1.0))
    sim.model = types.SimpleNamespace(site_name2id=lambda n: 0)
    sim.data.site_xpos = np.array([[1.0, 2.0, 3.0]])
    sim.data.site_xmat = np.eye(3).reshape(1, 9)
    robot = types.SimpleNamespace(sim=sim, eef_site_id=0)
    p = SensorPort.__new__(SensorPort)
    p._robot = robot
    p.env = types.SimpleNamespace()
    p.output_dir = str(tmp_path)
    p._count = 0
    p.reference = {}
    with pytest.raises(RuntimeError):
        SensorPort.capture(p)


def test_capture_writes_files(monkeypatch, tmp_path):
    from sensors import SensorPort
    from scipy.spatial.transform import Rotation as R

    def fake_render(width, height, camera_name, depth):
        return (np.full((height, width, 3), 128, np.uint8),
                np.full((height, width), 0.5, np.float32))

    class CU:
        @staticmethod
        def get_real_depth_map(sim, d):
            return np.asarray(d)
        @staticmethod
        def get_camera_intrinsic_matrix(sim, name, w, h):
            return np.eye(3)
        @staticmethod
        def get_camera_extrinsic_matrix(sim, name):
            return np.eye(4)

    monkeypatch.setattr('sensors._cu', lambda: CU)

    sid_data = {'site_xpos': np.array([[0.1, 0.2, 0.3]]),
                'site_xmat': np.eye(3).reshape(1, 9),
                'time': 2.5}
    sim = types.SimpleNamespace(render=fake_render, data=types.SimpleNamespace(**sid_data))
    sim.model = types.SimpleNamespace(site_name2id=lambda n: 0)
    robot = types.SimpleNamespace(sim=sim, eef_site_id=0)

    p = SensorPort.__new__(SensorPort)
    p._robot = robot
    p.env = types.SimpleNamespace()
    p.output_dir = str(tmp_path)
    p._count = 0
    p.reference = {}

    import scene_demo.joint_home as jh
    monkeypatch.setattr(jh, 'read_robot_state', lambda env: {'eef_position': [0.1, 0.2, 0.3],
                                                             'eef_rotation': np.eye(3).tolist()})
    out = SensorPort.capture(p)
    assert len(out['frames']) == 2
    views = {f['view'] for f in out['frames']}
    assert views == {'agentview', 'wrist'}
    for f in out['frames']:
        assert os.path.exists(f['rgb_path'])
        assert os.path.exists(f['depth_path'])
        T = np.asarray(f['T_world_camera'])
        assert T.shape == (4, 4)
        K = np.asarray(f['K'])
        assert K.shape == (3, 3)
    assert out['eef_position'] == [0.1, 0.2, 0.3]
    assert out['timestamp'] == 2.5


def test_start_failure_preserves(monkeypatch, tmp_path):
    from sensors import SensorPort
    import scene_demo.joint_home as jh
    import scene_demo.safe_exit as se
    orig_controller = object()
    orig_action = np.array([0.2, -0.3])
    fake_robot = types.SimpleNamespace(
        controller=orig_controller,
        gripper=types.SimpleNamespace(current_action=orig_action),
    )
    monkeypatch.setattr(jh, '_single_robot', lambda env: fake_robot)
    monkeypatch.setattr(jh, 'read_robot_state', lambda env: {})
    monkeypatch.setattr(se, '_finger_hold', lambda env, state: np.array([1.0, 1.0]))
    def _boom(*a, **k):
        raise RuntimeError('constructor')
    monkeypatch.setattr(jh, 'JointHomeAdapter', _boom)
    p = SensorPort(object(), {}, tmp_path)
    with pytest.raises(RuntimeError):
        p.start()
    assert fake_robot.controller is orig_controller
    assert np.array_equal(fake_robot.gripper.current_action, orig_action)
    assert not p._started
