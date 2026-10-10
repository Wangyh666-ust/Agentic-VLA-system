import os, json, time
import numpy as np
from scipy.spatial.transform import Rotation as R
import scene_demo.joint_home as jh
import scene_demo.safe_exit as safe_exit


def _cu():
    from robosuite.utils import camera_utils as cu
    return cu


class SensorPort:
    def __init__(self, env, reference, output_dir, step_callback=None):
        self.env = env
        self.reference = reference
        self.output_dir = output_dir
        self.step_callback = step_callback
        self.phase = None
        self._started = False
        self._original = None
        self._adapter = None
        self._robot = None
        self._count = 0
        os.makedirs(output_dir, exist_ok=True)

    def _robot_(self):
        if self._robot is None:
            self._robot = jh._single_robot(self.env)
        return self._robot

    def start(self):
        if self._started:
            return
        robot = self._robot_()
        state = jh.read_robot_state(self.env)
        hold = safe_exit._finger_hold(self.env, state)
        if len(hold) != 2:
            raise RuntimeError('finger hold must have length 2')
        original = robot.controller
        grip_orig = np.asarray(robot.gripper.current_action, dtype=np.float32).copy()
        adapter = jh.JointHomeAdapter(original, self.env, self.reference)
        self._original = original
        self._grip_orig = grip_orig
        self._adapter = adapter
        robot.controller = adapter
        robot.gripper.current_action = np.asarray(hold, dtype=np.float32).copy()
        self._started = True

    def _finger_norm(self, physical):
        robot = self._robot_()
        sim = robot.sim
        raw = sim.model._model
        import mujoco
        acts = robot._ref_joint_gripper_actuator_indexes
        lo = np.array([raw.actuator_ctrlrange[i][0] for i in acts], dtype=np.float64)
        hi = np.array([raw.actuator_ctrlrange[i][1] for i in acts], dtype=np.float64)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        half = np.where(half == 0, 1.0, half)
        return np.clip((np.asarray(physical, float) - mid) / half, -1.0, 1.0).astype(np.float32)

    def step(self, qtarget, finger_target=None):
        if not self._started:
            self.start()
        robot = self._robot_()
        q7 = np.asarray(qtarget, dtype=np.float64).reshape(7)
        self._adapter.select_target(q7)
        if finger_target is not None:
            robot.gripper.current_action = self._finger_norm(finger_target)
        action = np.zeros(7, dtype=np.float32)
        self.env.step(action)
        state = jh.read_robot_state(self.env)
        if self.step_callback is not None:
            row = dict(state)
            row['phase'] = self.phase
            row['qtarget'] = q7.tolist()
            row['finger_target'] = None if finger_target is None else np.asarray(finger_target, float).tolist()
            row['action'] = action.tolist()
            self.step_callback(row)
        return state

    def read_robot(self):
        return jh.read_robot_state(self.env)

    def close(self):
        if not self._started:
            return
        robot = self._robot_()
        try:
            robot.controller = self._original
            robot.gripper.current_action = self._grip_orig
            self._original.update(force=True)
            self._original.reset_goal()
        finally:
            self._started = False

    def _site_id(self):
        robot = self._robot_()
        sid = getattr(robot, 'eef_site_id', None)
        if isinstance(sid, (int, np.integer)):
            return int(sid)
        name = getattr(robot.controller, 'eef_name', None)
        if name is None:
            raise RuntimeError('no eef site id or name')
        sim = robot.sim
        return sim.model.site_name2id(name)

    def capture(self):
        import PIL.Image as Image
        robot = self._robot_()
        sim = robot.sim
        cu = _cu()
        out = {'timestamp': float(sim.data.time), 'frames': []}
        for view, out_view in (('agentview', 'agentview'), ('robot0_eye_in_hand', 'wrist')):
            res = sim.render(width=256, height=256, camera_name=view, depth=True)
            if not (isinstance(res, tuple) and len(res) == 2):
                raise RuntimeError('sim.render must return (rgb,depth) tuple')
            rgb, depth = res[0], res[1]
            rgb = np.asarray(rgb)[::-1]
            dmap = cu.get_real_depth_map(sim, depth)[::-1]
            K = cu.get_camera_intrinsic_matrix(sim, view, 256, 256)
            T = cu.get_camera_extrinsic_matrix(sim, view)
            K = np.asarray(K, float)
            T = np.asarray(T, float)
            if K.shape != (3, 3) or T.shape != (4, 4):
                raise RuntimeError('bad K/T shapes')
            self._count += 1
            base = os.path.join(self.output_dir, 'cap%06d_%s' % (self._count, out_view))
            rp = base + '.png'
            dp = base + '.npy'
            cp = base + '.json'
            Image.fromarray(rgb.astype(np.uint8)).save(rp)
            np.save(dp, dmap.astype(np.float32))
            with open(cp, 'w') as f:
                json.dump({'K': K.tolist(), 'T': T.tolist(), 'view': out_view}, f)
            out['frames'].append({'view': out_view, 'rgb_path': rp, 'depth_path': dp,
                                  'K': K.tolist(), 'T_world_camera': T.tolist()})
        out['robot'] = jh.read_robot_state(self.env)
        sid = self._site_id()
        out['eef_position'] = np.asarray(robot.sim.data.site_xpos[sid], float).tolist()
        out['eef_rotation'] = np.asarray(robot.sim.data.site_xmat[sid], float).reshape(3, 3).tolist()
        return out

    def _scratch(self, q):
        import mujoco
        robot = self._robot_()
        sim = robot.sim
        raw = sim.model._model
        data = mujoco.MjData(raw)
        data.qpos[:] = sim.data.qpos[:]
        data.qvel[:] = sim.data.qvel[:]
        data.time = sim.data.time
        state = jh.read_robot_state(self.env)
        idx = np.asarray(state['qpos_indexes'], int)
        data.qpos[idx] = np.asarray(q, float)
        mujoco.mj_forward(raw, data)
        return raw, data

    def ik(self, target_xyz, orientation, seedq):
        import mujoco
        robot = self._robot_()
        state = jh.read_robot_state(self.env)
        self._qpos_indexes = state['qpos_indexes']
        vi = list(state['qvel_indexes'])
        low = np.asarray(state['joint_limits_low'], float)
        high = np.asarray(state['joint_limits_high'], float)
        sid = self._site_id()
        q = np.asarray(seedq, float).copy()
        tgt = np.asarray(target_xyz, float).reshape(3)
        Rt = np.asarray(orientation, float).reshape(3, 3)
        lam = 0.03
        for _ in range(80):
            raw, rd = self._scratch(q)
            p = np.asarray(rd.site_xpos[sid], float)
            Rc = np.asarray(rd.site_xmat[sid], float).reshape(3, 3)
            pe = tgt - p
            rv = R.from_matrix(Rt @ Rc.T).as_rotvec()
            if np.linalg.norm(pe) < 1e-3 and np.linalg.norm(rv) < 1e-2:
                break
            jacp = np.zeros((3, raw.nv))
            jacr = np.zeros((3, raw.nv))
            mujoco.mj_jacSite(raw, rd, jacp, jacr, sid)
            J = np.vstack([jacp[:, vi], jacr[:, vi]])
            err = np.concatenate([pe, rv])
            dq = np.linalg.solve(J.T @ J + lam * lam * np.eye(len(vi)), J.T @ err)
            n = np.linalg.norm(dq)
            if n > 0.015:
                dq = dq * (0.015 / n)
            q = q + dq
            if np.any(q < low - 1e-9) or np.any(q > high + 1e-9):
                return None
        if np.any(q < low - 1e-9) or np.any(q > high + 1e-9):
            return None
        raw, rd = self._scratch(q)
        p = np.asarray(rd.site_xpos[sid], float)
        Rc = np.asarray(rd.site_xmat[sid], float).reshape(3, 3)
        pe = np.linalg.norm(tgt - p)
        rv = np.linalg.norm(R.from_matrix(Rt @ Rc.T).as_rotvec())
        if pe > 1e-3 or rv > 1e-2:
            return None
        return q

    def robot_spheres(self, q):
        import mujoco
        robot = self._robot_()
        sim = robot.sim
        mdl = sim.model
        raw = mdl._model
        rm = getattr(robot, 'robot_model', None)
        gr = getattr(robot, 'gripper', None)
        if rm is None or not hasattr(rm, 'contact_geoms'):
            raise RuntimeError('robot_model.contact_geoms missing')
        arm_names = list(rm.contact_geoms)
        hand_names = list(gr.contact_geoms) if (gr is not None and hasattr(gr, 'contact_geoms')) else []
        raw, rd = self._scratch(q)
        out = []
        for name, hand in [(n, False) for n in arm_names] + [(n, True) for n in hand_names]:
            gid = mdl.geom_name2id(name)
            if gid < 0:
                raise RuntimeError('geom not found: ' + name)
            gt = raw.geom_type[gid]
            size = np.asarray(raw.geom_size[gid], float)
            if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = size[0]
            elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r = size[0] + size[1]
            elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r = float(np.hypot(size[0], size[1]))
            elif gt == mujoco.mjtGeom.mjGEOM_BOX:
                r = float(np.linalg.norm(size[:3]))
            else:
                r = float(raw.geom_rbound[gid])
            if not np.isfinite(r) or r <= 0:
                raise RuntimeError('bad radius for ' + name)
            c = np.asarray(rd.geom_xpos[gid], float)
            out.append({'id': name, 'center': c.tolist(), 'radius': float(r), 'hand': bool(hand)})
        return out
