"""Bounded visual association + previously validated geometric side grasp.

This pilot deliberately retains simulator pose/orientation/contact geometry.
It does not claim to estimate a 6-D object pose from real-camera RGBD.
"""
import json
import sys
from pathlib import Path

import numpy as np
from . import sensors, vision
from .simple_checker import SimpleQwenClient
from scene_demo import service, joint_home

# Legacy validated helpers use their original top-level sibling imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collision_grasp as cg
import local_grasp as lg
import side_grasp as sg
import preparation_diagnostics as pd


def json_write(path, row):
    Path(path).write_text(json.dumps(row, ensure_ascii=False, indent=2,
        default=lambda value: np.asarray(value).tolist()) + '\n', encoding='utf-8')


def project(point, K, T, width, height):
    camera = np.linalg.inv(np.asarray(T)) @ np.r_[point, 1.]
    if camera[2] <= 0:
        raise ValueError('target behind camera')
    uv = np.asarray(K) @ camera[:3]
    return [float(uv[0] / uv[2] / width), float(uv[1] / uv[2] / height)]


def association(box, points, tolerance=.03):
    if (not isinstance(box, list) or len(box) != 4 or
            not all(type(v) in (int, float) and np.isfinite(v) for v in box)):
        return False
    a, b, c, d = box
    if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
        return False
    return any(a-tolerance <= x <= c+tolerance and b-tolerance <= y <= d+tolerance for x,y in points)


class GraspVision(SimpleQwenClient):
    request_options = {'enable_thinking': False}

    def parse_verdict(self, content):
        self.localization = vision._parse_verdict_content(content)
        for key in ('target_visible', 'upright'):
            if self.localization.get(key) is not None and type(self.localization[key]) is not bool:
                raise ValueError('invalid ' + key)
        # Use the existing transport/audit protocol without treating localization
        # as a completion observation. This verdict is never sent to the finisher.
        result = {key: None for key in vision._BOOL_FIELDS}
        result.update(boxes={}, evidence=[], unknown_reasons=[])
        return result

    def locate(self, context, paths):
        prompt = '\n'.join([
            '你是抓取前视觉定位检查器，不是任务完成检查器。机械臂准备夹取酒瓶；先确认找对目标，几何辅助器随后校准侧面抓取位姿。',
            '两张图为同一时刻：wrist为腕部第一视角，sideview为外部第三视角。',
            '仅依据图像定位待抓酒瓶，给sideview完整瓶身的紧致矩形框。坐标为相对图像宽高归一化的0到1，格式[xmin,ymin,xmax,ymax]，不可用0到1000。',
            'target_visible表示酒瓶是否可辨认；upright表示瓶身是否直立，明显倾倒或严重遮挡则false或null。未知不猜测。',
            '不要判断是否放到架子，也不要把酒瓶架、手或其他物品当瓶子。',
            '只输出JSON：{"target_visible":true或false或null,"upright":true或false或null,"sideview_box":[xmin,ymin,xmax,ymax]或null,"reason":"画面依据"}',
            json.dumps(context, ensure_ascii=False),
        ])
        response = self._request(context, paths, prompt)
        return dict(response, localization=self.localization)


class CalibratedAssistService(service.SceneService):
    def _run_capability(self, session, plan, job, capability_id):
        self._actual_vla = self._local_grasp_actions = 0
        self._pending_source = None
        self._last_action_source = 'vla'
        self._grasp = cg.CollisionGraspController() if capability_id == 'wine_to_rack' else None
        self._grasp_returned = False
        self._visual_calibration = None
        self._grasp_job, self._grasp_session = job, session
        env = self._env
        original_step = env.step

        def tracked_step(action):
            source = self._pending_source or 'finish'
            self._pending_source = None
            result = original_step(action)
            if source == 'vla':
                self._actual_vla += 1
            elif source == 'local_grasp':
                self._local_grasp_actions += 1
                self._grasp.observe_after(cg.read_geometry(env))
            self._last_action_source = source
            with (job.run_dir/'action_sources.jsonl').open('a') as handle:
                handle.write(json.dumps({'source':source,'actual_vla':self._actual_vla,
                    'local_grasp_actions':self._local_grasp_actions,
                    'timestamp':float(joint_home._single_robot(env).sim.data.time),
                    'action':np.asarray(action).tolist(),
                    'grasp_phase':None if self._grasp is None else self._grasp.phase})+'\n')
            return result

        env.step = tracked_step
        try:
            return super()._run_capability(session, plan, job, capability_id)
        finally:
            env.step = original_step
            if self._grasp is not None:
                json_write(job.run_dir/'grasp_assist.json', {
                    'helper':self._grasp.summary(), 'visual_calibration':self._visual_calibration,
                    'actual_vla':self._actual_vla,'local_grasp_actions':self._local_grasp_actions,
                    'limitation':'Visual target association; precise pose and collision/contact geometry still from simulation',
                })

    def _calibrate(self, reading):
        mismatch = pd.controller_mismatch(pd.controller_facts(self._env))
        if mismatch:
            raise RuntimeError('grasp controller mismatch: ' + str(mismatch))
        job, session = self._grasp_job, self._grasp_session
        port = sensors.SensorPort(self._env, session.home_reference, str(job.run_dir/'pregrasp/sensors'))
        cap = port.capture()  # rendering/encoders only; no controller swap or step
        wrist = next(frame for frame in cap['frames'] if frame['view'] == 'wrist')
        side = next(frame for frame in cap['verification_frames'] if frame['view'] == 'sideview')
        context = {'current_subtask':{'description':'pick up the wine bottle for placement on the rack'},
            'sensor_frames':[{'frame':i,'view':frame['view'],'timestamp':cap['timestamp']} for i,frame in enumerate((wrist,side))],
            'actuator_measurements':{'timestamp':cap['timestamp'],'finger_qpos':list(cap['robot']['finger_qpos'])}}
        client = GraspVision(max_calls=1)
        client.audit_dir = str(job.run_dir/'pregrasp/model_requests')
        response = client.locate(context, [wrist['rgb_path'], side['rgb_path']])
        loc = response['localization']
        cu = sensors._cu()
        sim = joint_home._single_robot(self._env).sim
        K = cu.get_camera_intrinsic_matrix(sim, 'sideview', 512, 512)
        T = cu.get_camera_extrinsic_matrix(sim, 'sideview')
        xyz = np.asarray(reading['snapshot']['objects']['wine_bottle_1']['position'])
        projected = [project(xyz + [0,0,z], K, T, 512, 512) for z in (.03,.06,.09)]
        position, orientation = sg.make_target(reading)
        good = loc.get('target_visible') is True and loc.get('upright') is True and association(loc.get('sideview_box'), projected)
        self._visual_calibration = {'ok':bool(good),'response':response,'capture':cap,
            'projected_body_points':projected,'side_K':np.asarray(K).tolist(),'side_T':np.asarray(T).tolist(),
            'target_position':position.tolist(),'target_orientation':orientation.tolist(),
            'geometry_source':'simulator measured pose; same validated relative side-grasp reference',
            'actual_vla_before':self._actual_vla,
            'initial_position_error_m':float(np.linalg.norm(position - reading['pose']['position']))}
        json_write(job.run_dir/'pregrasp/calibration.json',self._visual_calibration)
        if not good:
            raise RuntimeError('pregrasp visual/geometric association rejected; no forced close')

    def _select_action(self, batch):
        helper = self._grasp
        if helper is not None and helper.phase == lg.FAILED:
            raise RuntimeError('local grasp failed: ' + str(helper.reason))
        if helper is not None and helper.phase == lg.CONFIRMED and not self._grasp_returned:
            self._reset_policy_queues()
            self._grasp_returned = True
        if helper is not None and helper.phase not in (lg.IDLE, lg.CONFIRMED, lg.BYPASS):
            action = helper.next_action(cg.read_geometry(self._env))
            if action is None:
                raise RuntimeError('local grasp stopped: ' + str(helper.reason))
            self._pending_source = 'local_grasp'
            return action
        if self._actual_vla >= 300:
            raise RuntimeError('actual VLA budget reached without accepted completion')
        proposed = super()._select_action(batch)
        if helper is not None and helper.phase == lg.IDLE:
            reading = cg.read_geometry(self._env)
            if lg.evaluate_trigger(reading, proposed)['satisfied']:
                self._calibrate(reading)
                action = helper.next_action(reading, proposed)
                if action is None:
                    raise RuntimeError('calibrated grasp did not arm')
                self._reset_policy_queues()
                self._pending_source = 'local_grasp'
                return action
        self._pending_source = 'vla'
        return proposed
