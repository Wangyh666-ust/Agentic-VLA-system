"""Opt-in X-VLA persistent service; physical operations stay on the base worker."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from scene_demo import service
from scene_demo.xvla_skills import route_capabilities
from scene_demo.xvla_backend import MODEL_PATH, MODEL_REVISION, NativeXvlaEnv, make_client, reference_for, reference_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)), encoding='utf-8')


def next_capability(capabilities, current):
    index = capabilities.index(current) + 1
    return capabilities[index] if index < len(capabilities) else 'global_home'


def bridge_port(port):
    """Temporarily satisfy the original joint adapter's relative-OSC contract."""
    original_start, original_close = port.start, port.close
    saved = {}

    def restore(event):
        controller = saved.get('controller')
        if controller is None:
            return
        before = controller.use_delta
        controller.use_delta = saved['use_delta']
        controller.update(force=True)
        controller.reset_goal()
        robot = port._robot_()
        row = {'event': event, 'before': before, 'after': controller.use_delta,
               'timestamp': float(robot.sim.data.time)}
        path = Path(port.output_dir) / 'controller_modes.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        saved.clear()

    def start():
        if port._started:
            return original_start()
        controller = port._robot_().controller
        saved['controller'], saved['use_delta'] = controller, controller.use_delta
        before = controller.use_delta
        controller.use_delta = True
        row = {'event': 'start', 'before': before, 'after': controller.use_delta,
               'timestamp': float(port._robot_().sim.data.time)}
        path = Path(port.output_dir) / 'controller_modes.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        try:
            return original_start()
        except BaseException:
            restore('start_error')
            raise

    def close():
        try:
            return original_close()
        finally:
            restore('close')

    port.start, port.close = start, close
    return port


def handoff_evidence(pilot: dict) -> dict:
    source = pilot.get('source_counts') or {}
    status = pilot.get('finish_status') or {}
    terminal = status.get('terminal') or {}
    finish = pilot.get('finish') or {}
    counts = finish.get('counts') or {}
    metrics = pilot.get('actual_final_metrics') or {}
    resets = pilot.get('cache_reset') or []
    cache = resets[-1] if resets else {}
    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    def at_least(value, minimum):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= minimum
    def within(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1e-8
    checks = {
        'vla_actions': positive(source.get('vla')),
        'vla_calls': positive(pilot.get('vla_calls')),
        'visual_trigger': positive(status.get('prechecks')),
        'finish_terminal': terminal.get('ok') is True,
        'release_open_actions': positive(finish.get('release_open_actions')),
        'movement_actions': positive(counts.get('movement')),
        'home_actions': positive(counts.get('home')),
        'independent_five': at_least(status.get('independent_streak'), 5),
        'reference_ready': metrics.get('ready') is True,
        'queue_empty': cache.get('queue_size') == 0 and isinstance(cache.get('queue_size'), int) and not isinstance(cache.get('queue_size'), bool),
        'proprio_cleared': cache.get('proprio_is_none') is True,
        'absolute_controller': cache.get('controller_use_delta') is False,
        'position_goal_reset': within(cache.get('goal_pos_error')),
        'orientation_goal_reset': within(cache.get('goal_ori_error')),
        'gripper_command_reset': within(cache.get('gripper_hold_error')),
    }
    return {'ok': all(checks.values()), 'checks': checks}


class XvlaSceneService(service.SceneService):
    def __init__(self, run_root: str, *, expected_capabilities: list[str] | None = None):
        self.capability_backend = 'xvla'
        service.BACKEND_NAME = 'xvla'
        service.WORKFLOW_NAME = 'persistent_scene_xvla_v2'
        service.CONTROL_MODE = 'absolute'
        service.MAX_BUDGET, service.DEFAULT_BUDGET, service.SESSION_STEP_LIMIT = 900, 800, 8000
        self.expected_capabilities = expected_capabilities
        self._xvla_client = None
        self._cache_evidence = []
        super().__init__(model_path=MODEL_PATH, run_root=run_root,
                         completion_mode='release_verified', grasp_guard_mode='off', finish_mode='visual')
        self._model_revision = MODEL_REVISION

    def _load_policy(self):
        self._xvla_client = make_client()
        self._model_revision = MODEL_REVISION

    def _do_create_session(self, record, seed, init_state_index):
        self._session_run_dir = record.run_dir
        result = super()._do_create_session(record, seed, init_state_index)
        record.xvla_global_reference = copy.deepcopy(record.home_reference)
        record.xvla_completed = []
        return result

    def _build_env(self, suite_name, task_id, seed, init_state_index):
        return NativeXvlaEnv(suite_name, task_id, seed, init_state_index,
                             self._session_run_dir / 'native', session_limit=8000)

    def _robot(self):
        from scene_demo import joint_home
        return joint_home._single_robot(self._env)

    def _controller(self):
        return self._robot().controller

    def _observation_batch(self, obs, instruction):
        Mat_to_Rotate6D = self._xvla_client.processor.Mat_to_Rotate6D
        raw = copy.deepcopy(obs)
        controller = self._controller()
        controller.update(force=True)
        raw['robo_pos'] = controller.ee_pos.copy()
        raw['robo_ori'] = Mat_to_Rotate6D(controller.ee_ori_mat)
        return raw, instruction

    def _select_action(self, batch):
        if self._controller().use_delta is not False:
            raise RuntimeError('xvla_requires_absolute_controller')
        self._env.action_source = 'vla'
        return np.asarray(self._xvla_client.step(*batch))

    def _reset_policy_queues(self):
        if self._xvla_client is not None:
            self._xvla_client.reset()
            assert len(self._xvla_client.action_plan) == 0
            assert self._xvla_client.proprio is None
        if self._env is not None:
            controller = self._controller()
            controller.update(force=True)
            controller.reset_goal()
            controller.use_delta = False
            from scene_demo import joint_home, safe_exit
            state = joint_home.read_robot_state(self._env)
            hold = safe_exit._finger_hold(self._env, state)
            self._robot().gripper.current_action = np.asarray(hold, dtype=np.float32).copy()
            self._last_obs = self._refresh_observation(self._env)
            self._cache_evidence.append(self._actual_state())

    def _actual_state(self):
        from scene_demo import joint_home, safe_exit
        inner = joint_home._inner_env(self._env)
        state = joint_home.read_robot_state(self._env)
        grip = np.asarray(self._robot().gripper.current_action, dtype=float)
        hold = np.asarray(safe_exit._finger_hold(self._env, state), dtype=float)
        return {'gripper_current_action': grip.tolist(),
                'gripper_hold_from_measurement': hold.tolist(),
                'gripper_hold_error': float(np.max(np.abs(grip-hold))), 'timestamp': float(inner.sim.data.time),
                'robot': state,
                'eef': joint_home._read_eef_pose(self._env),
                'queue_size': len(self._xvla_client.action_plan),
                'proprio_is_none': self._xvla_client.proprio is None,
                'controller_use_delta': self._controller().use_delta,
                'goal_pos_error': float(np.linalg.norm(np.asarray(self._controller().goal_pos)-self._controller().ee_pos)),
                'goal_ori_error': float(np.linalg.norm(np.asarray(self._controller().goal_ori)-self._controller().ee_ori_mat))}

    def submit_plan(self, payload):
        if not isinstance(payload, dict):
            return super().submit_plan(payload)
        ids = payload.get('capability_ids')
        session_id = payload.get('session_id')
        record = getattr(self, '_sessions', {}).get(session_id) if isinstance(session_id, str) else None
        if (payload.get('decision') == 'execute' and record is not None
                and isinstance(ids, list) and 0 < len(ids) <= service.MAX_CAPABILITIES
                and all(isinstance(cid, str) for cid in ids) and len(set(ids)) == len(ids)):
            available = [c['capability_id'] for c in record.public()['capabilities']
                         if c.get('skill_kind') == 'native_long_horizon']
            route = route_capabilities(record.scene_id, ids, available)
            payload = copy.deepcopy(payload)
            payload['capability_ids'] = route['executed_capability_ids']
            payload['_xvla_native_route'] = route
        if (payload.get('decision') == 'execute' and self.expected_capabilities
                and payload.get('capability_ids') != self.expected_capabilities):
            return {'ok': False, 'reason': 'pilot_plan_mismatch',
                    'detail': {'expected': self.expected_capabilities, 'actual': payload.get('capability_ids')}}
        return super().submit_plan(payload)

    def _prepare_first(self, reference, session, plan, job):
        from scene_demo import joint_home, safe_exit_contacts
        from scene_demo.finish_v1.sensors import SensorPort
        if safe_exit_contacts.contacts(self._env):
            raise RuntimeError('prepare_start_hand_contact')
        state = joint_home.read_robot_state(self._env)
        if max(abs(a-b) for a,b in zip(state['finger_qpos'], reference['finger_home'])) > .002:
            raise RuntimeError('prepare_start_not_open')
        port = bridge_port(SensorPort(self._env, reference, job.run_dir / 'prepare_sensors'))
        waypoints = joint_home.joint_waypoints(state['arm_qpos'], reference['homeq'], step_rad=.01)
        screen = port.home_screen(waypoints)
        if not screen['ok']:
            raise RuntimeError('prepare_path_blocked: ' + json.dumps(screen))
        streak, count, rows = 0, 0, []
        try:
            while count < 360:
                with self._lock:
                    if self._stop.is_set() or plan.cancel_event.is_set() or self._total_steps >= 8000:
                        raise RuntimeError('prepare_stopped_cancelled_or_session_limit')
                    q = waypoints[min(count, len(waypoints)-1)]
                    self._env.action_source = 'prepare'
                    port.step(q, reference['finger_home'])
                    count += 1
                    self._total_steps += 1
                    session.total_steps = job.total_steps = self._total_steps
                metrics = reference_metrics(self._env, reference)
                ready = metrics['ready'] and not safe_exit_contacts.contacts(self._env)
                streak = streak + 1 if ready else 0
                rows.append({'action': count, 'metrics': metrics, 'ready': ready, 'streak': streak})
                if streak >= 5:
                    return {'ok': True, 'actions': count, 'metrics': metrics, 'streak': streak}
            raise RuntimeError('prepare_reference_not_reached')
        finally:
            port.close()
            write_json(job.run_dir / 'prepare_result.json', {'actions': count, 'streak': streak,
                'path_screen': screen, 'samples': rows})
            self._reset_policy_queues()

    def _run_capability(self, session, plan, job, capability_id):
        job.run_dir.mkdir(parents=True, exist_ok=True)
        self._xvla_client.directory = job.run_dir / 'xvla_calls'
        self._xvla_client.job_calls = 0
        before_counts = dict(self._env.counts)
        cache_start = len(self._cache_evidence)
        before = self._actual_state()
        target = None
        result = {'ok': False, 'reason': 'pilot_precondition_failed'}
        try:
            current = reference_for(self._env, capability_id, session.xvla_global_reference)
            if not session.xvla_completed:
                self._prepare_first(current, session, plan, job)
            elif not reference_metrics(self._env, current)['ready']:
                raise RuntimeError('previous_handoff_not_ready')
            nxt = next_capability(plan.capability_ids, capability_id)
            target = (copy.deepcopy(session.xvla_global_reference) if nxt == 'global_home'
                      else reference_for(self._env, nxt, session.xvla_global_reference))
            if nxt == 'global_home':
                target['reference_kind'] = 'global_home'
            session.home_reference = copy.deepcopy(target)
            write_json(job.run_dir / 'handoff_reference.json', {'next_capability': nxt,
                'target_kind': 'global_home' if nxt == 'global_home' else 'next_native', 'reference': target})
            result = super()._run_capability(session, plan, job, capability_id)
            self._reset_policy_queues()
        except Exception as exc:
            job.success, job.state = False, 'error'
            job.ended_reason, job.error = 'pilot_precondition_failed', str(exc)
            result = {'ok': False, 'reason': job.ended_reason, 'detail': str(exc)}
        finally:
            monitor = getattr(job, '_xvla_monitor', None)
            pilot = {'capability': capability_id,
                'outcome': result, 'job': job.public(), 'source_counts':
                {k: self._env.counts[k]-before_counts[k] for k in before_counts},
                'vla_calls': self._xvla_client.job_calls, 'handoff_reference': target,
                'actual_final_metrics': reference_metrics(self._env, target) if target else None,
                'finish_status': getattr(job, 'finish_status', None),
                'finish': getattr(monitor, '_finish', None), 'before': before,
                'after': self._actual_state(), 'cache_reset': self._cache_evidence[cache_start:]}
            if result.get('ok') is True:
                pilot['handoff_evidence'] = handoff_evidence(pilot)
                if pilot['handoff_evidence']['ok']:
                    session.xvla_completed.append(capability_id)
                else:
                    job.success = False
                    job.ended_reason = 'handoff_evidence_failed'
                    result['ok'] = False
                    result['reason'] = 'handoff_evidence_failed'
                pilot['job'] = job.public()
            base_path = job.run_dir / 'result.json'
            if base_path.exists():
                (job.run_dir / 'base_result.json').write_bytes(base_path.read_bytes())
            write_json(base_path, {'ok': result.get('ok') is True, 'job': job.public(),
                                  'outcome': result})
            write_json(job.run_dir / 'pilot_result.json', pilot)
        return result

    def _build_finish_context(self, capability: dict, user_request: str) -> dict:
        from scene_demo.xvla_finish import build_skill_context
        return build_skill_context(capability, user_request)

    def _make_finish_monitor(self, session, plan, job, env, capability, budget, frames, latest_png):
        from scene_demo.xvla_finish import XvlaQwenClient
        from scene_demo import safe_exit_contacts
        monitor = super()._make_finish_monitor(session, plan, job, env, capability, budget, frames, latest_png)
        bridge_port(monitor.port)
        monitor.call_client = XvlaQwenClient(max_calls=10)
        monitor._client_explicit = True
        cid = job.capability_id
        monitor.port.feedback_retreat_enabled = cid in ('wine_to_rack', 'stove_on')
        monitor.port.checker_views = ('wrist', 'agentview', 'sideview')
        def independent_ready():
            ids = session.xvla_completed + [cid]
            gates, predicates = {}, {}
            for identifier in ids:
                goals = [list(g) for g in service.catalog.CAPABILITIES[identifier]['goals']]
                values = {service.catalog.goal_key(g): bool(service.eval_goal_predicate(env, g)) for g in goals}
                predicates.update(values)
                gates[identifier] = self._completion_ready(env, goals, values)
            metrics = reference_metrics(env, session.home_reference)
            contacts = safe_exit_contacts.contacts(env)
            ready = all(gates.values()) and metrics['ready'] and contacts == []
            row = {'predicates': predicates, 'completion_gates': gates, 'metrics': metrics,
                   'contacts': contacts, 'actual': self._actual_state(), 'ready': ready}
            with (job.run_dir / 'independent_finish.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            return ready
        job._xvla_monitor = monitor
        monitor.independent_ready = independent_ready
        return monitor
