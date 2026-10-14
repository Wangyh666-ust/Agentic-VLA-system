"""Visual finish bridge for the scene demo.

Candidate, unused module. Reuses the existing control.run_finish pipeline without
modifying controllers, sensors, or homing. Pure imports, no side effects.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque


_PLACE_OPS = ("on", "in")
_TURN_OPS = ("turnon", "turnoff")
_ARTICULATE_OPS = ("open", "close")
_SIMPLE_CHECK_INTERVAL = 150


def build_context(capability, user_request):
    if not isinstance(capability, dict):
        raise ValueError("capability must be a dict")
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("user_request must be a non-empty string")
    goals = capability.get("goals")
    if not isinstance(goals, list) or len(goals) != 1:
        raise ValueError("capability must contain exactly one goal")
    goal = goals[0]
    if not isinstance(goal, list):
        raise ValueError("goal must be a list")
    if len(goal) == 3:
        op_raw = goal[0]
        if not isinstance(op_raw, str) or op_raw.strip() not in ("on", "in"):
            raise ValueError("unsupported operation: %r" % (op_raw,))
        kind = "place"
        target = goal[1]
        destination = goal[2]
    elif len(goal) == 2:
        op_raw = goal[0]
        if not isinstance(op_raw, str) or op_raw.strip() not in _TURN_OPS + _ARTICULATE_OPS:
            raise ValueError("unsupported operation: %r" % (op_raw,))
        kind = "turn" if op_raw.strip() in _TURN_OPS else "articulate"
        target = goal[1]
        destination = None
    else:
        raise ValueError("goal must have length 2 or 3")

    for item in goal:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("goal items must be non-empty strings")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("goal target must be a non-empty string")

    op = op_raw.strip()
    if kind == "place":
        criteria = {
            "must_release": True,
            "must_stabilize": True,
            "must_clear": True,
            "must_preserve_target": True,
        }
        criteria["operation_semantics"] = (
            "place target at destination; if the destination is a container, "
            "place the target inside it, then open the fingers and retreat away "
            "without disturbing it"
        )
    elif kind == "turn":
        criteria = {
            "must_release": True,
            "must_stabilize": True,
            "must_clear": True,
            "must_preserve_target": True,
            "desired_state": "on" if op == "turnon" else "off",
        }
        criteria["operation_semantics"] = (
            "rotate the stove knob to ignite or extinguish the burner; "
            "this is not opening a lid, and is not grasping the target again"
        )

    else:
        criteria = {
            "must_release": True, "must_stabilize": True,
            "must_clear": True, "must_preserve_target": True,
            "desired_state": "closed" if op == "close" else "open",
            "operation_semantics": "move the articulated door to its requested open or closed state; then release contact and retreat without changing that state",
        }
    if capability.get("instruction"):
        criteria["task_description"] = capability["instruction"]

    return {
        "user_request": user_request,
        "current_subtask": {
            "operation": kind,
            "target": target,
            "destination": destination,
            "phase": "executing",
        },
        "criteria": criteria,
    }


class FinishMonitor:
    def __init__(self, env, reference, output_dir, context, guarded_step, on_action,
                 independent_ready, budget, client=None):
        import numpy as _np  # noqa: F401  (lazy, imported for availability check)
        from scene_demo.finish_v1 import vision as _vision  # noqa: F401
        from scene_demo.finish_v1.sensors import SensorPort as _SensorPort

        self.env = env
        self.reference = reference
        from pathlib import Path as _Path
        self.output_dir = _Path(output_dir)
        self.context = context
        self.guarded_step = guarded_step
        self.on_action = on_action
        self.independent_ready = independent_ready
        self.budget = int(budget)

        self.call_client = None
        self._client_explicit = client is not None
        if client is None:
            from scene_demo.finish_v1.vision import QwenClient as _QwenClient
            client = _QwenClient(max_calls=10)
        self.call_client = client

        self.port = _SensorPort(
            env, reference, self.output_dir / "sensors",
            step_callback=self._on_action,
        )
        # Guard this port instance's step only; do not touch env.step or control.
        self._orig_step = self.port.step
        self.port.step = lambda q, f=None: self.guarded_step(
            lambda: self._orig_step(q, f)
        )

        self.history = deque(maxlen=3)
        self.calls = 0
        self.prechecks = 0
        self._simple_checked_steps = set()
        self.interval = max(25, int(math.ceil(self.budget / 5.0)))
        self.next_due = self.interval
        self.last_candidate = False
        self.terminal = None
        self.independent_streak = 0
        self.helper_counts = {"release": 0, "movement": 0, "confirm": 0, "home": 0}
        self._helper_actions = 0
        self._last_decision = None
        self._closed = False
        self._max_calls = 10
        self._max_prechecks = 5
        self._checks_path = self.output_dir / "checks.jsonl"
        self._finish = None
        self._phase = "observing"

    # -- serialization helpers -------------------------------------------

    def _obs_key(self, obs):
        if not isinstance(obs, dict):
            return None
        ts = obs.get("timestamp")
        if ts is None:
            return None
        try:
            return float(ts)
        except (TypeError, ValueError):
            return None

    def _obs_frames(self, obs):
        if not isinstance(obs, dict):
            return []
        out = []
        for frame in obs.get("verification_frames") or []:
            out.append(frame)
        if not out:
            for frame in obs.get("frames") or []:
                out.append(frame)
        return out

    def _collect_observations(self, extra_observations):
        merged = {}
        for obs in list(self.history) + list(extra_observations or []) + [self.port.capture()]:
            merged[float(obs['timestamp'])] = obs
        selected = [merged[t] for t in sorted(merged)[-3:]]
        self.history.clear()
        for obs in selected:
            self.history.append(obs)
        return selected

    def _obs_public(self, obs):
        out = []
        verification = obs.get("verification_frames") or []
        frames = verification if verification else (obs.get("frames") or [])
        for idx, frame in enumerate(frames[:6]):
            if not isinstance(frame, dict):
                continue
            out.append({
                "frame": idx,
                "view": frame.get("view"),
                "timestamp": obs.get("timestamp"),
                "rgb_path": frame.get("rgb_path"),
                "depth_path": frame.get("depth_path"),
            })
        return out

    def _obs_paths(self, obs):
        out = []
        verification = obs.get("verification_frames") or []
        frames = verification if verification else (obs.get("frames") or [])
        for frame in frames[:6]:
            if isinstance(frame, dict) and frame.get("rgb_path"):
                out.append(frame["rgb_path"])
        return out

    def _robot_public(self, robot):
        if not isinstance(robot, dict):
            return {
                "finger_qpos": None,
                "max_abs_arm_qvel": None,
                "arm_qpos": None,
                "stage": None,
            }
        arm_qvel = robot.get("arm_qvel")
        max_v = None
        if arm_qvel is not None:
            try:
                max_v = float(max(abs(float(x)) for x in arm_qvel))
            except (TypeError, ValueError):
                max_v = None
        stage = self.port.phase if self.port.phase is not None else "executing"
        return {
            "finger_qpos": robot.get("finger_qpos"),
            "max_abs_arm_qvel": max_v,
            "arm_qpos": robot.get("arm_qpos"),
            "stage": stage,
        }

    def _write_check(self, row):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self._checks_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x),
            ) + "\n")

    # -- action callback --------------------------------------------------

    def _on_action(self, row):
        if isinstance(row, dict):
            phase = row.get("phase")
            if phase in self.helper_counts:
                self.helper_counts[phase] += 1
            self._helper_actions += 1
        if self.on_action is not None:
            self.on_action(row)
        ready = bool(self.independent_ready())
        if ready:
            self.independent_streak += 1
        else:
            self.independent_streak = 0
        if self._helper_actions % 5 == 0:
            obs = self.port.capture()
            key = self._obs_key(obs)
            if key is not None and not any(
                self._obs_key(h) == key for h in self.history
            ):
                self.history.append(obs)

    # -- record / poll ----------------------------------------------------

    def record(self, step):
        if self.terminal is not None:
            return
        step = int(step)
        if step % 5 != 0:
            return
        obs = self.port.capture()
        key = self._obs_key(obs)
        if key is None:
            return
        if any(self._obs_key(h) == key for h in self.history):
            return
        self.history.append(obs)

    def _check(self, context, observations):
        selected = self._collect_observations(observations)
        simple = bool(getattr(self.call_client, 'simple_handoff', False))
        if simple:
            # Exactly two current views, not two historical samples per view.
            selected = selected[-1:]
        paths = []
        sensor_frames = []
        for cap in selected:
            frames = cap.get('verification_frames') or cap.get('frames') or []
            if simple:
                wrist = next((fr for fr in cap.get('frames', []) if fr['view'] in ('wrist', 'robot0_eye_in_hand')), None)
                external_view = getattr(self.port, 'checker_external_view', 'sideview')
                if external_view not in ('sideview', 'agentview'):
                    raise RuntimeError('unsupported checker external view')
                side = next((fr for fr in frames + cap.get('frames', []) if fr['view'] == external_view), None)
                if wrist is None or side is None:
                    raise RuntimeError('simple checker requires synchronized wrist and %s images' % external_view)
                frames = [wrist, side]
            for fr in frames[:3 if simple else 2]:
                frame = len(paths)
                if frame >= 6:
                    raise RuntimeError('maximum six frames exceeded')
                paths.append(fr['rgb_path'])
                sensor_frames.append({'frame': frame, 'view': fr['view'], 'timestamp': float(cap['timestamp'])})
        robot = selected[-1].get('robot')
        if not isinstance(robot, dict):
            raise RuntimeError('latest captured observation missing robot feedback')
        measurements = {
            'timestamp': float(selected[-1]['timestamp']),
            'stage': self.port.phase or 'executing',
            'finger_qpos': [float(x) for x in robot['finger_qpos']],
            'max_abs_arm_qvel': max(abs(float(x)) for x in robot['arm_qvel']),
            'standard_open_finger_qpos': [float(x) for x in self.reference['finger_home']],
        }
        if len(measurements['finger_qpos']) != 2:
            raise RuntimeError('finger_qpos must have two elements')
        if len(measurements['standard_open_finger_qpos']) != 2:
            raise RuntimeError('standard_open_finger_qpos must have two elements')
        ctx = {
            'user_request': context.get('user_request'),
            'current_subtask': {**context['current_subtask'], 'phase': self.port.phase or 'executing'},
            'criteria': context.get('criteria'),
            'sensor_frames': sensor_frames,
            'actuator_measurements': measurements,
        }
        if self.calls >= 10:
            raise RuntimeError('visual check call budget exhausted')
        self._phase = 'visual_check'
        self.guarded_step(lambda: None)
        self.calls += 1
        if getattr(self.call_client, 'simple_handoff', False):
            self.call_client.audit_dir = self.output_dir / 'model_requests'
        meta = self.call_client.analyze(ctx, paths)
        self.guarded_step(lambda: None)
        self._phase = 'observing'
        if not isinstance(meta, dict):
            raise RuntimeError('analyze must return a dict')
        raw = meta['verdict']
        from scene_demo.finish_v1 import vision
        simple = bool(getattr(self.call_client, 'simple_handoff', False))
        if simple:
            from scene_demo.finish_v1 import simple_checker
            reconcile = simple_checker.reconcile
            decide = simple_checker.decide
        else:
            reconcile = vision.reconcile_verdict
            decide = vision.decide
        fused = reconcile(ctx, raw, robot, self.reference)
        initial_check = {'metadata': dict(meta), 'raw_verdict': raw, 'verdict': fused,
                         'decision': decide(ctx, fused)}
        relation_review = None
        trigger = not simple and vision.needs_relation_review(ctx, raw, robot, self.reference)
        if trigger:
            self._write_check({'initial_check': initial_check, 'ctx': ctx, 'robot': robot,
                               'relation_review_pending': self.calls < 10})
            from scene_demo.finish_v1.relation_images import prepare_relation_images
            prepared = prepare_relation_images(selected, initial_check['raw_verdict'], self.output_dir / 'relation_details' / str(self.calls), operation=ctx['current_subtask']['operation']) if self.calls < 10 else {'ready': False, 'reason': 'review_skipped_budget'}
            if self.calls < 10 and prepared['ready']:
                review_ctx = dict(ctx)
                review_ctx['sensor_frames'] = prepared['sensor_frames']
                mapping_keys = ('frame', 'source_frame', 'view', 'timestamp', 'source_size_px', 'pixel_bounds', 'original_roi')
                review_ctx['relation_derivatives'] = [{k: d[k] for k in mapping_keys} for d in prepared['derivative_provenance']]
                self._phase = 'visual_check'
                self.guarded_step(lambda: None)
                self.calls += 1
                review_meta = self.call_client.inspect_relations(review_ctx, prepared['image_paths'])
                self.guarded_step(lambda: None)
                self._phase = 'observing'
                raw = review_meta['verdict']
                fused = vision.reconcile_verdict(ctx, raw, robot, self.reference)
                relation_review = {'trigger': True, 'prepared_inputs': prepared, 'review_context': review_ctx, 'metadata': review_meta,
                                   'raw_verdict': raw, 'verdict': fused,
                                   'decision': vision.decide(ctx, fused)}
                meta = review_meta
            else:
                relation_review = {'trigger': True, 'review_skipped_budget': self.calls >= 10, 'prepared_inputs': prepared, 'reason': prepared['reason']}
        self._last_decision = decide(ctx, fused)
        row = {
            'initial_check': initial_check,
            'relation_review': relation_review,
            'raw_verdict': raw,
            'verdict': fused,
            'usage': meta.get('usage'),
            'status': meta.get('status'),
            'model': meta.get('model'),
            'http_status': meta.get('http_status'),
            'request_sha256': meta.get('request_sha256'),
            'response_sha256': meta.get('response_sha256'),
            'call_index': meta.get('call_index'),
            'image_paths': paths,
            'sensor_frames': sensor_frames,
            'ctx': ctx,
            'robot': robot,
        }
        self._write_check(row)
        return fused

    def _finish_outcome(self, ok, reason):
        self.terminal = {'ok': bool(ok), 'reason': str(reason), 'checks': self.calls}
        if self._finish is not None:
            self.terminal['finish'] = self._finish
        self.close()
        return self.terminal

    def poll(self, step, candidate=False, force=False):
        if self.terminal is not None:
            return self.terminal
        simple = bool(getattr(self.call_client, 'simple_handoff', False))
        if simple:
            # Inspect the client here: the pilot replaces it after construction.
            # Native candidates and force cannot cause an early visual handoff.
            if step <= 0 or step % _SIMPLE_CHECK_INTERVAL or step > self.budget or step in self._simple_checked_steps:
                return None
            self._simple_checked_steps.add(step)
        else:
            first_candidate = bool(candidate) and not self.last_candidate
            distinct = len({float(cap['timestamp']) for cap in self.history})
            due = step >= self.next_due
            if not (distinct >= 2 and self.prechecks < 5 and (due or first_candidate or force)):
                return None
        try:
            verdict = self._check(self.context, [])
        except Exception:
            self.close()
            raise
        self.last_candidate = bool(candidate)
        self.prechecks += 1
        self.next_due = step + self.interval
        decision = self._last_decision
        state = decision.get('state') if isinstance(decision, dict) else None
        if state == 'incomplete':
            self._phase = 'observing'
            return None
        if state == 'unknown':
            # A valid inconclusive observation is not a failed execution. Keep
            # VLA running until the next scheduled observation / action budget.
            # Missing sensors, invalid replies and transport errors still raise.
            if simple and step < self.budget:
                self._phase = 'observing'
                return None
            return self._finish_outcome(False, 'visual_unknown')
        if state not in ('needs_release', 'needs_retreat', 'complete'):
            self.close()
            raise RuntimeError('invalid visual state: %r' % (state,))
        from scene_demo.finish_v1 import control, vision
        try:
            if getattr(self.call_client, 'simple_handoff', False):
                from scene_demo.finish_v1 import simple_checker
                finish = control.run_finish(self.port, self.context, verdict, simple_checker.decide,
                                            self._check, emit=lambda _: None, simple_handoff=True)
            else:
                finish = control.run_finish(self.port, self.context, verdict, vision.decide, self._check, emit=lambda _: None)
        except Exception:
            self.close()
            raise
        self._finish = finish
        try:
            self.guarded_step(lambda: None)
        except Exception:
            self.close()
            raise
        if isinstance(finish, dict) and finish.get('ok'):
            if self.independent_streak >= 5:
                return self._finish_outcome(True, 'complete')
            return self._finish_outcome(False, 'independent_finish_failed')
        reason = 'independent_finish_failed'
        if isinstance(finish, dict) and finish.get('reason'):
            reason = finish['reason']
        return self._finish_outcome(False, reason)

    def status(self):
        if self.terminal is not None:
            phase = 'done' if self.terminal.get('ok') else 'blocked'
        elif self._phase == 'visual_check':
            phase = 'visual_check'
        else:
            phase = self.port.phase or self._phase
        terminal = None
        if self.terminal is not None:
            terminal = {'ok': self.terminal['ok'], 'reason': self.terminal['reason'], 'checks': self.terminal['checks']}
        return {
            'phase': phase,
            'calls': self.calls,
            'prechecks': self.prechecks,
            'helper_counts': dict(self.helper_counts),
            'last_decision': dict(self._last_decision) if self._last_decision is not None else None,
            'terminal': terminal,
            'independent_streak': self.independent_streak,
        }

    def close(self):
        if self._closed:
            return
        self.port.close()
        self._closed = True
