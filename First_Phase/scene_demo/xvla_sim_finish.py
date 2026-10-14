"""Privileged simulator predicates with the unchanged finishing controller."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from scene_demo.finish_v1.runtime import FinishMonitor
from scene_demo.xvla_process_evidence import ReleaseSampler, capture_frames


def validate_checker_mode(mode):
    if mode not in ('visual', 'simulator'):
        raise ValueError('checker_mode must be visual or simulator')
    return mode


class SimFinishMonitor(FinishMonitor):
    def __init__(self, *args, **kwargs):
        kwargs['client'] = SimpleNamespace(simple_handoff=True)
        super().__init__(*args, **kwargs)
        self.goals = []
        self.preserved_goals = []
        self.candidate_streak = 0
        self.trigger_count = 0
        self.simulator_checks = 0
        self.sample_count = 0
        self._sampled_steps = set()
        self._last_sample_step = None
        self._observed_vla_steps = set()
        self.release_sampler = ReleaseSampler(list(self.reference['finger_home']))
        self.release_events = 0
        self.release_external_views = False
        self.skipped_policy_captures = 0
        self.skipped_helper_captures = 0

    def configure_simulation(self, goals: list[list[str]], preserved_goals: list[list[str]]) -> None:
        if not isinstance(goals, list) or not goals or any(not isinstance(g, list) or not g for g in goals):
            raise ValueError('nonempty current native goals required')
        if not isinstance(preserved_goals, list) or any(not isinstance(g, list) or not g for g in preserved_goals):
            raise ValueError('invalid preserved goals')
        self.goals, self.preserved_goals = copy.deepcopy(goals), copy.deepcopy(preserved_goals)

    def _predicates(self):
        from scene_demo import service
        values = []
        for goal in self.goals + self.preserved_goals:
            try:
                value = service.eval_goal_predicate(self.env, goal)
                if type(value) is not bool:
                    raise ValueError('invalid simulator predicate value: '+repr(value))
                values.append({'goal': list(goal), 'value': value})
            except Exception as error:
                self._append('simulator_samples.jsonl', {'goal':list(goal), 'value':None,
                    'error_type':type(error).__name__, 'error':str(error), 'status':'predicate_error'})
                raise
        return values, bool(self.goals) and all(row['value'] is True for row in values)

    def _append(self, name, row):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / name).open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')

    def visual_check_due(self, step: int) -> bool:
        if step <= 0 or step > self.budget or step in self._sampled_steps:
            return False
        predicates, passed = self._predicates()
        self._sampled_steps.add(step)
        self.sample_count += 1
        self.candidate_streak = (self.candidate_streak + 1 if passed and
            (self._last_sample_step is None or step == self._last_sample_step + 1) else 1 if passed else 0)
        self._last_sample_step = step
        trigger = self.candidate_streak >= 5
        if trigger:
            self.trigger_count += 1
        self._append('simulator_samples.jsonl', {'step':step, 'predicates':predicates,
            'all_true':passed, 'candidate_streak':self.candidate_streak, 'trigger':trigger})
        return trigger or step == self.budget

    def _check(self, context: dict, observations: list) -> dict:
        from scene_demo.finish_v1 import vision, simple_checker
        cap = self.port.capture()
        robot = cap['robot']
        timestamp = float(cap['timestamp'])
        ctx = copy.deepcopy(context)
        frames = cap.get('checker_frames') or (cap.get('frames', []) + cap.get('verification_frames', []))
        ctx['sensor_frames'] = [{'frame':i,'view':f['view'],'timestamp':timestamp} for i,f in enumerate(frames)]
        ctx['actuator_measurements'] = {'timestamp':timestamp,'finger_qpos':list(robot['finger_qpos']),
            'standard_open_finger_qpos':list(self.reference['finger_home']),
            'max_abs_arm_qvel':max(abs(float(v)) for v in robot['arm_qvel'])}
        predicates, passed = self._predicates()
        passed = passed and self.trigger_count > 0
        verdict = {key:None for key in vision._BOOL_FIELDS}
        verdict.update(operation_achieved=passed, at_destination=passed, boxes={},
            evidence=['Privileged simulator native predicates; no visual inference.'], unknown_reasons=[])
        fused = simple_checker.reconcile(ctx, verdict, robot, self.reference)
        self._last_decision = simple_checker.decide(ctx, fused)
        self.simulator_checks += 1
        self._write_check({'checker_mode':'simulator','decision_source':'simulator_predicates',
            'predicates':predicates,'robot':robot,'ctx':ctx,'verdict':fused,'decision':self._last_decision})
        return fused

    def _observe_release(self):
        robot = self.port.read_robot()
        event = self.release_sampler.observe(list(robot['finger_qpos']), self._capture_release)
        if event is not None:
            event['images'] = [{'view':f['view'],'path':str(f['rgb_path']),
                'sha256':hashlib.sha256(Path(f['rgb_path']).read_bytes()).hexdigest()} for f in capture_frames(event['capture'])]
            event['status'] = 'recorded_only_simulator'
            self.release_events += 1
            self._append('process_events.jsonl', event)

    def _capture_release(self) -> dict:
        cap = self.port.capture()
        if self.release_external_views:
            from scene_demo.xvla_grounding import render_external
            wrist = next(f for f in cap['frames'] if f['view'] == 'wrist')
            stem = Path(wrist['rgb_path']).stem + '_checker'
            cap['checker_frames'] = [copy.deepcopy(wrist)] + render_external(self.port.env, self.port.output_dir, stem)
        return cap

    def record(self, step):
        if self.terminal is None and step not in self._observed_vla_steps:
            self._observed_vla_steps.add(step)
            if int(step) % 5 == 0:
                self.skipped_policy_captures += 1
            self._observe_release()

    def _on_action(self, row):
        if isinstance(row, dict):
            phase = row.get('phase')
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
            self.skipped_helper_captures += 1
        self._observe_release()

    def _finish_outcome(self, ok, reason):
        if self._finish is not None:
            self._finish['handoff_mode'] = 'simulator_predicates'
            if self._finish.get('reason') == 'final visual failed':
                self._finish['reason'] = 'final_simulator_check_failed'
        if reason == 'final visual failed':
            reason = 'final_simulator_check_failed'
        return super()._finish_outcome(ok, reason)

    def status(self):
        result = super().status()
        result.update(checker_mode='simulator', decision_source='simulator_predicates',
            simulator_checks=self.simulator_checks, sample_count=self.sample_count,
            candidate_streak=self.candidate_streak, trigger_count=self.trigger_count,
            actual_visual_requests=0, calls=0, release_events=self.release_events)
        result.update(skipped_policy_captures=self.skipped_policy_captures,
            skipped_helper_captures=self.skipped_helper_captures)
        for name in ('observation_cache_hits', 'observation_cache_misses', 'native_step_seconds', 'native_step_count'):
            if hasattr(self.env, name):
                result[name] = getattr(self.env, name)
        return result
