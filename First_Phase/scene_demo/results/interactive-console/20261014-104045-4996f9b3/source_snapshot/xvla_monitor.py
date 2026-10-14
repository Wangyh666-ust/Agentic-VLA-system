"""X-VLA task-local release evidence around the unchanged finish runtime."""
import copy
import hashlib
import json
from pathlib import Path
from scene_demo.finish_v1.runtime import FinishMonitor
from scene_demo.xvla_process_evidence import ReleaseSampler, capture_frames, VIEWS

RELEASE_CHECK_LIMIT = 4

class XvlaFinishMonitor(FinishMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_sampler = ReleaseSampler(list(self.reference['finger_home']))
        self.pending_releases = []
        self.release_checks = 0
        self.release_events = 0
        self._observed_vla_steps = set()

    def _append_process(self, filename, row):
        self.output_dir.mkdir(parents=True,exist_ok=True)
        with (self.output_dir/filename).open('a',encoding='utf-8',newline='\n') as handle:
            handle.write(json.dumps(row,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')

    def _observe_release(self):
        robot=self.port.read_robot()
        event=self.release_sampler.observe(list(robot['finger_qpos']),self.port.capture)
        if event is not None:
            frames=capture_frames(event['capture'])
            event['images']=[{'view':f['view'],'path':str(f['rgb_path']),
                             'sha256':hashlib.sha256(Path(f['rgb_path']).read_bytes()).hexdigest()} for f in frames]
            self.pending_releases.append(event)
            self.release_events += 1
            self.call_client.remember_release(event['capture'], self._release_context(event['capture']), event['event_id'])
            self._append_process('process_events.jsonl',event)

    def _release_context(self, capture):
        robot=capture['robot']
        ctx=copy.deepcopy(self.context)
        ctx['sensor_frames']=[{'frame':i,'view':view,'timestamp':float(capture['timestamp'])} for i,view in enumerate(VIEWS)]
        ctx['actuator_measurements']={'timestamp':float(capture['timestamp']),
            'finger_qpos':list(robot['finger_qpos']),
            'standard_open_finger_qpos':list(self.reference['finger_home']),
            'max_abs_arm_qvel':max(abs(float(x)) for x in robot['arm_qvel'])}
        return ctx

    def record(self, step):
        if self.terminal is not None:
            return
        if step not in self._observed_vla_steps:
            self._observe_release()
            self._observed_vla_steps.add(step)
        super().record(step)
        if self.history and hasattr(self.call_client, 'remember_observation'):
            capture = self.history[-1]
            self.call_client.remember_observation(capture, self._release_context(capture))

    def _on_action(self, row):
        super()._on_action(row)
        self._observe_release()

    def _check(self, context, observations):
        from scene_demo.xvla_finish import XvlaQwenClient
        if not isinstance(self.call_client,XvlaQwenClient):
            raise ValueError('X monitor requires XvlaQwenClient')
        while self.pending_releases:
            event=self.pending_releases.pop(0)
            if self.release_checks >= RELEASE_CHECK_LIMIT:
                self.call_client.note_release_status(event['event_id'], 'unchecked_due_budget')
                self._append_process('process_checks.jsonl',{'event_id':event['event_id'],
                    'status':'unchecked_due_budget','capture':event['capture']})
                continue
            capture=event['capture'];frames=capture_frames(capture);robot=capture['robot']
            ctx=self._release_context(capture)
            self.call_client.audit_dir=self.output_dir/'model_requests'
            self.call_client.observe_capture(capture)
            self.guarded_step(lambda:None)
            self.release_checks += 1
            try:
                meta=self.call_client.analyze_release(ctx,[str(f['rgb_path']) for f in frames],event['event_id'])
            except Exception as error:
                self._append_process('process_checks.jsonl',{'event_id':event['event_id'],'status':'error',
                    'error_type':type(error).__name__,'error':str(error),'context':ctx,'capture':capture})
                raise
            self.guarded_step(lambda:None)
            self._append_process('process_checks.jsonl',{'event_id':event['event_id'],'status':'classified',
                'context':ctx,'capture':capture,'metadata':meta})
        return super()._check(context, observations)

    def status(self):
        result=super().status()
        result.update(current_view_calls=self.calls,release_evidence_calls=self.release_checks,
            calls=self.calls+self.release_checks,release_events=self.release_events,
            process_evidence=self.call_client.process_evidence_snapshot() if hasattr(self.call_client,'process_evidence_snapshot') else [])
        if hasattr(self.call_client, '_calls'):
            result['actual_visual_requests'] = self.call_client._calls
        return result
