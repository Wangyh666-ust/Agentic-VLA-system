"""Visual finish service contracts using the existing GPU-free fake environment."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import test_service_contract as existing
from scene_demo import joint_home, safe_exit_contacts
from scene_demo.finish_v1 import runtime

service = existing.service
catalog = existing.catalog
REFERENCE = {
    'homeq': [0.0] * 7, 'finger_home': [0.04, -0.04],
    'home_joint_speed': [0.0] * 7, 'origin_sha256': 'fake',
    'qpos_indexes': list(range(7)), 'qvel_indexes': list(range(7)),
    'finger_qpos_indexes': [7, 8], 'finger_qvel_indexes': [7, 8],
}


class FakeMonitor:
    def __init__(self, env, reference, output_dir, context, guarded_step,
                 on_action, independent_ready, budget, client=None):
        self.env = env
        self.context = context
        self.guarded_step = guarded_step
        self.on_action = on_action
        self.independent_ready = independent_ready
        self.closed = False

    def record(self, step):
        pass

    def poll(self, step, candidate=False, force=False):
        return None

    def status(self):
        return {'phase': 'observing', 'calls': 0}

    def close(self):
        self.closed = True


class VisualFinishServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.services = []
        self.addCleanup(self.stop_services)
        self.patches = [
            mock.patch.object(joint_home, 'capture_home_reference', return_value=dict(REFERENCE)),
            mock.patch.object(service, '_seed_everything', return_value=None),
            mock.patch.object(service, '_save_video', return_value=None),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def stop_services(self):
        for svc in self.services:
            svc.stop()

    def make(self):
        svc = service.SceneService(
            run_root=self.temp.name,
            env_factory=lambda **kwargs: existing._FakeEnv(),
            policy_loader=lambda s: setattr(s._v1, '_n_action_steps', 10),
            action_function=lambda batch: np.zeros(7),
            batch_builder=lambda obs, instruction: {},
            completion_mode='native', grasp_guard_mode='off', finish_mode='visual',
        )
        self.services.append(svc)
        svc.start()
        session = svc.create_session('goal_table', seed=0, init_state_index=0)
        self.assertTrue(session['ok'], session)
        return svc, session['session_id']

    def payload(self, svc, sid, rid, **extra):
        payload = {'session_id': sid, 'scene_version': svc.session(sid)['scene_version'],
                   'request_id': rid, 'capability_ids': ['bowl_to_plate'],
                   'decision': 'execute', 'budget_per_subgoal': 20,
                   'rationale': 'planner rationale is not the user request'}
        payload.update(extra)
        return payload

    def track_plan_end(self, svc):
        done = threading.Event()
        original = svc._do_run_plan
        def execute(plan):
            try:
                return original(plan)
            finally:
                done.set()
        svc._do_run_plan = execute
        return done

    def test_health_context_idempotency_conflict_and_tombstone(self):
        with self.assertRaises(ValueError):
            service.SceneService(run_root=self.temp.name, finish_mode='invalid')
        svc, sid = self.make()
        self.assertEqual(svc.health()['finish_mode'], 'visual')
        original = 'Please place the bowl on the plate.'
        first = svc.bind_request_context('same', sid, original)
        same = svc.bind_request_context('same', sid, original)
        self.assertTrue(first['ok'])
        self.assertFalse(first['idempotent'])
        self.assertTrue(same['idempotent'])
        conflict = svc.bind_request_context('same', sid, original + ' Changed')
        self.assertEqual(conflict['reason'], 'request_conflict')
        self.assertTrue(svc.cancel_request('cancelled-before-plan', sid)['ok'])
        denied = svc.bind_request_context('cancelled-before-plan', sid, original)
        self.assertEqual(denied['reason'], 'cancelled')

    def test_submit_requires_binding_and_preserves_original_text(self):
        svc, sid = self.make()
        payload = self.payload(svc, sid, 'bound', decision='clarify', capability_ids=[])
        denied = svc.submit_plan(payload)
        self.assertFalse(denied['ok'])
        self.assertEqual(denied['reason'], 'invalid_request')
        original = 'Original request, independent of the planner rationale.'
        self.assertTrue(svc.bind_request_context('bound', sid, original)['ok'])
        done = self.track_plan_end(svc)
        self.assertTrue(svc.submit_plan(payload)['ok'])
        self.assertTrue(done.wait(2), 'worker did not finish bound request')
        self.assertEqual(svc._plans['bound'].user_request, original)
        self.assertNotEqual(svc._plans['bound'].user_request, payload['rationale'])

    def test_monitor_guard_post_action_log_and_independent_gate(self):
        svc, sid = self.make()
        session = svc._sessions[sid]
        plan = service.PlanRecord('direct', sid, self.payload(svc, sid, 'direct'))
        plan.user_request = 'Put the bowl on the plate, please.'
        job = service.JobRecord('job-direct', 'direct', sid, 'bowl_to_plate', Path(self.temp.name, 'direct'))
        job.run_dir.mkdir()
        with mock.patch.object(runtime, 'FinishMonitor', FakeMonitor):
            monitor = svc._make_finish_monitor(session, plan, job, svc._env,
                                              catalog.CAPABILITIES['bowl_to_plate'], 20, [], job.run_dir / 'latest.png')
        self.assertEqual(monitor.context['user_request'], plan.user_request)
        self.assertEqual(monitor.context['current_subtask']['target'], 'akita_black_bowl_1')
        called = []
        plan.cancel_event.set()
        with self.assertRaises(service.SceneError) as caught:
            monitor.guarded_step(lambda: called.append(True))
        self.assertEqual(caught.exception.reason, 'cancelled')
        self.assertEqual(called, [])
        plan.cancel_event.clear()
        before = svc._total_steps
        svc._last_obs = 'old observation'
        monitor.on_action({'phase': 'release', 'finger_qpos': np.array([0.04, -0.04])})
        self.assertEqual(svc._total_steps, before + 1)
        self.assertEqual(job.total_steps, before + 1)
        self.assertEqual(session.total_steps, before + 1)
        self.assertEqual(svc._last_obs, {})
        rows = [json.loads(line) for line in (job.run_dir / 'finish_actions.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['finger_qpos'], [0.04, -0.04])
        svc._env._inner.steps = 3
        with mock.patch.object(joint_home, 'read_robot_state', return_value={'finger_qpos': [0.04, -0.04]}), \
             mock.patch.object(joint_home, 'home_metrics', return_value={'ready': True}) as metrics, \
             mock.patch.object(safe_exit_contacts, 'contacts', return_value=[]) as contacts:
            self.assertTrue(monitor.independent_ready())
        metrics.assert_called_once()
        contacts.assert_called_once_with(svc._env)

    def test_native_five_true_cannot_bypass_visual_failure(self):
        class RejectAfterEight(FakeMonitor):
            def poll(self, step, candidate=False, force=False):
                return {'ok': False, 'reason': 'final visual failed'} if step >= 8 else None
        svc, sid = self.make()
        self.assertTrue(svc.bind_request_context('visual-reject', sid, 'Put bowl on plate')['ok'])
        done = self.track_plan_end(svc)
        with mock.patch.object(runtime, 'FinishMonitor', RejectAfterEight):
            self.assertTrue(svc.submit_plan(self.payload(svc, sid, 'visual-reject'))['ok'])
            self.assertTrue(done.wait(2), 'visual failure did not terminate job')
        plan = svc.plan('visual-reject')
        job = svc.job(plan['job_ids'][0])
        self.assertFalse(job['success'])
        self.assertEqual(job['steps'], 8)
        self.assertEqual(job['ended_reason'], 'final visual failed')
        rows = [json.loads(line) for line in (Path(job['run_dir']) / 'events.jsonl').read_text().splitlines()]
        self.assertTrue(all(all(row['declared_predicates'].values()) for row in rows[-5:]))

    def test_cancel_during_blocked_poll_acknowledged_before_api_returns(self):
        entered = threading.Event()
        release = threading.Event()
        class BlockingMonitor(FakeMonitor):
            def poll(self, step, candidate=False, force=False):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('fake API release timeout')
                self.guarded_step(lambda: self.env.step(np.zeros(7)))
                return {'ok': False, 'reason': 'unexpected continuation'}
        svc, sid = self.make()
        self.assertTrue(svc.bind_request_context('api-cancel', sid, 'Put bowl on plate')['ok'])
        done = self.track_plan_end(svc)
        ack_done = threading.Event()
        acknowledgements = []
        def cancel():
            acknowledgements.append(svc.cancel_request('api-cancel', sid))
            ack_done.set()
        with mock.patch.object(runtime, 'FinishMonitor', BlockingMonitor):
            try:
                self.assertTrue(svc.submit_plan(self.payload(svc, sid, 'api-cancel'))['ok'])
                self.assertTrue(entered.wait(2), 'poll was not entered')
                steps_before_cancel = svc._env._inner.steps
                worker = threading.Thread(target=cancel, daemon=True)
                started = time.monotonic()
                worker.start()
                self.assertTrue(ack_done.wait(2), 'cancel blocked on visual API')
                self.assertLess(time.monotonic() - started, 2)
                self.assertFalse(release.is_set())
                self.assertTrue(acknowledgements[0]['ok'])
            finally:
                release.set()
            self.assertTrue(done.wait(2), 'cancelled job did not finish after API release')
        job = svc.job(svc.plan('api-cancel')['job_ids'][0])
        self.assertEqual(job['state'], 'cancelled')
        self.assertEqual(job['ended_reason'], 'cancelled')
        self.assertFalse(job['success'])
        self.assertEqual(svc._env._inner.steps, steps_before_cancel)


if __name__ == '__main__':
    unittest.main(verbosity=2)
