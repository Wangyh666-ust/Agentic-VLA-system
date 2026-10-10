import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.environ.get('FYP_PATH', 'D:/FYP/First_Phase'))
from scene_demo import catalog
from scene_demo.finish_v1 import control, runtime, sensors, vision


ALL_TRUE = {
    **{key: True for key in vision._BOOL_FIELDS},
    'boxes': {}, 'evidence': [], 'unknown_reasons': [],
}


class FakePort:
    def __init__(self, env, reference, output_dir, step_callback=None):
        self.reference = reference
        self.step_callback = step_callback
        self.phase = None
        self.timestamp = 0.0
        self.capture_version = 0
        self.steps = 0
        self.closed = False
        self.robot = {
            'finger_qpos': np.array(reference['finger_home'], dtype=float),
            'arm_qvel': np.zeros(7),
            'arm_qpos': np.zeros(7),
        }

    def capture(self):
        return {
            'timestamp': self.timestamp,
            'robot': {key: [float(x) for x in value] for key, value in self.robot.items()},
            'frames': [
                {'view': view, 'rgb_path': f'{self.timestamp}_{self.capture_version}_{view}.png'}
                for view in ('agentview', 'wrist')
            ],
        }

    def read_robot(self):
        return {key: [float(x) for x in value] for key, value in self.robot.items()}

    def step(self, qtarget, finger_target=None):
        self.steps += 1
        self.timestamp += 0.05
        self.robot['arm_qpos'] = np.asarray(qtarget, dtype=float)
        if finger_target is not None:
            self.robot['finger_qpos'] = np.asarray(finger_target, dtype=float)
        row = {'phase': self.phase, **{key: value.copy() for key, value in self.robot.items()},
               'qtarget': np.asarray(qtarget)}
        self.step_callback(row)
        return self.read_robot()

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, verdict=None, after=None):
        self.verdict = copy.deepcopy(ALL_TRUE if verdict is None else verdict)
        self.after = after
        self.calls = []

    def analyze(self, context, image_paths):
        self.calls.append((copy.deepcopy(context), list(image_paths)))
        if self.after:
            self.after()
        return {'verdict': copy.deepcopy(self.verdict), 'usage': {'total_tokens': 1},
                'status': 'ok', 'model': 'fake', 'http_status': 200,
                'request_sha256': 'request', 'response_sha256': 'response',
                'call_index': len(self.calls)}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = mock.patch.object(sensors, 'SensorPort', FakePort)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.cancelled = False
        self.rows = []
        self.reference = {'finger_home': [0.04, -0.04]}
        self.context = runtime.build_context(catalog.CAPABILITIES['bowl_to_plate'], 'Put bowl on plate')

    def guard(self, fn):
        if self.cancelled:
            raise RuntimeError('cancelled')
        return fn()

    def monitor(self, client=None, ready=lambda: True, on_action=None):
        return runtime.FinishMonitor(
            None, self.reference, Path(self.temp.name), self.context,
            self.guard, self.rows.append if on_action is None else on_action,
            ready, 100, FakeClient() if client is None else client,
        )

    def record_pair(self, monitor):
        for step in (5, 10):
            monitor.port.timestamp = float(step)
            monitor.record(step)

    def test_catalog_context_and_compound_rejection(self):
        bowl = self.context['current_subtask']
        self.assertEqual((bowl['operation'], bowl['target'], bowl['destination']),
                         ('place', 'akita_black_bowl_1', 'plate_1'))
        stove = runtime.build_context(catalog.CAPABILITIES['stove_on'], 'Turn on stove')
        self.assertEqual(stove['current_subtask']['operation'], 'turn')
        self.assertIsNone(stove['current_subtask']['destination'])
        self.assertEqual(stove['criteria']['desired_state'], 'on')
        vision.build_prompt(stove)
        with self.assertRaises(ValueError):
            runtime.build_context({'goals': [['on', 'a', 'b'], ['turnon', 'stove']]}, 'do both')

    def test_record_only_every_five_with_two_timestamps(self):
        monitor = self.monitor()
        for step in (1, 5, 6, 10):
            monitor.port.timestamp = float(step)
            monitor.record(step)
        self.assertEqual([cap['timestamp'] for cap in monitor.history], [5.0, 10.0])
        self.assertEqual(monitor.calls, 0)
        self.assertEqual(monitor.port.steps, 0)

    def test_check_frame_binding_and_measurement_whitelist(self):
        monitor = self.monitor()
        self.record_pair(monitor)
        monitor._check(self.context, [])
        ctx, paths = monitor.call_client.calls[-1]
        self.assertEqual(len(paths), 4)
        self.assertEqual([frame['frame'] for frame in ctx['sensor_frames']], list(range(4)))
        self.assertEqual(set(ctx), {'user_request', 'current_subtask', 'criteria', 'sensor_frames', 'actuator_measurements'})
        self.assertEqual(ctx['actuator_measurements']['standard_open_finger_qpos'], [0.04, -0.04])
        for frame, path in zip(ctx['sensor_frames'], paths):
            self.assertEqual(set(frame), {'frame', 'view', 'timestamp'})
            self.assertEqual(path, f"{frame['timestamp']}_0_{frame['view']}.png")
        for forbidden in ('gold', 'native', 'contacts', 'arm_qpos', 'robot'):
            self.assertNotIn(forbidden, json.dumps(ctx))
        monitor.port.timestamp = 15.0
        monitor._check(self.context, [])
        ctx, paths = monitor.call_client.calls[-1]
        self.assertEqual(len(paths), 6)
        self.assertEqual([frame['frame'] for frame in ctx['sensor_frames']], list(range(6)))
        json.loads(Path(self.temp.name, 'checks.jsonl').read_text().splitlines()[0])

    def test_simple_checker_receives_two_synced_three_view_samples(self):
        client = FakeClient()
        client.simple_handoff = True
        monitor = self.monitor(client)
        monitor.port.phase = 'release'
        original_capture = monitor.port.capture
        def capture():
            cap = original_capture()
            cap['verification_frames'] = [cap['frames'][0], {'view': 'sideview', 'rgb_path': 'side.png'}]
            return cap
        monitor.port.capture = capture
        for step in (5, 10, 15):
            monitor.port.timestamp = float(step)
            monitor.record(step)
        monitor._check(self.context, [])
        ctx, paths = client.calls[-1]
        self.assertEqual(len(paths), 6)
        self.assertEqual([fr['view'] for fr in ctx['sensor_frames']], ['agentview', 'sideview', 'wrist'] * 2)
        self.assertEqual([fr['timestamp'] for fr in ctx['sensor_frames']], [10.] * 3 + [15.] * 3)
        self.assertEqual(ctx['actuator_measurements']['timestamp'], 15.)
        monitor.port.phase = 'confirm'
        monitor._check(self.context, [])
        ctx, paths = client.calls[-1]
        self.assertEqual(len(paths), 4)
        self.assertEqual([fr['view'] for fr in ctx['sensor_frames']], ['agentview','sideview'] * 2)
        monitor.port.phase = 'home'
        monitor._check(self.context, [])
        ctx, paths = client.calls[-1]
        self.assertEqual(len(paths),2)
        self.assertEqual([fr['view'] for fr in ctx['sensor_frames']],['agentview'] * 2)

    def test_latest_capture_overwrites_same_timestamp(self):
        monitor = self.monitor()
        monitor.port.timestamp = 5.0
        monitor.record(5)
        monitor.port.capture_version = 2
        selected = monitor._collect_observations([])
        self.assertEqual(len(selected), 1)
        self.assertIn('_2_agentview', selected[0]['frames'][0]['rgb_path'])

    def test_persistent_candidate_not_per_step_and_periodic_check(self):
        raw = copy.deepcopy(ALL_TRUE)
        raw['at_destination'] = False
        monitor = self.monitor(FakeClient(raw))
        self.record_pair(monitor)
        for step in range(11, 20):
            self.assertIsNone(monitor.poll(step, candidate=True))
        self.assertEqual(monitor.calls, 1)
        monitor.poll(36, candidate=True)
        self.assertEqual(monitor.calls, 2)

    def test_unknown_does_not_control_and_status_is_json(self):
        raw = copy.deepcopy(ALL_TRUE)
        raw['target_visible'] = None
        monitor = self.monitor(FakeClient(raw))
        self.record_pair(monitor)
        with mock.patch.object(control, 'run_finish') as finish:
            outcome = monitor.poll(11, candidate=True)
        finish.assert_not_called()
        self.assertFalse(outcome['ok'])
        self.assertEqual(monitor.port.steps, 0)
        self.assertTrue(monitor.port.closed)
        status = json.loads(json.dumps(monitor.status()))
        self.assertEqual(status['phase'], 'blocked')
        self.assertEqual(set(status['terminal']), {'ok', 'reason', 'checks'})

    def test_closed_fingers_veto_visual_all_true(self):
        monitor = self.monitor()
        self.record_pair(monitor)
        monitor.port.robot['finger_qpos'] = np.array([0.003, -0.006])
        fused = monitor._check(self.context, [])
        self.assertFalse(fused['released'])
        self.assertFalse(fused['clear_of_target'])
        self.assertEqual(monitor._last_decision['state'], 'needs_release')

    def test_api_cancellation_after_response_has_zero_actions(self):
        def cancel():
            self.cancelled = True
        monitor = self.monitor(FakeClient(after=cancel))
        self.record_pair(monitor)
        with mock.patch.object(control, 'run_finish') as finish:
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                monitor.poll(11, candidate=True)
        finish.assert_not_called()
        self.assertEqual(monitor.calls, 1)
        self.assertEqual(monitor.port.steps, 0)
        self.assertTrue(monitor.port.closed)

    def test_finish_uses_verdict_and_five_actual_posts(self):
        monitor = self.monitor()
        self.record_pair(monitor)
        def finish(port, context, initial_verdict, decide_fn, check_fn, emit):
            self.assertIn('target_visible', initial_verdict)
            self.assertNotIn('state', initial_verdict)
            self.assertEqual(decide_fn(context, initial_verdict)['state'], 'complete')
            port.phase = 'home'
            for _ in range(5):
                port.step(np.zeros(7), np.array(self.reference['finger_home']))
            self.assertEqual(monitor.status()['phase'], 'home')
            return {'ok': True, 'reason': 'complete', 'array': np.zeros(2)}
        with mock.patch.object(control, 'run_finish', side_effect=finish):
            outcome = monitor.poll(11, candidate=True)
        self.assertTrue(outcome['ok'])
        self.assertEqual(monitor.independent_streak, 5)
        self.assertEqual(monitor.port.steps, 5)
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(monitor.helper_counts['home'], 5)
        self.assertTrue(monitor.port.closed)
        self.assertEqual(json.loads(json.dumps(monitor.status()))['phase'], 'done')

    def test_four_posts_fail_and_callback_errors_propagate(self):
        monitor = self.monitor()
        self.record_pair(monitor)
        def finish(port, *args, **kwargs):
            port.phase = 'home'
            for _ in range(4):
                port.step(np.zeros(7))
            return {'ok': True}
        with mock.patch.object(control, 'run_finish', side_effect=finish):
            outcome = monitor.poll(11, candidate=True)
        self.assertFalse(outcome['ok'])
        self.assertEqual(outcome['reason'], 'independent_finish_failed')
        self.assertEqual(monitor.independent_streak, 4)
        def broken_measurement():
            raise RuntimeError('measurement failed')
        second = self.monitor(ready=broken_measurement)
        second.port.phase = 'release'
        before = len(self.rows)
        with self.assertRaisesRegex(RuntimeError, 'measurement failed'):
            second.port.step(np.zeros(7))
        self.assertEqual(len(self.rows), before + 1)
        def broken_action(row):
            raise RuntimeError('callback failed')
        third = self.monitor(on_action=broken_action)
        with self.assertRaisesRegex(RuntimeError, 'callback failed'):
            third.port.step(np.zeros(7))
        self.assertEqual(third.port.steps, 1)


if __name__ == '__main__':
    unittest.main()
