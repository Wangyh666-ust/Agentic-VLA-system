import json
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scene_demo.finish_v1 import simple_checker as checker, control


class SimpleCheckerTests(unittest.TestCase):
    def setUp(self):
        self.ctx = {'user_request': '把酒瓶放到架子上',
                    'current_subtask': {'operation': 'place', 'target': 'wine_bottle_1', 'destination': 'wine_rack_1_top_region'},
                    'criteria': {}, 'sensor_frames': [{'frame': 0, 'view': 'sideview', 'timestamp': 1.}],
                    'actuator_measurements': {'timestamp': 1., 'finger_qpos': [.01, -.01], 'standard_open_finger_qpos': [.04, -.04]}}
        self.v = {'target_visible': True, 'destination_visible': True, 'at_destination': True,
                  'supported': True, 'released': None, 'stable': None, 'operation_achieved': True,
                  'clear_of_target': None, 'boxes': {}, 'evidence': [], 'unknown_reasons': []}

    def test_goal_reached_unknown_hold_starts_finish(self):
        self.assertTrue(checker.decide(self.ctx, self.v)['allow_release'])

    def test_closed_fingers_cannot_veto_placement(self):
        out = checker.reconcile(self.ctx, self.v, {'finger_qpos': [.01, -.01]}, {'finger_home': [.04, -.04]})
        self.assertTrue(out['operation_achieved'])
        self.assertTrue(out['supported'])
        self.assertTrue(checker.decide(self.ctx, out)['allow_release'])

    def test_unachieved_goal_never_opens(self):
        self.v['operation_achieved'] = False
        self.assertFalse(checker.decide(self.ctx, self.v)['allow_release'])

    def test_encoder_sync_required(self):
        self.ctx['actuator_measurements']['timestamp'] = .9
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            checker.reconcile(self.ctx, self.v, {'finger_qpos': [.01, -.01]}, {'finger_home': [.04, -.04]})

    def test_decoder_missing_fields_never_claims_success(self):
        client = checker.SimpleQwenClient(max_calls=0)
        result = client.parse_verdict('{"holding_or_contact":true,"reason":"目标遮挡"}')
        self.assertIsNone(result['supported'])
        self.assertFalse(checker.decide(self.ctx, result)['allow_release'])

    def test_goal_alone_can_trigger_without_support_conditions(self):
        client = checker.SimpleQwenClient(max_calls=0)
        result = client.parse_verdict('{"goal_achieved":true,"holding_or_contact":true,"reason":"瓶子已在架上"}')
        self.assertIsNone(result['supported'])
        self.assertTrue(checker.decide(self.ctx, result)['allow_release'])

    def test_prompt_requests_only_visual_goal_and_explains_encoder_authority(self):
        prompt = checker.build_prompt(self.ctx)
        self.assertNotIn('仅悬在', prompt)
        self.assertNotIn('supported', prompt)
        self.assertNotIn('boxes', prompt)
        self.assertIn('goal_achieved', prompt)
        self.assertNotIn('holding_or_contact', prompt)
        self.assertIn('遮挡、二维投影重叠', prompt)
        self.assertIn('这仅说明检查器的用途', prompt)
        self.assertIn('put the wine bottle on the rack',prompt)
        self.assertNotIn('wine_rack_1_top_region',prompt)
        self.assertNotIn('top region',prompt)
        self.assertIn('结束状态检查器', prompt)
        self.assertIn('遗漏最后的松手', prompt)
        self.assertNotIn('放架任务', prompt)
        self.assertNotIn('放盘任务', prompt)

    def test_prompt_drops_stage_labels_but_retains_synchronized_encoders(self):
        self.ctx['current_subtask']['phase'] = 'post_execution'
        self.ctx['actuator_measurements']['stage'] = 'post_execution'
        self.ctx['actuator_measurements']['max_abs_arm_qvel'] = .9
        prompt = checker.build_prompt(self.ctx)
        payload = json.loads(prompt.splitlines()[-1])
        self.assertNotIn('post_execution', prompt)
        self.assertNotIn('stage', payload['actuator_measurements'])
        self.assertNotIn('phase', payload['current_subtask'])
        measurements = payload['actuator_measurements']
        self.assertEqual({key: measurements[key] for key in ('timestamp', 'finger_qpos', 'standard_open_finger_qpos')}, {
            'timestamp': 1., 'finger_qpos': [.01, -.01],
            'standard_open_finger_qpos': [.04, -.04],
        })
        self.assertFalse(measurements['at_standard_open'])
        self.assertEqual(measurements['position_unit'], 'm')
        self.assertAlmostEqual(measurements['max_standard_open_error_mm'], 30.)
        self.assertIn('同一采样时刻', prompt)
        self.assertIn('第一人称', prompt)
        self.assertIn('第三人称', prompt)

    def test_stove_visual_definition_is_static_and_does_not_claim_completion(self):
        from scene_demo.finish_v1.runtime import build_context
        from scene_demo import catalog
        for operation in ('turnon', 'turnoff'):
            capability = copy.deepcopy(catalog.CAPABILITIES['stove_on'])
            capability['goals'][0][0] = operation
            ctx = build_context(capability, '操作灶台')
            ctx.update(sensor_frames=self.ctx['sensor_frames'], actuator_measurements=self.ctx['actuator_measurements'])
            prompt = checker.build_prompt(ctx)
            payload = json.loads(prompt.splitlines()[-1])
            self.assertIn('红色加热圈', payload['visual_state_definition']['on'])
            self.assertIn('熄灭', payload['visual_state_definition']['off'])
            self.assertNotIn('goal_achieved', payload)
            self.assertNotIn('post_execution', prompt)
        self.assertNotIn('visual_state_definition', json.loads(checker.build_prompt(self.ctx).splitlines()[-1]))

    def fused_open(self, goal=True, contact=True):
        ctx = copy.deepcopy(self.ctx)
        ctx['actuator_measurements']['finger_qpos'] = [.04, -.04]
        raw = checker.SimpleQwenClient(max_calls=0).parse_verdict(json.dumps({
            'goal_achieved': goal, 'holding_or_contact': contact, 'reason': '目标画面依据'}))
        return checker.reconcile(ctx, raw, {'finger_qpos': [.04, -.04]}, {'finger_home': [.04, -.04]})

    def test_open_encoder_overrides_visual_contact_and_occlusion(self):
        for contact in (True, False, None):
            with self.subTest(contact=contact):
                fused = self.fused_open(contact=contact)
                self.assertTrue(fused['released'])
                self.assertIsNone(fused['clear_of_target'])  # does not invent physical clearance
                self.assertEqual(checker.decide(self.ctx, fused)['state'], 'complete')

    def test_encoder_cannot_overwrite_negative_or_unknown_goal(self):
        for goal, expected in ((False, 'incomplete'), (None, 'unknown')):
            fused = self.fused_open(goal=goal)
            self.assertTrue(fused['released'])
            self.assertEqual(checker.decide(self.ctx, fused)['state'], expected)
            self.assertFalse(checker.decide(self.ctx, fused)['allow_release'])

    def test_visual_release_claim_cannot_override_closed_encoder(self):
        raw = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":false}')
        self.assertIsNone(raw['released'])
        fused = checker.reconcile(self.ctx, raw, {'finger_qpos': [.01, -.01]}, {'finger_home': [.04, -.04]})
        self.assertFalse(fused['released'])
        self.assertEqual(checker.decide(self.ctx, fused)['state'], 'needs_release')

    def test_new_goal_only_response_requires_measured_opening(self):
        raw = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"reason":"在目标处"}')
        self.assertIsNone(raw['released'])
        self.assertEqual(checker.decide(self.ctx, raw)['state'], 'needs_release')

    def test_every_view_must_match_encoder_timestamp(self):
        self.ctx['sensor_frames'].append({'frame': 1, 'view': 'wrist', 'timestamp': .9})
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            checker.encoder_observation(self.ctx)

    def test_missing_nonfinite_or_malformed_encoder_never_proves_open(self):
        for bad in ([.04], [float('nan'), -.04], [float('inf'), -.04], [True, -.04]):
            ctx = copy.deepcopy(self.ctx)
            ctx['actuator_measurements']['finger_qpos'] = bad
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                checker.encoder_observation(ctx)
        for key in ('finger_qpos', 'standard_open_finger_qpos', 'timestamp'):
            ctx = copy.deepcopy(self.ctx)
            del ctx['actuator_measurements'][key]
            with self.subTest(missing=key), self.assertRaises(KeyError):
                checker.encoder_observation(ctx)

    def test_nonfinite_timestamp_and_mismatched_feedback_rejected(self):
        for timestamp in (float('nan'), float('inf'), True):
            ctx = copy.deepcopy(self.ctx)
            ctx['actuator_measurements']['timestamp'] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                checker.encoder_observation(ctx)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            checker.reconcile(self.ctx, self.v, {'finger_qpos': [.04, -.04]}, {'finger_home': [.04, -.04]})

    def test_original_open_tolerance_is_retained(self):
        for error, opened in ((.0019, True), (.0021, False)):
            ctx = copy.deepcopy(self.ctx)
            ctx['actuator_measurements']['finger_qpos'] = [.04 - error, -.04]
            self.assertEqual(checker.encoder_observation(ctx)['at_standard_open'], opened)

    def test_first_action_really_opens_without_settle_gate(self):
        class Port:
            reference = {'finger_home': [.04, -.04]}
            def __init__(self): self.commands = []
            def read_robot(self): return {'arm_qpos': np.zeros(7), 'finger_qpos': [.01, -.01]}
            def start(self): pass
            def step(self, q, f):
                self.commands.append(list(f))
                return {'arm_qpos': np.zeros(7), 'finger_qpos': f, 'arm_qvel': np.ones(7) * .03}
            def capture(self): return {}
            def close(self): pass
        port = Port()
        fail_after_open = dict(self.v, operation_achieved=False, at_destination=False)
        with patch.object(control, 'plan_retreat', return_value={'ok':False}):
            result = control.run_finish(port, self.ctx, self.v, checker.decide,
                                        lambda *_: fail_after_open, lambda _: None, simple_handoff=True)
        self.assertEqual(port.commands[0], [.04, -.04])
        self.assertEqual(result['release_settle_actions'], 0)
        self.assertEqual(result['release_open_actions'], 2)
        self.assertEqual(result['reason'], 'retreat plan blocked')

    def run_retreat(self, follow):
        from scene_demo.finish_v1.tests.test_core import FakePort
        port = FakePort(follow=follow)
        path = [np.full(7, i * .025) for i in range(21)]
        initial = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":true}')
        final = self.fused_open()
        with patch.object(control, 'plan_retreat', return_value={'ok': True, 'waypoints': path}), patch.object(control, 'screen_shapes', return_value={'ok': True}), patch.object(control.jh, 'home_metrics', side_effect=lambda ref, st: {'ready': np.max(np.abs(st['arm_qpos'])) < .004}):
            result = control.run_finish(port, self.ctx, initial, checker.decide, lambda *_: final, lambda _: None, simple_handoff=True)
        return result

    def test_progressing_retreat_can_exceed_old_cap(self):
        result = self.run_retreat(.2)
        self.assertGreater(result['counts']['movement'], 120)
        self.assertGreater(result['counts']['home'], 0, result)
        self.assertTrue(result['ready'], result)

    def test_stalled_retreat_stops_without_starting_home(self):
        result = self.run_retreat(0.)
        self.assertEqual(result['reason'], 'movement no progress')
        self.assertEqual(result['counts']['home'], 0)

    def test_initial_open_checker_acceptance_still_runs_release_retreat_home(self):
        from scene_demo.finish_v1.tests.test_core import FakePort
        port = FakePort()
        port.f = np.asarray(port.reference['finger_home'])
        accepted = self.fused_open()
        with patch.object(control, 'plan_retreat', return_value={'ok': True, 'waypoints': [np.zeros(7), np.full(7, .04)]}), patch.object(control, 'screen_shapes', return_value={'ok': True}), patch.object(control.jh, 'home_metrics', side_effect=lambda ref, st: {'ready': np.max(np.abs(st['arm_qpos'])) < .004}):
            result = control.run_finish(port, self.ctx, accepted, checker.decide, lambda *_: accepted, lambda _: None, simple_handoff=True)
        self.assertTrue(result['ok'], result)
        for phase in ('release', 'movement', 'home'):
            self.assertGreater(result['counts'][phase], 0)

    def test_atomic_finish_latches_handoff_but_final_failure_is_retained(self):
        from scene_demo.finish_v1.tests.test_core import FakePort
        port = FakePort()
        initial = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":true}')
        rejected = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":false,"holding_or_contact":false}')
        check = unittest.mock.Mock(return_value=rejected)
        with patch.object(control,'plan_retreat',return_value={'ok':True,'waypoints':[np.zeros(7),np.full(7,.04)]}), patch.object(control,'screen_shapes',return_value={'ok':True}), patch.object(control.jh,'home_metrics',side_effect=lambda ref,st:{'ready':np.max(np.abs(st['arm_qpos']))<.004}):
            result = control.run_finish(port,self.ctx,initial,checker.decide,check,lambda _:None,simple_handoff=True)
        self.assertGreater(result['counts']['home'],0)
        check.assert_called_once()
        self.assertFalse(result['ready'])
        self.assertEqual(result['reason'],'final visual failed')

    def test_near_open_feedback_allows_withdrawal_but_not_automatic_completion(self):
        from scene_demo.finish_v1.tests.test_core import FakePort
        class RackPort(FakePort):
            def step(self, q, f=None):
                out = super().step(q, f)
                if self.phase == 'release':
                    self.q = np.asarray(q) + .018
                    out['arm_qpos'] = self.q.copy()
                self.f = np.asarray([.035, -.039])
                out['finger_qpos'] = self.f.copy()
                out['finger_qvel'] = [0., 0.]
                return out
        port = RackPort()
        initial = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":true}')
        final = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":false}')
        with patch.object(control, 'plan_retreat', return_value={'ok': True, 'waypoints':[np.zeros(7),np.full(7,.04)]}), patch.object(control, 'screen_shapes', return_value={'ok':True}):
            result = control.run_finish(port,self.ctx,initial,checker.decide,lambda *_:final,lambda _:None,simple_handoff=True)
        self.assertGreater(result['counts']['movement'],0)
        self.assertEqual(result['counts']['home'],0)
        self.assertFalse(result['ready'])
        self.assertFalse(result['withdraw_with_partial_open']['release_proven'])


if __name__ == '__main__':
    unittest.main()
