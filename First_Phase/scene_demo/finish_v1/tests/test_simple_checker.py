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

    def test_prompt_requests_only_two_observations(self):
        prompt = checker.build_prompt(self.ctx)
        self.assertNotIn('仅悬在', prompt)
        self.assertNotIn('supported', prompt)
        self.assertNotIn('boxes', prompt)
        self.assertIn('goal_achieved', prompt)
        self.assertIn('put the wine bottle on the rack',prompt)
        self.assertNotIn('wine_rack_1_top_region',prompt)
        self.assertNotIn('top region',prompt)
        self.assertIn('结束状态检查器', prompt)
        self.assertIn('遗漏最后的松手', prompt)
        self.assertNotIn('放架任务', prompt)
        self.assertNotIn('放盘任务', prompt)

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
        final = checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":false}')
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
