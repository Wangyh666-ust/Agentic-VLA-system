import unittest
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
        result = client.parse_verdict('{"operation_achieved":true,"boxes":{"sideview":{"target":[0.1,0.1,0.2,0.2]}}}')
        self.assertIsNone(result['supported'])
        self.assertFalse(checker.decide(self.ctx, result)['allow_release'])

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
        result = control.run_finish(port, self.ctx, self.v, checker.decide,
                                    lambda *_: fail_after_open, lambda _: None, simple_handoff=True)
        self.assertEqual(port.commands[0], [.04, -.04])
        self.assertEqual(result['release_settle_actions'], 0)
        self.assertEqual(result['release_open_actions'], 2)
        self.assertEqual(result['reason'], 'release visual failed')


if __name__ == '__main__':
    unittest.main()
