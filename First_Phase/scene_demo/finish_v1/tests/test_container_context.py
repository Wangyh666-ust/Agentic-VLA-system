import json
import unittest
from scene_demo import catalog
from scene_demo.finish_v1 import runtime, simple_checker, vision


class ContainerContextTests(unittest.TestCase):
    def context(self, capability):
        context = runtime.build_context(catalog.CAPABILITIES[capability], '放好杯子，然后关门')
        context['sensor_frames'] = [{'frame': 0, 'view': view, 'timestamp': 1.} for view in ('wrist', 'sideview')]
        context['actuator_measurements'] = {'timestamp': 1., 'finger_qpos': [.02, -.02],
            'standard_open_finger_qpos': [.04, -.04]}
        vision._validate_context(context)
        return context

    def test_door_context_is_close_not_stove(self):
        context = self.context('microwave_close')
        self.assertEqual(context['current_subtask']['operation'], 'articulate')
        self.assertEqual(context['criteria']['desired_state'], 'closed')
        prompt = simple_checker.build_prompt(context)
        payload = json.loads(prompt.splitlines()[-1])
        self.assertEqual(payload['current_subtask']['description'], 'close the microwave')
        self.assertIn('closed', payload['visual_state_definition'])
        self.assertNotIn('phase', payload['current_subtask'])
        self.assertNotIn('stage', prompt)
        self.assertNotIn('红色', prompt)

    def test_place_instruction_keeps_current_subgoal(self):
        for cap in ('mug_to_microwave', 'sauce_to_basket', 'bowl_to_plate', 'wine_to_rack'):
            context = self.context(cap)
            payload = json.loads(simple_checker.build_prompt(context).splitlines()[-1])
            self.assertEqual(payload['current_subtask']['description'], catalog.CAPABILITIES[cap]['instruction'])
            self.assertNotIn('visual_state_definition', payload)

    def test_door_goal_controls_trigger_not_closed_fingers(self):
        context = self.context('microwave_close')
        client = simple_checker.SimpleQwenClient(max_calls=0)
        for goal, allowed in ((True, True), (False, False), (None, False)):
            verdict = client.parse_verdict(json.dumps({'goal_achieved': goal, 'reason': 'door state'}))
            verdict = simple_checker.reconcile(context, verdict, {'finger_qpos': [.02, -.02]}, {'finger_home': [.04, -.04]})
            self.assertEqual(simple_checker.decide(context, verdict)['allow_release'], allowed)


if __name__ == '__main__':
    unittest.main()
