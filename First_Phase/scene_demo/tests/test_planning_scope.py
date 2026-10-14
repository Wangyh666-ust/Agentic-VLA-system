"""No simulation/network: partial coverage survives service/host boundaries."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import service
from planning_scope import validate_request_scope


class ScopeContract(unittest.TestCase):
    def setUp(self):
        self.scope = dict(coverage='partial', executable_outcomes=['put mug in microwave', 'close door'],
                          unmet_outcomes=['start heating'], reason='preparation only; activation unavailable')
        self.payload = dict(request_id='r', session_id='s', scene_version=0,
                            decision='execute', capability_ids=['xvla_microwave_place_close'],
                            request_scope=self.scope, budget_per_subgoal=300)
        self.session = dict(session_id='s', state='ready', scene_version=0,
                            scene_id='microwave_mug', capability_backend='xvla')

    def test_partial_plan_can_execute_but_cannot_become_full_success(self):
        self.assertIsNone(service.validate_plan_request(self.session, None, self.payload))
        plan = service.PlanRecord('r', 's', self.payload)
        plan.state, plan.plan_success = 'completed', True
        output = plan.public()
        self.assertTrue(output['plan_success'])
        self.assertFalse(output['request_fulfilled'])
        self.assertEqual(output['request_scope']['unmet_outcomes'], ['start heating'])
        self.scope['unmet_outcomes'].clear()
        output['request_scope']['unmet_outcomes'].clear()
        self.assertEqual(plan.public()['request_scope']['unmet_outcomes'], ['start heating'])

    def test_full_claim_with_unmet_outcome_rejected(self):
        self.scope['coverage'] = 'full'
        self.assertEqual(service.validate_plan_request(self.session, None, self.payload)['reason'],
                         'invalid_request_scope')

    def test_partial_without_supported_part_rejected(self):
        self.scope['executable_outcomes'] = []
        with self.assertRaises(ValueError):
            validate_request_scope(self.scope, 'execute')

    def test_none_cannot_execute(self):
        value = dict(coverage='none', executable_outcomes=[], unmet_outcomes=['discard'], reason='no bin')
        self.assertEqual(validate_request_scope(value, 'unsupported'), value)
        with self.assertRaises(ValueError):
            validate_request_scope(value, 'execute')

    def test_full_plan_needs_actual_completion_and_legacy_stays_unknown(self):
        self.scope.update(coverage='full', unmet_outcomes=[])
        plan = service.PlanRecord('r', 's', self.payload)
        self.assertIsNone(plan.public()['request_fulfilled'])
        plan.state, plan.plan_success = 'completed', True
        self.assertTrue(plan.public()['request_fulfilled'])
        payload = copy.deepcopy(self.payload)
        payload.pop('request_scope')
        legacy = service.PlanRecord('r2', 's', payload)
        legacy.state, legacy.plan_success = 'completed', True
        self.assertIsNone(legacy.public()['request_fulfilled'])


if __name__ == '__main__':
    unittest.main()
