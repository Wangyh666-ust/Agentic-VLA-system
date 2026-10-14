"""Software contracts for all ten native final conditions; no model or sim."""
import copy
import json
import unittest
from scene_demo import catalog
from scene_demo.xvla_skills import NATIVE_LONG_SKILLS
from scene_demo.xvla_finish import build_skill_context, build_prompt, XvlaQwenClient
from scene_demo.finish_v1.runtime import build_context

IDS = ['xvla_basket_both','xvla_basket_cheese_butter','xvla_stove_moka',
       'xvla_bowl_drawer_close','xvla_mugs_both','xvla_book_caddy',
       'xvla_mug_pudding','xvla_basket_soup_cheese','xvla_moka_both',
       'xvla_microwave_place_close']
COUNTS = [2,2,2,2,2,1,2,2,3,2]

class FullChecksTests(unittest.TestCase):
    def context(self, identifier):
        ctx = build_skill_context(catalog.CAPABILITIES[identifier], 'private full request')
        ctx['sensor_frames'] = [{'frame': i, 'view': v, 'timestamp': 1.0}
                                for i,v in enumerate(XvlaQwenClient.required_views)]
        ctx['actuator_measurements'] = {'timestamp': 1.0, 'finger_qpos': [.04,-.04],
                                       'standard_open_finger_qpos': [.04,-.04]}
        return ctx

    def test_ten_contexts_and_each_incomplete_or_unknown(self):
        self.assertEqual(set(NATIVE_LONG_SKILLS), set(IDS))
        for identifier, count in zip(IDS, COUNTS):
            with self.subTest(identifier=identifier):
                capability = catalog.CAPABILITIES[identifier]
                ctx = self.context(identifier)
                components = [catalog.CAPABILITIES[i] for i in capability['component_capability_ids']]
                conditions = components + capability.get('preserved_checks', [])
                checks = ctx['criteria']['success_checks']
                self.assertEqual(len(checks), count)
                self.assertEqual([c['description'] for c in checks], [c['instruction'] for c in conditions])
                self.assertEqual([c['relation'] for c in checks], [c.get('visual_relation',c['goals'][0][0]) for c in conditions])
                self.assertEqual(ctx['current_subtask'], build_context(components[-1], 'x')['current_subtask'])
                self.assertEqual([c.get('preserved', False) for c in checks], [False]*len(components)+[True]*(len(conditions)-len(components)))
                client = XvlaQwenClient(env_path='/nonexistent')
                client.set_check_contract(ctx)
                for index in range(count):
                    for value in (False, None):
                        reply = {'checks': [{'id':c['id'], 'achieved':value if i==index else True, 'reason':'software fake'} for i,c in enumerate(checks)],'goal_achieved':True}
                        self.assertIs(client.parse_verdict(json.dumps(reply))['operation_achieved'], value)
                reply = {'checks':[{'id':c['id'],'achieved':True,'reason':'software fake'} for c in checks]}
                self.assertIs(client.parse_verdict(json.dumps(reply))['operation_achieved'],True)

    def test_raw_goal_union_validation(self):
        for identifier in IDS:
            for mutation in ('missing','duplicate'):
                capability=copy.deepcopy(catalog.CAPABILITIES[identifier])
                if mutation=='missing':capability['goals'].pop()
                else:capability['goals'].append(copy.deepcopy(capability['goals'][0]))
                with self.subTest(identifier=identifier,mutation=mutation),self.assertRaises(ValueError):
                    build_skill_context(capability,'x')
        capability=copy.deepcopy(catalog.CAPABILITIES['xvla_moka_both'])
        capability['goals']=capability['goals'][:2]
        with self.assertRaises(ValueError):build_skill_context(capability,'x')
        capability=copy.deepcopy(catalog.CAPABILITIES['xvla_moka_both']);capability['component_capability_ids']=[]
        with self.assertRaises(ValueError):build_skill_context(capability,'x')
        capability=copy.deepcopy(catalog.CAPABILITIES['xvla_moka_both'])
        capability['preserved_checks'].append(copy.deepcopy(capability['preserved_checks'][0]));capability['goals'].append(['turnon','flat_stove_1'])
        with self.assertRaises(ValueError):build_skill_context(capability,'x')
        capability=copy.deepcopy(catalog.CAPABILITIES['xvla_moka_both'])
        capability['preserved_checks'].append({'instruction':'turn off stove','goals':[['turnoff','flat_stove_1']]});capability['goals'].append(['turnoff','flat_stove_1'])
        with self.assertRaises(ValueError):build_skill_context(capability,'x')
        capability=copy.deepcopy(catalog.CAPABILITIES['xvla_moka_both']);capability['preserved_checks'][0]['goals']=[['turnon','flat_stove_1'],['close','microwave_1']]
        with self.assertRaises(ValueError):build_skill_context(capability,'x')

    def test_special_semantics_and_safe_prompt(self):
        for identifier in IDS:
            prompt=build_prompt(self.context(identifier));safe=json.loads(prompt.splitlines()[-1])
            self.assertEqual(set(safe),{'current_skill','success_checks','sensor_frames','actuator_measurements'})
            for token in ('private full request','phase','stage','predicates','contact','home','goals'):
                self.assertNotIn(token,json.dumps(safe))
        pudding=self.context('xvla_mug_pudding')['criteria']['success_checks'][1]
        self.assertEqual(pudding['relation'],'right_of');self.assertIn('桌面',pudding['destination_definition'])
        self.assertIn('right_of指盘子指定右侧的桌面区域',build_prompt(self.context('xvla_mug_pudding')))
        drawer=self.context('xvla_bowl_drawer_close')
        self.assertEqual(drawer['current_subtask']['operation'],'articulate')
        definition=drawer['criteria']['success_checks'][1]['visual_state_definition']
        self.assertIn('最下层抽屉',definition['closed']);self.assertNotIn('微波',json.dumps(definition,ensure_ascii=False))
        moka=self.context('xvla_moka_both');checks=moka['criteria']['success_checks']
        self.assertEqual(checks[2]['relation'],'turnon');self.assertTrue(checks[2]['preserved'])
        self.assertNotEqual(moka['current_subtask']['target'],'flat_stove_1')
        prompt=build_prompt(moka);self.assertIn('两件不同实物',prompt);self.assertIn('不能用同一只摩卡壶',prompt);self.assertIn('不请求额外开关动作',prompt);self.assertIn('容器关着不能证明',prompt)
        self.assertIn('at_standard_open=true',prompt)
