import copy
import unittest
from unittest import mock
from scene_demo.finish_v1 import vision
from scene_demo.finish_v1.tests.test_runtime import RuntimeTests, ALL_TRUE

class JointVerifierTests(RuntimeTests):
    def strict(self):
        ctx=copy.deepcopy(self.context)
        ctx['sensor_frames']=[{'frame':0,'view':'agentview','timestamp':1.0}]
        ctx['actuator_measurements']={'timestamp':1.0,'stage':'post_execution','finger_qpos':[0.04,-0.04],'standard_open_finger_qpos':[0.04,-0.04],'max_abs_arm_qvel':0.0}
        return ctx

    def test_timestamp_mismatch_http_zero(self):
        ctx=self.strict();ctx['actuator_measurements']['timestamp']=2.0
        client=vision.QwenClient(max_calls=1)
        with mock.patch('urllib.request.urlopen') as http:
            with self.assertRaises(ValueError): client.analyze(ctx, [])
            http.assert_not_called()
        self.assertEqual(client._calls,0)

    def test_missing_timestamp_and_mismatched_robot_unknown(self):
        for mode in ('missing','robot'):
            ctx=self.strict();robot={'finger_qpos':[0.04,-0.04]}
            if mode=='missing':del ctx['actuator_measurements']['timestamp']
            else:robot['finger_qpos']=[0.0,0.0]
            fused=vision.reconcile_verdict(ctx,ALL_TRUE,robot,self.reference)
            self.assertIsNone(fused['released'])
            self.assertEqual(vision.decide(ctx,fused)['state'],'unknown')
            self.assertIn('missing_or_stale_encoder',fused['unknown_reasons'])

    def test_closed_raw_complete_and_open_unknown_and_negative(self):
        ctx=self.strict();ctx['actuator_measurements']['finger_qpos']=[0.0,0.0]
        fused=vision.reconcile_verdict(ctx,ALL_TRUE,{'finger_qpos':[0.0,0.0]},self.reference)
        decision=vision.decide(ctx,fused)
        self.assertEqual(decision['state'],'needs_release');self.assertTrue(decision['allow_release'])
        ctx=self.strict();raw=copy.deepcopy(ALL_TRUE);raw['released']=None
        fused=vision.reconcile_verdict(ctx,raw,{'finger_qpos':[0.04,-0.04]},self.reference)
        self.assertIsNone(fused['released'])
        raw['at_destination']=False
        self.assertEqual(vision.decide(ctx,vision.reconcile_verdict(ctx,raw,{'finger_qpos':[0.04,-0.04]},self.reference))['state'],'incomplete')

    def test_captured_feedback_without_later_read(self):
        mon=self.monitor();self.record_pair(mon)
        mon.port.read_robot=lambda: (_ for _ in ()).throw(AssertionError('late read forbidden'))
        mon._check(self.context,[])
        ctx,_=mon.call_client.calls[-1]
        self.assertEqual(ctx['actuator_measurements']['timestamp'],10.0)
        self.assertEqual(ctx['actuator_measurements']['finger_qpos'],[0.04,-0.04])

    def test_missing_captured_feedback_zero_model_actions(self):
        mon=self.monitor();self.record_pair(mon);capture=mon.port.capture
        def missing():
            cap=capture();del cap['robot'];return cap
        mon.port.capture=missing
        with self.assertRaisesRegex(RuntimeError,'missing robot'):mon._check(self.context,[])
        self.assertEqual(mon.calls,0);self.assertEqual(mon.port.steps,0)

class RelationReviewTests(JointVerifierTests):
    def eligible(self):
        ctx=self.strict()
        ctx['sensor_frames']=[{'frame':i,'view':v,'timestamp':t} for i,(v,t) in enumerate([('agentview',0.9),('wrist',0.9),('agentview',1.0),('wrist',1.0)])]
        raw=copy.deepcopy(ALL_TRUE);raw.update(released=None,stable=None,clear_of_target=False)
        raw['boxes']={v:{'target':[0.1,0.1,0.2,0.2],'destination':[0.1,0.1,0.3,0.3],'gripper':[0.7,0.7,0.9,0.9]} for v in ('agentview','wrist')}
        return ctx,raw

    def test_relation_trigger_and_exclusions(self):
        ctx,raw=self.eligible();robot={'finger_qpos':[0.04,-0.04]}
        self.assertTrue(vision.needs_relation_review(ctx,raw,robot,self.reference))
        for mode in ('closed','stale','missing','singleview','singletime','stablefalse','unsupported','away','gap'):
            c,r=copy.deepcopy(ctx),copy.deepcopy(raw);rb=copy.deepcopy(robot)
            if mode=='closed':c['actuator_measurements']['finger_qpos']=rb['finger_qpos']=[0.0,0.0]
            if mode=='stale':c['actuator_measurements']['timestamp']=2.0
            if mode=='missing':del c['actuator_measurements']['timestamp']
            if mode=='singleview':c['sensor_frames']=[f for f in c['sensor_frames'] if f['view']=='agentview']
            if mode=='singletime':c['sensor_frames']=[f for f in c['sensor_frames'] if f['timestamp']==1.0]
            if mode=='stablefalse':r['stable']=False
            if mode=='unsupported':r['supported']=False
            if mode=='away':r['at_destination']=False
            if mode=='gap':r['boxes']['wrist']['gripper']=[0.31,0.1,0.4,0.2]
            if mode in ('unsupported','away'):
                self.assertTrue(vision.needs_relation_review(c,r,rb,self.reference),mode)
                decision=vision.decide(c,vision.reconcile_verdict(c,r,rb,self.reference))
                self.assertFalse(decision['allow_release']);self.assertFalse(decision['allow_retreat'])
            else:
                self.assertFalse(vision.needs_relation_review(c,r,rb,self.reference),mode)

    def test_independent_prompt(self):
        ctx,_=self.eligible();ctx['user_request']='SECRET_REQUEST';ctx['current_subtask']['phase']='SECRET_PHASE';ctx['actuator_measurements']['stage']='SECRET_STAGE'
        prompt=vision.build_relation_prompt(ctx)
        category='LIBERO asset aliases alphabet_soup and tomato_sauce refer to cylindrical food cans'
        self.assertIn(category,prompt)
        self.assertIn(category,vision.build_prompt(ctx))
        self.assertNotIn('texture_map',prompt)
        self.assertNotIn('texture_map',vision.build_prompt(ctx))
        for required in ('inner wall', 'round burner ring is not a plate', 'outer rim', 'no longer hold or contact', 'distinct from support by the table', 'arm velocity', 'visible contrary evidence'):
            self.assertIn(required,prompt)
        for forbidden in ('SECRET_REQUEST','SECRET_PHASE','SECRET_STAGE','expected','grounding_roi','old_raw'):
            self.assertNotIn(forbidden,prompt)

    def relation_monitor(self):
        patch=mock.patch('scene_demo.finish_v1.relation_images.prepare_relation_images', return_value={'ready':True,'reason':'test prepared','image_paths':['a','b','c','d','e','f'],'sensor_frames':[{'frame':i,'view':v,'timestamp':t} for i,(v,t) in enumerate([('agentview',.9),('wrist',.9),('agentview',1.),('wrist',1.),('agentview',1.),('wrist',1.)])],'derivative_provenance':[{'frame':4+i,'source_frame':2+i,'view':v,'timestamp':1.,'source_size_px':[100,100],'pixel_bounds':[0,0,100,100],'original_roi':[0.,0.,1.,1.]} for i,v in enumerate(['agentview','wrist'])]})
        patch.start();self.addCleanup(patch.stop)
        ctx,raw=self.eligible();mon=self.monitor();mon.context=ctx
        mon.call_client.verdict=raw
        mon.port.timestamp=.9;mon.record(5);mon.port.timestamp=1.0;mon.record(10)
        # FakePort camera names are agentview/wrist.
        mon.call_client.inspect_relations=lambda c,p: {'verdict':{**copy.deepcopy(ALL_TRUE),'at_destination':False},'model':'fake-review'}
        return mon

    def test_runtime_two_calls_whole_negative_result(self):
        mon=self.relation_monitor();fused=mon._check(mon.context,[])
        self.assertEqual(mon.calls,2);self.assertFalse(fused['at_destination'])
        self.assertEqual(mon._last_decision['state'],'incomplete');self.assertEqual(mon.port.steps,0)

    def test_review_budget_and_cancellation(self):
        mon=self.relation_monitor();mon.calls=9
        mon._check(mon.context,[]);self.assertEqual(mon.calls,10)
        self.assertEqual(mon._last_decision['state'],'unknown')
        for when in ('before','after'):
            mon=self.relation_monitor()
            if when=='before':mon.call_client.after=lambda:setattr(self,'cancelled',True)
            else:
                def review(c,p):
                    self.cancelled=True
                    return {'verdict':copy.deepcopy(ALL_TRUE)}
                mon.call_client.inspect_relations=review
            with self.assertRaisesRegex(RuntimeError,'cancelled'):mon._check(mon.context,[])
            self.assertEqual(mon.port.steps,0);self.cancelled=False

class TurnRelationTests(RelationReviewTests):
    def test_turn_trigger_and_exclusions(self):
        ctx,raw=self.eligible();ctx['current_subtask'].update(operation='turn',destination=None);ctx['criteria']['desired_state']='on'
        ctx['actuator_measurements']['finger_qpos']=[0.,0.];robot={'finger_qpos':[0.,0.]};raw['operation_achieved']=False
        self.assertTrue(vision.needs_relation_review(ctx,raw,robot,self.reference))
        raw['operation_achieved']=None;self.assertTrue(vision.needs_relation_review(ctx,raw,robot,self.reference))
        for mode in ('done','hidden','stale','missing','ref','target','time','view'):
            c,r=copy.deepcopy(ctx),copy.deepcopy(raw)
            if mode=='done':r['operation_achieved']=True
            if mode=='hidden':r['target_visible']=False
            if mode=='stale':c['actuator_measurements']['timestamp']=2.
            if mode=='missing':del c['actuator_measurements']['timestamp']
            if mode=='ref':c['actuator_measurements']['standard_open_finger_qpos']=[.1,-.1]
            if mode=='target':r['boxes']['wrist']['target']=None
            if mode=='time':c['sensor_frames']=[f for f in c['sensor_frames'] if f['timestamp']==1.]
            if mode=='view':c['sensor_frames']=[f for f in c['sensor_frames'] if f['view']=='wrist']
            self.assertFalse(vision.needs_relation_review(c,r,robot,self.reference),mode)
        prompt=vision.build_relation_prompt(ctx)
        self.assertIn('does not override visible evidence',prompt);self.assertIn('red burner ring',prompt)
        ctx['criteria']['desired_state']='bad'
        with mock.patch('urllib.request.urlopen') as http:
            with self.assertRaises(ValueError):vision.QwenClient(max_calls=1).inspect_relations(ctx,[])
            http.assert_not_called()

if __name__=='__main__':unittest.main()
