import copy
import json
from pathlib import Path
import unittest
from scene_demo.finish_v1 import vision

class PlaceTriggerTests(unittest.TestCase):
    def fixture(self):
        root=Path(__file__).resolve().parents[4]
        path=root/'First_Phase/scene_demo/results/2026-10-10-live-handoff/pilot/wine_exit/runtime_checks/checks.jsonl'
        row=json.loads(path.read_text(encoding='utf8').splitlines()[-1])
        return copy.deepcopy(row['ctx']),copy.deepcopy(row['raw_verdict']),copy.deepcopy(row['robot']),{'finger_home':row['ctx']['actuator_measurements']['standard_open_finger_qpos']}
    def test_actual_reply_gate_not_decision(self):
        ctx,raw,robot,ref=self.fixture()
        self.assertTrue(vision.needs_relation_review(ctx,raw,robot,ref))
        d=vision.decide(ctx,vision.reconcile_verdict(ctx,raw,robot,ref))
        self.assertEqual(d['state'],'incomplete');self.assertFalse(d['allow_release']);self.assertFalse(d['allow_retreat'])
    def test_exclusions(self):
        for mode in ('far','touch','tv','dv','missing','stale','time','latestview'):
            c,r,rb,ref=self.fixture()
            if mode in ('far','touch'):
                for boxes in r['boxes'].values():
                    boxes['target']=[.1,.1,.2,.2];boxes['destination']=[.2 if mode=='touch' else .3,.1,.4,.2]
            if mode=='tv':r['target_visible']=False
            if mode=='dv':r['destination_visible']=False
            if mode=='missing':del c['actuator_measurements']['timestamp']
            if mode=='stale':c['actuator_measurements']['timestamp']=3.
            if mode=='time':c['sensor_frames']=[f for f in c['sensor_frames'] if f['timestamp']==c['actuator_measurements']['timestamp']]
            if mode=='latestview':c['sensor_frames']=[f for f in c['sensor_frames'] if f['timestamp']!=c['actuator_measurements']['timestamp'] or f['view']=='agentview']
            self.assertFalse(vision.needs_relation_review(c,r,rb,ref),mode)
if __name__=='__main__':unittest.main()
