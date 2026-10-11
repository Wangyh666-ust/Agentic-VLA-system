"""Meaningful branch regressions, not physical success claims."""
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scene_demo.finish_v1 import control, simple_checker
from scene_demo.finish_v1.calibrated_grasp import association, project, CalibratedAssistService
from scene_demo import service, joint_home
from scene_demo.finish_v1.tests.test_core import FakePort


class ConstrainedPort(FakePort):
    def __init__(self, remain_blocked=False):
        super().__init__()
        self.remain_blocked=remain_blocked

    def opening_shapes(self,q):
        return self.robot_shapes(q)

    def step(self,q,f=None):
        state=super().step(q,f)
        if self.phase == 'release' or self.remain_blocked:
            self.f=np.array([.021,-.038])
            state['finger_qpos']=self.f.copy()
        return state


class AssistedTests(unittest.TestCase):
    def run_finish(self,operation='turn',plan_ok=True,remain_blocked=False):
        port=ConstrainedPort(remain_blocked)
        initial=simple_checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":true}')
        final=simple_checker.SimpleQwenClient(max_calls=0).parse_verdict('{"goal_achieved":true,"holding_or_contact":false}')
        events=[]
        with patch.object(control,'plan_retreat',return_value={'ok':plan_ok,'waypoints':[np.zeros(7),np.full(7,.02)]}), patch.object(control,'screen_shapes',return_value={'ok':True}), patch.object(control.jh,'home_metrics',side_effect=lambda ref,st:{'ready':np.max(np.abs(st['arm_qpos']))<.004}):
            result=control.run_finish(port,{'current_subtask':{'operation':operation}},initial,
                simple_checker.decide,lambda *_:final,events.append,simple_handoff=True)
        return port,result,events

    def test_turn_opens_during_retreat_and_original_home(self):
        port,result,events=self.run_finish()
        self.assertTrue(result['ok'],result)
        self.assertEqual(result['counts']['release'],20)
        self.assertGreater(result['counts']['movement'],0)
        self.assertGreater(result['counts']['home'],0)
        index=port.phases.index('movement')
        np.testing.assert_allclose(port.targets_fingers[index],port.reference['finger_home'])
        self.assertTrue(result['home_entry']['measured_open'])
        self.assertTrue(any(e.get('command_kind')=='open_and_retreat' for e in events))

    def test_unsafe_turn_route_zero_movement(self):
        _,result,_=self.run_finish(plan_ok=False)
        self.assertEqual(result['reason'],'retreat plan blocked')
        self.assertEqual(result['counts']['movement'],0)
        self.assertEqual(result['counts']['home'],0)

    def test_cannot_home_if_still_not_open(self):
        _,result,_=self.run_finish(remain_blocked=True)
        self.assertEqual(result['reason'],'home recovery actuator gate')
        self.assertEqual(result['counts']['home'],0)

    def test_place_release_gate_unchanged(self):
        _,result,_=self.run_finish(operation='place')
        self.assertEqual(result['reason'],'release not confirmed')
        self.assertEqual(result['counts']['movement'],0)

    def test_visual_association_rejects_wrong_object_and_units(self):
        self.assertTrue(association([.4,.4,.6,.6],[[.5,.5]]))
        self.assertFalse(association([.1,.1,.2,.2],[[.5,.5]]))
        self.assertFalse(association([400,400,600,600],[[.5,.5]]))
        self.assertFalse(association(None,[[.5,.5]]))
        self.assertFalse(association([.6,.4,.4,.6],[[.5,.5]]))

    def test_projection_uses_camera_pose(self):
        T=np.eye(4);T[0,3]=1.
        K=np.array([[100,0,50],[0,100,50],[0,0,1]])
        np.testing.assert_allclose(project([1,0,1],K,T,100,100),[.5,.5])
        with self.assertRaises(ValueError):project([1,0,-1],K,T,100,100)

    def test_finish_commands_do_not_advance_actual_vla_counter(self):
        obj=object.__new__(CalibratedAssistService)
        env=SimpleNamespace(step=lambda action:None)
        obj._env=env
        def body(instance, session, plan, job, capability):
            instance._pending_source='vla'
            env.step(np.zeros(7))
            env.step(np.zeros(7))  # finish action: no pending VLA source
            env.step(np.zeros(7))
            self.assertEqual(instance._actual_vla,1)
            self.assertEqual(instance._local_grasp_actions,0)
            return {'ok':True}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(service.SceneService,'_run_capability',body), patch.object(joint_home,'_single_robot',return_value=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=1.)))):
                result=obj._run_capability(None,None,SimpleNamespace(run_dir=Path(folder)),'bowl_to_plate')
            rows=[json.loads(line) for line in (Path(folder)/'action_sources.jsonl').read_text().splitlines()]
        self.assertEqual([row['source'] for row in rows],['vla','finish','finish'])
        self.assertTrue(result['ok'])


if __name__ == '__main__':
    unittest.main()
