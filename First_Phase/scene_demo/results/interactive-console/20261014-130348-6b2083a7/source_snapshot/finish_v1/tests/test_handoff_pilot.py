import ast
from collections import deque
import inspect
import textwrap
import unittest
from unittest import mock
from scene_demo.finish_v1 import handoff_pilot, runtime

class HandoffTests(unittest.TestCase):
    def test_stop_eligible(self):
        case={'allow_helper':True,'expected_release':True,'expected_retreat':True}
        counts={'release':1,'movement':1,'confirm':1,'home':1}
        for status in ('blocked','unknown'):
            self.assertTrue(handoff_pilot.should_stop_after_case(case,{'status':status,'phase_counts':counts,'release_open_actions':1}))
        self.assertFalse(handoff_pilot.should_stop_after_case(case,{'status':'physical_success','phase_counts':counts,'release_open_actions':1}))

    def test_actual_monitor_binding_and_calls(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(handoff_pilot.PilotRunner.run_case)))
        body=next(n.body for n in ast.walk(tree) if isinstance(n,ast.Try) and any(isinstance(x,ast.ImportFrom) and x.module=='scene_demo.finish_v1.runtime' for x in n.body))
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.ImportFrom) and n.module=='scene_demo.finish_v1.runtime')
        selected=[]
        for n in body[start:]:
            selected.append(n)
            if isinstance(n,ast.FunctionDef) and n.name=='check_fn':break
        # Extract exactly the production binding and nested function; supply its closure.
        fn=ast.FunctionDef(name='factory',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[ast.Assign(targets=[ast.Name(id='last_verdict',ctx=ast.Store())],value=ast.Constant(None))]+selected+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='monitor',ctx=ast.Load()),ast.Name(id='check_fn',ctx=ast.Load())],ctx=ast.Load()))],decorator_list=[])
        module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        client=mock.Mock(_calls=0);port=mock.Mock();history=deque();rows=[]
        from pathlib import Path
        values=dict(env=None,reference={},case_out=Path('unused'),context={},qwen_client=client,port=port,frame_deque=history,qwen_budget={'n':0,'max':10},checks_fp=None,_append_jsonl=lambda fp,row:rows.append(row),vision=mock.Mock(decide=lambda ctx,v:{'state':'unknown'}))
        with mock.patch.object(runtime,'FinishMonitor',autospec=True) as constructor:
            mon=constructor.return_value
            mon.port=mock.Mock()
            def check(ctx,obs):client._calls+=2;return {'answer':'fused'}
            mon._check.side_effect=check
            exec(compile(module,'production_binding','exec'),values)
            monitor,check_fn=values['factory']()
            constructor.assert_called_once();self.assertIs(monitor.port,port);self.assertIs(monitor.history,history)
            self.assertEqual(check_fn({},[]),{'answer':'fused'});self.assertEqual(values['qwen_budget']['n'],2);self.assertEqual(rows[0]['call_delta'],2)
            mon._check.assert_called_once_with({},[])
            with self.assertRaises(TypeError):check_fn({},None)

if __name__=='__main__':unittest.main()
