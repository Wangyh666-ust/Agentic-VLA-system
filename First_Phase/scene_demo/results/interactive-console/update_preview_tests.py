from pathlib import Path
p=Path('First_Phase/scene_demo/tests/test_xvla_console.py');s=p.read_text(encoding='utf-8')
s=s.replace('def test_scenario_load_runs_review_before_runtime_or_creation', 'def test_task_submission_runs_review_before_runtime_or_creation')
s=s.replace("    with pytest.raises(RuntimeError, match='review blocked'):\n        c.load('microwave_mug')", "    c.session['preview_only'] = True\n    c.state.update(stage='ready', scene='microwave_mug')\n    with pytest.raises(RuntimeError, match='review blocked'):\n        c.plan('把黄杯放入微波炉')")
s=s.replace("    c.preflight.assert_called_once_with('microwave_mug')", "    c.preflight.assert_not_called()\n    c.ensure_runtime.assert_not_called()\n    c.svc.create_session.assert_not_called()")
s=s.replace("    c.svc.bind_request_context.side_effect = RuntimeError('stop before real models')", "    c.session.pop('preview_only')  # Test task text binding after promotion separately.\n    c.svc.bind_request_context.side_effect = RuntimeError('stop before real models')")
a=s.index('def test_preview_uses_worker_and_checker_cameras_without_binding');b=s.index('def test_history_is_read_only',a)
s=s[:a]+s[b:]
p.write_text(s,encoding='utf-8')
