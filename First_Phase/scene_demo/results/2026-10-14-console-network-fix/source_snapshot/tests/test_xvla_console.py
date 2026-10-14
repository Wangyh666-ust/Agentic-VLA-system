"""Exercise the user confirmation boundary without models or simulation."""
import copy
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import Mock

import pytest
from scene_demo import xvla_console as ui
from scene_demo import service


@pytest.fixture
def console(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'MAX_BUDGET', 900)
    monkeypatch.setattr(service, 'SESSION_STEP_LIMIT', 8000)
    c = ui.Console(tmp_path)
    c.directory = tmp_path / 'attempt'
    c.directory.mkdir()
    c.session = ui.read(ui.OFFLINE / 'heat_yellow_mug/session_public.json')
    # Offline capture fixture omitted this runtime field; real SessionRecord
    # supplies it. Preserve the actual X-VLA contract in this service stub.
    c.session['capability_backend'] = 'xvla'
    plan = ui.read(ui.OFFLINE / 'heat_yellow_mug/submitted_plans.json')[0]
    c.state = {'mode': 'live', 'stage': 'planning', 'request_id': plan['request_id'], 'events': []}
    c.svc = Mock()
    c.svc.session.return_value = copy.deepcopy(c.session)
    return c, plan


def test_capture_never_executes_and_rejects_duplicate(console):
    c, plan = console
    assert c.capture_plan(plan)['execution_deferred'] is True
    c.svc.submit_plan.assert_not_called()
    with pytest.raises(ValueError):
        c.capture_plan(plan)
    c.svc.submit_plan.assert_not_called()


@pytest.mark.parametrize('field,value', [('request_id', 'wrong'), ('session_id', 'wrong'),
                                        ('scene_version', 99), ('request_scope', None)])
def test_invalid_draft_does_not_reach_service(console, field, value):
    c, plan = console
    plan[field] = value
    with pytest.raises(ValueError):
        c.capture_plan(plan)
    assert c.draft is None
    c.svc.submit_plan.assert_not_called()


def test_confirmation_preserves_partial_scope_and_exact_plan(console):
    c, original = console
    c.capture_plan(original)
    expected = copy.deepcopy(c.draft)
    c.state['stage'] = 'draft'
    c.svc.submit_plan.return_value = {'ok': True}
    c.svc.plan.return_value = {**expected, 'state': 'completed', 'plan_success': True,
                               'request_fulfilled': False, 'job_ids': []}
    c.execute()
    c.svc.submit_plan.assert_called_once_with(expected)
    assert c.state['plan']['plan_success'] is True
    assert c.state['plan']['request_fulfilled'] is False
    with pytest.raises(ValueError):
        c.execute()
    assert c.svc.submit_plan.call_count == 1


def test_stale_scene_cannot_execute(console):
    c, plan = console
    c.capture_plan(plan)
    c.state['stage'] = 'draft'
    c.svc.session.return_value['scene_version'] += 1
    with pytest.raises(ValueError):
        c.execute()
    c.svc.submit_plan.assert_not_called()


def test_cancelled_planning_rejects_late_submission(console):
    c, plan = console
    c.stop()
    with pytest.raises(ValueError):
        c.capture_plan(plan)
    c.svc.submit_plan.assert_not_called()


def test_stop_tracks_actual_backend_terminal(console, monkeypatch):
    c, plan = console
    c.capture_plan(plan)
    c.state['stage'] = 'draft'
    c.svc.submit_plan.return_value = {'ok': True}
    running = {**c.draft, 'state': 'running', 'job_ids': []}
    c.svc.plan.side_effect = [running, running, {**running, 'state': 'cancelled'}]
    c.svc.cancel_request.return_value = {'ok': True}
    stages = []
    def tick(_):
        stages.append(c.state['stage'])
        c.cancel.set()
    monkeypatch.setattr(ui.time, 'sleep', tick)
    c.execute()
    assert stages == ['executing', 'stopping']
    assert c.state['stage'] == 'cancelled'
    c.svc.cancel_request.assert_called_once_with(plan['request_id'], plan['session_id'])


def test_task_submission_runs_setup_before_runtime_or_creation(console, monkeypatch):
    c, _ = console
    def blocked(*args):
        raise RuntimeError('setup blocked')
    monkeypatch.setattr(c, 'preflight', blocked)
    monkeypatch.setattr(c, 'ensure_runtime', Mock())
    c.session['preview_only'] = True
    c.state.update(stage='ready', scene='microwave_mug')
    with pytest.raises(RuntimeError, match='setup blocked'):
        c.plan('把黄杯放入微波炉')
    c.ensure_runtime.assert_not_called()
    c.svc.create_session.assert_not_called()


def test_setup_never_reads_registry_or_starts_review_subprocess(console, monkeypatch):
    c, _ = console
    monkeypatch.setattr(ui, 'read', Mock(side_effect=AssertionError('no history reads')))
    launcher = Mock(side_effect=AssertionError('no audit subprocess'))
    monkeypatch.setattr(ui.subprocess, 'Popen', launcher)
    c.preflight('book_caddy', '把书归位')
    record = json.loads((c.directory / 'run_setup.json').read_text(encoding='utf-8'))
    assert record['history_audit'] == 'not_run_in_interactive_path'
    assert record['execution_requires_confirmation'] is True
    assert record['helpers'] is False and record['source_sha256']
    assert record['request'] == '把书归位' and 'xvla_book_caddy' in record['candidate_capability_ids']
    assert not (c.directory / 'review_check.json').exists()
    launcher.assert_not_called()
    c.svc.submit_plan.assert_not_called()


def test_load_without_task_shows_triplet_before_any_grounding(console, monkeypatch):
    c, _ = console
    monkeypatch.setattr(c, 'preflight', Mock())
    monkeypatch.setattr(c, 'ensure_runtime', Mock())
    preview = ui.frames(ui.read(ui.OFFLINE / 'heat_yellow_mug/grounding/capture.json'))
    monkeypatch.setattr(c, 'preview_scene', Mock(return_value=preview))
    c.svc.create_session.return_value = {'ok': True, 'session': c.session}
    c.load('microwave_mug')
    assert c.state['stage'] == 'ready'
    assert c.state['request'] == ''
    assert c.state['frames'] == preview and len(preview) == 3
    assert c.state['preview_only'] is True
    c.preflight.assert_not_called()
    c.ensure_runtime.assert_not_called()
    c.svc.create_session.assert_not_called()
    c.svc.bind_request_context.assert_not_called()
    c.svc.submit_plan.assert_not_called()
    assert not (c.directory / 'task_request.json').exists()
    with pytest.raises(ValueError):
        c.plan('   ')
    assert c.state['stage'] == 'ready'
    c.svc.bind_request_context.assert_not_called()
    c.session.pop('preview_only')  # Test task text binding after promotion separately.
    c.svc.bind_request_context.side_effect = RuntimeError('stop before real models')
    with pytest.raises(RuntimeError, match='stop before real models'):
        c.plan('  把黄杯放进微波炉  ')
    c.svc.bind_request_context.assert_called_once_with(c.state['request_id'], c.session['session_id'], '把黄杯放进微波炉')
    assert ui.read(c.directory / 'task_request.json')['request'] == '把黄杯放进微波炉'


def test_history_is_read_only_and_uses_original_triplet(tmp_path):
    c = ui.Console(tmp_path)
    for example in ui.EXAMPLES:
        c.history(example['id'])
        state = c.snapshot()
        assert state['mode'] == 'history'
        assert [f['view'] for f in state['frames']] == ['wrist', 'agentview', 'sideview']
        assert all(ui.Path(f['path']).is_file() for f in state['frames'])
        assert c.svc is None and c.model is None
        with pytest.raises(ValueError):
            c.execute()


def test_http_requires_token_blocks_arbitrary_file_and_stale_confirm(console):
    c, _ = console
    server = ThreadingHTTPServer(('127.0.0.1', 0), ui.Handler)
    server.console = c
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:' + str(server.server_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for route, body, headers, code in [
            ('/api/execute', {'request_id': c.state['request_id']}, {}, 403),
            ('/api/execute', {'request_id': 'stale'}, {'X-Console-Token': c.token}, 400),
            ('/api/stop', {}, {'X-Console-Token': c.token, 'Origin': 'https://untrusted.example'}, 403),
            ('/media?path=/etc/passwd', None, {}, 400),
        ]:
            request = urllib.request.Request(base + route, data=None if body is None else json.dumps(body).encode(), headers=headers)
            with pytest.raises(urllib.error.HTTPError) as caught:
                opener.open(request)
            assert caught.value.code == code
        c.svc.submit_plan.assert_not_called()
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def runtime_console(tmp_path, monkeypatch):
    """Mirror deploy's actual info.json refusal without loading GPU weights."""
    from scene_demo import xvla_service
    c = ui.Console(tmp_path)
    c.directory = tmp_path / 'attempt'
    c.directory.mkdir()
    sock = Mock()
    sock.connect_ex.return_value = 1
    socket_context = Mock()
    socket_context.__enter__ = Mock(return_value=sock)
    socket_context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(ui.socket, 'socket', Mock(return_value=socket_context))
    opener = Mock()
    opener.open.side_effect = lambda *a, **k: io.StringIO(json.dumps({'paths': {'/act': {'post': {}}}}))
    monkeypatch.setattr(ui.urllib.request, 'build_opener', Mock(return_value=opener))
    scene = Mock()
    scene.health.return_value = {'ready': True}
    factory = Mock(return_value=scene)
    monkeypatch.setattr(xvla_service, 'XvlaSceneService', factory)
    starts = []
    def launch(argv, **kwargs):
        output = ui.Path(argv[argv.index('--output_dir') + 1])
        output.mkdir(parents=True, exist_ok=True)
        proc = Mock(pid=700 + len(starts))
        if (output / 'info.json').exists():
            proc.poll.return_value = 1
            kwargs['stdout'].write(b'Error: info.json already exists\n')
        else:
            ui.write(output / 'info.json', {'host': '127.0.0.1', 'port': 8794})
            proc.poll.return_value = None
            kwargs['stdout'].write(b'Model successfully loaded; service started\n')
        proc.terminate.side_effect = lambda: setattr(proc.poll, 'return_value', 0)
        starts.append({'output': output, 'proc': proc, 'argv': list(argv)})
        return proc
    launcher = Mock(side_effect=launch)
    monkeypatch.setattr(ui.subprocess, 'Popen', launcher)
    return c, sock, launcher, starts, factory, scene


def test_runtime_restarts_preserve_old_info_and_logs(runtime_console):
    c, _, launcher, starts, _, scene = runtime_console
    stale = c.root / 'model_output/info.json'
    ui.write(stale, {'old_process': 'exited'})
    legacy_log = c.root / 'model.log'
    legacy_log.write_bytes(b'old failure evidence\n')
    stale_bytes = stale.read_bytes()
    c.ensure_runtime()
    first_log = ui.Path(c.state['model_log'])
    first_bytes = first_log.read_bytes()
    c.close_runtime()
    starts[0]['proc'].terminate.assert_called_once()
    c.ensure_runtime()
    second_log = ui.Path(c.state['model_log'])
    assert launcher.call_count == 2
    assert starts[0]['output'] != starts[1]['output']
    assert first_log != second_log and first_log.read_bytes() == first_bytes
    assert stale.read_bytes() == stale_bytes
    assert legacy_log.read_bytes() == b'old failure evidence\n'
    assert all((s['output'] / 'info.json').is_file() for s in starts)
    assert ui.read(c.root / 'model_process.json')['log'] == str(second_log)
    assert ui.read(first_log.parent / 'model_process.json')['log'] == str(first_log)
    scene.create_session.assert_not_called()
    scene.bind_request_context.assert_not_called()
    scene.submit_plan.assert_not_called()
    c.close_runtime()


def test_runtime_occupied_port_never_starts_or_removes_old_files(runtime_console):
    c, sock, launcher, _, _, _ = runtime_console
    stale = c.root / 'model_output/info.json'
    ui.write(stale, {'keep': True})
    sock.connect_ex.return_value = 0
    with pytest.raises(RuntimeError, match='8794端口'):
        c.ensure_runtime()
    launcher.assert_not_called()
    assert ui.read(stale) == {'keep': True}
    assert not list(c.directory.glob('model-*'))


def test_runtime_failure_reports_its_own_log_without_starting_scene(runtime_console):
    c, _, launcher, _, factory, _ = runtime_console
    def fail(argv, **kwargs):
        kwargs['stdout'].write(b'actual startup failure\n')
        proc = Mock(pid=999)
        proc.poll.return_value = 7
        return proc
    launcher.side_effect = fail
    with pytest.raises(RuntimeError, match='退出码 7') as caught:
        c.ensure_runtime()
    log = ui.Path(c.state['model_log'])
    assert str(log) in str(caught.value)
    assert log.read_bytes() == b'actual startup failure\n'
    assert c.model is None and c.svc is None
    factory.assert_not_called()
