"""Transport failures keep usable, credential-free evidence and call budgets."""
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest
from scene_demo.finish_v1 import vision
from scene_demo.finish_v1.simple_checker import SimpleQwenClient
from scene_demo.xvla_grounding import InventoryClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-secret-do-not-log')
    monkeypatch.setattr(vision.urllib.request, 'urlopen', Mock(side_effect=AssertionError('network forbidden')))
    c = SimpleQwenClient(max_calls=1)
    c.audit_dir = tmp_path / 'audit'
    image = tmp_path / 'sample.png'
    image.write_bytes(b'fixture-only-never-sent')
    return c, [str(image)]


@pytest.mark.parametrize('error', [TimeoutError('hidden secret'), urllib.error.URLError(TimeoutError('hidden secret'))])
def test_timeout_preserves_prompt_frames_and_specific_error(client, monkeypatch, error):
    c, paths = client
    call = Mock(side_effect=error)
    monkeypatch.setattr(vision.urllib.request, 'urlopen', call)
    ctx = {'sensor_frames': [{'frame': 0, 'view': 'wrist', 'timestamp': 1.0}]}
    with pytest.raises(RuntimeError, match='timed out after 90s'):
        c._request(ctx, paths, 'look at target')
    record = json.loads((c.audit_dir / 'call_01_request.json').read_text())
    failure = json.loads((c.audit_dir / 'call_01_transport_error.json').read_text())
    assert record['prompt'] == 'look at target' and record['context'] == ctx
    assert record['requested_model'] == 'qwen3.8-max' and len(record['images']) == 1
    assert failure['reason_type'] == 'TimeoutError' and failure['stage'] == 'awaiting_response'
    assert failure['timeout_seconds'] == 90
    assert not (c.audit_dir / 'call_01_response.json').exists()
    audit_text = ''.join(p.read_text() for p in c.audit_dir.iterdir())
    assert 'test-secret' not in audit_text and 'hidden secret' not in audit_text
    with pytest.raises(RuntimeError, match='budget exhausted'):
        c._request(ctx, paths, 'again')
    assert call.call_count == 1


def test_http_error_records_status_without_response_secrets(client, monkeypatch):
    c, paths = client
    error = urllib.error.HTTPError('url', 429, 'private-text', {}, io.BytesIO(b'private-text'))
    monkeypatch.setattr(vision.urllib.request, 'urlopen', Mock(side_effect=error))
    with pytest.raises(RuntimeError, match='HTTP status 429'):
        c._request({}, paths, 'inspect')
    saved = (c.audit_dir / 'call_01_transport_error.json').read_text()
    assert json.loads(saved)['http_status'] == 429 and 'private-text' not in saved


def test_response_read_timeout_is_audited_and_closed(client, monkeypatch):
    c, paths = client
    response = Mock()
    response.getcode.return_value = 200
    response.read.side_effect = TimeoutError()
    monkeypatch.setattr(vision.urllib.request, 'urlopen', Mock(return_value=response))
    with pytest.raises(RuntimeError, match='reading_response'):
        c._request({}, paths, 'inspect')
    failure = json.loads((c.audit_dir / 'call_01_transport_error.json').read_text())
    assert failure['stage'] == 'reading_response'
    response.close.assert_called_once()


def test_inventory_has_longer_wait_without_changing_execution_checkers():
    assert InventoryClient().request_timeout == 180
    assert SimpleQwenClient().request_timeout == 90
    assert vision.QwenClient().request_timeout == 90


def test_service_wait_covers_both_inventory_requests(monkeypatch):
    from scene_demo import service
    from scene_demo.xvla_service import XvlaSceneService
    monkeypatch.setattr(service.SceneService, 'bind_request_context', Mock(return_value={'ok': True}))
    scene = object.__new__(XvlaSceneService)
    scene._grounding_enabled = True
    scene._sync_work = Mock(return_value={'ok': True})
    assert scene.bind_request_context('r', 's', 'book')['ok']
    assert scene._sync_work.call_args.kwargs['timeout'] > 2 * InventoryClient().request_timeout


@pytest.mark.parametrize('bad', [0, -1, True, float('inf'), float('nan'), '90'])
def test_invalid_timeout_is_rejected(bad):
    with pytest.raises(ValueError, match='request_timeout'):
        vision.QwenClient(request_timeout=bad)
