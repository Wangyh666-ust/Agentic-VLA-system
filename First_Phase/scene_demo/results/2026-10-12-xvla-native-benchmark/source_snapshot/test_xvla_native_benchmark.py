"""Tests for the audited native X-VLA LIBERO benchmark wrapper.

These tests lazy-load the runner module inside each test so that a tests-only
run fails explicitly when the runner is absent. HTTP and physical step
endpoints are faked; no model or simulator is constructed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import json_numpy

MODULE_PATH = Path(__file__).resolve().parent / "xvla_native_benchmark.py"


def _load_runner():
    assert MODULE_PATH.exists(), "runner module xvla_native_benchmark.py is absent"
    spec = importlib.util.spec_from_file_location("xvla_native_benchmark", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["xvla_native_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def _fake_response(action, status=200):
    class _Resp:
        def __init__(self):
            self.status_code = status
            self.content = json.dumps({"action": action}).encode()

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")

        def json(self):
            return json.loads(self.content.decode())

    return _Resp()


def _identity_action(grip_values):
    rows = []
    for g in grip_values:
        row = [0.4, -0.1, 1.15, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] + [0.0] * 11
        row[9] = g
        rows.append(row)
    return rows


def _obs():
    agent = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    wrist = (np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3) + 7) % 256
    return {
        "agentview_image": agent,
        "robot0_eye_in_hand_image": wrist,
        "robo_pos": np.array([0.0, 0.0, 1.0]),
        "robo_ori": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    }


def test_episode_slots_exact_order():
    mod = _load_runner()
    slots = mod.episode_slots()
    assert len(slots) == 200
    assert len(set(slots)) == 200
    assert slots[0] == ("libero_10", 0, 0)
    assert slots[-1] == ("libero_object", 9, 4)
    assert slots[:5] == [("libero_10", 0, e) for e in range(5)]
    assert slots[5:10] == [("libero_10", 1, e) for e in range(5)]
    assert slots[50] == ("libero_spatial", 0, 0)


def test_step_uses_one_post_then_two_and_double_flip(monkeypatch):
    mod = _load_runner()
    posted = []

    def fake_post(url, json=None, timeout=None):
        posted.append(json)
        grip = 0.4 if len(posted) == 1 else 0.6
        return _fake_response(_identity_action([grip] * 30))

    monkeypatch.setattr(mod.requests, "post", fake_post)
    with tempfile.TemporaryDirectory() as tmp:
        counters = {}
        client = mod.AuditedClient("127.0.0.1", 8794, Path(tmp), counters)
        out = client.step(_obs(), "do something")
        assert out.shape == (7,)
        assert abs(out[2] - 1.15) < 1e-6
        assert out[-1] == -1.0
        assert len(posted) == 1
        assert counters["model_requests"] == 1
        assert counters["model_calls"] == 1
        obs = _obs()
        query0 = np.asarray(json_numpy.loads(posted[0]["image0"]))
        query1 = np.asarray(json_numpy.loads(posted[0]["image1"]))
        assert query0.shape == (256, 256, 3)
        assert query1.shape == (256, 256, 3)
        assert np.array_equal(query0, obs["agentview_image"][::-1, ::-1])
        assert np.array_equal(query1, obs["robot0_eye_in_hand_image"])
        assert posted[0]["domain_id"] == 3
        assert posted[0]["steps"] == 10
        for _ in range(29):
            client.step(_obs(), "do something")
        assert len(posted) == 1
        out2 = client.step(_obs(), "do something")
        assert len(posted) == 2
        assert out2[-1] == 1.0
        assert abs(out2[2] - 1.15) < 1e-6
        proprio = np.asarray(json_numpy.loads(posted[1]["proprio"]))
        assert np.allclose(proprio[:9], np.array(_identity_action([0.4] * 30)[-1][:9], dtype=float))


def test_reset_clears_queue_and_proprio(monkeypatch):
    mod = _load_runner()
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return _fake_response(_identity_action([0.4] * 30))

    monkeypatch.setattr(mod.requests, "post", fake_post)
    with tempfile.TemporaryDirectory() as tmp:
        counters = {}
        client = mod.AuditedClient("127.0.0.1", 8794, Path(tmp), counters)
        client.step(_obs(), "goal")
        assert len(client.action_plan) == 29
        client.reset()
        assert len(client.action_plan) == 0
        assert client.proprio is None


def test_reject_malformed_and_nan(monkeypatch):
    mod = _load_runner()

    def bad_shape(url, json=None, timeout=None):
        return _fake_response([[0.0] * 9])

    monkeypatch.setattr(mod.requests, "post", bad_shape)
    with tempfile.TemporaryDirectory() as tmp:
        client = mod.AuditedClient("127.0.0.1", 8794, Path(tmp), {})
        with pytest.raises(Exception):
            client.step(_obs(), "goal")

    def nan_action(url, json=None, timeout=None):
        rows = _identity_action([0.4] * 30)
        rows[0][0] = float("nan")
        return _fake_response(rows)

    monkeypatch.setattr(mod.requests, "post", nan_action)
    with tempfile.TemporaryDirectory() as tmp:
        client = mod.AuditedClient("127.0.0.1", 8794, Path(tmp), {})
        with pytest.raises(Exception):
            client.step(_obs(), "goal")


def test_validate_episode_rejects_bad_rows():
    mod = _load_runner()
    base = {
        "status": "completed",
        "valid": True,
        "success": True,
        "author_score": 1,
        "independent_success": True,
        "policy_actions": 20,
        "setup_actions": 10,
        "model_calls": 1,
        "horizon": 900,
        "end_reason": "native_success",
    }
    assert mod.validate_episode(dict(base), 900)
    early = dict(base, policy_actions=5, end_reason="budget_exhausted", success=False, author_score=0,
                 independent_success=False)
    assert not mod.validate_episode(early, 900)
    disagree = dict(base, author_score=0)
    assert not mod.validate_episode(disagree, 900)
    zero_calls = dict(base, model_calls=0)
    assert not mod.validate_episode(zero_calls, 900)
    missing = {}
    assert not mod.validate_episode(missing, 900)


def test_summarize_excludes_unknown_and_keeps_attempts_separate():
    mod = _load_runner()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "metadata.json").write_text(json.dumps({"fingerprint": "test"}))
        suite = out / "libero_10" / "task_00"
        slot = suite / "episode_0"
        slot.mkdir(parents=True)
        row = {
            "status": "completed",
            "valid": True,
            "success": True,
            "author_score": 1,
            "independent_success": True,
            "policy_actions": 10,
            "setup_actions": 10,
            "model_calls": 1,
            "horizon": 900,
            "suite": "libero_10",
            "task_id": 0,
            "episode": 0,
            "end_reason": "native_success",
            "fingerprint": "test",
        }
        (slot / "result.json").write_text(json.dumps(row))
        attempt = slot / "attempt_001"
        attempt.mkdir()
        (attempt / "result.json").write_text(json.dumps(row))
        unknown = out / "libero_10" / "task_00" / "episode_1" / "attempt_001"
        unknown.mkdir(parents=True)
        (unknown / "result.json").write_text(json.dumps(dict(row, status="unknown", valid=False, success=None)))
        summary = mod.summarize(out)
        assert summary["valid_episodes"] == 1
        assert summary["successes"] == 1
        assert summary["planned"] == 200
        assert len(summary["unknown_paths"]) == 1
        csv_rows = (out / "tasks.csv").read_text().strip().splitlines()
        assert len(csv_rows) == 41


def test_summarize_fingerprint_mismatch_rejected():
    mod = _load_runner()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "metadata.json").write_text(json.dumps({"fingerprint": "test"}))
        slot = out / "libero_10" / "task_00" / "episode_0"
        slot.mkdir(parents=True)
        row = {
            "status": "completed", "valid": True, "success": True, "author_score": 1,
            "independent_success": True, "policy_actions": 10, "setup_actions": 10,
            "model_calls": 1, "horizon": 900, "suite": "libero_10", "task_id": 0,
            "episode": 0, "end_reason": "native_success", "fingerprint": "other",
        }
        (slot / "result.json").write_text(json.dumps(row))
        with pytest.raises(mod.ContractError):
            mod.summarize(out)


def test_summarize_missing_metadata_with_result_rejected():
    mod = _load_runner()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        slot = out / "libero_10" / "task_00" / "episode_0"
        slot.mkdir(parents=True)
        row = {
            "status": "completed", "valid": True, "success": True, "author_score": 1,
            "independent_success": True, "policy_actions": 10, "setup_actions": 10,
            "model_calls": 1, "horizon": 900, "suite": "libero_10", "task_id": 0,
            "episode": 0, "end_reason": "native_success", "fingerprint": "test",
        }
        (slot / "result.json").write_text(json.dumps(row))
        with pytest.raises(mod.ContractError):
            mod.summarize(out)


class _FakeController:
    name = "OSC_POSE"
    use_delta = False
    control_dim = 6
    ee_pos = np.array([0.0, 0.0, 0.0])


class _FakeSimData:
    time = 0.0


class _FakeSim:
    data = _FakeSimData()


class _FakeRobot:
    controller = _FakeController()
    sim = _FakeSim()


def test_audited_env_tracks_action_obs_and_double_close(tmp_path, monkeypatch):
    mod = _load_runner()
    actions_seen = []
    close_calls = []
    done_flag = {"value": False}
    obs = _obs()

    class _FakeNative:
        def __init__(self):
            self.env = type("E", (), {})()
            self.env.robots = [_FakeRobot()]
            self.env.sim = _FakeSim()

        def check_success(self):
            return done_flag["value"]

        def step(self, action):
            actions_seen.append(action)
            return obs, 0.0, False, {}

        def close(self):
            close_calls.append(1)

    native = _FakeNative()
    counters = {"policy_actions": 0, "setup_actions": 0, "last_native_success": False}
    ctx = {
        "out": tmp_path,
        "episode_dir": tmp_path,
        "counters": counters,
        "phase": "policy",
        "env": None,
        "disagreement": False,
    }
    wrapped = mod._AuditedEnv(native, ctx)
    ctx["env"] = wrapped
    action = np.array([0.4, -0.1, 1.15, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    wrapped.step(action)
    assert len(actions_seen) == 1
    assert actions_seen[0] is action
    assert actions_seen[0][2] == 1.15
    assert ctx["last_obs"] is obs
    rows = (tmp_path / "steps.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    done_flag["value"] = True
    with pytest.raises(mod.ContractError):
        wrapped.step(action)
    rows_after = (tmp_path / "steps.jsonl").read_text().strip().splitlines()
    assert len(rows_after) == 2
    assert counters["policy_actions"] == 2
    wrapped.close()
    wrapped.close()
    assert len(close_calls) == 1


def test_audited_env_stop_blocks_before_underlying_call(tmp_path):
    mod = _load_runner()
    actions_seen = []
    obs = _obs()

    class _FakeNative:
        def __init__(self):
            self.env = type("E", (), {})()
            self.env.robots = [_FakeRobot()]
            self.env.sim = _FakeSim()

        def check_success(self):
            return False

        def step(self, action):
            actions_seen.append(action)
            return obs, 0.0, False, {}

        def close(self):
            pass

    native = _FakeNative()
    (tmp_path / "STOP").write_text("")
    counters = {"policy_actions": 0, "setup_actions": 0, "last_native_success": False}
    ctx = {
        "out": tmp_path,
        "episode_dir": tmp_path,
        "counters": counters,
        "phase": "policy",
        "env": None,
        "disagreement": False,
    }
    wrapped = mod._AuditedEnv(native, ctx)
    ctx["env"] = wrapped
    action = np.array([0.4, -0.1, 1.15, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(mod.StopRequested):
        wrapped.step(action)
    assert len(actions_seen) == 0


def test_freeze_metadata_real_installed_contract(tmp_path):
    mod = _load_runner()
    config = mod.load_config()
    metadata = mod.freeze_metadata(
        tmp_path,
        config,
        mod.sha256_file(MODULE_PATH),
        mod.sha256_file(Path(mod.author.__file__)),
    )
    assert metadata["git_heads"]["libero"] == config["source_libero"]
    assert metadata["git_heads"]["upstream"] == config["source_xvla"]
    assert metadata["weight_verification"]["sha256"] == config["model_sha256"]
    again = mod.freeze_metadata(
        tmp_path,
        config,
        mod.sha256_file(MODULE_PATH),
        mod.sha256_file(Path(mod.author.__file__)),
    )
    assert metadata["fingerprint"] == again["fingerprint"]
