#!/usr/bin/env python3
"""GPU-free regression tests for the normal scene_demo launcher's reuse gate.

The production launcher is two files that are *never executed* here:

* ``scene_demo/start_demo.ps1`` embeds a fixed stdlib-only Python launcher inside
  a PowerShell here-string that is base64-encoded into WSL.  This test reads the
  ACTUAL ``$launcher = @' ... '@`` block back out of the script with an anchored
  multiline regex, then compiles and execs it into a private namespace
  (``__name__ == "launcher_test"``), so ``main`` is never called.  Every service
  / port / subprocess / filesystem interaction is faked, so the tests are GPU-,
  network- and process-free.
* ``scene_demo/run_service.sh`` is read as text and asserted, never run.

Behaviour covered:

* the validated normal launcher pins ``--completion-mode release_verified`` and
  ``--grasp-guard-mode enforce`` (before the trailing ``"$@"`` override) while
  keeping every offline / GL / port flag;
* ``start`` reuses an already-running service only when its ``/health`` reports
  the exact workflow, model revision, completion mode AND grasp-guard mode; a
  missing / null / ``shadow`` / ``off`` guard mode (or a workflow / revision /
  completion mismatch) returns 2 before any spawn, PID write, kill or app-health
  query.
"""

from __future__ import annotations

import re
import types
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
START_DEMO_PS1 = _SCENE_DEMO / "start_demo.ps1"
RUN_SERVICE_SH = _SCENE_DEMO / "run_service.sh"

EXPECTED_WORKFLOW = "persistent_scene_v2"
EXPECTED_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
EXPECTED_COMPLETION_MODE = "release_verified"
EXPECTED_GRASP_GUARD_MODE = "enforce"

# Anchored, multiline extraction of the PowerShell here-string body:
#   $launcher = @'
#   ...python...
#   '@
_LAUNCHER_RE = re.compile(
    r"^\$launcher = @'\r?\n(?P<body>.*?)\r?\n'@[ \t]*\r?$",
    re.MULTILINE | re.DOTALL,
)


def _extract_launcher_source() -> str:
    """Return the embedded Python body of the ``$launcher`` here-string."""

    text = START_DEMO_PS1.read_text(encoding="utf-8-sig")
    match = _LAUNCHER_RE.search(text)
    if match is None:
        raise AssertionError("could not locate the $launcher here-string in start_demo.ps1")
    return match.group("body")


def _load_namespace() -> dict:
    """Compile/exec the real embedded launcher without ever calling ``main``."""

    source = _extract_launcher_source()
    namespace: dict = {"__name__": "launcher_test"}
    exec(compile(source, "scene_demo_launcher", "exec"), namespace)
    return namespace


def _service_health(**overrides) -> dict:
    health = {
        "workflow": EXPECTED_WORKFLOW,
        "model_revision": EXPECTED_REVISION,
        "completion_mode": EXPECTED_COMPLETION_MODE,
        "grasp_guard_mode": EXPECTED_GRASP_GUARD_MODE,
    }
    health.update(overrides)
    return health


class _Harness:
    """Drive the real extracted ``start`` with every side effect faked."""

    def __init__(self, service_health=None, app_health=None):
        self.namespace = _load_namespace()
        self.service_health = _service_health() if service_health is None else service_health
        self.app_health = (
            {"frontend": "scene_demo_frontend"} if app_health is None else app_health
        )
        self.service_queries: list = []
        self.app_queries: list = []
        self.spawn = mock.Mock(name="spawn")
        self.save_pid = mock.Mock(name="save_pid")
        self.kill = mock.Mock(name="kill")
        self.makedirs = mock.Mock(name="makedirs")
        self.log = mock.Mock(name="log")
        self.run_setup = mock.Mock(return_value=True, name="run_setup")
        self.isfile = mock.Mock(return_value=True, name="isfile")

        self.namespace["run_setup"] = self.run_setup
        self.namespace["port_open"] = mock.Mock(return_value=True, name="port_open")
        self.namespace["spawn"] = self.spawn
        self.namespace["save_pid"] = self.save_pid
        self.namespace["log"] = self.log
        self.namespace["http_json"] = self._http_json
        # A fake ``os`` keeps every filesystem/process call off the real machine.
        self.namespace["os"] = types.SimpleNamespace(
            path=types.SimpleNamespace(isfile=self.isfile),
            makedirs=self.makedirs,
            kill=self.kill,
            remove=mock.Mock(name="remove"),
        )

    def _http_json(self, url, timeout=5):  # noqa: ARG002
        if url.endswith("/api/health"):
            self.app_queries.append(url)
            return self.app_health
        self.service_queries.append(url)
        return self.service_health

    def start(self, service_port=8767, app_port=8081, min_free_gpu=6000):
        return self.namespace["start"](service_port, app_port, min_free_gpu)

    def assert_no_process_side_effects(self, case):
        case.assertFalse(self.spawn.called, "spawn must never be called")
        case.assertFalse(self.save_pid.called, "save_pid must never be called")
        case.assertFalse(self.kill.called, "os.kill must never be called")


class ReuseSuccessTests(unittest.TestCase):
    def test_enforce_health_reuses_service_and_returns_zero(self):
        harness = _Harness()
        self.assertEqual(harness.start(), 0)
        self.assertEqual(harness.service_queries, ["http://127.0.0.1:8767/health"])
        self.assertEqual(harness.app_queries, ["http://127.0.0.1:8081/api/health"])
        harness.run_setup.assert_called_once_with()
        harness.assert_no_process_side_effects(self)


class GuardModeMismatchTests(unittest.TestCase):
    def _assert_rejected(self, service_health):
        harness = _Harness(service_health=service_health)
        self.assertEqual(harness.start(), 2)
        self.assertEqual(harness.service_queries, ["http://127.0.0.1:8767/health"])
        self.assertEqual(harness.app_queries, [])
        harness.assert_no_process_side_effects(self)

    def test_missing_grasp_guard_mode_returns_two(self):
        health = _service_health()
        del health["grasp_guard_mode"]
        self._assert_rejected(health)

    def test_null_grasp_guard_mode_returns_two(self):
        self._assert_rejected(_service_health(grasp_guard_mode=None))

    def test_shadow_grasp_guard_mode_returns_two(self):
        self._assert_rejected(_service_health(grasp_guard_mode="shadow"))

    def test_off_grasp_guard_mode_returns_two(self):
        self._assert_rejected(_service_health(grasp_guard_mode="off"))


class PriorProtectionTests(unittest.TestCase):
    """The pre-existing workflow / revision / completion reuse gate still holds."""

    def _assert_rejected(self, service_health):
        harness = _Harness(service_health=service_health)
        self.assertEqual(harness.start(), 2)
        self.assertEqual(harness.app_queries, [])
        harness.assert_no_process_side_effects(self)

    def test_workflow_mismatch_returns_two(self):
        self._assert_rejected(_service_health(workflow="some_other_workflow"))

    def test_revision_mismatch_returns_two(self):
        self._assert_rejected(_service_health(model_revision="0" * 40))

    def test_completion_mode_mismatch_returns_two(self):
        self._assert_rejected(_service_health(completion_mode="native"))


class LauncherSourceContractTests(unittest.TestCase):
    def test_embedded_namespace_pins_both_expected_modes(self):
        namespace = _load_namespace()
        self.assertEqual(namespace["EXPECTED_COMPLETION_MODE"], EXPECTED_COMPLETION_MODE)
        self.assertEqual(namespace["EXPECTED_GRASP_GUARD_MODE"], EXPECTED_GRASP_GUARD_MODE)

    def test_embedded_health_comparison_covers_grasp_guard_mode(self):
        source = _extract_launcher_source()
        self.assertIn('health.get("grasp_guard_mode")', source)
        self.assertIn(
            'mismatches.append("grasp_guard_mode=%r" % health.get("grasp_guard_mode"))',
            source,
        )
        self.assertLess(
            source.index('health.get("completion_mode")'),
            source.index('health.get("grasp_guard_mode")'),
        )

    def test_ps1_documents_that_enforce_is_required(self):
        source = START_DEMO_PS1.read_text(encoding="utf-8-sig")
        self.assertIn("grasp_guard_mode=enforce", source)

    def test_ps1_keeps_utf8_bom(self):
        self.assertTrue(START_DEMO_PS1.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_run_service_pins_enforce_before_trailing_overrides(self):
        source = RUN_SERVICE_SH.read_text(encoding="utf-8")
        guard_index = source.index("--grasp-guard-mode enforce")
        override_index = source.index('"$@"')
        self.assertLess(guard_index, override_index)
        self.assertLess(source.index("--completion-mode release_verified"), guard_index)

    def test_run_service_keeps_offline_and_environment_flags(self):
        source = RUN_SERVICE_SH.read_text(encoding="utf-8")
        for needle in (
            "set -euo pipefail",
            "export MUJOCO_GL=egl",
            "export LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config",
            "export LD_LIBRARY_PATH=/usr/lib/wsl/lib",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY || true",
            "--port 8767",
            "--run-root /home/yhwang/fyp/scene_demo/runs",
            "--completion-mode release_verified",
            "--grasp-guard-mode enforce",
        ):
            self.assertIn(needle, source)

    def test_run_service_is_lf_and_keeps_trailing_override_last(self):
        raw = RUN_SERVICE_SH.read_bytes()
        self.assertNotIn(b"\r", raw)
        text = raw.decode("utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertTrue(text.rstrip("\n").endswith('"$@"'))
        self.assertNotIn("--grasp-guard-mode shadow", text)
        self.assertNotIn("--grasp-guard-mode off", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
