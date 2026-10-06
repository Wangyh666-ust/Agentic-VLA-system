#!/usr/bin/env python3
"""GPU-free contract tests for ``scene_demo/service.py``.

These tests never load a checkpoint, never build a LIBERO scene and never touch
CUDA.  They exercise:

* the pure request validators (scene version, cross-scene, busy, budgets,
  audit gating, resume rules);
* artifact path resolution (directory traversal, dot-files, suffix allow-list);
* the independent-oracle ownership / cross-scene rules;
* the public session schema (no private positions leak);
* one end-to-end worker run against an injected fake simulator, to prove the
  persistent-session plan loop works without a GPU.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_SCENE_DEMO = _TESTS_DIR.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import catalog  # noqa: E402
import service  # noqa: E402


def _ready_session(**overrides):
    """A minimal *public* session dict, as ``SessionRecord.public`` yields it."""

    base = {
        "ok": True,
        "session_id": "sess-1",
        "scene_id": "goal_table",
        "state": "ready",
        "scene_version": 0,
        "env_instance_id": 1,
        "episode_resets": 1,
        "policy_resets": 0,
        "total_steps": 0,
        "capabilities": ["bowl_to_plate", "wine_to_rack", "stove_on"],
        "images": [],
        "run_dir": "/tmp/none",
        "active_request_id": None,
        "error": None,
    }
    base.update(overrides)
    return base


class PlanValidationTests(unittest.TestCase):
    """``validate_plan_request`` -- the POST /plans gate."""

    def _payload(self, **overrides):
        base = {
            "session_id": "sess-1",
            "request_id": "req-1",
            "scene_version": 0,
            "decision": "execute",
            "capability_ids": ["bowl_to_plate"],
            "rationale": "test",
            "budget_per_subgoal": 300,
        }
        base.update(overrides)
        return base

    def test_unknown_session(self):
        error = service.validate_plan_request(None, None, self._payload())
        self.assertEqual(error["reason"], "unknown_session")
        self.assertEqual(service.status_for_reason(error["reason"]), 404)

    def test_session_not_ready(self):
        error = service.validate_plan_request(_ready_session(state="closed"), None, self._payload())
        self.assertEqual(error["reason"], "session_closed")
        self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_busy_when_plan_active(self):
        active = {"request_id": "other", "state": "running"}
        error = service.validate_plan_request(_ready_session(), active, self._payload())
        self.assertEqual(error["reason"], "busy")
        self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_stale_scene_version(self):
        error = service.validate_plan_request(
            _ready_session(scene_version=3), None, self._payload(scene_version=1)
        )
        self.assertEqual(error["reason"], "stale_scene_version")
        self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_scene_version_must_be_int(self):
        error = service.validate_plan_request(
            _ready_session(), None, self._payload(scene_version="0")
        )
        self.assertEqual(error["reason"], "stale_scene_version")

    def test_cross_scene_capability_rejected(self):
        error = service.validate_plan_request(
            _ready_session(scene_id="goal_table"),
            None,
            self._payload(capability_ids=["white_mug_left"]),
        )
        self.assertEqual(error["reason"], "cross_scene")
        self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_unknown_capability_rejected(self):
        error = service.validate_plan_request(
            _ready_session(), None, self._payload(capability_ids=["not_a_capability"])
        )
        self.assertEqual(error["reason"], "unknown_capability")
        self.assertEqual(service.status_for_reason(error["reason"]), 404)

    def test_audit_only_requires_explicit_audit(self):
        payload = self._payload(capability_ids=["table_both"], budget_per_subgoal=600)
        error = service.validate_plan_request(_ready_session(), None, payload)
        self.assertEqual(error["reason"], "audit_required")
        self.assertEqual(service.status_for_reason(error["reason"]), 409)
        # The same request is accepted once audit=true is explicit.
        payload["audit"] = True
        self.assertIsNone(service.validate_plan_request(_ready_session(), None, payload))

    def test_budget_bounds(self):
        for budget in (0, -5, 601):
            error = service.validate_plan_request(
                _ready_session(), None, self._payload(budget_per_subgoal=budget)
            )
            self.assertEqual(error["reason"], "invalid_budget", budget)

    def test_composite_audit_budget_matches_two_atomic(self):
        # One audit capability at 600 == two atomic capabilities at 300.
        one = self._payload(capability_ids=["table_both"], budget_per_subgoal=600, audit=True)
        self.assertIsNone(service.validate_plan_request(_ready_session(), None, one))
        two = self._payload(
            capability_ids=["bowl_to_plate", "wine_to_rack"],
            budget_per_subgoal=300,
            scene_version=0,
        )
        self.assertIsNone(service.validate_plan_request(_ready_session(), None, two))

    def test_total_budget_over_session_limit(self):
        error = service.validate_plan_request(
            _ready_session(),
            None,
            self._payload(
                capability_ids=["bowl_to_plate", "wine_to_rack", "stove_on", "table_both"],
                budget_per_subgoal=600,
                audit=True,
            ),
        )
        self.assertEqual(error["reason"], "invalid_budget")

    def test_duplicate_and_too_many_capabilities(self):
        duplicate = self._payload(capability_ids=["bowl_to_plate", "bowl_to_plate"])
        self.assertEqual(
            service.validate_plan_request(_ready_session(), None, duplicate)["reason"],
            "invalid_capabilities",
        )
        too_many = self._payload(capability_ids=["bowl_to_plate"] * 7, budget_per_subgoal=100)
        self.assertEqual(
            service.validate_plan_request(_ready_session(), None, too_many)["reason"],
            "invalid_capabilities",
        )

    def test_non_execute_decision_must_be_empty(self):
        clarify_with_caps = self._payload(decision="clarify", capability_ids=["bowl_to_plate"])
        self.assertEqual(
            service.validate_plan_request(_ready_session(), None, clarify_with_caps)["reason"],
            "invalid_capabilities",
        )
        clarify_empty = self._payload(decision="clarify", capability_ids=[])
        self.assertIsNone(service.validate_plan_request(_ready_session(), None, clarify_empty))

    def test_non_execute_decision_still_checks_scene_version(self):
        for decision in ("clarify", "unsupported"):
            error = service.validate_plan_request(
                _ready_session(scene_version=2),
                None,
                self._payload(decision=decision, capability_ids=[], scene_version=1),
            )
            self.assertEqual(error["reason"], "stale_scene_version", decision)
            self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_invalid_decision(self):
        error = service.validate_plan_request(
            _ready_session(), None, self._payload(decision="do_something")
        )
        self.assertEqual(error["reason"], "invalid_decision")

    def test_request_id_required(self):
        error = service.validate_plan_request(_ready_session(), None, self._payload(request_id=""))
        self.assertEqual(error["reason"], "invalid_request_id")


class ResumeValidationTests(unittest.TestCase):
    def _plan(self, **overrides):
        base = {
            "ok": True,
            "request_id": "req-1",
            "session_id": "sess-1",
            "state": "blocked",
            "decision": "execute",
            "capability_ids": ["bowl_to_plate"],
            "completed_capability_ids": [],
            "repair_history": [],
        }
        base.update(overrides)
        return base

    def test_only_blocked_plan_can_resume(self):
        error = service.validate_resume_request(
            _ready_session(), self._plan(state="completed"),
            {"scene_version": 0, "capability_ids": ["wine_to_rack"]},
        )
        self.assertEqual(error["reason"], "plan_not_blocked")

    def test_single_repair_only(self):
        error = service.validate_resume_request(
            _ready_session(),
            self._plan(repair_history=[{"capability_ids": ["x"]}]),
            {"scene_version": 0, "capability_ids": ["wine_to_rack"]},
        )
        self.assertEqual(error["reason"], "resume_exhausted")

    def test_resume_ownership(self):
        error = service.validate_resume_request(
            _ready_session(session_id="sess-2"),
            self._plan(session_id="sess-1"),
            {"scene_version": 0, "capability_ids": ["wine_to_rack"]},
        )
        self.assertEqual(error["reason"], "ownership")

    def test_resume_ok(self):
        error = service.validate_resume_request(
            _ready_session(),
            self._plan(),
            {"scene_version": 0, "capability_ids": ["wine_to_rack"]},
        )
        self.assertIsNone(error)

    def test_resume_audit_defaults_to_false(self):
        # An audit-only capability requires an explicit audit flag even on resume.
        without = service.validate_resume_request(
            _ready_session(),
            self._plan(),
            {"scene_version": 0, "capability_ids": ["table_both"], "budget_per_subgoal": 100},
        )
        self.assertEqual(without["reason"], "audit_required")
        with_audit = service.validate_resume_request(
            _ready_session(),
            self._plan(),
            {
                "scene_version": 0,
                "capability_ids": ["table_both"],
                "budget_per_subgoal": 100,
                "audit": True,
            },
        )
        self.assertIsNone(with_audit)

    def test_resume_budget_must_be_int_in_range(self):
        for bad in (True, False, 0, -1, 601, "300", 3.0):
            error = service.validate_resume_request(
                _ready_session(),
                self._plan(),
                {"scene_version": 0, "capability_ids": ["wine_to_rack"], "budget_per_subgoal": bad},
            )
            self.assertEqual(error["reason"], "invalid_budget", bad)

    def test_resume_respects_remaining_session_budget(self):
        error = service.validate_resume_request(
            _ready_session(total_steps=1900),
            self._plan(),
            {
                "scene_version": 0,
                "capability_ids": ["wine_to_rack", "stove_on"],
                "budget_per_subgoal": 100,
            },
        )
        self.assertEqual(error["reason"], "invalid_budget")


class EvaluateValidationTests(unittest.TestCase):
    def _plan(self, **overrides):
        base = {
            "request_id": "req-1",
            "session_id": "sess-1",
            "decision": "execute",
            "state": "completed",
            "scene_version": 0,
        }
        base.update(overrides)
        return base

    def test_unknown_case(self):
        error = service.validate_evaluate_request(_ready_session(), self._plan(), None, "req-1")
        self.assertEqual(error["reason"], "unknown_case")
        self.assertEqual(service.status_for_reason(error["reason"]), 404)

    def test_cross_scene_case(self):
        case = {"case_id": "mugs_standard", "scene_id": "mugs_two"}
        error = service.validate_evaluate_request(
            _ready_session(scene_id="goal_table"), self._plan(), case, "req-1"
        )
        self.assertEqual(error["reason"], "cross_scene")
        self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_ownership_requires_a_plan(self):
        case = {"case_id": "table_tidy", "scene_id": "goal_table"}
        error = service.validate_evaluate_request(_ready_session(), None, case, "req-1")
        self.assertEqual(error["reason"], "ownership")

    def test_ownership_rejects_other_session_plan(self):
        case = {"case_id": "table_tidy", "scene_id": "goal_table"}
        error = service.validate_evaluate_request(
            _ready_session(session_id="sess-1"),
            self._plan(session_id="sess-2"),
            case,
            "req-1",
        )
        self.assertEqual(error["reason"], "ownership")

    def test_ownership_ok(self):
        case = {"case_id": "table_tidy", "scene_id": "goal_table"}
        error = service.validate_evaluate_request(_ready_session(), self._plan(), case, "req-1")
        self.assertIsNone(error)

    def test_running_plan_cannot_be_evaluated(self):
        case = {"case_id": "table_tidy", "scene_id": "goal_table"}
        for state in ("queued", "running"):
            error = service.validate_evaluate_request(
                _ready_session(), self._plan(state=state), case, "req-1"
            )
            self.assertEqual(error["reason"], "busy", state)
            self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_stale_evaluation_rejected(self):
        case = {"case_id": "table_tidy", "scene_id": "goal_table"}
        error = service.validate_evaluate_request(
            _ready_session(scene_version=2),
            self._plan(state="completed", scene_version=1),
            case,
            "req-1",
        )
        self.assertEqual(error["reason"], "stale_evaluation")
        self.assertEqual(service.status_for_reason(error["reason"]), 409)

    def test_terminal_plan_with_current_version_ok(self):
        case = {"case_id": "table_tidy", "scene_id": "goal_table"}
        for state in service.TERMINAL_PLAN_STATES:
            error = service.validate_evaluate_request(
                _ready_session(scene_version=3),
                self._plan(state=state, scene_version=3),
                case,
                "req-1",
            )
            self.assertIsNone(error, state)


class ArtifactPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "sess").mkdir()
        (self.root / "sess" / "latest.png").write_bytes(b"png")
        (self.root / "sess" / "rollout.mp4").write_bytes(b"mp4")
        (self.root / "result.json").write_text("{}", encoding="utf-8")
        (self.root / "secret.env").write_text("TOKEN=1", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_artifacts_resolve(self):
        for rel in ("sess/latest.png", "sess/rollout.mp4", "result.json"):
            self.assertIsNotNone(service.resolve_artifact_path(self.root, rel), rel)

    def test_directory_traversal_rejected(self):
        for rel in ("../secret.env", "sess/../../etc/passwd", "..\\secret.env"):
            self.assertIsNone(service.resolve_artifact_path(self.root, rel), rel)

    def test_absolute_path_rejected(self):
        self.assertIsNone(service.resolve_artifact_path(self.root, "/etc/passwd.png"))

    def test_dotfiles_rejected(self):
        self.assertIsNone(service.resolve_artifact_path(self.root, "secret.env"))
        self.assertIsNone(service.resolve_artifact_path(self.root, ".git/config.json"))

    def test_disallowed_suffix_rejected(self):
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        self.assertIsNone(service.resolve_artifact_path(self.root, "notes.txt"))

    def test_missing_file_rejected(self):
        self.assertIsNone(service.resolve_artifact_path(self.root, "sess/missing.png"))


class PublicSchemaTests(unittest.TestCase):
    def test_health_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service.SceneService(run_root=tmp)
            health = svc.health()
            self.assertTrue(health["ok"])
            self.assertEqual(health["workflow"], "persistent_scene_v2")
            self.assertFalse(health["ready"])
            self.assertIsNone(health["worker_error"])

    def test_scenes_lists_all_seven(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service.SceneService(run_root=tmp)
            payload = svc.scenes()
            ids = [entry["scene_id"] for entry in payload["scenes"]]
            self.assertEqual(len(ids), 7)
            self.assertEqual(set(ids), set(catalog.SCENES))

    def test_create_session_rejects_unknown_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service.SceneService(run_root=tmp)
            result = svc.create_session("no_such_scene")
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "unknown_scene")
            self.assertEqual(service.status_for_reason(result["reason"]), 404)

    def test_unknown_lookups(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = service.SceneService(run_root=tmp)
            self.assertIsNone(svc.session("nope"))
            self.assertIsNone(svc.plan("nope"))
            self.assertIsNone(svc.job("nope"))
            self.assertEqual(svc.cancel("nope")["reason"], "unknown_plan")
            self.assertEqual(svc.observe("nope")["reason"], "unknown_session")

    def test_session_public_never_leaks_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = service.SessionRecord(
                "sid", "goal_table", catalog.SCENES["goal_table"], 0, 0, Path(tmp)
            )
            record.initial_positions = {"akita_black_bowl_1": [1.0, 2.0, 3.0]}
            record.initial_state_path = "/private/initial_state.npy"
            public = record.public()
            for leaked in ("initial_positions", "positions", "initial_state_path", "xml_sha"):
                self.assertNotIn(leaked, public)

    def test_v1_bridge_reuse(self):
        self.assertIn("v1_libero_service", sys.modules)
        # ``_load_policy`` is a method of the v1 ``LiberoService`` class, not a
        # module-level function.
        self.assertTrue(hasattr(service.v1_libero_service.LiberoService, "_load_policy"))
        self.assertEqual(
            service.MODEL_REVISION_DEFAULT, service.v1_libero_service.MODEL_REVISION_DEFAULT
        )

    def test_persistent_env_overrides_ensure_env(self):
        self.assertIn("_ensure_env", service.PersistentLiberoEnv.__dict__)
        self.assertTrue(issubclass(service.PersistentLiberoEnv, service.LiberoEnv))


# --- end-to-end worker test with an injected fake simulator -----------------


class _FakeSimData:
    """Minimal native ``sim.data``: joint qpos by name + body centres by index."""

    def __init__(self):
        self.qpos = np.zeros(7)
        self.qvel = np.zeros(7)
        self.body_xpos = np.zeros((8, 3))
        self.joint_qpos: dict[str, np.ndarray] = {}

    def get_joint_qpos(self, name):
        return self.joint_qpos.setdefault(name, np.zeros(7, dtype=np.float64))

    def set_joint_qpos(self, name, value):
        self.joint_qpos[name] = np.asarray(value, dtype=np.float64).copy()


class _FakeMjSimState:
    """Native ``MjSimState`` stand-in: ``state_sha`` only uses ``flatten()``.

    The real object does not support ``np.asarray`` directly, so the service must
    call ``.flatten()`` first; ``flat_calls`` lets a test prove that path ran.
    """

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float64)
        self.flat_calls = 0

    def flatten(self):
        self.flat_calls += 1
        return self._values.copy()


class _FakeSim:
    def __init__(self):
        self.data = _FakeSimData()
        self.model = types.SimpleNamespace(
            body_names=[], camera_names=[], get_xml=lambda: "<xml/>"
        )
        # A value that lives *only* in the returned MjSimState (never in
        # qpos/qvel): a test can mutate it to prove the get_state()/.flatten()
        # path is genuinely used rather than the qpos/qvel fallback.
        self.extra_state = np.zeros(1, dtype=np.float64)
        self.get_state_calls = 0
        self.last_state = None

    def get_state(self):
        self.get_state_calls += 1
        self.last_state = _FakeMjSimState(
            np.concatenate(
                [
                    np.asarray(self.data.qpos, dtype=np.float64),
                    np.asarray(self.data.qvel, dtype=np.float64),
                    np.asarray(self.extra_state, dtype=np.float64),
                ]
            )
        )
        return self.last_state

    def forward(self):
        pass


class _FakeObject:
    """Native robosuite object: ``joints`` is a list of joint-name strings."""

    def __init__(self, joint_name):
        self.joints = [joint_name]


class _FakeInnerEnv:
    """Native-shaped inner env: object dict, body ids, one-arg predicate probe."""

    def __init__(self, default_truth_at=3):
        self.sim = _FakeSim()
        self.steps = 0
        self.default_truth_at = default_truth_at
        self.truth_after: dict[str, int] = {}
        self.objects_dict = {
            "akita_black_bowl_1": _FakeObject("akita_black_bowl_1_joint0"),
            "plate_1": _FakeObject("plate_1_joint0"),
        }
        self.object_states_dict = {"akita_black_bowl_1": "On"}
        self.obj_body_id = {"akita_black_bowl_1": 0, "plate_1": 1}
        self.sim.data.body_xpos[0] = [0.0, 0.0, 0.0]
        self.sim.data.body_xpos[1] = [0.5, 0.6, 0.7]

    def _get_observations(self):
        return {}

    def _eval_predicate(self, predicate):
        key = "|".join(str(part) for part in predicate)
        threshold = self.truth_after.get(key, self.default_truth_at)
        return self.steps >= threshold

    def step(self, action):
        self.steps += 1
        # Advance the mock simulator state so state hashes genuinely move: the
        # no-reset seam then really depends on the env persisting between jobs.
        self.sim.data.qpos[0] = float(self.steps)
        return {}


def _wrap(inner):
    """Wrap a native inner env the way ``PersistentLiberoEnv`` does."""

    return types.SimpleNamespace(_env=types.SimpleNamespace(env=inner))


class _FakeObs:
    pixels = {}
    robot_state = {}


class _FakeEnv:
    def __init__(self, gate=None):
        self._inner = _FakeInnerEnv(default_truth_at=3)
        self._env = types.SimpleNamespace(env=self._inner)
        self.action_space = types.SimpleNamespace(low=-np.ones(7), high=np.ones(7))
        self.closed = False
        self.gate = gate  # optional threading.Event that throttles env.step

    def reset(self, seed=0):
        return _FakeObs(), {}

    def step(self, action):
        if self.gate is not None:
            self.gate.wait(timeout=15.0)
        self._inner.step(action)
        return _FakeObs(), 0.0, False, False, {"is_success": False}

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class NativeApiShapeTests(unittest.TestCase):
    """The patch/position/predicate helpers match the verified native API."""

    def test_object_joint_is_a_name_string(self):
        inner = _FakeInnerEnv()
        name = service._object_joint(_wrap(inner), "akita_black_bowl_1")
        self.assertIsInstance(name, str)
        self.assertEqual(name, "akita_black_bowl_1_joint0")

    def test_offset_patch_reads_and_writes_joint_qpos(self):
        inner = _FakeInnerEnv()
        inner.sim.data.joint_qpos["akita_black_bowl_1_joint0"] = np.array(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        )
        recorded: dict = {}

        def _set(name, value):
            recorded["name"] = name
            recorded["value"] = np.asarray(value, dtype=np.float64).copy()

        inner.sim.data.set_joint_qpos = _set
        service.apply_scene_patch(
            _wrap(inner),
            {"kind": "offset", "object_id": "akita_black_bowl_1", "xyz": [0.1, -0.2, 0.3]},
        )
        self.assertEqual(recorded["name"], "akita_black_bowl_1_joint0")
        np.testing.assert_allclose(recorded["value"][:3], [1.1, 1.8, 3.3])
        # the quaternion tail is preserved
        np.testing.assert_allclose(recorded["value"][3:], [1.0, 0.0, 0.0, 0.0])

    def test_place_on_patch_centres_on_target_body(self):
        inner = _FakeInnerEnv()
        inner.sim.data.joint_qpos["akita_black_bowl_1_joint0"] = np.array(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        )
        recorded: dict = {}

        def _set(name, value):
            recorded["value"] = np.asarray(value, dtype=np.float64).copy()

        inner.sim.data.set_joint_qpos = _set
        service.apply_scene_patch(
            _wrap(inner),
            {"kind": "place_on", "object_id": "akita_black_bowl_1", "target_id": "plate_1"},
        )
        # plate_1 body centre [0.5, 0.6, 0.7] + [0, 0, 0.05]
        np.testing.assert_allclose(recorded["value"][:3], [0.5, 0.6, 0.75])
        np.testing.assert_allclose(recorded["value"][3:], [1.0, 0.0, 0.0, 0.0])

    def test_body_position_uses_obj_body_id_not_body_names(self):
        inner = _FakeInnerEnv()
        # deliberately wrong body_names: the id must never be looked up there
        inner.sim.model.body_names = ["totally_different_body"]
        inner.obj_body_id = {"plate_1": 5}
        inner.sim.data.body_xpos = np.zeros((6, 3))
        inner.sim.data.body_xpos[5] = [9.0, 8.0, 7.0]
        np.testing.assert_allclose(
            service._body_position(_wrap(inner), "plate_1"), [9.0, 8.0, 7.0]
        )

    def test_object_states_uses_exact_native_mapping(self):
        inner = _FakeInnerEnv()
        inner.object_states = {"speculative": True}  # decoy must be ignored
        inner.object_states_dict = {"akita_black_bowl_1": "On"}
        self.assertEqual(service._object_states(inner), {"akita_black_bowl_1": "On"})


class RefreshObservationTests(unittest.TestCase):
    """``_refresh_observation`` must never step, and must fail closed."""

    def _service(self):
        return service.SceneService(run_root=tempfile.mkdtemp())

    def test_real_path_uses_native_getter_and_formatter(self):
        seen: dict = {}

        class _Inner:
            def _get_observations(self, force_update=False):
                seen["force_update"] = force_update
                return {"raw": 1}

        class _Env:
            def __init__(self):
                self._env = types.SimpleNamespace(env=_Inner())

            def _format_raw_obs(self, raw):
                seen["raw"] = raw
                return {"formatted": raw}

        obs = self._service()._refresh_observation(_Env())
        self.assertTrue(seen["force_update"])
        self.assertEqual(seen["raw"], {"raw": 1})
        self.assertEqual(obs, {"formatted": {"raw": 1}})

    def test_fake_getter_without_force_update_is_used_without_stepping(self):
        inner = _FakeInnerEnv()
        before = inner.steps
        obs = self._service()._refresh_observation(_wrap(inner))
        self.assertEqual(obs, {})
        self.assertEqual(inner.steps, before)  # no hidden control step

    def test_refresh_impossible_fails_closed(self):
        env = types.SimpleNamespace(_env=types.SimpleNamespace(env=types.SimpleNamespace()))
        with self.assertRaises(service.SceneError) as ctx:
            self._service()._refresh_observation(env)
        self.assertEqual(ctx.exception.reason, "invalid_fixture")
        self.assertEqual(service.status_for_reason("invalid_fixture"), 400)


class PredicateApiTests(unittest.TestCase):
    def test_eval_predicate_receives_single_list_argument(self):
        seen: list = []

        class _Recorder:
            def _eval_predicate(self, predicate):
                seen.append(predicate)
                return True

        env = types.SimpleNamespace(_env=types.SimpleNamespace(env=_Recorder()))
        self.assertTrue(service.eval_goal_predicate(env, ["on", "a", "b"]))
        self.assertEqual(seen, [["on", "a", "b"]])

    def test_predicate_exception_raises_not_false(self):
        class _Boom:
            def _eval_predicate(self, predicate):
                raise RuntimeError("boom")

        env = types.SimpleNamespace(_env=types.SimpleNamespace(env=_Boom()))
        with self.assertRaises(service.SceneError) as ctx:
            service.eval_goal_predicate(env, ["on", "a", "b"])
        self.assertEqual(ctx.exception.reason, "predicate_error")

    def test_predicate_exception_under_not_still_raises(self):
        class _Boom:
            def _eval_predicate(self, predicate):
                raise RuntimeError("boom")

        env = types.SimpleNamespace(_env=types.SimpleNamespace(env=_Boom()))
        with self.assertRaises(service.SceneError) as ctx:
            # a failed inner query must never become True through negation
            service.eval_goal_predicate(env, ["not", "on", "a", "b"])
        self.assertEqual(ctx.exception.reason, "predicate_error")

    def test_predicate_exception_under_nested_not_still_raises(self):
        class _Boom:
            def _eval_predicate(self, predicate):
                raise RuntimeError("boom")

        env = types.SimpleNamespace(_env=types.SimpleNamespace(env=_Boom()))
        with self.assertRaises(service.SceneError) as ctx:
            # even double negation cannot launder a failed inner query into truth
            service.eval_goal_predicate(env, ["not", "not", "on", "a", "b"])
        self.assertEqual(ctx.exception.reason, "predicate_error")

    def test_not_negates_only_successful_booleans(self):
        true_env = _wrap(_FakeInnerEnv(default_truth_at=0))
        false_env = _wrap(_FakeInnerEnv(default_truth_at=10 ** 9))
        self.assertFalse(service.eval_goal_predicate(true_env, ["not", "on", "a", "b"]))
        self.assertTrue(service.eval_goal_predicate(false_env, ["not", "on", "a", "b"]))


class StateHashTests(unittest.TestCase):
    def test_state_sha_sees_a_tiny_real_float64_change(self):
        inner = _FakeInnerEnv()
        env = _wrap(inner)
        first = service.state_sha(env)
        inner.sim.data.qvel[0] = 1e-12  # far below any rounding grid
        second = service.state_sha(env)
        self.assertNotEqual(first, second)
        inner.sim.data.qvel[0] = 0.0
        self.assertEqual(first, service.state_sha(env))

    def test_state_sha_uses_get_state_flatten_path(self):
        inner = _FakeInnerEnv()
        env = _wrap(inner)
        sim = inner.sim
        first = service.state_sha(env)
        # The native getter ran and its MjSimState was actually flattened.
        self.assertGreaterEqual(sim.get_state_calls, 1)
        self.assertIsNotNone(sim.last_state)
        self.assertGreaterEqual(sim.last_state.flat_calls, 1)
        # A change that exists only in the MjSimState (never in qpos/qvel) must
        # change the digest; the qpos/qvel fallback would ignore it entirely.
        sim.extra_state[0] = 0.5
        self.assertNotEqual(first, service.state_sha(env))

    def test_state_sha_falls_back_to_qpos_qvel_only_on_exception(self):
        inner = _FakeInnerEnv()
        env = _wrap(inner)
        inner.sim.data.qpos[0] = 2.0

        def _boom():
            raise RuntimeError("no get_state here")

        inner.sim.get_state = _boom
        # Must not raise; the digest must come from the raw qpos/qvel vectors.
        flat = np.concatenate(
            [inner.sim.data.qpos, inner.sim.data.qvel]
        ).astype(np.float64)
        expected = service._sha256_bytes(
            np.ascontiguousarray(flat, dtype=np.float64).tobytes()
        )
        self.assertEqual(service.state_sha(env), expected)


class FinalisePlanTests(unittest.TestCase):
    """``_finalise_plan_success`` must never mark unsatisfied goals completed."""

    def _service(self, truth_at):
        svc = service.SceneService(run_root=tempfile.mkdtemp())
        svc._env = _FakeEnv()
        svc._env._inner.default_truth_at = truth_at
        return svc

    def test_missing_declared_goal_blocks_without_regressions(self):
        svc = self._service(10 ** 9)  # every predicate is false
        plan = service.PlanRecord(
            "req", "sess", {"decision": "execute", "capability_ids": ["bowl_to_plate"]}
        )
        plan.completed_capability_ids = []  # no completed goals -> no regressions
        svc._finalise_plan_success(plan)
        self.assertEqual(plan.state, "blocked")
        self.assertFalse(plan.plan_success)
        self.assertEqual(plan.regressions, [])
        self.assertIn("on|akita_black_bowl_1|plate_1", plan.error)

    def test_completed_goal_regressions_are_preserved(self):
        svc = self._service(10 ** 9)
        plan = service.PlanRecord(
            "req", "sess", {"decision": "execute", "capability_ids": ["bowl_to_plate"]}
        )
        plan.completed_capability_ids = ["bowl_to_plate"]
        svc._finalise_plan_success(plan)
        self.assertEqual(plan.state, "blocked")
        self.assertEqual(plan.regressions, [["on", "akita_black_bowl_1", "plate_1"]])

    def test_repair_cannot_erase_originally_declared_goals(self):
        # Only the bowl goal is true; the wine goal stays false.
        svc = self._service(10 ** 9)
        svc._env._inner.truth_after["on|akita_black_bowl_1|plate_1"] = 0
        plan = service.PlanRecord(
            "req",
            "sess",
            {
                "decision": "execute",
                "capability_ids": ["bowl_to_plate", "wine_to_rack"],
            },
        )
        # The snapshot is an independent copy of the initial declaration.
        self.assertEqual(plan.original_capability_ids, ["bowl_to_plate", "wine_to_rack"])
        snapshot = plan.public()
        self.assertEqual(
            snapshot["original_capability_ids"], ["bowl_to_plate", "wine_to_rack"]
        )
        self.assertIsNot(snapshot["original_capability_ids"], plan.original_capability_ids)
        snapshot["original_capability_ids"].append("stove_on")
        self.assertEqual(
            plan.original_capability_ids, ["bowl_to_plate", "wine_to_rack"]
        )

        # A repair dropped the failed wine goal and kept only the bowl: the
        # originally declared wine goal must still block the plan.
        plan.capability_ids = ["bowl_to_plate"]
        plan.pending_capability_ids = ["bowl_to_plate"]
        plan.completed_capability_ids = ["bowl_to_plate"]
        svc._finalise_plan_success(plan)
        self.assertEqual(plan.state, "blocked", plan.public())
        self.assertFalse(plan.plan_success)
        self.assertIn("on|wine_bottle_1|wine_rack_1_top_region", plan.error)


class PersistentWorkerTests(unittest.TestCase):
    """One real worker run against a fake env: no torch, no LIBERO, no GPU."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gate = threading.Event()
        self.gate.set()
        self.service = service.SceneService(
            run_root=self._tmp.name,
            env_factory=lambda **kwargs: _FakeEnv(gate=self.gate),
            policy_loader=lambda svc: setattr(svc._v1, "_n_action_steps", 10),
            action_function=lambda batch: np.zeros(7),
            batch_builder=lambda obs, instruction: {},
        )
        self.service.start()
        deadline = time.time() + 20.0
        while time.time() < deadline and not self.service.health()["ready"]:
            time.sleep(0.05)

    def tearDown(self):
        self.gate.set()
        self.service.stop()
        self._tmp.cleanup()

    def _wait_for_plan(self, request_id, deadline_s=20.0):
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            payload = self.service.plan(request_id)
            if payload is not None and payload["state"] in service.TERMINAL_PLAN_STATES:
                return payload
            time.sleep(0.05)
        self.fail("plan %s did not reach a terminal state" % request_id)

    def test_session_and_plan_lifecycle(self):
        session = self.service.create_session("goal_table", seed=0, init_state_index=0)
        self.assertTrue(session["ok"], session)
        self.assertEqual(session["state"], "ready")
        self.assertEqual(session["episode_resets"], 1)
        self.assertEqual(session["env_instance_id"], 1)
        session_id = session["session_id"]

        submitted = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-1",
                "capability_ids": ["bowl_to_plate"],
                "rationale": "goal_table test",
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(submitted["ok"], submitted)
        self.assertEqual(submitted["state"], "queued")

        plan = self._wait_for_plan("req-1")
        self.assertEqual(plan["state"], "completed", plan)
        self.assertTrue(plan["plan_success"])
        self.assertEqual(plan["completed_capability_ids"], ["bowl_to_plate"])

        job_id = plan["job_ids"][0]
        job = self.service.job(job_id)
        self.assertEqual(job["success"], True)
        self.assertEqual(job["ended_reason"], "success")
        self.assertGreaterEqual(job["steps"], service.GOAL_CONSECUTIVE_STEPS)

        after = self.service.session(session_id)
        self.assertEqual(after["scene_version"], 1)  # incremented per capability
        self.assertEqual(after["episode_resets"], 1)  # still a single reset
        self.assertEqual(after["active_request_id"], None)
        # A terminal plan records the owning session's current scene version.
        self.assertEqual(plan["scene_version"], after["scene_version"])

    def test_busy_rejection_via_service(self):
        session = self.service.create_session("goal_table")
        session_id = session["session_id"]
        # Close the gate so the first plan stays deterministically running.
        self.gate.clear()
        first = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-a",
                "capability_ids": ["bowl_to_plate"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(first["ok"], first)
        deadline = time.time() + 10.0
        while time.time() < deadline and self.service.plan("req-a")["state"] != "running":
            time.sleep(0.02)
        self.assertEqual(self.service.plan("req-a")["state"], "running")

        second = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-b",
                "capability_ids": ["stove_on"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertFalse(second["ok"])
        self.assertEqual(second["reason"], "busy")

        self.gate.set()
        self._wait_for_plan("req-a")

    def test_clarify_plan_creates_zero_jobs(self):
        session = self.service.create_session("goal_table")
        result = self.service.submit_plan(
            {
                "session_id": session["session_id"],
                "scene_version": 0,
                "request_id": "req-c",
                "capability_ids": [],
                "rationale": "ambiguous",
                "decision": "clarify",
            }
        )
        self.assertTrue(result["ok"], result)
        plan = self._wait_for_plan("req-c")
        self.assertEqual(plan["state"], "completed")
        self.assertEqual(plan["job_ids"], [])
        self.assertIsNone(plan["plan_success"])

    def test_cumulative_total_steps_are_session_wide(self):
        session = self.service.create_session("goal_table")
        self.assertTrue(session["ok"], session)
        # Make the second capability satisfiable only well after the first, so a
        # per-job (local) counter would report a strictly smaller total.
        self.service._env._inner.truth_after["turnon|flat_stove_1"] = 9
        submitted = self.service.submit_plan(
            {
                "session_id": session["session_id"],
                "scene_version": 0,
                "request_id": "req-cum",
                "capability_ids": ["bowl_to_plate", "stove_on"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(submitted["ok"], submitted)
        plan = self._wait_for_plan("req-cum")
        self.assertEqual(plan["state"], "completed", plan)
        job1 = self.service.job(plan["job_ids"][0])
        job2 = self.service.job(plan["job_ids"][1])
        after = self.service.session(session["session_id"])
        self.assertEqual(job2["total_steps"], after["total_steps"])
        self.assertEqual(job2["total_steps"], 13)
        self.assertLess(job2["steps"], job2["total_steps"])  # local != cumulative
        self.assertGreater(job2["total_steps"], job1["total_steps"])

    def test_stale_evaluation_after_scene_change(self):
        session = self.service.create_session("goal_table")
        session_id = session["session_id"]
        first = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-e1",
                "capability_ids": ["bowl_to_plate"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(first["ok"], first)
        plan = self._wait_for_plan("req-e1")
        self.assertEqual(plan["state"], "completed", plan)
        # The finished plan now matches the session, so evaluation is valid.
        self.assertEqual(
            self.service.plan("req-e1")["scene_version"],
            self.service.session(session_id)["scene_version"],
        )
        # A later capability bumps the scene version -> the first plan is stale.
        second = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": self.service.session(session_id)["scene_version"],
                "request_id": "req-e2",
                "capability_ids": ["stove_on"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(second["ok"], second)
        self._wait_for_plan("req-e2")
        result = self.service.evaluate(session_id, "table_tidy", "req-e1")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "stale_evaluation")
        self.assertEqual(service.status_for_reason(result["reason"]), 409)

    def test_two_sequential_subgoals_share_one_env(self):
        session = self.service.create_session("goal_table", seed=0, init_state_index=0)
        self.assertTrue(session["ok"], session)
        session_id = session["session_id"]
        # Make stove_on satisfiable only after the first capability's steps: the
        # second job is then neither already-satisfied nor a per-job counter.
        self.service._env._inner.truth_after["turnon|flat_stove_1"] = 9
        submitted = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-seq",
                "capability_ids": ["bowl_to_plate", "stove_on"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(submitted["ok"], submitted)
        plan = self._wait_for_plan("req-seq")
        self.assertEqual(plan["state"], "completed", plan)
        job1 = self.service.job(plan["job_ids"][0])
        job2 = self.service.job(plan["job_ids"][1])
        after = self.service.session(session_id)

        # One persistent environment and exactly one reset across both subgoals.
        self.assertEqual(job1["env_instance_id"], job2["env_instance_id"])
        self.assertEqual(after["env_instance_id"], job1["env_instance_id"])
        self.assertEqual(after["episode_resets"], 1)

        # The scene flows contiguously: the end of job1 is the start of job2.
        self.assertEqual(job1["state_after_sha"], job2["state_before_sha"])

        # The step counter is session-wide, not per job (fresh session).
        self.assertEqual(job2["total_steps"], job1["steps"] + job2["steps"])
        self.assertEqual(job2["total_steps"], after["total_steps"])
        self.assertLess(job2["steps"], job2["total_steps"])

        # The terminal plan records the session's current scene version.
        self.assertEqual(plan["scene_version"], after["scene_version"])

    def test_request_conflict_rejects_reused_request_id(self):
        session = self.service.create_session("goal_table")
        session_id = session["session_id"]
        first = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-dup",
                "capability_ids": [],
                "decision": "clarify",
                "rationale": "ambiguous",
            }
        )
        self.assertTrue(first["ok"], first)
        completed = self._wait_for_plan("req-dup")
        self.assertEqual(completed["state"], "completed", completed)
        snapshot = self.service.plan("req-dup")

        again = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-dup",
                "capability_ids": [],
                "decision": "clarify",
                "rationale": "again",
            }
        )
        self.assertFalse(again["ok"])
        self.assertEqual(again["reason"], "request_conflict")
        self.assertEqual(service.status_for_reason(again["reason"]), 409)
        # The original record is untouched: no overwrite, no idempotent replace.
        self.assertEqual(self.service.plan("req-dup"), snapshot)

    def test_evaluate_busy_while_new_request_runs(self):
        session = self.service.create_session("goal_table")
        session_id = session["session_id"]
        first = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-old",
                "capability_ids": ["bowl_to_plate"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(first["ok"], first)
        old = self._wait_for_plan("req-old")
        self.assertEqual(old["state"], "completed", old)
        version = self.service.session(session_id)["scene_version"]

        # The newer request must genuinely step (not be already satisfied), so
        # the closed gate deterministically holds it in the running state.
        self.service._env._inner.truth_after["turnon|flat_stove_1"] = 10 ** 9
        self.gate.clear()
        second = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": version,
                "request_id": "req-new",
                "capability_ids": ["stove_on"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(second["ok"], second)
        deadline = time.time() + 10.0
        while time.time() < deadline and self.service.plan("req-new")["state"] != "running":
            time.sleep(0.02)
        self.assertEqual(self.service.plan("req-new")["state"], "running")

        # Scoring the old request now would score a later scene: refuse as busy.
        result = self.service.evaluate(session_id, "table_tidy", "req-old")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "busy")
        self.assertEqual(service.status_for_reason(result["reason"]), 409)

        self.gate.set()
        self._wait_for_plan("req-new")

    def test_resume_keeps_originally_declared_goals(self):
        session = self.service.create_session("goal_table", seed=0, init_state_index=0)
        session_id = session["session_id"]
        # The wine goal can never be satisfied; only the bowl goal can be.
        self.service._env._inner.truth_after[
            "on|wine_bottle_1|wine_rack_1_top_region"
        ] = 10 ** 9
        submitted = self.service.submit_plan(
            {
                "session_id": session_id,
                "scene_version": 0,
                "request_id": "req-repair",
                "capability_ids": ["bowl_to_plate", "wine_to_rack"],
                "decision": "execute",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(submitted["ok"], submitted)
        self.assertEqual(
            submitted["original_capability_ids"], ["bowl_to_plate", "wine_to_rack"]
        )
        blocked = self._wait_for_plan("req-repair")
        self.assertEqual(blocked["state"], "blocked", blocked)

        # A repair that drops the failed wine goal: the original declaration
        # must survive the resume untouched.
        resumed = self.service.resume_plan(
            {
                "session_id": session_id,
                "request_id": "req-repair",
                "scene_version": self.service.session(session_id)["scene_version"],
                "capability_ids": ["bowl_to_plate"],
                "rationale": "drop the failed wine goal",
                "budget_per_subgoal": 50,
            }
        )
        self.assertTrue(resumed["ok"], resumed)
        final = self._wait_for_plan("req-repair")
        self.assertEqual(
            final["original_capability_ids"], ["bowl_to_plate", "wine_to_rack"]
        )
        self.assertEqual(final["state"], "blocked", final)
        self.assertFalse(final["plan_success"])
        self.assertIn("on|wine_bottle_1|wine_rack_1_top_region", final["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
