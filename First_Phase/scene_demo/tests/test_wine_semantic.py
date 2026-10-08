#!/usr/bin/env python3
"""GPU-free tests for ``scene_demo/wine_semantic.py`` (semantic_wine_rack_v1).

The tests build a *native-shaped* fake environment (objects_dict with named
free joints, a gripper, a contact-geom grasp probe, a ``sim.model.geom_name2id``
resolver, ``sim.data.contact`` entries and a ``mujoco.mj_contactForce``
stand-in) and drive the read-only ``read_wine_semantic`` probe plus the pure
``score_wine_semantic`` / ``SemanticTracker`` helpers.  ``wine_rack_1`` is a
LIBERO *fixture*: it is absent from ``objects_dict`` and is reachable only
through the native ``inner.get_object`` getter, so the fake env reproduces that
exact fixture interface.  No simulator, model, checkpoint, GPU or Hermes process
is ever created.
"""

from __future__ import annotations

import dataclasses
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_SCENE_DEMO = Path(__file__).resolve().parent.parent
if str(_SCENE_DEMO) not in sys.path:
    sys.path.insert(0, str(_SCENE_DEMO))

import wine_semantic as ws  # noqa: E402

FREE = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
TILTED = [0.0, 0.0, 0.1, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0]
REST6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

GEOM_NAME2ID = {"wine_geom": 10, "rack_geom": 11, "plate_geom": 12, "bowl_geom": 13}
WINE_GEOM_ID = 10
RACK_GEOM_ID = 11

_UNSET = object()


class _Contact:
    """One native contact entry (``geom1``/``geom2``/``pos``/``frame``/``dist``)."""

    def __init__(self, geom1, geom2, pos, normal, dist):
        self.geom1 = geom1
        self.geom2 = geom2
        self.pos = pos
        self.frame = list(normal) + [0.0] * 6
        self.dist = dist


def _support_contact(pos_z=0.08, normal=(0.0, 0.0, 1.0), dist=0.0):
    """An upward-support-like rack/wine contact at height ``pos_z``."""

    return _Contact(WINE_GEOM_ID, RACK_GEOM_ID, [0.0, 0.0, pos_z], list(normal), dist)


class _RawData:
    """The raw (unwrapped) ``sim.data._data`` mujoco data token."""

    def __init__(self, contacts):
        self.contact = list(contacts)


class _Model:
    """A native-shaped ``sim.model`` exposing ``geom_name2id`` + ``_model``."""

    def __init__(self):
        self._model = ("raw-model-token",)

    def geom_name2id(self, name):
        return GEOM_NAME2ID[name]


class _Data:
    """A native-shaped ``sim.data`` with free-joint reads and raw ``_data``."""

    def __init__(self, qpos_map, qvel_map, contacts):
        self._qpos_map = qpos_map
        self._qvel_map = qvel_map
        self._data = _RawData(contacts)

    def get_joint_qpos(self, name):
        if name not in self._qpos_map:
            raise KeyError(name)
        return np.asarray(self._qpos_map[name], dtype=np.float64)

    def get_joint_qvel(self, name):
        if name not in self._qvel_map:
            raise KeyError(name)
        return np.asarray(self._qvel_map[name], dtype=np.float64)


def _make_env(
    *,
    predicate=True,
    wine_qpos=None,
    wine_qvel=None,
    contacts=(),
    wine_grasp_raw=_UNSET,
    rack_grasp_raw=_UNSET,
    extra_objects=(),
    contact_result=False,
    wine_contact_geoms=("wine_geom",),
    rack_contact_geoms=("rack_geom",),
    wine_has_geoms=True,
    rack_has_geoms=True,
    rack_in_objects_dict=False,
    has_get_object=True,
    get_object_raises=False,
    rack_object_none=False,
):
    """Build a mock env wrapping a native-shaped read-only inner env.

    ``extra_objects`` is a list of ``(object_id, joint, geoms, has_geoms,
    raw_grasp)`` entries; a ``raw_grasp`` of ``None`` means the probe returns
    ``None`` (unknown) and ``"raise"`` means it raises.  No ``step``/``reset``/
    ``forward`` is ever expected to be called; the counters prove it.

    ``wine_rack_1`` is a LIBERO *fixture*: by default it is NOT a key of
    ``inner.objects_dict`` and is reachable only through the native
    ``inner.get_object('wine_rack_1')`` getter (with the wine likewise read via
    ``inner.get_object('wine_bottle_1')``).  ``has_get_object=False`` models an
    env without the getter; ``get_object_raises=True`` models a raising getter;
    ``rack_object_none=True`` models a getter that returns nothing.
    """

    objects: dict = {}
    fixtures: dict = {}
    qpos_map: dict = {}
    qvel_map: dict = {}
    grasp: dict = {}
    calls = {"step": 0, "reset": 0, "forward": 0}
    get_calls: list = []
    check_calls: list = []

    def _add(object_id, joint, geoms, has_geoms, qpos, qvel, raw):
        obj = types.SimpleNamespace(joints=[joint])
        if has_geoms and geoms is not None:
            obj.contact_geoms = list(geoms)
        # The native getter registry holds every object/fixture; ``objects_dict``
        # is populated separately (the rack fixture stays out of it by default).
        fixtures[object_id] = obj
        qpos_map[joint] = list(qpos)
        qvel_map[joint] = list(qvel)
        if has_geoms and geoms is not None:
            key = tuple(geoms)
            if raw is not _UNSET:
                grasp[key] = raw
            else:
                grasp.setdefault(key, False)
        return obj

    objects["wine_bottle_1"] = _add(
        "wine_bottle_1",
        "wine_j",
        wine_contact_geoms,
        wine_has_geoms,
        wine_qpos if wine_qpos is not None else FREE,
        wine_qvel if wine_qvel is not None else REST6,
        wine_grasp_raw,
    )
    rack_obj = _add(
        "wine_rack_1",
        "rack_j",
        rack_contact_geoms,
        rack_has_geoms,
        FREE,
        REST6,
        rack_grasp_raw,
    )
    # A fixture is absent from objects_dict by default; opt-in only.
    if rack_in_objects_dict:
        objects["wine_rack_1"] = rack_obj
    for object_id, joint, geoms, has_geoms, raw in extra_objects:
        objects[object_id] = _add(object_id, joint, geoms, has_geoms, FREE, REST6, raw)

    predicate_map = {ws.STANDARD_PREDICATE_KEY: predicate}

    class _Sim:
        def __init__(self):
            self.model = _Model()
            self.data = _Data(qpos_map, qvel_map, contacts)

        def forward(self):
            calls["forward"] += 1

        def step(self, action):  # noqa: ARG002
            calls["step"] += 1

    class _Inner:
        def __init__(self):
            self.objects_dict = objects
            self.object_states_dict = {}
            self.robots = [types.SimpleNamespace(gripper=types.SimpleNamespace())]
            self.sim = _Sim()
            # Native fixture registry + recorded contact/getter evidence.
            self.fixtures = fixtures
            self.get_calls = get_calls
            self.check_calls = check_calls

        def _check_grasp(self, gripper, geoms):  # noqa: ARG002
            key = tuple(geoms) if isinstance(geoms, (list, tuple)) else geoms
            outcome = grasp.get(key, False)
            if outcome == "raise":
                raise RuntimeError("grasp probe failed")
            return outcome

        def _eval_predicate(self, predicate):
            key = "|".join(str(part) for part in predicate)
            outcome = predicate_map.get(key, False)
            if outcome == "raise":
                raise RuntimeError("predicate failed")
            return outcome

        def check_contact(self, rack_object, wine_object):
            check_calls.append((rack_object, wine_object))
            if contact_result == "raise":
                raise RuntimeError("check_contact failed")
            return contact_result

        def get_object(self, object_id):
            get_calls.append(object_id)
            if get_object_raises:
                raise RuntimeError("get_object failed")
            if object_id == "wine_rack_1" and rack_object_none:
                return None
            return fixtures[object_id]

        def step(self, action):  # noqa: ARG002
            calls["step"] += 1

        def reset(self):
            calls["reset"] += 1

        def forward(self):
            calls["forward"] += 1

    if not has_get_object:
        # Model an environment that does not expose the native fixture getter.
        _Inner.get_object = None

    inner = _Inner()
    env = types.SimpleNamespace(_env=types.SimpleNamespace(env=inner))
    return env, inner, calls


class _FakeMujoco:
    """A ``mujoco.mj_contactForce`` stand-in recording its arguments."""

    def __init__(self, normal_forces=None, raise_all=False):
        self.calls: list = []
        self.raise_all = raise_all
        self.normal_forces = dict(normal_forces or {})

    def mj_contactForce(self, model, data, index, out):
        self.calls.append(
            {"model": model, "data": data, "index": index, "out_len": len(out)}
        )
        if self.raise_all:
            raise RuntimeError("force evidence unavailable")
        out[:] = 0.0
        value = self.normal_forces.get(index)
        if value is not None:
            out[0] = float(value)


def _read(env, mujoco_fake):
    with mock.patch.dict(sys.modules, {"mujoco": mujoco_fake}):
        return ws.read_wine_semantic(env)


def _valid_sample(**overrides):
    sample = {
        "spec_id": ws.SPEC.spec_id,
        "object_id": ws.SPEC.object_id,
        "target_id": ws.SPEC.target_id,
        "rack_contact": True,
        "support_contact": True,
        "linear_speed": 0.0,
        "angular_speed": 0.0,
        "held_objects": [],
        "observation_complete": True,
        "standard_predicate": False,
        "semantic_candidate": True,
        "contacts": [],
    }
    sample.update(overrides)
    return sample


class SpecContractTests(unittest.TestCase):
    def test_spec_is_fixed_printable_and_immutable(self):
        spec = ws.SPEC
        self.assertEqual(spec.spec_id, "semantic_wine_rack_v1")
        self.assertEqual(spec.object_id, "wine_bottle_1")
        self.assertEqual(spec.target_id, "wine_rack_1")
        self.assertEqual(spec.stable_samples, 20)
        self.assertEqual(spec.linear_speed_max, 0.02)
        self.assertEqual(spec.angular_speed_max, 0.2)
        self.assertEqual(spec.support_abs_normal_z_min, 0.5)
        self.assertEqual(spec.contact_height_slack_m, 0.01)
        self.assertEqual(spec.signed_distance_max, 0.001)
        # Print-able and copy-able.
        self.assertIn("semantic_wine_rack_v1", repr(spec))
        self.assertEqual(spec.as_dict()["spec_id"], "semantic_wine_rack_v1")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.stable_samples = 1  # type: ignore[misc]

    def test_standard_goal_literal_is_separate_from_semantic_support(self):
        self.assertEqual(
            ws.STANDARD_GOAL, (("on", "wine_bottle_1", "wine_rack_1_top_region"),)
        )
        self.assertEqual(
            ws.STANDARD_PREDICATE_KEY, "on|wine_bottle_1|wine_rack_1_top_region"
        )
        # The limited rubric is documented, not a certified guarantee.
        self.assertIn("not a certified safety", ws.OPERATIONAL_RUBRIC)


class ReadWineSemanticTests(unittest.TestCase):
    def test_native_false_with_support_is_semantically_true(self):
        env, _inner, calls = _make_env(
            predicate=False,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
        )
        fake = _FakeMujoco({0: 0.5})
        sample = _read(env, fake)
        self.assertIs(sample["standard_predicate"], False)  # recorded separately
        self.assertIs(sample["rack_contact"], True)
        self.assertIs(sample["support_contact"], True)
        self.assertEqual(sample["held_objects"], [])
        self.assertIs(sample["observation_complete"], True)
        self.assertEqual(sample["spec_id"], "semantic_wine_rack_v1")
        self.assertEqual(sample["object_id"], "wine_bottle_1")
        self.assertEqual(sample["target_id"], "wine_rack_1")
        self.assertIs(sample["semantic_candidate"], True)
        self.assertEqual(calls, {"step": 0, "reset": 0, "forward": 0})

    def test_twenty_stable_samples_complete_and_native_is_irrelevant(self):
        env, _inner, _calls = _make_env(
            predicate=False, contacts=[_support_contact()], wine_grasp_raw=False
        )
        sample = _read(env, _FakeMujoco({0: 0.4}))
        self.assertIs(sample["semantic_candidate"], True)
        tracker = ws.SemanticTracker()
        for _ in range(19):
            result = tracker.update(sample)
        self.assertEqual(result["state"], "incomplete")
        self.assertIs(result["semantic_success"], False)
        self.assertEqual(result["candidate_streak"], 19)
        result = tracker.update(sample)
        self.assertEqual(result["state"], "semantic_complete")
        self.assertIs(result["semantic_success"], True)
        self.assertEqual(result["candidate_streak"], 20)

    def test_native_true_but_held_wine_is_false(self):
        env, _inner, _calls = _make_env(
            predicate=True, contacts=[_support_contact()], wine_grasp_raw=True
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["standard_predicate"], True)
        self.assertIn("wine_bottle_1", sample["held_objects"])
        self.assertIs(sample["semantic_candidate"], False)

    def test_native_true_but_excessive_speed_is_false(self):
        env, _inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            wine_qvel=[0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["standard_predicate"], True)
        self.assertAlmostEqual(sample["linear_speed"], 0.05)
        self.assertIs(sample["semantic_candidate"], False)

    def test_native_true_but_support_false_is_false(self):
        env, _inner, _calls = _make_env(
            predicate=True, contacts=[], wine_grasp_raw=False, contact_result=False
        )
        sample = _read(env, _FakeMujoco({}))
        self.assertIs(sample["standard_predicate"], True)
        self.assertIs(sample["rack_contact"], False)
        self.assertIs(sample["support_contact"], False)
        self.assertIs(sample["semantic_candidate"], False)

    def test_side_contact_with_low_normal_z_is_not_support(self):
        side = _Contact(WINE_GEOM_ID, RACK_GEOM_ID, [0.0, 0.0, 0.08], [1.0, 0.0, 0.0], 0.0)
        env, _inner, _calls = _make_env(
            predicate=False, contacts=[side], wine_grasp_raw=False
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["rack_contact"], True)
        self.assertIs(sample["support_contact"], False)
        self.assertIs(sample["semantic_candidate"], False)

    def test_contact_above_com_and_open_distance_are_not_support(self):
        # Contact point is 0.05 m above the bottle COM (slack is 0.01 m) and the
        # signed distance is positive (separating).
        high = _support_contact(pos_z=0.15, dist=0.002)
        env, _inner, _calls = _make_env(
            predicate=False, contacts=[high], wine_grasp_raw=False
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["support_contact"], False)
        self.assertIs(sample["semantic_candidate"], False)

    def test_missing_force_evidence_is_unknown(self):
        env, _inner, _calls = _make_env(
            predicate=False, contacts=[_support_contact()], wine_grasp_raw=False
        )
        sample = _read(env, _FakeMujoco(raise_all=True))
        self.assertIs(sample["rack_contact"], True)
        self.assertIsNone(sample["support_contact"])
        self.assertIsNone(sample["semantic_candidate"])

    def test_absent_mujoco_binding_is_unknown(self):
        env, _inner, _calls = _make_env(
            predicate=False, contacts=[_support_contact()], wine_grasp_raw=False
        )
        sample = _read(env, None)  # import mujoco -> ImportError
        self.assertIsNone(sample["support_contact"])
        self.assertIsNone(sample["semantic_candidate"])

    def test_nonfinite_velocity_is_unknown(self):
        env, _inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            wine_qvel=[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIsNone(sample["linear_speed"])
        self.assertIsNone(sample["semantic_candidate"])

    def test_incomplete_foreign_held_screen_is_unknown(self):
        # The naive capture_snapshot bool-converts the raw None grasp to False
        # ("released"); the reused completion helper correctly reports the
        # screen as incomplete -> fail closed.
        env, _inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            extra_objects=[("plate_1", "plate_j", ("plate_geom",), True, None)],
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["observation_complete"], False)
        self.assertIsNone(sample["semantic_candidate"])

    def test_known_foreign_held_is_false(self):
        env, _inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            extra_objects=[("plate_1", "plate_j", ("plate_geom",), True, True)],
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertEqual(sample["held_objects"], ["plate_1"])
        self.assertIs(sample["observation_complete"], True)
        self.assertIs(sample["semantic_candidate"], False)

    def test_raw_check_grasp_none_on_wine_is_unknown(self):
        env, _inner, _calls = _make_env(
            predicate=True, contacts=[_support_contact()], wine_grasp_raw=None
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["observation_complete"], False)
        self.assertIsNone(sample["semantic_candidate"])

    def test_missing_contact_geoms_on_wine_is_unknown(self):
        env, _inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            wine_has_geoms=False,
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["observation_complete"], False)
        self.assertIsNone(sample["semantic_candidate"])

    def test_tilted_bottle_is_permitted(self):
        # A tilted free-joint quaternion must not be penalised: the rubric makes
        # no orientation assumption.
        env, _inner, _calls = _make_env(
            predicate=False,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            wine_qpos=TILTED,
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["semantic_candidate"], True)

    def test_unknown_standard_predicate_does_not_gate(self):
        env, _inner, _calls = _make_env(
            predicate="raise", contacts=[_support_contact()], wine_grasp_raw=False
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIsNone(sample["standard_predicate"])
        self.assertIs(sample["semantic_candidate"], True)

    def test_zero_step_reset_forward_calls(self):
        env, inner, calls = _make_env(
            predicate=False, contacts=[_support_contact()], wine_grasp_raw=False
        )
        _read(env, _FakeMujoco({0: 0.5}))
        self.assertEqual(calls, {"step": 0, "reset": 0, "forward": 0})
        # The simulator objects themselves keep their own counters untouched.
        self.assertEqual(inner.sim.model.__class__.__name__, "_Model")

    def test_mj_contactforce_receives_raw_model_data_index_and_six_vector(self):
        env, inner, _calls = _make_env(
            predicate=False, contacts=[_support_contact()], wine_grasp_raw=False
        )
        fake = _FakeMujoco({0: 0.5})
        sample = _read(env, fake)
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertIs(call["model"], inner.sim.model._model)  # raw unwrapped model
        self.assertIs(call["data"], inner.sim.data._data)  # raw unwrapped data
        self.assertEqual(call["index"], 0)
        self.assertEqual(call["out_len"], 6)
        self.assertIs(sample["support_contact"], True)
        self.assertEqual(sample["contacts"][0]["normal_force"], 0.5)
        self.assertEqual(sample["contacts"][0]["geom1"], WINE_GEOM_ID)
        self.assertEqual(sample["contacts"][0]["geom2"], RACK_GEOM_ID)

    def test_rack_fixture_absent_from_objects_dict_is_read_via_native_getter(self):
        # The confirmed LIBERO fixture interface: the rack is a fixture, absent
        # from objects_dict, and reachable only via inner.get_object.
        env, inner, calls = _make_env(
            predicate=False, contacts=[_support_contact()], wine_grasp_raw=False
        )
        self.assertNotIn("wine_rack_1", inner.objects_dict)
        self.assertIn("wine_bottle_1", inner.objects_dict)
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIs(sample["rack_contact"], True)
        self.assertIs(sample["support_contact"], True)
        self.assertIs(sample["semantic_candidate"], True)
        self.assertEqual(calls, {"step": 0, "reset": 0, "forward": 0})

    def test_exact_getter_ids_and_check_contact_object_identity(self):
        env, inner, _calls = _make_env(
            predicate=False, contacts=[_support_contact()], wine_grasp_raw=False
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        # Exactly the two fixed fixture/object ids, in order.
        self.assertEqual(inner.get_calls, ["wine_rack_1", "wine_bottle_1"])
        # The native parent contact receives the exact rack/wine objects.
        self.assertEqual(len(inner.check_calls), 1)
        rack_arg, wine_arg = inner.check_calls[0]
        self.assertIs(rack_arg, inner.fixtures["wine_rack_1"])
        self.assertIs(wine_arg, inner.fixtures["wine_bottle_1"])
        self.assertIs(sample["semantic_candidate"], True)

    def test_missing_native_getter_is_unknown_not_support(self):
        env, inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            has_get_object=False,
        )
        self.assertIs(getattr(inner, "get_object", None), None)
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIsNone(sample["rack_contact"])
        self.assertIsNone(sample["support_contact"])
        self.assertIsNone(sample["semantic_candidate"])

    def test_raising_native_getter_is_unknown_not_support(self):
        env, inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            get_object_raises=True,
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIsNone(sample["rack_contact"])
        self.assertIsNone(sample["support_contact"])
        self.assertIsNone(sample["semantic_candidate"])

    def test_getter_returning_no_object_is_unknown_not_support(self):
        env, inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            rack_object_none=True,
        )
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIsNone(sample["support_contact"])
        self.assertIsNone(sample["semantic_candidate"])

    def test_objects_dict_is_not_a_fixture_contact_fallback(self):
        # Even with the rack literally present in objects_dict, an unavailable
        # getter keeps the support evidence unknown: objects_dict is never used
        # to retrieve the fixture contact objects.
        env, inner, _calls = _make_env(
            predicate=True,
            contacts=[_support_contact()],
            wine_grasp_raw=False,
            rack_in_objects_dict=True,
            has_get_object=False,
        )
        self.assertIn("wine_rack_1", inner.objects_dict)
        sample = _read(env, _FakeMujoco({0: 0.5}))
        self.assertIsNone(sample["rack_contact"])
        self.assertIsNone(sample["support_contact"])
        self.assertIsNone(sample["semantic_candidate"])


class ScoreWineSemanticTests(unittest.TestCase):
    def test_fixed_criteria_only(self):
        self.assertIs(ws.score_wine_semantic(_valid_sample()), True)
        self.assertIsNone(ws.score_wine_semantic("not a mapping"))
        self.assertIsNone(ws.score_wine_semantic({}))
        # Incomplete all-object held screen -> unknown.
        self.assertIsNone(ws.score_wine_semantic(_valid_sample(observation_complete=False)))
        self.assertIsNone(ws.score_wine_semantic(_valid_sample(held_objects=None)))
        # Known false rack/support -> False.
        self.assertIs(ws.score_wine_semantic(_valid_sample(rack_contact=False)), False)
        self.assertIs(ws.score_wine_semantic(_valid_sample(support_contact=False)), False)
        # Unknown rack/support -> None; excessive speed -> False.
        self.assertIsNone(ws.score_wine_semantic(_valid_sample(rack_contact=None)))
        self.assertIsNone(ws.score_wine_semantic(_valid_sample(support_contact=None)))
        self.assertIs(ws.score_wine_semantic(_valid_sample(linear_speed=0.03)), False)
        self.assertIs(ws.score_wine_semantic(_valid_sample(angular_speed=0.3)), False)

    def test_arbitrary_task_plan_fields_cannot_alter_the_fixed_oracle(self):
        base = _valid_sample()
        self.assertIs(ws.score_wine_semantic(base), True)
        hostile = dict(
            base,
            task="open the drawer",
            plan={"capability_ids": ["stove_on"]},
            capability="table_both",
            goal=["in", "tomato_sauce_1", "basket_1_contain_region"],
            hermes_decision="skip",
            # A caller cannot force the verdict by writing the field either.
            semantic_candidate=False,
            standard_predicate=False,
            spec_id="attacker_spec",
            object_id="tomato_sauce_1",
            target_id="basket_1_contain_region",
        )
        self.assertIs(ws.score_wine_semantic(hostile), True)
        # And a hostile field cannot turn a fail into success.
        self.assertIs(
            ws.score_wine_semantic(dict(_valid_sample(support_contact=False), task="x")),
            False,
        )


class SemanticTrackerTests(unittest.TestCase):
    def test_unknown_resets_streak_and_success(self):
        tracker = ws.SemanticTracker()
        for _ in range(5):
            tracker.update(_valid_sample())
        self.assertEqual(tracker.candidate_streak, 5)
        result = tracker.update(_valid_sample(observation_complete=False))
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["candidate_streak"], 0)
        self.assertIsNone(result["semantic_success"])

    def test_false_resets_streak(self):
        tracker = ws.SemanticTracker()
        for _ in range(5):
            tracker.update(_valid_sample())
        result = tracker.update(_valid_sample(support_contact=False))
        self.assertEqual(result["state"], "incomplete")
        self.assertEqual(result["candidate_streak"], 0)
        self.assertIs(result["semantic_success"], False)

    def test_achieved_then_falls_becomes_false(self):
        tracker = ws.SemanticTracker()
        for _ in range(20):
            tracker.update(_valid_sample())
        self.assertIs(tracker.semantic_success, True)
        result = tracker.update(_valid_sample(rack_contact=False))
        self.assertIs(result["semantic_success"], False)
        self.assertEqual(result["candidate_streak"], 0)

    def test_stable_samples_is_configurable(self):
        tracker = ws.SemanticTracker(stable_samples=3)
        self.assertIs(tracker.update(_valid_sample())["semantic_success"], False)
        self.assertIs(tracker.update(_valid_sample())["semantic_success"], False)
        result = tracker.update(_valid_sample())
        self.assertEqual(result["state"], "semantic_complete")
        self.assertIs(result["semantic_success"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
