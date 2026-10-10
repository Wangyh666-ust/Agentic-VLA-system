"""Pure unit tests for improve_001 pilot runner helpers."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_pilot_module():
    """Load pilot.py from the expected source path without executing heavy imports."""
    # The source path relative to this test file: First_Phase/scene_demo/finish_v1/pilot.py
    candidate = Path(__file__).resolve().parents[1] / "pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_under_test", candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pilot_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestPureHelpers(unittest.TestCase):
    """Actual DeepSeek PRO tests, with GPT-directed mechanical corrections."""
    @classmethod
    def setUpClass(cls):
        cls.mod = load_pilot_module()

    def test_validate_context_origin_ok(self):
        self.mod.validate_context_origin({"context_origin": "fixed_historical_subtask_fixture"})

    def test_validate_context_origin_bad(self):
        with self.assertRaises(RuntimeError):
            self.mod.validate_context_origin({"context_origin": "something_else"})
        with self.assertRaises(RuntimeError):
            self.mod.validate_context_origin({})

    def test_sensor_frames_list_orders_and_sets_frame(self):
        caps = [
            {"timestamp": 3.0, "frames": [{"view": "wrist"}, {"view": "agentview"}]},
            {"timestamp": 1.0, "frames": [{"view": "agentview"}]},
        ]
        rows = self.mod.sensor_frames_list(caps, existing_frames=[])
        self.assertEqual([r["timestamp"] for r in rows], [1.0, 3.0, 3.0])
        self.assertEqual([r["frame"] for r in rows], [0, 1, 2])
        self.assertTrue(all("view" in r for r in rows))

    def test_sensor_frames_list_existing_merged(self):
        existing = [{"timestamp": 0.5, "view": "agentview", "frame": 0}]
        caps = [{"timestamp": 1.5, "frames": [{"view": "wrist"}]}]
        rows = self.mod.sensor_frames_list(caps, existing_frames=existing)
        self.assertEqual([r["timestamp"] for r in rows], [0.5, 1.5])

    def test_dedup_last_n_ordered_dedups_timestamp(self):
        rows = [
            {"timestamp": 1.0, "x": 1},
            {"timestamp": 1.0, "x": 99},
            {"timestamp": 2.0, "x": 2},
            {"timestamp": 3.0, "x": 3},
            {"timestamp": 4.0, "x": 4},
        ]
        uniq = self.mod.dedup_last_n_ordered(rows, 3)
        self.assertEqual([r["timestamp"] for r in uniq], [2.0, 3.0, 4.0])
        self.assertEqual(uniq[0]["x"], 2)
        self.assertEqual(self.mod.dedup_last_n_ordered(rows, 4)[0]["x"], 99)

    def test_predicate_key(self):
        self.assertEqual(self.mod.predicate_key(["on", "a", "b"]), "on|a|b")

    def test_negative_routing(self):
        self.assertEqual(self.mod.negative_routing({"state": "incomplete"}, False), "correct_negative")
        self.assertEqual(self.mod.negative_routing({"state": "unknown"}, False), "correct_negative")
        self.assertEqual(self.mod.negative_routing({"state": "complete"}, False), "false_positive")
        self.assertIsNone(self.mod.negative_routing({"state": "incomplete"}, True))

    def test_negative_routing_all_fixed_states(self):
        for state in ("incomplete", "unknown", "needs_release", "needs_retreat", "complete"):
            self.assertIsNone(self.mod.negative_routing({"state": state}, True))
            expected = "correct_negative" if state in ("incomplete", "unknown") else "false_positive"
            self.assertEqual(self.mod.negative_routing({"state": state}, False), expected)

    def test_eligible_branch_uses_allow_helper(self):
        import inspect
        source = inspect.getsource(self.mod.PilotRunner.run_case)
        self.assertIn("if not allow_helper:", source)
        self.assertNotIn('if decision_state != "complete":', source)

    def test_no_phantom_state_or_bare_sha_call(self):
        import ast
        source = Path(self.mod.__file__).read_text()
        self.assertNotIn("port.state()", source)
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sha256" for node in ast.walk(tree)))

    def test_false_goal_predicate_is_not_unknown(self):
        snap = {"predicates": {"on|a|b": False}}
        self.assertEqual(self.mod.PilotRunner._eval_gold_predicates(None, snap, [["on", "a", "b"]]), [False])

    def test_turn_ready_real_gold_layer_home_outside(self):
        runner = object.__new__(self.mod.PilotRunner)
        gold_snapshot = {
            "predicates": {"turnon|flat_stove_1": True},
            "grasp_observation_complete": True,
            "objects": {"bowl": {"grasped": False}},
            "held_objects": [],
            "strict_candidate": None,
        }
        outer = {"gold": gold_snapshot, "home": {"ready": True}, "contacts": []}
        self.assertTrue(runner._is_ready_for_goal(outer["gold"], [["turnon", "flat_stove_1"]], "stove_exit"))
        outer["gold"]["predicates"]["turnon|flat_stove_1"] = False
        self.assertFalse(runner._is_ready_for_goal(outer["gold"], [["turnon", "flat_stove_1"]], "stove_exit"))

    def test_compute_unique_actual_streak(self):
        rows = [
            {"actual_step": 0, "ready": False, "home_ready": True, "contacts_empty": True},
            {"actual_step": 1, "ready": True, "home_ready": True, "contacts_empty": True},
            {"actual_step": 2, "ready": True, "home_ready": True, "contacts_empty": True},
            {"actual_step": 3, "ready": True, "home_ready": True, "contacts_empty": True},
            {"actual_step": 4, "ready": True, "home_ready": True, "contacts_empty": True},
            {"actual_step": 5, "ready": True, "home_ready": True, "contacts_empty": True},
        ]
        self.assertEqual(self.mod.compute_unique_actual_streak(rows), 5)

    def test_compute_unique_actual_streak_rejects_duplicate(self):
        rows = [
            {"actual_step": 1, "ready": True, "home_ready": True, "contacts_empty": True},
            {"actual_step": 1, "ready": True, "home_ready": True, "contacts_empty": True},
        ]
        with self.assertRaises(RuntimeError):
            self.mod.compute_unique_actual_streak(rows)

    def test_turn_ready_without_static_entry(self):
        snap = {
            "predicates": {"turnon|flat_stove_1": True},
            "grasp_observation_complete": True,
            "objects": {"stove": {"grasped": False}, "obj2": {"grasped": False}},
            "held_objects": [],
        }
        self.assertTrue(self.mod.turn_ready_without_static_entry(snap))
        snap["objects"]["obj2"]["grasped"] = True
        self.assertFalse(self.mod.turn_ready_without_static_entry(snap))

    def test_eval_gold_predicates_missing_key_raises(self):
        snap = {"predicates": {"on|a|b": True}}
        with self.assertRaises(RuntimeError):
            self.mod.PilotRunner._eval_gold_predicates(None, snap, [["on", "a", "c"]])

    def test_preflight_source_validation_with_tempfile(self):
        # Validate PilotRunner.verify_preflight checks source_sha256 mappings using temp dirs
        import types, tempfile, hashlib
        runner = object.__new__(self.mod.PilotRunner)
        runner.proposal_path = Path(tempfile.gettempdir()) / "proposal_mock.txt"
        runner.inputs_path = Path(tempfile.gettempdir()) / "inputs_mock.json"
        # Create temp proposal file
        runner.proposal_path.write_text("proposal")
        runner.inputs_path.write_text("inputs")
        # Create mock preflight directory under ROOT? We'll monkeypatch ROOT to temp
        tmp_root = Path(tempfile.mkdtemp())
        evidence = tmp_root / "First_Phase" / "scene_demo" / "results" / "2026-10-10-visual-finish" / "evidence"
        evidence.mkdir(parents=True)
        # source file
        src_file = tmp_root / "source.txt"
        src_file.write_text("src")
        # preflight json
        prop_sha = self.mod._sha256_file(runner.proposal_path)
        inputs_sha = self.mod._sha256_file(runner.inputs_path)
        src_sha = self.mod._sha256_file(src_file)
        pre = {"proposal_sha256": prop_sha, "inputs_sha256": inputs_sha, "source_sha256": {"source.txt": src_sha}}
        (evidence / "preflight.json").write_text(json.dumps(pre))
        old_root = self.mod.ROOT
        self.mod.ROOT = tmp_root
        try:
            runner.verify_preflight()
        finally:
            self.mod.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
