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
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runner = object.__new__(self.mod.PilotRunner)
            runner.proposal_path = tmp_root / "proposal_mock.txt"
            runner.inputs_path = tmp_root / "inputs_mock.json"
            runner.proposal_path.write_text("proposal")
            runner.inputs_path.write_text("inputs")
            src_file = tmp_root / "source.txt"
            src_file.write_text("src")
            prop_sha = self.mod._sha256_file(runner.proposal_path)
            inputs_sha = self.mod._sha256_file(runner.inputs_path)
            src_sha = self.mod._sha256_file(src_file)
            pre = {"proposal_sha256": prop_sha, "inputs_sha256": inputs_sha, "source_sha256": {"source.txt": src_sha}}
            (tmp_root / "preflight.json").write_text(json.dumps(pre))
            old_root = self.mod.ROOT
            self.mod.ROOT = tmp_root
            try:
                runner.verify_preflight()
            finally:
                self.mod.ROOT = old_root

    def test_parse_args_defaults_and_bounds(self):
        a = self.mod._parse_args(["--proposal", "p", "--inputs", "i", "--output", "o"])
        self.assertEqual(a.limit, 5)
        self.assertEqual(a.start_index, 0)
        self.assertEqual(a.max_qwen_calls, 18)
        a = self.mod._parse_args([
            "--proposal", "p", "--inputs", "i", "--output", "o",
            "--start-index", "1", "--limit", "4", "--max-qwen-calls", "12",
        ])
        self.assertEqual(a.start_index, 1)
        self.assertEqual(a.limit, 4)
        self.assertEqual(a.max_qwen_calls, 12)

    def test_parse_args_rejects_bad_bounds(self):
        for extra in (
            ["--start-index", "5"],
            ["--start-index", "-1"],
            ["--max-qwen-calls", "0"],
            ["--max-qwen-calls", "19"],
            ["--start-index", "2", "--limit", "4"],
        ):
            with self.assertRaises(SystemExit):
                self.mod._parse_args(
                    ["--proposal", "p", "--inputs", "i", "--output", "o"] + extra
                )

    def test_takeover_metrics_negative_zero_not_failure(self):
        case = {"allow_helper": True, "expected_release": True, "expected_retreat": False}
        m = self.mod.takeover_metrics(case, {"release": 0, "movement": 0})
        self.assertTrue(m["eligible"])
        self.assertFalse(m["target_executed"])
        self.assertEqual(m["release_actions"], 0)

    def test_takeover_metrics_missing_release(self):
        case = {"allow_helper": True, "expected_release": True}
        m = self.mod.takeover_metrics(case, {"release": 0, "movement": 3})
        self.assertFalse(m["target_executed"])

    def test_takeover_metrics_missing_retreat(self):
        case = {"allow_helper": True, "expected_retreat": True}
        m = self.mod.takeover_metrics(case, {"release": 2, "movement": 0})
        self.assertFalse(m["target_executed"])

    def test_takeover_metrics_both_positive(self):
        case = {"allow_helper": True, "expected_release": True, "expected_retreat": True}
        m = self.mod.takeover_metrics(case, {"release": 1, "movement": 2, "confirm": 1, "home": 1}, 1)
        self.assertTrue(m["target_executed"])
        self.assertEqual(m["confirm_actions"], 1)
        self.assertEqual(m["home_actions"], 1)

    def test_takeover_metrics_noneligible_none(self):
        case = {"allow_helper": False}
        m = self.mod.takeover_metrics(case, {"release": 0})
        self.assertFalse(m["eligible"])
        self.assertIsNone(m["target_executed"])

    def test_should_stop_after_case_unknown(self):
        case = {"allow_helper": False}
        self.assertTrue(self.mod.should_stop_after_case(case, {"status": "unknown"}))

    def test_should_stop_after_case_eligible_untriggered(self):
        case = {"allow_helper": True, "expected_release": True}
        res = {"status": "complete", "phase_counts": {"release": 0}}
        self.assertTrue(self.mod.should_stop_after_case(case, res))

    def test_should_stop_after_case_eligible_satisfied(self):
        case = {"allow_helper": True, "expected_release": True}
        res = {"status": "complete", "phase_counts": {"release": 1}, "release_open_actions": 1}
        self.assertFalse(self.mod.should_stop_after_case(case, res))

    def test_should_stop_after_case_noneligible_zero_not_failure(self):
        case = {"allow_helper": False}
        res = {"status": "complete", "phase_counts": {"release": 0, "movement": 0}}
        self.assertFalse(self.mod.should_stop_after_case(case, res))

    def test_fake_runner_slicing(self):
        import argparse
        cases = []
        for i in range(5):
            c = {"case_id": f"c{i}", "context_origin": "fixed_historical_subtask_fixture"}
            cases.append(c)
        fake = object.__new__(self.mod.PilotRunner)
        fake.limit = 4
        fake.start_index = 1
        fake.max_qwen_calls = 12
        selected = cases[fake.start_index : fake.start_index + fake.limit]
        self.assertEqual([c["case_id"] for c in selected], ["c1", "c2", "c3", "c4"])
        a = self.mod._parse_args(["--proposal", "p", "--inputs", "i", "--output", "o"])
        self.assertEqual(getattr(a, "start_index", 0), 0)
        self.assertEqual(getattr(a, "max_qwen_calls", 18), 18)

    def test_run_case_source_reconciliation(self):
        import inspect
        src = inspect.getsource(self.mod.PilotRunner.run_case)
        self.assertIn("raw_verdict", src)
        self.assertIn(
            "vision.reconcile_verdict(\n                    context_arg, raw_verdict, robot, reference\n                )",
            src,
        )
        self.assertIn("port.read_robot()", src)


    def test_run_slice_executes_only_indices_one_and_two(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self.mod.PilotRunner(SimpleNamespace(
                proposal="x", inputs="y", output=str(Path(tmpdir) / "fresh"),
                limit=2, start_index=1, max_qwen_calls=12,
            ))
            cases = [{"case_id": f"case{i}", "allow_helper": True,
                      "expected_release": True, "expected_retreat": True}
                     for i in range(5)]
            runner.vision = SimpleNamespace(QwenClient=lambda max_calls: object())
            runner.verify_preflight = Mock()
            runner.load_inputs = Mock(return_value=({}, cases))
            runner.run_review = Mock()
            runner.import_runtime = Mock()
            runner.run_case = Mock(side_effect=lambda case, idx, client, budget: {
                "case_id": case["case_id"], "status": "blocked",
                "phase_counts": {"release": 1, "movement": 1, "confirm": 1, "home": 0},
                "release_open_actions": 1,
            })
            summary = runner.run()
            self.assertEqual([call.args[1] for call in runner.run_case.call_args_list], [1, 2])
            self.assertEqual([call.args[0]["case_id"] for call in runner.run_case.call_args_list],
                             ["case1", "case2"])
            self.assertEqual([row["case_id"] for row in summary["cases"]], ["case1", "case2"])
            self.assertTrue(all(row["target_metrics"]["target_executed"]
                                for row in summary["cases"]))
            self.assertTrue(all(row["status"] == "blocked" for row in summary["cases"]))
            self.assertEqual(runner.run_case.call_args_list[0].args[3]["max"], 12)
            runner.verify_preflight.assert_called_once()
            runner.run_review.assert_called_once()

    def test_run_stops_remaining_after_expected_retreat_not_executed(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self.mod.PilotRunner(SimpleNamespace(
                proposal="x", inputs="y", output=str(Path(tmpdir) / "fresh"),
                limit=3, start_index=0, max_qwen_calls=12,
            ))
            cases = [{"case_id": f"case{i}", "allow_helper": True,
                      "expected_release": True, "expected_retreat": True}
                     for i in range(5)]
            runner.vision = SimpleNamespace(QwenClient=lambda max_calls: object())
            runner.verify_preflight = Mock()
            runner.load_inputs = Mock(return_value=({}, cases))
            runner.run_review = Mock()
            runner.import_runtime = Mock()
            runner.run_case = Mock(return_value={
                "status": "blocked",
                "phase_counts": {"release": 1, "movement": 0, "confirm": 0, "home": 0},
            })
            summary = runner.run()
            runner.run_case.assert_called_once()
            self.assertEqual(runner.run_case.call_args.args[1], 0)
            self.assertEqual(summary["statuses"], ["blocked", "not_run", "not_run"])
            self.assertEqual(summary["stop_reason"], "expected takeover actions not executed")
            self.assertFalse(summary["cases"][0]["target_metrics"]["target_executed"])
            saved = json.loads((Path(tmpdir) / "fresh" / "case0" / "result.json").read_text())
            self.assertFalse(saved["target_metrics"]["target_executed"])


if __name__ == "__main__":
    unittest.main()
