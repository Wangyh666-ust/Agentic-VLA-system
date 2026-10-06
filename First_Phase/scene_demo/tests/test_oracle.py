"""Standard-library unittest suite for the fixture oracle and the catalog.

The tests add ``scene_demo`` to ``sys.path`` themselves so no package file is
needed. Nothing here is a plan: expectations are fixed by the preauthored
fixtures.
"""

import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCENE_DEMO = os.path.dirname(_HERE)
if _SCENE_DEMO not in sys.path:
    sys.path.insert(0, _SCENE_DEMO)

import catalog
import oracle

_BOWL = "on|akita_black_bowl_1|plate_1"
_WINE = "on|wine_bottle_1|wine_rack_1_top_region"
_STOVE = "turnon|flat_stove_1"


def _load():
    return oracle.load_cases()


class CatalogTests(unittest.TestCase):
    def test_scene_and_capability_counts(self):
        self.assertEqual(len(catalog.SCENES), 7)
        self.assertEqual(len(catalog.CAPABILITIES), 11)
        atomic = [c for c in catalog.CAPABILITIES.values() if not c["audit_only"]]
        composite = [c for c in catalog.CAPABILITIES.values() if c["audit_only"]]
        self.assertEqual(len(atomic), 8)
        self.assertEqual(len(composite), 3)

    def test_scene_fields_and_patches(self):
        shifted = catalog.SCENES["goal_table_shifted"]
        self.assertEqual(shifted["suite"], "libero_goal")
        self.assertEqual(shifted["task_id"], 8)
        self.assertEqual(shifted["variant"], "layout_shift")
        self.assertEqual(
            shifted["patches"],
            [
                {"kind": "offset", "object_id": "akita_black_bowl_1", "xyz": [0.06, 0, 0]},
                {"kind": "offset", "object_id": "wine_bottle_1", "xyz": [-0.06, 0, 0]},
            ],
        )
        self.assertEqual(
            catalog.SCENES["mugs_both_occupied"]["patches"],
            [
                {"kind": "place_on", "object_id": "red_coffee_mug_1", "target_id": "plate_1"},
                {"kind": "place_on", "object_id": "white_yellow_mug_1", "target_id": "plate_2"},
            ],
        )

    def test_scene_capabilities_filters_scene_and_audit(self):
        base = catalog.scene_capabilities("goal_table")
        self.assertEqual(
            [c["capability_id"] for c in base],
            ["bowl_to_plate", "wine_to_rack", "stove_on"],
        )
        with_audit = catalog.scene_capabilities("goal_table", include_audit=True)
        self.assertEqual(len(with_audit), 4)
        self.assertIn("table_both", [c["capability_id"] for c in with_audit])

    def test_scene_capabilities_returns_deep_copies(self):
        first = catalog.scene_capabilities("goal_table")
        first[0]["goals"][0][0] = "MUTATED"
        first[0]["scene_ids"].append("bogus")
        second = catalog.scene_capabilities("goal_table")
        self.assertEqual(second[0]["goals"][0][0], "on")
        self.assertNotIn("bogus", second[0]["scene_ids"])

    def test_scene_capabilities_unknown_scene_raises(self):
        with self.assertRaises(ValueError):
            catalog.scene_capabilities("nope")

    def test_deduplicate_goals_preserves_first_occurrence(self):
        result = catalog.deduplicate_goals(
            ["wine_to_rack", "bowl_to_plate", "wine_to_rack"]
        )
        self.assertEqual(
            result,
            [
                ["on", "wine_bottle_1", "wine_rack_1_top_region"],
                ["on", "akita_black_bowl_1", "plate_1"],
            ],
        )
        self.assertEqual(len(catalog.deduplicate_goals(["mugs_both"])), 2)

    def test_deduplicate_goals_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            catalog.deduplicate_goals(["nope"])

    def test_goal_key(self):
        self.assertEqual(catalog.goal_key(["on", "a", "b"]), "on|a|b")
        self.assertEqual(catalog.goal_key(["turnon", "flat_stove_1"]), _STOVE)


class FixtureTests(unittest.TestCase):
    def test_eleven_cases_and_version(self):
        cases = _load()
        self.assertEqual(len(cases), 11)
        fixtures_path = os.path.join(_SCENE_DEMO, "fixtures.json")
        with open(fixtures_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(raw["version"], 1)
        self.assertEqual(len(raw["cases"]), 11)
        self.assertEqual(
            set(cases),
            {
                "table_tidy",
                "table_wine_only",
                "table_bowl_only",
                "table_shifted_tidy",
                "basket_two_cans",
                "mugs_standard",
                "mugs_free_right",
                "mugs_free_left",
                "mugs_full",
                "missing_bin",
                "ambiguous_cleanup",
            },
        )

    def test_every_case_has_explicit_protection_fields(self):
        for case in _load().values():
            self.assertIn("protected_goals", case)
            self.assertIn("protected_positions", case)

    def test_refusal_cases_have_specified_decisions(self):
        cases = _load()
        self.assertEqual(cases["mugs_full"]["allowed_decisions"], ["unsupported", "clarify"])
        self.assertEqual(cases["missing_bin"]["allowed_decisions"], ["unsupported", "clarify"])
        self.assertEqual(cases["ambiguous_cleanup"]["allowed_decisions"], ["clarify"])
        self.assertEqual(cases["mugs_full"]["goal_options"], [])
        self.assertEqual(cases["missing_bin"]["allowed_objects"], [])

    def test_load_cases_explicit_path(self):
        fixtures_path = os.path.join(_SCENE_DEMO, "fixtures.json")
        cases = oracle.load_cases(fixtures_path)
        self.assertEqual(len(cases), 11)
        self.assertIn("table_tidy", cases)


class OracleGoalTests(unittest.TestCase):
    def setUp(self):
        self.cases = _load()
        self.tidy = self.cases["table_tidy"]

    def _tidy_truth(self):
        return {_BOWL: True, _WINE: True, _STOVE: False}

    def test_correct_combined_goal_succeeds(self):
        result = oracle.evaluate_case(
            self.tidy,
            self._tidy_truth(),
            ["akita_black_bowl_1", "wine_bottle_1"],
            "execute",
        )
        self.assertTrue(result["task_success"])
        self.assertEqual(result["goal_option_satisfied"], [True])
        self.assertTrue(result["protected_satisfied"])
        self.assertTrue(result["decision_ok"])
        self.assertTrue(result["objects_ok"])
        self.assertEqual(result["missing_truth"], [])
        self.assertEqual(result["oracle_source"], "preauthored_fixture")
        self.assertEqual(
            set(result),
            {
                "task_success",
                "goal_option_satisfied",
                "protected_satisfied",
                "decision_ok",
                "objects_ok",
                "position_checks",
                "missing_truth",
                "oracle_source",
            },
        )

    def test_half_done_goal_fails(self):
        truth = self._tidy_truth()
        truth[_WINE] = False
        result = oracle.evaluate_case(self.tidy, truth, ["wine_bottle_1"], "execute")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["goal_option_satisfied"], [False])
        self.assertEqual(result["missing_truth"], [])

    def test_wrong_target_predicates_true_still_fail(self):
        truth = {
            "on|akita_black_bowl_1|plate_2": True,
            _WINE: True,
            _STOVE: False,
        }
        result = oracle.evaluate_case(
            self.tidy,
            truth,
            ["akita_black_bowl_1", "wine_bottle_1"],
            "execute",
        )
        self.assertFalse(result["task_success"])
        self.assertEqual(result["goal_option_satisfied"], [False])
        self.assertIn(_BOWL, result["missing_truth"])

    def test_final_regression_fails(self):
        # bowl reached the plate earlier, but the final truth shows it regressed
        truth = self._tidy_truth()
        truth[_BOWL] = False
        result = oracle.evaluate_case(self.tidy, truth, ["wine_bottle_1"], "execute")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["goal_option_satisfied"], [False])

    def test_protected_condition_violated_fails(self):
        truth = self._tidy_truth()
        truth[_STOVE] = True
        result = oracle.evaluate_case(
            self.tidy,
            truth,
            ["akita_black_bowl_1", "wine_bottle_1"],
            "execute",
        )
        self.assertTrue(result["goal_option_satisfied"][0])
        self.assertFalse(result["protected_satisfied"])
        self.assertFalse(result["task_success"])

    def test_missing_truth_fails(self):
        truth = {_BOWL: True, _STOVE: False}
        result = oracle.evaluate_case(self.tidy, truth, ["akita_black_bowl_1"], "execute")
        self.assertFalse(result["task_success"])
        self.assertIn(_WINE, result["missing_truth"])

    def test_missing_negation_base_fails(self):
        truth = {_BOWL: True, _WINE: True}
        result = oracle.evaluate_case(
            self.tidy,
            truth,
            ["akita_black_bowl_1", "wine_bottle_1"],
            "execute",
        )
        self.assertFalse(result["task_success"])
        self.assertFalse(result["protected_satisfied"])
        self.assertIn(_STOVE, result["missing_truth"])

    def test_allowed_different_action_order_succeeds(self):
        result = oracle.evaluate_case(
            self.tidy,
            self._tidy_truth(),
            ["wine_bottle_1", "akita_black_bowl_1"],
            "execute",
        )
        self.assertTrue(result["objects_ok"])
        self.assertTrue(result["task_success"])

    def test_decision_not_allowed_fails(self):
        result = oracle.evaluate_case(self.tidy, self._tidy_truth(), [], "clarify")
        self.assertFalse(result["decision_ok"])
        self.assertFalse(result["task_success"])

    def test_object_outside_allowed_fails(self):
        result = oracle.evaluate_case(
            self.tidy,
            self._tidy_truth(),
            ["akita_black_bowl_1", "porcelain_mug_1"],
            "execute",
        )
        self.assertFalse(result["objects_ok"])
        self.assertFalse(result["task_success"])

    def test_or_of_and_options(self):
        case = dict(self.tidy)
        case["goal_options"] = [
            [["on", "akita_black_bowl_1", "plate_1"], ["on", "wine_bottle_1", "wine_rack_1_top_region"]],
            [["on", "akita_black_bowl_1", "plate_1"]],
        ]
        truth = {_BOWL: True, _STOVE: False}
        result = oracle.evaluate_case(case, truth, ["akita_black_bowl_1"], "execute")
        self.assertEqual(result["goal_option_satisfied"], [False, True])
        self.assertTrue(result["task_success"])

    def test_missing_truth_dedup_preserves_first_occurrence(self):
        case = dict(self.tidy)
        case["goal_options"] = [
            [["on", "akita_black_bowl_1", "plate_1"]],
            [["on", "akita_black_bowl_1", "plate_1"]],
        ]
        case["protected_goals"] = []
        result = oracle.evaluate_case(case, {}, [], "execute")
        self.assertEqual(result["missing_truth"], [_BOWL])

    def test_mugs_free_right_succeeds(self):
        case = self.cases["mugs_free_right"]
        truth = {"on|porcelain_mug_1|plate_2": True, "on|red_coffee_mug_1|plate_1": True}
        result = oracle.evaluate_case(case, truth, ["porcelain_mug_1"], "execute")
        self.assertTrue(result["task_success"])

    def test_mugs_free_right_protection_violation_fails(self):
        case = self.cases["mugs_free_right"]
        truth = {"on|porcelain_mug_1|plate_2": True, "on|red_coffee_mug_1|plate_1": False}
        result = oracle.evaluate_case(case, truth, ["porcelain_mug_1"], "execute")
        self.assertTrue(result["goal_option_satisfied"][0])
        self.assertFalse(result["protected_satisfied"])
        self.assertFalse(result["task_success"])


class OracleRefusalTests(unittest.TestCase):
    def setUp(self):
        self.cases = _load()
        self.full = self.cases["mugs_full"]
        self.protected_truth = {
            "on|red_coffee_mug_1|plate_1": True,
            "on|white_yellow_mug_1|plate_2": True,
        }

    def test_allowed_refusal_with_zero_actions_succeeds(self):
        result = oracle.evaluate_case(self.full, self.protected_truth, [], "clarify")
        self.assertTrue(result["task_success"])
        self.assertEqual(result["goal_option_satisfied"], [])
        self.assertTrue(result["decision_ok"])
        self.assertTrue(result["objects_ok"])

    def test_unsupported_decision_also_succeeds(self):
        result = oracle.evaluate_case(self.full, self.protected_truth, [], "unsupported")
        self.assertTrue(result["task_success"])

    def test_refusal_with_execution_fails(self):
        result = oracle.evaluate_case(
            self.full, self.protected_truth, ["porcelain_mug_1"], "clarify"
        )
        self.assertFalse(result["objects_ok"])
        self.assertFalse(result["task_success"])

    def test_refusal_with_protection_violation_fails(self):
        truth = dict(self.protected_truth)
        truth["on|red_coffee_mug_1|plate_1"] = False
        result = oracle.evaluate_case(self.full, truth, [], "clarify")
        self.assertFalse(result["protected_satisfied"])
        self.assertFalse(result["task_success"])

    def test_missing_bin_unsupported_succeeds(self):
        case = self.cases["missing_bin"]
        result = oracle.evaluate_case(case, {}, [], "unsupported")
        self.assertTrue(result["task_success"])

    def test_ambiguous_cleanup_only_clarify(self):
        case = self.cases["ambiguous_cleanup"]
        self.assertTrue(oracle.evaluate_case(case, {}, [], "clarify")["task_success"])
        result = oracle.evaluate_case(case, {}, [], "unsupported")
        self.assertFalse(result["decision_ok"])
        self.assertFalse(result["task_success"])


class OraclePositionTests(unittest.TestCase):
    def setUp(self):
        self.cases = _load()
        self.case = self.cases["table_wine_only"]
        self.truth = {_WINE: True, _STOVE: False}

    def test_missing_position_maps_fail(self):
        result = oracle.evaluate_case(self.case, self.truth, ["wine_bottle_1"], "execute")
        self.assertFalse(result["task_success"])
        self.assertFalse(result["position_checks"]["akita_black_bowl_1"]["ok"])

    def test_within_tolerance_succeeds(self):
        initial = {"akita_black_bowl_1": [0.0, 0.0, 0.0], "cream_cheese_1": [0.0, 0.0, 0.0]}
        final = {"akita_black_bowl_1": [0.03, 0.0, 0.0], "cream_cheese_1": [0.0, 0.0, 0.0]}
        result = oracle.evaluate_case(
            self.case, self.truth, ["wine_bottle_1"], "execute", initial, final
        )
        self.assertTrue(result["task_success"])
        check = result["position_checks"]["akita_black_bowl_1"]
        self.assertTrue(check["ok"])
        self.assertAlmostEqual(check["displacement"], 0.03, places=6)

    def test_displacement_exceeds_tolerance_fails(self):
        initial = {"akita_black_bowl_1": [0.0, 0.0, 0.0], "cream_cheese_1": [0.0, 0.0, 0.0]}
        final = {"akita_black_bowl_1": [0.5, 0.0, 0.0], "cream_cheese_1": [0.0, 0.0, 0.0]}
        result = oracle.evaluate_case(
            self.case, self.truth, ["wine_bottle_1"], "execute", initial, final
        )
        self.assertFalse(result["task_success"])
        check = result["position_checks"]["akita_black_bowl_1"]
        self.assertFalse(check["ok"])
        self.assertAlmostEqual(check["displacement"], 0.5, places=6)

    def test_absent_protected_position_fails(self):
        initial = {"akita_black_bowl_1": [0.0, 0.0, 0.0], "cream_cheese_1": [0.0, 0.0, 0.0]}
        final = {"akita_black_bowl_1": [0.0, 0.0, 0.0]}
        result = oracle.evaluate_case(
            self.case, self.truth, ["wine_bottle_1"], "execute", initial, final
        )
        self.assertFalse(result["task_success"])
        check = result["position_checks"]["cream_cheese_1"]
        self.assertFalse(check["ok"])
        self.assertIsNone(check["displacement"])

    def test_map_style_coordinates_succeed(self):
        initial = {
            "akita_black_bowl_1": {"x": 0.0, "y": 0.0, "z": 0.0},
            "cream_cheese_1": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        final = {
            "akita_black_bowl_1": {"x": 0.0, "y": 0.0, "z": 0.0},
            "cream_cheese_1": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        result = oracle.evaluate_case(
            self.case, self.truth, ["wine_bottle_1"], "execute", initial, final
        )
        self.assertTrue(result["task_success"])

    def test_empty_coordinates_fail(self):
        # A protected object can never report success for an empty position,
        # even when the initial and final inputs are identical.
        identical = {
            "akita_black_bowl_1": [],
            "cream_cheese_1": [0.0, 0.0, 0.0],
        }
        result = oracle.evaluate_case(
            self.case, self.truth, ["wine_bottle_1"], "execute", identical, identical
        )
        self.assertFalse(result["task_success"])
        self.assertFalse(result["position_checks"]["akita_black_bowl_1"]["ok"])

    def test_partial_axis_mapping_fails(self):
        # A mapping that carries only x is not a legal three-dimensional point.
        identical = {
            "akita_black_bowl_1": {"x": 0.0},
            "cream_cheese_1": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        result = oracle.evaluate_case(
            self.case, self.truth, ["wine_bottle_1"], "execute", identical, identical
        )
        self.assertFalse(result["task_success"])
        self.assertFalse(result["position_checks"]["akita_black_bowl_1"]["ok"])

    def test_two_dimensional_coordinates_fail(self):
        # Nested list/tuple coordinates are not a flat three-vector.
        bad_inputs = (
            [[0.0, 0.0], [0.0, 0.0]],
            ((0.0, 0.0), (0.0, 0.0)),
        )
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                identical = {
                    "akita_black_bowl_1": bad,
                    "cream_cheese_1": [0.0, 0.0, 0.0],
                }
                result = oracle.evaluate_case(
                    self.case,
                    self.truth,
                    ["wine_bottle_1"],
                    "execute",
                    identical,
                    identical,
                )
                self.assertFalse(result["task_success"])
                self.assertFalse(
                    result["position_checks"]["akita_black_bowl_1"]["ok"]
                )

    def test_non_finite_coordinates_fail(self):
        # NaN and both infinities are rejected however they are presented.
        bad_inputs = (
            [float("nan"), 0.0, 0.0],
            [0.0, float("inf"), 0.0],
            [0.0, 0.0, float("-inf")],
            {"x": float("nan"), "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": float("inf")},
        )
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                identical = {
                    "akita_black_bowl_1": bad,
                    "cream_cheese_1": [0.0, 0.0, 0.0],
                }
                result = oracle.evaluate_case(
                    self.case,
                    self.truth,
                    ["wine_bottle_1"],
                    "execute",
                    identical,
                    identical,
                )
                self.assertFalse(result["task_success"])
                self.assertFalse(
                    result["position_checks"]["akita_black_bowl_1"]["ok"]
                )


if __name__ == "__main__":
    unittest.main()
