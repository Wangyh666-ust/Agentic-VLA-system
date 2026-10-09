"""Offline behavior tests for scripts/review.py.

All registries, evidence and proposals are synthesized in temporary
registries.  No simulator, model, network, subprocess or real project data is
touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SKILL_ROOT = THIS_FILE.parent.parent
REVIEW_PATH = SKILL_ROOT / "scripts" / "review.py"


def _load_review():
    spec = importlib.util.spec_from_file_location("fyp_review", REVIEW_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review = _load_review()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry_path = self.root / "experiment_registry.json"

    def tearDown(self):
        self._tmp.cleanup()

    def write_evidence(self, relpath, content=b"evidence-bytes"):
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        return relpath, digest

    def make_entry(self, entry_id, question_keys, evidence, **extra):
        entry = {
            "id": entry_id,
            "title": f"title {entry_id}",
            "date": "2024-01-01",
            "question_keys": list(question_keys),
            "tags": ["tag-a"],
            "conditions": ["condition one"],
            "findings": ["finding one"],
            "limits": ["limit one"],
            "evidence": [
                {"path": item[0], "sha256": item[1]} for item in evidence
            ],
        }
        entry.update(extra)
        return entry

    def write_registry(self, entries):
        payload = {"schema_version": 1, "entries": entries}
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return self.registry_path

    def base_registry(self, entry_id="e1", question_key="q1", **extra):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry(entry_id, [question_key], [(relpath, digest)], **extra)
        self.write_registry([entry])
        return entry

    def run_cli(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = review.main(argv)
        output = buffer.getvalue()
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            payload = None
        return code, payload, output

    def write_proposal(self, proposal):
        path = self.root / "proposal.json"
        path.write_text(json.dumps(proposal, ensure_ascii=False), encoding="utf-8")
        return path

    def valid_new_proposal(self, question_key="q1", **overrides):
        proposal = {
            "schema_version": 1,
            "id": "p1",
            "intent": "new_experiment",
            "question_key": question_key,
            "purpose": "explore the question",
            "prior_evidence": [],
            "history_review": [],
            "new_physical_actions": 2,
            "hypothesis": "h",
            "information_gain": "ig",
            "success_criteria": "sc",
            "failure_criteria": "fc",
            "unknown_criteria": "uc",
            "changed_factors": [
                {"name": "f", "before": "a", "after": "b", "reason": "why"}
            ],
            "fixed_factors": {"seed": 1},
            "scope": {
                "max_cases": 2,
                "max_vla_actions_per_case": 1,
                "max_helper_actions_per_case": 0,
                "stop_rule": "stop when done",
            },
            "configuration": {"weights": "w"},
            "configuration_complete": True,
            "repeat": {"needed": False},
        }
        proposal.update(overrides)
        return proposal


# --------------------------------------------------------------------------- #
# registry / evidence
# --------------------------------------------------------------------------- #

class RegistryTests(Base):
    def test_valid_evidence_hash(self):
        self.base_registry()
        entries = review.load_registry(self.root, self.registry_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "e1")

    def test_hash_tampering_rejected(self):
        relpath, _ = self.write_evidence("evidence/tamper.txt", b"original")
        entry = self.make_entry("e1", ["q1"], [(relpath, "0" * 64)])
        self.write_registry([entry])
        with self.assertRaises(review.ReviewError):
            review.load_registry(self.root, self.registry_path)

    def test_missing_evidence_rejected(self):
        entry = self.make_entry("e1", ["q1"], [("evidence/nope.txt", "a" * 64)])
        self.write_registry([entry])
        with self.assertRaises(review.ReviewError):
            review.load_registry(self.root, self.registry_path)

    def test_lexical_traversal_rejected(self):
        entry = self.make_entry("e1", ["q1"], [("../escape.txt", "a" * 64)])
        self.write_registry([entry])
        with self.assertRaises(review.ReviewError):
            review.load_registry(self.root, self.registry_path)

    def test_absolute_path_rejected(self):
        entry = self.make_entry("e1", ["q1"], [("/etc/passwd", "a" * 64)])
        self.write_registry([entry])
        with self.assertRaises(review.ReviewError):
            review.load_registry(self.root, self.registry_path)

    def test_symlink_escape_rejected(self):
        outside = Path(self._tmp.name + "_outside")
        try:
            outside.mkdir(parents=True, exist_ok=True)
            outside_file = outside / "secret.txt"
            outside_file.write_bytes(b"secret")
            link = self.root / "link.txt"
            try:
                link.symlink_to(outside_file)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unsupported on this platform")
            digest = hashlib.sha256(b"secret").hexdigest()
            entry = self.make_entry("e1", ["q1"], [("link.txt", digest)])
            self.write_registry([entry])
            with self.assertRaises(review.ReviewError):
                review.load_registry(self.root, self.registry_path)
        finally:
            if outside.exists():
                for child in outside.iterdir():
                    child.unlink()
                outside.rmdir()

    def test_duplicate_registry_ids_rejected(self):
        relpath, digest = self.write_evidence("evidence/dup.txt")
        e1 = self.make_entry("same", ["q1"], [(relpath, digest)])
        e2 = self.make_entry("same", ["q2"], [(relpath, digest)])
        self.write_registry([e1, e2])
        with self.assertRaises(review.ReviewError):
            review.load_registry(self.root, self.registry_path)

    def test_bool_is_not_integer_schema(self):
        payload = {"schema_version": True, "entries": []}
        self.registry_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(review.ReviewError):
            review.load_registry(self.root, self.registry_path)

    def test_unknown_config_does_not_become_complete(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry(
            "e1", ["q1"], [(relpath, digest)], configuration_complete=True
        )
        self.write_registry([entry])
        with self.assertRaises(review.ReviewError):
            review.load_registry(self.root, self.registry_path)


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #

class QueryTests(Base):
    def test_case_insensitive_or_preserves_order_and_excludes_hash(self):
        relpath, digest = self.write_evidence("evidence/a.txt")
        e1 = self.make_entry("e1", ["Alpha"], [(relpath, digest)])
        e1["tags"] = ["Beta"]
        e2 = self.make_entry("e2", ["gamma"], [(relpath, digest)])
        e3 = self.make_entry("e3", ["delta"], [(relpath, digest)])
        e3["findings"] = ["needle-in-findings"]
        result = review.query_entries([e1, e2, e3], ["ALPHA", "NEEDLE"])
        self.assertEqual([item["id"] for item in result], ["e1", "e3"])

        hash_only = review.query_entries([e1], [digest])
        self.assertEqual(hash_only, [])

    def test_query_excludes_id_and_evidence_path(self):
        relpath, digest = self.write_evidence("evidence/unique-token.txt")
        e1 = self.make_entry("idtoken", ["q1"], [(relpath, digest)])
        e1["title"] = "neutral title"
        self.assertEqual(review.query_entries([e1], ["idtoken"]), [])
        self.assertEqual(review.query_entries([e1], ["unique-token"]), [])


# --------------------------------------------------------------------------- #
# review proposal
# --------------------------------------------------------------------------- #

class ProposalTests(Base):
    def test_reuse_and_analyze_zero_actions_pass_with_full_history(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        for intent in ("reuse_evidence", "analyze_existing"):
            proposal = {
                "schema_version": 1,
                "id": f"p-{intent}",
                "intent": intent,
                "question_key": "q1",
                "purpose": "reuse prior work",
                "prior_evidence": ["e1"],
                "history_review": [
                    {
                        "experiment_id": "e1",
                        "disposition": "reuse",
                        "reason": "directly reusable",
                    }
                ],
                "new_physical_actions": 0,
                "analysis_steps": ["step one"],
            }
            verdict = review.review_proposal(proposal, [entry])
            self.assertTrue(verdict["ok"], verdict)
            self.assertEqual(verdict["matched_history_ids"], ["e1"])
            self.assertTrue(verdict["scientific_review_required"])
            self.assertFalse(verdict["launch_authorized"])

    def test_missing_relevant_history_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = {
            "schema_version": 1,
            "id": "p1",
            "intent": "reuse_evidence",
            "question_key": "q1",
            "purpose": "reuse",
            "prior_evidence": [],
            "history_review": [],
            "new_physical_actions": 0,
            "analysis_steps": ["step"],
        }
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_unknown_prior_and_history_ids_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = {
            "schema_version": 1,
            "id": "p1",
            "intent": "reuse_evidence",
            "question_key": "q1",
            "purpose": "reuse",
            "prior_evidence": ["ghost"],
            "history_review": [
                {
                    "experiment_id": "ghost2",
                    "disposition": "reuse",
                    "reason": "reason",
                }
            ],
            "new_physical_actions": 0,
            "analysis_steps": ["step"],
        }
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_duplicate_history_ids_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = {
            "schema_version": 1,
            "id": "p1",
            "intent": "reuse_evidence",
            "question_key": "q1",
            "purpose": "reuse",
            "prior_evidence": ["e1"],
            "history_review": [
                {"experiment_id": "e1", "disposition": "reuse", "reason": "r"},
                {"experiment_id": "e1", "disposition": "reuse", "reason": "r"},
            ],
            "new_physical_actions": 0,
            "analysis_steps": ["step"],
        }
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_no_question_hit_requires_related_search(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["other"], [(relpath, digest)])
        proposal = self.valid_new_proposal(question_key="fresh")
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])
        proposal["related_search"] = "rg fresh over registry found nothing"
        verdict = review.review_proposal(proposal, [entry])
        self.assertTrue(verdict["ok"], verdict)

    def test_new_missing_unknown_criteria_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        del proposal["unknown_criteria"]
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_before_equals_after_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal["changed_factors"] = [
            {"name": "f", "before": "x", "after": "x", "reason": "why"}
        ]
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_complete_valid_new_proposal_passes_without_launch(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        verdict = review.review_proposal(proposal, [entry])
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["intent"], "new_experiment")
        self.assertTrue(verdict["scientific_review_required"])
        self.assertFalse(verdict["launch_authorized"])

    def test_new_physical_actions_zero_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal["new_physical_actions"] = 0
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_over_budget_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal["new_physical_actions"] = 100
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_bool_budget_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal["new_physical_actions"] = True
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_same_config_repeat_false_rejected_repeat_true_passes(self):
        configuration = {"weights": "w", "seed": 7}
        relpath, digest = self.write_evidence("evidence/e1.txt")
        historical = self.make_entry(
            "e1",
            ["q1"],
            [(relpath, digest)],
            configuration=configuration,
            configuration_complete=True,
        )
        entries = [historical]

        proposal = self.valid_new_proposal(configuration=dict(configuration))
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        verdict = review.review_proposal(proposal, entries)
        self.assertFalse(verdict["ok"])
        self.assertIn("e1", verdict["exact_duplicate_ids"])

        proposal2 = self.valid_new_proposal(configuration=dict(configuration))
        proposal2["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal2["repeat"] = {"needed": True, "reason": "replicate for variance"}
        verdict2 = review.review_proposal(proposal2, entries)
        self.assertTrue(verdict2["ok"], verdict2)
        self.assertIn("e1", verdict2["exact_duplicate_ids"])
        self.assertTrue(verdict2["scientific_review_required"])

    def test_same_config_changed_id_repeat_false_rejected(self):
        configuration = {"weights": "w"}
        relpath, digest = self.write_evidence("evidence/e1.txt")
        historical = self.make_entry(
            "e1",
            ["q1"],
            [(relpath, digest)],
            configuration=configuration,
            configuration_complete=True,
        )
        proposal = self.valid_new_proposal(configuration=dict(configuration))
        proposal["id"] = "different-id"
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        verdict = review.review_proposal(proposal, [historical])
        self.assertFalse(verdict["ok"])

    def test_repeat_true_requires_reason(self):
        configuration = {"weights": "w"}
        relpath, digest = self.write_evidence("evidence/e1.txt")
        historical = self.make_entry(
            "e1",
            ["q1"],
            [(relpath, digest)],
            configuration=configuration,
            configuration_complete=True,
        )
        proposal = self.valid_new_proposal(configuration=dict(configuration))
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal["repeat"] = {"needed": True}
        verdict = review.review_proposal(proposal, [historical])
        self.assertFalse(verdict["ok"])
        proposal["repeat"] = {"needed": True, "reason": "explicit purpose"}
        verdict = review.review_proposal(proposal, [historical])
        self.assertTrue(verdict["ok"], verdict)

    def test_incomplete_historical_configuration_not_exact_duplicate(self):
        configuration = {"weights": "w"}
        relpath, digest = self.write_evidence("evidence/e1.txt")
        historical = self.make_entry(
            "e1",
            ["q1"],
            [(relpath, digest)],
            configuration=configuration,
            configuration_complete=False,
        )
        proposal = self.valid_new_proposal(configuration=dict(configuration))
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        verdict = review.review_proposal(proposal, [historical])
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["exact_duplicate_ids"], [])

    def test_multiple_factors_warning_only(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal["changed_factors"] = [
            {"name": "a", "before": "1", "after": "2", "reason": "r"},
            {"name": "b", "before": "x", "after": "y", "reason": "r"},
        ]
        verdict = review.review_proposal(proposal, [entry])
        self.assertTrue(verdict["ok"], verdict)
        self.assertIn("multiple_changed_factors", verdict["warnings"])

    def test_schema_integer_bool_rejected(self):
        proposal = self.valid_new_proposal()
        proposal["schema_version"] = True
        verdict = review.review_proposal(proposal, [])
        self.assertFalse(verdict["ok"])


# --------------------------------------------------------------------------- #
# CLI behavior
# --------------------------------------------------------------------------- #

class WhitespaceTests(Base):
    def test_whitespace_purpose_rejected(self):
        proposal = self.valid_new_proposal()
        proposal["purpose"] = "   "
        verdict = review.review_proposal(proposal, [])
        self.assertFalse(verdict["ok"])

    def test_whitespace_history_reason_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "  "}
        ]
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_whitespace_repeat_reason_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        proposal["repeat"] = {"needed": True, "reason": "   "}
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_whitespace_analysis_steps_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = {
            "schema_version": 1,
            "id": "p1",
            "intent": "analyze_existing",
            "question_key": "q1",
            "purpose": "analyze",
            "prior_evidence": ["e1"],
            "history_review": [
                {"experiment_id": "e1", "disposition": "reuse", "reason": "r"}
            ],
            "new_physical_actions": 0,
            "analysis_steps": ["   "],
        }
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])

    def test_whitespace_query_rejected(self):
        with self.assertRaises(review.ReviewError):
            review.query_entries([], ["   "])


class PathTests(Base):
    def test_registry_outside_root_rejected_before_read(self):
        outside = Path(tempfile.mkdtemp())
        try:
            other = outside / "other_registry.json"
            other.write_text("{not json}", encoding="utf-8")
            with self.assertRaises(review.ReviewError):
                review.load_registry(self.root, other)
        finally:
            for child in outside.iterdir():
                child.unlink()
            outside.rmdir()

    def test_proposal_cli_outside_root_exit2(self):
        self.base_registry()
        outside = Path(tempfile.mkdtemp())
        try:
            other = outside / "proposal.json"
            other.write_text(json.dumps(self.valid_new_proposal()), encoding="utf-8")
            code, payload, raw = self.run_cli(
                [
                    "--root",
                    str(self.root),
                    "--registry",
                    str(self.registry_path),
                    "check",
                    "--proposal",
                    str(other),
                ]
            )
            self.assertEqual(code, 2)
            self.assertIsNotNone(payload)
            self.assertFalse(payload["ok"])
            self.assertNotIn("Traceback", raw)
        finally:
            for child in outside.iterdir():
                child.unlink()
            outside.rmdir()


class Utf8Tests(Base):
    def test_registry_invalid_utf8_cli_json_exit2(self):
        self.registry_path.write_bytes(b"\xff\xfe\x00")
        code, payload, raw = self.run_cli(
            ["--root", str(self.root), "--registry", str(self.registry_path), "validate"]
        )
        self.assertEqual(code, 2)
        self.assertIsNotNone(payload)
        self.assertFalse(payload["ok"])
        self.assertNotIn("Traceback", raw)

    def test_proposal_invalid_utf8_cli_json_exit2(self):
        self.base_registry()
        bad = self.root / "bad_utf8_proposal.json"
        bad.write_bytes(b"\xff\xfe\x00")
        code, payload, raw = self.run_cli(
            [
                "--root",
                str(self.root),
                "--registry",
                str(self.registry_path),
                "check",
                "--proposal",
                str(bad),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIsNotNone(payload)
        self.assertFalse(payload["ok"])
        self.assertNotIn("Traceback", raw)


class CliFlagTests(Base):
    def test_distinct_rejection_branches_retain_flags(self):
        cases = [
            [],
            ["--root", str(self.root), "--registry", str(self.registry_path), "bogus"],
            ["--root", str(self.root), "--registry", str(self.registry_path), "validate"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                code, payload, raw = self.run_cli(argv)
                self.assertEqual(code, 2)
                self.assertIsNotNone(payload)
                self.assertFalse(payload["ok"])
                self.assertTrue(payload.get("scientific_review_required") is True)
                self.assertTrue(payload.get("launch_authorized") is False)
                self.assertNotIn("Traceback", raw)


class CliTests(Base):
    def test_cli_validate_and_query(self):
        self.base_registry(question_key="UniqueTerm")
        code, payload, raw = self.run_cli(
            ["--root", str(self.root), "--registry", str(self.registry_path), "validate"]
        )
        self.assertEqual(code, 0)
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["entry_count"], 1)
        self.assertEqual(payload["evidence_count"], 1)

        code, payload, raw = self.run_cli(
            [
                "--root",
                str(self.root),
                "--registry",
                str(self.registry_path),
                "query",
                "--text",
                "uniqueterm",
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(payload["matches"]), 1)

    def test_cli_malformed_registry_json_exit2(self):
        self.registry_path.write_text("{not json", encoding="utf-8")
        code, payload, raw = self.run_cli(
            ["--root", str(self.root), "--registry", str(self.registry_path), "validate"]
        )
        self.assertEqual(code, 2)
        self.assertIsNotNone(payload)
        self.assertFalse(payload["ok"])
        self.assertNotIn("Traceback", raw)

    def test_cli_malformed_proposal_json_exit2(self):
        self.base_registry()
        bad = self.root / "bad_proposal.json"
        bad.write_text("{oops", encoding="utf-8")
        code, payload, raw = self.run_cli(
            [
                "--root",
                str(self.root),
                "--registry",
                str(self.registry_path),
                "check",
                "--proposal",
                str(bad),
            ]
        )
        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertNotIn("Traceback", raw)

    def test_cli_stale_evidence_exit2(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        self.write_registry([entry])
        (self.root / relpath).write_bytes(b"changed")
        code, payload, raw = self.run_cli(
            ["--root", str(self.root), "--registry", str(self.registry_path), "validate"]
        )
        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertNotIn("Traceback", raw)

    def test_cli_check_ok_without_launch(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        self.write_registry([entry])
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        path = self.write_proposal(proposal)
        code, payload, raw = self.run_cli(
            [
                "--root",
                str(self.root),
                "--registry",
                str(self.registry_path),
                "check",
                "--proposal",
                str(path),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["scientific_review_required"])
        self.assertFalse(payload["launch_authorized"])

    def test_cli_rejection_exit2(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        self.write_registry([entry])
        proposal = self.valid_new_proposal()
        path = self.write_proposal(proposal)
        code, payload, raw = self.run_cli(
            [
                "--root",
                str(self.root),
                "--registry",
                str(self.registry_path),
                "check",
                "--proposal",
                str(path),
            ]
        )
        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["launch_authorized"] is False)

    def test_cli_does_not_mutate_evidence_and_does_not_run_model(self):
        relpath, digest = self.write_evidence("evidence/e1.txt", b"immutable")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        self.write_registry([entry])
        before = (self.root / relpath).read_bytes()
        proposal = self.valid_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        path = self.write_proposal(proposal)
        code, payload, raw = self.run_cli(
            [
                "--root",
                str(self.root),
                "--registry",
                str(self.registry_path),
                "check",
                "--proposal",
                str(path),
            ]
        )
        after = (self.root / relpath).read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(code, 0)


def _fingerprint():
    return review.configuration_fingerprint({"b": 2, "a": 1})


class HelperOnlyProposalTests(Base):
    def _make_new_proposal(self, question_key="q1", **overrides):
        proposal = {
            "schema_version": 1,
            "id": "p-helper",
            "intent": "new_experiment",
            "question_key": question_key,
            "purpose": "helper-only safety case",
            "prior_evidence": [],
            "history_review": [],
            "new_physical_actions": 1654,
            "hypothesis": "h",
            "information_gain": "ig",
            "success_criteria": "sc",
            "failure_criteria": "fc",
            "unknown_criteria": "uc",
            "changed_factors": [
                {"name": "f", "before": "a", "after": "b", "reason": "why"}
            ],
            "fixed_factors": {"seed": 1},
            "scope": {
                "max_cases": 3,
                "max_vla_actions_per_case": 0,
                "max_helper_actions_per_case": 620,
                "stop_rule": "stop when done",
            },
            "configuration": {"weights": "w"},
            "configuration_complete": True,
            "repeat": {"needed": False},
        }
        proposal.update(overrides)
        return proposal

    def test_valid_new_proposal(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self._make_new_proposal()
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        verdict = review.review_proposal(proposal, [entry])
        self.assertTrue(verdict["ok"], verdict)
        self.assertFalse(verdict["launch_authorized"])

    def test_helper_zero_with_physical_rejected(self):
        relpath, digest = self.write_evidence("evidence/e1.txt")
        entry = self.make_entry("e1", ["q1"], [(relpath, digest)])
        proposal = self._make_new_proposal()
        proposal["scope"]["max_helper_actions_per_case"] = 0
        proposal["new_physical_actions"] = 1
        proposal["history_review"] = [
            {"experiment_id": "e1", "disposition": "not_applicable", "reason": "r"}
        ]
        verdict = review.review_proposal(proposal, [entry])
        self.assertFalse(verdict["ok"])


class FingerprintTests(Base):
    def test_fingerprint_is_canonical_and_stable(self):
        first = review.configuration_fingerprint({"b": 2, "a": 1})
        second = review.configuration_fingerprint({"a": 1, "b": 2})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_fingerprint_rejects_non_serializable(self):
        with self.assertRaises(review.ReviewError):
            review.configuration_fingerprint({"a": object()})


if __name__ == "__main__":
    unittest.main()
