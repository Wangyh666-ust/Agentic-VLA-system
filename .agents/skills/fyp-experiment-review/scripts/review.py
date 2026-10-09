"""Offline experiment registry validation and proposal review.

Standard library only.  Reads registry/proposal JSON, validates structure and
evidence hashes, performs term search, and emits JSON review verdicts.  No
subprocess, network, simulator or experiment launch is ever performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
INTENTS = ("reuse_evidence", "analyze_existing", "new_experiment")
DISPOSITIONS = ("reuse", "insufficient", "not_applicable")


class ReviewError(ValueError):
    """Raised for any structural/data error.  Never carries a traceback."""


# --------------------------------------------------------------------------- #
# primitive validators
# --------------------------------------------------------------------------- #

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_blank(value):
    return isinstance(value, str) and value.strip() == ""


def _req_str(obj, key, where):
    value = obj.get(key)
    if not isinstance(value, str) or _is_blank(value):
        raise ReviewError(f"{where}.{key} must be a nonempty string")
    return value


def _opt_str(obj, key, where):
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or _is_blank(value):
        raise ReviewError(f"{where}.{key} must be a nonempty string")
    return value


def _req_str_array(obj, key, where):
    value = obj.get(key)
    if not isinstance(value, list):
        raise ReviewError(f"{where}.{key} must be a string array")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or _is_blank(item):
            raise ReviewError(f"{where}.{key}[{i}] must be a nonempty string")
        out.append(item)
    if not out:
        raise ReviewError(f"{where}.{key} must be nonempty")
    return out


def _opt_str_array(obj, key, where):
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ReviewError(f"{where}.{key} must be a string array")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or _is_blank(item):
            raise ReviewError(f"{where}.{key}[{i}] must be a nonempty string")
        out.append(item)
    if not out:
        raise ReviewError(f"{where}.{key} must be nonempty")
    return out


def _req_nonneg_int(obj, key, where):
    value = obj.get(key)
    if not _is_int(value) or value < 0:
        raise ReviewError(f"{where}.{key} must be a nonnegative integer")
    return value


def _req_pos_int(obj, key, where):
    value = obj.get(key)
    if not _is_int(value) or value < 1:
        raise ReviewError(f"{where}.{key} must be a positive integer")
    return value


def _req_bool(obj, key, where):
    value = obj.get(key)
    if not isinstance(value, bool):
        raise ReviewError(f"{where}.{key} must be a boolean")
    return value


def _req_dict(obj, key, where):
    value = obj.get(key)
    if not isinstance(value, dict) or not value:
        raise ReviewError(f"{where}.{key} must be a nonempty object")
    return value


def _resolve_inside_root(root: Path, input_path, where: str) -> Path:
    root = Path(root)
    try:
        root_resolved = root.resolve()
    except OSError as exc:
        raise ReviewError(f"cannot resolve root: {exc}")
    if isinstance(input_path, str):
        original = input_path
        raw_path = Path(input_path)
    else:
        raw_path = Path(input_path)
        original = str(raw_path)
    if original == "":
        raise ReviewError(f"{where} must be a nonempty string")
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ReviewError(f"cannot resolve {where} {original!r}: {exc}")
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ReviewError(f"{where} escapes root: {original!r}")
    return resolved


def _path_is_inside(root: Path, candidate: Path, original: str):
    if not isinstance(original, str) or original == "":
        raise ReviewError("evidence path must be a nonempty string")
    if "\\" in original:
        raise ReviewError(f"evidence path must be POSIX: {original!r}")
    if original.startswith("/"):
        raise ReviewError(f"evidence path must be relative: {original!r}")
    for part in original.split("/"):
        if part == "..":
            raise ReviewError(f"lexical traversal in evidence path: {original!r}")
    try:
        resolved = (root / original).resolve()
        root_resolved = root.resolve()
    except OSError as exc:  # pragma: no cover - rare FS errors
        raise ReviewError(f"cannot resolve evidence path {original!r}: {exc}")
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ReviewError(f"evidence path escapes root: {original!r}")
    return resolved


def _file_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# registry loading
# --------------------------------------------------------------------------- #

def _validate_evidence(entry, root: Path, where: str):
    evidence = entry.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ReviewError(f"{where}.evidence must be a nonempty array")
    for i, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ReviewError(f"{where}.evidence[{i}] must be an object")
        path_str = item.get("path")
        if not isinstance(path_str, str) or path_str == "":
            raise ReviewError(f"{where}.evidence[{i}].path must be a nonempty string")
        sha = item.get("sha256")
        if not isinstance(sha, str) or not SHA256_RE.match(sha):
            raise ReviewError(f"{where}.evidence[{i}].sha256 must be 64 hex chars")
        resolved = _path_is_inside(root, root / path_str, path_str)
        if not resolved.is_file():
            raise ReviewError(f"{where}.evidence[{i}] file missing: {path_str}")
        actual = _file_sha256(resolved)
        if actual != sha.lower():
            raise ReviewError(
                f"{where}.evidence[{i}] sha256 mismatch for {path_str}: "
                f"expected {sha.lower()} got {actual}"
            )


def load_registry(root: Path, registry_path: Path) -> list:
    root = Path(root)
    registry_path = Path(registry_path)
    registry_path = _resolve_inside_root(root, registry_path, "registry path")
    if not registry_path.is_file():
        raise ReviewError(f"registry file missing: {registry_path}")
    try:
        raw = registry_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewError(f"cannot decode registry as utf-8: {exc}")
    except OSError as exc:
        raise ReviewError(f"cannot read registry: {exc}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewError(f"registry is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise ReviewError("registry must be a JSON object")
    schema_version = data.get("schema_version")
    if not _is_int(schema_version) or schema_version != SCHEMA_VERSION:
        raise ReviewError(f"registry schema_version must be {SCHEMA_VERSION}")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ReviewError("registry.entries must be a nonempty array")

    seen_ids = set()
    validated = []
    for index, entry in enumerate(entries):
        where = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise ReviewError(f"{where} must be an object")
        entry_id = _req_str(entry, "id", where)
        if entry_id in seen_ids:
            raise ReviewError(f"duplicate entry id: {entry_id!r}")
        seen_ids.add(entry_id)
        _req_str(entry, "title", where)
        _req_str(entry, "date", where)
        _req_str_array(entry, "question_keys", where)
        _req_str_array(entry, "tags", where)
        _req_str_array(entry, "conditions", where)
        _req_str_array(entry, "findings", where)
        _req_str_array(entry, "limits", where)
        _validate_evidence(entry, root, where)

        if "configuration" in entry:
            configuration = entry["configuration"]
            if not isinstance(configuration, dict):
                raise ReviewError(f"{where}.configuration must be an object")
        if "configuration_complete" in entry:
            complete = entry["configuration_complete"]
            if not isinstance(complete, bool):
                raise ReviewError(f"{where}.configuration_complete must be a boolean")
            if complete and not entry.get("configuration"):
                raise ReviewError(
                    f"{where}.configuration_complete=true requires nonempty configuration"
                )
        validated.append(entry)
    return validated


# --------------------------------------------------------------------------- #
# canonical configuration fingerprint
# --------------------------------------------------------------------------- #

def configuration_fingerprint(configuration: dict) -> str:
    if not isinstance(configuration, dict):
        raise ReviewError("configuration must be an object")
    try:
        canonical = json.dumps(
            configuration,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ReviewError(f"configuration is not JSON-serializable: {exc}")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #

def query_entries(entries: list, terms: list) -> list:
    if not isinstance(terms, list) or not terms:
        raise ReviewError("query requires at least one term")
    lowered = []
    for term in terms:
        if not isinstance(term, str) or _is_blank(term):
            raise ReviewError("query terms must be nonempty strings")
        lowered.append(term.lower())
    matches = []
    for entry in entries:
        haystacks = []
        for key in ("title", "tags", "conditions", "findings", "question_keys"):
            value = entry.get(key)
            if isinstance(value, str):
                haystacks.append(value.lower())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        haystacks.append(item.lower())
        if any(term in hay for term in lowered for hay in haystacks):
            matches.append(entry)
    return matches


# --------------------------------------------------------------------------- #
# proposal review
# --------------------------------------------------------------------------- #

def _collect_history(entries, question_key):
    return [
        entry
        for entry in entries
        if any(qk == question_key for qk in entry.get("question_keys", []))
    ]


def _exact_complete_matches(entries, fingerprint):
    matches = []
    for entry in entries:
        if entry.get("configuration_complete") is not True:
            continue
        configuration = entry.get("configuration")
        if not isinstance(configuration, dict) or not configuration:
            continue
        try:
            fp = configuration_fingerprint(configuration)
        except ReviewError:
            continue
        if fp == fingerprint:
            matches.append(entry)
    return matches


def _review_common(proposal, where="proposal"):
    schema_version = proposal.get("schema_version")
    if not _is_int(schema_version) or schema_version != SCHEMA_VERSION:
        raise ReviewError(f"{where}.schema_version must be {SCHEMA_VERSION}")
    _req_str(proposal, "id", where)
    intent = _req_str(proposal, "intent", where)
    if intent not in INTENTS:
        raise ReviewError(f"{where}.intent must be one of {INTENTS}")
    _req_str(proposal, "question_key", where)
    _req_str(proposal, "purpose", where)
    return intent


def _check_prior_ids(value, where):
    if not isinstance(value, list):
        raise ReviewError(f"{where} must be a string array")
    seen = set()
    for i, item in enumerate(value):
        if not isinstance(item, str) or _is_blank(item):
            raise ReviewError(f"{where}[{i}] must be a nonempty string")
        if item in seen:
            raise ReviewError(f"{where} has duplicate id: {item!r}")
        seen.add(item)
    return seen


def review_proposal(proposal: dict, entries: list) -> dict:
    errors = []
    warnings = []
    matched_history_ids = []
    exact_duplicate_ids = []

    def fail(message):
        errors.append(message)

    try:
        intent = _review_common(proposal)
    except ReviewError as exc:
        return {
            "ok": False,
            "errors": [str(exc)],
            "scientific_review_required": True,
            "launch_authorized": False,
        }

    question_key = proposal["question_key"]
    prior_ids = set()
    try:
        prior_ids = _check_prior_ids(proposal.get("prior_evidence"), "prior_evidence")
    except ReviewError as exc:
        fail(str(exc))

    history_review = proposal.get("history_review")
    history_by_id = {}
    if not isinstance(history_review, list):
        fail("history_review must be an array")
        history_review = []
    else:
        for i, item in enumerate(history_review):
            where = f"history_review[{i}]"
            if not isinstance(item, dict):
                fail(f"{where} must be an object")
                continue
            experiment_id = item.get("experiment_id")
            if not isinstance(experiment_id, str) or _is_blank(experiment_id):
                fail(f"{where}.experiment_id must be a nonempty string")
                continue
            if experiment_id in history_by_id:
                fail(f"duplicate history_review id: {experiment_id!r}")
                continue
            disposition = item.get("disposition")
            if disposition not in DISPOSITIONS:
                fail(f"{where}.disposition must be one of {DISPOSITIONS}")
            reason = item.get("reason")
            if not isinstance(reason, str) or _is_blank(reason):
                fail(f"{where}.reason must be a nonempty string")
            history_by_id[experiment_id] = item

    known_ids = {entry["id"] for entry in entries}
    for prior_id in prior_ids:
        if prior_id not in known_ids:
            fail(f"prior_evidence references unknown id: {prior_id!r}")
    for history_id in history_by_id:
        if history_id not in known_ids:
            fail(f"history_review references unknown id: {history_id!r}")

    history_entries = _collect_history(entries, question_key)
    matched_history_ids = [entry["id"] for entry in history_entries]
    for entry in history_entries:
        record = history_by_id.get(entry["id"])
        if record is None:
            fail(f"missing history_review for question_key entry: {entry['id']!r}")
            continue
        disposition = record.get("disposition")
        if disposition in ("reuse", "insufficient"):
            if entry["id"] not in prior_ids:
                fail(
                    f"disposition {disposition!r} for {entry['id']!r} requires "
                    "prior_evidence"
                )

    if not history_entries:
        related_search = proposal.get("related_search")
        if not isinstance(related_search, str) or _is_blank(related_search):
            fail("no question_key match requires nonempty related_search")

    new_physical_actions = proposal.get("new_physical_actions")
    if not _is_int(new_physical_actions) or new_physical_actions < 0:
        fail("new_physical_actions must be a nonnegative integer")
        new_physical_actions = None

    complete_matches = []

    if intent in ("reuse_evidence", "analyze_existing"):
        if new_physical_actions is not None and new_physical_actions != 0:
            fail(f"{intent} requires new_physical_actions=0")
        try:
            _req_str_array(proposal, "analysis_steps", "proposal")
        except ReviewError as exc:
            fail(str(exc))

    if intent == "new_experiment":
        for key in (
            "hypothesis",
            "information_gain",
            "success_criteria",
            "failure_criteria",
            "unknown_criteria",
        ):
            try:
                _req_str(proposal, key, "proposal")
            except ReviewError as exc:
                fail(str(exc))

        changed_factors = proposal.get("changed_factors")
        if not isinstance(changed_factors, list) or not changed_factors:
            fail("changed_factors must be a nonempty array")
        else:
            for i, factor in enumerate(changed_factors):
                where = f"changed_factors[{i}]"
                if not isinstance(factor, dict):
                    fail(f"{where} must be an object")
                    continue
                if not isinstance(factor.get("name"), str) or _is_blank(factor.get("name")):
                    fail(f"{where}.name must be a nonempty string")
                if not isinstance(factor.get("reason"), str) or _is_blank(factor.get("reason")):
                    fail(f"{where}.reason must be a nonempty string")
                if "before" not in factor or "after" not in factor:
                    fail(f"{where} requires before and after")
                elif factor["before"] == factor["after"]:
                    fail(f"{where}.before must differ from after")
            if len(changed_factors) > 1:
                warnings.append("multiple_changed_factors")

        try:
            _req_dict(proposal, "fixed_factors", "proposal")
        except ReviewError as exc:
            fail(str(exc))

        scope = proposal.get("scope")
        scope_budget = None
        if not isinstance(scope, dict):
            fail("scope must be an object")
        else:
            try:
                max_cases = _req_pos_int(scope, "max_cases", "scope")
                max_vla = _req_nonneg_int(scope, "max_vla_actions_per_case", "scope")
                max_helper = _req_nonneg_int(scope, "max_helper_actions_per_case", "scope")
                _req_str(scope, "stop_rule", "scope")
                scope_budget = max_cases * (max_vla + max_helper)
            except ReviewError as exc:
                fail(str(exc))

        try:
            _req_dict(proposal, "configuration", "proposal")
        except ReviewError as exc:
            fail(str(exc))
        complete = proposal.get("configuration_complete")
        if complete is not True:
            fail("new_experiment requires configuration_complete=true")

        repeat = proposal.get("repeat")
        if not isinstance(repeat, dict):
            fail("repeat must be an object")
            repeat_needed = None
        else:
            needed = repeat.get("needed")
            if not isinstance(needed, bool):
                fail("repeat.needed must be a boolean")
                repeat_needed = None
            else:
                repeat_needed = needed
                if needed:
                    reason = repeat.get("reason")
                    if not isinstance(reason, str) or _is_blank(reason):
                        fail("repeat.needed=true requires nonempty reason")

        if new_physical_actions is not None:
            if new_physical_actions <= 0:
                fail("new_experiment requires new_physical_actions > 0")
            elif scope_budget is not None and new_physical_actions > scope_budget:
                fail(
                    "new_physical_actions exceeds scope budget "
                    f"({new_physical_actions} > {scope_budget})"
                )

        configuration = proposal.get("configuration")
        if isinstance(configuration, dict) and configuration and complete is True:
            try:
                fingerprint = configuration_fingerprint(configuration)
                complete_matches = _exact_complete_matches(entries, fingerprint)
            except ReviewError as exc:
                fail(str(exc))
            exact_duplicate_ids = [entry["id"] for entry in complete_matches]
            if complete_matches and repeat_needed is False:
                fail(
                    "exact duplicate configuration found but repeat.needed=false: "
                    + ", ".join(exact_duplicate_ids)
                )

    if errors:
        return {
            "ok": False,
            "errors": errors,
            "matched_history_ids": matched_history_ids,
            "exact_duplicate_ids": exact_duplicate_ids,
            "warnings": warnings,
            "scientific_review_required": True,
            "launch_authorized": False,
        }

    return {
        "ok": True,
        "intent": intent,
        "matched_history_ids": matched_history_ids,
        "exact_duplicate_ids": exact_duplicate_ids,
        "warnings": warnings,
        "scientific_review_required": True,
        "launch_authorized": False,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _default_root():
    return Path(__file__).resolve().parents[4]


def _default_registry(root: Path):
    return Path(root) / "First_Phase" / "scene_demo" / "experiment_registry.json"


def _emit(payload, code):
    if "scientific_review_required" not in payload:
        payload["scientific_review_required"] = True
    if "launch_authorized" not in payload:
        payload["launch_authorized"] = False
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return code


def _summarize_entry(entry):
    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "date": entry.get("date"),
        "question_keys": entry.get("question_keys"),
        "tags": entry.get("tags"),
        "conditions": entry.get("conditions"),
        "findings": entry.get("findings"),
        "limits": entry.get("limits"),
        "evidence": [
            {"path": item.get("path"), "sha256": item.get("sha256")}
            for item in entry.get("evidence", [])
            if isinstance(item, dict)
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="review.py", add_help=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--registry", default=None)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validate")

    query_parser = sub.add_parser("query")
    query_parser.add_argument("--text", nargs="+", required=True)

    check_parser = sub.add_parser("check")
    check_parser.add_argument("--proposal", required=True)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return _emit({"ok": False, "errors": ["invalid command line"]}, 2) if code else 0

    if args.command is None:
        return _emit({"ok": False, "errors": ["missing command"]}, 2)

    root = Path(args.root) if args.root else _default_root()
    registry_path = Path(args.registry) if args.registry else _default_registry(root)

    try:
        entries = load_registry(root, registry_path)
    except ReviewError as exc:
        return _emit(
            {
                "ok": False,
                "errors": [str(exc)],
                "scientific_review_required": True,
                "launch_authorized": False,
            },
            2,
        )
    except OSError as exc:
        return _emit(
            {
                "ok": False,
                "errors": [f"io error: {exc}"],
                "scientific_review_required": True,
                "launch_authorized": False,
            },
            2,
        )

    if args.command == "validate":
        evidence_count = sum(len(entry.get("evidence", [])) for entry in entries)
        return _emit(
            {
                "ok": True,
                "entry_count": len(entries),
                "evidence_count": evidence_count,
            },
            0,
        )

    if args.command == "query":
        try:
            matches = query_entries(entries, args.text)
        except ReviewError as exc:
            return _emit({"ok": False, "errors": [str(exc)]}, 2)
        return _emit(
            {
                "ok": True,
                "matches": [_summarize_entry(entry) for entry in matches],
            },
            0,
        )

    if args.command == "check":
        try:
            proposal_path = _resolve_inside_root(root, args.proposal, "proposal path")
        except ReviewError as exc:
            return _emit(
                {
                    "ok": False,
                    "errors": [str(exc)],
                    "scientific_review_required": True,
                    "launch_authorized": False,
                },
                2,
            )
        if not proposal_path.is_file():
            return _emit(
                {"ok": False, "errors": [f"proposal file missing: {proposal_path}"]},
                2,
            )
        try:
            raw = proposal_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return _emit(
                {"ok": False, "errors": [f"cannot decode proposal as utf-8: {exc}"]}, 2
            )
        except OSError as exc:
            return _emit({"ok": False, "errors": [f"cannot read proposal: {exc}"]}, 2)
        try:
            proposal = json.loads(raw)
        except json.JSONDecodeError as exc:
            return _emit(
                {"ok": False, "errors": [f"proposal is not valid JSON: {exc}"]}, 2
            )
        if not isinstance(proposal, dict):
            return _emit(
                {"ok": False, "errors": ["proposal must be a JSON object"]}, 2
            )
        verdict = review_proposal(proposal, entries)
        return _emit(verdict, 0 if verdict.get("ok") else 2)

    return _emit({"ok": False, "errors": ["unknown command"]}, 2)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
