"""Standard-library validation and I/O helpers for the safe-exit pilot.

No simulator, model, network, or environment is imported or launched here.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "results" / "2026-10-09-safe-exit" / "pilot"
INPUTS_SHA = "5fae1983b80eedbc38f4db30ce6ac5607ebb7ed24b612fbe5814085e1fbc0f71"
CASES = ("case_02", "case_07", "case_08")

_REQUIRED_SKILLS = (
    ".agents/skills/fyp-experiment-review/scripts/review.py",
    ".agents/skills/fyp-experiment-review/tests/test_review.py",
    ".agents/skills/fyp-experiment-review/references/schema.md",
)

_CASE_EXPECT = {
    "case_02": {"vla_actions": 180, "init_state_index": 1, "capability": "wine_to_rack"},
    "case_07": {"vla_actions": 78, "init_state_index": 0, "capability": "stove_on"},
    "case_08": {"vla_actions": 76, "init_state_index": 1, "capability": "stove_on"},
}


def sha(p):
    digest = hashlib.sha256()
    with open(p, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(p):
    with open(p, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _plain(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(inner) for inner in value]
    raise TypeError("cannot serialize value of type %r" % (type(value).__name__,))


def write_json(target, value):
    destination = Path(target)
    data = _plain(value)
    with open(destination, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def load_module(name, source):
    spec = importlib.util.spec_from_file_location(name, str(source))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module %r from %r" % (name, source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def path(value):
    raw = str(value)
    normalized = raw.replace("\\", "/")
    lowered = normalized.lower()
    if lowered == "d:/fyp" or lowered.startswith("d:/fyp/"):
        remainder = normalized[6:].lstrip("/")
        return ROOT / remainder if remainder else ROOT
    if Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        return Path(normalized)
    return ROOT / normalized


def verify_facts(facts):
    if facts is None:
        return True
    if isinstance(facts, Mapping):
        records = []
        for key, value in facts.items():
            if not isinstance(value, Mapping):
                raise ValueError("fact value for %r must be an object" % (key,))
            record = dict(value)
            record["path"] = key
            records.append(record)
    elif isinstance(facts, (list, tuple)):
        records = list(facts)
    else:
        raise ValueError("facts must be an object or a list")

    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("each fact record must be an object")
        record = dict(raw)
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError("fact record requires a non-empty string path")
        target = path(raw_path)
        if not target.is_file():
            raise FileNotFoundError("fact source is not a file: %s" % (target,))
        actual_bytes = target.stat().st_size
        actual_sha = sha(target)
        if "bytes" in record:
            try:
                declared = int(record["bytes"])
            except Exception as exc:
                raise ValueError("fact bytes value is not an integer: %s" % (target,)) from exc
            if declared != actual_bytes:
                raise ValueError("fact bytes mismatch for %s" % (target,))
        registry_style = "registry_sha256" in record and "sha256" not in record
        if registry_style:
            if record.get("matches") is not True:
                raise ValueError("registry fact matches must be True: %s" % (target,))
            nested = record.get("actual")
            if not isinstance(nested, Mapping):
                raise ValueError("registry fact missing actual object: %s" % (target,))
            if "bytes" in nested:
                try:
                    nested_bytes = int(nested["bytes"])
                except Exception as exc:
                    raise ValueError(
                        "registry actual bytes value is not an integer: %s" % (target,)
                    ) from exc
                if nested_bytes != actual_bytes:
                    raise ValueError("registry actual bytes mismatch: %s" % (target,))
            if nested.get("sha256") != actual_sha:
                raise ValueError("registry actual sha256 mismatch: %s" % (target,))
            if record.get("registry_sha256") != actual_sha:
                raise ValueError("registry_sha256 mismatch: %s" % (target,))
        else:
            expected = record.get("sha256")
            if not isinstance(expected, str):
                raise ValueError("fact record missing sha256: %s" % (target,))
            if expected != actual_sha:
                raise ValueError("fact sha256 mismatch: %s" % (target,))
    return True


def _require_absolute(name, value):
    if not isinstance(value, str) or value == "":
        raise ValueError("args.%s must be a non-empty string" % (name,))
    normalized = value.replace("\\", "/")
    if not Path(normalized).is_absolute() and not PureWindowsPath(normalized).is_absolute():
        raise ValueError("args.%s must be an absolute path: %r" % (name, value))
    return normalized


def _mapping_field(container, key, label):
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError("%s.%s must be an object" % (label, key))
    return value


def _verify_inputs(args):
    source = path(args.inputs)
    if not source.is_file():
        raise FileNotFoundError("inputs file is missing: %s" % (source,))
    if sha(source) != INPUTS_SHA:
        raise ValueError("inputs file sha256 does not equal INPUTS_SHA")
    inputs = read(source)
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a JSON object")
    if inputs.get("ok") is not True:
        raise ValueError("inputs.ok must be the boolean True")
    cases = inputs.get("cases")
    if not isinstance(cases, list):
        raise ValueError("inputs.cases must be a list")
    if [case.get("case_id") if isinstance(case, Mapping) else None for case in cases] != list(CASES):
        raise ValueError("inputs case_id order must equal %r" % (list(CASES),))

    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("each inputs.cases entry must be an object")
        cid = case["case_id"]
        if cid not in _CASE_EXPECT:
            raise ValueError("unexpected case_id %r" % (cid,))
        spec = _mapping_field(case, "spec", cid)
        worker = _mapping_field(case, "worker", cid)
        home = _mapping_field(case, "home", cid)
        expected = _CASE_EXPECT[cid]
        vla = worker.get("vla_actions")
        if type(vla) is not int or vla != expected["vla_actions"]:
            raise ValueError(
                "%s.worker.vla_actions must be int %d" % (cid, expected["vla_actions"])
            )
        index = spec.get("init_state_index")
        if type(index) is not int or index != expected["init_state_index"]:
            raise ValueError(
                "%s.spec.init_state_index must be int %d" % (cid, expected["init_state_index"])
            )
        if case.get("capability_id") != expected["capability"]:
            raise ValueError("%s.capability_id must be %r" % (cid, expected["capability"]))
        reference = case.get("reference")
        native_reference = case.get("native_home_reference")
        if not isinstance(reference, Mapping) or reference != native_reference:
            raise ValueError("%s.reference must equal native_home_reference" % (cid,))
        initial = case.get("initial_state_sha")
        before = worker.get("before_sha256")
        origin = reference.get("origin_sha256")
        native_origin = case.get("native_home_origin_sha256")
        if not isinstance(initial, str) or initial == "":
            raise ValueError("%s.initial_state_sha is missing" % (cid,))
        if not (initial == before == origin == native_origin):
            raise ValueError(
                "%s: initial_state_sha == worker.before_sha256 == reference.origin_sha256 "
                "== native_home_origin_sha256 must hold" % (cid,)
            )
        after = worker.get("after_sha256")
        if not isinstance(after, str) or after == "":
            raise ValueError("%s.worker.after_sha256 is missing" % (cid,))
        if after != home.get("final_state_sha256"):
            raise ValueError(
                "%s.worker.after_sha256 must equal home.final_state_sha256" % (cid,)
            )

    frozen = inputs.get("frozen_files")
    if not isinstance(frozen, Mapping):
        raise ValueError("inputs.frozen_files must be an object keyed by file path")
    verify_facts(frozen)

    registry = inputs.get("registry_evidence")
    if not isinstance(registry, list) or len(registry) != 21:
        raise ValueError("inputs.registry_evidence must be a list of 21 records")
    verify_facts(registry)

    user8 = inputs.get("user8_before")
    if not isinstance(user8, list) or len(user8) != 8:
        raise ValueError("inputs.user8_before must be a list of 8 records")
    verify_facts(user8)

    historical = inputs.get("historical_runner_facts")
    if historical is not None:
        verify_facts(historical)
    return inputs


def _verify_software(args):
    source = path(args.software_proof)
    if not source.is_file():
        raise FileNotFoundError("software proof file is missing: %s" % (source,))
    software = read(source)
    if not isinstance(software, Mapping):
        raise ValueError("software proof must be a JSON object")
    if software.get("ok") is not True:
        raise ValueError("software.ok must be the boolean True")
    exit_code = software.get("test_exit_code")
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError("software.test_exit_code must be int 0")

    declared_sources = software.get("source_sha256")
    if not isinstance(declared_sources, Mapping):
        raise ValueError("software.source_sha256 must be an object")
    source_files = sorted(
        (entry for entry in HERE.iterdir() if entry.is_file() and entry.suffix == ".py"),
        key=lambda entry: entry.name,
    )
    expected = {entry.name: entry for entry in source_files}
    if set(declared_sources.keys()) != set(expected.keys()):
        raise ValueError(
            "software.source_sha256 keys must exactly match the basenames of HERE/*.py"
        )
    for name, entry in expected.items():
        if declared_sources[name] != sha(entry):
            raise ValueError("software.source_sha256 mismatch for %s" % (name,))

    proposal_path = path(args.proposal)
    if not proposal_path.is_file():
        raise FileNotFoundError("proposal file is missing: %s" % (proposal_path,))
    if software.get("proposal_sha256") != sha(proposal_path):
        raise ValueError("software.proposal_sha256 must equal sha(args.proposal)")

    declared_skills = software.get("skill_sha256")
    if not isinstance(declared_skills, Mapping):
        raise ValueError("software.skill_sha256 must be an object")
    for required in _REQUIRED_SKILLS:
        if required not in declared_skills:
            raise ValueError("software.skill_sha256 missing %s" % (required,))
    root_resolved = ROOT.resolve()
    for key, declared in declared_skills.items():
        if not isinstance(key, str) or key == "":
            raise ValueError("software.skill_sha256 keys must be non-empty strings")
        normalized = key.replace("\\", "/")
        if Path(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
            raise ValueError("software.skill_sha256 key must be relative: %r" % (key,))
        target = (ROOT / normalized).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                "software.skill_sha256 path escapes ROOT: %r" % (key,)
            ) from exc
        if not target.is_file():
            raise FileNotFoundError(
                "software.skill_sha256 target is not a file: %s" % (target,)
            )
        if not isinstance(declared, str) or declared != sha(target):
            raise ValueError("software.skill_sha256 mismatch for %r" % (key,))
    return software


def _verify_proposal(args):
    source = path(args.proposal)
    if not source.is_file():
        raise FileNotFoundError("proposal file is missing: %s" % (source,))
    proposal = read(source)
    if not isinstance(proposal, Mapping):
        raise ValueError("proposal must be a JSON object")
    if proposal.get("intent") != "new_experiment":
        raise ValueError("proposal.intent must be 'new_experiment'")
    if proposal.get("question_key") != "joint-home-handoff":
        raise ValueError("proposal.question_key must be 'joint-home-handoff'")

    configuration = proposal.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("proposal.configuration must be an object")
    cap = configuration.get("actual_vla_cap")
    if type(cap) is not int or cap != 0:
        raise ValueError("proposal.configuration.actual_vla_cap must be int 0")
    retries = configuration.get("retries")
    if type(retries) is not int or retries != 0:
        raise ValueError("proposal.configuration.retries must be int 0")
    if configuration.get("cases") != list(CASES):
        raise ValueError("proposal.configuration.cases must equal %r" % (list(CASES),))
    frozen_sha = configuration.get("frozen_inputs_sha256")
    alias_sha = configuration.get("inputs_sha256")
    if frozen_sha is None and alias_sha is None:
        raise ValueError("proposal.configuration.frozen_inputs_sha256 is required")
    if frozen_sha is not None and alias_sha is not None and frozen_sha != alias_sha:
        raise ValueError(
            "proposal.configuration.frozen_inputs_sha256 and inputs_sha256 disagree"
        )
    effective = frozen_sha if frozen_sha is not None else alias_sha
    if effective != INPUTS_SHA:
        raise ValueError(
            "proposal.configuration.frozen_inputs_sha256 must equal INPUTS_SHA"
        )

    scope = proposal.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("proposal.scope must be an object")
    max_cases = scope.get("max_cases")
    if type(max_cases) is not int or max_cases != 3:
        raise ValueError("proposal.scope.max_cases must be int 3")
    max_vla = scope.get("max_vla_actions_per_case")
    if type(max_vla) is not int or max_vla != 0:
        raise ValueError("proposal.scope.max_vla_actions_per_case must be int 0")
    max_helper = scope.get("max_helper_actions_per_case")
    if type(max_helper) is not int or max_helper != 620:
        raise ValueError("proposal.scope.max_helper_actions_per_case must be int 620")
    new_actions = proposal.get("new_physical_actions")
    if type(new_actions) is not int or new_actions != 1654:
        raise ValueError("proposal.new_physical_actions must be int 1654")
    return proposal


def _run_skill_check(proposal_path):
    script = ROOT / ".agents" / "skills" / "fyp-experiment-review" / "scripts" / "review.py"
    registry = HERE / "experiment_registry.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--root",
            str(ROOT),
            "--registry",
            str(registry),
            "check",
            "--proposal",
            str(proposal_path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "skill check exited %d: %s" % (completed.returncode, completed.stderr.strip())
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "skill check stdout is not valid JSON: %r" % (completed.stdout,)
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("skill check stdout must be a JSON object")
    if payload.get("ok") is not True:
        raise RuntimeError("skill check payload.ok must be the boolean True")
    if payload.get("intent") != "new_experiment":
        raise RuntimeError("skill check payload.intent must be 'new_experiment'")
    if payload.get("launch_authorized") is not False:
        raise RuntimeError(
            "skill check payload.launch_authorized must be the boolean False"
        )
    return payload


def gates(args):
    for name in ("inputs", "software_proof", "proposal", "output_dir"):
        _require_absolute(name, getattr(args, name, None))
    if path(args.output_dir).resolve() != OUTPUT.resolve():
        raise ValueError("args.output_dir must equal OUTPUT")
    if OUTPUT.exists():
        raise FileExistsError("OUTPUT already exists: %s" % (OUTPUT,))

    inputs = _verify_inputs(args)
    software = _verify_software(args)
    proposal = _verify_proposal(args)
    _run_skill_check(path(args.proposal))
    return inputs, software, proposal
