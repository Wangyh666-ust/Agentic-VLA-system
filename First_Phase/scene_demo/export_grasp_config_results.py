#!/usr/bin/env python3
"""Deterministic passive exporter for completed paired grasp/config campaign evidence.

The exporter reads two *completed* paired-configuration campaign reports (the
``discovery`` report and the ``holdout`` report) plus any optional guard reports,
copies their exact input bytes into a fresh output directory and publishes a
self-describing artifact manifest, a per-trial ``summary.csv`` and a sorted
``SHA256SUMS`` list.  It never runs a model, a simulator, the network or git: it
only reads the explicitly declared source files and writes new output files.

Only ``pathlib``/``json``/``csv``/``hashlib``/``gzip``/``argparse`` standard-library
modules are imported.  Every source path is limited to a declared campaign
``run_root`` (symlinks resolved); directories are never copied recursively;
credential-like path components are refused.  Every recognized declared artifact
path is preflighted against *its own* report's ``metadata.run_root`` before the
output directory is created: a relative, out-of-root or symlink-escaping path is
refused up front, while a contained but absent optional file stays an explicit
missing record.  A value that is not present in the accepted paired schema is
recorded as *missing* rather than invented, and every output path is a safe,
collision-free, contained relative path.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

EXPERIMENT_NAME = "paired_config_experiments"
DISCOVERY_MIN_TRIALS = 19
_CHUNK_SIZE = 1024 * 1024

_SUMMARY_COLUMNS = (
    "phase",
    "profile",
    "state_index",
    "model_seed",
    "control",
    "policy_action_count",
    "assessment_action_count",
    "strict_policy_success",
    "semantic_before_assessment",
    "semantic_after_assessment",
    "first_grasp_confirmed_step",
    "first_failed_grasp_step",
    "policy_wall_s",
    "ended_reason",
    "initial_state_sha",
    "input_digest",
)

# (role, fixed file name) in deterministic export order.  The paired service
# writes these exact names inside each capability job's measured ``run_dir``.
_PNG_ROLES = (("first_png", "first.png"), ("last_png", "last.png"), ("latest_png", "latest.png"))
_JSON_ROLES = (("result", "result.json"), ("wine_diagnostic", "wine_diagnostic.json"))
_JSONL_ROLES = (
    ("wine_telemetry", "wine_telemetry.jsonl"),
    ("guard_semantic", "guard_semantic.jsonl"),
    ("semantic_status", "semantic_status.jsonl"),
    ("events", "events.jsonl"),
    ("assessment", "assessment.jsonl"),
)
_ROLLOUT_NAME = "rollout.mp4"

# The exact per-job declared path fields the exporter reads.  Both the campaign
# ``trials[].jobs[]`` records and the guard report's top-level ``jobs[]`` records
# use these exact keys; only these nonempty declared paths are preflighted.
_JOB_PATH_KEYS = (
    "run_dir",
    "rollout_path",
    "latest_png",
    "wine_telemetry_path",
    "wine_diagnostic_path",
    "events_path",
    "result_path",
)

_CREDENTIAL_TOKENS = (
    "credential",
    "secret",
    "password",
    "passwd",
    "token",
    "apikey",
    "api_key",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "private_key",
    "privatekey",
    ".env",
    "authorized_keys",
    "known_hosts",
    "keystore",
    "keychain",
)

_FIELD_SOURCES = {
    "phase": "trials[].phase",
    "profile": "trials[].profile",
    "state_index": "trials[].state_index",
    "model_seed": "trials[].model_seed",
    "control": "trials[].is_control",
    "policy_action_count": "trials[].policy_action_count",
    "assessment_action_count": "trials[].assessment_action_count",
    "strict_policy_success": (
        "trials[].strict_policy_success (benchmark strict score; kept separate from the "
        "semantic assessment columns)"
    ),
    "semantic_before_assessment": "trials[].semantic_before_assessment",
    "semantic_after_assessment": "trials[].semantic_after_assessment",
    "first_grasp_confirmed_step": (
        "trials[].observer.first_grasp_confirmed_step, else the paired top-level "
        "alternative trials[].grasp_confirmed_step (there is no top-level "
        "first_grasp_confirmed_step)"
    ),
    "first_failed_grasp_step": "trials[].first_failed_grasp_step",
    "policy_wall_s": "trials[].policy_wall_s",
    "ended_reason": (
        "trials[].jobs[-1].ended_reason (a measured per-job field; the paired trial "
        "record declares no trial-level ended_reason, so an absent value is left empty)"
    ),
    "initial_state_sha": "trials[].initial_state_sha",
    "input_digest": "trials[].first_input.combined_sha256",
}


class ExportError(Exception):
    """A deterministic refusal.  Any output written before the error is preserved."""


# --- pure helpers ------------------------------------------------------------


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(src, dst):
    digest = hashlib.sha256()
    size = 0
    with open(src, "rb") as reader, open(dst, "wb") as writer:
        for block in iter(lambda: reader.read(_CHUNK_SIZE), b""):
            digest.update(block)
            size += len(block)
            writer.write(block)
    return digest.hexdigest(), size


def _gzip_bytes_to_file(data, dst):
    with open(dst, "wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=9) as gz:
            gz.write(data)


def _safe_component(value):
    text = "" if value is None else str(value)
    if not text:
        raise ExportError("cannot derive a safe path component from an empty value")
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in text)
    if cleaned in (".", "..") or cleaned.strip(".") == "":
        raise ExportError("unsafe path component derived from %r" % (value,))
    return cleaned


def _contained(child, root):
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def _credential_part(path):
    for part in path.parts:
        lowered = part.lower()
        for token in _CREDENTIAL_TOKENS:
            if token in lowered:
                return part
    return None


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _first_grasp_step(trial):
    observer = trial.get("observer")
    if isinstance(observer, dict) and observer.get("first_grasp_confirmed_step") is not None:
        return observer.get("first_grasp_confirmed_step")
    return trial.get("grasp_confirmed_step")


def _ended_reason(trial):
    jobs = trial.get("jobs")
    if isinstance(jobs, list):
        for job in reversed(jobs):
            if isinstance(job, dict) and job.get("ended_reason") is not None:
                return job.get("ended_reason")
    return None


def _input_digest(trial):
    first_input = trial.get("first_input")
    if isinstance(first_input, dict):
        return first_input.get("combined_sha256")
    return None


def _summary_row(trial):
    return {
        "phase": trial.get("phase"),
        "profile": trial.get("profile"),
        "state_index": trial.get("state_index"),
        "model_seed": trial.get("model_seed"),
        "control": trial.get("is_control"),
        "policy_action_count": trial.get("policy_action_count"),
        "assessment_action_count": trial.get("assessment_action_count"),
        "strict_policy_success": trial.get("strict_policy_success"),
        "semantic_before_assessment": trial.get("semantic_before_assessment"),
        "semantic_after_assessment": trial.get("semantic_after_assessment"),
        "first_grasp_confirmed_step": _first_grasp_step(trial),
        "first_failed_grasp_step": trial.get("first_failed_grasp_step"),
        "policy_wall_s": trial.get("policy_wall_s"),
        "ended_reason": _ended_reason(trial),
        "initial_state_sha": trial.get("initial_state_sha"),
        "input_digest": _input_digest(trial),
    }


# --- input loading and validation -------------------------------------------


def _require_abs_json(raw, label):
    if not isinstance(raw, str) or not raw.strip():
        raise ExportError("%s is required" % label)
    path = Path(raw)
    if not path.is_absolute():
        raise ExportError("%s must be an absolute path: %r" % (label, raw))
    if path.suffix.lower() != ".json":
        raise ExportError("%s must be a .json path: %r" % (label, raw))
    if not path.is_file():
        raise ExportError("%s is not a readable file: %r" % (label, raw))
    return path.resolve()


def _require_fresh_dir(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ExportError("--output is required")
    path = Path(raw)
    if not path.is_absolute():
        raise ExportError("--output must be an absolute path: %r" % (raw,))
    if path.exists():
        raise ExportError("--output already exists; refusing to overwrite %r" % (raw,))
    return path


def _load_json_object(path, label):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - report the real failure
        raise ExportError("%s is not readable JSON: %s" % (label, exc))
    if not isinstance(obj, dict):
        raise ExportError("%s root is not a JSON object" % label)
    return obj


def _load_json_optional(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001 - the raw bytes are still exported verbatim
        return None, str(exc)


def _validate_campaign(report, label):
    if not isinstance(report, dict):
        raise ExportError("%s is not a JSON object" % label)
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ExportError("%s metadata is missing" % label)
    fatal = metadata.get("fatal_error")
    if fatal:
        raise ExportError("%s metadata.fatal_error is set: %r" % (label, fatal))
    trials = report.get("trials")
    if not isinstance(trials, list):
        raise ExportError("%s trials is missing" % label)
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ExportError("%s trials[%d] is not an object" % (label, index))
        operational = trial.get("operational_errors")
        if operational:
            raise ExportError(
                "%s trials[%d] carries operational_errors: %r" % (label, index, operational)
            )
    return metadata, trials


def _preregistration(report, label):
    preregistration = report.get("preregistration")
    if not isinstance(preregistration, dict):
        raise ExportError("%s preregistration is missing" % label)
    path = preregistration.get("path")
    plan = preregistration.get("plan")
    if not isinstance(path, str) or not path.strip():
        raise ExportError("%s preregistration.path is missing" % label)
    if not isinstance(plan, list):
        raise ExportError("%s preregistration.plan is missing" % label)
    return path, plan


# --- declared-path preflight (before ANY output is created) ------------------


def _preflight_path(raw, root, label, field):
    """Resolve one declared path and require containment in ``root``.

    A non-string/empty value is not a declared path and is skipped.  A relative
    path, an out-of-root path or a symlink-escaping absolute path raises
    ``ExportError``.  A contained but absent optional file is *not* fatal: it
    stays an explicit missing record later.  No file contents are read here.
    """

    if not isinstance(raw, str) or not raw.strip():
        return
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ExportError("%s %s must be an absolute path: %r" % (label, field, raw))
    try:
        resolved = candidate.resolve()
    except Exception as exc:  # noqa: BLE001 - an unresolvable path is refused
        raise ExportError("%s %s could not be resolved: %s" % (label, field, exc))
    if not _contained(resolved, root):
        raise ExportError(
            "%s %s resolves outside its own run_root %r: %r" % (label, field, str(root), raw)
        )


def _preflight_declared_paths(report, root, label):
    """Preflight every recognized declared artifact path BEFORE creating output.

    Only the exact fields the exporter reads are checked: each nonempty
    ``trials[].artifacts`` value, the exact ``trials[].jobs[]`` path keys, each
    ``trials[].assessment.path``, the top-level ``jobs[]`` path keys (guard
    reports) and ``preregistration.path``.  A declared absolute path must resolve
    inside *this* report's own ``metadata.run_root`` -- never a different
    campaign's root.  A relative, out-of-root or symlink-escaping path refuses
    before the output directory is created; a contained but absent optional file
    is left as an explicit missing record rather than a fatal error.
    """

    if not isinstance(report, dict):
        raise ExportError("%s is not a JSON object" % label)

    declared = []  # (field label, raw value)

    preregistration = report.get("preregistration")
    if isinstance(preregistration, dict):
        declared.append(("preregistration.path", preregistration.get("path")))

    def _collect_job(prefix, job):
        if isinstance(job, dict):
            for key in _JOB_PATH_KEYS:
                declared.append(("%s.%s" % (prefix, key), job.get(key)))

    trials = report.get("trials")
    if isinstance(trials, list):
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            prefix = "trials[%d]" % index
            artifacts = trial.get("artifacts")
            if isinstance(artifacts, dict):
                for key, value in artifacts.items():
                    declared.append(("%s.artifacts.%s" % (prefix, key), value))
            assessment = trial.get("assessment")
            if isinstance(assessment, dict):
                declared.append(("%s.assessment.path" % prefix, assessment.get("path")))
            jobs = trial.get("jobs")
            if isinstance(jobs, list):
                for job_index, job in enumerate(jobs):
                    _collect_job("%s.jobs[%d]" % (prefix, job_index), job)

    jobs = report.get("jobs")
    if isinstance(jobs, list):
        for job_index, job in enumerate(jobs):
            _collect_job("jobs[%d]" % job_index, job)

    for field, raw in declared:
        _preflight_path(raw, root, label, field)


def _validate_guard_report(obj, label):
    """Reject a parsed guard report that reports a fatal/operational failure.

    The real ``guard_validation`` schema carries a top-level ``ok``, a top-level
    ``fatal_error``, a ``metadata.fatal_error`` and a top-level
    ``operational_errors`` list.  An unparseable/non-object guard report cannot be
    checked and is left to be preserved raw as an explicit missing record.
    """

    if not isinstance(obj, dict):
        return
    if obj.get("ok") is False:
        raise ExportError("%s reports ok=False" % label)
    if obj.get("fatal_error"):
        raise ExportError("%s fatal_error is set: %r" % (label, obj.get("fatal_error")))
    metadata = obj.get("metadata")
    if isinstance(metadata, dict) and metadata.get("fatal_error"):
        raise ExportError(
            "%s metadata.fatal_error is set: %r" % (label, metadata.get("fatal_error"))
        )
    if obj.get("operational_errors"):
        raise ExportError(
            "%s carries operational_errors: %r" % (label, obj.get("operational_errors"))
        )


# --- the exporter run --------------------------------------------------------


class _Run:
    def __init__(self, output):
        self.output = output
        self.entries = []
        self.missing = []
        self.unsupported = []
        self.allowed_roots = []
        self.used_paths = set()
        self.seen_trial_ids = set()
        self.trial_index = {}
        self.summary_rows = 0

    # -- roots / containment -----------------------------------------------

    def add_root(self, root):
        if not isinstance(root, str) or not root.strip():
            return None
        path = Path(root)
        if not path.is_absolute():
            return None
        try:
            resolved = path.resolve()
        except Exception:  # noqa: BLE001 - an unresolvable root is not a root
            return None
        if resolved not in self.allowed_roots:
            self.allowed_roots.append(resolved)
        return resolved

    def resolve_source(self, raw):
        if not isinstance(raw, str) or not raw.strip():
            return None, "path not declared"
        candidate = Path(raw)
        if not candidate.is_absolute():
            return None, "declared path is not absolute"
        try:
            resolved = candidate.resolve()
        except Exception as exc:  # noqa: BLE001
            return None, "path resolve failed: %s" % exc
        if not any(_contained(resolved, root) for root in self.allowed_roots):
            return None, "resolved path is outside the declared run roots"
        bad = _credential_part(resolved)
        if bad is not None:
            return None, "refused credential-like path component %r" % (bad,)
        if not resolved.is_file():
            return None, "file does not exist"
        return resolved, None

    def _declared_dir(self, raw):
        if not isinstance(raw, str) or not raw.strip():
            return None, "run_dir not declared"
        candidate = Path(raw)
        if not candidate.is_absolute():
            return None, "run_dir is not absolute"
        try:
            resolved = candidate.resolve()
        except Exception as exc:  # noqa: BLE001
            return None, "run_dir resolve failed: %s" % exc
        if not any(_contained(resolved, root) for root in self.allowed_roots):
            return None, "run_dir is outside the declared run roots"
        bad = _credential_part(resolved)
        if bad is not None:
            return None, "refused credential-like run_dir component %r" % (bad,)
        if not resolved.is_dir():
            return None, "run_dir does not exist"
        return resolved, None

    # -- output path bookkeeping -------------------------------------------

    def register(self, rel):
        if not rel or rel.startswith("/") or "\\" in rel:
            raise ExportError("unsafe output relative path %r" % (rel,))
        parts = [part for part in rel.split("/") if part != ""]
        if not parts or any(part in (".", "..") for part in parts):
            raise ExportError("unsafe output relative path %r" % (rel,))
        normalized = "/".join(parts)
        if normalized in self.used_paths:
            raise ExportError("output path collision: %r" % (normalized,))
        self.used_paths.add(normalized)
        dst = self.output.joinpath(*parts)
        if not _contained(dst, self.output):
            raise ExportError("output path escapes the output directory: %r" % (normalized,))
        return dst

    def record_missing(self, meta, role, expected, reason):
        entry = {"role": role, "expected_source": expected, "reason": reason}
        entry.update(meta)
        self.missing.append(entry)

    # -- payload publishing ------------------------------------------------

    def _publish(self, src, out_rel, role, meta, gzip_it):
        dst = self.register(out_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if gzip_it:
            data = src.read_bytes()
            source_sha = hashlib.sha256(data).hexdigest()
            source_size = len(data)
            _gzip_bytes_to_file(data, dst)
            kind = "gzip"
        else:
            source_sha, source_size = _copy_file(src, dst)
            kind = "copy"
        entry = {
            "role": role,
            "kind": kind,
            "relative_path": out_rel,
            "source_path": str(src),
            "source_sha256": source_sha,
            "source_size": source_size,
            "published_sha256": _sha256_file(dst),
            "published_size": dst.stat().st_size,
        }
        entry.update(meta)
        self.entries.append(entry)

    def copy_payload(self, src_raw, out_rel, role, meta, gzip_it=False):
        src, reason = self.resolve_source(src_raw)
        if src is None:
            self.record_missing(meta, role, src_raw, reason)
            return
        self._publish(src, out_rel, role, meta, gzip_it)

    def copy_plain(self, src_path, out_rel, role, meta, gzip_it=False):
        src = Path(src_path)
        if not src.is_file():
            self.record_missing(meta, role, str(src), "file does not exist")
            return
        self._publish(src, out_rel, role, meta, gzip_it)

    # -- campaigns / guards -------------------------------------------------

    def index_trial(self, trial, phase):
        trial_id = trial.get("trial_id")
        if trial_id is None:
            return
        self.trial_index.setdefault(trial_id, []).append(trial)

    def register_guard_roots(self, obj):
        """Register only the guard report's own ``metadata.run_root``.

        The real ``guard_validation`` report declares its root at
        ``metadata.run_root``.  A job's ``run_dir`` is *never* trusted to introduce
        a new unbounded allowed root; it must stay contained in the declared guard
        root (enforced by :func:`_preflight_declared_paths` and ``resolve_source``).
        Returns the resolved guard root, or ``None`` when it is absent/relative.
        """

        if not isinstance(obj, dict):
            return None
        metadata = obj.get("metadata")
        if not isinstance(metadata, dict):
            return None
        return self.add_root(metadata.get("run_root"))

    # -- per-trial artifacts -----------------------------------------------

    def export_trial(self, trial, campaign_phase):
        trial_id = trial.get("trial_id")
        safe = _safe_component(trial_id)
        if trial_id in self.seen_trial_ids:
            raise ExportError("duplicate trial id %r" % (trial_id,))
        self.seen_trial_ids.add(trial_id)
        base = "trials/" + safe
        meta = {
            "trial_id": trial_id,
            "phase": trial.get("phase", campaign_phase),
            "profile": trial.get("profile"),
            "state_index": trial.get("state_index"),
            "model_seed": trial.get("model_seed"),
            "is_control": trial.get("is_control"),
        }
        jobs = [job for job in (trial.get("jobs") or []) if isinstance(job, dict)]
        if not jobs:
            for role, _name in _PNG_ROLES + _JSON_ROLES + _JSONL_ROLES:
                self.record_missing(meta, role, None, "trial declares no job run_dir")
            self.record_missing(meta, "rollout_video", None, "trial declares no job run_dir")
        multi = len(jobs) > 1
        for index, job in enumerate(jobs):
            job_meta = dict(meta)
            job_meta["job_id"] = job.get("job_id")
            job_meta["job_index"] = index
            if multi:
                job_base = base + "/job_%02d_%s" % (
                    index,
                    _safe_component(job.get("job_id") or ("job%d" % index)),
                )
            else:
                job_base = base
            self._export_job(job_meta, job_base, job)
        self.record_missing(
            meta,
            "session_initial_state",
            None,
            "no accepted paired trial field locates the session initial_state path",
        )

    def _export_job(self, job_meta, job_base, job):
        run_dir, _reason = self._declared_dir(job.get("run_dir"))

        def fixed(name):
            return str(run_dir / name) if run_dir is not None else None

        for role, name in _PNG_ROLES:
            declared = job.get("latest_png") if role == "latest_png" and isinstance(job.get("latest_png"), str) else None
            self.copy_payload(declared or fixed(name), job_base + "/" + name, role, job_meta)
        for role, name in _JSON_ROLES:
            key = "result_path" if role == "result" else "wine_diagnostic_path"
            declared = job.get(key) if isinstance(job.get(key), str) else None
            self.copy_payload(declared or fixed(name), job_base + "/" + name, role, job_meta)
        for role, name in _JSONL_ROLES:
            key = {"wine_telemetry": "wine_telemetry_path", "events": "events_path"}.get(role)
            declared = job.get(key) if key and isinstance(job.get(key), str) else None
            self.copy_payload(
                declared or fixed(name),
                job_base + "/" + name + ".gz",
                role,
                job_meta,
                gzip_it=True,
            )

    # -- videos -------------------------------------------------------------

    def _rollout_candidate(self, trial):
        artifacts = trial.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        candidates = []
        if isinstance(artifacts.get("rollout"), str):
            candidates.append(artifacts.get("rollout"))
        for job in trial.get("jobs") or []:
            if not isinstance(job, dict):
                continue
            if isinstance(job.get("rollout_path"), str):
                candidates.append(job.get("rollout_path"))
            run_dir = job.get("run_dir")
            if isinstance(run_dir, str):
                candidates.append(str(Path(run_dir) / _ROLLOUT_NAME))
        for raw in candidates:
            src, _reason = self.resolve_source(raw)
            if src is not None:
                return raw
        return None

    def resolve_videos(self, video_ids):
        # The actual guard_validation report declares no trial_id, so there is no
        # special guard video-ID lookup: a guard report is only raw-preserved.
        resolved = []
        seen = set()
        for video_id in video_ids:
            if not isinstance(video_id, str) or not video_id.strip():
                raise ExportError("--video-trial-ids contains an empty value")
            if video_id in seen:
                raise ExportError("duplicate --video-trial-ids value %r" % (video_id,))
            seen.add(video_id)
            candidates = []
            for trial in self.trial_index.get(video_id, []):
                raw = self._rollout_candidate(trial)
                if raw is not None:
                    candidates.append(raw)
            unique = []
            for raw in candidates:
                if raw not in unique:
                    unique.append(raw)
            if not unique:
                raise ExportError(
                    "unknown --video-trial-ids value %r; no completed trial declares a "
                    "resolvable rollout video" % (video_id,)
                )
            if len(unique) > 1:
                raise ExportError(
                    "ambiguous --video-trial-ids value %r matches %d distinct sources"
                    % (video_id, len(unique))
                )
            resolved.append({"trial_id": video_id, "safe": _safe_component(video_id), "source": unique[0]})
        return resolved

    def export_video(self, video):
        src, reason = self.resolve_source(video["source"])
        if src is None:
            raise ExportError(
                "selected video for trial %r is unavailable: %s" % (video["trial_id"], reason)
            )
        meta = {"trial_id": video["trial_id"], "phase": None}
        self._publish(src, "videos/" + video["safe"] + ".mp4", "rollout_video", meta, False)

    # -- guard declared logs ------------------------------------------------

    def export_guard(self, index, obj, err):
        meta = {"phase": None, "guard_index": index}
        if obj is None:
            self.record_missing(
                meta,
                "guard_fields",
                None,
                "guard report JSON could not be parsed (%s); raw bytes exported verbatim" % (err,),
            )
            return
        if not isinstance(obj, dict):
            self.record_missing(
                meta, "guard_fields", None, "guard report root is not an object; raw bytes exported verbatim"
            )
            return
        jobs = obj.get("jobs")
        if not isinstance(jobs, list):
            self.record_missing(
                meta,
                "guard_jobs",
                None,
                "guard report declares no jobs[] list; guard job log paths are unsupported",
            )
            return
        for job_index, job in enumerate(jobs):
            if not isinstance(job, dict):
                continue
            job_meta = {
                "phase": None,
                "guard_index": index,
                "job_id": job.get("job_id"),
                "job_index": job_index,
            }
            copied = False
            for key, value in job.items():
                if key == "run_dir" or not isinstance(value, str) or "/" not in value:
                    continue
                lowered = value.lower()
                if lowered.endswith(".jsonl") or lowered.endswith(".log"):
                    gzip_it = True
                    extension = ".gz"
                elif lowered.endswith(".json"):
                    gzip_it = False
                    extension = ""
                else:
                    continue
                rel = "guard/guard_%02d/job_%02d/%s_%s%s" % (
                    index,
                    job_index,
                    _safe_component(key),
                    _safe_component(Path(value).name),
                    extension,
                )
                before = len(self.entries)
                self.copy_payload(value, rel, "guard_" + str(key), job_meta, gzip_it=gzip_it)
                if len(self.entries) > before:
                    copied = True
            if not copied:
                self.record_missing(
                    job_meta,
                    "guard_job_logs",
                    job.get("run_dir"),
                    "guard job declares no supported log path field; optional fields unsupported",
                )

    # -- summary / manifest / checksums ------------------------------------

    def write_summary(self, rows):
        rel = "summary.csv"
        dst = self.register(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(list(_SUMMARY_COLUMNS))
            for row in rows:
                writer.writerow([_cell(row.get(column)) for column in _SUMMARY_COLUMNS])
        self.summary_rows = len(rows)
        sha = _sha256_file(dst)
        size = dst.stat().st_size
        self.entries.append(
            {
                "role": "summary",
                "kind": "generated",
                "relative_path": rel,
                "source_path": None,
                "source_sha256": sha,
                "source_size": size,
                "published_sha256": sha,
                "published_size": size,
                "phase": None,
            }
        )

    def write_manifest(self, discovery_path, holdout_path, guard_paths, video_ids):
        manifest = {
            "experiment": EXPERIMENT_NAME,
            "exporter": "export_grasp_config_results",
            "inputs": {
                "discovery": str(discovery_path),
                "holdout": str(holdout_path),
                "guard_reports": [str(path) for path in guard_paths],
                "run_roots": [str(root) for root in self.allowed_roots],
            },
            "video_trial_ids": list(video_ids),
            "summary": {
                "relative_path": "summary.csv",
                "columns": list(_SUMMARY_COLUMNS),
                "rows": self.summary_rows,
            },
            "field_sources": dict(_FIELD_SOURCES),
            "unsupported": self.unsupported,
            "entries": self.entries,
            "missing": self.missing,
        }
        dst = self.register("artifact_manifest.json")
        dst.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def write_sha256sums(self):
        files = [path for path in self.output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
        files.sort(key=lambda path: path.relative_to(self.output).as_posix())
        lines = []
        for path in files:
            rel = path.relative_to(self.output).as_posix()
            lines.append("%s  %s" % (_sha256_file(path), rel))
        text = "\n".join(lines)
        if text:
            text += "\n"
        (self.output / "SHA256SUMS").write_text(text, encoding="utf-8")


# --- top-level export --------------------------------------------------------


def export_results(args):
    discovery_raw = getattr(args, "discovery", None)
    holdout_raw = getattr(args, "holdout", None)
    guard_raws = list(getattr(args, "guard_reports", None) or [])
    output_raw = getattr(args, "output", None)
    video_ids = list(getattr(args, "video_trial_ids", None) or [])

    discovery_path = _require_abs_json(discovery_raw, "--discovery")
    holdout_path = _require_abs_json(holdout_raw, "--holdout")
    guard_paths = [_require_abs_json(raw, "--guard-reports") for raw in guard_raws]
    output = _require_fresh_dir(output_raw)

    discovery = _load_json_object(discovery_path, "--discovery")
    holdout = _load_json_object(holdout_path, "--holdout")
    guard_objects = []
    for path in guard_paths:
        obj, err = _load_json_optional(path)
        guard_objects.append((path, obj, err))

    discovery_meta, discovery_trials = _validate_campaign(discovery, "discovery")
    holdout_meta, holdout_trials = _validate_campaign(holdout, "holdout")
    discovery_prereg_path, _discovery_plan = _preregistration(discovery, "discovery")
    holdout_prereg_path, holdout_plan = _preregistration(holdout, "holdout")

    if len(discovery_trials) < DISCOVERY_MIN_TRIALS:
        raise ExportError(
            "discovery completed campaign has %d trials; at least %d are required"
            % (len(discovery_trials), DISCOVERY_MIN_TRIALS)
        )
    if len(holdout_trials) != len(holdout_plan):
        raise ExportError(
            "holdout trial count %d != preregistration.plan length %d"
            % (len(holdout_trials), len(holdout_plan))
        )

    run = _Run(output)
    discovery_root = run.add_root(discovery_meta.get("run_root"))
    holdout_root = run.add_root(holdout_meta.get("run_root"))
    if discovery_root is None:
        raise ExportError("discovery metadata.run_root is missing or not absolute")
    if holdout_root is None:
        raise ExportError("holdout metadata.run_root is missing or not absolute")
    guard_roots = [
        run.register_guard_roots(obj) for _path, obj, _err in guard_objects
    ]

    # --- refuse any invalid declared path BEFORE creating the output dir ------
    # Each report's declared artifacts must resolve inside its *own*
    # ``metadata.run_root``; a relative, out-of-root or symlink-escaping path (or
    # a guard report that failed operationally) refuses here, before ``mkdir``.
    _preflight_declared_paths(discovery, discovery_root, "discovery")
    _preflight_declared_paths(holdout, holdout_root, "holdout")
    for (guard_path, obj, _err), guard_root in zip(guard_objects, guard_roots):
        _validate_guard_report(obj, "--guard-reports %s" % guard_path)
        if guard_root is not None:
            _preflight_declared_paths(obj, guard_root, "guard %s" % guard_path)

    for label, prereg_path in (("discovery", discovery_prereg_path), ("holdout", holdout_prereg_path)):
        src, reason = run.resolve_source(prereg_path)
        if src is None:
            raise ExportError("%s preregistration.path is unusable: %s" % (label, reason))

    for trial in discovery_trials:
        run.index_trial(trial, "discovery")
    for trial in holdout_trials:
        run.index_trial(trial, "holdout")

    videos = run.resolve_videos(video_ids)

    # --- validate everything above BEFORE creating the output directory ---
    output.mkdir(parents=True, exist_ok=False)

    run.unsupported = [
        {
            "field": "session.initial_state",
            "reason": (
                "the accepted paired trial schema declares no session initial_state path, so it "
                "is recorded missing and no file is fabricated"
            ),
        },
    ]

    run.copy_plain(str(discovery_path), "raw/discovery.json", "raw_discovery", {"phase": "discovery"})
    run.copy_plain(str(holdout_path), "raw/holdout.json", "raw_holdout", {"phase": "holdout"})
    run.copy_plain(
        str(discovery_prereg_path), "raw/preregistration_discovery.json", "raw_preregistration", {"phase": "discovery"}
    )
    run.copy_plain(
        str(holdout_prereg_path), "raw/preregistration_holdout.json", "raw_preregistration", {"phase": "holdout"}
    )
    for index, path in enumerate(guard_paths):
        run.copy_plain(
            str(path), "raw/guard_%02d.json" % index, "raw_guard", {"phase": None, "guard_index": index}
        )

    for trial in discovery_trials:
        run.export_trial(trial, "discovery")
    for trial in holdout_trials:
        run.export_trial(trial, "holdout")

    for video in videos:
        run.export_video(video)

    for index, (_path, obj, err) in enumerate(guard_objects):
        run.export_guard(index, obj, err)

    rows = [_summary_row(trial) for trial in list(discovery_trials) + list(holdout_trials)]
    run.write_summary(rows)

    manifest = run.write_manifest(discovery_path, holdout_path, guard_paths, video_ids)
    run.write_sha256sums()
    return manifest


# --- CLI ---------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic passive exporter for completed paired grasp/config campaign "
            "evidence (read-only; no model, simulator, network or git)."
        )
    )
    parser.add_argument(
        "--discovery",
        required=True,
        type=str,
        help="absolute path of the completed discovery campaign JSON",
    )
    parser.add_argument(
        "--holdout",
        required=True,
        type=str,
        help="absolute path of the completed holdout campaign JSON",
    )
    parser.add_argument(
        "--guard-reports",
        dest="guard_reports",
        action="append",
        default=[],
        type=str,
        help="absolute path of a guard report JSON (repeatable, optional)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="fresh absolute output directory (must not exist)",
    )
    parser.add_argument(
        "--video-trial-ids",
        dest="video_trial_ids",
        action="append",
        default=[],
        type=str,
        help="exact trial id whose rollout video is exported (repeatable, optional)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        export_results(args)
    except ExportError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
