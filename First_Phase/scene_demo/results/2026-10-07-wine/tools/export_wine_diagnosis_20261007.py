#!/usr/bin/env python3
"""Private deterministic six-trial wine-evidence exporter (generate/reread only).

This script is a *pure, offline* exporter.  It reads one already-finished
campaign report (given on the command line) plus the exact evidence files that
report references, and it writes a self-describing, hash-audited evidence
bundle into a brand-new output directory.  It never imports the VLA stack, the
scene service, a model, or any GPU/network client; it never initialises an
interpreter environment beyond the already-running CPython interpreter; and it
never runs the physical experiment.

Interface facts this exporter is pinned to (read from the exact sources, never
guessed):

* ``scene_demo/catalog.py`` -- ``catalog.goal_key(goal)`` is ``"|".join(goal)``,
  so the fixed wine predicate key for
  ``["on", "wine_bottle_1", "wine_rack_1_top_region"]`` is literally
  ``on|wine_bottle_1|wine_rack_1_top_region``.  That literal key is hard-coded
  below; there is NO alternative key lookup and NO fallback.
* ``scene_demo/wine_diagnostics.py`` -- each telemetry sample is a JSON object
  with ``step``, ``raw_policy_action``, ``postprocessed_action``,
  ``sent_action`` (a float vector whose gripper command is its last element),
  ``instruction``, ``policy_input_state``, ``before_snapshot``,
  ``after_snapshot``, ``gripper_before``, ``gripper_after`` and
  ``native_success``.  ``gripper_after.width_m`` is the jaw gap in metres;
  ``after_snapshot.objects.wine_bottle_1.position`` is the object position;
  ``after_snapshot.objects.wine_bottle_1.grasped`` is the contact/kinematic
  grasp proxy; ``after_snapshot.predicates[<key>]`` is the fixed predicate.

Design rules:

* Inputs are fully validated *before* the output directory is created, and a
  pre-existing destination is refused (never overwritten).
* The campaign report is copied byte-for-byte; every copied artifact is
  byte-identical to its source; the wine telemetry is gzip-compressed with a
  deterministic header (``mtime=0``, empty header filename) and the round-trip
  is verified against the source bytes and SHA-256.
* ``summary.csv`` and ``timeline.png`` are *generated* (never claimed to be raw
  byte copies); they record their exact sources and source hashes.
* Rollout-video selection is a media choice only: exactly three of the fixed
  six trials are selected -- pair 0:0 for BOTH conditions (the two videos
  predetermined BEFORE any results were seen) plus shared_goal8 pair 1:1 (an
  additional video added because it demonstrates a distinct placement-stage
  failure identified by the primary agent).  ALL SIX trials remain in the
  statistics, summary CSV and raw evidence, and the timeline stays pair 0:0
  (both conditions, unchanged columns/scales).  The selection is independent of
  success and implies no success judgement and no new causal diagnosis.
* Missing source files are recorded as missing; nothing is substituted,
  fabricated or inferred.  No diagnosis, rating or success judgement is made.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- fixed, literal interface constants --------------------------------------

# The two fixed conditions, in their fixed column order.
CONDITIONS = ("native_goal9", "shared_goal8")

# --- rollout-video selection (a media choice only) ---------------------------
# Exactly three of the fixed six trials have their rollout video exported:
#   * pair 0:0 for BOTH conditions -- two videos, predetermined BEFORE any
#     results were seen (native_goal9 pair 0:0 and shared_goal8 pair 0:0);
#   * shared_goal8 pair 1:1 -- ONE additional selected video, added because it
#     demonstrates a distinct placement-stage failure identified by the primary
#     agent.
# This selection is by condition/pair only and is independent of success.  It
# is a media choice and implies no success claim and no new causal diagnosis.
# It does NOT drop any trial: ALL SIX trials remain in the statistics, summary
# CSV and raw evidence -- only the rollout.mp4 of the other three is unselected.
# The timeline stays pair 0:0 (both conditions) exactly as before.
PREDETERMINED_MEDIA_PAIR = "0:0"
ADDITIONAL_MEDIA_CONDITION = "shared_goal8"
ADDITIONAL_MEDIA_PAIR = "1:1"

# catalog.goal_key(goal) == "|".join(goal)  ->  literal fixed wine key.
WINE_PREDICATE_KEY = "on|wine_bottle_1|wine_rack_1_top_region"
WINE_OBJECT_ID = "wine_bottle_1"

EXPECTED_TRIAL_COUNT = 6

# EXACT ordered summary columns (order is part of the interface).
SUMMARY_COLUMNS = (
    "condition",
    "pair",
    "steps",
    "ended_reason",
    "strict_task_success",
    "task_success",
    "first_grasp_proxy_step",
    "first_lift_2cm_step",
    "max_lift_m",
    "first_goal_true_step",
    "sent_open_count",
    "sent_close_count",
    "open_command_width_increased_count",
    "width_min_m",
    "width_max_m",
)

# Per-trial files copied from ``artifacts.run_dir`` when present.
RUN_DIR_COPY_FILES = (
    "result.json",
    "wine_diagnostic.json",
    "diagnostic.json",
    "events.jsonl",
    "first.png",
    "last.png",
)

CAMPAIGN_COPY_BASENAME = "campaign.json"
SUMMARY_BASENAME = "summary.csv"
PLOT_BASENAME = "timeline.png"
MANIFEST_BASENAME = "manifest.json"
TRIALS_DIRNAME = "trials"
TRIAL_JSON_BASENAME = "trial.json"
TELEMETRY_GZ_BASENAME = "wine_telemetry.jsonl.gz"
ROLLOUT_BASENAME = "rollout.mp4"

FIG_SIZE_INCHES = (12.0, 12.0)
FIG_DPI = 140
ROW_COUNT = 5

ROW_YLABELS = (
    "Gripper command\n- open / + close",
    "Actual jaw gap (mm)",
    "Bottle rise (cm)",
    "Holding proxy (0/1)",
    "In rack target (0/1)",
)

# Per-condition fixed title prefixes for the two rendered timeline columns.
COLUMN_TITLE_PREFIXES = {
    "native_goal9": "Native wine scene (goal 9)",
    "shared_goal8": "Shared scene (goal 8)",
}


class ExportError(Exception):
    """A fatal, non-recoverable exporter error (input validation / integrity)."""


# --- small numeric / hashing helpers -----------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object) -> "float | None":
    """A finite float scalar, or ``None`` for a missing/malformed/nonfinite one."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _is_plot_number(value: object) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _bundle_rel(output_dir: Path, path: Path) -> str:
    rel = os.path.relpath(str(path), str(output_dir))
    return rel.replace(os.sep, "/")


def _is_safe_component(name: str) -> bool:
    if name in ("", ".", ".."):
        return False
    if "/" in name or "\\" in name or os.sep in name:
        return False
    return True


def _resolve_within(base_real: str, candidate: Path) -> str:
    """Resolve ``candidate`` and require it to stay under ``base_real``."""

    real = os.path.realpath(str(candidate))
    try:
        common = os.path.commonpath([base_real, real])
    except ValueError:
        raise ExportError("destination escapes the output directory: %s" % candidate)
    if common != base_real:
        raise ExportError("destination escapes the output directory: %s" % candidate)
    return real


# --- deterministic gzip ------------------------------------------------------


def _gzip_deterministic(data: bytes) -> bytes:
    """Deterministic gzip: ``mtime=0`` and an empty header filename field."""

    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="", fileobj=buffer, mode="wb", compresslevel=9, mtime=0
    ) as handle:
        handle.write(data)
    return buffer.getvalue()


def _gunzip(data: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as handle:
        return handle.read()


def _copy_file(source: str, dest: Path) -> None:
    with open(source, "rb") as src, open(str(dest), "wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)


# --- campaign loading / validation -------------------------------------------


def _load_campaign(path: Path):
    if not path.is_file():
        raise ExportError("campaign path is not a file: %s" % path)
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation failure
        raise ExportError("campaign is not valid JSON: %s" % exc)
    if not isinstance(data, dict):
        raise ExportError("campaign root is not a JSON object")
    return raw, data


def _validate_campaign(data: dict):
    """Require ``metadata.fatal_error`` to be explicitly present and null.

    The actual campaign root keys are ``experiment``, ``generated_utc``,
    ``metadata``, ``trials`` and ``aggregate`` -- there is NO root-level
    ``fatal_error`` key, and no fallback to one is permitted.  The fatal-error
    flag lives at ``metadata.fatal_error`` and must be explicitly present and
    JSON ``null``; a missing/non-dict ``metadata``, a missing ``fatal_error``,
    or a non-null ``fatal_error`` is a fatal, non-recoverable export error.
    """

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ExportError("campaign metadata is not a JSON object; refusing to export")
    if "fatal_error" not in metadata:
        raise ExportError(
            "campaign metadata.fatal_error is absent; refusing to export"
        )
    if metadata["fatal_error"] is not None:
        raise ExportError(
            "campaign metadata.fatal_error is not null; refusing to export"
        )
    trials = data.get("trials")
    if not isinstance(trials, list):
        raise ExportError("campaign root 'trials' is not a list")
    if len(trials) != EXPECTED_TRIAL_COUNT:
        raise ExportError(
            "campaign has %d trials; exactly %d are required"
            % (len(trials), EXPECTED_TRIAL_COUNT)
        )
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ExportError("trial %d is not a JSON object" % index)
    return trials


def _require_nonexistent_destination(output_dir: Path) -> None:
    if os.path.lexists(str(output_dir)):
        raise ExportError(
            "destination already exists; refusing to overwrite: %s" % output_dir
        )


# --- trial field extraction --------------------------------------------------


def _as_int(value: object) -> "int | None":
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _trial_pair_text(trial: dict) -> "str | None":
    """Trial pair, formatted ``seed:init_state_index`` when it is structured."""

    pair = trial.get("pair")
    if isinstance(pair, str) and pair:
        return pair
    if isinstance(pair, dict):
        seed = pair.get("seed", trial.get("seed"))
        index = pair.get("init_state_index", trial.get("init_state_index"))
        if seed is not None and index is not None:
            return "%s:%s" % (seed, index)
        return None
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return "%s:%s" % (pair[0], pair[1])
    seed = trial.get("seed")
    index = trial.get("init_state_index")
    if seed is not None and index is not None:
        return "%s:%s" % (seed, index)
    return None


def _trial_seed_index(trial: dict):
    """``(seed, init_state_index)`` preferring the trial fields, else the pair."""

    pair = trial.get("pair")
    if isinstance(pair, str) and ":" in pair:
        seed_text, _, state_text = pair.partition(":")
        try:
            return int(seed_text), int(state_text)
        except ValueError:
            pass
    seed = _as_int(trial.get("seed"))
    index = _as_int(trial.get("init_state_index"))
    return seed, index


def _csv_cell(value: object) -> str:
    """Null/nonfinite -> empty cell; bool -> explicit ``true``/``false``."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _summary_row(trial: dict) -> list:
    wine = trial.get("wine_statistics")
    if not isinstance(wine, dict):
        wine = {}
    jobs = trial.get("jobs")
    first_job = jobs[0] if isinstance(jobs, list) and jobs and isinstance(jobs[0], dict) else {}
    return [
        _csv_cell(trial.get("condition")),
        _csv_cell(_trial_pair_text(trial)),
        _csv_cell(first_job.get("steps")),
        _csv_cell(first_job.get("ended_reason")),
        _csv_cell(trial.get("strict_task_success")),
        _csv_cell(trial.get("task_success")),
        _csv_cell(wine.get("first_grasp_proxy_step")),
        _csv_cell(wine.get("first_lift_2cm_step")),
        _csv_cell(wine.get("max_lift_m")),
        _csv_cell(wine.get("first_goal_true_step")),
        _csv_cell(wine.get("sent_open_count")),
        _csv_cell(wine.get("sent_close_count")),
        _csv_cell(wine.get("open_command_width_increased_count")),
        _csv_cell(wine.get("width_min_m")),
        _csv_cell(wine.get("width_max_m")),
    ]


def _render_summary_csv(trials: list) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SUMMARY_COLUMNS)
    for trial in trials:
        writer.writerow(_summary_row(trial))
    return buffer.getvalue().encode("utf-8")


# --- telemetry readers (for the observational plot only) ---------------------


def _read_telemetry_samples(path: str) -> list:
    samples: list = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001 - a malformed line is skipped
                    continue
                if isinstance(record, dict):
                    samples.append(record)
    except OSError:
        return []
    return samples


def _snapshot_of(sample: object, key: str):
    if not isinstance(sample, dict):
        return None
    value = sample.get(key)
    return value if isinstance(value, dict) else None


def _sample_step(sample: object, index: int) -> int:
    if isinstance(sample, dict):
        step = sample.get("step")
        if isinstance(step, int) and not isinstance(step, bool):
            return step
    return index + 1


def _sent_last(sample: object):
    if not isinstance(sample, dict):
        return None
    sent = sample.get("sent_action")
    if not isinstance(sent, (list, tuple)) or not sent:
        return None
    return _finite(sent[-1])


def _gripper_width_after(sample: object):
    gripper = _snapshot_of(sample, "gripper_after")
    if not isinstance(gripper, dict):
        return None
    return _finite(gripper.get("width_m"))


def _object_entry(snapshot: object, object_id: str):
    if not isinstance(snapshot, dict):
        return None
    objects = snapshot.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(object_id)
    return entry if isinstance(entry, dict) else None


def _wine_z(snapshot: object):
    entry = _object_entry(snapshot, WINE_OBJECT_ID)
    if not isinstance(entry, dict):
        return None
    position = entry.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    return _finite(position[2])


def _wine_grasped(snapshot: object):
    entry = _object_entry(snapshot, WINE_OBJECT_ID)
    if not isinstance(entry, dict):
        return None
    value = entry.get("grasped")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = _finite(value)
    if number is None:
        return None
    return 1.0 if number != 0.0 else 0.0


def _predicate01(snapshot: object):
    if not isinstance(snapshot, dict):
        return None
    predicates = snapshot.get("predicates")
    if not isinstance(predicates, dict) or WINE_PREDICATE_KEY not in predicates:
        return None
    value = predicates.get(WINE_PREDICATE_KEY)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return None


def _extract_series(samples: list) -> dict:
    """Five observational series over the recorded control-step x values."""

    xs: list = []
    row1: list = []
    row2: list = []
    row3: list = []
    row4: list = []
    row5: list = []

    reference_z = None
    if samples:
        reference_z = _wine_z(_snapshot_of(samples[0], "before_snapshot"))

    for index, sample in enumerate(samples):
        xs.append(_sample_step(sample, index))

        sent = _sent_last(sample)
        row1.append(sent if sent is not None else math.nan)

        width = _gripper_width_after(sample)
        row2.append(width * 1000.0 if width is not None else math.nan)

        after_snapshot = _snapshot_of(sample, "after_snapshot")
        after_z = _wine_z(after_snapshot)
        if reference_z is not None and after_z is not None:
            row3.append((after_z - reference_z) * 100.0)
        else:
            row3.append(math.nan)

        grasped = _wine_grasped(after_snapshot)
        row4.append(grasped if grasped is not None else math.nan)

        predicate = _predicate01(after_snapshot)
        row5.append(predicate if predicate is not None else math.nan)

    return {
        "x": xs,
        "row1": row1,
        "row2": row2,
        "row3": row3,
        "row4": row4,
        "row5": row5,
        "n": len(samples),
    }


def _render_timeline(output_path: Path, entries: list):
    """Render ``timeline.png`` (2 columns x 5 rows) and return its sources."""

    series = {condition: None for condition in CONDITIONS}
    sources: list = []
    for entry in entries:
        if entry["pair_text"] != PREDETERMINED_MEDIA_PAIR:
            continue
        condition = entry["condition"]
        if condition not in series:
            continue
        path = entry.get("telemetry_path")
        if not isinstance(path, str) or not path or not os.path.isfile(path):
            continue
        samples = _read_telemetry_samples(path)
        series[condition] = _extract_series(samples)
        try:
            sources.append(
                {
                    "condition": condition,
                    "path": path,
                    "sha256": _sha256_path(path),
                    "bytes": os.path.getsize(path),
                }
            )
        except OSError:
            pass

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - reported as a clean export failure
        raise ExportError("matplotlib Agg backend unavailable: %s" % exc)

    fig, axes = plt.subplots(
        ROW_COUNT, len(CONDITIONS), figsize=FIG_SIZE_INCHES, dpi=FIG_DPI, squeeze=False
    )

    # --- shared physical comparison scales (IDENTICAL in BOTH columns) --------
    # x: 0 .. maximum RECORDED control step over BOTH selected series.
    recorded_steps: list = []
    for condition in CONDITIONS:
        data = series.get(condition)
        if data is not None:
            recorded_steps.extend(
                step
                for step in data["x"]
                if isinstance(step, int) and not isinstance(step, bool)
            )
    shared_xmax = max(recorded_steps) if recorded_steps else 1
    if shared_xmax <= 0:
        shared_xmax = 1

    # row3 measured lift (cm): ONE shared limit from ALL finite both-column values.
    lift_values: list = []
    for condition in CONDITIONS:
        data = series.get(condition)
        if data is not None:
            lift_values.extend(y for y in data["row3"] if _is_plot_number(y))
    if lift_values:
        lift_lower = min(-1.0, min(lift_values) - 1.0)
        lift_upper = max(2.0, max(lift_values) + 1.0)
    else:
        lift_lower, lift_upper = -1.0, 2.0

    for col, condition in enumerate(CONDITIONS):
        data = series.get(condition)
        if data is None:
            data = {
                "x": [],
                "row1": [],
                "row2": [],
                "row3": [],
                "row4": [],
                "row5": [],
                "n": 0,
            }
        xs = data["x"]

        # Row 1: sent_action[-1] gripper command; negative OPEN, positive CLOSE,
        # exactly zero HOLD (np.sign(0) == 0 -> no commanded gap increment;
        # zero is NEVER labelled CLOSE).
        axis = axes[0][col]
        ys = data["row1"]
        axis.plot(xs, ys, color="0.55", linewidth=0.8)
        open_x = [x for x, y in zip(xs, ys) if _is_plot_number(y) and y < 0]
        open_y = [y for y in ys if _is_plot_number(y) and y < 0]
        close_x = [x for x, y in zip(xs, ys) if _is_plot_number(y) and y > 0]
        close_y = [y for y in ys if _is_plot_number(y) and y > 0]
        hold_x = [x for x, y in zip(xs, ys) if _is_plot_number(y) and y == 0]
        hold_y = [y for y in ys if _is_plot_number(y) and y == 0]
        axis.plot(open_x, open_y, linestyle="none", marker="o", markersize=3,
                  color="tab:green", label="OPEN (command < 0)")
        axis.plot(close_x, close_y, linestyle="none", marker="s", markersize=3,
                  color="tab:red", label="CLOSE (command > 0)")
        axis.plot(hold_x, hold_y, linestyle="none", marker="D", markersize=3,
                  color="tab:gray", label="HOLD (command == 0)")
        if open_x or close_x or hold_x:
            axis.legend(fontsize=7, loc="best")
        title_prefix = COLUMN_TITLE_PREFIXES.get(condition, condition)
        seed_text, _, state_text = PREDETERMINED_MEDIA_PAIR.partition(":")
        axis.set_title(
            "%s\nseed=%s / state=%s / %d samples"
            % (title_prefix, seed_text, state_text, data["n"]),
            fontsize=10,
        )
        axis.set_ylabel(ROW_YLABELS[0], fontsize=8)
        axis.set_ylim(-1.15, 1.15)
        axis.grid(True, linewidth=0.3, alpha=0.5)

        # Rows 2-5: jaw width (mm), lift (cm), grasped 0/1, predicate 0/1.
        for row in (1, 2, 3, 4):
            axis = axes[row][col]
            values = data["row%d" % (row + 1)]
            axis.plot(xs, values, linewidth=0.9, marker=".", markersize=2, color="tab:blue")
            axis.set_ylabel(ROW_YLABELS[row], fontsize=8)
            axis.grid(True, linewidth=0.3, alpha=0.5)
            if row == 1:
                axis.set_ylim(0.0, 85.0)
            elif row == 2:
                axis.set_ylim(lift_lower, lift_upper)
            elif row in (3, 4):
                axis.set_ylim(-0.2, 1.2)
                axis.set_yticks([0, 1])

        axes[ROW_COUNT - 1][col].set_xlabel("control step", fontsize=8)

    # IDENTICAL x limits on every panel: shared 0 .. recorded maximum control step.
    for row in range(ROW_COUNT):
        for col in range(len(CONDITIONS)):
            axes[row][col].set_xlim(0, shared_xmax)

    fig.suptitle(
        "Wine diagnosis: commands, jaw motion and bottle placement",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(str(output_path), dpi=FIG_DPI, format="png")
    plt.close(fig)
    return sources


# --- manifest record builders -------------------------------------------------


def _copy_record(rel_path: str, source_path: str, published_path: Path):
    original_sha = _sha256_path(source_path)
    original_bytes = os.path.getsize(source_path)
    published_sha = _sha256_path(str(published_path))
    published_bytes = os.path.getsize(str(published_path))
    if original_sha != published_sha or original_bytes != published_bytes:
        raise ExportError("copy is not byte-identical to its source: %s" % rel_path)
    return {
        "path": rel_path,
        "type": "copy",
        "source_paths": [source_path],
        "original_sha256": original_sha,
        "original_bytes": original_bytes,
        "published_sha256": published_sha,
        "published_bytes": published_bytes,
        "gzip": None,
    }


def _generated_record(rel_path: str, published_path: Path, derived_from: list, note: str):
    return {
        "path": rel_path,
        "type": "generated",
        "derived_from": list(derived_from),
        "original_sha256": None,
        "original_bytes": None,
        "published_sha256": _sha256_path(str(published_path)),
        "published_bytes": os.path.getsize(str(published_path)),
        "gzip": None,
        "note": note,
    }


def _gzip_record(rel_path: str, source_path: str, gz_bytes: bytes, raw_bytes: bytes):
    decompressed = _gunzip(gz_bytes)
    if decompressed != raw_bytes:
        raise ExportError("gzip round-trip is not lossless for %s" % rel_path)
    raw_sha = _sha256_bytes(raw_bytes)
    if _sha256_bytes(decompressed) != raw_sha:
        raise ExportError("gzip decompressed SHA mismatch for %s" % rel_path)
    return {
        "path": rel_path,
        "type": "copy-gzip",
        "source_paths": [source_path],
        "original_sha256": raw_sha,
        "original_bytes": len(raw_bytes),
        "published_sha256": _sha256_bytes(gz_bytes),
        "published_bytes": len(gz_bytes),
        "gzip": {
            "mtime": 0,
            "header_filename": "",
            "decompressed_sha256": _sha256_bytes(decompressed),
            "decompressed_bytes": len(decompressed),
        },
        "note": (
            "deterministic gzip (mtime=0, empty header filename) of the source "
            "bytes; decompressed bytes equal the source"
        ),
    }


# --- per-trial export --------------------------------------------------------


def _process_trial(output_dir, base_real, entry, trial, campaign_path, campaign_sha, campaign_bytes):
    trials_root = output_dir / TRIALS_DIRNAME
    trial_dir = trials_root / entry["dirname"]
    _resolve_within(base_real, trial_dir)
    trial_dir.mkdir()

    file_records: list = []
    created: list = []
    missing: list = []

    # trial.json -- verbatim serialization of the source trial object.
    trial_json_bytes = (json.dumps(trial, indent=2, sort_keys=True) + "\n").encode("utf-8")
    trial_json_path = trial_dir / TRIAL_JSON_BASENAME
    trial_json_path.write_bytes(trial_json_bytes)
    rel = _bundle_rel(output_dir, trial_json_path)
    created.append(rel)
    file_records.append(
        {
            "path": rel,
            "type": "generated",
            "derived_from": [
                {
                    "path": str(campaign_path),
                    "sha256": campaign_sha,
                    "bytes": campaign_bytes,
                    "locator": "trials[%d]" % entry["index"],
                }
            ],
            "original_sha256": None,
            "original_bytes": None,
            "published_sha256": _sha256_bytes(trial_json_bytes),
            "published_bytes": len(trial_json_bytes),
            "gzip": None,
            "note": (
                "verbatim serialization of campaign trials[%d]; not a byte copy "
                "of any source file" % entry["index"]
            ),
        }
    )

    artifacts = trial.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    run_dir = artifacts.get("run_dir")
    run_dir_path = Path(run_dir) if isinstance(run_dir, str) and run_dir else None

    # Lossless deterministic gzip of the actual wine telemetry bytes.
    telemetry_path = artifacts.get("wine_telemetry")
    if isinstance(telemetry_path, str) and telemetry_path and os.path.isfile(telemetry_path):
        raw_telemetry = Path(telemetry_path).read_bytes()
        gz_bytes = _gzip_deterministic(raw_telemetry)
        gz_path = trial_dir / TELEMETRY_GZ_BASENAME
        gz_path.write_bytes(gz_bytes)
        rel = _bundle_rel(output_dir, gz_path)
        created.append(rel)
        file_records.append(_gzip_record(rel, telemetry_path, gz_bytes, raw_telemetry))
    else:
        missing.append(
            {
                "file": TELEMETRY_GZ_BASENAME,
                "source": telemetry_path,
                "reason": "wine telemetry source not present",
            }
        )

    # Byte-identical copies of the run_dir artifacts, when present.
    for name in RUN_DIR_COPY_FILES:
        source = run_dir_path / name if run_dir_path is not None else None
        if source is not None and source.is_file():
            destination = trial_dir / name
            _resolve_within(base_real, destination)
            _copy_file(str(source), destination)
            rel = _bundle_rel(output_dir, destination)
            created.append(rel)
            file_records.append(_copy_record(rel, str(source), destination))
        else:
            missing.append(
                {
                    "file": name,
                    "source": str(source) if source is not None else None,
                    "reason": "source file not present",
                }
            )

    # rollout.mp4 -- exactly the three selected (condition, pair) trials, and
    # no others:
    #   * pair 0:0 for BOTH conditions (native_goal9 and shared_goal8) -- the two
    #     videos predetermined BEFORE any results were seen;
    #   * shared_goal8 pair 1:1 -- one additional selected video, added because
    #     it demonstrates a distinct placement-stage failure identified by the
    #     primary agent.
    # Selection is by condition/pair only, is independent of success, and uses
    # the actual artifacts.rollout source with no fallback.  It is a media
    # choice only and implies no success claim and no new causal diagnosis; the
    # other three trials are still fully exported as raw evidence/statistics
    # (only their rollout.mp4 is unselected).
    rollout_selected = entry["pair_text"] == PREDETERMINED_MEDIA_PAIR or (
        entry["condition"] == ADDITIONAL_MEDIA_CONDITION
        and entry["pair_text"] == ADDITIONAL_MEDIA_PAIR
    )
    if rollout_selected:
        rollout_source = artifacts.get("rollout")
        if isinstance(rollout_source, str) and rollout_source and os.path.isfile(rollout_source):
            destination = trial_dir / ROLLOUT_BASENAME
            _resolve_within(base_real, destination)
            _copy_file(rollout_source, destination)
            rel = _bundle_rel(output_dir, destination)
            created.append(rel)
            file_records.append(_copy_record(rel, rollout_source, destination))
        else:
            missing.append(
                {
                    "file": ROLLOUT_BASENAME,
                    "source": rollout_source,
                    "reason": "selected trial rollout source not present",
                }
            )

    record = {
        "index": entry["index"],
        "trial_id": trial.get("trial_id"),
        "condition": entry["condition"],
        "pair": entry["pair_text"],
        "seed": entry["seed"],
        "init_state_index": entry["init_state_index"],
        "directory": _bundle_rel(output_dir, trial_dir),
        "artifacts_run_dir": run_dir,
        "rollout_selected": rollout_selected,
        "files": created,
        "missing": missing,
    }
    return record, file_records


# --- top-level export --------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Private deterministic six-trial wine-evidence exporter. Reads one "
            "finished campaign report plus its referenced evidence files and "
            "writes a hash-audited bundle into a new directory."
        )
    )
    parser.add_argument("--campaign", required=True, type=str,
                        help="path to the finished campaign report JSON")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="path of the bundle directory (must not exist)")
    return parser


def _export(campaign_path: Path, output_dir: Path) -> dict:
    # --- validate ALL inputs before creating any output ---
    raw, campaign = _load_campaign(campaign_path)
    trials = _validate_campaign(campaign)
    _require_nonexistent_destination(output_dir)

    campaign_sha = _sha256_bytes(raw)
    campaign_bytes = len(raw)

    entries: list = []
    seen_dirs: set = set()
    for index, trial in enumerate(trials):
        condition = trial.get("condition")
        if not isinstance(condition, str) or not condition:
            raise ExportError("trial %d has no string 'condition'" % index)
        if not _is_safe_component(condition):
            raise ExportError(
                "trial %d condition is not a safe path component: %r" % (index, condition)
            )
        seed, init_index = _trial_seed_index(trial)
        if seed is None or init_index is None:
            raise ExportError(
                "trial %d has no resolvable seed/init_state_index" % index
            )
        dirname = "%s_%d-%d" % (condition, seed, init_index)
        if dirname in seen_dirs:
            raise ExportError("duplicate trial directory requested: %s" % dirname)
        seen_dirs.add(dirname)
        artifacts = trial.get("artifacts")
        telemetry = artifacts.get("wine_telemetry") if isinstance(artifacts, dict) else None
        entries.append(
            {
                "index": index,
                "condition": condition,
                "pair_text": _trial_pair_text(trial),
                "seed": seed,
                "init_state_index": init_index,
                "dirname": dirname,
                "telemetry_path": telemetry if isinstance(telemetry, str) else None,
            }
        )

    for entry in entries:
        if not _is_safe_component(entry["dirname"]):
            raise ExportError("unsafe trial directory name: %r" % entry["dirname"])

    # --- create the (still nonexistent) destination ---
    output_dir.mkdir(parents=True, exist_ok=False)
    base_real = os.path.realpath(str(output_dir))
    trials_root = output_dir / TRIALS_DIRNAME
    trials_root.mkdir()

    file_records: list = []
    trial_records: list = []

    # Byte-for-byte campaign copy.
    campaign_copy = output_dir / CAMPAIGN_COPY_BASENAME
    campaign_copy.write_bytes(raw)
    rel = _bundle_rel(output_dir, campaign_copy)
    campaign_record = _copy_record(rel, str(campaign_path), campaign_copy)
    campaign_record["note"] = "byte-for-byte copy of the CLI campaign report"
    file_records.append(campaign_record)

    # Per trial.
    for entry, trial in zip(entries, trials):
        trial_record, records = _process_trial(
            output_dir, base_real, entry, trial, campaign_path, campaign_sha, campaign_bytes
        )
        trial_records.append(trial_record)
        file_records.extend(records)

    # summary.csv (generated from the campaign trial fields).
    summary_bytes = _render_summary_csv(trials)
    summary_path = output_dir / SUMMARY_BASENAME
    summary_path.write_bytes(summary_bytes)
    file_records.append(
        _generated_record(
            _bundle_rel(output_dir, summary_path),
            summary_path,
            [
                {
                    "path": str(campaign_path),
                    "sha256": campaign_sha,
                    "bytes": campaign_bytes,
                    "locator": "trials[*] (condition/pair/jobs/wine_statistics/task flags)",
                }
            ],
            "generated CSV; values are copied from the campaign trial fields and "
            "wine_statistics (never inferred); not a byte copy of any source file",
        )
    )

    # timeline.png (generated from the actual pair 0:0 telemetry).
    timeline_path = output_dir / PLOT_BASENAME
    plot_sources = _render_timeline(timeline_path, entries)
    file_records.append(
        _generated_record(
            _bundle_rel(output_dir, timeline_path),
            timeline_path,
            [
                {"path": str(campaign_path), "sha256": campaign_sha, "bytes": campaign_bytes}
            ]
            + plot_sources,
            "generated plot; observational series read from the actual pair 0:0 "
            "wine telemetry sources listed above; not a byte copy of any source",
        )
    )

    # Manifest (last), covering every exported file except itself.
    bundle = {
        "generated_utc": _utc_now(),
        "output_dir": str(output_dir),
        "campaign_copy": CAMPAIGN_COPY_BASENAME,
        "summary_csv": SUMMARY_BASENAME,
        "timeline_png": PLOT_BASENAME,
        "trials_dir": TRIALS_DIRNAME,
        "predicate_key": WINE_PREDICATE_KEY,
        "predicate_key_source": "catalog.goal_key(goal) == '|'.join(goal) (literal, no fallback)",
        "conditions": list(CONDITIONS),
        "predetermined_media_pair": PREDETERMINED_MEDIA_PAIR,
        "figure_size_inches": list(FIG_SIZE_INCHES),
        "figure_dpi": FIG_DPI,
        "row_count": ROW_COUNT,
        "file_count": len(file_records),
        "missing_count": sum(len(item["missing"]) for item in trial_records),
    }
    campaign_identity = {
        "path": str(campaign_path),
        "sha256": campaign_sha,
        "bytes": campaign_bytes,
        "experiment": campaign.get("experiment"),
        "generated_utc": campaign.get("generated_utc"),
        "fatal_error": (campaign.get("metadata") or {}).get("fatal_error"),
        "trial_count": len(trials),
        "expected_trial_count": EXPECTED_TRIAL_COUNT,
        "conditions": [entry["condition"] for entry in entries],
        "pairs": [entry["pair_text"] for entry in entries],
    }
    manifest = {
        "bundle": bundle,
        "campaign": campaign_identity,
        "trials": trial_records,
        "files": file_records,
    }
    manifest_path = output_dir / MANIFEST_BASENAME
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    # Sanity: every trial directory must still be unique and inside the bundle.
    dirs = [item["directory"] for item in trial_records]
    if len(set(dirs)) != len(dirs):
        raise ExportError("trial directories are not unique")

    return manifest


def main(argv: "list | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    campaign_path = Path(args.campaign)
    output_dir = Path(args.output_dir)
    try:
        manifest = _export(campaign_path, output_dir)
    except ExportError as exc:
        sys.stderr.write("wine-evidence export failed: %s\n" % exc)
        return 1
    sys.stdout.write(
        "wine-evidence bundle written to %s (%d files, %d trials)\n"
        % (output_dir, manifest["bundle"]["file_count"], manifest["campaign"]["trial_count"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
