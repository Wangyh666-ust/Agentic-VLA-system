#!/usr/bin/env python3
"""Provision and verify the LIBERO benchmark environment.

The script is executed inside WSL with the shared launcher interpreter::

    wsl -d Ubuntu -- /home/yhwang/fyp/vla/venv/bin/python \
        /mnt/d/FYP/First_Phase/libero_demo/setup_env.py

It only relies on the Python standard library; every heavy operation is
delegated to a subprocess running inside the freshly created virtual
environment ``<newroot>/venv``. The steps are:

1. Create ``<newroot>/venv`` via
   ``/usr/bin/python3 -m venv --without-pip <newroot>/venv`` so ``ensurepip``
   is never invoked (it is unreliable in this image), then (re)write
   ``shared_vla.pth`` so the shared VLA / lerobot packages -- including
   ``pip`` itself -- are importable. ``torch`` is never pip-installed, it
   comes from the shared tree.
2. Confirm ``python -m pip`` resolves through the ``shared_vla.pth`` entry
   above (the shared tree provides ``pip``; ``get-pip``/``ensurepip`` are
   never used) and that its install target is ``<newroot>/venv``, then install
   the LIBERO runtime dependencies with it.
3. Download the SmolVLA checkpoint (fixed revision, ``local_dir``) and the
   SmolVLM2 processor files (fixed revision, ``*.json``/``*.txt``/``*.model``).
   The LIBERO assets (``repo_type="dataset"``, no ``local_dir``) are pulled into
   the default Hugging Face cache.
4. Locate the installed ``libero`` package (``importlib.util.find_spec``), resolve
   the cached asset snapshot offline (``local_files_only=True``), symlink the
   package's ``assets`` directory to it and write
   ``<newroot>/libero_config/config.yaml`` (JSON, which is valid YAML).
5. Pin the base SmolVLM2 revision inside ``HF_HUB_CACHE`` so that the commit
   resolves offline without ever downloading a base weight.
6. Verify the stack in a subprocess: CUDA, the checkpoint feature shapes
   (state 8 / action 7), the >1GB weights and their SHA256, and that the LIBERO
   benchmark exposes the expected number of tasks (40).
7. Verify the base and LIBERO snapshots resolve with ``local_files_only=True``,
   record the lerobot source commit (read-only ``git rev-parse``/``status``) and
   the resolved assets snapshot revision, and write ``runtime_manifest.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Fixed filesystem layout
# --------------------------------------------------------------------------
NEWROOT = Path("/home/yhwang/fyp/libero_demo")
VENV = NEWROOT / "venv"
VENV_PYTHON = VENV / "bin" / "python"
MODELS_DIR = NEWROOT / "models"
SMOLVLA_DIR = MODELS_DIR / "smolvla_libero"

MANIFEST_PATH = Path("/mnt/d/FYP/First_Phase/libero_demo/runtime_manifest.json")

BASE_PYTHON = "/usr/bin/python3"
SHARED_SITE_PACKAGES = "/home/yhwang/fyp/vla/venv/lib/python3.12/site-packages"
SHARED_LEROBOT_SRC = "/home/yhwang/fyp/vla/lerobot/src"

# The lerobot Git checkout backing ``SHARED_LEROBOT_SRC``. The probe records the
# exact commit it sits on (read-only ``git rev-parse`` / ``git status``) so the
# manifest ties the verified runtime to a specific source revision. The
# repository is only ever queried, never modified.
LEROBOT_REPO = "/home/yhwang/fyp/vla/lerobot"

SMOLVLA_REPO = "HuggingFaceVLA/smolvla_libero"
SMOLVLA_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SMOLVLM2_REPO = "HuggingFaceTB/SmolVLM2-500M-Instruct"
SMOLVLM2_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SMOLVLM2_ALLOW_PATTERNS = ["*.json", "*.txt", "*.model"]
LIBERO_ASSETS_REPO = "lerobot/libero-assets"

# LIBERO configuration lives next to the venv. ``LIBERO_CONFIG_PATH`` (exported
# below and by run_service.sh) points at this directory; the file itself must be
# named ``config.yaml`` for the ``libero`` package to pick it up.
LIBERO_CONFIG_DIR = NEWROOT / "libero_config"
LIBERO_CONFIG_FILE = LIBERO_CONFIG_DIR / "config.yaml"
DATASETS_DIR = NEWROOT / "datasets"

# Sub-directories that the ``lerobot/libero-assets`` snapshot must provide;
# they are symlinked into the ``libero`` package as its ``assets`` folder.
LIBERO_ASSET_SUBDIRS = [
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "stable_hope_objects",
]

# Environment exported for every subprocess that imports LIBERO: the headless
# rendering backend and the directory holding the config written in step 4.
RUNTIME_ENV = {
    "LIBERO_CONFIG_PATH": str(LIBERO_CONFIG_DIR),
    "MUJOCO_GL": "egl",
}

PIP_PACKAGES = [
    "hf-libero==0.1.4",
    "imageio-ffmpeg==0.6.0",
    "scipy>=1.14,<2",
]

# Expected SmolVLA LIBERO checkpoint contract.
EXPECTED_STATE_DIM = 8
EXPECTED_ACTION_DIM = 7
EXPECTED_TASK_COUNT = 40
MIN_WEIGHTS_BYTES = 1_000_000_000

# The checkpoint was trained on exactly these four 10-task suites. They are the
# *only* suites that contribute to ``task_count``; any extra suite the installed
# LIBERO build happens to expose (e.g. ``libero_90``/``libero_130``) is ignored
# so that the fixed total stays at 40.
LIBERO_BENCHMARK_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# These variables are stripped from every subprocess environment so that a
# stale offline setting inherited from the launcher shell can never block a
# download. Offline verification uses ``local_files_only=True`` instead.
OFFLINE_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# --------------------------------------------------------------------------
# Embedded subprocess programs
# --------------------------------------------------------------------------
_SNAPSHOT_CODE = (
    "import json, sys\n"
    "from huggingface_hub import snapshot_download\n"
    "kwargs = json.loads(sys.argv[1])\n"
    "print(snapshot_download(**kwargs))\n"
)

# Step 4 (inside the venv): locate the installed ``libero`` package, resolve the
# already-downloaded asset snapshot strictly offline, prove it carries the
# expected LIBERO asset folders, symlink it in as the package's ``assets``
# directory and write the LIBERO ``config.yaml`` (JSON text is valid YAML).
#
# Failure handling mirrors the task contract: a *missing* snapshot or a
# ``benchmark_root/assets`` entry that is a real directory is a hard error --
# nothing is ever deleted, so an existing tree can never be lost. A symlink
# that points somewhere else is reported and the run aborts; a correct symlink
# is left untouched so repeated runs are idempotent.
_PROVISION_CODE = r'''
import importlib.util
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

spec = json.loads(sys.argv[1])
assets_repo = spec["assets_repo"]
config_path = Path(spec["config_path"])
datasets_dir = Path(spec["datasets_dir"])
required_subdirs = spec["required_subdirs"]

module_spec = importlib.util.find_spec("libero")
if module_spec is None or not module_spec.submodule_search_locations:
    print(
        "LIBERO_SETUP_FAIL: cannot locate the installed 'libero' package",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)
package_root = Path(list(module_spec.submodule_search_locations)[0]).resolve()
benchmark_root = package_root / "libero"

snapshot = Path(
    snapshot_download(assets_repo, repo_type="dataset", local_files_only=True)
).resolve()

errors = []
if not snapshot.is_dir():
    errors.append("assets snapshot is not a directory: %s" % snapshot)
for name in required_subdirs:
    if not (snapshot / name).is_dir():
        errors.append("assets snapshot lacks %s/ (%s)" % (name, snapshot))
if errors:
    print("LIBERO_SETUP_FAIL: " + "; ".join(errors), file=sys.stderr, flush=True)
    sys.exit(1)

bddl_files = benchmark_root / "bddl_files"
init_states = benchmark_root / "init_files"
for directory in (config_path.parent, datasets_dir, benchmark_root):
    directory.mkdir(parents=True, exist_ok=True)

assets_link = benchmark_root / "assets"
if assets_link.is_symlink():
    if assets_link.resolve() != snapshot:
        print(
            "LIBERO_SETUP_FAIL: %s is a symlink to %s (expected %s); "
            "refusing to repoint it"
            % (assets_link, assets_link.resolve(), snapshot),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
elif assets_link.exists():
    print(
        "LIBERO_SETUP_FAIL: %s exists and is not a symlink; refusing to delete it"
        % assets_link,
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)
else:
    assets_link.symlink_to(snapshot, target_is_directory=True)

if assets_link.resolve() != snapshot:
    print(
        "LIBERO_SETUP_FAIL: %s resolves to %s (expected %s)"
        % (assets_link, assets_link.resolve(), snapshot),
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

config = {
    "benchmark_root": str(benchmark_root),
    "bddl_files": str(bddl_files),
    "init_states": str(init_states),
    "datasets": str(datasets_dir),
    "assets": str(snapshot),
}
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

print(
    json.dumps(
        {
            "benchmark_root": str(benchmark_root),
            "assets_snapshot_path": str(snapshot),
            "libero_config_path": str(config_path),
        }
    ),
    flush=True,
)
'''

# Proves ``python -m pip`` is importable inside the venv and that the running
# interpreter resolves ``sys.prefix`` to the venv itself (so ``pip install``
# targets ``<newroot>/venv``). The venv is created with ``--without-pip``, so
# ``pip`` can only be coming from the shared ``shared_vla.pth`` tree.
_PIP_CHECK_CODE = r'''
import pathlib
import sys

import pip

expected_prefix = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
pip_path = pathlib.Path(pip.__file__).resolve()
print("[setup] python sys.prefix: %s" % prefix, flush=True)
print("[setup] pip resolved from: %s" % pip_path, flush=True)
if prefix != expected_prefix:
    print(
        "LIBERO_SETUP_FAIL: python sys.prefix=%s (expected %s)"
        % (prefix, expected_prefix),
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)
'''

_PROBE_CODE = r'''
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

import torch
import libero  # noqa: F401
import mujoco  # noqa: F401
import robosuite  # noqa: F401
import lerobot  # noqa: F401
from huggingface_hub import snapshot_download
from libero.libero import benchmark

spec = json.loads(sys.argv[1])
model_dir = Path(spec["model_dir"])
manifest_path = Path(spec["manifest_path"])
expected_state_dim = int(spec["expected_state_dim"])
expected_action_dim = int(spec["expected_action_dim"])
expected_task_count = int(spec["expected_task_count"])
min_weights_bytes = int(spec["min_weights_bytes"])
libero_config_path = Path(spec["libero_config_path"])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_snapshot(**kwargs):
    """Resolve a repository purely from the local cache."""
    return snapshot_download(local_files_only=True, **kwargs)


def _git(*args):
    """Run a read-only git query against the lerobot checkout.

    ``check=True`` makes a missing repository / bad command fail the probe;
    ``capture_output``/``text`` give clean, decoded stdout. The repository is
    never mutated -- only ``rev-parse`` and ``status`` are ever invoked.
    """
    return subprocess.run(
        ["git", "-C", spec["lerobot_repo"], *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _version(distribution):
    try:
        return md.version(distribution)
    except md.PackageNotFoundError:
        return None


# --- SmolVLA checkpoint contract ------------------------------------------
config_path = model_dir / "config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
state_dim = int(config["input_features"]["observation.state"]["shape"][-1])
action_dim = int(config["output_features"]["action"]["shape"][-1])

weights_path = model_dir / "model.safetensors"
weights_size = weights_path.stat().st_size
weights_sha256 = _sha256(weights_path)

# --- Offline snapshot verification ----------------------------------------
base_kwargs = {
    "repo_id": spec["base_repo"],
    "allow_patterns": spec["base_allow_patterns"],
}
if spec.get("base_revision"):
    base_kwargs["revision"] = spec["base_revision"]
base_snapshot = Path(_cached_snapshot(**base_kwargs))

# The base repo must also resolve with the *default* revision while fully
# offline, and any returned directory has to be the very same snapshot as the
# explicit commit above. This proves the ``refs/main`` pin written by
# ``write_base_cache_ref`` works, and that no model is substituted and no base
# weight is ever re-downloaded.
base_default_snapshot = Path(
    snapshot_download(
        spec["base_repo"],
        local_files_only=True,
        allow_patterns=spec["base_allow_patterns"],
    )
).resolve()

assets_kwargs = {"repo_id": spec["assets_repo"], "repo_type": "dataset"}
if spec.get("assets_revision"):
    assets_kwargs["revision"] = spec["assets_revision"]
assets_snapshot = Path(_cached_snapshot(**assets_kwargs))

# --- lerobot source provenance --------------------------------------------
# The commit the shared lerobot checkout is parked on, plus whether that
# checkout has uncommitted changes (any non-empty ``--porcelain`` output means
# the verified runtime was built from a dirty tree).
lerobot_source_commit = _git("rev-parse", "HEAD")
lerobot_source_dirty = bool(_git("status", "--porcelain"))

# --- LIBERO benchmark contract --------------------------------------------
# Only the four fixed 10-task suites contribute to ``task_count``. Every other
# suite the installed LIBERO build may expose is deliberately skipped so a
# task_count of exactly 40 is enforced.
benchmark_dict = benchmark.get_benchmark_dict()
suite_names = list(spec["benchmark_suites"])
task_counts = {
    name: int(benchmark_dict[name]().n_tasks)
    for name in suite_names
    if name in benchmark_dict
}
task_count = sum(task_counts.values())

cuda_available = bool(torch.cuda.is_available())

tracked_packages = [
    "hf-libero",
    "lerobot",
    "mujoco",
    "robosuite",
    "torch",
    "huggingface-hub",
    "transformers",
    "numpy",
    "scipy",
    "imageio-ffmpeg",
]
packages = {name: _version(name) for name in tracked_packages}

manifest = {
    "python": sys.version.split()[0],
    "cuda_available": cuda_available,
    "torch_version": getattr(torch, "__version__", None),
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "state_dim": state_dim,
    "action_dim": action_dim,
    "task_count": task_count,
    "task_counts": task_counts,
    "libero_config_path": str(libero_config_path),
    "libero_config_exists": libero_config_path.is_file(),
    "model_dir": str(model_dir),
    "model_revision": spec.get("model_revision"),
    "model_safetensors_bytes": weights_size,
    "model_safetensors_sha256": weights_sha256,
    "base_repo": spec["base_repo"],
    "base_revision": spec.get("base_revision"),
    "base_snapshot_path": str(base_snapshot),
    "assets_repo": spec["assets_repo"],
    "assets_snapshot_path": str(assets_snapshot),
    "assets_revision": assets_snapshot.name,
    "lerobot_source_commit": lerobot_source_commit,
    "lerobot_source_dirty": lerobot_source_dirty,
    "packages": packages,
}

manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
)

errors = []
if not cuda_available:
    errors.append("torch.cuda.is_available() is False")
if state_dim != expected_state_dim:
    errors.append("state_dim=%s (expected %s)" % (state_dim, expected_state_dim))
if action_dim != expected_action_dim:
    errors.append("action_dim=%s (expected %s)" % (action_dim, expected_action_dim))
if task_count != expected_task_count:
    errors.append(
        "task_count=%s (expected %s; per-suite %s)"
        % (task_count, expected_task_count, task_counts)
    )
missing_suites = [name for name in suite_names if name not in task_counts]
if missing_suites:
    errors.append(
        "benchmark suites missing: %s (available %s)"
        % (missing_suites, sorted(benchmark_dict))
    )
if not libero_config_path.is_file():
    errors.append("libero config missing: %s" % libero_config_path)
if weights_size <= min_weights_bytes:
    errors.append(
        "model.safetensors is %s bytes (must exceed %s)"
        % (weights_size, min_weights_bytes)
    )
for label, snapshot in (("base", base_snapshot), ("libero-assets", assets_snapshot)):
    if not snapshot.is_dir():
        errors.append("%s snapshot path missing: %s" % (label, snapshot))
if base_default_snapshot != base_snapshot.resolve():
    errors.append(
        "base default revision resolved to %s (expected %s)"
        % (base_default_snapshot, base_snapshot.resolve())
    )

if errors:
    print("LIBERO_SETUP_FAIL: " + "; ".join(errors), file=sys.stderr, flush=True)
    sys.exit(1)

print(
    "LIBERO_SETUP_PASS cuda=%s state_dim=%s action_dim=%s"
    % (
        "true" if cuda_available else "false",
        state_dim,
        action_dim,
    ),
    flush=True,
)
'''


def site_packages_dir() -> Path:
    """Return the site-packages directory of the freshly created venv."""
    for candidate in sorted(VENV.glob("lib/python*/site-packages")):
        return candidate
    version = "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
    return VENV / "lib" / version / "site-packages"


def run(cmd, *, env=None, cwd=None):
    """Invoke *cmd* (an argument list) and fail the whole script on error.

    Stale ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` values inherited from
    the launcher shell are dropped so the download steps can always reach the
    Hub (the verify step controls its own offline behaviour through
    ``local_files_only``). ``RUNTIME_ENV`` is then merged in for *every*
    subprocess, so each setup child always carries ``LIBERO_CONFIG_PATH`` and
    ``MUJOCO_GL=egl``; an explicit *env* is merged last and wins.
    """
    merged = dict(os.environ)
    for key in OFFLINE_KEYS:
        merged.pop(key, None)
    merged.update(RUNTIME_ENV)
    if env:
        merged.update(env)
    print("[setup] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=merged, cwd=cwd)


def create_venv():
    """Step 1: create the Python virtual environment.

    The venv is created with ``--without-pip`` so ``ensurepip`` is never
    invoked: that bootstrap is broken in this WSL image. The run time ``pip``
    instead comes from ``SHARED_SITE_PACKAGES`` through ``shared_vla.pth``
    (see :func:`ensure_pip`). An existing venv -- together with its
    interpreter and ``pyvenv.cfg`` -- is never deleted or rebuilt.
    """
    NEWROOT.mkdir(parents=True, exist_ok=True)
    if not VENV_PYTHON.exists():
        run([BASE_PYTHON, "-m", "venv", "--without-pip", str(VENV)])


def write_shared_pth():
    """Step 1: expose the shared VLA site-packages and lerobot sources.

    The ``shared_vla.pth`` file is rewritten on every run so a stale copy is
    never reused. Because the venv is created with ``--without-pip``, this
    ``.pth`` is also what makes ``pip`` importable inside the venv.
    """
    pth_file = site_packages_dir() / "shared_vla.pth"
    pth_file.parent.mkdir(parents=True, exist_ok=True)
    if pth_file.exists():
        pth_file.unlink()
    pth_file.write_text(
        "\n".join([SHARED_SITE_PACKAGES, SHARED_LEROBOT_SRC]) + "\n",
        encoding="utf-8",
    )
    print("[setup] wrote " + str(pth_file), flush=True)


def ensure_pip():
    """Step 1b: verify ``python -m pip`` resolves from the shared tree.

    Runs *after* :func:`write_shared_pth` (the shared ``site-packages`` entry,
    which ships ``pip``, is already on ``sys.path``) and *before*
    :func:`install_packages` so a broken pip bootstrap fails fast. Nothing is
    installed here and ``get-pip``/``ensurepip`` are never invoked: the check
    only proves that ``python -m pip`` imports inside the venv and that
    ``sys.prefix`` targets ``<newroot>/venv``.
    """
    run([str(VENV_PYTHON), "-c", _PIP_CHECK_CODE, str(VENV)])


def install_packages():
    """Step 2: install the runtime deps (torch comes from the shared tree)."""
    run([str(VENV_PYTHON), "-m", "pip", "install", *PIP_PACKAGES])


def snapshot_download(
    repo_id,
    revision=None,
    local_dir=None,
    *,
    allow_patterns=None,
    repo_type=None,
    max_workers=None,
):
    """Download a Hugging Face repository through the venv interpreter.

    ``local_dir`` is optional: when omitted the snapshot lands in the default
    Hugging Face cache (``HF_HUB_CACHE``), which is what the offline
    verification stage later resolves against.
    """
    kwargs = {"repo_id": repo_id}
    if revision is not None:
        kwargs["revision"] = revision
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    if repo_type is not None:
        kwargs["repo_type"] = repo_type
    if max_workers is not None:
        kwargs["max_workers"] = max_workers
    run([str(VENV_PYTHON), "-c", _SNAPSHOT_CODE, json.dumps(kwargs)])


def write_base_cache_ref() -> Path:
    """Step 4: pin the base repo revision inside the Hugging Face cache.

    The base snapshot is fetched with an explicit commit hash; writing the
    ``refs/main`` pointer makes that commit resolvable offline without ever
    downloading a single base weight.
    """
    from huggingface_hub import constants

    cache_root = Path(constants.HF_HUB_CACHE)
    ref_path = (
        cache_root
        / "models--HuggingFaceTB--SmolVLM2-500M-Instruct"
        / "refs"
        / "main"
    )
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(SMOLVLM2_REVISION, encoding="utf-8")
    print("[setup] pinned base revision at " + str(ref_path), flush=True)
    return ref_path


def download_models():
    """Step 3: fetch the SmolVLA checkpoint and the SmolVLM2 processor files."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        SMOLVLA_REPO,
        SMOLVLA_REVISION,
        SMOLVLA_DIR,
        max_workers=4,
    )
    snapshot_download(
        SMOLVLM2_REPO,
        SMOLVLM2_REVISION,
        allow_patterns=SMOLVLM2_ALLOW_PATTERNS,
        max_workers=4,
    )
    write_base_cache_ref()


def download_assets():
    """Step 3: fetch the LIBERO assets repository into the default cache.

    ``lerobot/libero-assets`` is a *dataset* repository; ``local_dir`` is left
    unset so the snapshot lands in ``HF_HUB_CACHE`` where
    :func:`provision_libero` can resolve it offline.
    """
    snapshot_download(LIBERO_ASSETS_REPO, repo_type="dataset")


def provision_libero():
    """Step 4: symlink the cached assets into ``libero`` and write the config.

    Runs inside the venv so it can import the installed ``libero`` package,
    resolve the just-downloaded snapshot with ``local_files_only=True`` and
    provision ``<newroot>/libero_config/config.yaml``.
    """
    spec = {
        "assets_repo": LIBERO_ASSETS_REPO,
        "config_path": str(LIBERO_CONFIG_FILE),
        "datasets_dir": str(DATASETS_DIR),
        "required_subdirs": LIBERO_ASSET_SUBDIRS,
    }
    run(
        [str(VENV_PYTHON), "-c", _PROVISION_CODE, json.dumps(spec)],
        env=RUNTIME_ENV,
    )


def verify_runtime():
    """Steps 6 and 7: probe the stack and write the runtime manifest."""
    spec = {
        "model_dir": str(SMOLVLA_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "model_revision": SMOLVLA_REVISION,
        "base_repo": SMOLVLM2_REPO,
        "base_revision": SMOLVLM2_REVISION,
        "base_allow_patterns": SMOLVLM2_ALLOW_PATTERNS,
        "assets_repo": LIBERO_ASSETS_REPO,
        "assets_revision": None,
        "lerobot_repo": LEROBOT_REPO,
        "expected_state_dim": EXPECTED_STATE_DIM,
        "expected_action_dim": EXPECTED_ACTION_DIM,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "benchmark_suites": list(LIBERO_BENCHMARK_SUITES),
        "min_weights_bytes": MIN_WEIGHTS_BYTES,
        "libero_config_path": str(LIBERO_CONFIG_FILE),
    }
    run(
        [str(VENV_PYTHON), "-c", _PROBE_CODE, json.dumps(spec)],
        env=RUNTIME_ENV,
    )


def main() -> None:
    print("[setup] launcher interpreter: " + sys.executable, flush=True)
    print("[setup] new root: " + str(NEWROOT), flush=True)
    create_venv()
    write_shared_pth()
    ensure_pip()
    install_packages()
    download_models()
    download_assets()
    provision_libero()
    verify_runtime()
    print("[setup] LIBERO environment setup complete", flush=True)
    print("[setup] runtime manifest: " + str(MANIFEST_PATH), flush=True)


if __name__ == "__main__":
    main()