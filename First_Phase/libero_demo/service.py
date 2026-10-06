#!/usr/bin/env python3
"""Resident HTTP service driving the public SmolVLA checkpoint on standard LIBERO.

One process, one GPU, one resident worker thread:

* the main thread only runs a stdlib ``ThreadingHTTPServer`` (JSON verbs below);
* a single daemon worker thread owns the CUDA policy *and* the MuJoCo/LIBERO
  environment.  Every ``LiberoEnv`` construction, ``reset``, ``step``,
  ``render`` and policy inference happens on that thread, so the EGL context
  and the simulation are never touched from two threads at once;
* HTTP threads only read lock-protected job state and push work onto a queue.

The point of the service is to make the *standard* LIBERO benchmark the source
of truth: 40 tasks come from ``libero.libero.benchmark.get_benchmark_dict()``,
the instruction and step budget are looked up from that catalogue (a client
cannot inject a free-form instruction), and the success flag is
``info["is_success"]`` returned by the LIBERO environment -- never an LLM
verdict.

Endpoints
---------
GET  /health                backend/model/revision/ready/worker_error
GET  /tasks                 the generated 40-task catalogue
POST /execute               {suite, task_id, seed, init_state_index} -> 202
GET  /status?job_id=...     job state machine
GET  /observe[?job_id=...]  most recent rendered RGB frame + job state

Run (see run_service.sh for the environment)::

    MUJOCO_GL=egl python -u service.py --port 8766
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import signal
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

# --- configuration -----------------------------------------------------------

SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Mirror of lerobot.envs.libero.TASK_SUITE_MAX_STEPS; used only as a fallback if
# that mapping cannot be imported.  The authoritative value is read from lerobot
# while building the task catalogue.
FALLBACK_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

DEFAULT_MODEL_PATH = "/home/yhwang/fyp/libero_demo/models/smolvla_libero"
DEFAULT_RUN_ROOT = "/home/yhwang/fyp/libero_demo/runs"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
MODEL_REVISION_DEFAULT = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"

OBS_WIDTH = 256
OBS_HEIGHT = 256
NUM_STEPS_WAIT = 10
CONTROL_MODE = "relative"
ACTION_DIM = 7
VIDEO_FPS = 20
PNG_EVERY = 5

ROBOT_NAME = "Franka Panda"
BACKEND_NAME = "smolvla"

_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    """Timestamped line to stderr (the launcher does not redirect it)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _LOG_LOCK:
        sys.stderr.write("[libero-service %s] %s\n" % (stamp, message))
        sys.stderr.flush()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _detect_model_revision(model_path: str) -> str:
    """Prefer a ``revision.txt`` sidecar, then env, then the audited constant."""
    try:
        sidecar = Path(model_path) / "revision.txt"
        if sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return text
    except OSError:
        pass
    env_value = os.environ.get("LIBERO_MODEL_REVISION")
    if env_value and env_value.strip():
        return env_value.strip()
    return MODEL_REVISION_DEFAULT


# --- small file helpers ------------------------------------------------------


def _save_png(path: Path, frame: np.ndarray) -> None:
    """Write one HWC uint8 RGB frame as PNG (Pillow is a lerobot core dep)."""
    from PIL import Image

    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(str(path))


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Write RGB frames to MP4 as H.264 (libx264) via imageio-ffmpeg.

    This is the single, fixed encoder path for the service: no lerobot helper
    and no OpenCV mp4v fallback.  ``imageio_ffmpeg.write_frames`` returns a
    generator that must be primed with ``send(None)`` before the first frame and
    closed in ``finally`` so ffmpeg flushes the moov atom.
    """
    import imageio_ffmpeg

    if not frames:
        raise ValueError("no frames to encode")
    height, width = np.asarray(frames[0]).shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("invalid frame size %dx%d" % (width, height))

    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    try:
        writer.send(None)  # prime the generator
        for frame in frames:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("frame must be HWC RGB, got shape %s" % (array.shape,))
            if array.shape[0] != height or array.shape[1] != width:
                raise ValueError(
                    "frame size %dx%d != %dx%d" % (array.shape[1], array.shape[0], width, height)
                )
            writer.send(np.ascontiguousarray(array, dtype=np.uint8))
    finally:
        writer.close()

    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError("video encoder produced an empty file: %s" % path)


def _batch_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Add a leading batch dimension to a single (un-vectorised) LiberoEnv obs.

    ``lerobot-eval`` feeds ``preprocess_observation`` the output of a
    *vectorised* env, so every leaf already carries a batch axis and
    ``LiberoProcessorStep`` can convert ``(B, 4)`` quaternions.  A single
    ``LiberoEnv`` returns unbatched leaves, so we restore that axis here before
    handing the observation to the official preprocessors.
    """

    def _b(value: Any) -> np.ndarray:
        return np.asarray(value)[None, ...]

    batched: dict[str, Any] = {}
    if isinstance(obs.get("pixels"), dict):
        batched["pixels"] = {key: _b(img) for key, img in obs["pixels"].items()}
    if isinstance(obs.get("robot_state"), dict):
        batched["robot_state"] = {
            group: {key: _b(value) for key, value in values.items()}
            for group, values in obs["robot_state"].items()
        }
    return batched


# --- job bookkeeping ---------------------------------------------------------


class BusyError(RuntimeError):
    """Raised when a second execution is submitted while one is still active."""

    def __init__(self, active_job_id: str) -> None:
        super().__init__(active_job_id)
        self.active_job_id = active_job_id


class JobRecord:
    """One execute request and its live/final state (guarded by the service lock)."""

    __slots__ = (
        "job_id",
        "suite",
        "task_id",
        "instruction",
        "max_steps",
        "seed",
        "init_state_index",
        "state",
        "steps",
        "success",
        "error",
        "wall_s",
        "ended_reason",
        "created_utc",
        "started_utc",
        "finished_utc",
        "run_dir",
        "result_path",
        "rollout_path",
        "latest_png",
        "first_png",
        "last_png",
        "_t0",
    )

    def __init__(
        self,
        job_id: str,
        suite: str,
        task_id: int,
        instruction: str,
        max_steps: int,
        seed: int,
        init_state_index: int,
    ) -> None:
        self.job_id = job_id
        self.suite = suite
        self.task_id = task_id
        self.instruction = instruction
        self.max_steps = max_steps
        self.seed = seed
        self.init_state_index = init_state_index
        self.state = "queued"
        self.steps = 0
        self.success: bool | None = None
        self.error: str | None = None
        self.wall_s: float | None = None
        self.ended_reason: str | None = None
        self.created_utc = _now_utc()
        self.started_utc: str | None = None
        self.finished_utc: str | None = None
        self.run_dir: str | None = None
        self.result_path: str | None = None
        self.rollout_path: str | None = None
        self.latest_png: str | None = None
        self.first_png: str | None = None
        self.last_png: str | None = None
        self._t0: float | None = None

    def wall_seconds(self) -> float | None:
        if self.wall_s is not None:
            return self.wall_s
        if self._t0 is not None and self.state == "running":
            return round(time.monotonic() - self._t0, 3)
        return None

    def public(self) -> dict[str, Any]:
        """JSON-safe snapshot; ``success`` stays null until the job finishes."""
        return {
            "job_id": self.job_id,
            "state": self.state,
            "steps": self.steps,
            "success": self.success,
            "suite": self.suite,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "init_state_index": self.init_state_index,
            "wall_s": self.wall_seconds(),
            "ended_reason": self.ended_reason,
            "error": self.error,
            "created_utc": self.created_utc,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "run_dir": self.run_dir,
            "result_path": self.result_path,
            "rollout_path": self.rollout_path,
            "latest_png": self.latest_png,
            "first_png": self.first_png,
            "last_png": self.last_png,
        }


# --- the service -------------------------------------------------------------


class LiberoService:
    """Owns the catalogue, the worker thread and all mutable job state."""

    def __init__(self, model_path: str, run_root: str) -> None:
        self.model_path = str(model_path)
        self.run_root = Path(run_root)
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._active_id: str | None = None
        self._ready = False
        self._worker_error: str | None = None
        self._catalog: dict[str, list[dict[str, Any]]] = {}
        self._model_revision = MODEL_REVISION_DEFAULT
        # Policy/pre/post are written and read only by the worker thread.
        self._policy: Any = None
        self._pre: Any = None
        self._post: Any = None
        self._n_action_steps: int | None = None
        self._dtype: str | None = None
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name="libero-worker", daemon=True)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._model_revision = _detect_model_revision(self.model_path)
        self._catalog = self.build_catalog()
        total = sum(len(tasks) for tasks in self._catalog.values())
        log("catalogue ready: %d suites, %d tasks" % (len(self._catalog), total))
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=60.0)

    @staticmethod
    def build_catalog() -> dict[str, list[dict[str, Any]]]:
        """Generate the task catalogue from the real LIBERO benchmark.

        Task names/instructions are read from ``get_task(i).name`` /
        ``.language``; step budgets come from lerobot's suite table.  Nothing is
        invented.
        """
        from libero.libero import benchmark

        try:
            from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
        except Exception:  # pragma: no cover - only if lerobot internals move
            TASK_SUITE_MAX_STEPS = FALLBACK_MAX_STEPS

        bench = benchmark.get_benchmark_dict()
        catalog: dict[str, list[dict[str, Any]]] = {}
        for suite_name in SUITE_NAMES:
            if suite_name not in bench:
                raise RuntimeError(
                    "LIBERO suite %r missing from get_benchmark_dict(); available=%s"
                    % (suite_name, sorted(bench.keys()))
                )
            suite = bench[suite_name]()
            task_count = len(suite.tasks)
            max_steps = int(TASK_SUITE_MAX_STEPS.get(suite_name, FALLBACK_MAX_STEPS.get(suite_name, 500)))
            entries: list[dict[str, Any]] = []
            for task_id in range(task_count):
                task = suite.get_task(task_id)
                entries.append(
                    {
                        "task_id": task_id,
                        "name": str(task.name),
                        "instruction": str(task.language),
                        "max_steps": max_steps,
                    }
                )
            catalog[suite_name] = entries
        return catalog

    # -- HTTP-facing reads ----------------------------------------------------

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = self._jobs.get(self._active_id) if self._active_id else None
            return {
                "ok": True,
                "backend": BACKEND_NAME,
                "robot": ROBOT_NAME,
                "model_revision": self._model_revision,
                "ready": bool(self._ready),
                "worker_error": self._worker_error,
                "model_path": self.model_path,
                "device": "cuda",
                "n_action_steps": self._n_action_steps,
                "dtype": self._dtype,
                "control_mode": CONTROL_MODE,
                "suites": list(self._catalog.keys()),
                "active_job": active.public() if active is not None else None,
            }

    def tasks(self) -> dict[str, Any]:
        with self._lock:
            suites: dict[str, list[dict[str, Any]]] = {}
            flat: list[dict[str, Any]] = []
            for suite_name in SUITE_NAMES:
                entries = self._catalog.get(suite_name, [])
                suites[suite_name] = [dict(entry) for entry in entries]
                for entry in entries:
                    flat.append({"suite": suite_name, **entry})
            return {
                "ok": True,
                "backend": BACKEND_NAME,
                "robot": ROBOT_NAME,
                "n_tasks": len(flat),
                "suites": suites,
                "tasks": flat,
            }

    def status(self, job_id: str | None) -> dict[str, Any] | None:
        with self._lock:
            if job_id is None:
                if not self._order:
                    return None
                job_id = self._order[-1]
            job = self._jobs.get(job_id)
            return job.public() if job is not None else None

    def observe(self, job_id: str | None) -> dict[str, Any] | None:
        with self._lock:
            if job_id is None:
                job = self._jobs.get(self._order[-1]) if self._order else None
            else:
                job = self._jobs.get(job_id)
            if job is None:
                return None
            snapshot = job.public()
        return {
            "ok": True,
            "job_id": snapshot["job_id"],
            "state": snapshot["state"],
            "steps": snapshot["steps"],
            "success": snapshot["success"],
            "image_path": snapshot["latest_png"],
            "latest_png": snapshot["latest_png"],
            "run_dir": snapshot["run_dir"],
            "job": snapshot,
        }

    # -- submission -----------------------------------------------------------

    def validate_request(self, suite: str, task_id: int) -> dict[str, Any]:
        with self._lock:
            if suite not in self._catalog:
                raise ValueError("unknown_suite")
            entries = self._catalog[suite]
            if not (0 <= task_id < len(entries)):
                raise ValueError("task_id_out_of_range")
            return dict(entries[task_id])

    def submit(self, suite: str, task_id: int, seed: int, init_state_index: int) -> JobRecord:
        entry = self.validate_request(suite, task_id)
        with self._lock:
            active = self._jobs.get(self._active_id) if self._active_id else None
            if active is not None and active.state in ("queued", "running"):
                raise BusyError(active.job_id)
            job = JobRecord(
                job_id=uuid.uuid4().hex,
                suite=suite,
                task_id=task_id,
                instruction=entry["instruction"],
                max_steps=entry["max_steps"],
                seed=seed,
                init_state_index=init_state_index,
            )
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._active_id = job.job_id
        self._queue.put(job.job_id)
        return job

    # -- worker thread --------------------------------------------------------

    def _worker_loop(self) -> None:
        try:
            self._load_policy()
        except BaseException as exc:  # noqa: BLE001 - report, never crash silently
            message = _format_exc(exc)
            with self._lock:
                self._worker_error = message
            log("policy load FAILED:\n%s" % message)
            return
        with self._lock:
            self._ready = True
        log("policy ready (n_action_steps=%s)" % self._n_action_steps)

        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:
                continue
            try:
                self._execute_job(job)
            except BaseException as exc:  # noqa: BLE001 - last-resort guard
                message = _format_exc(exc)
                log("job %s crashed:\n%s" % (job.job_id, message))
                with self._lock:
                    job.state = "error"
                    job.error = message
                    job.ended_reason = "error"
                    job.finished_utc = _now_utc()
            finally:
                with self._lock:
                    if self._active_id == job.job_id:
                        self._active_id = None

    def _load_policy(self) -> None:
        """Load the public SmolVLA LIBERO checkpoint on the worker thread."""
        import torch

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        log("loading config from %s" % self.model_path)
        cfg = SmolVLAConfig.from_pretrained(self.model_path)
        # Inference-time overrides only; all architecture fields and n_action_steps
        # stay exactly as saved in the checkpoint.
        cfg.device = "cuda"
        cfg.load_vlm_weights = False  # reuse the VLM already inside the checkpoint
        cfg.use_amp = True

        log("loading SmolVLA weights (n_action_steps=%s, dtype=%s)" % (cfg.n_action_steps, cfg.dtype))
        # strict=True: a missing/unexpected key must raise, not be silently skipped.
        policy = SmolVLAPolicy.from_pretrained(self.model_path, config=cfg, strict=True)
        policy = policy.to("cuda").eval()

        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=self.model_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self._policy = policy
        self._pre = pre
        self._post = post
        self._n_action_steps = int(cfg.n_action_steps)
        self._dtype = str(cfg.dtype) if cfg.dtype is not None else "float32"

    def _execute_job(self, job: JobRecord) -> None:
        """Run one full LIBERO episode; always persists result.json + events."""
        import torch

        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv
        from lerobot.envs.utils import preprocess_observation
        from lerobot.processor.env_processor import LiberoProcessorStep

        started = time.monotonic()
        run_dir = self.run_root / ("%s_%s" % (_utc_stamp(), uuid.uuid4().hex[:8]))
        run_dir.mkdir(parents=True, exist_ok=True)
        events_path = run_dir / "events.jsonl"
        result_path = run_dir / "result.json"
        rollout_path = run_dir / "rollout.mp4"
        first_png = run_dir / "first.png"
        last_png = run_dir / "last.png"
        latest_png = run_dir / "latest.png"

        with self._lock:
            job.state = "running"
            job.started_utc = _now_utc()
            job.run_dir = str(run_dir)
            job.result_path = str(result_path)
            job.rollout_path = str(rollout_path)
            job.first_png = str(first_png)
            job.last_png = str(last_png)
            job.latest_png = str(latest_png)
            job._t0 = started

        env = None
        frames: list[np.ndarray] = []
        steps_done = 0
        success = False
        ended_reason: str | None = None
        error_text: str | None = None
        video_error: str | None = None

        try:
            torch.manual_seed(job.seed)
            torch.cuda.manual_seed_all(job.seed)
            np.random.seed(job.seed % (2**32))
            random.seed(job.seed)

            suite = benchmark.get_benchmark_dict()[job.suite]()
            env = LiberoEnv(
                task_suite=suite,
                task_id=job.task_id,
                task_suite_name=job.suite,
                obs_type="pixels_agent_pos",
                observation_width=OBS_WIDTH,
                observation_height=OBS_HEIGHT,
                control_mode=CONTROL_MODE,
                init_states=True,
                episode_index=job.init_state_index,
                hard_reset=True,
                num_steps_wait=NUM_STEPS_WAIT,
            )
            # Fresh episode: clear policy chunk queue and processor state.
            self._policy.reset()
            self._pre.reset()
            self._post.reset()
            obs, _reset_info = env.reset(seed=job.seed)
            processor = LiberoProcessorStep()

            # Capture the true initial frame before any action is taken, so the
            # recording starts at t=0 and first.png is the reset state.
            initial_frame = np.asarray(env.render())
            frames.append(initial_frame)
            _save_png(first_png, initial_frame)
            _save_png(latest_png, initial_frame)

            with open(events_path, "w", encoding="utf-8") as events_file:
                for step in range(1, job.max_steps + 1):
                    if self._stop.is_set():
                        ended_reason = "shutdown"
                        error_text = "service shutdown requested mid-episode"
                        break

                    batch = preprocess_observation(_batch_observation(obs))
                    batch = processor._process_observation(batch)
                    batch["task"] = [job.instruction]
                    batch = self._pre(batch)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        action = self._policy.select_action(batch)
                    action = self._post(action)
                    act = np.asarray(action.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

                    if act.shape != (ACTION_DIM,):
                        raise RuntimeError(
                            "policy action shape %s != (%d,); refusing to truncate"
                            % (act.shape, ACTION_DIM)
                        )
                    if not np.all(np.isfinite(act)):
                        raise RuntimeError("policy produced a non-finite action: %s" % (act.tolist(),))

                    send = np.clip(act, env.action_space.low, env.action_space.high).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(send)

                    is_success = bool(info.get("is_success", False))
                    if is_success:
                        success = True
                    steps_done = step

                    frame = np.asarray(env.render())
                    frames.append(frame)
                    if step % PNG_EVERY == 0:
                        _save_png(latest_png, frame)

                    events_file.write(
                        json.dumps(
                            {
                                "step": step,
                                "action": [float(v) for v in send.tolist()],
                                "policy_action": [float(v) for v in act.tolist()],
                                "reward": float(reward),
                                "is_success": is_success,
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                                "wall_s": round(time.monotonic() - started, 4),
                            }
                        )
                        + "\n"
                    )
                    events_file.flush()

                    with self._lock:
                        job.steps = step

                    if success:
                        ended_reason = "success"
                        break
                    if terminated or truncated:
                        ended_reason = "env_done"
                        break
                else:
                    ended_reason = "max_steps"
        except BaseException as exc:  # noqa: BLE001 - persist the real exception
            error_text = _format_exc(exc)
            ended_reason = "error"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:  # noqa: BLE001
                    log("env.close failed: %s" % exc)

            if frames:
                try:
                    _save_png(last_png, frames[-1])
                    _save_png(latest_png, frames[-1])
                except Exception as exc:  # noqa: BLE001
                    log("final PNG save failed: %s" % exc)
                if len(frames) >= 2:
                    try:
                        _save_video(rollout_path, frames, VIDEO_FPS)
                    except Exception as exc:  # noqa: BLE001
                        video_error = _format_exc(exc)
                        log("video write failed: %s" % exc)

            wall_s = round(time.monotonic() - started, 3)
            finished = _now_utc()
            final_reason = "error" if error_text else (ended_reason or "unknown")
            final_state = "error" if error_text else "completed"

            result = {
                "ok": error_text is None,
                "job_id": job.job_id,
                "state": final_state,
                "suite": job.suite,
                "task_id": job.task_id,
                "instruction": job.instruction,
                "seed": job.seed,
                "init_state_index": job.init_state_index,
                "steps": steps_done,
                "max_steps": job.max_steps,
                "wall_s": wall_s,
                "success": bool(success),
                "ended_reason": final_reason,
                "benchmark_truth": "info['is_success'] from the LIBERO environment",
                "model_path": self.model_path,
                "model_revision": self._model_revision,
                "n_action_steps": self._n_action_steps,
                "dtype": self._dtype,
                "control_mode": CONTROL_MODE,
                "obs_type": "pixels_agent_pos",
                "observation_width": OBS_WIDTH,
                "observation_height": OBS_HEIGHT,
                "num_steps_wait": NUM_STEPS_WAIT,
                "video_fps": VIDEO_FPS,
                "autocast_dtype": "bfloat16",
                "error": error_text,
                "video_error": video_error,
                "created_utc": job.created_utc,
                "finished_utc": finished,
                "run_dir": str(run_dir),
                "files": {
                    "events": str(events_path),
                    "result": str(result_path),
                    "rollout": str(rollout_path) if rollout_path.exists() else None,
                    "first": str(first_png) if first_png.exists() else None,
                    "last": str(last_png) if last_png.exists() else None,
                    "latest": str(latest_png) if latest_png.exists() else None,
                },
            }
            try:
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                log("result.json write failed: %s" % exc)

            with self._lock:
                job.steps = steps_done
                job.success = bool(success)
                job.wall_s = wall_s
                job.finished_utc = finished
                job.ended_reason = final_reason
                job.error = error_text
                job.state = final_state
            log(
                "job %s %s | suite=%s task=%s steps=%d success=%s ended=%s"
                % (job.job_id, final_state, job.suite, job.task_id, steps_done, success, final_reason)
            )


# --- HTTP layer --------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    service: LiberoService


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LiberoService/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102 - stderr trace
        log("http %s - %s" % (self.address_string(), fmt % args))

    # -- helpers --------------------------------------------------------------

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("body must be a JSON object")
        return parsed

    # -- verbs ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        service = self.server.service
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        if path == "/health":
            return self._send_json(200, service.health())
        if path == "/tasks":
            return self._send_json(200, service.tasks())
        if path == "/status":
            job_id = (query.get("job_id") or [None])[0]
            payload = service.status(job_id)
            if payload is None:
                return self._send_json(404, {"ok": False, "reason": "unknown_job", "job_id": job_id})
            return self._send_json(200, {"ok": True, **payload})
        if path == "/observe":
            job_id = (query.get("job_id") or [None])[0]
            payload = service.observe(job_id)
            if payload is None:
                return self._send_json(404, {"ok": False, "reason": "no_jobs"})
            return self._send_json(200, payload)
        return self._send_json(404, {"ok": False, "reason": "not_found", "path": path})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        service = self.server.service
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path != "/execute":
            return self._send_json(404, {"ok": False, "reason": "not_found", "path": path})

        try:
            payload = self._read_json()
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"ok": False, "reason": "invalid_json", "detail": str(exc)})

        suite = payload.get("suite")
        if not isinstance(suite, str) or not suite:
            return self._send_json(
                400, {"ok": False, "reason": "invalid_suite", "detail": "suite must be a non-empty string"}
            )

        task_id = payload.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            return self._send_json(
                400, {"ok": False, "reason": "invalid_task_id", "detail": "task_id must be an integer"}
            )

        seed = payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            return self._send_json(
                400, {"ok": False, "reason": "invalid_seed", "detail": "seed must be an integer"}
            )

        init_state_index = payload.get("init_state_index", 0)
        if isinstance(init_state_index, bool) or not isinstance(init_state_index, int) or init_state_index < 0:
            return self._send_json(
                400,
                {
                    "ok": False,
                    "reason": "invalid_init_state_index",
                    "detail": "init_state_index must be a non-negative integer",
                },
            )

        try:
            service.validate_request(suite, task_id)
        except ValueError as exc:
            return self._send_json(400, {"ok": False, "reason": str(exc), "suite": suite, "task_id": task_id})

        if service.health()["worker_error"] is not None:
            return self._send_json(
                500, {"ok": False, "reason": "worker_error", "detail": service.health()["worker_error"]}
            )

        try:
            job = service.submit(suite, task_id, seed, init_state_index)
        except BusyError as exc:
            return self._send_json(
                409, {"ok": False, "reason": "busy", "active_job_id": exc.active_job_id}
            )
        except ValueError as exc:
            return self._send_json(400, {"ok": False, "reason": str(exc), "suite": suite, "task_id": task_id})

        return self._send_json(
            202,
            {
                "ok": True,
                "job_id": job.job_id,
                "state": "queued",
                "suite": job.suite,
                "task_id": job.task_id,
                "instruction": job.instruction,
            },
        )


# --- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resident SmolVLA-on-LIBERO HTTP service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args(argv)

    service = LiberoService(args.model, args.run_dir)
    service.start()

    httpd = _Server((args.host, args.port), _Handler)
    httpd.service = service

    def _handle_signal(signum: int, _frame: Any) -> None:
        log("received signal %s; shutting down" % signum)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log("listening on http://%s:%d (backend=%s, robot=%s)" % (args.host, args.port, BACKEND_NAME, ROBOT_NAME))
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        service.stop()
        log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
