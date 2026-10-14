#!/usr/bin/env python3
"""Isolated, wine-only *local grasp assistance* service (experimental).

This module hosts :class:`GraspAssistService` -- a subclass of the actual
:class:`wine_diagnostics.WineDiagnosticService` (itself a subclass of
``placement_experiments.DiagnosticService`` and of the real
``scene_demo/service.py`` ``SceneService``) -- plus a tiny HTTP host that reuses
the existing ``service._Server`` / ``service._Handler`` verbatim on its own port.

The base worker, the single ``strict=True`` model load, the persistent
environment (one reset per session), the release-verified completion gate, the
wine-only grasp guard, ``events.jsonl`` / PNG / MP4, cancellation, the
capability-scoped policy queues, the base action clipping/counting loop and the
wine telemetry are inherited unchanged.  This module adds exactly three seams:

* :meth:`GraspAssistService._do_create_session` -- after the base session is
  created, the live OSC delta-controller contract is required via the existing
  ``preparation_diagnostics.controller_facts`` / ``controller_mismatch``; a
  mismatch closes *only* this service's environment and refuses the session (no
  extra scenario reset);
* :meth:`GraspAssistService._select_action` -- for exactly one capability
  (``wine_to_rack``) a fresh :class:`local_grasp.LocalGraspController` may
  *replace* one VLA proposal with an explicit auxiliary 7-D action and then
  continue driving the active ``above/descend/close/lift`` stages, returning to
  the VLA after ``confirmed``;
* :meth:`GraspAssistService._run_capability` -- wraps this job's worker-owned
  ``env.step`` to write one ``action_sources.jsonl`` row per *actual* step (the
  base still owns rendering, counters and the single physical step) and writes
  ``grasp_assist.json``.

Honest boundaries
=================

``assisted`` is ``True`` only when **actual** local auxiliary actions were
stepped; a controller ``confirmed`` phase is **never** promoted into a placement
success (``job.success`` remains the base predicate/probe verdict).  A physical
assist failure raises ``SceneError('local_grasp_failed', reason)`` and an
unreadable geometry during active control or trigger detection raises
``SceneError('local_grasp_unknown', detail)`` -- these are experimental
physical/probe limitations, never fabricated successes and never backend-crash
claims; the caller's ``job.error`` retains the honest detail.  No recovery,
retry, forced release, teleport or extra cancel action is ever injected.

Only stdlib + numpy are imported at module import time; ``torch`` is imported
lazily by the pinned base ``_select_action``, so ``--help`` and the GPU-free
unit tests never initialise CUDA, load a model, open a socket or build a scene.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import local_grasp  # noqa: E402
import placement_experiments as pe  # noqa: E402
import preparation_diagnostics as pd  # noqa: E402
import service  # noqa: E402
import wine_diagnostics as wd  # noqa: E402

EXPERIMENT_NAME = "grasp_assist_service"

# The single assisted capability (literal, never derived from a schedule).
WINE_CAPABILITY_ID = wd.WINE_CAPABILITY_ID  # "wine_to_rack"
ACTION_DIM = 7

# Fixed defaults of this diagnostic: the released completion gate and the wine
# grasp guard in ``enforce`` (the base worker behaviour is inherited verbatim).
ASSIST_COMPLETION_MODE = wd.COMPLETION_MODE  # "release_verified"
ASSIST_GRASP_GUARD_MODE = "enforce"

# The fixed pilot budget override.  This is a preregistered HARNESS design
# constant (a fixed pilot budget), not a Hermes-selected budget; the caller's
# own payload is never mutated.  Auxiliary local actions count toward it.
ASSIST_BUDGET_PER_SUBGOAL = 500

# Process environment, mirroring ``scene_demo/run_service.sh`` EXACTLY: headless
# EGL, the LIBERO config directory, the WSL CUDA/GL library path, the offline
# checkpoint/VLM flags and no inherited proxy configuration.  Nothing global is
# patched and the pre-existing 8767/8081 services are never touched.
MUJOCO_BACKEND = "egl"
LIBERO_CONFIG_PATH = "/home/yhwang/fyp/libero_demo/libero_config"
WSL_LIBRARY_PATH = "/usr/lib/wsl/lib"
PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
    "all_proxy",
    "ALL_PROXY",
)

# The assist host uses its own port (never the base 8767 or the legacy 8081).
DEFAULT_ASSIST_PORT = 8779

ACTION_SOURCES_FILENAME = "action_sources.jsonl"
ASSIST_REPORT_FILENAME = "grasp_assist.json"
ASSIST_RESULT_FILENAMES = ("diagnostic.json", "wine_diagnostic.json")


# --- small helpers -----------------------------------------------------------


def _format_exc(exc: BaseException) -> str:
    import traceback

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _flat_list(value: Any) -> list[float] | None:
    """A flat finite-agnostic float list of whatever length ``value`` has."""

    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        return [float(component) for component in array.tolist()]
    except Exception:  # noqa: BLE001 - an unconvertible action stays unknown
        return None


def _safe_summary(helper: Any) -> dict[str, Any] | None:
    """``helper.summary()`` when it is a dict; a failed read stays ``None``."""

    if helper is None:
        return None
    try:
        summary = helper.summary()
    except Exception:  # noqa: BLE001 - a failed summary is unknown evidence
        return None
    return summary if isinstance(summary, dict) else None


def _json_default(value: Any) -> Any:
    """Serialize numpy values numerically; anything else degrades to ``str``.

    A ``np.ndarray`` becomes a nested list of Python numbers (a matrix is never
    stringified) and a numpy scalar becomes its Python ``item()``; every other
    object falls back to ``str(value)`` exactly like the previous ``default=str``.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


# --- the assist service ------------------------------------------------------


class GraspAssistService(wd.WineDiagnosticService):
    """``WineDiagnosticService`` plus one wine-only local grasp assistant.

    Everything not named below is the accepted wine diagnostic behaviour
    verbatim.  The assistant is *additive*: it can only replace a VLA proposal
    for the one wine capability, and it never renders, counts, cancels or steps
    on its own -- the base step loop owns the single physical ``env.step``.
    """

    def __init__(
        self,
        *args: Any,
        completion_mode: str = ASSIST_COMPLETION_MODE,
        grasp_guard_mode: str = ASSIST_GRASP_GUARD_MODE,
        grasp_module: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            completion_mode=completion_mode,
            grasp_guard_mode=grasp_guard_mode,
            **kwargs,
        )
        # Backward-compatible grasp-module injection: consume the keyword (never
        # forward it to the base) and default to the existing local_grasp module.
        # The caller may inject a compatible controller module instead.
        self._grasp_module = grasp_module if grasp_module is not None else local_grasp
        # Per-request budget bookkeeping for the assist report (requested vs the
        # fixed effective pilot override).  Keyed by request id; never mutates a
        # caller payload.
        self._assist_budgets: dict[str, dict[str, Any]] = {}
        # Per-capability assistant state (worker thread only).
        self._grasp_assist_helper: Any = None
        self._grasp_assist_capability: str | None = None
        self._grasp_assist_job_id: str | None = None
        self._grasp_assist_source = "vla"
        self._grasp_assist_trigger_proposal: list[float] | None = None
        # Worker-owned fail-closed latch.  A post-real-action observer/audit
        # failure is recorded here (first error wins) and the NEXT selection
        # stops before any further inference or physical action.
        self._grasp_assist_observer_error: str | None = None
        self._grasp_assist_handoff_reset_done = False
        self._grasp_assist_return_reset_done = False
        self._grasp_assist_step = 0
        self._grasp_assist_local_actions = 0
        self._grasp_assist_vla_actions = 0

    # -- controller contract, before any movement (worker thread only) --------

    def _do_create_session(self, record: Any, seed: int, init_state_index: int) -> dict[str, Any]:
        """Create the session, then require the fixed OSC delta-controller contract.

        Runs on the worker thread (it is the ``create_session`` task).  A
        mismatch closes *only* this service's environment and refuses the
        session; there is no extra scenario reset and no extra action.
        """

        created = super()._do_create_session(record, seed, init_state_index)
        if not isinstance(created, dict) or not created.get("ok"):
            return created
        facts = pd.controller_facts(self._env)
        mismatch = pd.controller_mismatch(facts)
        if mismatch is not None:
            self._close_env()
            return service._err("controller_mismatch", mismatch)
        return created

    # -- plan submission: fixed pilot budget override -------------------------

    def submit_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Override the pilot budget for ``execute`` only; leave everything else.

        The caller's payload object is never mutated: a shallow copy is handed
        to the base implementation with ``budget_per_subgoal`` set to the fixed
        pilot budget.  All non-``execute`` decisions and every other base
        behaviour (validation, tombstones, conflict rules) are unchanged.
        """

        if not isinstance(payload, dict) or payload.get("decision") != "execute":
            return super().submit_plan(payload)
        requested = payload.get("budget_per_subgoal")
        effective = dict(payload)
        effective["budget_per_subgoal"] = ASSIST_BUDGET_PER_SUBGOAL
        result = super().submit_plan(effective)
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and isinstance(result, dict) and result.get("ok"):
            self._assist_budgets[request_id] = {
                "requested": requested,
                "effective": ASSIST_BUDGET_PER_SUBGOAL,
            }
        return result

    # -- action selection (worker thread only) --------------------------------

    def _clear_wine_evidence(self) -> None:
        """Clear stale VLA raw/input evidence so it cannot masquerade as auxiliary."""

        self._wine_raw_action = None
        self._wine_post_action = None
        self._wine_input_task = None
        self._wine_input_state = None

    def _assist_read_geometry(self) -> Any:
        """Read the wine geometry; an unreadable geometry is an honest unknown."""

        try:
            return getattr(self, "_grasp_module", local_grasp).read_geometry(self._env)
        except service.SceneError:
            raise
        except BaseException as exc:  # noqa: BLE001 - never blind-continue
            raise service.SceneError("local_grasp_unknown", _format_exc(exc)) from exc

    @staticmethod
    def _assist_failure_detail(helper: Any) -> str:
        reason = getattr(helper, "reason", None)
        return str(reason) if reason else "local_grasp_failed"

    def _assist_next(self, helper: Any, reading: Any, proposed: Any = None) -> Any:
        try:
            return helper.next_action(reading, proposed)
        except service.SceneError:
            raise
        except BaseException as exc:  # noqa: BLE001 - never blind-continue
            raise service.SceneError("local_grasp_unknown", _format_exc(exc)) from exc

    def _latch_observer_error(self, detail: str) -> None:
        """Latch the FIRST post-real-action observer/audit error (never overwrite)."""

        if not self._grasp_assist_observer_error:
            self._grasp_assist_observer_error = detail

    def _select_action(self, batch: Any) -> np.ndarray:
        """Select one action, letting the wine assistant drive its active stages.

        The base behaviour is preserved for every other capability and while the
        assistant is idle/bypassed/confirmed.  An auxiliary action replaces the
        VLA proposal exactly once (queues reset once at that handoff); after the
        controller confirms, the queues are reset once more and the VLA resumes
        from the freshly constructed current batch.

        Fail closed first: if a post-real-action observer/audit read failed on a
        previous step, no further inference or physical action is allowed.
        """

        latched = getattr(self, "_grasp_assist_observer_error", None)
        if latched:
            raise service.SceneError("local_grasp_unknown", latched)

        helper = self._grasp_assist_helper
        if helper is None:
            self._grasp_assist_source = "vla"
            return super()._select_action(batch)

        phase = helper.phase
        if phase == local_grasp.FAILED:
            raise service.SceneError("local_grasp_failed", self._assist_failure_detail(helper))

        if phase in local_grasp.STAGE_ORDER:
            # Active control: read the geometry and continue the auxiliary stage.
            reading = self._assist_read_geometry()
            action = self._assist_next(helper, reading)
            if action is None:
                if helper.phase == local_grasp.FAILED:
                    raise service.SceneError(
                        "local_grasp_failed", self._assist_failure_detail(helper)
                    )
                raise service.SceneError(
                    "local_grasp_unknown",
                    "local grasp returned no action in phase %r" % (phase,),
                )
            self._clear_wine_evidence()
            self._grasp_assist_source = "local_grasp"
            return np.asarray(action, dtype=np.float32).reshape(-1)

        if phase == local_grasp.IDLE:
            proposal = super()._select_action(batch)
            reading = self._assist_read_geometry()
            action = self._assist_next(helper, reading, proposal)
            if helper.phase == local_grasp.FAILED:
                raise service.SceneError(
                    "local_grasp_failed", self._assist_failure_detail(helper)
                )
            if action is None:
                # Not triggered (still idle) or bypassed: the VLA action is kept.
                self._grasp_assist_source = "vla"
                return proposal
            if not self._grasp_assist_handoff_reset_done:
                self._reset_policy_queues()
                self._grasp_assist_handoff_reset_done = True
            self._grasp_assist_trigger_proposal = _flat_list(proposal)
            self._clear_wine_evidence()
            self._grasp_assist_source = "local_grasp"
            return np.asarray(action, dtype=np.float32).reshape(-1)

        # CONFIRMED (or BYPASS): return control to the VLA.  After a confirmed
        # grasp the capability-scoped queues are reset exactly once, then the
        # base runs against the freshly constructed current batch.
        if phase == local_grasp.CONFIRMED and not self._grasp_assist_return_reset_done:
            self._reset_policy_queues()
            self._grasp_assist_return_reset_done = True
        self._grasp_assist_source = "vla"
        return super()._select_action(batch)

    # -- capability execution (worker thread only) ----------------------------

    def _patch_assist_result(
        self,
        path: Path,
        *,
        local_actions: int,
        vla_actions: int,
        sources_path: Path,
    ) -> None:
        """Mark a *newly generated* result file as assisted (never a prior file)."""

        if local_actions <= 0 or not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - an unreadable result is left untouched
            return
        if not isinstance(payload, dict):
            return
        payload["assisted"] = True
        payload["action_sources_path"] = str(sources_path)
        payload["actual_vla_actions"] = vla_actions
        payload["actual_local_actions"] = local_actions
        try:
            pe._write_json_atomic(path, payload)
        except Exception as exc:  # noqa: BLE001 - never break the run on a patch
            service.log("assist result patch failed: %s" % exc)

    def _run_capability(
        self,
        session: service.SessionRecord,
        plan: service.PlanRecord,
        job: service.JobRecord,
        capability_id: str,
    ) -> dict[str, Any]:
        """Wrap this capability's ``env.step`` with an action-source audit.

        One fresh :class:`local_grasp.LocalGraspController` is created for the
        wine capability only.  Every actual ``env.step`` is recorded exactly once
        (the original step is invoked exactly once); the post-action geometry and
        ``helper.observe_after`` run ONLY for an actual local auxiliary action.
        Rendering, counters and the physical step remain owned by the base.
        """

        env = self._env
        if env is None or self._env_session_id != session.session_id:
            return super()._run_capability(session, plan, job, capability_id)

        job.run_dir.mkdir(parents=True, exist_ok=True)
        helper = (
            getattr(self, "_grasp_module", local_grasp).LocalGraspController()
            if capability_id == WINE_CAPABILITY_ID
            else None
        )
        self._grasp_assist_helper = helper
        self._grasp_assist_capability = capability_id
        self._grasp_assist_job_id = job.job_id
        self._grasp_assist_source = "vla"
        self._grasp_assist_trigger_proposal = None
        self._grasp_assist_observer_error = None
        self._grasp_assist_handoff_reset_done = False
        self._grasp_assist_return_reset_done = False
        self._grasp_assist_step = 0
        self._grasp_assist_local_actions = 0
        self._grasp_assist_vla_actions = 0

        sources_path = job.run_dir / ACTION_SOURCES_FILENAME
        original_step = env.step
        try:
            sources_file: Any = open(sources_path, "w", encoding="utf-8")
        except BaseException as exc:  # noqa: BLE001 - fail closed, never step un-audited
            # The outer base only finalizes the plan *before* entering the base
            # capability loop, so this pre-action failure never reaches the base
            # worker's own job finalization.  This wrapper must therefore make its
            # job publicly terminal here: an ``error`` job with zero actions taken.
            error = service.SceneError("audit_unavailable", _format_exc(exc))
            with self._lock:
                job.state = "error"
                job.ended_reason = "audit_unavailable"
                job.error = str(error)
                job.success = False
                job.steps = 0
                job.total_steps = self._total_steps
                job.wall_s = 0.0
            raise error from exc

        def wrapped_step(action: Any) -> Any:
            source = self._grasp_assist_source
            phase_before = helper.phase if helper is not None else None
            proposed = self._grasp_assist_trigger_proposal if source == "local_grasp" else None
            raw_before = None if source == "local_grasp" else self._wine_raw_action
            # The real step is called exactly once, with the unchanged action, and
            # its real result is returned so the base can finish counting/rendering
            # the just-sent action.  Any observer/audit failure is latched instead
            # of raised here; the next selection stops before the next action.
            result = original_step(action)
            self._grasp_assist_step += 1
            geometry = None
            if source == "local_grasp" and helper is not None:
                try:
                    geometry = getattr(self, "_grasp_module", local_grasp).read_geometry(env)
                except BaseException as exc:  # noqa: BLE001 - latch, never fatal here
                    geometry = {"error": _format_exc(exc)}
                    self._latch_observer_error(_format_exc(exc))
                try:
                    helper.observe_after(geometry)
                except BaseException as exc:  # noqa: BLE001 - latch, never fatal here
                    self._latch_observer_error(_format_exc(exc))
            if source == "local_grasp":
                self._grasp_assist_local_actions += 1
            else:
                self._grasp_assist_vla_actions += 1
            record = {
                "step": self._grasp_assist_step,
                "source": source,
                "sent_action": _flat_list(action),
                "proposed_vla_action": _flat_list(proposed),
                "phase_before": phase_before,
                "phase_after": helper.phase if helper is not None else None,
                "summary": _safe_summary(helper),
                "geometry": geometry,
                "raw_action": raw_before,
            }
            if source == "local_grasp":
                # The replaced VLA proposal is recorded for the FIRST local action
                # only; later local actions must never repeat a stale proposal.
                self._grasp_assist_trigger_proposal = None
            if sources_file is not None:
                try:
                    sources_file.write(json.dumps(record, default=_json_default) + "\n")
                    sources_file.flush()
                except BaseException as exc:  # noqa: BLE001 - latch, never fatal here
                    self._latch_observer_error(_format_exc(exc))
            return result

        env.step = wrapped_step
        try:
            result = super()._run_capability(session, plan, job, capability_id)
        finally:
            try:
                env.step = original_step
            except Exception:  # noqa: BLE001
                pass
            if sources_file is not None:
                try:
                    sources_file.close()
                except Exception:  # noqa: BLE001
                    pass

        local_actions = self._grasp_assist_local_actions
        vla_actions = self._grasp_assist_vla_actions
        budget = self._assist_budgets.get(plan.request_id) or {}
        report = {
            "job_id": job.job_id,
            "capability_id": capability_id,
            "condition": self._diag_condition,
            "profile": self._diag_profile_name,
            "completion_mode": self.completion_mode,
            "grasp_guard_mode": self.grasp_guard_mode,
            "helper": helper is not None,
            "assisted": bool(local_actions > 0),
            "confirmation_is_not_placement_success": True,
            "requested_budget": budget.get("requested"),
            "effective_budget": int(plan.budget_per_subgoal),
            "actual_vla_actions": vla_actions,
            "actual_local_actions": local_actions,
            "steps": self._grasp_assist_step,
            "action_sources_path": str(sources_path),
            "phase": helper.phase if helper is not None else None,
            "reason": getattr(helper, "reason", None) if helper is not None else None,
            "helper_summary": _safe_summary(helper),
        }
        try:
            pe._write_json_atomic(job.run_dir / ASSIST_REPORT_FILENAME, report)
        except Exception as exc:  # noqa: BLE001
            service.log("grasp assist report write failed: %s" % exc)
        for name in ASSIST_RESULT_FILENAMES:
            self._patch_assist_result(
                job.run_dir / name,
                local_actions=local_actions,
                vla_actions=vla_actions,
                sources_path=sources_path,
            )

        self._grasp_assist_helper = None
        self._grasp_assist_capability = None
        self._grasp_assist_job_id = None
        return result


# --- process environment + CLI -----------------------------------------------


def configure_process_environment() -> None:
    """Set the process EGL / LIBERO / WSL / offline variables, clear proxies.

    Exactly mirrors the existing ``run_service.sh`` launcher; it only sets this
    process's own ``os.environ`` and never modifies another service.
    """

    os.environ["MUJOCO_GL"] = MUJOCO_BACKEND
    os.environ["LIBERO_CONFIG_PATH"] = LIBERO_CONFIG_PATH
    os.environ["LD_LIBRARY_PATH"] = WSL_LIBRARY_PATH
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated wine-only local grasp-assist diagnostic host: the accepted "
            "wine service plus one experimental local grasp assistant on its own port."
        )
    )
    parser.add_argument("--port", type=int, default=DEFAULT_ASSIST_PORT)
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="run root for this assist service's per-job artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_process_environment()

    assist = GraspAssistService(service.DEFAULT_MODEL_PATH, args.run_root)
    assist.start()

    httpd = service._Server((service.DEFAULT_HOST, args.port), service._Handler)
    httpd.service = assist

    def _handle_signal(signum: int, _frame: Any) -> None:
        service.log("received signal %s; shutting down" % signum)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    service.log("listening on http://%s:%d (workflow=%s)" % (service.DEFAULT_HOST, args.port, EXPERIMENT_NAME))
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        # Close ONLY this service's own httpd/worker -- never a global resource.
        httpd.server_close()
        assist.stop()
        service.log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
