#!/usr/bin/env python3
"""Unified subtask-assistance experimental service (fixed design).

This module hosts :class:`UnifiedAssistService`, a subclass of the existing
:class:`grasp_assist_service.GraspAssistService`, which layers a general
subtask-aware assistance pipeline (geometry classification, context
construction, strategy selection, generic preparation, and one bounded
backward-compatible calibrated grasp adapter) on top of the accepted wine
service.  It reuses the existing ``service._Server`` / ``service._Handler``
verbatim on its own experimental port and never touches the production
services on 8767/8081.

No environment, model or socket is created at import time.  Only stdlib +
numpy are imported here; ``gas``, ``wd``, ``service``, ``catalog``, ``pd``,
``pe``, ``collision_grasp``, ``subtask_context``, ``subtask_geometry`` and
``subtask_preparation`` are imported (none of which create a scene or open a
socket at import).
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

import catalog  # noqa: E402
import collision_grasp  # noqa: E402
import grasp_assist_service as gas  # noqa: E402
import placement_experiments as pe  # noqa: E402
import preparation_diagnostics as pd  # noqa: E402
import service  # noqa: E402
import subtask_context  # noqa: E402
import subtask_geometry  # noqa: E402
import subtask_preparation  # noqa: E402
import wine_diagnostics as wd  # noqa: E402

EXPERIMENT_NAME = "subtask_assist_service"

DEFAULT_ASSIST_PORT = 8782
DEFAULT_HOST = "127.0.0.1"

# Production ports are strictly forbidden.
PRODUCTION_PORTS = (8767, 8081)

PREPARE_MAX_ACTIONS = 200

SUBTASK_REPORT_FILENAME = "subtask_assist.json"
ACTION_SOURCES_FILENAME = "action_sources.jsonl"

MODE_ENABLED = "enabled"
MODE_DISABLED = "disabled"
VALID_MODES = (MODE_ENABLED, MODE_DISABLED)

KIND_GENERIC = "generic_preparation"
KIND_CALIBRATED = "calibrated_side"
KIND_VLA = "vla_only"

SOURCE_PREPARE = "prepare"
SOURCE_LOCAL_GRASP = "local_grasp"
SOURCE_VLA = "vla"
SOURCE_DISABLED = "disabled"


def _format_exc(exc: BaseException) -> str:
    import traceback

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _flat_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        return [float(component) for component in array.tolist()]
    except Exception:  # noqa: BLE001
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _load_profiles(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("calibration profiles must be a JSON object")
    return payload


class UnifiedAssistService(gas.GraspAssistService):
    """``GraspAssistService`` plus one unified subtask-assistance pipeline.

    The parent's wine-only local gripper assist is preserved (retained under
    the ``calibrated_side`` execution kind when the object binding matches);
    generic preparation, VLA-only execution, and disabled mode are added on
    top.  All env reads and actions occur inside the worker overrides.
    """

    def __init__(
        self,
        *args: Any,
        assist_mode: str = MODE_ENABLED,
        calibration_profiles: dict | None = None,
        grasp_module: Any = None,
        **kwargs: Any,
    ) -> None:
        if assist_mode not in VALID_MODES:
            raise ValueError("assist_mode must be one of %r" % (VALID_MODES,))
        super().__init__(
            *args,
            grasp_module=grasp_module if grasp_module is not None else collision_grasp,
            **kwargs,
        )
        self._subtask_assist_mode = assist_mode
        self._subtask_calibration_profiles: dict = (
            dict(calibration_profiles) if isinstance(calibration_profiles, dict) else {}
        )
        # Per-session assist state.
        self._subtask_home_orientation: Any = None
        # Fresh per-job state.
        self._subtask_kind: str | None = None
        self._subtask_context: Any = None
        self._subtask_selection: Any = None
        self._subtask_reading: Any = None
        self._subtask_helper: Any = None
        self._subtask_context_error: str | None = None
        self._subtask_observer_error: str | None = None
        self._subtask_prepare_confirmed = False
        self._subtask_prepare_local_grasp_actions = 0
        self._subtask_prepare_actions = 0
        self._subtask_vla_actions = 0
        self._subtask_step = 0
        self._subtask_queue_reset_done = False
        self._subtask_extra_goals = None
        self._subtask_prepare_execution = None
        self._grasp_assist_capability = None
        self._grasp_assist_job_id = None
        self._grasp_assist_step = 0
        self._grasp_assist_trigger_proposal = None
        # Gas-compatible legacy fields expected by the parent's orchestration.
        self._grasp_assist_local_actions = 0
        self._grasp_assist_vla_actions = 0

    # -- session creation -----------------------------------------------------

    def _do_create_session(
        self, record: Any, seed: int, init_state_index: int
    ) -> dict[str, Any]:
        created = super()._do_create_session(record, seed, init_state_index)
        if not isinstance(created, dict) or not created.get("ok"):
            return created
        env = self._env
        try:
            pose = pd.read_eef_pose(env)
            home_orientation = pose["orientation_matrix"]
            matrix = np.asarray(home_orientation, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError("home orientation matrix is not finite 3x3")
        except service.SceneError:
            self._close_env()
            return service._err(
                "home_orientation_unavailable",
                "end-effector home orientation is unreadable",
            )
        except BaseException as exc:  # noqa: BLE001
            self._close_env()
            return service._err("home_orientation_unavailable", _format_exc(exc))
        self._subtask_home_orientation = matrix
        self._reset_assist_session_state()
        return created

    def _reset_assist_session_state(self) -> None:
        self._subtask_kind = None
        self._subtask_context = None
        self._subtask_selection = None
        self._subtask_reading = None
        self._subtask_helper = None
        self._subtask_context_error = None
        self._subtask_observer_error = None
        self._subtask_prepare_confirmed = False
        self._subtask_prepare_local_grasp_actions = 0
        self._subtask_prepare_actions = 0
        self._subtask_vla_actions = 0
        self._subtask_step = 0
        self._subtask_queue_reset_done = False
        self._subtask_extra_goals = None
        self._subtask_prepare_execution = None
        self._grasp_assist_capability = None
        self._grasp_assist_job_id = None
        self._grasp_assist_step = 0
        self._grasp_assist_trigger_proposal = None
        self._grasp_assist_helper = None
        self._grasp_assist_source = SOURCE_VLA
        self._grasp_assist_observer_error = None
        self._grasp_assist_handoff_reset_done = False
        self._grasp_assist_return_reset_done = False
        self._grasp_assist_local_actions = 0
        self._grasp_assist_vla_actions = 0

    # -- capability execution -------------------------------------------------

    def _run_capability(
        self,
        session: Any,
        plan: Any,
        job: Any,
        capability_id: str,
    ) -> dict[str, Any]:
        env = self._env
        if env is None or self._env_session_id != session.session_id:
            self._clear_assist_job_state()
            return wd.WineDiagnosticService._run_capability(
                self, session, plan, job, capability_id
            )

        capability = catalog.CAPABILITIES[capability_id]

        # Fresh per-job state.
        self._subtask_kind = None
        self._subtask_context = None
        self._subtask_selection = None
        self._subtask_reading = None
        self._subtask_helper = None
        self._subtask_context_error = None
        self._subtask_observer_error = None
        self._subtask_prepare_confirmed = False
        self._subtask_prepare_local_grasp_actions = 0
        self._subtask_prepare_actions = 0
        self._subtask_vla_actions = 0
        self._subtask_step = 0
        self._subtask_queue_reset_done = False
        self._subtask_extra_goals = None
        self._subtask_prepare_execution = None
        self._grasp_assist_helper = None
        self._grasp_assist_source = SOURCE_VLA
        self._grasp_assist_observer_error = None
        self._grasp_assist_handoff_reset_done = False
        self._grasp_assist_return_reset_done = False
        self._grasp_assist_local_actions = 0
        self._grasp_assist_vla_actions = 0
        self._grasp_assist_capability = None
        self._grasp_assist_job_id = None
        self._grasp_assist_step = 0
        self._grasp_assist_trigger_proposal = None
        self._grasp_assist_capability = capability_id
        self._grasp_assist_job_id = job.job_id

        job.run_dir.mkdir(parents=True, exist_ok=True)

        # --- pre-run geometry / selection / execution-kind decision ---------
        try:
            decide = self._decide_execution_kind(
                env, capability, capability_id, plan
            )
        except BaseException as exc:  # noqa: BLE001
            self._subtask_context_error = _format_exc(exc)
            decide = {"kind": KIND_VLA, "reason": "context_error", "blocked": True}

        if decide.get("blocked"):
            self._terminalize_blocked(job, decide.get("reason", "blocked"))
            raise service.SceneError(
                "subtask_assist_blocked", decide.get("reason", "blocked")
            )

        self._subtask_kind = decide.get("kind", KIND_VLA)

        sources_path = job.run_dir / ACTION_SOURCES_FILENAME
        try:
            sources_file: Any = open(sources_path, "w", encoding="utf-8")
        except BaseException as exc:  # noqa: BLE001
            self._subtask_context_error = _format_exc(exc)
            self._terminalize_blocked(job, "audit_unavailable")
            raise service.SceneError("audit_unavailable", _format_exc(exc)) from exc

        original_step = env.step
        helper = self._subtask_helper
        kind = self._subtask_kind

        def wrapped_step(action: Any) -> Any:
            source = self._grasp_assist_source
            phase_before = getattr(helper, "phase", None) if helper is not None else None
            instruction = (
                self._wine_input_task if source == SOURCE_VLA else None
            )
            result = original_step(action)
            self._subtask_step += 1
            helper_summary = None
            if source == SOURCE_PREPARE and helper is not None:
                try:
                    reading = subtask_geometry.read_geometry(
                        env,
                        capability,
                        self._subtask_home_orientation,
                        self._subtask_calibration_profiles,
                        self._subtask_extra_goals or None,
                    )
                    self._subtask_reading = reading
                    helper.observe_after(reading)
                except BaseException as exc:  # noqa: BLE001
                    self._latch_observer_error(_format_exc(exc))
                try:
                    helper_summary = helper.summary()
                except BaseException as exc:  # noqa: BLE001
                    self._latch_observer_error(_format_exc(exc))
            elif source == SOURCE_LOCAL_GRASP and helper is not None:
                try:
                    reading = collision_grasp.read_geometry(env)
                    helper.observe_after(reading)
                except BaseException as exc:  # noqa: BLE001
                    self._latch_observer_error(_format_exc(exc))
                try:
                    helper_summary = helper.summary()
                except BaseException as exc:  # noqa: BLE001
                    self._latch_observer_error(_format_exc(exc))

            if source == SOURCE_PREPARE:
                self._subtask_prepare_actions += 1
            elif source == SOURCE_LOCAL_GRASP:
                self._subtask_prepare_local_grasp_actions += 1
            else:
                self._subtask_vla_actions += 1

            record = {
                "step": self._subtask_step,
                "source": source,
                "sent_action": _flat_list(action),
                "phase_before": phase_before,
                "phase_after": getattr(helper, "phase", None) if helper is not None else None,
                "helper_summary": helper_summary,
                "instruction": instruction if instruction is not None else None,
            }
            if sources_file is not None:
                try:
                    sources_file.write(json.dumps(record, default=_json_default) + "\n")
                    sources_file.flush()
                except BaseException as exc:  # noqa: BLE001
                    self._latch_observer_error(_format_exc(exc))
            return result

        env.step = wrapped_step
        try:
            result = wd.WineDiagnosticService._run_capability(
                self, session, plan, job, capability_id
            )
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

            report = self._compose_report(job, capability_id, kind, helper)
            try:
                pe._write_json_atomic(
                    job.run_dir / SUBTASK_REPORT_FILENAME, report
                )
            except Exception as exc:  # noqa: BLE001
                self._clear_assist_job_state()
                raise service.SceneError("subtask_assist_save_failed", _format_exc(exc))
            self._clear_assist_job_state()

        return result

    def _clear_assist_job_state(self) -> None:
        self._subtask_kind = None
        self._subtask_context = None
        self._subtask_selection = None
        self._subtask_reading = None
        self._subtask_helper = None
        self._subtask_context_error = None
        self._subtask_observer_error = None
        self._subtask_prepare_confirmed = False
        self._subtask_prepare_local_grasp_actions = 0
        self._subtask_prepare_actions = 0
        self._subtask_vla_actions = 0
        self._subtask_step = 0
        self._subtask_queue_reset_done = False
        self._subtask_extra_goals = None
        self._subtask_prepare_execution = None
        self._grasp_assist_helper = None
        self._grasp_assist_source = SOURCE_VLA
        self._grasp_assist_observer_error = None
        self._grasp_assist_handoff_reset_done = False
        self._grasp_assist_return_reset_done = False
        self._grasp_assist_local_actions = 0
        self._grasp_assist_vla_actions = 0
        self._grasp_assist_capability = None
        self._grasp_assist_job_id = None
        self._grasp_assist_step = 0
        self._grasp_assist_trigger_proposal = None

    def _compose_report(
        self, job: Any, capability_id: str, kind: str, helper: Any
    ) -> dict[str, Any]:
        ctx = self._subtask_context
        selection = self._subtask_selection
        helper_summary = None
        helper_summary_error = None
        if helper is not None:
            try:
                helper_summary = helper.summary()
            except BaseException as exc:  # noqa: BLE001
                helper_summary_error = _format_exc(exc)
        prepare_confirmed = False
        if (
            helper is not None
            and kind == KIND_GENERIC
            and getattr(helper, "phase", None) == "ready"
        ):
            prepare_confirmed = True
        self._subtask_prepare_confirmed = prepare_confirmed
        total = (
            self._subtask_prepare_actions
            + self._subtask_prepare_local_grasp_actions
            + self._subtask_vla_actions
        )
        action_sources_path = job.run_dir / ACTION_SOURCES_FILENAME
        return {
            "job_id": job.job_id,
            "capability_id": capability_id,
            "mode": self._subtask_assist_mode,
            "execution_kind": kind,
            "prepare_execution": getattr(self, "_subtask_prepare_execution", None),
            "context": ctx.to_dict() if ctx is not None else None,
            "selection": selection.to_dict() if selection is not None else None,
            "helper_summary": helper_summary,
            "helper_summary_error": helper_summary_error,
            "prepare_confirmed": prepare_confirmed,
            "confirmation_is_not_placement_success": True,
            "prepare_source_actions": self._subtask_prepare_actions,
            "local_grasp_source_actions": self._subtask_prepare_local_grasp_actions,
            "vla_source_actions": self._subtask_vla_actions,
            "actual_total_steps": total,
            "context_error": self._subtask_context_error,
            "observer_error": self._subtask_observer_error,
            "assisted": bool(
                self._subtask_prepare_actions > 0
                or self._subtask_prepare_local_grasp_actions > 0
            ),
            "action_sources_path": str(action_sources_path),
        }

    def _latch_observer_error(self, detail: str) -> None:
        if not self._subtask_observer_error:
            self._subtask_observer_error = detail
        if not self._grasp_assist_observer_error:
            self._grasp_assist_observer_error = detail

    # -- pre-run decision -----------------------------------------------------

    def _decide_execution_kind(
        self, env: Any, capability: dict, capability_id: str, plan: Any
    ) -> dict[str, Any]:
        if self._subtask_assist_mode == MODE_DISABLED:
            self._subtask_kind = KIND_VLA
            service.log(
                "subtask_assist vla_only/disabled for capability=%s" % capability_id
            )
            return {"kind": KIND_VLA, "reason": "disabled"}

        operation = subtask_context.operation_from_goals(capability["goals"])
        if operation != "pick_place":
            service.log(
                "subtask_assist vla_only for non-pick_place operation=%s" % operation
            )
            return {"kind": KIND_VLA, "reason": "non_pick_place"}

        extra_goals = self._collect_extra_goals(plan)
        self._subtask_extra_goals = extra_goals

        reading = subtask_geometry.read_geometry(
            env,
            capability,
            self._subtask_home_orientation,
            self._subtask_calibration_profiles,
            extra_goals or None,
        )
        self._subtask_reading = reading

        ctx = subtask_context.build_context(capability, reading)
        self._subtask_context = ctx
        selection = subtask_context.select_strategies(ctx)
        self._subtask_selection = selection

        if selection.status == "blocked":
            return {
                "kind": KIND_VLA,
                "reason": selection.reason,
                "blocked": True,
            }

        if selection.status == "vla_only":
            service.log(
                "subtask_assist vla_only reason=%s" % selection.reason
            )
            return {"kind": KIND_VLA, "reason": selection.reason}

        native_goals = capability.get("goals") or []
        reading_snapshot = reading.get("snapshot") or {}
        reading_predicates = reading_snapshot.get("predicates") or {}
        if native_goals and all(
            reading_predicates.get(catalog.goal_key(goal)) is True
            for goal in native_goals
        ):
            self._subtask_kind = KIND_VLA
            return {
                "kind": KIND_VLA,
                "reason": "native_goals_already_true_use_original_completion_gate",
            }

        # selected
        preview = self._preview_action(env, capability, reading, selection)
        if preview is None:
            return {
                "kind": KIND_VLA,
                "reason": "selection_preview_failed",
                "blocked": True,
            }

        if preview.get("kind") == KIND_VLA:
            return preview

        kind = preview.get("kind", KIND_VLA)
        self._subtask_kind = kind
        self._subtask_helper = preview.get("helper")
        self._subtask_prepare_execution = preview.get("prepare_execution")
        return {"kind": kind, "reason": preview.get("reason", selection.reason)}

    def _preview_action(
        self,
        env: Any,
        capability: dict,
        reading: dict,
        selection: Any,
    ) -> dict[str, Any] | None:
        ctx = self._subtask_context
        if (
            selection.grasp_strategy == "calibrated_side_v1"
            and ctx is not None
            and ctx.object_id == collision_grasp.WINE_OBJECT_ID
        ):
            try:
                base = self._build_calibrated_controller(reading)
            except BaseException as exc:  # noqa: BLE001
                self._subtask_context_error = _format_exc(exc)
                raise service.SceneError(
                    "subtask_assist_blocked", _format_exc(exc)
                ) from exc
            return {
                "kind": KIND_CALIBRATED,
                "reason": "calibrated_side",
                "helper": base,
                "prepare_execution": "delegated_to_calibrated_side_route",
            }

        if (
            selection.grasp_strategy == "calibrated_side_v1"
            and ctx is not None
            and ctx.object_id != collision_grasp.WINE_OBJECT_ID
        ):
            service.log(
                "subtask_assist grasp_adapter_unverified object_id=%s"
                % getattr(ctx, "object_id", None)
            )
            try:
                helper = self._build_generic_helper(ctx, reading)
            except BaseException as exc:  # noqa: BLE001
                self._subtask_context_error = _format_exc(exc)
                raise service.SceneError(
                    "subtask_assist_blocked", _format_exc(exc)
                ) from exc
            return {
                "kind": KIND_GENERIC,
                "reason": "grasp_adapter_unverified",
                "helper": helper,
                "prepare_execution": "generic_preparation_adapter_unverified",
            }

        if selection.prepare_strategy is not None:
            service.log(
                "subtask_assist grasp_strategy_unverified reason=%s"
                % selection.reason
            )
            try:
                helper = self._build_generic_helper(ctx, reading)
            except BaseException as exc:  # noqa: BLE001
                self._subtask_context_error = _format_exc(exc)
                raise service.SceneError(
                    "subtask_assist_blocked", _format_exc(exc)
                ) from exc
            return {
                "kind": KIND_GENERIC,
                "reason": "grasp_strategy_unverified",
                "helper": helper,
                "prepare_execution": "generic_preparation",
            }

        return {"kind": KIND_VLA, "reason": selection.reason}

    def _build_generic_helper(self, ctx: Any, reading: dict) -> Any:
        helper = subtask_preparation.PreparationController(
            ctx, reading, max_actions=PREPARE_MAX_ACTIONS
        )
        return helper

    def _build_calibrated_controller(self, reading: dict) -> Any:
        controller = collision_grasp.LocalGraspController()
        return controller

    def _collect_extra_goals(self, plan: Any) -> list:
        completed = list(getattr(plan, "completed_capability_ids", []) or [])
        extra: list = []
        for capability_id in completed:
            if capability_id not in catalog.CAPABILITIES:
                continue
            capability = catalog.CAPABILITIES[capability_id]
            for goal in capability.get("goals", []):
                extra.append(list(goal))
        return extra

    # -- terminalize blocked job ---------------------------------------------

    def _terminalize_blocked(self, job: Any, reason: str) -> None:
        error = service.SceneError("subtask_assist_blocked", reason)
        with self._lock:
            job.state = "error"
            job.ended_reason = "subtask_assist_blocked"
            job.error = str(error)
            job.success = False
            job.steps = 0
            job.total_steps = self._total_steps
            job.wall_s = 0.0
        job.run_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "job_id": job.job_id,
            "mode": self._subtask_assist_mode,
            "execution_kind": KIND_VLA,
            "blocked": True,
            "blocked_reason": reason,
            "context": self._subtask_context.to_dict()
            if self._subtask_context is not None
            else None,
            "selection": self._subtask_selection.to_dict()
            if self._subtask_selection is not None
            else None,
            "prepare_confirmed": False,
            "confirmation_is_not_placement_success": True,
            "prepare_source_actions": 0,
            "local_grasp_source_actions": 0,
            "vla_source_actions": 0,
            "actual_total_steps": 0,
            "context_error": self._subtask_context_error,
            "observer_error": self._subtask_observer_error,
            "assisted": False,
        }
        try:
            pe._write_json_atomic(job.run_dir / SUBTASK_REPORT_FILENAME, report)
        except Exception as exc:  # noqa: BLE001
            self._clear_assist_job_state()
            raise service.SceneError(
                "subtask_assist_save_failed", _format_exc(exc)
            ) from exc
        self._clear_assist_job_state()

    # -- action selection -----------------------------------------------------

    def _select_action(self, batch: Any) -> np.ndarray:
        latched = getattr(self, "_subtask_observer_error", None)
        if latched:
            raise service.SceneError("subtask_assist_unknown", latched)

        kind = self._subtask_kind

        if kind == KIND_GENERIC:
            helper = self._subtask_helper
            if helper is None:
                raise service.SceneError(
                    "subtask_assist_unknown", "generic helper missing"
                )
            phase = getattr(helper, "phase", None)
            if phase == "failed":
                reason = getattr(helper, "reason", None) or "subtask_prepare_failed"
                raise service.SceneError("subtask_prepare_failed", str(reason))
            if phase == "preparing":
                try:
                    reading = subtask_geometry.read_geometry(
                        self._env,
                        catalog.CAPABILITIES[self._grasp_assist_capability],
                        self._subtask_home_orientation,
                        self._subtask_calibration_profiles,
                        self._subtask_extra_goals or None,
                    )
                except BaseException as exc:  # noqa: BLE001
                    raise service.SceneError(
                        "subtask_assist_unknown", _format_exc(exc)
                    ) from exc
                self._subtask_reading = reading
                action = helper.next_action(reading)
                if action is None:
                    # Transitioned to ready without an action: fall through.
                    phase = getattr(helper, "phase", None)
                else:
                    self._clear_wine_evidence()
                    self._grasp_assist_source = SOURCE_PREPARE
                    self._grasp_assist_helper = helper
                    return np.asarray(action, dtype=np.float32).reshape(-1)
            if getattr(helper, "phase", None) == "ready":
                if not self._subtask_queue_reset_done:
                    self._reset_policy_queues()
                    self._subtask_queue_reset_done = True
                self._grasp_assist_source = SOURCE_VLA
                self._grasp_assist_helper = helper
                return wd.WineDiagnosticService._select_action(self, batch)
            # Phase unknown/unexpected: fail closed.
            raise service.SceneError(
                "subtask_assist_unknown",
                "unexpected generic phase %r" % (getattr(helper, "phase", None),),
            )

        if kind == KIND_CALIBRATED:
            self._grasp_assist_helper = self._subtask_helper
            return gas.GraspAssistService._select_action(self, batch)

        # KIND_VLA (or unset): parent VLA behaviour unchanged.
        self._grasp_assist_source = SOURCE_VLA
        self._grasp_assist_helper = None
        return wd.WineDiagnosticService._select_action(self, batch)


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified subtask-assist experimental service on its own port. "
            "Never touches production services on 8767/8081."
        )
    )
    parser.add_argument("--port", type=int, default=DEFAULT_ASSIST_PORT)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="absolute run root for this service's per-job artifacts",
    )
    parser.add_argument(
        "--calibration-profiles",
        type=str,
        default=None,
        help="optional absolute path to a calibration profiles JSON file",
    )
    parser.add_argument(
        "--disable-assist",
        action="store_true",
        help="run in disabled mode (vla_only, no helper, no extra inference)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.port in PRODUCTION_PORTS:
        parser.error(
            "port %d is a production port and is refused" % args.port
        )
    if not os.path.isabs(args.run_root):
        parser.error("--run-root must be an absolute path")
    if args.calibration_profiles is not None and not os.path.isabs(
        args.calibration_profiles
    ):
        parser.error("--calibration-profiles must be an absolute path")

    profiles = None
    if args.calibration_profiles is not None:
        try:
            profiles = _load_profiles(args.calibration_profiles)
        except BaseException as exc:  # noqa: BLE001
            parser.error("failed to read calibration profiles: %s" % (exc,))

    # Environment configuration happens only after parsing/validation.
    gas.configure_process_environment()

    mode = MODE_DISABLED if args.disable_assist else MODE_ENABLED
    assist = UnifiedAssistService(
        model_path=service.DEFAULT_MODEL_PATH,
        run_root=args.run_root,
        assist_mode=mode,
        calibration_profiles=profiles,
    )

    httpd = service._Server((args.host, args.port), service._Handler)
    httpd.service = assist

    stopped = {"value": False}
    serve_started = {"value": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        service.log("received signal %s; shutting down" % signum)
        if serve_started["value"]:
            threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    service.log(
        "listening on http://%s:%d (workflow=%s mode=%s)"
        % (args.host, args.port, EXPERIMENT_NAME, mode)
    )
    try:
        assist.start()
        serve_started["value"] = True
        httpd.serve_forever(poll_interval=0.5)
    finally:
        if not stopped["value"]:
            stopped["value"] = True
            if serve_started["value"]:
                try:
                    httpd.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            try:
                httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
            try:
                assist.stop()
            except Exception:  # noqa: BLE001
                pass
            service.log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())