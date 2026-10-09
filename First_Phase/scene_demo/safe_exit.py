import copy
import numpy as np

from . import joint_home as jh
from . import safe_exit_contacts as sc
from . import safe_exit_planner as sp

read_contacts = sc.contacts
contact_check = sc.contact_check
plan_safe_exit = sp.plan_safe_exit

EXIT_CAP = 80
CONFIRM = 5


def _finger_hold(env, state):
    robot = jh._single_robot(env)
    model = robot.sim.model
    gripper = robot.gripper
    act_idx = list(robot._ref_joint_gripper_actuator_indexes)
    if len(act_idx) != 2:
        raise jh.HomeError("gripper actuator indexes must be exactly two")
    for i in act_idx:
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise jh.HomeError("gripper actuator index not integer")
        if int(i) < 0 or int(i) >= int(model.nu):
            raise jh.HomeError("gripper actuator index out of range")
    finger_qpos = np.asarray(state["finger_qpos"], dtype=float)
    finger_qi = list(state["finger_qpos_indexes"])
    if finger_qpos.shape != (2,):
        raise jh.HomeError("finger_qpos shape must be (2,)")
    if len(finger_qi) != 2:
        raise jh.HomeError("finger qpos indexes must be exactly two")
    norms = []
    for axis in range(2):
        actuatorID = int(act_idx[axis])
        if int(model.actuator_trntype[actuatorID]) != 0:
            raise jh.HomeError("actuator_trntype must be joint 0")
        jointID = int(model.actuator_trnid[actuatorID, 0])
        if jointID < 0 or jointID >= int(model.njnt):
            raise jh.HomeError("jointID out of range")
        if isinstance(finger_qi[axis], bool) or not isinstance(
                finger_qi[axis], (int, np.integer)):
            raise jh.HomeError("finger qpos index not integer")
        qadr = int(model.jnt_qposadr[jointID])
        if qadr != int(finger_qi[axis]):
            raise jh.HomeError("finger qpos index mismatch")
        if int(model.actuator_dyntype[actuatorID]) != 0:
            raise jh.HomeError("actuator dyntype must be 0")
        if int(model.actuator_gaintype[actuatorID]) != 0:
            raise jh.HomeError("actuator gaintype must be 0")
        if int(model.actuator_biastype[actuatorID]) != 1:
            raise jh.HomeError("actuator biastype must be 1")
        gear = np.asarray(model.actuator_gear[actuatorID], dtype=float)
        if not np.all(np.isfinite(gear)):
            raise jh.HomeError("actuator gear not finite")
        if gear.size < 2:
            raise jh.HomeError("actuator gear too short")
        if float(gear[0]) != 1.0:
            raise jh.HomeError("actuator gear first must be 1")
        for k in range(1, gear.size):
            if float(gear[k]) != 0.0:
                raise jh.HomeError("actuator gear rest must be 0")
        gainprm0 = float(model.actuator_gainprm[actuatorID, 0])
        biasprm1 = float(model.actuator_biasprm[actuatorID, 1])
        if not np.isfinite(gainprm0) or not np.isfinite(biasprm1):
            raise jh.HomeError("gainprm0/biasprm1 not finite")
        if not (gainprm0 > 0.0):
            raise jh.HomeError("gainprm0 must be positive")
        if biasprm1 != -gainprm0:
            raise jh.HomeError("biasprm1 must equal -gainprm0")
        cr = np.asarray(model.actuator_ctrlrange[actuatorID], dtype=float)
        if cr.shape != (2,) or not np.all(np.isfinite(cr)) or cr[0] >= cr[1]:
            raise jh.HomeError("ctrlrange invalid")
        actual = float(finger_qpos[axis])
        if not np.isfinite(actual):
            raise jh.HomeError("finger qpos not finite")
        if actual < float(cr[0]) or actual > float(cr[1]):
            raise jh.HomeError("finger qpos outside ctrlrange")
        mid = 0.5 * (float(cr[0]) + float(cr[1]))
        half = 0.5 * (float(cr[1]) - float(cr[0]))
        norms.append((actual - mid) / half)
    ca = np.asarray(gripper.current_action, dtype=float)
    if ca.shape != (2,) or not np.all(np.isfinite(ca)):
        raise jh.HomeError("gripper current_action shape must be (2,)")
    return np.asarray(norms, dtype=np.float32)


def _deepcopy_state(state):
    return copy.deepcopy(state)


def run_safe_exit(env, reference, guard, emit, stop_requested=None):
    result = {
        "ok": True,
        "ready": False,
        "actions": 0,
        "confirmed_samples": 0,
        "reason": None,
        "plan": None,
        "restored": True,
        "controller_facts": None,
        "errors": [],
    }

    if stop_requested is not None:
        try:
            if stop_requested():
                result["reason"] = "stop_requested"
                return result
        except Exception as e:
            result["ok"] = False
            result["errors"].append(str(e))
            return result

    try:
        gdict = guard()
        if not isinstance(gdict, dict):
            raise jh.HomeError("guard must return dict")
        if type(gdict.get("ok")) is not bool:
            result["ok"] = False
            result["reason"] = "protocol_error"
            return result
        if gdict["ok"] is False:
            result["reason"] = "guard_blocked"
            return result
    except Exception as e:
        result["ok"] = False
        result["errors"].append(str(e))
        return result

    try:
        plan = plan_safe_exit(env, reference)
    except Exception as e:
        result["ok"] = False
        result["errors"].append(str(e))
        return result
    result["plan"] = plan
    if not isinstance(plan, dict):
        result["ok"] = False
        result["reason"] = "protocol_error"
        return result
    if type(plan.get("ok")) is not bool:
        result["ok"] = False
        result["reason"] = "protocol_error"
        return result
    if plan["ok"] is False:
        result["reason"] = plan.get("reason") or "no_safe_exit_candidate"
        result["ok"] = True
        result["ready"] = False
        result["actions"] = 0
        return result
    if type(plan.get("needed")) is not bool:
        result["ok"] = False
        result["reason"] = "protocol_error"
        return result
    if plan["needed"] is False:
        result["ready"] = True
        result["reason"] = "already_clear"
        result["actions"] = 0
        return result

    waypoints = plan.get("waypoints")
    if not waypoints:
        result["reason"] = "bad_waypoints"
        result["ok"] = False
        return result
    for w in waypoints:
        arr = np.asarray(w, dtype=float)
        if arr.shape != (7,) or not np.all(np.isfinite(arr)):
            result["reason"] = "bad_waypoints"
            result["ok"] = False
            return result

    try:
        hold = _finger_hold(env, jh.read_robot_state(env))
    except Exception as e:
        result["ok"] = False
        result["errors"].append(str(e))
        return result

    robot = None
    original = None
    installed = False
    completed = 0

    try:
        try:
            robot = jh._single_robot(env)
            original = robot.controller
            adapter = jh.JointHomeAdapter(original, env, reference)
            robot.controller = adapter
            installed = True
            robot.gripper.current_action = np.asarray(hold, dtype=np.float32)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(str(e))
            return result

        try:
            initial_contacts = read_contacts(env)
            initial_map = sc.pair_map(plan["initial_contacts"])
        except Exception as e:
            result["ok"] = False
            result["errors"].append(str(e))
            return result
        prev_map = dict(initial_map)
        cleared = set()
        idx = 0
        last = len(waypoints) - 1
        confirmed = 0

        while completed < EXIT_CAP:
            if stop_requested is not None:
                try:
                    if stop_requested():
                        result["reason"] = "stop_requested"
                        break
                except Exception as e:
                    result["ok"] = False
                    result["errors"].append(str(e))
                    break

            try:
                g = guard()
                if not isinstance(g, dict):
                    raise jh.HomeError("guard must return dict")
                if type(g.get("ok")) is not bool:
                    result["ok"] = False
                    result["reason"] = "protocol_error"
                    result["errors"].append("guard ok not bool")
                    break
                if g["ok"] is False:
                    result["reason"] = "guard_blocked"
                    break
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break

            try:
                before_state = _deepcopy_state(jh.read_robot_state(env))
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break
            try:
                before_contacts = read_contacts(env)
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break

            try:
                cur_map_pre = sc.pair_map(before_contacts)
                verdict_pre = contact_check(
                    initial_map, prev_map, cur_map_pre, cleared)
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break
            if not isinstance(verdict_pre, dict) or type(
                    verdict_pre.get("ok")) is not bool:
                result["ok"] = False
                result["reason"] = "protocol_error"
                result["errors"].append("contact_check pre not strict dict ok")
                break
            if verdict_pre["ok"] is not True:
                result["reason"] = "contact_blocked"
                break
            if "newcleared" not in verdict_pre:
                result["ok"] = False
                result["reason"] = "protocol_error"
                result["errors"].append("missing newcleared")
                break
            prev_map = dict(cur_map_pre)
            cleared = set(verdict_pre["newcleared"])

            before_arm = np.asarray(before_state["arm_qpos"], dtype=float)
            while idx < last and float(np.max(
                    np.abs(before_arm - np.asarray(
                        waypoints[idx], dtype=float)))) <= 0.004:
                idx += 1
            target = np.asarray(waypoints[idx], dtype=float)
            adapter.select_target(target)

            sent = np.zeros(7, dtype=np.float32)
            try:
                env.step(sent)
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break
            completed += 1
            result["actions"] = completed
            step_row = completed

            try:
                after_state = _deepcopy_state(jh.read_robot_state(env))
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break
            try:
                after_contacts = read_contacts(env)
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break
            try:
                pg = guard()
                if not isinstance(pg, dict):
                    raise jh.HomeError("guard must return dict")
                if type(pg.get("ok")) is not bool:
                    result["ok"] = False
                    result["reason"] = "protocol_error"
                    result["errors"].append("guard ok not bool")
                    break
                pg_ok = pg["ok"]
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break

            try:
                cur_map_post = sc.pair_map(after_contacts)
                verdict = contact_check(
                    initial_map, prev_map, cur_map_post, cleared)
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break
            if not isinstance(verdict, dict) or type(
                    verdict.get("ok")) is not bool:
                result["ok"] = False
                result["reason"] = "protocol_error"
                result["errors"].append("contact_check post not strict dict ok")
                break
            if "newcleared" not in verdict:
                result["ok"] = False
                result["reason"] = "protocol_error"
                result["errors"].append("missing newcleared")
                break
            prev_map = dict(cur_map_post)
            cleared = set(verdict["newcleared"])
            verdict_ok = verdict["ok"]

            after_arm = np.asarray(after_state["arm_qpos"], dtype=float)
            err_after = float(np.max(np.abs(
                after_arm - np.asarray(waypoints[last], dtype=float))))

            streak = 0
            if (idx == last and err_after <= 0.004
                    and len(after_contacts) == 0 and pg_ok and verdict_ok):
                confirmed += 1
                streak = confirmed
            else:
                confirmed = 0
            result["confirmed_samples"] = confirmed

            row = {
                "step": step_row,
                "qtarget": np.asarray(target, dtype=float).tolist(),
                "sent_action": np.asarray(sent, dtype=float).tolist(),
                "before_robot_state": _deepcopy_state(before_state),
                "after_robot_state": _deepcopy_state(after_state),
                "before_contacts": before_contacts,
                "after_contacts": after_contacts,
                "guard": pg,
                "contact_check": verdict,
                "streak": streak,
            }
            try:
                emit(row)
            except Exception as e:
                result["ok"] = False
                result["errors"].append(str(e))
                break

            if not verdict_ok:
                result["reason"] = "contact_blocked"
                break

            if not pg_ok:
                result["reason"] = result["reason"] or "guard_blocked"
                break

            if confirmed >= CONFIRM:
                result["ready"] = True
                result["reason"] = "exit_ready"
                break

        if result["reason"] is None and completed >= EXIT_CAP:
            result["reason"] = "exit_budget"
        if result["reason"] is None and not result["ready"]:
            result["reason"] = "exit_budget"

        return result
    finally:
        if installed and robot is not None:
            try:
                robot.controller = original
                if robot.controller is not original:
                    result["ok"] = False
                    result["ready"] = False
                    result["restored"] = False
                    result["errors"].append("controller restore mismatch")
            except Exception as e:
                result["ok"] = False
                result["ready"] = False
                result["restored"] = False
                result["errors"].append(str(e))
            try:
                original.update(force=True)
                original.reset_goal()
            except Exception as e:
                result["ok"] = False
                result["ready"] = False
                result["restored"] = False
                result["errors"].append(str(e))
            try:
                from . import preparation_diagnostics as pd
                facts = pd.controller_facts(env)
                result["controller_facts"] = facts
                mismatch = pd.controller_mismatch(facts)
                if mismatch is not None:
                    result["ok"] = False
                    result["ready"] = False
                    result["restored"] = False
                    result["errors"].append(str(mismatch))
            except Exception as e:
                result["ok"] = False
                result["ready"] = False
                result["restored"] = False
                result["errors"].append(str(e))
        if result["errors"]:
            result["ready"] = False
