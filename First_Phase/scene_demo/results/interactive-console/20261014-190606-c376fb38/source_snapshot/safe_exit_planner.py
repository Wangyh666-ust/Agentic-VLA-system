import struct

import numpy as np

from . import joint_home as jh
from . import safe_exit_contacts as sc


def plan_safe_exit(env, reference):
    import mujoco
    from . import preparation_diagnostics as pd

    robot = jh._single_robot(env)
    sim = robot.sim
    wrapper = sim.model
    rawmodel = wrapper._model
    livedata = sim.data

    site_id = wrapper.site_name2id(robot.controller.eef_name)
    HomeError = jh.HomeError

    base_qpos = np.array(livedata.qpos, copy=True)
    base_qvel = np.array(livedata.qvel, copy=True)
    base_time = float(livedata.time)
    live_qpos_bytes = livedata.qpos.tobytes()
    live_qvel_bytes = livedata.qvel.tobytes()
    live_time_bytes = struct.pack("<d", base_time)

    state = jh.read_robot_state(env)
    jh._reference_identity(reference, state)
    arm_qpos = np.array(state["arm_qpos"], dtype=float, copy=True)
    finger_qpos = np.array(state["finger_qpos"], dtype=float, copy=True)
    qpos_indexes = list(state["qpos_indexes"])
    qvel_indexes = list(state["qvel_indexes"])
    finger_qpos_indexes = list(state["finger_qpos_indexes"])
    joint_low = np.array(state["joint_limits_low"], dtype=float, copy=True)
    joint_high = np.array(state["joint_limits_high"], dtype=float, copy=True)

    homeq = np.array(reference["homeq"], dtype=float, copy=True)
    home_finger = np.array(reference["finger_home"], dtype=float, copy=True)
    if not (np.all(np.isfinite(homeq)) and np.all(np.isfinite(home_finger))):
        raise HomeError("non-finite home reference")

    scratch = mujoco.MjData(rawmodel)
    nv = rawmodel.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    result = {
        "ok": False,
        "needed": True,
        "reason": None,
        "initial_contacts": [],
        "candidate_reports": [],
        "home_nominal_clear": False,
        "live_unchanged": True,
        "waypoints": [],
        "direction": None,
        "distance": None,
    }

    def check_live():
        if livedata.qpos.tobytes() != live_qpos_bytes:
            raise HomeError("live qpos changed during planning")
        if livedata.qvel.tobytes() != live_qvel_bytes:
            raise HomeError("live qvel changed during planning")
        if struct.pack("<d", float(livedata.time)) != live_time_bytes:
            raise HomeError("live time changed during planning")

    def finish():
        result["live_unchanged"] = True
        return result

    def set_scratch(q, finger):
        scratch.qpos[:] = base_qpos
        scratch.qvel[:] = base_qvel
        scratch.time = base_time
        for i, idx in enumerate(qpos_indexes):
            scratch.qpos[idx] = q[i]
        for i, idx in enumerate(finger_qpos_indexes):
            scratch.qpos[idx] = finger[i]
        mujoco.mj_forward(rawmodel, scratch)

    startq = arm_qpos.copy()
    actual_finger = finger_qpos.copy()

    set_scratch(startq, actual_finger)
    pos0 = scratch.site_xpos[site_id].copy()
    R0 = scratch.site_xmat[site_id].reshape(3, 3).copy()

    def detect():
        return list(sc.contacts(env, data=scratch))

    def pairs(contacts):
        return sc.pair_map(contacts)

    def normal_of(c):
        return np.array(c["outward_normal"], dtype=float)

    def depth_of(c):
        return max(-float(c["dist"]), 0.0)

    def unit(v):
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return None
        return v / n

    def build_directions(contacts):
        out = []
        if contacts:
            acc = np.zeros(3)
            wsum = 0.0
            for c in contacts:
                acc = acc + normal_of(c) * depth_of(c)
                wsum += depth_of(c)
            if wsum > 0:
                u = unit(acc / wsum)
                if u is not None:
                    out.append(u)
        z = np.array([0.0, 0.0, 1.0])
        for extra in [z] + [normal_of(c) for c in contacts] + [
            normal_of(c) + z for c in contacts
        ]:
            u = unit(extra)
            if u is None:
                continue
            if any(np.allclose(u, p) for p in out):
                continue
            out.append(u)
        return out

    def solve(seed, target):
        q = np.array(seed, dtype=float, copy=True)
        for _ in range(50):
            set_scratch(q, actual_finger)
            cur = scratch.site_xpos[site_id].copy()
            Rcur = scratch.site_xmat[site_id].reshape(3, 3).copy()
            ep = target - cur
            eo = pd._matrix_to_rotvec(R0 @ Rcur.T)
            err = np.concatenate([ep, eo])
            if not np.all(np.isfinite(err)):
                raise HomeError("non-finite ik error")
            if np.linalg.norm(ep) <= 1e-4 and np.linalg.norm(eo) <= 0.005:
                set_scratch(q, actual_finger)
                if not np.all(np.isfinite(q)):
                    raise HomeError("non-finite ik solution")
                return q
            mujoco.mj_jacSite(rawmodel, scratch, jacp, jacr, site_id)
            J = np.vstack([jacp, jacr])[:, qvel_indexes]
            A = J @ J.T + (0.03 ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
            dq = np.clip(dq, -0.015, 0.015)
            q = q + dq
            q = np.clip(q, joint_low, joint_high)
        return None

    try:
        initial_contacts = list(sc.contacts(env))
        result["initial_contacts"] = initial_contacts

        if any(c["self"] for c in initial_contacts):
            result["ok"] = False
            result["reason"] = "initial_self_contact"
            return finish()

        if not initial_contacts:
            result["ok"] = True
            result["needed"] = False
            result["reason"] = "no_initial_contact"
            return finish()

        directions = build_directions(initial_contacts)

        for distance in (0.02, 0.04, 0.06):
            for direction in directions:
                cand = {
                    "distance": float(distance),
                    "direction": [float(x) for x in direction],
                    "reason": None,
                }
                result["candidate_reports"].append(cand)

                warm = startq.copy()
                prev_q = startq.copy()
                initial_map = pairs(initial_contacts)
                prev_map = dict(initial_map)
                cleared = set()
                joint_path = [startq.copy()]
                cart_solutions = []
                reason = None

                steps = max(1, int(round(distance / 0.001)))
                offsets = np.linspace(0.001, distance, steps)

                for off in offsets:
                    target = pos0 + direction * float(off)
                    sol = solve(warm, target)
                    if sol is None:
                        reason = "ik_failed"
                        break

                    wps = jh.joint_waypoints(
                        list(prev_q), list(sol), step_rad=0.003
                    )
                    rejected = False
                    for w in wps:
                        qn = np.array(w, dtype=float)
                        set_scratch(qn, actual_finger)
                        rows = detect()
                        if any(r["self"] for r in rows):
                            rejected = True
                            reason = "contact_progression"
                            break
                        cur_map = pairs(rows)
                        verdict = sc.contact_check(
                            initial_map, prev_map, cur_map, cleared
                        )
                        if verdict["ok"] is not True:
                            rejected = True
                            reason = "contact_progression"
                            break
                        prev_map = cur_map
                        cleared = set(verdict["newcleared"])
                        joint_path.append(qn)

                    if rejected:
                        break

                    prev_q = sol
                    warm = sol.copy()
                    cart_solutions.append(sol.copy())

                if reason is None:
                    set_scratch(prev_q, actual_finger)
                    if detect():
                        reason = "endpoint_not_clear"

                if reason is None:
                    tail = jh.joint_waypoints(
                        list(prev_q), list(homeq), step_rad=0.01
                    )
                    full_home = [list(prev_q)] + tail
                    for hq in full_home:
                        set_scratch(np.array(hq, dtype=float), home_finger)
                        if detect():
                            reason = "home_not_clear"
                            break

                if reason is None:
                    cand["reason"] = "accepted"
                    result["ok"] = True
                    result["needed"] = True
                    result["reason"] = "planned"
                    result["home_nominal_clear"] = True
                    result["direction"] = [float(x) for x in direction]
                    result["distance"] = float(distance)
                    result["waypoints"] = [
                        [float(x) for x in q] for q in cart_solutions
                    ]
                    return finish()

                cand["reason"] = reason

        result["reason"] = "no_safe_exit_candidate"
        return finish()
    finally:
        check_live()
