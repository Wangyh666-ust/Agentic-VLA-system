import math
import numpy as np
from . import joint_home as jh

PENETRATION_SLACK_M = 0.00005


def _finite_float(x, name):
    if isinstance(x, (bool, np.bool_)):
        raise jh.HomeError(f"{name} must not be boolean")
    try:
        v = float(x)
    except Exception as exc:
        raise jh.HomeError(f"{name} is not a finite number") from exc
    if not math.isfinite(v):
        raise jh.HomeError(f"{name} is not finite")
    return v


def _int_id(x, ngeom, name):
    if isinstance(x, (bool, np.bool_)):
        raise jh.HomeError(f"{name} must not be boolean")
    if not isinstance(x, (int, np.integer)):
        raise jh.HomeError(f"{name} must be an integer")
    i = int(x)
    if i < 0 or i >= ngeom:
        raise jh.HomeError(f"{name} out of range")
    return i


def _finite_array(x, size, name):
    arr = np.asarray(x)
    if arr.dtype.kind == "b":
        raise jh.HomeError(f"{name} must not be boolean")
    if arr.size != size:
        raise jh.HomeError(f"{name} must have size {size}")
    arr = arr.astype(float).reshape(size)
    if not np.all(np.isfinite(arr)):
        raise jh.HomeError(f"{name} must be finite")
    return arr


def _ngeom(model):
    if hasattr(model, "ngeom"):
        return int(model.ngeom)
    if hasattr(model, "_model") and hasattr(model._model, "ngeom"):
        return int(model._model.ngeom)
    raise jh.HomeError("model ngeom unavailable")


def contacts(env, data=None):
    robot = jh._single_robot(env)
    model = robot.sim.model
    live = robot.sim.data
    if data is None:
        data = live

    ngeom = _ngeom(model)

    names = []
    rm = getattr(robot, "robot_model", None)
    if rm is None:
        raise jh.HomeError("robot_model missing")
    names.extend(getattr(rm, "contact_geoms", []) or [])

    gr = getattr(robot, "gripper", None)
    if gr is None:
        raise jh.HomeError("gripper missing")
    names.extend(getattr(gr, "contact_geoms", []) or [])

    robot_ids = set()
    for name in names:
        if not isinstance(name, str):
            raise jh.HomeError("contact geom name must be str")
        gid = model.geom_name2id(name)
        robot_ids.add(_int_id(gid, ngeom, "geom_name2id"))

    contact = getattr(data, "contact", None)
    if contact is None:
        raise jh.HomeError("data.contact missing")

    ncon = getattr(data, "ncon", None)
    if isinstance(ncon, (bool, np.bool_)) or not isinstance(ncon, (int, np.integer)):
        raise jh.HomeError("ncon must be integer")
    ncon = int(ncon)
    if ncon < 0 or ncon > len(contact):
        raise jh.HomeError("ncon out of range")

    rows = []
    for i in range(ncon):
        c = contact[i]
        dist = _finite_float(getattr(c, "dist"), "dist")
        if dist > 0.0:
            continue

        g1 = _int_id(getattr(c, "geom1"), ngeom, "geom1")
        g2 = _int_id(getattr(c, "geom2"), ngeom, "geom2")

        pos = _finite_array(getattr(c, "pos"), 3, "pos")
        frame = _finite_array(getattr(c, "frame"), 9, "frame")

        normal_raw = frame[:3]
        n = float(np.linalg.norm(normal_raw))
        if n == 0.0 or not math.isfinite(n):
            raise jh.HomeError("contact normal is zero")
        normal = normal_raw / n

        if g1 not in robot_ids and g2 not in robot_ids:
            continue

        pair = sorted([g1, g2])
        g1_name = model.geom_id2name(g1)
        g2_name = model.geom_id2name(g2)
        if not isinstance(g1_name, str) or not isinstance(g2_name, str):
            raise jh.HomeError("geom_id2name must return str")

        self_contact = (g1 in robot_ids and g2 in robot_ids)
        if g2 in robot_ids:
            outward = normal
        else:
            outward = -normal

        rows.append({
            'geom1': g1,
            'geom2': g2,
            'pair': pair,
            'geom1_name': g1_name,
            'geom2_name': g2_name,
            'dist': dist,
            'pos': pos.tolist(),
            'normal': normal.tolist(),
            'outward_normal': outward.tolist(),
            'self': self_contact,
        })

    return rows


def _validate_contact_dict(d, name):
    if not isinstance(d, dict):
        raise jh.HomeError(f"{name} must be a dict")
    for pair, dist in d.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise jh.HomeError(f"{name} key must be a 2-tuple")
        _int_id(pair[0], 2**31, f"{name} pair id")
        _int_id(pair[1], 2**31, f"{name} pair id")
        _finite_float(dist, f"{name} distance")


def _normalize_contact_dict(d, name):
    _validate_contact_dict(d, name)
    out = {}
    for pair, dist in d.items():
        a = int(pair[0])
        b = int(pair[1])
        key = (a, b) if a <= b else (b, a)
        val = _finite_float(dist, f"{name} distance")
        if key not in out or val < out[key]:
            out[key] = val
    return out


def pair_map(rows):
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise jh.HomeError("row must be dict")
        pair_raw = row.get('pair')
        if not isinstance(pair_raw, (list, tuple)) or len(pair_raw) != 2:
            raise jh.HomeError("row pair must be 2 elements")
        a = _int_id(pair_raw[0], 2**31, "pair id")
        b = _int_id(pair_raw[1], 2**31, "pair id")
        key = (a, b) if a <= b else (b, a)
        dist = _finite_float(row.get('dist'), "dist")
        if key not in out or dist < out[key]:
            out[key] = dist
    return out


def contact_check(initial, previous, current, cleared):
    initial_n = _normalize_contact_dict(initial, "initial")
    previous_n = _normalize_contact_dict(previous, "previous")
    current_n = _normalize_contact_dict(current, "current")

    cleared_set = set()
    if cleared is None:
        cleared = []
    for pair in cleared:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise jh.HomeError("cleared entries must be 2-tuples")
        p0 = _int_id(pair[0], 2**31, "cleared pair id")
        p1 = _int_id(pair[1], 2**31, "cleared pair id")
        key = (p0, p1) if p0 <= p1 else (p1, p0)
        cleared_set.add(key)

    reasons = []
    for pair, dist in current_n.items():
        if pair not in initial_n:
            reasons.append(f"new_pair:{pair}")
            continue
        if pair in cleared_set:
            reasons.append(f"cleared_recontact:{pair}")
            continue

        init_dist = initial_n[pair]
        if dist < init_dist - PENETRATION_SLACK_M:
            reasons.append(f"deeper_than_initial:{pair}")

        if pair in previous_n and dist < previous_n[pair] - PENETRATION_SLACK_M:
            reasons.append(f"previous_penetration:{pair}")

    new_cleared = set(cleared_set)
    new_cleared.update(set(initial_n) - set(current_n))
    new_cleared = sorted(new_cleared)

    return {'ok': not reasons, 'reasons': reasons, 'newcleared': new_cleared}
