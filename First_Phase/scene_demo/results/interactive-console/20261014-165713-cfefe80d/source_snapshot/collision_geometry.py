import math
import numpy as np

import service


def rotation_lower_extent(descriptor: object, current_orientation: object, target_orientation: object) -> float | None:
    """CPU upper bound on hand downward extent over any rotation <= required angle + 0.05 rad."""
    if not isinstance(descriptor, dict):
        return None
    if 'hand_sweep_radius_m' not in descriptor:
        return None
    hand_sweep = descriptor.get('hand_sweep_radius_m')
    try:
        sweep = float(hand_sweep)
    except Exception:
        return None
    if not math.isfinite(sweep) or sweep < 0.0:
        return None
    if 'hand_spheres' not in descriptor:
        return sweep
    spheres = descriptor.get('hand_spheres')
    if not isinstance(spheres, list) or len(spheres) == 0:
        return None
    try:
        current = np.asarray(current_orientation, dtype=np.float64)
        target = np.asarray(target_orientation, dtype=np.float64)
    except Exception:
        return None
    if current.shape != (3, 3) or target.shape != (3, 3):
        return None
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        return None
    trace = float(np.trace(target @ current.T))
    angle = math.acos(min(1.0, max(-1.0, (trace - 1.0) / 2.0)))
    theta = min(math.pi, angle + 0.05)
    result = 0.0
    for sphere in spheres:
        if not isinstance(sphere, dict):
            return None
        offset = sphere.get('offset_world')
        radius = sphere.get('radius')
        try:
            v = np.asarray(offset, dtype=np.float64)
            r = float(radius)
        except Exception:
            return None
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            return None
        if not math.isfinite(r) or r < 0.0:
            return None
        d = float(np.linalg.norm(v))
        if d <= 1.0e-12:
            z_min = 0.0
        else:
            beta = math.acos(min(1.0, max(-1.0, v[2] / d)))
            z_min = d * math.cos(min(math.pi, beta + theta))
        result = max(result, r - z_min)
    return max(0.0, float(result))


def describe(env: object, eef_position: object) -> dict | None:
    """Describe fixed three-segment path collision geometry.

    wine_robot_contacts are post-action samples, not all physics substeps.
    """
    try:
        inner = service._inner_env(env)
        model = inner.sim.model
        data = inner.sim.data

        hand = []
        wine = []
        for index in range(model.ngeom):
            if model.geom_contype[index] == 0 and model.geom_conaffinity[index] == 0:
                continue
            name = model.geom_id2name(index)
            if not name:
                continue
            if name.startswith("gripper0"):
                hand.append(index)
            elif name.startswith("wine_bottle_1"):
                wine.append(index)

        if not hand or not wine:
            return None

        eef = np.asarray(eef_position, dtype=float)
        if eef.shape != (3,) or not np.all(np.isfinite(eef)):
            return None

        bottle_top_z = None
        for index in wine:
            center = np.asarray(data.geom_xpos[index], dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                return None
            rbound = float(model.geom_rbound[index])
            if not math.isfinite(rbound) or rbound < 0.0:
                return None
            top = float(center[2] + rbound)
            bottle_top_z = top if bottle_top_z is None else max(bottle_top_z, top)

        hand_sweep = None
        hand_spheres = []
        for index in hand:
            center = np.asarray(data.geom_xpos[index], dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                return None
            rbound = float(model.geom_rbound[index])
            if not math.isfinite(rbound) or rbound < 0.0:
                return None
            offset_world = center - eef
            reach = float(np.linalg.norm(center - eef) + rbound)
            hand_sweep = reach if hand_sweep is None else max(hand_sweep, reach)
            hand_spheres.append({
                'offset_world': offset_world.tolist(),
                'radius': float(rbound),
            })

        ncon = int(data.ncon)
        if ncon < 0 or ncon > len(data.contact):
            return None

        contacts = []
        for i in range(ncon):
            contact = data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if not (0 <= g1 < model.ngeom and 0 <= g2 < model.ngeom):
                return None
            n1 = model.geom_id2name(g1)
            n2 = model.geom_id2name(g2)
            if not n1 or not n2:
                return None
            wine_first = n1.startswith("wine_bottle_1") or n2.startswith("wine_bottle_1")
            robot_other = (
                (n2.startswith("gripper0") or n2.startswith("robot0"))
                if n1.startswith("wine_bottle_1")
                else (n1.startswith("gripper0") or n1.startswith("robot0"))
            )
            if not (wine_first and robot_other):
                continue
            dist = float(contact.dist)
            pos = np.asarray(contact.pos, dtype=float)
            if not math.isfinite(dist) or pos.shape != (3,) or not np.all(np.isfinite(pos)):
                return None
            contacts.append({"pair": [str(n1), str(n2)], "dist": dist, "pos": [float(v) for v in pos]})

        return {
            "bottle_top_z_m": float(bottle_top_z),
            "hand_sweep_radius_m": float(hand_sweep),
            "hand_spheres": hand_spheres,
            "wine_robot_contacts": contacts,
        }
    except Exception:
        return None
