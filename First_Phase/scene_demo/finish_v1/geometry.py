import numpy as np
from scipy.spatial import cKDTree


def clouds(frames):
    out = []
    if not isinstance(frames, (list, tuple)):
        return np.zeros((0, 3), dtype=float)
    for frame in frames:
        try:
            depth_path = frame["depth_path"]
            K = np.asarray(frame["K"], dtype=float)
            T = np.asarray(frame["T_world_camera"], dtype=float)
        except Exception:
            continue
        if K.shape != (3, 3) or T.shape != (4, 4):
            continue
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(T)):
            continue
        try:
            depth = np.load(depth_path)
        except Exception:
            continue
        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 2:
            continue
        H, W = depth.shape
        if H == 0 or W == 0:
            continue
        us = np.arange(0, W, 4)
        vs = np.arange(0, H, 4)
        if us.size == 0 or vs.size == 0:
            continue
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu]
        mask = np.isfinite(z) & (z > 0.1) & (z < 2.0)
        if not np.any(mask):
            continue
        try:
            offset = float(frame.get('pixel_center_offset', 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(offset) or offset < 0.0 or offset > 1.0:
            continue
        u_flat = uu[mask].astype(float) + offset
        v_flat = vv[mask].astype(float) + offset
        z_flat = z[mask]
        pix = np.stack([u_flat, v_flat, np.ones_like(u_flat)], axis=1)
        try:
            Kinv = np.linalg.inv(K)
        except Exception:
            continue
        cam = (pix @ Kinv.T) * z_flat[:, None]
        R = T[:3, :3]
        t = T[:3, 3]
        world = cam @ R.T + t
        out.append(world)
    if not out:
        return np.zeros((0, 3), dtype=float)
    pts = np.concatenate(out, axis=0)
    return np.ascontiguousarray(pts, dtype=float)


def mesh_surface_distance(points, vertices, faces):
    points = np.asarray(points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be Nx3')
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('vertices must be Vx3')
    if not np.all(np.isfinite(points)):
        raise ValueError('points must be finite')
    if not np.all(np.isfinite(vertices)):
        raise ValueError('vertices must be finite')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('faces must be Fx3')
    if faces.size == 0:
        raise ValueError('no valid faces')
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.all(np.equal(np.mod(faces, 1), 0)):
            raise ValueError('faces must be integer indices')
        faces = faces.astype(np.int64)
    else:
        faces = faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise ValueError('face index out of range')
    tri = vertices[faces]
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    area2 = np.linalg.norm(cross, axis=1)
    good = area2 > 1e-12
    if not np.any(good):
        raise ValueError('no valid faces')
    a = a[good]
    b = b[good]
    c = c[good]
    ab = ab[good]
    ac = ac[good]
    cross = cross[good]
    area2 = area2[good]
    n_hat = cross / area2[:, None]
    e0 = b - a
    e1 = c - b
    e2 = a - c
    out = np.empty(points.shape[0], dtype=float)
    chunk = 64
    nverts = a.shape[0]
    for s in range(0, points.shape[0], chunk):
        ps = points[s:s + chunk]
        p = ps[:, None, :]
        av = a[None, :, :]
        bv = b[None, :, :]
        cv = c[None, :, :]
        ap = p - av
        d_plane = np.sum(ap * n_hat[None, :, :], axis=2)
        proj = p - d_plane[:, :, None] * n_hat[None, :, :]
        v0 = bv - av
        v1 = cv - bv
        v2 = av - cv
        def _closest_seg(p, s0, e0_):
            d = e0_ - s0
            dd = np.sum(d * d, axis=2)
            dd = np.where(dd > 1e-30, dd, 1.0)
            t = np.sum((p - s0) * d, axis=2) / dd
            t = np.clip(t, 0.0, 1.0)
            closest = s0 + t[:, :, None] * d
            return np.linalg.norm(p - closest, axis=2)
        d0 = _closest_seg(p, av, bv)
        d1 = _closest_seg(p, bv, cv)
        d2 = _closest_seg(p, cv, av)
        d_edge = np.minimum(np.minimum(d0, d1), d2)
        a0 = bv - av
        a1 = cv - av
        a2 = proj - av
        dot00 = np.sum(a0 * a0, axis=2)
        dot01 = np.sum(a0 * a1, axis=2)
        dot11 = np.sum(a1 * a1, axis=2)
        dot20 = np.sum(a2 * a0, axis=2)
        dot21 = np.sum(a2 * a1, axis=2)
        denom = dot00 * dot11 - dot01 * dot01
        denom = np.where(np.abs(denom) > 1e-30, denom, 1.0)
        u = (dot11 * dot20 - dot01 * dot21) / denom
        v = (dot00 * dot21 - dot01 * dot20) / denom
        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        d_plane_abs = np.abs(d_plane)
        d_tri = np.where(inside, d_plane_abs, d_edge)
        out[s:s + ps.shape[0]] = np.min(d_tri, axis=1)
    return out

def _sphere_fields(s):
    try:
        sid = s["id"]
        c = np.asarray(s["center"], dtype=float)
        r = float(s["radius"])
        hand = bool(s["hand"])
    except Exception:
        return None
    if c.shape != (3,) or not np.all(np.isfinite(c)):
        return None
    if not np.isfinite(r) or r <= 0:
        return None
    return sid, c, r, hand


def screen(points, spheres_start, spheres_path, margin=0.01):
    blocked = {
        "ok": False,
        "reason": "blocked",
        "min_distance": 0.0,
        "endpoint_hand_clear": False,
        "limitation": "visible geometry only",
    }
    try:
        P = np.asarray(points, dtype=float)
    except Exception:
        return dict(blocked, reason="invalid_points")
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return dict(blocked, reason="invalid_points")
    if not np.all(np.isfinite(P)):
        return dict(blocked, reason="nonfinite_points")
    if not isinstance(spheres_start, (list, tuple)) or len(spheres_start) == 0:
        return dict(blocked, reason="invalid_start")
    if not isinstance(spheres_path, (list, tuple)) or len(spheres_path) == 0:
        return dict(blocked, reason="invalid_path")
    starts = []
    for s in spheres_start:
        f = _sphere_fields(s)
        if f is None:
            return dict(blocked, reason="invalid_start_sphere")
        starts.append(f)
    ids = [f[0] for f in starts]
    poses = []
    for pose in spheres_path:
        if not isinstance(pose, (list, tuple)) or len(pose) != len(starts):
            return dict(blocked, reason="pose_mismatch")
        row = []
        for i, s in enumerate(pose):
            f = _sphere_fields(s)
            if f is None:
                return dict(blocked, reason="invalid_path_sphere")
            if f[0] != ids[i]:
                return dict(blocked, reason="id_order_mismatch")
            row.append(f)
        if not any(f[3] for f in row):
            return dict(blocked, reason="pose_no_hand")
        poses.append(row)
    if not any(f[3] for f in starts):
        return dict(blocked, reason="start_no_hand")
    remaining = P
    for (_, c, r, _) in starts:
        d = np.linalg.norm(remaining - c, axis=1)
        keep = d > (r + 0.002)
        remaining = remaining[keep]
        if remaining.shape[0] == 0:
            return dict(blocked, reason="no_remaining")
    if remaining.shape[0] == 0:
        return dict(blocked, reason="no_remaining")
    tree = cKDTree(remaining)
    def dist(center):
        d, _ = tree.query(center, k=1)
        return float(d)
    n_s = len(starts)
    init_d = np.full(n_s, np.inf)
    for i in range(n_s):
        _, c, r, _ = starts[i]
        init_d[i] = dist(c) - r
    min_dist = float(np.min(init_d))
    all_poses = [starts] + poses
    for pi, pose in enumerate(poses):
        prev = starts if pi == 0 else poses[pi - 1]
        for i in range(n_s):
            _, c, r, _ = pose[i]
            d = dist(c) - r
            if d < min_dist:
                min_dist = d
            if init_d[i] < margin:
                _, pc, _, _ = prev[i]
                pd = dist(pc) - prev[i][2]
                if not (d >= pd - 0.001):
                    return dict(blocked, reason="approach_violation", min_distance=min_dist)
            else:
                if d < margin:
                    return dict(blocked, reason="path_margin_violation", min_distance=min_dist)
    endpoint_clear = True
    final = poses[-1]
    for i in range(n_s):
        _, c, r, hand = final[i]
        if hand:
            if (dist(c) - r) < margin:
                endpoint_clear = False
    ok = bool(endpoint_clear)
    return {
        "ok": ok,
        "reason": "clear" if ok else "endpoint_hand_clear_fail",
        "min_distance": min_dist,
        "endpoint_hand_clear": endpoint_clear,
        "limitation": "visible geometry only",
    }

def shape_distance(points, shape):
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError('points must be Nx3')
    center = np.asarray(shape['center'], dtype=float)
    rotation = np.asarray(shape['rotation'], dtype=float)
    half_extents = np.asarray(shape['half_extents'], dtype=float)
    if center.shape != (3,) or rotation.shape != (3, 3) or half_extents.shape != (3,):
        raise ValueError('bad shape fields')
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(half_extents)):
        raise ValueError('nonfinite shape')
    if not np.all(half_extents > 0):
        raise ValueError('nonpositive half_extents')
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError('rotation not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError('rotation not right-handed')
    local = (P - center) @ rotation
    d = np.abs(local) - half_extents
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside

def screen_shapes(points, starts, path, margin=0.01, require_endpoint_clear=True):
    blocked = {
        "ok": False,
        "reason": "blocked",
        "min_distance": 0.0,
        "endpoint_hand_clear": False,
        "limitation": "visible geometry only",
        "diagnostic": {},
    }
    try:
        P = np.asarray(points, dtype=float)
    except Exception:
        return dict(blocked, reason="invalid_points")
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return dict(blocked, reason="invalid_points")
    if not np.all(np.isfinite(P)):
        return dict(blocked, reason="nonfinite_points")
    if not isinstance(starts, (list, tuple)) or len(starts) == 0:
        return dict(blocked, reason="invalid_start")
    if not isinstance(path, (list, tuple)) or len(path) == 0:
        return dict(blocked, reason="invalid_path")
    if not np.isfinite(margin) or margin < 0.0:
        return dict(blocked, reason="invalid_margin")
    start_ids = []
    for s in starts:
        if not isinstance(s, dict) or 'id' not in s or 'hand' not in s:
            return dict(blocked, reason="invalid_start_shape")
        try:
            shape_distance(P[:1], s)
        except (KeyError, TypeError, ValueError):
            return dict(blocked, reason="invalid_start_shape")
        start_ids.append(s['id'])
    poses = []
    for pose in path:
        if not isinstance(pose, (list, tuple)) or len(pose) != len(starts):
            return dict(blocked, reason="pose_mismatch")
        row = []
        for i, s in enumerate(pose):
            if not isinstance(s, dict) or 'id' not in s or 'hand' not in s:
                return dict(blocked, reason="invalid_path_shape")
            if s['id'] != start_ids[i]:
                return dict(blocked, reason="id_order_mismatch")
            try:
                shape_distance(P[:1], s)
            except (KeyError, TypeError, ValueError):
                return dict(blocked, reason="invalid_path_shape")
            row.append(s)
        if not any(bool(s['hand']) for s in row):
            return dict(blocked, reason="pose_no_hand")
        poses.append(row)
    if not any(bool(s['hand']) for s in starts):
        return dict(blocked, reason="start_no_hand")

    def clearance(shape):
        return float(np.min(shape_distance(P, shape)))

    init_clear = np.array([clearance(s) for s in starts], float)
    min_dist = float(np.min(init_clear))
    all_poses = [starts] + poses
    for pi, pose in enumerate(poses):
        prev = starts if pi == 0 else poses[pi - 1]
        for i, shape in enumerate(pose):
            d = clearance(shape)
            if d < min_dist:
                min_dist = d
            if init_clear[i] < margin:
                prev_d = clearance(prev[i])
                if not (d >= prev_d - 0.001):
                    return dict(
                        blocked,
                        reason="approach_violation",
                        min_distance=min_dist,
                        diagnostic={
                            "geom": shape['id'],
                            "initial_clearance": float(init_clear[i]),
                            "previous_clearance": float(prev_d),
                            "current_clearance": float(d),
                            "self_mask_policy": "sensor robot CAD surface matching only; no OBB point deletion",
                        },
                    )
            else:
                if d < margin:
                    return dict(
                        blocked,
                        reason="path_margin_violation",
                        min_distance=min_dist,
                        diagnostic={
                            "geom": shape['id'],
                            "initial_clearance": float(init_clear[i]),
                            "previous_clearance": float(clearance(prev[i])),
                            "current_clearance": float(d),
                            "self_mask_policy": "sensor robot CAD surface matching only; no OBB point deletion",
                        },
                    )
    endpoint_clear = True
    failed_hand = None
    final = poses[-1]
    for i, shape in enumerate(final):
        if bool(shape['hand']):
            d = clearance(shape)
            if d < min_dist:
                min_dist = d
            if d < margin:
                endpoint_clear = False
                if failed_hand is None:
                    failed_hand = {
                        "geom": shape['id'],
                        "initial_clearance": float(init_clear[i]),
                        "previous_clearance": float(clearance(poses[-2][i] if len(poses) > 1 else starts[i])),
                        "current_clearance": float(d),
                        "self_mask_policy": "sensor robot CAD surface matching only; no OBB point deletion",
                    }
    ok = bool(endpoint_clear or not require_endpoint_clear)
    return {
        "ok": ok,
        "reason": ("clear" if endpoint_clear else "screened_exit") if ok else "endpoint_hand_clear_fail",
        "min_distance": min_dist,
        "endpoint_hand_clear": endpoint_clear,
        "limitation": "mesh OBB is conservative approximation; not exact mesh collision",
        "diagnostic": failed_hand if failed_hand is not None else {
            "self_mask_policy": "sensor robot CAD surface matching only; no OBB point deletion",
        },
    }

def plan_retreat(port, observation, verdict, directions_override=None):
    q0 = np.asarray(observation['robot']['arm_qpos'], float)
    eef = np.asarray(observation['eef_position'], float)
    R = np.asarray(observation['eef_rotation'], float)
    if q0.shape != (7,) or eef.shape != (3,) or R.shape != (3, 3):
        return {'ok': False, 'reason': 'invalid input shapes', 'waypoints': [], 'candidate_reports': [], 'sensor_screen': None}
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(eef)) or not np.all(np.isfinite(R)):
        return {'ok': False, 'reason': 'non-finite input', 'waypoints': [], 'candidate_reports': [], 'sensor_screen': None}
    points = port.obstacle_cloud(observation['frames'])
    reports = []
    fail = {'ok': False, 'reason': 'no sensor-safe route', 'waypoints': [], 'candidate_reports': reports, 'sensor_screen': None}
    if len(points) == 0:
        return fail
    target = None
    for fr in observation['frames']:
        view = fr['view']
        box = verdict['boxes'].get(view, {}).get('target')
        if box is None and view == 'robot0_eye_in_hand':
            box = verdict['boxes'].get('wrist', {}).get('target')
        if box is not None:
            depth = np.load(fr['depth_path'])
            H, W = depth.shape
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = x0 * W, y0 * H, x1 * W, y1 * H
            xlo = int((3 * x0 + x1) / 4)
            xhi = int((x0 + 3 * x1) / 4)
            ylo = int((3 * y0 + y1) / 4)
            yhi = int((y0 + 3 * y1) / 4)
            xlo = max(0, min(xlo, W))
            xhi = max(0, min(xhi, W))
            ylo = max(0, min(ylo, H))
            yhi = max(0, min(yhi, H))
            roi = depth[ylo:yhi, xlo:xhi]
            valid = roi[np.isfinite(roi) & (roi >= 0.1) & (roi <= 2.0)]
            if len(valid) >= 8:
                z = np.median(valid)
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                pixel = np.array([cx, cy, 1.0])
                fr_T = np.asarray(fr['T_world_camera'], float)
                fr_K = np.asarray(fr['K'], float)
                campoint = np.linalg.inv(fr_K) @ (pixel * z)
                worldpoint = fr_T[:3, :3] @ campoint + fr_T[:3, 3]
                target = worldpoint
                break
    directions = [np.array([0.0, 0.0, 1.0])]
    if target is not None:
        away = eef - target
        away[2] = 0.0
        if np.linalg.norm(away) > 1e-9:
            away = away / np.linalg.norm(away)
            directions.extend([away, (away + np.array([0.0, 0.0, 1.0])) / np.linalg.norm(away + np.array([0.0, 0.0, 1.0]))])
        else:
            directions.append(np.array([0.0, 0.0, 1.0]))
    if directions_override is not None:
        directions = [np.asarray(direction, float) for direction in directions_override]
    start = port.robot_shapes(q0)
    for dist in (0.04, 0.06):
        for direction in directions:
            qs = [q0]
            valid_candidate = True
            steps = np.linspace(0, dist, int(np.ceil(dist / 0.002)) + 1)[1:]
            for alpha in steps:
                pos = eef + direction * alpha
                q = port.ik(pos, R, qs[-1])
                if q is None or np.asarray(q).shape != (7,) or not np.all(np.isfinite(q)):
                    valid_candidate = False
                    break
                qs.append(q)
            if not valid_candidate:
                continue
            poses = []
            for i in range(len(qs) - 1):
                qa, qb = qs[i], qs[i + 1]
                n = max(1, int(np.ceil(np.max(np.abs(np.asarray(qb) - np.asarray(qa))) / 0.003)))
                for a in np.linspace(0, 1, n + 1)[1:]:
                    q_interp = np.asarray(qa) + (np.asarray(qb) - np.asarray(qa)) * a
                    poses.append(port.robot_shapes(q_interp))
            scr = screen_shapes(points, start, poses)
            reports.append({'direction': direction.tolist(), 'distance': dist, 'reason': scr['reason']})
            if scr['ok']:
                return {'ok': True, 'reason': 'ok', 'waypoints': [q.tolist() for q in qs], 'candidate_reports': reports, 'sensor_screen': scr}
    return fail
