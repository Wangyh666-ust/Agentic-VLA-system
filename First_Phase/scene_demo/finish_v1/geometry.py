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
        u_flat = uu[mask].astype(float)
        v_flat = vv[mask].astype(float)
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

def plan_retreat(port, observation, verdict):
    q0 = np.asarray(observation['robot']['arm_qpos'], float)
    eef = np.asarray(observation['eef_position'], float)
    R = np.asarray(observation['eef_rotation'], float)
    if q0.shape != (7,) or eef.shape != (3,) or R.shape != (3, 3):
        return {'ok': False, 'reason': 'invalid input shapes', 'waypoints': [], 'candidate_reports': [], 'sensor_screen': None}
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(eef)) or not np.all(np.isfinite(R)):
        return {'ok': False, 'reason': 'non-finite input', 'waypoints': [], 'candidate_reports': [], 'sensor_screen': None}
    points = clouds(observation['frames'])
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
    start = port.robot_spheres(q0)
    for dist in (0.02, 0.04, 0.06):
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
                    poses.append(port.robot_spheres(q_interp))
            scr = screen(points, start, poses)
            reports.append({'direction': direction.tolist(), 'distance': dist, 'reason': scr['reason']})
            if scr['ok']:
                return {'ok': True, 'reason': 'ok', 'waypoints': [q.tolist() for q in qs], 'candidate_reports': reports, 'sensor_screen': scr}
    return fail
