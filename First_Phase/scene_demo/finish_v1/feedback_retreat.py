"""Measured-pose withdrawal; screened short commands without waypoint gates."""
import numpy as np
from .geometry import plan_retreat, screen_shapes, shape_distance
from scene_demo import joint_home

LOOKAHEAD = .008
MIN_TRAVEL = .03
REFRESH_EVERY = 10
MAX_REPLANS = 3


def directions(port, observation, current=None):
    """Generic exits derived from nearby sensed geometry, with upward bias."""
    up = np.array([0., 0., 1.])
    points = port.obstacle_cloud(observation['frames'])
    q = observation['robot']['arm_qpos']
    nearby = []
    for shape in port.robot_shapes(q):
        if shape['hand'] and len(points):
            ds = shape_distance(points, shape)
            idx = int(np.argmin(ds))
            nearby.append((float(ds[idx]), shape, points[idx]))
    exits = [up]
    if nearby:
        _, shape, point = min(nearby, key=lambda item: item[0])
        rotation = np.asarray(shape['rotation'])
        local = (point - np.asarray(shape['center'])) @ rotation
        half = np.asarray(shape['half_extents'])
        normal = local - np.clip(local, -half, half)
        if np.linalg.norm(normal) < 1e-9:
            axis = int(np.argmax(np.abs(local) - half))
            normal[axis] = 1. if local[axis] >= 0 else -1.
        away = -(rotation @ normal)
        away[2] = 0.
        if np.linalg.norm(away) > 1e-9:
            away /= np.linalg.norm(away)
            exits += [(away + up) / np.sqrt(2), away]
    for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        exits.append(np.array([x, y, 1.]) / np.sqrt(2))
    unique = []
    for direction in exits:
        if current is not None and float(np.dot(direction, current)) > .98:
            continue
        if not any(float(np.dot(direction, d)) > .98 for d in unique):
            unique.append(direction)
    return unique


def run(port, verdict, act, limit=480):
    result = {'ok': False, 'reason': '', 'mode': 'measured_pose_feedback',
              'screens': [], 'replans': [], 'home_candidates': [], 'trace': [], 'trace_actions': [], 'plan': None,
              'removed_gate': 'per-waypoint joint_error<=0.004',
              'retained_guards': 'CAD path, current endpoint clearance, joint limits, opening and home readiness'}
    observation = port.capture()
    result['initial_frames'] = observation['frames']
    origin = np.array(observation['eef_position'], float)
    def home_eligible(q):
        return port.home_screen(joint_home.joint_waypoints(q, port.reference['homeq']))
    current_plan = plan_retreat(port, observation, verdict,
                               directions_override=directions(port, observation), endpoint_eligible=port.candidate_home_screen)
    result['plan'] = current_plan
    if not current_plan['ok']:
        result['reason'] = 'retreat plan blocked'
        return result
    direction = np.array(current_plan['candidate_reports'][-1]['direction'])
    active_distance = current_plan['candidate_reports'][-1]['distance']
    cloud = port.obstacle_cloud(observation['frames'])
    cloud_frames = observation['frames']
    checkpoint = origin.copy()
    route_origin = origin.copy()
    since_refresh = 0
    next_home_check_at = 0

    def replan(cause):
        nonlocal direction, active_distance, cloud, cloud_frames, checkpoint, route_origin, since_refresh
        if len(result['replans']) >= MAX_REPLANS:
            result['reason'] = 'retreat replans exhausted'
            return False
        fresh = port.capture()
        alternatives = directions(port, fresh, current=direction)
        new_plan = plan_retreat(port, fresh, verdict, directions_override=alternatives,
                               endpoint_eligible=port.candidate_home_screen)
        result['replans'].append({'cause': cause, 'actual_q': fresh['robot']['arm_qpos'],
                                  'eef_position': fresh['eef_position'], 'frames': fresh['frames'], 'plan': new_plan})
        if not new_plan['ok']:
            result['reason'] = 'no screened retreat alternative'
            return False
        direction = np.array(new_plan['candidate_reports'][-1]['direction'])
        active_distance = new_plan['candidate_reports'][-1]['distance']
        cloud = port.obstacle_cloud(fresh['frames'])
        cloud_frames = fresh['frames']
        checkpoint = np.array(fresh['eef_position'])
        route_origin = checkpoint.copy()
        since_refresh = 0
        return True

    while len(result['trace_actions']) < limit:
        pose = port.measured_pose()
        q = np.array(pose['robot']['arm_qpos'])
        xyz = np.array(pose['eef_position'])
        shapes = port.robot_shapes(q)
        clearance = screen_shapes(cloud, shapes, [shapes])
        travel = float(np.linalg.norm(xyz - origin))
        if travel >= MIN_TRAVEL and clearance['ok'] and len(result['trace_actions']) >= next_home_check_at:
            # Refresh before accepting an endpoint; cached geometry only guides steps.
            fresh = port.capture()
            cloud = port.obstacle_cloud(fresh['frames'])
            cloud_frames = fresh['frames']
            qfresh = fresh['robot']['arm_qpos']
            actual_shapes = port.robot_shapes(qfresh)
            clearance = screen_shapes(cloud, actual_shapes, [actual_shapes])
            result['endpoint_frames'] = fresh['frames']
            if clearance['ok']:
                eligibility = home_eligible(qfresh)
                result['home_candidates'].append({'actions': len(result['trace_actions']),
                                                  'actual_q': qfresh, 'frames': fresh['frames'],
                                                  'home_screen': eligibility})
                if eligibility.get('ok') is True:
                    # Stop where the arm actually is, never pull it back to an old endpoint.
                    result['endpoint_clearance'] = clearance
                    result['travel_m'] = travel
                    break
                # Current separation is insufficient when the original return
                # would sweep an obstacle. Continue exiting, without moving home.
                next_home_check_at = len(result['trace_actions']) + REFRESH_EVERY
                if np.dot(xyz - route_origin, direction) >= active_distance - .003:
                    if not replan('home_path_not_clear_at_planned_exit'):
                        return result
                    continue
        if since_refresh >= REFRESH_EVERY:
            fresh = port.capture()
            xyz = np.array(fresh['eef_position'])
            cloud = port.obstacle_cloud(fresh['frames'])
            cloud_frames = fresh['frames']
            distance = xyz - route_origin
            lateral = np.linalg.norm(distance - direction * np.dot(distance, direction))
            progress = float(np.dot(xyz - checkpoint, direction))
            result['trace'].append({'actions': len(result['trace_actions']),
                                    'travel_m': float(np.linalg.norm(xyz - origin)),
                                    'progress_m': progress, 'lateral_m': float(lateral)})
            if progress < .0005 or lateral > .006:
                if not replan('stalled' if progress < .0005 else 'actual_pose_deviation'):
                    return result
            else:
                checkpoint = xyz.copy()
                since_refresh = 0
            continue
        target = port.ik(xyz + direction * LOOKAHEAD,
                         np.array(pose['eef_rotation']), q)
        if target is None:
            if not replan('local_ik_failed'):
                return result
            continue
        target = np.array(target)
        subdivisions = max(1, int(np.ceil(np.max(abs(target - q)) / .003)))
        path = [port.robot_shapes(q + (target - q) * alpha)
                for alpha in np.linspace(0, 1, subdivisions + 1)[1:]]
        checked = screen_shapes(cloud, shapes, path, require_endpoint_clear=False)
        if not checked['ok']:
            result['screens'].append({'actual_q': q.tolist(), 'candidate_q': target.tolist(),
                                       'frames': cloud_frames, 'report': checked})
            if not replan('local_path_' + checked['reason']):
                return result
            continue
        act('movement', target, port.reference['finger_home'], command_kind='feedback_open_and_retreat')
        result['trace_actions'].append({'actual_q': q.tolist(), 'target_q': target.tolist(),
                                       'from_xyz': xyz.tolist(), 'direction': direction.tolist(),
                                       'screen': checked})
        since_refresh += 1
    else:
        result['reason'] = 'movement budget'
        return result
    # Only the final hold checks opening and settling; no intermediate joint gate.
    hold = np.asarray(port.read_robot()['arm_qpos'])
    streak = 0
    for _ in range(20):
        state = act('confirm', hold, port.reference['finger_home'])
        settled = np.max(abs(np.asarray(state['arm_qvel']))) <= .02
        opened = np.max(abs(np.asarray(state['finger_qpos']) - port.reference['finger_home'])) <= .002
        streak = streak + 1 if settled and opened else 0
        if streak >= 2:
            result.update(ok=True, reason='retreated and settled', hold_q=hold.tolist())
            return result
    result['reason'] = 'retreat final opening or settling failed'
    return result
