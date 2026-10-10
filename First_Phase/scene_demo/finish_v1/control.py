import numpy as np
from .geometry import clouds, screen, plan_retreat, screen_shapes
from scene_demo import joint_home as jh


def run_finish(port, context, initial_verdict, decide_fn, check_fn, emit, simple_handoff=False):
    r = {
        'ready': False, 'ok': False, 'reason': '',
        'counts': {'release': 0, 'movement': 0, 'confirm': 0, 'home': 0},
        'release_settle_actions': 0, 'release_open_actions': 0,
        'plan': None, 'checks': [], 'restored': False,
        'limitation': 'visible geometry only',
        'opening_confirmed': False,
        'object_release_not_yet_confirmed': False
    }
    v = initial_verdict
    caps = {'release': 20, 'movement': 120, 'confirm': 20, 'home': 360}

    def act(ph, q, f=None, command_kind=None):
        if r['counts'][ph] >= caps[ph]:
            raise RuntimeError('budget')
        port.phase = ph
        port.start()
        st = port.step(q, f)
        r['counts'][ph] += 1
        row = {'phase': ph, 'count': r['counts'][ph], 'state': st}
        if command_kind is not None:
            row['command_kind'] = command_kind
        emit(row)
        return st

    def verify():
        obs = port.capture()
        vnew = check_fn(context, [obs])
        dnew = decide_fn(context, vnew)
        r['checks'].append(vnew)
        return vnew, dnew

    def route_screen(remaining, delta):
        fresh = port.capture()
        actualq = np.asarray(port.read_robot()['arm_qpos'])
        poses = []
        previous = actualq
        for q in remaining:
            q = np.asarray(q)
            n = max(1, int(np.ceil(np.max(np.abs(q - previous)) / delta)))
            for alpha in np.linspace(0, 1, n + 1)[1:]:
                poses.append(port.robot_shapes(previous + (q - previous) * alpha))
            previous = q
        report = dict(screen_shapes(port.obstacle_cloud(fresh['frames']), port.robot_shapes(actualq), poses))
        report['frames'] = fresh['frames']
        if 'depth_path' in fresh:
            report['depth_path'] = fresh['depth_path']
        if 'calibration_path' in fresh:
            report['calibration_path'] = fresh['calibration_path']
        if 'json_path' in fresh:
            report['json_path'] = fresh['json_path']
        return report

    try:
        d = decide_fn(context, v)
        op = context['current_subtask']['operation']
        ref = port.reference

        if d['state'] in ('incomplete', 'unknown'):
            r['reason'] = 'initial stop'
            return r

        if simple_handoff:
            # Purpose reached: explicitly open, including already-open fingers.
            d = {'state': 'needs_release', 'allow_release': True, 'allow_retreat': False}
            r['handoff_mode'] = 'simple_goal_contact_v1'

        if d['state'] == 'needs_release':
            q = np.asarray(port.read_robot()['arm_qpos'])
            f = np.asarray(port.read_robot()['finger_qpos'])

            settle_streak = 0
            for _ in range(0 if simple_handoff else 10):
                st = act('release', q, f, command_kind='release_settle')
                r['release_settle_actions'] += 1
                if np.max(np.abs(st['arm_qvel'])) <= 0.02:
                    settle_streak += 1
                else:
                    settle_streak = 0
                if settle_streak >= 5:
                    break
            if not simple_handoff and settle_streak < 5:
                r['reason'] = 'settle not stable'
                return r

            if not d['allow_release']:
                v, d = verify()
                if not d['allow_release']:
                    r['reason'] = 'release denied'
                    return r

            streak = 0
            while r['counts']['release'] < 20:
                st = act('release', q, ref['finger_home'], command_kind='release_open')
                r['release_open_actions'] += 1
                condition = (
                    np.max(np.abs(q - st['arm_qpos'])) <= 0.01 and
                    np.max(np.abs(np.asarray(ref['finger_home']) - np.asarray(st['finger_qpos']))) <= 0.002 and
                    (simple_handoff or np.max(np.abs(st['arm_qvel'])) <= 0.02)
                )
                if condition:
                    streak += 1
                else:
                    streak = 0
                if streak >= (2 if simple_handoff else 5):
                    break
            if streak < (2 if simple_handoff else 5):
                r['reason'] = 'release not confirmed'
                return r
            v, d = verify()
            if op == 'place':
                opening_confirmed = (
                    v.get('target_visible') is True and
                    v.get('destination_visible') is True and
                    v.get('at_destination') is True and
                    v.get('supported') is True and
                    v.get('stable') is not False
                )
            else:
                opening_confirmed = (
                    v.get('target_visible') is True and
                    v.get('operation_achieved') is True
                )
            if not opening_confirmed:
                r['reason'] = 'release visual failed'
                return r
            r['opening_confirmed'] = True
            r['object_release_not_yet_confirmed'] = True
            d = {'state': 'needs_retreat', 'reason': 'release followed by withdrawal', 'allow_release': False, 'allow_retreat': True}
            r['_release_path'] = True

        if d['state'] == 'needs_retreat':
            if not d['allow_retreat']:
                r['reason'] = 'retreat denied'
                return r
            if not r.get('_release_path'):
                if op == 'turn':
                    if v.get('operation_achieved') is not True or v.get('released') is not True:
                        r['reason'] = 'retreat denied'
                        return r
                if op == 'place':
                    if not all(v.get(k) is True for k in ('at_destination', 'supported', 'released', 'stable')):
                        r['reason'] = 'retreat denied'
                        return r

            p = plan_retreat(port, port.capture(), v)
            r['plan'] = p
            r['active_retreat_plan'] = p
            if not p['ok']:
                r['reason'] = 'retreat plan blocked'
                return r
            qs = p['waypoints']

            i = 1
            while i < len(qs):
                while np.max(np.abs(np.asarray(port.read_robot()['arm_qpos']) - np.asarray(qs[i]))) > 0.004:
                    if r['counts']['movement'] >= 120:
                        r['reason'] = 'movement budget'
                        return r
                    st = act('movement', qs[i])
                    if r['counts']['movement'] % 10 == 0:
                        check = route_screen(qs[i:], 0.003)
                        r.setdefault('movement_screens', []).append(check)
                        if not check['ok']:
                            if check['reason'] == 'endpoint_hand_clear_fail' and not r.get('retreat_replans'):
                                r.setdefault('retreat_replans', []).append({'failed_screen': check, 'new_plan': None})
                                newp = plan_retreat(port, port.capture(), v)
                                r['retreat_replans'][-1]['new_plan'] = newp
                                if newp['ok']:
                                    r['active_retreat_plan'] = newp
                                    qs = newp['waypoints']
                                    i = 1
                                    break
                                else:
                                    r['reason'] = 'movement screen blocked'
                                    return r
                            else:
                                r['reason'] = 'movement screen blocked'
                                return r
                else:
                    i += 1
                    continue
                continue

            streak = 0
            for _ in range(20):
                st = act('confirm', qs[-1])
                qerr = np.max(np.abs(np.asarray(port.read_robot()['arm_qpos']) - np.asarray(qs[-1])))
                if qerr <= 0.004 and np.max(np.abs(st['arm_qvel'])) <= 0.02:
                    streak += 1
                else:
                    streak = 0
                if streak >= 5:
                    break
            if streak < 5:
                r['reason'] = 'confirm not stable'
                return r
            v, d = verify()

        r['home_entry'] = {
            'visible_semantics': False,
            'measured_open': False,
            'settled': False,
            'task_complete_before_home': False,
            'label': 'safety eligibility under review; task completion pending final verification',
        }
        semantic_ok = False
        try:
            if v.get('target_visible') is True:
                if op == 'place':
                    semantic_ok = (
                        v.get('destination_visible') is True
                        and v.get('at_destination') is True
                        and v.get('supported') is True
                        and v.get('stable') is not False
                    )
                elif op == 'turn':
                    semantic_ok = (
                        v.get('operation_achieved') is True
                        and v.get('stable') is not False
                    )
        except (TypeError, ValueError, KeyError):
            semantic_ok = False
        r['home_entry']['visible_semantics'] = bool(semantic_ok)
        decided_complete = False
        try:
            decided_complete = d['state'] == 'complete' and v.get('released') is True
        except (TypeError, ValueError, KeyError):
            decided_complete = False
        r['home_entry']['task_complete_before_home'] = bool(decided_complete)

        actuator_ok = True
        try:
            st = port.read_robot()
            arm_qpos = np.asarray(st['arm_qpos'], dtype=float)
            arm_qvel = np.asarray(st['arm_qvel'], dtype=float)
            finger_qpos = np.asarray(st['finger_qpos'], dtype=float)
            finger_home = np.asarray(ref['finger_home'], dtype=float)
            if not (
                arm_qpos.shape == (7,)
                and arm_qvel.shape == (7,)
                and finger_qpos.shape == (2,)
                and finger_home.shape == (2,)
                and np.all(np.isfinite(arm_qpos))
                and np.all(np.isfinite(arm_qvel))
                and np.all(np.isfinite(finger_qpos))
                and np.all(np.isfinite(finger_home))
            ):
                actuator_ok = False
            else:
                measured_open = bool(
                    np.max(np.abs(finger_qpos - finger_home)) <= 0.002
                )
                settled = bool(np.max(np.abs(arm_qvel)) <= 0.02)
                r['home_entry']['measured_open'] = measured_open
                r['home_entry']['settled'] = settled
                if not (measured_open and settled):
                    actuator_ok = False
        except (TypeError, ValueError, KeyError):
            actuator_ok = False
        if not actuator_ok:
            r['reason'] = 'home recovery actuator gate'
            return r

        if not semantic_ok:
            r['reason'] = 'not eligible for home'
            return r

        actualq = np.asarray(port.read_robot()['arm_qpos'])
        clearance_screen = route_screen([actualq], 0.003)
        r['home_entry']['clearance_screen'] = clearance_screen
        clearance_ok = clearance_screen.get('ok') if isinstance(clearance_screen, dict) else None
        if clearance_ok is not True:
            r['reason'] = 'home recovery clearance gate'
            return r


        actualq = np.asarray(port.read_robot()['arm_qpos'])
        points = jh.joint_waypoints(actualq, ref['homeq'])
        r['home_screen'] = port.home_screen(points)
        screen_ok = r['home_screen'].get('ok') if isinstance(r['home_screen'], dict) else None
        if screen_ok is not True:
            r['reason'] = 'home screen blocked'
            return r

        r['home_mode'] = 'original_joint_home_streaming'
        r['home_streak'] = 0
        for i in range(360):
            target = points[min(i, len(points) - 1)]
            st = act('home', target, ref['finger_home'])
            ready = jh.home_metrics(ref, st)['ready']
            if ready:
                r['home_streak'] += 1
            else:
                r['home_streak'] = 0
            if r['home_streak'] >= 5:
                break

        if r['home_streak'] < 5:
            r['reason'] = 'home not ready'
            return r

        v, d = verify()
        if d['state'] == 'complete':
            r['ready'] = True
            r['ok'] = True
            r['reason'] = 'complete'
        else:
            r['reason'] = 'final visual failed'
        return r

    except Exception as exc:
        r['reason'] = 'exception:' + str(exc)
        return r
    finally:
        port.close()
        r['restored'] = True
