import numpy as np
from .geometry import clouds, screen, plan_retreat
from scene_demo import joint_home as jh


def run_finish(port, context, initial_verdict, decide_fn, check_fn, emit):
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
    caps = {'release': 20, 'movement': 80, 'confirm': 20, 'home': 360}

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
                poses.append(port.robot_spheres(previous + (q - previous) * alpha))
            previous = q
        report = dict(screen(clouds(fresh['frames']), port.robot_spheres(actualq), poses))
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

        if d['state'] == 'needs_release':
            q = np.asarray(port.read_robot()['arm_qpos'])
            f = np.asarray(port.read_robot()['finger_qpos'])

            settle_streak = 0
            for _ in range(10):
                st = act('release', q, f, command_kind='release_settle')
                r['release_settle_actions'] += 1
                if np.max(np.abs(st['arm_qvel'])) <= 0.02:
                    settle_streak += 1
                else:
                    settle_streak = 0
                if settle_streak >= 5:
                    break
            if settle_streak < 5:
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
                    np.max(np.abs(st['arm_qvel'])) <= 0.02
                )
                if condition:
                    streak += 1
                else:
                    streak = 0
                if streak >= 5:
                    break
            if streak < 5:
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
                    if r['counts']['movement'] >= 80:
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

        if d['state'] != 'complete' or v.get('released') is not True:
            r['reason'] = 'not eligible for home'
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
