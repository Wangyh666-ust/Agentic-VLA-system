import numpy as np
from .geometry import clouds, screen, plan_retreat
from scene_demo import joint_home as jh


def run_finish(port, context, initial_verdict, decide_fn, check_fn, emit):
    r = {
        'ready': False, 'ok': False, 'reason': '',
        'counts': {'release': 0, 'movement': 0, 'confirm': 0, 'home': 0},
        'plan': None, 'checks': [], 'restored': False,
        'limitation': 'visible geometry only'
    }
    v = initial_verdict
    caps = {'release': 20, 'movement': 80, 'confirm': 20, 'home': 360}

    def act(ph, q, f=None):
        if r['counts'][ph] >= caps[ph]:
            raise RuntimeError('budget')
        port.phase = ph
        port.start()
        st = port.step(q, f)
        r['counts'][ph] += 1
        emit({'phase': ph, 'count': r['counts'][ph], 'state': st})
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
        return screen(clouds(fresh['frames']), port.robot_spheres(actualq), poses)['ok']

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

            if not d['allow_release']:
                streak = 0
                for _ in range(10):
                    st = act('release', q, f)
                    if np.max(np.abs(st['arm_qvel'])) <= 0.02:
                        streak += 1
                    else:
                        streak = 0
                    if streak >= 5:
                        break
                if streak < 5:
                    r['reason'] = 'settle not stable'
                    return r
                v, d = verify()
                if not d['allow_release']:
                    r['reason'] = 'release denied'
                    return r

            streak = 0
            while r['counts']['release'] < 20:
                st = act('release', q, ref['finger_home'])
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
            if not all(v.get(k) is True for k in ('at_destination', 'supported', 'released', 'stable')):
                r['reason'] = 'release visual failed'
                return r

        if d['state'] == 'needs_retreat':
            if not d['allow_retreat'] or (op == 'turn' and v['operation_achieved'] is not True):
                r['reason'] = 'retreat denied'
                return r
            if op == 'place' and not all(v.get(k) is True for k in ('at_destination', 'supported', 'released', 'stable')):
                r['reason'] = 'retreat denied'
                return r

            p = plan_retreat(port, port.capture(), v)
            r['plan'] = p
            if not p['ok']:
                r['reason'] = 'retreat plan blocked'
                return r
            qs = p['waypoints']

            for i in range(1, len(qs)):
                while np.max(np.abs(np.asarray(port.read_robot()['arm_qpos']) - np.asarray(qs[i]))) > 0.004:
                    if r['counts']['movement'] >= 80:
                        r['reason'] = 'movement budget'
                        return r
                    st = act('movement', qs[i])
                    if r['counts']['movement'] % 10 == 0 and not route_screen(qs[i:], 0.003):
                        r['reason'] = 'movement screen blocked'
                        return r

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
        homeq = np.asarray(ref['homeq'])
        n = max(1, int(np.ceil(np.max(np.abs(homeq - actualq)) / 0.01)))
        qs = [actualq + (homeq - actualq) * alpha for alpha in np.linspace(0, 1, n + 1)]

        if not route_screen(qs[1:], 0.01):
            r['reason'] = 'home screen blocked'
            return r

        for i in range(1, len(qs)):
            while np.max(np.abs(np.asarray(port.read_robot()['arm_qpos']) - np.asarray(qs[i]))) > 0.004:
                if r['counts']['home'] >= 360:
                    r['reason'] = 'home budget'
                    return r
                st = act('home', qs[i], ref['finger_home'])
                if r['counts']['home'] % 10 == 0 and not route_screen(qs[i:], 0.01):
                    r['reason'] = 'home refresh blocked'
                    return r

        streak = 0
        while r['counts']['home'] < 360:
            st = act('home', homeq, ref['finger_home'])
            if jh.home_metrics(ref, st)['ready']:
                streak += 1
            else:
                streak = 0
            if streak >= 5:
                break
        if streak < 5:
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
