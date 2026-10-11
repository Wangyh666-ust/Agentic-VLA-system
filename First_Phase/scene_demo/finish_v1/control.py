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
    caps = {'release': 20, 'movement': 480 if simple_handoff else 120, 'confirm': 20, 'home': 360}
    concurrent_open = False

    class OpeningPort:
        def __getattr__(self, name):
            return getattr(port, name)

        def robot_shapes(self, q):
            return port.opening_shapes(q)

    def geometry_port():
        return OpeningPort() if concurrent_open else port

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
                poses.append(geometry_port().robot_shapes(previous + (q - previous) * alpha))
            previous = q
        report = dict(screen_shapes(port.obstacle_cloud(fresh['frames']), geometry_port().robot_shapes(actualq), poses))
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
            r['handoff_mode'] = 'two_question_goal_contact_v2'
            r['handoff_goal_latched'] = v.get('operation_achieved') is True

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
                if simple_handoff and not condition:
                    measured = np.abs(np.asarray(st['finger_qpos'], float))
                    standard = np.abs(np.asarray(ref['finger_home'], float))
                    # Near-full opening can be constrained by a support surface.
                    # This only permits screened withdrawal with open still commanded;
                    # it does NOT establish release or bypass the exact home gate.
                    partial = bool(
                        np.all(standard > .001) and
                        measured.sum() >= .90 * standard.sum() and
                        np.all(measured >= .85 * standard) and
                        np.max(np.abs(st.get('finger_qvel', [float('inf')] * 2))) <= .005 and
                        np.max(np.abs(st['arm_qvel'])) <= .02
                    )
                    if partial:
                        condition = True
                        r['withdraw_with_partial_open'] = {
                            'finger_qpos': list(st['finger_qpos']),
                            'aperture_fraction': float(measured.sum() / standard.sum()),
                            'standard_open_pending': True,
                            'release_proven': False,
                        }
                if condition:
                    streak += 1
                else:
                    streak = 0
                if streak >= (2 if simple_handoff else 5):
                    break
            if streak < (2 if simple_handoff else 5):
                # A stove contact may physically constrain one finger. Requiring
                # full opening before retreat deadlocks. Keep opening commanded
                # along a CAD-screened exit; exact opening is still required by
                # the unchanged home-entry gate below. Place tasks retain their gate.
                state = port.read_robot()
                finite = all(np.all(np.isfinite(state[k])) for k in ('arm_qpos', 'arm_qvel', 'finger_qpos'))
                if (simple_handoff and op == 'turn' and r.get('handoff_goal_latched')
                        and finite and np.max(np.abs(state['arm_qvel'])) <= .02
                        and hasattr(port, 'opening_shapes')):
                    concurrent_open = True
                    r['concurrent_open_retreat'] = {
                        'triggered': True, 'finger_qpos_before': list(state['finger_qpos']),
                        'exact_open_pending': True, 'release_proven': False,
                        'geometry': 'measured/half/standard-open CAD envelope',
                    }
                else:
                    r['reason'] = 'release not confirmed'
                    return r
            if simple_handoff:
                # Once accepted, finishing is an atomic recovery sequence.
                # Task acceptance is checked again after home, not used to
                # strand an open, safe robot midway through recovery.
                opening_confirmed = r['handoff_goal_latched']
            else:
                v, d = verify()
            if not simple_handoff and op == 'place':
                opening_confirmed = (
                    v.get('target_visible') is True and
                    v.get('destination_visible') is True and
                    v.get('at_destination') is True and
                    v.get('supported') is True and
                    v.get('stable') is not False
                )
            elif not simple_handoff:
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

            feedback_enabled = simple_handoff and getattr(port, 'feedback_retreat_enabled', False)
            if feedback_enabled:
                from .feedback_retreat import run as feedback_run
                feedback = feedback_run(geometry_port(), v, act, limit=480)
                r['feedback_retreat'] = feedback
                r['plan'] = feedback['plan']
                r['movement_budget'] = 480
                if not feedback['ok']:
                    r['reason'] = feedback['reason']
                    return r
            else:
                p = plan_retreat(geometry_port(), port.capture(), v)
                r['plan'] = p
                r['active_retreat_plan'] = p
                if not p['ok']:
                    r['reason'] = 'retreat plan blocked'
                    return r
                qs = p['waypoints']
                movement_limit = min(480, max(120, 16 * (len(qs) - 1) + 40)) if simple_handoff else 120
                r['movement_budget'] = movement_limit

                i = 1
                while i < len(qs):
                    best_error = float('inf')
                    stagnant = 0
                    while np.max(np.abs(np.asarray(port.read_robot()['arm_qpos']) - np.asarray(qs[i]))) > 0.004:
                        if r['counts']['movement'] >= movement_limit:
                            r['reason'] = 'movement budget'
                            return r
                        st = act('movement', qs[i], ref['finger_home'] if concurrent_open else None,
                                 command_kind='open_and_retreat' if concurrent_open else None)
                        error = float(np.max(np.abs(np.asarray(st['arm_qpos']) - np.asarray(qs[i]))))
                        r['movement_progress'] = {'waypoint': i, 'waypoints': len(qs), 'joint_error': error}
                        if error < best_error - 0.0002:
                            best_error, stagnant = error, 0
                        else:
                            stagnant += 1
                        if simple_handoff and stagnant >= 40:
                            r['reason'] = 'movement no progress'
                            return r
                        if r['counts']['movement'] % 10 == 0:
                            check = route_screen(qs[i:], 0.003)
                            r.setdefault('movement_screens', []).append(check)
                            if not check['ok']:
                                if check['reason'] == 'endpoint_hand_clear_fail' and not r.get('retreat_replans'):
                                    r.setdefault('retreat_replans', []).append({'failed_screen': check, 'new_plan': None})
                                    newp = plan_retreat(geometry_port(), port.capture(), v)
                                    r['retreat_replans'][-1]['new_plan'] = newp
                                    if newp['ok']:
                                        r['active_retreat_plan'] = newp
                                        qs = newp['waypoints']
                                        if simple_handoff:
                                            movement_limit = min(480, r['counts']['movement'] + max(120, 16 * (len(qs) - 1) + 40))
                                            r['movement_budget'] = movement_limit
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
                    st = act('confirm', qs[-1], ref['finger_home'] if concurrent_open else None)
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
                if not simple_handoff:
                    v, d = verify()
        # Opening-envelope screening is only for withdrawal. Original measured
        # geometry and the original home path/controller remain in use for home.
        if concurrent_open:
            r['concurrent_open_retreat']['finger_qpos_after'] = list(port.read_robot()['finger_qpos'])
            concurrent_open = False

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
        if simple_handoff:
            semantic_ok = r.get('handoff_goal_latched') is True
            r['home_entry']['semantic_basis'] = 'initial handoff purpose; final task acceptance pending'
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
