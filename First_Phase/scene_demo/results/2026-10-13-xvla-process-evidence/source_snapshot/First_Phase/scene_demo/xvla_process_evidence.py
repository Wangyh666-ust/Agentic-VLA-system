"""Measured release sampling and task-local, visually verified placement memory."""
import copy
import math
from scene_demo.finish_v1.simple_checker import encoder_observation

VIEWS = ('wrist', 'agentview', 'sideview')
PLACEMENT_RELATIONS = {'on', 'in', 'right_of'}

def finite_pair(values):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError('measured fingers must be two values')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('measured fingers must be finite numbers')
    return [float(v) for v in values]

def capture_frames(capture):
    timestamp = capture.get('timestamp')
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int,float)) or not math.isfinite(timestamp):
        raise ValueError('capture timestamp must be finite')
    available = capture.get('frames', []) + capture.get('verification_frames', [])
    frames = []
    for view in VIEWS:
        matching = [f for f in available if f.get('view') == view and f.get('rgb_path')]
        if not matching:
            raise ValueError('missing synchronized view: ' + view)
        frame = matching[0]
        for item in matching:
            time = item.get('timestamp', timestamp)
            if isinstance(time, bool) or not isinstance(time,(int,float)) or not math.isfinite(time) or time != timestamp:
                raise ValueError('capture image time mismatch')
        frames.append(frame)
    if len({str(f['rgb_path']) for f in frames}) != 3:
        raise ValueError('three independent image paths required')
    finite_pair(capture['robot']['finger_qpos'])
    return frames

class ReleaseSampler:
    def __init__(self, finger_home: list[float]):
        self.finger_home = finite_pair(finger_home)
        self.armed = False
        self.open_streak = 0
        self.events = 0

    def observe(self, finger_qpos: list[float], capture) -> dict | None:
        values = finite_pair(finger_qpos)
        error = max(abs(a-b) for a,b in zip(values,self.finger_home))
        if error >= .004:
            self.armed = True
            self.open_streak = 0
            return None
        if not self.armed:
            return None
        self.open_streak = self.open_streak + 1 if error <= .002 else 0
        if self.open_streak < 2:
            return None
        observation = capture()
        capture_frames(observation)
        measured = finite_pair(observation['robot']['finger_qpos'])
        if max(abs(a-b) for a,b in zip(measured,self.finger_home)) > .002:
            raise ValueError('release capture is not measured open')
        self.armed = False
        self.open_streak = 0
        self.events += 1
        return {'event_id': 'release_%04d' % self.events, 'capture': copy.deepcopy(observation),
                'detector': 'measured_open_transition'}

class PlacementEvidence:
    def __init__(self, success_checks: list[dict]):
        self.contract = copy.deepcopy(success_checks)
        self.ids = [c['id'] for c in success_checks]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('invalid check contract')
        self.definitions = {c['id']:c for c in self.contract}
        self.evidence = {}

    def validate(self, checks):
        if not isinstance(checks,list) or len(checks) != len(self.ids):
            raise ValueError('complete checks required')
        ids=[]
        for check in checks:
            if not isinstance(check,dict) or not isinstance(check.get('id'),str) or not isinstance(check.get('reason'),str):
                raise ValueError('invalid check fields')
            if 'achieved' not in check or (check['achieved'] is not None and type(check['achieved']) is not bool):
                raise ValueError('achieved must be bool or null')
            ids.append(check['id'])
        if len(set(ids)) != len(ids) or set(ids) != set(self.ids):
            raise ValueError('invalid check IDs')

    def record_release(self, event_id: str, context: dict, raw_checks: list[dict], provenance: dict) -> None:
        self.validate(raw_checks)
        if not isinstance(event_id,str) or not event_id:
            raise ValueError('event ID required')
        measured = encoder_observation(context)
        if not measured['at_standard_open']:
            raise ValueError('release evidence requires synchronized measured opening')
        for check in raw_checks:
            definition=self.definitions[check['id']]
            if check['achieved'] is False and definition['relation'] in PLACEMENT_RELATIONS:
                self.evidence.pop(check['id'],None)
            if (check['achieved'] is True and definition['relation'] in PLACEMENT_RELATIONS
                    and not definition.get('preserved')):
                self.evidence[check['id']] = {'id':check['id'],'event_id':event_id,
                    'timestamp':measured['timestamp'],'relation':definition['relation'],
                    'reason':check['reason'],'provenance':copy.deepcopy(provenance)}

    def combine(self, current_checks: list[dict]) -> list[dict]:
        self.validate(current_checks)
        combined=copy.deepcopy(current_checks)
        for check in combined:
            identifier=check['id']
            if check['achieved'] is False:
                self.evidence.pop(identifier,None)
            elif check['achieved'] is None and identifier in self.evidence:
                definition=self.definitions[identifier]
                if definition['relation'] in PLACEMENT_RELATIONS and not definition.get('preserved'):
                    check['achieved']=True
                    check['history_event_id']=self.evidence[identifier]['event_id']
                    check['current_reason']=check['reason']
                    check['reason']=check['reason']+'；采用已验证放置证据 '+identifier+'/'+check['history_event_id']
        return combined

    def snapshot(self) -> list[dict]:
        return copy.deepcopy([self.evidence[k] for k in self.ids if k in self.evidence])
