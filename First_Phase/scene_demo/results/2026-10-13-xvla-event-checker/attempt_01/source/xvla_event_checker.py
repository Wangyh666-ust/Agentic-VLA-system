"""Visual event observations and a deterministic, task-independent goal reducer.

Only this adapter changes the X-VLA inspection protocol. The original finish,
retreat, joint reference and simulator-independent acceptance remain unchanged.
"""
import copy
import hashlib
import json
from collections import deque
from pathlib import Path

from scene_demo.xvla_finish import XvlaQwenClient
from scene_demo.finish_v1.simple_checker import SimpleQwenClient, encoder_observation
from scene_demo.finish_v1 import vision

PROTOCOL = 'xvla_event_facts_v4'
MAX_EVENTS = 4
MAX_IMAGES = 42  # reference + 4 * (before, release, after) + current triplets
PLACEMENTS = ('on', 'in', 'right_of')


def image_source(context, paths):
    encoder_observation(context)
    return {'context': copy.deepcopy(context), 'paths': list(map(str, paths)),
            'sha256': [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]}


def build_event_prompt(packet):
    return '\n'.join([
        '你是视觉事实观察员，程序根据事实汇总完整任务。不要直接输出整体完成与否。',
        '先用reference和before照片辨认任务目标与干扰物，再看每个release/after，最后看current。reference只用于身份对应，不能证明当前位置。time=-1表示旧身份参考照片采样时间未记录，不用于时间顺序或编码器判断。',
        'wrist是腕部第一人称，agentview与sideview是两个外部视角；照片不裁剪、不镜像。每张照片的frame唯一。每个实测开口事件可能只是空手张开，不表示成功持物或放置。before/release/after只描述采样顺序，不能据标签猜测动作成功。',
        '对每个events中的event_id，必须逐项返回全部goal的observations；不相关或看不清也要返回unknown并说明。再对current逐项返回全部goal。不能跳过过程照片，也不能用旧检查True/False/Null替代图像；previous_checks仅供对照纠错。',
        '一项观察格式：{"id":"goal_0","identity":"matched|uncertain|distractor","object_key":"视觉跟踪的同一实物标识或null","relation":"satisfied|violated|occluded|unknown","frames":[关系证据frame整数],"identity_frames":[身份或连续对应证据frame整数],"supported":true或false或null,"continuity":"intact|broken|unknown","reason":"具体画面依据"}。',
        'object_key根据参考图的外观和位置给实际物体稳定命名（例如yellow_handle_mug），跨事件与当前保持同一标识，不是复制目标ID。不同目标物必须对应不同实物；同一物体的放置与开关等多个要求可共用标识。相同类别的两件实物也须区分，不能凭事件序号分配。',
        'identity=matched须由外观或连续视觉对应确认指定目标；标签文字、花纹无需每帧可读。桌上其他杯或其他罐不能当作指定目标的反证。uncertain/distractor均不可确证目标关系。',
        'relation描述这组图对指定目标关系的证据：satisfied=明确满足；violated=明确同一目标未满足；occluded=目标被容器/门/机械臂遮住；unknown=其他证据不足。frames必须引用本组release或after（current则引用当前组）确切照片，identity_frames可引用参考和更早图片。不能用before位置证明release之后已放好。',
        '历史放置satisfied还须supported=true，即放在目的地得到支撑，不是正在落下或入口外；同步实测开口只证明已张开，不证明放对。当前到位可仍夹持，松手、退离、归位由后续收尾完成，勿添加这些完成条件。',
        'current看不到目标时可返回occluded。continuity=intact表示结合这次原过程图与当前场景，同一目标的身份和所在容器对应仍成立且无取出、移位或掉落反证；broken表示有明确变化，unknown表示对应无法建立。当前被门遮住不等于历史放置证据失效，关着门也不能独自证明物品放入。',
        '门/抽屉/开关及preserved状态只用当前图判断；侧面或斜视可辨门板与机身关系即可，不要求正面。只检查goals要求的条件；on盘为盘面，in容器为内部，right_of为指定物旁的区域，不额外要求进入盘内部。只在目标含close时要求关门。',
        '只输出JSON：{"events":[{"event_id":"release_0001","observations":[上述观察]}],"current":[上述观察],"reason":"简短说明"}。release_review时current仍是本次最新图，事件分类不会直接触发收尾。',
        json.dumps(packet, ensure_ascii=False, separators=(',', ':')),
    ])


def reduce_observations(definitions, events, current):
    """All observations have already passed frame/time/identity schema checks."""
    effective = []
    for goal in definitions:
        gid = goal['id']
        now = next(r for r in current if r['id'] == gid)
        placement = goal['relation'] in PLACEMENTS and not goal.get('preserved')
        proof = None
        for event in events:
            row = next(r for r in event['observations'] if r['id'] == gid)
            if row['identity'] == 'matched':
                if row['relation'] == 'satisfied' and row['supported'] is True:
                    proof = (event['event_id'], row)
                elif row['relation'] == 'violated' or row['continuity'] == 'broken':
                    proof = None
        matched = now['identity'] == 'matched'
        direct = matched and now['relation'] in ('satisfied', 'violated')
        achieved = (now['relation'] == 'satisfied') if direct else None
        event_ids = []
        if (placement and not direct and proof and matched
                and now['relation'] == 'occluded' and now['continuity'] == 'intact'
                and now['object_key'] == proof[1]['object_key']):
            achieved = True
            event_ids = [proof[0]]
        row = {'id': gid, 'achieved': achieved, 'target_matches_definition': matched,
               'current_evidence': direct, 'evidence_event_ids': event_ids,
               'object_key': now['object_key'], 'reason': now['reason']}
        if event_ids:
            row.update(history_event_id=event_ids[0], history_event_ids=event_ids)
            row['reason'] += '；原图事件确认 ' + event_ids[0]
        effective.append(row)
    # A model cannot satisfy distinct asset IDs by counting one physical object twice.
    owners = {}
    for goal, row in zip(definitions, effective):
        if row['achieved'] is True:
            owners.setdefault(row['object_key'], set()).add(goal['target_id'])
    for row in effective:
        if row['achieved'] is True and len(owners[row['object_key']]) > 1:
            row['achieved'] = None
            row['reason'] += '；不同目标复用了同一实物'
    return effective


class XvlaEventQwenClient(XvlaQwenClient):
    request_options = {'enable_thinking': True, 'thinking_budget': 1024, 'max_tokens': 4096}

    def set_check_contract(self, context):
        old = getattr(self, '_contract_signature', None)
        super().set_check_contract(context)
        if old != self._contract_signature:
            self._reference = None
            self._recent = deque(maxlen=4)
            self._events = {}
            self._omitted = []
            self._states = []
            self._facts = []

    def remember_observation(self, capture, context):
        from scene_demo.xvla_process_evidence import capture_frames
        paths = [str(f['rgb_path']) for f in capture_frames(capture)]
        self._validate_source(capture, context, paths)
        self.set_check_contract(context)
        source = image_source(context, paths)
        time = context['actuator_measurements']['timestamp']
        if self._reference is None:
            self._reference = source
        if self._recent and time <= self._recent[-1]['context']['actuator_measurements']['timestamp']:
            return
        self._recent.append(source)
        for event in self._events.values():
            if event['after'] is None and time > event['time']:
                event['after'] = source

    def remember_identity_images(self, paths, views):
        """Offline identity-only photos with explicitly unknown time/feedback."""
        if self._reference is not None:
            raise ValueError('identity reference already assigned')
        if len(paths) != 3 or tuple(views) != self.required_views:
            raise ValueError('three identity views required')
        self._reference = {'paths': list(map(str, paths)),
            'sha256': [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],
            'context': {'sensor_frames': [{'frame':i, 'view':v, 'timestamp':-1} for i,v in enumerate(views)]}}

    def remember_release(self, capture, context, event_id):
        super().remember_release(capture, context, event_id)
        if event_id in self._events or event_id in self._omitted:
            return
        if len(self._events) >= MAX_EVENTS:
            self._omitted.append(event_id)
            return
        time = context['actuator_measurements']['timestamp']
        earlier = [s for s in self._recent if s['context']['actuator_measurements']['timestamp'] < time]
        self._events[event_id] = {'time':time, 'before':earlier[0] if earlier else None,
            'release':image_source(context, self._timeline.releases[event_id]['image_paths']),
            'after':None, 'status':'pending'}

    def note_release_status(self, event_id, status):
        super().note_release_status(event_id, status)
        if event_id in self._events:
            self._events[event_id]['status'] = status

    def _load_images(self, paths):
        if not isinstance(paths, list) or not 1 <= len(paths) <= MAX_IMAGES:
            raise ValueError('event image budget exceeded')
        result = []
        for offset in range(0, len(paths), 6):
            result.extend(SimpleQwenClient._load_images(self, paths[offset:offset+6]))
        return result

    def _packet(self, context, paths, role):
        time = context['actuator_measurements']['timestamp']
        packet = {'protocol':PROTOCOL, 'role':role,
                  'goals':list(self._check_definitions.values()), 'groups':[], 'events':[],
                  'previous_checks':copy.deepcopy([s for s in self._states if s['time'] <= time][-10:]),
                  'omitted_events':list(self._omitted)}
        frames, image_paths = [], []
        def add(source, kind, event_id=None):
            if source is None:
                return []
            own = source['context']
            stamp = own['sensor_frames'][0]['timestamp']
            if stamp > time:
                raise ValueError('future evidence')
            indices = []
            for frame, path, sha in zip(own['sensor_frames'],source['paths'],source['sha256']):
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                    raise ValueError('evidence image bytes changed')
                indices.append(len(frames))
                frames.append({**frame,'frame':len(frames)})
                image_paths.append(path)
            group = {'kind':kind, 'event_id':event_id, 'time':stamp, 'frames':indices}
            if 'actuator_measurements' in own:
                group['encoder'] = encoder_observation(own)
            else:
                group['time_known'] = False
            packet['groups'].append(group)
            return indices
        add(self._reference, 'reference')
        for eid, event in sorted(self._events.items(),key=lambda item:item[1]['time']):
            if event['time'] > time:
                if role == 'release_review':
                    continue
                raise ValueError('future release')
            before = add(event['before'], 'before', eid)
            release = add(event['release'], 'release', eid)
            after = add(event['after'], 'after', eid)
            packet['events'].append({'event_id':eid,'time':event['time'],
                'identity_frames':before+release+after,
                'relation_frames':release+after,'status':event['status']})
        packet['current_frames'] = add(image_source(context, paths), 'current')
        packet['sensor_frames'] = frames
        if len(image_paths) > MAX_IMAGES:
            raise ValueError('event image budget exceeded')
        return packet, {**context,'sensor_frames':frames}, image_paths

    def _analyze_events(self, context, paths, role):
        self._validate_capture(context, paths)
        self.set_check_contract(context)
        packet, request_context, request_paths = self._packet(context, paths, role)
        self._active_packet = packet
        prompt = build_event_prompt(packet)
        result = self._request(request_context, request_paths, prompt)
        effective = copy.deepcopy(self.last_effective_checks)
        raw = copy.deepcopy(self.last_event_observations)
        reason = raw['reason']
        self._facts = [{'id':r['id'],'event_id':r['history_event_id'],'object_key':r['object_key']}
                       for r in effective if r.get('achieved') is True and 'history_event_id' in r]
        result.update(checker=PROTOCOL, prompt_text=prompt, input_context=request_context,
            request_image_paths=request_paths, event_packet=packet, event_observations=raw,
            raw_visual_goal_verdict={'checks':effective,'reason':reason},
            effective_checks=effective, process_evidence=self.process_evidence_snapshot(),
            visual_goal_verdict={'checks':effective,'goal_achieved':self._aggregate(effective),'reason':reason})
        return result

    def analyze_release(self, context, image_paths, event_id):
        self.remember_release(self._capture, context, event_id)
        # Review after the nearby passive sample, using that sample's own feedback.
        event = self._events.get(event_id)
        latest = event['after'] if event and event['after'] else image_source(context,image_paths)
        latest_context = latest['context']
        original = self._capture
        self._capture = {'timestamp':latest_context['actuator_measurements']['timestamp'],
            'robot':{'finger_qpos':latest_context['actuator_measurements']['finger_qpos']},
            'frames':[{'view':f['view'],'rgb_path':p} for f,p in zip(latest_context['sensor_frames'],latest['paths'])]}
        try:
            result = self._analyze_events(latest_context, latest['paths'], 'release_review')
        finally:
            self._capture = original
        self.note_release_status(event_id, 'classified')
        result.update(observation_role='release_snapshot',event_id=event_id)
        return result

    def analyze(self, context, image_paths):
        result = self._analyze_events(context,image_paths,'current_review')
        self._states.append({'time':context['actuator_measurements']['timestamp'],
            'goals':[{k:r[k] for k in ('id','achieved','object_key','evidence_event_ids')}
                     for r in result['effective_checks']]})
        return result

    def process_evidence_snapshot(self):
        return copy.deepcopy(getattr(self,'_facts',[]))

    def parse_verdict(self, content):
        raw = vision._parse_verdict_content(content)
        packet = self._active_packet
        events = raw.get('events')
        if not isinstance(events,list) or not isinstance(raw.get('reason'),str):
            raise ValueError('event observations and reason required')
        expected = [e['event_id'] for e in packet['events']]
        if [e.get('event_id') for e in events] != expected:
            raise ValueError('missing, reordered or unknown events')
        def validate(rows, relation_frames, latest_time):
            if not isinstance(rows,list) or [r.get('id') for r in rows] != self._check_ids:
                raise ValueError('every goal must be observed exactly once in order')
            allowed_identity = {f['frame'] for f in packet['sensor_frames'] if f['timestamp'] <= latest_time}
            for row in rows:
                if row.get('identity') not in ('matched','uncertain','distractor') or row.get('relation') not in ('satisfied','violated','occluded','unknown'):
                    raise ValueError('invalid identity/relation')
                if row.get('continuity') not in ('intact','broken','unknown') or not isinstance(row.get('reason'),str):
                    raise ValueError('invalid continuity/reason')
                if 'supported' not in row or (row['supported'] is not None and type(row['supported']) is not bool):
                    raise ValueError('supported must be bool or null')
                for field, allowed in [('frames',set(relation_frames)),('identity_frames',allowed_identity)]:
                    refs=row.get(field)
                    if not isinstance(refs,list) or any(type(n) is not int for n in refs) or len(set(refs))!=len(refs) or not set(refs)<=allowed:
                        raise ValueError('invalid '+field)
                if row['identity']=='matched' and (not isinstance(row.get('object_key'),str) or not row['object_key'].strip() or not row['identity_frames']):
                    raise ValueError('matched identity requires visual anchor')
                if row['relation'] in ('satisfied','violated','occluded') and not row['frames']:
                    raise ValueError('relation requires source image')
        for event, spec in zip(events, packet['events']):
            latest=max(packet['sensor_frames'][i]['timestamp'] for i in spec['relation_frames'])
            validate(event.get('observations'),spec['relation_frames'],latest)
        latest=max(packet['sensor_frames'][i]['timestamp'] for i in packet['current_frames'])
        validate(raw.get('current'),packet['current_frames'],latest)
        effective=reduce_observations(list(self._check_definitions.values()),events,raw['current'])
        self.last_event_observations=copy.deepcopy(raw)
        self.last_effective_checks=effective
        return SimpleQwenClient.parse_verdict(self,json.dumps({'goal_achieved':self._aggregate(effective),'reason':raw['reason']}))
