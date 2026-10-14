"""X-VLA whole-skill checker using synchronized measured three-view input."""
import copy
import json
import re
from scene_demo.finish_v1.runtime import build_context
from scene_demo.finish_v1.simple_checker import SimpleQwenClient, encoder_observation
from scene_demo.finish_v1 import vision

IDENTITIES = {
 'porcelain_mug': '白/浅灰瓷杯，白把手；不是红花纹或黄把手杯。',
 'white_yellow_mug': '浅色杯身与黄色区域/黄色把手。',
 'alphabet_soup': '圆柱罐、银盖、蓝色/橙色标签；与Milk盒及Tomato Sauce罐区分。',
 'cream_cheese': '白/浅灰长方体食品盒、蓝紫标签/Cream Cheese字样。',
 'butter': '红橙色长方体食品包装、蓝色区域/Butter字样。',
 'chocolate_pudding': '棕色与浅蓝包装的小盒、Chocolate Pudding字样。',
 'moka_pot': '金属咖啡壶、黑把手。',
 'black_book': '扁平书本，不是杯子或食品盒。',
 'akita_black_bowl': '深灰黑碗，不是盘。',
 'wine_bottle': '深绿近黑长颈瓶、白标签。',
}

STATE_DEFINITIONS = {
    'white_cabinet_bottom_region': {
        'closed': '最下层抽屉前板归入柜体、未明显向外伸。',
        'open': '最下层抽屉明显拉出。',
    },
}

def build_skill_context(capability: dict, user_request: str) -> dict:
    from collections import Counter
    from scene_demo import catalog
    native = capability.get('skill_kind') == 'native_long_horizon'
    components = ([catalog.CAPABILITIES[i] for i in capability['component_capability_ids']]
                  if native else [capability])
    if not components:
        raise ValueError('native skill must contain action components')
    preserved = capability.get('preserved_checks', []) if native else []
    if not isinstance(preserved, list):
        raise ValueError('preserved_checks must be a list')
    conditions = components + preserved
    # Validate every atomic condition through the unchanged control context
    # parser; preserved conditions never become the control representative.
    for condition in conditions:
        build_context(condition, user_request)
        if not isinstance(condition.get('instruction'), str) or not condition['instruction'].strip():
            raise ValueError('condition instruction required')
    goals = [condition['goals'][0] for condition in conditions]
    if len(set(map(tuple, goals))) != len(goals):
        raise ValueError('duplicate final conditions')
    opposites = {'turnon': 'turnoff', 'turnoff': 'turnon', 'open': 'close', 'close': 'open'}
    goal_set = set(map(tuple, goals))
    if any((opposites[g[0]], *g[1:]) in goal_set for g in goals if g[0] in opposites):
        raise ValueError('contradictory final conditions')
    if native:
        raw_goals = capability.get('goals')
        if not isinstance(raw_goals, list) or any(not isinstance(g, list) for g in raw_goals):
            raise ValueError('native goals must be lists')
        for goal in raw_goals:
            build_context({'goals': [goal]}, user_request)
        if Counter(map(tuple, raw_goals)) != Counter(map(tuple, goals)):
            raise ValueError('native goals must equal complete component and preserved goals')
    context = build_context(components[-1], user_request)
    checks = []
    for index, condition in enumerate(conditions):
        goal = condition['goals'][0]
        relation = condition.get('visual_relation', goal[0])
        if relation not in ('on', 'in', 'turnon', 'turnoff', 'open', 'close', 'right_of'):
            raise ValueError('unsupported visual relation')
        check = {'id': 'goal_' + str(index), 'description': condition['instruction'],
                 'relation': relation, 'target_id': goal[1],
                 'destination_id': goal[2] if len(goal) > 2 else None}
        if index >= len(components):
            check['preserved'] = True
        if 'destination_definition' in condition:
            definition = condition['destination_definition']
            if not isinstance(definition, str) or not definition.strip():
                raise ValueError('destination_definition must be nonempty string')
            check['destination_definition'] = definition
        category = re.sub(r'_\d+(?=_|$)', '', goal[1])
        identity = IDENTITIES.get(category, catalog.OBJECT_VISUAL_IDENTITY_DEFINITIONS.get(category))
        if identity:
            check['target_visual_identity'] = copy.deepcopy(identity)
        state = STATE_DEFINITIONS.get(category, catalog.OBJECT_VISUAL_STATE_DEFINITIONS.get(category))
        if goal[0] in ('turnon', 'turnoff', 'open', 'close') and state:
            check['visual_state_definition'] = copy.deepcopy(state)
        checks.append(check)
    context['criteria'].update(skill_description=capability['instruction'], success_checks=checks)
    return context

def build_prompt(context: dict, observation_role=None) -> str:
    feedback_context = context
    if 'visual_timeline' in context:
        indices = context['visual_timeline']['current_frame_indices']
        feedback_context = {**context, 'sensor_frames': [context['sensor_frames'][i] for i in indices]}
    safe = {'current_skill': {'description': context['criteria']['skill_description']},
            'success_checks': context['criteria']['success_checks'],
            'sensor_frames': context['sensor_frames'],
            'actuator_measurements': encoder_observation(feedback_context)}
    if observation_role is not None:
        if observation_role != 'release_snapshot':
            raise ValueError('invalid observation role')
        safe['observation_role'] = observation_role
    if 'visual_timeline' in context:
        safe['visual_timeline'] = copy.deepcopy(context['visual_timeline'])
    goal_prompt = ('逐项检查success_checks明确要求的物品与目标，部分完成不能结束。仅验证过程照片，placement True要实际放在正确目的地并已放下；夹爪张开或事件标签本身不证明目标，其他目标可False/Null。' if observation_role else '逐项检查success_checks明确要求的物品与目标。部分完成不能结束。只检查明确要求的目标，不附加松手、解除接触、退离或归位条件；仍夹持不否定目标到位。')
    role_prompt = ('这只是实测开口触发的照片，不证明曾持物；placement true必须明确指定物品已经放在正确目的地并得到支撑，不是空手张开、正掉落或入口外。其他子目标可false/null，不得补全；此分类不触发收尾。' if observation_role else '当前遮挡应返回null；过程照片独立验证，不能凭历史忽略当前取出、移位或掉落的反证。')
    return '\n'.join([
        role_prompt,
        '你是机器人当前完整skill的结束状态检查器。每组三图来自一个采样时刻：wrist是夹爪第一人称；agentview与sideview是两个不同外部视角。按frame/view/timestamp和visual_timeline区分过程与当前照片。',
        '时间相同的照片属于同一物理瞬间，不能当成多个时刻或先后动作的证据。每组照片的夹爪反馈只对应该组时刻，不能套用其他时刻的反馈。',
        '先逐项确认任务指定的target_id、外观和destination_id，再判断关系。同类别的其他对象是潜在干扰物；没有确认是同一个目标，不能用其位置证明目标未到位、被取出或掉落。多目标必须分别跟踪，不能重复用同一实物。身份不确定时target_matches_definition=null且achieved=null。',
        'process_history包含实测过程照片和当时判定，previous_check_states包含先前检查状态；其中True/False/Null都只是可纠正的旧判断，不是真值或计划。必须重新对照原图，可纠正先前误识别；不能仅因旧False否定新证据，也不能仅因旧True宣称完成。没有提供图片的旧判定不能独自证明成功。',
        '放置关系可结合过程照片确认同一目标曾放入正确位置，并检查当前是否有同一目标被取出、移动或掉落的明确反证；关门、抽屉、开关及preserved状态必须由最新一组照片确认，不能用历史状态替代。侧面、斜视或其他视角只要门板与机身等状态关系可辨即可，不要求正面；看不到状态时返回null，不臆造炉腔或入口。',
        goal_prompt,
        'on盘指在盘面；in指进入内部，旁边、入口外或二维投影重叠不算。只有对应检查项含close才要求关门，不添加层板等额外目的地条件。',
        'right_of指盘子指定右侧的桌面区域，而非盘面或盘内。destination_definition只解释静态目的地，不声明当前状态。preserved保持条件只检查继续满足，不请求额外开关动作。',
        '两个检查项指定两件同类物品时，必须是两件不同实物，不能用同一只摩卡壶重复满足两个放置检查。容器关着不能证明物品在内部；位置遮挡且证据不足时返回null。',
        '外观提示只解释资产身份或状态，不声明当前状态；细纹理和标签字样无需全部清晰，不是必须满足的识别条件。指定物品不能被其他物品替代。',
        '每项achieved三值：true目标达到；false有明确未达证据；null身份或目标关系看不清。一视角能明确判断且无相反证据即可，其他视角遮挡不是反证。',
        'at_standard_open=true表示同步实测夹爪达到标准开口，按已松手、未夹持处理；遮挡、投影重叠或接近不能推翻。false表示需要开爪，不单凭闭合证明持物。开口不能证明物品位置，也不能证明路径安全。',
        '不接受计划、检查时机或阶段名称作为完成证据。每项必须输出id、achieved、target_matches_definition、current_evidence、evidence_event_ids和reason。achieved、target_matches_definition、current_evidence都是true/false/null；current_evidence表示当前照片是否有对应同一目标的明确关系或状态证据；evidence_event_ids只列本次process_history提供、实际用作证据的event_id。放置True可引用过程图，False必须有当前同一目标的未达反证；状态类判断必须有当前证据。只输出JSON：{"checks":[{"id":"goal_0","achieved":true或false或null,"target_matches_definition":true或false或null,"current_evidence":true或false或null,"evidence_event_ids":[],"reason":"简短依据"}],"reason":"简短依据"}。',
        json.dumps(safe, ensure_ascii=False)])

class XvlaQwenClient(SimpleQwenClient):
    def __init__(self, *args, **kwargs):
        if len(args) < 4:
            kwargs.setdefault('max_calls',14)
        super().__init__(*args, **kwargs)

    required_views = ('wrist', 'agentview', 'sideview')
    simple_handoff = True

    def observe_capture(self, capture: dict) -> None:
        self._capture = copy.deepcopy(capture)

    def set_check_contract(self, context: dict) -> None:
        from scene_demo.xvla_process_evidence import PlacementEvidence
        from scene_demo.xvla_timeline import ObservationTimeline
        checks = context['criteria']['success_checks']
        signature = [(c['id'],c['relation'],c['description'],c.get('target_id'),
                      c.get('destination_id'),c.get('preserved',False)) for c in checks]
        if signature != getattr(self, '_contract_signature', None):
            self._placement_evidence = PlacementEvidence(checks)
            self._timeline = ObservationTimeline()
            self._active_event_ids = set()
            self._contract_signature = signature
        self._check_ids = [c['id'] for c in checks]
        self._check_definitions = {c['id']:c for c in checks}

    def _validate_capture(self, context, image_paths):
        self._validate_source(getattr(self, '_capture', None), context, image_paths)

    @classmethod
    def _validate_source(cls, capture, context, image_paths):
        from scene_demo.xvla_process_evidence import capture_frames, finite_pair
        if capture is None:
            raise ValueError('synchronized capture required')
        source = capture_frames(capture)
        frames = context['sensor_frames']
        if len(frames) != 3 or len(image_paths) != 3 or len(set(map(str,image_paths))) != 3:
            raise ValueError('exactly three independent images required')
        for index,(view,frame,path,item) in enumerate(zip(cls.required_views,frames,image_paths,source)):
            if frame != {'frame':index,'view':view,'timestamp':float(capture['timestamp'])}:
                raise ValueError('capture frame labels/time mismatch')
            if str(item['rgb_path']) != str(path):
                raise ValueError('capture image path/time mismatch')
        measured=context['actuator_measurements']
        if measured['timestamp'] != capture['timestamp'] or finite_pair(measured['finger_qpos']) != finite_pair(capture['robot']['finger_qpos']):
            raise ValueError('capture encoder mismatch')
        encoder_observation(context)

    def _load_images(self, image_paths):
        # Extend only X-VLA's temporal input; preserve the original finish client.
        from scene_demo.xvla_timeline import MAX_INPUT_IMAGES
        if not isinstance(image_paths, list) or not 1 <= len(image_paths) <= MAX_INPUT_IMAGES:
            raise ValueError('X-VLA timeline image budget exceeded')
        result = []
        for offset in range(0, len(image_paths), 6):
            result.extend(super()._load_images(image_paths[offset:offset+6]))
        return result

    def remember_release(self, capture, context, event_id):
        from scene_demo.xvla_process_evidence import capture_frames
        paths = [str(f['rgb_path']) for f in capture_frames(capture)]
        self._validate_source(capture, context, paths)
        if not encoder_observation(context)['at_standard_open']:
            raise ValueError('release observation requires measured standard opening')
        self.set_check_contract(context)
        self._timeline.remember_release(event_id, context, paths)

    def note_release_status(self, event_id, status):
        self._timeline.classify_release(event_id, [], [], status)

    @staticmethod
    def _aggregate(checks):
        values=[c['achieved'] for c in checks]
        return False if any(v is False for v in values) else True if all(v is True for v in values) else None

    def process_evidence_snapshot(self) -> list[dict]:
        ledger=getattr(self,'_placement_evidence',None)
        return ledger.snapshot() if ledger is not None else []

    def analyze_release(self, context: dict, image_paths: list[str], event_id: str) -> dict:
        import hashlib
        from pathlib import Path
        self._validate_capture(context,image_paths)
        self.set_check_contract(context)
        self.remember_release(self._capture, context, event_id)
        if not encoder_observation(context)['at_standard_open']:
            raise ValueError('release classification requires measured standard opening')
        prompt=build_prompt(context, observation_role='release_snapshot')
        self._active_event_ids = set()
        result=self._request(context,image_paths,prompt)
        raw=copy.deepcopy(self.last_skill_verdict)
        effective=copy.deepcopy(self.last_effective_checks)
        provenance={'images':[{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in image_paths],
                    'request_sha256':result.get('request_sha256'), 'response_sha256':result.get('response_sha256'),
                    'call_index':result.get('call_index')}
        self._placement_evidence.record_release(event_id,context,effective,provenance)
        self._timeline.classify_release(event_id, raw['checks'], effective)
        result.update(prompt_text=prompt,checker='xvla_timeline_skill_v3',observation_role='release_snapshot',
                      event_id=event_id,raw_visual_goal_verdict=raw,
                      visual_goal_verdict={'checks':effective,'goal_achieved':self._aggregate(effective),'reason':raw['reason']},
                      effective_checks=effective,
                      process_evidence=self.process_evidence_snapshot())
        return result

    def analyze(self, context: dict, image_paths: list[str]) -> dict:
        self._validate_capture(context,image_paths)
        self.set_check_contract(context)
        request_context, request_paths = self._timeline.request_input(context, image_paths)
        self._active_event_ids = {row['event_id'] for row in request_context['visual_timeline']['process_history']}
        prompt=build_prompt(request_context)
        result=self._request(request_context,request_paths,prompt)
        raw=copy.deepcopy(self.last_skill_verdict)
        # Re-read the actual timeline; an old boolean cannot override a new unknown.
        effective=copy.deepcopy(self.last_effective_checks)
        self._placement_evidence.invalidate_false(effective)
        self._timeline.remember_check(context, raw['checks'], effective)
        reused=[c['id']+'/'+c['history_event_id'] for c in effective if 'history_event_id' in c]
        reason=raw['reason']+('；本轮对照过程照片 '+', '.join(reused) if reused else '')
        effective_verdict={'checks':effective,'goal_achieved':self._aggregate(effective),'reason':reason}
        if 'reported_goal_achieved' in raw:
            effective_verdict['reported_goal_achieved']=raw['reported_goal_achieved']
        result['verdict']=SimpleQwenClient.parse_verdict(self,json.dumps({'goal_achieved':effective_verdict['goal_achieved'],'reason':reason}))
        result.update(prompt_text=prompt,checker='xvla_timeline_skill_v3',raw_visual_goal_verdict=raw,
                      effective_checks=effective,visual_goal_verdict=effective_verdict,
                      input_context=request_context,request_image_paths=request_paths,
                      inspection_scope='current_and_process',
                      process_evidence=self.process_evidence_snapshot())
        return result

    def parse_verdict(self, content: str) -> dict:
        raw = vision._parse_verdict_content(content)
        checks = raw.get('checks')
        if not isinstance(checks, list) or not hasattr(self, '_check_ids'):
            raise ValueError('checks and contract required')
        ids = []
        effective = []
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get('id'), str) or not isinstance(check.get('reason'), str):
                raise ValueError('invalid check fields')
            if 'achieved' not in check or (check['achieved'] is not None and type(check['achieved']) is not bool):
                raise ValueError('achieved must be bool or null')
            for field in ('target_matches_definition', 'current_evidence'):
                if field not in check or (check[field] is not None and type(check[field]) is not bool):
                    raise ValueError(field+' must be bool or null')
            events = check.get('evidence_event_ids')
            if (not isinstance(events, list) or any(not isinstance(e,str) for e in events)
                    or len(set(events)) != len(events)
                    or not set(events) <= getattr(self, '_active_event_ids', set())):
                raise ValueError('invalid evidence_event_ids')
            normalized = copy.deepcopy(check)
            definition = self._check_definitions.get(check['id'])
            if definition is None:
                raise ValueError('unknown check ID')
            placement = definition['relation'] in ('on','in','right_of') and not definition.get('preserved')
            identified = check['target_matches_definition'] is True
            has_current = check['current_evidence'] is True
            if (not identified or (not placement and not has_current)
                    or (check['achieved'] is False and not has_current)
                    or (check['achieved'] is True and not has_current and not events)):
                normalized['achieved'] = None
                normalized['reason'] = check['reason']+'；目标身份或有效时刻证据未确认'
            if normalized['achieved'] is True and placement and events:
                normalized['history_event_ids'] = list(events)
                normalized['history_event_id'] = events[0]
                normalized['reason'] = check['reason']+'；本轮复核过程照片 '+', '.join(events)
            effective.append(normalized)
            ids.append(check['id'])
        if len(ids) != len(set(ids)) or set(ids) != set(self._check_ids):
            raise ValueError('missing, duplicate or unknown check IDs')
        values = [c['achieved'] for c in effective]
        goal = False if False in values else True if all(v is True for v in values) else None
        reason = raw.get('reason', '')
        if not isinstance(reason, str):
            raise ValueError('reason must be string')
        verdict = super().parse_verdict(json.dumps({'goal_achieved': goal, 'reason': reason}))
        self.last_skill_verdict = {'checks': copy.deepcopy(checks), 'goal_achieved': self._aggregate(checks), 'reason': reason}
        self.last_effective_checks = effective
        if 'goal_achieved' in raw:
            self.last_skill_verdict['reported_goal_achieved'] = raw['goal_achieved']
        return verdict
