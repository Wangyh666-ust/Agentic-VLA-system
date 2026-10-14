"""Confirmed release evidence survives occlusion; only grounded changes revoke it.

One observation request per capture. Device states are observed, not voted on by
separate goal-conditioned requests. All original finish/control interfaces remain.
"""
import copy
import hashlib
import json
from collections import deque
from pathlib import Path

from scene_demo.xvla_finish import XvlaQwenClient
from scene_demo.xvla_process_evidence import capture_frames
from scene_demo.finish_v1.simple_checker import SimpleQwenClient, encoder_observation
from scene_demo.finish_v1 import vision

PROTOCOL = 'xvla_evidence_memory_v5'
PLACEMENTS = ('in', 'on', 'right_of')
STATES = {'open':'open', 'close':'closed', 'turnon':'on', 'turnoff':'off'}
WINDOW_SECONDS = 1.0
MAX_IMAGES = 54


def source(context, paths):
    encoder_observation(context)
    return {'context':copy.deepcopy(context), 'paths':list(map(str,paths)),
            'sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]}


def is_placement(goal):
    return goal['relation'] in PLACEMENTS and not goal.get('preserved')


def explicit_counterexample(row, proof):
    change = row['change']
    return (row['identity'] == 'matched' and row['object_key'] == proof['object_key']
            and row['relation'] == 'outside' and change['kind'] in ('removed','moved','dropped')
            and change['same_object'] is True and bool(change['frames'])
            and bool(change['reason'].strip()))


def update_memory(definitions, memory, rows, timestamp, event_id=None, provenance=None):
    """Never infer removal from missing identity/visibility, or from a bare label."""
    result = copy.deepcopy(memory)
    owners = {}
    for goal, row in zip(definitions, rows):
        if row['identity']=='matched' and row['relation']=='at_destination':
            owners.setdefault(row['object_key'],set()).add(goal['target_id'])
    for goal, row in zip(definitions, rows):
        if not is_placement(goal):
            continue
        gid=goal['id']; proof=result.get(gid)
        if proof and proof['status']=='confirmed' and timestamp > proof['timestamp'] and explicit_counterexample(row,proof):
            proof.update(status='revoked', revoked_at=timestamp,
                         counterexample=copy.deepcopy(row['change']),
                         revocation_provenance=copy.deepcopy(provenance))
        if (event_id and row['identity']=='matched' and row['relation']=='at_destination'
                and row['supported'] is True and len(owners[row['object_key']])==1):
            if proof and proof['status']=='confirmed':
                # Preserve the original proof/identity; an unrelated opening adds nothing.
                continue
            result[gid]={'id':gid,'target_id':goal['target_id'],'destination_id':goal.get('destination_id'),
                         'object_key':row['object_key'],'event_id':event_id,'timestamp':timestamp,
                         'status':'confirmed','observation':copy.deepcopy(row),
                         'provenance':copy.deepcopy(provenance)}
    return result


def evaluate(definitions, memory, rows):
    effective=[]
    for goal,row in zip(definitions,rows):
        proof=memory.get(goal['id']); known=proof is not None and proof['status']=='confirmed'
        matched=row['identity']=='matched'
        direct=matched and row['relation'] in ('at_destination','outside')
        history=[]
        if goal['relation'] in STATES:
            direct=matched and row['observed_state']!='unknown'
            achieved=(row['observed_state']==STATES[goal['relation']]) if direct else None
        elif known and is_placement(goal):
            # Uncertain/distractor is no new evidence. A contradictory same-object
            # outside observation without a grounded change is held for review,
            # not silently deleted and not allowed to pass.
            unresolved=(matched and row['object_key']==proof['object_key'] and row['relation']=='outside')
            achieved=None if unresolved else True
            if achieved is True:history=[proof['event_id']]
        else:
            achieved=(row['relation']=='at_destination') if direct else None
        item={'id':goal['id'],'achieved':achieved,
              'target_matches_definition':True if history else (True if matched else None),
              'current_evidence':direct, 'evidence_event_ids':history,
              'object_key':proof['object_key'] if history else row['object_key'], 'reason':row['reason']}
        if history:
            item.update(history_event_id=history[0],history_event_ids=history,
                        history_mode='confirmed_memory')
            item['reason']+='；保留已确认放置证据 '+history[0]
        effective.append(item)
    owners={}
    for goal,row in zip(definitions,effective):
        if row['achieved'] is True:
            owners.setdefault(row['object_key'],set()).add(goal['target_id'])
    for row in effective:
        if row['achieved'] is True and len(owners[row['object_key']])>1:
            row['achieved']=None;row['reason']+='；不同目标不能重复计同一物体'
    return effective


def build_prompt(packet):
    return '\n'.join([
        '观察机器人照片中的实际事实，程序负责判定完整任务。每项goal输出一条observation，顺序与goals一致。',
        '图片按frame/view/time标注：wrist腕部，agentview和sideview两个外部视角。reference仅用于辨认身份；time=-1表示旧参考图时间未知。memory为此前一次真实松手检查确认的放置记录及原图，不是本次待重新评分的目标。',
        '先辨认指定目标和干扰物：用reference和memory对照当前。黄色把手、标签配色、形状可识别即可，不要求每帧读出标签字样。不能把仍在桌上的其他杯或包装盒当成目标。两件同类目标必须对应两件不同实物，object_key要稳定沿用memory里的标识。',
        'role=release_review时，仅检查这一次release/current所见的目标关系；空手开爪、操作容器后的开爪可能与某放置目标无关，无关或遮挡返回unknown，不能否定之前的放置。before/after严格限相邻1秒，缺失就是缺失，不能臆造动作连续性。',
        '放置项relation只填at_destination、outside、unknown。at_destination指确见指定目标满足所要求in/on/right_of关系；outside指明确同一目标不在目的地；unknown指遮挡/无法辨认/本次无关。on盘是盘面，不额外要求入盘内部。current仍夹持不否定到位，松手/退离/归位由后续流程执行。',
        '只有release_review的at_destination并且supported=true才会存为放置证据。supported表示看见物体在目的地得到支撑，不是正在掉落或容器入口外；实测开口只证明张开，不能证明成功放置。',
        '已有memory时，当前看不清、同一目标暂时辨认不出、后续无关开爪都不撤销。不要输出broken之类推断标签。只有明确看到同一个目标离开原目的地，才填change.kind=removed/moved/dropped；必须确认same_object=true、引用当前反证frame并说明目标和位置如何变化。其他情况kind=none。',
        '装置状态项先报告实际看到的observed_state=open/closed/on/off/unknown，由程序比较目标。只看current图判断状态，不能用memory里的旧门状态。辨认出打开的门也是identity=matched。侧面/斜视只要门板相对机身的位置可辨即可，不要求正面；黑色机身/背板不是门，不能因看不到炉腔就推断关门。',
        '放置项observed_state=unknown；装置状态项relation=unknown。reason只说明本目标的图像依据，不猜测其他检查项。状态只能有一个事实，不得同时解释为开和关。',
        'frames只列release/current对应本项关系或状态的图；identity_frames可引用本次参考/记忆/当前图。明确判断需引用照片，看不清允许空数组。change.frames只允许current图。identity=matched必须有明确object_key和identity_frames；uncertain/distractor不算明确目标反证。',
        '只输出JSON：{"observations":[{"id":"goal_0","identity":"matched|uncertain|distractor","object_key":"同一实物标识或null","relation":"at_destination|outside|unknown","observed_state":"open|closed|on|off|unknown","supported":true或false或null,"frames":[整数],"identity_frames":[整数],"change":{"kind":"none|removed|moved|dropped","same_object":true或false或null,"frames":[整数],"reason":"具体反证，没有则空串"},"reason":"简短画面依据"}],"reason":"简短说明"}。',
        json.dumps(packet,ensure_ascii=False,separators=(',',':')),
    ])


class XvlaEvidenceMemoryClient(XvlaQwenClient):
    request_options={'enable_thinking':True,'thinking_budget':1024,'max_tokens':3072}

    def set_check_contract(self,context):
        previous=getattr(self,'_contract_signature',None)
        super().set_check_contract(context)
        if previous!=self._contract_signature:
            self._reference=None;self._recent=deque(maxlen=4)
            self._events={};self._memory={};self._states=[];self._classified={}
            self._last_capture_result=None

    def remember_identity_images(self,paths,views):
        if self._reference is not None or tuple(views)!=self.required_views or len(paths)!=3:
            raise ValueError('one three-view identity reference required')
        self._reference={'paths':list(map(str,paths)),
            'sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],
            'context':{'sensor_frames':[{'frame':i,'view':v,'timestamp':-1} for i,v in enumerate(views)]}}

    def remember_observation(self,capture,context):
        paths=[str(f['rgb_path']) for f in capture_frames(capture)]
        self._validate_source(capture,context,paths);self.set_check_contract(context)
        item=source(context,paths);time=capture['timestamp']
        if self._reference is None:self._reference=item
        if self._recent and time<=self._recent[-1]['context']['actuator_measurements']['timestamp']:return
        self._recent.append(item)
        for event in self._events.values():
            delta=time-event['time']
            if event['after'] is None and .2<=delta<=WINDOW_SECONDS and event['status']=='pending':
                event['after']=item

    def remember_release(self,capture,context,event_id):
        super().remember_release(capture,context,event_id)
        if event_id in self._events:return
        if event_id not in self._timeline.releases:return
        time=capture['timestamp']
        near=[s for s in self._recent if 0<time-s['context']['actuator_measurements']['timestamp']<=WINDOW_SECONDS]
        self._events[event_id]={'time':time,'before':near[-1] if near else None,
            'release':source(context,self._timeline.releases[event_id]['image_paths']),
            'after':None,'status':'pending'}

    def note_release_status(self,event_id,status):
        super().note_release_status(event_id,status)
        if event_id in self._events:self._events[event_id]['status']=status

    def _load_images(self,paths):
        if not isinstance(paths,list) or not 1<=len(paths)<=MAX_IMAGES:
            raise ValueError('evidence image budget exceeded')
        result=[]
        for i in range(0,len(paths),6):result.extend(SimpleQwenClient._load_images(self,paths[i:i+6]))
        return result

    def _packet(self,context,paths,event_id=None):
        time=context['actuator_measurements']['timestamp']
        packet={'protocol':PROTOCOL,'role':'release_review' if event_id else 'current_review',
            'event_id':event_id,'goals':list(self._check_definitions.values()),'groups':[],
            'memory':[],'previous_checks':copy.deepcopy([r for r in self._states if r['time']<=time][-10:])}
        frames=[];images=[];keys={};retained_sources=[]
        def add(item,kind,eid=None):
            if item is None:return []
            own=item['context'];stamp=own['sensor_frames'][0]['timestamp']
            if stamp>time:raise ValueError('future evidence')
            indices=[]
            for frame,path,digest in zip(own['sensor_frames'],item['paths'],item['sha256']):
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('evidence image bytes changed')
                key=(frame['timestamp'],frame['view'],digest)
                if key not in keys:
                    keys[key]=len(frames);frames.append({**frame,'frame':len(frames)});images.append(path)
                indices.append(keys[key])
            group={'kind':kind,'event_id':eid,'time':stamp,'frames':indices}
            if 'actuator_measurements' in own:group['encoder']=encoder_observation(own)
            else:group['time_known']=False
            packet['groups'].append(group)
            return indices
        add(self._reference,'reference')
        for proof in self._memory.values():
            if proof['timestamp']>time:raise ValueError('future placement proof')
            indices=[]
            for item in proof['provenance']['sources']:
                indices+=add(item,'memory',proof['event_id'])
            packet['memory'].append({k:copy.deepcopy(proof[k]) for k in ('id','target_id','object_key','event_id','timestamp','status')})
            packet['memory'][-1]['evidence_frames']=sorted(set(indices))
        event=self._events[event_id] if event_id else None
        relation=[]
        if event:
            for kind in ('before','release','after'):
                item=event[kind];indices=add(item,kind,event_id)
                if kind!='before':relation+=indices
                if item is not None:retained_sources.append(item)
            packet['window']={'max_gap_seconds':WINDOW_SECONDS,'before_available':event['before'] is not None,
                              'after_available':event['after'] is not None}
        current=source(context,paths)
        packet['current_frames']=add(current,'current')
        packet['relation_frames']=sorted(set(relation+packet['current_frames']))
        packet['sensor_frames']=frames
        if len(images)>MAX_IMAGES:raise ValueError('evidence image budget exceeded')
        return packet,{**context,'sensor_frames':frames},images,retained_sources

    def _source_key(self,context,paths):
        return (context['actuator_measurements']['timestamp'],
                tuple(context['actuator_measurements']['finger_qpos']),
                tuple(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths))

    def _inspect(self,context,paths,event_id=None):
        self.set_check_contract(context)
        packet,request_context,request_paths,sources=self._packet(context,paths,event_id)
        self._active_packet=packet
        prompt=build_prompt(packet)
        result=self._request(request_context,request_paths,prompt)
        rows=copy.deepcopy(self.last_observations['observations'])
        time=context['actuator_measurements']['timestamp']
        provenance={'sources':copy.deepcopy(sources),
            'request_sha256':result.get('request_sha256'),'response_sha256':result.get('response_sha256'),
            'call_index':result.get('call_index'),'packet':copy.deepcopy(packet)}
        self._memory=update_memory(list(self._check_definitions.values()),self._memory,rows,time,event_id,provenance)
        effective=evaluate(list(self._check_definitions.values()),self._memory,rows)
        reason=self.last_observations['reason']
        result['verdict']=SimpleQwenClient.parse_verdict(self,json.dumps({'goal_achieved':self._aggregate(effective),'reason':reason}))
        result.update(checker=PROTOCOL,prompt_text=prompt,input_context=request_context,
            request_image_paths=request_paths,event_packet=packet,event_observations=copy.deepcopy(self.last_observations),
            effective_checks=effective,process_evidence=self.process_evidence_snapshot(),
            raw_visual_goal_verdict=copy.deepcopy(self.last_observations),
            visual_goal_verdict={'checks':effective,'goal_achieved':self._aggregate(effective),'reason':reason})
        self.last_effective_checks=copy.deepcopy(effective)
        self._last_capture_result=(self._source_key(context,paths),copy.deepcopy(result))
        self._states.append({'time':time,'role':packet['role'],'event_id':event_id,
            'goals':[{k:r[k] for k in ('id','achieved','object_key','evidence_event_ids')} for r in effective]})
        return result

    def analyze_release(self,context,image_paths,event_id):
        self._validate_capture(context,image_paths)
        self.remember_release(self._capture,context,event_id)
        if event_id in self._classified:return copy.deepcopy(self._classified[event_id])
        if event_id not in self._events:raise ValueError('release exceeds retained event budget')
        event=self._events[event_id];latest=event['after'] or event['release']
        result=self._inspect(latest['context'],latest['paths'],event_id)
        result.update(observation_role='release_snapshot',event_id=event_id)
        self._classified[event_id]=copy.deepcopy(result)
        self.note_release_status(event_id,'classified')
        return result

    def analyze(self,context,image_paths):
        self._validate_capture(context,image_paths);self.set_check_contract(context)
        key=self._source_key(context,image_paths)
        # Exactly the same synchronized capture cannot acquire a second contrary
        # device-state vote. Reuse the observation, not a later/older state.
        if self._last_capture_result and self._last_capture_result[0]==key:
            result=copy.deepcopy(self._last_capture_result[1]);result['reused_same_capture']=True
            return result
        return self._inspect(context,image_paths)

    def process_evidence_snapshot(self):
        return copy.deepcopy(list(getattr(self,'_memory',{}).values()))

    def parse_verdict(self,content):
        raw=vision._parse_verdict_content(content);packet=self._active_packet;rows=raw.get('observations')
        if not isinstance(rows,list) or [r.get('id') for r in rows]!=self._check_ids or not isinstance(raw.get('reason'),str):
            raise ValueError('complete ordered observations required')
        all_frames=set(range(len(packet['sensor_frames'])));current=set(packet['current_frames'])
        def refs(value,allowed):
            return isinstance(value,list) and all(type(n) is int for n in value) and len(set(value))==len(value) and set(value)<=allowed
        for goal,row in zip(packet['goals'],rows):
            if row.get('identity') not in ('matched','uncertain','distractor') or row.get('relation') not in ('at_destination','outside','unknown'):
                raise ValueError('invalid identity or relation')
            if row.get('observed_state') not in ('open','closed','on','off','unknown') or not isinstance(row.get('reason'),str):
                raise ValueError('invalid observed state or reason')
            if not isinstance(row.get('object_key'),(str,type(None))):raise ValueError('invalid object key')
            if 'supported' not in row or (row['supported'] is not None and type(row['supported']) is not bool):raise ValueError('invalid support')
            allowed=current if goal['relation'] in STATES or goal.get('preserved') else set(packet['relation_frames'])
            if not refs(row.get('frames'),allowed) or not refs(row.get('identity_frames'),all_frames):raise ValueError('invalid source frames')
            if row['identity']=='matched' and (not row['object_key'] or not row['identity_frames']):raise ValueError('identity anchor missing')
            if (row['relation']!='unknown' or row['observed_state']!='unknown') and not row['frames']:raise ValueError('fact source missing')
            if goal['relation'] in STATES:
                if row['relation']!='unknown':raise ValueError('state goal must report observed state')
                domain=('open','closed','unknown') if goal['relation'] in ('open','close') else ('on','off','unknown')
                if row['observed_state'] not in domain:raise ValueError('wrong state domain')
            elif row['observed_state']!='unknown':raise ValueError('placement must not vote on device state')
            change=row.get('change')
            if not isinstance(change,dict) or change.get('kind') not in ('none','removed','moved','dropped') or not isinstance(change.get('reason'),str):raise ValueError('invalid change')
            if 'same_object' not in change or (change['same_object'] is not None and type(change['same_object']) is not bool) or not refs(change.get('frames'),current):raise ValueError('invalid counterexample source')
            if change['kind']!='none' and (not change['frames'] or not change['reason'].strip()):raise ValueError('counterexample needs actual source')
        self.last_observations=copy.deepcopy(raw)
        effective=evaluate(packet['goals'],self._memory,rows)
        return SimpleQwenClient.parse_verdict(self,json.dumps({'goal_achieved':self._aggregate(effective),'reason':raw['reason']}))
