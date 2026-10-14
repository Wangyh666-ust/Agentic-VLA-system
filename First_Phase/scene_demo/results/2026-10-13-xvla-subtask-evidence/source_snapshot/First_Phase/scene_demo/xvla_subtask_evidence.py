"""Immutable observations and versioned opinions for one logical task attempt."""
import copy
import hashlib
import json
import uuid
from pathlib import Path
from scene_demo.finish_v1.simple_checker import encoder_observation

PROTOCOL = 'xvla_subtask_review_v7'
MAX_IMAGES = 54

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def goal_key(goal):
    return json.dumps([goal['target_id'],goal['relation'],goal.get('destination_id'),bool(goal.get('preserved'))])

def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0

class SubtaskEvidence:
    def __init__(self,scope=None,directory=None):
        self.scope=copy.deepcopy(scope or {'session_id':'local','logical_subtask_id':uuid.uuid4().hex,'attempt_id':'1'})
        if set(self.scope)!={'session_id','logical_subtask_id','attempt_id'} or any(not isinstance(v,str) or not v for v in self.scope.values()):
            raise ValueError('explicit nonempty task scope required')
        self.scope_id=hashlib.sha256(json.dumps(self.scope,sort_keys=True).encode()).hexdigest()[:16]
        self.directory=Path(directory) if directory is not None else None
        self.observations={};self._source_keys={};self.initial_id=None;self.events={}
        self.bindings={};self.proofs={};self.reviews=[];self.revision=0
        self.initial_review_done=False;self.repair_used=False;self.cache={}

    def append(self,kind,data):
        if self.directory is not None:
            self.directory.mkdir(parents=True,exist_ok=True)
            with (self.directory/(kind+'.jsonl')).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(data,ensure_ascii=False)+'\n')

    def observe(self,context,paths,role='observation',unknown_time=False):
        hashes=[digest(p) for p in paths]
        if len(paths)!=3:raise ValueError('three source images required')
        stamp=None if unknown_time else encoder_observation(context)['timestamp']
        feedback=None if unknown_time else encoder_observation(context)
        key=json.dumps([stamp,context['sensor_frames'],hashes,feedback],sort_keys=True)
        if key in self._source_keys:return self.observations[self._source_keys[key]]
        cid=self.scope_id+'/capture_%06d'%(len(self.observations)+1)
        item={'capture_id':cid,'scope_id':self.scope_id,'captured_at':stamp,
              'sequence_index':len(self.observations),'sampling_role':role,
              'context':copy.deepcopy(context),'paths':list(map(str,paths)),
              'sha256':hashes,'encoder':feedback}
        self.observations[cid]=item;self._source_keys[key]=cid;self.revision+=1
        if self.initial_id is None:self.initial_id=cid
        self.append('observations',item)
        return item

    def release(self,event_id,item):
        if not isinstance(event_id,str) or not event_id:raise ValueError('event id required')
        if not item['encoder'] or not item['encoder']['at_standard_open']:
            raise ValueError('release requires synchronized measured opening')
        if event_id in self.events:
            if self.events[event_id]['capture_id']!=item['capture_id']:raise ValueError('release source changed')
            return
        self.events[event_id]={'event_id':event_id,'capture_id':item['capture_id'],'status':'unclassified'}
        self.revision+=1;self.append('events',self.events[event_id])

    def review(self,record):
        self.reviews.append(copy.deepcopy(record));self.revision+=1;self.append('reviews',record)

def service_evidence_store(session,plan,job):
    """Native skills and a matching atomic decomposition share explicit scope only."""
    from scene_demo.xvla_skills import route_capabilities
    cid=job.capability_id
    ids=getattr(plan,'capability_ids',[cid]);logical=cid
    for group in route_capabilities(session.scene_id,ids)['replacements']:
        if cid in group['component_capability_ids']:
            logical=group['native_capability_id'];break
    request=str(getattr(plan,'request_id',getattr(job,'request_id',job.run_dir)))
    attempt=str(len(getattr(plan,'repair_history',[])))
    scope={'session_id':str(getattr(session,'session_id',getattr(job,'session_id',job.run_dir))),
           'logical_subtask_id':request+'/'+logical,'attempt_id':attempt}
    if not hasattr(session,'_xvla_evidence_stores'):session._xvla_evidence_stores={}
    key=json.dumps(scope,sort_keys=True)
    if key not in session._xvla_evidence_stores:
        session._xvla_evidence_stores[key]=SubtaskEvidence(scope,Path(job.run_dir)/'finish/subtask_evidence')
    return session._xvla_evidence_stores[key]

def build_prompt(packet):
    return '\n'.join([
        '你审核同一个机器人任务段的初态、过程和当前照片。按task和goals指定的实物判断，不能把同类干扰物替代目标。',
        '每张图有frame、capture_id、view、captured_at。三视图是同一时刻；wrist会随手臂移动，不能用画面位置变化直接推断物体移动。初始图时间null仅表示绝对时间未知，顺序在动作之前。',
        '先用初始/清晰图定位各指定目标，bindings给出外观、图内框和易混淆物。框为0到1000归一化[x1,y1,x2,y2]，须包围实际物体。按照goals中外观定义辨认，不能用同类或相似颜色的另一物体替代。已绑定目标需用相同参照物核对，目标名称相同不证明是同一实物。',
        '比较初态、各次松手和当前：逐个核对什么物体从哪里去了哪里。目标只露局部时可结合初态与过程核对身份，不要求每张都看清标签；不能把看不到标签说成容器空置。仅从源位置消失不能证明到达目的地，还需目的地/过程对应。',
        'current只描述当前图可见事实。history单独重审review_requests指定的旧capture，即使旧回答Unknown/False也可纠正。历史图里的放置不能填成当前看得到。历史结论必须引用该旧capture的图；identity_refs可以引用初始、过程或当前，用comparison说明为什么是同一目标。',
        '放置state=at_destination/outside/unknown；supported表示视觉上在目的地受支撑。编码器仅证明对应时刻开口，不能证明位置。当前仍夹持不妨碍到位判断，收尾另行执行。门/抽屉/开关state=open/closed/on/off/unknown，仅用当前图；侧面可辨门板相对机身即可，不要求正面。',
        '已有放置证据遇遮挡时current可unknown；若称同一目标outside，必须核对身份和原图。change=removed/moved/dropped需要当前明确同物移出证据，否则none。不要把桌上另一个目标/瓶/盒当成已放入目标的反证。每件目标分别定位，不能把同一件实物计两次。',
        '只输出JSON。bindings可为空（沿用已有绑定）；首次或之前身份未知时应补充。所有matched=true必须有features和anchors。current按goals顺序；history按review_requests逐条，不能漏项。看不清填unknown和same_target=null，说明缺失信息，不抄模板。',
        '结构：{"bindings":[{"target_id":"目标ID","matched":true,"anchors":[{"frame":整数,"box":[x1,y1,x2,y2]}],"features":"实际可见特征","distractors":[{"frame":整数,"box":[x1,y1,x2,y2],"features":"不同物体的可见差异"}]}],"current":[{"id":"goal_0","state":"unknown","same_target":null,"evidence":[],"identity_refs":[],"supported":null,"change":"none","comparison":"初态与当前实际对比"}],"history":[{"id":"goal_0","capture_id":"指定旧capture_id","state":"unknown","same_target":null,"evidence":[],"identity_refs":[],"supported":null,"change":"none","comparison":"初态、旧过程图与后来图的身份和位置对应"}]}。evidence与identity_refs均为[{"frame":整数,"box":[x1,y1,x2,y2]}]；明确事实必须填写evidence，明确目标身份必须引用bindings参照图的identity_refs。',
        'mode=initial时只输出bindings，current=[]、history=[]。mode=history_page时current=[]，仅审核该页history。旧判断是可修订意见，不是真值。',
        '若旧bindings认错实物，可输出同target_id的新bindings并加corrects_previous=true、correction_reason说明新旧图的具体区别；旧身份所建放置证据必须重新确认，不能凭改名称直接继承。',
        json.dumps(packet,ensure_ascii=False,separators=(',',':')),
    ])
