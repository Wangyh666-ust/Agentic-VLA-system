"""Task-relation evidence for handoff; never commands robot motion."""
import math
from scene_demo.xvla_evidence_memory import is_placement

# Extra evidence for occluded-contact inference, not a placement-success oracle
# or the final settling tolerance. Keep the same limit across task identities.
INFERRED_HANDOFF_MAX_ARM_SPEED = .2  # rad/s


def motion_measurement(observation):
    measured = observation.get('context', {}).get('actuator_measurements', {})
    value = measured.get('max_abs_arm_qvel')
    stamp = measured.get('timestamp')
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or type(stamp) not in (int, float) or not math.isfinite(stamp)
            or observation.get('captured_at') is None
            or abs(stamp - observation['captured_at']) > 1e-9):
        return None
    return {'timestamp': stamp, 'max_abs_arm_qvel': value, 'unit': 'rad/s'}


def validate_placement(row, goal, frames, refs, observations=()):
    """Require observable relation evidence, not a confident aggregate boolean."""
    if not is_placement(goal) or row['state'] != 'at_destination':
        return
    detail = row.get('placement')
    if not isinstance(detail, dict):
        raise ValueError('at_destination requires placement assessment')
    for name in ('region_description', 'boundary_observation', 'support_observation'):
        if not isinstance(detail.get(name), str) or not detail[name].strip():
            raise ValueError('placement ' + name + ' missing')
    for name in ('region_confirmed', 'needs_motion', 'contradiction'):
        if detail.get(name) is not None and type(detail[name]) is not bool:
            raise ValueError('placement ' + name + ' must be bool/null')
    current = next((o for o in observations if set(o['frames']) == frames), None)
    now = current.get('captured_at') if current else None
    earlier_frames = {f for o in observations if now is not None
                      and o.get('captured_at') is not None and o['captured_at'] < now
                      for f in o['frames']}
    # Preserve supplemental, correctly timestamped references without letting
    # them substitute for current evidence or silently relabelling old pixels.
    detail['supplemental_context_evidence'] = {}
    for field in ('destination_evidence', 'relation_evidence'):
        supplied = refs(detail.get(field), frames | earlier_frames)
        detail[field] = [r for r in supplied if r['frame'] in frames]
        detail['supplemental_context_evidence'][field] = [r for r in supplied if r['frame'] not in frames]
    if not detail['destination_evidence'] or not detail['relation_evidence']:
        raise ValueError('placement requires current destination and relation evidence')
    views = detail.get('views')
    if (not isinstance(views, list) or len(views) != len(frames)
            or {v.get('frame') for v in views if isinstance(v, dict)} != frames):
        raise ValueError('placement must assess all synchronized views')
    for view in views:
        if (view.get('relation') not in ('satisfied', 'violated', 'occluded', 'unclear')
                or not isinstance(view.get('reason'), str) or not view['reason'].strip()):
            raise ValueError('invalid per-view relation assessment')
    contact=detail.get('support_contact')
    if not isinstance(contact,dict) or 'visible' not in contact or (contact['visible'] is not None and type(contact['visible']) is not bool):
        raise ValueError('support_contact.visible must be bool/null')
    if not isinstance(contact.get('observation'),str) or not contact['observation'].strip():
        raise ValueError('support contact observation required')
    contact['evidence']=refs(contact.get('evidence',[]),frames)
    if contact['visible'] is True and not contact['evidence']:
        raise ValueError('visible support contact requires contact image references')
    # A model cannot grant itself this marker. Validate the alternate evidence
    # against the actual supplied captures, including their sampling times.
    detail.pop('_geometry_sequence_validated', None)
    detail.pop('_inferred_motion_settled', None)
    support = detail.get('support_inference')
    if support is not None:
        if not isinstance(support, dict):
            raise ValueError('support_inference must be an object')
        if support.get('depth_sufficient') is not None and type(support['depth_sufficient']) is not bool:
            raise ValueError('support_inference.depth_sufficient must be bool/null')
        for field in ('spatial_observation', 'temporal_observation'):
            if not isinstance(support.get(field), str) or not support[field].strip():
                raise ValueError('support_inference ' + field + ' missing')
        current = next((o for o in observations if set(o['frames']) == frames), None)
        now = current.get('captured_at') if current else None
        eligible = [o for o in observations if now is not None and o.get('captured_at') is not None
                    and o['captured_at'] <= now and o['role'] != 'initial']
        allowed = {f for o in eligible for f in o['frames']}
        support['evidence'] = refs(support.get('evidence', []), allowed)
        cited = {r['frame'] for r in support['evidence']}
        detail['_geometry_sequence_validated'] = (
            support.get('depth_sufficient') is True
            and bool(cited & frames)
            and any(0 < now - o['captured_at'] <= 1.0 and cited & set(o['frames']) for o in eligible))
        neighbours = [o for o in eligible if 0 <= now - o['captured_at'] <= 1.0]
        detail['_inferred_motion_settled'] = (
            len({o['captured_at'] for o in neighbours}) >= 2
            and all(o.get('motion') is not None
                    and o['motion']['max_abs_arm_qvel'] <= INFERRED_HANDOFF_MAX_ARM_SPEED
                    for o in neighbours))


def placement_ready(row, region, before_release=False):
    detail = row.get('placement', {})
    views = detail.get('views', [])
    satisfied = {v['frame'] for v in views if v['relation'] == 'satisfied'}
    evidence = {r['frame'] for r in detail.get('relation_evidence', [])}
    inferred = (detail.get('_geometry_sequence_validated') is True
                and detail.get('_inferred_motion_settled') is True)
    return (region is not None and region.get('confirmed') is True
            and row.get('same_target') is True and row.get('state') == 'at_destination'
            and (row.get('supported') is True or
                 (before_release and row.get('supported') is None
                  and inferred))
            and (not before_release or detail.get('support_contact',{}).get('visible') is True
                 or inferred)
            and detail.get('region_confirmed') is True
            and detail.get('needs_motion') is False and detail.get('contradiction') is False
            and bool(satisfied & evidence)
            and not any(v['relation'] == 'violated' for v in views))


STATE_PROMPT = '''门/盖/抽屉状态必须依据当前活动部件相对固定机身的几何位置判断。门窗可能透明、半透明或无明显反光，透过窗口看见内部物品/腔体不等于门打开；内部不可见也不等于门关闭。即使外观提示写到“入口可见”，也不能将透过窗户看见内部当作敞开的入口。
逐一比较当前和初始图中的门板边框、把手、铰链侧与机身开口的位置：门板是否回到关闭平面、把手是否回到对应位置，还是门板确实向外转出/抽屉伸出。正面或侧面都可提供依据。comparison必须说明这些实际位置证据；无法分辨活动件位置则unknown，不以能否看见里面的目标代替。此规则适用于所有门盖抽屉任务，不增加额外动作条件。'''


PROMPT = '''先报告可见关系，再决定是否到位。严格区分物体身份、目的地区域、允许接管和松手后完成。
仍被夹持可以已经到位；仅到达上方、入口附近、口沿或仍需继续下降／推进，不能因此判定放置动作结束。容器旁边、入口外、二维投影重叠不等于进入内部；碰到盒壁不证明目的地承接。不能从初始位置消失推断放置成功。
按任务原有语义检查：in检查指定内部/隔间；on检查指定承载面；right_of检查正确参照物右侧区域且物体已受桌面等支撑。不能把盘面改为盘内，不能要求物体完全低于容器口沿、居中、额外关门。门/开关不适用放置支撑要求。
每个放置goal在初始模式必须给regions；destination_id是具体任务区域，不等于整个容器身份。description说明是哪一格/面/侧，boundary说明实际可见的边沿、隔板或参照物如何界定它，anchors定位该区域而非只框整个装置。看不清可confirmed=false，不得依据ID猜位置。
regions结构为[{"destination_id":"goals里的目的地ID","confirmed":true或false,"description":"具体任务区域","boundary":"可见边界与参照","anchors":[{"frame":整数,"box":[x1,y1,x2,y2]}]}]。首次必须逐一提供；后续可省略沿用。纠正旧区域必须加corrects_previous=true和correction_reason，并重新审核旧放置证据。
所有current/history的at_destination放置回答必须加placement对象：{"region_confirmed":true或false或null,"region_description":"任务指定区域","destination_evidence":[{"frame":整数,"box":[x1,y1,x2,y2]}],"boundary_observation":"目标相对入口/边界/承载面的直接可见事实","relation_evidence":[{"frame":整数,"box":[x1,y1,x2,y2]}],"support_observation":"实际承接位置；区分容器/台面与夹爪，不能把贴近当接触","needs_motion":true或false或null,"contradiction":true或false或null,"views":[{"frame":整数,"relation":"satisfied/violated/occluded/unclear","reason":"此视角实际看见什么"}]}。
views必须包含该时刻全部三图；一图明确成立且其余遮挡可接受，任何同刻明确反证都不能忽略。引用必须来自所审核时刻。身份清楚而关系不清时same_target=true、state=unknown；不要混淆身份未知和关系未知。
placement还必须包含support_contact={"visible":true或false或null,"observation":"实际可见接触，或接触处被什么遮挡","evidence":[{"frame":整数,"box":[x1,y1,x2,y2]}]}。visible只记录直接可见性，不等于是否已承接；接触处被容器壁/物体自身遮挡时填false或null，不能因此自动判未到位。夹爪闭合只说明开口状态，不证明物体悬空，也不妨碍目的地已经承接。
松手前有两条通用证据路径：直接可见承接接触；或指定区域的多视角几何关系结合最近连续过程，足以确认放置深度已到位、无需继续下降或推进。后者须在placement增加support_inference={"depth_sufficient":true或false或null,"spatial_observation":"结合物体整体尺度、可见露出部分、入口/边沿/隔板/承载平面说明实际深度及有无悬空间隙","temporal_observation":"同一目标相对固定目的地从接近到当前的变化或已到位状态如何延续，不能仅说机械臂停住","evidence":[{"frame":整数,"box":[x1,y1,x2,y2]}]}。引用必须包括被审核时刻以及此前1秒内至少一个非初态的过程时刻，不得用初始桌面位置或后来图片证明当时已到位。全部同步视角共同核对，一图清晰、其他被遮挡可接受；任何明确反证均不能忽略。无法提供足够几何/时序依据时保持unknown。
几何深度必须先成立，静止不能补足深度证据。仅物体尖端/下端刚越入口、主体仍明显悬在外部、卡在口沿或需要继续降低/推进，均不满足接管；不能把前述情况仅因停住而写成已承接。对in检查实际进入指定内部/隔间的深度，对on/right_of检查相对指定承载平面的高度与间隙，不要求目标完全低于口沿。位置投影重叠、源位置消失和夹爪握持都不能单独证明这些关系。
support_inference可额外引用同任务段更早的接近/进入照片，以比较进入前后物体整体与边界的关系；但仍须引用当前及此前1秒内的近邻图核对现在没有移出或仍需推进。current.evidence、destination_evidence、relation_evidence只能引用当前时刻；更早的过程引用只填support_inference.evidence或identity_refs，不能混填当前事实。
observations.motion是与各组三视图同步的实测关节速度。它不能证明物体到位，但能核对“已经停在最终放置深度”的说法：接触遮挡而依赖几何/过程推断时，程序要求当前和至少一个1秒内近邻样本的最大关节速度均不高于0.2 rad/s（缺测则此路径暂不接管）。若连续图仍在下降/推进、速度也未减缓，不得以“正在进入”当作“放置结束”。速度低也不证明到位，仍须独立满足身份、目的地区域和几何深度条件；此规则不是最终静止验收阈值。
严格分开位置到位和接触已确认：state=at_destination表示已达到任务要求的放置位置/深度，剩余只需松手，不表示已释放或最终完成。supported只描述目的地承接是否能确认；接触处遮挡可填null，同时state仍可at_destination、needs_motion=false，并用充分的support_inference申请接管。不要为了允许接管编造supported=true；也不要把supported=null自动等同于位置未知。明确悬空、挂在外沿或需要继续放置位移时supported=false/needs_motion=true，必须拦截。
判断放置深度要比较初态中物体整体尺度与当前露出部分、跨越哪条边界、进入前后的相对位置，而不是必须看到落底瞬间或证明零间隙。已明显深入正确区域、主体由内部边界包围且近邻过程保持该关系时，不应仅因想象“也许卡在看不见的边沿”而否认到位；有实际卡沿/间隙/未完成推进的图像依据则须明确引用并拦截。位置或深度本身仍无法判断时保留unknown。
允许接管不等于最终完成。松手后仍须依据脱手及仍留在目标处的过程确认承接和稳定，不能把松手前的几何接管许可写成松手后历史完成证据。needs_motion不包含松手、退离和归位，不能把这些后续动作变成松手前的前置条件。
若decision_stage=handoff_review，这是独立接管复核：依据原图与最近过程确认是否仍需定位移动，不继承之前的到位结论；不得因将要收尾而认为已经完成。'''
