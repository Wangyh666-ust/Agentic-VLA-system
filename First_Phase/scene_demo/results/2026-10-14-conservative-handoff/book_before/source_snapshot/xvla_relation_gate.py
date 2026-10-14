"""Task-relation evidence for handoff; never commands robot motion."""
from scene_demo.xvla_evidence_memory import is_placement


def validate_placement(row, goal, frames, refs):
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
    detail['destination_evidence'] = refs(detail.get('destination_evidence'), frames)
    detail['relation_evidence'] = refs(detail.get('relation_evidence'), frames)
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


def placement_ready(row, region):
    detail = row.get('placement', {})
    views = detail.get('views', [])
    satisfied = {v['frame'] for v in views if v['relation'] == 'satisfied'}
    evidence = {r['frame'] for r in detail.get('relation_evidence', [])}
    return (region is not None and region.get('confirmed') is True
            and row.get('same_target') is True and row.get('state') == 'at_destination'
            and row.get('supported') is True
            and detail.get('region_confirmed') is True
            and detail.get('needs_motion') is False and detail.get('contradiction') is False
            and bool(satisfied & evidence)
            and not any(v['relation'] == 'violated' for v in views))


PROMPT = '''先报告可见关系，再决定是否到位。严格区分物体身份、目的地区域、允许接管和松手后完成。
仍被夹持可以已经到位；仅到达上方、入口附近、口沿或仍需继续下降／推进，不能因此判定放置动作结束。容器旁边、入口外、二维投影重叠不等于进入内部；碰到盒壁不证明目的地承接。不能从初始位置消失推断放置成功。
按任务原有语义检查：in检查指定内部/隔间；on检查指定承载面；right_of检查正确参照物右侧区域且物体已受桌面等支撑。不能把盘面改为盘内，不能要求物体完全低于容器口沿、居中、额外关门。门/开关不适用放置支撑要求。
每个放置goal在初始模式必须给regions；destination_id是具体任务区域，不等于整个容器身份。description说明是哪一格/面/侧，boundary说明实际可见的边沿、隔板或参照物如何界定它，anchors定位该区域而非只框整个装置。看不清可confirmed=false，不得依据ID猜位置。
regions结构为[{"destination_id":"goals里的目的地ID","confirmed":true或false,"description":"具体任务区域","boundary":"可见边界与参照","anchors":[{"frame":整数,"box":[x1,y1,x2,y2]}]}]。首次必须逐一提供；后续可省略沿用。纠正旧区域必须加corrects_previous=true和correction_reason，并重新审核旧放置证据。
所有current/history的at_destination放置回答必须加placement对象：{"region_confirmed":true或false或null,"region_description":"任务指定区域","destination_evidence":[{"frame":整数,"box":[x1,y1,x2,y2]}],"boundary_observation":"目标相对入口/边界/承载面的直接可见事实","relation_evidence":[{"frame":整数,"box":[x1,y1,x2,y2]}],"support_observation":"实际承接位置；区分容器/台面与夹爪，不能把贴近当接触","needs_motion":true或false或null,"contradiction":true或false或null,"views":[{"frame":整数,"relation":"satisfied/violated/occluded/unclear","reason":"此视角实际看见什么"}]}。
views必须包含该时刻全部三图；一图明确成立且其余遮挡可接受，任何同刻明确反证都不能忽略。引用必须来自所审核时刻。身份清楚而关系不清时same_target=true、state=unknown；不要混淆身份未知和关系未知。
supported只指目的地/台面实际承接，不是夹爪支撑。needs_motion指是否还需继续放置移动，不包含松手、退离和归位。放手前不要求已经松手、退离或证明松手后稳定；这些在后续检验。若只看见靠近、悬空或仍需推进，应outside；无法确定深度/承接应unknown，不猜true。
若decision_stage=handoff_review，这是独立接管复核：依据原图与最近过程确认是否仍需定位移动，不继承之前的到位结论；不得因将要收尾而认为已经完成。'''
