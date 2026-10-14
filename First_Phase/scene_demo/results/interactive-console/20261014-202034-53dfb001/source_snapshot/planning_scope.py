"""Capability coverage metadata; never changes robot goals or control."""
import copy

PLANNING_SCOPE_GUIDANCE = """[request coverage]
先理解完整用户需求，再按公开动作库区分能执行的部分与未支持的部分。
若有明确推进原需求的准备步骤或独立子目标，可先执行这些部分，不因缺少后续能力而全部拒绝；必须明确说出未完成步骤。
部分执行不能改变原意、对象或目的地，不能为了使用现有技能而添加无关动作。例如把物品放进收纳篮不等于丢弃；没有证据表明篮子是垃圾桶时，不得把入篮当成丢弃的准备步骤。
例如加热所需的入炉和关门可以作为准备步骤；若库中没有启动加热，只提交准备步骤并说明不能启动加热，不能声称完全覆盖或完成加热。
提交 submit_scene_plan 时填写 request_scope={"coverage":"full|partial|none","executable_outcomes":["所选技能能实现的结果"],"unmet_outcomes":["当前不能实现或待澄清的结果"],"reason":"与原需求的关系及能力缺口"}。
full：完整需求都有对应能力，unmet_outcomes=[]；partial：能执行有用部分且仍有缺口，两项列表均非空；none：无法确定任何可执行部分，executable_outcomes=[]，说明缺失能力或歧义。
full/partial 使用 decision="execute" 和非空能力列表；none 使用 unsupported 或 clarify 和空列表。复合请求中某一项不支持，不自动否定其他明确可执行且互不依赖的子目标。
只有本次明确选择的可执行部分，其动作、对象、目的地与顺序均匹配原生长程技能时才优先调用该技能；不得把“部分匹配”写成完整需求“完全匹配”。
rationale 和提交后的用户回答都必须用自然语言说明执行范围与剩余限制，不能只输出能力编号或“已提交”。提交不等于执行完成。
检查器仍按所选技能原有判据验收，不增加库中不存在的动作；plan_success仅代表所选动作目标，不能替代完整用户需求的完成情况。
修复时保留原request_scope及未支持步骤，只修复已声明的执行部分，不得借修复把partial改成full。
"""


def validate_request_scope(value, decision):
    """Optional for legacy callers; supplied metadata must be consistent."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        'coverage', 'executable_outcomes', 'unmet_outcomes', 'reason'
    }:
        raise ValueError('request_scope requires coverage, executable_outcomes, unmet_outcomes, reason')
    coverage = value['coverage']
    if coverage not in ('full', 'partial', 'none'):
        raise ValueError('invalid request coverage')
    for name in ('executable_outcomes', 'unmet_outcomes'):
        if not isinstance(value[name], list) or any(not isinstance(s, str) or not s.strip() for s in value[name]):
            raise ValueError(name + ' must be a list of nonempty strings')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('request_scope reason required')
    executable, unmet = value['executable_outcomes'], value['unmet_outcomes']
    if coverage == 'full' and (not executable or unmet or decision != 'execute'):
        raise ValueError('full coverage requires executable outcomes only and execute')
    if coverage == 'partial' and (not executable or not unmet or decision != 'execute'):
        raise ValueError('partial coverage requires executable and unmet outcomes and execute')
    if coverage == 'none' and (executable or not unmet or decision not in ('clarify', 'unsupported')):
        raise ValueError('none coverage requires unmet outcomes only and clarify/unsupported')
    return copy.deepcopy(value)


def request_fulfilled(scope, state, plan_success):
    """Do not promote completed preparation into completed user intent."""
    if scope is None:
        return None
    if scope['coverage'] != 'full':
        return False
    return plan_success if state == 'completed' else None
