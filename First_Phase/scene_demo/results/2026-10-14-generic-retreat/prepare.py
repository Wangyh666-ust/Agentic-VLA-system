"""Freeze the authorized retreat-only replay and subsequent timed full run."""
import copy, hashlib, json
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[3]
previous=out.parent/'2026-10-14-native-joint-planning'
base=json.loads((previous/'online_proposal.json').read_text(encoding='utf-8'))
for stage in ('replay','online'):
    p=copy.deepcopy(base)
    p.update(id='20261014-generic-retreat-'+stage, question_key='xvla-drawer-retreat-block',
        purpose='修复通用退离接口；在刚才完整关抽屉的例子上确认实际退离通过，再完整重跑记录总耗时。',
        prior_evidence=['20261014-native-joint-planning'],
        history_review=[{'experiment_id':'20261014-native-joint-planning','disposition':'reuse',
          'reason':'复用完整关抽屉成功但竖直4/6厘米退离均失败的原始动作和状态；原代码及旧半段成功的退离均保留作为对照。'}],
        hypothesis='当前RGBD与机器人几何补充退出方向，可在保持原路径/净空/归位约束的前提下覆盖竖直受阻的结束姿态。',
        information_gain='旧试验仅生成竖直候选；本轮只改变通用候选生成及诊断记录，先确认实际退离有效再测试正常完整链路耗时。',
        success_criteria='预期旧向上两候选失败、新当前几何分支被选择；实际movement>0，当前终点净空通过，碗入抽屉且抽屉关闭始终保持，原home恢复到初始参考。在线另需正常最终视觉通过及5/5，记录载入/规划/执行/全程墙钟时间。不以候选计划ok替代实际运动。',
        failure_criteria='复现不匹配或新候选未触发/实际movement为0为目标未验证，停止后续在线；运动中保护失败、终点不清、目标状态破坏或home失败分别记录。',
        unknown_criteria='重放匹配不足、视觉网络/协议错误或缺少观测，记录未知；不得把未验收当成功。',
        changed_factors=[{'name':'retreat_candidate_interface','before':'无目标框仅尝试上移4/6cm',
            'after':'保留原候选顺序；失败后根据同刻RGBD和机器人几何补充水平/斜向候选，保留逐条筛查诊断',
            'reason':'不依赖任务名称或历史目标框，让通用退离覆盖关抽屉后等贴近障碍的姿态。'}])
    p['fixed_factors'].update(finish_reset='原控制循环、深度、净空阈值、home、5/5不变',helpers=False)
    p['configuration'].update(retreat='legacy_first_then_current_geometry_v1',stage=stage)
    p['repeat']={'needed':True,'reason':'用户明确要求修复后复现退离并完整重跑计时，不重测原生基线。'}
    p['scope']['stop_rule']='一次本阶段试验；旧姿态匹配、新分支触发、实际退离与独立效果任一不通过，停止依赖的在线计时；不降低筛查阈值，不自动重跑。程序启动故障且零动作可修入口后重启并保留原记录。'
    if stage=='replay':
        p['configuration'].update(entry='replay.py historical actions, then original run_finish',new_model_calls=0)
        p['scope'].update(max_vla_actions_per_case=240,max_helper_actions_per_case=1260)
        p['new_physical_actions']=1500
        p['success_criteria']+=' 最小重放仅验退离与home；末端视觉明确标未调用/未知，不能作为整链视觉验收。'
    (out/(stage+'_proposal.json')).write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
protected=['finish_v1/control.py','finish_v1/sensors.py','finish_v1/runtime.py',
           'finish_v1/feedback_retreat.py','finish_v1/simple_checker.py','joint_home.py','xvla_backend.py','xvla_service.py']
record={p:hashlib.sha256((out/'before'/p).read_bytes()).hexdigest() for p in protected}
(out/'protected_controls.json').write_text(json.dumps(record,indent=2))
print('Two proposals and protected source hashes saved')
