# 目的检查与退离预算修复

**结论：灶台在线完整通过；酒瓶实际松手→退离→归位及独立五连验收通过，原在线最后视觉判错。最终检查修正后，在同一真实终态上复核通过，没有再跑归位。不能称最终统一版整链联测通过。**

实际旧提示词仍添加了“仅悬在上方不算”和单独承托门槛，未落实用户要求。新检查仅输出目的区域是否达到、是否仍夹持/接触、简短依据。接管与最终放置验收分开。

灶台旧120周期仅走完约14/20个2mm路径段，并未停滞。新上限为 `min(480,max(120,16×路径段数+40))`，40周期无误差改善停止，保留路径筛查和到达容差。原归位控制器和streaming执行代码未变。

酒瓶进一步暴露开爪前退离门槛问题：一侧指尖接近架面，实测开口约标准92%，关节指令误差约1度。近全开且手指/手臂停稳时允许从实际姿态规划有筛查的退离，继续全开命令；这不是松手成功证据，归位前仍要求精确标准开口和无障碍。

最终接管采用原子恢复：首次目的信号为真后连续开爪、退离，按实时开口/速度/RGBD几何筛查进入原归位。中途不反复用波动的视觉目的判断打断恢复；最终重新拍照验收，失败仍保留，不拿初次目的判定冒充最终成功。

酒瓶最终视觉修正：改用清晰主视图；子任务使用catalog已有自然语言，内部region ID不作为额外视觉要求；删除“目的地区域/层板目标位置”措辞，字面核对是否在架子上，开启1024预算的Qwen短推理。最后回答goal_achieved=true、holding_or_contact=false。视角/措辞/推理多次修正，不能分别归因。

| 运行 | 子任务 | 实际开爪周期 | 退离周期 | 归位周期 | 最终归位 | 完整验收 | 停止原因 |
|---|---|---:|---:|---:|---|---|---|
| pilot_wine | 酒瓶放架 | 0 | 0 | 0 | False | blocked | initial stop |
| pilot_wine_repair | 酒瓶放架 | 0 | 0 | 0 | False | blocked | initial stop |
| pilot_wine_views | 酒瓶放架 | 20 | 0 | 0 | False | blocked | release not confirmed |
| pilot_stove | 灶台开火 | 0 | 0 | 0 | False | blocked | initial stop |
| pilot_wine_withdraw | 酒瓶放架 | 20 | 0 | 0 | False | blocked | release not confirmed |
| pilot_stove_repair | 灶台开火 | 17 | 182 | 68 | True | physical_success | complete |
| pilot_wine_escape | 酒瓶放架 | 12 | 135 | 0 | False | blocked | not eligible for home |
| pilot_wine_stage_views | 酒瓶放架 | 12 | 0 | 0 | False | blocked | release visual failed |
| pilot_wine_atomic | 酒瓶放架 | 12 | 135 | 214 | True | blocked | final visual failed |

实际Qwen调用 20 次、记录usage 30759 tokens；子agent0、新VLA动作0。失败及修复版本均保留。

录像前段为原动作真实回放，后段为本轮新增收尾动作；阶段计数不把回放计为新VLA。独立目标、接触、归位从oracle日志验收，没有输入Qwen。

酒瓶与灶台分别在修复过程中验证；灶台已有成功后复用，不为了统一源码版本重复归位。各运行绑定对应冻结源码，不能当成最终同版两任务或组合联测。碗、番茄罐及演示自动收尾没有在本轮启用。

[上轮实际完整提示词](evidence/actual_old_prompt.txt)；[本轮逐阶段原始摘要与模型答复](evidence/summary.json)。

## pilot_wine：酒瓶放架

![酒瓶放架](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_wine/wine_exit/all_actions.mp4)

## pilot_wine_repair：酒瓶放架

![酒瓶放架](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_wine_repair/wine_exit/all_actions.mp4)

## pilot_wine_views：酒瓶放架

![酒瓶放架](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_wine_views/wine_exit/all_actions.mp4)

## pilot_stove：灶台开火

![灶台开火](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_stove/stove_exit/all_actions.mp4)

## pilot_wine_withdraw：酒瓶放架

![酒瓶放架](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_wine_withdraw/wine_exit/all_actions.mp4)

## pilot_stove_repair：灶台开火

![灶台开火](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_stove_repair/stove_exit/all_actions.mp4)

## pilot_wine_escape：酒瓶放架

![酒瓶放架](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_wine_escape/wine_exit/all_actions.mp4)

## pilot_wine_stage_views：酒瓶放架

![酒瓶放架](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_wine_stage_views/wine_exit/all_actions.mp4)

## pilot_wine_atomic：酒瓶放架

![酒瓶放架](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-goal-finish/pilot_wine_atomic/wine_exit/all_actions.mp4)

## 当前终态复核

| 复核版本 | 目的达成 | 仍夹持/接触 | 判定 | 新机器人动作 |
|---|---|---|---|---:|
| posthoc_primary | False | False | incomplete | 0 |
| posthoc_primary_names | False | False | incomplete | 0 |
| posthoc_primary_task_text | False | False | incomplete | 0 |
| posthoc_primary_literal_thinking | True | False | complete | 0 |

[最终实际完整提示词（文字原样，图片以文件/哈希索引表示）](posthoc_primary_literal_thinking/complete_prompt.txt)。
补充番茄罐负例的一次启动没有保留可核实的模型响应，结果及usage未知，不计作检查通过、不重试；上方调用/token数只统计有原始响应的记录。
