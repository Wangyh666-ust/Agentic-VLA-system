# 视觉与同步夹爪反馈检查

已修好编码器同步与未松手否决，视觉碗盘定位仍错，整体未通过，UI自动收尾未启用。

| 状态 | 预期 | 实际最终检查 | 是否准确验证 |
|---|---|---|---|
| 酒瓶退出旧状态 | needs_release，允许释放 | needs_release，允许释放 | 通过 |
| 初始持碗状态 | needs_release，允许释放 | needs_release，允许释放 | 通过 |
| 碗完成归位旧状态 | complete，五核心字段True | revision3 incomplete | 未通过 |
| 灶台退出状态 | operation_achieved=True，needs_release | 未运行 | 未验证 |
| 番茄酱负例 | at_destination=False，incomplete，两allow=False | 未运行 | 未验证 |

碗初版unknown；revision2复核incomplete，目的plate框落在灶台区域，并自述手无接触却released=False；revision3复核仍定位错，但released/clear=True正确识别手已分离。三个版本原输出均保留，未改标签，失败后按停止规则不运行余例。

本轮五次Qwen请求：初版三次、revision2和revision3复核各一次，共计 13747 tokens（prompt 11715，completion 2032）。五个预登记状态只有三个产生新输出，两次零调用复算不是新增样本；不能报告5/5或机器人总体可靠性。触发是检查器分支，物理/VLA/helper/home/render动作均0。历史真实归位1/1由shape-home原证据复用，本轮不重跑归位，未证明归位失败或训练缺陷。

历史同步限定为inferred_same_state：旧测量没有直接timestamp，冻结代码及state SHA链证明最新画面处在同一无动作状态，不能称新硬件显式同步。新runtime使用capture同次robot，缺反馈即中止。旧oracle仅抽取可测夹爪编码器与关节速度，物体真值、接触和评分ROI不上送模型。阶段标签不能证明执行成功；开口不证明放置或支撑；二维间隙只调度独立复核，不是物理安全距离或接触证据。

完整原图（最新同刻双视角）：

![0](/D:/FYP/First_Phase/scene_demo/results/2026-10-10-shape-home/pilot_home_gate/bowl_held83/sensors/cap001045_agentview.png)

![1](/D:/FYP/First_Phase/scene_demo/results/2026-10-10-shape-home/pilot_home_gate/bowl_held83/sensors/cap001047_sideview.png)

证据：[五例冻结输入](evidence/fixtures.json)、[同步来源审计](evidence/synchronization_audit.json)、[汇总](evidence/summary.json)、[初版软件](evidence/software_checks.json)、[revision2软件](evidence/revision2_software_checks.json)、[revision3验证](evidence/revision3_commands.json)、[proposal](../../plans/2026-10-10-joint-verifier.json)。原输出见[saved_checks](saved_checks/)，源冻结为[source_snapshot.zip](source_snapshot.zip)、[revision2](source_snapshot_revision2.zip)、[revision3](source_snapshot_revision3.zip)。

后续方向：动态感知定位+局部视觉细节辅证，先原存图验证；尚未实施，也未擅自扩大预登记或启用UI。
