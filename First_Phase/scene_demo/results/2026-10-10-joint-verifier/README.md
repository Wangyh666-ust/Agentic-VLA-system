# 视觉与同步夹爪反馈检查

画面+同步编码器接口及四个保存状态判定已修正；第五负例身份识别仍unknown，整体未通过，自动收尾保持关闭。

| 保存状态 | 预期 | 选用实际输出 | 准确验证 |
|---|---|---|---|
| 酒瓶 | needs_release，allow_release=True | 初版needs_release / True | 通过 |
| 初始持碗 | needs_release，allow_release=True | 初版needs_release / True | 通过 |
| 碗完成 | complete，五核心True | revision4 complete，八字段True | 通过 |
| 灶台 | operation_achieved=True，needs_release / True | revision5符合 | 通过 |
| 番茄酱负例 | at_destination=False，incomplete，两allow=False | revision6 unknown，两allow=False | 安全拒绝，准确失败 |

五个历史保存状态准确输出4/5，不是机器人成功率估计。实际Qwen 10请求，26977 tokens（prompt 22633，completion 4344）。前三个碗失败版本、首次灶台失败和两次番茄酱unknown全部保留，见[saved_checks](saved_checks/)；不同版本输出复用，不是最终同一版五个独立API样本。两次零调用复算不计新增样本。所有新增机器人/VLA/replay/helper/home/render0；历史真实归位1/1复用，本轮home0。

真实修复：

- 最新画面绑定capture同次robot，避免随后读取另时刻编码器。
- 标准开口偏差大于2mm否决完成中的release/clear，缺失或过期同步反馈为未知；闭爪不证明持物，开口不证明放对位置。
- 四完整原图加双view动态无损crop，最多一次独立复核；公开can/carton类别避免瓶/罐误认，仍有目标身份未知时停止。

旧测量同步仍是inferred_same_state：旧日志无直接measurement timestamp，冻结只读源码与state SHA链推定最新帧同一无动作状态；新runtime采用同次capture反馈。阶段不证明动作成功，二维框只调度观察，crop不是新view/time或接触真值。公开asset类别语义证明tomato_sauce为can；纹理模板仅本地核查，不上送模型；评分ROI、仿真物体真值和接触不上送。

原完整双视角：

![原完整图0](D:/FYP/First_Phase/scene_demo/results/2026-10-10-shape-home/pilot_home_gate/bowl_held83/sensors/cap001045_agentview.png)

![原完整图1](D:/FYP/First_Phase/scene_demo/results/2026-10-10-shape-home/pilot_home_gate/bowl_held83/sensors/cap001047_sideview.png)

原保存图的派生细节（无重采样，不是新视频或新动作）：

![碗细节agent](D:/FYP/First_Phase/scene_demo/results/2026-10-10-joint-verifier/saved_checks/revision4/bowl_final_home_details/detail_0.png)

![碗细节side](D:/FYP/First_Phase/scene_demo/results/2026-10-10-joint-verifier/saved_checks/revision4/bowl_final_home_details/detail_1.png)

证据：[冻结输入](evidence/fixtures.json)、[来源审计](evidence/synchronization_audit.json)、[类别依据](evidence/revision6_category_sources.json)、[汇总](evidence/summary.json)、[初版报告](README_before_detail.md)、[初版软件](evidence/software_checks.json)、[最终类别软件](evidence/revision6_commands.json)。所有版本源zip保留，[最终源码](source_snapshot_final.zip)；三计划见[初版](../../plans/2026-10-10-joint-verifier.json)、[细节](../../plans/2026-10-10-joint-verifier-detail.json)、[类别](../../plans/2026-10-10-joint-verifier-category.json)。

剩余方向只有目标身份识别和更充分可见信息，尚未实施；不重复归位或微调。没有off灶台反例，不能宣称点火检测普遍可靠。本轮未重新验证机器人松手/退离/归位效果，UI保持off。
