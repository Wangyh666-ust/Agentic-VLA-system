# 接管与收尾 v2：原碗失败状态小范围验证（2026-10-10）

**本轮修正了接管触发与松手/退离顺序，已真实发出开口和退离动作，但完整子任务交接仍未通过。最终版本只测了原碗失败状态：11 个控制周期持续下达标准开口命令、60 次退离、确认 0、归位 0；因此完整交接 0/1，余四例暂停。** 编码器已确认夹爪实际张开；这里的 11 个周期不是反复开关夹爪 11 次。

这次有三个不同源码版本的定点试点，均使用同一个旧碗失败状态。它们是修正过程，不能合并成可靠性统计；最后版本的效果分母只有 1。最早一次启动因 LIBERO 导入询问配置路径而发生 EOF，物理动作和 Qwen 调用都是 0，不进入物理效果分母。

## 各版本实际动作与结果

每个物理试点都有 10 步初始化和 83 步历史动作回放。下表只列新增收尾动作；“等待”与“实际开口”是 release 阶段的两种动作，不能把整个 release 数量都算成松手。

| 版本 / 输出目录 | 闭合等待 | 标准开口命令控制周期 | 退离 movement | 确认 confirm | 归位 home | 停止原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 入口修正版 `pilot_first_entry_fix` | 6 | 0 | 0 | 0 | 0 | `release denied`：稳定未知，未发出开口 |
| 松手修正版 `pilot_first_releasefix` | 6 | 11 | 30 | 0 | 0 | `movement screen blocked`：在线路线筛查阻断 |
| 一次终点重规划版 `pilot_first_retreatfix` | 6 | 11 | 60 | 0 | 0 | `movement screen blocked`：一次重规划后，第二段终点仍清场不足 |

最终独立仿真检查保留了碗 `on|akita_black_bowl_1|plate_1=True`、`strict_candidate=True`、`released_stable`、`held_objects=[]`、机器人接触数 0。但完整交接要求的连续五次真实 post-action 样本计数仍是 `physical_streak=0`，`home.ready=False`，确认与归位也没有启动。**这是退离阶段阻断，不是一次已执行归位的失败。**

日志里的 `target_metrics.target_executed=True` 只检查实际开口和退离是否发生，不能代替完整验收。首例必须实际开口、退离、确认都大于 0；之后还要分别核对最终视觉完成、独立目标与释放状态、无接触和五次连续归位达标。本轮没有通过这套完整条件。

## 看视频

松手修正版：6 步等待后真正开启夹爪，随后发生 30 步退离，确认和归位未启动。[原视频文件](pilot_first_releasefix/bowl_held83/all_actions.mp4)

![松手修正版原视频](D:/FYP/First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/pilot_first_releasefix/bowl_held83/all_actions.mp4)

一次终点重规划版：真正开启夹爪后，两段退离共 60 步，第二次终点筛查仍不通过而停止。[原视频文件](pilot_first_retreatfix/bowl_held83/all_actions.mp4)

![一次终点重规划版原视频](D:/FYP/First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/pilot_first_retreatfix/bowl_held83/all_actions.mp4)

## 这次改对了什么，还卡在哪里

之前的条件形成了循环：先要求“物体已经放手后稳定”，才允许放手；先要求“夹爪已经离开物体”，才允许退离。于是程序可能一直等待，或者直接跳过真正的收尾动作。现在的顺序是：**确认目标到位且由目标支撑面承托 → 停稳 → 实际开启夹爪并用编码器确认 → 受保护的 RGBD 退离 → 独立完成检查 → 旧归位流程**。编码器开口只是交接条件，不能独自证明物体已释放、放对并稳定。

松手修正版在线原日志只记录了路线筛查的布尔阻断，没有保存精确底层原因。之后使用保存的 RGBD 和机械臂 CAD 做[离线 FK 复算](evidence/movement_screen_fk.json)：首次退离剩余路径的末端手部间隙约 **7.959 mm**，低于原 **10 mm** 清场要求。这个具体间隙是离线复算所得，不能写成原运行日志当时就记录了它。

最后版本只对 `endpoint_hand_clear_fail` 增加一次从实际姿态、用新 RGBD 重新规划的机会。movement 总预算仍为 80，不因重规划重置；geometry 的 margin 0.01、self-mask 0.002、approach 0.001 阈值均未改变。新版在线记录明确显示：第一次终点不足后发生一次重规划，第二次仍是 `endpoint_hand_clear_fail`，随后停止，不能强制开始归位。当前证据说明清场检查否决了路径，不能直接说出现了新的实际碰撞。

下一步只做离线核对：检查手部约 **119 mm** 的保守球包络与 RGBD 自身点剔除，是否使新观测反复否决余下终点。当前尚不能判断是哪部分造成问题；不增加物理试验、不降低阈值、不扩大样例、不微调。酒瓶、灶台、汤罐和番茄酱四例保持 `not_run`。

## 范围、调用量与来源

本轮不是新的 VLA 或 Hermes 能力实验：新 VLA 推理 **0**，新 Hermes 规划 **0**，没有训练。运行环境是 LIBERO / robosuite / MuJoCo + Franka Panda，尚未改演示 UI 默认链路。评分是独立的碗在盘子上的子任务目标，不是整个原场景的全部目标或完整多任务链。独立 oracle 仍负责仿真评分与紧急监督，不输入视觉检查或 RGBD 路线规划，不能称为已验收真机方案。

三个物理试点累计 **409 个实际动作 = 初始化 30 + 历史回放 249 + 辅助 130**；辅助包括等待 18、实际开口 22、退离 90，确认和归位均为 0。实际新 Qwen 调用 **6 次，11152 tokens**，每个物理版本 2 次。三次离线诊断均为零实际动力执行：前两次被守卫阻断并保留失败记录，第三次 FK 构建时将一次 `mj_step1` 替换为 no-op，未执行动力步进。

开发由 GPT 判断、定方案与验收，实际 DeepSeek 实现，本地 worker 应用、运行和归档。作者证据核对实际 DeepSeek 共 **8 次**：视觉 2、控制 3、runner 1、释放修正 1、退离修正 1；输入 **33503**、输出 **22014 tokens**。释放修正请求输出 6000 tokens 截断，整体返回没有直接应用；GPT 逐项审阅其中 12 个完整生产补丁对象，机械恢复 11 个，另 1 个与现有片段完全相同而无需修改，生成的测试没有照单应用，测试由 worker 按 GPT 固定规格完成。随后退离只请求两个准确片段，没有重生成整个模块。原请求与响应私有保留，公开作者记录含 SHA 和 usage；这些数字不能用于估计 GPT 账户消耗。

## 原始材料

- [机械汇总与计数定义](evidence/observed_results.json)；[初次 EOF 汇总](pilot_first/summary.json)。
- 入口修正：[summary](pilot_first_entry_fix/summary.json)、[result](pilot_first_entry_fix/bowl_held83/result.json)、[checks](pilot_first_entry_fix/bowl_held83/checks.jsonl)、[helper](pilot_first_entry_fix/bowl_held83/helper.jsonl)、[events](pilot_first_entry_fix/bowl_held83/events.jsonl)。
- 松手修正：[summary](pilot_first_releasefix/summary.json)、[result](pilot_first_releasefix/bowl_held83/result.json)、[checks](pilot_first_releasefix/bowl_held83/checks.jsonl)、[helper](pilot_first_releasefix/bowl_held83/helper.jsonl)、[events](pilot_first_releasefix/bowl_held83/events.jsonl)。
- 一次终点重规划：[summary](pilot_first_retreatfix/summary.json)、[result](pilot_first_retreatfix/bowl_held83/result.json)、[checks](pilot_first_retreatfix/bowl_held83/checks.jsonl)、[helper](pilot_first_retreatfix/bowl_held83/helper.jsonl)、[events](pilot_first_retreatfix/bowl_held83/events.jsonl)。
- 离线诊断：[首次守卫阻断](evidence/movement_screen_offline.json)、[第二次守卫阻断](evidence/movement_screen_kinematics.json)、[第三次 FK 复算](evidence/movement_screen_fk.json)。
- 作者记录：[vision](evidence/vision_author.json)、[control](evidence/control_author.json)、[pilot](evidence/pilot_author.json)、[release](evidence/release_author.json)、[retreat](evidence/retreat_author.json)。
- 固定输入与源码：[inputs](evidence/inputs.json)、[当前 preflight](evidence/preflight.json)、[原启动 preflight](evidence/preflight_at_entry_failure.json)、[入口版 preflight](evidence/preflight_at_entry_fix.json)、[释放版 preflight](evidence/preflight_at_releasefix.json)；[原源码 ZIP](evidence/runtime_sources_at_launch.zip)、[释放版 ZIP](evidence/runtime_sources_releasefix.zip)、[退离版 ZIP](evidence/runtime_sources_retreatfix.zip)。
- 方案：[入口版原字节备份](evidence/proposal_at_entry_fix.json)、[释放版原字节备份](evidence/proposal_at_releasefix.json)、[当前退离版](../../plans/2026-10-10-finish-takeover-v2.json)。原始输出均不可覆盖。

生成的大量逐步 sensors 数据保留本机，不删除。Git 仅忽略本 campaign 的 sensors 目录；Git 已精确纳入 44 个关键帧及各自 PNG/NPY/JSON 和三个实际视频，其余逐步传感数据留在本机。原 case 日志与证据保持原字节。此次归档只追加第 19 条台账，旧 18 条各记录及其证据 SHA 不变。
