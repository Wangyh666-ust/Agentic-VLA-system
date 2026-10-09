# 安全退离试点：接口已修正，归位效果尚待验证（2026-10-10）

## 结论先行

上一轮 case02/07/08 的前任务已经成功，但夹爪仍接触架子或灶台，原归位预检因此拒绝开始，实际归位动作数为0。新方法保持的流程是：读取夹爪实际开口 → 沿接触远离方向做短距离真实退离 → 再执行原严格 7 关节归位。但本次尚未测到任何一例的退离或归位有效性。

三例中，case02 回放了 180 个旧动作后，因 SDK 字段名错误中止，退离 0、归位 0；case07 与 case08 均为 not_run。SDK 报错为 `'SingleArm' object has no attribute '_ref_gripper_actuator_indexes'`。这不是新的 VLA 抓取失败，也不代表修复成功；候选方法仅通过规划层面，尚未进入物理验证。

## 三例状态

| case | 状态 | 回放动作 | 退离动作 | 归位动作 |
|---|---:|---:|---:|---:|
| case02 | unknown | 180 | 0 | 0 |
| case07 | not_run | 0 | 0 | 0 |
| case08 | not_run | 0 | 0 | 0 |

unknown 触发停止全 campaign。无调参、无重试、无新 VLA / Hermes 调用、无权重加载、无训练。

## SHA 与候选

case02 原态与终态完整 SHA 均与旧记录精确相同。2cm 候选在 scratch 环境被接受，但未执行。最终酒瓶谓词为 true / 未持有接触代理；完整最终 guard 未保存，因此不能称“全部保护通过”。视频只有1帧，显示重放旧动作后的状态；本次没有归位运动，不能作为归位演示。

最初空 CLI 因缺 `__main__` 退出，零物理执行；已补入口及入口测试，并保留原记录，这不计入机器人试验。

## 软件与证据边界

软件层面区分“运行冻结版”与“运行后 SDK 修正”：错误 `_ref_gripper_actuator_indexes` 已改为 installed SDK 的 `_ref_joint_gripper_actuator_indexes`；SDK 新增测试会调用真实 `SingleArm.setup_references`，但伪模型不创建机器人环境。接口修正已通过35项软件测试（0跳过），其中调用了当前安装版本的 SingleArm.setup_references；未创建仿真环境。软件验收以 `evidence/software_proof_final.json` 为准，不能代替物理验证。

证据链接：[实际结果](pilot/summary.json)、[运行命令](evidence/pilot_real_command.json)、[运行后审核](evidence/pilot_real_postflight.json)、[视频审核](evidence/pilot_real_video_check.json)、[运行前冻结软件证明](evidence/software_proof_entry.json)、[SDK修正验证](evidence/gripper_sdk_fix_final_proof.json)、[当前版本软件验收](evidence/software_proof_final.json)、[原输入快照](evidence/frozen_inputs.json)、[原15条台账快照](evidence/registry_before.json)、[原方案](../../plans/2026-10-09-safe-exit.json)。

## 离线酒瓶分析

旧首酒瓶 case01/02 成功 262/180 动作；旧碗后酒瓶 case03/04 各 336 失败，目标从未 true，记录中“抓住酒瓶”的接触检查样本为0。归位后关节离标准约 .0027 rad，酒瓶位置不变，碗移动约 14.9/16.0 cm。缺两路原始/处理后策略输入与每子任务 RNG 快照；旧 runner 只在 case 开始 seed、边界清队列，不能把差异只归因于机器人姿态或训练。详见 [离线差异表](analysis/wine_existing.json)、[旧12例报告](../2026-10-09-joint-home-20/README.md)，本轮不重做原生基线。

## 边界声明

新增安全退离路径目前用于独立诊断试点，默认8081演示尚未接入；需先验证实际运动效果再接入。

本轮有效性尚未测到，不能称已修复物理效果，不合并历史样本成功率。后续计划见 [../../plans/2026-10-10-safe-exit-next.md](../../plans/2026-10-10-safe-exit-next.md)，该计划仅作审阅，不代替新 proposal 检查与授权。
