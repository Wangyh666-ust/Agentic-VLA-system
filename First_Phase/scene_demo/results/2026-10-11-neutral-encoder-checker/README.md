# 检查器修复：视觉判断目标，编码器判断开口

已修改运行期检查器 `visual_goal_encoder_v3`。Qwen 只回答当前子任务目标是否达到及画面依据；程序用同一时刻的实测双指编码器与标准开口参考判断是否已松手。达到标准开口时，遮挡、二维投影重叠或旧回答里的“仍接触”不能否决松手。保留原开口容差 2 mm，未增加新门槛。

目标达到就移交收尾器，实际执行开爪、退离和原归位流程。即使初次检查已经开爪，也不会跳过这些动作。开爪不代替“物品到位”，也不代替退离路径检查；视觉目标为假或未知时不会强行改成成功。原归位的动作循环、配置和路径保护未改，控制文件仅更新检查器版本标签。

新提示词按当前子任务描述检查，没有列举任务供选择，没有 phase/stage 标签，也不预设本次已到位或仍夹持。检查时机仍为第 150、300 个 VLA 动作。完整新提示词示例：[酒瓶末次输入](wine_final_new_prompt.txt)、[碗末次输入](bowl_final_veto_new_prompt.txt)、[灶台输入](stove_trigger_new_prompt.txt)。这些提示词由真实旧输入生成，本轮没有发送给 Qwen。

## 本轮验证结论

45 项相关软件回归通过，包括开口覆盖视觉接触误判、目标假/未知不被编码器覆盖、双视角与编码器必须同步、无效反馈拒绝，以及“初次已开爪仍执行收尾及原归位”的控制分支测试。软件中的模拟 Port 动作不计作实际机械臂动作。

另复用上轮已核对的五份原始请求、原图、原回答及同步机器人反馈，验证新融合规则。来源 SHA256 均匹配；没有新 Qwen 请求，没有新增 VLA/辅助/回放动作。

| 保存记录 | 原视觉目标 | 实测标准开口 | 新融合结果 |
|---|---|---|---|
| 酒瓶初次检查 | 达到 | 否 | 允许收尾、需要开爪 |
| 酒瓶归位后检查 | 未达到 | 是 | 确认已松手，但保留目标否定 |
| 灶台初次检查 | 达到 | 否 | 允许收尾、需要开爪 |
| 碗归位后旧通过回答 | 达到 | 是 | 检查通过 |
| 相同碗末态旧“仍接触”回答 | 达到 | 是 | 检查通过，与上行一致 |

碗的两份相反“夹持”回答不再导致相反结果。酒瓶末次原回答还把 `goal_achieved` 判成 false，本次没有重新请求 Qwen，不能宣称新中性提示词已修复视觉目标识别。灶台修订后的退出路线仍缺实际验证；编码器修复不会把几何阻断改写成视觉问题，也不改写旧实验结果。

本轮完成的是程序修复及旧证据离线融合验证，尚没有新版本组合任务视频或物理成功结论。下一次沿用原范围的小试应核对：150/300 动作检查的实际回答、目标达成是否触发、实际开爪/退离/home 动作数和独立效果；不单独重跑归位。

## 复现与证据

在 WSL 中使用原项目 venv，设置 `PYTHONPATH=/mnt/d/FYP/First_Phase:/mnt/d/FYP/First_Phase/scene_demo`：

```text
python -B -m unittest scene_demo.finish_v1.tests.test_simple_checker scene_demo.finish_v1.tests.test_runtime scene_demo.finish_v1.tests.test_assisted_combos
python -B First_Phase/scene_demo/results/2026-10-11-neutral-encoder-checker/verify_existing.py
```

派生数据和原证据哈希：[verification.json](verification.json)。测试原始输出：[stderr](tests.stderr.log)、[stdout](tests.stdout.log)。分析门禁通过：[方案](../../plans/2026-10-11-neutral-encoder-checker.json)、[门禁输出](preflight.log)。上轮实际 HTTP 输入核查：[报告](../2026-10-11-final-input-audit/README.md)。本轮按用户要求由当前主会话与本地工具完成，没有子agent或新增 DeepSeek/Qwen 请求，不声称存在 low 档实现模型调用。
