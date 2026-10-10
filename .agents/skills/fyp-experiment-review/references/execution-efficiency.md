# 执行复盘：关键验证太晚与流程开销（2026-10-10）

用于FYP的软件排错和实验执行；规则见[skill主文件](../SKILL.md)。这是具体案例记录，不增加机器人动作授权或固定耗时承诺。

## 这次发生了什么

[安全退离试点报告](../../../../First_Phase/scene_demo/results/2026-10-09-safe-exit/README.md)及[实际运行命令](../../../../First_Phase/scene_demo/results/2026-10-09-safe-exit/evidence/pilot_real_command.json)保留原始结果。

- 第一次命令缺少main入口，退出0却没有启动仿真；最初只测试函数，没验实际CLI入口。
- 后来真正启动的试点进程约40秒；case02回放180个旧动作后，因夹爪接口字段名称错误停止，退离0、归位0，case07/08未运行。
- mock用了同样猜错的字段，未及时暴露真实SDK不匹配。修正后35项SDK相关软件测试通过；最终97项测试及3子测试通过，仍没有修正后的退离/归位实测。
- 实现、测试与文档多轮跨模型往返；证明材料过重。Git检查又因原始日志空白和Windows输出换行被阻断；最终保留原字节，对四份原始输出作精确格式例外后提交。

## 下次如何作决策

| 情境 | 应采取的动作 |
|---|---|
| 退出0但没有预期输出或动作 | 查真实CLI入口；不宣布运行通过 |
| 新增SDK调用 | 先查已安装版本的字段/签名，再写独立mock；单元测试不能冒充实际执行 |
| 同一修改出现多个确定错误 | GPT列一次完整修正清单，按六节模板合并派发；不逐行循环 |
| 只改说明文字 | worker复制确定文本，做结构/链接审阅；不启动机器人验收或重复模型生成 |
| 原始日志格式告警 | 保留原字节，精确标记例外；源码/文档检查继续有效 |
| 真实运行出现unknown | 按登记停止规则结束；保留错误，修复不能自动授权旧campaign重试 |
| 软件通过但实际动作0 | 明确效果未验证；不报告用户目标已完成 |

本轮慢主要来自执行安排和返工；不能把软件检查数当作机器人效果，也不把这些问题归因于VLA训练。

## 视觉收尾试点：目标未触发却转向归位（2026-10-10）

[原试点报告](../../../../First_Phase/scene_demo/results/2026-10-10-visual-finish/README.md)记录：原目标是验证视觉检查接管与松手／退离收尾，五例 release、movement、confirm 实际动作均为零；碗误判为已完成后直接归位并耗尽预算，酒瓶直接归位路线阻断，灶台开火被误解为开盖。负例零辅助动作须与应介入的正例区分。

本轮没有得到收尾动作效果证据；检查器触发与语义存在问题。不能把这批结果解释为收尾器无效，也不能用归位失败替代原实验目的。GPT 在首个应介入样例发现目标未执行时，就应停止依赖验证并定位；保留原记录，修正须作为后续独立工作，不能悄悄覆盖历史。

另一个偏离是只复用了归位控制器类，却更换实际轨迹执行循环与路径筛查。已验证能力应复用到执行流程与配置层面；若修改，必须显式列为改变因素。此案例用于约束未来派发、首例检查与验收，不增加机器人动作授权。

## DeepSeek 返工与空输出：请求组织问题（2026-10-10）

[初始调用记录](../../../../First_Phase/scene_demo/results/2026-10-10-visual-finish/evidence/control_author.json)保存错误接口、输出截断与修正经过；[控制作者记录](../../../../First_Phase/scene_demo/results/2026-10-10-visual-finish/evidence/finish_control_author.json)保存多轮拒绝，包括两次耗尽 12000 输出 token 但代码文本为空。初始实际请求先写一套字段、末尾再覆盖为另一套，且同时要求多模块与测试的完整 JSON 源码映射，没有落实六节模板。

这些记录证明存在可避免的返工，不是只归因于 DeepSeek 能力。GPT 应在派发前统一接口、按功能缩小交付，在截断或空输出后停止盲试并检查调用参数；代码实现与确定的机械处理分别交付。该规则不增加模型、机器人或扩大测试范围的授权。

## 首例启动遗漏与释放阶段误计（2026-10-10）

[首次启动命令](../../../../First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/evidence/first_command.json)漏了旧成功运行使用的 LIBERO_CONFIG_PATH，LIBERO 导入询问路径时发生 EOF；物理动作与 Qwen 调用均为零，不计入物理效果分母。保留该记录后，[入口修正命令](../../../../First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/evidence/first_entry_fix_command.json)只补回已证实的配置变量，单独运行首例。

入口修正首例的 [checks](../../../../First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/pilot_first_entry_fix/bowl_held83/checks.jsonl)与 [helper](../../../../First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/pilot_first_entry_fix/bowl_held83/helper.jsonl)显示：六步 release 标签实为保持闭合等待，实际标准开口命令为零、退离为零；stable 未知造成放手前先决条件循环。阶段名称和计数不能证明已经松手，不能据此宣称收尾效果或物理修复成功。原记录保持不变。

本次[释放修正请求](../../../../First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/evidence/release_author.json)把生产补丁与多个测试放在一起，9951 input / 6000 output tokens 后截断；停止再生成，GPT 审阅完整生产对象后局部恢复，未将截断整体或生成测试直接应用。[后续退离请求](../../../../First_Phase/scene_demo/results/2026-10-10-finish-takeover-v2/evidence/retreat_author.json)只给两个准确片段，982 input / 1129 output tokens，end_turn。下一次按已定位目标提供必要片段，避免整模块重生成；token 更少或软件测试通过都不等于物理成功。
