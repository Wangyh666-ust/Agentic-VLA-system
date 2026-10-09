# FYP 实验历史审查 skill：工作流验收（2026-10-09）

本次验收检验证据复用与方案审查工作流，没有新增机器人动作、VLA或Hermes调用，不是新的机器人效果试验。

## 入口与现有证据

[Skill](../../../../.agents/skills/fyp-experiment-review/SKILL.md)位于仓库`.agents/skills/fyp-experiment-review`。[AGENTS.md](../../../../AGENTS.md)的触发规则要求：涉及本项目实验解释、排错、计划、执行或报告时先读取skill；普通文档排版不触发。

[实验索引](../../EXPERIMENT_INDEX.md)与[台账](../../experiment_registry.json)包含15个历史campaign、21份原始证据，逐字节哈希核验保持不变。它们是证据导航，复用样本可能重叠，不能合成总体成功率。

## 软件与真实方案检查

- Linux：47项通过、0项跳过，见[原始日志](linux-tests.log)。
- Windows：46项通过、1项跳过，见[原始日志](windows-tests.log)。两种环境的计数分开保留。
- 官方skill校验输出`Skill is valid!`，见[校验日志](skill-validate.log)。
- 台账验证15条、21份证据，见[原始JSON](registry-validate.json)。
- [真实离线分析方案](../../plans/2026-10-09-evidence-analysis.json)检查退出0，`ok=true`、`intent=analyze_existing`、`launch_authorized=false`，见[实际输出](offline-proposal-check.json)。检查没有执行方案中的分析步骤。
- [漏查历史夹具](missing-history-proposal.json)故意不列原生酒瓶基线。检查退出2，报告`missing history_review`并命中原生/共享诊断、配置配对与起始条件三个旧campaign，见[拒绝输出](missing-history-check.json)。夹具不是真实实验方案。

第一次软件测试存在夹具错误，定向修复后重新验收；原失败证明和日志仍保留在本机私有忽略目录，没有改成首轮全过。

## 五个独立GPT行为请求

[原始回答](forward_answers.json)保留Q1–Q5。主Agent逐条人工复核，5/5通过：

- Q1：识别旧原生、配置、篮子对照并避免重复。
- Q2：区分准备与抓取，引用兼容入口和完整归位成功反例，不直接决定训练。
- Q3：区分12完整、13部分中止与14–20未启动；保留3成功、6次VLA失败、3次归位阻断及11/17子任务成功。
- Q4：区分未确认抓稳与未知训练因果，先复用离线记录。
- Q5：最新实验Hermes调用0，不拿3/12评价规划增益。

这些评分由GPT主审确定，本地执行者没有再次模型评分。有限的5个请求不是永不出错的保证。

## 来源与边界

GPT负责设计、证据判断与主审；实际DeepSeek API负责作者实现和定向修复；worker机械应用、运行与归档。两次真实作者调用均为HTTP200、`deepseek-flash`、`end_turn`；净化元数据和文件SHA见[CHECKS.json](CHECKS.json)，不归档请求、响应、鉴权头或凭据。

当前15条历史的`configuration_complete`全部为false，已核实完整配置数为0。精确配置重复检查目前仅由合成测试覆盖；真实历史依靠问题检索与GPT复核，不能声称已实现真实状态自动精确去重。

现有runner尚未统一强制接入审查脚本；检查通过不表示启动已获授权，也不证明科学结论成立。后续工作仍为[待审计划](../../plans/2026-10-09-evidence-first-next.md)，物理分支未执行。历史失败、未知和用户中止范围保持原样，不自动续跑或扩大。
