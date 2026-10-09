---
name: fyp-experiment-review
description: Review FYP Hermes/SmolVLA robot experiments, explain existing failures, and plan or verify follow-up work using the experiment registry and original evidence. Use before proposing or running new FYP physical experiments, and before making experimental conclusions. Ordinary formatting and unrelated software tasks do not require this workflow.
---

# FYP 实验历史与证据审查

目标：复用已做实验，用明确的证据缺口决定下一步，区分程序通过与物理成功。此 skill 不运行机器人，不改变运行期 Hermes/VLA 控制链，也不提供新实验授权。

## 先查历史

仓库根目录下运行：

```text
python .agents/skills/fyp-experiment-review/scripts/review.py validate
python .agents/skills/fyp-experiment-review/scripts/review.py query --text 酒瓶 原生
```

优先阅读 `First_Phase/scene_demo/EXPERIMENT_INDEX.md` 和 `experiment_registry.json` 的匹配条目，再打开其相关原始 JSON、CSV 或逐步记录。索引是导航，不是全部实验的汇总成功率。检索不到不等于从未做过：继续用 rg 搜索 results、plans、源码和相关任务关键词。证据缺失、过期或条件不明应明确标记，不凭记忆补成事实。

先给出：相关实验 ID/证据链接、已知结果、适用条件、未解问题。不要重新安排已完成的原生基线、配置检查或辅助器对照；确需重复测量时说明新增信息或统计目的。

## 决定复用、离线分析或新实验

只解释历史结果时，不要求创建新方案文件或重跑测试。先复用已保存证据。要重新统计或比较已有记录时，用 `analyze_existing`；计划未来物理运行时，用 `new_experiment`。

设计或执行前读 [schema.md](references/schema.md)，填写结构化 proposal，运行：

```text
python .agents/skills/fyp-experiment-review/scripts/review.py check --proposal <方案JSON>
```

脚本只检查字段、证据完整性、相关历史是否被纳入以及完整配置重复；不判断因果，不调用模型，不启动实验。只有脚本通过且主 Agent 复核科学设计与现有授权范围后，才可在另一步调用原实验入口；现有入口尚无统一的程序级强制门禁，不能声称该脚本防住了任意绕过。每次新启动都先检查，保留输出。不要因本 skill 增设重复审批。

新实验必须说明：旧实验不能回答的具体问题；与旧实验的区别；成功/失败/未知判据；固定条件与改变因素；最大样本和动作预算、停止规则。诊断对照尽量隔离一项因素；多因素干预必须明确无法分开归因。配对控制应包含真实状态、两路原生图像/状态/语言输入、策略队列、采样随机流和实际动作预算；初态编号相同并不证明实际输入相同。无法取得时记为未知。

完全相同配置可以为了有效复现或补样本重测，但必须有明确目的；仅换名称或日期没有信息增益。原始成功/失败样本保留，分析不能覆盖旧记录。

## 结论与验收

GPT/主 Agent 保留问题定位、设计选择和证据判断；DeepSeek 执行明确的代码/统计任务，按 AGENTS.md 六节模板派发。本地 worker 可以应用与运行；不得冒称它就是 DeepSeek。验收对照命令、原始输出、Git diff 与实测，不直接相信“已完成”。

报告必须分开：
- 程序/协议完成、规划正确、准备动作完成、VLA 物理任务完成。
- 新 VLA 动作、历史动作回放、辅助动作；未启动、准备阻断、执行失败、用户中止。
- 单目标与全任务、原生位置分与独立语义观察、样本计数与总体成功率。

`ok=true` 不代替物理成功；VLA 未启动不能算 VLA 失败；已松手不等于放对位置。严格条件不是失败的自动解释，先查目标谓词是否曾真、抓取/高度/释放与轨迹。接触筛查是观测代理。少量成功证明能力存在，不证明可靠；多次失败不直接证明训练缺陷。当前 LIBERO 环境是 robosuite/MuJoCo + Franka Panda，不是 ManiSkill；辅助几何来自仿真，RGBD 实测未实现。

沿用已通过的接口核查，只有源码、模型、观测、控制配置有相关变化才重查受影响部分。软件验证不能代替物理验证，历史科学证据可在条件与来源明确时复用。用户已中止或缩小的试验不可擅自续跑。

完成新的工作后更新索引/台账，给出结论、证据、限制、尚缺信息和下一步分支；保留原始数值和未知。只为当前问题检索相关材料，不把所有日志加载进上下文。
