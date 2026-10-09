# 台账与方案格式

路径均为仓库相对 POSIX 路径；脚本解析后必须仍位于仓库根目录内。台账的 evidence 每项有 path 与原始字节 sha256。哈希不一致、缺文件或配置未知，不能凭空修正；核对原始证据后更新索引派生条目，保留旧实验。

## Registry v1

`schema_version=1`，`entries` 是数组。每项包含唯一 `id`、`title`、`date`、`question_keys`（非空字符串数组）、`tags`、`conditions`（说明实际场景、状态/种子、动作来源、判据、范围）、`findings`、`limits`、`evidence`（非空 `{path, sha256}` 数组）。可选 `configuration` 对象与 `configuration_complete` 布尔值。仅 complete=true 且完整配置已从来源核实的条目参与完全配置重复检查；其余只有问题/主题匹配，不能当成精确同状态对照。

台账一期按15个历史 campaign组织，并非15次独立试验；不同 campaign 的复用样本不得合并计算成功率。查询用 tags/title/conditions/findings/question_keys，仍由主 Agent 阅读实际证据。

## Proposal v1

所有方案要求：`schema_version=1`、非空 `id`、`intent`（`reuse_evidence` / `analyze_existing` / `new_experiment`）、非空 `question_key`、`purpose`、`prior_evidence`（历史 ID 数组）、`history_review`（`{experiment_id, disposition, reason}` 数组；disposition 为 `reuse` / `insufficient` / `not_applicable`）、`new_physical_actions`（非负整数）。

同 question_key 的历史条目每个都必须进入 history_review；`reuse` 与 `insufficient` 还必须进入 prior_evidence。reason 必须解释依据，不以索引命中代替科学判断。无主题命中时仍要在 related_search 写明额外 rg 搜索与未找到证据的范围。

`reuse_evidence` 与 `analyze_existing` 要求 new_physical_actions=0，并提供非空 `analysis_steps` 字符串数组；不要求物理实验预算、假设或成功率。

`new_experiment` 还要求非空 `hypothesis`、`information_gain`、`success_criteria`、`failure_criteria`、`unknown_criteria`；`changed_factors` 为非空 `{name,before,after,reason}` 数组（before/after 不同），`fixed_factors` 为非空对象；`scope={max_cases,max_vla_actions_per_case,max_helper_actions_per_case,stop_rule}`，max_cases/max_vla_actions_per_case 为正整数，max_helper_actions_per_case 为非负整数，stop_rule 非空；`configuration` 为非空对象，`configuration_complete=true`；`repeat={needed,reason}`，needed 为布尔值，needed=true 时 reason 非空。new_physical_actions 必须为正整数且不超过 scope 的总预算。

若 configuration 与完整历史配置完全相同，repeat.needed 必须为 true 且说明复现/补样本的具体目的；新名称或 changed_factors 的声明不会免除重复检查。多项因素变化输出提醒，由主 Agent 判断设计，脚本不替代判断。完整配置要求研究者记录实际对照所需的权重、输入/状态/语言、控制、种子/随机流、动作预算与停止条件；脚本不能证明这些值真实或语义等价。

## 脚本接口

`review.py [--root PATH] [--registry PATH] validate`：全台账路径、字段和证据哈希检查。
`review.py [--root PATH] [--registry PATH] query --text TERM [TERM ...]`：不区分大小写的词项 OR 检索；输出匹配条目、条件、结论、证据。
`review.py [--root PATH] [--registry PATH] check --proposal PATH`：先验证台账，再检查方案；输出 JSON，`ok` 仅表示结构审查通过、`scientific_review_required=true`、`launch_authorized=false`。退出0为结构通过，退出2为拒绝/数据错误。不自动执行任何命令。默认 root 由脚本位置指向仓库，registry 为 First_Phase/scene_demo/experiment_registry.json。所有命令输出必须为 JSON，失败不能有 traceback。
