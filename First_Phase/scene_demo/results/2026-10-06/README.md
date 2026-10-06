# v2 pilot 实测原始证据（2026-10-06）

本目录保存「持久场景 v2」**已完成单次 pilot** 的原始证据：seed 0、初始状态 0、**n=1/条件**；运行时 **qwen3-vl-plus / alibaba-cn**；checkpoint revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。上层说明见 [../../README.md](../../README.md)。

## 一、实测表

下表是**历史 table pilot 五行**（seed 0、初始状态 0、n=1/条件）：这五行的 `task_success` 均为 `false`，且其 blocked 记录里 **`plan_success` 为 `null`（既不是 `false` 也不是 `true`）**。本目录后来新增的**物理执行**、**意图 / 容量**与**修复范围探针**证据见第三、四、五节；其中 bowl-only、`missing_bin`、`mugs_full` 与 ambiguity 澄清重测的 `task_success` 为 `true`（含义分别见对应小节，`task_success=true` 不等于 VLA 操作成功）。

| 记录 | 场景 / case | 执行方式 | 子目标步数 | plan_success | task_success | 时间 |
|---|---|---|---|---|---|---|
| `table_forward`（manual） | `goal_table` / `table_tidy` | manual_subgoals（无 Hermes） | 碗 `bowl_to_plate` 95 步成功；酒瓶 `wine_to_rack` 300 步失败 | `null` | `false` | elapsed_s 235.657 |
| `table_tidy`（真实 Hermes） | `goal_table` / `table_tidy` | Hermes 规划 + 一次修复 | 碗 95 步成功；酒瓶 300 步失败；修复后酒瓶再 300 步失败（累计 695 步） | `null` | `false` | wall_s 457.966 |
| `table_reverse` | `goal_table` / `table_tidy` | manual_subgoals（无 Hermes） | 酒瓶 300 步失败；碗**从未执行** | `null` | `false` | elapsed_s 187.545 |
| `table_direct` | `goal_table` / `table_tidy` | direct_vla | 复合 `table_both` 单次 600 步失败 | `null` | `false` | elapsed_s 361.806 |
| `table_shifted` | `goal_table_shifted` / `table_shifted_tidy` | manual_subgoals（无 Hermes） | 碗 300 步失败；酒瓶**从未执行** | `null` | `false` | elapsed_s 197.762 |

- 表中 `elapsed_s` / `wall_s` 均取自对应 JSON 原始字段，未做换算。
- **`run_ok` / 进程退出码不等同于任务成功**：Hermes 行的 `run_ok=true` 只表示计划流程跑完并如实返回 blocked。
- 真实 Hermes 行为 **2 次 Hermes CLI session**（`20261006_195554_6151d6` initial、`20261006_200009_7d48b2` repair），**CLI session 数不是 API 请求数**。
- pilot **n=1** 只支持这些**观测**，**不能**推断一般成功率，也**不能**建立 Hermes 信息增益结论；`basket` / `mugs` / 占位（occupied）场景此后已各做过 pilot 实测：其中**受测的物理放置执行均失败**（见第三节），而**完全占满的 `mugs_full` 以 `unsupported` 正确拒绝**（**0 个机器人动作**、`task_success=true`，见第四节）——正确拒绝是**决策成功**，**不是** VLA 物理成功，因此**不声称**可靠执行。

## 二、原始证据链接（全部现存 JSON / PNG / MP4）

JSON（oracle 的 `oracle_source` 保持 `preauthored_fixture`）：

- [table_forward_manual.json](table_forward_manual.json) —— `table_forward` manual 行。
- [hermes_table_tidy.json](hermes_table_tidy.json) —— 真实 Hermes `table_tidy` 行（含 `usage`、`repair_history`）。
- [hermes_table_trace.json](hermes_table_trace.json) —— 两个 Hermes CLI session 的工具调用 / 图像证据轨迹。
- [table_comparisons.json](table_comparisons.json) —— `table_reverse` / `table_direct` / `table_shifted` 三行。
- [secondary_manual_pilots.json](secondary_manual_pilots.json) —— `basket_direct` / `free_left` 两条 direct/manual 物理执行记录。
- [hermes_free_right_initial_failed.json](hermes_free_right_initial_failed.json) —— free-right Hermes 记录（白杯 300 步失败 + 错误对象修复 `yellow_mug_right` 300 步失败）。
- [hermes_basket_two.json](hermes_basket_two.json) —— basket 双罐头 Hermes 多作业记录（两作业累计 600 步）。
- [hermes_bowl_only.json](hermes_bowl_only.json) —— 单目标 bowl-only 正对照（成功）。
- [hermes_missing_bin.json](hermes_missing_bin.json) —— 缺垃圾桶请求（`unsupported`）。
- [hermes_mugs_full.json](hermes_mugs_full.json) —— 满盘请求（`unsupported`）。
- [hermes_ambiguous_cleanup_initial_failed.json](hermes_ambiguous_cleanup_initial_failed.json) —— 含糊「清理」首次（incorrect execute，blocked）。
- [hermes_ambiguous_cleanup_retest.json](hermes_ambiguous_cleanup_retest.json) —— 同请求澄清重测（`clarify`）。
- [repair_scope_live_probe.json](repair_scope_live_probe.json) —— 修复范围守卫的手工协议探针（HTTP 409 / `repair_scope`）。

图片（agentview）：

- [hermes_table_initial.png](hermes_table_initial.png) —— Hermes `table_tidy` 初始画面。
- [hermes_table_final.png](hermes_table_final.png) —— Hermes `table_tidy` 终态画面。
- [shifted_bowl_initial.png](shifted_bowl_initial.png) —— `table_shifted` 初始画面。
- [shifted_bowl_final.png](shifted_bowl_final.png) —— `table_shifted` 终态画面。
- [basket_direct_initial.png](basket_direct_initial.png) / [basket_direct_final.png](basket_direct_final.png) —— `basket_direct` 初始 / 终态画面。
- [free_left_initial.png](free_left_initial.png) / [free_left_final.png](free_left_final.png) —— `free_left` 初始 / 终态画面。
- [free_right_initial.png](free_right_initial.png) / [free_right_final.png](free_right_final.png) —— free-right 初始 / 终态画面。
- [hermes_basket_initial.png](hermes_basket_initial.png) / [hermes_basket_final.png](hermes_basket_final.png) —— 真实 Hermes basket 双作业记录的初始 / 终态画面。
- [hermes_bowl_initial.png](hermes_bowl_initial.png) / [hermes_bowl_final.png](hermes_bowl_final.png) —— 单目标 bowl-only 正对照的初始 / 终态画面。
- [mugs_full_scene.png](mugs_full_scene.png) —— 完全占满场景（`mugs_full` 正确 `unsupported` 决策所用画面）。

原始回放（在 [../../demos/](../../demos/)，共 11 个 v2 回放）：

- [hermes-table-bowl.mp4](../../demos/hermes-table-bowl.mp4) —— Hermes `table_tidy` 碗 95 步**成功**段。
- [hermes-table-wine-failed.mp4](../../demos/hermes-table-wine-failed.mp4) —— 酒瓶 300 步**失败**段。
- [hermes-table-wine-retry-failed.mp4](../../demos/hermes-table-wine-retry-failed.mp4) —— 修复后酒瓶再次 300 步**失败**段。
- [shifted-bowl-failed.mp4](../../demos/shifted-bowl-failed.mp4) —— `table_shifted` 碗 300 步**失败**段。
- [basket-direct-failed.mp4](../../demos/basket-direct-failed.mp4) —— `basket_direct` 复合 `basket_both` 600 步**失败**段。
- [free-left-failed.mp4](../../demos/free-left-failed.mp4) —— `free_left` 白杯 300 步**失败**段。
- [free-right-failed.mp4](../../demos/free-right-failed.mp4) —— free-right 白杯（`white_mug_right`）300 步**失败**段。
- [mugs-wrong-object-repair.mp4](../../demos/mugs-wrong-object-repair.mp4) —— free-right 记录中那次**错误对象修复**（`yellow_mug_right`）段。
- [hermes-basket-step-01.mp4](../../demos/hermes-basket-step-01.mp4) —— basket 第一个执行子目标（字母汤）300 步**失败**段。
- [hermes-basket-step-02.mp4](../../demos/hermes-basket-step-02.mp4) —— basket 第二个执行子目标（番茄酱）300 步**失败**段。
- [hermes-bowl-only.mp4](../../demos/hermes-bowl-only.mp4) —— 单目标 bowl-only 碗 95 步**成功**段。

回放只是仿真动作帧（原始 20 fps 的动作帧），**不是**实时 / 墙钟录屏，也**不包含** Hermes 规划与等待时间。

## 三、物理执行记录（direct / manual / Hermes，n=1/条件）

所有行同为 seed 0、初始状态 0、n=1/条件；`plan_success=null` 表示被 blocked（既非 `false` 也非 `true`）。`elapsed_s` 取自 driver 原始字段，作业的 `wall_s` 是动作阶段墙钟时间；**Hermes 行的 `wall_s` 包含 Hermes 与等待时间**。CLI session 数**不是** API 往返数。

| 记录 | 场景 / case | 执行方式 | 作业 / 步数 | plan_success | task_success | 时间 |
|---|---|---|---|---|---|---|
| `basket_direct`（manual/direct） | `basket_two` / `basket_two_cans` | direct_vla | `basket_both` 单作业 600 步失败 | `null` | `false` | elapsed_s 344.572；job wall_s 339.083 |
| `free_left`（manual） | `mugs_right_occupied` / `mugs_free_left` | manual_subgoals | `white_mug_left` 单作业 300 步失败 | `null` | `false` | elapsed_s 179.101；job wall_s 173.785 |
| free-right（真实 Hermes，初次失败） | `mugs_left_occupied` / `mugs_free_right` | Hermes + 一次修复 | 白杯 `white_mug_right` 300 步失败；修复改成 `yellow_mug_right` 300 步失败（2 作业，累计 600 步） | `null` | `false` | wall_s 372.179 |
| basket（真实 Hermes） | `basket_two` / `basket_two_cans` | Hermes + 一次修复 | 字母汤 `soup_to_basket` 300 步失败；修复重排为 `[sauce_to_basket, soup_to_basket]` 后番茄酱 `sauce_to_basket` 300 步失败；待执行（`pending`）字母汤**没有再次执行**（2 作业，累计 600 步） | `null` | `false` | wall_s 375.921 |
| bowl-only（真实 Hermes，正对照） | `goal_table` / `table_bowl_only` | Hermes（单目标） | `bowl_to_plate` 单作业 95 步**成功** | `true` | `true` | wall_s 72.191；job wall_s 54.055 |

- bowl-only 是**单目标正对照**：`run_ok` / `chain_ok` / `plan_success` / `task_success` 均为 `true`，`protected_satisfied` / `objects_ok` 为 `true`，仅 **1 次 Hermes CLI session**。它是**单目标成功**，**不是**多目标成功。
- free-right 行 `protected_satisfied=false`、`objects_ok=false`（`decision_ok=true`）。
- basket 行 `protected_satisfied` / `objects_ok` 均为 `true`，`pending_capability_ids=["soup_to_basket"]`；pending **不等于**正在运行或已完成。
- `elapsed_s`（manual/direct driver 字段）与 job `wall_s` 是**不同**字段；不要混用。
- 每行各自绑定其软件版本；free-right 的错误对象修复**早于**修复范围守卫（见第五节），**未**在守卫上线后重测。

## 四、意图 / 容量记录（unsupported / clarify，n=1）

| 记录 | 用户请求 | case | decision | 作业 | plan_success | task_success | 时间 |
|---|---|---|---|---|---|---|---|
| [hermes_missing_bin.json](hermes_missing_bin.json) | 把酒瓶扔进垃圾桶。 | `missing_bin` | `unsupported` | 0 | `null` | `true` | wall_s 18.933；1 CLI |
| [hermes_mugs_full.json](hermes_mugs_full.json) | 把白杯放到一个空盘子上，不能移动已有的杯子。 | `mugs_full` | `unsupported` | 0 | `null` | `true` | wall_s 20.931；1 CLI |
| [hermes_ambiguous_cleanup_initial_failed.json](hermes_ambiguous_cleanup_initial_failed.json) | 清理一下酒瓶。 | `ambiguous_cleanup` | `execute`（错误） | `wine_to_rack` 300 步失败 | `null` | `false` | wall_s 210.399；2 CLI |
| [hermes_ambiguous_cleanup_retest.json](hermes_ambiguous_cleanup_retest.json) | 清理一下酒瓶。 | `ambiguous_cleanup` | `clarify` | 0 | `null` | `true` | wall_s 18.599；1 CLI |

- 场景无垃圾桶时，`unsupported` 是**正确决策**：**0 个机器人动作**、`task_success=true`，这**不**代表 VLA 操作成功；`clarify` 同理。
- 含糊「清理」首次为 **incorrect execute**：执行 `wine_to_rack` 300 步失败后被 **blocked**；`decision_ok=false`、`objects_ok=false`、`task_success=false`；`repair_history` 为空。修复阶段解释预算和执行失败、未提交新执行；首次歧义解读仍错误。取消请求到达时计划已是 blocked，因此这不是一次成功取消执行的记录。
- 同请求重测改为 `clarify`（0 作业、`task_success=true`）。

## 五、修复范围守卫手工探针

[repair_scope_live_probe.json](repair_scope_live_probe.json) 是一条 `manual_contract_probe`（`source_commit` `6815bf98e77dd50f468a3254a0c29ffdccf43841`）：

- `before`：`mugs_left_occupied` 会话，原声明目标 `white_mug_right`，仅 1 步即 `budget_exhausted` 被 blocked（`plan_success=null`）。
- 探针请求把 `white_mug_right` 改成 `yellow_mug_right`：**新增了一个物体**（`white_yellow_mug_1`），**而目的地仍是右侧盘子**（`plate_2`，与原目标相同）；通用守卫**仍**拒绝**新引入的物体或目的地**（本条探针只改变了物体）。
- 服务端返回 **HTTP 409、`reason=repair_scope`**，detail：修复只能重排 / 重试 / 恢复原计划**已声明**的目标，新增物体或目的地需要**新的用户请求**。
- `after` 与 `before` 的 plan、jobs、`repair_history`、`steps`、`scene_version`、`env_instance_id`、`episode_resets` **完全一致**：守卫在入队 / 执行**之前**即以 409 拒绝。
- `contract_ok=true`；独立 oracle 评测 `task_success=false`（守卫**不读** oracle）。
- 该守卫只把修复限制在**最初声明的目标**内，**不能**保证初始 Hermes 解读正确，也**不能**避免 VLA 的附带碰撞。
- 更早那次 free-right 错误对象修复**早于**本守卫上线，**保留未改**；这**不**意味着物理 free-right 任务在守卫上线后有重测。

## 六、Hermes 会话与原生图像证据

- **首次 Hermes pilot 遇到 lazy MCP 错误调用，并误解了修复预算**：`hermes_table_trace.json` 里可见先以 lazy `tool_search` / `tool_describe` / `tool_call` 方式反复摸索 MCP 工具，repair 相位只提交了 `["wine_to_rack"]` 并在 rationale 中解释「一次预算片内塞两步导致预算耗尽」。后续源码已针对这些 **prompt / 工具引导**问题做了修正：**修正后的正确 MCP 调用**已在真实的 missing-bin 与后续试验（missing_bin / cleanup / basket）中被验证，**改进后的预算证据解释**也在真实的 cleanup / basket 试验中被使用。但**该首次 trace 与评分保持不变**，**不代表任何新的成功**，也不代表原试验失败被改写。
- **CLI session 数不是 API 请求数**：`hermes_table_tidy.json` 的 `hermes_invocations=2` 指两次 CLI session，并非 API 调用次数。
- **原生图像用量未被导出**：已安装 Hermes chat native-image 路径未导出 usage，`usage` 中 `available=false`、`api_calls` / `input_tokens` / `output_tokens` 均为 `null`，**没有可估计的零成本**结论。
- **真正的 native-image 证据**由三部分组成：
  1. CLI attachment 标记（`native_attachment_markers`）；
  2. 精确 session 的运行时行（runtime line）含 `[1 image]`；
  3. `vision_analyze` 的 native fast-path 返回 `already_in_context=true`。
  initial 与 repair 两个精确 session **各有一条 attachment marker、各自一条 `[1 image]` 行**。
- **满盘正确拒绝（`mugs_full`）用到了公开的场景 ID / 描述**以及**图像**，**不是**孤立的「视觉增益」证明。
- DB 中 `stored_content_image_blocks=0`，是因为用户消息按文本序列化；**API image block 未被保留**，原始网络 payload 也未捕获。因此**不能**用 DB 的 image-block 计数来证明 vision。

## 七、限制与注意事项

- **历史 v1 的酒瓶成功用的是 `libero_goal/9`**（见 v0.1.0 demo），**当前 table 场景使用 `goal8`**；两者是**不同的任务初始化**，所以本次酒瓶失败**不能仅归因于动作顺序**。
- **shifted 场景**是沿 x 施加 **±0.06 m 偏移**的复制场景，并带**初始化稳定（settle）**；它**不是**纯单因子对照，也**不是**一个完全合并的通用场景。
- 两个占位（occupied）左右变体的 300 步失败**不能**证明模型从未学过左 / 右，也**不能**证明它完全没有该能力。
- 目前的**物理多目标 / table / basket / 占位 / shifted** pilot **均未成功**；每条件 **n=1** 且绑定其软件版本，部分旧失败**早于** prompt / 修复守卫改动。
- direct/manual **不是** Hermes，且预算 / 修复不同；**不**据此推断泛化可靠性或 Hermes 信息增益。
- **执行顺序 / 版本身份**：只有精确 `request_id` 的最终终态参与评分，防止陈旧评测。
- **持久性证据**以状态哈希为准：`table_forward`(manual)、Hermes table、free-right 与 basket 多作业记录共享**不变的** `env_instance_id` / `episode_resets`，且每一段的前一 `state_after_sha` 等于下一段的 `state_before_sha`。
- **basket 记录**：两作业累计 600 步，修复顺序 `[sauce, soup]`；sauce 失败**阻止**了第二次 soup 重试；`pending` **不**表示正在运行或已完成。
- oracle 只保护**显式声明的谓词**（含可选的 `protected_positions` 位移检查）；`table_wine_only` 中碗 / 奶酪位移 **≤ 0.08 m** 的阈值是 **fixture 约束**，**不是**实测性能结果；`executed_objects` 只是 job 元数据，**不是**接触或碰撞传感。
- bowl-only 当前样本中 `wine_bottle_1` 位移 `3.2907010479152855e-13 m`、`cream_cheese_1` 位移 `1.4210854715202004e-14 m`（阈值均 `0.08 m`）——这只是**一个样本**里的**近数值零**，**不是**通用零误差或「从未接触」保证。
- **满盘正确拒绝**同时用到了公开的场景 ID / 描述**与**图像，**不是**孤立的视觉增益证明。
- **修复范围守卫**只把修复限制在**最初声明的目标**内，且**不读** oracle；它**不**修复所有初始意图错误，也**不**防止 VLA 的附带接触。
- **没有**全面碰撞 / 接触审计，也**没有**训练得到的 robot look-around / 搜物、可靠物体跟踪或任意货架空位控制；额外仿真相机视图**只辅助 Hermes**，VLA 仍保留**两路原生相机输入**。
- **任意自然语言**若无预先编写的 fixture，`task_success` 保持 `null`。
- 已安装 Hermes chat native-image 路径**不导出 usage**：`available=false`、`api_calls` / `input_tokens` / `output_tokens` 为 `null`，**不能**解读为「零成本」。
- host-local 的 direct/manual 审计**不是** Hermes 实验；pilot **n=1** 不构成成功率或信息增益结论。
