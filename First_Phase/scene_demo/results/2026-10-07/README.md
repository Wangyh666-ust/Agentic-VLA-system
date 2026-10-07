# 放置完成实验：结论与证据（2026-10-07）

本目录记录 **2026-10-07** 一轮「放置完成诊断与修复」：释放验证（`release_verified`）与原生谓词（`native`）停止判据的对照（筛选 / 候选对比 / 基线对比）、释放验证碗的复现、原生续跑实测，以及一轮**独立**的 live 守卫 / 真实 Hermes 集成验证。相关 JSON 报告、统计、前缀校验、资源清单与所选视频均以**仓库相对路径**链接。

## 结论概述

这轮实验要分清两类问题。第一类是执行程序的完成判据本身有错：旧的原生判据要求原生目标谓词连续五帧为真之后才结束作业，它只看谓词，不检验物体是否真的已经松手、是否放稳，于是碗被过早记为完成。这个判据缺陷现已更正。第二类是 VLA 的执行是否成功：判据更正并不会让模型变强，汤等目标仍然失败，可靠的多个子目标执行尚未达成。

新的释放验证判据在「连续五帧满足原生目标谓词」之上再加两项：物体经接触筛查判定为未被持有、放置目标速度足够低。所以「连续五帧」是旧判据本来就有的要求，新的严格判据只是额外增加了接触筛查与低速要求。

## 结论对照表

| 实验 | 实际观察 | 可以得出的结论 |
|---|---|---|
| 碗 / baseline_bf16：旧 95 步 → 新 102 步 | 旧停止点上碗已松手但未放稳；新旧动作前缀逐元素相同 | 差异只来自停止规则，额外动作由 VLA 产生；仅此一样本 |
| 碗 / fp32_h5：旧 83 步 → 新 104 步 | 旧停止点上碗仍被夹爪持有；新旧动作前缀逐元素相同 | 同上，差异只来自停止规则；仅此一样本 |
| 汤：单独请求 300 步、600 步、300+300 重试全部失败 | 一次受测的完整两罐头指令原生判据运行里，汤在第 212 步被筛查为稳定，同一轮的另一个罐头与整条任务仍失败 | 只是本样本、只是原生判据诊断运行，不是释放门对照；不能说释放门修好了汤或 basket |
| 共享场景单酒瓶 300 步失败，之后的碗未被执行 | 此前的单酒瓶测试（goal/9）只由旧原生谓词报告成功，已释放 / 稳定尚未核实 | 不能据此声称具备可靠多目标能力；此前的成功也没有释放门对照 |

两条碗的对照里，每一个额外动作都由 VLA 自己产生：没有强制张开夹爪，没有调用 Hermes，也没有任何训练或微调。因此结论是：过早完成的判据缺陷已被修正，但仍有部分 VLA 失败尚未解决，可靠的多个子目标执行尚未达成。每个条件只有 n=1，不给出整体成功率，也不声称 Hermes 获得物理技能增益。

软件单元测试通过与停止（Stop）测试成功，度量的是控制 / 协议的正确性，不是 VLA 的物理放置成功；32 步的取消测试是刻意的中断，不算放置成功，其计数与原 17 条诊断试验分开，不并入同一总数。

## 常见问题（FAQ）

- **模型是不是根本不会松手？** 不是。碗的释放已经成功复现，所以不能再说模型从不张开夹爪。
- **本地固定的模型卡怎么说？** 固定的[模型卡](https://huggingface.co/HuggingFaceVLA/smolvla_libero)把数据集列为 unknown；当前 HuggingFace main 就是同一个 pinned **`6721902`** revision，**没有** train_config，也**没有**数据集列表。因此确切的训练样本与释放尾部（release tail）都尚未核实；数据集列表缺失本身**不**能证明训练覆盖范围。这里既不声称训练时缺少释放段（不声称存在「训练尾部缺失」），也不声称它一定存在。
- **原生任务成功就等于真的释放了吗？** 不是。本地已经演示过原生任务成功可以先于实际释放出现。所谓的「训练尾部缺失」只是一个假设，并不能证明该 checkpoint 在训练时就省略了释放。
- **v0.1 的单酒瓶示例算成功吗？** 它只由旧的原生谓词报告为成功，**尚未**用新的严格判据核实其已释放 / 稳定，需要以**同一严格判据重新检查**；这**不**能说明它已证明释放 / 稳定。
- **需要分清哪几种失败？** 一是执行程序停得太早；二是策略始终没放对，或者动作序列不稳定；三是虽然下达了张开的指令，但物理释放或物体位置仍然不达标。**下达张开指令不等于真正释放。**
- **`blocked` 是什么意思？** 指的是当前子目标没有完成，或受到约束而无法完成，其余子目标因此暂停；它并不总是等于第一个目标，也不是某个 agent 忘了放碗。
- **`run_ok` 代表成功吗？** 不代表。它只说明程序跑完了；没有匹配的 fixture 时 `task_success` 是 `null`。
- **那一次取消的 32 步运行呢？** 它是一次**刻意的中断**，只测试停止（Stop）控制，不能算作放置成功，也不能算作原来的诊断试验；其计数与原 **17** 条诊断试验**分开**，不并入同一总数。
- **什么算严格完成？** 位置正确、经接触筛查判定未被持有、速度足够低，并且连续五帧成立；其中接触与运动学筛查只是近似。

## 尚未执行的诊断计划

本节只是计划，**尚未执行**；这份文档没有进行任何新的训练或物理实验。

- 用同一个 checkpoint，在固定的 seed / 初始状态下，对比「隔离的单酒瓶 goal9」与「共享场景 goal8」，使用同样的清晰任务文本、正常视觉输入与释放门。
- 对齐并比较原始第七个动作、后处理 / 实际发送的动作、手指关节张开度、物体位姿 / 接触，以及目标谓词。
- 审计确切的训练来源（目前**尚未**核实任何训练来源），如有可能再审计释放尾部。
- 只有在定位到失败点之后，才考虑微调。

<details>
<summary>展开实验设置、逐项数据和原始证据</summary>

## 一、结论要点（accepted finding）

- 在同一 seed 0 / 初始状态 0 / n=1、同一 checkpoint、同一场景与同一指令下，**仅改变停止判据**：旧的**原生谓词**停止在 **95 步**（`baseline_bf16`）与 **83 步**（`fp32_h5`）即结束，而在这两步上**严格释放 / 稳定判据均为 `false`**（`strict_task_success=false`）。
- **同动作前缀反事实**：把同一策略在**完全相同的前缀**下继续执行到 **102 步**（`baseline_bf16`）与 **104 步**（`fp32_h5`）后，**严格判据变为 `true` 并通过**。
- 两次对照的**完整动作前缀逐元素相等**：`prefix_length` 分别为 95 与 83，`all_actions_equal=true`，`first_difference_step=null`，`max_abs_difference=0.0`；且**旧停止步的快照与新运行在同一步的快照完全相等**（`full_snapshot_equal=true`）。因此在这些样本里，**唯一改变量就是停止规则本身**。
- 新增的动作**全部由 VLA 生成**（`assisted=false`，`hermes_calls=0`）；**没有**强制张开夹爪、**没有**脚本化移动、**没有**重训练。
- 本轮由 **GPT 设计 / 复核**，**DeepSeek 实际执行源码编辑**；本地被忽略的 proof 日志**未提交**，本文**从不链接**任何密钥 / 日志文件。

## 二、冻结的环境与来源（frozen setup）

| 项目 | 值 |
|---|---|
| 遗留实测源码（screening / candidate / baseline 三组） | `4357b61e11f1000ee1bf1b5121460219c77485d1` |
| 释放验证门（`release_verified`）源码 | `96c6a375c3a9d29007f96dc99ccf4e147f670df4` |
| 持有守卫（holding guard） | `7deea902a24f5f83dcb99995819e620caf600e6c`（**独立的 live 集成验证来源**，见第八节；本节其余冻结诊断实测的源码为 `4357b61e…` / `96c6a375…`） |
| checkpoint revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| 采样 | seed 0、初始状态 0、**n=1 / 条件** |
| 场景 | `goal_table` = `libero_goal` / `8`；`basket_two` = `libero_10` / `0`，均为**原始未修改**场景 |
| 机器人 / 渲染 | Franka Panda，**20 Hz**，**256 px**，**7 维**动作 |
| 依赖版本 | torch `2.11.0+cu128`、lerobot `0.6.2`、robosuite `1.4.0`、mujoco `3.8.1`、transformers `5.5.4`（**无环境升级**） |
| 保存的默认设置（saved default） | `use_amp=False`、denoising `num_steps=10`、`n_action_steps=1`（**保存**于配置中的值） |
| 实测 `baseline_bf16` 档位（loaded baseline） | 加载器**覆盖**为 `use_amp=True`（BF16），`num_steps=10`、`n_action_steps=1`；即「保存的 `use_amp=False`」与「实测的 BF16」**不是**同一件事，筛选表与既有冻结历史成绩**均按实测档位记录、保持不变** |

- 上述源码 SHA 与哈希见资源清单 [artifact_manifest.json](artifact_manifest.json) 及各统计文件的 `source_metadata` / `source_provenance`。
- 释放验证一组另记录 `source_revision_expected=6721902b…`、`assisted=false`、`hermes_calls=0`、`campaign_wall_s=98.496`。

## 三、完成模式与「已验证放置」判据

本项目有两类完成（终止 + 评分）模式：

- **`native`（原生谓词）**：要求场景的**全部已声明原生目标谓词**在**连续五个（post-action samples）**采样上**均为真**，才判定完成并停止。2026-10-06 起以及本轮 screening / candidate / baseline 三组**默认**使用此模式。
- **`release_verified`（释放验证）**：在原生谓词为真之外，还要求
  1. **原生已声明目标谓词为真**；
  2. **所有具名关节的物理物体**（**全部**此类物体，**不只**已声明的目标物体）经接触筛查判定为**未被持有**（`grasped=False`）；
  3. **已声明的放置目标物体**的 6 维速度向量有限，且**线性范数 ≤ 0.02 m/s**、**角速度范数 ≤ 0.2 rad/s**；
  4. **连续五帧（post-action samples）**均满足以上条件。
- **初始即就绪（`already_satisfied`）例外**：若某作业在**初始完成探针（initial completion probe）**时即为 ready，则该作业可在**两种模式**（`native` 与 `release_verified`）下以**零动作**完成；上述「连续五帧（post-action samples）」的要求**仅适用于已执行作业**。**初始即被持有或持有状态未知**的状态**不能**通过 `release_verified`。
- 「严格（strict）判据」：`strict_task_success` = **最终 FIXED-ORACLE 快照的合取** **AND** **来自最终作业（FINAL job）的最后五个 fixed-oracle strict 候选全部为真**，**绝不**跨重试作业（retry jobs）合并样本；它**不是**「只看最后一次快照」，也**不是**「曾达到过（ever-peak）标志」。三个字段**彼此独立、含义不同、不得混为一谈**：`native_benchmark_success`（「native」列）是**原始环境 full-task 信息成功（info success）在作业期间是否曾出现（EVER）**的判定结果，**不是**最终快照真值；`task_success` 是**为该实验条件独立声明的固定字面 FINAL_ORACLE_GOALS** 的**最终谓词合取**；`strict_task_success` 则在此之上**还要求最终固定 oracle 合取与最终作业最后五个 fixed-oracle strict 候选全部为真**。该固定 oracle 目标**独立于已提交的能力与 Hermes**，**不**由能力调度（capability schedule）或 Hermes 派生。上述字段的历史取值**原样保留**；本次更正**仅为文档说明，不涉及评测代码改动**。
- **接触筛查只是代理**：`grasped` 由接触式判定给出，是**筛查代理**，**不是**权威的物理释放真值；速度是否「有限」也只对**目标物体**的 6 维向量判定。资源清单中的 `measurement_note` 明确记为「screening proxies, not authoritative physical release truth」。

## 四、接受的正常 v2 工作流（正常启动器）

- 正常的 v2 提交流程由 [../../run_service.sh](../../run_service.sh) **显式启用** `release_verified` 完成模式。
- **构造 / 基准对比实验**（[../../placement_experiments.py](../../placement_experiments.py) 的 `--completion-mode`）**默认保留原生谓词**（`native`），以便与历史结果对齐；本轮的对照实验即用 `native` 复现旧停止点。
- **持有守卫**在任何 VLA / 环境步进**之前**运行：当已经持有一个物体时，**阻止切换到另一个物体**；**同一已声明目标**的继续执行**允许**；**持有状态未知**时**阻止**。
- **Hermes 的角色不变**：只做**一次场景内初始规划**，**至多一次**失败触发的修复（`resume_scene_plan`），**绝不逐动作调用**；正常机器人运动期间不调用 Hermes。
- 该守卫**不声称**提升已学 VLA 的能力；它只约束「已持有物体时不得改换目标」这一**程序化边界**。

## 五、配置档位与逐动作延迟（latency screening）

| 档位 | 精度 | denoising `num_steps` | `n_action_steps` | 含队列的逐动作均值（秒） |
|---|---|---|---|---|
| `baseline_bf16` | BF16 | 10 | 1 | 0.684703 |
| `fp32` | FP32 | 10 | 1 | 0.618007 |
| `fp32_d1` | FP32 | 1 | 1 | 0.151394 |
| `fp32_h5` | FP32 | 10 | 5 | 0.096310 |
| `fp32_d1_h5` | FP32 | 1 | 5 | 0.030723 |

- 上表数值来自 screening 组各档位 `per_action_latency.mean`（`includes_queue_hit_steps=true`），按档位匹配顺序列出：`baseline_bf16 / fp32 / fp32_d1 / fp32_h5 / fp32_d1_h5`。
- 这些数字**只**是**延迟筛选**，**不是**成功率或金钱成本；screening 组**五个 soup 300 步试验全部失败**，因此**不能**据此得出任何「生产档位变更」的结论。

## 六、全部十七条已完成试验（truthful outcomes）

全部为 seed 0 / 初始状态 0 / n=1；「时点」列给出结束步数与结束原因；`blocked` 时计划 `plan_success=null`（**既非 `false` 也非 `true`**），**不等于**任务成功。`native` 列即各次自己的 `native_benchmark_success` 原值。

### 6.1 筛选组（screening，`native`，预算 300 步，源码 `4357b61e…`，场景 `basket_two`，条件 `soup_fresh`）

| trial_id | 档位 | capability | 时点 | task_success | native | strict | 结束时 held |
|---|---|---|---|---|---|---|---|
| `soup_fresh_baseline_bf16_0-0_75949dfc` | baseline_bf16 | soup_to_basket | 300（budget_exhausted） | false | false | false | `alphabet_soup_1` |
| `soup_fresh_fp32_0-0_68d6e4e7` | fp32 | soup_to_basket | 300（budget_exhausted） | false | false | false | `alphabet_soup_1` |
| `soup_fresh_fp32_d1_0-0_19c47d0f` | fp32_d1 | soup_to_basket | 300（budget_exhausted） | false | false | false | 无 |
| `soup_fresh_fp32_h5_0-0_79212887` | fp32_h5 | soup_to_basket | 300（budget_exhausted） | false | false | false | `alphabet_soup_1` |
| `soup_fresh_fp32_d1_h5_0-0_0e2f9433` | fp32_d1_h5 | soup_to_basket | 300（budget_exhausted） | false | false | false | 无 |

### 6.2 候选对比组（candidate_comparisons，档位 `fp32_h5`，源码 `4357b61e…`）

| trial_id | 场景 | capability | 时点 | task_success | native | strict | 结束时 held | 备注 |
|---|---|---|---|---|---|---|---|---|
| `basket_native_fp32_h5_0-0_e53bee90` | basket_two | basket_both | 600（budget_exhausted） | false | false | false | 无 | 完整指令 |
| `sauce_fresh_fp32_h5_0-0_5e8f0036` | basket_two | sauce_to_basket | 300（budget_exhausted） | false | false | false | 无 | |
| `basket_split_fp32_h5_0-0_023addbf` | basket_two | soup_to_basket | 300（budget_exhausted） | false | false | false | `alphabet_soup_1` | `pending=[sauce_to_basket]`（未执行） |
| `bowl_control_fp32_h5_0-0_f20de3b1` | goal_table | bowl_to_plate | 83（success） | true | true | false | `akita_black_bowl_1` | 原生提前完成：**仍被持有**（`goal_held`） |

### 6.3 基线对比组（baseline_comparisons，档位 `baseline_bf16`，源码 `4357b61e…`）

| trial_id | 场景 | capability | 时点 | task_success | native | strict | 结束时 held | 备注 |
|---|---|---|---|---|---|---|---|---|
| `basket_native_baseline_bf16_0-0_ddadbe07` | basket_two | basket_both | 600（budget_exhausted） | false | false | false | 无 | 完整指令 |
| `sauce_fresh_baseline_bf16_0-0_8f65bb5d` | basket_two | sauce_to_basket | 300（budget_exhausted） | false | false | false | `milk_1` | **错误物体** |
| `basket_forced_handoff_baseline_bf16_0-0_a71f1be6` | basket_two | soup_to_basket + sauce_to_basket | 300 + 300（各 budget_exhausted） | false | false | false | `alphabet_soup_1` | 受迫交接，见 7.4 |
| `bowl_control_baseline_bf16_0-0_b2e3a268` | goal_table | bowl_to_plate | 95（success） | true | true | false | 无 | 原生提前完成：**已释放但未稳定**（`released_unsettled`） |

### 6.4 释放验证组（release_verified_bowls，`release_verified`，源码 `96c6a375…`，`goal_table` / `bowl_control`）

| trial_id | 档位 | 时点 | task_success | native | strict | 结束时 held |
|---|---|---|---|---|---|---|
| `bowl_control_baseline_bf16_0-0_a44bf704` | baseline_bf16 | **102**（success） | true | true | **true** | 无 |
| `bowl_control_fp32_h5_0-0_f9818df9` | fp32_h5 | **104**（success） | true | true | **true** | 无 |

释放验证逐步里程碑：

| 档位 | first_native_goal_true | first_release | first_ready | first_stable_five_end | 目标线性 / 角速度范数（末帧） | 最大连续 strict 帧数 |
|---|---|---|---|---|---|---|
| baseline_bf16 | 78 | 79 | 98 | 102 | `1.3039925911622895e-07` / `1.903439490695052e-06` | 5 |
| fp32_h5 | 79 | 85 | 86 | 104 | `1.4735441597011624e-07` / `1.25729773756531e-06` | 5 |

- 两行的 `last_five_completion_ready` 均为 `[true, true, true, true, true]`。
- 总计 **5 + 4 + 4 + 2 + 2 = 17 条已完成试验**（资源清单 `trials_total=17`；`jobs_total=19`，因 `basket_forced_handoff` 有 2 个 job、`soup_retry` 有 2 个 job）。

### 6.5 同动作前缀反事实（prefix checks）

| 档位 | old 报告 / job | old 步数 | new 报告 / job | new 步数 | prefix_length | all_actions_equal | max_abs_difference | 旧 native/strict | 新 native/strict |
|---|---|---|---|---|---|---|---|---|---|
| baseline_bf16 | baseline_comparisons.json / `6095190f…` | 95 | release_verified_bowls.json / `3eef289a…` | 102 | 95 | true | 0.0 | true / **false** | true / **true** |
| fp32_h5 | candidate_comparisons.json / `46e9ca4b…` | 83 | release_verified_bowls.json / `51677b42…` | 104 | 83 | true | 0.0 | true / **false** | true / **true** |

- 两条对照均满足：`same_initial_state_hash=true`、`initial_state_hash=8b8be634a018202744096727ae139fd1401257b2046592ef8e5f8dc164e467bd`、`same_instruction=true`（`"put the bowl on the plate"`）、`same_model_revision=true`、`same_profile_config=true`；`first_difference_step=null`。
- 旧停止步的整份快照与新运行同一步快照 `full_snapshot_equal=true`（`target_object_equal` / `all_objects_equal` / `gripper_qpos_equal` / `held_objects_equal` 均为 `true`）。完整校验见 [release_verified_bowl_prefix_checks.json](release_verified_bowl_prefix_checks.json)。

### 6.6 原生续跑组（soup_continuations，`native`，源码 `96c6a375…`）

两个条件，均为 seed 0 / 初始状态 0 / n=1 / `basket_two` / `soup_to_basket`，各含一次「更大预算」或「同目标重试」运行：

| trial_id | 档位 | 时点（步数） | job / wall_s | task_success | native | strict | 结束时 held |
|---|---|---|---|---|---|---|---|
| `soup_extended_baseline_bf16_0-0_d7571b04` | baseline_bf16 | 600（budget_exhausted） | `09d850c2145c47e2b518e60a99918932` / wall_s 327.893、trial_wall_s 332.112 | false | false | false | `alphabet_soup_1` |
| `soup_retry_baseline_bf16_0-0_b6ec5130` | baseline_bf16 | 300 + 300（各 budget_exhausted） | `4bf7ece72ba94a3ca0ccf5442a3da62e` / wall_s 159.685；`fb9794387d564709a8873627bc2b02cf` / wall_s 165.314、trial_wall_s 328.941 | false | false | false | `alphabet_soup_1` |

- 两行的 `last_five_strict_flags` 均为全 `false`；固定 oracle 目标均为 `[in, alphabet_soup_1, basket_1_contain_region]`。
- `soup_extended`：`state_before_sha=9fb2f99b…`、`state_after_sha=6cb426dc…`、`env_instance_id=1`、`episode_resets=1`。
- `soup_retry`：两段以 `6b0435…` 衔接（`exact_state_chain_equal=true`），`env=[2,2]`、`resets=[1,1]`。
- **前缀一致性**（[soup_continuation_prefix_checks.json](soup_continuation_prefix_checks.json)）：两者的**首个 300 个动作**与原 screening `baseline_bf16` 试验（job `0eed368c…`）**及**受迫交接基线（job `8f183679…`）**逐元素完全一致**（`all_actions_equal=true`、`prefix_length=300`、`max_abs_difference=0.0`、`all_snapshots_equal=true`）。
- **连续 600 段 vs 拼接 retry 600 段**：`prefix_length=600` 下 `all_actions_equal=true` 且 `all_snapshots_equal=true`（`continuous_steps=600`、`retry_concatenated_steps=600`）——**两者逐动作与整份快照完全相等**。
- 结论仅限本样本：**更大预算（600）与同目标重试（300+300）都没有让这一个样本成功**；本文**不**据此宣称全局「无法能力」或任何训练层面的因果结论。

## 七、现象解释（with relative evidence links）

1. **汤始终没进篮子（原子 screening 的 IN 谓词始终为假）**：screening 五个 `soup_fresh` 试验里，原子谓词 `in|alphabet_soup_1|basket_1_contain_region` **全程为假**（如 `baseline_bf16` 抓住汤 226/300 帧，却从未满足「在篮内」）。证据：[screening_statistics.json](screening_statistics.json)。
2. **酱料抓成了错误的牛奶**：`baseline_bf16` 的 `sauce_fresh` 试验抓取的是 `milk_1`（300 步里 112 帧 `grasped=true`、末 20 帧全为 `milk_1`），**不是** `tomato_sauce_1`；末帧 `tomato_sauce_1` 仍 `grasped=false`。证据：[baseline_comparison_statistics.json](baseline_comparison_statistics.json)。
3. **碗的「提前完成」，且两档位末态不同**：`native` 判据在 95 步（`baseline_bf16`）与 83 步（`fp32_h5`）即因原生谓词连续为真而停止，两步上严格判据皆为假，故均为「提前完成」；但**两档位末态不同、不可混为一谈**——`baseline_bf16` 在 95 步**已释放但未稳定**（末态相位 `released_unsettled`、`held_objects=[]`，目标速度仍显著），而 `fp32_h5` 在 83 步**仍被持有**（末态相位 `goal_held`、`held_objects=[akita_black_bowl_1]`、`grasped=true`）。证据：[release_verified_bowl_prefix_checks.json](release_verified_bowl_prefix_checks.json)（`old_strict_task_success=false`）与 [candidate_comparison_statistics.json](candidate_comparison_statistics.json) / [baseline_comparison_statistics.json](baseline_comparison_statistics.json)（`final_fixed_oracle_snapshot.held_objects`）。
4. **受迫交接（forced handoff）证明载体延续**：`basket_forced_handoff_baseline_bf16_0-0_a71f1be6` 的第二个 job 的 `state_before_sha` **等于**第一个 job 的 `state_after_sha` = `6b04350450dcb5cd6174b76f905a2c6d1fd15edb61fe8f18c362555c02dac055`（**以 `6b0435` 起、以 `…055` 止**），且全程 `env_instance_id=3`、`episode_resets=1`——**同一个仿真器、无重置**，因此物体状态**确实被带到下一段**。交接**没有**在篮子之外强制张开夹爪放置；物体在交接边界**仍被持有**（第 2 个 job 末帧 `held_objects=[alphabet_soup_1]`）。该字段同时标注 `deliberately_forced_after_failure=true`、`note="strictly experimental; never a production default recovery"`。证据：[artifact_manifest.json](artifact_manifest.json)。
5. **完整指令在 212 步内放好汤，而原子 300 步失败**：`basket_native_baseline_bf16`（完整指令）的固定 oracle 显示 `alphabet_soup_1` 的 `first_true_step=208`、`first_released_stable_true_step=212`；同一条件里 `tomato_sauce_1` 的 `first_true_step` 为 `null`（这是该目标**首真步为空**，**并非** oracle 真值为 `null`），整任务（fulltask）判定为 `false`。相比之下，单独的原子 `soup_fresh` 跑满 300 步仍从未满足。**原子轨迹与完整指令轨迹的差异只是「观察到的行为」，并非语言因果性的证明**。证据：[baseline_comparison_statistics.json](baseline_comparison_statistics.json)（`fixed_oracle_goal_first_steps`、`fixed_oracle_goal_first_steps_rule`，`score_updated=false`）。
6. **「额外预算」假设在本样本上未被证实**：第六节 6.6 的 `soup_extended`（600 步）与 `soup_retry`（300+300）都**没有**让这一个样本成功；但更大的预算 / 同目标重试仍只是**单一样本**证据，本文**不**据此宣称全局「无法能力」或任何训练层面的因果结论。与此同时，完整指令确实在 212 步内达成了汤的稳定放置，说明该场景**并非**在结构上不可行。

## 八、独立的 live 集成验证（2 checks / 3 jobs，源码 `7deea902…`）

这是一组**独立于**第六节诊断实测的 live 集成验证，来源为 **`7deea902a24f5f83dcb99995819e620caf600e6c`**，证据为 [holding_guard_live_probe.json](holding_guard_live_probe.json) / [hermes_bowl_release_verified.json](hermes_bowl_release_verified.json) / [live_validation_metadata.json](live_validation_metadata.json) / [hermes_bowl_trace.json](hermes_bowl_trace.json)。

### 8.1 手动守卫检查（manual guard，2 个作业，100+0 步）

- 请求 `d22b106e…`、会话 `99ddfea8…`；先执行 **100 步 soup**，随后尝试 **0 步 sauce**。
- 结束原因 `holding_other_object`：**在持有 soup 时，守卫阻止切换到另一个物体**；`expected_guard_block=true`、`assistance=false`、`independent_evaluation_called=false`。
- 结束时 `held=[alphabet_soup_1]`；`state_sha=06f1e751c5df3e05b482327a7c1543bbfb78b5037adb198f1299e266f0ee1c31`；`env_instance_id=1`、`episode_resets=1`。
- 此项**验证守卫的程序化边界正确**，**不**证明 Hermes / VLA 本身的任务能力。

### 8.2 真实 Hermes 检查（real hermes，102 步）

- 请求 `5da476c4…`、会话 `d07ec8f0…`；`capability=bowl_to_plate`，共 **102 步**，`last_five_completion_ready=true`，结束时 `held=[]`。
- 解码后的实际 Hermes 请求为：**`把碗放到盘子上，不要动酒瓶，也不要打开炉灶。`**；所选能力为 `bowl_to_plate`。
- `case_id=table_bowl_only`：独立 fixture 的 `task` / `protection` / `decision` 三项检查均为 `true`；**无修复（repair）发生**。
- 运行期 Hermes 为 **`qwen3-vl-plus`（provider `alibaba-cn`）**，**不是**开发期的 GPT / DeepSeek。**GPT 仅在设计期规划 / 复核，DeepSeek 仅在开发期做实际源码编辑。**
- **一次** Hermes CLI 调用，记录了 **5 次**同一会话的 API 调用（`5x true`）。运行日志中的 `[1 image]` 只证明**原生图像附件**，**不是**「视觉推理」结论。
- 原始 `agent_result` 的 usage 导出仍**不可用 / 为 `null`**；单独的**经净化（sanitized）trace** 恢复了这些计数器。两套计数**含义不同、并非未解释的差异**：日志 trace 的 `input=43129` **包含缓存输入**，canonical DB 的 `input=32505` 是**未缓存输入**、另有 `cache_read=10624`，且 **43129 = 32505 + 10624**；两者 `output` 均为 **621**。本文**不**据此做任何 token / 价格推断，也**不**解读为「零成本」。
- `final_session`：`env_instance_id=2`、`episode_resets=1`、`total_steps=102`。

> 本节的 **2 项检查 / 3 个 job** 与第六节诊断的 **17 trials / 19 jobs** **彼此独立、分开统计**，**不**并入同一总数。

## 九、证据、统计与媒体链接

### 9.1 报告与统计（仓库相对路径）

| 内容 | 链接 |
|---|---|
| 资源清单 / SHA 来源 | [artifact_manifest.json](artifact_manifest.json) |
| screening 报告（gz） | [screening.json.gz](screening.json.gz) |
| screening 统计 | [screening_statistics.json](screening_statistics.json) |
| 候选对比报告（gz） | [candidate_comparisons.json.gz](candidate_comparisons.json.gz) |
| 候选对比统计 | [candidate_comparison_statistics.json](candidate_comparison_statistics.json) |
| 基线对比报告（gz） | [baseline_comparisons.json.gz](baseline_comparisons.json.gz) |
| 基线对比统计 | [baseline_comparison_statistics.json](baseline_comparison_statistics.json) |
| 释放验证报告（gz） | [release_verified_bowls.json.gz](release_verified_bowls.json.gz) |
| 释放验证统计 | [release_verified_bowl_statistics.json](release_verified_bowl_statistics.json) |
| 释放验证前缀校验 | [release_verified_bowl_prefix_checks.json](release_verified_bowl_prefix_checks.json) |
| soup 续跑报告（gz） | [soup_continuations.json.gz](soup_continuations.json.gz) |
| soup 续跑统计 | [soup_continuation_statistics.json](soup_continuation_statistics.json) |
| soup 续跑前缀校验 | [soup_continuation_prefix_checks.json](soup_continuation_prefix_checks.json) |

### 9.2 所选诊断视频（9 条，均为 20 fps 仿真器运动，**不含**模型 / 墙钟等待）

1. [artifacts/screening/baseline_bf16/soup_fresh/job_01/rollout.mp4](artifacts/screening/baseline_bf16/soup_fresh/job_01/rollout.mp4)
2. [artifacts/candidate_comparisons/fp32_h5/bowl_control/job_01/rollout.mp4](artifacts/candidate_comparisons/fp32_h5/bowl_control/job_01/rollout.mp4)
3. [artifacts/baseline_comparisons/baseline_bf16/basket_native/job_01/rollout.mp4](artifacts/baseline_comparisons/baseline_bf16/basket_native/job_01/rollout.mp4)
4. [artifacts/baseline_comparisons/baseline_bf16/basket_forced_handoff/job_02/rollout.mp4](artifacts/baseline_comparisons/baseline_bf16/basket_forced_handoff/job_02/rollout.mp4)
5. [artifacts/baseline_comparisons/baseline_bf16/bowl_control/job_01/rollout.mp4](artifacts/baseline_comparisons/baseline_bf16/bowl_control/job_01/rollout.mp4)
6. [artifacts/release_verified_bowls/baseline_bf16/bowl_control/job_01/rollout.mp4](artifacts/release_verified_bowls/baseline_bf16/bowl_control/job_01/rollout.mp4)
7. [artifacts/release_verified_bowls/fp32_h5/bowl_control/job_01/rollout.mp4](artifacts/release_verified_bowls/fp32_h5/bowl_control/job_01/rollout.mp4)
8. [artifacts/soup_continuations/baseline_bf16/soup_extended/job_01/rollout.mp4](artifacts/soup_continuations/baseline_bf16/soup_extended/job_01/rollout.mp4)
9. [artifacts/soup_continuations/baseline_bf16/soup_retry/job_02/rollout.mp4](artifacts/soup_continuations/baseline_bf16/soup_retry/job_02/rollout.mp4)

### 9.3 旧 / 新「已验证碗」成对首末帧

- `baseline_bf16`（旧 → 新）：
  [旧首帧](artifacts/baseline_comparisons/baseline_bf16/bowl_control/job_01/first.png) ·
  [旧末帧](artifacts/baseline_comparisons/baseline_bf16/bowl_control/job_01/last.png) →
  [新首帧](artifacts/release_verified_bowls/baseline_bf16/bowl_control/job_01/first.png) ·
  [新末帧](artifacts/release_verified_bowls/baseline_bf16/bowl_control/job_01/last.png)
- `fp32_h5`（旧 → 新）：
  [旧首帧](artifacts/candidate_comparisons/fp32_h5/bowl_control/job_01/first.png) ·
  [旧末帧](artifacts/candidate_comparisons/fp32_h5/bowl_control/job_01/last.png) →
  [新首帧](artifacts/release_verified_bowls/fp32_h5/bowl_control/job_01/first.png) ·
  [新末帧](artifacts/release_verified_bowls/fp32_h5/bowl_control/job_01/last.png)

### 9.4 关于 gzip 与 SHA 溯源

- `.json.gz` 为**确定性 gzip**（`gzip_mtime_0`），可无损解压回原始 JSON；检查方式：

  ```python
  import gzip, json
  with gzip.open("screening.json.gz", "rt", encoding="utf-8") as f:
      report = json.load(f)
  ```

- 资源清单同时保留**原始字节 SHA** 与**解压后 SHA**（`original.sha256`、`decompressed_sha256`、`roundtrip_match=true`），因此 `gzip` 包与原始报告可逐字节核对。**完整 JSON 报告以无损 gzip 提供**（可无损解压回原始 JSON），**所选 PNG / MP4 为原始字节拷贝**；本文**不链接**任何密钥 / 日志文件。

### 9.5 独立 live 集成验证证据与视频（第八节）

| 内容 | 链接 |
|---|---|
| 手动守卫 live 探针 | [holding_guard_live_probe.json](holding_guard_live_probe.json) |
| 真实 Hermes 碗释放验证 | [hermes_bowl_release_verified.json](hermes_bowl_release_verified.json) |
| live 验证元数据 | [live_validation_metadata.json](live_validation_metadata.json) |
| Hermes 碗追踪 | [hermes_bowl_trace.json](hermes_bowl_trace.json) |

- live 视频（1 条，20 fps 仿真器运动，**不含**模型 / 墙钟等待）：
  [artifacts/live_hermes_bowl/baseline_bf16/table_bowl_only/job_01/rollout.mp4](artifacts/live_hermes_bowl/baseline_bf16/table_bowl_only/job_01/rollout.mp4)

## 十、复现命令（reproduction commands）

在 WSL 内使用**既有**解释器与配置。下面的示例使用 **`96c6a375`（或之后）的当前源码**，并**显式**给出 `--completion-mode`；它们**不是**当时的历史命令原文（见下文「历史 CLI 分野」）。

```bash
export LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY

# 延迟筛选（native）
/home/yhwang/fyp/libero_demo/venv/bin/python -u /mnt/d/FYP/First_Phase/scene_demo/placement_experiments.py \
  --profiles baseline_bf16 fp32 fp32_d1 fp32_h5 fp32_d1_h5 \
  --conditions soup_fresh \
  --pairs 0:0 \
  --completion-mode native \
  --timeout 1200 \
  --output /home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/repro_screening.json \
  --run-root /home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/repro_screening_runs

# 释放验证碗（release_verified）
/home/yhwang/fyp/libero_demo/venv/bin/python -u /mnt/d/FYP/First_Phase/scene_demo/placement_experiments.py \
  --profiles baseline_bf16 fp32_h5 \
  --conditions bowl_control \
  --pairs 0:0 \
  --completion-mode release_verified \
  --timeout 1200 \
  --output /home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/repro_bowls.json \
  --run-root /home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/repro_bowls_runs

# 原生续跑（native，soup_extended + soup_retry）
/home/yhwang/fyp/libero_demo/venv/bin/python -u /mnt/d/FYP/First_Phase/scene_demo/placement_experiments.py \
  --profiles baseline_bf16 \
  --conditions soup_extended soup_retry \
  --pairs 0:0 \
  --completion-mode native \
  --timeout 1200 \
  --output /home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/repro_continuations.json \
  --run-root /home/yhwang/fyp/scene_demo/placement_experiments/2026-10-07/repro_continuation_runs
```

- **`--timeout 1200` 作为独立参数**给出（与其它 flag 分开写）。
- 每次运行前请设置 `LD_LIBRARY_PATH=/usr/lib/wsl/lib`，并 `unset` 上面列出的全部代理变量（`http_proxy` / `https_proxy` / `HTTP_PROXY` / `HTTPS_PROXY` / `no_proxy` / `NO_PROXY` / `all_proxy` / `ALL_PROXY`）。
- `repro_*` 的**输出 JSON 与运行根目录（`--run-root`）必须是新的、不存在的路径**：runner **拒绝覆盖**已存在的输出 / 运行根。
- 实验前**只能**通过其**作用域内启动器（scoped launcher）**停止**自己注册的 demo GPU 服务**，**绝不**杀其它进程。
- **历史 CLI 分野**：`4357b61e…` 的历史 CLI **没有** `--completion-mode`，也**没有** `soup_extended` / `soup_retry` 条件；因此**逐字**的历史命令**省略**该选项。上面的示例是**当前**源码的复现用法，**不能**当作当时的历史命令原文。

## 十一、局限（limits）

- 全部结论**只**适用于**已采样的 n=1 条件**（seed 0 / 初始状态 0），**不能**外推为一般成功率。
- 接触式抓取（`grasped`）是**筛查代理**，可能**误判**；它不是权威的物理释放真值。
- 「连续五帧」在 20 Hz 下等于 **0.25 仿真秒**，**不足以**保证长期稳定性。
- **没有**任意货架空位控制、**没有**可靠泛化结论、**没有** Hermes 信息增益结论。
- **不修订**任何更早日期的证据声明；本文只补充 2026-10-07 这一轮。
- **没有**测得的长期在线成功率结论；live 部署**已被验证**（第八节的 live 集成验证证明守卫的程序化边界与一次 Hermes 碗任务在固定检查下成立，2 checks / 3 jobs），**不**构成成功率结论。
- 第六节的 `soup_extended`（600）与 `soup_retry`（300+300）**已有结果**：在本样本上**均未成功**（task / strict / native 全为 `false`）；「额外预算」假设在本样本上**未被证实**，但仍**不**构成全局「无法能力」或训练层面因果结论。

</details>
