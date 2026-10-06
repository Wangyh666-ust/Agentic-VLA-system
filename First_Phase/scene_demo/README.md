# 持久场景 v2（实验）— 场景内多子目标 SmolVLA 演示

本目录（`First_Phase/scene_demo/`）是「持久场景 v2」实验线：在**同一个** LIBERO 仿真器内，用 SmolVLA 策略连续执行多个子目标，Hermes 只在场景内做一次规划（失败时最多一次修复）。

> **状态：待实测。** 本文描述的是**已实现的工程设计**；live 证据仍在等待，本文件**不声明任何已实测的 v2 成功率或结果**。

## 一、范围（scope）

- 运行时链路固定为**既有**组件：Hermes（`qwen3-vl-plus`）+ `scene_tools` MCP + 预训练 SmolVLA checkpoint（revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`）+ LIBERO 场景中的 Franka Panda。**不训练、不微调、不下载。**
- GPT 的规划/验收与 DeepSeek 的实际源码执行**仅用于开发期**，不属于运行时控制链。
- v2 与 v0.1.0 单任务演示并存：v0.1.0 见 [../libero_demo/README.md](../libero_demo/README.md)。

## 二、工作流与职责（workflow and responsibilities）

固定流程（职责划分固定，不由模型自由改写）：

1. **先固定场景**：host/服务端一次性创建固定场景（一个常驻 LIBERO 仿真器），并公开 agentview / wrist 画面与公开 `storage_policy`。
2. 用户给出自然语言指令。
3. **Hermes 收到原生图像**（通过 `--image` 传入 agentview PNG），据此挑选物体与能力（capability）**执行顺序**，提交计划后立即结束。
4. **本地 host/harness 等待**：只读轮询精确 `request_id` 的计划终态；host 不替 Hermes 选择能力、不生成计划。
5. **同一 SmolVLA 控制多段动作**：子目标之间**保留同一仿真器**，只重置 VLA 的**动作队列**（不改场景、不重置物体）。
6. **本地判定**：逐子目标用本地谓词判定，最后对全部目标做**最终 AND** 检查。
7. **不逐动作调用 Hermes**，也**不按例程逐子目标**调用 Hermes；正常机器人运动期间不调用 Hermes。
8. 仅当计划 `blocked` 时，最多触发**一次**由失败驱动的 Hermes 修复（`resume_scene_plan`），之后不再空转。

关键区分：

- `plan_success` 来自服务端对已声明目标的最终 AND；`task_success` 是**独立**评测结果。**没有预授权用例（case）时 `task_success` 为 `null`。**
- **用例期望绝不进入 Hermes 的 prompt / MCP**：`case_id` 与 fixture 只在计划终态后用于一次独立评测。

## 三、使用（usage）

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File D:\FYP\First_Phase\scene_demo\start_demo.ps1
```

启动后访问 http://127.0.0.1:8081 。

- `-Stop` **仅影响 v2**：只停 v2 PID 文件中记录、且 cmdline 含 `scene_demo` 的进程，不动 v1 服务。
- 复用旧的 WSL venv / 模型 / Hermes 凭据；**不训练、不下载**，也**不是新机器一键运行**。
- 默认要求 **≥ 6000 MiB 空闲 GPU 显存**。若已有旧的常驻服务占着显存/端口，需要**用户自行**用**旧脚本** [../libero_demo/start_demo.ps1](../libero_demo/start_demo.ps1) `-Stop` 停掉；**本启动器绝不自己杀旧服务**。
- `-MinFreeGpuMiB 3500` 是 root 为**当前主机 pilot** 选定的调试覆盖值，**不是**通用显存要求。注意参数写法：`-MinFreeGpuMiB 3500`。

## 四、场景与能力审计（scene/capability audit）

固定场景（见 [catalog.py](catalog.py)）：

| scene_id | 来源 | 说明 |
|---|---|---|
| `goal_table` | original | goal8：碗 + 酒瓶 + 炉灶 |
| `goal_table_shifted` | experimental | 沿 x 施加 ±0.06 m 偏移的复制场景 |
| `basket_two` | original | libero_10 / 0：两个罐头 |
| `mugs_two` | original | libero_10 / 4：两个杯子 / 两个盘子 |
| `mugs_left_occupied` | experimental | 脚本预置占位物体 |
| `mugs_right_occupied` | experimental | 脚本预置占位物体 |
| `mugs_both_occupied` | experimental | 脚本预置占位物体 |

- **没有**「完全合并的通用场景」，也**没有微调**。
- 能力目录共 **11 条**：**8 条原子能力 + 3 条仅审计的复合指令**，全部使用**同一**模型。
- `candidate` 证据**不是**可靠性保证。
- **原生 original goal 不是唯一终止条件**。
- 额外仿真相机只是**观测辅助**，**不是**训练得到的 wrist-search 技能。
- VLA 保持其**两路原生相机输入**；可达性/视角差异**未经验证**。

## 五、独立评测（independent evaluation）

- 共 **10 个预授权 fixture**（见 [fixtures.json](fixtures.json) / [oracle.py](oracle.py)）：判定为**最终合取（AND）+ 保护物体约束 + decision 合规**；**缺失真值 fail-closed**（视为不满足）。
- oracle **只保护被显式声明的谓词**：即 fixture 中的 `protected_goals`，以及可选的 Euclidean `protected_positions` 约束。例如 `table_wine_only` 额外检查 `akita_black_bowl_1`（碗）与 `cream_cheese_1`（奶酪）相对初始位置的位移 **≤ 0.08 m**；该 **0.08 m 阈值是 fixture 约束，不是实测性能结果**。
- `executed_objects` 只是**被非零 executed 作业寻址**的物体元数据（并须为 `allowed_objects` 的子集），**不是**接触传感器证据。
- 上述约束是**可测代理（measurable proxies）**，与**完整物理接触验证**不同：目前**没有**对附带接触/碰撞的**全面审计**，因此**不能保证字面上从未触碰**受保护物体。
- **不提供**面向任意自然语言请求的**自动 oracle 生成**：`task_success` 必须取自某个**预授权 fixture**，否则保持 `null`。
- 宽泛的整理遵循公开 `storage_policy`；含糊的「清理」要求**澄清**；场景**没有垃圾桶**时「丢弃」**不受支持**。
- **左/右盘不能证明**在单个货架上任意左/右空位放置；占位独占目标会被 **blocked**，而不是堆叠。
- **API 终态/版本归属防止陈旧评测**：只评测精确 `request_id` 的**最终终态**。

## 六、实验命令（experiment commands）

在 WSL 内运行（公共运行环境）：

```bash
export LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config
export MUJOCO_GL=egl

/home/yhwang/fyp/libero_demo/venv/bin/python /mnt/d/FYP/First_Phase/scene_demo/run_experiments.py \
  --conditions table_direct table_forward table_reverse table_shifted \
  --states 0 --seed 0 \
  --output /home/yhwang/fyp/scene_demo/table_pilot.json
```

- 次要条件：`basket_direct basket_split mugs_direct mugs_split free_left free_right`。
- 应提供 `LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config`。
- [run_experiments.py](run_experiments.py) 只用**标准库**，所以仅跑这条 HTTP driver 时**无需**初始化 GL；通用运行时环境可使用 `MUJOCO_GL=egl`。
- **host-local 的 direct/manual 审计不能建立 Hermes 信息增益结论**。
- **pilot n=1 不能支撑宽泛的成功率结论**。
- 所有 flag 的值必须是**独立参数**，尤其 `--states 0 --seed 0` 与 `--max-repairs 1`。

Oracle fixture CLI（**复用现有 session**，不隐藏场景创建——场景需先用网页/服务预先创建）：

```bash
export LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config
export MUJOCO_GL=egl

/home/yhwang/fyp/libero_demo/venv/bin/python /mnt/d/FYP/First_Phase/scene_demo/run_agent.py \
  --session-id <existing> \
  --request '请帮我整理一下桌面，把碗和酒瓶收好。' \
  --case-id table_tidy --max-repairs 1
```

## 七、当前局限（present limits）

- **live 验证仍待实测**；本文件不声明任何已实测的 v2 成功率。
- host-local 的 direct/manual 审计**不能**建立 Hermes 信息增益结论；pilot **n=1** 不能支撑宽泛成功率结论。
- 泛化（可达性/视角差异、新场景、改顺序、空位选择与占位避让）**未经验证**。
- **没有完全合并的通用场景**，也**没有微调**；`candidate` 能力不是可靠性保证。
- **不是新机器一键运行**（复用旧环境与凭据）。

## 相关文件

| 内容 | 相对链接 |
|---|---|
| 执行服务（常驻仿真器） | [service.py](service.py) |
| 场景/能力目录 | [catalog.py](catalog.py) |
| 独立评测 oracle | [oracle.py](oracle.py) |
| 预授权 fixture | [fixtures.json](fixtures.json) |
| Hermes MCP 桥（6 工具） | [mcp_server.py](mcp_server.py) |
| Hermes 视觉规划入口 | [run_agent.py](run_agent.py) |
| direct/manual 审计 runner | [run_experiments.py](run_experiments.py) |
| 本地网页/HTTP API | [app.py](app.py) |
| Windows 一键启动/停止 | [start_demo.ps1](start_demo.ps1) |
| WSL 服务启动器 | [run_service.sh](run_service.sh) |
| 隔离 Hermes profile | [setup_profile.py](setup_profile.py) |
| v0.1.0 历史演示 | [../libero_demo/README.md](../libero_demo/README.md) |
