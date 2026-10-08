# Agentic VLA System

最新（2026-10-08）：新的「规划与 VLA 起始状态诊断」已完成，完整报告见 [First_Phase/scene_demo/results/2026-10-08-skill-context/README.md](First_Phase/scene_demo/results/2026-10-08-skill-context/README.md)。在给定场景与能力目录下，Hermes 五次请求中有四次给出正确的真实提交。基线酒瓶试验为 native 2/2、shared_initial 1/2、after_bowl 0/2，仅限所测起始状态与两个种子；预注册的准备对照在酒瓶阶段之前就被阻断，**不是**一次新的酒瓶失败。本轮没有训练，也没有生产技能改进。

上一轮（2026-10-08）：酒瓶抓取与配对配置实验已完成，完整报告见 [First_Phase/scene_demo/results/2026-10-08-grasp-config/README.md](First_Phase/scene_demo/results/2026-10-08-grasp-config/README.md)。默认配置保留 baseline，本轮没有训练；抓取守卫在失败或停滞时把当前执行轮提前停下（原生失败 119 步、共享 109 步，对比原本的 300 步预算），覆盖范围仅限酒瓶子目标。

本仓库有两条线：

- **v0.1.0（历史基线，已实测）**：单任务集成演示。运行链为自然语言 → Hermes（qwen3-vl-plus）→ MCP → SmolVLA LIBERO checkpoint → Franka Panda。GPT 与 DeepSeek 仅负责开发期的规划与代码执行，不参与这条运行时控制链。
- **v2（当前实验入口，已完成单次 pilot 的实验版）**：持久场景多子目标。**同一个** LIBERO 仿真器内由 SmolVLA 连续执行多个子目标，Hermes 在收到**原生图像**后于场景内做一次规划（失败时最多一次修复）。该线已完成**一次 pilot 实测**（n=1/条件，seed 0，初始状态 0），其后另补充了 **direct/manual 物理执行**、**意图 / 容量**与**单目标正对照**记录（唯一 `task_success=true` 的物理结果是**单目标** bowl-only，**不是**多目标成功）。本文档不声称任何成功率结论，全部实测行与限制见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md)。

本文档是该仓库的**根入口**，汇总 v0.1.0 的已实现范围、运行链、启动方式与精选 Demo，并给出 v2 实验线的入口。更细的运行、环境、结果字段与历史实测说明见子目录 README。

## 相关文件

| 内容 | 相对链接 |
|---|---|
| 当前实验入口（持久场景 v2，已完成单次 pilot 的实验版） | [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md) |
| v2 pilot 实测原始证据（2026-10-06） | [First_Phase/scene_demo/results/2026-10-06/README.md](First_Phase/scene_demo/results/2026-10-06/README.md) |
| 酒瓶分阶段诊断六次试验报告（2026-10-07，wine-only） | [First_Phase/scene_demo/results/2026-10-07-wine/README.md](First_Phase/scene_demo/results/2026-10-07-wine/README.md) |
| v0.1.0 详细运行说明（子目录 README） | [First_Phase/libero_demo/README.md](First_Phase/libero_demo/README.md) |
| 验收结果记录 | [First_Phase/libero_demo/acceptance_results.json](First_Phase/libero_demo/acceptance_results.json) |
| 精选 Demo 元数据（权威值来源） | [First_Phase/libero_demo/demos/manifest.json](First_Phase/libero_demo/demos/manifest.json) |
| 早期阶段记录 | [First_Phase/README.md](First_Phase/README.md)（**早期阶段记录**，不代表当前实现） |

## v0.1.0 初始范围（历史）

以下描述适用于 v0.1.0 历史单任务演示：

- 仅支持**从任务目录选择一个标准任务**，完成**单次仿真执行**。
- 不支持同场景多子目标规划。
- 此版本使用预训练 checkpoint，没有训练或微调。

当前实验入口 v2 改为**持久场景多子目标**：**同一个**仿真器内连续执行多个子目标，并由 Hermes 接收**原生图像**做场景内规划。v2 的范围、工作流与命令见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md)。

## 运行链（v0.1.0，历史已实现）

```mermaid
flowchart TD
    A[自然语言请求] --> B["Hermes qwen3-vl-plus<br/>选择一个 suite / task_id"]
    B --> C["MCP 工具桥<br/>execute_libero_task"]
    C --> S["执行服务 service.py<br/>按 task_id 创建并重置场景<br/>取标准英文 instruction"]
    S --> E["LIBERO 场景中的 Franka Panda"]
    S -- "标准指令" --> D["SmolVLA"]
    D -- "末端与夹爪动作" --> E
    E -- "两路图像＋机器人状态" --> D
    E -- "环境成功判定／进度" --> F["执行服务保存结果"]
    F -- "查询与等待" --> B
    F --> W["网页结果与视频"]
    B -- "最终说明" --> W
```

- **Hermes 不控制每一个动作步骤**：v0.1.0 中 Hermes 只负责理解自然语言、读取任务目录、选择 suite / task_id 并编排调用。
- MCP 只转发工具调用；执行服务创建与重置场景并提供标准英文 instruction，SmolVLA 与环境的图像、机器人状态和动作循环构成真正控制闭环。

> v2 实验入口不再受此 v0.1.0 限制：Hermes 通过 `--image` 接收**原生 agentview 图像**做场景内规划，并在**同一个**仿真器内执行多子目标，职责划分见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md)。

## 启动

### v0.1.0（历史）

```powershell
# [Windows PowerShell] 启动本地网页与执行服务
powershell -ExecutionPolicy Bypass -File D:\FYP\First_Phase\libero_demo\start_demo.ps1
```

推荐访问 `http://127.0.0.1:8080`。

### v2 实验入口（持久场景，已完成单次 pilot 的实验版）

```powershell
# [Windows PowerShell] 启动持久场景 v2（场景服务 8767 + 网页 8081）
powershell -ExecutionPolicy Bypass -File D:\FYP\First_Phase\scene_demo\start_demo.ps1
```

推荐访问 `http://127.0.0.1:8081`；使用、停止与显存要求见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md)。

当前启动依赖本机已有的 WSL Ubuntu、旧 torch / lerobot 环境，以及 Hermes 配置与凭据。项目仍使用本机绝对路径和共享环境，不能声称新机器一键运行。

## 精选 Demo 视频

下图为精选 Demo 的网页画面：

![web-success](First_Phase/libero_demo/demos/web-success.jpg)

| 视频 | 原始请求 | suite / task_id | 步数 | 真实仿真秒数 | 完整请求秒数 |
|---|---|---|---|---|---|
| [alphabet-soup.mp4](First_Phase/libero_demo/demos/alphabet-soup.mp4) | 把字母汤罐头放进篮子里。 | libero_object / 0 | 152 | 80.857 | 95.964 |
| [bowl-on-plate.mp4](First_Phase/libero_demo/demos/bowl-on-plate.mp4) | put the bowl on the plate | libero_goal / 8 | 78 | 45.770 | 58.902 |
| [stove-on.mp4](First_Phase/libero_demo/demos/stove-on.mp4) | 开启一下烹饪的炉灶 | libero_goal / 7 | 72 | 41.185 | 55.624 |
| [wine-on-rack.mp4](First_Phase/libero_demo/demos/wine-on-rack.mp4) | 请收拾一下酒瓶，放到架子上 | libero_goal / 9 | 173 | 90.998 | 105.763 |

四个视频仅包含动作帧的 20 fps 回放，不包含 Hermes 规划与等待时间，不是实时录屏。仿真耗时与完整请求耗时分别列出，不能用视频时长代表真实运行速度。上表为 v0.1.0 历史精选视频，数值保持不变。

v2 pilot 另选了**十一个**原始回放（bowl 成功段、酒瓶首次失败段、Hermes 修复后酒瓶再次失败段、shifted 碗失败段，以及后补的 basket / free-plate / Hermes basket / bowl-only 段），逐帧数值与限制见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md) 与 [First_Phase/scene_demo/results/2026-10-06/README.md](First_Phase/scene_demo/results/2026-10-06/README.md)：

| 视频 | 对应 pilot 记录 |
|---|---|
| [hermes-table-bowl.mp4](First_Phase/scene_demo/demos/hermes-table-bowl.mp4) | Hermes `table_tidy`：碗 95 步成功 |
| [hermes-table-wine-failed.mp4](First_Phase/scene_demo/demos/hermes-table-wine-failed.mp4) | Hermes `table_tidy`：酒瓶 300 步失败 |
| [hermes-table-wine-retry-failed.mp4](First_Phase/scene_demo/demos/hermes-table-wine-retry-failed.mp4) | Hermes `table_tidy`：修复后酒瓶再次 300 步失败 |
| [shifted-bowl-failed.mp4](First_Phase/scene_demo/demos/shifted-bowl-failed.mp4) | `table_shifted`：碗 300 步失败 |
| [basket-direct-failed.mp4](First_Phase/scene_demo/demos/basket-direct-failed.mp4) | `basket_direct`：复合 `basket_both` 600 步失败 |
| [free-left-failed.mp4](First_Phase/scene_demo/demos/free-left-failed.mp4) | `free_left`：白杯 300 步失败 |
| [free-right-failed.mp4](First_Phase/scene_demo/demos/free-right-failed.mp4) | free-right：白杯（`white_mug_right`）300 步失败 |
| [mugs-wrong-object-repair.mp4](First_Phase/scene_demo/demos/mugs-wrong-object-repair.mp4) | free-right：错误对象修复（`yellow_mug_right`）段 |
| [hermes-basket-step-01.mp4](First_Phase/scene_demo/demos/hermes-basket-step-01.mp4) | basket：第一个执行子目标（字母汤）300 步失败 |
| [hermes-basket-step-02.mp4](First_Phase/scene_demo/demos/hermes-basket-step-02.mp4) | basket：第二个执行子目标（番茄酱）300 步失败 |
| [hermes-bowl-only.mp4](First_Phase/scene_demo/demos/hermes-bowl-only.mp4) | bowl-only：单目标碗 95 步成功 |

这些回放同样是**原始 20 fps 的仿真动作帧**，**不是**实时 / 墙钟录屏，也**不包含** Hermes 规划与等待时间；`run_ok` / 进程退出码**不等同于**任务成功。

版本管理约定：运行视频默认排除；仅 **v0.1.0 的四个精选小视频 + v2 的十一个原始回放**（上表）明确入 Git（共 **十五个** 精选视频）。凭据、权重、虚拟环境和日志仍排除。

## 局限与后续计划

**v0.1.0（历史）局限：**

1. 单任务入口；宽泛的整理目标会先澄清。
2. 任务切换会重置到不同场景，未实现同场景多子目标执行。
3. WebSearch 与可配置用户偏好尚未接入。
4. 40 个任务入口不代表 40 个任务全部成功；已有 drawer 任务在 300 步上限失败。
5. 公开 checkpoint 模型卡标注训练数据 unknown，不能断言具体训练覆盖。
6. 本机绝对路径与共享旧环境使可移植性有限。

**v2（当前实验入口）已完成单次 pilot 的实验版 —— 实测与待验证项（保留）：**

- 已完成**一次 pilot 实测**（seed 0、初始状态 0、n=1/条件）并补充了 direct/manual、意图 / 容量与单目标正对照记录；全部实测行见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md) 与 [First_Phase/scene_demo/results/2026-10-06/README.md](First_Phase/scene_demo/results/2026-10-06/README.md)。本文档不声称任何成功率结论，`run_ok` / 进程退出码**不等同于**任务成功。
- 目前的**物理多目标 / table / basket / 占位 / shifted** pilot **均未成功**；**唯一**的 `task_success=true` 物理结果是**单目标** bowl-only 正对照（**不是**多目标成功）。`basket` / `mugs` / 占位场景受测的物理放置均失败；满盘请求以 `unsupported` 正确拒绝（0 动作、`task_success=true`），属于决策成功。偏移与占位已测试，但可靠泛化仍未建立；更广的新场景与视角变化仍待验证。
- host-local 的 direct/manual 审计不能建立 Hermes 信息增益结论；pilot n=1 不能支撑宽泛成功率结论。
- **修复范围守卫**只把修复限制在原计划**已声明**的目标内（新增物体/目的地需新的用户请求），且**不读** oracle；它不能修复所有初始意图错误，也不能防止 VLA 的附带接触。
- 独立 oracle 只保护**显式声明的谓词**（含 `table_wine_only` 的碗/奶酪位移 ≤ 0.08 m 等 **fixture 约束**），属于**可测代理**而非完整物理接触验证：尚无附带接触/碰撞的全面审计（`executed_objects` 只是 job 元数据，不是接触传感），也不能保证从未触碰受保护物体；且**不为任意自然语言请求自动生成 oracle**——`task_success` 必须来自预先编写的测试用例，否则为 `null`。详见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md)。
- 没有「完全合并的通用场景」，没有训练 / 微调 / 新 VLA 权重下载；额外仿真相机视图只辅助 Hermes，VLA 保持**两路原生相机输入**；`candidate` 能力不是可靠性保证。
- 根仓库独立通过了 **178 个自动化测试**，这属于**软件验证**，**不是**机器人任务成功。

**后续计划：** 能力审计、独立测试用例、持续场景与真实视觉输入已在 v2 实验线中落地（pilot 证据见 [First_Phase/scene_demo/README.md](First_Phase/scene_demo/README.md)）；WebSearch 与可配置用户偏好等仍为计划，尚未实现。

以上**只声明已实现的链路与已知不足**，不列出任何未经实测的成功率、微调结果、训练覆盖或泛化结论。

## 初版发布

首次远端发布采用当前已验收版本的独立文件快照；本地开发历史保留，后续远端开发以 main 为基线。
