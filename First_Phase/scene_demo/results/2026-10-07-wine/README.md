# 酒瓶分阶段诊断（2026-10-07）：六次试验的结论与证据

## 本轮结论

这是一次**只针对酒瓶**（wine-only）的六次仿真诊断。用同一套判断标准、同一个预训练 checkpoint，在两种场景里各跑三个配对（seed 0 / 1 / 2），结果分成三类：**1 次严格成功**（瓶子被抬起、放到货架上、松手并稳定）；**2 次卡在抓取阶段**（瓶子没有被稳定抬起）；**3 次其实已经松手并接触到了货架，但瓶子的落点落在 LIBERO 目标区域之外**。

因此，这次诊断**没有**证据表明当前权重普遍无法松手；模型在训练时是否包含「释放尾部」（release tail）仍然未知。本文只做诊断，**不**做自动修正，**不**微调，也**不**修改任何判断标准。

## 六个试验（按固定顺序）

下表的 `seed:init_state` 明确给出每行的采样，`步数` 与 `最大抬升` 直接取自 [summary.csv](summary.csv) / [campaign.json](campaign.json)。

两个条件的含义（通俗说明）：

- `native_goal9`：LIBERO 原始的标准酒瓶任务场景。
- `shared_goal8`：复用的标准碗/盘任务桌面，但本轮只下发酒瓶指令，**不**执行任何碗的子目标。

| 条件 | seed:init_state | 步数 | 最大抬升 | 结果 |
|---|---|---|---|---|
| 原生酒瓶任务（native_goal9） | 0:0 | 178 | 33.12 cm | 严格成功（已释放且稳定） |
| 共享碗/盘桌面（shared_goal8） | 0:0 | 300 | 0.15 cm | 未稳定抬起（抓取失败） |
| 原生酒瓶任务（native_goal9） | 1:1 | 300 | 0.99 cm | 未稳定抬起（抓取失败） |
| 共享碗/盘桌面（shared_goal8） | 1:1 | 300 | 28.79 cm | 已释放并接触货架，但在目标区域之外 |
| 原生酒瓶任务（native_goal9） | 2:2 | 300 | 34.58 cm | 已释放并接触货架，但在目标区域之外 |
| 共享碗/盘桌面（shared_goal8） | 2:2 | 300 | 29.74 cm | 已释放并接触货架，但在目标区域之外 |

五个失败的试验都跑到 300 步预算用尽（budget_exhausted），**零个操作错误**；六个试验各自是**独立的全新会话**。每个条件只有 3 个样本，**不能**据此给出一般成功率；「严格成功」在这批试验里只观测到 **1/6**，且仅指这六个样本本身。

## 成功的那一次（native_goal9, 0:0）是怎么走完的

这一次的逐步时间线（逐动作事件取自 [trials/native_goal9_0-0/wine_telemetry.jsonl.gz](trials/native_goal9_0-0/wine_telemetry.jsonl.gz)）：

- 第 **92** 步：抓取接触信号首次出现（接触代理）。
- 第 **96** 步：瓶子相对**初始**高度被抬起超过 **2 cm**。
- 第 **169** 步：发出张开夹爪的指令。
- 第 **170** 步：接触筛查判定**不再持有**瓶子。
- 第 **173** 步：瓶子进入货架目标区域。
- 第 **174–178** 步：瓶子稳定、已释放、目标条件连续为真；作业在第 **178** 步结束。

要强调：本轮直接下发固定酒瓶指令，Hermes调用次数为0；逐动作记录由本地模拟器完成，不调用Hermes或GPT。另外，这里的「抓取」是**接触代理**：该信号**仅**来自 robosuite 的接触筛查；主诊断只是把这一信号与瓶高、指缝宽度、视频信号一并考虑，**不是**把四个信号融合成一个代理。接触代理是近似的，**不是**权威物理真值，也**不是**力/触觉传感器。

## 三个放置失败的共同点

三个「已释放并接触货架、但落在目标区域之外」的试验，结束时都有 `rack_contact=true`、`region_z=true`，且最后指缝间距约为 **79 mm**、判定为**未被持有**。问题出在一条**很窄的货架局部轴**上：该判据要求瓶子落点相对参考坐标的横向偏差**绝对值小于 22 mm**，也就是总宽度只有 **44 mm** 的一条带；它**不是**世界坐标的 Y 轴，也**不是**整只瓶子的包围盒。三个试验的实测横向偏差分别是：

- shared_goal8 1:1：`delta_y = -24.3000 mm`，落在带外约 **2.3 mm**；
- native_goal9 2:2：`delta_y = -24.3654 mm`，落在带外约 **2.4 mm**；
- shared_goal8 2:2：`delta_y = -32.4316 mm`，落在带外约 **10.4 mm**。

从画面看，这三个瓶子**可能**已经停在货架上；因此本文**不**声称它们仍然被夹着，也**不**声称这是基准（benchmark）的缺陷。判断标准**没有改动**，严格成功仍然是 **1/6**。

另外，在独立模拟器中复放**四条已有动作轨迹**（原始记录保持不变）：四条初始 SHA 一致，1078 个逐动作瓶子位置/夹爪关节位置差为 **0**、目标不匹配为 **0**（见 [predicate_replay/replay_summary.json](predicate_replay/replay_summary.json)）。这四条轨迹对应**1 条成功参考**与**3 条放置失败**，用于验证既有记录；策略样本数仍为 **6**。

## 证据与媒体

- 成功样本：[trials/native_goal9_0-0/rollout.mp4](trials/native_goal9_0-0/rollout.mp4)
- 抓取失败样本：[trials/shared_goal8_0-0/rollout.mp4](trials/shared_goal8_0-0/rollout.mp4)
- 已释放但落点区域不符的样本：[trials/shared_goal8_1-1/rollout.mp4](trials/shared_goal8_1-1/rollout.mp4)
- 观察时间线图：[timeline.png](timeline.png)

前两条 0:0 视频在**看到结果之前**就已选定；第三条 `shared_goal8 1:1` 是在观察到「已释放但落点区域不符」这类不同的放置阶段失败**之后**才补选的，这里明确说明。**全部六个试验**都保留在总表与原始证据里（下表），补选视频不影响任何统计口径。

`timeline.png` 仍是 pair 0:0（两种条件）的观察曲线，共 **5 行**：① 发给夹爪的指令（负值 OPEN、正值 CLOSE、零 HOLD）；② 实际指缝间距（mm）；③ 瓶子相对初始高度的抬升（cm）；④ 接触/持有代理（0/1）；⑤ 目标条件是否成立（0/1）。两种条件使用**相同的物理量纲**，缺失处留空（null），不做任何填补。

报告与审计文件：

| 内容 | 相对链接 |
|---|---|
| 六个试验的完整报告 | [campaign.json](campaign.json) |
| 逐试验汇总表 | [summary.csv](summary.csv) |
| 输入/动作审计（维数、裁剪、归一化） | [input_action_audit.json](input_action_audit.json) |
| 模型接口审计（哈希、维数、归一化均值/方差） | [model_interface_audit.json](model_interface_audit.json) |
| 场景定义审计（BDDL 块比较、逐 seed XML 哈希） | [scene_definition_audit.json](scene_definition_audit.json) |
| 发布清单（文件与 SHA 溯源） | [manifest.json](manifest.json) |
| 谓词复算汇总 | [predicate_replay/replay_summary.json](predicate_replay/replay_summary.json) |
| 谓词复算清单 | [predicate_replay/manifest.json](predicate_replay/manifest.json) |
| 全部发布文件的 SHA256（含报告、工具与补充审计） | [SHA256SUMS](SHA256SUMS) |

各清单的覆盖范围不同：`manifest.json` 覆盖**导出的 BASE 证据**；[predicate_replay/manifest.json](predicate_replay/manifest.json) 覆盖**复放载荷**；而 [SHA256SUMS](SHA256SUMS) 覆盖**全部已完成的酒瓶发布文件**，除最终报告与工具副本外，还包括**三份后续补充审计**（`input_action_audit.json`、`model_interface_audit.json`、`scene_definition_audit.json`）等。这三份补充审计**不**在初始导出清单 `manifest.json` 的覆盖范围内。`SHA256SUMS` 由原生桥在最终 README 定稿**之后**生成。

每个试验的原始遥测（gz，逐试验）：

- [trials/native_goal9_0-0/wine_telemetry.jsonl.gz](trials/native_goal9_0-0/wine_telemetry.jsonl.gz)
- [trials/shared_goal8_0-0/wine_telemetry.jsonl.gz](trials/shared_goal8_0-0/wine_telemetry.jsonl.gz)
- [trials/native_goal9_1-1/wine_telemetry.jsonl.gz](trials/native_goal9_1-1/wine_telemetry.jsonl.gz)
- [trials/shared_goal8_1-1/wine_telemetry.jsonl.gz](trials/shared_goal8_1-1/wine_telemetry.jsonl.gz)
- [trials/native_goal9_2-2/wine_telemetry.jsonl.gz](trials/native_goal9_2-2/wine_telemetry.jsonl.gz)
- [trials/shared_goal8_2-2/wine_telemetry.jsonl.gz](trials/shared_goal8_2-2/wine_telemetry.jsonl.gz)

（如需逐试验的完整字段，也可读取同目录的 `trials/<条件>/trial.json`。）

## 局限

- 本轮**只针对酒瓶**，**没有**任何 Hermes 调用（`hermes_calls=0`），也**没有**测到多目标的技能增益。
- `goal8` 与 `goal9` 的 regions / fixtures / objects / init **块在空白归一化之后**相同（**不是**整份 BDDL 逐字节相同），且**同一个 seed 的 XML 哈希相同**；但**六个初始状态 SHA 全部不同**，所以这**不能**证明存在一般的「统一场景 OOD」，也**不能**证明环境完全兼容。
- 动作与状态维数正确（**7 维动作 / 8 维状态**）；`post → sent` 裁剪逐步完全一致（差为 **0**）；`raw → post` 反归一化（转换到控制尺度）最大误差为 **1.12e-7**；七个模型文件哈希**未变**——证据见 [model_interface_audit.json](model_interface_audit.json)（只链接，不粘贴数组）。
- 固定的 checkpoint revision 为 `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`；实验源码为 `4b2941b161e79fda804251c4c49308df67a4a391`。
- 本地固定的[模型卡](https://huggingface.co/HuggingFaceVLA/smolvla_libero)把 datasets 记为 `unknown`；确切的训练样本与释放尾部都**尚未核实**（UNVERIFIED）。因此文中**不**做「训练缺少释放段」这类结论。

## 后续计划（尚未实现）

以下是**计划**，本任务**没有**执行：

1. 在 UI 上区分「已获取 / 已释放 / 已接触 / 区域是否命中」等状态。
2. 增加抓取阶段守卫、同场景**有界**重试，以及独立评分。
3. 对比失败状态与参考推理的观测 / 配置，并测试放置精度。

只有在核实出确实存在**残余技能差距**之后，才考虑微调。**不**对已经松手的瓶子强制张开夹爪；**不**悄悄放宽基准判据。本轮已完成上述六次仿真与诊断，未做微调或自动纠偏；后续三项尚未实现。

<details>
<summary>复现附录（隔离 CLI、默认设置、测试与工具）</summary>

### 隔离复现命令

下面这条命令在固定 WSL 虚拟环境与固定的 5 个环境变量下运行；**输出文件名与 `--run-root` 必须是新的、不存在的路径**（runner 拒绝覆盖）：

```text
wsl.exe -d Ubuntu -- /usr/bin/env MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/yhwang/fyp/libero_demo/venv/bin/python -u /mnt/d/FYP/First_Phase/scene_demo/wine_diagnostics.py --output /home/yhwang/fyp/scene_demo/wine_diagnostics/reproduce-01/campaign.json --run-root /home/yhwang/fyp/scene_demo/wine_diagnostics/reproduce-01/runs
```

- 默认跑 **3 个配对 0:0 / 1:1 / 2:2 × 2 个条件**。
- 固定档位：baseline **BF16**、`num_steps=10`、`n_action_steps=1`、指令 `put the wine bottle on the rack`、完成模式 `release_verified`、每子目标预算 **300**。
- **没有** `--profiles` 这个受支持的 flag（不要写成其它档位）。
- 该命令**不重启 UI、不改动策略**，进程本身可被中止。

### 测试与工具

- 纯协议/软件测试（无 GPU、无 checkpoint）：**33 + 79 + 21** 通过——它们验证的是**软件与协议正确性**，与物理技能不同。
- 仓库 runner 与测试：[../../wine_diagnostics.py](../../wine_diagnostics.py)、[../../tests/test_wine_diagnostics.py](../../tests/test_wine_diagnostics.py)。
- 公共工具：[tools/replay_wine_predicate_20261007.py](tools/replay_wine_predicate_20261007.py)（记录动作复放核验）与 [tools/export_wine_diagnosis_20261007.py](tools/export_wine_diagnosis_20261007.py)（离线证据导出）。
- 复算脚本导入的是**已部署仓库** `/mnt/d/FYP`，**不加载任何权重**；工具路径按**实际所在目录**相对复现。
- 六次实测用相同配置运行；这里的reproduce-01是新的示例输出路径，本次没有重复运行该路径。

</details>
