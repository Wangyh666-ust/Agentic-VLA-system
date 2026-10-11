# SmolVLA / LIBERO 设置核查（2026-10-11）

结论：本地文件确实与 `HuggingFaceVLA/smolvla_libero` 的固定公开版本一致；当前组合任务流程并不等同于论文评测。公开 checkpoint 与论文主模型的结构描述也存在可核实的差异，不能直接把论文的 87.3% 当作这份权重在本项目中的保证。

本轮只读取配置、SDK 源码、权重文件头和既有实验，核对公开来源；新增机器人动作、策略推理、Qwen 调用、训练和子 agent 均为零。未改动运行代码、模型、环境或演示服务。Windows localhost 的 8766/8767 health 请求未成功，因此以下运行设置依据源码及已保存的实际配置回读，不声称读取了当前常驻进程。

## 论文数字究竟是什么

原论文 v1 表 2 的 SmolVLA 0.45B：Spatial **90%**、Object **96%**、Goal **92%**、Long **71%**，平均 **87.3%**。评测覆盖四类共 40 个任务，每个任务 10 次，共 400 次。该数字既不是每个任务的成功率，也不覆盖本项目新增的任意组合与收尾归位要求。[论文 §4.1、§4.3、表 2](https://arxiv.org/html/2506.01844v1)

论文描述的仿真推理是预测 50 步动作，每执行 1 步就读取新观测并重新预测，flow matching 推理 10 步。50 是预测长度，1 是实际执行后更新观测的间隔，二者不能混淆。[论文 §4.3](https://arxiv.org/html/2506.01844v1#S4.SS3)

## 已核实一致的部分

| 核查项目 | 本地结果 | 一致范围 |
|---|---|---|
| 权重来源 | `HuggingFaceVLA/smolvla_libero`，revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | 与公开仓库的这份文件一致；不自动等同于论文 checkpoint |
| 权重完整性 | 1,218,047,032 bytes；实际 SHA-256 与远端 LFS SHA-256 相同 | 本轮重新读取本地文件计算，排除下载损坏或拿错文件 |
| 配置与处理器 | config、preprocessor、postprocessor JSON 与同 revision 的远端 JSON 内容相等 | 未用自己的配置替换文件内容；加载时仍有下述覆盖 |
| 归一化 | 两个统计文件的实际 SHA-256 均与远端相同；state/action 的 mean/std 为真实非平凡值 | 复用权重自带统计；未使用零均值/单位方差占位统计 |
| VLA 图像输入 | 原生 agentview + wrist，两路 256×256 RGB，经官方 LiberoProcessorStep 翻转高和宽，再按模型配置缩放到 512×512 | 与该 checkpoint 的特征契约和当前官方处理流程一致；额外展示相机不自动成为 VLA 输入 |
| 状态与动作 | state 8 维：末端位置 3 + axis-angle 3 + 两指编码器 2；action 7 维，relative OSC，20 Hz | 与当前官方 LIBERO 接口一致 |
| 动作预测 | `chunk_size=50`、`n_action_steps=1`、`num_steps=10` | 与论文上述仿真推理描述一致 |
| 初次场景加载 | 原生 init state、hard reset、10 个 no-op 等待步 | 与本地官方 SDK 默认流程一致；不是论文版本环境的逐位等价证明 |

本地源码入口：[原版 loader](../libero_demo/service.py)、[当前场景 service](service.py)、[场景与指令目录](catalog.py)。公开接口说明见 [LeRobot LIBERO 文档](https://huggingface.co/docs/lerobot/libero#policy-inputs-and-outputs)，公开配置见 [checkpoint config](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/main/config.json)。

## 不一致与尚未确认的部分

| 项目 | 当前实际条件 | 为什么影响比较 |
|---|---|---|
| 模型结构 | 文件头计数 604,934,176 个 tensor elements；语言模型与 action expert 均含编号 0–31 的层。`num_vlm_layers=0` 在实际 SDK 表示不裁剪，保留 32 层 | 论文主模型写的是约 0.45B、保留语言模型前 16 层。这是公开权重与论文描述的差异，不能据此判断哪个 checkpoint 更强 |
| 指令 | 原生篮子任务为 `put both the alphabet soup and the tomato sauce in the basket`，现在还会单独下发 `put the tomato sauce in the basket`；微波炉原指令为 `put the yellow and white mug in the microwave and close it`，现在拆成放杯和关门 | 句子与训练/标准评测任务发生变化。拆分合理，但模型是否可靠接受这些原子指令需要单独验证 |
| 第二个子任务起点 | `goal_table` 使用碗任务 `libero_goal:8` 的初始状态；执行前一任务后再归位，物体与环境仍保留前一任务成果 | 不等于把酒瓶原生任务 `libero_goal:9` 或灶台任务重新 reset 到各自标准初态；机械臂归位也不能恢复所有物体状态 |
| 成功判定 | 当前流程含子目标连续五次成立、释放/稳定要求；试点还包括 Qwen 检查、辅助器、收尾、退离、归位 | 官方 evaluator 以环境原生 `is_success` 结束并计分。项目 strict、语义分和整链完成必须分别报告 |
| 动作预算 | 旧酒瓶对照 300；最新容器试点每子任务 600；组合共享 2000 会话上限 | 本地当前官方 SDK 的 suite 上限为 Spatial/Object 280、Goal 300、Long 520；600 是本项目扩展，不能混作标准评测 |
| 混合精度 | 下载 config 为 `use_amp=false`；loader 覆盖 true，执行循环显式使用 BF16 autocast | 与公开配置默认不同。论文提及 BF16 训练，但没有由此证明其评测精度与本项目完全相同 |
| 软件版本 | LeRobot 0.6.2、hf-libero 0.1.4、MuJoCo 3.8.1、robosuite 1.4.0、Torch 2.11.0+cu128、Transformers 5.5.4 | 未取得论文运行环境锁文件，不能声称版本全一致 |
| 评测采样 | 当前是围绕失败技能挑选的小规模诊断、固定状态/种子或组合试点 | 不是 40 任务 × 10 次的评测样本，也未获得与论文一致的模型 RNG 与初态列表 |
| 训练来源与论文对应关系 | 此 revision 无 train_config；模型卡 datasets 为 unknown，未提供该权重对应表 2 哪一行的明确说明 | 尚不能核实训练步数、数据 revision、checkpoint 选择，以及原作者是否用同一权重覆盖全部 suite |

上游 [issue #2354](https://github.com/huggingface/lerobot/issues/2354) 有对同一公开模型的低于论文分数的复现报告，并指出 32/16 层差异；本报告的层数结论另由本地文件头和 SDK 源码核实，不依赖讨论者猜测。该讨论没有提供作者确认的统一失败原因。

[issue #4390](https://github.com/huggingface/lerobot/issues/4390) 提供不同 MuJoCo 版本改变 `libero_spatial:5` 碗在 ramekin 上沉降状态的对照，涉及 3.8.1。我们正使用这个版本，这是具体的复现风险，但该任务不是本项目常测的 `libero_goal:8`，不能将其结论直接推广到酒瓶、番茄罐或所有抓取失败。暂未更改版本。

## LIBERO 任务集是否更新：补充实证核查

当前安装的是 Hugging Face fork 的 `hf-libero 0.1.4`，不是原始仓库某个明确锁定的论文版本。这个包发布于 **2026-06-10**，晚于 SmolVLA v1 论文的 2025-06-02；因此软件版本确实不能称为与论文相同。[PyPI 发布记录](https://pypi.org/project/hf-libero/0.1.4/)、[fork 说明](https://github.com/huggingface/LIBERO)

但“软件更新”不等于“任务目标换了”。本轮取原始 `Lifelong-Robot-Learning/LIBERO` 在论文发布日之前的最新 commit：`8f1084e3132a39270c3a13ebe37270a43ece2a01`（2025-03-15），通过 GitHub tree 的文件 blob SHA-1 与本地实际文件逐一比较：

| 文件类别 | Spatial | Object | Goal | Long | 总计 |
|---|---|---|---|---|---|
| 原生任务定义 `.bddl` | 10/10 相同 | 10/10 相同 | 10/10 相同 | 10/10 相同 | **40/40 字节一致** |
| 实际使用的初始状态文件 `.pruned_init` | 10/10 相同 | 10/10 相同 | 10/10 相同 | 10/10 相同 | **40/40 字节一致** |

运行配置的 `bddl_files` / `init_states` 路径确实指向上述已比较的本地目录，没有只比较备用文件。原生任务定义包含物体、区域、初始关系及目标谓词；这些文件与该历史参照一致。本项目新增的拆分指令、布局变体和组合成功条件另算。

对该历史参照中本地仍存在的 56 个 `libero/libero` Python 文件，45 个相同，11 个有变化，包含 `bddl_base_domain.py`、benchmark 初始化及若干物体/资源加载模块。资源目录使用 `lerobot/libero-assets` 的固定 snapshot `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，本轮未全量比对几何和纹理资源。再加上 MuJoCo 3.8.1 的物理变化，不能从“任务文件一致”推出“运行行为完全一致”。

逐文件比较结果保存在 [LIBERO_VERSION_AUDIT.json](LIBERO_VERSION_AUDIT.json)。此历史 commit 是公开仓库的时间参照，**不是已经证明作者实际采用的 commit**；作者没有在所查论文/模型卡中给出可核实的完整环境锁文件。当前结论是：没有证据表明本地标准 40 个任务定义或预存初态被换掉；运行软件与论文时期不同，实际沉降后初态和执行行为仍需独立对照。

## 复用既有实验，避免重复

- `20261008-grasp-config`：原生酒瓶 scene、原指令、无 Hermes/辅助器的配对实验已经做过。默认 BF16 的项目 strict 为 2/6，关闭 AMP 为 2/6，关闭 AMP 且每次执行 5 步为 1/6；默认配置另三个留出初态 strict 3/3。这些口径含释放和连续采样要求，不当作官方 benchmark 成功率。原始 [discovery.json](results/2026-10-08-grasp-config/raw/discovery.json)、[holdout.json](results/2026-10-08-grasp-config/raw/holdout.json) 及 [报告](results/2026-10-08-grasp-config/README.md)。不能把关闭 AMP 当作已证明有效的修复，再重跑同样配置。
- `20261011-container-budget`：番茄罐入篮与杯子入炉各 600 个 VLA 动作，相关原生子目标均 0 次为真；收尾均未执行，微波炉关门子任务未启动。前者操作了 Milk，传入 preprocessor 前的 task 文本却确实是 tomato sauce。这两次使用拆分指令，不能视为原生 Long benchmark 复现。[原始汇总](results/2026-10-11-container-budget/summary.json)。
- 索引与 registry 只用于导航：[实验索引](EXPERIMENT_INDEX.md)、[registry](experiment_registry.json)。本轮 validate 输出 `ok=true`，40 entries / 3106 evidence；这是归档结构检查，不是模型效果结论。

## 下一步如何取得可归因的答案

先保留当前演示环境。不要继续从组合任务失败推导训练缺陷，也不立即微调或把模型裁成 16 层。

1. 为比较论文建立独立的标准评测条件：固定公开权重 hash、明确版本/精度、原生任务和原始完整指令、官方初态、suite 原生预算与 `is_success`；不调用 Hermes、辅助器、Qwen、收尾或归位。现有酒瓶原始记录优先复用，不能把做过的基线说成没做过。
2. 目前缺口是篮子/微波炉拆分指令相对原指令的影响，以及 simulator 版本对应性。下一轮如安排最小对照，只改一个因素并预登记；不能同时换模型、版本、提示词与控制器后再归因。本轮没有启动该实验。
3. 标准单任务通过后，再单独测试改写指令、任务后状态和 harness 增益。完整 400 次评测用于估计总体成功率；少数失败样例用于诊断，不替代它。

## 本轮核查锚点

- 源码 HEAD：`eb789e9`；LeRobot HEAD：`8c920c4270460851cedd2737657584586d3dc66f`，该 SDK checkout 无未提交改动。
- 模型 SHA-256：`71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`。
- 两个 normalizer safetensors 均为 416 bytes，SHA-256：`7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e`。
- 三个本地 JSON 与固定远端 revision 的解析内容相等；本地 config SHA-256：`e346a63094732ea28e05b08302a218306cf6541960b663a3cb4b6948e4efe6e0`。
- 在线核查日期 2026-10-11；未取得作者的原始逐任务结果与完整环境锁文件，未知项保留为未知。
