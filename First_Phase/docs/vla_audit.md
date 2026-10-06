# VLA 改造:现状审计与选型(阶段 A)

> 本文是 `First_Phase/changing_to_VLA_structure.md` 阶段 A 的产出:实测现状审计、兼容性矩阵与主方案选型。
> 所有"实测"行均可在 2026-10-04 会话中复核;安装后的固定版本见 `/home/yhwang/fyp/vla/requirements_frozen.txt`。

## 1. 实测环境事实

| 项 | 实测值 | 实测方式 |
|---|---|---|
| WSL2 | Ubuntu 24.04,用户 `yhwang`,代码根 `/home/yhwang/fyp`(ws/ perception/ robot_tools/ tools/ worlds/ config/ 均在) | `wsl.exe -d Ubuntu` |
| GPU | RTX 5070 **Laptop**,显存 **8151 MiB(≈8 GB)**,驱动 610.88 | Windows 侧 `nvidia-smi` |
| WSL 内 CUDA | 可用,CUDA UMD 13.3 | WSL 内 `nvidia-smi` |
| WSL 内 Vulkan | 仅 llvmpipe(CPU 软渲染,Vulkan 1.4.318);`/usr/share/vulkan/icd.d/` 无 Dozen(dzn)ICD | `vulkaninfo --summary`、`ls /usr/share/vulkan/icd.d/` |
| 磁盘/内存 | WSL `/` 余 932 GB;WSL 内存上限 17 GB | `df -h`、`free -h` |
| Python | 系统 3.12.3(externally-managed);无 conda | `python3 --version` |
| 网络 | 代理 `127.0.0.1:7897` 时通时断;失败时先 unset 代理直连 | `docs/environment.md` §2 |
| 现有栈(保留) | ROS 2 Jazzy + Gazebo Harmonic + MoveIt 2 + Hermes(qwen3-vl-plus)+ robot_tools MCP 9 工具 | `First_Phase/README.md` |

## 2. WSL2 能不能用 GPU?(结论)

- **CUDA 计算:可用。** SmolVLA 推理与微调走 CUDA,不受 Vulkan 限制。
- **GPU 图形渲染:基本不可用。** NVIDIA 对 WSL2 只提供 CUDA 支持,不提供原生 Vulkan 驱动;WSL2 图形走 `/dev/dxg` → D3D12 半虚拟化,Vulkan 只能经 Mesa Dozen 层。本机未装 Dozen ICD;历史上 D3D12 路径实测更慢(5.7 Hz,走 Intel 集显),llvmpipe 反而 12.7 Hz。
- 装新版 Mesa 可试 Dozen + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`,效果未知,列为阶段 B 的一次性有界实验(B-2a)。
- **大规模 GPU 并行渲染(ManiSkill 的核心卖点)在 WSL2 上任何路径都不可用**,留给未来原生 Linux 机器。
- 缓解因素:策略传感器仅 128×128(`SO100GraspCube-v1` 默认 `base_camera`),CPU 渲染负担小;物理仿真 PhysX 可试 GPU(CUDA),与渲染解耦。

## 3. 兼容性矩阵与选型

| 决策点 | 选择 | 依据与被淘汰项 |
|---|---|---|
| VLA 模型 | SmolVLA(`lerobot/smolvla_base`,450M) | 8 GB 显存下唯一可同时承载推理+微调的候选;OpenVLA-OFT(7B,bf16 推理约 14 GB+)与 π0(3B)被显存淘汰;SmolVLA 官方生态即 SO-100/SO-101 + LeRobot |
| 仿真器 | ManiSkill 3(proposal 指定) | `SO100GraspCube-v1` 官方数字孪生任务源码已核实:控制 `pd_joint_target_delta_pos`、sim 100 Hz / control 20 Hz、`base_camera` 128×128、`max_episode_steps=64`、成功判据 is_grasped & cube_lifted & 回 rest 位、内置域随机化。SmolVLA 路径依据 ManiSkill 作者公开的 SO-100 仿真训练工作流,**兼容性以阶段 B 实测为准** |
| 渲染方案 | llvmpipe CPU(现状);Dozen 实验封顶一次 | 见 §2 |
| 深度学习栈 | torch cu128+(Blackwell sm_120 必需)+ LeRobot 源码 `.[smolvla]` | 固定版本见 requirements_frozen.txt |
| 环境隔离 | 独立 venv `/home/yhwang/fyp/vla/venv`,不碰系统 Python / ROS / Hermes | proposal §8 |
| Hermes 集成 | VLA 执行服务以 MCP stdio 暴露(复用 robot_tools 的 mcp 2.3.0 模式) | 沿用 C-1/C-2 已验证链路 |
| Gazebo/ROS 栈 | 整体保留,作工程对照与未来真机迁移底座 | 指示文档 §4 |

## 4. 风险与回退门

| 风险 | 门槛/回退 |
|---|---|
| SAPIEN 在 llvmpipe 渲染不可用或过慢 | 门槛 B1:≥5 Hz@128×128 继续 ManiSkill;否则回退 Gazebo 原生 VLA 路线并记录差异与迁移成本 |
| 8 GB 显存微调 SmolVLA 不现实 | 门槛 C1:小 batch + 梯度检查点实测;不可行则云端训练,如实记录环境与成本 |
| `SO100GraspCube-v1` 只覆盖抓起 | 阶段 C-4 子类化放置任务;阶段 E 自建堆叠任务 |
| 官方示范生成/转换脚本版本漂移 | 先验证官方路径;失败则按 LeRobot 数据集格式自写转换器 |
| WSL 17 GB 内存并发紧张 | 阶段 B/C 记录内存峰值;必要时降 batch、关域随机化、串行化 |

## 5. 参考来源

- `First_Phase/changing_to_VLA_structure.md`(任务边界)
- `proposal/FYP_Research_Proposal_v1.md`(正式研究计划)
- ManiSkill `SO100GraspCube-v1` 源码(官方文档站)
- SmolVLA / LeRobot 官方文档(huggingface.co/docs/lerobot/smolvla)

> 本审计的结论以实测命令输出为准;后续阶段门槛(B1/C1)裁决记录将追加到本文档。

## 6. 门槛裁决记录

### 门槛 B1(2026-10-04,K3 裁决):**通过**

证据:`logs/phaseVLA_B2_smoke.log`(bench 脚本 `/home/yhwang/fyp/vla/bench_maniskill.py`)。

| 指标 | 实测值 | 门槛 |
|---|---|---|
| 128×128 传感器渲染 env-step FPS(llvmpipe) | **84.4** | ≥ 5 Hz ✅ |
| 512×512 人工渲染 FPS | 86.1 | 参考值 |
| state 模式物理步频 | 548.0 steps/s | 参考值 |
| 峰值内存 | ≤ 1.29 GB | WSL 上限 17 GB ✅ |
| **torch GPU + SAPIEN CPU 同进程共存** | **可行**:`gym.make(..., sim_backend="cpu", render_backend="cpu")` 使 SAPIEN 不再请求 cuda 渲染设备,torch 独占 GPU;coexist 段 200 步 FPS 83.6,CUDA context 在仿真后仍存活 | 阶段 B-3 的前提 ✅ |
| Dozen 有界实验 | Ubuntu 24.04 仓库 mesa-vulkan-drivers 未编译 dzn ICD;PPA+sudo 路线超出授权,**弃用,llvmpipe 为最终渲染路径** | 封顶一次,已执行 |

附注(子 Agent 控制实验,日志 D/E/F 段):84 FPS 由约 8–9 ms 固定开销主导,512 分辨率天花板 ~88 FPS,不要当像素吞吐量读;批量并行 env 的 CPU 渲染吞吐未验证(后续如需并行评测再压);`TORCH_MATMUL_1024_MS 95.7` 含 cuBLAS 初始化,不代表 GPU 吞吐,真实策略推理延迟待 B-3 实测。

### 阶段 B-3 结论(2026-10-04):真实策略链已通

证据:`logs/phaseVLA_B3_rollout.log`。`lerobot/smolvla_base` 真实权重,3 episode × 64 步,图像(base_camera 复制进 camera1/2/3)+ qpos(6)+ 指令进策略,动作链逐步断言通过(`ACTION_CHAIN_OK` ×3),无随机/脚本动作。前向延迟 ~250 ms/chunk(50 步),零样本成功率 0/3(base 模型预期内)。发现:该 checkpoint 的归一化统计键(so100.buffer.* 等)与策略特征键不匹配,归一化实为 no-op——阶段 C 用自建数据集训练时会生成正确统计量,自动修复。

### 门槛 C1 前置:数据与训练(2026-10-04,K3 裁决记录)

证据:`logs/phaseVLA_C1_demos.log`、`logs/phaseVLA_C2_train.log`。

- **示范数据**:100 条 `evaluate()` 验证成功的示范 / 8882 帧 / 256×256 base_camera,LeRobot 数据集校验通过(`DATASET_CHECK: OK`)。训练种子 0–594,评测种子 1000–1049 分离。
- **决策 1:mplib 本机弃用**。0.1.1 首次 C++ 调用即 SIGSEGV;0.2.1 依赖 numpy<2 与本栈不可共存(force-reinstall 尝试已还原环境)。官方运动规划专家不可用,改用脚本内 IK 专家(`--planner ik`)。后续若需 mplib(如堆叠任务)需源码编译,另行授权。
- **决策 2:抓取几何参数适配**(官方值按 4 cm 方块调,本任务方块 2.5 cm):`--grip-open -0.35 --descend-dz 0.005`,实测运动阶段成功率 0/12 → 6/12。
- **决策 3:专家产率规则**:质量门槛不变(仅 evaluate 判成功落盘),尝试上限 600,前 40 次 <30% 中止。实际 100/595(IK 失败占 63.4%,运动阶段成功率 45.9%)。
- **决策 4:`--rename_map` 解决相机键映射**;训练时仅 camera1 有真实图像,camera2/3 由模型内部零填充+mask=0(modeling_smolvla.py 确认)。**评测时也必须只喂 camera1**(与训练分布一致),B-3 的三键复制写法仅用于冷管线测试。
- **决策 5:保持 expert-only 微调**(checkpoint 默认 `train_expert_only=True` + `freeze_vision_encoder=True`,可训参数 100M/450M)。理由:官方小数据微调协议;100 条示范解冻 VLM 只会增加过拟合风险;课题变量在 Agent 层,底层训练配方保持标准。
- **训练实测**:batch 8,0.187 s/step,20000 步 ETA ~1.04 h,显存峰值 4570 MB / 8151 MB,loss 1.29 → 0.37(前 4000 步),`--policy.push_to_hub=false`(默认值会导致无 repo_id 报错)。**门槛 C1(8 GB 显存可微调):PASS**。

### 门槛 C-3(独立基线 ≥80%/50 种子,预登记):首轮 **FAIL**,进入归因修复

证据:`logs/phaseVLA_C3_eval.log`、`phaseVLA_C3_metrics.csv`(2026-10-04)。

- **结果**:2/50(4.0%,Wilson95 [1.1%, 13.5%])。成功 seed 1000/1004;43/50 `never_lifted`,全 50 例仅 3 例真正抓起过。
- **harness 可信度已验证**:专家示范在同一评测配置重放 6/6 成功、渲染逐像素一致——低分是策略结果,不是评测脚本 bug。
- **归因(证据强度递减)**:(a) `n_action_steps=50` 开环 chunk——逐帧预测误差 ~0.1 rad(优于零动作基线 2.5×),盲执行 50 步把手臂带离示范流形,之后全程离分布;(b) 训练量(100 示范 / 20k 步)或冻结视觉编码器;(c) 推理 bf16 vs 训练 fp32 不一致。
- **修复阶梯(K3 预登记,评测侧先行,不重训)**:① `n_action_steps` 50→10(闭环 ×5,加载后改 `policy.config.n_action_steps`);② 若仍 FAIL,fp32 加载复评;③ 若仍 FAIL,fp32+n_action_steps=10 组合复评;④ 仍 FAIL 才考虑训练侧(步数/数据量/解冻)。每级 50 种子同种子集(1000–1049)配对比较,全部结果如实记录。(注:③ 原为"n_action_steps=5",因 ① 证明步长轴无效应而调整为 dtype×步长组合,避免无信息重复。)
- **阶梯 ① 结果(2026-10-04,`phaseVLA_C3b_*)**:2/50,聚合零变化;成功集合在噪声水平换人(1000↔1030)。override 生效已验证(deque=10,每 10 步重规划,推理代价 ×3.6)。结论:`n_action_steps` 不是瓶颈;策略质量本身(视觉定位精度)才是约束。佐证:本任务方块 2.5 cm、官方专家经适配后运动阶段成功率也仅 45.9%——任务本身对抓取精度敏感。
- **阶梯 ②③ 结果(2026-10-04,`phaseVLA_C3c/C3d_*`)**:fp32+50 = 1/50,fp32+10 = 3/50;四轮(bf16_50, bf16_10, fp32_50, fp32_10)合计 250 次试验 9 次成功(3.6%),Fisher 精确检验最小成对 p=0.617(Bonferroni 0.005),**无任何一级与合并速率分离;250 次中仅 6 个不同种子成功过**。checkpoint 500 张量中 474 个以 bf16 存盘(冻结 VLM 主干),fp32 级实为"bf16 权重 + fp32 算术"。**评测侧变量全部证伪,瓶颈在训练侧(策略能力)**。
- **训练侧阶梯(K3,单变量序贯)**:T1 = 步数 20k→60k、同 100 示范(loss 曲线仍在降,最便宜假设);T2 = 若 T1 仍 FAIL,示范 100→500 后 60k 重训;T3 = 若 T2 仍 FAIL,解冻视觉编码器(`train_expert_only=false`,batch 视显存降至 4–8)。数据扩充(+400 条,种子 595 起、跳过评测种子 1000–1049)与 T1 并行执行(GPU/CPU 资源不冲突),产物为独立 dataset v2,不污染 T1 的对照性。
- **T2 数据已就绪(2026-10-04,`phaseVLA_C1b_demos.log`)**:v2 = 500 episodes / 44385 帧,`DATASET_CHECK: OK`;EVAL_SEED_GUARD 实测 500 条种子与 1000–1049 零交集;v1 未动(stat 双匹配)。两项记录:① 首轮扩充被我误设的中止门槛(40 次 ≥30%)误杀——p≈16.8% 产率下该门槛误杀率 97%,已按 1% 尾重校准为"300 次窗口 ≤36 条中止";② 授权 IK 重试 37→128,效果为中性(63.4%→62.2%):探针证实 IK 失败是**工作空间边界**问题(方块过远时垂直接近无解,姿态误差恒 <0.4°,纯位置差与基座距离强相关),非搜索预算不足;剩余杠杆(接近角/出生区)属冻结项,未动。
- **评测配置勘误(已记录并采纳)**:传感器须显式 256×256(注册默认 128);horizon 取 250(注册默认 64 会截断 93% 的成功示范,64 下结论为 0/50,两者皆 FAIL,250 才是真实能力)。

### 训练侧阶梯 T1(60k 步 / v1 100 示范):被 WSL 虚拟机重启中断,评测 **2/50 FAIL**

证据:`logs/phaseVLA_T1_eval.log`、`phaseVLA_T1_metrics.csv`、`vla/outputs/overnight_decision.txt`(2026-10-05 02:14)。

- **训练**:跑到 50000/60000 步被杀,最后可用 checkpoint `050000`。已排除休眠与 OOM——电源待机设为"从不"、电池 100% 交流,WSL 内无 dmesg 痕迹,判定为**整个 WSL 虚拟机被重启**(当时断网所致)。
- **结果**:`success=2/50 rate=0.0400 wilson95=[0.0110,0.1346]`;`ok=2 never_lifted=41 lifted_but_dropped=7 grasped_not_rest=0 other=0`;`mean_steps=242.78`,`EVAL_TRUNCATED_EPISODES: 48/50`(跑满 250 步 horizon)。
- **诚实指标**:`EVAL_LIFT_MAGNITUDE: episodes_with_cube_lifted=22 ; with_lift_mm_max>=5mm=4 ; with_grasp_ever=5 ; lift_mm_max_mean=3.974 lift_mm_max_max=112.992`。env 的 `cube_lifted` 阈值仅为离桌 1 mm,所以 22 这个数 mostly 是碰撞噪声;**真正抓住过方块的只有 5/50 集**。
- **harness 自身无恙**:`ACTION_CHAIN_FAILURES_TOTAL: 0`、`TRUTH_ISOLATION_OK: True`、`POLICY_DRIVEN` 声明 12139 个动作全部来自微调策略前向;`POSTPROCESSOR_STEPS: ['UnnormalizerProcessorStep','DeviceProcessorStep']` 且 `NORMALIZATION_EMPIRICAL_ACTION ... identical=False`,证明反归一化确实在工作(实测 `0.276×0.1966−0.0449=0.0094`,与日志逐位吻合)。
- **裁决**:T1 = FAIL。步数 20k→50k 对成功率无实质改善(T0 2/50 → T1 2/50)。

### 过夜 T2 首次启动失败:根因是继承自 Windows 的死代理(已修复)

证据:`vla/outputs/overnight_t2_train.log` 第 219–316 行。

- **现象**:守护链 round 1 在 200 秒内以 rc=1 结束,`latest_step=none`,一个 checkpoint 都没产出;脚本按 `FATAL_FAST_FRESH_FAILURE` 逻辑放弃并直接转入评测,又因无 checkpoint 可评而空转退出,并 `touch` 了 `overnight_t2_done` 假完成标记(该标记会让后续自愈重启认为链路已完成而立即退出)。
- **根因**:`lerobot-train` 在 `make_policy → SmolVLAPolicy.__init__ → SmolVLMWithExpertModel.__init__ → AutoProcessor.from_pretrained → transformers.processing_utils.get_processor_dict → list_repo_templates → hf_api.list_repo_tree` 处发起网络请求,`httpx.ConnectTimeout: [Errno 110]`。WSL 环境继承了 Windows 的 `http_proxy/https_proxy=http://127.0.0.1:7897`,而 WSL2 处于 NAT 模式(默认路由 `192.168.5.1`,resolv.conf `10.255.255.254`),该回环地址到不了宿主,代理进程也未监听。
- **关键鉴别**:`curl --noproxy '*' https://huggingface.co/api/models/lerobot/smolvla_base` → **HTTP 200,0.257 s**。所以外网本身是通的,坏的只是代理变量。注意 `AutoProcessor` 之前的权重加载是成功的(`Loading weights: 100%|489/489`),因为它命中本地缓存;只有"枚举 repo 模板"这一步强制联网。
- **修复**:在 `overnight_t2.sh` 的 `export LD_LIBRARY_PATH` 之后加入 `export HF_HUB_OFFLINE=1`、`export TRANSFORMERS_OFFLINE=1`、`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY`。两个资产(`lerobot/smolvla_base`、`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`)的本地缓存经核对是完整的(processor 的 `preprocessor_config.json`/`processor_config.json`/`chat_template.json`/tokenizer 全在)。
- **验证**:2 步离线冒烟训练 rc=0(数据集 500 episodes / 44385 帧正确加载,loss 2.409→1.801,step_s 0.183);重启后读 `/proc/<train_pid>/environ` 确认 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 已到位且无任何 `*_proxy`。
- **旁证**:`run_eval_round.sh` 本来就带 `HF_HUB_OFFLINE=1`,这正是 T1 评测在同样的死代理环境下仍能跑通的原因——缺陷只在训练侧。
- **T2 重启状态**:02:29:24 fresh 起,02:38 时真实进度 step 2689、5.95–6.06 step/s(tqdm 自报剩余 2:38),预计 ~05:16 训完,距 07:30 硬期限约 2 h 余量。注:`ot_train.py` 的 INFO step 行走 stdout 块缓冲,会滞后于走 stderr 的 tqdm 进度条,判断进度要看 tqdm 的 `N/60000`,不能只看最后一条 INFO 行。

### 夹爪维的目标分布确实病态(相隔 5σ 的双峰回归目标)——但它**不是**根因,该假设已证伪,见下节

证据:`logs/tmp_bench/gripper_diag.py`、`gripper_trace.py`、`gripper_law_check.py`(三者只读 v2 parquet,不占 GPU)。

- **动作第 5 维(夹爪)的分布是病态的**:`min=-1.0 max=0.0 mean=-0.0449 std=0.1966`。按 lerobot 的 `(a−mean)/std` 归一化后,该维取值范围为 **[−4.858, +0.228]**,中位数 +0.228。相比之下其余 5 维都是对称的:dim0 ±3.15、dim1 ±2.66、dim2 ±1.93、dim3 ±1.58、dim4 ±1.92。**只有夹爪维畸形**。
- **它是双峰的**:44385 帧中 44.67% 落在 [−0.01, 0)、49.7% 是极小正值,两者合计 ~94.4% 都是"≈0 保持";另有 3.67% 恰为 −1.0、0.71% 在 −0.8~−0.6、1.13% 在 −0.25~−0.15;而 −0.45~−0.15 区间**一帧都没有**。归一化后两个峰分别位于 +0.228 与 −4.858,**相隔约 5 个标准差**。
- **每集的时序结构完全一致**(500/500 无例外):夹爪指令只出现在两个爆发点,每集恰好 5 帧。
  - 第 0、1 帧:把夹爪从 env 静止值(`rest_qpos[5]=0`,加 `initial_qpos_noise_scale=0.02` 的噪声)收到专家定义的"张开"口径 **−0.35**(即 `--grip-open -0.35`)。实测 ep0 为 −1.0000、−0.6101。
  - 第 K、K+1、K+2 帧:真正抓取,从 −0.35 收到 **−0.80**(`planner.close_gripper(gripper_state=-0.8)`)。实测 ep0(K=43)为 −1.0000、−1.0000、−0.2500。
  - K 的分布:min 13、p25 34、**中位 41**、p75 50、max 70;相对位置中位 0.532。
  - 其余 ~94% 帧夹爪指令恒为 0(目标保持),`state_grip` 相应停在 −0.3500 长达十余帧(专家在抓取位姿上 dwell)。
- **专家的夹爪执行机构已被完全重构并验证**:`gen_demos_so100.py:182-186` 的 `gripper_action(g) = clip((g − target5)/0.2, −1, 1)`,配合 `pd_joint_target_delta_pos`(`lower=[-0.05]*5+[-0.2]`、`normalize_action=True`)即 `target5 += 0.2·a5`。以 `g=−0.35 → −0.80` 单次切换重放,**500 集全部逐帧比特级复现,最大残差 0.000000**(`GRIP_SERVO_SELFTEST ... exact=500/500`)。手算校验:ep0 第 1 帧 target=−0.228 → (−0.35+0.228)/0.2 = −0.61 ✓;第 45 帧 target=−0.75 → (−0.8+0.75)/0.2 = −0.25 ✓。
- **原假设(据上述分布提出)**:夹爪的"何时闭合"本质是一个由视觉触发的二值事件,却被编码成用 mean/std 归一化、两峰相隔 5σ 的连续回归目标;流匹配/MSE 类损失会偏向主峰(94.4% 的 ≈0),模型需要闭合时只能输出中间值(如归一化 −1 → 实际 a5 ≈ −0.24),夹爪半闭不足以抓住 2.5 cm 方块。**这个假设随后被证伪,见下一节。** 分布本身病态是事实,但它不是本例的失效机制。

### 上述夹爪假设的证伪,以及真正的根因:垂直下降精度(2026-10-05 03:20)

证据:`logs/tmp_bench/mine_policy_actions.py`、`mine_tcp_dist.py`,均只解析已有的 `phaseVLA_T1_eval.log`,不重跑、不占 GPU。

- **关键便利**:harness 对前 2 集逐步打印完整动作(`EP%d_STEP%03d qpos= pre_clip6= exec6= reward= is_grasped= cube_z= dist_rest=`),所以**策略自己的输出分布可以从已有日志里直接挖出来**,无需新跑评测。T1 的 ep0/ep1 共 500 个逐步样本。
- **证伪 1:策略确实输出了双峰夹爪指令**。实测 dim5 分布为 **96.40% 落在 [−0.010,+0.010]、1.20% 落在 [−1.010,−0.750)**,与示范的 94.4%/5.6% 双峰结构几乎一致。ep0 在 step 42 输出 `a5=−0.9930`、ep1 在 step 31 输出 `−1.0086`,都是接近满幅的闭合指令;而示范的抓取时刻 K 中位数正是 **41**(范围 13–70)。**幅度对、时机也对,"向主峰塌缩"这个机制在本例中不成立。**
- **证伪 2:夹爪确实物理闭合了**。qpos5 在 ep0 收到 −0.6699、ep1 收到 −0.8123(示范的闭合值是 −0.80)。ep1 是满闭,仍 `ever_grasped=False`。
- **真正的根因:手压到桌面高度才闭爪,手指夹的是桌面而不是方块。** 由 reward 反解(方法见下节)得到 TCP 到方块中心的距离与触桌标志:

  | | ep0 | ep1 |
  |---|---|---|
  | TCP 最近距离 | **5.4 mm**(step 60) | 30.8 mm(step 15) |
  | steps 25–49 触桌率 | **25/25 = 100%** | 15/25 |
  | 闭爪瞬间 TCP 距离 | 14.3 mm | **35.1 mm**(对空气闭合) |
  | 闭爪瞬间 `touching_table` | **True** | **True** |
  | 闭爪时 qpos5 | −0.2155(连示范接近口径 −0.35 都没到) | −0.3176 |
  | cube_z 全程 | **0.01250,一动未动** | 峰值 0.01391(1.4 mm,碰撞级) |

  两集的共性是**闭爪瞬间手指正压在桌面上**。ep0 的 TCP 甚至进到方块中心 5.4 mm 处却完全没推动方块,说明手是在桌面高度横扫过去、而非在方块抓取高度合围。
- **为什么这与既有阶梯的空效应一致**:`n_action_steps` 50→10、bf16→fp32 这四个轴都不改变**视觉深度分辨能力**,因此全部为空效应(250 次试验 3.6%,无一级显著)——不是阶梯设计不当,是这些轴与真正的失效机制正交。
- **机理判断**:示范的下降高度是用 `--descend-dz 0.005` 专门精调出来的(见 §6 C1 决策 2,该参数把运动阶段成功率从 0/12 拉到 6/12),即毫米级。策略要从**单目 256×256 图像**复现这个高度,而本轮 SmolVLA 的视觉编码器是**冻结**的(`freeze_vision_encoder=True`,可训参数 100M/450M);SmolVLM2 的 ViT 预训练于自然图像,不太可能天然分辨这个合成场景里的细深度线索。观测里也没有 TCP 位姿可用(`_get_obs_extra` 只在 `obs_mode_struct.state` 为真时才给 `tcp_pos`/`tcp_to_obj_pos`,而本轮 obs_mode 是 `rgb+segmentation`),策略只有单目图像 + 6 维关节角。
- **这给预登记阶梯的下一级 T3(解冻视觉编码器)提供了明确的机理理由**,而不再是"前两级失败所以试第三级"。
- **对 T2 的预期(在 T2 结果产出前记录,避免事后解释)**:T2 只把示范量 100→500,既没有解冻视觉编码器,也没有增加单目深度线索的信息量。预期成功率高于 T1 的 2/50,但**预计仍达不到 40/50 门槛**。

### 从 reward 反解 TCP 距离:方法、无歧义性与踩过的坑

- env 的 `reward_mode=None` → 取 `SUPPORTED_REWARD_MODES[0] = "normalized_dense"`(`sapien_env.py:128, 300-301`),所以日志里的 reward 是 `compute_dense_reward()/3`。
- `dense = (1 − tanh(5·d)) + is_grasped + exp(−2·dist_rest)·is_grasped − 2·touching_table`,其中 `d = ‖cube.pose.p − agent.tcp_pose.p‖` 是 TCP 到**方块中心**的距离(`grasp_cube.py:417-420, 456-473`);`touching_table` 是 finger1/finger2 对桌面的接触力 ≥1e-2。
- **无歧义性**:reaching 项 `1 − tanh(5d)` 被限制在 (0,1],所以在 `is_grasped=False` 时,`dense<0 ⟺ touching_table=True` 且 `reaching = dense+2`;`dense≥0 ⟹` 未触桌且 `reaching = dense`。给定日志已有的 `is_grasped` 列,反解唯一。
- **踩过的坑(记录以免重犯)**:第一版脚本误按**未归一化**的 dense reward 反解,负 reward 会得出 `reaching>1` 的不可能值,距离被系统性放大约 1.6×(例如把 14.3 mm 报成 164.2 mm)。是"负 reward 却解出 reaching>1"这个矛盾暴露了口径错误。修正后 T1 的 ep0/ep1 共 **500/500 步全部可反解**。
- 自洽性校验:修正口径下 ep0 step 0 的 reward=+0.1742 → d=103.9 mm(开局手离方块约 10 cm),step 42 的 −0.3571 → d=14.3 mm 且触桌;与物理图景一致。
- 脚本:`logs/tmp_bench/mine_tcp_dist.py`(`invert()` 为参考实现)。同一反解正在被加进 `eval_smolvla_grip.py --contact-audit`,以便把它跑到全部 50 集;该开关**不修改任何下发动作**,所以它的 `EVAL_SUMMARY` 必须与登记轮次完全相同——这同时又是一道交叉校验。

### T2g 诊断设计(预登记,不改动已登记协议)

目的:把"策略不知道何时闭合"与"策略表达不出闭合指令"这两种失败分离开。

- **手段**:复制 `eval_smolvla.py` → `eval_smolvla_grip.py`,**只替换动作第 5 维**为上面已验证的专家伺服律,由策略自己的二值意图驱动;第 0–4 维(手臂)保持 100% 策略输出不变。原件 `eval_smolvla.py` 一个字节都不改,以保证 T2 的登记评测不受影响。
- **锁存规则**:当 `target5 ≤ −0.34` 且策略输出的 `a5 < θ` 时锁存到 `g=−0.80`,此后不再张开(与专家一致——500 集里没有任何一集在闭合后重新张开)。前置条件 `target5 ≤ −0.34` 是**必需的**:示范第 0、1 帧的 a5 同样是 −1.0,若无此条件,每一集都会在开局立刻闭爪抓空气,诊断将毫无信息量。
- **θ 预登记为 −0.125**(示范"保持"模 ≈0 与最小闭合指令 −0.25 的中点)。先跑 10 集并记录策略 a5 的完整分布(`GRIP_INTENT_DISTRIBUTION`),再决定是否做 θ ∈ {−0.05, −0.25} 的敏感性分析;敏感性分析结果一律标注为探索性,不与登记轮次比较。
- **诚实性约束**:该脚本必须打印 `GRIP_SERVO_INTERVENTION: ACTIVE` 与 `GRIP_SERVO_NOTE`,并改写 `POLICY_DRIVEN` 声明以说明第 5 维被替换;其 80% 门槛不适用,任何 `EVAL_THRESHOLD` 行不得与登记轮次并列比较。证据文件独立命名 `phaseVLA_T2g*`,不覆盖 `phaseVLA_T2*`。
- **自证**:脚本内置 `--grip-selftest`,在构建 env 之前先用 v2 parquet 复现全部 500 集的夹爪动作,`PASS` 才继续,失败以 rc=2 退出——这样执行机构本身的正确性不依赖我的手工推算。

### T3 可行性静态核算(2026-10-05 03:46):预登记的 T3 在 8 GB 上很可能跑不动

证据:`logs/tmp_bench/t3_param_count.py`(只读 safetensors **头部**拿 shape,不载权重、不占 GPU),冻结逻辑出处 `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:152-192`。

- **参数清单(从 T2 的 `checkpoints/005000` 实读,540 个张量)**:

  | 分组 | 张量数 | 参数量 |
  |---|---|---|
  | `vision_model`(ViT) | 197 | 86.4 M |
  | `connector` | 1 | 11.8 M |
  | `text_model` | 146 | 204.6 M |
  | `vlm.lm_head` | 1 | 47.3 M |
  | `lm_expert` | 145 | 98.2 M |
  | 动作头(in/out/time_mlp) | 48 | 1.6 M |
  | `state_proj` | 2 | 0.03 M |
  | **合计** | | **450.0 M** |

- **脚本自校验通过**:算出当前配方可训 **99.9 M(22.2%)**,与 §6 C1 决策 5 独立记录的"100M/450M"吻合。
- **T3(`train_expert_only=false` + `freeze_vision_encoder=false`)可训 392.9 M(87.3%)**,新增 **293.0 M**,其中 ViT 占 86.4 M。T3 下 VLM 仍被冻结的子集只有 `lm_head` + `text_model.norm.weight` + `text_model.layers.15.*`,共 11 个张量 / 57.1 M。
- **一个会误导人的陷阱(已核实并排除)**:checkpoint 的 `config.json` 写着 `num_expert_layers = 0`,而解冻分支里有 `num_vlm_layers % num_expert_layers`,看起来必然 `ZeroDivisionError`。**实际不会**:`smolvlm_with_expert.py:112-120` 在构造时把它解析成 `len(lm_expert.layers) = 16`,与 `num_vlm_layers` 相等,于是第 165 行的 `!=` 先短路,取模根本不执行。按 config 原值推断会得出错误结论——必须按运行时解析值。
- **显存**:新增 293.0 M 可训参数带来 **grad bf16 0.59 GB + Adam m/v fp32 各 1.17 GB = +2.93 GB**,叠加 C1 实测的 batch-8 峰值 4570 MB → **约 7.5 GB / 8151 MB**。而这**还不包含激活值**:冻结时 ViT 与 15 层文本走 `no_grad`、不建图,中间激活即用即弃;解冻后它们必须全部留存以供反传。**结论:batch 8 极可能 OOM,实际可用 batch 大概在 1–2。**
- **没有缓解手段**:`grep -rn gradient_checkpointing lerobot/src/lerobot/policies/smolvla/` **零命中**(pi0 路径有,smolvla 没有),所以 batch size 是唯一杠杆,不能拿计算换显存。
- **时间**:当前 `step_s ≈ 0.224`(batch 8、expert-only,取自正在跑的 T2 日志)。全模型反传后单步时间会显著上升,预登记的 60k 步保守估计 **9–17 小时,一夜跑不完**。这一点必须在启动 T3 之前告诉用户,而不是跑到一半发现。
- **因此布署了实测探针**(`t3_probe.sh`,已挂后台,等 `outputs/t2contact_done` 且 GPU 空闲后自动跑):4 臂 × 3 步,测**整机峰值显存 / 是否 OOM / step_s**。
  - 臂 0 是**对照臂**:重跑当前 expert-only 配方 batch 8。它必须复现约 4570 MB;**若复现不了,探针本身就不可信,T3 各臂的数字一律作废**。这是把"估算"变"实测"时必要的自校验。
  - 臂 1–3:T3 配方 batch 8 / 4 / 2。
  - 产物:`outputs/t3_probe_summary.txt`、`logs/phaseVLA_T3probe_*.log`;每臂用独立的一次性 `outputs/t3_probe_*` 目录,跑完即删(带路径前缀白名单,拒绝删任何非预期路径),**不触碰任何已登记证据**。
- **备选 T3a(需用户决策,今夜不启动)**:只解冻 ViT(86.4 M,+0.86 GB)。理由——真正的机理假设是"冻结的 ViT 分辨不出这个合成场景的深度线索",那么**最小、最贴合假设的干预就是只解冻 ViT**;而 `train_expert_only=false` 会连带解冻 15 层文本,既贵又引入了与假设无关的自由度。**注意两个配置标志无法表达"只解冻 ViT"**:`train_expert_only=True` 会冻结整个 `vlm`(含 `vision_model`),所以要单独训 ViT 必须加约 10 行补丁(构造后重新设 `requires_grad`)。这是设计变更,留给用户批准,不擅自改 lerobot 源码。

#### 根因的定量形式:闭爪瞬间手比方块中心低约 12.5 mm(几何反推,2026-10-05 03:58)

上表的"闭爪瞬间 TCP 距离 14.3 mm + `touching_table=True`"两条,联合方块尺寸可以把误差**分解到垂直方向**,不需要额外实验:

- 方块边长 2.5 cm(§6 C1 决策 2),静止时中心高度 = 1.25 cm;T1 日志实测 `cube_z = 0.01250` **正好等于边长的一半**,与"方块平放在桌面上"自洽。
- `touching_table` 的判据是 finger1/finger2 对桌面的接触力 ≥1e-2(`grasp_cube.py`),即**指尖已经压在桌面上**。TCP 位于两指尖之间,故 `z_TCP ≈ 0 ~ 数 mm`,远低于方块中心的 12.5 mm。
- ep0 闭爪瞬间 3D 距离 d = 14.3 mm。若 `z_TCP ≈ 0`,则垂直分量就是 12.5 mm,水平分量 = √(14.3² − 12.5²) ≈ **6.9 mm**。
- 也就是说:**手已经到了方块正侧方 7 mm 处——水平定位基本是对的——但低了整整一个方块半高。** 在这个高度合围,手指夹到的只能是桌面。
- 交叉验证:ep0 的 TCP 最近距离 5.4 mm 出现在 step 60,而闭爪在 step 42。若 step 60 时手仍在桌面高度,则 3D 距离不可能小于垂直分量 12.5 mm;5.4 mm < 12.5 mm 说明**闭爪失败之后手才抬到了正确高度**。时机与高度两个误差是耦合的:它在错误的高度上提前闭合,随后才到达正确高度,而夹爪已锁死。
- 这条推理把根因从"下降精度不够"细化为**"垂直方向存在约一个方块半高(12.5 mm)的系统性偏低,水平方向误差反而只有约 7 mm"**。它同时解释了为什么 `n_action_steps`、dtype 这些轴全是空效应——它们都不改变单目图像里的深度可分辨性。
- **待接触审计量化的部分**:上面只用了 ep0/ep1 两集(harness 只对前 2 集逐步打印)。`--contact-audit` 会把 `touching_table` 比例、闭爪瞬间 TCP 距离、以及反解一致率统计到全部 50 集(`CONTACT_SUMMARY` / `CONTACT_AT_CLOSE_BURST` / `CONTACT_INVERSION_AGREEMENT`)。若"闭爪瞬间正在触桌"在 50 集里是普遍现象而非 ep0/ep1 的偶发,这个根因就成立;若只是少数,则结论要退回"多种失败模式并存"。**判据在此预先写下,以免事后挑数据。**

### 训练侧阶梯 T2(60k 步 / v2 500 示范):**5/50,FAIL**

证据:`logs/phaseVLA_T2_eval.log`、`phaseVLA_T2_metrics.csv`;独立复算 `logs/tmp_bench/verify_round.py`。

- **训练**:60000/60000 完整跑完,总耗时 3:15:06,末段 loss 0.171;checkpoint `060000`,`last → 060000`,`model.safetensors` mtime 05:44:43、906712552 字节。
- **结果**:`success=5/50 rate=0.1000 wilson95=[0.0435,0.2136]`;`ok=5 never_lifted=38 lifted_but_dropped=7 grasped_not_rest=0`;`mean_steps=233.36`;`EVAL_TRUNCATED_EPISODES: 45/50`;成功种子 `[1004, 1012, 1015, 1035, 1043]`。
- **独立复算通过**:从 CSV 重算得 k=5、rate=0.1000、wilson95=[0.0435,0.2136]、分类逐位相同,`VERIFY_VERDICT: CONSISTENT`。抬升指标在扣除方块静止高度 12.5 mm 后也逐项吻合(verify 的 max_z 均值 22.04 − 12.5 = harness 的 9.54;122.94 − 12.5 = 110.44)。
- **完整性**:`ACTION_CHAIN_FAILURES_TOTAL: 0`、`TRUTH_LEAK_TOTAL: 0`、`TRUTH_ISOLATION_OK: True`、`CSV_ROWS: 50`、50 个唯一种子恰好覆盖 1000–1049、grip-servo 未启用。
- **checkpoint 身份已严格核实**:`eval_smolvla.py:288,315` 是 `ck = os.path.abspath(args.ckpt)` → `SmolVLAPolicy.from_pretrained(ck)`,`CHECKPOINT_USED` 忠实反映它,指向 `smolvla_grasp_cube_v2_s60k`。
- **harness 缺陷(装饰性,不影响结论,但必须记录)**:`POLICY_DRIVEN` 那行引用的是 `eval_smolvla.py:62` 的模块级常量 `CKPT`(指向 T0 的 `smolvla_grasp_cube` 目录),而不是 `args.ckpt`。所以**每一轮的 `POLICY_DRIVEN` 字符串都指错了路径**(T1 日志里它指向第三个目录 `smolvla_grasp_cube_s60k`)。加载的权重始终由 `args.ckpt` 决定,结论不受影响,但这行"完整性声明"本身不可信,应改为插值实际路径。
- **裁决:T2 = FAIL**(k=5 < 预登记门槛 40)。
- **预登记的预期被证实**:结果产出前我写下"预期高于 T1 的 2/50,但预计仍达不到 40/50"。

#### 但 5/50 对 2/50 **不是**真提升:统计上与噪声不可区分

证据:`logs/tmp_bench/fisher_rounds.py`(自写双侧 Fisher 精确检验,超几何分布直接求和,无 scipy)。

- T1(2/48)vs T2(5/45):**p = 0.4360**。
- 六个登记轮次(T0、C3b、C3c、C3d、T1、T2)对 T2 的两两比较,最小 p = 0.2044,Bonferroni α = 0.0100,**无一可分离**。
- 六轮合并 **15/300 = 5.00%**。
- 要达到门槛 40/50,相对实测 5/50 的 p = 7.3e-13。
- **结论:训练侧阶梯到此为止(步数 20k→50k→60k、示范 100→500)没有产生任何统计上可检出的改善。** 5 倍数据量买到的是 +6 个百分点的点估计,而这个幅度在 n=50 下无法与噪声区分。

#### 根因修订:50 集接触审计**推翻**了我基于 2 集写下的表述

预登记判据(写在结果之前):"若'闭爪瞬间正在触桌'在 50 集里是普遍现象而非 ep0/ep1 的偶发,这个根因就成立;若只是少数,则结论要退回'多种失败模式并存'。判据在此预先写下,以免事后挑数据。"

证据:`logs/phaseVLA_T2contact_eval.log`(`--contact-audit`,无干预),解析脚本 `logs/tmp_bench/analyze_contact.py`。

- **反解方法 100% 验证**:11606/11606 步可反解,且反解出的 `touching_table` 与 env 自己的标志**逐步完全一致**(`CONTACT_INVERSION_AGREEMENT 11606/11606 = 1.000000`)。此前只在 T1 的 2 集上验过,现在是全部 50 集。
- **触桌不是判别量 → 撤回原表述**:50/50 集都触过桌;成功集 `touch_frac` 均值 **23.3%** 反而**高于**失败集 16.9%(Mann-Whitney p=0.0199)。**"手压到桌面高度才闭爪、手指夹的是桌面"这个说法不成立,予以撤回。** 它只是 ep0/ep1 两个特例的表象。
- **夹爪假设的最后一钉**:成功与失败的 `raw_a5_min` 都 ≈ −1.0,**p=0.72 不可分离**。策略在两种结局里都发出了满幅闭合指令——夹爪既不是幅度问题也不是时机问题。
- **真正的判别量是 TCP 到方块中心的三维定位精度**:

  | 指标 | 成功(n=6) | 失败(n=44) | Mann-Whitney p |
  |---|---|---|---|
  | 闭爪瞬间 TCP 距离 中位 | **10.7 mm**(2.4–15.8) | **50.0 mm**(8.5–123.4) | **0.0004** |
  | 全集最近 TCP 距离 中位 | 3.6 mm | **36.7 mm** | **0.0002** |
  | `raw_a5_min` | −1.0 | −1.0 | 0.72(不可分离) |
  | `touch_frac` 均值 | 23.3% | 16.9% | 0.0199(**方向与假设相反**) |

- **物理不可能性检验**:**34/44 失败集在 TCP 距方块中心 > 25 mm(整整一个边长)时就下令闭爪**,31/44 甚至 > 30 mm;而 **0/6 成功集超过 15.8 mm**。夹爪在离方块一个边长以外闭合,几何上不可能把 25 mm 的方块围住。
- **"送进去"近乎必要但不充分**:只有 11/50 集曾把 TCP 送进 15 mm 以内(成功 6 + 失败 5);其中 5 个进去了仍没抓住,说明除了接近还有第二重误差(高度/姿态)。
- **修订后的根因**:失效机制是**单目图像下方块三维位置的定位精度不足,误差量级是厘米级**(失败集最近距离中位 36.7 mm、p75 73.5 mm),而**不是**我先前写的"毫米级垂直下降精度"。
- **这一修订加强而非削弱 T3 的机理理由**:问题更明确地落在视觉表征上。信息在单目图里原则上是可得的(相机固定、方块在已知平面上、尺寸已知),所以是"冻结的 ViT 能不能提取出这个信息"的问题,而不是"观测里有没有这个信息"的问题——而本轮视觉编码器正是冻结的。

### T3 可行性:静态估算被实测推翻(今夜第二次)

证据:`logs/tmp_bench/t3_param_count.py`(静态,safetensors 头部)、`outputs/t3_probe_summary.txt`(粗测,2 s 轮询/3 步)、`outputs/t3_dense_summary.txt`(密测,0.2 s 轮询/10 步)、`logs/phaseVLA_T3probe_*.log`、`phaseVLA_T3dense_*.log`。

- **静态账**:可训参数 99.9 M(22.2%)→ 392.9 M(87.3%),新增 293.0 M(含 ViT 86.4 M),优化器+梯度 **+2.93 GB**;T3 下 VLM 仍冻结的只有 `lm_head` + `text_model.norm.weight` + `text_model.layers.15.*`(11 张量 / 57.1 M)。`lerobot/policies/smolvla/` **无 gradient_checkpointing**(pi0 有),batch size 是唯一杠杆。
- **我据此写下"batch 8 极可能 OOM、实际可用 batch 大概 1–2、60k 步 9–17 小时、一夜跑不完"。实测全部推翻**:

  | 臂 | 配方 | 峰值 MiB | 余量 MiB | step_s | 60k 步 |
  |---|---|---|---|---|---|
  | ctrl_b8 | 当前(expert-only) | 3754 | 4397 | 0.163 | 2.72 h |
  | **unfrozen_b8** | T3 全解冻 | **7700** | **451** | 0.338 | **5.63 h** |
  | unfrozen_b4 | T3 全解冻 | 6162 | 1989 | 0.248 | 4.13 h |

  batch 8 全解冻**跑满 10 步 rc=0、无 OOM**。按 T2 实跑与探针的 step_s 比值(0.195/0.163 = 1.20)校正长时运行的额外开销,T3 batch 8 约 **6.7 h**、batch 4 约 5.0 h——**都在一夜窗口内**,不是我原先写的 9–17 h。
- **标定不一致,如实记录**:对照臂本应复现文档记录的 4570 MB,实测 **3754 MiB**(97 个采样点),粗测 3885(3–4 点)——两者相差 3.5% 说明**探针自身自洽**,差异出在那个 4570 的测量口径从未被记录。影响:绝对余量的不确定度达数百 MiB,**大于 batch 8 的 451 MiB 余量**,所以"batch 8 安全"**不能**仅凭余量断言。但"batch 8 不 OOM"是**直接观测**,且固定形状训练循环的峰值出现在首次反传、不随步数累积,证据强度高于余量推算。
- **给用户的建议(今夜不启动)**:T3 = 全解冻、batch 8、60k 步、`save_freq` 降到 2500–5000、由守护自动 resume(万一 OOM 最多损失一个保存间隔);batch 4 为安全回退,但样本量减半(240k vs T2 的 480k),与 T2 的可比性变弱。**这是预登记阶梯的正常升级,不是推翻**——§6 C1 决策 5 的"保持 expert-only"依据是官方小数据协议,而现在有了 50 集的机理证据(厘米级定位误差 + 冻结 ViT)指向视觉表征,升级理由充分。

#### 评测 harness 自身不可复现:同一配置三次运行得 3 / 5 / 6(2026-10-05 06:08)

证据:`logs/phaseVLA_T2_eval.log`(登记轮次)、`phaseVLA_T2contact_eval.log`(审计)、`phaseVLA_T2rep_eval.log`(重跑);三次对比脚本内联于本节末尾的命令记录。

起因是接触审计的 `CROSSCHECK_EVAL_SUMMARY: MISMATCH`——审计跑出 6/50 而登记轮次 5/50。审计**不修改任何下发动作**(已核:`GRIP_SERVO_ACTIVE_IN_AUDIT: 0`、`ACTION_CHAIN_FAILURES_TOTAL: 0`、代码 diff 里对 `pre_clip`/`clipped64`/`sent` 只有读没有写),所以我没有接受"审计改变了行为"这个解释,而是用**原件脚本 + 原驱动器**重跑了一次(`run_eval_round.sh T2rep`,同 checkpoint、同 50 种子),得到 **3/50**。

| 运行 | 脚本 | k/50 | 成功种子 |
|---|---|---|---|
| T2 登记 | `eval_smolvla.py` | 5 | 1004, 1012, 1015, 1035, 1043 |
| T2contact 审计 | `eval_smolvla_grip.py --contact-audit` | 6 | 1000, 1004, 1012, 1015, 1035, 1043 |
| T2rep 重跑 | `eval_smolvla.py`(与登记轮次逐字节同一入口) | **3** | **1011, 1045, 1046** |

- **没有任何一个种子在三次运行里都成功**;T2rep 的成功集与前两次**完全不相交**。
- 三次合并 14/150 = **9.33%**;k 的均值 4.67、标准差 1.53、极差 3(即 **±6 个百分点**)。
- **发散是高度结构化的,不是全面混乱**:登记 vs 重跑只有 **8/50** 集步数不同,而这 8 集**恰好就是全部成功种子的并集**;其余 **42 集逐位相同**(步数、结局全等)。登记 vs 审计同理,只有 6 集发散,且全是接触丰富的成功集。
- **机理**:无接触的失败轨迹是**完全确定性**的;只有当手指真正接触到方块时,微小数值差异(PhysX CPU 接触求解顺序 / GPU 归约顺序)才会被接触动力学放大,决定这一次抓没抓住。
- **这本身就是关于策略的重要结论,而不只是测量学脚注**:**策略的每一次抓取都处在临界状态,没有一次是稳健的。** 一个真正学会了抓取的策略,其成功集应当跨运行稳定;而这里成功集近乎随机地换人,说明它处在"勉强能合围"的边缘,结局由数值噪声决定。这也解释了为什么 `EVAL_CLASSES` 里 `lifted_but_dropped` 一直有 7 例——抓起了但拿不稳。

**对整个阶梯历史的影响(必须回溯修正解读)**:

- 六个登记轮次的单次结果为 T0=2、C3b=2、C3c=1、C3d=3、T1=2、T2=5。现在已知**同一配置的单次 50 集结果自带 ±3 集(sd≈1.5)的重复性噪声**,这个 band 覆盖了 1 到 6 的全部观测值。
- 也就是说:**这些轮次之间的差异,单靠重复性噪声就能完全解释**。此前"评测侧四轴(n_action_steps、dtype)全为空效应"的结论因此**被加强**——不仅它们互相不可分离,它们与"什么都没改"也不可分离。
- 同时这也给 T2 vs T1 的判读加了第二重保险:T2 三次合并 9.33%(14/150)对 T1 单次 4%(2/50),Fisher 仍不显著,而 T1 的重复性未测,其真值同样可能落在同一 band 内。
- **对后续的方法学要求(预登记,适用于 T3 及以后)**:
  1. 门槛 ≥40/50 远在噪声底之上,**PASS 判定不受影响**——单次跑出 40+ 不可能是重复性噪声造成的。
  2. 但**任何"小幅改善"的断言都不成立**,除非重复运行。若 T3 落在 10–20/50 这个区间,**必须跑 3 次取合并**,否则无法与 T2 的 9.33% 区分。
  3. 报告成功率时应写成"单次 k/50 + 已知的 ±3 集重复性 band",而不是把单次 k 当作点估计直接比较。
- **诚实声明**:这一项是**意外发现**,不是预登记假设。它由审计链自带的交叉校验(`EVAL_SUMMARY` 必须相同)触发——那道校验本来是为了证明审计无副作用,结果证明了更强的东西:harness 本身不可复现。这是把校验写进链条的回报。

#### 新发现:LR 调度使 T2 的后 40000 步近乎惰性(2026-10-05 06:55)

工具:`logs/tmp_bench/lr_loss_trace.py`(只读训练日志)。调度器为 `cosine_decay_with_warmup`,`num_warmup_steps=1000`、`num_decay_steps=30000`、`peak_lr=1e-4`、`decay_lr=2.5e-6`(以上直接来自 T2 日志里的 config dump)。

**LR 在第 30000 步到达地板值 2.5e-6(峰值的 2.5%),而 T2 跑了 60000 步。** 分段重算的训练 loss 均值:

| 步数区间 | 平均 loss | 平均 LR |
|---|---|---|
| 1–5000 | 0.899 | 5.8e-05 |
| 5001–10000 | 0.310 | 8.2e-05 |
| 10001–15000 | 0.267 | 5.9e-05 |
| 15001–20000 | 0.212 | 3.4e-05 |
| 20001–25000 | **0.190** | 1.4e-05 |
| 25001–30000 | 0.195 | 3.7e-06 |
| 30001–60000(6 段) | 0.186 / 0.189 / 0.188 / 0.202 / 0.191 / 0.186,**无趋势** | 2.5e-06 恒定 |

三点锚定值(原始 INFO 行,未缩写):`step:150 loss:1.457 lr:1.3e-05`、`step:30K loss:0.199 lr:2.5e-06`、`step:60K loss:0.190 lr:2.5e-06`。

**结论:loss 在约 20000 步就停在 ≈0.19,其后 40000 步(2/3 算力、约 1.9 小时)无可测量下降。**

含义:
1. **加强 T1 vs T2 的空效应结论**。T2 不只是"5 倍数据未买到可分离的提升",它在优化上也早已饱和。数据量与步数两个杠杆均已拉满 → 瓶颈不在二者,与 §6 修订后的根因(单目三维定位厘米级误差)及 C1 决策记录里的专家产率 45.9% 互相印证。
2. **效率杠杆(待验证)**:后续轮次可能只需 ~25–30k 步,墙钟减半而质量无损。建议 T3 出结果后单独做一次"30k vs 60k 等价性检验"(约 3.4 小时)来坐实,不要仅凭本条外推。

**测量限制(如实记录)**:lerobot 的 INFO 行把大数字缩写(`30050` → `30K`),1200 条日志中只有 **79 个可区分步数值**,step>10000 的每段仅约 5 个采样点。平台期横跨 8 个分段,结论对该稀疏度稳健;曲线细节不可过度解读。首次运行本脚本时因正则只匹配 `\d+` 而把 `30K` 之类全部丢弃,报出"19 unique steps, range 50..950"的错误结论,已修正为同时接受 K/M 缩写与科学计数法(`lr:1.3e-05`)。

#### T3 启动记录与预登记(2026-10-05 06:48 启动 / 06:58 登记,**早于任何结果**)

**配方**(与 T2 的唯一差异是可训参数集合):`--policy.train_expert_only=false --policy.freeze_vision_encoder=false`,其余全部保持 T2 值 —— 数据集 v2(500 集 / 44385 帧)、batch 8、60000 步、seed 42、同一 LR 调度、`num_workers 4`。输出目录 `outputs/smolvla_grasp_cube_v3_t3_unfrozen`(全新,T2 目录未被触碰)。

**已在真跑日志中证实解冻生效**:`num_learnable_params=392904096`、`num_total_params=450046176`,与 `t3_param_count.py` 的静态核算(392.9M / 450.0M)逐位吻合。lerobot 侧默认值确认为 `freeze_vision_encoder: bool = True`、`train_expert_only: bool = True`(`configuration_smolvla.py:69-70`),故两个 `=false` 覆盖是必要且正确的。

**启动决策的依据**:会话处于自动权限模式(不得向用户提问),而用户指令为"完成工作"。四个候选中,B 需改 lerobot 源码、C 会改变任务定义、D 是战略转向,三者都需用户授权;A 是预登记阶梯的下一级、仅用两个配置标志、写入全新目录、可随时 kill,**是唯一既不需授权又不锁死其它选项的动作**,故执行 A 并原样保留 B/C/D。

**磁盘约束导致的必要设计变更**:WSL 的 vhdx 位于 C:,**仅剩 54 GB**。实测 T2 单 checkpoint = 1.32 GB(907 MB 模型 + 413 MB 优化器 ≈ 4.1 字节/可训参数),故 T3 单 checkpoint ≈ **2.53 GB**(907 MB + 1.62 GB);lerobot 不清理旧 checkpoint(T2 的 12 个全在,共 15 GB)。`save_freq 2500` × 24 = **61 GB > 54 GB**,会写满 C:。因此 `t3_chain.sh` 在守护循环内剪枝,**只保留最新 2 个**(峰值 ~5.1 GB),可用空间 <8 GB 时收紧到 1 个。剪枝是本链唯一的删除代码路径,配 5 用例自测 `--selftest-prune`(只删 6 位数字目录;4 位目录与普通文件不动;`last` 指向旧 checkpoint 时该 checkpoint 必须存活;keep≥总数为空操作;目录缺失不报错;真实输出目录绝不被触碰),**PASS**,且由子 Agent 与主 Agent 各独立跑过一次。输出目录留在 ext4 内而非 `/mnt/d`,因为 `checkpoints/last` 是符号链接,drvfs 上创建符号链接不可靠。

**预登记的预期(在结果产生之前写下,供事后对照)**:

1. **T3 不会通过 ≥40/50 的门槛。** 上限被示范质量压住:C1 记录适配后专家在运动阶段自身成功率仅 **45.9%**(100/595 产率,IK 失败占 63.4%),模仿学习难以系统性超越其示范。
2. **T3 三次合并(150 集)预期落在 8%–30% 区间(约 k=12–45),最可能 15%–25%。** 依据:根因是视觉三维定位,而解冻 ViT + 文本层是对该根因最直接的杠杆,应有可观改善;但受第 1 条的上限约束。
3. **可检测性**:150 vs 150 的双侧 Fisher 在 α=0.05 下,足以把 9.33%(14/150)与 20%(30/150)分开(p≈0.008)。因此若 T3 真的翻倍,**应当**能测出显著;若测不出,说明改善不足一倍。
4. **证伪判据(关键)**:若 T3 三次合并 **≤ 14/150**,则"冻结的视觉主干是瓶颈"这一假设**被证伪**。届时剩余解释只有:任务几何/示范质量(选项 C)、或单目深度歧义这类**微调无法解决**的表征问题。那将是对选项 D(把研究变量放回 Agent 层)的强支持,并意味着不应再向底层抓取率投入。
5. **不预先宣称的方向**:我不会把任何单次 k/50 当点估计比较。裁决只认三次合并 + Fisher/McNemar,且必须写明 ±3 集的重复性 band。

**裁决落点**:`outputs/t3_verdict.txt`,内含逐次 `verify_round.py` 独立重算(必须 `VERIFY_VERDICT: CONSISTENT`)、150 vs 150 合并 Fisher、按种子的精确 McNemar 配对检验(附功效 caveat)、以及 `STEPS_REACHED` 与 `DEADLINE_HIT`(防止把被硬期限截断的不完整跑当成完整跑)。

**已知会印错但故意不修的一处**:`eval_smolvla.py:728` 的 `POLICY_DRIVEN` 插值模块级常量 `CKPT`(`:62`,指向 T0 目录)而非 `args.ckpt`,故 T3 的该行路径也会是错的。实际加载走 `:288 → :315 from_pretrained(args.ckpt)`,正确;权威记录是每个评测日志开头的 `T3_CHECKPOINT_ARG`(由 `run_eval_round.sh` 从真实参数写入)。不在无人值守时修改承载全部历史轮次可比性的 792 行脚本;T3 出结果后再改,并在本节记录 md5 迁移。

**硬期限 16:00**:到点则终止训练并用最新可用 checkpoint 评测,`t3_verdict.txt` 会打印 `WARNING: INCOMPLETE RUN`。

**启动后 15 分钟的实测健康度(2026-10-05 07:04)**:step 2511 / 60000,loss 0.564,GPU 7734 MiB / 89% util,零 OOM、零 Traceback、零 ConnectTimeout(离线环境变量生效)。**首个 checkpoint `002500` 保存成功,实测 2.4 GB**(静态预估 2.53 GB,误差 5%),且保存期间 `nvidia-smi` 读数不变(仍 7734 MiB)→ **排除了"451 MiB 余量在存盘瞬间被峰值击穿"这一最主要的失败模式**,这是启动前唯一无法用 10 步探针覆盖的风险(探针只跑到 step 10 就存了一次,规模与真实保存相同但当时无法确证长时行为)。

ETA 修订:训练 ~12:30–13:25(tqdm 自估剩余 5.4–6.3 h,取决于监控开销;启动初期 3.02 step/s,step 2511 时 2.52 step/s),三次评测 + 合并分析后**裁决约 13:40 前落盘**,距 16:00 硬期限仍有余量。

**已知显示缺陷(记录在案,故意不修)**:`T3_STATUS.txt` 的 `LATEST_STEP` 在轮中不刷新,会持续显示 `none`,因为 `write_status` 只在每轮训练开始与结束时调用。真实进度须读 `t3_train.log`。不热修的理由:bash 边读边执行,修改正在运行的脚本会破坏它(昨夜已踩过)。

**一个必须诚实记录的设计局限**:脚本里的低磁盘安全阀(`disk_avail_gb` < 8 GB 时把保留数收紧到 1)**看不见真正的约束**。它执行的是 WSL 内部的 `df --output=avail /`,返回的是 vhdx 的**虚拟**可用空间(877 GB),而不是宿主 C: 的**真实**剩余(43 GB)。所以那道阀门永远不会触发。**真正保护 C: 的是"只保留最新 2 个"这条剪枝规则本身**,而不是那个阈值判断。修正方案(下次改脚本时做):从 Windows 侧取 C: 真实剩余,例如 `powershell.exe -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,0)"`,或直接检测 vhdx 文件大小。

**C: 空间的实测复核(2026-10-05 07:10)**:06:19 时 C: 剩 54 GB,07:06 降到 43 GB。核查后确认**不是泄漏**:同期 vhdx 只增长约 1.15 GB(83771 MB → 84923 MB),差额来自 Windows 页面文件的一次性扩容(`Win32_PageFileUsage`:AllocatedBaseSize 14658 MB、CurrentUsage 仅 543 MB,即 WSL 虚拟机拉到 17 GB 内存后页面文件随之扩容)。07:08 与 07:10 两次测量均为 42.96 GB,**已稳定**。按剪枝后峰值 ~5 GB 计,余量充足。WSL 内部 `free -g`:total 17 G、used 5 G、buff/cache 9 G、swap 5 G 未用。

**剪枝在真实目录树上的实测确认(2026-10-05 07:42,step ~9025)**:启动前唯一无法验证、而整个 C: 磁盘安全都依赖它的一环,现已闭环。

| 检查项 | 期望 | 实测 |
|---|---|---|
| `checkpoints/` 内 6 位数字目录数 | ≤ 2 | **2 个:`005000`、`007500`**(各 2.4 G) |
| 最旧 checkpoint 已被删 | `002500` 不存在 | **不存在** ✓ |
| `PRUNE_DELETED` 日志条数 | ≥1 | **1 条**:`2026-10-05T07:33:34+08:00 \| PRUNE_DELETED step=002500` |
| 异常标记 | 0 | **0**(无 `PRUNE_SKIP_UNEXPECTED_PATH` / `PRUNE_FAILED` / `PRUNE_SKIP_LAST_TARGET`) |
| 续跑点未被删 | `last` 目标存在且可用 | `last -> 007500`,`model.safetensors` 在位,`LAST_TARGET_USABLE=yes` |
| 宿主 C: 真实剩余 | > 40 GB | **43.62 GB**(07:08 为 42.96,不降反微升 → 稳定) |
| 训练仍在推进 | 进程存活、无错误 | step 9025/60000、2.97 step/s、ETA 4:46 → ~12:32;`t3_chain.sh`(pid 594)与 `lerobot-train`(pid 646)均存活;OOM/Traceback/CUDA error 计数 **0**;GPU 7660 MiB / 94% |

结论:**剪枝按设计工作,磁盘占用稳定在 ~4.8 GB(2 个 checkpoint),不会写满 C:。** 稳态行为已确立:此后每新增一个 checkpoint 就删除一个最旧的,`PRUNE_DELETED` 应累计到 22 条(step 60000 时保留 `057500` 与 `060000`)。

一处读取日志的坑,记录备忘:`PRUNE_DELETED` 是守护进程写的,而 tqdm 用裸 `\r` 刷新进度条,两者混在同一个"行"里。**直接 `grep PRUNE_ t3_train.log` 会拖出几万字节的进度条噪声**,必须先 `tr "\r" "\n"` 再 grep。
