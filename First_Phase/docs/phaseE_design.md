# 阶段 E 设计:组合任务、多指令底层策略与受扰恢复

> 前提:单指令抓取技能在声明变体上已可靠(dev 50/50,见 `review_2026-10-05.md` §3b)。
> 目的:构造**高层决策确实会影响结果**的条件,从而能检验 proposal 的 RQ1(分解)与 RQ2(恢复)。在单技能、无扰动的抓取上,高层没有可发挥的余地,E1–E4 必然没有差别。

## 1. 环境 `SO100StackCube-v1`(自建子类)

- 文件:`/home/yhwang/fyp/vla/envs/so100_stack.py`;继承 `SO100GraspCubeEnv`,import 时注册。
- 物体:红方块(沿用原环境)+ 蓝方块(2.5 cm,动态刚体,颜色 (0,0,1),摩擦与红方块相同)。
- 出生区域 R:先用修正专家测绘再定。候选:x ∈ [0.18, 0.23],y ∈ [−0.06, 0.10](径向距离 ≤ 0.25 m);两方块中心距 ≥ 0.07 m,yaw 各自随机。若某候选区内专家成功率 < 95%,就缩小区域,不放宽专家门槛。
- 评估谓词(只给独立评估器,不进入策略或 Agent):
  - `grasp_success`:沿用原 `evaluate()`(红块被夹住、抬起、手臂回到 rest)。
  - `stack_success`:红块在蓝块上方,水平偏差 < 12 mm 且 0.020 < Δz < 0.030 m;夹爪未夹持红块;红块线速度 < 0.01 m/s;以上条件连续保持 10 步。
- 策略相机与 v3 相同(base_camera 256×256);Agent 侧视相机沿用服务 v1.1 的低机位。

## 2. 技能与指令

| 指令 | 起始状态 | 结束状态 | 判据 |
|---|---|---|---|
| `Grasp the red cube.` | 手臂在 rest,夹爪张开 | 夹住红块回到 rest | grasp_success |
| `Place the red cube on the blue cube.` | 夹住红块在 rest | 红块叠在蓝块上,夹爪松开,手臂回到 rest | stack_success |
| `Put the red cube on the blue cube.` | 手臂在 rest | 同上 | stack_success |

- 同一个多指令 checkpoint 承载三条指令。比较各条件时,底层 checkpoint 固定不变。
- 第三条是完整任务指令,供 E1(直接 VLA)使用,保证"直接执行完整指令"有对应的训练覆盖,不把它做成一个没有训练过的弱基线。

## 3. 专家与示范

- 专家 = v3 抓取专家 + 放置段:移到蓝块正上方 5 cm → 下降到红块底面贴住蓝块顶面(TCP 目标 z = 蓝块中心 + 0.025)→ 张开到 −0.35 → 上抬 5 cm → 回到 rest。TCP 偏移沿用 v3 的参考开度修正。
- 示范录成完整的 episode,在"抓取段回到 rest"那一步切分,按帧打 task 标签:
  - 切分点之前的帧标 `Grasp the red cube.`,之后的帧标 `Place the red cube on the blue cube.`;
  - 另外把同一批 episode 整段复制一份,全部标为 `Put the red cube on the blue cube.`。LeRobot 的 task 是逐帧的,这样做可行。
- **恢复覆盖**:约 30% 的 episode 从"事后"状态开始,例如夹爪已闭合停在 rest、红块被挪到新位置,使"失败后再抓一次"处于训练分布之内。闭环本身并不会凭空带来恢复能力(见 `changing_to_VLA_structure.md` §3)。
- 数据格式沿用 v3:无损图像、绝对关节目标、state = qpos。

## 4. 扰动(只给 runner,服务 v2 增加 `/perturb`)

| 类型 | 触发 | 效果 |
|---|---|---|
| `displace_red` | 指定 episode 步(抓取执行中、闭爪之前) | 把红块瞬移到 R 内的新位置,平放 |
| `drop_red` | 指定 episode 步(抓起之后) | 若红块被夹持,把它瞬移到夹爪正下方的桌面上 |
| `displace_blue` | 两个技能之间 | 把蓝块瞬移到 R 内的新位置 |

- 扰动的时刻、类型与幅度在最终测试之前固定成计划表,每个 seed 的扰动计划对所有条件完全相同。每次扰动都写入 trace。
- 恢复只能在 episode 内完成。复位只发生在 episode 之间。

## 5. 条件(同一 checkpoint、同一 seed 集、同一扰动计划、同一 episode 预算)

| ID | 条件 | 实现 |
|---|---|---|
| E1 | 直接 VLA | runner 调用一次 `execute_skill("Put the red cube on the blue cube.")`(单次上限 300 步,见 §5c A1) |
| E2 | 固定分解 | 计划固定为 [Grasp, Place](由 runner 给出,见 §5c A1);执行期间不观察、不重规划 |
| E3 | 固定分解 + 按执行状态重试 | E2 的计划;某技能 `ended_reason ≠ settled`(超时、预算耗尽)时原样重试,最多 2 次;不看场景观测 |
| E4r | 规则式观测驱动(无 LLM) | 每个技能结束后调用 `inspect_scene`:Grasp 后 `red_held` 为假则重抓;Place 后 `red_on_blue` 为假时,若仍 `red_held` 就重放,否则先 Grasp 再 Place;直到成功或预算耗尽 |
| E4 | LLM 观测驱动 | Hermes 每个技能结束后调用 `inspect_scene`(可选 `vision_analyze`),自行决定重试、改选技能或终止 |

**修订说明(2026-10-05)**
- 原先设想的 E3 是"整段盲重放"。它在成功之后仍会去抓已经叠好的方块,会人为毁掉成功,使对照被不公平地压低。现改为只依据执行状态重试,用来控制"额外执行机会"本身的作用。
- 新增 E4r,是为了区分两种贡献:"有了感知反馈"带来的提升,和"LLM 推理"带来的额外提升。E4 对 E4r 的差值才是 LLM 的增益。
- D2 实测显示,Hermes 直接读图(vision_analyze)判断抓取结果,与真值的一致率只有 2/5,全部是假阴性;每个 episode 约 15.5 万输入 token。因此 E4 的主要感知通道改为结构化的 `inspect_scene`(侧视 RGB-D:颜色阈值 + 深度反投影,加上关节正运动学),它的准确率先对照真值单独测量(proposal §5.1:结构化感知摘要)。
- 有了文本形式的感知结果,高层模型不再必须是视觉模型,可以评估 proposal 建议的 DeepSeek。

- 预算:每个 episode 共 300 个动作步,所有执行共享;LLM 调用次数与墙钟时间另行记录。(已由 §5c A1 改为 450 步,并加每条指令的单次上限。)
- 指标:任务成功率(Wilson 区间)、扰动条件下的成功率、恢复率(分母 = 发生了可恢复扰动的 episode)、所用步数、墙钟时间、LLM 调用数与 token,以及 E4 的 `AGENT_VERDICT` 与真值的一致率。
- 样本:每个条件 × 每种扰动先做 20 个 episode 的 pilot,用以估计效应量;正式规模在锁定协议之后再定(proposal 目标是 100)。

## 5b. 进度与 pilot 预登记(2026-10-05 15:50,写于任何多指令策略结果之前)

**已完成**
| 项 | 结果 | 证据 |
|---|---|---|
| 堆叠环境 + 专家 | 区域 R 内 200/200;恢复起步 9/9 | `review/v4_map.*`、`review/v4b_smoke.*` |
| 示范 v4 | 400/401 成功(其中恢复起步 120/121),种子 0–400,每条都有切分点 | `review/v4_demos.*`、`demos/so100_stack_v4.*` |
| 多指令数据集 | 800 episode / 112,672 帧(Grasp 31,040 + Place 25,296 = Put 56,336),3 条指令,无损图像,绝对动作 | `review/convert_v4.log`、`review/validate_v4.log` |
| 服务 v2 | v1 回归 9/9;v2 自测 9/9;MCP 7 个工具;inspect 平均误差 xy 1.8 mm、z 0.6 mm(侧视 + base 双视角融合,解决了相互遮挡) | `service/selftest_*` |
| runner | E1/E2/E3/E4r/E4 在 replay 下跑通;G 条件、单次上限、真值快照、绕过代理在 replay 下验证 | `experiments/results/smoke_*`、`smoke3_*`、`smoke4_*`、`smoke5_*` |
| 多指令训练 | batch 16,30k 步,15:32 启动,17:52 完成;内存最低剩余 12.9 GB;checkpoint `outputs/smolvla_v4_b16_s30k/checkpoints/030000` | `outputs/train_v4_b16_s30k.log` |

**门槛评测(训练完成后,dev 种子 3000–3049,无扰动,checkpoint 030000)**
- G(只执行 Grasp)的 grasp_success ≥ 80%;E1(Put 直达)的 stack_success ≥ 70%;E2(Grasp→Place)的 stack_success ≥ 70%。
- 任一门槛不达标:先定位数据或适配问题(例如 settled 判据、指令标签、动作链),不得进入 pilot。

**Pilot(门槛全部通过后)**
- 种子 3100–3119(20 个,与门槛评测用的 dev 种子不重叠,也不碰测试集 1000–1099);
- 扰动 ∈ {none, displace_red_early(第 20 步), drop_red_mid(第 85 步)};
- 条件 E1、E2、E3、E4r 各跑 20 × 3;E4(Hermes)先每种扰动 10 个 episode(token 成本约 17 万输入/episode);
- 预算 300 步;同一 seed 在所有条件下扰动计划完全相同。(已由 §5c A1 改为 450 步 + 单次上限;门槛评测同样适用。)
- **预期(写在结果之前)**:无扰动时 E1–E4r 接近,差距 ≤ 10 个百分点;drop_red_mid 下 E4r 与 E4 明显高于 E1/E2/E3(E3 只按执行状态重试,发现不了"空夹回到 rest");displace_red_early 下各条件差别取决于策略自身的闭环能力,方向不预设。
- pilot 只用来估计效应量和发现问题,不作为最终结论;正式实验的规模与种子在 pilot 之后另行锁定。

## 5c. 预登记修正 A1 与感知验证(2026-10-05 21:20,写于任何 SmolVLA 堆叠结果之前)

**修正 A1:单次执行上限与 episode 预算**
- 问题:核对 runner 时发现,每次执行都以"剩余全部预算"为上限。一次没有 settled 的执行会一直跑到预算耗尽,E3 的"按执行状态重试"永远轮不到;E4r/E4 同样会被一次卡住的执行吃光预算,失去再次观察和决策的机会。E3 因此实际等同于 E2,对照失效。
- 修正:
  1. 服务 v2.1 为每条指令设单次执行上限:Grasp 150 步、Place 150 步、Put 300 步。依据是 400 条专家示范的长度:Grasp 段 p99 = 125、最大 144;Place 段 p99 = 100、最大 116;全程 p99 = 194、最大 223。Put 的上限等于 Grasp 与 Place 上限之和,使 E1 与 E2 的最大可用步数相同(300)。上限由服务端执行,runner 和 Hermes(E4)一视同仁:runner 不传 max_steps,E4 提示词 `experiments/prompt_e4_v2.txt` 也要求 Hermes 不传。
  2. episode 预算由 300 改为 450,即 300 再加一次完整技能执行(150)。E1/E2 最多只会用到 300 步;E3/E4r/E4 多出的 150 步只能通过额外执行来使用,而这正是这些条件要检验的"额外执行机会"。
  3. E2 的计划由 runner 固定为 [Grasp, Place],不再由 Hermes 生成。计划本身是确定的,用 LLM 生成只会增加噪声,不增加信息。
  4. §6 第 2 条中的 Place-from-holding 门槛,由 §5b 的 E2 门槛代替。runner 现在会在每次执行结束后记录一次真值快照(`executions[].truth_after`),报告中给出 E2 的"第一次执行后确实抓住"比例,以及在此条件下的堆叠成功率(`stack_given_held`),作为 Place 的条件成功率。
- 其余(种子、扰动、样本量、预期)不变。
- 验证:replay 后端下,未知种子(1000)使 E3 产生 3 次 150 步的 Grasp 重试(共 450 步),E2 为 150 + 150,E1 为 300;已知种子 G/E2 的执行均在上限内 settled(`experiments/results/smoke3_*`)。

**工程修正(不影响协议)**
- WSL 的登录 shell 会注入 http_proxy,runner、MCP 客户端和 mcp_check 访问 127.0.0.1 时原本会经过 Clash 代理(曾导致一次 30 秒超时)。三者现在都显式绕过代理,并用指向不存在端口的代理做过验证(`smoke4_*`)。
- mcp_check.py 现在把 VLA_SERVICE_URL 传给 MCP 子进程(MCP SDK 默认只传 6 个环境变量)。
- Hermes 配置中 vla_tools 固定连接 8765,因此 pilot 的服务必须开在 8765 端口。

**感知准确率(inspect_scene 对照真值,2026-10-05 21:15)**
- 样本:replay 后端 50 个 episode(正常 30、恢复起步 10、drop_red 扰动 10),共 934 个采样点。脚本 `experiments/validate_perception.py`,结果 `experiments/results/perception_validation.jsonl` 与 `perception_validation_summary.txt`。
- 真值定义:lifted 表示红块底面高出桌面 2 cm 以上;held 取环境的 is_grasped;on_blue 表示两块水平距离 < 2 cm,且高度差在 1.5–3.5 cm 之间。

| 估计 | 准确率 | TP / FP / TN / FN |
|---|---|---|
| red_lifted | 99.9% | 598 / 0 / 335 / 1 |
| red_on_blue | 99.5% | 261 / 5 / 668 / 0 |
| red_held | 90.8% | 399 / 66 / 449 / 20 |

- red_held 的 66 个假阳性全部出现在"红块已叠在蓝块上、夹爪已松开、手臂正在回撤"的阶段(第 96–144 步)。E4r 先判断 red_on_blue,所以这类误判不会改变它的决策。排除已叠好的样本后,red_held 的准确率为 97.0%(673 个样本,FP 0、FN 20)。20 个假阴性都出现在刚夹住、尚未抬起的瞬间(红块中心高度 1.4–3.5 cm),而决策点(技能结束、手臂回到 rest)不会处于这种状态。
- 定位误差(两相机融合):红块 xy 中位数 1.7 mm、p95 4.4 mm,z 中位数 1.4 mm、p95 3.3 mm;蓝块 xy 中位数 1.7 mm、p95 2.9 mm。每个采样点都至少有一个相机看到两块方块(没有弃权)。
- 污染说明:另一项测试误连 8765,有一次外来的 /execute 落进了恢复组 seed 20 的 episode,导致该 episode 第 83–139 步的 8 个采样点来自偏离示范的状态。估计与真值仍在同一时刻比较,因此这些数据保留。
- pilot 中 runner 会在 E4r 的每个决策点同时记录估计值和真值(`decisions[].truth`),届时再报告决策点处的一致率。

**修正 A2(2026-10-05 21:50,pilot 之前)**
- drop_red 的落点改为:红块被夹住时,把它放到任务区域内随机采样、距蓝块中心 ≥ 7 cm(min_sep)的桌面位置,朝向随机,均由扰动种子决定。原实现把红块直接放到 TCP 正下方;E4 冒烟(seed 3000)中,第 85 步时手臂已在蓝块上方,红块被放在距蓝块中心 12 mm 处,与蓝块穿模,物理引擎会把两块方块不可控地弹开。新落点保持在训练分布内,恢复所需的步骤仍然是"发现空夹 → 重新抓取 → 放置"。验证:replay 下 5/5 落点在区域内,距蓝块 ≥ 9 cm(`experiments/results/smoke7_*`);自测 9/9。
- 扰动仍然只在红块被夹住时施加,否则记为 not_grasped、不施加。恢复率的分母是 applied = True 的 episode。
- runner 解析 E4 结论时兼容 `AGENTS_VERDICT` 等写法(冒烟中 Hermes 写成了 AGENTS_VERDICT,导致原解析为空)。

**门槛评测结果(2026-10-05 21:23–21:39;dev 种子 3000–3049,无扰动,A1 设置,checkpoint 030000)**

| 条件 | 指标 | 结果 | Wilson 95% | 门槛 |
|---|---|---|---|---|
| G | grasp_success | 43/50 = 86% | [73.8, 93.0] | ≥ 80%,通过 |
| E1 | stack_success | 42/50 = 84% | [71.5, 91.7] | ≥ 70%,通过 |
| E2 | stack_success | 35/50 = 70% | [56.2, 80.9] | ≥ 70%,恰好通过 |

- E2 第一次执行后确实抓住的有 43/50,在抓住的条件下 Place 成功 34/43 = 79%。
- 失败模式:Grasp 的 7 例失败集中在固定种子(3001、3015、3019、3033、3042、3043、3045),同一种子在 E1/E2 下也大多失败。Place 失败多为放偏:落在蓝块旁 3–5 cm,或叠上了但水平偏 12–13 mm,未被判为成功。种子 3042 下 Grasp 指令直接完成了叠放,是指令混淆的个例。
- 结果文件:`experiments/results/gate_G.jsonl`、`gate_E1.jsonl`、`gate_E2.jsonl`、`gate_summary.txt`。

**E4 冒烟(seed 3000,drop_red_mid,修正 A2 之前的旧落点)**
- Hermes 共 15 次模型调用,输入约 22.3 万 token、输出约 1 千 token,墙钟 49 s。执行序列:Grasp(70 步,settled)→ Place(空夹,跑满 150 步上限)→ Put(230 步,预算耗尽)。结论为 failure,与真值一致。
- 按此估算,E4 pilot 30 个 episode 约需 670 万输入 token。

## 5d. Pilot v1 结果与修正 A3(2026-10-05 23:10,写于 pilot v2 之前)

**Pilot v1(21:47–22:56;种子 3100–3119,E4 为 3100–3109;A1 + A2 设置)**

| 条件 | none | displace_red_early | drop_red_mid |
|---|---|---|---|
| E1 | 18/20(90%) | 9/20(45%) | 5/20(25%) |
| E2 | 16/20(80%) | 8/20(40%) | 3/20(15%) |
| E3 | 16/20(80%) | 8/20(40%) | 5/20(25%) |
| E4r | 16/20(80%) | 10/20(50%) | 4/20(20%) |
| E4 | 8/10(80%) | 3/10(30%) | 4/10(40%) |

- drop_red 实际施加的次数:E1 16/20,E2/E3/E4r 各 18/20,E4 8/10(未施加的,是第 85 步时红块不在夹爪里)。displace 全部施加。
- E4r 决策点处感知与真值一致:held 225/226,on_blue 226/226。
- E4:结论与真值一致 28/29(另有 1 例 Hermes 把标记写成了 AGENCY_VERDICT,未被解析);每个 episode 12–20 次模型调用,平均输入 18.1 万(none)/ 22.4 万(displace)/ 22.0 万(drop)token,墙钟 34–45 s。
- 对照预登记预期:无扰动时各条件差距 ≤ 10 个百分点,符合。drop_red_mid 下 E4r/E4 明显高于 E1/E2/E3,**不符合**(E4r 20%、E4 40%,E1–E3 为 15–25%,样本小)。
- 诊断:感知不是瓶颈,瓶颈在底层技能的起始状态。
  1. 扰动之后,失败的那次执行通常会跑满上限,手臂停在半空(空夹的 Place 每次都跑满 150 步)。
  2. 紧接着的 Grasp 从非 rest 姿态、并且夹爪闭合的状态起步,超出了训练分布(训练中的恢复起步只覆盖 rest ±0.15 rad)。E3/E4r 中:首次 Grasp 成功 92/120(77%);紧跟在未 settled 执行之后的 Grasp 只成功 18/61(30%);紧跟在 settled 执行之后的 Grasp 只成功 5/29(17%),其中多数是"上一次 Grasp settled 但夹空,夹爪仍然闭合"的情形。E4 多次在同样的半空姿态下改用 Put,也都失败了。
  3. 另一类失败:放偏之后红块离蓝块不到 6 cm(训练中两块至少相距 7 cm),此后的 Grasp 全部失败。这是策略本身的分布外问题,A3 不处理。

**修正 A3:两个脚本原语与统一的复位规则**
- 服务 v2.4 增加两个脚本动作。它们不经过 VLA,计入 episode 预算,单次上限 60 步:
  - `Return to rest.`:把 5 个关节移回 rest,夹爪目标不变(抓着的方块仍然抓着);
  - `Return to rest and open the gripper.`:同时把夹爪恢复到本 episode 复位时的值(张开)。
  两者都按控制器的每步限幅移动,实测从离 rest 0.6–0.9 rad 处返回需要 16–17 步。
- 复位规则:E2/E3/E4r 由 runner 执行,E4 写进提示词 `prompt_e4_v3.txt`(Hermes 也可以自行调用)。
  - 非首次执行的 Grasp 之前,先执行 `Return to rest and open the gripper.`;
  - 其他技能若前一次执行没有 settled,先执行 `Return to rest.`;
  - E4r 在必要时先复位,再 inspect,再决策。
- E1 只执行一次 Put,不受影响,沿用 pilot v1 的 E1 数据。
- **Pilot v2 预期(写在结果之前)**:
  - none:E2/E3/E4r/E4 两两差距 ≤ 10 个百分点;
  - drop_red_mid:E4r 与 E4 都比 E2 高至少 20 个百分点;E3 比 E2 高不超过 10 个百分点(空夹的 Place 跑满后,E3 只会重试 Place);
  - displace_red_early:E4r 比 E2 高至少 10 个百分点;
  - 红块被放偏到蓝块 7 cm 以内导致的失败,预计仍会存在。
- Pilot v2:种子与扰动同 v1;E2/E3/E4r 各 20 × 3,E4 各 10 × 3(种子 3100–3109);服务 run_dir 为 `service/runs/pilot2_20261005`;结果为 `experiments/results/pilot2_*`。

## 5e. Pilot v2 结果(2026-10-06 00:35;A3 设置)

| 条件 | none | displace_red_early | drop_red_mid |
|---|---|---|---|
| E1(沿用 v1) | 18/20(90%) | 9/20(45%) | 5/20(25%) |
| E2 | 16/20(80%) | 10/20(50%) | 4/20(20%) |
| E3 | 15/20(75%) | 9/20(45%) | 8/20(40%) |
| E4r | 16/20(80%) | 13/20(65%) | 16/20(80%) |
| E4 | 7/10(70%) | 7/10(70%) | 6/10(60%) |

- drop_red_mid 的 Wilson 95% 区间:E2 [8.1, 41.6],E3 [21.9, 61.3],E4r [58.4, 91.9],E4 [31.3, 83.2]。
- drop_red 实际施加的次数:E1 16/20,E2/E3/E4r 各 18/20,E4 7/10。
- E4r 决策点处感知与真值一致:held 217/217,on_blue 217/217。
- 中断说明:00:15 前后 WSL 虚拟机重启,E4 displace 在 seed 3104 之后中断。之后用新的服务 run_dir(`service/runs/pilot2b_20261005`)补跑 displace 3105–3109 与 drop 3100–3109,结果追加在同一结果文件中。E1–E4r 不受影响。

**对照预登记预期(§5d)**
- none:两两差距 ≤ 10 个百分点。实测 80 / 75 / 80 / 70,最大差 10,符合(在边界上)。
- drop_red_mid:E4r 比 E2 高 60 个百分点,E4 比 E2 高 40 个百分点,都 ≥ 20,符合。E3 比 E2 高 20 个百分点,超出预期的 ≤ 10,**不符合**。原因:空夹的 Place 跑满后,`Return to rest.` 让 E3 有机会再次执行 Place,而 Place 策略有时会自己把掉落的方块重新抓起再放上去。E3 drop 的 8 例成功中,有 5 例属于这种情形(seed 3100、3101、3103、3107、3117)。
- displace_red_early:E4r 比 E2 高 15 个百分点,符合。

**E4(Hermes)与 E4r 的比较**(轨迹见 `experiments/results/pilot2_E4_traces.jsonl`,v1 为 `pilot_E4_traces.jsonl`)
- 三种扰动下,E4 都不高于 E4r(70 / 70 / 60 对 80 / 65 / 80)。
- 81 个可比较的技能决策中,68 个(84%)与 E4r 的规则一致。不一致的 13 个里:6 次在 inspect 显示未抓住时仍执行 Place,3 次在已抓住时执行 Grasp,4 次改用 Put。
- 30 个 episode 中有 28 个把 `Return to rest and open the gripper.` 当作第一步(提示词只要求在"重新 Grasp"之前使用)。这让 E4 的执行比其他条件晚 6 步,扰动落在 Grasp 的不同阶段,drop 的施加率因此只有 7/10。也就是说,E4 与其他条件的扰动时序并不完全一致,解读 E4 时需要注意。
- 视觉工具几乎不用:30 个 episode 中只有 1 次 vision_analyze,决策几乎全靠 inspect_scene 的文字估计。
- 成本:每个 episode 13–22 次模型调用,平均输入 23.6 万(none)/ 26.3 万(displace)/ 26.0 万(drop)token,墙钟 40–50 s。结论与真值一致 25/26,另有 4 例没有写出可解析的结论标记。

**结论(pilot 级别,样本小)**
1. 底层技能支持复位之后,观测驱动的恢复(E4r)在掉落扰动下把成功率从 20% 提高到 80%;无扰动时没有代价。
2. 在这个任务上,LLM 规划(E4)没有超过拿到同样信息的规则策略(E4r),每个 episode 却要消耗约 25 万输入 token。LLM 的价值需要在规则难以覆盖的情形下检验,例如多物体、指令变化、事先没有预料到的失败类型。
3. 剩余的主要失败来自底层策略本身:放偏后红块离蓝块过近,以及 Place 的精度(E2 中,抓住条件下 Place 成功约 80%)。

**正式实验前需要决定的事项**
- 规模:proposal 的目标是每个条件 100 个 episode;
- E4 的规划模型与成本(例如改用 DeepSeek,或精简 Hermes 的系统提示);
- 是否修改 E4 提示词,避免第一步多余的复位;
- 是否加入 LLM 可能占优的任务变体。

## 6. 推进门槛

1. 专家在 R 内:grasp 段与 stack 段都 ≥ 95%(调参种子 4000+)。
2. 多指令策略(dev 种子 3000–3049,每条指令各 50 集):Grasp ≥ 80%,Place-from-holding ≥ 80%,Put ≥ 70%。未达门槛先定位数据与适配问题,不用 Agent 去掩盖底层缺陷。
3. 服务 v2 的自测(含 `/perturb`)全部通过,才开始 E1–E4。
