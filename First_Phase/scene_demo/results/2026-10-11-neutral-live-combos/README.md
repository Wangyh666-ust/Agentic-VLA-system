# 新检查器：两个组合实测

**两个组合均未完整跑通。两次碗任务都完成了收尾、退离和归位；酒瓶实际已放架并松手，但收尾控制器启动时被手指边界检查阻断；灶台实际已开启，但 Qwen 返回“无法判断”，没有触发收尾。**

本轮使用 `1364b62` 的 `visual_goal_encoder_v3`，预登记、源码冻结及实际启动版本为 `06deba6`（完整 Git 版本见 `launch_*.json`）。各组合新执行一次，未改词重试、未微调、未开子agent、未单独测试归位。顺序由用户指定，验证执行链路，不是新增 Hermes 自主规划评估。

## 哪一步做到了，哪一步没做到

| 组合 | 子任务 | 新 VLA | 抓取辅助 | 收尾开爪 | 退离 | 归位 | 独立连续通过 | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 碗→酒瓶 | 碗放盘 | 150 | 0 | 2 | 103 | 131 | 5 | 完整通过 |
| 碗→酒瓶 | 酒瓶放架 | 300 | 182 | 0 | 0 | 0 | 0 | 目标达到、VLA已松手；收尾启动被阻断 |
| 碗→灶台 | 碗放盘 | 150 | 0 | 2 | 103 | 131 | 5 | 完整通过 |
| 碗→灶台 | 灶台开启 | 150 | 0 | 0 | 0 | 0 | 0 | 已开火；视觉未知，未调用收尾器 |

四个子任务均实际启动；三个得到检查器收尾许可，两个实际完成收尾及原归位。完整组合 0/2。这是两例小试的结果，不是总体成功率。酒瓶和灶台的收尾/退离/home 动作为零，不能说“已验证收尾器无效”或“原归位再次失败”。它们分别卡在控制器启动与视觉触发，后续效果仍未验证。

## 酒瓶：目标已经达到，启动控制器时出错

第 150 个 VLA 动作，Qwen 的实际回答：

```json
{"goal_achieved":false,"reason":"酒瓶仍被夹爪夹持，未接触架子表面"}
```

对应总动作第 332 步（150 VLA＋182 抓辅）的原生放架位置谓词为 true，几何记录也有瓶子与架子的接触；其支撑判据仍为 false。这说明 Qwen 的“未接触架子”理由与保存记录不一致，而且仍将夹持写进了位置目标判断。用户的目标触发条件没有额外要求支撑；不能据本轮证明新提示词已彻底消除这类视觉否定。原始侧视图与完整提示词均随请求归档，没有现场改词复查。

按原定流程继续到第 300 个 VLA 动作，Qwen 回答：

```json
{"goal_achieved":true,"reason":"wine bottle is on the rack in sideview"}
```

这次同步编码器也确认标准开口，融合检查通过。此时 VLA 自己已经松手，原生放架和语义放置均为 true，几何记录无持物，语义连续达标 122 次。但随后 `SensorPort.start → safe_exit._finger_hold` 在发第一条收尾命令前抛出 `finger qpos outside ctrlrange`，所以收尾开爪、退离、归位都为零。

实测左指位置 `0.04001874449 m`，原 Panda 位置执行器的指令上限 `0.04 m`，超出约 **0.01874449 mm**。它仍在标准开口参考的原 2 mm 容差内，故“编码器已张开”和“控制器入口拒绝”并不矛盾：两个模块检查的是不同的范围。这里把指令范围直接当成实测位置的严格拒绝条件，产生了不必要的启动阻断。诊断见 [wine_controller_diagnosis.json](wine_controller_diagnosis.json)，原装执行器定义见 [panda_gripper_source.xml](panda_gripper_source.xml)。本轮未修改这个边界检查。

![酒瓶第150步实际外部输入](D:/FYP/First_Phase/scene_demo/results/2026-10-11-neutral-live-combos/bowl_wine/runs/20261011T100138Z_goal_table_746c1a26/cap_02_wine_to_rack_b8708bf3/finish/sensors/cap000201_sideview.png)

## 灶台：已经开火，视觉检查却返回未知

第 150 个 VLA 动作的实际回答：

```json
{"goal_achieved":null,"reason":"sideview中灶台无明确开启状态视觉标识，无法判断是否已turn on"}
```

实际外部输入清楚显示红色加热圈，原始动作记录 `declared_predicates["turnon|flat_stove_1"]=true`、`native_success=true`。腕部图主要看到旋钮与手，开启标识应从外部图判断。Qwen 没有识别这个状态标志，不能归因于松手或归位失败。

按预登记的未知停止规则，第 150 步返回 null 后停止，没有继续到 300 步，也没有发出收尾动作。当前修订退离路线仍未实际进入；不能用前置碗归位通过替代灶台退离验证。

![灶台实际外部输入：红色加热圈](D:/FYP/First_Phase/scene_demo/results/2026-10-11-neutral-live-combos/bowl_stove/runs/20261011T101200Z_goal_table_223413f1/cap_02_stove_on_dfbfb1e5/finish/sensors/cap000093_sideview.png)

![灶台实际腕部输入](D:/FYP/First_Phase/scene_demo/results/2026-10-11-neutral-live-combos/bowl_stove/runs/20261011T101200Z_goal_table_223413f1/cap_02_stove_on_dfbfb1e5/finish/sensors/cap000092_wrist.png)

## 本次新视频

碗→酒瓶：末段瓶子已在架上、夹爪已打开，但未启动收尾归位。

![碗→酒瓶](D:/FYP/First_Phase/scene_demo/results/2026-10-11-neutral-live-combos/bowl_wine/sequence.mp4)

碗→灶台：末段已出现红色加热圈，视觉检查未知而停止。

![碗→灶台](D:/FYP/First_Phase/scene_demo/results/2026-10-11-neutral-live-combos/bowl_stove/sequence.mp4)

视频与动作时间线逐帧数量一致：682 帧、350 帧。VLA/抓辅每动作一帧，收尾每五步一帧，统一 20 fps；模型等待未录入，收尾在视频里加速，不代表墙钟速度。

## 输入与版本核对

七份本轮完成检查请求均已核对：运行期确为 `visual_goal_encoder_v3`；当前腕部＋外部两图与编码器同采样时刻，图片哈希匹配；完整提示词没有 phase/stage，模型只回答目标，开口由程序决定；仿真物体位置、接触和目标谓词未输入模型。这里的目标/接触记录只用于独立效果诊断。见 [live_input_audit.json](live_input_audit.json)，各请求旁的 `.txt` 是实际完整提示词及图片路径。

执行中检查点为 VLA150、300，首次 true 即移交收尾；成功归位后另有一次目标确认。null 停止与 false 继续到下一检查点的规则不同。本轮没有复用 60 步回答，也没有调用保存画面重新检查。所有过程保留原成功与失败，不重写历史记录。

新 VLA **750**、抓辅 **182**、收尾 **482**；Qwen **8 次 / 12627 tokens**（含一次抓前定位），历史回放 0、子agent 0。两个启动命令退出 0 只表示试验运行和归档完成，不表示组合成功。实际命令和环境在 `launch_*.json`，门禁输出在各组合的 `preflight.log`，控制与模型源码在 `source_snapshot.zip`。

## 下一步应处理的两个明确问题

1. 修复收尾初始化的指令范围转换：保留实测位置，用合法范围内的握爪保持/开爪命令接管；不要因这类微小实测越界拒绝第一条恢复命令，也不改写物理状态或取消路径保护。
2. 改善视觉目标判断：灶台应通过场景元数据明确“红色加热圈表示开启”，不预告当前已开；酒瓶检查仍需防止把夹持/接触条件混入位置目标。当前只凭更换中性提示词，尚不能保证 Qwen 正确判断这两类目标。

先解决这两个入口问题，再补原定组合验证；不重新调已成功归位，不据本轮建议微调 VLA。详细数值见 [summary.json](summary.json) 和原始请求/动作记录。
