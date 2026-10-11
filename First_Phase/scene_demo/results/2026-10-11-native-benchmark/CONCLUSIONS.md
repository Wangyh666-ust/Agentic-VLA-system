# 原生 SmolVLA 标准40任务×5次：已完成

本轮40任务各5次，139/200原生成功，成功率69.5%。40任务审计、实际动作和原生分数一致；原有88份完整证据文件SHA256保持不变。原生成功使用is_success，不附加松手、退离或归位。

| Suite | 本轮成功/有效 | 本轮成功率 | 论文表2 |
|---|---:|---:|---:|
| Spatial | 34/50 | 68% | 90% |
| Object | 44/50 | 88% | 96% |
| Goal | 39/50 | 78% | 92% |
| Long | 22/50 | 44% | 71% |
| 平均 | 139/200 | 69.5% | 87.3% |

[SmolVLA论文表2](https://arxiv.org/html/2506.01844v1#S4)每任务10次、总400次，本轮每任务5次、总200次。当前公开权重是32层，尚未证实对应论文主0.45B、16层模型；当前MuJoCo3.8.1、hf-libero0.1.4也未证明与论文环境栈等价。本轮测量当前公开权重与安装环境的组合，不是论文严格复现，也不直接证明训练缺陷。

| 用户关心的任务 | 原生成功/有效 |
|---|---:|
| 酒瓶放架 | 4/5 |
| 碗放盘 | 4/5 |
| 灶台 | 5/5 |
| 双罐入篮（alphabet soup与tomato sauce罐，非后方ketchup挤压瓶） | 1/5 |
| 双杯左右盘 | 0/5 |
| 双moka壶上灶台 | 0/5 |
| 碗入底抽屉并关门 | 5/5 |
| 杯子入微波炉并关门 | 2/5 |

微波炉两个成功样本的计分步数是312和326。本轮没有Hermes、Qwen、helper或训练，已经证明当前native栈存在失败且技能差异较大，不能把全部失败归到Hermes，也不能笼统说模型完全不会组合任务。尚未量化自定义harness的因果贡献，尚未决定微调。

## 已有双罐入篮样本的离线观察

Long0 episode0的第一蓝罐已移入篮，末帧红绿罐倒在篮外；此样本未抓牛奶。episode1双罐原生成功。每段固定6帧只是静态证据，不能定位精确失败动作；没有逐目标原生谓词记录，不据此证明训练缺陷。

- [失败episode0固定帧](visual_review/libero_10_00_episode_0.png)
- [成功episode1固定帧](visual_review/libero_10_00_episode_1.png)
- [原帧数、抽帧索引与原生分数](visual_review/frame_manifest.json)

## 正常速度代表视频

原视频保留；源metadata为80fps，副本20fps，全部原帧保留，源目标帧数已验证一致，没有新增动作。以下T/F是原生success。

入篮失败 episode0（native F）

![入篮失败 episode0（native F）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_00/eval_episode_0.mp4)

入篮成功 episode1（native T）

![入篮成功 episode1（native T）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_00/eval_episode_1.mp4)

微波炉成功 episode0（native T）

![微波炉成功 episode0（native T）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_09/eval_episode_0.mp4)

微波炉失败 episode1（native F）

![微波炉失败 episode1（native F）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_09/eval_episode_1.mp4)

酒瓶成功 episode0（native T）

![酒瓶成功 episode0（native T）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_09/eval_episode_0.mp4)

酒瓶失败 episode2（native F）

![酒瓶失败 episode2（native F）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_09/eval_episode_2.mp4)

碗成功 episode0（native T）

![碗成功 episode0（native T）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_08/eval_episode_0.mp4)

碗失败 episode2（native F）

![碗失败 episode2（native F）](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_08/eval_episode_2.mp4)

## 中断恢复与实际执行

恢复时确认旧进程已退出，已有22个完整任务（110次）可复用；另有goal02不完整记录保留且不计分。110次原生结果86成功保留且未重跑。旧goal02不完整组已保存于interrupted_attempts目录，不填成5个失败。续跑仅补18任务90次，最终有效分母200。没有pause.json或可靠暂停信号记录，不能精确确认用户提出暂停瞬间的任务数。

有效40任务实际计分VLA环境动作44486，SDK padding环境动作15244，有效任务policy_select_calls从audit合计11984。这些是环境动作与策略选择计数，不是外部API调用数；progress累计12062包含中断期已持久化计数，不是全部实际历史调用的精确总数。SDK reset noops总量没有完整逐步记录，不补猜。验收与归档均零新增物理/VLM动作。

实现实际使用gpt-6.1-sol low，调用依据是本轮spawn与launch记录；物理执行使用本地SmolVLA。本轮没有DeepSeek/Qwen API调用。未把主Agent推理档位写成无法验证的运行设置。

[动作与计分验收](verification.json)、[完整逐任务报告](README.md)、[逐任务CSV](native_tasks.csv)、[来源与文件SHA256清单](manifest.json)。run_metadata记录启动前source_head=c932322adf8ce90f827837a7c5c76badb94215f0；首次runner提交是352095e14eb61c1bd07fb0333b4b176948f9fa6b，续跑提交是87ef8eeb84f909fddde9b772a94b1e856a4bf083。这三个来源点不同，source_snapshot保留首次和当前runner及reporter/验收脚本原字节，不用可变library源码替代历史快照。

下一步建议仅离线定位已有低分样本、核实公开权重与论文模型对应关系及环境差异，然后选择单因素最小对照。后续物理实验未授权，不自动运行，不微调。
