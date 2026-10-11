# 原生 SmolVLA LIBERO 评测：完成

有效次数：200/计划200；已完成任务：40/40；成功数：139。
总成功率：69.5%。未完成或异常样本不填失败。

| Suite | 完成任务 | 有效次数/计划 | 成功数 |
|---|---:|---:|---:|
| libero_spatial | 10/10 | 50/50 | 34 |
| libero_object | 10/10 | 50/50 | 44 |
| libero_goal | 10/10 | 50/50 | 39 |
| libero_10 | 10/10 | 50/50 | 22 |

计分使用原生 `is_success`：单次为真即成功，不附加松手或归位要求。无 Hermes、Qwen、helper 或训练。
AMP关闭，沿用下载默认配置；动作上限依次为280/280/300/520。每任务使用官方前5个初始状态、环境seed1000..1004、模型seed1000。
当前MuJoCo3.8.1、hf-libero0.1.4，checkpoint为32层；尚未证明与论文设置相同。因此结果测量当前权重与环境组合，低分不直接证明训练缺陷，每任务5次仅作初筛。

原始SDK视频metadata为80fps，control为20Hz，因此源视频约4倍速播放；不补动作，原视频保留。

| Task | 成功/有效 | 五次成功 | 五次计分步数 | 原始完整指令 |
|---|---:|---|---|---|
| libero_spatial_00 | 2/5 | T,F,F,T,F | 80,280,280,74,280 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| libero_spatial_01 | 4/5 | T,T,T,F,T | 107,98,99,280,98 | pick up the black bowl next to the ramekin and place it on the plate |
| libero_spatial_02 | 4/5 | F,T,T,T,T | 280,108,88,96,103 | pick up the black bowl from table center and place it on the plate |
| libero_spatial_03 | 2/5 | T,F,F,T,F | 91,280,280,84,280 | pick up the black bowl on the cookie box and place it on the plate |
| libero_spatial_04 | 4/5 | T,T,F,T,T | 125,144,280,129,141 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| libero_spatial_05 | 2/5 | T,F,F,T,F | 97,280,280,94,280 | pick up the black bowl on the ramekin and place it on the plate |
| libero_spatial_06 | 4/5 | T,T,T,T,F | 116,114,105,102,280 | pick up the black bowl next to the cookie box and place it on the plate |
| libero_spatial_07 | 5/5 | T,T,T,T,T | 196,126,120,137,123 | pick up the black bowl on the stove and place it on the plate |
| libero_spatial_08 | 4/5 | T,T,T,T,F | 91,87,96,83,280 | pick up the black bowl next to the plate and place it on the plate |
| libero_spatial_09 | 3/5 | F,F,T,T,T | 280,280,125,119,126 | pick up the black bowl on the wooden cabinet and place it on the plate |
| libero_object_00 | 4/5 | T,T,F,T,T | 217,170,280,273,150 | pick up the alphabet soup and place it in the basket |
| libero_object_01 | 4/5 | T,F,T,T,T | 123,280,129,130,161 | pick up the cream cheese and place it in the basket |
| libero_object_02 | 5/5 | T,T,T,T,T | 123,125,106,125,127 | pick up the salad dressing and place it in the basket |
| libero_object_03 | 5/5 | T,T,T,T,T | 157,142,128,125,143 | pick up the bbq sauce and place it in the basket |
| libero_object_04 | 5/5 | T,T,T,T,T | 199,156,149,156,182 | pick up the ketchup and place it in the basket |
| libero_object_05 | 3/5 | F,T,T,T,F | 280,200,131,208,280 | pick up the tomato sauce and place it in the basket |
| libero_object_06 | 4/5 | T,F,T,T,T | 150,280,153,156,152 | pick up the butter and place it in the basket |
| libero_object_07 | 5/5 | T,T,T,T,T | 130,134,164,129,138 | pick up the milk and place it in the basket |
| libero_object_08 | 5/5 | T,T,T,T,T | 199,154,168,160,153 | pick up the chocolate pudding and place it in the basket |
| libero_object_09 | 4/5 | T,T,F,T,T | 133,127,280,120,145 | pick up the orange juice and place it in the basket |
| libero_goal_00 | 4/5 | T,F,T,T,T | 144,300,120,112,277 | open the middle drawer of the cabinet |
| libero_goal_01 | 4/5 | T,T,T,T,F | 91,97,79,90,300 | put the bowl on the stove |
| libero_goal_02 | 5/5 | T,T,T,T,T | 88,101,96,85,87 | put the wine bottle on top of the cabinet |
| libero_goal_03 | 2/5 | F,T,F,T,F | 300,193,300,208,300 | open the top drawer and put the bowl inside |
| libero_goal_04 | 2/5 | F,T,F,T,F | 300,92,300,85,300 | put the bowl on top of the cabinet |
| libero_goal_05 | 4/5 | T,F,T,T,T | 133,300,139,140,134 | push the plate to the front of the stove |
| libero_goal_06 | 5/5 | T,T,T,T,T | 87,100,105,105,109 | put the cream cheese in the bowl |
| libero_goal_07 | 5/5 | T,T,T,T,T | 69,72,73,72,69 | turn on the stove |
| libero_goal_08 | 4/5 | T,T,F,T,T | 83,84,300,80,85 | put the bowl on the plate |
| libero_goal_09 | 4/5 | T,T,F,T,T | 139,130,300,144,148 | put the wine bottle on the rack |
| libero_10_00 | 1/5 | F,T,F,F,F | 520,429,520,520,520 | put both the alphabet soup and the tomato sauce in the basket |
| libero_10_01 | 1/5 | F,F,F,T,F | 520,520,520,277,520 | put both the cream cheese box and the butter in the basket |
| libero_10_02 | 3/5 | F,F,T,T,T | 520,520,245,296,217 | turn on the stove and put the moka pot on it |
| libero_10_03 | 5/5 | T,T,T,T,T | 239,236,229,211,226 | put the black bowl in the bottom drawer of the cabinet and close it |
| libero_10_04 | 0/5 | F,F,F,F,F | 520,520,520,520,520 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| libero_10_05 | 5/5 | T,T,T,T,T | 191,240,238,172,172 | pick up the book and place it in the back compartment of the caddy |
| libero_10_06 | 3/5 | F,T,F,T,T | 520,251,520,229,219 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| libero_10_07 | 2/5 | F,T,F,T,F | 520,347,520,397,520 | put both the alphabet soup and the cream cheese box in the basket |
| libero_10_08 | 0/5 | F,F,F,F,F | 520,520,520,520,520 | put both moka pots on the stove |
| libero_10_09 | 2/5 | T,F,T,F,F | 312,520,326,520,520 | put the yellow and white mug in the microwave and close it |

已验证正常速度代表视频：源目标帧数相同，副本20fps呈现全部原帧。

- [libero_spatial_00/eval_episode_0.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_spatial_00/eval_episode_0.mp4)：80帧，20fps。
- [libero_spatial_00/eval_episode_1.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_spatial_00/eval_episode_1.mp4)：280帧，20fps。
- [libero_goal_02/eval_episode_0.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_02/eval_episode_0.mp4)：88帧，20fps。
- [libero_goal_07/eval_episode_0.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_07/eval_episode_0.mp4)：69帧，20fps。
- [libero_goal_08/eval_episode_0.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_08/eval_episode_0.mp4)：83帧，20fps。
- [libero_goal_08/eval_episode_2.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_08/eval_episode_2.mp4)：300帧，20fps。
- [libero_goal_09/eval_episode_0.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_09/eval_episode_0.mp4)：139帧，20fps。
- [libero_goal_09/eval_episode_2.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_goal_09/eval_episode_2.mp4)：300帧，20fps。
- [libero_10_00/eval_episode_1.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_00/eval_episode_1.mp4)：429帧，20fps。
- [libero_10_00/eval_episode_0.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_00/eval_episode_0.mp4)：520帧，20fps。
- [libero_10_09/eval_episode_0.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_09/eval_episode_0.mp4)：312帧，20fps。
- [libero_10_09/eval_episode_1.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-11-native-benchmark/presentation/libero_10_09/eval_episode_1.mp4)：520帧，20fps。
