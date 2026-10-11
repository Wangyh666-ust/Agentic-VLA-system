# 番茄酱入篮、杯子入炉→关门：初次小试

**两例均未完整通过。发现新场景的外部检查视角有问题，不能把这轮失败归结为归位能力或训练缺陷。**

| 样例 | 实际发生什么 | 新VLA动作 | 开爪 / 退离 / 归位 |
|---|---|---:|---:|
| 番茄酱入篮 | 第150步Qwen因外部图未拍到篮子返回未知，停止；独立入篮目标也为假 | 150 | 0 / 0 / 0 |
| 杯子入炉→关门 | 第300步Qwen误判已入炉，收尾归位实际执行；独立入炉判定仍假，因此关门未启动 | 300 | 9 / 26 / 107 |

番茄酱实际回答：`{"goal_achieved":null,"reason":"sideview图像中未显示篮子，无法确认番茄酱罐是否在篮子内"}`。核对发送图后确认该外部相机漏掉篮子；同一时刻已有agentview包含篮子，但未送入检查器。这例没有收尾动作，不能评价收尾效果。

杯子第150步返回false；第300步回答：`{"goal_achieved":true,"reason":"sideview显示黄白杯已位于微波炉内部腔体中，满足'put the yellow and white mug in the microwave'目标"}`。外部相机主要看到微波炉背面，正面agentview可见杯子在入口附近。原生入炉目标在300次VLA采样中一次也未为真，收尾后独立五连为0。机器人原归位五连通过，**这不代表杯子入炉成功**。关门动作0，关门能力未验证。

本轮固定用户顺序，不评估Hermes自主规划。环境是LIBERO/robosuite/MuJoCo、Franka Panda；同一预训练SmolVLA，没有微调。150/300检查、同步腕部/外部图与夹爪编码器；无阶段标签或仿真目标输入Qwen。三步潜在子任务中实际只启动两步；Qwen4次、8483 tokens，新VLA450、收尾144，无抓辅、回放、子agent或独立归位试验。

后续只修正新场景外部检查相机，并明确任务要求的“入内”关系；保留初测，修改后各一次补测，不扩大或重测原归位。该补测同时改变相机选择与空间关系提示，不能独立归因二者贡献。

![番茄酱初测](D:/FYP/First_Phase/scene_demo/results/2026-10-11-container-pilot/sauce_basket/sequence.mp4)

![杯子入炉初测](D:/FYP/First_Phase/scene_demo/results/2026-10-11-container-pilot/mug_microwave/sequence.mp4)

详见[逐步统计和原始请求审计](summary.json)。源码与预登记提交：a3c6edd。
