# 150/300步真实在线组合试点

**结论：两个组合都实际启动并执行了第二任务，但完整流程均未通过。两次碗任务均正确到位并完成收尾、归位和独立验收。酒瓶第150步仍在桌面，Qwen误报已在架上；实际退离归位完成，但放架失败。灶台第150步已经打开，Qwen正确触发收尾；开爪20周期后仍未达到开口门槛，因此退离和归位均未启动。不能把这次称为两个组合已跑通。**

本次从头运行预训练SmolVLA，生成全新录像和原始记录。没有复用旧60步判定、回放旧动作或再次调用保存图片检查。每个组合仅初始化一次场景；子任务间保留物体状态，实际收尾并归位、重置策略队列、读取新观察后执行下一步。

| 组合 | 子任务 | 新VLA动作 | 在线检查步 | 实际开爪周期 | 退离周期 | 归位周期 | 独立连续通过 | 结果 |
|---|---|---:|---|---:|---:|---:|---:|---|
| bowl_wine | bowl_to_plate | 150 | 150 | 2 | 103 | 131 | 5 | complete |
| bowl_wine | wine_to_rack | 150 | 150 | 14 | 131 | 133 | 0 | independent_finish_failed |
| bowl_stove | bowl_to_plate | 150 | 150 | 2 | 103 | 131 | 5 | complete |
| bowl_stove | stove_on | 150 | 150 | 20 | 0 | 0 | 0 | release not confirmed |

## 具体发生了什么

### bowl_wine：blocked

**bowl_to_plate**：归位连续通过 5，全目标独立验收连续通过 5。

第150步在线检查：

```json
{"goal_achieved":true,"holding_or_contact":true,"reason":"sideview shows bowl resting on plate, gripper still in contact with bowl"}
```

收尾后最终检查（不是新的VLA检查点）：

```json
{"goal_achieved":true,"holding_or_contact":false,"reason":"sideview shows bowl resting on plate with gripper lifted and not in contact"}
```

最终独立位置/开关判据：`{"on|akita_black_bowl_1|plate_1": true}`。

**wine_to_rack**：归位连续通过 5，全目标独立验收连续通过 0。

第150步在线检查：

```json
{"goal_achieved":true,"holding_or_contact":false,"reason":"sideview显示酒瓶已稳定放置在架子下层木板上，wrist view确认位置关系；sideview中夹爪悬停于酒瓶上方无接触，编码器数据结合图像判断未持物"}
```

收尾后最终检查（不是新的VLA检查点）：

```json
{"goal_achieved":true,"holding_or_contact":false,"reason":"sideview shows wine bottle placed on rack, gripper above bottle without contact"}
```

最终独立位置/开关判据：`{"on|akita_black_bowl_1|plate_1": true, "on|wine_bottle_1|wine_rack_1_top_region": false}`。

![本次新执行 bowl_wine](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-online-150-300-combos/bowl_wine/sequence.mp4)

### bowl_stove：blocked

**bowl_to_plate**：归位连续通过 5，全目标独立验收连续通过 5。

第150步在线检查：

```json
{"goal_achieved":true,"holding_or_contact":true,"reason":"sideview显示碗已稳定放置在盘子上，满足子任务位置要求；wrist和sideview均显示夹爪仍接触碗，编码器数据与开合状态一致但需结合图像确认接触"}
```

收尾后最终检查（不是新的VLA检查点）：

```json
{"goal_achieved":true,"holding_or_contact":false,"reason":"sideview显示碗已稳定放置在盘子中央，位置符合要求；wrist和sideview均显示夹爪悬停于碗上方无接触，编码器开合状态与标准开合值接近且图像确认无持物"}
```

最终独立位置/开关判据：`{"on|akita_black_bowl_1|plate_1": true}`。

**stove_on**：归位连续通过 0，全目标独立验收连续通过 0。

第150步在线检查：

```json
{"goal_achieved":true,"holding_or_contact":false,"reason":"sideview中灶台红色加热圈激活表示已开启，夹爪未接触灶台旋钮"}
```

最终独立位置/开关判据：`{"on|akita_black_bowl_1|plate_1": true, "turnon|flat_stove_1": true}`。

![本次新执行 bowl_stove](/mnt/d/FYP/First_Phase/scene_demo/results/2026-10-11-online-150-300-combos/bowl_stove/sequence.mp4)

## 配置与结论边界

在线检查只在每个子任务的第150、300个实际VLA动作发生；150步判断达到目标后立即停止VLA并接管，因而不再运行到300步。收尾后的最终确认属于另一个环节，不是第三个VLA定时检查点。每次Qwen接收同一时刻腕部第一视角、外部侧视第三视角两张原图及同步夹爪编码器；完整提示词中不发送phase/stage标签。
检查器判断目标位置/开关状态，不要求已经松手或退离才允许接管。实际收尾由原有控制器执行；归位实现和参数保持原版本。本轮没有单独复测归位，没有训练或微调，没有子agent。
独立验收使用启动前固定的目标：碗在盘子上、酒瓶在架子目标区域、灶台打开；第二步同时验证前一步目标仍成立，并检查无持物、无手接触、归位连续五个真实动作后采样。仿真谓词和接触真值不输入Qwen。不以Qwen说完成或单纯归位代替物理目标。
此前保存负例中Qwen的误判仍保留为失败。本次根据用户明确的新请求直接验证真实在线配置，没有把旧负例伪造为通过。两个顺序由用户指定，没有新Hermes规划调用，因此本次验证执行/检查/交接，不衡量Hermes的规划增益；小范围结果不代表可靠性。
录像为本次真实动作，顶部标注NEW ONLINE 150/300及当前子任务。沿用原始记录的缩时采样：VLA每动作一帧，辅助动作每五步一帧，统一20fps，不含模型等待。原始未加标注录像也保存。
本轮新VLA 600动作，辅助 785周期，历史回放0，Qwen 7次/13758tokens（只统计有原始响应的Qwen调用，不代表整个Codex会话用量）。

## 两处剩余故障与下一步

酒瓶：第150步输入两张原图均显示瓶子仍在桌面前方。Qwen误把它判断为“架子下层”，收尾后又误报到位；独立验收正确拦截。这里有位置关系识别错误，也有放架尚未完成；由于误判在150步就停止VLA，本轮不能断言继续到300步仍会失败，更不能据此认定训练缺陷。下一步先处理这个空间关系误判，复用本次已保存的错误输入，不扩大物理测试。
灶台：开火目标真实达到，目标触发不是问题。最后开爪反馈为[0.0214956,-0.0380312]，标准为[0.0387306,-0.0387278]；一根手指明显未全开。独立接触记录显示gripper0_finger1_collision仍接触flat_stove_1_g5。当前收尾器先要求全开，未达到后在20周期停止，而未进入解除接触的退离。这说明当前末态仍存在“先全开才能退离，但接触妨碍全开”的阻断；不能仅由预算耗尽推断增加等待就会解决。Qwen另称夹爪未接触，与独立接触记录不一致。下一步修的是带开爪指令的受保护脱离入口，复用已成功的归位模块；本轮不修改、不重试。
本轮程序结果保存后，两个进程退出时各出现EGL_NOT_INITIALIZED析构告警；运行结果和视频已保存，未将退出清理告警当成物理失败原因。
