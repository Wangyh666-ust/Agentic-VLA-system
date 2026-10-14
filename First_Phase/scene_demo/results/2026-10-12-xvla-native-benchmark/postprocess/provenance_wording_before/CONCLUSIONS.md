# X-VLA 原生 200 次：结论与边界

## 主结论

**本次 40 个原生任务 × 每任务 5 次，200 次完整完成，197/200 = 98.5%。Spatial 48/50、Object 50/50、Goal 50/50、Long 49/50。**

模型 FP32 在 **RTX 5070 Laptop 8GB** 上真实执行推理，原生动作在 LIBERO/robosuite/MuJoCo 仿真中实际执行。成功采用作者 native goal，**无需附加释放、退离或 home**。

本轮的复现结论是：**“作者公开权重/客户端协议下的缩小样本复现，取得相近量级表现”**。

作者项目公布平均 98.1%；本轮 98.5% 不能解释为严格复现论文或超越作者。完整 40 任务各 5 次结果见 [README](README.md#2-40-个任务各-5-次结果)，逐例原语言/结果/路径见 [200 行 CSV](episodes.csv)。

## 独立核验结果

- 固定 slot 200 个且唯一、每任务5、每例唯一 `attempt_001`；`summary.complete=true`、`unknown_paths=[]`。
- 全部 policy 行 `done==independent_success`，197 个成功 episode 仅最后一行 true，三个失败全 false。这里是**独立读取原生谓词**，与 done 使用同一个 native goal，不是外部视觉或额外真实物理标注。
- 实际 **1,160 次 X-VLA 调用**，**32,184 policy + 2,000 settle 动作**。逐例调用数为 `ceil(policy_actions/30)`；辅助0、Hermes/Qwen0、训练0；内部 reset 物理子步未单独观测。
- 全部 response 有限 `30×20`，发送动作有限7维/夹爪±1。用原作者 `LiberoAbsActionProcessor` 数值转换全部 response，截到实际 policy 动作后，与 sent float32 **逐元素、逐字节精确相同**。未构造环境或执行回放动作。
- 400 份 slot/attempt result 的 fingerprint 与 metadata 一致，freeze 源文件和快照哈希无变化；原始 result/request/response/steps/video 保持。
- **200 段原视频全部完整解码成功**，合计32,184帧，逐例帧数等于 policy_actions，作者fps30；九段代表视频原字节副本SHA相同。[代表视频](README.md#5-视频原-30fps无转码)
- 控制频率 **20Hz**，原视频 **30fps**，播放时长不能当真实控制时间。
- episodewall_sum **6,142.17249584198 秒**，单例 **17.181641578674316–176.32755184173584 秒**；这是各例执行耗时总和，不是包含开发准备与首例暂停的总墙钟。
- 实际机器人 model：MountedPanda125例、OnTheGroundPanda75例，以原result为准。

完整逐项断言：[final_verification.json](final_verification.json)。

## 三个保留的失败

| Slot | 原语言 | 终止时 policy / 预算 |
| --- | --- | ---: |
| [Long03 ep2](libero_10/task_03/episode_2/result.json) | put the black bowl in the bottom drawer of the cabinet and close it | 900/900 |
| [Spatial04 ep1](libero_spatial/task_04/episode_1/result.json) | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 800/800 |
| [Spatial07 ep2](libero_spatial/task_07/episode_2/result.json) | pick up the black bowl on the stove and place it on the plate | 800/800 |

三例均 `budget_exhausted`，全程 policy 记录的原生成功谓词均 false。本报告只描述终止事实，不编写失败原因，不重试或修改原结果。

## 复现边界与尚未验证事项

1. 本轮每task5，而作者当前 CLI 默认50；公开 HF 格式转换还有约1%性能变化/下降提示，样本和模型发布条件不同。
2. Python客户端3.10与作者示例3.8不同；GPU torch2.11+cu128/RTX5070不同；MuJoCo2.3.7为本地固定选择，不是作者锁文件；模型RNG沿作者server默认未固定。
3. 保留作者core：30chunk、10denoise、domain3、absolute控制、800/900预算、原语言、原init、两路图像及原处理方式；实际配置见 [metadata](metadata.json)。
4. 旧SmolVLA139/200仅作**非配对历史背景**：模型、控制方式、预算、MuJoCo同时不同，不能分离单因素因果。
5. **持续场景拆分任务与任务间归位接续尚未验证**。本次原生长任务成功不等同于任意组合已修复，也不证明真机可靠性。

## 开发证据、停服与归档

三次实际历史 `deepseek-flash` 请求 usage 为原7553/9166、fix10107/7374、final1369/1304（input/output）。按原proof合计 **input19,029 + output17,844 = 36,873 tokens**，cache creation/read均0。contextrepair为**GPT诊断 + worker机械定位**，保留原失败记录，不能说3次实现均一次通过。

本次离线核验与归档**新增模型调用0、环境/机器人动作0**。实际执行者模型 `openai/gpt-6-astra`，推理档位未暴露，本地工具执行，无新子agent/DeepSeek请求。仅补一次既有11pytest：**11 passed、21 warnings、3.52s**，[完整原输出](software/final_pytest.log) 已归档。

client878已退出；server423完整argv/starttime匹配后仅发送SIGTERM并等其退出。GPU当前显存快照从5161MiB降至1307MiB，计算进程列表为空；**这不是峰值显存记录**。[停服证据](postprocess/server_stop.json)

稳定证据见 [manifest](manifest.json)。registry仅追加本campaign，旧41 entries数据哈希保留证明见 [registry_preservation](postprocess/registry_preservation.json)；原索引历史字节不变。最终全台账门禁以 [finalize.log](postprocess/finalize.log) 中 `review.py validate` 的 `ok=true` 为准。
