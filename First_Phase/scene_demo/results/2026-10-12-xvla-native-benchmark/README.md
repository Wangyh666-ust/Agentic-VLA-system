# X-VLA 原生 benchmark：40 任务 × 5 次

**本次完成 200/200，成功 197/200 = 98.5%。Spatial 48/50、Object 50/50、Goal 50/50、Long 49/50。**

这是**作者公开权重/客户端协议下的缩小样本复现，取得相近量级表现**。X-VLA FP32 在 RTX 5070 Laptop 8GB 上真实执行推理，动作在 LIBERO/robosuite/MuJoCo 仿真中实际执行。原生成功判据无需附加释放、退离或 home。此结果不代表真实机械臂实测，也不代表持续场景任意任务组合已修复。

- [简明结论与限制](CONCLUSIONS.md)
- [完整独立核验及逐 episode 断言](final_verification.json)
- [200 行 episode 明细](episodes.csv) · [原 40 行任务统计](tasks.csv) · [原 summary](summary.json)
- [稳定文件 SHA256/size 清单](manifest.json) · [原来源冻结](freeze.json) · [实际配置](metadata.json)

## 1. 固定样本与实际计数

| Suite | 原生任务数 | 每任务次数 | 成功 / 完成 | 成功率 |
| --- | ---: | ---: | ---: | ---: |
| Spatial (`libero_spatial`) | 10 | 5 | 48/50 | 96% |
| Object (`libero_object`) | 10 | 5 | 50/50 | 100% |
| Goal (`libero_goal`) | 10 | 5 | 50/50 | 100% |
| Long (`libero_10`) | 10 | 5 | 49/50 | 98% |
| 合计 | 40 | 5 | 197/200 | 98.5% |

200 个固定 slot 唯一完整，每个只有 `attempt_001`；原 `summary.complete=true`、`unknown_paths=[]`。首例沿用已验收的原执行，没有重跑；其余 199 例由同一配置完成。软件启动/预检查失败不计入物理样本分母，历史失败日志保留。

| 项目 | 独立读取原始记录得到的数值 |
| --- | ---: |
| X-VLA HTTP 模型调用 / 请求 | 1,160 |
| 已执行 policy 动作 | 32,184 |
| 原生 settling 动作 | 2,000（每例 10） |
| 辅助动作 | 0 |
| Hermes / Qwen 调用 | 0 / 0 |
| 训练 / 微调 | 0 |
| 原作者 episode 视频 | 200 |
| 原视频完整解码帧数合计 | 32,184 |
| episodewall_sum | 6,142.17249584198 秒（约 102.37 分钟） |
| 单例 wall_seconds 范围 | 17.181641578674316–176.32755184173584 秒 |

`episodewall_sum` 是各例执行耗时总和，**不等于包含开发准备与首例暂停的总墙钟**。内部 reset 物理子步未单独观测，不包含在可枚举的 policy/settle 计数中。

## 2. 40 个任务各 5 次结果

下表编号均从 0 开始；`✓/✗` 依次对应 episode 0、1、2、3、4。原语言从各 episode 首个 `request_001.json.gz` 的 `language_instruction` 字段读取。完整路径、动作数、调用数和耗时见 [episodes.csv](episodes.csv)；原 [tasks.csv](tasks.csv) 保持原字节。

| Suite / task | 原始语言 | ep0–ep4 | 成功 |
| --- | --- | --- | ---: |
| Spatial00 | pick up the black bowl between the plate and the ramekin and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Spatial01 | pick up the black bowl next to the ramekin and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Spatial02 | pick up the black bowl from table center and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Spatial03 | pick up the black bowl on the cookie box and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Spatial04 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | ✓ ✗ ✓ ✓ ✓ | 4/5 |
| Spatial05 | pick up the black bowl on the ramekin and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Spatial06 | pick up the black bowl next to the cookie box and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Spatial07 | pick up the black bowl on the stove and place it on the plate | ✓ ✓ ✗ ✓ ✓ | 4/5 |
| Spatial08 | pick up the black bowl next to the plate and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Spatial09 | pick up the black bowl on the wooden cabinet and place it on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object00 | pick up the alphabet soup and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object01 | pick up the cream cheese and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object02 | pick up the salad dressing and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object03 | pick up the bbq sauce and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object04 | pick up the ketchup and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object05 | pick up the tomato sauce and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object06 | pick up the butter and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object07 | pick up the milk and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object08 | pick up the chocolate pudding and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Object09 | pick up the orange juice and place it in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal00 | open the middle drawer of the cabinet | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal01 | put the bowl on the stove | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal02 | put the wine bottle on top of the cabinet | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal03 | open the top drawer and put the bowl inside | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal04 | put the bowl on top of the cabinet | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal05 | push the plate to the front of the stove | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal06 | put the cream cheese in the bowl | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal07 | turn on the stove | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal08 | put the bowl on the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Goal09 | put the wine bottle on the rack | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long00 | put both the alphabet soup and the tomato sauce in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long01 | put both the cream cheese box and the butter in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long02 | turn on the stove and put the moka pot on it | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long03 | put the black bowl in the bottom drawer of the cabinet and close it | ✓ ✓ ✗ ✓ ✓ | 4/5 |
| Long04 | put the white mug on the left plate and put the yellow and white mug on the right plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long05 | pick up the book and place it in the back compartment of the caddy | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long06 | put the white mug on the plate and put the chocolate pudding to the right of the plate | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long07 | put both the alphabet soup and the cream cheese box in the basket | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long08 | put both moka pots on the stove | ✓ ✓ ✓ ✓ ✓ | 5/5 |
| Long09 | put the yellow and white mug in the microwave and close it | ✓ ✓ ✓ ✓ ✓ | 5/5 |

## 3. 三个失败：仅记录终止事实

| Slot / 原结果 | 原始语言 | policy / budget | 模型调用 | 终止 |
| --- | --- | ---: | ---: | --- |
| [Long03 ep2](libero_10/task_03/episode_2/result.json) | put the black bowl in the bottom drawer of the cabinet and close it | 900/900 | 30 | budget_exhausted |
| [Spatial04 ep1](libero_spatial/task_04/episode_1/result.json) | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 800/800 | 27 | budget_exhausted |
| [Spatial07 ep2](libero_spatial/task_07/episode_2/result.json) | pick up the black bowl on the stove and place it on the plate | 800/800 | 27 | budget_exhausted |

三个失败的全部 policy 行 `done=false` 且 `independent_success=false`，到作者预算耗尽停止，原结果和视频保留。此归档不对失败原因作推断。

## 4. 计分与动作证据核验

核验独立解析每个 slot 和 attempt 的原 `result.json`、全部 `steps.jsonl`、`calls.jsonl`、`response_NNN.json`，并读取首个压缩 request 的语言。未依赖 summary 作为唯一统计来源。

1. 全部 policy 行的 `done` 与 `independent_success` 一致；197 个成功 episode 仅最后一行 true，三个失败全 false。作者 `results.json`、result 的 `author_score/success/independent_success` 一致。
2. 每例 policy 行数等于 `result.policy_actions`，settle 行数严格为 10，无第三类动作；调用数等于 `ceil(policy_actions/30)`，也等于 `model_calls/model_requests`。
3. 全部 1,160 个原 response 都是有限 `30×20`。全部发送动作有限且 7 维，夹爪仅 ±1。
4. 在原 sim_venv Python 和原 PYTHONPATH/CONFIG 中导入原作者 `LiberoAbsActionProcessor`，按原 `ClientModel` 的位置、6D旋转、夹爪列和 float32 阈值规则转换全部 response，再截到实际动作数。全部 32,184 个 policy 动作逐元素精确相等，float32 原始字节也相等；各例双方 SHA256 均保存。
5. 400 份 slot/attempt result 的 fingerprint 全等于 metadata 的重算值 `e0cb4f80464dd8f610cda924e710551603b42b06d851361aa27e802560e4abaa`。原 freeze 文件、全部冻结源/快照、首例既有哈希、BDDL/init 来源均核对无漂移。

“独立”准确含义是**独立读取原生谓词**，与 done 使用同一原生 goal；并非外部视觉验收或额外真实物理标注。离线转换未构造环境，未调用 `env.step` 或模型；网络连接被阻断。本次后处理新增DeepSeek/Qwen/VLA调用0，机器人/仿真动作0。

机器人状态以实际 result 为准：`MountedPanda` 125 例、`OnTheGroundPanda` 75 例，原生 PandaGripper、OSC_POSE 绝对控制；未将首例安装形式外推给所有任务。

## 5. 视频：原 30fps，无转码

200 个 episode 均存在作者生成的唯一 mp4。使用当前 `imageio_ffmpeg.get_ffmpeg_exe()` 对所有视频完整解码到 null，关闭补帧，逐例得到精确帧数；均满足 **frames = policy_actions、fps = 30**。原始输出见 [video_decode.log](postprocess/video_decode.log)，结构化逐例结果见 [final_verification.json](final_verification.json)。

**物理控制是 20Hz，原视频编码是 30fps。** 作者每个 policy 动作保存一帧，30fps 播放长度为 policy_actions/30；物理控制步对应 policy_actions/20 的仿真时间。视频播放时长不能当成真实控制耗时或 episode wall_seconds；视频也不包含单独的 10 个 settling 帧。

以下均为原视频**原字节复制**，不是新轨迹；[副本映射及 SHA256](postprocess/representative_videos.json) 已核验源/副本相同。

| 代表样例 | 原字节副本 |
| --- | --- |
| Long00 ep0：双罐入篮 | [视频](videos/long00_ep0_two_cans.mp4) |
| Long04 ep0：双杯上盘 | [视频](videos/long04_ep0_two_mugs.mp4) |
| Long08 ep0：双壶上灶 | [视频](videos/long08_ep0_two_moka_pots.mp4) |
| Long09 ep0：杯子入微波炉并关门 | [视频](videos/long09_ep0_microwave.mp4) |
| Goal08 ep0：碗上盘 | [视频](videos/goal08_ep0_bowl.mp4) |
| Goal09 ep0：酒瓶上架 | [视频](videos/goal09_ep0_wine.mp4) |
| 失败 Long03 ep2 | [视频](videos/failed_long03_ep2.mp4) |
| 失败 Spatial04 ep1 | [视频](videos/failed_spatial04_ep1.mp4) |
| 失败 Spatial07 ep2 | [视频](videos/failed_spatial07_ep2.mp4) |

## 6. 复现范围及配置差异

结论固定为：**作者公开权重/客户端协议下的缩小样本复现，取得相近量级表现**。不能称严格复现论文，也不能据 98.5% 与 98.1% 的数值比较声称超越作者。

- [作者项目](https://github.com/2toinf/X-VLA/blob/6bc2513f5f1cbec715cc668b414392a6cae5c671/README.md) 公布平均 98.1%；同时提示原模型转换为 HF 格式后在不同数据集有约 1% 的性能变化/下降。本轮使用公开 HF 权重 `2toINF/X-VLA-Libero`。
- 本轮每 task 5 次、init 0–4；[原客户端快照](source_snapshot/libero_client.py) 的当前 CLI `eval_time` 默认 50。本轮样本规模与作者评估不同。
- Python 客户端为 3.10.22，作者示例为 3.8；模型端 Python 3.12.3 / torch 2.11.0+cu128 / RTX 5070 Laptop，与作者环境不同。
- 客户端 torch 1.11.0+cpu、numpy 1.22.4、robosuite 1.4.0、MuJoCo 2.3.7；MuJoCo 2.3.7 是本地固定选择，不是作者锁文件。
- 模型 RNG 沿作者 server 默认未固定，不能宣称模型采样随机流精确复现。
- 保留 author core：30-action chunk、10 denoise、domain 3、absolute 控制、Spatial/Object/Goal 800 与 Long 900 预算、原语言、原 init、两路图像。agentview 旋转180°、wrist 原样，原处理器224×224与第三路遮零、原 proprio 更新方式均沿冻结配置。
- 权重 revision：`129e71460678b7236cee6fc9707f09d9fa0c3590`；权重 SHA256：`260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8`。权重校验复用原安装与冻结记录，本轮不加载模型。
- X-VLA source：`6bc2513f5f1cbec715cc668b414392a6cae5c671`；LIBERO source：`8f1084e3132a39270c3a13ebe37270a43ece2a01`。完整配置见 [metadata](metadata.json) 与 [原 proposal](source_snapshot/20261012-xvla-native-benchmark.json)。

[旧 SmolVLA 原生 139/200](../2026-10-11-native-benchmark/CONCLUSIONS.md) 可作历史背景，但不是配对比较：模型、控制方式、预算、MuJoCo 同时不同，不能分离单因素效果。**持续场景拆分任务与任务间归位接续尚未验证**，原生长任务成功不等同于任意组合已修复。

## 7. 开发调用与实际归档分工

三次历史 DeepSeek 请求均有原 request/response/proof，按 proof 的实际 usage 求和：

| 阶段 / 原 proof | 实际模型 | input tokens | output tokens |
| --- | --- | ---: | ---: |
| [author](authoring/author/proof.json) | deepseek-flash | 7,553 | 9,166 |
| [author_fix](authoring/author_fix/proof.json) | deepseek-flash | 10,107 | 7,374 |
| [author_final](authoring/author_final/proof.json) | deepseek-flash | 1,369 | 1,304 |
| 外部DeepSeek HTTP请求合计：3 次 | | **19,029** | **17,844** |

上述三次外部DeepSeek HTTP请求总 tokens **36,873**，cache creation/read 均 0；不包含GPT规划、验收及工具worker子会话的模型用量。这里是历史DeepSeek开发调用；与真实评测的 1,160 次 X-VLA 调用分开统计。

contextrepair 是 GPT 诊断与 worker 机械定位，原候选并非整体逐字应用，曾拒绝不合接口的候选测试。旧 proof 中 10 passed/1 failed 与后续环境修正记录保留，**不能说三次实现均一次通过**。此次归档实际使用GPT6 Astra工具worker子会话执行本地命令，执行者实际模型为 `openai/gpt-6-astra`，推理档位未暴露；不假称调用了其他模型或指定 high/low。后处理新增DeepSeek/Qwen/VLA调用0；GPT工具worker子会话自身的模型用量不计入上述DeepSeek HTTP用量。

## 8. 软件验证、停服与证据清单

- [首例原验收](first_acceptance.json) 保留，后续只复用。
- 离线 `review.py check` 为 `analyze_existing`，`ok=true`，`launch_authorized=false`：[命令与原输出](postprocess/review_analysis.json)。结构门禁不代替科学结论。
- 仅补跑既有 11 pytest：**11 passed, 21 warnings in 3.52s**。完整原始输出含 warnings：[software/final_pytest.log](software/final_pytest.log)；[命令、环境与退出码](postprocess/final_pytest.json)。未改 runner/test，未扩大测试。
- client878 已退出；server423 完整 argv 和 start_ticks=491 与 `launch_remaining` 一致，只向该 PID 发送一次 SIGTERM，等到 `/proc/423` 消失。[server_stop.json](postprocess/server_stop.json) 保留原进程身份和 GPU 输出。
- 当前 GPU 快照：停止前 5,161 MiB、停止后 1,307 MiB，GPU 总量 8,151 MiB；停止后 compute-apps 列表为空。本任务模型计算资源已释放。**这是两个时点的当前信息，不是运行峰值，也不表示整机显存归零。** [停止前原输出](postprocess/gpu_before_stop.json)。
- [manifest.json](manifest.json) 记录 campaign 稳定原始/派生文件 SHA256 与 size，排除自身和生成时仍在写的 `postprocess/finalize.log`，避免循环自哈希。日志闭合后另行只读验 hash。
- 追加 registry 的条目 ID 为 `20261012-xvla-native-benchmark`；仅登记稳定报告、summary/tasks/episodes/final_verification/manifest/freeze/first_acceptance/metadata、source_snapshot 和 200 slot result。索引、registry 本体和活跃日志不登记为该条 evidence。
- [旧41 entries 保留证明](postprocess/registry_preservation.json) 按规范化数据 SHA256 验证全部旧 entry 不变，旧索引字节前缀保持。最终 `review.py validate` 的原始输出及完整性收据见 [finalize.log](postprocess/finalize.log)，以其中 `ok=true` 为归档通过依据。
- [归档工具快照](postprocess/xvla_20261012_archive.py) · [执行记录与已修正的工具路径映射错误](postprocess/WORK_LOG.md)。后处理工具曾在 Windows→WSL 路径映射处报错，修正后完成核验；未改变原始数据或判据。
- [来源措辞更正记录](postprocess/provenance_wording_correction.md) · [更正前报告原字节](postprocess/provenance_wording_before/) · [不变性证明](postprocess/provenance_wording_preservation.json) · [更正后台账门禁](postprocess/provenance_wording_validation.json)。本次更正由GPT工具worker子会话执行，未重跑pytest、视频解码或模型。

只读复核 manifest 的命令（不启动模型/环境）：

```text
wsl -d Ubuntu --cd /mnt/d/FYP --exec python3 /mnt/d/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-native-benchmark/postprocess/xvla_20261012_archive.py verify
```
