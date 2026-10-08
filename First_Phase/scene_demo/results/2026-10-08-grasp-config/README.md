# 酒瓶抓取与配对配置实验（2026-10-08）

## 结论先说

当前权重本身可以完成“释放”：抓住酒瓶、把它放到货架上、并松开夹爪。本轮失败主要有两类：部分试验没有观察到稳定抓起；另一些已经把瓶子放上货架并松手，但没有命中 LIBERO 的指定区域。

本轮没有训练、没有微调，也没有强制张开夹爪。新加的抓取守卫改善的是“什么时候停下来”，并不提升模型已经学到的技能——它只做“发现不对就停”，不会凭空教会模型抓取。默认配置仍然是 A `baseline_bf16`，保留的就是它。

术语先按物理动作解释。“释放（release）”指瓶子不再被夹爪抓住、由货架托住，就像把手从杯子上拿开、杯子留在桌上。“原生窄目标区”指 LIBERO 官方规定货架上那一小块该放瓶子的地方，很窄，偏出一点就算没放进。“strict（严格分）”要求同时满足原生窄目标区且已释放、稳定；“semantic（语义分）”是一条独立的、事前写死的判据，见下文。

开发期由 GPT 做规划与复核，实际的源码执行由 DeepSeek 完成，两者都不在运行期控制链里。运行期仍是 Hermes（`qwen3-vl-plus`）加同一个 SmolVLA 策略。本实验直接下发固定指令，零次 Hermes 调用，因此不能据此得出任何 Hermes 规划增益的结论；也没有强制张开夹爪。实验只覆盖 `wine_to_rack`（把酒瓶放到货架上）这一个子目标，不代表整个 LIBERO 任务。

## 本轮改动的三件事

1. 酒瓶抓取监视器加就地早停：检测到瓶子没被抬起、而夹爪却在空手后退时，判定为抓取失败，当场停下、不重置场景，并阻止后续子目标继续执行。
2. 独立显示标准分与语义分：把原生“标准分”和独立“语义分”分开显示，不改动标准分，也不改动运行期成功判定。
3. 完全相同实际状态下的三配置对比：三个配置在实际状态、模型输入与配对模型种子的前提下运行，保证可比。

抓取监视器要连续 5 次动作后的检查都显示瓶子比起点高至少 2 厘米，而且接触检查显示夹爪持有瓶子，才确认已抓起。一次没被确认本身不能证明模型从未抓住过，只是没观察到足够证据。18 次 discovery 试验里共出现 5 次失败标记，这些被标记的运行里没有一次随后出现标准分或语义分成功；另有 2 次卡住的尝试漏掉了标记，一直跑到 300 步预算耗尽。7 条语义失败记录缺少稳定确认。

## 三配置对比

表里的“每次预测后执行几步（action horizon）”指每次预测之后要执行多少步，不是预测之前。

| 配置 | AMP | 去噪步数 | 每次预测后执行几步 | strict 严格分 | semantic 语义分（评估后） | 策略墙钟中位数 |
|---|---|---|---|---|---|---|
| A `baseline_bf16` | 开 | 10 | 1 | 2/6 | 4/6 | 173.4365 s |
| B `fp32` | 关 | 10 | 1 | 2/6 | 4/6 | 184.9355 s |
| C `fp32_h5` | 关 | 10 | 5 | 1/6 | 3/6 | 68.6675 s |

- AMP 是自动混合精度。“AMP 关”只表示关闭 BF16 混合精度，并不承诺每个计算 kernel 都改用 IEEE FP32。
- 选择规则在收集数据前就定死：先比 strict 计数最高，相同再比 semantic 计数最高，再相同取策略墙钟中位数最低者；只有三项指标（strict 计数、semantic 计数、策略墙钟中位数）完全相同时，才按固定顺序 A 先于 B、B 先于 C。
- 因此 C 虽然更快，但成功更低，未被选中；胜出并保留的是 A `baseline_bf16`。
- 统计只算 18 次 discovery 试验；1 次 A/A 对照不计入任何成功率。A/A 对照重复同一条 178 步轨迹，发出的动作逐条完全一致、最大绝对差为 0，只作可重复性证据，不声称逐位确定性。

配对不只看种子标签：同一实际状态、XML、相机图像与进入策略处理流程前的实际图像、状态和指令摘要都逐一比对通过。

## Holdout（选型后的留出检查）

选出 A 之后，用固定 state 3/4/5、model seed 3/4/5 做留出检查，只跑胜出的 baseline。三次 strict 全部通过，分别在 180、157、147 个策略动作处结束。这是配置选型的留出状态，不做其他验证、不重新选型、不给总体成功率，也不声称这些状态从未见过训练数据。

## 标准分与语义分

标准分沿用既有判据：原生窄目标区，加上已释放 / 稳定门；连续 5 次动作后的检查都通过，才完成标准任务。语义分是独立且事前固定的酒瓶 / 货架操作性判据：瓶子已不被夹住，货架有承托接触，而且连续 20 次动作后的检查都显示它保持稳定，才显示语义达成。它只用于展示与诊断，绝不覆盖 strict 完成、运行期成功或官方分数。

discovery 的描述性分组是：5 次 strict 通过、6 次仅语义成功、7 次两者都失败；holdout 的 3 次 strict 通过单独列出。6 个仅语义成功案例都在策略运行中首次完成，开始步依次为 163、220、224、182、183、191。例如配置 C、state 0 / model seed 1000 在策略步 163 语义完成，此刻原生目标区仍为否，运行继续跑到 300 步。这些瓶子已经释放、由货架托住，不是仍被抓住。释放只要求不再被夹住并得到货架支撑，不要求手指宽度恰好、也不要求执行张爪指令。

## 评估动作（assessment）

每条跑完的策略试验都会额外做 20 个独立的真实七维零动作，夹爪保持中立、不强制张开。全部 22 条记录（含 A/A）各做 20 次，合计 440 个评估动作。这些零动作会推进物理、可能帮助物体沉降，不是只读回放，也不是 VLA 输出。语义分在评估前 / 后分别记录：strict 为正的记录在 5 个稳定采样后即结束，因此语义评估前为假——原因是 20 个采样的观察时长还没凑满——评估后为真。绝不把评估后的成功提升为策略 strict 分数。四次守卫验证运行则不含任何额外评估。

## 实时守卫验证（四个用例）

- 原生失败参考（state 1 / model seed 1，enforce）：在 119 步以 `failed_grasp` 停下；同输入的 discovery baseline 原本跑到 300。
- 原生正参考（state 0 / model seed 0，enforce）：178 步成功，与精确一致的 baseline 相同。
- 共享 off（`goal_table`，state 0 / model seed 0）：酒瓶跑满 300 步、以 `budget_exhausted` 结束，碗被跳过。
- 共享 enforce（同一共享状态）：109 步以 `failed_grasp` 停下，碗被跳过。

四个用例在 0.25 s 冻结窗口内队列长度都保持 0，状态、步数与动作选择计数都不变。原生 119/178 与共享 109 的动作前缀逐条精确一致、最大绝对差为 0，首个输入也一致。作为对照继续跑到 300 步的失败样本，后来也没有达到标准或语义放置条件；这几项验证未观察到提前误停本来能成功的任务。四次验证与 18 次 discovery 分开，不合并比率，也不含 Hermes 调用、重试、强制张爪或额外评估。守卫只停当前这一轮执行，不是技能提升。

## 启动与运行

正常启动器现在显式使用 `--completion-mode release_verified` 与 `--grasp-guard-mode enforce`，健康检查要求 guard 为 enforce 才能复用。构造器 / 直接 CLI 的默认仍是 shadow：off 完全关闭，shadow 只记录不停，enforce 才停。正常酒瓶监视器按 enforce 运行。运行期在 blocked 之后仍可做至多一次既有的 Hermes 修复：守卫停的是当前这一轮，不是永久取消；网页已有的“停止任务”按钮会取消整个请求，并阻止修复重新打开。语义分只是解释结果，不会自动结束或推进计划。

```powershell
powershell -ExecutionPolicy Bypass -File D:\FYP\First_Phase\scene_demo\start_demo.ps1
```

启动后访问 <http://127.0.0.1:8081>。

## 视频

- [`videos/discovery_s0_m0_baseline_bf16_085e81c8.mp4`](videos/discovery_s0_m0_baseline_bf16_085e81c8.mp4) — strict 正例。
- [`videos/discovery_s0_m1000_fp32_h5_903f26f5.mp4`](videos/discovery_s0_m1000_fp32_h5_903f26f5.mp4) — 语义在货架上成立，但原生目标区未命中。
- [`videos/native_failed_grasp_stopped.mp4`](videos/native_failed_grasp_stopped.mp4) — 在 119 步停止的抓取失败。

视频是在看到结果之后挑选的，但全部 22 条配置实验记录（含 A/A）都保留。视频是 20 fps 的动作回放，不是真实墙钟速度录像。

## 证据文件

- 原始数据：[`raw/discovery.json`](raw/discovery.json)、[`raw/holdout.json`](raw/holdout.json)
- 预注册：[`raw/preregistration_discovery.json`](raw/preregistration_discovery.json)、[`raw/preregistration_holdout.json`](raw/preregistration_holdout.json)
- 守卫原始报告：[`raw/guard_00.json`](raw/guard_00.json)、[`raw/guard_01.json`](raw/guard_01.json)、[`raw/guard_02.json`](raw/guard_02.json)、[`raw/guard_03.json`](raw/guard_03.json)
- 交叉验收：[`verification/guard_validation_20261008_crosscheck_acceptance.json`](verification/guard_validation_20261008_crosscheck_acceptance.json)
- 汇总与清单：[`summary.csv`](summary.csv)、[`artifact_manifest.json`](artifact_manifest.json)、[`SHA256SUMS`](SHA256SUMS)

保留的 `initial_state.npy` 是真实的 float64 仿真器状态；只有在同一仿真模型 / 资产与所需 RNG 状态 / 种子的条件下，才能重建同一个起点。它本身不是完整仿真快照，也不是 XML 备份。文件 SHA（整个文件的哈希）与载荷 SHA（文件内数据的哈希）是两个不同的东西，不能混用；源码、XML、输入与模型的指纹都保留。模型 checkpoint revision 为 `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。

## 记录布局

- `trials/<trial_id>/` 保存每个试验（discovery 与 holdout）的初始状态与逐步遥测；`guard/guard_00` 到 `guard/guard_03` 对应四个守卫用例。
- `raw/` 是授权导出的原始报告，`videos/` 是三条入选回放；全部 22 条配置实验记录（含 A/A）都在这套布局内保留。

## 局限

只覆盖酒瓶子目标；仍有两个漏标记的 persistent-attempt 停滞；货架落点精度尚未修复；不是通用的抓取 / 重试 / 控制修复方案，也不代表整个 LIBERO 的可靠性。在考虑训练之前，先把问题诊断清楚。

## 附录：口径、判据与哈希

- 一步（step）= 一条七维控制指令，仿真 20 Hz；5 个采样 = 0.25 仿真秒，20 个采样 = 1 仿真秒；实际墙钟时间要慢得多。
- 检查都是本地的，不逐动作调用大模型。
- 选择判据（预注册）：strict 计数 → semantic 计数 → 策略墙钟中位数；三项指标（strict 计数、semantic 计数、策略墙钟中位数）完全相同时固定顺序 A 先于 B、B 先于 C。
- 冻结判据：四个守卫用例各在 0.25 s 睡眠前后两次探测，物理状态哈希、总步数、动作选择计数与动作队列长度相等，且动作前缀与参考逐条一致、最大绝对差为 0。
- 记录的 SHA-256 锚点（均为既有记录，不在此重算）：discovery 报告 `0c493abe860efb7bd130f7c2e2ae44141a3c8c237d7320790c2e2804eed63b5c`；holdout 报告 `036f8375b3223b50b685bdedc20dc9417734ee5fec57aad44a859879b9179a4a`；守卫报告依次为 `f2685beb91998cc116bf409e68772adc81311ccd95a47e471fe40ceb11fc94ee`、`e0c92758d9881d0eb5083a113507903a178097c71c57d2f3f1ff3ef4075091e2`、`061fa585a31b2c7337ef0f7879e257b0c3ce6203d6d33567915eb6820b8f0041`、`18d51feb4cea7bfa87e833e7cf5f55c1481cdd3c87ae063466bef4645449087f`；模型 `model.safetensors` 为 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`。
