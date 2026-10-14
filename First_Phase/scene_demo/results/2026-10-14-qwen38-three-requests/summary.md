# 当前流程：三条自然语言任务实测

使用用户三条原文，三个独立场景各尝试一次。Hermes、初识别与执行前选择复核使用Qwen3.8-Max；末端检查保留Qwen3-VL-Plus。使用原X-VLA、原收尾/退离/下一起点复位，无新增抓取搬运辅助器，无子agent。没有在测试过程中改提示词或放宽判据。

| 用户请求 | 实际结果 | 验收 |
|---|---|---|
| 整理桌面，包括碗、酒瓶，并且打开灶台 | Hermes实际提交碗放盘子→酒瓶放架子→开灶。三个动作、收尾和两次下一起点交接均通过，最后回global home。三个用户目标最终同时为真 | 本次整链成功，n=1 |
| 帮我将黄杯子加热一下 | Hermes错误认为“加热”与“入炉关门”原生能力完全匹配，未说明缺少启动加热动作。入炉关门从VLA第256步至900步持续满足原生AND；直到900步才触发收尾。机械臂最终到home，释放/静止独立筛查只连续4次通过，第5次失败，整链blocked | 加热目标没有得到覆盖；入炉关门本体达标，但完整验收失败 |
| 帮我扔一下桌面上的两个罐头 | 第一轮初识别请求报Qwen network failure，未收到模型回答，未启动Hermes规划或VLA；仅10步原生初始化settling | 丢弃意图判断和动作效果未验证；不能记作正确unsupported或VLA失败 |

## 整理任务：真实交接完成

- 碗：150 VLA步、217收尾步、5准备步；独立五连及handoff全部检查通过；复位目标为酒瓶原生任务起始状态。
- 酒瓶：150 VLA步、243收尾步；独立五连及handoff全部检查通过；复位目标为灶台原生任务起始状态。
- 灶台：150 VLA步、122收尾步；独立五连及handoff全部检查通过；末任务复位到global home。
- 三次共450 VLA步、15次X-VLA请求；582收尾/退离/归位步。原始记录确认关节、末端位置/姿态、夹爪、控制目标与策略缓存均按原检查通过，第二、第三策略实际接续启动。
- 最终独立目标：碗在盘子上=True；酒瓶在架子上=True；灶台开启=True。这些判据只用于独立验收，没有作为视觉模型输入。

![整理任务结束原图：碗在盘子上、酒瓶在架子上、灶台红圈亮起](D:/FYP/First_Phase/scene_demo/results/2026-10-14-qwen38-three-requests/table_tidy_stove/runs/20261013T171256Z_goal_table_0a0268d3/cap_03_stove_on_9b327f93/finish/sensors/cap000119_wrist_checker_sideview.png)

[完整结果](table_tidy_stove/result.json) · [真实Hermes回答](table_tidy_stove/agent/hermes_initial.log) · [Hermes原始调用日志](table_tidy_stove/agent/hermes_runtime_trace.log)

## 黄杯：三个问题需分开

**意图匹配错误。** 用户要求加热，Hermes的rationale却认为对象、目的地、顺序与`xvla_microwave_place_close`“完全匹配”。这个技能只有杯子入炉和门关闭两个goal，当前动作库没有启动微波炉加热的能力。不能把它记为完成加热，也没有温度变化的独立测量。

**视觉放行迟到。** 原生AND首次在第256步为真，随后到900步以及收尾结束持续为真。第300至750步的检查仍把门报unknown，理由是“炉腔入口可见、门未回到关闭位置”。在第880步的事件审查中首次回答closed，第900步当前检查才实际触发收尾。已保存同刻原图与时间标签，不能把这个结果解释成策略一直未关门。

**最终独立确认失败。** 原收尾控制器返回complete，实际执行release=2、movement=188、confirm=5、home=161，共356步。最终机械臂reference_metrics.ready=True，末端位置误差约0.285mm、姿态误差约0.00488rad，策略队列/控制目标已重置。原生两goal为真、手接触清单为空，但release_verified完成筛查仅连续4次为真，末次变成False，独立五连未通过，terminal=independent_finish_failed。失败处属于目标释放/静止筛查；没有保存该帧完整的杯子速度及抓取代理探针，不能断言具体是哪一项导致False，也不能只因机械臂到home就宣称全任务通过。

三个松手事件各自的三视图原图均已保留。杯子历史证据确实建立并留存：release_0001，observed_at=8.95、reviewed_at=9.25，status=confirmed，证据来自旧agentview且引用初始杯子参照。这轮历史输入与引用机制实际触发，但不能用它替代门状态或最终独立筛查。

![黄杯任务末态原图：机械臂已退离回位，杯子在炉内、门关闭](D:/FYP/First_Phase/scene_demo/results/2026-10-14-qwen38-three-requests/heat_yellow_mug/runs/20261013T173304Z_microwave_mug_a19ae68c/cap_01_xvla_microwave_place_close_f479dff1/finish/sensors/cap000854_wrist_checker_agentview.png)

[完整结果](heat_yellow_mug/result.json) · [规划原文](heat_yellow_mug/plan_during_execution.json) · [真实Hermes回答](heat_yellow_mug/agent/hermes_initial.log)

## 双罐丢弃：尚未进入规划

既有规则应把“扔掉”与收纳区分，当前公开场景没有垃圾桶或丢弃能力。但本轮不能给出Hermes是否遵守这一规则的结论：初识别第一轮就失败，没有初稿、复核或Hermes提交。

最后一张初态图片生成到报错日志约94.35秒；现有HTTP请求超时为90秒，超时是合理疑点。既有QwenClient将具体异常统一包装为Qwen network failure并抑制原异常，记录无法确认是读取超时、连接中断还是其他网络错误。请求尝试1次、响应0次，失败请求的服务端推理/用量未知。没有现场重试或改参数；失败原图、原始stderr与代码版本保留，需修复请求诊断后补测该项。

退出时还出现EGL析构warning，发生在Qwen失败之后，不能把它当作先发生的识别失败原因。

[原始失败结果](discard_two_cans/result.json) · [原始stderr](runtime/discard_two_cans.stderr) · [请求时长核对](discard_transport_timing.json)

## 调用与范围

- X-VLA实际45请求/1350步；初始化settling30步、准备10步、原收尾/退离/归位938步，总仿真动作2328步。
- 独立Qwen客户端成功响应25份：Qwen3.8-Max 6份、Qwen3-VL-Plus 19份；另有双罐初识别一次失败尝试。成功响应259944tokens，失败尝试用量未知。
- Hermes实际入口2次，原始日志各4个API调用，共8个Qwen3.8-Max调用、144469tokens。CLI的usage文件仍记录 unavailable，没有覆盖原文件；统计来自本轮session对应的原始API日志。成功响应可统计总tokens=404413。
- 两项实际执行任务各记录3个松手事件，每事件3张原图，共18张松手图；所有成功图片请求与原图哈希均已核验。
- 实际每请求图片最多33张；继承方案中的旧max_input_images=15字段未约束当前v7.1请求，统计按原始请求，不按该旧字段推算。
- 原收尾/退离/复位等九项受保护源码哈希匹配冻结版本。没有训练、策略回放或新增抓取搬运辅助器；进程已清理，8769/8794均不再监听。
- 本轮每项只有一次尝试，场景/指令不同，不能合并成可靠成功率。根线程实际模型/推理档位没有工具证据，不虚构设置；执行由本地脚本完成，子agent数为0。

[结果审计](audit.json) · [完整派生评估](assessment.json) · [Hermes日志用量](hermes_logged_usage.json) · [模型进程清理](runtime/server_cleanup.json)
