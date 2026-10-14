# 同一子任务持续证据与历史重审：实现完成，离线验收未通过

用户目标：所有长程任务保留同一任务段的初态、过程、末态原图与拍摄时间，允许回看旧松手图补建证据，重点补上双罐任务第一次松手的汤罐；保留原收尾、退离和下一子任务复位。此次没有新环境动作、辅助器或子 agent。

**结论：通用机制已实现；汤罐的指定历史补证仍未触发，不能宣称任务已修好。** 入炉实际触发了旧松手补证，并得到模型整体 True，但图像区域和“当前可见”描述仍有错误。候选没有替换默认检查器，也没有据此启动收尾或复位。

| 检查 | 实际结果 | 验收含义 |
|---|---|---|
| 双罐，按旧时间线到第 300 步 | v7 整体 Unknown；第一次松手汤罐未确认 | 按停止规则停后续依赖验证 |
| 双罐，v7.1 针对性重审 | 整体 True；两项证据都来自当前第 300 步，event_id=null；旧第一次、第二次松手仍被回答为 outside | **没有完成指定的历史汤罐补证**；初始汤罐框还落在桌面空处，不能把当前 True 当作视觉验收通过 |
| 入炉，旧时间线 | 到第 300 步仍 Unknown；下一次第 450 步请求网络失败 | 完整时间线未跑完；失败与原始响应保留 |
| 入炉，v7.1 初态＋两次松手＋末态 | 起初换参照视角缺乏身份对应，被拦截；通用身份增补与一次协议修复后，杯子历史证据＋当前关门得到 True | 实际触发历史补证：observed_at=8.95、reviewed_at=45.75、release_0001。但框位置错误、当前可见性描述错误，**视觉质量仍未验收通过** |
| 无历史入炉末帧负例 | 杯子 Unknown、门关闭 True，整体 Unknown | 没有仅凭门关闭或桌面杯子消失推断入炉；仅这一项负例通过 |

这些时间是原仿真采样/检查进度时间，不是墙钟时间。旧初始参考没有可核实绝对拍摄时间，保存为 null，仅保留动作前顺序；没有补造反馈。

## 改动

- `xvla_subtask_evidence.py`：按 session、logical_subtask、attempt 隔离记录；稳定照片编号、原图 SHA、拍摄时间与同步编码器；保留所有已采集图和松手事件。原图不改写，文件改变会被拒绝。
- `xvla_subtask_review.py`：初态身份绑定、当前检查、历史重审分开；旧 Unknown/False 可以重新审核。保存旧意见，证据另记 observed_at、reviewed_at、真实请求/响应 SHA。事件多时按时间分页并记录覆盖范围；没有旧首四条上限，也不永久缓存第一次分类结果。
- 可用带旧参照关联的新视角补充身份；明确认错时可修订身份并要求原放置证据重新确认。不能仅靠同名 target_id 接受新物体。缺字段或身份引用缺链只影响对应目标，每任务段最多一次协议修复，修复证据指向实际修复响应。
- 关门、抽屉、开关、保持条件仍检查当前状态；历史放置可在遮挡时保留。相同区域不能重复计为两件待放物品；容器与其内容物允许在图中重叠。明确移出后不能用更旧松手图恢复证据。
- `xvla_service.py` 只改检查器工厂的候选接线和任务段元数据；匹配的拆分调用共享逻辑任务段，事件编号加调用范围以防碰撞。`xvla_pilot.py` 仅增加新模块的源码冻结记录。

## 为什么不能算通过

**双罐的第一处错误发生在初始身份对应。** Qwen 将初始 agentview 的 `[350,450,420,580]` 区域标为汤罐。对照 256×256 原图，这相当于约 x=90–108、y=115–149，主要是桌面空处。后续沿用这处错误参照，回答旧松手时汤罐“仍在原位”。第 300 步虽然另建当前证据，却没有使用第一次松手图补汤罐，不能替代本轮目标。

- [初始外部原图](../2026-10-13-xvla-timeline-trial/basket_independent/basket_two/runs/20261013T074733Z_basket_two_405c2b77/cap_01_xvla_basket_both_7ef2a1b9/finish/sensors/cap000001_agentview.png)
- [初始另一侧视原图：可见蓝黄圆柱罐](../2026-10-13-xvla-timeline-trial/basket_independent/basket_two/runs/20261013T074733Z_basket_two_405c2b77/cap_01_xvla_basket_both_7ef2a1b9/finish/sensors/cap000003_sideview.png)
- [第一次松手 wrist 原图：篮内可见罐顶](../2026-10-13-xvla-timeline-trial/basket_independent/basket_two/runs/20261013T074733Z_basket_two_405c2b77/cap_01_xvla_basket_both_7ef2a1b9/finish/sensors/cap000083_wrist.png)
- [真实重审及 current/history 分离结果](v7_1_basket/historical_probe.json)

**入炉存在协议问题，也存在视觉依据问题。** 原回答改用了同一初始时刻的 wrist 参照，却没有与已经记录的 agentview 参照建立对应。程序原先直接拒绝；现在已支持明确关联的参照增补，并让缺链走既定的单次协议修复。这次确实补出了旧事件证明，证明接口可达。但最终证据框 `[0,0,1000,500]` 指向旧 wrist 图上半部，黄色把手主要在下半部；末帧 wrist 明显被门板遮挡，模型却描述“当前看到炉内黄杯”。这说明有效编号、合法 JSON 和 True 还不足以证明视觉依据可靠。

- [第一次松手 wrist 原图](../2026-10-13-xvla-timeline-trial/microwave_mug/runs/20261013T071938Z_microwave_mug_1b637b05/cap_01_xvla_microwave_place_close_b844a1f4/finish/sensors/cap000101_wrist.png)
- [末帧 wrist 原图：门板遮挡](../2026-10-13-xvla-timeline-trial/microwave_mug/runs/20261013T071938Z_microwave_mug_1b637b05/cap_01_xvla_microwave_place_close_b844a1f4/finish/sensors/cap000563_wrist.png)
- [末帧侧视原图](../2026-10-13-xvla-timeline-trial/microwave_mug/runs/20261013T071938Z_microwave_mug_1b637b05/cap_01_xvla_microwave_place_close_b844a1f4/finish/sensors/cap000564_sideview.png)
- [补证元数据与实际修复请求来源](v7_1_microwave_identityfix/final_comparison.json)

## 验证与范围

最终 **361 tests、30 subtests 通过**，含全部 10 个原生长程的“首次未知→旧图补证→遮挡沿用”、作用域隔离、旧反馈不能借用当前开口、证据版本、分页覆盖、源文件变化拒绝、多物去重、撤销、协议修复与参照增补。这是软件覆盖；真实 Qwen 只使用入炉和双罐两个既有物理样本的照片，不代表 10 项实测成功。

本轮 **22 次实际 Qwen 请求尝试，21 份响应，1 次网络失败**；5 次完全相同的本轮既有请求按 prompt/context/原图 SHA 严格核对后复用，没有重新抽样。没有导入上一轮模型判断来强制结论；没有把仿真目标谓词送入 Qwen。原始请求体、原图字节、视角与时间、21 份响应来源均已核对。wrist/agentview 保持 256×256，sideview 保持 512×512。

回放入口最初漏合并旧日志的 `verification_frames`，发生了三视图校验错误，原错误记录保留。修正为复用既有 `capture_frames` 后继续，线上采集和此前补证探针未漏图。后续发现的字段含义、装置身份漏项、参照增补和容器重叠问题均作为通用接口修复，未加入杯子或汤罐专用通过规则。

43 个受保护控制文件哈希未变；`xvla_monitor.py` 全文件未变；service 检查器工厂之外 AST 未变。原收尾、退离、下一子任务位置/姿态/夹爪/控制器/策略缓存复位实现和参数保留。此次未执行这些动作，不能评价它们的新物理效果。默认检查入口保持原配置；未使用子 agent，主 Agent 的具体模型/推理档位调用元数据不可核实，记录 unknown。

还缺的关键证据是：**正确的汤罐初态身份对应，以及以第一次松手原图为来源的汤罐补证。** 后续工作应先解决图中目标定位和跨视角对应，验证引用区域确实包含目标，再评价整条长程检查；继续堆入图片或把 Unknown 放宽成 True 无法消除本轮已看到的问题。

补充更正历史摘要：上一轮 evidence-recovery 中第 450/600/750 步仍有模型当前到位判断，不能单凭存在 history 字段称为“纯遮挡沿用”；独立 recovery_probe 才实际出现当前 Unknown 后沿用番茄酱证据。历史条目与原始文件保持不变。

[原始请求与照片审计](audit.json) · [最终软件输出](pytest-identity.stdout.txt) · [结构化结果](summary.json)
