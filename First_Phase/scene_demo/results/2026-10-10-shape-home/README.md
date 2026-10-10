# 收尾与归位定点试点

归位验证通过，完整任务验收仍未通过。最后一版实际执行松手、退离和137个归位控制周期，连续五个动作后的采样归位到位且碗留在盘上；最终Qwen视觉误判盘子为空，所以原始结果仍为blocked / final visual failed。

本轮只恢复同一个历史 `bowl_held83` 停点，回放原83个VLA动作，再执行收尾。四版是修订过程，不是四个独立可靠性样本；前三版归位未启动，不能算执行归位失败。

| 版本 | 实际释放阶段 | 退离 | 确认 | 归位 | 原始结果与卡点 |
| --- | --- | ---: | ---: | ---: | --- |
| [几何与表面过滤](pilot_first/bowl_held83/result.json) | 6停稳 + 11开口控制周期 | 44 | 5 | 0 | blocked / not eligible for home；最终视觉误认仍持碗 |
| [加入实际阶段与编码器](pilot_visualfix/bowl_held83/result.json) | 6停稳 + 11开口控制周期 | 44 | 5 | 0 | blocked / not eligible for home；最终视觉因遮挡无法确认 |
| [实时侧视与较长退离](pilot_observation/bowl_held83/result.json) | 6停稳 + 11开口控制周期 | 89 | 5 | 0 | blocked / not eligible for home；最终视觉仍称接触/持碗 |
| [安全归位资格](pilot_home_gate/bowl_held83/result.json) | 6停稳 + 11开口控制周期 | 89 | 5 | 137 | blocked / final visual failed；实际归位通过，最终视觉误判 |

`release=17` 是6个停稳周期加11个标准开口命令周期，不是17次松手。每版实际开口均已核对 `helper.jsonl` 的 `finger_target` 与初始标准开口相等。最后一版的248条辅助动作与248条事件相符；最后五条真实动作后记录均为目标在盘上、释放稳定、无持有物、无机器人接触、归位到位。终点最大关节误差0.002632 rad；最终归位1/1，完整任务验收0/1。

## 改了什么

把约12 cm的手部保守球体换成实际机器人几何的保守包围盒；RGBD中只剔除匹配机器人CAD表面3 mm以内的自身点（网格使用三角表面距离），保留障碍点。包围盒仍是近似，不能称精确碰撞。2D图像重叠只作提示，不能当作3D接触。几何保护阈值仍为1 cm，没有随意降低。

第三版起，视觉检查使用实时agentview和sideview，退离候选改为4/6 cm，movement预算改为120。路线几何仍使用原两路RGBD，并修正像素中心半像素偏移；此前成功的实际归位控制循环保持。最后一版把“允许安全归位”与“任务完成”分开：先核对视觉到位/支撑、实际开口/停稳、新鲜几何清场和原归位路线安全，再允许归位；它不会宣告任务完成，最终完整视觉验收仍必须通过。归位路线123个采样均无碰撞，五次动作后归位确认通过。

## 看原始画面与结果

[最后一版视频](pilot_home_gate/bowl_held83/all_actions.mp4) · [agentview原图](pilot_home_gate/bowl_held83/sensors/cap001045_agentview.png) · [sideview原图](pilot_home_gate/bowl_held83/sensors/cap001047_sideview.png)

GPT已直接查看上述两张最终真实图像：碗在盘上，夹爪在上方远处且未持碗。这项人眼复核与独立物理记录一致。最终Qwen却称盘子为空、碗在桌上；错误输出完整保留，没有覆盖，也没有把归位通过改写成全链成功。

[最终result](pilot_home_gate/bowl_held83/result.json) · [逐次视觉检查](pilot_home_gate/bowl_held83/checks.jsonl) · [实际命令](pilot_home_gate/bowl_held83/helper.jsonl) · [事件](pilot_home_gate/bowl_held83/events.jsonl) · [独立物理记录](pilot_home_gate/bowl_held83/oracle.jsonl) · [四版机器可读汇总](evidence/observed.json)

## 分工、验证与范围

GPT负责设计和验收，实际DeepSeek共7次确定代码调用，本地worker做机械应用、运行和汇总。出处：[几何作者](evidence/author.json)、[表面过滤作者](evidence/surface_author.json)、[视觉作者](evidence/vision_author.json)、[观察作者](evidence/observation_author.json)、[归位资格作者](evidence/home_gate_author.json)。最后软件验证是69项中68项通过，1项因旧mock缺少新增当前筛查结果失败；精确补齐该mock后，仅失败的1项重跑通过，不能称一次全69项干净通过。同图离线Qwen诊断1次不是新物理效果样本。

本轮没有执行其余4个fixture，没有运行新VLA/Hermes或微调，没有扩大多任务测试或真实机器人测试。环境仍是LIBERO/robosuite/MuJoCo + Franka Panda。`finish_v1.pilot`尚未接入默认演示UI。多个条件一起改变，不能单独归因，也不能据一个状态宣称泛化或可靠性。

下一步只利用保存终态图像校准目标/盘子识别与最终验收，再考虑其余失败样例；不立即再跑归位，不微调。原始四版结果、输入、冻结与视频均保留，当前成果是已验证这一个状态的实际收尾归位能力，视觉最终验收仍需修正。
