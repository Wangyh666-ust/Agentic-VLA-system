# Qwen 3.8 Max 初始识别对照

已实际调用 qwen3.8-max 两次，分别重新识别旧实验的双罐、黄白杯初始同步三视图。两次 HTTP 200，服务端 model 均为 qwen3.8-max，输出未截断。除模型外，提示词、原图字节、视角顺序、时间标签及请求参数与旧实验相同，旧请求重建哈希匹配。

| 场景 | 人工对照原图的结果 | 验收范围 |
|---|---|---|
| 双罐 | 两个外部视角均正确指向蓝黄汤罐与红绿番茄酱罐；Milk纸盒、果汁盒、橙色扁包装已分开。腕部文字仍把番茄酱罐说成右下角，而实际在左下；汤罐位于右下，回答却称下中。还把腕部中央橙色扁包装对应成果汁盒，并把同刻Milk盒描述成腕部侧躺、外部直立 | 两外部视角目标识别改善；完整三视角一致性仍未通过 |
| 入炉 | 黄白杯三个视角的框均对应目标杯，与灰白斑点杯区分正确；两个外部视角识别到打开的门 | 目标杯的初始身份识别通过；未测试执行中关门判断、历史证据或完整在线任务 |

## 模型返回的目标框

坐标为每张原图0..1000归一化值，均来自本轮首次回答，未人工修改。

- 双罐：番茄酱obj_1，frame1=[455,435,505,505]、frame2=[560,460,610,530]；汤罐obj_2，frame1=[335,490,390,570]、frame2=[440,415,485,485]。模型没有给出两罐的腕部框，腕部对应仅在文字中声明。
- 入炉：黄白杯obj_1，frame0=[345,0,690,270]、frame1=[432,468,532,562]、frame2=[455,465,550,562]；灰杯obj_2的两外部框对应另一只杯。
- 本轮不是三次复核流程：每个场景只跑首次识别一次，未调用清单复核、Hermes规划、选择复核或末端检查。

## 配置与验证

- X-VLA专用Hermes profile的model.default已从qwen3-vl-plus改为qwen3.8-max，provider和supports_vision保留；原配置已在同目录备份为config.before-qwen38-20261014.yaml。
- setup_profile.py生成的新profile与InventoryClient默认使用qwen3.8-max；SCENE_PLANNER_MODEL可统一覆盖两者，显式model参数仍可覆盖初识别客户端。
- 当前旧SmolVLA专用profile未改；本轮操作的运行配置是/home/yhwang/fyp/xvla_native/hermes_home/config.yaml。
- 末端检查器仍为qwen3-vl-plus。原收尾、退离、复位、VLA后端和末端审核实现的九项源码哈希与上轮冻结版本一致。
- 已有三视图身份接口测试15项通过；未把这些软件测试作为视觉准确性的证明。
- 真实Qwen调用2次；Hermes推理0次；仿真/VLA/辅助动作0步；子agent0个。当前主agent的实际模型/推理档位未由工具确认，不作虚构记录。

## 原始证据

- [双罐请求](basket_two/request.json)、[原响应](basket_two/response.json)、[物体清单](basket_two/inventory.json)、[调用结果与用量](basket_two/result.json)。
- [入炉请求](microwave_mug/request.json)、[原响应](microwave_mug/response.json)、[物体清单](microwave_mug/inventory.json)、[调用结果与用量](microwave_mug/result.json)。
- [输入核对](input_validation.json)、[配置与受保护源码核验](configuration_verification.json)、[测试原始输出](pytest.stdout)。

双罐耗时89.45秒、5694tokens；入炉耗时85.94秒、5251tokens。合计10945tokens，每次thinking_budget=2048、max_tokens=6500、temperature=0。每场景n=1，旧模型未同期重测，不能据此估计成功率或宣称整个检查流程修复。
