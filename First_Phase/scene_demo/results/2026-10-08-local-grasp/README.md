# 2026-10-08 本地抓取辅助标定：失败报告

## 结论
本次辅助标定失败，原定五个组合请求全部未开始执行。这次未运行新VLA推理，测到的是辅助器下降失败，不能据此归因于VLA训练。机械臂到达了酒瓶上方，但向瓶口下降时卡住，距离抓取目标仍约3.3厘米；达到80步下降上限后自动停止。夹爪还没闭合，所以这次还没检验“能否抓稳酒瓶”。本次标定没有解决抓取问题或提升 VLA，标定过程中也从未下发闭合夹爪指令。校准结果 ok=false、fatal_error=null、kind=physical、helper_reason=descend_timeout。

## 背景与分工
方案由开发用 GPT 设计与评审，实际实现由 DeepSeek 完成。未来运行时的目标链路为 Hermes(qwen3-vl-plus) → SmolVLA + wine 本地辅助器。本次标定使用 0 次 Hermes 调用、0 次新 VLA 推理：实际重放 102 条历史碗动作与 91 条历史酒动作，随后执行 124 条真实本地辅助动作（44 到上方 + 80 下降 + 0 闭合 + 0 抬升）。重放动作不等于新生成的 VLA 动作。

## 动作计数
| 阶段 | 次数 |
| --- | --- |
| above（移动到瓶口上方） | 44 |
| descend（下降） | 80 |
| close（闭合） | 0 |
| lift（抬升） | 0 |
| 本地辅助合计 | 124 |
| 历史重放 | 102 碗 + 91 酒 |

## 五个组合请求状态
全部为“已计划、未执行”（因标定失败），按顺序如下：
1. 请帮我整理桌面上的碗和酒瓶。
2. 把碗放到盘子上，然后把酒瓶放到架子上。
3. 请把酒瓶和碗都收回各自的位置。
4. 先把酒瓶放到架子上，再把碗放到盘子上。
5. 整理这两件东西：碗放盘子，酒瓶放架子。

## 实测事实
- 控制器与夹爪抓取点误差为 0。
- above 末态误差 2.246 mm / 0.435°。
- descend 末态三维位置误差 33.04 mm，其中垂直 24.143 mm、水平约 22.6 mm；结束时为 descend_timeout。
- 最后 10 步 Z 峰值约 1.024418 m，相对目标 EEF Z 1.000275 m 停滞。
- 无受保护物体（akita_black_bowl_1、plate_1、cream_cheese_1）违规。
- 源文件未改动。
- 上述数据不能区分可达性、碰撞或控制饱和，也不代表训练缺陷。

## 来源区分
- 原标定 JSON 的 final_snapshot / final_bowl_predicate 为 null，原样复制、未改写。
- 最后一条真实动作后测量仍记录 on|akita_black_bowl_1|plate_1=True、wine grasped=False。
- 最后一行 phase_after=descend，helper phase=descend；最终终态 helper summary=failed / descend_timeout。未知导出字段未被覆盖。
- 代理持有或失败辅助都不等于酒瓶放置成功，也不是对 VLA 技能的判定。

## 建议（仅建议，未执行）
下一步建议：从控制器、关节范围、接触三方面定位下降停滞，调整本地辅助器，并在评审后只做一次新标定。现在不建议训练，规模不得自动扩大。控制器23项、服务27项软件测试通过；它们不能替代物理抓取成功。新原型未改动生产 UI 默认值，未做微调，也未放宽成功判据。

## 证据与媒体
- [视频](calibration/calibration.mp4)
- [首帧](calibration/first.png)
- [末帧](calibration/last.png)
- [清单](manifest.json)
- [计划用例](planned_cases.json)
- [标定记录](calibration/calibration.json)

## 复现（此处未执行）
输出目录必须不存在，且与既有 calibration-v1 不同：
```bash
wsl -d Ubuntu -- env MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/yhwang/fyp/libero_demo/venv/bin/python -u /mnt/d/FYP/First_Phase/scene_demo/grasp_calibration.py --output-dir /home/yhwang/fyp/scene_demo/grasp_assist/FRESH_CALIBRATION_OUTPUT
```
