# 当前检查器实际接管小试

本轮尚未打通检查→松手→退离→归位。酒瓶两次因视觉把已搭在架上的瓶子判断为未到位而未触发收尾；修正后额外复核已实际触发，但仍判断错误。灶台复核确认开火并允许收尾，但收尾器在开爪前等待停稳10次后停止，真正松手/退离/归位均零。两个组合未运行，不计失败。

| 尝试 | 检查器实际触发 | 实际执行 | 独立效果 |
|---|---|---|---|
| 酒瓶初版 | 判断未到位（incomplete），无复核 | 新收尾动作0 | 目标动作未触发 |
| 酒瓶修正版 | 独立复核真实触发，仍判未到位（incomplete） | open/movement/confirm/home均0 | 目标动作未触发 |
| 独立灶台 | 复核后需要松手（needs_release），允许收尾 | 10次闭爪settle等待，真正open/movement/confirm/home均0 | settle not stable，未完成 |

旧wine输入expected_release=False，本轮要求actual_open>0；修正版在新inputs标True，原source不改。两个wine尝试是同状态两版本，不能合并为可靠性估计。两chain not_run，不计失败；UI保持off。stove末max arm speed=0.0261014993，超过0.02且未达五连；不能据此断言等久就成功。home实际0，不评价归位效果或修改旧成功home。

实际Qwen5次、15009 tokens；两次wine primary请求体SHA相同但响应SHA不同，按每次运行的call_index去重pending/final重复记录；request/response哈希仅审计补充，保留两个真实调用。新VLA/Hermes0、新saved检查0。初始化30+历史回放438+新settle helper10=478仿真动作；10等待不是松手10，actual_open0。

录像已逐帧完整解码（20fps）：两个wine各181帧，stove89帧。wine只有历史恢复回放，无新收尾；stove末10等待不是松手，均无新VLA。

![修正版wine录像](D:/FYP/First_Phase/scene_demo/results/2026-10-10-live-handoff/pilot_revision2/wine_exit/all_actions.mp4)

![独立stove录像](D:/FYP/First_Phase/scene_demo/results/2026-10-10-live-handoff/pilot_stove_independent/stove_exit/all_actions.mp4)

[首wine结果](pilot/wine_exit/result.json)、[修正版wine结果](pilot_revision2/wine_exit/result.json)、[stove结果](pilot_stove_independent/stove_exit/result.json)、[汇总](evidence/summary.json)、[逐帧验证](evidence/video_validation.json)、[最终精确源码](evidence/source_snapshot_final.zip)。原来源、门禁、launch及失败日志保留，版本sourcezip未覆盖。

Git只归档每个实际模型输入PNG、原crop来源和provenance、原oracle/helper/events/checks/result/video及门禁/源冻结；恢复时中间RGBD和标定数据留在本地，不删除，不声称全部本地帧已提交。具体清单见[evidence/manifest.json](evidence/manifest.json)。
