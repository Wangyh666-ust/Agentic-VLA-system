X-VLA已通过真实Hermes进入持久场景，无抓取/运输辅助；本轮四组整链均未完成。酒瓶和汤罐实际收尾并到达下一任务起点，独立五连通过，但最终视觉否决；白杯和入炉均没有触发收尾，下一任务未启动。下一策略真实接续与末任务回globalhome尚未验证。

| 主试点目标 | VLA/prepare/finish | 终态 | 未启动 |
|---|---|---|---|
| wine_to_rack | 150/15/254 | final visual failed，complete=false | bowl_to_plate, stove_on |
| soup_to_basket | 750/16/222 | final visual failed，complete=false | sauce_to_basket |
| white_mug_left | 800/5/0 | budget_exhausted，complete=false | yellow_mug_right |
| mug_to_microwave | 800/5/0 | budget_exhausted，complete=false | microwave_close |

真实Hermes顺序4/4匹配；分母为4个独立主case，各一次，整链完成0/4。v2前置模式失败VLA0/finish0；v3传感器缺h5py，VLA5/调用1、prepare15、finish0；二者单列启动阻断，不作为整链主样本，实际用量仍保留。不能用未触发的零收尾动作断言收尾机制无效。

Wine下一碗起点末端误差0.83525757mm/0.00345894rad；Soup下一罐起点0.82989416mm/0.00271399rad。两者原15项handoff核查中14项通过、finish_terminal=false，完整handoff不通过。独立原生位置谓词/无持物/无手接触/机器人到位五连通过，并不替代最终视觉或总体物理成功。

wine_bowl_stove 原动作事件代理：{"on|wine_bottle_1|wine_rack_1_top_region": {"first_true_step": 131, "true_count": 5, "last": true, "sample_count": 150}}；completion_ready {"first_true_step": 150, "true_count": 1, "last": true, "sample_count": 150}
实测最大joint误差/速度=0.003223464419963906/0.005807712812297226，finger误差/速度=1.8999741280209537e-06/1.1109189870434826e-06；队列=0，proprioNone=True，controller.use_delta=False，OSC position/orientation goal误差=0.0/0.0，夹爪内部保持命令实测误差=0.0。下一模型真实输入重建尚未发生。
同步原检查（累计VLA、位置谓词、completion_ready、visual）：[{"cumulative_vla": 150, "declared_predicates": {"on|wine_bottle_1|wine_rack_1_top_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": true, "reason": "wrist和sideview视角均显示酒瓶已进入架子内部，位于木质层板上"}}, {"cumulative_vla": 150, "declared_predicates": {"on|wine_bottle_1|wine_rack_1_top_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "sideview显示酒瓶未在架子上，架子空置且酒瓶不在目的地内"}}]

basket_two 原动作事件代理：{"in|alphabet_soup_1|basket_1_contain_region": {"first_true_step": 245, "true_count": 506, "last": true, "sample_count": 750}}；completion_ready {"first_true_step": 257, "true_count": 494, "last": true, "sample_count": 750}
实测最大joint误差/速度=0.0034402225202155856/0.0056082910056190404，finger误差/速度=5.4087760189468204e-05/4.690157089055851e-05；队列=0，proprioNone=True，controller.use_delta=False，OSC position/orientation goal误差=0.0/0.0，夹爪内部保持命令实测误差=0.0。下一模型真实输入重建尚未发生。
同步原检查（累计VLA、位置谓词、completion_ready、visual）：[{"cumulative_vla": 150, "declared_predicates": {"in|alphabet_soup_1|basket_1_contain_region": false}, "completion_ready": false, "visual_goal_verdict": {"goal_achieved": false, "reason": "agentview明确显示字母汤罐在篮子外的桌面上，未进入篮子内部"}}, {"cumulative_vla": 300, "declared_predicates": {"in|alphabet_soup_1|basket_1_contain_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "agentview显示字母汤罐（橙色盒子）仍在桌面上，未进入篮子内部；wrist view中篮子内部物体身份无法确认为字母汤罐，且agentview明确显示指定物品在目的地外"}}, {"cumulative_vla": 450, "declared_predicates": {"in|alphabet_soup_1|basket_1_contain_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "agentview明确显示字母汤罐（橙色盒子）在桌面上，未进入篮子内部；wrist view的篮子内部视觉可能因视角遮挡误判，但agentview作为外部视角确认篮子为空"}}, {"cumulative_vla": 600, "declared_predicates": {"in|alphabet_soup_1|basket_1_contain_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "字母汤罐（黄色盒子）在agentview中位于篮子外的桌面，未进入篮子内部；wrist view显示篮子内有深色物体但非指定字母汤罐，且篮子入口外有其他物品。"}}, {"cumulative_vla": 750, "declared_predicates": {"in|alphabet_soup_1|basket_1_contain_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": true, "reason": "wrist view显示字母汤罐（深色圆形物体）完全位于篮子内部；agentview确认篮子内有物体且无其他物品替代，符合放进容器要求"}}, {"cumulative_vla": 750, "declared_predicates": {"in|alphabet_soup_1|basket_1_contain_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "字母汤罐（橙色罐子）在agentview中位于桌子中央，未进入篮子内部；篮子在右下角，罐子与篮子有明显距离，二维投影未重叠。"}}]

mugs_two 原动作事件代理：{"on|porcelain_mug_1|plate_1": {"first_true_step": 86, "true_count": 714, "last": true, "sample_count": 800}}；completion_ready {"first_true_step": 94, "true_count": 703, "last": true, "sample_count": 800}
同步原检查（累计VLA、位置谓词、completion_ready、visual）：[{"cumulative_vla": 150, "declared_predicates": {"on|porcelain_mug_1|plate_1": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "wrist view显示白杯（红白条纹）在盘子上，但agentview显示白杯（红色杯子）未在任何盘子上，仅位于桌面；指定目标要求白杯必须在左边盘子内部，当前画面明确显示白杯未进入盘子内部。"}}, {"cumulative_vla": 300, "declared_predicates": {"on|porcelain_mug_1|plate_1": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "白杯（红白条纹杯）在agentview中位于桌面，未进入左边盘子内部；左边盘子为空，指定物品未达目的地"}}, {"cumulative_vla": 450, "declared_predicates": {"on|porcelain_mug_1|plate_1": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "wrist视角显示白杯（红条纹）未进入左盘内部，仍在盘外；agentview视角中左盘未见白杯，且白杯位置与左盘无明确接触关系"}}, {"cumulative_vla": 600, "declared_predicates": {"on|porcelain_mug_1|plate_1": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "wrist视角显示左边盘子（红白条纹）空置，agentview视角未见白杯在左边盘子上；指定物品白杯未进入目的地内部"}}, {"cumulative_vla": 750, "declared_predicates": {"on|porcelain_mug_1|plate_1": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "白杯（带红条纹）未在左边盘子内部；agentview显示白杯位于桌面，左边盘子未见白杯，且wrist视角中白杯未与盘子接触"}}]

microwave_mug 原动作事件代理：{"in|white_yellow_mug_1|microwave_1_heating_region": {"first_true_step": 142, "true_count": 659, "last": true, "sample_count": 800}}；completion_ready {"first_true_step": 154, "true_count": 576, "last": true, "sample_count": 800}
同步原检查（累计VLA、位置谓词、completion_ready、visual）：[{"cumulative_vla": 150, "declared_predicates": {"in|white_yellow_mug_1|microwave_1_heating_region": true}, "completion_ready": false, "visual_goal_verdict": {"goal_achieved": false, "reason": "黄白杯在微波炉外部，未进入内部（agentview显示杯子位于微波炉旁桌面）"}}, {"cumulative_vla": 300, "declared_predicates": {"in|white_yellow_mug_1|microwave_1_heating_region": true}, "completion_ready": false, "visual_goal_verdict": {"goal_achieved": false, "reason": "黄白杯在agentview中位于微波炉外部桌面，未进入微波炉内部；wrist view显示杯子仍在夹爪中，确认未放置到位"}}, {"cumulative_vla": 450, "declared_predicates": {"in|white_yellow_mug_1|microwave_1_heating_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "黄白杯在agentview中位于微波炉外的台面上，未进入微波炉内部；wrist view显示夹爪附近有杯但未确认在炉内，且微波炉门未关闭。"}}, {"cumulative_vla": 600, "declared_predicates": {"in|white_yellow_mug_1|microwave_1_heating_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "agentview显示黄白杯位于微波炉外部台面上，未进入微波炉内部；wrist view未见杯在微波炉内，符合未达目标状态。"}}, {"cumulative_vla": 750, "declared_predicates": {"in|white_yellow_mug_1|microwave_1_heating_region": true}, "completion_ready": true, "visual_goal_verdict": {"goal_achieved": false, "reason": "黄白杯在agentview中位于微波炉外部地面，未进入微波炉内部；wrist view未显示杯体，但agentview明确显示杯在目的地外"}}]

白杯step450位置on谓词/完成门禁为true但视觉false；入炉step150位置in谓词true但completion_ready=false，不能称完整语义到位。四组末位置谓词true和整链失败同时保留。酒瓶、汤罐末视觉否决与独立证据冲突，等待原图判断。

旧control/sensors/feedback_retreat/joint_home/safe_exit源码SHA保持在冻结manifest内；新mode bridge因v2真实relative-only入口与absolute冲突而必要，仅桥接模式，不改旧算法/保护。唯一依赖新增h5py3.10，所有旧包版本保持。X-VLA FP32沿原作者30chunk/domain3/denoise10/absolute协议、20Hz；下一reference来自settling后真实初态，只改机器人返回目标而保留场景物体。8769入口为opt-in，不改默认Smol服务。

两个offline checker对照各false/true/false，各3真实Qwen、物理0/X0，未应用在线源码，不能宣称检查器已修复。

真实新增用量（含启动阻断）：{"new_actions": {"settling": 60, "vla": 2505, "finish": 476, "prepare": 56}, "physical_actions_total": 3097, "xvla_calls": 85, "hermes_invocations": 6, "online_qwen_calls": 18, "online_qwen_tokens": 40283, "offline_qwen_calls": 6, "hermes_token_usage": null, "offline_qwen_tokens": 12747, "all_qwen_tokens": 53030}。Hermes tokens/API usage不可用，保持unknown；权重无训练、无抓取/运输辅助和回放；模型随机流未固定，非配对比较。软件只引用原输出：backend10passed，pipeline旧28passed、最终independent41passed；不把检查数量当独立物理样本。

实际实现/执行代理gpt-6.1-sol low，依据实际spawn/dispatch设置；主规划宿主未公开实际model/effort，标unknown。

证据链接：
[summary](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/summary.json)
[summarize](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/summarize.py)

[v4/wine_bowl_stove result](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/result.json)
[pilot](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/pilot_result.json)
[rollout.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/rollout.mp4)
[independent_finish.jsonl](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/independent_finish.jsonl)
[events.jsonl](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/events.jsonl)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/finish/sensors/cap000263_wrist.png)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/finish/sensors/cap000264_sideview.png)
[原request/response finish/model_requests](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/finish/model_requests)
[原request/response xvla_calls](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v4/wine_bowl_stove/runs/20261012T135129Z_goal_table_4359e1a2/cap_01_wine_to_rack_c51276c4/xvla_calls)

[v5/basket_two result](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/result.json)
[pilot](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/pilot_result.json)
[rollout.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/rollout.mp4)
[independent_finish.jsonl](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/independent_finish.jsonl)
[events.jsonl](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/events.jsonl)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/finish/sensors/cap000647_wrist.png)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/finish/sensors/cap000646_agentview.png)
[原request/response finish/model_requests](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/finish/model_requests)
[原request/response xvla_calls](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/basket_two/runs/20261012T141538Z_basket_two_b5fdc461/cap_01_soup_to_basket_c2e42f33/xvla_calls)

[v5/mugs_two result](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/result.json)
[pilot](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a/pilot_result.json)
[rollout.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a/rollout.mp4)
[events.jsonl](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a/events.jsonl)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a/finish/sensors/cap000464_wrist.png)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a/finish/sensors/cap000463_agentview.png)
[原request/response finish/model_requests](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a/finish/model_requests)
[原request/response xvla_calls](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/mugs_two/runs/20261012T142339Z_mugs_two_56230ab5/cap_01_white_mug_left_d507c75a/xvla_calls)

[v5/microwave_mug result](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/result.json)
[pilot](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f/pilot_result.json)
[rollout.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f/rollout.mp4)
[events.jsonl](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f/events.jsonl)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f/finish/sensors/cap000464_wrist.png)
[最终实际checker图片](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f/finish/sensors/cap000463_agentview.png)
[原request/response finish/model_requests](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f/finish/model_requests)
[原request/response xvla_calls](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v5/microwave_mug/runs/20261012T142927Z_microwave_mug_0959aa4a/cap_01_mug_to_microwave_b778a47f/xvla_calls)

[v2/wine_bowl_stove result](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v2/wine_bowl_stove/result.json)
[pilot](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v2/wine_bowl_stove/runs/20261012T133345Z_goal_table_ec481cc8/cap_01_wine_to_rack_7194c91b/pilot_result.json)

[v3/wine_bowl_stove result](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v3/wine_bowl_stove/result.json)
[pilot](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v3/wine_bowl_stove/runs/20261012T134308Z_goal_table_4d9462a8/cap_01_wine_to_rack_197381cc/pilot_result.json)
[rollout.mp4](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v3/wine_bowl_stove/runs/20261012T134308Z_goal_table_4d9462a8/cap_01_wine_to_rack_197381cc/rollout.mp4)
[events.jsonl](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v3/wine_bowl_stove/runs/20261012T134308Z_goal_table_4d9462a8/cap_01_wine_to_rack_197381cc/events.jsonl)
[原request/response xvla_calls](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/v3/wine_bowl_stove/runs/20261012T134308Z_goal_table_4d9462a8/cap_01_wine_to_rack_197381cc/xvla_calls)
[原proposal 20261012-xvla-persistent-independent-v5.json](D:/FYP/First_Phase/scene_demo/plans/20261012-xvla-persistent-independent-v5.json)
[原proposal 20261012-xvla-persistent-pilot-v2.json](D:/FYP/First_Phase/scene_demo/plans/20261012-xvla-persistent-pilot-v2.json)
[原proposal 20261012-xvla-persistent-pilot-v3.json](D:/FYP/First_Phase/scene_demo/plans/20261012-xvla-persistent-pilot-v3.json)
[原proposal 20261012-xvla-persistent-pilot-v4.json](D:/FYP/First_Phase/scene_demo/plans/20261012-xvla-persistent-pilot-v4.json)
[原proposal 20261012-xvla-persistent-pilot.json](D:/FYP/First_Phase/scene_demo/plans/20261012-xvla-persistent-pilot.json)
[offline checker_agentview](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/checker_agentview/result.json)
[offline checker_rack_description](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/checker_rack_description/result.json)

下一步分支建议：先修检查器静态物品身份、on/in关系和终态视角覆盖，用保存的正负例验收；保持已验证收尾复位实现；随后再做一次首链。本次不执行新试验。
