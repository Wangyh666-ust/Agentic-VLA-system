import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_bytes())
audit=load(OUT/'audit.json');second=load(OUT/'attempt_02/summary.json');tail=load(OUT/'continuation/summary.json')
assert audit['real_Qwen_requests']==32 and tail['final']['goal'] is True
basket=second['cases']['basket_two']['records'][-1];assert basket['goal'] is True
findings=[
 '已确认放置记录不再因未知、遮挡、干扰物或无关开爪被删除；只有同一目标、明确移出/移动/掉落及当前反证图可撤销。未充分支持的同物outside冲突保留记录、输出Unknown等待核实。',
 '所有10个原生长程共用关系类型规则；门/抽屉/开关和preserved条件只比较最新视觉事实，不继承历史状态。同期同图不重复请求状态票。',
 '入炉末帧：当前杯不可见，goal_0仍True且引用release_0001，goal_1按当前门图True，完整目标True；已实际触发并执行记忆复用。',
 '入炉开门负例：第二轮150步正确False，164步首次松手却把开门判closed，450/750步又把实际已关判open；门状态视觉判断仍不稳定，负例验收失败。松手快照整体True不等于线上已移交，当前轮未运行环境。',
 '入篮第二轮：首次松手汤罐被描述为桌上橙色小盒，未建立goal_0证据；第二次松手确认goal_1番茄酱罐并保存。最终两目标True，但只有番茄酱引用过程证据；未达到两罐均由松手证据确认的预登记判据。',
 '入篮中途仍把桌上其他物品当成番茄酱罐：程序保留原证据，对缺明确变化证据的同物outside冲突输出Unknown，未误撤销。',
 '单罐负例整体False、无历史遮挡杯Unknown；这些负例结果不能抵销门开误报或缺失汤罐过程确认。',
 '两处协议边界已修：状态项多余relation不覆盖observed_state；identity不确定的无依据outside不能中断整帧或撤销证据。最后补测严格复用了第二轮7条原响应，每条请求body SHA（图片/新中间状态/提示）相同，只新增末帧1次真实请求。',
 '共32次真实Qwen请求，所有实际发送PNG原字节、帧标签和body SHA匹配；没有将仿真目标真值发给模型。过程信息的传输与保存已验证，但视觉整体效果未通过。']
limits=[
 '全部10技能仅有软件规则与真实服务接口测试；真实Qwen旧图回放只有两类任务，不能声称10任务实测成功。',
 '原图256x256；这不是新增降采样。初始参考时刻未知标-1；松手前后仅复用1秒内实际有记录的帧，旧图缺密集窗口，不能补造连续动作。',
 '32为累计实际请求数，不是独立任务样本数或成功率；7条精确原响应重放不另计请求。两个提示版本有限图片，不把不同版本最好输出拼成通过。',
 '零新仿真/实体动作、零VLA/Hermes请求、零辅助器/子agent。默认检查器未替换，未启动收尾/退离/复位，不评价这些下游模块的本轮效果。',
 '主Agent实际型号/推理档位无法从本地调用证据确认，记录unknown；没有冒称按AGENTS默认模型派发。']
summary={'id':'20261013-xvla-evidence-memory','title':'确认放置证据保存/撤销修复：真实旧图回放仍未完整达标',
 'status':'通用规则已修、真实遮挡复用已触发；两项末帧True，但开门负例错误与汤罐过程确认缺失，离线资格未通过。',
 'qualification':False,'default_enabled':False,'software':audit['software'],
 'actual_calls':{'Qwen':32,'VLA':0,'Hermes':0,'subagents':0},'environment_actions':0,
 'final_cases':{'microwave_mug':tail['final'],'basket_two':basket},
 'attempts':[{'id':'attempt_01','requests':16,'original_complete':False,'original_passed':False},
             {'id':'attempt_02','requests':15,'original_complete':False,'original_passed':False},
             {'id':'continuation','requests':1,'reused_exact_v5_responses':7,'complete':True,'passed':False}],
 'findings':findings,'limitations':limits,
 'next_gap':'在同一套已保存照片上诊断目标定位/身份关联与门板位置观察；不能继续放宽完成条件或绕过负例。',
 'preserved':'finish_v1全py文件、safe_exit/joint_home/placement_completion等哈希与本轮前一致；service仅更换显式候选检查器入口，pilot仅加入源文件冻结/资格校验。'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text='''已按规则实现通用候选检查器，并完成真实 Qwen 旧图回放。**保存/撤销机制已生效，整体视觉资格仍未通过，候选未替换默认检查器。**

| 样例 | 末帧结果 | 过程证据及剩余问题 |
|---|---|---|
| 黄白杯入炉再关门 | True | 杯子不可见时仍引用首次松手 release_0001；但首次松手时门仍开，模型误报关闭，负例不通过 |
| 双罐入篮 | True | 番茄酱罐引用 release_0002；汤罐首次松手被错认成桌上盒子，缺少已确认的松手证据 |
| 单罐完成负例 | False | 整体没有放行；不能由此证明每个物品都识别正确 |
| 入炉末帧，无过程历史 | Unknown | 没有凭遮挡或任务描述推断杯子已入炉 |

本轮修复覆盖全部 **10 个原生长程技能**，按放置关系、装置状态及 preserved 条件统一处理。确认放置后，遮挡、身份未知、干扰物及无关开爪不会擦掉证据；撤销必须有同一实物和当前移出/移动/掉落反证。没有完整变化依据的同物 outside 冲突保留证据并输出 Unknown，避免直接放行或误删除。门/抽屉/开关只取最新视觉状态。

原图、同步编码器、目标定义及本次新中间状态继续传入检查器；历史照片不能变成当前视觉事实。松手前后窗口限定在 1 秒内，缺图即缺图。同期同图复用观察，避免对同一捕获再请求相反的状态判断。

协议中两处已修：门状态不再受无意义的 placement 字段干扰；身份不确定且没有当前依据的 outside 标签不再中断整帧或撤销证据。明确 matched 事实仍需当前图，越界帧和错误状态类型仍拒绝。

验证结果：319 tests、30 subtests 通过。真实 Qwen 共 **32 次**：首轮 16、修订轮 15、补完末帧 1。补完前按时间顺序重放修订轮已有 7 条原始响应，每条实际请求 body 的 SHA 都相同（包括图像、提示和新中间状态），没有重新采样或挑选较好回答，也没有灌入旧 v3 判断。原始两次解析失败与全部失败响应均保留。

当前问题已有区别：入炉的放置证据确实保存且能在遮挡后复用；剩余是模型对炉门开闭的观察错误。入篮第二轮建立了番茄酱罐证据，但汤罐首次松手被错认成小盒；中途还把桌上的干扰物当成罐。最后一帧 True 不能替代“松手确认及后续复用完整”的本轮验收目标。

原图为 256×256，32 次请求的 PNG 原字节、标签和请求 SHA 均核对通过；仿真目标真值只用于离线评分，没有发给视觉模型。现有旧图缺少部分紧邻松手的照片，无法据此评价新采样窗口在线效果。

原收尾、退离、下一子任务起始位/姿态与策略状态复位实现保留。本轮零新增仿真/实体动作、零 VLA/Hermes 请求、零辅助器、零子agent。只测试了两类旧图，不能报告所有长程的实际成功率。默认仍使用原检查器，新候选需继续通过视觉负例与完整过程证据验收。

证据：[代码快照清单](source_manifest.json)、[请求原图审计](audit.json)、[末帧补完及7条同源重放](continuation/summary.json)、[第二轮原始结果](attempt_02/summary.json)、[软件测试](tests-final.stdout)、[结构化报告](summary.json)。
'''
(OUT/'summary.md').write_text(text,encoding='utf-8')
print(json.dumps({'status':summary['status'],'actual_Qwen_requests':32},ensure_ascii=False))
