# 实验历史索引（截至2026-10-11）

本索引截至2026-10-11，共28个历史campaign。每行是问题/证据导航，不是一个独立统计样本；记录中的回放、复用、未启动和原始判据差异保留。最近一次扩大组合测试为前12例；20例目录名只保留原注册历史。环境是LIBERO/robosuite/MuJoCo + Franka Panda，非ManiSkill。早期开发采用GPT规划/验收、DeepSeek实现；当前分工遵循最新授权，本轮未开子agent。运行期Hermes/Qwen3-VL与SmolVLA不等于开发分工。

[结构化台账](experiment_registry.json)记录各条适用条件、结论、限制和原始证据哈希。

[后续计划](plans/2026-10-09-evidence-first-next.md) · [Skill验收](review_checks/2026-10-09/ACCEPTANCE.md)

| ID | 实验问题 | 实际结果 | 证据链接 |
| --- | --- | --- | --- |
| 20261006-persistent-pilot | 持久场景初版与Hermes意图/容量试点 | 碗95步曾报告原生成功，后酒瓶300失败并修复再失败；桌面完整指令600失败，篮子/杯子物理试点失败；缺垃圾桶/容量满的正确拒绝是决策成功，不是物理成功 | [报告](results/2026-10-06/README.md) · [台账](experiment_registry.json) |
| 20261007-placement | 停止判据与篮子完整/拆分指令对照 | 碗在旧95/83步停点未达到释放稳定，新102/104步达标，动作前缀完全相同；完整篮子指令中汤罐第212步稳定放置，但全任务失败；原子汤罐300、600、300+300均失败，番茄酱原子失败 | [报告](results/2026-10-07/README.md) · [台账](experiment_registry.json) |
| 20261007-wine-stages | 原生/共享酒瓶抓取与放置六次诊断 | 六次：1次严格成功、2次抓取阶段失败、3次已释放接触架但目标区外；native0:0在178动作完成 | [报告](results/2026-10-07-wine/README.md) · [台账](experiment_registry.json) |
| 20261008-grasp-config | 酒瓶抓取守卫与三推理配置配对 | 默认baseline严格2/6、评估后语义4/6；fp32严格2/6，fp32_h5严格1/6；默认留出3/3严格成功，默认未变；守卫能更早停止部分空手撤退，不提升技能 | [报告](results/2026-10-08-grasp-config/README.md) · [台账](experiment_registry.json) |
| 20261008-skill-context | Hermes规划与酒瓶起始条件六次对照 | 原生酒瓶2/2、共享初始1/2、放碗后0/2；Hermes五次规划检查四次正确提交，另一项没有实际提交；准备分支阻断，VLA未启动 | [报告](results/2026-10-08-skill-context/README.md) · [台账](experiment_registry.json) |
| 20261008-wine-start-pose | 放碗后酒瓶：直接执行与夹爪位姿准备 | 直接0/2、准备0/2，均无稳定抓取确认；准备位姿误差2.03mm/0.535度、空手与碗保护通过 | [报告](results/2026-10-08-wine-start-pose/README.md) · [台账](experiment_registry.json) |
| 20261008-local-grasp | 顶部抓取辅助标定失败 | 下降超时，离抓取目标约3.3cm，未进入闭合；五条组合请求全部未运行 | [报告](results/2026-10-08-local-grasp/README.md) · [台账](experiment_registry.json) |
| 20261009-side-grasp | 侧向抓取五条组合请求试点 | 全链1/5，两个酒瓶完成搬运与释放；三例辅助接近时偏转超限，另一例先瓶后碗拾取失败 | [报告](results/2026-10-09-side-grasp/README.md) · [台账](experiment_registry.json) |
| 20261009-collision-approach | 同失败状态旧路线/v1/v2避碰抓取对照 | 旧路线实际推碰瓶身20.18度；v1超时；v2用191辅助动作抓起2.73cm；本轮未测运输/置架 | [报告](results/2026-10-09-collision-approach/README.md) · [台账](experiment_registry.json) |
| 20261009-collision-chain | 避碰抓取后放架与两步组合五例 | 酒瓶5/5按原生释放稳定完成，全链3/5；两例酒瓶后碗500动作未抓起；单旧状态经191辅助+82新VLA完成置架 | [报告](results/2026-10-09-collision-chain/README.md) · [台账](experiment_registry.json) |
| 20261009-subtask-assist | 统一子任务辅助入口四分支对照 | 关闭各500新VLA失败；开启143/144准备朝向超时，VLA0；统一入口已接通，但没有有效准备后VLA样本 | [报告](results/2026-10-09-subtask-assist/README.md) · [台账](experiment_registry.json) |
| 20261009-assist-repair | 碗上方接近准备修正两例 | 两例准备达标，但都未抓起碗；位置误差2.70/2.63mm，酒瓶目标保留 | [报告](results/2026-10-09-subtask-assist-repair/README.md) · [台账](experiment_registry.json) |
| 20261009-entry-compatibility | 参考轨迹入口兼容性四例 | 原轨迹重放通过；原第30步接续61新VLA通过；原生辅助准备165后59新VLA通过；酒瓶后224准备朝向超时，VLA0 | [报告](results/2026-10-09-entry-compatibility/README.md) · [台账](experiment_registry.json) |
| 20261009-joint-home | 两旧酒瓶后状态完整关节归位再抓碗 | 243/241归位后，93/105新VLA放碗完成，两例成功；旧直接/碗上方准备失败背景保留 | [报告](results/2026-10-09-joint-home/README.md) · [台账](experiment_registry.json) |
| 20261009-joint-home-12 | 关节归位六种组合×两条件，前12例 | 全链3/12；首技能失败4、归位入口受阻3、归位后酒瓶失败2；实际17VLA：碗5/5、酒瓶2/4、灶台4/4、汤0/2、番茄酱0/2；5次归位达标后接续3成功；六个VLA失败目标谓词从未真 | [报告](results/2026-10-09-joint-home-20/README.md) · [台账](experiment_registry.json) |
| 20261009-safe-exit-3 | 保持实际夹爪开口→安全退离→严格归位 | case02接口异常unknown，回放180/退离0/归位0；case07/08未启动；接口修正软件35通过，尚无有效归位结果 | [报告](results/2026-10-09-safe-exit/README.md) · [台账](experiment_registry.json) |
| 20261010-safe-exit-sdk-3 | SDK修正后三旧终态退离实测 | 完整退离并归位0/3；退离80/14/9、home0；酒瓶预算耗尽仅3/5确认，灶台新浅接触阻断；没有新VLA | [报告](results/2026-10-10-safe-exit-sdk/README.md) · [台账](experiment_registry.json) |

| 20261010-visual-finish | 视觉检查与通用收尾五旧样例 | 子任务完整交接0/3：碗home360未归位，酒瓶直接home路线阻断，灶台开火误判开盖；负例2/2零辅助（汤罐未完成、番茄酱unknown），release/movement/confirm均0 | [报告](results/2026-10-10-visual-finish/README.md) · [台账](experiment_registry.json) |

| 20261010-finish-takeover-v2 | 接管、实际松手与受保护退离定点修正 | 最终原碗状态完整交接0/1，11开口/60退离/确认0/归位0，退离终点清场阻断；余四暂停，三个源码版本不合并可靠性 | [报告](results/2026-10-10-finish-takeover-v2/README.md) · [台账](experiment_registry.json) |

| 20261010-shape-home | 收尾与安全归位定点验证：归位1/1，最终视觉误判，全链0/1 | 11开口周期/89退离/5确认/137归位，五真实post达标且目标保持；最终Qwen误判，原始blocked；前三版home0，非归位执行失败 | [报告](results/2026-10-10-shape-home/README.md) · [台账](experiment_registry.json) |
| 20261010-vision-grounding | 保存画面校准：碗盘识别修正，松手退离未验收 | 最终保留版两种碗状态融合判定正确（持物raw unknown）；酒瓶raw/fused错误complete，指尖候选使正例退化后撤回；9Qwen/零物理，余3未运行，自动视觉收尾默认off | [报告](results/2026-10-10-vision-grounding/README.md) |

| 20261010-joint-verifier | 同步编码器与视觉保存状态检查 | 不同修订选定四状态判对、一未知；无物理收尾，不等于同版完整流程通过 | [报告](results/2026-10-10-joint-verifier/README.md) |
| 20261010-live-handoff | 当前检查器实际接管 | 酒瓶未触发；灶台允许后闭爪等待10次阻断，实际开爪/归位0 | [报告](results/2026-10-10-live-handoff/README.md) |
| 20261010-simple-checker | 用户两问检查器四项独立实测 | 酒瓶/碗误判未触发；灶台实际开爪17、退离120、归位0；番茄未入篮拒绝。正例完整归位0/3 | [报告与四视频](results/2026-10-10-simple-checker/README.md) |

## 当前结论

- 当前权重有抓取/释放能力，原生酒瓶、碗以及部分组合有成功证据；能力存在不等于可靠性足够。
- 原生酒瓶基线、三推理配置、篮子完整/拆分、夹爪位姿准备和完整关节归位已做，优先复用，不能重列为未做。
- 辅助器准备失败、归位入口受阻与真正VLA失败应分开。20261010-safe-exit-sdk-3的三个旧终态当时均在退离阶段阻断、归位未启动；最新20261010-shape-home已在原碗终态完成退离和归位，酒瓶／灶台等其余终态尚未验证新修正。
- 碗/灶台在最新受测条件中较稳定；酒瓶对起点/前序条件敏感，汤罐抓住后未入篮，番茄酱未确认抓取；仍未确定训练因果。
- 酒瓶四例的离线差异分析已完成；两路策略输入和子任务随机状态仍缺记录，不能只归因于姿态或训练。当前先校准保存终态图像的最终验收；不重复已有原生基线，不自动微调、扩大或续跑已中止20例。
- 20261009-safe-exit-3 因SDK字段错误在case02中止，退离/归位均0；20261010-safe-exit-sdk-3完成三个旧终态实测，但退离80/14/9均阻断、归位0，不能宣称修复有效。

- 20261010-visual-finish 的三个受测子任务收尾到可交接状态0/3：碗视觉误报已松手/退离且归位未完成，酒瓶与架子接近造成直接归位路线阻断，灶台开火被误解为开盖。两个负例没有辅助动作，其中番茄酱只是unknown保守拒绝；新VLA/Hermes均0，不能当完整多任务链成功率。先校准存帧检查器与归位流程，暂不扩大、不微调。

- 20261010-finish-takeover-v2 已真实执行开口与退离，但最终原碗样例完整子任务交接0/1，确认/归位均0，余四例暂停；这是退离终点清场阻断，不能称已执行归位失败或VLA训练缺陷。该退离误阻断的后续修正和单例实际归位结果见20261010-shape-home；其余四例仍暂停。

- 20261010-shape-home 最新原碗状态归位验证1/1，完整验收0/1：实际松手/退离/归位均执行，五真实post归位到位且碗在盘上；最终Qwen误判盘空，原结果blocked。GPT复核两路原图与独立物理记录一致。仅同一历史状态四个修订，不能合并可靠性；下一步只利用保存终态图像校准最终验收，暂不再跑归位、扩大样例或微调。

| 20261010-joint-verifier | 视觉与同步夹爪反馈 | 五保存状态4准确1unknown，十Qwen、多版本复用；整体未通过，UIoff，动作0 | [报告](results/2026-10-10-joint-verifier/README.md) |

当前联合检查：旧酒瓶无编码器视觉误判历史保留；当前同步融合及动态细节已使四保存状态准确，番茄酱身份仍unknown，整体未通过，自动收尾off。

| 20261010-live-handoff | 实际接管小试 | wine未触发，stove许可后10闭爪等待阻断，actualopen/move/confirm/home0；整体未通过，两chain未运行 | [报告](results/2026-10-10-live-handoff/README.md) |

## 2026-10-11 最新修复

[酒瓶/灶台收尾与归位实测](results/2026-10-11-goal-finish/README.md)：以逐阶段动作和独立效果记录为准；保留所有先前失败，不合并版本成功率。

## 2026-10-11 通用检查器组合首步

[本轮报告与录像](results/2026-10-11-generic-combos/README.md)：碗第60步误报到位并提前松手；实际归位完成，碗未在盘上，独立验收阻断酒瓶；碗→灶台依共同首步骤失败停止，未启动。通用检查器仍有假阳性，不能宣称组合跑通。

## 2026-10-11 双视角与无阶段标签负例门禁

[报告、实际图片与来源录像](results/2026-10-11-two-view-combos/README.md)：两组保存负例依序各一次Qwen；松手前回答未到位，归位后碗在盘旁却回答到位，整体未通过。4031 tokens、零新VLA/辅助/回放，两个组合均未启动；未单独测试归位，未改词重试、未开子agent。下一步先修物体位置关系识别，不以延后到150/300或归位成功代替验收。

## 2026-10-11 150/300步真实在线两组合

[本次新录像、原始回答与完整报告](results/2026-10-11-online-150-300-combos/README.md)：两个组合均实际进入第二任务，完整流程0/2。两次碗任务各150新VLA后完成开爪、退离、归位，独立连续5次通过。酒瓶仍在桌面，Qwen误报放架，虽归位但独立验收失败；灶台已开，检查器触发收尾，但一指仍接触灶台，开爪20周期未达到阈值，退离/归位0。新VLA600、辅助785、Qwen7次/13758tokens、回放0、子agent0。原归位不变，未改词重试、不微调；此次直接在线试验由用户明确要求，旧保存负例失败仍保留，不能称两组合已跑通或仅归因训练。

## 2026-10-11 酒瓶预抓辅助与灶台同时开爪退离

[本次报告、两段新录像与原始记录](results/2026-10-11-assisted-combos/README.md)：完整组合0/2。酒瓶抓辅真实182动作、confirmed并抬升2.79厘米，VLA放架目标成立；放架后退离54周期停滞，归位0。灶台新开爪同时退离分支真实执行10周期，但中途剩余路径复查阻断，归位0。两组首碗均完成收尾和原归位，前碗目标保持。新VLA600、抓辅182、收尾576、Qwen7次/12667tokens、回放0、子agent0、不微调。只证明目标动作已发生和抓瓶改善，不能称完整多任务链或真机RGBD泛化成功；下一步针对退离路径及开爪协调，不重复原归位。

### 20261011-withdrawal-diagnosis：退离停滞与旧路径阻断

零新动作的原始证据诊断：酒瓶卡旧首目标，灶台原生碰撞形状复核显示回追旧目标会挤入保存深度点。原成功归位执行保持不变，修复尚未实施。见 [诊断与明细](results/2026-10-11-withdrawal-diagnosis/README.md)。

### 20261011-feedback-retreat：实测反馈退离两组合

退离2/2，第二任务归位1/2；酒瓶物理五连通过但末次视觉否决，灶台当前已退出但原回程预测碰旋钮，在线完整0/2。见[原始结果与视频](results/2026-10-11-feedback-retreat/README.md)。

- `20261011-stove-exit-repair`：首碗通过；灶台候选检查把当前接触姿态误插第0点，退离0/home0。[完整失败记录](results/2026-10-11-stove-exit-repair/README.md)。

- `20261011-stove-candidate-start`：假想回程起点已修，前碗实测归位/目标五连通过却遭末次视觉否决；灶台未启动，修订目标未验证。[本轮结论与三段视频](results/2026-10-11-stove-candidate-start/README.md)。

- `20261011-final-input-audit`：五份完整HTTP请求体与原记录匹配，实际图片和同步编码器正确、没有阶段标签；酒瓶/前碗是归位后的视觉否决，灶台初测则是回程碰撞阻断。零真实模型/机械臂动作。[核查与原图、完整prompt](results/2026-10-11-final-input-audit/README.md)。

- `20261011-neutral-encoder-checker`：中性提示词与编码器主导开口判定已接入检查器；45项软件回归、五份旧记录离线融合通过，相同碗末态不再因视觉接触回答相反而改变结果。酒瓶旧目标否定保留；零新Qwen/机械臂动作，不宣称新组合物理成功。[修复、完整提示词与限制](results/2026-10-11-neutral-encoder-checker/README.md)。

- `20261011-neutral-live-combos`：新版两组合各一次，完整0/2；两次碗均收尾归位五连通过。酒瓶300步目标/开口通过，但左指超指令上限0.0187mm使收尾启动动作0；灶台150步已开、红圈可见，Qwen却返回null，收尾0。新VLA750/抓辅182/收尾482、Qwen8次12627tokens；七完成请求图片、同步编码器、实际v3提示词核对通过。无现场重跑、子agent、训练、单独归位。[视频与明确停止点](results/2026-10-11-neutral-live-combos/README.md)。

- `20261011-takeover-stove-cue`：限幅保持命令、补充静态红圈定义后，两组合各一次均完整通过；酒瓶同样0.0187mm实测越界状态实际接管、退离与归位成功；灶台识别红圈并触发同时开爪退离及原归位，前碗成果保留，独立五连均过。新VLA750/抓辅182/收尾900、Qwen10次16386tokens；无子agent、微调、独立归位或现场重跑。小范围2/2不等于总体可靠性。[两段新视频与明确结论](results/2026-10-11-takeover-stove-cue/README.md)。

- `20261011-container-pilot`：番茄酱与微波炉初测：相机覆盖不足、杯子入炉误判，关门未启动。[报告、视频与明确结论](results/2026-10-11-container-pilot/README.md)。
