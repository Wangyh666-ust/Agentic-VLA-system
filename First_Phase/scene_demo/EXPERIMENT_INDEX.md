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

- `20261011-container-view`：容器补测：正面检查避免杯子误松手，入篮/入炉仍未完成，关门未启动。[报告、视频与明确结论](results/2026-10-11-container-view/README.md)。

- `20261011-container-budget`：600步容器补测：未到位均false且继续执行；牛奶误抓/杯子炉外失败，收尾与关门未触发。[报告、视频与明确结论](results/2026-10-11-container-budget/README.md)。

- `20261011-native-benchmark`：原生40任务×5次完成，139/200=69.5%；Spatial34/50、Object44/50、Goal39/50、Long22/50。110次原结果保留，仅补90次，88证据哈希不变。无Hermes/Qwen/辅助或微调；非论文严格复现，不直接归因训练缺陷。[结论与代表视频](results/2026-10-11-native-benchmark/CONCLUSIONS.md)、[完整逐任务报告](results/2026-10-11-native-benchmark/README.md)。

- `20261012-xvla-native-benchmark`：X-VLA原生40任务×5次，197/200=98.5%；Spatial48/50、Object50/50、Goal50/50、Long49/50；1160模型调用、32184policy+2000settle、200原视频逐帧核验，三失败均budget_exhausted。作者公开权重/客户端协议下的缩小样本复现，取得相近量级表现；非严格论文复现或配对SmolVLA比较，持续场景拆分及归位接续尚未验证。离线归档零新增模型/物理动作，仅关闭本任务server。[结论与视频](results/2026-10-12-xvla-native-benchmark/CONCLUSIONS.md)、[完整报告](results/2026-10-12-xvla-native-benchmark/README.md)。

- 20261012-xvla-persistent-pilot：四整链未完成；酒瓶/汤罐实际复位但末视觉否决，白杯/入炉未触发收尾；下一策略/globalhome未验证。[报告](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-persistent-pilot/README.md)

- 20261012-xvla-checker-grounding：9次保存画面判定，3项未过、1项允许未知；候选未接在线。[结果](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-checker-grounding/result.json)
- 20261012-xvla-input-audit：33请求body/PNGbytes/标签/时间匹配，旧offline6项测值来源unknown，零真实请求。[审计](D:/FYP/First_Phase/scene_demo/results/2026-10-12-xvla-input-audit/audit.json)

- 20261013-xvla-native-three-view-software：[X-VLA原生长程优先与三视图软件资格](results/2026-10-13-xvla-native-three-view/summary.json)：软件通过，三技能+三视图；真实VLA/Qwen/Hermes调用与动作0，物理/视觉效果未实测。

- 20261013-xvla-all-native-software：[全10个原生长程接口与整体检查软件资格](results/2026-10-13-xvla-all-native/summary.json)：10/10技能接口、7scene/13候选拆分、完整条件与三图；1277 passed/106 subtests；真实VLA/Qwen/Hermes模型调用与动作0，实际效果待测。

- 20261013-xvla-process-evidence-software：[全10长程统一松手三视图过程证据软件资格](results/2026-10-13-xvla-process-evidence/summary.json)：全10共用、实测松手三图、已验证placement遮挡复用/反证撤销；1316 passed/106 subtests，真实模型/环境动作0，实际效果未验证。

- 20261013-xvla-release-trial：[入炉关门真实松手三视图首例](results/2026-10-13-xvla-release-trial/summary.json)：gate未通过/budget_exhausted；2事件采图与8请求协议通过，分类False且复用0，收尾未启动，入篮未启动。


### 20261013-xvla-timeline-software：通用对象身份与过程图/中间状态联合检查

- 软件资格：全部10 native统一输入时间线与身份门禁；278测试/34 subtests通过，原失败trial 9图与中间状态离线序列化核验。
- 真实模型/环境动作0，子agent0；原相机/收尾/退离/归位不变；实际复测及入篮未启动。
- [报告](results/2026-10-13-xvla-timeline-software/summary.md) · [软件方案](plans/20261013-xvla-timeline-software.json)


### 20261013-xvla-timeline-trial：新时间线两项独立实测

- 入炉关门与双罐入篮各一次，原生完整目标达到，视觉检查未放行；每项两次松手三图、末次9图与5次先前状态已传入，过程引用0。
- 原收尾未启动；无抓取/搬运辅助，无重试，子agent0。
- [报告](results/2026-10-13-xvla-timeline-trial/summary.md) · [独立入篮方案](plans/20261013-xvla-timeline-basket-independent.json)


### 20261013-xvla-event-checker：通用事件事实候选，离线未通过

- 三版共18次真实Qwen旧图请求；过程开始被引用，但最终仍有开门负例误放行及身份连续性错误。
- 192 tests / 30 subtests通过只代表程序资格；原收尾/退离/归位源未改，候选未替换默认检查器。
- 离线门禁未通过，入炉关门与双罐入篮均未新增实测；零VLA/环境动作、零子agent。
- [结构化报告](results/2026-10-13-xvla-event-checker/summary.json) · [请求原图审计](results/2026-10-13-xvla-event-checker/audit.json)


### 20261013-xvla-evidence-memory：确认放置证据保存/撤销修复：真实旧图回放仍未完整达标

- 通用规则已修、真实遮挡复用已触发；两项末帧True，但开门负例错误与汤罐过程确认缺失，离线资格未通过。
- 全10长程共用规则；319 tests / 30 subtests；原收尾、退离、归位未改。
- 32次真实Qwen旧图请求；零新增VLA/环境动作、零子agent，候选未替换默认检查器。
- [报告](results/2026-10-13-xvla-evidence-memory/summary.md) · [审计](results/2026-10-13-xvla-evidence-memory/audit.json)


### 20261013-xvla-evidence-recovery：全部松手原图与后续补建，视觉资格未通过

- 漏判/未知/未分类松手原图全部入后续请求；普通检查可按实际时刻补建，真实Qwen已触发并复用。
- 336 tests / 30 subtests覆盖全部10长程；原收尾、退离、复位等43个保护文件未改。
- 14次真实旧图Qwen：入炉300步通过、450步缺事实帧停止；入篮汤罐未确认，末帧身份冲突Unknown。离线未通过，零新增环境动作、辅助器、子agent。
- [报告](results/2026-10-13-xvla-evidence-recovery/summary.md) · [原图与请求审计](results/2026-10-13-xvla-evidence-recovery/audit.json)

### 待实施方案：同一子任务的初态、过程、末态联合审核

- 保存同段全部已采集原图及同步反馈，明确任务目标与干扰物的图像对应；允许重审旧松手事件，分别记录拍摄时间与审核时间。
- 方案阶段已通过历史证据结构检查；尚未实施、未新增Qwen或环境调用。拟先覆盖全10长程软件规则，再独立旧图验证入炉关门与双罐入篮，保护原收尾和复位。
- [结构化方案](plans/20261013-xvla-subtask-evidence-review.json)


### 20261013-xvla-subtask-evidence：同段原图重审，关键汤罐补证仍未通过

- 全10长程通用实现：图像时间/稳定编号/同步反馈、身份参照、旧图重审与意见版本；361 tests、30 subtests。
- 22次旧图Qwen尝试（21响应、1网络失败），无新动作/辅助器/子agent。入炉旧事件补证触发但框与可见性有误；汤罐仍只得当前证明，未引用第一次松手补证。
- 整体离线资格未通过，候选未切换默认；原收尾、退离、下一子任务复位43个保护文件及monitor保持。
- [报告与原图](results/2026-10-13-xvla-subtask-evidence/summary.md) · [请求审计](results/2026-10-13-xvla-subtask-evidence/audit.json)


## 2026-10-14：三视角规划身份在线试点

- 实验 `20261013-xvla-grounded-online`：[报告与原图](results/2026-10-13-xvla-grounded-online/summary.md)。
- 入炉黄杯初态识别正确，但150步提前误判关门；双罐初态错认被复核沿用，87步人工停止。两项均未通过完整流程，历史补证未验证。
- 相机只读接口启动故障已修并零步核验；334 tests/4 subtests。原收尾/退离/复位控制保留，无辅助器/子agent。


## 2026-10-14：Qwen3.8 初始识别对照

- 实验 `20261014-qwen38-initial-recognition`：[报告](results/2026-10-14-qwen38-initial-recognition/summary.md)。
- 同图同提示词仅换模型，两次真实识别、零机器人动作。双罐两外部正确但腕部对应仍错；黄杯三视角目标身份正确。
- X-VLA Hermes与初识别默认切到qwen3.8-max；末端检查、收尾和复位保留。15项已有接口测试通过。


## 2026-10-14：Qwen3.8三条自然语言任务

- 实验 `20261014-qwen38-three-requests`：[报告与原图](results/2026-10-14-qwen38-three-requests/summary.md)。
- 整理碗、酒瓶并开灶整链通过，两次下一起点复位与末次home真实执行。
- 加热被误匹配为入炉关门；原生AND256至900为真，独立释放/静止只4连而整体阻断。丢弃初识别网络失败，未进入规划，不记作正确拒绝。
- 当前流程未现场修改，无新增抓取搬运辅助器/子agent；完整请求、过程原图与失败记录保留。


## 2026-10-14：部分执行与能力边界的旧图规划

- 实验 `20261014-partial-scope-offline`：[报告](results/2026-10-14-partial-scope-offline/summary.md)。
- 三条旧图规划通过：整理full；黄杯入炉关门partial且明确不能启动加热；丢弃unsupported且不擅自入篮。在线未启动。
- 更正前项“加热误匹配”的表述：选择加热准备技能合理，旧问题是漏说能力缺口；物理失败记录仍保留。
- 双罐腕图身份框仍交叉，不能宣称操控视觉验收通过；本轮第三项仅能力边界判断。


## 2026-10-14：X-VLA交互工作台（软件验证）

- 记录 `20261014-xvla-console-software`：[使用说明](XVLA_CONSOLE.md)，[验证记录](results/2026-10-14-xvla-console/validation.json)。
- 真实Hermes草稿与执行隔离，用户确认后交给原X-VLA服务；显示三视图身份、能力缺口、过程照片、检查与收尾/复位。
- 17项软件测试及三组历史图片浏览器检查通过，生产源码20文件未变；零新模型调用、零仿真动作、零子agent。新入口在线动作尚未验收。


## 2026-10-14：界面改为先看场景再填任务

- `20261014-xvla-console-scene-first`：[验证](results/2026-10-14-xvla-console-scene-first/validation.json)。
- 载入不要求任务，立即拍摄实际三视图；图片下填写任务后才调用识别与Hermes。19项软件测试通过，浏览器已刷新；零新仿真、零新模型调用。

- `20261014-all-vlm-qwen38-book`：[全VLM切换与书收纳复测](results/2026-10-14-all-vlm-qwen38/summary.json)。初识别/复核/Hermes/全部检查统一Qwen3.8；原图对照1次仍误报到位。在线1次VLA150步仍提前允许松手，收尾291步，书在盒外；新模型松手后与最终均识别outside，整链final visual failed；归位通过不替代入格。原提示词/机位/收尾与复位不变，无助手/子agent。

### 20261014-conservative-handoff-offline

- 具体目的地区域 + 保守接管：书松手前 unknown 拦截，盒外 False；盘上碗未标准开口仍 True，并实际触发接管复核（仅模型分支，无运动）。
- 黄杯复用 8.95 秒松手证据；63.55 秒当前门状态按门板几何判 closed，整体 True。初版误放行/透明门误判均保留。
- 软件 245 passed、30 subtests；10 原生接口覆盖。真实照片 3 场景，19 次成功 Qwen3.8 请求；不代表在线成功率。零新动作、无子agent、原控制不变。
- 测试隔离遗漏曾发出 11 次无效图片 HTTP400，修复并归档，不计视觉样本。
- 证据：[summary.json](results/2026-10-14-conservative-handoff/summary.json)。

### 20261014-console-startup-fix

- 界面复用输出目录残留 info.json，模型权重已加载但服务退出；不是任务检查结论。
- 改为每次独立输出和日志；16项软件测试及真实“启动→释放→再次启动”通过。
- 后台已更新并恢复书收纳场景三视图；零新推理、零任务动作，原失败记录保留。
- 证据：[summary.json](results/2026-10-14-console-startup-fix/summary.json)。

### 20261014-console-network-and-setup

- 交互入口不再扫描全台账，实测X-VLA启动前准备248秒降至约1秒。
- 旧图正常复核94.34秒超过原90秒；初识别180秒/请求、外层390秒，错误细节留档。
- 实际三视图识别与Hermes书收纳草稿通过，总415秒；识别/规划仍慢。
- 147项测试及6项子测试；10步原初始化静置、零VLA任务动作；未执行放书/收尾。
- 证据：[报告](results/2026-10-14-console-network-fix/summary.md)。

### 20261014-book-held-support-gate

- 用户执行书收纳：原生位置165至900步持续True，VLA持续闭爪；6次视觉Unknown、0次收尾释放。
- 未松手时“直接可见支撑接触”的提示词与代码硬门槛，遇不透明盒壁遮挡阻断。只诊断，未改动或重跑。
- [诊断](results/2026-10-14-book-held-diagnosis/summary.md)。

### 20261014-occluded-support-handoff

- 通用几何/过程/同步速度接管：旧提前松手负例禁止，新300/900步已入格正例通过独立复核并允许原松手；原控制7文件未改。
- 碗已松手回归通过；入炉历史杯子补证通过、炉门误判，额外回归未通过。472项测试及30项子测试；零新动作，无子agent。
- [离线报告](results/2026-10-14-occluded-support-handoff/summary.md)。

### 20261014-book-latency-diagnosis

- 用户书收纳1138秒；记录模型等待约860秒（76%），X-VLA10次约5.64秒。识别/重复检查、Hermes新环境和多图协议是优先优化点。
- 实测归位ready=True；最终JSON格式错误，同时松手后原生入格False。仅分析、未加速改代码或重跑。
- [耗时诊断与优化顺序](results/2026-10-14-book-latency-diagnosis/summary.md)。

### 20261014-latency-optimization

- 40项原生首次检查起点、复用Hermes环境、任务清单差异复核、合并身份与区域审核、滑动窗口；原收尾复位/深度未改。
- 旧书提前松手负例拒绝、已入格正例独立复核允许；常规9图，遮挡关键复核27图。入炉额外历史请求网络超时未验收，在线提速未测。
- [实现与离线验收](results/2026-10-14-latency-optimization/summary.md)。

### 20261014-native-joint-planning

- 完整原生技能目录及服务端门禁；真实Hermes单轮识别规划+一次独立复核，JSON模式和逐回答一次修复。原控制/复位/5/5未改。
- 碗瓶旧图规划56.2秒，在线43.3秒；完整碗入抽屉并关闭第228步达标，240步停止VLA，视觉目的通过。
- 整链仍未通过：原退离器没有目标定位框，上抬4/6cm候选均净空失败，退离/home未执行；执行6分18秒，其中视觉请求约5分13秒。
- [修改、离线与一例在线报告](results/2026-10-14-native-joint-planning/summary.md)。


### 20261014-generic-retreat — 通用退离与完整在线复测

- 旧240步重放误差小于1e-9；旧上移失败，新斜向4cm实际退离84步、home135步、5/5，任务全程保持。
- 在线第228至240步碗入抽屉和抽屉关闭同时达标；第4次Qwen证据引用修复请求90秒超时，收尾未启动，release/movement/confirm/home均为0，整链blocked。
- 从场景加载至终态452.30秒；规划46.13秒；执行379.74秒。
- 348项软件测试与53子测试通过；保留首次重放入口不匹配和测试入口缺路径的原失败记录。
- 证据：[测量摘要](results/2026-10-14-generic-retreat/summary.json)，[旧动作重放](results/2026-10-14-generic-retreat/replay_v2/summary.json)。仅此样例效果，不代表所有任务成功率。

| 20261014-sim-fast | 仿真真值快速版 | 2条流程/4次技能通过；执行VLM0，原5/5、下一起点接续与末项home通过；暖启动执行153s/392s，含规划总210s/493s；规划拒绝与启动错误保留 | [报告](results/2026-10-14-sim-fast/README.md) |
| 20261015-sim-io | 仿真观测I/O优化 | 2流程/4技能原5/5与handoff通过；执行152.773→74.421s、392.139→181.086s；观测/RNG未配对，单例新增冷启动不做严格因果比较 | [报告](results/2026-10-15-sim-io/README.md) |
