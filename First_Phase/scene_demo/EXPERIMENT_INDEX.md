# 实验历史索引（截至2026-10-09）

本索引截至2026-10-09，共15个历史campaign。每行是问题/证据导航，不是一个独立统计样本；记录中的回放、复用、未启动和原始判据差异保留。当前最近完整样本是前12例；20例目录名只保留原注册历史。环境是LIBERO/robosuite/MuJoCo + Franka Panda，非ManiSkill。开发GPT规划/验收、DeepSeek实现；运行期Hermes/Qwen3-VL与SmolVLA不等于开发分工。

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

## 当前结论

- 当前权重有抓取/释放能力，原生酒瓶、碗以及部分组合有成功证据；能力存在不等于可靠性足够。
- 原生酒瓶基线、三推理配置、篮子完整/拆分、夹爪位姿准备和完整关节归位已做，优先复用，不能重列为未做。
- 辅助器准备失败、归位入口受阻与真正VLA失败分开；安全退离仍是独立机制缺口。
- 碗/灶台在最新受测条件中较稳定；酒瓶对起点/前序条件敏感，汤罐抓住后未入篮，番茄酱未确认抓取；仍未确定训练因果。
- 下一步先离线比较现有成功/失败输入与轨迹；只有明确缺项才注册小对照，不自动微调/扩大或续跑已中止20例。
