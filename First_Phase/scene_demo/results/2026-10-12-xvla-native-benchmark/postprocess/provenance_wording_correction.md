# 2026-10-12 来源与调用范围措辞更正

本次是派生报告更正。执行者为 GPT 6 Astra 工具 worker 子会话，实际模型 `openai/gpt-6-astra`，推理档位未暴露。

## 更正内容

- `README.md`、`CONCLUSIONS.md` 将离线后处理的零新增调用限定为 DeepSeek、Qwen、VLA，明确 GPT 工具 worker 子会话参与执行。
- 三次实际外部 DeepSeek HTTP 请求合计 input 19,029、output 17,844，总计 36,873 tokens；不包含 GPT 规划、验收与工具 worker 会话用量。
- `final_verification.json` 仅更改顶层 `execution` 字段；统计、逐例断言、哈希检查及其余 metadata 保持。既有 11 pytest 实际结果保留，本次未重跑。
- registry 仅更新 `20261012-xvla-native-benchmark` 的 README、CONCLUSIONS、final_verification、manifest 四个 evidence SHA256。按本次授权范围，registry 的 findings 等非哈希字段保留；其中历史“新增模型/物理动作0”表述应参照本更正理解为未新增 DeepSeek/Qwen/VLA 调用、环境动作0，不涵盖 GPT 子会话模型用量。

## 追溯与机械证明

1. 修改前先逐文件核对原 manifest 的 4,880 个文件 SHA256/size，并将三个报告的原字节保存到 `provenance_wording_before/`。副本中保留被更正的历史措辞。
2. `provenance_wording_baseline.json` 保存原 manifest、修改前文件清单、旧41条目规范化哈希、当前条目原数据，以及仅屏蔽四个允许变动 evidence 哈希的 registry 全字节哈希。
3. `provenance_wording_preservation.json` 给出删除 `execution` 后 JSON 对象相等、规范化 SHA256 相等，以及仅替换该字符串后文件其余字节完全相同的断言。
4. 所有既有文件（除三个派生报告及 manifest）按修改前 SHA256/size 逐一比较；单列200个slot result、1,160个response、200个steps及source_snapshot。旧41条目和registry其余字节一并校验。
5. manifest 收入更正记录、工具脚本、基线、不变性证明、三个旧派生副本及已闭合的 validate 输出；继续排除 manifest 自身和活动日志 `postprocess/finalize.log`。首次 validate 输出闭合后更新 manifest 与当前条目 manifest 哈希，再运行 validate 核对最终链，要求输出原字节相同。

## 执行与复核命令

在 `D:\FYP` 运行：

```text
python First_Phase/scene_demo/results/2026-10-12-xvla-native-benchmark/postprocess/provenance_wording_tools.py snapshot
# 三个派生报告通过 apply_patch 更正后执行：
python First_Phase/scene_demo/results/2026-10-12-xvla-native-benchmark/postprocess/provenance_wording_tools.py finalize
# 可重复的只读复核：
python First_Phase/scene_demo/results/2026-10-12-xvla-native-benchmark/postprocess/provenance_wording_tools.py verify
```

工具仅使用标准库进行字节复制、JSON比较、SHA256校验及指定范围的索引更新；子进程仅执行 `.agents/skills/fyp-experiment-review/scripts/review.py validate`。其原始标准输出保存为 `provenance_wording_validation.json`。没有新增物理/仿真动作、DeepSeek/Qwen/VLA调用，没有重跑pytest或视频解码。
