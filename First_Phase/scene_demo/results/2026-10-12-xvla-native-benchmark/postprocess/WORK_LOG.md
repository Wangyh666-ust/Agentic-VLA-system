# 固定判据离线归档记录

- 执行者实际模型：`openai/gpt-6-astra`；推理档位未暴露，不假称 gpt-6.1-sol high/low。直接执行用户六节机械任务，本轮无子agent、无新增模型请求。
- 用户给定任务是本轮规范。仅使用允许的归档路径；源码与文档用 apply_patch，数据产物由本地工具生成。无控制实现改动，无新增测试，无 git commit/push。
- 初始 `review.py validate`：`ok=true, entry_count=41, evidence_count=3517`。相关历史 `20261011-native-benchmark` 是 SmolVLA 139/200，配置不同，仅作非配对背景；X-VLA已完成结果见原 summary/metadata/200 slot。
- 200固定slot完整、每task5、唯一attempt_001；summary.complete=true、unknown_paths=[]。client878已退出。server423完整argv/start_ticks=491匹配launch_remaining；08:33:20 UTC仅发送一次SIGTERM并等其退出。
- GPU是当前快照而非峰值：停止前5161MiB，停止后1307MiB；停止后compute-apps列表为空。详见gpu_before_stop.json和server_stop.json。
- 首次离线解析在source_check前段停止：工具将 `D:\FYP\...` 错映射为 `/mnt/d/FYP/FYP/...`，报 FileNotFoundError。根因是只裁掉盘符而ROOT已经含FYP；仅修正本归档工具的固定路径前缀。没有出现原始数据判据差异，也未构造环境、执行动作或调用模型。原inventory/registry/index快照保留，续跑先验证其完全相同。
- 计划执行按用户范围优先：不引入worktree、提交、额外测试或模型审阅。后续仅补用户允许的既有11pytest持久输出。最终按产物和原证据作自审。
- 修正路径映射后离线核验完成：200/197/3，四suite48/50/50/49，1160调用，32184policy/2000settle，200视频完整解码32184帧且全部30fps。原作者转换逐元素及float32字节完全相同。freeze所有源/快照和原campaign inventory保持。
- 生成200行episodes.csv，复制九段代表原视频并核对SHA。DeepSeek三proof实算19029input/17844output/36873total，未新增模型调用。
- 唯一一次既有软件测试补验：`sim_venv/bin/python -m pytest -q -p no:cacheprovider test_xvla_native_benchmark.py`，11 passed、21 warnings、3.52s；原输出已写software/final_pytest.log。原runner/test及来源再次核对无变化。
- README/CONCLUSIONS机械填充用户固定结论和已核实数值；索引仅在末尾追加本campaign一行。最终自审（无子agent工具/新模型请求）：核对原作者转换列与float32阈值、分母、三失败终止事实、20Hz/30fps和总耗时定义、复现边界、DeepSeek真实usage及历史修正表述。
- manifest与registry按finalize阶段闭合：排除自身及活跃finalize.log；旧41 entries逐条规范化hash对照；仅本campaign登记manifest最后hash，然后运行全台账validate并将原输出写入finalize.log。归档通过与否以该实际日志为准，本文不预先代替结果。
