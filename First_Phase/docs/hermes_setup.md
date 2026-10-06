# Hermes 接入指南（阶段 C-2，**已实测 2026-10-04**）

> **状态说明**：本文所有条目均为 **2026-10-04 在本机实测**（WSL2 Ubuntu 24.04 + Hermes Agent v0.21.5），
> 每条结论都能指到 `logs/` 下的原始日志（主体为 `phaseC2_*`；图像链路为 `phaseC2_vision*`）。
> 本文**已全部落锤为实测口径，无遗留规划项**；与更早规划不一致之处，一律以本文的实测结论为准。
> 凭据纪律：真实 DashScope key 只写在 `hermes config env-path` 指向的 `.env`
> （实测为 `/home/yhwang/.hermes/.env`，`chmod 600`）；仓库内只保留占位符 `.env.example`。

---

## 1. 已确认的事实（阶段 C0）

| 项 | 值 |
|---|---|
| 安装方式 | `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash` |
| 安装目录 | `/home/yhwang/.hermes/hermes-agent`（install method: git） |
| 可执行文件 | `/home/yhwang/.local/bin/hermes`（已把 `$HOME/.local/bin` 写入 `~/.bashrc` 的 PATH） |
| 版本 | `Hermes Agent v0.21.5+6662.g343500b (2026.9.24)`，upstream `343500b3` |
| 自带 Python | 3.14.7（独立的 runtime venv）。**注意**：Hermes 会把该解释器目录插到 stdio 子进程 `PATH` 的最前面，这是 §2.3 必须钉死解释器的根因 |
| 家目录 | `/home/yhwang/.hermes`（含 `config.yaml`、`.env`、`skills/`、`tools/`、`sessions/`、`state.db` 等） |
| 系统服务 | **无**：安装时没有 `systemd --user`，未安装任何常驻服务 |
| 凭据状态 | 阶段 C0 时**未配置任何模型凭据**；阶段 C-2 已配置（见 §2.1） |
| 默认模型字段 | 阶段 C0 为 `model.default = "anthropic/claude-opus-4.6"` + `provider: auto`；阶段 C-2 已替换为 §2.2 的实测组合 |
| MCP 服务器 | 阶段 C0 时为空（`No MCP servers configured.`）；阶段 C-2 已注册 `robot_tools`（见 §2.3） |
| 安装期网络 | 直连可用（HTTP 200），安装时未走代理 |
| 已知告警 | `hermes doctor` 报若干 npm 依赖漏洞（`agent-browser` 3 项、`web` workspace 7 项、`ui-tui` 5 项），均为上游 lockfile 问题，本地手修不会持久化 |
| MCP 依赖导入身份 | 阶段 C0 时 `mcp 2.3.0` 只在 `/root/.local/lib/python3.12/site-packages`，`yhwang` 身份下导入失败。**阶段 C-2 已解决**：`yhwang` 下 `python3 -c "import mcp"` 成功，`importlib.metadata.version("mcp")` = `2.3.0`，实际加载路径为 `/home/yhwang/.local/lib/python3.12/site-packages` |

相关证据：`logs/phaseC0_hermes_install.log`。

与 ROS/Gazebo 的关系：Hermes 自身**不**访问 Gazebo。它通过 MCP 调用 `robot_tools` 的 9 个工具，由工具层去和 ROS 2 交互。因此 Hermes 可以在**没有**仿真在跑的情况下启动（只有涉及运动的工具会失败）。

---

## 2. 阶段 C-2 实测结论

### 2.1 凭据（实测）

- key 写入位置 = `hermes config env-path` 打印的路径，实测为 **`/home/yhwang/.hermes/.env`**。
- 写入形式为单行 **`DASHSCOPE_API_KEY=<key>`**（该文件已被 `chmod 600` 保护）。
- 仓库内**只保留占位符**（`.env.example`）；所有日志、会话记录、报告里都没有 key 原文（C-2 全部日志已逐文件核验，密钥占位串计数均为 0）。
- 实测确认该站点：**中国大陆站（`alibaba-cn`）直连一次成功**，国际站 provider `alibaba` 未用到。

### 2.2 OpenAI 兼容端点与主模型（实测）

DashScope 的 OpenAI 兼容模式端点：

```
https://dashscope.aliyuncs.com/compatible-mode/v1            # 中国大陆（alibaba-cn），本次使用
https://dashscope-intl.aliyuncs.com/compatible-mode/v1       # 国际站（alibaba），备用
```

不需要走 `provider: custom` + 手填 `base_url`：Hermes **内置 alibaba provider 插件**已覆盖这两个站点，配置 provider 即可，`base_url` 自动解析：

```bash
hermes config set model.provider alibaba-cn      # 环境变量 DASHSCOPE_API_KEY；base_url 自动指向大陆站
hermes config set model.default qwen3-vl-plus
hermes config set model.supports_vision true     # 必须，见 §2.4
```

实测终态：

| 项 | 实测值 |
|---|---|
| `model.provider` | `alibaba-cn` |
| `model.default` | `qwen3-vl-plus` |
| `model.supports_vision` | `true` |
| 冒烟测试 | `env -u <代理变量> hermes -z "只回答一个数字:1+1等于几?"` → 输出 `2`，exit 0（证据 `logs/phaseC2_model.log`） |

#### 型号选择实测（为什么不是 `qwen-vl-max`）

任务书原定 `qwen-vl-max`，实测**不可用**：

| 模型 | DashScope compatible-mode 下的原生 function calling | 结论 |
|---|---|---|
| `qwen-vl-max` | **不支持**：curl 直连返回 `finish_reason=stop`、响应无 `tool_calls`；模型把工具调用写成**纯文本**，Hermes 不会执行 → agent 原地不动 | 弃用 |
| `qwen-vl-plus` | **不支持**（同上） | 弃用 |
| `qwen3-vl-plus` | **支持**：返回 `finish_reason=tool_calls` + 合法 `tool_calls`；工具往返后 `api_calls=2` 真实生效 | **选用** |

实测判定依据：`logs/phaseC2_toolcall_probe.log`（`api_calls=1`、模型只把命令写成文本）、`logs/phaseC2_toolprobe.log`（MCP schema 缓存里 9 个工具齐全，但模型自报工具列表里没有 → 问题在模型侧）、`logs/phaseC2_model_verify.log`（切模型后 `api_calls=2`，工具真实执行）。

> 注意：`qwen3-vl-plus` 同时具备工具调用与视觉能力，满足「既会调工具、又能看图」的双重要求。

### 2.3 MCP 注册 `robot_tools`（实测）

`config.yaml` 中最终条目（`~/.hermes/config.yaml`，实测通过）：

```yaml
mcp_servers:
  robot_tools:
    command: /usr/bin/python3
    args:
      - -u
      - /home/yhwang/fyp/robot_tools/mcp_server.py
    enabled: true
    env:
      PATH: /usr/bin:/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/yhwang/.local/bin
    cwd: /home/yhwang/fyp
```

注册命令（`--args` 必须是最后一个选项）：

```bash
hermes mcp add robot_tools --command /usr/bin/python3 \
  --args -u /home/yhwang/fyp/robot_tools/mcp_server.py
hermes config set mcp_servers.robot_tools.env.PATH "/usr/bin:/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/yhwang/.local/bin"
hermes config set mcp_servers.robot_tools.cwd /home/yhwang/fyp
hermes config set mcp_servers.robot_tools.enabled true
```

**为什么必须钉死解释器 + 覆盖 `env.PATH`**（本次踩过的坑，实测根因）：

1. Hermes 会把**自带的 Python 3.14**（`~/.hermes/tools/python-3.14.7+.../bin`）插到 stdio 子进程 `PATH` 的最前面；
2. 于是子进程里的 `python3` 解析成 3.14，而 `anyio` / `mcp` 装在**系统 Python 3.12 的 user site**（`/home/yhwang/.local/lib/python3.12/site-packages`）；
3. 结果是注册时报 `✗ Failed to connect: Connection closed`，`~/.hermes/logs/mcp-stderr.log` 里是
   `ModuleNotFoundError: No module named 'anyio'`；
4. 只把 `command` 写成绝对路径 `/usr/bin/python3` **还不够**：`mcp_server.py` 在检测不到 `rclpy` 时会 `re-exec` 到 `source ROS ... && exec python3 "$@"`，那一步又会按 `PATH` 找到 3.14。所以 `env.PATH` 覆盖是必需的。

实测结果（证据 `logs/phaseC2_mcp.log`、`phaseC2_mcp_add.log`）：

```
Testing 'robot_tools'...   Transport: stdio → /usr/bin/python3
✓ Connected (1549ms)       ✓ Tools discovered: 9

hermes mcp list:
  Name             Transport                      Tools        Status
  robot_tools      /usr/bin/python3 -u /home...   all          ✓ enabled
```

9 个工具全部列出：`observe_scene`、`locate_object`、`propose_grasps`、`plan_grasp`、`execute_plan`、`get_execution_status`、`verify_task`、`reset_scene`、`stop_execution`。

### 2.4 图像链路实测结论（关键）

**结论一：MCP 工具结果里的图像块无法把像素交给模型。**
Hermes 对 MCP 返回的 `ImageContent` 块的处理是**无条件**的：缓存成文件后只把 `MEDIA:<path>` 文本标签回灌给模型
（源码 `tools/mcp_tool_content.py` 的 `_cache_mcp_image_block` → `_cache_mcp_media_block`，没有任何 provider 开关）。
实测证据：`observe_scene` 的 tool 结果里 `data:image` / `image_url` / `"type": "image"` 计数均为 0，只有一行 `MEDIA:/home/yhwang/.hermes/cache/images/img_*.png`。

**结论二：正路是 `observe_scene` → 内置 `vision_analyze` → native fast path。**

1. `observe_scene` 返回 `image_path`（落盘帧，`data/frames/frame_*.png`）；
2. 把该路径交给内置工具 `vision_analyze`（参数 `image` = 路径，`question` = 问题）；
3. 前提是 `model.supports_vision: true`（`agent/image_routing.py` 的 `_supports_vision_override` 逃生开关）；
4. 于是 native fast path（`tools/vision_tools.py` 的 `_should_use_native_vision_fast_path` / `_build_native_vision_tool_result`）
   把图片以多模态块 `{"type": "image_url", "image_url": {"url": <data URL>}}` **挂进主模型的 tool-result**，像素直接进上下文。

**已实测验证**（证据：`logs/phaseC2_vision.log`、`logs/phaseC2_vision_verdict.log`、`logs/phaseC2_vision_frame.png`）：

- 工具返回原文：`Image loaded into your context — you can see it natively now. Use your built-in vision to answer the user.`
- 模型答出的内容包含**只在像素里存在**的事实：红色立柱在画面中右侧偏上、立在蓝色方块上；左下角黄色机械臂及其末端一小块红色夹爪。
  独立肉眼核对同一帧（`phaseC2_vision_frame.png`，640×480）与之一致；而这些事实**不在任何工具文本输出里**（工具文本只有 `center_base` / `mask_px` 等数值）。
- 该次 `auxiliary` 用量只有 `title_generation` 一项 → **没有走 aux vision 客户端**，确认是 native fast path。

> **重要陷阱（别踩）**：会话库/导出是**纯文本投影**——图像 part 会被替换成 `[screenshot]` 占位符
> （源码 `agent/session_persistence.py` 的 `_durable_content`）。
> 因此**不要**用 `grep 'image_url'` 或 `grep '"type": "image"'` 作为「像素是否进模型」的判据：
> 前者会命中 `vision_analyze` 的**参数名**造成假阳性，后者恒为 0。
> 判据应改为：工具返回文本是否含 `Image loaded into your context`、导出里是否出现 `[screenshot]`、以及模型是否答出像素级事实。

### 2.5 端到端实测（三次，全部真实记录）

| 尝试 | prompt 要点 | verify_task 结果 | 结论 |
|---|---|---|---|
| attempt 1 | 任务书原样 prompt（未给放置坐标、未要求轮询到终态） | `error_kind=task_failed`、`reason=place_out_of_tolerance`、`dxy=0.1283`、`carried=None` | **失败**。agent 自造 `expect_place_xy=[0.15,-0.25]`（并非蓝色区），且只 `sleep` 约 22 秒就 verify —— 工具日志显示执行当时仍在 `lift`（`exec_0001 stage: close -> lift`），arm 在 verify 之后仍在收 goal |
| attempt 2 | 补上正确坐标 + 要求轮询到终态 | `task_failed`、`dxy=0.0458`、`carried=true`、`truth_pose=[0.0922,-0.432,0.014]` | **失败**。已真实抓起并释放（`carried=true`），但落点超过 0.03 m 容差 |
| attempt 3 | 正确坐标 + 轮询到终态 + 允许最多 3 轮重试 | **`task_success: true`**、`carried: true`、`dxy=0.0175`、`z=0.0320`、`truth_pose=[0.0569,-0.4339,0.032]` | **成功**（第 1 轮即成功，轮询约 152 秒） |

attempt 3 的 `verify_task` 原文：

```json
{
  "ok": true, "error_kind": null, "reason": "",
  "task_success": true, "carried": true,
  "dxy": 0.017497864864467305, "z": 0.03199996939271488,
  "expect_place_xy": [0.05, -0.45],
  "truth_pose": [0.0569, -0.4339, 0.032],
  "truth_only_for_eval": true, "exec_id": "exec_0001",
  "detail": "dxy=0.0175 m (tol 0.030), z=0.0320 m (tol 0.040); pillar pose read from the simulator (ground truth, evaluation only); carried=True from execution exec_0001"
}
```

用量（`logs/phaseC2_e2e_attempt3_usage.json`）：主模型 **33 次调用 / 约 72.8 万 token**，`provider=alibaba-cn`、`model=qwen3-vl-plus`。
主证据：`logs/phaseC2_e2e_attempt3.log`、`logs/phaseC2_e2e_attempt3_session.jsonl`、`logs/phaseC2_e2e_attempt3_trace.txt`。

**经验**：任务能否成功，主要取决于 prompt 有没有把「放置坐标」和「轮询到终态」写死；这两点不写，模型大概率提前 verify 并误判失败。

#### 推荐任务 prompt 模板（逐字可用）

```
使用 robot_tools 的 MCP 工具完成抓取任务:把红色立柱放到蓝色区域(放置目标 base 坐标 (0.05, -0.45))。流程:observe_scene 观察 → locate_object 定位红色立柱 → propose_grasps → plan_grasp → execute_plan → get_execution_status 轮询直到 state 到达终态(succeeded/failed)再继续 → verify_task(expect_place_xy 用 [0.05, -0.45])验证。若失败可重试最多 2 次。如实报告每一步结果,最后明确给出成功或失败结论,并原文引用 verify_task 的返回。
```

想让它真正"看图"时，在 prompt 里追加一句（并按 §2.4 的链路走）：
「并用内置 `vision_analyze` 分析 `observe_scene` 返回的 `image_path`，描述画面内容」。

### 2.6 运行期注意事项（实测补充）

- **oneshot 会连带杀掉进行中的执行**：`hermes -z` 会话退出时会回收 MCP 子进程，工具层后台的执行线程随之消失（attempt 1 结束时 arm 仍在收 goal）。**必须轮询到终态再结束会话**。
- **一律用 `yhwang` 身份操作**：以 root 跑 phase 脚本会把 `data/`、`logs/` 变成 root 拥有，`yhwang` 之后无法写入（实测报 `Permission denied`，且本机没有免密 sudo）。root 遗留已改名保留为 `data.phaseC1_root/` 与 `logs/prev_phaseC1/`。
- **场景状态跨会话保留**：评测/演示"从零抓放"前先让 agent 调 `reset_scene`（当前立柱已在蓝色区上，是 attempt 3 的成果）。
- **调 DashScope 必须绕开代理**：本机代理 `127.0.0.1:7897` 时通时断，所有 hermes 调用前缀
  `env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY`。
- 起 Hermes 之前需要仿真栈在跑（`phaseC1_up.sh`），否则运动类工具会失败。
- `ros2 daemon` 必须常驻；不要用会杀 daemon 的清理方式（`ros2 daemon status` 在本机会崩，别用它判断状态）。
- VLM 只负责理解意图、选目标、解释结果与给重试策略；**逐帧关节指令与凭空坐标都不允许进入控制器**——运动目标只能来自 `locate_object` 的数值输出。

---

## 3. 明确不做的事

- 不让 VLM 逐帧输出关节角或直接给坐标作为控制目标。
- 不用通用 VLM 的"看图估计坐标"替代感知模块。
- 不让模型从 MCP 工具结果里直接读像素：该通路在设计上就不存在（图像块被无条件降级成 `MEDIA:` 文本标签），**要看图必须走 `vision_analyze`**。
- 不在文档、日志或会话记录里出现真实 API key。
