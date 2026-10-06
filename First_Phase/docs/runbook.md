# 运行手册（runbook）

本手册是日常操作的**照抄版**：起栈 → 跑任务 → 复跑评测 → 收尾 → 故障处置。
背景与实测细节见 `docs/hermes_setup.md`；工具契约见 `docs/robot_tools_api.md`。

**通用纪律**：全程用 `yhwang` 身份，**不要** `sudo` / root（root 会把 `data/`、`logs/` 变成 root 拥有，之后普通用户写不进去）；Hermes 必须**免代理**启动（`~/.bashrc` 里已配好别名，直接敲 `hermes` 即可）。

```bash
# 别名（已写入 ~/.bashrc）—— 直接敲 hermes 就行，不用手动处理代理
alias hermes='env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY hermes'

# 别名不生效时（脚本内调用 / 用全路径）手动加这段前缀
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
```

---

## 一、启动

```bash
# [Windows] 进入 WSL
wsl.exe -d Ubuntu
```

```bash
# [WSL] 1) 起 ros2 daemon —— 必须先于 up.sh
#    实测：起栈时若 daemon 没跑，bringup 的 load_controller 会全部超时静默失败
#    （ros2 control list_controllers 显示 No controllers are currently loaded，机械臂不会动）
#    注意：ros2 daemon status 在本机会崩，不要用它判断状态；daemon 必须常驻
ros2 daemon start

# [WSL] 2) 一键起栈：Gazebo + 相机桥 + move_group（约 130 秒）
bash /home/yhwang/fyp/robot_tools/phaseC1_up.sh
```

起栈成功的自检标准（`phaseC1_up.sh` 结尾会打印，日志在 `/home/yhwang/fyp/logs/phaseC1_*.log`）：

| 检查项 | 期望 |
|---|---|
| 进程 | `gz sim`、`parameter_bridge`（两处：clock/tf 与 rgbd）、`move_group` 都在 |
| 话题 | 含 `/rgbd/image`、`/rgbd/depth_image`、`/rgbd/camera_info`、`/joint_states` 的行 |
| 服务 | `/plan_kinematic_path` |

三项不齐就不要往下走。

`phaseC1_up.sh` 自带防重复启动保护:若提示栈已在运行,说明可以直接用;想重起就先跑 `phaseC1_down.sh` 再跑 `phaseC1_up.sh`。

### 实时观察（可选，开机后随手开）

想随时看相机画面（与 VLM 同款视角，约 12 Hz）：

```bash
# [WSL] 打开实时图像观察窗（WSLg 会把它显示成 Windows 窗口）
setsid nohup ros2 run rqt_image_view rqt_image_view /rgbd/image >/tmp/rqt_live.log 2>&1 < /dev/null &
sleep 10 && pgrep -af rqt_image_view     # 有输出即存活；日志见 /tmp/rqt_live.log
```

**若 WSLg 窗口画不出来**（本机现状：图标注册了但窗口不显示），用**浏览器通道**（完全不依赖 GUI）：双击打开 `D:\FYP\First_Phase\logs\live.html`，约 2 Hz 刷新，页面每 500 ms 只换 `<img>` 的 src（带 cache-buster，不整页闪）。它依赖 `/home/yhwang/fyp/tools/live_view.py` 常驻，起栈后跑一句：

```bash
# [WSL] 起浏览器通道：把 /rgbd/image 以 ~2Hz 落成 live.jpg（原子替换）
setsid nohup bash -c 'source /opt/ros/jazzy/setup.bash && exec python3 /home/yhwang/fyp/tools/live_view.py' >/tmp/live_view.log 2>&1 &
sleep 10 && pgrep -af "[l]ive_view.py"    # 有输出即存活；日志见 /tmp/live_view.log
```

画面落在 `/mnt/d/FYP/First_Phase/logs/live.jpg`（先写 `.tmp` 再 `os.replace`，浏览器不会读到半截图）；日志每 30 秒一行心跳（收到的帧数 / 实际写入数 / 跳过数 / 字节数）。

两条实测教训：

- **gz 的 GUI 窗口点不开属预期**：本机只有 llvmpipe 软渲染，GUI 常驻约 261% CPU，去点它只会更卡；**不要点它，也不要管它**，让它在后台待着即可。（`gz.launch.py` 没有关 GUI 的开关，up.sh 起的栈一定带 GUI。）
- **严禁 `pkill gz sim gui`**：gz 的监管进程把「GUI 退出」当成用户关窗信号，杀掉 GUI 会**连带把整个仿真杀掉**（实测：gz 全死，`move_group` 与 `parameter_bridge` 变成无用残留）。要停栈只走 `phaseC1_down.sh`。

---

## 二、跑任务

两种入口（都要求栈已在跑）：

```bash
# [WSL] 交互式 —— 直接敲 hermes，~/.bashrc 里的别名会自动免代理
hermes

# [WSL] 一次性（非交互、自动批准工具调用、stdout 只打印最终回复）
hermes -z "<任务 prompt>"
```

> 免代理别名（已写入 `~/.bashrc`，登录 shell 生效）：
> `alias hermes='env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY hermes'`。
> 登录 shell 被 `/etc/profile.d/99-proxy.sh` 注入代理 `127.0.0.1:7897`，该代理实测已宕（curl 12 秒超时、`http=000`），去掉代理后 0.22 秒即连通 DashScope。
> **绕过别名时**（脚本内调用、或直接用全路径 `/home/yhwang/.local/bin/hermes`）别名不生效，仍需手动加 `env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY` 前缀。

推荐任务 prompt 模板（逐字可用）：

```
使用 robot_tools 的 MCP 工具完成抓取任务:把红色立柱放到蓝色区域(放置目标 base 坐标 (0.05, -0.45))。流程:observe_scene 观察 → locate_object 定位红色立柱 → propose_grasps → plan_grasp → execute_plan → get_execution_status 轮询直到 state 到达终态(succeeded/failed)再继续 → verify_task(expect_place_xy 用 [0.05, -0.45])验证。若失败可重试最多 2 次。如实报告每一步结果,最后明确给出成功或失败结论,并原文引用 verify_task 的返回。
```

两个必须写死的点（不写就会失败）：**放置坐标 `[0.05, -0.45]`** 与 **「轮询到终态再继续」**。
另外 `hermes -z` 会话退出会连带杀掉进行中的执行，所以一定要等轮询到终态。

想让它**真正看图**（而不只是读数值），在 prompt 里追加：

```
并用内置 vision_analyze 分析 observe_scene 返回的 image_path,描述画面内容。
```

（原理：MCP 工具结果里的图像块只会变成 `MEDIA:<path>` 文本标签，像素进不来；必须靠 `vision_analyze` 的 native fast path，且 `model.supports_vision: true` 已开。）

---

## 三、复跑评测

场景状态**跨会话保留**（立柱不会自己回到起点），所以"从零抓放"的评测必须先把场景复位：

```bash
# [WSL] 让 agent 先复位（或在任务 prompt 里要求第一步调 reset_scene）
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  hermes -z "调用 robot_tools 的 reset_scene 复位场景(归位机械臂并把红色立柱送回起点),报告返回。"
```

复位后再跑第二节的任务 prompt。想看当前状态可以直接问：

```
调用 observe_scene 和 locate_object('红色立柱'),报告红色立柱的 center_base 与 surface_base。
```

---

## 四、收尾

```bash
# [WSL] 停仿真栈（脚本刻意保留 ros2 daemon）
bash /home/yhwang/fyp/robot_tools/phaseC1_down.sh
```

- **不要**杀 `ros2 daemon`：daemon 停掉后本机 `ros2` CLI 会长时间卡住（mirrored 网络的 loopback 怪癖）。
- **不要**放 root/sudo 去跑任何 phase 脚本。
- Hermes 侧无需收尾：MCP 子进程随会话退出自动回收。

---

## 五、故障表

| 症状 | 原因 | 处置 |
|---|---|---|
| `hermes mcp add` / `mcp test` 报 `✗ Failed to connect: Connection closed`；`~/.hermes/logs/mcp-stderr.log` 里是 `ModuleNotFoundError: No module named 'anyio'` | 解释器没钉死：Hermes 把自带 Python 3.14 插到子进程 `PATH` 最前，而 `anyio`/`mcp` 装在系统 Python 3.12 | 按 `docs/hermes_setup.md` §2.3 把 `command` 设为 `/usr/bin/python3` **并**覆盖 `mcp_servers.robot_tools.env.PATH`（两者缺一不可） |
| 模型调用报 401 / 403 / 鉴权类错误 | key 错，或 key 不属于大陆站 | 核对 `~/.hermes/.env` 的 `DASHSCOPE_API_KEY`；改 `hermes config set model.provider alibaba` 试国际站 |
| agent 只输出文本、不复述工具结果、完全不行动（`api_calls=1`） | 模型不支持原生 function calling（`qwen-vl-max` / `qwen-vl-plus` 在 DashScope compatible-mode 下都不支持） | 固定用 `hermes config set model.default qwen3-vl-plus` |
| verify 时任务明显没做完（`carried=None`、dxy 很大、状态还是 `running`） | prompt 没要求「轮询到终态」，agent 提前 verify 并提前结束会话 | 用第二节的 prompt 模板（写死终态轮询）；同时确认 `expect_place_xy` 用的是 `[0.05, -0.45]` 而不是模型自造的坐标 |
| 写 `data/`、`logs/` 报 `Permission denied` | 之前用 root 跑过 phase 脚本，目录变成 root 拥有 | 改回 `yhwang` 身份；已 root 化的目录需改名保留后由 `yhwang` 重建（本机现有遗留：`data.phaseC1_root/`、`logs/prev_phaseC1/`，可作对照） |
| `hermes` 报 network / request 错误（请求超时、连接被拒） | 登录 shell 被 `/etc/profile.d/99-proxy.sh` 注入代理 `127.0.0.1:7897`，该代理实测已宕（curl 12 秒超时、`http=000`）；**旧终端里跑的 hermes 进程环境就带着这个死代理** | **三层防护**：① `~/.hermes/.env` 已加 `NO_PROXY` / `no_proxy` 绕行域名（`localhost,127.0.0.1,aliyuncs.com,dashscope.aliyuncs.com`）——**进程级**，任何启动方式（交互 / 脚本 / 旧终端）都免疫；实测「显式带死代理变量跑 `hermes -z`」仍正常返回；② 交互 shell 用 `~/.bashrc` 里的 `hermes` 别名（自动 `env -u` 六个代理变量）；③ **别名只对"新"终端生效**——改完 `~/.bashrc` 必须重开终端或 `source ~/.bashrc`，**旧终端里跑的 hermes 依然带死代理** |
| `ros2 control list_controllers` 显示 `No controllers are currently loaded`（机械臂不会动） | 起栈时 `ros2 daemon` 没在跑，bringup 的 `load_controller` 全部超时静默失败 | 先 `phaseC1_down.sh` → `ros2 daemon start` → `phaseC1_up.sh` 重起；应急也可手动补：`ros2 control load_controller --set-state active arm_controller`、`ros2 control load_controller --set-state active gripper_controller` |
| `ros2` CLI 长时间卡住 | `ros2 daemon` 被停掉了 | `ros2 daemon start`（`ros2 daemon status` 在本机会崩，不要用它判断） |
| 运动类工具报错、话题/service 缺失 | 仿真栈没起或没起全 | 重跑 `phaseC1_up.sh`，按第一节的三项自检确认 |
