# 环境与操作 SOP

本文汇总本机（Windows 11 + WSL2 Ubuntu 24.04）在搭建与运行仿真链路时踩到的坑，以及对应的标准操作。
所有命令除特别标注外均在 **WSL 内**执行。

---

## 1. 环境基线

| 项 | 值 |
|---|---|
| 宿主 | Windows 11，RTX 5070，24 GB 内存 |
| WSL | WSL2，kernel `6.18.33.2-microsoft-standard-WSL2` |
| 内存上限 | 18 GB（`.wslconfig` 配置），`free -g` 显示 `total 17` |
| 网络 | mirrored 模式 |
| 发行版 | Ubuntu 24.04.1 LTS |
| 代码根 | `/home/yhwang/fyp` |
| 证据落盘 | `/mnt/d/FYP/First_Phase/logs`（即 Windows 的 `D:\FYP\First_Phase\logs`） |

每次开新 shell 必须先执行（顺序不能颠倒）：

```bash
source /opt/ros/jazzy/setup.bash
source /home/yhwang/fyp/ws/install/setup.bash
```

---

## 2. 网络与代理

- 代理地址：`127.0.0.1:7897`，已写入 `/etc/apt/apt.conf.d/99proxy` 与 `/etc/profile.d/99-proxy.sh`。
- 代理**时通时断**。若 apt/curl 报连接失败，先取消代理环境变量再直连重试：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

- 判断当前是否走代理：直接看 `env | grep -i proxy`，不要凭印象。
- Hermes 安装那次是**直连**成功的（HTTP 200），说明代理不是所有场景的必需项。

---

## 3. 权限：sudo 与 root

- `sudo` 需要交互输入密码，脚本里**不要**依赖 `sudo`。
- 需要 root 的文件操作（例如清理 `data/` 下 root 所有的目录）走：

```bash
wsl.exe -u root -d Ubuntu -- bash -lc '<command>'      # 在 Windows 侧执行
```

WSL 内等价写法：

```bash
wsl -u root -- bash -lc '<command>'
```

- 现象参考：`/home/yhwang/fyp/data` 与 `__pycache__` 曾由 root 创建（`drwxr-xr-x root root`），普通用户写入会失败。

---

## 4. ros2 daemon 必须常驻

- mirrored 网络下有一个 loopback 超时怪癖：daemon 一旦被杀，后续 `ros2 topic list` / `ros2 service list` 会长时间卡住或超时。
- 因此所有 teardown 脚本（`tools/stop_all.sh`、`robot_tools/phaseC1_down.sh`）都**刻意不杀 daemon**，结束时还会打印 `ros2 daemon status` 确认它活着。
- 若发现 daemon 掉了，先 `ros2 daemon start` 再继续。

---

## 5. 清理进程必须用方括号写法

`pkill -f "pattern"` 会匹配到**执行 pkill 的这条 shell 命令自身**（命令行里含有该字符串），导致自杀、脚本中断。
标准写法是把匹配模式的第一个字符放进方括号：

```bash
pkill -f "[g]z sim"
pkill -f "[m]ove_group"
pkill -f "[p]arameter_bridge"
pkill -f "[r]obot_state_publisher"
pkill -f "[r]os2 launch so_arm_100_bringup"
pkill -f "[r]os2 launch so_arm_100_moveit_config"
pkill -f "[r]vr2"
```

顺序：先杀子进程（move_group / bridge / gz sim），再杀 launch 父进程，避免中途被重新拉起。

---

## 6. 写脚本时注意 git-bash 的 `$` 提前展开

在 Windows 侧用 git-bash 组织 WSL 命令时（例如把一段脚本经 `wsl.exe bash -lc "..."` 传进去），`$VAR`、`$@`、`$(...)` 会在 **git-bash 这一层**就被展开，传到 WSL 时已经变空。

对策：**先把脚本内容写到 Windows 侧文件，再让 WSL 执行该文件**；或者用单引号包裹、并对确实需要保留的 `$` 转义。

现象参考：`for f in a b c; do cat "$f"; done` 经 git-bash 传递后 `$f` 变成空串，`cat` 报 ``cat: '': No such file or directory``。

---

## 7. Python 与 pip

- 系统 Python 为 3.12.3，是 externally-managed 环境，直接 `pip install` 会被拒绝，必须：

```bash
pip install --break-system-packages <pkg>
```

- 工具层依赖 `mcp==2.3.0`。该版本的 `FastMCP` 已改名：`mcp.server.fastmcp` 不存在，高层服务类现在是 `mcp.server.mcpserver.MCPServer`。写新代码时不要沿用旧导入路径。
- **`mcp` 的安装位置是个坑（已验证）**：本机用 `sudo pip install --break-system-packages mcp==2.3.0` 安装，包落到了 **root 的 user site**（`/root/.local/lib/python3.12/site-packages/mcp`，2026-10-04 04:47）。后果：

  ```bash
  # 以 root 身份
  wsl -u root -d Ubuntu -- python3 -c "import mcp; print(mcp.__file__)"
  # → /root/.local/lib/python3.12/site-packages/mcp/__init__.py   OK

  # 以 yhwang 身份
  python3 -c "import mcp"        # → ModuleNotFoundError: No module named 'mcp'
  ```

  阶段 C-1 的 MCP 验证（`phaseC1_mcp_list*.log`、`phaseC1_tools_test.log`）就是在 root 身份下跑的——同一时期 WSL `logs/` 下的 `phaseC1_*.log`、`robot_tools.log` 均为 `root:root`。
  阶段 C-2 之前必须把 `mcp` 装到 `yhwang` 可导入的位置（`pip install --break-system-packages --user mcp==2.3.0`），否则 Hermes 拉起 `robot_tools/mcp_server.py` 会因 `ModuleNotFoundError` 立刻退出。**改完必须用将要运行 Hermes 的那个身份复验导入。**
- MCP 客户端（SDK 的 stdio client）会构造**过滤过的子进程环境**，会剥掉 `PYTHONPATH` / `AMENT_PREFIX_PATH` 等 ROS overlay 变量，导致子进程 `import rclpy` 失败。`mcp_server.py` 的对策是在检测不到 `rclpy` 时自动 re-exec：

```
bash -c "source /opt/ros/jazzy/setup.bash && source /home/yhwang/fyp/ws/install/setup.bash && exec python3 \"$@\""
```

（用**非登录** shell，避免 profile 脚本往 stdout 打印内容污染 JSON-RPC 线缆。想要禁用这一行为可设 `ROBOT_TOOLS_NO_REEXEC=1`。）

- 另一个 MCP 相关坑：工具内部（尤其 `pick_place` 的动作辅助函数）会往 stdout 打印。mcp 2.x 的 stdio 传输会把 fd 1 复制到私有句柄并把 fd 1 改指向 stderr，所以这些打印不会污染协议流；服务端另加了一层 `_stray_print_guard()` 兜底。

---

## 8. 渲染与帧率（重要）

- WSL 内 Gazebo **无法 GPU 渲染**：Vulkan 只暴露 `llvmpipe`（Mesa 软件光栅化）。
- 曾经尝试让 EGL/GL 走 Intel 集显（`GALLIUM_DRIVER=d3d12`），实测**更慢**：
  - llvmpipe（软件）：`/rgbd/image` 平均 **12.655 Hz**（min 6.844 / max 15.156，55 样本）→ 当前采用
  - D3D12（集显）：平均 **5.723 Hz**（min 3.836 / max 9.512，51 样本）→ 已弃用
- 结论：固定使用 640×480 + 软件渲染，约 12.7 Hz；相机 `update_rate` 为 15 Hz。
- 因此**不要**在文档或代码里假设有可用的 GPU 加速；渲染性能相关的判断一律以实测 `ros2 topic hz /rgbd/image` 为准。

---

## 9. 相机话题与取帧

三个话题必须**同时**桥接，缺任何一个都会让下游失败：

| 话题 | 类型 | 用途 |
|---|---|---|
| `/rgbd/image` | `sensor_msgs/msg/Image` | RGB 分割 |
| `/rgbd/depth_image` | `sensor_msgs/msg/Image` | 深度反投影 |
| `/rgbd/camera_info` | `sensor_msgs/msg/CameraInfo` | 内参 K |

- `tools/save_frame.py` 取 RGB + 深度出图；`perception/locate_object.py` 还需要 `camera_info`。
- 桥接命令（GZ→ROS 单方向）：

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo
```

- 注意 `tools/save_frame.py` 的固定输出路径是 `logs/phase0b_rgb.png` / `logs/phase0b_depth.png`：任何调用它的脚本都会**覆盖**这两个文件。阶段 C-1 的 `robot_tools/tools.py` 因此刻意不调用 `pick_place.attempt()`，而是自己镜像同样的步骤序列，把帧写到 `data/frames/`。

---

## 10. Gazebo 的无害噪音

启动日志里会出现：

```
[Err] [Physics.cc:960] Failed to find plugin [gz-physics-ode-plugin]
```

这是**无害噪音**：当前世界使用 DART 物理引擎，ODE 插件未安装不影响任何功能与结果。不要为了"消掉这条报错"去装 ODE 插件。

另一类无害告警来自 move_group 启动：

```
Failed to load sensor: `default_sensor` of type: `occupancy_map_monitor/DepthImageOctomapUpdater`
Failed to load sensor: `kinect_depthimage` of type: `occupancy_map_monitor/DepthImageOctomapUpdater`
```

octomap 深度图更新器加载失败，属于既有配置问题；规划与执行正常，可忽略。

---

## 11. 时钟与观测时效

- 整条链路使用仿真时间（`use_sim_time:=true`）。`/clock` 未就绪前 ROS 时间为 0。
- `locate_object.py` 会校验帧时间戳与 ROS 当前时间的偏差，超过 `STALE_SEC = 1.0 s` 直接判 `stale`。
- 工具层用**观测戳**防止执行过期计划：计划登记时记录观测时间，观测超过 `PLAN_STALE_SEC = 30 s` 时 `execute_plan` 直接拒绝（`reason=stale_plan`）。
- 排查问题时注意：墙钟与仿真钟的 skew 会出现在日志里（例如 `[clock] ros_now=226.617 skew=0.170`）。

---

## 12. 常用 SOP

### 起栈

```bash
bash /home/yhwang/fyp/robot_tools/phaseC1_up.sh    # sim(90s) + bridge(10s) + move_group(30s)
```

### 停栈（保留 daemon）

```bash
bash /home/yhwang/fyp/robot_tools/phaseC1_down.sh
```

### 健康检查

```bash
ros2 topic list | grep -E "rgbd|joint_states|clock"
ros2 service list | grep plan_kinematic_path
timeout 30 ros2 topic hz /rgbd/image --window 40      # 期望 ~12.7 Hz
```

### 单次定位

```bash
python3 -u /home/yhwang/fyp/perception/locate_object.py
# stdout: 一行 JSON;stderr: [extrinsics] / [clock] 诊断
```

### 工具层回归

```bash
cd /home/yhwang/fyp
python3 -u robot_tools/test_tools.py       # 30 项检查
python3 robot_tools/mcp_list_tools.py      # MCP 协议级验证
```

---

## 13. 已知未解决项

- 放置阶段立柱倒伏导致放置成功率低（列为阶段 D 改进项）。
- HOME 位红夹爪饰件遮挡 `(-0.02, -0.28)` 角落。
- 抓取高度窗口仅 ±2 mm。
- DashScope/Qwen-VL 链路未实测（缺少 API key），见 `docs/hermes_setup.md`。
