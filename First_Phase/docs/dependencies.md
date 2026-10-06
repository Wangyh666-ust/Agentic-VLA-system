# 依赖版本固定清单与安装顺序

本文件记录本机（WSL2 Ubuntu 24.04）实际使用的版本与安装顺序，并给出每一步的**验证命令**。
标记含义：**[已验证]** = 有本机实测输出（见 `logs/` 或下方直接给出的命令输出）；**[标准流程]** = 上游官方安装步骤，本项目安装时未逐条留存输出，重复搭建时按此执行并逐条核对。

---

## 1. 版本固定清单

### 系统

| 组件 | 固定版本 | 验证命令 |
|---|---|---|
| Ubuntu | 24.04.1 LTS (noble) | `lsb_release -d` |
| WSL 内核 | 6.18.33.2-microsoft-standard-WSL2 | `uname -r` |
| Python | 3.12.3（`/usr/bin/python3`） | `python3 -V` |
| WSL 内存上限 | 18 GB（`.wslconfig`） | `free -g \| head -2` → `total 17` |

### ROS / 仿真 / 规划 **[已验证：`logs/phase0a_install.log`]**

| 包 | 版本 |
|---|---|
| `ros-jazzy-desktop` | 0.11.0-1noble.20260905.070740 |
| `gz-harmonic` | 1.0.0-1~noble |
| Gazebo `gz sim` | 8.15.0 |
| `ros-jazzy-ros-gz` | 1.0.24-1noble.20260905.090616 |
| `ros-jazzy-ros-gz-bridge` | 1.0.24-1noble.20260905.070149 |
| `ros-jazzy-ros-gz-image` | 1.0.24-1noble.20260905.072745 |
| `ros-jazzy-ros-gz-interfaces` | 1.0.24-1noble.20260902.041547 |
| `ros-jazzy-ros-gz-sim` | 1.0.24-1noble.20260905.085151 |
| `ros-jazzy-ros-gz-sim-demos` | 1.0.24-1noble.20260905.085920 |
| `ros-jazzy-gz-ros2-control` | 1.2.20-1noble.20260905.083607 |
| `ros-jazzy-moveit` / `ros-jazzy-moveit-core` | 2.12.4 |
| `ros-jazzy-ros2-control` | 4.48.0 |
| `ros-jazzy-ros2-controllers` | 4.42.1 |

### 项目代码 **[已验证]**

| 项 | 值 | 验证命令 |
|---|---|---|
| 机械臂仓库 | `https://github.com/brukg/SO-100-arm.git` | `git -C ws/src/SO-100-arm remote -v` |
| commit | `789b6b2c32819d792105b068a4c70c32767d4e46`（`789b6b2`, 2026-08-05, "Merge pull request #20 … ikfast-plugin"） | `git -C ws/src/SO-100-arm log -1 --format=%H` |
| 包 | `so_arm_100_description`、`so_arm_100_bringup`、`so_arm_100_moveit_config`、`so_arm_100_5dof_arm_ikfast_plugin` | `ls ws/install/` |
| 工作区 | `/home/yhwang/fyp/ws`（overlay 入口 `ws/install/setup.bash`） | — |

### 工具层与 Agent

| 项 | 版本 | 验证命令 |
|---|---|---|
| Python `mcp` | **2.3.0** | `python3 -m pip show mcp` |
| `mcp-types` | 2.3.0 | `python3 -m pip show mcp-types` |
| `anyio` | 4.15.1 | `python3 -m pip show anyio` |
| `numpy` | 1.26.4（root 环境）/ 由 ROS 提供系统版本 | `python3 -c "import numpy; print(numpy.__version__)"` |
| `PyYAML` | 6.0.1 | `python3 -c "import yaml; print(yaml.__version__)"` |
| Hermes Agent | v0.21.5+6662.g343500b (2026.9.24)，upstream `343500b3`，Python 3.14.7，OpenAI SDK 2.24.0 | `hermes --version` |

> **⚠ 重要偏差（已验证）**：`mcp 2.3.0` 被装到了 **root 的 user site-packages**
> （`/root/.local/lib/python3.12/site-packages/mcp`，安装时间 2026-10-04 04:47），
> 普通用户 `yhwang` 下 `python3 -c "import mcp"` **会 ModuleNotFoundError**，root 下正常。
> 阶段 C-1 的 MCP 验证因此是在 root 身份下完成的（同时期 WSL `logs/phaseC1_*.log`、`robot_tools.log` 均为 `root:root`）。
> 阶段 C-2 前必须解决：把 `mcp` 装到 `yhwang` 可导入的位置，否则 Hermes（以 `yhwang` 身份）拉起
> `robot_tools/mcp_server.py` 会立刻失败。见 §4。

---

## 2. 安装顺序

顺序不可调换：ROS 源 → desktop → 仿真器 → 桥 → ros2_control → 工作区编译 → Python 工具依赖 → Agent。

### 步骤 1：apt 源 **[标准流程]**

```bash
sudo apt update && sudo apt install -y curl software-properties-common
# 添加 ROS 2 apt 源(Ubuntu 24.04 用 ros2-apt-source 包,不再用 apt-key)
# 具体 URL/版本以 https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html 为准
sudo apt update
```

验证：

```bash
ls /etc/apt/sources.list.d/ | grep -i ros
```

### 步骤 2：`ros-jazzy-desktop` **[标准流程]**

```bash
sudo apt install -y ros-jazzy-desktop
```

验证：

```bash
dpkg -l | grep ros-jazzy-desktop          # 期望 0.11.0-1noble.*
source /opt/ros/jazzy/setup.bash && ros2 --help >/dev/null && echo ROS_OK
```

### 步骤 3：Gazebo Harmonic **[标准流程]**

```bash
sudo apt install -y gz-harmonic
```

验证：

```bash
gz sim --version            # 期望 Gazebo Sim, version 8.15.0
dpkg -l | grep gz-harmonic  # 期望 1.0.0-1~noble
```

### 步骤 4：`ros-gz`（ROS↔Gazebo 桥） **[标准流程]**

```bash
sudo apt install -y ros-jazzy-ros-gz
```

验证：

```bash
dpkg -l | grep ros-jazzy-ros-gz | awk '{print $2, $3}'   # 期望 1.0.24-1noble.*
ros2 pkg list | grep ros_gz_bridge
```

### 步骤 5：`gz-ros2-control` **[标准流程]**

```bash
sudo apt install -y ros-jazzy-gz-ros2-control
```

验证：

```bash
dpkg -l | grep gz-ros2-control     # 期望 1.2.20-1noble.*
```

### 步骤 6：工作区克隆与编译 **[已验证：`ws/log/build_*` 的构建记录]**

```bash
mkdir -p /home/yhwang/fyp/ws/src && cd /home/yhwang/fyp/ws
git clone https://github.com/brukg/SO-100-arm.git src/SO-100-arm
cd src/SO-100-arm && git checkout 789b6b2 && cd /home/yhwang/fyp/ws

colcon build --symlink-install
source install/setup.bash
```

本机实际发生的三次构建（`ws/log/` 目录名即时间戳）：

| 构建 | 包 | 说明 |
|---|---|---|
| `build_2026-10-03_22-17-09` | `so_arm_100_description`、`so_arm_100_bringup`、`so_arm_100_moveit_config` | 首次全量构建 |
| `build_2026-10-04_00-53-28` | `so_arm_100_5dof_arm_ikfast_plugin` | 补建 IKFast 插件 |
| `build_2026-10-04_00-58-19` | `so_arm_100_5dof_arm_ikfast_plugin` | 修好后重编（≈16.6 s） |

即：**先全量 `colcon build`，再单独编译 IKFast 插件包**。包之间的编译顺序由 `package.xml` 依赖自动决定，不需要手工排序。

验证：

```bash
ls install/                                    # 期望含上述 4 个包目录(另有 COLCON_IGNORE)
ros2 pkg list | grep so_arm_100
source install/setup.bash && ros2 launch so_arm_100_bringup gz.launch.py world:=/home/yhwang/fyp/worlds/tabletop_v6
```

### 步骤 7：Python 工具依赖 `mcp` **[已验证，但装错位置]**

```bash
sudo pip install --break-system-packages mcp==2.3.0
```

- `--break-system-packages` 是必需的：Ubuntu 24.04 的 Python 是 externally-managed，普通 `pip install` 会被拒绝。
- **本机实际效果**：用 `sudo pip` 安装，包落到了 **root 的 user site**（`/root/.local/...`），`yhwang` 无法导入。
  重复搭建时请改为下面任一方式，并核对导入身份：

```bash
# 方式 A:装到 yhwang 可导入的用户目录(推荐)
pip install --break-system-packages --user mcp==2.3.0
# 方式 B:装到系统目录
sudo pip install --break-system-packages mcp==2.3.0   # 注意首次用 sudo 会走 root user site
```

验证（**必须用将要运行 Hermes 的那个身份**执行）：

```bash
python3 -c "import mcp; print(mcp.__version__ if hasattr(mcp,'__version__') else mcp.__file__)"
python3 -m pip show mcp | head -3
```

> 本仓库代码用的是 `mcp 2.3.0` 的 `MCPServer`（`from mcp.server.mcpserver import MCPServer`）。
> 该版本已经**没有** `mcp.server.fastmcp.FastMCP`，旧写法会 `ModuleNotFoundError`。

### 步骤 8：Hermes 安装 **[已验证：`logs/phaseC0_hermes_install.log`]**

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

- 安装目录 `/home/yhwang/.hermes/hermes-agent`，可执行文件 `/home/yhwang/.local/bin/hermes`（PATH 已写入 `~/.bashrc`）。
- 安装器最后会启动交互式 `hermes setup`；本次**未配置模型与 API key**（详见 `docs/hermes_setup.md`）。
- 安装期直连可用（HTTP 200），未使用代理。

验证：

```bash
hermes --version       # 期望: Hermes Agent v0.21.5+6662.g343500b (2026.9.24)
hermes doctor          # 查看依赖与凭据状态
hermes mcp list        # 期望当前输出: No MCP servers configured.
```

---

## 3. 一键整体验证

```bash
# 1) 系统与包版本
lsb_release -d
dpkg -l | grep -E 'ros-jazzy-desktop|gz-harmonic|ros-gz|gz-ros2-control|moveit-core'
gz sim --version

# 2) 工作区
source /opt/ros/jazzy/setup.bash
source /home/yhwang/fyp/ws/install/setup.bash
ros2 pkg list | grep so_arm_100

# 3) 工具层(需要仿真栈已起)
cd /home/yhwang/fyp
python3 -u robot_tools/test_tools.py        # 期望 30 checks, 30 passed
python3 robot_tools/mcp_list_tools.py       # 期望 COUNT 9 / MATCH True

# 4) Agent
hermes --version
```

---

## 4. 待确认 / 风险项

1. **`mcp` 装在 root 用户目录**（已确认现状，见 §1 脚注）。阶段 C-2 之前必须把 `mcp==2.3.0` 装到 `yhwang` 可导入的位置并复验，否则 Hermes 侧 stdio 服务器起不来。
2. **apt 源引导步骤未留存输出**：步骤 1 的 ROS 源添加过程没有日志记录（只有安装后的 `dpkg -l` 输出），重复搭建时按上游文档执行并逐条核对。
3. **代理依赖**：apt/pip 是否走 `127.0.0.1:7897` 视网络而定；代理失效时应 `unset` 相关变量后直连。Hermes 安装器本次是直连成功的。
4. **Gazebo 渲染固定为 llvmpipe 软件渲染**（约 12.7 Hz @640×480）；不要假设有 GPU 加速可用。
5. **`move_group` 的 octomap 更新器报错**（`DepthImageOctomapUpdater` 加载失败）属既有配置问题，不影响规划与执行。
6. `gz-physics-ode-plugin` 缺失是无害噪音（当前用 DART），不要为消错而安装。
