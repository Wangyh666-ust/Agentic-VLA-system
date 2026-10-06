# FYP 第一阶段交付:SO-ARM100 桌面抓放仿真链路

本目录（Windows 侧 `D:\FYP\First_Phase\`）是第一阶段的交付物：README、启动说明、依赖清单、接口文档与全部运行证据日志。
项目代码在 WSL 内 `/home/yhwang/fyp/`，Windows 侧只承载文档与证据（日志/截图落盘）。

---

## 一、项目简介

第一阶段的目标是在纯仿真环境里搭出一条"自然语言 → 抓取执行 → 结果验证"的完整链路的底座：
用户指令交给 Hermes，由 VLM 理解画面与意图，再调用专用工具完成"感知定位 → 几何计算 → 抓取候选 → MoveIt 2 规划 → Gazebo 物理执行 → 结果判定 → 必要时重试"。
本阶段不接真实机械臂、不训练模型，所有环节都要求可独立测试；感知、规划、控制、任务四层接口彼此独立，以便后续迁移到 SO-ARM101 真机。
当前进度：阶段 0、A、B、C-1、C-2、D 均已完成并有日志证据。**阶段 C-2** 打通 Hermes（qwen3-vl-plus）+ MCP 工具层全链路，自然语言抓放任务 `verify_task` 实测成功（`task_success: true`，dxy=17.5 mm）。**阶段 D**（先于 C-2 完成）做了 D-1 / D-2 重试对照实验：换档真正生效 + 重试前复位再感知 + 放置失败纳入重试，成功来自放置重试，代价是平均 1514 s/成功（耗时约 ×2.2）。

---

## 二、架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Windows 11 主机 (RTX 5070, 24 GB 内存)                                       │
│   D:\FYP\First_Phase\   本交付目录:README / docs / logs(证据与截图落盘)      │
│   WSL 侧日志通过 /mnt/d/FYP/First_Phase/logs 直接写回 Windows                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ wsl.exe -d Ubuntu   (显示与人机入口在 Windows)
┌───────────────────────────────▼──────────────────────────────────────────────┐
│ WSL2 Ubuntu 24.04.1 LTS  (内存 18 GB, mirrored 网络, /dev/dxg 可见但不用于渲染)│
│                                                                              │
│   Gazebo Harmonic 8.15.0  ──  worlds/tabletop_v6.sdf                         │
│      │   gz-ros2-control 1.2.20 / ros_gz_bridge 1.0.24                       │
│      ├── so_arm_100  5-DOF 臂 + 平行夹爪 (commit 789b6b2)                    │
│      ├── RGB-D 相机 640×480 @15 Hz → /rgbd/image /rgbd/depth_image           │
│      │                                /rgbd/camera_info                      │
│      └── 渲染:llvmpipe CPU 软渲染,实测约 12.7 Hz                             │
│              (D3D12 走 Intel 集显只有 5.7 Hz,已弃用)                          │
│                                                                              │
│   MoveIt 2 2.12.4 (IKFast 求解器 + OMPL) ── /plan_kinematic_path             │
│                          ↕ ros2_control (arm_controller / gripper_controller)│
│                                                                              │
│   perception/locate_object.py   HSV 红分割 + 深度反投影 + base_link 外参      │
│                     │                                                        │
│   robot_tools/tools.py   9 个工具(坐标 base_link,米,弧度)                   │
│                     │                                                        │
│   robot_tools/mcp_server.py   MCP stdio 服务(mcp 2.3.0,MCPServer)            │
│                     │                                                        │
│   Hermes v0.21.5(qwen3-vl-plus) ──→ Qwen-VL(阶段已实测端到端跑通)           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、目录结构

### WSL 侧 `/home/yhwang/fyp/`（代码与运行环境）

| 路径 | 内容 |
|---|---|
| `ws/` | colcon 工作区。`ws/src/SO-100-arm/` 为 brukg/SO-100-arm（commit `789b6b2`），含 `so_arm_100_description`、`so_arm_100_bringup`、`so_arm_100_moveit_config`、`so_arm_100_5dof_arm_ikfast_plugin` 四个包；`ws/install/setup.bash` 是每次必须 source 的 overlay |
| `worlds/` | `tabletop.sdf` 与 `tabletop_v2..v6.sdf` 六个场景；当前使用 `tabletop_v6.sdf`（立柱 + 相机） |
| `perception/` | `locate_object.py`（视觉定位 CLI，输出一行 JSON）、`gate_check.py`、`eval_closed_loop.py` |
| `robot_tools/` | `tools.py`（9 个工具的实现）、`mcp_server.py`（MCP stdio 包装）、`mcp_list_tools.py`（协议级验证）、`test_tools.py`（独立测试）、`phaseC1_up.sh` / `phaseC1_down.sh`（一键起停） |
| `tools/` | 阶段 A/B 的底层原语 `pick_place.py`，相机取帧 `save_frame.py`，以及各类诊断/探针脚本与 `phaseB_up_*.sh` 启动脚本 |
| `config/` | `camera_extrinsics.yaml`（相机外参 `T_base_opt`） |
| `data/frames/` | 运行期观测帧落盘目录 |
| `logs/` | WSL 侧运行日志（`robot_tools.log`、`phaseC1_*.log` 等） |

### Windows 侧 `D:\FYP\First_Phase\`（本交付目录）

```
First_Phase/
├── README.md              本文件
├── .env.example           密钥占位模板(不含真实值)
├── docs/                  环境、接口、外参、Hermes、依赖、运行手册(runbook)文档
├── logs/                  阶段 0/A/B/C 的证据日志与截图 + INDEX.md
└── working_requirement.txt 原始需求文本(只读参考)
```

---

## 四、环境与依赖

| 组件 | 版本 |
|---|---|
| 宿主 | Windows 11；RTX 5070；24 GB 内存 |
| WSL | WSL2，kernel 6.18.33.2-microsoft-standard-WSL2，内存上限 18 GB（`.wslconfig`，mirrored 网络） |
| Linux | Ubuntu 24.04.1 LTS |
| Python | 3.12.3（Hermes 自带 venv 用 3.14.7） |
| ROS 2 | Jazzy，`ros-jazzy-desktop` 0.11.0 |
| 仿真器 | Gazebo Harmonic，`gz sim` 8.15.0（`gz-harmonic` 1.0.0-1~noble） |
| ROS↔Gazebo | `ros-gz` 1.0.24、`gz-ros2-control` 1.2.20 |
| 规划 | MoveIt 2 2.12.4 |
| 机械臂模型 | SO-100-arm（brukg），commit `789b6b2` |
| 工具层 MCP | Python `mcp` 2.3.0（`MCPServer`） |
| Agent | Hermes Agent v0.21.5（Python 3.14.7，装在 `/home/yhwang/.hermes`） |

详细安装顺序、固定版本与逐步验证命令见 `docs/dependencies.md`。

---

## 五、启动步骤

每条命令都标注了执行环境；**所有 ROS 命令都在 WSL 内执行**，Windows 侧只负责进入 WSL 与查看落盘文件。
下文 `[Windows]` = 在 Windows 终端（PowerShell 或 git-bash）执行；`[WSL]` = 在 WSL Ubuntu shell 内执行。

> **快速开始**：只想尽快把仿真栈起起来、再跑一条自然语言抓放任务，请直接照抄 **`docs/runbook.md`**（五节：启动 / 跑任务 / 复跑评测 / 收尾 / 故障表）。下面的 0~8 步是展开版说明。

### 0. 进入 WSL

```bash
# [Windows] 打开 WSL 会话
wsl.exe -d Ubuntu
```

### 1. 每次开新 shell 的第一件事（source 顺序不能颠倒）

```bash
# [WSL]
source /opt/ros/jazzy/setup.bash
source /home/yhwang/fyp/ws/install/setup.bash
```

### 2. 启动仿真（Gazebo + ros2_control）

```bash
# [WSL] 首次就绪约需 90 秒
ros2 launch so_arm_100_bringup gz.launch.py world:=/home/yhwang/fyp/worlds/tabletop_v6
```

### 3. 启动相机桥（三个话题必须全桥，缺一不可）

```bash
# [WSL] 另开一个 shell（记得先 source 第 1 步的两个 setup.bash）
ros2 run ros_gz_bridge parameter_bridge \
  /rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo
```

### 4. 启动 move_group

```bash
# [WSL] 另开一个 shell（同样先 source）；首次就绪约需 30 秒
ros2 launch so_arm_100_moveit_config moveit.launch.py use_sim_time:=true rviz:=false
```

### 5. 验证工具层（9 个工具，不需要 API key）

```bash
# [WSL] 另开一个 shell
cd /home/yhwang/fyp
python3 -u robot_tools/test_tools.py      # 期望:FINAL ... 30 checks, 30 passed
python3 robot_tools/mcp_list_tools.py     # 期望:COUNT 9 / MATCH True / CALL_VERDICT ok
```

### 6. 一键起停（等价于上面的 2~4 步，日志写入 logs/phaseC1_*.log）

```bash
# [WSL]
bash /home/yhwang/fyp/robot_tools/phaseC1_up.sh     # 起 sim + bridge + move_group
bash /home/yhwang/fyp/robot_tools/phaseC1_down.sh   # 停栈,保留 ros2 daemon
```

### 7. 最小演示任务（真值基线抓放，阶段 A/B 用过的入口）

```bash
# [WSL] 前提:第 2、3、4 步已起；该脚本会把观测帧写到 Windows 侧 logs/
cd /home/yhwang/fyp
python3 -u tools/pick_place.py        # 阶段 A 真值基线:抓取 -> 搬运 -> 放置 -> 判定
python3 -u perception/eval_closed_loop.py   # 阶段 B 视觉闭环:10 个随机位评估
```

### 8. 查看证据

```bash
# [Windows] 日志与截图都在本目录
ls /d/FYP/First_Phase/logs/
```

> 阶段 C-2（Hermes + Qwen-VL）已实测跑通：日常起停/跑任务/复跑评测/故障处置请照抄 **`docs/runbook.md`**；接入细节与全部实测结论见 `docs/hermes_setup.md`。

---

## 六、已完成阶段与证据

| 阶段 | 结论 | 证据文件（`logs/`） |
|---|---|---|
| 阶段 0 | 最小验证通过：环境版本探明、机械臂可动、RGB-D 相机可出图 | `phase0a_install.log`、`phase0b_arm.log`、`phase0b_camera.log`、`phase0b_rgb.png`、`phase0b_depth.png` |
| 阶段 A | 真值抓放基线：共 20 次真实尝试；首次完整成功出现在 `phaseA2c_pickplace_run2.log` 第 1 次尝试（放置偏差 dxy=7.3 mm，`success=True`）。失败模式分类为闭空 / 推开 / 滑落 / 碰倒 / 放置偏差 | `phaseA1_moveit.log`、`phaseA2a_gate.log`、`phaseA2b/c/d/e/f_*`、`phaseA2c_pickplace_run2.log` |
| 阶段 B | RGB-D 视觉闭环：HSV 分割 + 深度反投影 + 外参，8 位置复测定位误差 mean 2.1 mm / median 2.0 mm / max 3.9 mm；10 个随机位 carried 8/10、place 1/10；实现首次端到端视觉抓放 | `phaseB_extrinsics_gate.log`、`phaseB_gate.log`、`phaseB_closedloop.log`、`phaseB_*.png` |
| 阶段 C-1 | robot_tools 9 工具模块 + 独立测试 30/30 PASS + MCP stdio `tools/list` 验证（MATCH True，真实 `tools/call` 往返） | `phaseC0_hermes_install.log`、`phaseC1_tools_test.log`、`phaseC1_mcp_list.log`、`phaseC1_mcp_list_cleanenv.log` |
| 阶段 C-2 | Hermes（`qwen3-vl-plus`，DashScope 大陆站）+ MCP `robot_tools` 全链路打通：9 工具注册并 `mcp test` 通过；自然语言抓放任务端到端实测 **`verify_task` 成功**（`task_success: true`、dxy=17.5 mm，容差 30 mm，轮询约 152 秒）。图像链路走 `vision_analyze` native fast path（像素实测进入主模型） | `phaseC2_model.log`、`phaseC2_mcp.log`、`phaseC2_stack.log`、`phaseC2_e2e_attempt3.log`、`phaseC2_e2e_attempt3_trace.txt`、`phaseC2_vision.log`、`phaseC2_vision_verdict.log` |
| 阶段 D | D-1 / D-2 重试对照实验（**先于 C-2 完成**）：换档真正生效 + 重试前复位再感知 + 放置失败纳入重试。换档真生效是前提（D-1 是假换档 → 重试原地踏步）；重试前复位把再感知误差从 45 mm 压回 3.8 mm；放置重试是唯一成功来源，代价平均 1514 s/成功（耗时约 ×2.2）；重观察回退一次都没触发（未修）。`tools.py:675` 改一行后回归 30/30 | `phaseD1_retry.log`、`phaseD1_trials.csv`、`phaseD1_metrics.csv`、`phaseD1_*.png`、`phaseD2_retry.log`、`phaseD2_trials.csv`、`phaseD2_metrics.csv`、`phaseD2_trials_pos1_5.csv`、`phaseD2_metrics_pos1_5.csv`、`phaseD2_regression_tools_test.log`、`phaseD2_*.png` |

阶段 A 的根因分析：指尖下探时存在约 22–27 mm 的"地板"残留，2 cm 方块在结构上难以被真正握住；改用 0.025×0.025×0.06 m 的立柱后实现了真实环握。

每个日志文件的阶段、内容与结论逐条列在 `logs/INDEX.md`。

---

## 七、已知限制

1. **放置成功率低**：立柱释放时容易倒伏，阶段 B 的 place 仅 1/16（10 次随机位评估中 1 次成功）。阶段 D-2 已把这一定量化并纳入重试：**放置失败纳入重试是唯一的成功来源**，代价是耗时约 ×2.2（平均 1514 s/成功）；根因在释放几何未调，列为后续改进项。
2. **HOME 位遮挡**：机械臂停在 HOME 时，红夹爪饰件会遮挡相机 `(-0.02, -0.28)` 角落，该位置的观测会被 `implausible_z` 门拒绝。
3. **伪目标风险**：夹爪上的红色饰件可能被 HSV 分割误认为目标，现有 `implausible_z` 高度带门（顶面 z 在 0.035–0.10 m）用于排除。
4. **抓取窗口窄**：候选抓取高度只有 ±2 mm 的有效区间，姿态/高度稍偏即失败。
5. **ros2 daemon 必须常驻**：受 mirrored 网络的 loopback 超时怪癖影响，daemon 停掉后 `ros2` CLI 会长时间卡住；teardown 脚本刻意保留 daemon。
6. **代理时通时断**：apt/shell 已配 `127.0.0.1:7897`；代理失效时需 `unset` 相关变量直连。
7. **WSL 内 Gazebo 无法 GPU 渲染**：Vulkan 只有 `llvmpipe`，D3D12 走集显反而更慢，因此固定使用 CPU 软渲染，640×480 RGB-D 约 12.7 Hz。
8. **`mcp` 曾装在 root 的用户目录（阶段 C-2 已解决）**：`mcp 2.3.0` 原先只位于 `/root/.local/lib/python3.12/site-packages`，普通用户 `yhwang` 下 `import mcp` 会失败（阶段 C-1 的 MCP 验证因此在 root 身份下完成）。阶段 C-2 已确认 `yhwang` 可正常导入（实际加载路径 `/home/yhwang/.local/lib/python3.12/site-packages`），`hermes mcp test robot_tools` 通过。
9. **MCP 工具结果里的图像不进模型上下文**：Hermes 会把 MCP 返回的图像块**无条件**降级为 `MEDIA:<path>` 文本标签（源码 `tools/mcp_tool_content.py`），模型从 MCP 结果拿不到像素。**要让 agent 真正"看图"，必须在任务 prompt 里让它调内置 `vision_analyze`（并保持 `model.supports_vision: true`）**，详见 `docs/hermes_setup.md` §2.4。
10. **模型固定 `qwen3-vl-plus`**：`qwen-vl-max` 与 `qwen-vl-plus` 在 DashScope compatible-mode 下**不支持原生 function calling**（实测 `finish_reason=stop`、响应无 `tool_calls`，模型会把工具调用写成纯文本导致 agent 完全不行动），因此不能换回这两个型号。
11. **`verify_task` 判据不含 `carried`（阶段 D 已记录，未修）**：成功判据只有 `dxy < 0.03 m` 与 `z < 0.04 m`，没有要求 `carried=True`，存在"物体被撞进目标区、却从未被真正抓起"也被判成功的误判风险；建议把 `carried=True` 加进判据。
12. **重观察回退是死代码（阶段 D 已记录，未修）**：D-1 / D-2 设计的"重观察回退"分支一次都没有触发——回退逻辑不可达，且"观测过期闸"存在语义漏洞；需要重写触发条件才算真正生效。
13. **`retry_runner` 两份副本并存（建议合并）**：`robot_tools/retry_runner.py` 与 `robot_tools/retry_runner_d2.py` 逻辑重叠，建议合并为带 `--policy` 开关的单脚本，避免两套重试策略各自漂移。

---

## 八、阶段 C-2 与阶段 D（均已实测完成）

### 阶段 C-2：Hermes + Qwen-VL 全链路（**已完成，2026-10-04**）

- 已完成：DashScope 大陆站（`provider: alibaba-cn`）+ `qwen3-vl-plus` 接入 Hermes；`robot_tools` 以 MCP 方式注册（9 工具）；图像链路走内置 `vision_analyze` 的 native fast path；自然语言抓放任务端到端实测 **`verify_task` 成功**（`task_success: true`，dxy=17.5 mm）。
- 日常操作（起栈 / 跑任务 / 复跑评测 / 收尾 / 故障处置）照抄 `docs/runbook.md`；一次性任务的推荐 prompt 模板见 `docs/hermes_setup.md` §2.5 与 `docs/runbook.md` 第二节。
- 接入细节、实测结论与踩坑记录见 `docs/hermes_setup.md`（**已全部落锤为实测口径，无遗留规划项**）。

### 阶段 D：失败反馈与重复测试（**已完成，先于 C-2**）

- 完成 **D-1 vs D-2 严格对照实验**：D-1 只加"有次数上限的换档重试"，D-2 在其上做三处策略修正。
- **D-2 的三处修正**：① 换档真正生效（D-1 是假换档，重试等于原地踏步）；② 重试前先复位、再重新感知；③ 把放置失败纳入重试范围。
- **四条实测结论**：
  1. 换档必须真生效，否则重试只是原地踏步；
  2. 重试前复位，把再感知误差从 45 mm 压回 3.8 mm；
  3. 放置失败纳入重试是唯一的成功来源，代价是耗时约 ×2.2（平均 1514 s/成功）；
  4. 重观察回退一次都没触发（死代码 + 观测过期闸语义漏洞，**未修**，见 §七 第 12 条）。
- D-2 另改了 `robot_tools/tools.py:675` 一行（执行改用 plan 登记的 grasp z），改后独立测试 **30/30 通过**（`phaseD2_regression_tools_test.log`）。
- 样本量小需注意：位置 1–5 子集（`phaseD2_*_pos1_5.csv`）里 ±1 个位置约等于 12.5 个百分点，百分比结论不宜过度解读。
- 其余未修项见 §七 第 11~13 条；16 个产物已逐条登记在 `logs/INDEX.md`。

---

## 九、团队协作说明

- **两侧分工**：Windows 侧 `D:\FYP\First_Phase\` 是交付与证据区（文档、日志、截图），WSL 侧 `/home/yhwang/fyp/` 是代码与运行区。改代码在 WSL 侧改，改文档在本目录改。
- **密钥纪律**：仓库内只允许出现占位符（`.env.example`）。真实 key 只放 WSL 侧未纳入版本控制的 `.env`，任何日志和文档都不得出现真实密钥。
- **证据纪律**：任何"跑通了"的说法都必须能指到 `logs/` 里的原始日志或截图；`logs/INDEX.md` 记录每个文件的阶段、内容与结论，新增日志请同步登记。
- **目录纪律**：日志与截图统一写到 `/mnt/d/FYP/First_Phase/logs/`（即本目录的 `logs/`），避免证据散落在 WSL 内部。
- **环境坑**：git-bash 的 `$` 提前展开、`pkill` 必须写方括号、pip 需 `--break-system-packages`、`save_frame.py` 依赖三个相机话题全通——完整清单见 `docs/environment.md`。

---

## 十、VLA 路线(ManiSkill 3 + SmolVLA + Hermes,2026-10-05 起)

第一阶段后期,底层执行从 MoveIt 脚本改为学习得到的 VLA 策略,规划见 `changing_to_VLA_structure.md`。代码与运行环境在 WSL 的 `/home/yhwang/fyp/vla/`;WSL 磁盘已迁到 `D:\WSL\Ubuntu`。

| 阶段 | 内容 | 结果 | 文档 / 证据 |
|---|---|---|---|
| 复核 | 旧 VLA 评测的问题清单:专家 TCP 偏移、评测集落在 IK 不可达区、视频压缩失真、未设种子等 | v3 之前的成功率不再引用 | `docs/review_2026-10-05.md` |
| C 基线重建 | 抓取任务:专家 v3、300 条无损示范、SmolVLA 训练 20k 步 | dev 50/50;test 97/100(Wilson [91.6, 99.0]%) | `logs/phaseVLA_v3test_s20k_*` |
| D 服务化 | 常驻 VLA 执行服务 + MCP 薄客户端(7 个工具)接入 Hermes;`inspect_scene` 结构化感知 | 感知准确率:lifted 99.9%,on_blue 99.5% | `docs/vla_service_design.md` |
| E 组合任务 | 堆叠环境、400 条示范、多指令策略(Grasp / Place / Put);条件 E1–E4 对照与受扰恢复 | 门槛:G 86%、E1 84%、E2 70%;pilot v2 掉落扰动下 E4r 80%,E2 20% | `docs/phaseE_design.md` |

启动方式(WSL):

```bash
# 常驻服务(SmolVLA 后端,堆叠环境)
bash /home/yhwang/fyp/vla/service/run_service.sh --backend smolvla --env stack \
  --ckpt /home/yhwang/fyp/vla/outputs/smolvla_v4_b16_s30k/checkpoints/030000/pretrained_model --port 8765

# 条件对照(另开一个 shell)
cd /home/yhwang/fyp/vla/experiments
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  /home/yhwang/fyp/vla/venv/bin/python run_conditions.py --condition E4r \
  --seed-start 3100 --count 20 --perturb drop_red_mid --tag demo
/home/yhwang/fyp/vla/venv/bin/python summarize.py results/demo.jsonl
```

注意事项:
- WSL 的登录 shell 会注入 http_proxy;凡是访问 127.0.0.1 的命令都要去掉代理(runner 和 MCP 客户端已在代码里绕过代理)。
- 长任务需要保持一个 `wsl.exe -d Ubuntu --exec sleep infinity` 会话,否则虚拟机空闲后会关闭,并杀掉其中所有进程。
- Hermes 配置里 vla_tools 固定连接 8765 端口;E4 条件运行时,服务必须开在 8765。
