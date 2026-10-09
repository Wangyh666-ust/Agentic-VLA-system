# 统一子任务辅助实验使用说明

本文档说明 `scene_demo` 中统一子任务辅助实验的工作方式、文件职责、启动方法、验证边界与当前状态。内容仅覆盖本机已验证事实，不包含训练或微调承诺。

## 1. 工作流

Hermes 负责根据场景选择对象、能力与顺序，不在每个动作上被调用。真正的执行由本地 harness 驱动：对每个子任务重新读取最新状态，统一入口再依据操作类型、形状、位姿、尺寸以及已验证标定证据选择 `prepare` 或 `grasp`。随后由同一个 SmolVLA 继续执行。原始 `release_verified` 评分保持不变。

对象并不各自拥有独立的 VLA。开发分工为：GPT 负责设计与复核，DeepSeek 负责实际核心实现，本地执行者负责测试与打包。

## 2. 文件职责

| 文件 | 职责 |
| --- | --- |
| `subtask_context.py` | 构建上下文，选择策略 |
| `subtask_geometry.py` | 读取原生 MuJoCo 几何、持有状态、障碍物，并生成指纹 |
| `subtask_preparation.py` | 空手退出障碍物、方向调整、带检查接近 |
| `subtask_assist_service.py` | 每个子任务的统一入口 |
| `subtask_assist_profiles.json` | 仅保存已验证的酒瓶碰撞指纹侧握 profile |

低宽碗当前使用 `prepare` 加原始 VLA，没有碗专用抓取控制器。酒瓶标定适配器仍绑定已验证的旧能力接口；其他物体即使形状相似，也尚未验证抓取。未知观测、外来持有物、碰撞以及受保护对象移动都会触发门控停止。

## 3. 验证边界

需要区分：准备到达、稳定抓取、最终放置。准备阶段最多 200 个动作，全部计入子任务预算 500。准备目标门控为位置误差不超过 5 mm、方向误差不超过 0.05 rad，并连续五次真实动作后采样。它只证明准备阶段，不改变原始 `release_verified` 完成判定。

本项目不进行训练或微调。路径采样会检查夹爪碰撞几何；完整机械臂只做实际接触监测，不保证完整臂运动。平台为 LIBERO/MuJoCo Franka；当前几何来自仿真，真实 RGBD 集成尚未实现。

## 4. 手动独立启动

以下为可选的实验启动命令，本轮尚未启动该页面。复用已有共享 venv、已有 EGL、LIBERO 与离线环境。路径是现有本机环境，不构成新机器一键承诺。

服务终端：

```bash
cd /mnt/d/FYP/First_Phase/scene_demo
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config LD_LIBRARY_PATH=/usr/lib/wsl/lib
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/yhwang/fyp/vla/lerobot/src:/mnt/d/FYP/First_Phase/scene_demo
export SCENE_SERVICE_URL=http://127.0.0.1:8782
export SCENE_RUNS_DIR=/mnt/d/FYP/First_Phase/tmp/unified_demo_runs
export HERMES_HOME=/mnt/d/FYP/First_Phase/tmp/subtask_assist_hermes_home
/home/yhwang/fyp/libero_demo/venv/bin/python subtask_assist_service.py --port 8782 --run-root "$SCENE_RUNS_DIR" --calibration-profiles /mnt/d/FYP/First_Phase/scene_demo/subtask_assist_profiles.json
```

另一终端重复同样的环境导出，再执行：

```bash
export SCENE_SERVICE_URL=http://127.0.0.1:8782
export SCENE_RUNS_DIR=/mnt/d/FYP/First_Phase/tmp/unified_demo_runs
export HERMES_HOME=/mnt/d/FYP/First_Phase/tmp/subtask_assist_hermes_home
/home/yhwang/fyp/libero_demo/venv/bin/python app.py --port 8082
```

其中 EGL、离线与 `PYTHONPATH` 可复用上一节的公共导出。新实验页面为 `http://127.0.0.1:8082`；当前默认 `8081` 未切换。CLI 会拒绝生产端口 `8767`/`8081`。

## 5. Hermes 私有克隆

私有 Hermes 克隆仅在本机准备，不通过 Git 分发。新机器需要独立 Hermes home，并设置 `scene_tools.env.SCENE_SERVICE_URL=http://127.0.0.1:8782` 以及指向仓库 `mcp_server.py` 的 MCP 路径。私有配置和凭据不随实验结果归档。

## 6. 当前状态

软件67项新测试与77项回归通过。四次对照完成，两条准备均调整朝向超时，未进入VLA；详见实验报告。

- [实现计划](plans/2026-10-09-subtask-assist.md)
- [实验报告](results/2026-10-09-subtask-assist/README.md)
