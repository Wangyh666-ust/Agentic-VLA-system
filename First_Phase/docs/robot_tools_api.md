# robot_tools 工具接口契约（阶段 C-1）

实现：`/home/yhwang/fyp/robot_tools/tools.py`（类 `RobotTools`）
MCP 包装：`/home/yhwang/fyp/robot_tools/mcp_server.py`（stdio，`mcp` 2.3.0 的 `MCPServer`）
本文的字段名与语义取自上述两个文件，不是推测。

---

## 1. 全局约定

| 项 | 约定 |
|---|---|
| 长度单位 | **米** |
| 角度单位 | **弧度** |
| 坐标系 | 一律 `base_link`（机器人基座系）；相机光学系仅在感知模块内部使用 |
| 时间戳 | 每个返回都带 `ts`（ROS/仿真秒）与 `ts_wall`（UTC epoch 秒） |
| 返回类型 | 每个工具都返回一个 dict；成功时 `ok=true`、`error_kind=null`、`reason=""` |
| 失败语义 | 失败时 `ok=false`，另有 `error_kind` 与 `reason` 字段 |
| 目标物体 | v1 只有一个目标 `kind = "red_tower"`（0.025×0.025×0.06 m 红色立柱） |
| 地面真值 | `verify_task` / `reset_scene` 会读仿真器真值位姿，字段显式标注 `truth_only_for_eval=true`；真值**只用于评分**，绝不作为感知输入或运动目标 |

### error_kind 四级分类

| error_kind | 含义 |
|---|---|
| `tool_error` | 调用本身失败：入参非法、id 未知、目标描述不支持、计划过期、观测取不到 |
| `plan_failed` | MoveIt 无法规划某个必需位姿 |
| `exec_failed` | 计划已下发但 `/execute_trajectory` 失败，或被 `stop_execution` 中止 |
| `task_failed` | 动作都执行了，但物体没被拿起/放置，或 `verify_task` 判定超差 |

### 关键常量（可在实现中查到）

| 常量 | 值 | 含义 |
|---|---|---|
| `PLAN_STALE_SEC` | 30.0 s | 观测超过此龄期，`execute_plan` 拒绝执行 |
| `PLAN_TOL` | `(0.15, 0.15, 3.14)` | 计划校验的姿态容差（来自阶段 A 第一档） |
| `PRE_Z` | 0.10 | pre_grasp 高度 |
| `GRASP_CANDIDATES` | `[0.010, 0.008, 0.012]` | 抓取高度候选(D-2 起,执行以 plan 登记的 z 为准)（主 + 两个备份） |
| `CARRY_Z_MIN` | 0.05 | 判定"已抬起"的 z 下限 |
| `TOPPLED_Z_MAX` | 0.02 | 判定"倒伏"的 z 上限 |
| `PLACE_TOL_XY` / `PLACE_TOL_Z` | 0.03 / 0.04 | 任务成功判据 |
| `GRIP_OPEN` / `GRIP_CLOSE` | 1.400 / -0.150 | 夹爪开合命令值（方向语义经阶段 A 实测确认） |
| `LOCATE_TIMEOUT_SEC` | 180.0 s | 感知子进程超时 |
| `EXEC_POLL_SEC` | 1.0 s | 状态轮询间隔 |

### 目标描述映射（v1）

`红色立柱`、`红方块`、`红色方块`、`red cube`、`red tower` → 均为 `red_tower`。
匹配时先做小写化并把连续空白折叠，因此 `Red Cube` 与 `red cube` 等价。
未命中的描述返回 `ok=false, error_kind=tool_error, reason=unsupported_target`。

### 典型调用顺序

```
observe_scene → locate_object → propose_grasps → plan_grasp
              → execute_plan → get_execution_status（轮询）
              → verify_task
辅助：reset_scene / stop_execution
```

---

## 2. 九个工具

### 2.1 `observe_scene()`

**输入**：无。

**输出**

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | bool | 观测成功与否 |
| `ts` / `ts_wall` | float | 观测时刻（ROS 秒 / UTC 秒） |
| `summary` | str | 人类可读摘要，含目标、`surface_base`、`center_base`、`mask_px`、图像路径 |
| `objects` | list | 每项含 `kind` / `surface_base` / `center_base` / `confidence`；检测失败时为空列表 |
| `image_path` | str \| null | 落盘帧路径（`data/frames/frame_<时间戳>.png`） |
| `frame_detail` | str | 取帧诊断（如 RGB 时间戳与尺寸） |
| `image_ts` | float \| null | 帧时间戳 |

MCP 层额外把该帧作为图像内容块一并返回（供 VLM 使用）。

**失败**：`error_kind=tool_error`，`reason` 取自感知模块，取值之一为
`no_frames` / `stale` / `not_found` / `depth_insufficient` / `implausible_z`。

**示例（成功）**

```json
{
  "ok": true, "error_kind": null, "reason": "",
  "ts": 226.617, "ts_wall": 1791061628.63,
  "summary": "1 object(s): red_tower surface_base=[0.0507,-0.3522,0.06] ...; image=/home/yhwang/fyp/data/frames/frame_...png",
  "objects": [{"kind": "red_tower",
               "surface_base": [0.0507, -0.3522, 0.06],
               "center_base": [0.0507, -0.3522, 0.03],
               "confidence": {"mask_px": 1527, "valid_depth_ratio": 1.0}}],
  "image_path": "/home/yhwang/fyp/data/frames/frame_20261004_043712_123.png"
}
```

> `surface_base` 是**顶面**点（真值 z=0.06），不是物体中心；中心由顶面下移 0.03 m（立柱半高）得到，见 `docs/camera_extrinsics.md`。

---

### 2.2 `locate_object(description)`

**输入**

| 字段 | 类型 | 说明 |
|---|---|---|
| `description` | str | 目标描述，见 §1 映射表 |

**输出**：`ok`、`ts`、`ts_wall`、`description`、`object_kind`、`surface_base`、`center_base`、`confidence`。

**失败**

- `error_kind=tool_error, reason=unsupported_target`：描述不在映射表内，另带 `supported` 列表。
- `error_kind=tool_error, reason=<感知原因>`：感知模块失败，另带 `object_kind` 与 `confidence`。

**示例（成功）**

```json
{"ok": true, "error_kind": null, "reason": "",
 "description": "红色立柱", "object_kind": "red_tower",
 "surface_base": [0.050703, -0.352163, 0.06],
 "center_base": [0.050703, -0.352163, 0.03],
 "confidence": {"mask_px": 1527, "valid_depth_ratio": 1.0},
 "ts": 226.617, "ts_wall": 1791061628.63}
```

该调用的 `ts` 同时被登记为"观测戳"，用于后续的计划时效检查。

---

### 2.3 `propose_grasps(surface_xy)`

**输入**

| 字段 | 类型 | 说明 |
|---|---|---|
| `surface_xy` | `[x, y]`（或含 `x`/`y` 的 dict） | 顶面点，单位米，`base_link` 系 |

**输出**：`ok`、`grasps`、`surface_xy`、`primary_grasp_id`、`detail`。

每个候选：

| 字段 | 说明 |
|---|---|
| `grasp_id` | `g1` / `g2` / `g3` |
| `pre_grasp` | `[x, y, 0.10]`，米 |
| `grasp` | `[x, y, z]`，z 依次为 0.010（主）、0.008、0.012 |
| `orientation` | 固定 `"top_down"` |
| `rank` | 1 起排序 |
| `source` | `"geometric_top_down_v1"` |

**失败**：`error_kind=tool_error, reason=invalid_surface_xy`（`surface_xy` 不是两个数）。

**示例**

```json
{"ok": true,
 "grasps": [{"grasp_id": "g1", "pre_grasp": [0.0507, -0.3522, 0.1],
             "grasp": [0.0507, -0.3522, 0.01], "orientation": "top_down",
             "rank": 1, "source": "geometric_top_down_v1"}],
 "surface_xy": [0.0507, -0.3522], "primary_grasp_id": "g1",
 "detail": "geometric top-down v1: single surface-point candidate plus 2 backup heights"}
```

---

### 2.4 `plan_grasp(grasp)`

**输入**：`propose_grasps` 返回的候选 dict，或已登记的 `grasp_id` 字符串。

**行为**：**只规划不执行**。用容差 `PLAN_TOL=(0.15, 0.15, 3.14)` 依次校验 `pre_grasp` 与 `grasp`（MoveIt 服务 `/plan_kinematic_path`），成功后把计划登记进计划表并绑定当前观测戳。

**输出**：`ok`、`plan_id`、`obs_ts`、`obs_ts_wall`、`grasp`、`tol`、`planned`（每个位姿的 `error_code` 与轨迹点数）、`detail`。

**失败**

| reason | 说明 |
|---|---|
| `unknown_grasp_id` | `grasp_id` 未登记 |
| `invalid_grasp` | 不是候选 dict 或 `pre_grasp`/`grasp` 不是三元数值 |
| `<tag>_plan_error_<code>` | MoveIt 返回非 1 的 `error_code`（`tag` ∈ `pre_grasp`/`grasp`） |
| `plan_service_unavailable` / `plan_call_timeout` / `plan_call_exception` / `plan_call_no_response` | 服务或调用层问题 |

以上失败均为 `error_kind=plan_failed`（输入类为 `tool_error`）。

**示例**

```json
{"ok": true, "plan_id": "plan_0007", "obs_ts": 226.617,
 "grasp": {"grasp_id": "g1", "pre_grasp": [0.0507, -0.3522, 0.1],
           "grasp": [0.0507, -0.3522, 0.01], "orientation": "top_down"},
 "tol": [0.15, 0.15, 3.14],
 "planned": {"pre_grasp": {"error_code": 1, "points": 154},
             "grasp": {"error_code": 1, "points": 169}},
 "detail": "pre_grasp and grasp plan only, no motion executed"}
```

---

### 2.5 `execute_plan(plan_id)`

**输入**：`plan_id`（来自 `plan_grasp`）。

**行为**：**立即返回** `exec_id`，完整抓放流程在后台线程运行：
`open → pre_grasp → grasp → close → lift → carry check → （失败时夹爪方向升级重试）→ place_above → place_down → release → retreat → settle → place check → home`。
实际运动在执行时由同一套可规划性门**重新规划**；登记的计划用于授权与时效检查，而不是回放固定轨迹;自 D-2 起,抓取高度以 plan 登记的 z 为准(其余位姿仍由可规划性门实时校验)。

**拒绝条件（均为 `tool_error`）**：`unknown_plan_id`、`execution_in_progress`、`no_observation`、
`stale_plan`（观测龄期 > 30 s，返回 `obs_age`）。

**输出**：`ok`、`exec_id`、`plan_id`、`state="running"`、`stage="queued"`、`obs_ts`、`detail`。

---

### 2.6 `get_execution_status(exec_id)`

**输入**：`exec_id`。

**输出**

| 字段 | 说明 |
|---|---|
| `state` | `running` / `succeeded` / `failed` |
| `stage` | 当前/最后阶段：`gate` `pre_grasp` `grasp` `lift` `carry_check` `place_above` `place_down` `release` `retreat` `settle` `place_check` `home` `done` `stopped` |
| `carried` | 是否真的把物体搬走（bool \| null） |
| `place_success` | 放置判定（bool \| null） |
| `dxy` / `z` | 放置偏差（米）/ 末端高度（米），未到该阶段为 null |
| `failed_stage` | 失败发生的阶段 |
| `stopped` | 是否被 `stop_execution` 置位 |
| `error_kind` | 四级分类 |
| `detail`、`elapsed`、`ts` | 说明文本、已耗时（秒）、时间戳 |

**失败**：`error_kind=tool_error, reason=unknown_exec_id`，另带 `known` 列表。

> 该工具是唯一可以在**没有仿真在跑**时调用的接口（对未知 id 是纯本地错误），因此被 MCP 协议级验证用作真实 `tools/call` 往返样例。

---

### 2.7 `verify_task(expect_place_xy)`

**输入**：`expect_place_xy = [x, y]`（米，`base_link`）。

**行为**：从 Gazebo world 读立柱真值位姿，计算 `dxy`（到期望点的平面距离）与 `z`；
`task_success = dxy < 0.03 且 z < 0.04`。

**输出**：`task_success`、`carried`（取自最近一次执行）、`dxy`、`z`、`expect_place_xy`、`truth_pose`、`truth_only_for_eval=true`、`exec_id`、`detail`。

**失败**：`task_success=false` 时 `error_kind=task_failed, reason=place_out_of_tolerance`；
真值读不到时 `error_kind=tool_error, reason=truth_unavailable`。

---

### 2.8 `reset_scene(target_xy=None)`

**输入**：`target_xy = [x, y]`（米，可选；默认回到立柱起始位 `(0.05, -0.35)`，z 取 `CUBE_START[2]=0.03`）。

**行为**：先回 HOME，再用 gz `set_pose` 把立柱传回起始位；返回前复测位姿。
**执行中拒绝**：有执行在跑时返回 `tool_error`。

**输出**：`target_xy`、`truth_pose`、`truth_only_for_eval=true`、`detail`。

**失败**

| reason | error_kind | 说明 |
|---|---|---|
| `invalid_target_xy` | `tool_error` | 入参不是两个数 |
| `busy` | `tool_error` | 运动锁被占用超过 900 s |
| `execution_in_progress` | `tool_error` | 有执行在跑，另带 `active_exec_id` |
| `reset_pose_not_reached` | `exec_failed` | 复位后实测位姿偏差 > 20 mm（平面/高度各自 < 0.02 判成功） |

---

### 2.9 `stop_execution()`

**输入**：无。

**行为**：给当前执行的停止标志置位（流程在**下一个阶段边界**退出并把 stage 置为 `stopped`），随后打开夹爪释放物体，机械臂停在原地。
注意语义：已在飞行中的运动不会被打断——这正是"松开夹爪、手臂不动"的实现方式。

**输出**：`exec_id`（无执行时为 null）、`stage_at_stop`、`stopped`、`gripper{ok, detail}`、`detail`。

**无执行在跑时**：仍返回 `ok=true`，`stopped=false`，`detail="no active execution; nothing to stop"`，并照常尝试释放夹爪。

---

## 3. 线程与并发约定

- 所有对阶段 A/B 原语的调用由一把 `_ros_lock` 串行化（rclpy 的阻塞式 helper 只能由一个线程驱动全局 executor）。
- 本模块自己的 ROS 节点由私有 `MultiThreadedExecutor` 驱动，不使用全局 executor。
- 结果：`reset_scene` 会等待正在跑的执行结束（避免把立柱从夹爪里瞬移走），而 `stop_execution` 因为只碰自己的节点，始终可响应。

---

## 4. 验证方式

| 验证 | 命令 | 结果 |
|---|---|---|
| 独立测试（9 工具，无需 API key） | `cd /home/yhwang/fyp && python3 -u robot_tools/test_tools.py` | 30 checks / 30 passed（`logs/phaseC1_tools_test.log`） |
| MCP 协议级 `tools/list` + 真实 `tools/call` | `python3 robot_tools/mcp_list_tools.py` | `COUNT 9`、`MATCH True`、`CALL_VERDICT ok`（`logs/phaseC1_mcp_list.log`） |

早期一次运行 `phaseC1_tools_test_run1.log` 为 12 项检查 11 通过 1 失败（`stop_execution` 的 live 停止语义），修正后为 30/30。
