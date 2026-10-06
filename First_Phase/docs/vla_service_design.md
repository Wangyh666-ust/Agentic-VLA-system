# VLA 执行服务设计(阶段 D,v1 契约)

> 对应 `changing_to_VLA_structure.md` §3/§6 与 proposal §5.2:策略执行有预算、可停止、可观测进度;底层短段闭环;高层只在决策点介入;真值只给独立评估器。

## 1. 进程结构

```text
Hermes (python3.14)            评测 runner (E1–E4)
   │ MCP stdio                      │ HTTP
   ▼                                ▼
vla_mcp_server.py (系统 python3,mcp 2.3.0,无 torch)
   │ HTTP JSON 127.0.0.1:8765        ▲
   ▼                                 │
vla_service.py (vla venv:torch + mani_skill + lerobot,常驻)
   ├─ env: SO100GraspCube-v1 v3 变体(spawn 0.215,0.02 / 0.035,256×256,pd_joint_target_delta_pos)
   ├─ backend: SmolVLABackend(ckpt) | ReplayBackend(h5)   ← 后者用于不占 GPU 的管线自测
   ├─ adapter: 绝对目标 → 增量指令(同 eval_smolvla_v3)
   └─ worker 线程:唯一推进仿真的线程
```

- 服务常驻:策略加载约 30 s,不适合每个 MCP 会话重载;runner 与 Hermes 共用同一执行语义。
- 仿真只在执行期间推进。高层思考期间世界静止(仿真特性,报告中注明)。

## 2. 状态机

`idle` →(execute)→ `running` →(结束)→ `completed` | `stopped` | `error` → 下一次 execute 前回到可执行状态。

- 同一时刻只允许一个执行。
- `completed` 只表示"本次执行结束",**不等于任务成功**。结束原因 `ended_reason` ∈ {`max_steps_reached`, `episode_budget_exhausted`, `settled`, `stopped`, `error`}。
- `settled`:本体感觉判据——手臂控制目标距 rest 位 < 0.2 rad 且夹爪目标 ≤ −0.7,并连续保持 10 步。只用机器人自身可得的量。

## 3. 预算与公平性

- **episode 动作预算** `episode_budget`(默认 250 步),由同一 episode 内所有执行共享;重试会消耗同一预算。
- `execute(max_steps)` 超出剩余预算时截断为剩余值,并在返回里标出 `max_steps_effective`。
- `reset` **只给 runner**,不经 MCP 暴露。episode 内不能靠复位伪装成恢复。

## 4. 指令白名单

- 只接受 backend 声明的 `supported_instructions`(当前 checkpoint 只有 `"Grasp the red cube."`);其它指令一律拒绝,返回 `reason="unsupported_instruction"` 并附带支持列表。
- 高层不得假定任意 grasp/release 之类的指令都受支持。

## 5. 短段闭环与动作过期

- 每 `n_action_steps`(默认 10 步 = 0.5 s)用最新观测重新推理一次;不会盲执行完整的 50 步动作块。
- 新执行开始时以及 stop 时都调用 `backend.reset()`,清空动作队列,旧动作不会跨执行残留。
- 每个 episode 开始时用 episode seed 设定 torch/numpy 随机种子;第 k 次执行开始时设为 `seed*1000 + k`,保证可复现。

## 6. HTTP 接口(JSON)

| 方法 | 路径 | 调用方 | 请求 | 返回要点 |
|---|---|---|---|---|
| POST | `/reset` | runner | `{seed:int}` | `{ok, episode_id, seed}` |
| GET | `/observe` | 两者 | — | `{ok, episode_id, episode_step, image_path, qpos[6], controller_target[6], gripper_blocked, arm_near_rest, executing, wall_time}` |
| POST | `/execute` | 两者 | `{instruction:str, max_steps:int}` | `{accepted, exec_id, max_steps_effective, reason?, supported_instructions?}` |
| GET | `/status` | 两者 | — | `{state, exec_id, instruction, steps_executed, max_steps_effective, episode_steps_used, episode_budget, ended_reason, gripper_blocked, arm_near_rest, last_infer_ms, started_wall, updated_wall}` |
| POST | `/stop` | 两者 | — | `{ok, state}`(当前步完成后停止,队列已清) |
| GET | `/evaluate` | 仅 runner | — | `{success, is_grasped, cube_lifted, distance_to_rest_qpos}`(**真值,绝不经 MCP 暴露**) |
| GET | `/health` | 两者 | — | `{ok, backend, ckpt, supported_instructions, episode_budget, n_action_steps}` |

- **v1.1 更正**:原设计的 `gripper_blocked`(夹爪目标 ≤ −0.7 且实测角 > −0.65)已删除。实测 300 条成功示范,夹住方块时夹爪关节也到达 −0.800,与空夹无法区分,关节角不携带"是否夹住"的信息。现改为 `gripper_closed_cmd = (controller_target[5] ≤ −0.7)`,只表示"已下发闭合指令"。是否夹住必须靠视觉判断。
- `image_path`:策略自己的 base_camera(256×256,高位前侧视角)。
- `agent_image_path`(v1.1 新增):只给高层 Agent 的侧视相机,384×384,eye (0.20, −0.40, 0.025),target (0.18, 0.02, 0.06),fov 1.0 rad,通过 `human_render_camera_configs` 覆盖 `render_camera`,不进入 VLA。
  - 机位选择有实测依据:最初 eye z=0.10,与抬起后方块的高度相同,方块投影落在桌面远端地平线上。Hermes(qwen3-vl-plus + vision_analyze)在同一条确定性 rollout(seed 7,71 步,真值 success)上两次判为 failure。降到 z=0.025 后判为 success,与真值一致。证据:`logs/phaseVLA_D2_hermes_replay{,_v2,_v3}.log`、`logs/review_D2v3_side_{start,end}.png`。
  - 高层视觉核验本身就是一个误差源。正式实验中逐 episode 记录 `AGENT_VERDICT` 与 `/evaluate` 的一致率,单独报告。
- MCP 返回的图像块在 Hermes 里会被降级为文本(见 README §七 第 9 条),所以图像一律以文件路径提供,由 `vision_analyze` 读取。
- 并发:`_sim_lock` 只包住 `env.reset`、`env.step`、`env.render`、`evaluate` 四个调用,观测渲染不会与物理步并发。

## 7. 日志

- 每步一行 JSONL trace:`{episode_id, exec_id, episode_step, instruction, qpos, policy_out, cmd, infer_ms, replanned}`。
- 每次执行结束写一行 summary。runner 另外记录 `/evaluate` 的真值,与 trace 分开存放。

## 8. MCP 工具(给 Hermes)

`observe_scene`、`execute_skill(instruction, max_steps)`、`get_execution_status`、`stop_execution`、`list_skills`。不暴露 reset 与 evaluate。
